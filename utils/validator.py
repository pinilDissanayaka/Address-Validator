import re
import uuid
import logging
from typing import Dict, Optional, List, Tuple
from schema import (
    EnhancedAddressValidationResponse,
    VerdictResponse,
    StructureResponse,
    PSGCResponse,
    GeocodeResponse,
    DeliveryHistoryResponse,
    DeliveryHistoryAddress
)
from utils.address_parser import AddressParser
from utils.philatlas_client import PhilAtlasClient
from core.gmaps_integration import GoogleMapsValidator
from core import database_client as db

logger = logging.getLogger(__name__)


class AddressValidator:
    """
    Comprehensive address validator integrating:
    1. LLM-based address parsing (OpenAI)
    2. PhilAtlas validation
    3. PSGC database matching
    4. Google Maps geocoding
    5. Delivery history lookup
    """
    
    def __init__(
        self,
        parser: AddressParser,
        philatlas_client: PhilAtlasClient,
        gmaps_api_key: Optional[str] = None
    ):
        """Initialize enhanced validator with all required clients."""
        self.parser = parser
        self.philatlas_client = philatlas_client
        self.gmaps_validator = GoogleMapsValidator(gmaps_api_key) if gmaps_api_key else None
        self.db_available = db.is_database_available()
        
        logger.info(f"EnhancedAddressValidator initialized (DB: {self.db_available}, GMaps: {self.gmaps_validator is not None})")
    
    async def validate_address(self, address_text: str) -> EnhancedAddressValidationResponse:
        """
        Main validation flow following bod-ph logic:
        1. Check if international or PH address (Google Address Validation)
        2. Parse address structure (OpenAI LLM)
        3. Validate against PhilAtlas
        4. Match against PSGC database
        5. Geocode the address
        6. Check delivery history
        7. Calculate confidence score
        """
        logger.info(f"Starting enhanced validation for: {address_text[:50]}...")
        
        validation_id = str(uuid.uuid4())
        _incoming_address = address_text.strip().replace(",", "")
        temp_data = self._init_temp_data()
        
        logger.info("STEP 1: Address country validation")
        if self.gmaps_validator:
            addr_validation = self.gmaps_validator.get_address_country(_incoming_address)
            logger.debug(f"Address validation result: {addr_validation}")
            
            if addr_validation.get("isValid") == True:
                detected_country = addr_validation.get("country", "Unknown")
                logger.info(f"Non-Philippine address detected: {detected_country}")
                temp_data["country"] = detected_country
                temp_data["reason"].append(f"We only validate Philippine addresses. The provided address appears to be from {detected_country}.")
                temp_data["structureOk"] = False
                temp_data["geocodeMatched"] = False
                temp_data["psgcMatched"] = False
                
                verdict = self._calculate_verdict(temp_data)
                return self._build_response(validation_id, address_text, temp_data, verdict)
                
            elif addr_validation.get("isValid") == "PH":
                temp_data["country"] = "PH"
            elif addr_validation.get("isValid") == "NOT_PH":
                detected_country = addr_validation.get("country", "")
                if detected_country:
                    logger.info(f"Non-Philippine address detected: {detected_country}")
                    temp_data["country"] = detected_country
                    temp_data["reason"].append(f"We only validate Philippine addresses. The provided address appears to be from {detected_country}.")
                else:
                    logger.info("Non-Philippine address detected (country unknown)")
                    temp_data["country"] = "Unknown"
                    temp_data["reason"].append("We only validate Philippine addresses. The provided address does not appear to be from the Philippines.")
                
                temp_data["structureOk"] = False
                temp_data["geocodeMatched"] = False
                temp_data["psgcMatched"] = False
                
                verdict = self._calculate_verdict(temp_data)
                return self._build_response(validation_id, address_text, temp_data, verdict)
            else:
                temp_data["country"] = addr_validation.get("country", "PH")
        else:
            temp_data["country"] = "PH"
        
        if temp_data["country"] == "PH" and not temp_data["structureOk"]:
            logger.info("STEP 2: Parsing PH address with OpenAI LLM")
            parsed = await self.parser.parse(_incoming_address)
            logger.debug(f"LLM parsed: province={parsed.province}, city={parsed.city}, barangay={parsed.barangay}")
            
            parsed = self._apply_aliases(parsed)
            
            logger.info("STEP 3: Validating against PhilAtlas")
            philatlas_results = self._validate_with_philatlas(parsed)
            temp_data.update(philatlas_results)
            
            if self.db_available:
                logger.info("STEP 4: Matching against PSGC database")
                psgc_results = self._match_psgc_database(temp_data)
                temp_data.update(psgc_results)
            
            if all([temp_data.get("streetAddress"), temp_data.get("barangay"), 
                    temp_data.get("city"), temp_data.get("province")]):
                temp_data["formattedAddress"] = self._format_ph_address(temp_data)
                temp_data["structureOk"] = True
        
        if temp_data["country"] in ["PH", "NOT_PH"] and self.gmaps_validator:
            logger.info("STEP 5: Geocoding address")
            geocode_result = self._geocode_address(temp_data, _incoming_address)
            temp_data.update(geocode_result)
            
            if self.db_available and (geocode_result.get("city") or geocode_result.get("barangay")):
                logger.info("Re-matching PSGC after geocode filled missing components")
                psgc_results = self._match_psgc_database(temp_data)
                temp_data.update(psgc_results)
                
                if all([temp_data.get("streetAddress"), temp_data.get("barangay"), 
                        temp_data.get("city"), temp_data.get("province")]):
                    temp_data["formattedAddress"] = self._format_ph_address(temp_data)
                    temp_data["structureOk"] = True
        
        # STEP 6: Check delivery history
        if self.db_available:
            logger.info("STEP 6: Checking delivery history")
            delivery_history = self._check_delivery_history(_incoming_address, temp_data.get("formattedAddress", ""))
            temp_data.update(delivery_history)  
        
        logger.info("STEP 7: Calculating confidence score")
        verdict = self._calculate_verdict(temp_data)
        
        return self._build_response(validation_id, address_text, temp_data, verdict)
    
    def _init_temp_data(self) -> Dict:
        """Initialize temporary data structure."""
        return {
            "structureOk": False,
            "psgcMatched": False,
            "geocodeMatched": False,
            "deliveryHistorySuccess": 0,
            "confidence": 0.0,
            "province": "",
            "city": "",
            "barangay": "",
            "streetAddress": "",
            "postalCode": "",
            "country": "PH",
            "latitude": 0,
            "longitude": 0,
            "place_id": "",
            "formattedAddress": "",
            "geocodeFormattedAddress": "",
            "region_id": "",
            "province_id": "",
            "city_id": "",
            "barangay_id": "",
            "reason": [],
            "suggestion": [],
            "deliveryHistory": {},
        }
    
    def _apply_aliases(self, parsed):
        """Apply known aliases for provinces and cities."""
        from utils.config import settings
        
        if parsed.province:
            province_lower = parsed.province.lower()
            if province_lower in settings.PROVINCE_ALIASES:
                parsed.province = settings.PROVINCE_ALIASES[province_lower].title()
        
        if parsed.city:
            city_lower = parsed.city.lower()
            if city_lower in settings.CITY_ABBREVIATIONS:
                parsed.city = settings.CITY_ABBREVIATIONS[city_lower].title()
        
        return parsed
    
    def _validate_with_philatlas(self, parsed) -> Dict:
        """Validate address components against PhilAtlas."""
        result = {
            "province": "",
            "city": "",
            "barangay": "",
            "streetAddress": parsed.street_address or "",
            "postalCode": parsed.postal_code or "",
        }
        
        if parsed.province:
            province_match = self.philatlas_client.search_province(parsed.province)
            if province_match:
                result["province"] = province_match.get('name', '').title()
                logger.info(f"Province validated: {parsed.province} -> {result['province']}")
            else:
                result["province"] = parsed.province.title()
                result["reason"] = result.get("reason", [])
                result["reason"].append(f"Province '{parsed.province}' not found in PhilAtlas")
        
        if parsed.city:
            city_match = self.philatlas_client.search_city_municipality(
                parsed.city, result.get("province")
            )
            if city_match:
                result["city"] = city_match.get('name', '').title()
                logger.info(f"City validated: {parsed.city} -> {result['city']}")
            else:
                result["city"] = parsed.city.title()
                result["reason"] = result.get("reason", [])
                result["reason"].append(f"City '{parsed.city}' not found in PhilAtlas")
        
        if parsed.barangay:
            barangay_match = self.philatlas_client.search_barangay(
                parsed.barangay, result.get("city"), result.get("province")
            )
            if barangay_match:
                result["barangay"] = barangay_match.get('name', '').title()
                logger.info(f"Barangay validated: {parsed.barangay} -> {result['barangay']}")
            else:
                result["barangay"] = parsed.barangay.title()
                result["reason"] = result.get("reason", [])
                result["reason"].append(f"Barangay '{parsed.barangay}' not found in PhilAtlas")
        
        return result
    
    def _match_psgc_database(self, temp_data: Dict) -> Dict:
        """Match validated components against PSGC database."""
        result = {}
        
        if temp_data.get("province"):
            province_details = db.get_province_details(temp_data["province"])
            if province_details:
                result["region_id"] = str(province_details.get("region_id", ""))
                result["province_id"] = str(province_details.get("province_id", ""))
                logger.debug(f"PSGC province matched: {province_details}")
        
        if temp_data.get("city"):
            city_details = db.get_city_details(temp_data["city"])
            if city_details:
                result["city_id"] = str(city_details.get("city_id", ""))
                logger.debug(f"PSGC city matched: {city_details}")
        
        if temp_data.get("barangay"):
            barangay_details = db.get_barangay_details(temp_data["barangay"])
            if barangay_details:
                result["barangay_id"] = str(barangay_details.get("barangay_id", ""))
                if not temp_data.get("postalCode"):
                    result["postalCode"] = str(barangay_details.get("postcode", ""))
                logger.debug(f"PSGC barangay matched: {barangay_details}")
        
        result["psgcMatched"] = all([
            result.get("region_id"),
            result.get("province_id"),
            result.get("city_id"),
            result.get("barangay_id")
        ])
        
        return result
    
    def _format_ph_address(self, temp_data: Dict) -> str:
        """Format Philippine address components."""
        components = []
        
        if temp_data.get("streetAddress"):
            components.append(temp_data["streetAddress"])
        if temp_data.get("barangay"):
            components.append(temp_data["barangay"])
        if temp_data.get("city"):
            components.append(temp_data["city"])
        if temp_data.get("province"):
            components.append(temp_data["province"])
        if temp_data.get("postalCode"):
            components.append(str(temp_data["postalCode"]))
        components.append("PH")
        
        formatted = " ".join(components)
        return re.sub(r"\s+", " ", formatted).strip()
    
    def _geocode_address(self, temp_data: Dict, original_address: str) -> Dict:
        """Geocode the address using Google Maps."""
        if not self.gmaps_validator:
            return {}
        
        address = temp_data.get("formattedAddress") or original_address
        geocode_result = self.gmaps_validator.get_geocode(address, temp_data.get("country", "PH"))
        
        result = {}
        
        if geocode_result.get("isValid") == False:
            result["reason"] = result.get("reason", [])
            result["reason"].append(geocode_result.get("reason", "Geocoding failed"))
            result["geocodeMatched"] = False
        elif temp_data.get("country") == "PH" and geocode_result.get("isValid"):
            result["latitude"] = geocode_result.get("latitude", 0)
            result["longitude"] = geocode_result.get("longitude", 0)
            result["place_id"] = geocode_result.get("place_id", "")
            result["geocodeFormattedAddress"] = geocode_result.get("formattedAddress", "")
            result["geocodeMatched"] = True
            
            if not temp_data.get("city") and geocode_result.get("city"):
                result["city"] = geocode_result.get("city")
                logger.info(f"City filled from geocode: {result['city']}")
            
            if not temp_data.get("barangay") and geocode_result.get("barangay"):
                result["barangay"] = geocode_result.get("barangay")
                logger.info(f"Barangay filled from geocode: {result['barangay']}")
            
            if not temp_data.get("postalCode") and geocode_result.get("postalCode"):
                result["postalCode"] = geocode_result.get("postalCode")
                logger.info(f"Postal code filled from geocode: {result['postalCode']}")
                
        elif geocode_result.get("isValid"):
            result["structureOk"] = True
            result["streetAddress"] = geocode_result.get("streetAddress", "")
            result["city"] = geocode_result.get("city", "")
            result["province"] = geocode_result.get("province", "")
            result["postalCode"] = geocode_result.get("postalCode", "")
            result["country"] = geocode_result.get("country", "")
            result["formattedAddress"] = geocode_result.get("formattedAddress", "")
            result["latitude"] = geocode_result.get("latitude", 0)
            result["longitude"] = geocode_result.get("longitude", 0)
            result["place_id"] = geocode_result.get("place_id", "")
            result["geocodeMatched"] = True
        
        return result
    
    def _check_delivery_history(self, original_address: str, formatted_address: str) -> Dict:
        """Check delivery history for both original and formatted addresses."""
        def format_history(history_records: List) -> DeliveryHistoryAddress:
            """Format delivery history records into a DeliveryHistoryAddress object.
            
            If history_records is not empty and has at least one record, return a
            DeliveryHistoryAddress object with address, status, last_delivery_at, and
            rts_reason filled from the first record. Otherwise, return an empty
            DeliveryHistoryAddress object.
            """
            if history_records and len(history_records) > 0:
                record = history_records[0]
                last_delivery = record.get("last_delivery")
                return DeliveryHistoryAddress(
                    address=record.get("address") or "",
                    status=record.get("status") or "",
                    last_delivery_at=str(last_delivery) if last_delivery else "",
                    rts_reason=record.get("failure_reason") or ""
                )
            return DeliveryHistoryAddress()
        
        original_history = db.get_delivery_history(original_address)
        input_delivery = format_history(original_history)
        
        formatted_history = db.get_delivery_history(formatted_address) if formatted_address else []
        formatted_delivery = format_history(formatted_history)
        
        delivery_success = 0
        if original_history and len(original_history) > 0:
            if original_history[0].get("status") == "DELIVERED":
                delivery_success = 1
            elif original_history[0].get("status") == "RETURN TO SENDER":
                delivery_success = -1
        
        if formatted_history and len(formatted_history) > 0:
            if formatted_history[0].get("status") == "DELIVERED":
                delivery_success = 1
            elif formatted_history[0].get("status") == "RETURN TO SENDER":
                delivery_success = -1
        
        return {
            "deliveryHistorySuccess": delivery_success,
            "deliveryHistory": DeliveryHistoryResponse(
                inputAddress=input_delivery,
                formattedAddress=formatted_delivery
            )
        }
    
    def _calculate_verdict(self, temp_data: Dict) -> VerdictResponse:
        """Calculate validation verdict and confidence score."""
        structure_ok = temp_data.get("structureOk", False)
        psgc_matched = temp_data.get("psgcMatched", False)
        geocode_matched = temp_data.get("geocodeMatched", False)
        delivery_success = temp_data.get("deliveryHistorySuccess", 0)
        
        reasons = temp_data.get("reason", [])
        is_non_ph_rejection = any("only validate Philippine addresses" in reason for reason in reasons)
        
        if structure_ok and psgc_matched and geocode_matched and delivery_success == 1:
            confidence = 99
        elif not structure_ok and not psgc_matched and not geocode_matched and delivery_success <= 0:
            confidence = 0
        else:
            confidence = 50
            confidence += 15 if delivery_success == 1 else (0 if delivery_success == 0 else -15)
            confidence += 20 if structure_ok else -10
            confidence += 10 if psgc_matched else -5
            confidence += 4 if geocode_matched else -2
            confidence = max(0, min(100, confidence)) 
        
        if is_non_ph_rejection:
            is_valid = False
        else:
            is_valid = structure_ok or psgc_matched or geocode_matched or delivery_success == 1
        
        return VerdictResponse(
            isValid=is_valid,
            structureOk=structure_ok,
            psgcMatched=psgc_matched,
            geocodeMatched=geocode_matched,
            deliveryHistorySuccess=(delivery_success == 1),
            confidence=confidence
        )
    
    def _build_response(
        self,
        validation_id: str,
        original_input: str,
        temp_data: Dict,
        verdict: VerdictResponse
    ) -> EnhancedAddressValidationResponse:
        """Build the final response object."""
        return EnhancedAddressValidationResponse(
            id=validation_id,
            input=original_input,
            formattedAddress=temp_data.get("formattedAddress", temp_data.get("geocodeFormattedAddress", "")),
            verdict=verdict,
            structure=StructureResponse(
                streetAddress=temp_data.get("streetAddress", ""),
                barangay=temp_data.get("barangay", ""),
                city=temp_data.get("city", ""),
                province=temp_data.get("province", ""),
                country=temp_data.get("country", "PH"),
                postalCode=str(temp_data.get("postalCode", "")),
                formattedAddress=temp_data.get("formattedAddress", "")
            ),
            psgc=PSGCResponse(
                regionCode=str(temp_data.get("region_id", "")),
                provinceCode=str(temp_data.get("province_id", "")),
                cityMuniCode=str(temp_data.get("city_id", "")),
                barangayCode=str(temp_data.get("barangay_id", ""))
            ),
            geocode=GeocodeResponse(
                lat=temp_data.get("latitude", 0),
                lng=temp_data.get("longitude", 0),
                place_id=temp_data.get("place_id", ""),
                formattedAddress=temp_data.get("geocodeFormattedAddress", "")
            ),
            deliveryHistory=temp_data.get("deliveryHistory", DeliveryHistoryResponse()),
            reason=temp_data.get("reason", []),
            suggestions=temp_data.get("suggestion", [])
        )