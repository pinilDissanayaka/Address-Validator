"""
LangGraph-based Address Validation Agent
Converts the validation flow into a state graph with tool calls for better observability and control.
"""

import uuid
import logging
from typing import Dict, Optional, List, TypedDict, Annotated, Literal
from operator import add

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for older versions
    from langgraph.graph import StateGraph
    END = "__end__"

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


class ValidationState(TypedDict):
    """State for the address validation workflow"""
    validation_id: str
    original_address: str
    clean_address: str
    
    country: str
    structure_ok: bool
    psgc_matched: bool
    geocode_matched: bool
    philatlas_validated: bool
    delivery_history_success: int
    
    province: str
    city: str
    barangay: str
    street_address: str
    postal_code: str
    
    latitude: float
    longitude: float
    place_id: str
    geocode_formatted_address: str
    
    region_id: str
    province_id: str
    city_id: str
    barangay_id: str
    
    formatted_address: str
    confidence: float
    reasons: Annotated[List[str], add]
    suggestions: Annotated[List[str], add]
    delivery_history: Optional[DeliveryHistoryResponse]
    
    current_step: str
    error: Optional[str]


class AddressValidatorAgent:
    """
    LangGraph-based address validator with tool calls.
    Each validation step is a node in the graph with clear transitions.
    """
    
    def __init__(
        self,
        parser: AddressParser,
        philatlas_client: PhilAtlasClient,
        gmaps_api_key: Optional[str] = None
    ):
        """Initialize the validator agent with required clients."""
        self.parser = parser
        self.philatlas_client = philatlas_client
        self.gmaps_validator = GoogleMapsValidator(gmaps_api_key) if gmaps_api_key else None
        self.db_available = db.is_database_available()
        
        self.graph = self._build_graph()
        
        logger.info(f"AddressValidatorAgent initialized (DB: {self.db_available}, GMaps: {self.gmaps_validator is not None})")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph validation workflow."""
        workflow = StateGraph(ValidationState)
        
        workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("validate_country", self._validate_country)
        workflow.add_node("parse_address", self._parse_address)
        workflow.add_node("validate_philatlas", self._validate_philatlas)
        workflow.add_node("match_psgc", self._match_psgc)
        workflow.add_node("geocode", self._geocode_address)
        workflow.add_node("check_delivery", self._check_delivery_history)
        workflow.add_node("calculate_verdict", self._calculate_verdict)
        
        workflow.set_entry_point("initialize")
        
        workflow.add_edge("initialize", "validate_country")
        workflow.add_conditional_edges(
            "validate_country",
            self._route_after_country_check,
            {
                "parse": "parse_address",
                "end": "calculate_verdict"
            }
        )
        
        workflow.add_edge("parse_address", "validate_philatlas")
        workflow.add_edge("validate_philatlas", "match_psgc")
        workflow.add_edge("match_psgc", "geocode")
        workflow.add_edge("geocode", "check_delivery")
        workflow.add_edge("check_delivery", "calculate_verdict")
        workflow.add_edge("calculate_verdict", END)
        
        return workflow.compile()
    
    async def validate_address(self, address_text: str) -> EnhancedAddressValidationResponse:
        """
        Main entry point for address validation.
        Executes the LangGraph workflow and returns the response.
        """
        logger.info(f"Starting LangGraph validation for: {address_text[:50]}...")
        
        initial_state: ValidationState = {
            "validation_id": str(uuid.uuid4()),
            "original_address": address_text,
            "clean_address": address_text.strip().replace(",", ""),
            "country": "PH",
            "structure_ok": False,
            "psgc_matched": False,
            "geocode_matched": False,
            "philatlas_validated": True,
            "delivery_history_success": 0,
            "province": "",
            "city": "",
            "barangay": "",
            "street_address": "",
            "postal_code": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "place_id": "",
            "geocode_formatted_address": "",
            "region_id": "",
            "province_id": "",
            "city_id": "",
            "barangay_id": "",
            "formatted_address": "",
            "confidence": 0.0,
            "reasons": [],
            "suggestions": [],
            "delivery_history": None,
            "current_step": "initialize",
            "error": None
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return self._build_response(final_state)
    
    
    def _initialize_state(self, state: ValidationState) -> ValidationState:
        """Initialize the validation state."""
        logger.info("STEP 1: Initializing validation state")
        state["current_step"] = "initialize"
        return state
    
    def _validate_country(self, state: ValidationState) -> ValidationState:
        """Validate if address is from Philippines."""
        logger.info("STEP 2: Validating country")
        state["current_step"] = "validate_country"
        
        if not self.gmaps_validator:
            state["country"] = "PH"
            return state
        
        addr_validation = self.gmaps_validator.get_address_country(state["clean_address"])
        logger.debug(f"Country validation result: {addr_validation}")
        
        if addr_validation.get("isValid") == True:
            detected_country = addr_validation.get("country", "Unknown")
            logger.info(f"Non-Philippine address detected: {detected_country}")
            state["country"] = detected_country
            state["reasons"].append(
                f"We only validate Philippine addresses. The provided address appears to be from {detected_country}."
            )
            state["structure_ok"] = False
            state["geocode_matched"] = False
            state["psgc_matched"] = False
            
        elif addr_validation.get("isValid") == "PH":
            state["country"] = "PH"
            
        elif addr_validation.get("isValid") == "NOT_PH":
            detected_country = addr_validation.get("country", "")
            if detected_country:
                logger.info(f"Non-Philippine address detected: {detected_country}")
                state["country"] = detected_country
                state["reasons"].append(
                    f"We only validate Philippine addresses. The provided address appears to be from {detected_country}."
                )
            else:
                logger.info("Non-Philippine address detected (country unknown)")
                state["country"] = "Unknown"
                state["reasons"].append(
                    "We only validate Philippine addresses. The provided address does not appear to be from the Philippines."
                )
            state["structure_ok"] = False
            state["geocode_matched"] = False
            state["psgc_matched"] = False
        else:
            state["country"] = addr_validation.get("country", "PH")
        
        return state
    
    async def _parse_address(self, state: ValidationState) -> ValidationState:
        """Parse address using LLM."""
        logger.info("STEP 3: Parsing address with LLM")
        state["current_step"] = "parse_address"
        
        parsed = await self.parser.parse(state["clean_address"])
        logger.debug(f"LLM parsed: province={parsed.province}, city={parsed.city}, barangay={parsed.barangay}")
        
        parsed = self._apply_aliases(parsed)
        
        state["province"] = parsed.province or ""
        state["city"] = parsed.city or ""
        state["barangay"] = parsed.barangay or ""
        state["street_address"] = parsed.street_address or ""
        state["postal_code"] = parsed.postal_code or ""
        
        return state
    
    def _validate_philatlas(self, state: ValidationState) -> ValidationState:
        """Validate address components against PhilAtlas."""
        logger.info("STEP 4: Validating against PhilAtlas")
        state["current_step"] = "validate_philatlas"
        
        philatlas_valid = True
        new_reasons = []
        
        if state["province"]:
            province_match = self.philatlas_client.search_province(state["province"])
            if province_match:
                state["province"] = province_match.get('name', '').title()
                logger.info(f"Province validated: {state['province']}")
            else:
                reason = f"Province '{state['province']}' not found in PhilAtlas"
                if reason not in state["reasons"]:
                    new_reasons.append(reason)
                state["province"] = state["province"].title()
                philatlas_valid = False
        
        if state["city"]:
            city_match = self.philatlas_client.search_city_municipality(
                state["city"], state["province"]
            )
            if city_match:
                state["city"] = city_match.get('name', '').title()
                logger.info(f"City validated: {state['city']}")
            else:
                reason = f"City '{state['city']}' not found in PhilAtlas"
                if reason not in state["reasons"]:
                    new_reasons.append(reason)
                state["city"] = state["city"].title()
                philatlas_valid = False
        
        if state["barangay"]:
            barangay_match = self.philatlas_client.search_barangay(
                state["barangay"], state["city"], state["province"]
            )
            if barangay_match:
                state["barangay"] = barangay_match.get('name', '').title()
                logger.info(f"Barangay validated: {state['barangay']}")
            else:
                reason = f"Barangay '{state['barangay']}' not found in PhilAtlas"
                if reason not in state["reasons"]:
                    new_reasons.append(reason)
                state["barangay"] = state["barangay"].title()
                philatlas_valid = False
        
        if new_reasons:
            state["reasons"].extend(new_reasons)
        
        state["philatlas_validated"] = philatlas_valid
        
        if all([state["street_address"], state["barangay"], state["city"], state["province"]]):
            state["formatted_address"] = self._format_ph_address(state)
            state["structure_ok"] = True
        
        return state
    
    def _match_psgc(self, state: ValidationState) -> ValidationState:
        """Match address components against PSGC database."""
        logger.info("STEP 5: Matching against PSGC database")
        state["current_step"] = "match_psgc"
        
        if not self.db_available:
            return state
        
        if state["province"]:
            province_details = db.get_province_details(state["province"])
            if province_details:
                state["region_id"] = str(province_details.get("region_id", ""))
                state["province_id"] = str(province_details.get("province_id", ""))
                logger.debug(f"PSGC province matched: {province_details}")
        
        if state["city"]:
            city_details = db.get_city_details(state["city"])
            if city_details:
                state["city_id"] = str(city_details.get("city_id", ""))
                logger.debug(f"PSGC city matched: {city_details}")
        
        if state["barangay"]:
            barangay_details = db.get_barangay_details(state["barangay"])
            if barangay_details:
                state["barangay_id"] = str(barangay_details.get("barangay_id", ""))
                if not state["postal_code"]:
                    state["postal_code"] = str(barangay_details.get("postcode", ""))
                logger.debug(f"PSGC barangay matched: {barangay_details}")
        
        psgc_codes_found = all([
            state["region_id"],
            state["province_id"],
            state["city_id"],
            state["barangay_id"]
        ])
        
        state["psgc_matched"] = psgc_codes_found and state["philatlas_validated"]
        
        if psgc_codes_found and not state["philatlas_validated"]:
            logger.warning("PSGC codes found but PhilAtlas validation failed - marking psgcMatched as False")
        
        return state
    
    def _geocode_address(self, state: ValidationState) -> ValidationState:
        """Geocode the address using Google Maps."""
        logger.info("STEP 6: Geocoding address")
        state["current_step"] = "geocode"
        
        if not self.gmaps_validator:
            return state
        
        address = state["formatted_address"] or state["clean_address"]
        geocode_result = self.gmaps_validator.get_geocode(address, state["country"])
        
        if geocode_result.get("isValid") == False:
            state["reasons"].append(geocode_result.get("reason", "Geocoding failed"))
            state["geocode_matched"] = False
            
        elif state["country"] == "PH" and geocode_result.get("isValid"):
            state["latitude"] = geocode_result.get("latitude", 0)
            state["longitude"] = geocode_result.get("longitude", 0)
            state["place_id"] = geocode_result.get("place_id", "")
            state["geocode_formatted_address"] = geocode_result.get("formattedAddress", "")
            state["geocode_matched"] = True
            
            if not state["city"] and geocode_result.get("city"):
                state["city"] = geocode_result.get("city")
                logger.info(f"City filled from geocode: {state['city']}")
                
                if self.db_available:
                    logger.info("Re-matching PSGC after geocode filled missing components")
                    city_details = db.get_city_details(state["city"])
                    if city_details:
                        state["city_id"] = str(city_details.get("city_id", ""))
            
            if not state["barangay"] and geocode_result.get("barangay"):
                state["barangay"] = geocode_result.get("barangay")
                logger.info(f"Barangay filled from geocode: {state['barangay']}")
                
                if self.db_available:
                    barangay_details = db.get_barangay_details(state["barangay"])
                    if barangay_details:
                        state["barangay_id"] = str(barangay_details.get("barangay_id", ""))
                        if not state["postal_code"]:
                            state["postal_code"] = str(barangay_details.get("postcode", ""))
            
            if not state["postal_code"] and geocode_result.get("postalCode"):
                state["postal_code"] = geocode_result.get("postalCode")
                logger.info(f"Postal code filled from geocode: {state['postal_code']}")
            
            if all([state["street_address"], state["barangay"], state["city"], state["province"]]):
                state["formatted_address"] = self._format_ph_address(state)
                state["structure_ok"] = True
                
        elif geocode_result.get("isValid"):
            state["structure_ok"] = True
            state["street_address"] = geocode_result.get("streetAddress", "")
            state["city"] = geocode_result.get("city", "")
            state["province"] = geocode_result.get("province", "")
            state["postal_code"] = geocode_result.get("postalCode", "")
            state["country"] = geocode_result.get("country", "")
            state["formatted_address"] = geocode_result.get("formattedAddress", "")
            state["latitude"] = geocode_result.get("latitude", 0)
            state["longitude"] = geocode_result.get("longitude", 0)
            state["place_id"] = geocode_result.get("place_id", "")
            state["geocode_matched"] = True
        
        return state
    
    def _check_delivery_history(self, state: ValidationState) -> ValidationState:
        """Check delivery history for the address."""
        logger.info("STEP 7: Checking delivery history")
        state["current_step"] = "check_delivery"
        
        if not self.db_available:
            return state
        
        def format_history(history_records: List) -> DeliveryHistoryAddress:
            """Format delivery history records."""
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
        
        original_history = db.get_delivery_history(state["clean_address"])
        input_delivery = format_history(original_history)
        
        formatted_history = db.get_delivery_history(state["formatted_address"]) if state["formatted_address"] else []
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
        
        state["delivery_history_success"] = delivery_success
        state["delivery_history"] = DeliveryHistoryResponse(
            inputAddress=input_delivery,
            formattedAddress=formatted_delivery
        )
        
        return state
    
    def _calculate_verdict(self, state: ValidationState) -> ValidationState:
        """Calculate final validation verdict and confidence score."""
        logger.info("STEP 8: Calculating verdict and confidence")
        state["current_step"] = "calculate_verdict"
        
        structure_ok = state["structure_ok"]
        psgc_matched = state["psgc_matched"]
        geocode_matched = state["geocode_matched"]
        philatlas_validated = state["philatlas_validated"]
        delivery_success = state["delivery_history_success"]
        
        # Check for rejection reasons
        reasons = state["reasons"]
        is_non_ph_rejection = any("only validate Philippine addresses" in reason for reason in reasons)
        has_philatlas_errors = any("not found in PhilAtlas" in reason for reason in reasons)
        
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
            
            if has_philatlas_errors:
                confidence -= 20
            
            confidence = max(0, min(100, confidence))
        
        state["confidence"] = confidence
        
        logger.info(f"Validation complete: confidence={confidence}, philatlas_validated={philatlas_validated}")
        
        return state
    
    
    def _route_after_country_check(self, state: ValidationState) -> Literal["parse", "end"]:
        """Route based on country validation result."""
        if state["country"] == "PH":
            return "parse"
        else:
            return "end"
    
    
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
    
    def _format_ph_address(self, state: ValidationState) -> str:
        """Format Philippine address components."""
        components = []
        
        if state["street_address"]:
            components.append(state["street_address"])
        if state["barangay"]:
            components.append(state["barangay"])
        if state["city"]:
            components.append(state["city"])
        if state["province"]:
            components.append(state["province"])
        if state["postal_code"]:
            components.append(str(state["postal_code"]))
        components.append("PH")
        
        formatted = " ".join(components)
        import re
        return re.sub(r"\s+", " ", formatted).strip()
    
    def _build_response(self, state: ValidationState) -> EnhancedAddressValidationResponse:
        """Build the final response from state."""
        seen = set()
        unique_reasons = []
        for reason in state["reasons"]:
            if reason not in seen:
                seen.add(reason)
                unique_reasons.append(reason)
        
        seen_suggestions = set()
        unique_suggestions = []
        for suggestion in state["suggestions"]:
            if suggestion not in seen_suggestions:
                seen_suggestions.add(suggestion)
                unique_suggestions.append(suggestion)
        
        reasons = unique_reasons
        is_non_ph_rejection = any("only validate Philippine addresses" in reason for reason in reasons)
        has_philatlas_errors = any("not found in PhilAtlas" in reason for reason in reasons)
        
        if is_non_ph_rejection:
            is_valid = False
        else:
            if has_philatlas_errors and state["delivery_history_success"] != 1:
                is_valid = False
            else:
                is_valid = (
                    state["structure_ok"] or 
                    state["psgc_matched"] or 
                    state["geocode_matched"] or 
                    state["delivery_history_success"] == 1
                )
        
        return EnhancedAddressValidationResponse(
            id=state["validation_id"],
            input=state["original_address"],
            formattedAddress=state["formatted_address"] or state["geocode_formatted_address"],
            verdict=VerdictResponse(
                isValid=is_valid,
                structureOk=state["structure_ok"],
                psgcMatched=state["psgc_matched"],
                geocodeMatched=state["geocode_matched"],
                deliveryHistorySuccess=(state["delivery_history_success"] == 1),
                confidence=state["confidence"]
            ),
            structure=StructureResponse(
                streetAddress=state["street_address"],
                barangay=state["barangay"],
                city=state["city"],
                province=state["province"],
                country=state["country"],
                postalCode=str(state["postal_code"]),
                formattedAddress=state["formatted_address"]
            ),
            psgc=PSGCResponse(
                regionCode=str(state["region_id"]),
                provinceCode=str(state["province_id"]),
                cityMuniCode=str(state["city_id"]),
                barangayCode=str(state["barangay_id"])
            ),
            geocode=GeocodeResponse(
                lat=state["latitude"],
                lng=state["longitude"],
                place_id=state["place_id"],
                formattedAddress=state["geocode_formatted_address"]
            ),
            deliveryHistory=state["delivery_history"] or DeliveryHistoryResponse(),
            reason=unique_reasons,
            suggestions=unique_suggestions
        )
