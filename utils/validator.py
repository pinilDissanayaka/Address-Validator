from typing import Optional, Dict, Any, List, Tuple
import logging
from schema import AddressValidationResponse, ParsedAddress, GeocodingData
from utils.address_parser import AddressParser
from utils.philatlas_client import PhilAtlasClient
from utils.geocoding_client import GeocodingClient
from utils.config import settings

logger = logging.getLogger(__name__)


class AddressValidator:
    def __init__(self, parser: AddressParser, philatlas_client: PhilAtlasClient, 
                 geocoding_client: Optional[GeocodingClient] = None):
        """
        Initializes an AddressValidator instance
        :param parser: An instance of AddressParser to parse addresses
        :param philatlas_client: An instance of PhilAtlasClient to interact with PhilAtlas
        :param geocoding_client: Optional GeocodingClient for geocoding valid addresses
        """
        self.parser = parser
        self.philatlas_client = philatlas_client
        self.geocoding_client = geocoding_client
        logger.info("AddressValidator initialized with PhilAtlas client")
        if self.geocoding_client:
            logger.info("Geocoding client enabled")

    
    def _normalize_name(self, name: Optional[str]) -> Optional[str]:
        """
        Normalize an address component name by converting to title case, stripping, and removing common prefixes
        :param name: The name to normalize
        :return: The normalized name, or None if the input was None
        """
        if not name:
            return None
        
        normalized = name.strip().title()
        
        prefixes_to_remove = ['Barangay ', 'Brgy. ', 'Brgy ', 'City Of ', 'Municipality Of ']
        for prefix in prefixes_to_remove:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        return normalized
    
    def _apply_aliases(self, parsed: ParsedAddress) -> ParsedAddress:
        """
        Apply known aliases for provinces and cities to parsed address components
        :param parsed: ParsedAddress to apply aliases to
        :return: ParsedAddress with aliases applied
        """
        if parsed.province:
            province_lower = parsed.province.lower()
            if province_lower in settings.PROVINCE_ALIASES:
                original = parsed.province
                parsed.province = settings.PROVINCE_ALIASES[province_lower].title()
                logger.debug(f"Applied province alias: {original} -> {parsed.province}")
        
        if parsed.city:
            city_lower = parsed.city.lower()
            if city_lower in settings.CITY_ABBREVIATIONS:
                original = parsed.city
                parsed.city = settings.CITY_ABBREVIATIONS[city_lower].title()
                logger.debug(f"Applied city abbreviation: {original} -> {parsed.city}")
        
        return parsed
    
    async def validate_address(self, address_text: str) -> AddressValidationResponse:
        """
        Main validation method:
        1. Parse the address using LLM
        2. Validate each component against PhilAtlas data
        3. Return structured response with validation reasons
        """
        logger.debug(f"Starting address validation for: {address_text[:50]}...")
        
        parsed = await self.parser.parse(address_text)
        logger.debug(f"Parsed address: province={parsed.province}, city={parsed.city}, barangay={parsed.barangay}")
        
        parsed = self._apply_aliases(parsed)
        
        validated_data = self._validate_components(parsed)
        logger.debug(f"Validated components: {validated_data}")
        
        # Check validity and collect reasons
        is_valid, validation_reasons = self._check_validity(validated_data)
        logger.info(f"Address validation complete: isValid={is_valid}")
        
        formatted = self._format_address(validated_data)
        
        # Geocode the address if it's valid and geocoding client is available
        geocoding_data = None
        all_reasons = list(validation_reasons)  # Copy validation reasons
        
        if is_valid and self.geocoding_client:
            logger.debug("Address is valid, attempting geocoding...")
            geocode_result = self.geocoding_client.geocode_address(formatted)
            
            if geocode_result:
                # Check if there's an error reason
                if 'error_reason' in geocode_result:
                    logger.warning(f"Geocoding failed: {geocode_result['error_reason']}")
                    all_reasons.append(f"Geocoding failed: {geocode_result['error_reason']}")
                else:
                    # Check for warnings (like unsupported region types)
                    if 'warning' in geocode_result:
                        all_reasons.append(geocode_result['warning'])
                    
                    geocoding_data = GeocodingData(
                        latitude=geocode_result.get('latitude'),
                        longitude=geocode_result.get('longitude'),
                        formatted_address=geocode_result.get('formatted_address'),
                        place_id=geocode_result.get('place_id'),
                        location_type=geocode_result.get('location_type')
                    )
                    logger.info(f"Geocoding successful: {geocoding_data.latitude}, {geocoding_data.longitude}")
            else:
                logger.warning("Geocoding failed for valid address")
                all_reasons.append("Geocoding failed: Unable to find geographic coordinates for this address")
        
        return AddressValidationResponse(
            isValid=is_valid,
            province=validated_data.get('province'),
            city=validated_data.get('city'),
            barangay=validated_data.get('barangay'),
            streetAddress=validated_data.get('street_address'),
            postalCode=validated_data.get('postal_code'),
            formattedAddress=formatted,
            geocoding=geocoding_data,
            reasons=all_reasons
        )

    
    def _validate_components(self, parsed: ParsedAddress) -> Dict[str, Optional[str]]:
        """
        Validate each component against PhilAtlas data
        Returns a dictionary with validated/normalized component names and validation status
        """
        logger.debug("Starting component validation against PhilAtlas data")
        result = {
            'province': None,
            'city': None,
            'barangay': None,
            'street_address': parsed.street_address,
            'postal_code': parsed.postal_code,
            'province_validated': False,
            'city_validated': False,
            'barangay_validated': False,
            'province_original': parsed.province,
            'city_original': parsed.city,
            'barangay_original': parsed.barangay
        }
        
        if parsed.province:
            logger.debug(f"Validating province: {parsed.province}")
            province_match = self.philatlas_client.search_province(parsed.province)
            if province_match:
                result['province'] = self._normalize_name(province_match.get('name'))
                result['province_validated'] = True
                logger.info(f"Province validated: {parsed.province} -> {result['province']}")
            else:
                result['province'] = self._normalize_name(parsed.province)
                result['province_validated'] = False
                logger.warning(f"Province not found in PhilAtlas: {parsed.province}")
        
        if parsed.city:
            logger.debug(f"Validating city: {parsed.city}")
            city_match = self.philatlas_client.search_city_municipality(
                parsed.city, 
                result.get('province')
            )
            if city_match:
                result['city'] = self._normalize_name(city_match.get('name'))
                result['city_validated'] = True
                logger.info(f"City validated: {parsed.city} -> {result['city']}")
            else:
                result['city'] = self._normalize_name(parsed.city)
                result['city_validated'] = False
                logger.warning(f"City not found in PhilAtlas: {parsed.city}")
        
        if parsed.barangay:
            logger.debug(f"Validating barangay: {parsed.barangay}")
            barangay_match = self.philatlas_client.search_barangay(
                parsed.barangay,
                result.get('city'),
                result.get('province')
            )
            if barangay_match:
                result['barangay'] = self._normalize_name(barangay_match.get('name'))
                result['barangay_validated'] = True
                logger.info(f"Barangay validated: {parsed.barangay} -> {result['barangay']}")
            else:
                result['barangay'] = self._normalize_name(parsed.barangay)
                result['barangay_validated'] = False
                logger.warning(f"Barangay not found in PhilAtlas: {parsed.barangay}")
        
        return result
    
    def _check_validity(self, validated_data: Dict[str, Optional[str]]) -> Tuple[bool, List[str]]:
        """
        Check if address is valid based on validation rules.
        Returns tuple of (is_valid, reasons)
        
        Validation rules:
        - Must have Province
        - Must have City/Municipality
        - Must have Barangay that is validated against PhilAtlas
        - Must have Street Address
        - Components must belong to each other (city in province, barangay in city)
        """
        reasons = []
        required_fields = ['province', 'city', 'barangay', 'street_address']
        
        # Check missing fields
        for field in required_fields:
            if not validated_data.get(field):
                field_name = field.replace('_', ' ').title()
                reasons.append(f"Missing required field: {field_name}")
        
        # Check province validation
        if validated_data.get('province') and not validated_data.get('province_validated'):
            province_name = validated_data.get('province_original') or validated_data.get('province')
            reasons.append(f"Province '{province_name}' not found in PhilAtlas database")
        
        # Check city validation
        if validated_data.get('city') and not validated_data.get('city_validated'):
            city_name = validated_data.get('city_original') or validated_data.get('city')
            province_name = validated_data.get('province', 'specified province')
            
            if validated_data.get('province_validated'):
                reasons.append(f"City '{city_name}' not found in {province_name}")
            else:
                reasons.append(f"City '{city_name}' not found in PhilAtlas database")
        
        # Check barangay validation
        if validated_data.get('barangay') and not validated_data.get('barangay_validated'):
            barangay_name = validated_data.get('barangay_original') or validated_data.get('barangay')
            city_name = validated_data.get('city', 'specified city')
            
            if validated_data.get('city_validated'):
                reasons.append(f"Barangay '{barangay_name}' not found in {city_name}")
            else:
                reasons.append(f"Barangay '{barangay_name}' cannot be validated (city not found)")
        
        is_valid = len(reasons) == 0
        
        if not is_valid:
            logger.debug(f"Address validation failed. Reasons: {', '.join(reasons)}")
        
        return is_valid, reasons
    
    def _format_address(self, validated_data: Dict[str, Optional[str]]) -> str:
        """
        Format the validated address components into a single string
        """
        components = []
        
        if validated_data.get('street_address'):
            components.append(validated_data['street_address'])
        
        if validated_data.get('barangay'):
            components.append(f"Barangay {validated_data['barangay']}")
        
        if validated_data.get('city'):
            components.append(validated_data['city'])
        
        if validated_data.get('province'):
            components.append(validated_data['province'])
        
        if validated_data.get('postal_code'):
            components.append(validated_data['postal_code'])
        
        return ', '.join(components)
