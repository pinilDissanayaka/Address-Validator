from typing import Optional, Dict, Any
import logging
from schema import AddressValidationResponse, ParsedAddress
from utils.address_parser import AddressParser
from utils.philatlas_client import PhilAtlasClient
from utils.config import settings

logger = logging.getLogger(__name__)


class AddressValidator:
    def __init__(self, parser: AddressParser, philatlas_client: PhilAtlasClient):
        """
        Initializes an AddressValidator instance
        :param parser: An instance of AddressParser to parse addresses
        :param philatlas_client: An instance of PhilAtlasClient to interact with PhilAtlas
        """
        self.parser = parser
        self.philatlas_client = philatlas_client
        logger.info("AddressValidator initialized with PhilAtlas client")
    
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
        3. Return structured response
        """
        logger.debug(f"Starting address validation for: {address_text[:50]}...")
        
        parsed = await self.parser.parse(address_text)
        logger.debug(f"Parsed address: province={parsed.province}, city={parsed.city}, barangay={parsed.barangay}")
        
        parsed = self._apply_aliases(parsed)
        
        validated_data = self._validate_components(parsed)
        logger.debug(f"Validated components: {validated_data}")
        
        is_valid = self._check_validity(validated_data)
        logger.info(f"Address validation complete: isValid={is_valid}")
        
        formatted = self._format_address(validated_data)
        
        return AddressValidationResponse(
            isValid=is_valid,
            province=validated_data.get('province'),
            city=validated_data.get('city'),
            barangay=validated_data.get('barangay'),
            streetAddress=validated_data.get('street_address'),
            postalCode=validated_data.get('postal_code'),
            formattedAddress=formatted
        )
    
    def _validate_components(self, parsed: ParsedAddress) -> Dict[str, Optional[str]]:
        """
        Validate each component against PhilAtlas data with hierarchical validation.
        Ensures barangay belongs to city, and city belongs to province.
        Returns a dictionary with validated/normalized component names.
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
            'hierarchy_valid': False
        }
        
        province_data = None
        if parsed.province:
            logger.debug(f"Validating province: {parsed.province}")
            province_data = self.philatlas_client.search_province(parsed.province)
            if province_data:
                result['province'] = self._normalize_name(province_data.get('name'))
                result['province_validated'] = True
                logger.info(f"Province validated: {parsed.province} -> {result['province']}")
            else:
                result['province'] = self._normalize_name(parsed.province)
                logger.warning(f"Province not found in PhilAtlas: {parsed.province}")
        
        city_data = None
        if parsed.city:
            logger.debug(f"Validating city: {parsed.city}")
            city_data = self.philatlas_client.search_city_municipality(
                parsed.city, 
                result.get('province')
            )
            if city_data:
                result['city'] = self._normalize_name(city_data.get('name'))
                
                if province_data and city_data.get('province'):
                    city_province = self._normalize_name(city_data.get('province'))
                    expected_province = self._normalize_name(province_data.get('name'))
                    if city_province == expected_province:
                        result['city_validated'] = True
                        logger.info(f"City validated and belongs to province: {result['city']} -> {result['province']}")
                    else:
                        logger.warning(f"City '{result['city']}' does not belong to province '{result['province']}'. City's actual province: {city_data.get('province')}")
                        result['city_validated'] = False
                elif not province_data:
                    result['city_validated'] = True
                    logger.info(f"City validated: {parsed.city} -> {result['city']}")
                else:
                    result['city_validated'] = True
                    logger.info(f"City validated: {parsed.city} -> {result['city']}")
            else:
                result['city'] = self._normalize_name(parsed.city)
                logger.warning(f"City not found in PhilAtlas: {parsed.city}")
        
        if parsed.barangay:
            logger.debug(f"Validating barangay: {parsed.barangay}")
            barangay_data = self.philatlas_client.search_barangay(
                parsed.barangay,
                result.get('city'),
                result.get('province')
            )
            if barangay_data:
                result['barangay'] = self._normalize_name(barangay_data.get('name'))
                
                if city_data and barangay_data.get('city'):
                    barangay_city = self._normalize_name(barangay_data.get('city'))
                    expected_city = self._normalize_name(city_data.get('name'))
                    if barangay_city == expected_city:
                        result['barangay_validated'] = True
                        logger.info(f"Barangay validated and belongs to city: {result['barangay']} -> {result['city']}")
                    else:
                        logger.warning(f"Barangay '{result['barangay']}' does not belong to city '{result['city']}'. Barangay's actual city: {barangay_data.get('city')}")
                        result['barangay_validated'] = False
                elif not city_data:
                    logger.warning(f"Barangay found but city not validated")
                    result['barangay_validated'] = False
                else:
                    result['barangay_validated'] = True
                    logger.info(f"Barangay validated: {parsed.barangay} -> {result['barangay']}")
            else:
                result['barangay'] = self._normalize_name(parsed.barangay)
                result['barangay_validated'] = False
                logger.warning(f"Barangay not found in PhilAtlas: {parsed.barangay}")
        
        result['hierarchy_valid'] = (
            result['province_validated'] and 
            result['city_validated'] and 
            result['barangay_validated']
        )
        
        if result['hierarchy_valid']:
            logger.info("Hierarchical validation successful: Province -> City -> Barangay")
        else:
            logger.warning(f"Hierarchical validation failed. Province: {result['province_validated']}, City: {result['city_validated']}, Barangay: {result['barangay_validated']}")
        
        return result
    
    def _check_validity(self, validated_data: Dict[str, Optional[str]]) -> bool:
        """
        Check if address is valid based on validation rules:
        - Must have Province
        - Must have City/Municipality
        - Must have Barangay
        - Must have Street Address
        - Barangay must belong to City
        - City must belong to Province
        """
        required_fields = ['province', 'city', 'barangay', 'street_address']
        missing_fields = [field for field in required_fields if not validated_data.get(field)]
        
        if missing_fields:
            logger.debug(f"Address validation failed. Missing fields: {', '.join(missing_fields)}")
            return False
        
        if not validated_data.get('hierarchy_valid', False):
            logger.debug("Address validation failed. Hierarchical validation failed (barangay must belong to city, city must belong to province)")
            return False
        
        return True
    
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
