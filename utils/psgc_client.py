import requests
import logging
from utils.config import settings
from typing import Optional, Dict, List, Any

logger = logging.getLogger(__name__)


class PSGCClient:
    def __init__(self):
        """
        Initializes a PSGCClient instance.

        :param timeout: Optional timeout in seconds for requests to the PSGC API (default is 30 seconds)
        """
        self.base_url = settings.PSGC_BASE_URL
        self.timeout = settings.PSGC_TIMEOUT
        self.session = requests.Session()
        logger.info(f"PSGCClient initialized with base_url={self.base_url}, timeout={self.timeout}s")
        
        
    def _get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Internal method to fetch data from the PSGC API.
        
        :param endpoint: The endpoint to fetch data from (e.g. "provinces", "cities-municipalities", "barangays")
        :param params: Optional parameters to pass to the request (e.g., {"provinceCode": "NCR"})
        :return: Data fetched from the PSGC API, or an empty list on failure
        """
        url = f"{self.base_url}/{endpoint}"
        logger.debug(f"Fetching PSGC data from {endpoint} with params: {params}")
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            logger.debug(f"Successfully fetched {len(data) if isinstance(data, list) else 'N/A'} items from {endpoint}")
            return data if data else []
        except requests.RequestException as e:
            logger.error(f"Error fetching PSGC data from {endpoint}: {e}")
            return []
    
    def get_provinces(self) -> List[Dict[str, Any]]:
        """
        Fetches all provinces in the Philippines from the PSGC API.

        :return: A list of dictionaries containing province data, or an empty list on failure
        """
        result = self._get("provinces")
        if isinstance(result, list):
            return result
        return []
    
    def get_cities_municipalities(self, province_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches all cities/municipalities in the Philippines from the PSGC API,
        optionally filtered by province code.

        :param province_code: Optional province code to filter by (e.g., "NCR")
        :return: A list of dictionaries containing city/municipality data, or an empty list on failure
        """
        params = {}
        if province_code:
            params['provinceCode'] = province_code
        
        result = self._get("cities-municipalities", params)
        if isinstance(result, list):
            return result
        return []
    
    def get_barangays(self, city_code: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetches all barangays in the Philippines from the PSGC API,
        optionally filtered by city code.

        :param city_code: Optional city code to filter by (e.g., "NCR-QUEZON CITY")
        :return: A list of dictionaries containing barangay data, or an empty list on failure
        """
        params = {}
        if city_code:
            params['cityCode'] = city_code
        
        result = self._get("barangays", params)
        if isinstance(result, list):
            return result
        return []
    
    def search_province(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Searches for a province in the Philippines from the PSGC API.

        :param name: The name of the province to search for
        :return: A dictionary containing the province data, or None if not found
        """
        logger.debug(f"Searching for province: {name}")
        provinces = self.get_provinces()
        if not provinces:
            logger.debug("No provinces returned from API")
            return None
        
        if not isinstance(provinces, list):
            logger.debug(f"provinces is not a list, it's a {type(provinces)}")
            return None
        
        name_lower = name.lower().strip()
        
        for province in provinces:
            if not isinstance(province, dict):
                logger.debug(f"province item is not a dict, it's a {type(province)}: {province}")
                continue
            if province.get('name', '').lower() == name_lower:
                logger.debug(f"Found exact province match: {province.get('name')}")
                return province
        
        for province in provinces:
            if not isinstance(province, dict):
                continue
            if name_lower in province.get('name', '').lower():
                logger.debug(f"Found partial province match: {province.get('name')}")
                return province
        
        logger.debug(f"Province not found: {name}")
        return None
    
    def search_city_municipality(self, name: str, province_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Searches for a city/municipality in the Philippines from the PSGC API.

        :param name: The name of the city/municipality to search for
        :param province_code: Optional province code to filter by (e.g., "NCR")
        :return: A dictionary containing the city/municipality data, or None if not found
        """
        logger.debug(f"Searching for city/municipality: {name} (province_code={province_code})")
        cities = self.get_cities_municipalities(province_code)
        if not cities:
            logger.debug("No cities returned from API")
            return None
        
        if not isinstance(cities, list):
            logger.debug(f"cities is not a list, it's a {type(cities)}")
            return None
        
        name_lower = name.lower().strip()
        
        name_cleaned = name_lower.replace('city of ', '').replace('municipality of ', '')
        
        for city in cities:
            if not isinstance(city, dict):
                logger.debug(f"city item is not a dict, it's a {type(city)}: {city}")
                continue
            city_name = city.get('name', '').lower()
            city_cleaned = city_name.replace('city of ', '').replace('municipality of ', '')
            
            if city_cleaned == name_cleaned or city_name == name_lower:
                logger.debug(f"Found exact city match: {city.get('name')}")
                return city
        
        for city in cities:
            if not isinstance(city, dict):
                continue
            city_name = city.get('name', '').lower()
            if name_cleaned in city_name or city_name in name_cleaned:
                logger.debug(f"Found partial city match: {city.get('name')}")
                return city
        
        logger.debug(f"City/municipality not found: {name}")
        return None
    
    def search_barangay(self, name: str, city_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Searches for a barangay in the Philippines from the PSGC API.

        :param name: The name of the barangay to search for
        :param city_code: Optional city code to filter by (e.g., "NCR-QUEZON CITY")
        :return: A dictionary containing the barangay data, or None if not found
        """
        logger.debug(f"Searching for barangay: {name} (city_code={city_code})")
        barangays = self.get_barangays(city_code)
        if not barangays:
            logger.debug("No barangays returned from API")
            return None
        
        if not isinstance(barangays, list):
            logger.debug("Barangays data is not a list")
            return None
        
        name_lower = name.lower().strip()
        
        name_cleaned = name_lower.replace('barangay ', '').replace('brgy. ', '').replace('brgy ', '')
        
        for barangay in barangays:
            if not isinstance(barangay, dict):
                continue
            brgy_name = barangay.get('name', '').lower()
            brgy_cleaned = brgy_name.replace('barangay ', '').replace('brgy. ', '').replace('brgy ', '')
            
            if brgy_cleaned == name_cleaned or brgy_name == name_lower:
                logger.debug(f"Found exact barangay match: {barangay.get('name')}")
                return barangay
        
        for barangay in barangays:
            if not isinstance(barangay, dict):
                continue
            brgy_name = barangay.get('name', '').lower()
            if name_cleaned in brgy_name or brgy_name in name_cleaned:
                logger.debug(f"Found partial barangay match: {barangay.get('name')}")
                return barangay
        
        logger.debug(f"Barangay not found: {name}")
        return None
