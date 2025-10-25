import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any
from functools import lru_cache
import logging
import re
from difflib import SequenceMatcher
from utils.config import settings

logger = logging.getLogger(__name__)


class PhilAtlasClient:
    """
    Client for scraping and accessing Philippine administrative division data from PhilAtlas.
    
    This client provides methods to dynamically scrape and search provinces, cities,
    municipalities, and barangays from the PhilAtlas website.
    """
    
    BASE_URL = settings.PHILATLAS_BASE_URL
    def __init__(self, timeout: float = 10.0):
        """
        Initializes a PhilAtlasClient instance with a given timeout for requests.

        Args:
            timeout: float, optional
                Timeout in seconds for requests to PhilAtlas. Defaults to 10.0.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info(f"PhilAtlasClient initialized with timeout={timeout}s")
    
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.75) -> float:
        """
        Calculate the fuzzy match ratio between a query string and a target string.
        
        Args:
            query: str
                Query string to compare against the target string.
            target: str
                Target string to compare against the query string.
            threshold: float, optional
                Minimum fuzzy match ratio to consider the query and target strings as matching. Defaults to 0.75.
        
        Returns:
            float
                Fuzzy match ratio between the query and target strings.
        """
        return SequenceMatcher(None, query.lower(), target.lower()).ratio()
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize an address component name by converting to lowercase, stripping, and removing common prefixes.
        
        :param name: The name to normalize
        :return: The normalized name, or an empty string if the input was None
        """
    
        if not name:
            return ""
        
        normalized = name.lower().strip()
        
        prefixes = ['city of ', 'municipality of ', 'barangay ', 'brgy. ', 'brgy ', 'the ']
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]
                break
        
        return normalized
    
    @lru_cache(maxsize=128)
    def _fetch_page(self, url: str) -> Optional[BeautifulSoup]:
        """
        Fetch a PhilAtlas webpage and return its parsed HTML content using BeautifulSoup.

        Args:
            url: str
                URL of the PhilAtlas webpage to fetch

        Returns:
            Optional[BeautifulSoup ]
                Parsed HTML content of the webpage, or None if the request fails
        """
        try:
            logger.debug(f"Fetching PhilAtlas page: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            logger.debug(f"Successfully fetched page: {url}")
            return soup
        except requests.RequestException as e:
            logger.error(f"Error fetching PhilAtlas page {url}: {e}")
            return None
    
    @lru_cache(maxsize=32)
    def get_provinces(self) -> List[Dict[str, Any]]:
        """
        Fetch all provinces from PhilAtlas and return a list of dictionaries containing the province name, URL, region, and code.
        
        Returns:
            List[Dict[str, Any]]
                List of dictionaries containing the province name, URL, region, and code.
        """
        logger.info("Fetching provinces from PhilAtlas")
        url = f"{self.BASE_URL}/provinces.html"
        soup = self._fetch_page(url)
        
        if not soup:
            logger.error("Failed to fetch provinces page")
            return []
        
        provinces = []
        
        links = soup.find_all('a', href=re.compile(r'/(luzon|visayas|mindanao)/[^/]+/[^/]+\.html'))
        
        for link in links:
            province_name = link.get_text(strip=True)
            province_url = link.get('href', '')
            
            if not province_name or not province_url:
                continue
            
            url_parts = province_url.split('/')
            region = url_parts[2] if len(url_parts) > 2 else 'unknown'
            
            provinces.append({
                'name': province_name,
                'url': province_url,
                'region': region.upper(),
                'code': self._normalize_name(province_name).upper().replace(' ', '_')
            })
        
        ncr_link = soup.find('a', href=re.compile(r'/luzon/ncr\.html'))
        if ncr_link:
            provinces.append({
                'name': 'Metro Manila',
                'url': '/luzon/ncr.html',
                'region': 'NCR',
                'code': 'METRO_MANILA'
            })
        
        logger.info(f"Found {len(provinces)} provinces from PhilAtlas")
        return provinces
    
    def search_province(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a province by name.
        
        Args:
            name: Province name to search for
            
        Returns:
            Dictionary with province data or None if not found
        """
        if not name:
            return None
        
        normalized_query = self._normalize_name(name)
        logger.debug(f"Searching for province: {name} (normalized: {normalized_query})")
        
        provinces = self.get_provinces()
        
        # Exact match first
        for province in provinces:
            if self._normalize_name(province['name']) == normalized_query:
                logger.info(f"Found exact province match: {province['name']}")
                return province
        
        # Check common aliases
        aliases = {
            'ncr': 'metro manila',
            'national capital region': 'metro manila',
            'compostela valley': 'davao de oro',
        }
        
        if normalized_query in aliases:
            alias_target = aliases[normalized_query]
            for province in provinces:
                if self._normalize_name(province['name']) == alias_target:
                    logger.info(f"Found province via alias: {province['name']}")
                    return province
        
        best_match = None
        best_score = 0.0
        
        for province in provinces:
            score = self._fuzzy_match(normalized_query, province['name'])
            if score > best_score and score >= 0.75:
                best_score = score
                best_match = province
        
        if best_match:
            logger.info(f"Found province via fuzzy match: {best_match['name']} (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"Province not found: {name}")
        return None
    
    def get_cities_municipalities(self, province_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch cities and municipalities from a province page or NCR.
        
        Args:
            province_url: URL of the province page (e.g., '/luzon/ncr.html')
            
        Returns:
            List of city/municipality dictionaries
        """
        if not province_url:
            return []
        
        url = f"{self.BASE_URL}{province_url}" if not province_url.startswith('http') else province_url
        logger.debug(f"Fetching cities/municipalities from: {url}")
        
        soup = self._fetch_page(url)
        if not soup:
            return []
        
        cities = []
        
        city_links = soup.find_all('a', href=re.compile(r'/[^/]+/[^/]+/[^/]+\.html'))
        
        for link in city_links:
            city_name = link.get_text(strip=True)
            city_url = link.get('href', '')
            
            if not city_name or not city_url:
                continue
            
            if city_url == province_url or city_url.count('/') < 3:
                continue
            
            cities.append({
                'name': city_name,
                'url': city_url,
                'code': self._normalize_name(city_name).upper().replace(' ', '_')
            })
        
        logger.debug(f"Found {len(cities)} cities/municipalities")
        return cities
    
    def search_city_municipality(self, name: str, province_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a city or municipality by name.
        
        Args:
            name: City/municipality name to search for
            province_name: Optional province name to narrow search
            
        Returns:
            Dictionary with city/municipality data or None if not found
        """
        if not name:
            return None
        
        normalized_query = self._normalize_name(name)
        logger.debug(f"Searching for city/municipality: {name} (province: {province_name})")
        
        province_data = None
        if province_name:
            province_data = self.search_province(province_name)
        
        if province_data:
            cities = self.get_cities_municipalities(province_data['url'])
            
            for city in cities:
                if self._normalize_name(city['name']) == normalized_query:
                    logger.info(f"Found exact city match: {city['name']}")
                    city['province'] = province_data['name']
                    return city
            
            best_match = None
            best_score = 0.0
            
            for city in cities:
                score = self._fuzzy_match(normalized_query, city['name'])
                if score > best_score and score >= 0.75:
                    best_score = score
                    best_match = city
            
            if best_match:
                logger.info(f"Found city via fuzzy match: {best_match['name']} (score: {best_score:.2f})")
                best_match['province'] = province_data['name']
                return best_match
        
        abbreviations = {
            'qc': 'quezon city',
            'q.c.': 'quezon city',
            'q.c': 'quezon city',
            'bgc': 'taguig',
        }
        
        if normalized_query in abbreviations:
            expanded_name = abbreviations[normalized_query]
            logger.info(f"Expanded abbreviation: {normalized_query} -> {expanded_name}")
            return self.search_city_municipality(expanded_name, province_name or 'Metro Manila')
        
        logger.warning(f"City/municipality not found: {name}")
        return None
    
    def get_barangays(self, city_url: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch barangays from a city page.

        Args:
            city_url: URL of the city page (e.g., '/luzon/ncr.html')

        Returns:
            List of barangay dictionaries
        """
        if not city_url:
            return []
        
        url = f"{self.BASE_URL}{city_url}" if not city_url.startswith('http') else city_url
        logger.debug(f"Fetching barangays from: {url}")
        
        soup = self._fetch_page(url)
        if not soup:
            return []
        
        barangays = []
        
        barangay_section = soup.find('h2', string=re.compile(r'Barangays', re.IGNORECASE))
        
        if barangay_section:
            table = barangay_section.find_next('table')
            if table:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if cells:
                        barangay_name = cells[0].get_text(strip=True)
                        if barangay_name and barangay_name not in ['Barangay', 'Name']:
                            barangays.append({
                                'name': barangay_name,
                                'code': self._normalize_name(barangay_name).upper().replace(' ', '_')
                            })
        
        logger.debug(f"Found {len(barangays)} barangays")
        return barangays
    
    def search_barangay(self, name: str, city_name: Optional[str] = None, 
                       province_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a barangay by name.
        
        Args:
            name: Barangay name to search for
            city_name: Optional city/municipality name to narrow search
            province_name: Optional province name to narrow search
            
        Returns:
            Dictionary with barangay data or None if not found
        """
        if not name:
            return None
        
        normalized_query = self._normalize_name(name)
        logger.debug(f"Searching for barangay: {name} (city: {city_name}, province: {province_name})")
        
        if not city_name:
            logger.warning("Cannot search barangay without city context")
            return None
        
        city_data = self.search_city_municipality(city_name, province_name)
        if not city_data:
            logger.warning(f"City not found, cannot search for barangay: {city_name}")
            return None
        
        barangays = self.get_barangays(city_data['url'])
        
        for barangay in barangays:
            if self._normalize_name(barangay['name']) == normalized_query:
                logger.info(f"Found exact barangay match: {barangay['name']}")
                barangay['city'] = city_data['name']
                barangay['province'] = city_data.get('province', province_name)
                return barangay
        
        best_match = None
        best_score = 0.0
        
        for barangay in barangays:
            score = self._fuzzy_match(normalized_query, barangay['name'])
            if score > best_score and score >= 0.75:
                best_score = score
                best_match = barangay
        
        if best_match:
            logger.info(f"Found barangay via fuzzy match: {best_match['name']} (score: {best_score:.2f})")
            best_match['city'] = city_data['name']
            best_match['province'] = city_data.get('province', province_name)
            return best_match
        
        logger.warning(f"Barangay not found: {name}")
        return None
