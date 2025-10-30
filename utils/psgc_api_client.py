import re
import requests
import logging
from typing import Optional, Dict, List, Any
from functools import lru_cache
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class PSGCAPIClient:
    """
    Client for accessing Philippine Standard Geographic Code (PSGC) data via psgc.cloud API v2.
    
    This client provides methods to validate and search provinces, cities, municipalities, 
    and barangays using the official PSGC API v2.
    """
    
    BASE_URL = "https://psgc.cloud/api/v2"
    
    def __init__(self, timeout: float = 10.0):
        """
        Initialize PSGC API Client.
        
        Args:
            timeout: Timeout in seconds for API requests. Defaults to 10.0.
        """
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        logger.info(f"PSGCAPIClient initialized with timeout={timeout}s")
    
    def _fuzzy_match(self, query: str, target: str, threshold: float = 0.75) -> float:
        """
        Calculate fuzzy match ratio between query and target strings using multiple algorithms.
        
        Args:
            query: Query string
            target: Target string
            threshold: Minimum match ratio
            
        Returns:
            Match ratio (0.0 to 1.0)
        """
        query_normalized = self._normalize_for_matching(query)
        target_normalized = self._normalize_for_matching(target)
        
        # Use SequenceMatcher as primary
        ratio = SequenceMatcher(None, query_normalized, target_normalized).ratio()
        
        # Boost score for substring matches
        if query_normalized in target_normalized or target_normalized in query_normalized:
            ratio = max(ratio, 0.85)
        
        # Levenshtein-based scoring for typos
        lev_distance = self._levenshtein_distance(query_normalized, target_normalized)
        max_len = max(len(query_normalized), len(target_normalized))
        if max_len > 0:
            lev_ratio = 1.0 - (lev_distance / max_len)
            # Take the better of the two ratios
            ratio = max(ratio, lev_ratio)
        
        return ratio
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate Levenshtein distance between two strings (edit distance).
        This helps catch typos better than SequenceMatcher.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            Edit distance (number of operations needed to transform s1 to s2)
        """
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # Cost of insertions, deletions, or substitutions
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _normalize_for_matching(self, text: str) -> str:
        """
        Enhanced normalization for matching with typo tolerance.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        normalized = text.lower().strip()
        
        # Common character substitutions (typos/variations)
        replacements = {
            'ñ': 'n',
            'á': 'a', 'à': 'a', 'â': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'ö': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            # Common double letter typos
            'aa': 'a',
            'ee': 'e',
            'ii': 'i',
            'oo': 'o',
            'uu': 'u',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Remove punctuation and extra spaces
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized.strip()
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize location name by removing common prefixes.
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
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
    
    @lru_cache(maxsize=1)
    def get_all_regions(self) -> List[Dict[str, Any]]:
        """
        Fetch all regions from PSGC API v2.
        
        Returns:
            List of region dictionaries with 'code', 'name' fields
        """
        try:
            url = f"{self.BASE_URL}/regions"
            logger.debug(f"Fetching regions from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            data = result.get('data', [])
            logger.info(f"Fetched {len(data)} regions from PSGC API v2")
            return data
        except Exception as e:
            logger.error(f"Error fetching regions: {e}")
            return []
    
    @lru_cache(maxsize=1)
    def get_all_provinces(self) -> List[Dict[str, Any]]:
        """
        Fetch all provinces from PSGC API v2.
        
        Returns:
            List of province dictionaries with 'code', 'name', 'region' fields
        """
        try:
            url = f"{self.BASE_URL}/provinces"
            logger.debug(f"Fetching provinces from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            data = result.get('data', [])
            logger.info(f"Fetched {len(data)} provinces from PSGC API v2")
            return data
        except Exception as e:
            logger.error(f"Error fetching provinces: {e}")
            return []
    
    @lru_cache(maxsize=1)
    def get_all_cities(self) -> List[Dict[str, Any]]:
        """
        Fetch all cities and municipalities from PSGC API v2.
        
        Returns:
            List of city/municipality dictionaries with 'code', 'name', 'type', 'district', 'zip_code', 'region', 'province' fields
        """
        try:
            url = f"{self.BASE_URL}/cities-municipalities"
            logger.debug(f"Fetching cities/municipalities from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            data = result.get('data', [])
            logger.info(f"Fetched {len(data)} cities/municipalities from PSGC API v2")
            return data
        except Exception as e:
            logger.error(f"Error fetching cities: {e}")
            return []
    
    @lru_cache(maxsize=1)
    def get_all_municipalities(self) -> List[Dict[str, Any]]:
        """
        Fetch all municipalities from PSGC API v2.
        Note: This is now redundant as v2 combines cities and municipalities in one endpoint.
        
        Returns:
            List of municipality dictionaries with 'code', 'name', 'type', 'district', 'zip_code', 'region', 'province' fields
        """
        try:
            url = f"{self.BASE_URL}/cities-municipalities"
            logger.debug(f"Fetching municipalities from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            data = result.get('data', [])
            municipalities = [item for item in data if item.get('type') == 'Mun']
            logger.info(f"Fetched {len(municipalities)} municipalities from PSGC API v2")
            return municipalities
        except Exception as e:
            logger.error(f"Error fetching municipalities: {e}")
            return []
    
    @lru_cache(maxsize=1)
    def get_all_barangays(self) -> List[Dict[str, Any]]:
        """
        Fetch all barangays from PSGC API v2.
        
        NOTE: This endpoint has pagination limitations and may not return all barangays.
        Use search_barangay() with direct API lookup for better results.
        
        Returns:
            List of barangay dictionaries with 'code', 'name', 'status', 'region', 'province', 'city_municipality', 'zip_code', 'district' fields
        """
        try:
            url = f"{self.BASE_URL}/barangays"
            logger.debug(f"Fetching barangays from: {url}")
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            result = response.json()
            data = result.get('data', [])
            logger.warning(f"Fetched only {len(data)} barangays from PSGC API v2 (API has pagination limits)")
            return data
        except Exception as e:
            logger.error(f"Error fetching barangays: {e}")
            return []
    
    def search_province(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Search for a province by name.
        
        Args:
            name: Province name to search
            
        Returns:
            Province dictionary or None if not found
        """
        if not name:
            return None
        
        normalized_query = self._normalize_name(name)
        logger.debug(f"Searching for province: {name} (normalized: {normalized_query})")
        
        provinces = self.get_all_provinces()
        
        for province in provinces:
            if self._normalize_name(province['name']) == normalized_query:
                logger.info(f"Found exact province match: {province['name']}")
                return province
        
        best_match = None
        best_score = 0.0
        
        for province in provinces:
            score = self._fuzzy_match(normalized_query, province['name'])
            if score > best_score and score >= 0.70:  # Lowered from 0.75 for better typo tolerance
                best_score = score
                best_match = province
        
        if best_match:
            logger.info(f"Found province via fuzzy match: {best_match['name']} (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"Province not found: {name}")
        return None
    
    def _get_city_name_variations(self, name: str) -> List[str]:
        """
        Generate common variations of city names to improve matching.
        
        Args:
            name: City name
            
        Returns:
            List of name variations to try
        """
        variations = [name.lower().strip()]
        base_name = self._normalize_name(name)
        
        # Add base normalized name
        variations.append(base_name)
        
        # Handle "X City" <-> "City of X" conversions
        if " city" in base_name:
            # "cebu city" -> "cebu"
            clean_name = base_name.replace(" city", "").strip()
            variations.append(clean_name)
            # "cebu city" -> "city of cebu"
            variations.append(f"city of {clean_name}")
        elif not base_name.startswith("city of"):
            # "cebu" -> "city of cebu", "cebu city"
            variations.append(f"city of {base_name}")
            variations.append(f"{base_name} city")
        
        # Handle "X Municipality" <-> "Municipality of X"
        if " municipality" in base_name:
            clean_name = base_name.replace(" municipality", "").strip()
            variations.append(clean_name)
            variations.append(f"municipality of {clean_name}")
        
        return list(set(variations))  # Remove duplicates
    
    def search_city_municipality(self, name: str, province_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a city or municipality by name using PSGC API v2.
        
        Args:
            name: City/municipality name
            province_code: Optional province code to narrow search
            
        Returns:
            City/municipality dictionary or None if not found
        """
        if not name:
            return None
        
        normalized_query = self._normalize_name(name)
        name_variations = self._get_city_name_variations(name)
        logger.debug(f"Searching for city/municipality: {name} (variations: {name_variations[:3]}...)")
        
        all_locations = self.get_all_cities()
        
        # If province_code provided, try province-filtered search first
        if province_code:
            filtered_locations = [loc for loc in all_locations if loc['code'].startswith(province_code[:4])]
            
            # Exact match in province (try all name variations)
            for location in filtered_locations:
                normalized_location = self._normalize_name(location['name'])
                for variation in name_variations:
                    if normalized_location == variation:
                        logger.info(f"Found exact city/municipality match in province: {location['name']}")
                        return location
            
            # Substring match in province
            for location in filtered_locations:
                normalized_location = self._normalize_name(location['name'])
                if normalized_query in normalized_location or normalized_location in normalized_query:
                    logger.info(f"Found city/municipality via substring match in province: {location['name']}")
                    return location
            
            # Fuzzy match in province
            best_match = None
            best_score = 0.0
            
            for location in filtered_locations:
                score = self._fuzzy_match(normalized_query, location['name'])
                if score > best_score and score >= 0.70:  # Lowered from 0.75 for better typo tolerance
                    best_score = score
                    best_match = location
            
            if best_match:
                logger.info(f"Found city/municipality via fuzzy match in province: {best_match['name']} (score: {best_score:.2f})")
                return best_match
            
            # If not found in province, fall back to nationwide search
            # This handles independent cities (HUCs) like Cagayan De Oro
            logger.debug(f"City not found in province, trying nationwide search for: {name}")
        
        # Nationwide search (or initial search if no province_code)
        # Exact match (try all name variations)
        for location in all_locations:
            normalized_location = self._normalize_name(location['name'])
            for variation in name_variations:
                if normalized_location == variation:
                    logger.info(f"Found exact city/municipality match: {location['name']}")
                    return location
        
        # Substring match (but only if query is longer to avoid false positives like "Anda" matching "Mandaue")
        for location in all_locations:
            normalized_location = self._normalize_name(location['name'])
            # Only use substring if the query is at least 5 chars and significantly present
            if len(normalized_query) >= 5:
                if normalized_query in normalized_location:
                    logger.info(f"Found city/municipality via substring match (query in location): {location['name']}")
                    return location
                # Be more strict when location name is in query - must be >50% of query length
                if normalized_location in normalized_query and len(normalized_location) > len(normalized_query) * 0.5:
                    logger.info(f"Found city/municipality via substring match (location in query): {location['name']}")
                    return location
        
        # Fuzzy match
        best_match = None
        best_score = 0.0
        
        for location in all_locations:
            score = self._fuzzy_match(normalized_query, location['name'])
            if score > best_score and score >= 0.70:  # Lowered from 0.75 for better typo tolerance
                best_score = score
                best_match = location
        
        if best_match:
            logger.info(f"Found city/municipality via fuzzy match: {best_match['name']} (score: {best_score:.2f})")
            return best_match
        
        logger.warning(f"City/municipality not found: {name}")
        return None
    
    def search_barangay(self, name: str, city_municipality_code: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a barangay by name using PSGC API v2.
        
        Args:
            name: Barangay name
            city_municipality_code: Optional city/municipality code to narrow search
            
        Returns:
            Barangay dictionary or None if not found
        """
        if not name:
            return None
        
        normalized_query = self._normalize_name(name)
        logger.debug(f"Searching for barangay: {name} (normalized: {normalized_query})")
        
        # If city code provided, fetch barangays from that specific city
        if city_municipality_code:
            try:
                # Use city-specific barangay endpoint
                url = f"{self.BASE_URL}/cities-municipalities/{city_municipality_code}/barangays"
                logger.debug(f"Fetching barangays from city: {url}")
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    city_barangays = result.get('data', [])
                    logger.debug(f"Fetched {len(city_barangays)} barangays for city {city_municipality_code}")
                    
                    # Exact match
                    for barangay in city_barangays:
                        if self._normalize_name(barangay['name']) == normalized_query:
                            logger.info(f"Found exact barangay match in city: {barangay['name']}")
                            return barangay
                    
                    # Fuzzy match
                    best_match = None
                    best_score = 0.0
                    for barangay in city_barangays:
                        score = self._fuzzy_match(normalized_query, barangay['name'])
                        if score > best_score and score >= 0.70:  # Lowered from 0.75 for better typo tolerance
                            best_score = score
                            best_match = barangay
                    
                    if best_match:
                        logger.info(f"Found barangay via fuzzy match in city: {best_match['name']} (score: {best_score:.2f})")
                        return best_match
                        
            except Exception as e:
                logger.debug(f"City-specific barangay lookup failed: {e}")
        
        # Fallback: Try direct barangay name lookup (without city filter)
        try:
            url = f"{self.BASE_URL}/barangays/{name}"
            logger.debug(f"Trying direct barangay lookup: {url}")
            response = self.session.get(url, timeout=self.timeout)
            if response.status_code == 200:
                result = response.json()
                barangay_data = result.get('data')
                if barangay_data:
                    # If city code provided, verify it matches
                    if city_municipality_code:
                        if barangay_data['code'].startswith(city_municipality_code[:6]):
                            logger.info(f"Found barangay via direct API lookup: {barangay_data['name']}")
                            return barangay_data
                    else:
                        logger.info(f"Found barangay via direct API lookup: {barangay_data['name']}")
                        return barangay_data
        except Exception as e:
            logger.debug(f"Direct barangay lookup failed: {e}")
        
        # Try compound name variations if city provided
        if city_municipality_code:
            variations = [name]
            if '-' not in name:
                for prefix in ['Funda-', 'San-', 'Santa-', 'Santo-']:
                    variations.append(f"{prefix}{name}")
            
            for variation in variations:
                try:
                    url = f"{self.BASE_URL}/barangays/{variation}"
                    logger.debug(f"Trying barangay variation: {url}")
                    response = self.session.get(url, timeout=self.timeout)
                    if response.status_code == 200:
                        result = response.json()
                        barangay_data = result.get('data')
                        if barangay_data and barangay_data['code'].startswith(city_municipality_code[:6]):
                            logger.info(f"Found barangay via compound name: {barangay_data['name']}")
                            return barangay_data
                except Exception as e:
                    logger.debug(f"Variation lookup failed for {variation}: {e}")
        
        logger.warning(f"Barangay not found: {name}")
        return None
    
    def get_postal_code(self, city_municipality_code: str) -> Optional[str]:
        """
        Get postal code for a city or municipality using PSGC API v2.
        
        Args:
            city_municipality_code: City/municipality code
            
        Returns:
            Postal code or None if not found
        """
        try:
            all_locations = self.get_all_cities()
            for location in all_locations:
                if location['code'] == city_municipality_code and location.get('zip_code'):
                    return location['zip_code']
            
            return None
        except Exception as e:
            logger.error(f"Error getting postal code: {e}")
            return None
