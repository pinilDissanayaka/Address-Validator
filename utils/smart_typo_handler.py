"""

Smart Typo Handler - Dynamic fuzzy matching without hardcoded corrections.
Uses RapidFuzz for fast fuzzy string matching and phonetic algorithms.
"""

import logging
from typing import Optional, List, Tuple, Dict
from functools import lru_cache

logger = logging.getLogger(__name__)

# Try to import rapidfuzz, fall back to difflib if not available
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
    logger.info("Using RapidFuzz for high-performance fuzzy matching")
except ImportError:
    from difflib import SequenceMatcher
    RAPIDFUZZ_AVAILABLE = False
    logger.info("RapidFuzz not available, using difflib (slower)")

# Try to import phonetics library for sound-alike matching
try:
    import phonetics
    PHONETICS_AVAILABLE = True
    logger.info("Phonetics library available for sound-alike matching")
except ImportError:
    PHONETICS_AVAILABLE = False
    logger.info("Phonetics library not available (optional - pip install phonetics)")


class SmartTypoHandler:
    """
    Intelligent typo correction using fuzzy matching algorithms.
    No hardcoded dictionaries - learns from the official database.
    """
    
    def __init__(self, min_score: int = 85, phonetic_enabled: bool = True):
        """
        Initialize SmartTypoHandler.
        
        Args:
            min_score: Minimum similarity score (0-100) for fuzzy matching
            phonetic_enabled: Enable phonetic (sound-alike) matching
        """
        self.min_score = min_score
        self.phonetic_enabled = phonetic_enabled and PHONETICS_AVAILABLE
        
        # Cache for API data
        self._province_cache: Optional[List[str]] = None
        self._city_cache: Optional[List[str]] = None
        self._barangay_cache: Dict[str, List[str]] = {}
    
    def find_best_match(
        self, 
        query: str, 
        candidates: List[str],
        threshold: int = None
    ) -> Optional[Tuple[str, float]]:
        """
        Find the best match for a query string from a list of candidates.
        
        Args:
            query: The misspelled string to match
            candidates: List of correct strings to match against
            threshold: Minimum score to consider (uses self.min_score if None)
            
        Returns:
            Tuple of (best_match, score) or None if no good match found
        """
        if not query or not candidates:
            return None
        
        threshold = threshold or self.min_score
        
        if RAPIDFUZZ_AVAILABLE:
            # RapidFuzz is much faster for large candidate lists
            result = process.extractOne(
                query,
                candidates,
                scorer=fuzz.WRatio,  # Weighted ratio - handles different lengths well
                score_cutoff=threshold
            )
            
            if result:
                match, score, _ = result
                logger.info(f"RapidFuzz match: '{query}' -> '{match}' (score: {score:.1f})")
                return (match, score)
        else:
            # Fallback to difflib
            best_match = None
            best_score = 0.0
            
            for candidate in candidates:
                ratio = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
                score = ratio * 100
                
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = candidate
            
            if best_match:
                logger.info(f"Difflib match: '{query}' -> '{best_match}' (score: {best_score:.1f})")
                return (best_match, best_score)
        
        return None
    
    def find_phonetic_match(
        self,
        query: str,
        candidates: List[str]
    ) -> Optional[Tuple[str, float]]:
        """
        Find matches based on phonetic similarity (sound-alike).
        Useful for names like "Cordoba" vs "Cordova".
        
        Args:
            query: The query string
            candidates: List of candidates to match
            
        Returns:
            Tuple of (best_match, confidence) or None
        """
        if not self.phonetic_enabled or not candidates:
            return None
        
        try:
            # Generate phonetic codes for query
            query_metaphone = phonetics.metaphone(query.lower())
            query_soundex = phonetics.soundex(query.lower())
            
            matches = []
            
            for candidate in candidates:
                candidate_lower = candidate.lower()
                candidate_metaphone = phonetics.metaphone(candidate_lower)
                candidate_soundex = phonetics.soundex(candidate_lower)
                
                # Calculate phonetic similarity
                phonetic_score = 0
                
                if query_metaphone == candidate_metaphone:
                    phonetic_score += 50
                if query_soundex == candidate_soundex:
                    phonetic_score += 50
                
                if phonetic_score > 0:
                    # Add string similarity for tiebreaking
                    if RAPIDFUZZ_AVAILABLE:
                        string_score = fuzz.ratio(query, candidate)
                    else:
                        string_score = SequenceMatcher(None, query, candidate).ratio() * 100
                    
                    # Weighted average: 60% phonetic, 40% string similarity
                    final_score = (phonetic_score * 0.6) + (string_score * 0.4)
                    matches.append((candidate, final_score))
            
            if matches:
                # Sort by score and return best
                matches.sort(key=lambda x: x[1], reverse=True)
                best_match, score = matches[0]
                
                if score >= self.min_score:
                    logger.info(f"Phonetic match: '{query}' -> '{best_match}' (score: {score:.1f})")
                    return (best_match, score)
        
        except Exception as e:
            logger.debug(f"Phonetic matching error: {e}")
        
        return None
    
    def correct_province(
        self,
        province_name: str,
        psgc_client
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Correct province name typos using official PSGC data.
        
        Args:
            province_name: Potentially misspelled province name
            psgc_client: PSGCAPIClient instance
            
        Returns:
            Tuple of (corrected_name, confidence_score)
        """
        if not province_name:
            return None, None
        
        # Get all official province names
        if not self._province_cache:
            provinces = psgc_client.get_all_provinces()
            self._province_cache = [p['name'] for p in provinces]
        
        # Try exact match first (case-insensitive)
        for province in self._province_cache:
            if province.lower() == province_name.lower():
                return province, 100.0
        
        # Try phonetic match
        if self.phonetic_enabled:
            phonetic_result = self.find_phonetic_match(province_name, self._province_cache)
            if phonetic_result:
                return phonetic_result
        
        # Try fuzzy match
        fuzzy_result = self.find_best_match(province_name, self._province_cache)
        if fuzzy_result:
            return fuzzy_result
        
        return None, None
    
    def correct_city(
        self,
        city_name: str,
        psgc_client,
        province_code: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Correct city/municipality name typos using official PSGC data.
        
        Args:
            city_name: Potentially misspelled city name
            psgc_client: PSGCAPIClient instance
            province_code: Optional province code to narrow search
            
        Returns:
            Tuple of (corrected_name, confidence_score)
        """
        if not city_name:
            return None, None
        
        # Get all official city names
        if not self._city_cache:
            cities = psgc_client.get_all_cities()
            self._city_cache = [c['name'] for c in cities]
        
        candidates = self._city_cache
        
        # Filter by province if provided
        if province_code:
            cities = psgc_client.get_all_cities()
            candidates = [
                c['name'] for c in cities 
                if c['code'].startswith(province_code[:4])
            ]
        
        # Try exact match first
        for city in candidates:
            if city.lower() == city_name.lower():
                return city, 100.0
        
        # Try phonetic match
        if self.phonetic_enabled:
            phonetic_result = self.find_phonetic_match(city_name, candidates)
            if phonetic_result:
                return phonetic_result
        
        # Try fuzzy match
        fuzzy_result = self.find_best_match(city_name, candidates)
        if fuzzy_result:
            return fuzzy_result
        
        return None, None
    
    def correct_barangay(
        self,
        barangay_name: str,
        psgc_client,
        city_code: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[float]]:
        """
        Correct barangay name typos using official PSGC data.
        
        Args:
            barangay_name: Potentially misspelled barangay name
            psgc_client: PSGCAPIClient instance
            city_code: Optional city code to narrow search
            
        Returns:
            Tuple of (corrected_name, confidence_score)
        """
        if not barangay_name or not city_code:
            return None, None
        
        # Get barangays for this city
        cache_key = city_code
        if cache_key not in self._barangay_cache:
            try:
                url = f"{psgc_client.BASE_URL}/cities-municipalities/{city_code}/barangays"
                response = psgc_client.session.get(url, timeout=psgc_client.timeout)
                if response.status_code == 200:
                    result = response.json()
                    barangays = result.get('data', [])
                    self._barangay_cache[cache_key] = [b['name'] for b in barangays]
                else:
                    return None, None
            except Exception as e:
                logger.debug(f"Error fetching barangays: {e}")
                return None, None
        
        candidates = self._barangay_cache.get(cache_key, [])
        
        if not candidates:
            return None, None
        
        # Try exact match first
        for barangay in candidates:
            if barangay.lower() == barangay_name.lower():
                return barangay, 100.0
        
        # Try phonetic match
        if self.phonetic_enabled:
            phonetic_result = self.find_phonetic_match(barangay_name, candidates)
            if phonetic_result:
                return phonetic_result
        
        # Try fuzzy match
        fuzzy_result = self.find_best_match(barangay_name, candidates)
        if fuzzy_result:
            return fuzzy_result
        
        return None, None
    
    def apply_corrections(self, parsed_address, psgc_client):
        """
        Apply smart typo corrections to a parsed address.
        
        Args:
            parsed_address: ParsedAddress object
            psgc_client: PSGCAPIClient instance
            
        Returns:
            Modified parsed_address with corrections applied and a list of suggestions
        """
        corrections_made = []
        suggestions = []
        
        # Correct province
        if hasattr(parsed_address, 'province') and parsed_address.province:
            corrected, score = self.correct_province(parsed_address.province, psgc_client)
            if corrected and corrected != parsed_address.province:
                corrections_made.append(f"Province: '{parsed_address.province}' -> '{corrected}' (confidence: {score:.0f}%)")
                parsed_address.province = corrected
        
        # Correct city (with province context if available)
        if hasattr(parsed_address, 'city') and parsed_address.city:
            province_code = None
            if hasattr(parsed_address, 'province') and parsed_address.province:
                provinces = psgc_client.get_all_provinces()
                for p in provinces:
                    if p['name'].lower() == parsed_address.province.lower():
                        province_code = p['code']
                        break
            
            original_city = parsed_address.city
            corrected, score = self.correct_city(parsed_address.city, psgc_client, province_code)
            if corrected and corrected != parsed_address.city:
                corrections_made.append(f"City: '{parsed_address.city}' -> '{corrected}' (confidence: {score:.0f}%)")
                
                # Add suggestion if the city name was significantly shortened
                if len(original_city) > len(corrected) + 5:
                    suggestions.append(
                        f"City name standardized from '{original_city}' to official name '{corrected}'"
                    )
                
                parsed_address.city = corrected
        
        # Correct barangay (with city context if available)
        if hasattr(parsed_address, 'barangay') and parsed_address.barangay:
            city_code = None
            if hasattr(parsed_address, 'city') and parsed_address.city:
                # First try exact match
                cities = psgc_client.get_all_cities()
                for c in cities:
                    if c['name'].lower() == parsed_address.city.lower():
                        city_code = c['code']
                        break
                
                # If no exact match, use PSGC search which handles variations
                if not city_code:
                    # Get province code if available
                    province_code = None
                    if hasattr(parsed_address, 'province') and parsed_address.province:
                        provinces = psgc_client.get_all_provinces()
                        for p in provinces:
                            if p['name'].lower() == parsed_address.province.lower():
                                province_code = p['code']
                                break
                    
                    # Use search_city_municipality which handles name variations
                    city_result = psgc_client.search_city_municipality(parsed_address.city, province_code)
                    if city_result:
                        city_code = city_result.get('code')
                        logger.debug(f"Found city code using search: {parsed_address.city} -> {city_code}")
            
            if city_code:
                corrected, score = self.correct_barangay(parsed_address.barangay, psgc_client, city_code)
                if corrected and corrected != parsed_address.barangay:
                    corrections_made.append(f"Barangay: '{parsed_address.barangay}' -> '{corrected}' (confidence: {score:.0f}%)")
                    parsed_address.barangay = corrected
        
        if corrections_made:
            logger.info(f"Smart typo corrections applied: {'; '.join(corrections_made)}")
        
        # Return both the parsed address and suggestions
        return parsed_address, suggestions
