import googlemaps
from typing import Optional, Dict, Any
import logging
from utils.config import settings

logger = logging.getLogger(__name__)


class GeocodingClient:
    """
    Client for geocoding Philippine addresses using Google Maps API.
    
    This client fetches latitude, longitude, and other geographic data
    for validated addresses.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes a GeocodingClient instance.
        
        Args:
            api_key: Optional Google Maps API key. If not provided, uses settings.
        
        Raises:
            ValueError: If Google Maps API key is not provided.
        """
        self.api_key = api_key or settings.GOOGLE_MAPS_API_KEY
        
        if not self.api_key:
            logger.error("Google Maps API key is missing")
            raise ValueError("Google Maps API key is required. Set GOOGLE_API_KEY in .env file.")
        
        self.gmaps = googlemaps.Client(key=self.api_key)
        logger.info("GeocodingClient initialized successfully")
    
    def geocode_address(self, address: str, region: str = "PH") -> Optional[Dict[str, Any]]:
        """
        Geocode a Philippine address to get coordinates and geographic data.
        
        Args:
            address: The formatted address string to geocode
            region: Region code for biasing results (default: "PH" for Philippines)
            
        Returns:
            Dictionary containing geocoding results with:
                - latitude: float
                - longitude: float
                - formatted_address: str (Google's formatted version)
                - place_id: str
                - location_type: str (ROOFTOP, RANGE_INTERPOLATED, etc.)
                - viewport: dict with northeast and southwest bounds
                - error_reason: str (if geocoding fails)
            Returns None if geocoding fails.
        """
        if not address:
            logger.warning("Empty address provided for geocoding")
            return {'error_reason': 'Empty address provided'}
        
        try:
            logger.debug(f"Geocoding address: {address[:50]}...")
            
            geocode_result = self.gmaps.geocode(address, region=region)
            
            if not geocode_result or len(geocode_result) == 0:
                logger.warning(f"No geocoding results found for address: {address}")
                return {'error_reason': 'No geographic coordinates found for this address'}
            
            result = geocode_result[0]
            geometry = result.get('geometry', {})
            location = geometry.get('location', {})
            
            types = result.get('types', [])
            unsupported_types = [t for t in types if t in ['political', 'sublocality', 'sublocality_level_1', 'sublocality_level_2', 'postal_code']]
            
            geocoding_data = {
                'latitude': location.get('lat'),
                'longitude': location.get('lng'),
                'formatted_address': result.get('formatted_address'),
                'place_id': result.get('place_id'),
                'location_type': geometry.get('location_type'),
                'viewport': geometry.get('viewport'),
                'address_components': result.get('address_components', []),
                'types': types
            }
            
            if unsupported_types and len([t for t in types if t not in unsupported_types]) == 0:
                logger.warning(f"Geocoding result has unsupported region types: {unsupported_types}")
                geocoding_data['warning'] = f"Unsupported Region Types on Geocoding API: {', '.join(unsupported_types)}"
            
            logger.info(f"Geocoding successful: lat={geocoding_data['latitude']}, lng={geocoding_data['longitude']}")
            return geocoding_data
            
        except googlemaps.exceptions.ApiError as e:
            error_msg = f"Google Maps API error: {str(e)}"
            logger.error(f"Google Maps API error during geocoding: {e}")
            return {'error_reason': error_msg}
        except Exception as e:
            error_msg = f"Geocoding error: {str(e)}"
            logger.error(f"Error geocoding address: {e}", exc_info=True)
            return {'error_reason': error_msg}
    
    def reverse_geocode(self, latitude: float, longitude: float) -> Optional[Dict[str, Any]]:
        """
        Reverse geocode coordinates to get address information.
        
        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate
            
        Returns:
            Dictionary containing the formatted address and components,
            or None if reverse geocoding fails.
        """
        try:
            logger.debug(f"Reverse geocoding: lat={latitude}, lng={longitude}")
            
            reverse_result = self.gmaps.reverse_geocode((latitude, longitude))
            
            if not reverse_result or len(reverse_result) == 0:
                logger.warning(f"No reverse geocoding results found for coordinates")
                return None
            
            result = reverse_result[0]
            
            return {
                'formatted_address': result.get('formatted_address'),
                'place_id': result.get('place_id'),
                'address_components': result.get('address_components', [])
            }
            
        except Exception as e:
            logger.error(f"Error reverse geocoding: {e}", exc_info=True)
            return None
    
    def validate_address_with_usps(self, address: str, locality: str = None) -> Optional[Dict[str, Any]]:
        """
        Validate an address using Google's Address Validation API.
        Note: This is primarily for US addresses with USPS validation.
        
        Args:
            address: Address to validate
            locality: Optional locality/city name
            
        Returns:
            Address validation results or None if validation fails.
        """
        try:
            logger.debug(f"Validating address with USPS: {address}")
            
            validation_result = self.gmaps.addressvalidation(
                [address],
                regionCode='US',
                locality=locality,
                enableUspsCass=True
            )
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating address: {e}", exc_info=True)
            return None
    
    def get_place_details(self, place_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a place using its Place ID.
        
        Args:
            place_id: Google Maps Place ID
            
        Returns:
            Detailed place information or None if request fails.
        """
        try:
            logger.debug(f"Fetching place details for place_id: {place_id}")
            
            place_result = self.gmaps.place(place_id)
            
            if place_result.get('status') == 'OK':
                return place_result.get('result')
            else:
                logger.warning(f"Place details request failed: {place_result.get('status')}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching place details: {e}", exc_info=True)
            return None
