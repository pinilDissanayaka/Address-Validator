import googlemaps
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class GoogleMapsValidator:
    """Wrapper for Google Maps address validation and geocoding."""
    
    def __init__(self, api_key: str):
        """Initialize Google Maps client."""
        self.gmaps = googlemaps.Client(key=api_key)
        logger.info("GoogleMapsValidator initialized")
    
    def get_address_country(self, address: str) -> Dict[str, Any]:
        """
        Validate address and determine country using Google Address Validation API.
        If that fails, fallback to Geocoding API to detect country.
        Returns structured address if valid internationally, or indicates PH address.
        """
        try:
            _result = self.gmaps.addressvalidation(address)
            
            if "result" in _result.keys() and _result["result"]["verdict"]["possibleNextAction"] == "ACCEPT":
                postal_address = _result["result"]["address"]["postalAddress"]
                country_code = postal_address.get("regionCode", "")
                logger.info(f"Address Validation API detected country: {country_code}")
                return {
                    "isValid": True,
                    "country": country_code,
                    "streetAddress": ", ".join(postal_address.get("addressLines", [])),
                    "barangay": "",
                    "postalCode": postal_address.get("postalCode", ""),
                    "city": postal_address.get("locality", ""),
                    "province": postal_address.get("administrativeArea", ""),
                    "formattedAddress": _result["result"]["address"].get("formattedAddress", ""),
                    "latitude": _result["result"]["geocode"]["location"].get("latitude", 0),
                    "longitude": _result["result"]["geocode"]["location"].get("longitude", 0),
                }
            elif "error" in _result.keys() and '"PH".' in _result["error"]["message"]:
                logger.info("Address Validation API detected PH address (error message)")
                return {
                    "isValid": "PH",
                    "country": "PH",
                }
            elif "error" in _result.keys() and "Unsupported region" in _result["error"]["message"]:
                country = _result["error"]["message"].split(":")[1].split(".")[0].replace('"', '').strip()
                logger.info(f"Address Validation API detected unsupported region: {country}")
                return {
                    "isValid": "NOT_PH",
                    "country": country,
                }
            elif "result" in _result.keys():
                logger.info("Address Validation API returned result but not ACCEPT - treating as NOT_PH")
                return {
                    "isValid": "NOT_PH"
                }
            else:
                logger.warning("Address Validation API returned unexpected response - falling back to Geocoding")
                # Fallback to geocoding to detect country
                return self._detect_country_via_geocoding(address)
        except Exception as e:
            logger.error(f"Error in address validation API: {e} - falling back to Geocoding")
            # Fallback to geocoding API
            return self._detect_country_via_geocoding(address)
    
    def _detect_country_via_geocoding(self, address: str) -> Dict[str, Any]:
        """
        Fallback method: Use Geocoding API to detect country.
        """
        try:
            logger.info("Using Geocoding API to detect country")
            _result = self.gmaps.geocode(address)
            
            if _result and len(_result) > 0:
                # Get country from first result
                for component in _result[0].get("address_components", []):
                    if "country" in component.get("types", []):
                        country_code = component.get("short_name", "")
                        logger.info(f"Geocoding API detected country: {country_code}")
                        
                        if country_code == "PH":
                            return {
                                "isValid": "PH",
                                "country": "PH",
                            }
                        else:
                            return {
                                "isValid": "NOT_PH",
                                "country": country_code,
                            }
            
            logger.warning("Could not detect country - defaulting to PH")
            return {
                "isValid": "PH",
                "country": "PH",
            }
        except Exception as e:
            logger.error(f"Error in geocoding fallback: {e} - defaulting to PH")
            return {
                "isValid": "PH",
                "country": "PH",
            }
    
    def get_geocode(self, address: str, region: str = "PH") -> Dict[str, Any]:
        """
        Geocode an address to get coordinates and location details.
        Returns geocoding results or error information.
        """
        try:
            _result = self.gmaps.geocode(address, region=region)
            
            for __r in _result:
                if "types" in __r.keys() and any(t in ["premise", "subpremise", "establishment"] for t in __r["types"]):
                    _premise, _street_number, _street_name = "", "", ""
                    _city, _province, _country, _postalCode = "", "", "", ""
                    
                    for k in __r["address_components"]:
                        if any(t in ["premise", "subpremise", "establishment"] for t in k["types"]):
                            _premise = k["long_name"]
                        elif any(t in ["street_number"] for t in k["types"]):
                            _street_number = k["long_name"]
                        elif any(t in ["route"] for t in k["types"]):
                            _street_name = k["long_name"]
                        elif any(t in ["locality"] for t in k["types"]):
                            _city = k["long_name"]
                        elif any(t in ["administrative_area_level_1", "administrative_area_level_2"] for t in k["types"]) and _province == "":
                            _province = k["short_name"]
                        elif any(t in ["country"] for t in k["types"]):
                            _country = k["short_name"]
                        elif any(t in ["postal_code"] for t in k["types"]):
                            _postalCode = k["long_name"]
                    
                    return {
                        "isValid": True,
                        "streetAddress": ((_premise or "") + " " + (_street_number or "") + " " + (_street_name or "")).strip(),
                        "barangay": "",
                        "postalCode": _postalCode or "",
                        "city": _city or "",
                        "province": _province or "",
                        "country": _country or "",
                        "formattedAddress": __r["formatted_address"] or "",
                        "latitude": __r["geometry"]["location"]["lat"] or 0,
                        "longitude": __r["geometry"]["location"]["lng"] or 0,
                        "place_id": __r["place_id"] or "",
                    }
            
            _location_types = ""
            for __r in _result:
                _location_types += ", ".join(__r["types"])
            
            logger.error(f"Unsupported location types: {_location_types}")
            return {
                "isValid": False,
                "reason": f"Unsupported Region Types on Geocoding API: {_location_types}",
            }
            
        except Exception as e:
            logger.exception(f"Error geocoding address: {e}")
            return {
                "isValid": False,
                "reason": f"Unknown Error on Geocoding API: {str(e)}"
            }
