from pydantic import BaseModel
from typing import Optional, List, Dict


class AddressValidationRequest(BaseModel):
    address: str


class GeocodingData(BaseModel):
    """Geographic coordinates and location data"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    formatted_address: Optional[str] = None
    place_id: Optional[str] = None
    location_type: Optional[str] = None


class AddressValidationResponse(BaseModel):
    isValid: bool
    province: Optional[str] = None
    city: Optional[str] = None
    barangay: Optional[str] = None
    streetAddress: Optional[str] = None
    postalCode: Optional[str] = None
    formattedAddress: str = ""
    geocoding: Optional[GeocodingData] = None
    reasons: List[str] = []  # Reasons why address is invalid or geocoding failed


class ParsedAddress(BaseModel):
    province: Optional[str] = None
    city: Optional[str] = None
    barangay: Optional[str] = None
    street_address: Optional[str] = None
    postal_code: Optional[str] = None
