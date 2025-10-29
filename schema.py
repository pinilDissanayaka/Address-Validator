from pydantic import BaseModel, Field
from typing import Optional, List




class AddressValidationRequest(BaseModel):
    """Request model for address validation"""
    address: str = Field(..., description="Address text to validate", min_length=1)


class ParsedAddress(BaseModel):
    """Parsed address components from LLM"""
    province: Optional[str] = None
    city: Optional[str] = None
    barangay: Optional[str] = None
    street_address: Optional[str] = None
    postal_code: Optional[str] = None


class GeocodingData(BaseModel):
    """Geographic coordinates and location data"""
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    formatted_address: Optional[str] = None
    place_id: Optional[str] = None
    location_type: Optional[str] = None


class AddressValidationResponse(BaseModel):
    """Simple validation response for basic endpoint"""
    isValid: bool
    province: Optional[str] = None
    city: Optional[str] = None
    barangay: Optional[str] = None
    streetAddress: Optional[str] = None
    postalCode: Optional[str] = None
    formattedAddress: str = ""
    geocoding: Optional[GeocodingData] = None
    reasons: List[str] = []


class VerdictResponse(BaseModel):
    """Comprehensive validation verdict with multiple validation dimensions"""
    isValid: bool = Field(..., description="Overall validation result")
    structureOk: bool = Field(False, description="Address structure is complete and valid")
    psgcMatched: bool = Field(False, description="Address components match PSGC database")
    geocodeMatched: bool = Field(False, description="Address geocoding successful")
    deliveryHistorySuccess: bool = Field(False, description="Previous delivery was successful")
    confidence: float = Field(0, ge=0, le=100, description="Confidence score (0-100)")



class StructureResponse(BaseModel):
    """Structured address components"""
    streetAddress: str = Field("", description="Street address with building/unit number")
    barangay: str = Field("", description="Barangay name")
    city: str = Field("", description="City or municipality name")
    province: str = Field("", description="Province name")
    country: str = Field("PH", description="Country code")
    postalCode: str = Field("", description="Postal code")
    formattedAddress: str = Field("", description="Formatted complete address")


class PSGCResponse(BaseModel):
    """PSGC codes for administrative divisions"""
    regionCode: str = Field("", description="Region code")
    provinceCode: str = Field("", description="Province code")
    cityMuniCode: str = Field("", description="City/Municipality code")
    barangayCode: str = Field("", description="Barangay code")


class GeocodeResponse(BaseModel):
    """Geocoding information"""
    lat: float = Field(0, description="Latitude")
    lng: float = Field(0, description="Longitude")
    place_id: str = Field("", description="Google Maps Place ID")
    formattedAddress: str = Field("", description="Google's formatted address")


class DeliveryHistoryAddress(BaseModel):
    """Delivery history for a specific address"""
    address: str = Field("", description="Address used for delivery")
    status: str = Field("", description="Delivery status")
    last_delivery_at: str = Field("", description="Last delivery timestamp")
    rts_reason: str = Field("", description="Return to sender reason if applicable")


class DeliveryHistoryResponse(BaseModel):
    """Delivery history for both input and formatted addresses"""
    inputAddress: DeliveryHistoryAddress = Field(default_factory=DeliveryHistoryAddress)
    formattedAddress: DeliveryHistoryAddress = Field(default_factory=DeliveryHistoryAddress)


class EnhancedAddressValidationResponse(BaseModel):
    """
    Comprehensive address validation response.
    Integrates structure validation, database matching, geocoding, and delivery history.
    """
    id: str = Field(..., description="Unique validation request ID")
    input: str = Field(..., description="Original input address")
    formattedAddress: str = Field("", description="Formatted validated address")
    verdict: VerdictResponse = Field(..., description="Validation verdict")
    structure: StructureResponse = Field(default_factory=StructureResponse, description="Structured address components")
    psgc: PSGCResponse = Field(default_factory=PSGCResponse, description="PSGC codes")
    geocode: GeocodeResponse = Field(default_factory=GeocodeResponse, description="Geocoding data")
    deliveryHistory: DeliveryHistoryResponse = Field(default_factory=DeliveryHistoryResponse, description="Delivery history")
    reason: List[str] = Field(default_factory=list, description="Validation failure reasons")
    suggestions: List[str] = Field(default_factory=list, description="Suggestions for improvement")
