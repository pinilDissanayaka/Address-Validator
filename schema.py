from pydantic import BaseModel
from typing import Optional, List, Dict


class AddressValidationRequest(BaseModel):
    address: str


class AddressValidationResponse(BaseModel):
    isValid: bool
    province: Optional[str] = None
    city: Optional[str] = None
    barangay: Optional[str] = None
    streetAddress: Optional[str] = None
    postalCode: Optional[str] = None
    formattedAddress: str = ""


class ParsedAddress(BaseModel):
    province: Optional[str] = None
    city: Optional[str] = None
    barangay: Optional[str] = None
    street_address: Optional[str] = None
    postal_code: Optional[str] = None
