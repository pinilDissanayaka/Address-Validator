from fastapi import APIRouter, HTTPException
import logging
from schema import AddressValidationRequest, AddressValidationResponse
from utils import AddressParser, PhilAtlasClient, GeocodingClient, AddressValidator

logger = logging.getLogger(__name__)

validate_router=APIRouter(
    prefix="/validator",
    tags=["validator"]
)

parser: AddressParser=None
philatlas_client: PhilAtlasClient = None
geocoding_client: GeocodingClient = None
validator: AddressValidator = None

@validate_router.on_event("startup")
async def startup_event():
    global parser, philatlas_client, geocoding_client, validator
    logger.info("Initializing validator components...")
    parser = AddressParser()
    
    # Initialize PhilAtlas client
    from utils.config import settings
    logger.info("Initializing PhilAtlas client...")
    philatlas_client = PhilAtlasClient(timeout=settings.PHILATLAS_TIMEOUT)
    logger.info("PhilAtlas client initialized")
    
    # Initialize Geocoding client
    try:
        logger.info("Initializing Geocoding client...")
        geocoding_client = GeocodingClient()
        logger.info("Geocoding client initialized")
    except ValueError as e:
        logger.warning(f"Geocoding client not initialized: {e}")
        geocoding_client = None
    
    validator = AddressValidator(parser, philatlas_client, geocoding_client)
    logger.info("Validator components initialized successfully")

    

@validate_router.post("/address", response_model=AddressValidationResponse)
async def validate_address(request: AddressValidationRequest):
    """
    Validate a Philippine address
    
    Accepts unstructured address text and returns structured, validated components.
    
    - **address**: Raw address text (e.g., "Unit 405, 23rd St., Barangay Libis, Quezon City, Metro Manila")
    
    Returns structured address with validation status.
    """
    try:
        if not request.address or not request.address.strip():
            logger.warning("Empty address validation attempt")
            raise HTTPException(status_code=400, detail="Address cannot be empty")
        
        logger.info(f"Validating address: {request.address[:50]}...")
        global validator
        result = await validator.validate_address(request.address)
        logger.info(f"Address validation result: isValid={result.isValid}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error validating address: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")