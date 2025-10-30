from fastapi import APIRouter, HTTPException
import logging
from schema import AddressValidationRequest, EnhancedAddressValidationResponse
from utils import AddressParser, PhilAtlasClient
from utils.validator import AddressValidator
from utils.psgc_api_client import PSGCAPIClient
from utils.config import settings

logger = logging.getLogger(__name__)

validator_router = APIRouter(
    prefix="/api/v1",
    tags=["Validation"]
)

address_validator: AddressValidator = None


@validator_router.on_event("startup")
async def startup_event():
    """Initialize validator components on startup."""
    global enhanced_validator
    logger.info("Initializing validator components...")
    
    try:
        parser = AddressParser()
        logger.info("Address parser initialized")
        
        psgc_client = PSGCAPIClient()
        logger.info("PSGC API client initialized")
        
        philatlas_client = PhilAtlasClient(timeout=settings.PHILATLAS_TIMEOUT)
        logger.info("PhilAtlas client initialized")
        
        gmaps_api_key = settings.GOOGLE_MAPS_API_KEY
        if not gmaps_api_key:
            logger.warning("Google Maps API key not found - geocoding will be limited")
        
        enhanced_validator = AddressValidator(
            parser=parser,
            psgc_client=psgc_client,
            philatlas_client=philatlas_client,
            gmaps_api_key=gmaps_api_key
        )
        logger.info("Enhanced validator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize enhanced validator: {e}", exc_info=True)
        raise


@validator_router.post(
    "/validate-address",
    response_model=EnhancedAddressValidationResponse,
    summary="Comprehensive Address Validation with Database Integration",
    description="Validate a Philippine address with comprehensive multi-layer validation."
)
async def validate_address_enhanced(request: AddressValidationRequest):
    try:
        if not request.address or not request.address.strip():
            logger.warning("Empty address validation attempt")
            raise HTTPException(status_code=400, detail="Address cannot be empty")
        
        logger.info(f"Enhanced validation request: {request.address[:50]}...")
        
        global enhanced_validator
        if enhanced_validator is None:
            logger.error("Enhanced validator not initialized")
            raise HTTPException(status_code=503, detail="Validator service not available")
        
        result = await enhanced_validator.validate_address(request.address)
        
        logger.info(f"Validation complete: isValid={result.verdict.isValid}, confidence={result.verdict.confidence}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in enhanced address validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
