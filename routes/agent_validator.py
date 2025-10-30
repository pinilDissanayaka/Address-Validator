from fastapi import APIRouter, HTTPException, Query
import logging
from enum import Enum
from schema import AddressValidationRequest, EnhancedAddressValidationResponse
from utils import AddressParser, PhilAtlasClient
from utils.validator import AddressValidator
from utils.validator_agent import AddressValidatorAgent
from utils.config import settings

logger = logging.getLogger(__name__)

agent_validator_router = APIRouter(
    prefix="/api/v2/agent-validator",
    tags=["Validation"]
)


agent_validator: AddressValidatorAgent = None


@agent_validator_router.on_event("startup")
async def startup_event():
    """Initialize validator components on startup."""
    global agent_validator
    logger.info("Initializing validator components...")
    
    try:
        parser = AddressParser()
        logger.info("Address parser initialized")
        
        philatlas_client = PhilAtlasClient(timeout=settings.PHILATLAS_TIMEOUT)
        logger.info("PhilAtlas client initialized")
        
        gmaps_api_key = settings.GOOGLE_MAPS_API_KEY
        if not gmaps_api_key:
            logger.warning("Google Maps API key not found - geocoding will be limited")
        
        
        agent_validator = AddressValidatorAgent(
            parser=parser,
            philatlas_client=philatlas_client,
            gmaps_api_key=gmaps_api_key
        )
        logger.info("LangGraph agent validator initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize validators: {e}", exc_info=True)
        raise


@agent_validator_router.post(
    "/validate-address",
    response_model=EnhancedAddressValidationResponse,
    summary="Comprehensive Address Validation",
    description="Validate a Philippine address with comprehensive multi-layer validation using LangGraph agent."
)
async def validate_address(
    request: AddressValidationRequest,
):
    """
    Validate a Philippine address with comprehensive multi-layer validation using LangGraph agent.

    Args:
        request (AddressValidationRequest): Address to validate

    Returns:
        EnhancedAddressValidationResponse: Validation result

    Raises:
        HTTPException: If address is empty or invalid
    """
    try:
        if not request.address or not request.address.strip():
            logger.warning("Empty address validation attempt")
            raise HTTPException(status_code=400, detail="Address cannot be empty")

        logger.info(f"Validation request (agent): {request.address[:50]}...")

        if agent_validator is None:
            logger.error("Agent validator not initialized")
            raise HTTPException(status_code=503, detail="Agent validator service not available")
        validator = agent_validator
        
        result = await validator.validate_address(request.address)
        
        logger.info(f"Validation complete: isValid={result.verdict.isValid}, confidence={result.verdict.confidence}")
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in address validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

