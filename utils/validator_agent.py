"""
LangGraph-based Address Validation Agent
Converts the validation flow into a state graph with tool calls for better observability and control.
"""

import uuid
import logging
from typing import Dict, Optional, List, TypedDict, Annotated, Literal
from operator import add

try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # Fallback for older versions
    from langgraph.graph import StateGraph
    END = "__end__"

from schema import (
    EnhancedAddressValidationResponse,
    VerdictResponse,
    StructureResponse,
    PSGCResponse,
    GeocodeResponse,
    DeliveryHistoryResponse,
    DeliveryHistoryAddress
)
from utils.address_parser import AddressParser
from utils.philatlas_client import PhilAtlasClient
from utils.psgc_api_client import PSGCAPIClient
from core.gmaps_integration import GoogleMapsValidator
from core import database_client as db

logger = logging.getLogger(__name__)


class ValidationState(TypedDict):
    """State for the address validation workflow"""
    validation_id: str
    original_address: str
    clean_address: str
    
    country: str
    structure_ok: bool
    psgc_matched: bool
    geocode_matched: bool
    psgc_api_validated: bool
    delivery_history_success: int
    
    province: str
    city: str
    barangay: str
    street_address: str
    postal_code: str
    
    province_code: str
    city_code: str
    barangay_code: str
    
    latitude: float
    longitude: float
    place_id: str
    geocode_formatted_address: str
    
    # Intermediate geocode results for agentic refinement
    geocode_city: str
    geocode_barangay: str
    geocode_postal: str
    
    region_id: str
    province_id: str
    city_id: str
    barangay_id: str
    
    formatted_address: str
    confidence: float
    reasons: Annotated[List[str], add]
    suggestions: Annotated[List[str], add]
    delivery_history: Optional[DeliveryHistoryResponse]
    
    current_step: str
    error: Optional[str]


class AddressValidatorAgent:
    """
    LangGraph-based address validator with tool calls.
    Each validation step is a node in the graph with clear transitions.
    """
    
    def __init__(
        self,
        parser: AddressParser,
        psgc_client: PSGCAPIClient,
        philatlas_client: Optional[PhilAtlasClient] = None,
        gmaps_api_key: Optional[str] = None
    ):
        """Initialize the validator agent with required clients."""
        self.parser = parser
        self.psgc_client = psgc_client
        self.philatlas_client = philatlas_client
        self.gmaps_validator = GoogleMapsValidator(gmaps_api_key) if gmaps_api_key else None
        self.db_available = db.is_database_available()
        
        self.graph = self._build_graph()
        
        logger.info(f"AddressValidatorAgent initialized (PSGC API: True, DB: {self.db_available}, GMaps: {self.gmaps_validator is not None}, PhilAtlas: {philatlas_client is not None})")
    
    def _build_graph(self) -> StateGraph:
        """Build the enhanced LangGraph validation workflow with intelligent routing."""
        workflow = StateGraph(ValidationState)
        
        # Add validation nodes
        workflow.add_node("initialize", self._initialize_state)
        workflow.add_node("validate_country", self._validate_country)
        workflow.add_node("parse_address", self._parse_address)
        workflow.add_node("apply_typo_corrections", self._apply_typo_corrections)
        workflow.add_node("validate_psgc_api", self._validate_psgc_api)
        workflow.add_node("match_psgc", self._match_psgc)
        workflow.add_node("geocode", self._geocode_address)
        workflow.add_node("refine_with_geocode", self._refine_with_geocode)
        workflow.add_node("check_delivery", self._check_delivery_history)
        workflow.add_node("retry_with_suggestions", self._retry_with_suggestions)
        workflow.add_node("calculate_verdict", self._calculate_verdict)
        
        workflow.set_entry_point("initialize")
        
        # Linear flow with conditional branching
        workflow.add_edge("initialize", "validate_country")
        
        # After country validation, decide next step
        workflow.add_conditional_edges(
            "validate_country",
            self._route_after_country_check,
            {
                "parse": "parse_address",
                "end": "calculate_verdict"
            }
        )
        
        # After parsing, apply typo corrections
        workflow.add_edge("parse_address", "apply_typo_corrections")
        workflow.add_edge("apply_typo_corrections", "validate_psgc_api")
        workflow.add_edge("validate_psgc_api", "match_psgc")
        
        # After PSGC matching, decide if we need geocoding help
        workflow.add_conditional_edges(
            "match_psgc",
            self._route_after_psgc,
            {
                "geocode": "geocode",
                "skip_geocode": "check_delivery"
            }
        )
        
        # After geocoding, decide if we need to refine
        workflow.add_conditional_edges(
            "geocode",
            self._route_after_geocode,
            {
                "refine": "refine_with_geocode",
                "continue": "check_delivery"
            }
        )
        
        workflow.add_edge("refine_with_geocode", "check_delivery")
        
        # After delivery check, decide if we need to retry
        workflow.add_conditional_edges(
            "check_delivery",
            self._route_after_delivery,
            {
                "retry": "retry_with_suggestions",
                "finish": "calculate_verdict"
            }
        )
        
        workflow.add_edge("retry_with_suggestions", "calculate_verdict")
        workflow.add_edge("calculate_verdict", END)
        
        return workflow.compile()
    
    async def validate_address(self, address_text: str) -> EnhancedAddressValidationResponse:
        """
        Main entry point for address validation.
        Executes the LangGraph workflow and returns the response.
        """
        logger.info(f"Starting LangGraph validation for: {address_text[:50]}...")
        
        initial_state: ValidationState = {
            "validation_id": str(uuid.uuid4()),
            "original_address": address_text,
            "clean_address": address_text.strip().replace(",", ""),
            "country": "PH",
            "structure_ok": False,
            "psgc_matched": False,
            "geocode_matched": False,
            "psgc_api_validated": True,
            "delivery_history_success": 0,
            "province": "",
            "city": "",
            "barangay": "",
            "street_address": "",
            "postal_code": "",
            "province_code": "",
            "city_code": "",
            "barangay_code": "",
            "latitude": 0.0,
            "longitude": 0.0,
            "place_id": "",
            "geocode_formatted_address": "",
            "geocode_city": "",
            "geocode_barangay": "",
            "geocode_postal": "",
            "region_id": "",
            "province_id": "",
            "city_id": "",
            "barangay_id": "",
            "formatted_address": "",
            "confidence": 0.0,
            "reasons": [],
            "suggestions": [],
            "delivery_history": None,
            "current_step": "initialize",
            "error": None
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        
        return self._build_response(final_state)
    
    
    def _initialize_state(self, state: ValidationState) -> ValidationState:
        """Initialize the validation state."""
        logger.info("STEP 1: Initializing validation state")
        state["current_step"] = "initialize"
        return state
    
    def _validate_country(self, state: ValidationState) -> ValidationState:
        """Validate if address is from Philippines."""
        logger.info("STEP 2: Validating country")
        state["current_step"] = "validate_country"
        
        if not self.gmaps_validator:
            state["country"] = "PH"
            return state
        
        addr_validation = self.gmaps_validator.get_address_country(state["clean_address"])
        logger.debug(f"Country validation result: {addr_validation}")
        
        if addr_validation.get("isValid") == True:
            detected_country = addr_validation.get("country", "Unknown")
            logger.info(f"Non-Philippine address detected: {detected_country}")
            state["country"] = detected_country
            state["reasons"].append(
                f"We only validate Philippine addresses. The provided address appears to be from {detected_country}."
            )
            state["structure_ok"] = False
            state["geocode_matched"] = False
            state["psgc_matched"] = False
            
        elif addr_validation.get("isValid") == "PH":
            state["country"] = "PH"
            
        elif addr_validation.get("isValid") == "NOT_PH":
            detected_country = addr_validation.get("country", "")
            if detected_country:
                logger.info(f"Non-Philippine address detected: {detected_country}")
                state["country"] = detected_country
                state["reasons"].append(
                    f"We only validate Philippine addresses. The provided address appears to be from {detected_country}."
                )
            else:
                logger.info("Non-Philippine address detected (country unknown)")
                state["country"] = "Unknown"
                state["reasons"].append(
                    "We only validate Philippine addresses. The provided address does not appear to be from the Philippines."
                )
            state["structure_ok"] = False
            state["geocode_matched"] = False
            state["psgc_matched"] = False
        else:
            state["country"] = addr_validation.get("country", "PH")
        
        return state
    
    async def _parse_address(self, state: ValidationState) -> ValidationState:
        """Parse address using LLM."""
        logger.info("STEP 3: Parsing address with LLM")
        state["current_step"] = "parse_address"
        
        parsed = await self.parser.parse(state["clean_address"])
        logger.debug(f"LLM parsed: province={parsed.province}, city={parsed.city}, barangay={parsed.barangay}")
        
        parsed = self._apply_aliases(parsed)
        
        state["province"] = parsed.province or ""
        state["city"] = parsed.city or ""
        state["barangay"] = parsed.barangay or ""
        state["street_address"] = parsed.street_address or ""
        state["postal_code"] = parsed.postal_code or ""
        
        return state
    
    def _apply_typo_corrections(self, state: ValidationState) -> ValidationState:
        """Apply intelligent typo corrections using smart handler."""
        logger.info("STEP 3.5: Applying typo corrections")
        state["current_step"] = "apply_typo_corrections"
        
        try:
            from utils.smart_typo_handler import SmartTypoHandler
            from schema import ParsedAddress
            
            typo_handler = SmartTypoHandler(min_score=80, phonetic_enabled=True)
            
            # Create a parsed address object
            parsed_address = ParsedAddress(
                province=state["province"],
                city=state["city"],
                barangay=state["barangay"],
                street_address=state["street_address"],
                postal_code=state["postal_code"]
            )
            
            # Apply corrections
            corrected = typo_handler.apply_corrections(parsed_address, self.psgc_client)
            
            # Update state with corrected values
            state["province"] = corrected.province or state["province"]
            state["city"] = corrected.city or state["city"]
            state["barangay"] = corrected.barangay or state["barangay"]
            
            logger.debug(f"After typo corrections: province={state['province']}, city={state['city']}, barangay={state['barangay']}")
            
        except ImportError:
            logger.debug("Smart typo handler not available, skipping")
        except Exception as e:
            logger.warning(f"Typo correction failed: {e}")
        
        return state
    
    def _validate_psgc_api(self, state: ValidationState) -> ValidationState:
        """Validate address components against PSGC API with PhilAtlas fallback."""
        logger.info("STEP 4: Validating against PSGC API (with PhilAtlas fallback)")
        state["current_step"] = "validate_psgc_api"
        
        psgc_api_valid = True
        philatlas_used = False
        new_reasons = []
        province_code = None
        city_code = None
        
        # Province validation - PSGC primary, PhilAtlas fallback
        if state["province"]:
            province_match = self.psgc_client.search_province(state["province"])
            if province_match:
                state["province"] = province_match.get('name', '').title()
                province_code = province_match.get('code')
                state["province_code"] = province_code
                logger.info(f"✓ Province validated via PSGC API: {state['province']} ({province_code})")
            elif self.philatlas_client:
                # Try PhilAtlas fallback
                logger.info(f"Province '{state['province']}' not in PSGC, trying PhilAtlas...")
                philatlas_provinces = self.philatlas_client.get_provinces()
                philatlas_match = next(
                    (p for p in philatlas_provinces if p['name'].lower() == state["province"].lower() or 
                     self.philatlas_client._normalize_name(p['name']) == self.philatlas_client._normalize_name(state["province"])),
                    None
                )
                if philatlas_match:
                    state["province"] = philatlas_match['name'].title()
                    philatlas_used = True
                    logger.info(f"✓ Province validated via PhilAtlas: {state['province']}")
                    new_reasons.append(f"Province verified via PhilAtlas (not found in PSGC)")
                else:
                    reason = f"Province '{state['province']}' not found in PSGC API or PhilAtlas"
                    if reason not in state["reasons"]:
                        new_reasons.append(reason)
                    state["province"] = state["province"].title()
                    psgc_api_valid = False
            else:
                reason = f"Province '{state['province']}' not found in PSGC API"
                if reason not in state["reasons"]:
                    new_reasons.append(reason)
                state["province"] = state["province"].title()
                psgc_api_valid = False
        
        # City validation - PSGC primary, PhilAtlas fallback
        if state["city"]:
            city_match = self.psgc_client.search_city_municipality(
                state["city"], province_code
            )
            if city_match:
                state["city"] = city_match.get('name', '').title()
                city_code = city_match.get('code')
                state["city_code"] = city_code
                
                # Get postal code from PSGC API if available
                if not state.get("postal_code") and city_match.get('zip_code'):
                    state["postal_code"] = city_match.get('zip_code')
                    logger.info(f"Postal code from PSGC API: {state['postal_code']}")
                
                logger.info(f"✓ City validated via PSGC API: {state['city']} ({city_code})")
            elif self.philatlas_client:
                # Try PhilAtlas fallback
                logger.info(f"City '{state['city']}' not in PSGC, trying PhilAtlas...")
                
                # Get province URL from PhilAtlas for province-specific city search
                province_url = None
                if state["province"]:
                    philatlas_provinces = self.philatlas_client.get_provinces()
                    province_match = next(
                        (p for p in philatlas_provinces if p['name'].lower() == state["province"].lower() or 
                         self.philatlas_client._normalize_name(p['name']) == self.philatlas_client._normalize_name(state["province"])),
                        None
                    )
                    if province_match:
                        province_url = province_match.get('url')
                
                # Get cities for the province
                philatlas_cities = self.philatlas_client.get_cities_municipalities(province_url) if province_url else []
                philatlas_match = next(
                    (c for c in philatlas_cities if c['name'].lower() == state["city"].lower() or
                     self.philatlas_client._normalize_name(c['name']) == self.philatlas_client._normalize_name(state["city"])),
                    None
                )
                if philatlas_match:
                    state["city"] = philatlas_match['name'].title()
                    philatlas_used = True
                    logger.info(f"✓ City validated via PhilAtlas: {state['city']}")
                    new_reasons.append(f"City verified via PhilAtlas (not found in PSGC)")
                else:
                    reason = f"City '{state['city']}' not found in PSGC API or PhilAtlas"
                    if reason not in state["reasons"]:
                        new_reasons.append(reason)
                    state["city"] = state["city"].title()
                    psgc_api_valid = False
            else:
                reason = f"City '{state['city']}' not found in PSGC API"
                if reason not in state["reasons"]:
                    new_reasons.append(reason)
                state["city"] = state["city"].title()
                psgc_api_valid = False
        
        # Barangay validation - PSGC primary, PhilAtlas fallback
        if state["barangay"]:
            barangay_match = self.psgc_client.search_barangay(
                state["barangay"], city_code
            )
            if barangay_match:
                state["barangay"] = barangay_match.get('name', '').title()
                state["barangay_code"] = barangay_match.get('code')
                logger.info(f"✓ Barangay validated via PSGC API: {state['barangay']} ({state['barangay_code']})")
            elif self.philatlas_client and state["city"]:
                # Try PhilAtlas fallback
                logger.info(f"Barangay '{state['barangay']}' not in PSGC, trying PhilAtlas...")
                
                # Get city URL from PhilAtlas for city-specific barangay search
                city_url = None
                if state["city"]:
                    # Get province URL first
                    province_url = None
                    if state["province"]:
                        philatlas_provinces = self.philatlas_client.get_provinces()
                        province_match = next(
                            (p for p in philatlas_provinces if p['name'].lower() == state["province"].lower() or 
                             self.philatlas_client._normalize_name(p['name']) == self.philatlas_client._normalize_name(state["province"])),
                            None
                        )
                        if province_match:
                            province_url = province_match.get('url')
                    
                    # Get city URL
                    philatlas_cities = self.philatlas_client.get_cities_municipalities(province_url) if province_url else []
                    city_match = next(
                        (c for c in philatlas_cities if c['name'].lower() == state["city"].lower() or
                         self.philatlas_client._normalize_name(c['name']) == self.philatlas_client._normalize_name(state["city"])),
                        None
                    )
                    if city_match:
                        city_url = city_match.get('url')
                
                # Get barangays for the city
                philatlas_barangays = self.philatlas_client.get_barangays(city_url) if city_url else []
                philatlas_match = next(
                    (b for b in philatlas_barangays if b['name'].lower() == state["barangay"].lower() or
                     self.philatlas_client._normalize_name(b['name']) == self.philatlas_client._normalize_name(state["barangay"])),
                    None
                )
                if philatlas_match:
                    state["barangay"] = philatlas_match['name'].title()
                    philatlas_used = True
                    logger.info(f"✓ Barangay validated via PhilAtlas: {state['barangay']}")
                    new_reasons.append(f"Barangay verified via PhilAtlas (not found in PSGC)")
                else:
                    reason = f"Barangay '{state['barangay']}' not found in PSGC API or PhilAtlas"
                    if reason not in state["reasons"]:
                        new_reasons.append(reason)
                    state["barangay"] = state["barangay"].title()
                    psgc_api_valid = False
            else:
                reason = f"Barangay '{state['barangay']}' not found in PSGC API"
                if reason not in state["reasons"]:
                    new_reasons.append(reason)
                state["barangay"] = state["barangay"].title()
                psgc_api_valid = False
        
        if new_reasons:
            state["reasons"].extend(new_reasons)
        
        # Mark as validated if PSGC succeeded OR PhilAtlas provided verification
        state["psgc_api_validated"] = psgc_api_valid or philatlas_used
        if philatlas_used and not psgc_api_valid:
            logger.info("⚠ PSGC validation failed but PhilAtlas provided verification - marking as partially validated")
        
        if all([state["street_address"], state["barangay"], state["city"], state["province"]]):
            state["formatted_address"] = self._format_ph_address(state)
            state["structure_ok"] = True
        
        return state
    
    def _match_psgc(self, state: ValidationState) -> ValidationState:
        """Match address components against PSGC database and API codes."""
        logger.info("STEP 5: Matching PSGC codes from API and database")
        state["current_step"] = "match_psgc"
        
        # Use codes from PSGC API if available
        if state.get("province_code"):
            state["province_id"] = state["province_code"]
            # Extract region code (first 2 digits of province code + zeros)
            state["region_id"] = state["province_code"][:2] + "0000000"
            logger.debug(f"Using PSGC API province code: {state['province_id']}")
        elif state["province"] and self.db_available:
            province_details = db.get_province_details(state["province"])
            if province_details:
                state["region_id"] = str(province_details.get("region_id", ""))
                state["province_id"] = str(province_details.get("province_id", ""))
                logger.debug(f"PSGC database province matched: {province_details}")
        
        if state.get("city_code"):
            state["city_id"] = state["city_code"]
            logger.debug(f"Using PSGC API city code: {state['city_id']}")
        elif state["city"] and self.db_available:
            city_details = db.get_city_details(state["city"])
            if city_details:
                state["city_id"] = str(city_details.get("city_id", ""))
                logger.debug(f"PSGC database city matched: {city_details}")
        
        if state.get("barangay_code"):
            state["barangay_id"] = state["barangay_code"]
            logger.debug(f"Using PSGC API barangay code: {state['barangay_id']}")
        elif state["barangay"] and self.db_available:
            barangay_details = db.get_barangay_details(state["barangay"], state.get("city"))
            if barangay_details:
                state["barangay_id"] = str(barangay_details.get("barangay_id", ""))
                if not state["postal_code"]:
                    postal_from_db = barangay_details.get("postcode")
                    if postal_from_db:
                        state["postal_code"] = str(postal_from_db)
                        logger.info(f"Postal code from database: {postal_from_db}")
                logger.debug(f"PSGC database barangay matched: {barangay_details}")
        
        # Try PhilAtlas for postal code if still not found
        if not state["postal_code"] and state["barangay"] and state["city"] and self.philatlas_client:
            logger.info(f"Postal code not found, attempting PhilAtlas lookup")
            philatlas_postal = self.philatlas_client.get_barangay_postal_code(
                state["barangay"],
                state["city"],
                state["province"]
            )
            if philatlas_postal:
                state["postal_code"] = philatlas_postal
                new_suggestion = f"Postal code {philatlas_postal} suggested from PhilAtlas for barangay {state['barangay']}"
                if new_suggestion not in state["suggestions"]:
                    state["suggestions"].append(new_suggestion)
                logger.info(f"Postal code from PhilAtlas: {philatlas_postal}")
        
        psgc_codes_found = all([
            state.get("region_id"),
            state.get("province_id"),
            state.get("city_id"),
            state.get("barangay_id")
        ])
        
        state["psgc_matched"] = psgc_codes_found and state.get("psgc_api_validated", True)
        
        if psgc_codes_found and not state.get("psgc_api_validated", True):
            logger.warning("PSGC codes found but PSGC API validation had issues - marking psgcMatched as False")
        
        return state
    
    def _geocode_address(self, state: ValidationState) -> ValidationState:
        """Geocode the address using Google Maps."""
        logger.info("STEP 6: Geocoding address")
        state["current_step"] = "geocode"
        
        if not self.gmaps_validator:
            return state
        
        address = state["formatted_address"] or state["clean_address"]
        geocode_result = self.gmaps_validator.get_geocode(address, state["country"])
        
        if geocode_result.get("isValid") == False:
            state["reasons"].append(geocode_result.get("reason", "Geocoding failed"))
            state["geocode_matched"] = False
            
        elif state["country"] == "PH" and geocode_result.get("isValid"):
            state["latitude"] = geocode_result.get("latitude", 0)
            state["longitude"] = geocode_result.get("longitude", 0)
            state["place_id"] = geocode_result.get("place_id", "")
            state["geocode_formatted_address"] = geocode_result.get("formattedAddress", "")
            state["geocode_matched"] = True
            
            # Store geocode results for potential refinement (agentic decision-making)
            state["geocode_city"] = geocode_result.get("city", "")
            state["geocode_barangay"] = geocode_result.get("barangay", "")
            state["geocode_postal"] = geocode_result.get("postalCode", "")
            
            if not state["city"] and geocode_result.get("city"):
                state["city"] = geocode_result.get("city")
                logger.info(f"City filled from geocode: {state['city']}")
                
                if self.db_available:
                    logger.info("Re-matching PSGC after geocode filled missing components")
                    city_details = db.get_city_details(state["city"])
                    if city_details:
                        state["city_id"] = str(city_details.get("city_id", ""))
            
            if not state["barangay"] and geocode_result.get("barangay"):
                state["barangay"] = geocode_result.get("barangay")
                logger.info(f"Barangay filled from geocode: {state['barangay']}")
                
                if self.db_available:
                    barangay_details = db.get_barangay_details(state["barangay"])
                    if barangay_details:
                        state["barangay_id"] = str(barangay_details.get("barangay_id", ""))
                        if not state["postal_code"]:
                            state["postal_code"] = str(barangay_details.get("postcode", ""))
            
            if not state["postal_code"] and geocode_result.get("postalCode"):
                state["postal_code"] = geocode_result.get("postalCode")
                logger.info(f"Postal code filled from geocode: {state['postal_code']}")
            
            if all([state["street_address"], state["barangay"], state["city"], state["province"]]):
                state["formatted_address"] = self._format_ph_address(state)
                state["structure_ok"] = True
                
        elif geocode_result.get("isValid"):
            state["structure_ok"] = True
            state["street_address"] = geocode_result.get("streetAddress", "")
            state["city"] = geocode_result.get("city", "")
            state["province"] = geocode_result.get("province", "")
            state["postal_code"] = geocode_result.get("postalCode", "")
            state["country"] = geocode_result.get("country", "")
            state["formatted_address"] = geocode_result.get("formattedAddress", "")
            state["latitude"] = geocode_result.get("latitude", 0)
            state["longitude"] = geocode_result.get("longitude", 0)
            state["place_id"] = geocode_result.get("place_id", "")
            state["geocode_matched"] = True
        
        return state
    
    def _check_delivery_history(self, state: ValidationState) -> ValidationState:
        """Check delivery history for the address."""
        logger.info("STEP 7: Checking delivery history")
        state["current_step"] = "check_delivery"
        
        if not self.db_available:
            return state
        
        def format_history(history_records: List) -> DeliveryHistoryAddress:
            """Format delivery history records."""
            if history_records and len(history_records) > 0:
                record = history_records[0]
                last_delivery = record.get("last_delivery")
                return DeliveryHistoryAddress(
                    address=record.get("address") or "",
                    status=record.get("status") or "",
                    last_delivery_at=str(last_delivery) if last_delivery else "",
                    rts_reason=record.get("failure_reason") or ""
                )
            return DeliveryHistoryAddress()
        
        original_history = db.get_delivery_history(state["clean_address"])
        input_delivery = format_history(original_history)
        
        formatted_history = db.get_delivery_history(state["formatted_address"]) if state["formatted_address"] else []
        formatted_delivery = format_history(formatted_history)
        
        delivery_success = 0
        if original_history and len(original_history) > 0:
            if original_history[0].get("status") == "DELIVERED":
                delivery_success = 1
            elif original_history[0].get("status") == "RETURN TO SENDER":
                delivery_success = -1
        
        if formatted_history and len(formatted_history) > 0:
            if formatted_history[0].get("status") == "DELIVERED":
                delivery_success = 1
            elif formatted_history[0].get("status") == "RETURN TO SENDER":
                delivery_success = -1
        
        state["delivery_history_success"] = delivery_success
        state["delivery_history"] = DeliveryHistoryResponse(
            inputAddress=input_delivery,
            formattedAddress=formatted_delivery
        )
        
        return state
    
    def _refine_with_geocode(self, state: ValidationState) -> ValidationState:
        """
        Refine address components using geocoding results.
        This is an agentic self-correction step.
        """
        logger.info("STEP 6.5: Refining with geocode results")
        state["current_step"] = "refine_with_geocode"
        
        try:
            # If geocoding provided components we're missing, re-validate with PSGC
            if not state["city"] and state.get("geocode_city"):
                state["city"] = state["geocode_city"]
                logger.info(f"Agent refinement: City filled from geocode: {state['city']}")
                
                # Re-validate city with PSGC
                city_match = self.psgc_client.search_city_municipality(state["city"])
                if city_match:
                    state["city"] = city_match.get('name', '').title()
                    state["city_code"] = city_match.get('code', '')
                    logger.info(f"Agent refinement: City re-validated: {state['city']}")
            
            if not state["barangay"] and state.get("geocode_barangay"):
                state["barangay"] = state["geocode_barangay"]
                logger.info(f"Agent refinement: Barangay filled from geocode: {state['barangay']}")
                
                # Re-validate barangay with PSGC if we have city code
                if state.get("city_code"):
                    barangay_match = self.psgc_client.search_barangay(
                        state["barangay"],
                        state["city_code"]
                    )
                    if barangay_match:
                        state["barangay"] = barangay_match.get('name', '').title()
                        state["barangay_code"] = barangay_match.get('code', '')
                        logger.info(f"Agent refinement: Barangay re-validated: {state['barangay']}")
            
            if not state["postal_code"] and state.get("geocode_postal"):
                state["postal_code"] = state["geocode_postal"]
                logger.info(f"Agent refinement: Postal code filled from geocode: {state['postal_code']}")
            
            # Update formatted address
            if all([state["street_address"], state["barangay"], state["city"], state["province"]]):
                state["formatted_address"] = self._format_ph_address(state)
                state["structure_ok"] = True
                logger.info("Agent refinement: Structure now complete!")
        
        except Exception as e:
            logger.warning(f"Refinement error: {e}")
        
        return state
    
    def _retry_with_suggestions(self, state: ValidationState) -> ValidationState:
        """
        Retry validation using agent-generated suggestions.
        This is an agentic error recovery step.
        """
        logger.info("STEP 7.5: Retrying with agent suggestions")
        state["current_step"] = "retry_with_suggestions_attempted"
        
        try:
            # Analyze reasons and generate recovery actions
            reasons = state["reasons"]
            
            # If barangay not found, try without barangay (city-level validation)
            if any("Barangay" in reason and "not found" in reason for reason in reasons):
                logger.info("Agent recovery: Attempting city-level validation without barangay")
                
                # Try to match city and province only
                if state["city"] and state["province"]:
                    city_match = self.psgc_client.search_city_municipality(state["city"])
                    if city_match:
                        state["city"] = city_match.get('name', '').title()
                        state["city_code"] = city_match.get('code', '')
                        state["city_id"] = city_match.get('code', '')
                        
                        # Mark as partially validated
                        state["suggestions"].append(
                            f"Validated at city level: {state['city']}, {state['province']}. " 
                            f"Barangay '{state['barangay']}' could not be verified."
                        )
                        state["confidence"] = max(state["confidence"], 40)
            
            # If province not found, try variations
            if any("Province" in reason and "not found" in reason for reason in reasons):
                logger.info("Agent recovery: Trying province variations")
                
                # Common variations
                province_variations = [
                    state["province"],
                    f"Province of {state['province']}",
                    state["province"].replace(" Province", "")
                ]
                
                for variation in province_variations:
                    province_match = self.psgc_client.search_province(variation)
                    if province_match:
                        state["province"] = province_match.get('name', '').title()
                        state["province_code"] = province_match.get('code', '')
                        state["suggestions"].append(
                            f"Province name corrected to: {state['province']}"
                        )
                        logger.info(f"Agent recovery: Province found with variation: {state['province']}")
                        break
        
        except Exception as e:
            logger.warning(f"Retry failed: {e}")
        
        return state
    
    def _calculate_verdict(self, state: ValidationState) -> ValidationState:
        """Calculate final validation verdict and confidence score."""
        logger.info("STEP 8: Calculating verdict and confidence")
        state["current_step"] = "calculate_verdict"
        
        structure_ok = state["structure_ok"]
        psgc_matched = state["psgc_matched"]
        geocode_matched = state["geocode_matched"]
        psgc_api_validated = state.get("psgc_api_validated", True)
        delivery_success = state["delivery_history_success"]
        
        # Check for rejection reasons
        reasons = state["reasons"]
        is_non_ph_rejection = any("only validate Philippine addresses" in reason for reason in reasons)
        has_psgc_api_errors = any("not found in PSGC API" in reason for reason in reasons)
        
        if structure_ok and psgc_matched and geocode_matched and delivery_success == 1:
            confidence = 99
        elif not structure_ok and not psgc_matched and not geocode_matched and delivery_success <= 0:
            confidence = 0
        else:
            confidence = 50
            confidence += 15 if delivery_success == 1 else (0 if delivery_success == 0 else -15)
            confidence += 20 if structure_ok else -10
            confidence += 10 if psgc_matched else -5
            confidence += 4 if geocode_matched else -2
            
            if has_psgc_api_errors:
                confidence -= 20
            
            confidence = max(0, min(100, confidence))
        
        state["confidence"] = confidence
        
        logger.info(f"Validation complete: confidence={confidence}, psgc_api_validated={psgc_api_validated}")
        
        return state
    
    
    def _route_after_country_check(self, state: ValidationState) -> Literal["parse", "end"]:
        """Route based on country validation result."""
        if state["country"] == "PH":
            return "parse"
        else:
            return "end"
    
    def _route_after_psgc(self, state: ValidationState) -> Literal["geocode", "skip_geocode"]:
        """
        Decide if geocoding is needed based on PSGC matching results.
        Skip geocoding if we have complete and confident data.
        """
        # If we have all components and PSGC matched, geocoding is optional
        has_all_components = all([
            state["province"],
            state["city"],
            state["barangay"],
            state["postal_code"]
        ])
        
        if has_all_components and state["psgc_matched"]:
            logger.info("Decision: Skipping geocoding (complete PSGC match)")
            return "skip_geocode"
        
        # Otherwise, geocoding can help fill gaps or validate
        logger.info("Decision: Performing geocoding (incomplete data or no PSGC match)")
        return "geocode"
    
    def _route_after_geocode(self, state: ValidationState) -> Literal["refine", "continue"]:
        """
        Decide if we need to refine data with geocoding results.
        Refine if geocoding filled in missing components.
        """
        # Check if geocoding provided new information
        geocode_provided_data = state["geocode_matched"] and (
            state["latitude"] != 0.0 or 
            state["longitude"] != 0.0
        )
        
        # Check if we're missing key components that geocoding might have filled
        missing_components = not all([state["province"], state["city"], state["barangay"]])
        
        if geocode_provided_data and missing_components:
            logger.info("Decision: Refining with geocode data (missing components)")
            return "refine"
        
        logger.info("Decision: Continuing (no refinement needed)")
        return "continue"
    
    def _route_after_delivery(self, state: ValidationState) -> Literal["retry", "finish"]:
        """
        Decide if we should retry validation with suggestions.
        Retry if we have low confidence and suggestions available.
        """
        # Check if we have low confidence and no positive signals
        low_confidence = state["confidence"] < 50
        no_positive_signals = (
            not state["structure_ok"] and
            not state["psgc_matched"] and
            not state["geocode_matched"] and
            state["delivery_history_success"] <= 0
        )
        
        # Check if we have actionable suggestions
        has_suggestions = len(state["suggestions"]) > 0
        
        # Only retry once (check if we're already in retry)
        already_retried = "retry_attempted" in state.get("current_step", "")
        
        if low_confidence and no_positive_signals and has_suggestions and not already_retried:
            logger.info("Decision: Retrying with suggestions (low confidence)")
            return "retry"
        
        logger.info("Decision: Finishing validation")
        return "finish"
    
    
    def _apply_aliases(self, parsed):
        """Apply known aliases for provinces and cities."""
        from utils.config import settings
        
        if parsed.province:
            province_lower = parsed.province.lower()
            if province_lower in settings.PROVINCE_ALIASES:
                parsed.province = settings.PROVINCE_ALIASES[province_lower].title()
        
        if parsed.city:
            city_lower = parsed.city.lower()
            if city_lower in settings.CITY_ABBREVIATIONS:
                parsed.city = settings.CITY_ABBREVIATIONS[city_lower].title()
        
        return parsed
    
    def _format_ph_address(self, state: ValidationState) -> str:
        """Format Philippine address components."""
        components = []
        
        if state["street_address"]:
            components.append(state["street_address"])
        if state["barangay"]:
            components.append(state["barangay"])
        if state["city"]:
            components.append(state["city"])
        if state["province"]:
            components.append(state["province"])
        if state["postal_code"]:
            components.append(str(state["postal_code"]))
        components.append("PH")
        
        formatted = " ".join(components)
        import re
        return re.sub(r"\s+", " ", formatted).strip()
    
    def _build_response(self, state: ValidationState) -> EnhancedAddressValidationResponse:
        """Build the final response from state."""
        seen = set()
        unique_reasons = []
        for reason in state["reasons"]:
            if reason not in seen:
                seen.add(reason)
                unique_reasons.append(reason)
        
        seen_suggestions = set()
        unique_suggestions = []
        for suggestion in state["suggestions"]:
            if suggestion not in seen_suggestions:
                seen_suggestions.add(suggestion)
                unique_suggestions.append(suggestion)
        
        reasons = unique_reasons
        is_non_ph_rejection = any("only validate Philippine addresses" in reason for reason in reasons)
        has_psgc_api_errors = any("not found in PSGC API" in reason for reason in reasons)
        
        if is_non_ph_rejection:
            is_valid = False
        else:
            if has_psgc_api_errors and state["delivery_history_success"] != 1:
                is_valid = False
            else:
                is_valid = (
                    state["structure_ok"] or 
                    state["psgc_matched"] or 
                    state["geocode_matched"] or 
                    state["delivery_history_success"] == 1
                )
        
        return EnhancedAddressValidationResponse(
            id=state["validation_id"],
            input=state["original_address"],
            formattedAddress=state["formatted_address"] or state["geocode_formatted_address"],
            verdict=VerdictResponse(
                isValid=is_valid,
                structureOk=state["structure_ok"],
                psgcMatched=state["psgc_matched"],
                geocodeMatched=state["geocode_matched"],
                deliveryHistorySuccess=(state["delivery_history_success"] == 1),
                confidence=state["confidence"]
            ),
            structure=StructureResponse(
                streetAddress=state["street_address"],
                barangay=state["barangay"],
                city=state["city"],
                province=state["province"],
                country=state["country"],
                postalCode=str(state["postal_code"]),
                formattedAddress=state["formatted_address"]
            ),
            psgc=PSGCResponse(
                regionCode=str(state["region_id"]),
                provinceCode=str(state["province_id"]),
                cityMuniCode=str(state["city_id"]),
                barangayCode=str(state["barangay_id"])
            ),
            geocode=GeocodeResponse(
                lat=state["latitude"],
                lng=state["longitude"],
                place_id=state["place_id"],
                formattedAddress=state["geocode_formatted_address"]
            ),
            deliveryHistory=state["delivery_history"] or DeliveryHistoryResponse(),
            reason=unique_reasons,
            suggestions=unique_suggestions
        )
