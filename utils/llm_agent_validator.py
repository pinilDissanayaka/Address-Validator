"""
LLM-Powered Address Validation Agent
Uses LLM reasoning for intelligent tool calling and address matching decisions.
No fuzzy string matching - pure LLM-based reasoning.
"""

import uuid
import logging
import json
from typing import Dict, Optional, List, Any
from datetime import datetime

from schema import (
    EnhancedAddressValidationResponse,
    VerdictResponse,
    StructureResponse,
    PSGCResponse,
    GeocodeResponse,
    DeliveryHistoryResponse,
)
from utils.address_parser import AddressParser
from utils.philatlas_client import PhilAtlasClient
from utils.psgc_api_client import PSGCAPIClient
from core.gmaps_integration import GoogleMapsValidator
from core import database_client as db

logger = logging.getLogger(__name__)


class LLMAddressValidatorAgent:
    """
    Intelligent address validator that uses LLM reasoning for all decisions.
    The LLM decides which tools to call and how to interpret the results.
    """
    
    def __init__(
        self,
        parser: AddressParser,
        psgc_client: PSGCAPIClient,
        philatlas_client: Optional[PhilAtlasClient] = None,
        gmaps_api_key: Optional[str] = None
    ):
        """Initialize the LLM-powered validator agent."""
        self.parser = parser
        self.psgc_client = psgc_client
        self.philatlas_client = philatlas_client
        self.gmaps_validator = GoogleMapsValidator(gmaps_api_key) if gmaps_api_key else None
        self.db_available = db.is_database_available()
        
        # Import LLM using LangChain
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from utils.config import settings
            if not settings.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY not configured")
            
            model_name = settings.GEMINI_MODEL or "gemini-2.0-flash-exp"
            temperature = getattr(settings, 'GEMINI_TEMPERATURE', 0)
            self.llm = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=temperature,
                convert_system_message_to_human=True
            )
            self.llm_available = True
            logger.info(f"LLM ({model_name} via LangChain) initialized for agent reasoning")
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            self.llm_available = False
        
        logger.info(f"LLMAddressValidatorAgent initialized (LLM: {self.llm_available}, DB: {self.db_available}, GMaps: {self.gmaps_validator is not None})")
    
    def _get_available_tools(self) -> List[Dict[str, Any]]:
        """Define the tools available to the LLM agent."""
        tools = [
            {
                "name": "search_province",
                "description": "Search for a province in the Philippine database. Returns exact matches only.",
                "parameters": {
                    "province_name": "string - the province name to search"
                }
            },
            {
                "name": "search_city",
                "description": "Search for a city/municipality in a province. Returns exact matches only.",
                "parameters": {
                    "province_name": "string - the province name",
                    "city_name": "string - the city/municipality name to search"
                }
            },
            {
                "name": "search_barangay",
                "description": "Search for a barangay in a city. Supports compound names (e.g., 'Funda-Dalipe') and partial names. Returns exact matches or close variations.",
                "parameters": {
                    "province_name": "string - the province name",
                    "city_name": "string - the city name",
                    "barangay_name": "string - the barangay name to search (can be compound name like 'Funda-Dalipe' or partial like 'Funda')"
                }
            },
            {
                "name": "fuzzy_match_component",
                "description": "Use fuzzy matching to find closest match for a component (province, city, or barangay) when exact match fails. Returns best match with confidence score.",
                "parameters": {
                    "component_type": "string - 'province', 'city', or 'barangay'",
                    "component_value": "string - the value to match",
                    "context": "dict - optional context like province_code for city/barangay search"
                }
            },
            {
                "name": "philatlas_search",
                "description": "Search PhilAtlas for barangay postal codes and geographic verification. More comprehensive than PSGC.",
                "parameters": {
                    "barangay": "string - barangay name",
                    "city": "string - city name",
                    "province": "string - province name"
                }
            },
            {
                "name": "get_postal_code",
                "description": "Get postal code for a city from the database.",
                "parameters": {
                    "city_name": "string - the city name",
                    "province_name": "string - the province name"
                }
            },
            {
                "name": "geocode_address",
                "description": "Use Google Maps to geocode an address and get coordinates, formatted address, and geographic components.",
                "parameters": {
                    "address": "string - the full address to geocode"
                }
            },
            {
                "name": "check_delivery_history",
                "description": "Check if this address has delivery history in the database.",
                "parameters": {
                    "address": "string - the address to check"
                }
            },
            {
                "name": "validate_country",
                "description": "Validate if the address is in the Philippines using Google Maps.",
                "parameters": {
                    "address": "string - the address to validate"
                }
            },
            {
                "name": "verify_geographic_hierarchy",
                "description": "Verify that barangay belongs to city and city belongs to province using multiple sources.",
                "parameters": {
                    "province": "string - province name",
                    "city": "string - city name",
                    "barangay": "string - barangay name (optional)"
                }
            }
        ]
        return tools
    
    def _call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call and return the result."""
        try:
            if tool_name == "search_province":
                province = self.psgc_client.search_province(parameters["province_name"])
                if province:
                    return {
                        "success": True,
                        "found": True,
                        "data": {
                            "name": province.get("name"),
                            "code": province.get("code")
                        }
                    }
                return {"success": True, "found": False, "data": None}
            
            elif tool_name == "search_city":
                # Get province code first
                province = self.psgc_client.search_province(parameters["province_name"])
                if not province:
                    return {"success": False, "error": "Province not found"}
                
                city = self.psgc_client.search_city_municipality(
                    parameters["city_name"],
                    province.get("code")
                )
                if city:
                    return {
                        "success": True,
                        "found": True,
                        "data": {
                            "name": city.get("name"),
                            "code": city.get("code"),
                            "zip_code": city.get("zip_code")
                        }
                    }
                return {"success": True, "found": False, "data": None}
            
            elif tool_name == "search_barangay":
                # Get city code first
                province = self.psgc_client.search_province(parameters["province_name"])
                if not province:
                    return {"success": False, "error": "Province not found"}
                
                city = self.psgc_client.search_city_municipality(
                    parameters["city_name"],
                    province.get("code")
                )
                if not city:
                    return {"success": False, "error": "City not found"}
                
                barangay = self.psgc_client.search_barangay(
                    parameters["barangay_name"],
                    city.get("code")
                )
                if barangay:
                    return {
                        "success": True,
                        "found": True,
                        "data": {
                            "name": barangay.get("name"),
                            "code": barangay.get("code")
                        }
                    }
                return {"success": True, "found": False, "data": None}
            
            elif tool_name == "fuzzy_match_component":
                # Use smart typo handler for fuzzy matching
                try:
                    from utils.smart_typo_handler import SmartTypoHandler
                    
                    typo_handler = SmartTypoHandler(min_score=85, phonetic_enabled=True)
                    component_type = parameters["component_type"]
                    component_value = parameters["component_value"]
                    context = parameters.get("context", {})
                    
                    if component_type == "province":
                        corrected, score = typo_handler.correct_province(
                            component_value, 
                            self.psgc_client
                        )
                        if corrected:
                            return {
                                "success": True,
                                "found": True,
                                "original": component_value,
                                "corrected": corrected,
                                "confidence": score
                            }
                    
                    elif component_type == "city":
                        province_code = context.get("province_code")
                        corrected, score = typo_handler.correct_city(
                            component_value,
                            self.psgc_client,
                            province_code
                        )
                        if corrected:
                            return {
                                "success": True,
                                "found": True,
                                "original": component_value,
                                "corrected": corrected,
                                "confidence": score
                            }
                    
                    elif component_type == "barangay":
                        city_code = context.get("city_code")
                        corrected, score = typo_handler.correct_barangay(
                            component_value,
                            self.psgc_client,
                            city_code
                        )
                        if corrected:
                            return {
                                "success": True,
                                "found": True,
                                "original": component_value,
                                "corrected": corrected,
                                "confidence": score
                            }
                    
                    return {"success": True, "found": False, "data": None}
                
                except ImportError:
                    return {"success": False, "error": "Smart typo handler not available"}
            
            elif tool_name == "philatlas_search":
                if not self.philatlas_client:
                    return {"success": False, "error": "PhilAtlas client not available"}
                
                try:
                    # Get city data from PhilAtlas to get the full city name
                    city_data = self.philatlas_client.search_city_municipality(
                        parameters["city"],
                        parameters["province"]
                    )
                    
                    postal_code = self.philatlas_client.get_barangay_postal_code(
                        parameters["barangay"],
                        parameters["city"],
                        parameters["province"]
                    )
                    
                    if postal_code or city_data:
                        result = {
                            "success": True,
                            "found": True,
                            "source": "PhilAtlas"
                        }
                        if postal_code:
                            result["postal_code"] = postal_code
                        if city_data:
                            result["city_name"] = city_data.get("name")  # PhilAtlas has full names like "San Jose de Buenavista"
                        return result
                    return {"success": True, "found": False, "data": None}
                
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            elif tool_name == "get_postal_code":
                postal_code = None
                
                # Try database first
                if self.db_available:
                    try:
                        city_details = db.get_city_details(parameters["city_name"])
                        if city_details and city_details.get("zip_code"):
                            postal_code = city_details.get("zip_code")
                    except:
                        pass
                
                # Try PSGC API
                if not postal_code:
                    try:
                        province = self.psgc_client.search_province(parameters["province_name"])
                        if province:
                            city = self.psgc_client.search_city_municipality(
                                parameters["city_name"],
                                province.get("code")
                            )
                            if city and city.get("zip_code"):
                                postal_code = city.get("zip_code")
                    except:
                        pass
                
                if postal_code:
                    return {"success": True, "postal_code": postal_code}
                return {"success": True, "postal_code": None}
            
            elif tool_name == "geocode_address":
                if not self.gmaps_validator:
                    return {"success": False, "error": "Google Maps not available"}
                
                try:
                    result = self.gmaps_validator.get_geocode(
                        parameters["address"],
                        "PH"
                    )
                    
                    if result and result.get("isValid"):
                        return {
                            "success": True,
                            "data": {
                                "latitude": result.get("latitude"),
                                "longitude": result.get("longitude"),
                                "formatted_address": result.get("formattedAddress"),
                                "place_id": result.get("place_id"),
                                "city": result.get("city"),
                                "barangay": result.get("barangay"),
                                "province": result.get("province"),
                                "postal_code": result.get("postalCode")
                            }
                        }
                    return {"success": False, "error": "Geocoding failed"}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            elif tool_name == "check_delivery_history":
                if not self.db_available:
                    return {"success": True, "found": False, "count": 0}
                
                try:
                    history = db.get_delivery_history(parameters["address"])
                    if history and len(history) > 0:
                        successful = sum(1 for h in history if h.get("status") == "DELIVERED")
                        failed = sum(1 for h in history if h.get("status") == "RETURN TO SENDER")
                        return {
                            "success": True,
                            "found": True,
                            "total": len(history),
                            "successful": successful,
                            "failed": failed,
                            "latest_status": history[0].get("status") if history else None
                        }
                    return {"success": True, "found": False, "count": 0}
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            elif tool_name == "validate_country":
                if not self.gmaps_validator:
                    return {"success": False, "error": "Country validation not available"}
                
                try:
                    result = self.gmaps_validator.get_address_country(parameters["address"])
                    is_ph = result.get("isValid") == "PH" or result.get("isValid") == True
                    return {
                        "success": True,
                        "is_philippines": is_ph,
                        "country_code": result.get("country", "PH" if is_ph else "Unknown")
                    }
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            elif tool_name == "verify_geographic_hierarchy":
                # Verify province -> city -> barangay hierarchy using multiple sources
                verification_results = {
                    "province_valid": False,
                    "city_valid": False,
                    "barangay_valid": False,
                    "city_in_province": False,
                    "barangay_in_city": False
                }
                
                try:
                    # Verify province
                    province = self.psgc_client.search_province(parameters["province"])
                    if province:
                        verification_results["province_valid"] = True
                        province_code = province.get("code")
                        
                        # Verify city belongs to province
                        city = self.psgc_client.search_city_municipality(
                            parameters["city"],
                            province_code
                        )
                        if city:
                            verification_results["city_valid"] = True
                            verification_results["city_in_province"] = True
                            city_code = city.get("code")
                            
                            # Verify barangay if provided
                            if parameters.get("barangay"):
                                barangay = self.psgc_client.search_barangay(
                                    parameters["barangay"],
                                    city_code
                                )
                                if barangay:
                                    verification_results["barangay_valid"] = True
                                    verification_results["barangay_in_city"] = True
                    
                    all_valid = (
                        verification_results["province_valid"] and
                        verification_results["city_valid"] and
                        verification_results["city_in_province"] and
                        (not parameters.get("barangay") or verification_results["barangay_valid"])
                    )
                    
                    return {
                        "success": True,
                        "hierarchy_valid": all_valid,
                        "details": verification_results
                    }
                
                except Exception as e:
                    return {"success": False, "error": str(e)}
            
            else:
                return {"success": False, "error": f"Unknown tool: {tool_name}"}
        
        except Exception as e:
            logger.error(f"Tool call failed: {tool_name} - {e}")
            return {"success": False, "error": str(e)}
    
    def _perform_self_reflection(
        self,
        original_address: str,
        parsed_components: Dict[str, Any],
        final_components: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        confidence: float,
        geocode_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Agent performs self-reflection on validation results.
        Analyzes for inconsistencies, missing data, and potential errors.
        Returns corrections if needed.
        """
        if not self.llm_available:
            return {"needs_correction": False}
        
        try:
            # Build reflection prompt
            reflection_prompt = f"""You are performing self-reflection on an address validation you just completed.

**Original Address:** {original_address}

**Parsed Components (Initial):**
- Province: {parsed_components.get('province', 'N/A')}
- City: {parsed_components.get('city', 'N/A')}
- Barangay: {parsed_components.get('barangay', 'N/A')}

**Final Validated Components:**
- Province: {final_components.get('province', 'N/A')} (Code: {final_components.get('province_code', 'N/A')})
- City: {final_components.get('city', 'N/A')} (Code: {final_components.get('city_code', 'N/A')})
- Barangay: {final_components.get('barangay', 'N/A')} (Code: {final_components.get('barangay_code', 'N/A')})
- Postal Code: {final_components.get('postal_code', 'N/A')}

**Tool Results Summary:**
{json.dumps([{{'tool': t.get('tool'), 'success': t.get('result', {}).get('success')}} for t in tool_results], indent=2)}

**Geocode Data:**
{json.dumps({'city': geocode_data.get('city'), 'barangay': geocode_data.get('barangay'), 'province': geocode_data.get('province')} if geocode_data else 'Not available')}

**Current Confidence:** {confidence}%

**Your Task - Self-Reflection:**
Analyze the validation results and identify any potential issues:

1. **Consistency Checks:**
   - Do the final components make geographic sense? (barangay ∈ city ∈ province)
   - Are there conflicts between tool results and final components?
   - Does geocode data contradict PSGC validation?

2. **Completeness Checks:**
   - Are critical components missing? (province, city)
   - Should we have a postal code but don't?
   - Did we lose any correctly parsed data?

3. **Confidence Assessment:**
   - Is the confidence score accurate given the validation results?
   - Should it be higher (all tools succeeded) or lower (many failures)?

4. **Data Quality:**
   - Are component names properly formatted? (title case, no typos)
   - Are there redundant words between components? (e.g., "Manila" in both city and province)
   - Does the barangay name make sense for the city?

**Response Format (JSON):**
{{
    "needs_correction": true/false,
    "issues": ["list of identified issues"],
    "corrected_components": {{
        "province": "corrected value or null",
        "city": "corrected value or null",
        "barangay": "corrected value or null",
        "postal_code": "corrected value or null"
    }},
    "confidence_adjustment": -10 to +10 (adjustment to confidence score),
    "reflection_notes": ["notes for the user about the validation"],
    "reasoning": "Your detailed reasoning about what you found"
}}

**Guidelines:**
- Only suggest corrections if you find actual issues
- Be conservative - don't change data that looks correct
- If confidence is >90% and all components validated, set needs_correction=false
- Focus on geographic consistency and data completeness
- Consider that geocode data may be approximate (less reliable than PSGC)
- CRITICAL: Keep complete city names intact (e.g., "San Jose De Buenavista", not "San Jose")

Respond with ONLY the JSON object, no additional text.
"""
            
            # Get LLM reflection
            response = self.llm.invoke(reflection_prompt)
            response_text = response.content.strip()
            
            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            reflection_result = json.loads(response_text)
            
            logger.info(f"Self-reflection reasoning: {reflection_result.get('reasoning', 'N/A')[:150]}...")
            
            return reflection_result
        
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse reflection response as JSON: {e}")
            return {"needs_correction": False}
        except Exception as e:
            logger.error(f"Self-reflection failed: {e}")
            return {"needs_correction": False}
    
    def _generate_llm_prompt(
        self,
        address: str,
        parsed_components: Dict[str, Any],
        tool_results: List[Dict[str, Any]],
        iteration: int
    ) -> str:
        """Generate the prompt for the LLM to reason about the address."""
        prompt = f"""You are an intelligent address validation agent for Philippine addresses with access to multiple validation tools.

**Current Task:** Validate and match address components using available tools strategically.

**Input Address:** {address}

**Parsed Components (from initial LLM parsing):**
- Province: {parsed_components.get('province', 'N/A')}
- City: {parsed_components.get('city', 'N/A')}
- Barangay: {parsed_components.get('barangay', 'N/A')}
- Street Address: {parsed_components.get('street_address', 'N/A')}
- Postal Code: {parsed_components.get('postal_code', 'N/A')}

**Tool Results from Previous Iteration:**
{json.dumps(tool_results, indent=2) if tool_results else 'No tools called yet'}

**Available Tools:**
{json.dumps(self._get_available_tools(), indent=2)}

**Intelligent Tool Usage Strategy:**

1. **First Priority - Exact Matching:**
   - Try search_province, search_city, search_barangay with the FULL EXACT names from the address
   - IMPORTANT: Keep complete city names intact (e.g., "San Jose De Buenavista", not "San Jose")
   - Many Philippine cities have compound names - do NOT truncate them
   
2. **Second Priority - Fuzzy Matching (if exact fails):**
   - Use fuzzy_match_component for potential typos
   - This tool handles phonetic matching (Cordoba/Cordova) and spelling errors
   
3. **Third Priority - Cross-Validation:**
   - Use verify_geographic_hierarchy to ensure barangay ∈ city ∈ province
   - Use philatlas_search for postal codes when missing
   
4. **Fourth Priority - Geocoding (if needed):**
   - Use geocode_address to fill missing components
   - Extract city, barangay, postal from geocode results
   
5. **Final Step - Additional Data:**
   - Use check_delivery_history for confidence boost
   - Use get_postal_code if still missing

**Smart Decision Making:**
- If exact search fails, IMMEDIATELY try fuzzy_match_component (don't wait for next iteration)
- Call verify_geographic_hierarchy after getting province/city/barangay to confirm they match
- Use geocoding when components are missing or to cross-validate
- Check delivery history for addresses with successful deliveries (confidence boost)
- Be strategic: don't call unnecessary tools if you already have high confidence

**Response Format (JSON):**
{{
    "reasoning": "Your step-by-step thought process and tool selection strategy",
    "next_tools": [
        {{
            "tool": "tool_name",
            "parameters": {{"param": "value"}},
            "reason": "why calling this tool and expected outcome"
        }}
    ],
    "validation_complete": false,
    "final_components": {{
        "province": "validated province name or null",
        "province_code": "PSGC code or null",
        "city": "validated city name or null",
        "city_code": "PSGC code or null",
        "barangay": "validated barangay name or null",
        "barangay_code": "PSGC code or null",
        "postal_code": "postal code or null"
    }},
    "confidence": 0-100,
    "suggestions": ["list of helpful suggestions for the user"]
}}

**Confidence Scoring Guidelines:**
- 90-100: All components validated with exact matches + hierarchy verified + geocoded
- 70-89: All major components (province, city) validated, barangay might be fuzzy matched
- 50-69: Major components validated but hierarchy not verified or missing barangay
- 30-49: Some components validated but significant gaps
- 0-29: Few or no components validated

**Important Guidelines:**
- Be proactive: if exact match fails, immediately try fuzzy matching in the SAME iteration
- Use multiple tools together for cross-validation
- Prioritize official PSGC data over geocoding
- Set validation_complete to true when confidence > 70 OR no more useful tools to call
- Current iteration: {iteration}/3 (max iterations)
- If iteration 3, you MUST complete regardless of confidence

Respond with ONLY the JSON object, no additional text.
"""
        return prompt
    
    async def validate_address(self, address_text: str) -> EnhancedAddressValidationResponse:
        """
        Main validation method using LLM reasoning.
        """
        validation_id = str(uuid.uuid4())
        logger.info(f"Starting LLM-powered validation for: {address_text[:50]}...")
        
        try:
            # Step 1: Parse address using LLM
            logger.info("Step 1: Parsing address with LLM")
            parsed = await self.parser.parse(address_text)
            
            parsed_components = {
                "province": parsed.province,
                "city": parsed.city,
                "barangay": parsed.barangay,
                "street_address": parsed.street_address,
                "postal_code": parsed.postal_code
            }
            
            logger.info(f"Parsed components: {parsed_components}")
            
            # Step 2: Iterative LLM-based validation with tool calling
            tool_results = []
            max_iterations = 3
            final_components = None
            confidence = 0
            suggestions = []
            
            for iteration in range(1, max_iterations + 1):
                logger.info(f"Step 2.{iteration}: LLM reasoning and tool calling")
                
                if not self.llm_available:
                    logger.warning("LLM not available, falling back to direct matching")
                    break
                
                # Generate prompt
                prompt = self._generate_llm_prompt(
                    address_text,
                    parsed_components,
                    tool_results,
                    iteration
                )
                
                # Get LLM response using LangChain
                try:
                    response = self.llm.invoke(prompt)
                    response_text = response.content.strip()
                    
                    # Extract JSON from response
                    if "```json" in response_text:
                        response_text = response_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in response_text:
                        response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                    llm_decision = json.loads(response_text)
                    logger.info(f"LLM reasoning: {llm_decision.get('reasoning', 'N/A')[:200]}...")
                    
                    # Execute tool calls
                    if llm_decision.get("next_tools"):
                        for tool_call in llm_decision["next_tools"]:
                            tool_name = tool_call["tool"]
                            params = tool_call["parameters"]
                            logger.info(f"Calling tool: {tool_name} with params: {params}")
                            
                            result = self._call_tool(tool_name, params)
                            tool_results.append({
                                "tool": tool_name,
                                "parameters": params,
                                "result": result,
                                "reason": tool_call.get("reason", "")
                            })
                            logger.info(f"Tool result: {result}")
                    
                    # Check if validation is complete
                    if llm_decision.get("validation_complete"):
                        final_components = llm_decision.get("final_components", {})
                        confidence = llm_decision.get("confidence", 0)
                        suggestions = llm_decision.get("suggestions", [])
                        logger.info(f"Validation complete! Confidence: {confidence}%")
                        break
                
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LLM response as JSON: {e}")
                    logger.debug(f"LLM response: {response_text}")
                except Exception as e:
                    logger.error(f"LLM reasoning failed: {e}")
            
            # Step 3: Extract additional data from tool results
            geocode_data = None
            delivery_data = None
            philatlas_city_name = None  # PhilAtlas has more complete city names than PSGC
            
            for tool_result in tool_results:
                if tool_result.get("tool") == "geocode_address":
                    result = tool_result.get("result", {})
                    if result.get("success"):
                        geocode_data = result.get("data", {})
                        logger.info(f"Geocode data available: {geocode_data.get('formatted_address')}")
                
                elif tool_result.get("tool") == "check_delivery_history":
                    result = tool_result.get("result", {})
                    if result.get("found"):
                        delivery_data = result
                        logger.info(f"Delivery history found: {delivery_data.get('successful')} successful deliveries")
                
                elif tool_result.get("tool") == "philatlas_search":
                    result = tool_result.get("result", {})
                    if result.get("success") and result.get("city_name"):
                        philatlas_city_name = result.get("city_name")
                        logger.info(f"PhilAtlas city name: {philatlas_city_name}")
            
            # Step 4: Build response
            if not final_components:
                # Fallback to parsed components
                final_components = {
                    "province": parsed.province,
                    "city": parsed.city,
                    "barangay": parsed.barangay,
                    "postal_code": parsed.postal_code,
                    "province_code": None,
                    "city_code": None,
                    "barangay_code": None
                }
                confidence = 50
                suggestions.append("Could not fully validate address components")
            else:
                # IMPORTANT: If LLM didn't validate a component but we have it from parsing, use it
                # This prevents losing correctly parsed data when tool calls fail
                if not final_components.get("barangay") and parsed.barangay:
                    final_components["barangay"] = parsed.barangay
                    logger.info(f"Using parsed barangay (LLM validation incomplete): {parsed.barangay}")
                    suggestions.append(f"Verify the barangay name '{parsed.barangay}'.")
                
                if not final_components.get("city") and parsed.city:
                    final_components["city"] = parsed.city
                    logger.info(f"Using parsed city (LLM validation incomplete): {parsed.city}")
                
                if not final_components.get("province") and parsed.province:
                    final_components["province"] = parsed.province
                    logger.info(f"Using parsed province (LLM validation incomplete): {parsed.province}")
                
                if not final_components.get("postal_code") and parsed.postal_code:
                    final_components["postal_code"] = parsed.postal_code
                    logger.info(f"Using parsed postal code (LLM validation incomplete): {parsed.postal_code}")
            
            # Step 5: Fill gaps and enhance data from PhilAtlas and geocoding
            # PhilAtlas has more complete city names than PSGC (e.g., "San Jose de Buenavista" vs "San Jose ")
            if philatlas_city_name and final_components.get("city"):
                psgc_city = final_components["city"].strip()
                philatlas_city = philatlas_city_name.strip()
                # Replace if PhilAtlas name is longer/more complete
                if len(philatlas_city) > len(psgc_city) and psgc_city.lower() in philatlas_city.lower():
                    logger.info(f"Using PhilAtlas city name '{philatlas_city}' instead of PSGC '{psgc_city}'")
                    final_components["city"] = philatlas_city
            
            if geocode_data:
                if not final_components.get("city") and geocode_data.get("city"):
                    final_components["city"] = geocode_data.get("city")
                    suggestions.append(f"City filled from geocoding: {geocode_data.get('city')}")
                
                if not final_components.get("barangay") and geocode_data.get("barangay"):
                    final_components["barangay"] = geocode_data.get("barangay")
                    suggestions.append(f"Barangay filled from geocoding: {geocode_data.get('barangay')}")
                
                if not final_components.get("postal_code") and geocode_data.get("postal_code"):
                    final_components["postal_code"] = geocode_data.get("postal_code")
            
            # Step 6: Boost confidence based on delivery history
            if delivery_data and delivery_data.get("successful", 0) > 0:
                confidence = min(100, confidence + 10)
                suggestions.append(f"Address has {delivery_data.get('successful')} successful delivery records")
            elif delivery_data and delivery_data.get("failed", 0) > 0:
                confidence = max(0, confidence - 5)
                suggestions.append(f"Warning: Address has {delivery_data.get('failed')} failed delivery attempts")
            
            # Step 7: Self-Reflection - Agent analyzes its own results
            logger.info("Step 7: Agent self-reflection")
            reflection_result = self._perform_self_reflection(
                address_text,
                parsed_components,
                final_components,
                tool_results,
                confidence,
                geocode_data
            )
            
            # Apply corrections from self-reflection
            if reflection_result.get("needs_correction"):
                logger.info(f"Self-reflection identified issues: {reflection_result.get('issues')}")
                
                # Update components based on reflection
                if reflection_result.get("corrected_components"):
                    corrected = reflection_result["corrected_components"]
                    for key, value in corrected.items():
                        if value:
                            final_components[key] = value
                            logger.info(f"Self-correction applied: {key} = {value}")
                
                # Adjust confidence based on reflection
                if reflection_result.get("confidence_adjustment"):
                    old_confidence = confidence
                    confidence = max(0, min(100, confidence + reflection_result["confidence_adjustment"]))
                    logger.info(f"Confidence adjusted by reflection: {old_confidence}% → {confidence}%")
                
                # Add reflection notes to suggestions
                if reflection_result.get("reflection_notes"):
                    suggestions.extend(reflection_result["reflection_notes"])
            else:
                logger.info("Self-reflection: No issues found, validation looks good")
            
            # Format address
            formatted_parts = []
            if parsed.street_address:
                formatted_parts.append(parsed.street_address)
            if final_components.get("barangay"):
                formatted_parts.append(final_components["barangay"])
            if final_components.get("city"):
                formatted_parts.append(final_components["city"])
            if final_components.get("province"):
                formatted_parts.append(final_components["province"])
            if final_components.get("postal_code"):
                formatted_parts.append(final_components["postal_code"])
            formatted_parts.append("PH")
            
            formatted_address = ", ".join(formatted_parts)
            
            # Determine validity
            is_valid = confidence >= 70
            geocode_matched = geocode_data is not None
            delivery_success = bool(delivery_data and delivery_data.get("successful", 0) > 0)
            
            # Build geocode response
            geocode_response = GeocodeResponse()
            if geocode_data:
                geocode_response = GeocodeResponse(
                    lat=geocode_data.get("latitude", 0.0),
                    lng=geocode_data.get("longitude", 0.0),
                    place_id=geocode_data.get("place_id", ""),
                    formattedAddress=geocode_data.get("formatted_address", "")
                )
            
            # Build delivery history response
            delivery_history_response = DeliveryHistoryResponse()
            if delivery_data and delivery_data.get("found"):
                from schema import DeliveryHistoryAddress
                delivery_history_response = DeliveryHistoryResponse(
                    inputAddress=DeliveryHistoryAddress(
                        address=address_text,
                        status=delivery_data.get("latest_status", ""),
                        last_delivery_at="",
                        rts_reason=""
                    ),
                    formattedAddress=DeliveryHistoryAddress()
                )
            
            # Count tool types used
            tool_types = set(t.get("tool") for t in tool_results if t.get("tool"))
            reason_parts = [
                f"Intelligent LLM validation with {len(tool_results)} tool calls",
                f"Tools used: {', '.join(tool_types)}" if tool_types else "Direct matching only"
            ]
            
            return EnhancedAddressValidationResponse(
                id=validation_id,
                input=address_text,
                formattedAddress=formatted_address,
                verdict=VerdictResponse(
                    isValid=is_valid,
                    structureOk=bool(final_components.get("city") and final_components.get("province")),
                    psgcMatched=bool(final_components.get("province_code") and final_components.get("city_code")),
                    geocodeMatched=geocode_matched,
                    deliveryHistorySuccess=delivery_success,
                    confidence=float(confidence)
                ),
                structure=StructureResponse(
                    streetAddress=parsed.street_address or "",
                    barangay=final_components.get("barangay") or "",
                    city=final_components.get("city") or "",
                    province=final_components.get("province") or "",
                    country="PH",
                    postalCode=final_components.get("postal_code") or "",
                    formattedAddress=formatted_address
                ),
                psgc=PSGCResponse(
                    regionCode="",
                    provinceCode=final_components.get("province_code") or "",
                    cityMuniCode=final_components.get("city_code") or "",
                    barangayCode=final_components.get("barangay_code") or ""
                ),
                geocode=geocode_response,
                deliveryHistory=delivery_history_response,
                reason=reason_parts,
                suggestions=suggestions
            )
        
        except Exception as e:
            logger.error(f"Validation failed: {e}", exc_info=True)
            return EnhancedAddressValidationResponse(
                id=validation_id,
                input=address_text,
                formattedAddress=address_text,
                verdict=VerdictResponse(
                    isValid=False,
                    structureOk=False,
                    psgcMatched=False,
                    geocodeMatched=False,
                    deliveryHistorySuccess=False,
                    confidence=0.0
                ),
                structure=StructureResponse(
                    streetAddress="",
                    barangay="",
                    city="",
                    province="",
                    country="PH",
                    postalCode="",
                    formattedAddress=address_text
                ),
                psgc=PSGCResponse(regionCode="", provinceCode="", cityMuniCode="", barangayCode=""),
                geocode=GeocodeResponse(),
                deliveryHistory=DeliveryHistoryResponse(),
                reason=[f"Validation error: {str(e)}"],
                suggestions=["Please check the address format and try again"]
            )
