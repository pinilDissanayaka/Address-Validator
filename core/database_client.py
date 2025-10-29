import os
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

SUPABASE_URL: str = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY")

SUPABASE_CLIENT = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import Client, create_client
        SUPABASE_CLIENT: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully")
    except ImportError:
        logger.warning("Supabase library not installed - database features unavailable")
    except Exception as e:
        logger.warning(f"Failed to initialize Supabase client: {e}")
else:
    logger.warning("SUPABASE_URL or SUPABASE_KEY not set - database features unavailable")


def is_database_available() -> bool:
    """Check if database connection is available."""
    return SUPABASE_CLIENT is not None


def get_provinces() -> List[str]:
    """Get list of all province names."""
    if not SUPABASE_CLIENT:
        logger.debug("Database not available for provinces lookup")
        return []
    
    try:
        res = SUPABASE_CLIENT.table("psgc_province").select("province").order("province_id").execute()
        data = res.data if hasattr(res, "data") else res
        return [item["province"] for item in data] if data else []
    except Exception as e:
        logger.exception("Failed to fetch provinces")
        return []


def get_province_details(province_name: str) -> Dict[str, Any]:
    """Get detailed information about a province."""
    if not SUPABASE_CLIENT:
        return {}
    
    try:
        query = SUPABASE_CLIENT.table("psgc_province").select("*").filter("province", "ilike", province_name)
        res = query.execute()
        data = res.data if hasattr(res, "data") else res
        return data[0] if len(data) > 0 else {}
    except Exception as e:
        logger.exception(f"Failed to fetch province details for {province_name}")
        return {}


def get_cities(province_name: Optional[str] = None) -> List[str]:
    """Get list of city names, optionally filtered by province."""
    if not SUPABASE_CLIENT:
        logger.debug("Database not available for cities lookup")
        return []
    
    try:
        query = SUPABASE_CLIENT.table("psgc_city").select("city, province_id").order("city_id")
        
        if province_name:
            province_details = get_province_details(province_name)
            if province_details and "province_id" in province_details:
                query = query.eq("province_id", province_details["province_id"])
        
        res = query.execute()
        data = res.data if hasattr(res, "data") else res
        return [item["city"] for item in data] if data else []
    except Exception as e:
        logger.exception(f"Failed to fetch cities for province {province_name}")
        return []


def get_city_details(city_name: str) -> Dict[str, Any]:
    """Get detailed information about a city."""
    if not SUPABASE_CLIENT:
        return {}
    
    try:
        _city = city_name.replace("city of ", "").replace("city ", "").strip()
        query = SUPABASE_CLIENT.table("psgc_city").select("*").filter("city", "ilike", _city)
        res = query.execute()
        data = res.data if hasattr(res, "data") else res
        return data[0] if len(data) > 0 else {}
    except Exception as e:
        logger.exception(f"Failed to fetch city details for {city_name}")
        return {}


def get_barangays(city_name: Optional[str] = None) -> List[str]:
    """Get list of barangay names, optionally filtered by city."""
    if not SUPABASE_CLIENT:
        logger.debug("Database not available for barangays lookup")
        return []
    
    try:
        query = SUPABASE_CLIENT.table("psgc_barangay").select("barangay, city_id").order("barangay")
        
        if city_name:
            city_details = get_city_details(city_name)
            if city_details and "city_id" in city_details:
                query = query.eq("city_id", city_details["city_id"])
        
        res = query.execute()
        data = res.data if hasattr(res, "data") else res
        return [item["barangay"] for item in data] if data else []
    except Exception as e:
        logger.exception(f"Failed to fetch barangays for city {city_name}")
        return []


def get_barangay_details(barangay_name: str) -> Dict[str, Any]:
    """Get detailed information about a barangay including postal code."""
    if not SUPABASE_CLIENT:
        return {}
    
    try:
        _barangay = barangay_name.replace("brgy ", "").replace("brgy.", "").replace("barangay ", "").strip()
        query = SUPABASE_CLIENT.table("psgc_barangay").select("*").filter("barangay", "ilike", _barangay)
        res = query.execute()
        data = res.data if hasattr(res, "data") else res
        return data[0] if len(data) > 0 else {}
    except Exception as e:
        logger.exception(f"Failed to fetch barangay details for {barangay_name}")
        return {}


def get_delivery_history(address: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Get delivery history for an address."""
    if not SUPABASE_CLIENT:
        logger.debug("Database not available for delivery history lookup")
        return []
    
    try:
        exact_result = (
            SUPABASE_CLIENT.table("delivery_history").select("*")
            .filter("address", "ilike", address)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        exact_data: List[Dict[str, Any]] = exact_result.data if hasattr(exact_result, "data") else exact_result
        if isinstance(exact_data, list) and len(exact_data) > 0:
            logger.debug(f"Delivery history match returned {len(exact_data)} rows")
            return exact_data
        
        return []
    except Exception as e:
        logger.exception("Failed to fetch delivery history")
        return []


def insert_delivery_history(address: str, status: str, failure_reason: Optional[str] = None) -> Dict[str, Any]:
    """Insert a delivery history record."""
    if not SUPABASE_CLIENT:
        logger.warning("Database not available for delivery history insert")
        return {}
    
    try:
        payload: Dict[str, Optional[str]] = {
            "address": address,
            "status": status,
            "failure_reason": failure_reason,
        }
        logger.info(f"Inserting delivery_history: {payload}")
        result = SUPABASE_CLIENT.table("delivery_history").insert(payload).execute()
        data = result.data if hasattr(result, "data") else result
        return data[0] if isinstance(data, list) and data else data
    except Exception as e:
        logger.exception("Failed to insert delivery history")
        return {}
