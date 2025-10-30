import os
import csv
import time
import requests
import logging
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

API_URL = "http://localhost:8000/api/v2/agent-validator/validate-address"
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  


def get_supabase_client():
    """Initialize Supabase client"""
    try:
        from supabase import create_client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("✓ Supabase client initialized")
        return client
    except ImportError:
        logger.error("✗ Supabase library not installed. Run: pip install supabase")
        return None
    except Exception as e:
        logger.error(f"✗ Failed to initialize Supabase: {e}")
        return None


def fetch_delivery_history(client, limit: int = None) -> List[Dict[str, Any]]:
    """Fetch delivery history records from database"""
    try:
        if limit:
            logger.info(f"Fetching delivery history records (limit: {limit})...")
            response = client.table("delivery_history").select("*").limit(limit).execute()
        else:
            logger.info("Fetching ALL delivery history records from database...")
            response = client.table("delivery_history").select("*").execute()
        
        data = response.data if hasattr(response, "data") else response
        logger.info(f"✓ Retrieved {len(data)} delivery history records")
        return data
    except Exception as e:
        logger.error(f"✗ Failed to fetch delivery history: {e}")
        return []


def test_api_endpoint(address: str, timeout: int = 60) -> Dict[str, Any]:
    """
    Test the API endpoint with an address
    Returns: dict with response data and timing
    """
    payload = {"address": address}
    
    start_time = time.time()
    try:
        response = requests.post(API_URL, json=payload, timeout=timeout)
        end_time = time.time()
        
        response_time = round((end_time - start_time) * 1000, 2)  # in milliseconds
        
        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "data": result
            }
        else:
            return {
                "success": False,
                "status_code": response.status_code,
                "response_time_ms": response_time,
                "error": response.text,
                "data": None
            }
    except requests.exceptions.Timeout:
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)
        return {
            "success": False,
            "status_code": 0,
            "response_time_ms": response_time,
            "error": "Request timeout",
            "data": None
        }
    except Exception as e:
        end_time = time.time()
        response_time = round((end_time - start_time) * 1000, 2)
        return {
            "success": False,
            "status_code": 0,
            "response_time_ms": response_time,
            "error": str(e),
            "data": None
        }





def save_results_to_csv(results: List[Dict[str, Any]], filename: str = None):
    """Save test results to CSV file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"api_test_results_{timestamp}.csv"
    
    fieldnames = [
        "test_id",
        "original_address",
        "db_status",
        "db_failure_reason",
        "test_type",
        "test_address",
        "expected_valid",
        "response_time_ms",
        "api_status_code",
        "api_success",
        "validation_id",
        "is_valid",
        "confidence",
        "structure_ok",
        "psgc_matched",
        "geocode_matched",
        "delivery_history_success",
        "formatted_address",
        "street_address",
        "barangay",
        "city",
        "province",
        "postal_code",
        "latitude",
        "longitude",
        "delivery_input_address",
        "delivery_input_status",
        "delivery_formatted_address",
        "delivery_formatted_status",
        "reasons",
        "suggestions",
        "error_message"
    ]
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow(result)
        
        logger.info(f"✓ Results saved to: {filename}")
        return filename
    except Exception as e:
        logger.error(f"✗ Failed to save CSV: {e}")
        return None


def run_tests():
    """Main test runner"""
    logger.info("="*70)
    logger.info("API TESTING WITH DELIVERY HISTORY DATA")
    logger.info("="*70)
    
    client = get_supabase_client()
    if not client:
        logger.error("Cannot proceed without database connection")
        return
    
    history_records = fetch_delivery_history(client, limit=None)  
    if not history_records:
        logger.error("No delivery history records found")
        return
    
    logger.info(f"\nTesting API with {len(history_records)} delivery history records")
    total_tests = len(history_records)
    estimated_time = (total_tests * 3) / 60  
    logger.info(f"⏱️  Estimated time: {estimated_time:.1f} minutes")
    logger.info(f"API Endpoint: {API_URL}\n")
    
    # Create CSV file and write header immediately
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"api_test_results_{timestamp}.csv"
    
    fieldnames = [
        "test_id",
        "original_address",
        "db_status",
        "db_failure_reason",
        "test_type",
        "test_address",
        "expected_valid",
        "response_time_ms",
        "api_status_code",
        "api_success",
        "validation_id",
        "is_valid",
        "confidence",
        "structure_ok",
        "psgc_matched",
        "geocode_matched",
        "delivery_history_success",
        "formatted_address",
        "street_address",
        "barangay",
        "city",
        "province",
        "postal_code",
        "latitude",
        "longitude",
        "delivery_input_address",
        "delivery_input_status",
        "delivery_formatted_address",
        "delivery_formatted_status",
        "reasons",
        "suggestions",
        "error_message"
    ]
    
    csvfile = open(csv_filename, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()  
    
    logger.info(f"📝 Writing results to: {csv_filename}\n")
    
    all_results = []
    test_counter = 1
    start_time = time.time()
    
    for idx, record in enumerate(history_records, 1):
        original_address = record.get("address", "")
        db_status = record.get("status", "")
        db_failure_reason = record.get("failure_reason", "")
        
        progress = (idx / len(history_records)) * 100
        
        logger.info(f"\n[{idx}/{len(history_records)} - {progress:.1f}%] Testing: {original_address[:60]}...")
        logger.info(f"  DB Status: {db_status}")
        
        api_result = test_api_endpoint(original_address)
        
        logger.info(f"  ⏱  Response time: {api_result['response_time_ms']}ms")
        
        result_row = {
            "test_id": test_counter,
            "original_address": original_address,
            "db_status": db_status,
            "db_failure_reason": db_failure_reason or "",
            "test_type": "original",
            "test_address": original_address,
            "expected_valid": True,
            "response_time_ms": api_result["response_time_ms"],
            "api_status_code": api_result["status_code"],
            "api_success": api_result["success"],
        }
        
        if api_result["success"] and api_result["data"]:
            data = api_result["data"]
            verdict = data.get("verdict", {})
            structure = data.get("structure", {})
            psgc = data.get("psgc", {})
            geocode = data.get("geocode", {})
            delivery = data.get("deliveryHistory", {})
            
            result_row.update({
                "validation_id": data.get("id", ""),
                "is_valid": verdict.get("isValid", False),
                "confidence": verdict.get("confidence", 0),
                "structure_ok": verdict.get("structureOk", False),
                "psgc_matched": verdict.get("psgcMatched", False),
                "geocode_matched": verdict.get("geocodeMatched", False),
                "delivery_history_success": verdict.get("deliveryHistorySuccess", False),
                "formatted_address": data.get("formattedAddress", ""),
                "street_address": structure.get("streetAddress", ""),
                "barangay": structure.get("barangay", ""),
                "city": structure.get("city", ""),
                "province": structure.get("province", ""),
                "postal_code": structure.get("postalCode", ""),
                "latitude": geocode.get("lat", 0),
                "longitude": geocode.get("lng", 0),
                "delivery_input_address": delivery.get("inputAddress", {}).get("address", ""),
                "delivery_input_status": delivery.get("inputAddress", {}).get("status", ""),
                "delivery_formatted_address": delivery.get("formattedAddress", {}).get("address", ""),
                "delivery_formatted_status": delivery.get("formattedAddress", {}).get("status", ""),
                "reasons": "; ".join(data.get("reason", [])),
                "suggestions": "; ".join(data.get("suggestions", [])),
                "error_message": ""
            })
            
            logger.info(f"  ✓ Valid: {verdict.get('isValid')} | Confidence: {verdict.get('confidence')}%")
            
        else:
            result_row.update({
                "validation_id": "",
                "is_valid": False,
                "confidence": 0,
                "structure_ok": False,
                "psgc_matched": False,
                "geocode_matched": False,
                "delivery_history_success": False,
                "formatted_address": "",
                "street_address": "",
                "barangay": "",
                "city": "",
                "province": "",
                "postal_code": "",
                "latitude": 0,
                "longitude": 0,
                "delivery_input_address": "",
                "delivery_input_status": "",
                "delivery_formatted_address": "",
                "delivery_formatted_status": "",
                "reasons": "",
                "suggestions": "",
                "error_message": api_result.get("error", "")
            })
            
            logger.error(f"  ✗ Error: {api_result.get('error', 'Unknown error')}")
        
        all_results.append(result_row)
        
        # Write this row immediately to CSV
        writer.writerow(result_row)
        csvfile.flush()  # Ensure row is written to disk immediately
        
        test_counter += 1
        
        time.sleep(0.5)
    
    # Close CSV file
    csvfile.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    logger.info("\n" + "="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)
    
    total_tests = len(all_results)
    successful_tests = sum(1 for r in all_results if r["api_success"])
    valid_addresses = sum(1 for r in all_results if r["is_valid"])
    avg_response_time = sum(r["response_time_ms"] for r in all_results) / total_tests if total_tests > 0 else 0
    
    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Total Records from DB: {len(history_records)}")
    logger.info(f"Successful API Calls: {successful_tests}")
    logger.info(f"Valid Addresses: {valid_addresses}")
    logger.info(f"Average Response Time: {avg_response_time:.2f}ms")
    logger.info(f"Total Execution Time: {total_time/60:.2f} minutes")
    logger.info(f"Results saved to: {csv_filename}")
    
    return all_results


if __name__ == "__main__":
    try:
        logger.info("Starting API tests...")
        results = run_tests()
        logger.info("\n✓ Testing completed successfully!")
    except KeyboardInterrupt:
        logger.warning("\n⚠ Testing interrupted by user")
    except Exception as e:
        logger.error(f"\n✗ Testing failed: {e}", exc_info=True)
