"""Test validation improvements"""
import asyncio
from utils.llm_agent_validator import LLMAddressValidatorAgent
from utils.address_parser import AddressParser
from utils.psgc_api_client import PSGCAPIClient
from utils.philatlas_client import PhilAtlasClient
from core.gmaps_integration import GoogleMapsValidator
import core.database_client as database_client

async def test_address(address_text):
    parser = AddressParser()
    psgc_client = PSGCAPIClient()
    philatlas_client = PhilAtlasClient()
    gmaps_validator = GoogleMapsValidator()
    
    validator = LLMAddressValidatorAgent(
        parser=parser,
        psgc_client=psgc_client,
        philatlas_client=philatlas_client,
        gmaps_validator=gmaps_validator,
        db_client=database_client
    )
    
    result = await validator.validate_address(address_text)
    
    print(f"\nAddress: {address_text}")
    print(f"City: {result.structure.city}")
    print(f"Province: {result.structure.province}")
    print(f"Barangay: {result.structure.barangay}")
    print(f"Is Valid: {result.verdict.isValid}")
    print(f"Confidence: {result.verdict.confidence}%")
    print(f"Formatted: {result.formattedAddress}")
    return result

async def main():
    # Test cases that were problematic
    test_cases = [
        "C/O MYRNA LEDESMA DONDONS SNACK BAR ,FUNDA-DALIPE SAN JOSE DE BUENAVISTA,ANTIQUE",
        "B13 L26 DOVE ST. MORNING MIST UPPER CARMEN CDOC,CARMEN CAGAYAN DE ORO CITY,MISAMIS ORIENTAL",
        "890 SAINT JUDE STREET, BARANGAY HIPODROMO CEBU CITY PH 6000",
    ]
    
    for address in test_cases:
        await test_address(address)
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
