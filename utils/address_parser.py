from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from schema import ParsedAddress
from utils.config import settings
from loguru import logger


class AddressParser:
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes an AddressParser instance.

        :param api_key: Optional OpenAI API key. If not provided, uses the environment variable OPENAI_API_KEY.

        :raises ValueError: If OpenAI API key is not provided.
        """
        self.api_key = api_key or settings.OPENAI_API_KEY
        if not self.api_key:
            logger.error("OpenAI API key is missing")
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")

        logger.info(f"Initializing AddressParser with model: {settings.OPENAI_MODEL}")
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            api_key=self.api_key
        )

        self.parser = PydanticOutputParser(pydantic_object=ParsedAddress)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at parsing Philippine addresses. Your task is to extract structured address components from unstructured text.

                Philippine address hierarchy:
                - Region (e.g., "NCR", "Region IV-A", "Metro Manila")
                - Province (e.g., "Metro Manila", "Cavite", "Batangas")
                - City/Municipality (e.g., "Quezon City", "Makati", "City of Manila")
                - Barangay (e.g., "Libis", "Baclaran", "San Antonio", "Barangay 175")
                - Street Address (e.g., "Unit 405, 23rd Street", "123 Main Ave")
                - Postal Code (e.g., "1102", "4000")

                Important rules:
                1. For Metro Manila/NCR addresses, the province is "Metro Manila" or "National Capital Region"
                2. Common abbreviations: QC = Quezon City, Manila = City of Manila
                3. Remove prefixes like "Barangay", "Brgy.", "City of" when extracting the base name
                4. Extract the street-level details (unit numbers, building names, street names) as street_address
                5. Postal codes are typically 4-digit numbers in the Philippines
                6. If a component cannot be identified, set it to null
                7. Normalize casing: use Title Case for proper nouns
                8. For cities/municipalities, determine which one it is based on context
                
                CRITICAL BARANGAY EXTRACTION RULES:
                9. Some cities (like Caloocan, Manila, Navotas) use NUMBERED barangays (e.g., "Barangay 1", "Barangay 175")
                10. When you see a standalone number (especially 1-188 for Caloocan, 1-306 for Manila) in the address, it likely refers to a barangay number
                11. District/subdivision names (like "Camarin", "Tondo", "Sampaloc") are NOT official barangay names - look for numbers nearby
                12. If you see both a district name AND a number, prioritize the NUMBER as the barangay (e.g., "Camarin 175" = "Barangay 175")
                13. Format numbered barangays as "Barangay [number]" (e.g., "Barangay 175", not just "175")
                14. Examples:
                    - "Camarin 175 Caloocan" → barangay: "Barangay 175"
                    - "Block 3 Lot 11 Franville II Camarin 175 Caloocan" → barangay: "Barangay 175"
                    - "Tondo 105 Manila" → barangay: "Barangay 105"
                    - "Barangay Libis Quezon City" → barangay: "Libis"

                {format_instructions}

                Parse the following Philippine address:"""),
                            ("user", "{address}")
                        ])

    async def parse(self, address_text: str) -> ParsedAddress:
        """
        Parse an unstructured address text into structured components.

        :param address_text: Raw address text (e.g., "Unit 405, 23rd St., Barangay Libis, Quezon City, Metro Manila")
        :return: Structured address with validation status.
        :raises Exception: If any error occurs during the parsing process.
        """
        logger.info(f"Parsing address with LLM: {address_text[:50]}...")
        try:
            chain = self.prompt | self.llm | self.parser

            result = await chain.ainvoke({
                "address": address_text,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            logger.debug(f"LLM parsing successful: {result}")
            return result

        except Exception as e:
            logger.error(f"Error parsing address with LLM: {e}", exc_info=True)
            return ParsedAddress()

    def parse_sync(self, address_text: str) -> ParsedAddress:
        """
        Synchronous version of parse.

        Parse unstructured address text into structured components.

        :param address_text: Raw address text (e.g., "Unit 405, 23rd St., Barangay Libis, Quezon City, Metro Manila")
        :return: Structured address with validation status.
        :raises Exception: If any error occurs during the parsing process.
        """
        logger.info(f"Parsing address with LLM (sync): {address_text[:50]}...")
        try:
            chain = self.prompt | self.llm | self.parser

            result = chain.invoke({
                "address": address_text,
                "format_instructions": self.parser.get_format_instructions()
            })

            logger.debug(f"LLM parsing successful (sync): {result}")
            return result

        except Exception as e:
            logger.error(f"Error parsing address with LLM (sync): {e}", exc_info=True)
            return ParsedAddress()
