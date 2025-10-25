import os
import logging
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:

    API_TITLE = "Philippine Address Validator API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Validates and structures unstructured Philippine addresses using PSGC data"

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_TEMPERATURE = float(os.getenv("OPENAI_TEMPERATURE", "0"))

    PSGC_BASE_URL = os.getenv("PSGC_BASE_URL")
    PSGC_TIMEOUT = float(os.getenv("PSGC_TIMEOUT"))

    CITY_ABBREVIATIONS: Dict[str, str] = {
        'qc': 'quezon city',
        'q.c.': 'quezon city',
        'q.c': 'quezon city',
        'manila': 'city of manila',
        'makati': 'city of makati',
        'pasig': 'city of pasig',
        'taguig': 'city of taguig',
        'bgc': 'taguig', 
        'paranaque': 'city of paranaque',
        'las pinas': 'city of las pinas',
        'muntinlupa': 'city of muntinlupa',
    }

    PROVINCE_ALIASES: Dict[str, str] = {
        'compostela valley': 'davao de oro',
        'ncr': 'metro manila',
        'national capital region': 'metro manila',
    }

    @classmethod
    def validate(cls) -> bool:
        """
        Validates the configuration settings.

        Checks if the OPENAI_API_KEY is set.

        :raises ValueError: If OPENAI_API_KEY is not set.
        :return: True if the configuration is valid, False otherwise.
        """
        logger.info("Validating configuration settings...")
        if not cls.OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY is not set")
            raise ValueError("OPENAI_API_KEY is required. Set it in your .env file.")
        logger.info("Configuration validated successfully")
        return True


settings = Settings()
logger.info(f"Settings loaded: API={settings.API_TITLE}, Host={settings.HOST}:{settings.PORT}")
