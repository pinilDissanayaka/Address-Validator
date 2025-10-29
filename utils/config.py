import os
import logging
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class Settings:

    API_TITLE = "Philippine Address Validator API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "Validates and structures unstructured Philippine addresses using PhilAtlas data"

    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", "8000"))

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0"))

    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_API_KEY") 

    PHILATLAS_BASE_URL = os.getenv("PHILATLAS_BASE_URL", "https://philatlas.ph")
    PHILATLAS_TIMEOUT = float(os.getenv("PHILATLAS_TIMEOUT", "10.0"))

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

        Checks if the GEMINI_API_KEY and GOOGLE_MAPS_API_KEY are set.

        :raises ValueError: If required API keys are not set.
        :return: True if the configuration is valid, False otherwise.
        """
        logger.info("Validating configuration settings...")
        if not cls.GEMINI_API_KEY:
            logger.error("GEMINI_API_KEY is not set")
            raise ValueError("GEMINI_API_KEY is required. Set it in your .env file.")
        if not cls.GOOGLE_MAPS_API_KEY:
            logger.warning("GOOGLE_MAPS_API_KEY is not set - geocoding features will be limited")
        logger.info("Configuration validated successfully")
        return True


settings = Settings()
logger.info(f"Settings loaded: API={settings.API_TITLE}, Host={settings.HOST}:{settings.PORT}")
