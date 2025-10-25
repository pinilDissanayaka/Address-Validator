from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from utils.config import settings
from routes import validate_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION
)

app.include_router(validate_router)

logger.info(f"FastAPI application initialized: {settings.API_TITLE} v{settings.API_VERSION}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """
    Root endpoint for the API.

    Returns a JSON object with the following fields:

    - service: The title of the service.
    - version: The version of the service.

    This endpoint is useful for checking if the service is up and running.
    """
    logger.debug("Root endpoint accessed")
    return JSONResponse(content={
        "service": settings.API_TITLE,
        "version": settings.API_VERSION,
    }, status_code=200)



if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)