"""Production FastAPI REST Application Entrypoint.

Tanzanian Social Commerce & Escrow Fraud Moderation Engine:
- Dual-Layer Detection (Vision-Language CLIP + Incremental River Model)
- Real-time mobile money and local rider delivery risk mitigation
- Continuous learning via /feedback and audit tracking in SQLite
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import get_orchestrator, router
from config.settings import DEFAULT_DB_FILE
from database.db import init_db
from services.moderation_layer1 import ContentModerationLayer1

# Configure structured application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle.

    Initializes database tables, pre-warms the ML orchestrator, River pipeline,
    and HuggingFace CLIP zero-shot vision model into memory.

    Parameters:
        app (FastAPI): Active FastAPI instance.

    Yields:
        None
    """
    logger.info("Initializing Tanzanian Fraud Moderation Engine...")
    init_db(DEFAULT_DB_FILE)
    # Pre-warm orchestrator and river pipeline
    orchestrator = get_orchestrator()
    # Pre-warm HuggingFace CLIP model in RAM
    logger.info("Pre-warming HuggingFace CLIP vision-language model...")
    ContentModerationLayer1.warmup_clip()
    logger.info("Orchestrator, River pipeline, and CLIP model ready in RAM. Active database: %s", DEFAULT_DB_FILE)
    yield
    logger.info("Shutting down Tanzanian Fraud Moderation Engine gracefully.")


app = FastAPI(
    title="Tanzanian Social Commerce Fraud & Moderation Engine",
    description=(
        "Production-ready, self-learning fraud detection and illegal goods moderation service "
        "tailored for Tanzanian social commerce, mobile money escrow (TZS), and rider deliveries."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web portals and mobile clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Register API Router
app.include_router(router)


@app.get("/", include_in_schema=False)
async def serve_workbench() -> FileResponse:
    """Serve the interactive Testing & Risk Workbench UI.

    Returns:
        FileResponse: static/index.html file.
    """
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        # Fallback if accessed before file creation
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"message": "Workbench UI is being prepared."},
        )
    return FileResponse(str(index_file))


@app.get("/health", tags=["Health"])
async def health_check() -> JSONResponse:
    """Health check endpoint to verify system availability.

    Returns:
        JSONResponse: Status OK and platform metadata.
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "Tanzanian Social Commerce Fraud Moderation Engine",
            "currency": "TZS",
            "version": "1.0.0",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler to ensure all unhandled errors return clean JSON.

    Parameters:
        request (Request): Current HTTP request.
        exc (Exception): Unhandled exception.

    Returns:
        JSONResponse: Standardized error response.
    """
    logger.error("Unhandled exception processing request %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred while processing the request.",
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
