"""Production FastAPI REST Application Entrypoint.

Tanzanian Social Commerce Content Moderation & Escrow Fraud Engine:
- ``POST /moderate`` for the store backend (text + images + videos)
- ``GET /health`` with per-model status
- Layer 2 escrow risk scoring, online-learning feedback and SQLite audit log

Run:  python -m uvicorn main:app --host 127.0.0.1 --port 8000
Set ``MODERATION_WARMUP=1`` to load all models at start-up instead of on first use.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routes import get_orchestrator, moderation_router, router
from config.settings import CONFIG, DEFAULT_DB_FILE, enable_utf8_console
from database.db import init_db
from services import model_registry
from services.child_safety import csam_matcher_configured
from services.fusion import meta_model_status
from services.lexicon import get_lexicon

enable_utf8_console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fraud_api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
APP_VERSION = "2.0.0"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise the database, lexicon and orchestrator; optionally warm up the models.

    Parameters:
        app (FastAPI): Active FastAPI instance.

    Yields:
        None
    """
    logger.info("Initializing Tanzanian Fraud Moderation Engine (device: %s)...", model_registry.get_device())
    init_db(DEFAULT_DB_FILE)
    get_lexicon()
    get_orchestrator()
    if not os.getenv("MODERATION_API_KEY"):
        logger.warning("MODERATION_API_KEY is not set: the API accepts unauthenticated requests.")
    if not csam_matcher_configured():
        logger.warning("CSAM hash matching (PhotoDNA / Thorn Safer) is NOT configured. Connect it before launch.")
    if os.getenv("MODERATION_WARMUP", "").lower() in ("1", "true", "yes"):
        model_registry.warmup()
    logger.info("Engine ready. Database: %s", DEFAULT_DB_FILE)
    yield
    logger.info("Shutting down Tanzanian Fraud Moderation Engine gracefully.")


app = FastAPI(
    title="Tanzanian Social Commerce Fraud & Moderation Engine",
    description=(
        "Content moderation (Swahili/English prohibited goods, nudity, photo mismatch, OCR, video) "
        "and escrow fraud scoring for Tanzanian social commerce."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

_origins = list(CONFIG["api"].get("cors_origins", []))
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_credentials=False,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.include_router(moderation_router)
app.include_router(router)


@app.get("/", include_in_schema=False, response_model=None)
async def serve_workbench() -> FileResponse | JSONResponse:
    """Serve the interactive Risk Workbench UI (static/index.html)."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"message": "Workbench UI not found."})
    return FileResponse(str(index_file))


@app.get("/health", tags=["Health"])
def health_check() -> JSONResponse:
    """Service health plus the status of every model/check.

    Model status values: ``ready``, ``not_loaded`` (loads on first use), ``disabled``
    (switched off in config.json) or ``unavailable`` (failed to load: that check is off).
    """
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "healthy",
            "service": "Tanzanian Social Commerce Fraud Moderation Engine",
            "version": APP_VERSION,
            "currency": "TZS",
            "device": model_registry.get_device(),
            "lexicon_categories": sorted(get_lexicon().categories),
            "models": model_registry.model_status(),
            "meta_model": meta_model_status(),
            "csam_hash_matching": "configured" if csam_matcher_configured() else "NOT_CONFIGURED",
            "auth": "api_key" if os.getenv("MODERATION_API_KEY") else "none",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return clean JSON for unhandled errors without leaking internals.

    Parameters:
        request (Request): Current HTTP request.
        exc (Exception): Unhandled exception.

    Returns:
        JSONResponse: Standardized error response.
    """
    logger.error("Unhandled exception processing request %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "InternalServerError", "message": "An unexpected error occurred.",
                 "path": request.url.path},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "8000")))
