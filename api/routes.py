"""FastAPI Route Handlers.

Endpoints:
- ``POST /moderate``: content moderation for the store backend (multipart: title,
  description, files[]) -> ALLOW / REVIEW / BLOCK / ESCALATE.
- ``/api/v1/...``: pre-listing moderation with seller/price risk (Layer 1 + Layer 2),
  escrow transaction scoring, online-learning feedback, metrics and audit queries.

Handlers are plain ``def`` functions so FastAPI runs the CPU-heavy model inference in
its thread pool instead of blocking the event loop. When ``MODERATION_API_KEY`` is set,
every endpoint here requires the header ``X-API-Key``.
"""

import hmac
import logging
import os
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile, status

from config.settings import CONFIG
from schemas.api_schemas import (
    AuditLogItem,
    AuditStatsResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModerateListingRequest,
    ModerateListingResponse,
    ModerateResponse,
    SystemMetricsResponse,
    TransactionRiskRequest,
    TransactionRiskResponse,
)
from services.moderation_layer1 import MediaFile
from services.moderation_orchestrator import ModerationOrchestrator

logger = logging.getLogger(__name__)

API_CFG: Dict[str, Any] = CONFIG["api"]
_RESPONSE_KEYS = {"decision", "score", "categories", "reasons", "seller_message_sw", "seller_message_en",
                  "category", "checks", "processing_ms", "audit_log_id"}


def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Check ``X-API-Key`` against ``MODERATION_API_KEY`` (no check when the variable is unset).

    Parameters:
        x_api_key (Optional[str]): Header value.

    Raises:
        HTTPException: 401 when the key is missing or wrong.
    """
    expected = os.getenv("MODERATION_API_KEY")
    if not expected:
        return
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing X-API-Key")


router = APIRouter(prefix="/api/v1", tags=["Fraud & Moderation Engine"], dependencies=[Depends(require_api_key)])
moderation_router = APIRouter(tags=["Content Moderation"], dependencies=[Depends(require_api_key)])

_orchestrator_instance: Optional[ModerationOrchestrator] = None


def get_orchestrator() -> ModerationOrchestrator:
    """Dependency provider for the ModerationOrchestrator singleton.

    Returns:
        ModerationOrchestrator: Initialized orchestrator instance.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = ModerationOrchestrator()
    return _orchestrator_instance


def read_uploads(uploads: List[UploadFile]) -> List[MediaFile]:
    """Read uploaded files, enforcing the count and size limits from config.json.

    Parameters:
        uploads (List[UploadFile]): Multipart files.

    Returns:
        List[MediaFile]: File contents.

    Raises:
        HTTPException: 413 when there are too many files or one is too large.
    """
    uploads = [u for u in uploads if u is not None and (u.filename or "")]
    max_files = int(API_CFG.get("max_files", 10))
    max_bytes = int(float(API_CFG.get("max_file_mb", 25)) * 1024 * 1024)
    if len(uploads) > max_files:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"At most {max_files} files per listing")
    media: List[MediaFile] = []
    for upload in uploads:
        data = upload.file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                                f"{upload.filename} is larger than {API_CFG.get('max_file_mb', 25)} MB")
        media.append(MediaFile(upload.filename or "upload", data, upload.content_type))
    return media


@moderation_router.post("/moderate", response_model=ModerateResponse, summary="Moderate one listing (multipart)")
def moderate(
    title: str = Form(..., min_length=1, max_length=300),
    description: str = Form("", max_length=10000),
    files: List[UploadFile] = File(default=[]),
    listing_id: Optional[str] = Form(None),
    seller_id: Optional[str] = Form(None),
    category: Optional[str] = Form(None),
    price_tzs: Optional[float] = Form(None, ge=0),
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> ModerateResponse:
    """Check a listing's text, images and videos.

    Returns ALLOW / REVIEW / BLOCK / ESCALATE with reasons and a Swahili seller message.
    ESCALATE means *urgent human review* (child-safety); hide the listing immediately.
    """
    media = read_uploads(files)
    try:
        result = orchestrator.moderate_content(
            title=title, description=description, files=media,
            listing_id=listing_id or f"LIST-{uuid.uuid4().hex[:12]}", seller_id=seller_id or "unknown",
            category=category, price_tzs=price_tzs or 0.0,
        )
    except Exception:
        logger.exception("Moderation failed for listing %s", listing_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Moderation failed; please retry.")
    payload = {k: result[k] for k in _RESPONSE_KEYS}
    payload["details"] = {k: v for k, v in result.items() if k not in _RESPONSE_KEYS}
    return ModerateResponse(**payload)


@router.post(
    "/moderate-listing",
    response_model=ModerateListingResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderate Listing Pre-Upload (Layer 1 + Layer 2)",
)
def moderate_listing(
    payload: ModerateListingRequest,
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> ModerateListingResponse:
    """Evaluate product listing for prohibited goods, image mismatch, and seller fraud risk.

    Parameters:
        payload (ModerateListingRequest): Listing metadata and optional base64 image.
        orchestrator (ModerationOrchestrator): Injected orchestrator service.

    Returns:
        ModerateListingResponse: Moderation findings, fraud probability and ALLOW/REVIEW/BLOCK/ESCALATE.
    """
    try:
        result = orchestrator.moderate_listing(
            listing_id=payload.listing_id,
            seller_id=payload.seller_id,
            title=payload.title,
            description=payload.description,
            price_tzs=payload.price_tzs,
            account_age_days=payload.account_age_days,
            seller_completed_orders=payload.seller_completed_orders,
            seller_dispute_rate=payload.seller_dispute_rate,
            image_data=payload.image_base64,
            hour_of_day=payload.hour_of_day,
            is_repeat_buyer_same_day=payload.is_repeat_buyer_same_day,
            category=payload.category,
        )
        return ModerateListingResponse(**result)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"Invalid input: {type(exc).__name__}")
    except Exception:
        logger.exception("Error moderating listing %s", payload.listing_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to moderate listing.")


@router.post(
    "/moderate-listing-upload",
    response_model=ModerateListingResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderate Listing with Multipart Image/Video Upload (Layer 1 + Layer 2)",
)
def moderate_listing_upload(
    listing_id: str = Form(...),
    seller_id: str = Form(...),
    title: str = Form(...),
    description: str = Form(...),
    price_tzs: float = Form(...),
    category: Optional[str] = Form(None),
    account_age_days: float = Form(30.0),
    seller_completed_orders: int = Form(5),
    seller_dispute_rate: float = Form(0.0),
    hour_of_day: int = Form(12),
    is_repeat_buyer_same_day: int = Form(0),
    image: Optional[UploadFile] = File(None),
    files: List[UploadFile] = File(default=[]),
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> ModerateListingResponse:
    """Evaluate a listing from multipart form-data (``image`` and/or ``files[]``).

    Returns:
        ModerateListingResponse: Full moderation evaluation.
    """
    media = read_uploads(([image] if image is not None else []) + list(files))
    try:
        result = orchestrator.moderate_listing(
            listing_id=listing_id,
            seller_id=seller_id,
            title=title,
            description=description,
            price_tzs=price_tzs,
            account_age_days=account_age_days,
            seller_completed_orders=seller_completed_orders,
            seller_dispute_rate=seller_dispute_rate,
            hour_of_day=hour_of_day,
            is_repeat_buyer_same_day=is_repeat_buyer_same_day,
            category=category,
            files=media,
        )
    except Exception:
        logger.exception("Error moderating uploaded listing %s", listing_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to moderate listing.")
    return ModerateListingResponse(**result)


@router.post(
    "/predict-transaction",
    response_model=TransactionRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Score Escrow Transaction Risk Pre-Lock",
)
def predict_transaction(
    payload: TransactionRiskRequest,
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> TransactionRiskResponse:
    """Evaluate payment risk before locking buyer funds in escrow.

    Parameters:
        payload (TransactionRiskRequest): Transaction feature attributes.
        orchestrator (ModerationOrchestrator): Injected orchestrator.

    Returns:
        TransactionRiskResponse: Fraud probability and ALLOW/REVIEW/BLOCK decision.
    """
    try:
        result = orchestrator.evaluate_transaction(
            transaction_id=payload.transaction_id,
            seller_id=payload.seller_id,
            buyer_id=payload.buyer_id or "anonymous",
            price_tzs=payload.price_tzs,
            hour_of_day=payload.hour_of_day,
            account_age_days=payload.account_age_days,
            seller_completed_orders=payload.seller_completed_orders,
            seller_dispute_rate=payload.seller_dispute_rate,
            image_text_mismatch_score=payload.image_text_mismatch_score,
            swahili_slang_risk_score=payload.swahili_slang_risk_score,
            is_repeat_buyer_same_day=payload.is_repeat_buyer_same_day,
            category=payload.category,
            price_to_category_median_ratio=payload.price_to_category_median_ratio,
        )
        return TransactionRiskResponse(**result)
    except Exception:
        logger.exception("Error evaluating transaction %s", payload.transaction_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to score transaction.")


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Online Learning Feedback (.learn_one)",
)
def submit_feedback(
    payload: FeedbackRequest,
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> FeedbackResponse:
    """Deliver ground-truth feedback to update the River online learning model in real-time.

    Parameters:
        payload (FeedbackRequest): Actual label (0 = Legit, 1 = Fraud) and feature set.
        orchestrator (ModerationOrchestrator): Injected orchestrator.

    Returns:
        FeedbackResponse: Comparison of prediction before and after online model update.
    """
    try:
        result = orchestrator.process_feedback(
            transaction_id=payload.transaction_id,
            features=payload.features,
            actual_label=payload.actual_label,
            comment=payload.comment,
        )
        return FeedbackResponse(**result)
    except Exception:
        logger.exception("Error processing feedback for %s", payload.transaction_id)
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, "Failed to process feedback.")


def _metrics(orchestrator: ModerationOrchestrator) -> SystemMetricsResponse:
    """Build the metrics response from the repository."""
    metrics = orchestrator.repository.get_metrics_summary()
    return SystemMetricsResponse(**{key: metrics.get(key, 0.0) for key in SystemMetricsResponse.model_fields})


@router.get("/metrics", response_model=SystemMetricsResponse, summary="Get Running System Metrics & Audit Stats")
def get_metrics(orchestrator: ModerationOrchestrator = Depends(get_orchestrator)) -> SystemMetricsResponse:
    """Retrieve cumulative statistics on inferences, feedbacks, and model accuracy."""
    return _metrics(orchestrator)


@router.get("/audit-stats", response_model=AuditStatsResponse, summary="Get Audit Ledger and Model Health")
def get_audit_stats(orchestrator: ModerationOrchestrator = Depends(get_orchestrator)) -> AuditStatsResponse:
    """Fetch recent audit logs alongside running accuracy and feedback telemetry."""
    recent_logs_raw = orchestrator.repository.get_recent_audit_logs(limit=15)
    return AuditStatsResponse(
        metrics=_metrics(orchestrator),
        recent_logs=[AuditLogItem(**item) for item in recent_logs_raw],
        model_type="River Incremental Pipeline (StandardScaler + LogisticRegression)",
        active_currency="TZS",
    )
