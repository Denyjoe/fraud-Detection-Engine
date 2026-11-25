"""FastAPI Route Handlers.

Exposes REST endpoints for:
- Pre-listing moderation (Layer 1 + Layer 2)
- Real-time escrow transaction risk scoring
- Continuous online learning feedback
- System metrics and audit queries
"""

import base64
import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from database.repository import AuditLogRepository
from schemas.api_schemas import (
    AuditLogItem,
    AuditStatsResponse,
    FeedbackRequest,
    FeedbackResponse,
    ModerateListingRequest,
    ModerateListingResponse,
    SystemMetricsResponse,
    TransactionRiskRequest,
    TransactionRiskResponse,
)
from services.moderation_orchestrator import ModerationOrchestrator

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Fraud & Moderation Engine"])

# Global singleton orchestrator
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


@router.post(
    "/moderate-listing",
    response_model=ModerateListingResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderate Listing Pre-Upload (Layer 1 + Layer 2)",
)
async def moderate_listing(
    payload: ModerateListingRequest,
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> ModerateListingResponse:
    """Evaluate product listing for prohibited goods, image mismatch, and seller fraud risk.

    Parameters:
        payload (ModerateListingRequest): Listing metadata and optional base64 image.
        orchestrator (ModerationOrchestrator): Injected orchestrator service.

    Returns:
        ModerateListingResponse: Moderation findings, fraud probability, and ALLOW/HOLD/BLOCK status.
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
    except Exception as exc:
        logger.error("Error moderating listing %s: %s", payload.listing_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to moderate listing: {str(exc)}",
        )


@router.post(
    "/moderate-listing-upload",
    response_model=ModerateListingResponse,
    status_code=status.HTTP_200_OK,
    summary="Moderate Listing with Direct Multipart Image File Upload",
)
async def moderate_listing_upload(
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
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> ModerateListingResponse:
    """Evaluate product listing accepting multipart form-data and binary file upload.

    Parameters:
        listing_id (str): Unique listing ID.
        seller_id (str): Seller ID.
        title (str): Title.
        description (str): Description.
        price_tzs (float): Price in TZS.
        category (Optional[str]): Product category.
        account_age_days (float): Age of account in days.
        seller_completed_orders (int): Past completed orders.
        seller_dispute_rate (float): Dispute rate (0.0 to 1.0).
        hour_of_day (int): Hour (0 to 23).
        is_repeat_buyer_same_day (int): Velocity flag.
        image (Optional[UploadFile]): Uploaded photo file.
        orchestrator (ModerationOrchestrator): Injected orchestrator.

    Returns:
        ModerateListingResponse: Full moderation evaluation.
    """
    image_bytes = None
    if image is not None:
        image_bytes = await image.read()

    result = orchestrator.moderate_listing(
        listing_id=listing_id,
        seller_id=seller_id,
        title=title,
        description=description,
        price_tzs=price_tzs,
        account_age_days=account_age_days,
        seller_completed_orders=seller_completed_orders,
        seller_dispute_rate=seller_dispute_rate,
        image_data=image_bytes,
        hour_of_day=hour_of_day,
        is_repeat_buyer_same_day=is_repeat_buyer_same_day,
        category=category,
    )
    return ModerateListingResponse(**result)


@router.post(
    "/predict-transaction",
    response_model=TransactionRiskResponse,
    status_code=status.HTTP_200_OK,
    summary="Score Escrow Transaction Risk Pre-Lock",
)
async def predict_transaction(
    payload: TransactionRiskRequest,
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> TransactionRiskResponse:
    """Evaluate payment risk before locking buyer funds in escrow.

    Parameters:
        payload (TransactionRiskRequest): Transaction feature attributes.
        orchestrator (ModerationOrchestrator): Injected orchestrator.

    Returns:
        TransactionRiskResponse: Fraud probability and ALLOW/HOLD/BLOCK decision.
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
    except Exception as exc:
        logger.error("Error evaluating transaction %s: %s", payload.transaction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to score transaction: {str(exc)}",
        )


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit Online Learning Feedback (.learn_one)",
)
async def submit_feedback(
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
    except Exception as exc:
        logger.error("Error processing feedback for %s: %s", payload.transaction_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback: {str(exc)}",
        )


@router.get(
    "/metrics",
    response_model=SystemMetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Running System Metrics & Audit Stats",
)
async def get_metrics(
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> SystemMetricsResponse:
    """Retrieve cumulative statistics on inferences, feedbacks, and model accuracy.

    Parameters:
        orchestrator (ModerationOrchestrator): Injected orchestrator.

    Returns:
        SystemMetricsResponse: Aggregated metrics dictionary.
    """
    metrics = orchestrator.repository.get_metrics_summary()
    return SystemMetricsResponse(
        total_inferences=metrics.get("total_inferences", 0.0),
        total_feedbacks=metrics.get("total_feedbacks", 0.0),
        running_accuracy=metrics.get("running_accuracy", 0.0),
        running_f1=metrics.get("running_f1", 0.0),
        total_listings_moderated=metrics.get("total_listings_moderated", 0.0),
        total_transactions_evaluated=metrics.get("total_transactions_evaluated", 0.0),
        total_feedback_events=metrics.get("total_feedback_events", 0.0),
    )


@router.get(
    "/audit-stats",
    response_model=AuditStatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Detailed Audit Ledger and Model Health for Workbench",
)
async def get_audit_stats(
    orchestrator: ModerationOrchestrator = Depends(get_orchestrator),
) -> AuditStatsResponse:
    """Fetch recent audit logs alongside running accuracy and feedback telemetry.

    Parameters:
        orchestrator (ModerationOrchestrator): Injected orchestrator.

    Returns:
        AuditStatsResponse: Latest 15 unified logs with system metrics.
    """
    metrics = orchestrator.repository.get_metrics_summary()
    recent_logs_raw = orchestrator.repository.get_recent_audit_logs(limit=15)
    
    return AuditStatsResponse(
        metrics=SystemMetricsResponse(
            total_inferences=metrics.get("total_inferences", 0.0),
            total_feedbacks=metrics.get("total_feedbacks", 0.0),
            running_accuracy=metrics.get("running_accuracy", 0.0),
            running_f1=metrics.get("running_f1", 0.0),
            total_listings_moderated=metrics.get("total_listings_moderated", 0.0),
            total_transactions_evaluated=metrics.get("total_transactions_evaluated", 0.0),
            total_feedback_events=metrics.get("total_feedback_events", 0.0),
        ),
        recent_logs=[AuditLogItem(**item) for item in recent_logs_raw],
        model_type="River Incremental Pipeline (StandardScaler + LogisticRegression)",
        active_currency="TZS",
    )
