"""Pydantic Request and Response Schemas.

Defines validated data models for listing moderation, escrow transaction scoring,
and online feedback updates for the Tanzanian Fraud & Illegal Goods Moderation API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class ModerateListingRequest(BaseModel):
    """Request payload for pre-listing illegal goods and fraud moderation."""

    listing_id: str = Field(..., description="Unique product listing identifier (e.g. LIST-9921)")
    seller_id: str = Field(..., description="Seller user ID (e.g. USR-KARIAKOO-01)")
    title: str = Field(..., min_length=2, max_length=255, description="Product listing title")
    description: str = Field(..., min_length=2, description="Product description in English or Swahili")
    price_tzs: float = Field(..., gt=0, description="Listing price in Tanzanian Shillings (TZS)")
    category: Optional[str] = Field(default=None, description="Category: fashion, electronics, groceries, home_goods, beauty, general (auto-inferred if omitted)")
    account_age_days: float = Field(default=30.0, ge=0, description="Seller account age in days")
    seller_completed_orders: int = Field(default=5, ge=0, description="Total completed orders by seller")
    seller_dispute_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Seller dispute rate (0.0 to 1.0)")
    image_base64: Optional[str] = Field(default=None, description="Optional Base64 encoded product photo")
    hour_of_day: int = Field(default=12, ge=0, le=23, description="Hour of listing submission (0 to 23)")
    is_repeat_buyer_same_day: int = Field(default=0, ge=0, le=1, description="High-velocity buyer indicator")

    @field_validator("title", "description")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """Validate that textual fields are not blank or purely whitespace."""
        if not value or not value.strip():
            raise ValueError("Field cannot be empty or solely whitespace")
        return value.strip()


class ModerateListingResponse(BaseModel):
    """Response payload for pre-listing moderation inspection."""

    listing_id: str
    seller_id: str
    category: str = Field(..., description="Resolved retail category")
    price_to_category_median_ratio: float = Field(..., description="Normalized price ratio against category median")
    decision: str = Field(..., description="Actionable decision: ALLOW, HOLD_FOR_REVIEW, or BLOCK")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Overall fraud probability (0.0 to 1.0)")
    risk_reasons: List[str] = Field(default_factory=list, description="Explanatory reasons if flagged")
    seller_claimed_concept: Optional[str] = Field(default=None, description="Inferred/translated English visual concept")
    top_visual_label: Optional[str] = Field(default=None, description="Top detected visual concept from zero-shot CLIP")
    top_visual_probability: Optional[float] = Field(default=None, description="Probability of top visual label")
    visual_comparison_status: Optional[str] = Field(default=None, description="Visual-text comparison status: MATCH, MISMATCH, CONTRABAND, NO_IMAGE")
    visual_comparison_detail: Optional[str] = Field(default=None, description="Detailed visual comparison rationale")
    visual_contraband_detected: bool = Field(default=False, description="Flag indicating contraband/weapons/NSFW in image")
    layer1_details: Dict[str, Any] = Field(..., description="Detailed Layer 1 vision & slang breakdown")
    layer2_features: Dict[str, Any] = Field(..., description="Sanitized feature vector sent to River model")
    audit_log_id: int = Field(..., description="Database audit log entry row ID")


class TransactionRiskRequest(BaseModel):
    """Request payload for real-time escrow transaction risk evaluation."""

    transaction_id: str = Field(..., description="Unique escrow transaction ID (e.g. TX-TZS-4481)")
    seller_id: str = Field(..., description="Seller user ID")
    buyer_id: Optional[str] = Field(default=None, description="Buyer user ID")
    price_tzs: float = Field(..., gt=0, description="Escrow payment amount in TZS")
    category: Optional[str] = Field(default=None, description="Retail category")
    hour_of_day: int = Field(default=12, ge=0, le=23, description="Local transaction hour (0-23)")
    account_age_days: float = Field(default=30.0, ge=0, description="Seller account age in days")
    seller_completed_orders: int = Field(default=10, ge=0, description="Total completed orders by seller")
    seller_dispute_rate: float = Field(default=0.0, ge=0.0, le=1.0, description="Dispute rate (0.0 to 1.0)")
    image_text_mismatch_score: float = Field(default=0.10, ge=0.0, le=1.0, description="Vision-text mismatch")
    swahili_slang_risk_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Prohibited slang risk")
    is_repeat_buyer_same_day: int = Field(default=0, ge=0, le=1, description="Repeat purchase velocity flag")
    price_to_category_median_ratio: Optional[float] = Field(default=None, description="Direct ratio if precomputed")


class TransactionRiskResponse(BaseModel):
    """Response payload for escrow risk evaluation before locking funds."""

    transaction_id: str
    category: str = Field(..., description="Retail category")
    price_to_category_median_ratio: float = Field(..., description="Price to category median ratio")
    decision: str = Field(..., description="Decision: ALLOW, HOLD_FOR_REVIEW, or BLOCK")
    fraud_probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability (0.0 to 1.0)")
    features: Dict[str, Any]
    audit_log_id: int


class FeedbackRequest(BaseModel):
    """Request payload for continuous online learning feedback loop."""

    transaction_id: str = Field(..., description="Transaction or listing identifier receiving feedback")
    actual_label: int = Field(..., ge=0, le=1, description="Ground truth label: 0 = Legit/OTP Confirmed, 1 = Fraud/Illegal")
    features: Dict[str, Any] = Field(..., description="Original transaction feature dictionary")
    comment: Optional[str] = Field(default=None, description="Optional administrative notes or OTP status")


class FeedbackResponse(BaseModel):
    """Response payload detailing online model weight adjustment."""

    transaction_id: str
    actual_label: int
    previous_probability: float = Field(..., description="Model probability before feedback update")
    updated_probability: float = Field(..., description="Model probability after online learning step")
    delta: float = Field(..., description="Probability shift caused by feedback")
    feedback_log_id: int


class SystemMetricsResponse(BaseModel):
    """Response payload containing system health, inference count, and running accuracy."""

    total_inferences: float
    total_feedbacks: float
    running_accuracy: float
    running_f1: float
    total_listings_moderated: float
    total_transactions_evaluated: float
    total_feedback_events: float


class AuditLogItem(BaseModel):
    """Individual audit log record for UI ledger display."""

    event_type: str = Field(..., description="LISTING or TRANSACTION")
    reference_id: str = Field(..., description="Listing or transaction identifier")
    seller_id: str = Field(..., description="Seller user identifier")
    detail: str = Field(..., description="Title or event description")
    price_tzs: float = Field(..., description="Amount in TZS")
    fraud_probability: float = Field(..., description="Model calculated fraud probability")
    decision: str = Field(..., description="ALLOW, HOLD_FOR_REVIEW, or BLOCK")
    created_at: str = Field(..., description="Timestamp of the event")


class AuditStatsResponse(BaseModel):
    """Aggregated audit statistics and recent logs for the UI workbench."""

    metrics: SystemMetricsResponse
    recent_logs: List[AuditLogItem]
    model_type: str = Field(default="River Incremental Pipeline (StandardScaler + LogisticRegression)")
    active_currency: str = Field(default="TZS")

