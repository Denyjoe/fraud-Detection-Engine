"""Moderation and Fraud Detection Orchestrator.

Combines Layer 1 (content moderation: lexicon, text ML, images, videos, OCR, child
safety -> ALLOW / REVIEW / BLOCK / ESCALATE) with Layer 2 (River incremental model with
category-aware price benchmarking) under the production guardrails (Rules A, B, C).
Layer 2 can only make a content decision stricter, never relax a BLOCK or ESCALATE.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import (
    ALLOW_UPPER_BOUND,
    CATEGORY_MEDIANS_TZS,
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_ESCALATE,
    DECISION_REVIEW,
    DECISION_SEVERITY,
    DEFAULT_CATEGORY,
    HIGH_PRICE_RATIO,
    NEW_ACCOUNT_DAYS,
    RULE_A_DISPUTE_THRESHOLD,
    RULE_A_FRAUD_PROB_THRESHOLD,
    RULE_A_MISMATCH_THRESHOLD,
    RULE_B_ALLOW_CAP,
    TRUST_DAMPENING_FACTOR,
    TRUSTED_ACCOUNT_AGE_DAYS,
    TRUSTED_COMPLETED_ORDERS,
    TRUSTED_DISPUTE_RATE_MAX,
)
from database.repository import AuditLogRepository
from services.fraud_model_layer2 import OnlineFraudModelLayer2
from services.moderation_layer1 import ContentModerationLayer1, MediaFile

logger = logging.getLogger(__name__)


class ModerationOrchestrator:
    """Unified service orchestrating dual-layer moderation and audit logging."""

    def __init__(
        self,
        layer1_moderator: Optional[ContentModerationLayer1] = None,
        layer2_model: Optional[OnlineFraudModelLayer2] = None,
        repository: Optional[AuditLogRepository] = None,
    ) -> None:
        """Initialize the orchestrator with required sub-services.

        Parameters:
            layer1_moderator (Optional[ContentModerationLayer1]): Layer 1 inspector.
            layer2_model (Optional[OnlineFraudModelLayer2]): Layer 2 River model.
            repository (Optional[AuditLogRepository]): Database audit repository.
        """
        self.layer1 = layer1_moderator or ContentModerationLayer1()
        self.layer2 = layer2_model or OnlineFraudModelLayer2()
        self.repository = repository or AuditLogRepository()

    def moderate_content(
        self,
        title: str,
        description: str,
        files: Sequence[MediaFile] = (),
        listing_id: str = "",
        seller_id: str = "",
        category: Optional[str] = None,
        price_tzs: float = 0.0,
    ) -> Dict[str, Any]:
        """Content-only moderation (used by ``POST /moderate``) with audit logging.

        Parameters:
            title (str): Listing title.
            description (str): Listing description.
            files (Sequence[MediaFile]): Uploaded images/videos.
            listing_id (str): Store listing id (for the audit log).
            seller_id (str): Seller id (for the audit log).
            category (Optional[str]): Optional explicit category.
            price_tzs (float): Optional price, stored in the audit log.

        Returns:
            Dict[str, Any]: Layer 1 result plus ``audit_log_id``.
        """
        result = self.layer1.moderate(title, description, files, explicit_category=category)
        result["audit_log_id"] = self._log(listing_id, seller_id, title, price_tzs, result, result["score"],
                                           result["decision"])
        return result

    def moderate_listing(
        self,
        listing_id: str,
        seller_id: str,
        title: str,
        description: str,
        price_tzs: float,
        account_age_days: float = 30.0,
        seller_completed_orders: int = 5,
        seller_dispute_rate: float = 0.0,
        image_data: Optional[Any] = None,
        hour_of_day: int = 12,
        is_repeat_buyer_same_day: int = 0,
        category: Optional[str] = None,
        files: Sequence[MediaFile] = (),
    ) -> Dict[str, Any]:
        """Pre-listing moderation combining Layer 1 content checks with Layer 2 seller/price risk.

        Parameters:
            listing_id (str): Unique listing ID.
            seller_id (str): Seller identifier.
            title (str): Product listing title.
            description (str): Detailed product description.
            price_tzs (float): Price in Tanzanian Shillings.
            account_age_days (float): Seller's account age.
            seller_completed_orders (int): Past successfully delivered orders.
            seller_dispute_rate (float): Seller dispute percentage (0.0 to 1.0).
            image_data (Optional[Any]): One image as base64 string or bytes (legacy field).
            hour_of_day (int): Hour listing is submitted (0-23).
            is_repeat_buyer_same_day (int): Rapid transaction indicator.
            category (Optional[str]): Product category.
            files (Sequence[MediaFile]): Additional uploaded images/videos.

        Returns:
            Dict[str, Any]: Full moderation decision, probabilities, and audit record.
        """
        media: List[MediaFile] = list(files)
        if image_data:
            import base64

            raw = image_data if isinstance(image_data, bytes) else base64.b64decode(
                image_data.split(",", 1)[1] if "," in image_data else image_data)
            media.insert(0, MediaFile("image_upload", raw))

        layer1_result = self.layer1.moderate(title, description, media, explicit_category=category)
        resolved_category = layer1_result["category"]
        mismatch_score = float(layer1_result["image_text_mismatch_score"])
        slang_score = float(layer1_result["swahili_slang_risk_score"])

        median_price = float(CATEGORY_MEDIANS_TZS.get(resolved_category, CATEGORY_MEDIANS_TZS[DEFAULT_CATEGORY]))
        price_ratio = round(price_tzs / median_price, 4) if median_price > 0 else 1.0
        feature_vector = {
            "category": resolved_category,
            "price_tzs": price_tzs,
            "price_to_category_median_ratio": price_ratio,
            "hour_of_day": hour_of_day,
            "account_age_days": account_age_days,
            "seller_completed_orders": seller_completed_orders,
            "seller_dispute_rate": seller_dispute_rate,
            "image_text_mismatch_score": mismatch_score,
            "swahili_slang_risk_score": slang_score,
            "is_repeat_buyer_same_day": is_repeat_buyer_same_day,
        }
        raw_probability = self.layer2.predict_fraud_probability(feature_vector)

        final_probability, final_decision, reasons = self._reconcile_decisions(
            layer1_result=layer1_result,
            raw_prob=raw_probability,
            price_ratio=price_ratio,
            account_age_days=account_age_days,
            seller_completed_orders=seller_completed_orders,
            seller_dispute_rate=seller_dispute_rate,
        )
        log_id = self._log(listing_id, seller_id, title, price_tzs, layer1_result, final_probability, final_decision)

        return {
            "listing_id": listing_id,
            "seller_id": seller_id,
            "category": resolved_category,
            "price_to_category_median_ratio": price_ratio,
            "decision": final_decision,
            "fraud_probability": final_probability,
            "risk_reasons": reasons,
            "content_decision": layer1_result["decision"],
            "categories": layer1_result["categories"],
            "seller_message_sw": layer1_result["seller_message_sw"],
            "seller_message_en": layer1_result["seller_message_en"],
            "seller_claimed_concept": layer1_result["seller_claimed_concept"],
            "top_visual_label": layer1_result["top_visual_label"],
            "top_visual_probability": layer1_result["top_visual_probability"],
            "visual_comparison_status": layer1_result["visual_comparison_status"],
            "visual_comparison_detail": next(
                (r["detail"] for r in layer1_result["reasons"] if r.get("source") in ("image", "video")), ""),
            "visual_contraband_detected": layer1_result["visual_contraband_detected"],
            "layer1_details": layer1_result,
            "layer2_features": feature_vector,
            "audit_log_id": log_id,
        }

    def _log(self, listing_id: str, seller_id: str, title: str, price_tzs: float, layer1_result: Dict[str, Any],
             probability: float, decision: str) -> int:
        """Write a listing decision to the audit log."""
        return self.repository.log_listing_moderation(
            listing_id=listing_id or "unknown",
            seller_id=seller_id or "unknown",
            title=title,
            price_tzs=price_tzs,
            mismatch_score=float(layer1_result["image_text_mismatch_score"]),
            slang_score=float(layer1_result["swahili_slang_risk_score"]),
            nsfw_detected=bool(layer1_result["nsfw_detected"]),
            fraud_probability=probability,
            decision=decision,
            flagged_keywords=layer1_result["categories"] + layer1_result["matched_slang_terms"],
        )

    @staticmethod
    def _reconcile_decisions(
        layer1_result: Dict[str, Any],
        raw_prob: float,
        price_ratio: float,
        account_age_days: float,
        seller_completed_orders: int,
        seller_dispute_rate: float,
    ) -> Tuple[float, str, List[str]]:
        """Combine the Layer 1 content decision with the Layer 2 ML probability.

        * Content BLOCK / ESCALATE always stand.
        * Rule C (trust): trusted merchants with clean content get a 30% risk reduction.
        * Rule A: Layer 2 probability > 0.85 blocks only when corroborated by a dispute rate
          > 25% or a severe photo mismatch; otherwise the listing is reviewed.
        * Rule B: price/time alone never blocks; an elevated Layer 2 score is reviewed,
          except for established clean sellers. A new account (< 30 days) listing an item
          priced >= 3x the category median is reviewed.

        Returns:
            Tuple[float, str, List[str]]: (final_probability, final_decision, reasons)
        """
        content_decision = layer1_result["decision"]
        reasons = [r["detail"] for r in layer1_result["reasons"] if r.get("level") != "WEAK"]
        if content_decision in (DECISION_BLOCK, DECISION_ESCALATE):
            return round(max(raw_prob, layer1_result["score"]), 4), content_decision, reasons

        trusted = (account_age_days >= TRUSTED_ACCOUNT_AGE_DAYS
                   and seller_completed_orders >= TRUSTED_COMPLETED_ORDERS
                   and seller_dispute_rate <= TRUSTED_DISPUTE_RATE_MAX)
        prob = raw_prob * TRUST_DAMPENING_FACTOR if trusted and content_decision == DECISION_ALLOW else raw_prob

        layer2_decision = DECISION_ALLOW
        if prob > RULE_A_FRAUD_PROB_THRESHOLD:
            corroborated = (seller_dispute_rate > RULE_A_DISPUTE_THRESHOLD
                            or layer1_result["image_text_mismatch_score"] > RULE_A_MISMATCH_THRESHOLD)
            layer2_decision = DECISION_BLOCK if corroborated else DECISION_REVIEW
            reasons.append("High seller/transaction risk corroborated by dispute history or photo mismatch."
                           if corroborated else "Elevated seller/transaction risk: routed to agent review.")
        elif prob >= ALLOW_UPPER_BOUND:
            clean_established = (seller_dispute_rate <= TRUSTED_DISPUTE_RATE_MAX
                                 and account_age_days >= NEW_ACCOUNT_DAYS and content_decision == DECISION_ALLOW)
            if not clean_established:
                layer2_decision = DECISION_REVIEW
                reasons.append("Elevated seller/transaction risk score.")
        if account_age_days < NEW_ACCOUNT_DAYS and price_ratio >= HIGH_PRICE_RATIO:
            layer2_decision = max(layer2_decision, DECISION_REVIEW, key=DECISION_SEVERITY.get)
            reasons.append("New account with high-value item: held for routine verification.")

        final = max(content_decision, layer2_decision, key=DECISION_SEVERITY.get)
        final_prob = max(prob, layer1_result["score"]) if final != DECISION_ALLOW else min(prob, RULE_B_ALLOW_CAP)
        return round(final_prob, 4), final, reasons

    def evaluate_transaction(
        self,
        transaction_id: str,
        seller_id: str,
        buyer_id: str,
        price_tzs: float,
        hour_of_day: int,
        account_age_days: float,
        seller_completed_orders: int,
        seller_dispute_rate: float,
        image_text_mismatch_score: float = 0.0,
        swahili_slang_risk_score: float = 0.0,
        is_repeat_buyer_same_day: int = 0,
        category: Optional[str] = None,
        price_to_category_median_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate escrow payment risk before locking buyer funds.

        Parameters:
            transaction_id (str): Escrow transaction identifier.
            seller_id (str): Seller identifier.
            buyer_id (str): Buyer identifier.
            price_tzs (float): Amount in Tanzanian Shillings.
            hour_of_day (int): Time of transaction (0-23).
            account_age_days (float): Age of account in days.
            seller_completed_orders (int): Past completed deliveries.
            seller_dispute_rate (float): Dispute rate (0.0 to 1.0).
            image_text_mismatch_score (float): Layer 1 mismatch score.
            swahili_slang_risk_score (float): Layer 1 slang score.
            is_repeat_buyer_same_day (int): Multiple orders flag.
            category (Optional[str]): Resolved category.
            price_to_category_median_ratio (Optional[float]): Normalized price ratio.

        Returns:
            Dict[str, Any]: Risk evaluation report with decision.
        """
        resolved_category = category or DEFAULT_CATEGORY
        median_price = float(CATEGORY_MEDIANS_TZS.get(resolved_category, CATEGORY_MEDIANS_TZS[DEFAULT_CATEGORY]))

        if price_to_category_median_ratio is None:
            price_ratio = round(price_tzs / median_price, 4) if median_price > 0 else 1.0
        else:
            price_ratio = price_to_category_median_ratio

        features = {
            "category": resolved_category,
            "price_tzs": price_tzs,
            "price_to_category_median_ratio": price_ratio,
            "hour_of_day": hour_of_day,
            "account_age_days": account_age_days,
            "seller_completed_orders": seller_completed_orders,
            "seller_dispute_rate": seller_dispute_rate,
            "image_text_mismatch_score": image_text_mismatch_score,
            "swahili_slang_risk_score": swahili_slang_risk_score,
            "is_repeat_buyer_same_day": is_repeat_buyer_same_day,
        }

        fraud_prob = self.layer2.predict_fraud_probability(features)

        # Rule C: Trust dampening
        is_trusted_merchant = (
            account_age_days >= TRUSTED_ACCOUNT_AGE_DAYS
            and seller_completed_orders >= TRUSTED_COMPLETED_ORDERS
            and seller_dispute_rate <= TRUSTED_DISPUTE_RATE_MAX
        )
        if is_trusted_merchant and swahili_slang_risk_score < 0.30:
            fraud_prob = round(fraud_prob * TRUST_DAMPENING_FACTOR, 4)

        # Rule B: Legitimate seller at night with matching image cannot be auto-blocked
        if image_text_mismatch_score <= 0.35 and seller_dispute_rate <= 0.05 and swahili_slang_risk_score == 0.0:
            if fraud_prob > 0.75:
                # Cap at HOLD_FOR_REVIEW for new accounts or ALLOW for established
                fraud_prob = min(fraud_prob, 0.70 if account_age_days < 30 else 0.35)

        decision = self.layer2.classify_decision(fraud_prob)

        log_id = self.repository.log_transaction_fraud(
            transaction_id=transaction_id,
            seller_id=seller_id,
            buyer_id=buyer_id,
            price_tzs=price_tzs,
            features=features,
            fraud_probability=fraud_prob,
            decision=decision,
        )

        return {
            "transaction_id": transaction_id,
            "category": resolved_category,
            "price_to_category_median_ratio": price_ratio,
            "decision": decision,
            "fraud_probability": fraud_prob,
            "features": features,
            "audit_log_id": log_id,
        }

    def process_feedback(
        self,
        transaction_id: str,
        features: Dict[str, Any],
        actual_label: int,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Process ground-truth feedback, update online model, and log learning metrics.

        Parameters:
            transaction_id (str): Escrow transaction identifier.
            features (Dict[str, Any]): Feature vector associated with transaction.
            actual_label (int): 0 = Legitimate/OTP confirmed, 1 = Fraud/Illegal goods.
            comment (Optional[str]): Admin review note.

        Returns:
            Dict[str, Any]: Update summary with previous vs new probability.
        """
        prev_prob, updated_prob = self.layer2.learn_one(
            features=features, actual_label=actual_label, auto_save=True
        )

        log_id = self.repository.log_feedback(
            transaction_id=transaction_id,
            actual_label=actual_label,
            previous_probability=prev_prob,
            updated_probability=updated_prob,
            comment=comment,
        )

        return {
            "transaction_id": transaction_id,
            "actual_label": actual_label,
            "previous_probability": prev_prob,
            "updated_probability": updated_prob,
            "delta": round(updated_prob - prev_prob, 4),
            "feedback_log_id": log_id,
        }

