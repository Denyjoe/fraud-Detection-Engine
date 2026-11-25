"""Moderation and Fraud Detection Orchestrator.

Combines Layer 1 (context-aware Swahili/English lexicon, NSFW, vision-text mismatch)
with Layer 2 (River incremental model with category-aware price benchmarking)
to provide unified evaluations under strict production guardrails (Rules A, B, C).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from config.settings import (
    CATEGORY_MEDIANS_TZS,
    CRITICAL_MISMATCH_THRESHOLD,
    CRITICAL_NSFW_THRESHOLD,
    CRITICAL_SLANG_THRESHOLD,
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_HOLD,
    DEFAULT_CATEGORY,
    RULE_A_DISPUTE_THRESHOLD,
    RULE_A_FRAUD_PROB_THRESHOLD,
    RULE_A_MISMATCH_THRESHOLD,
    TRUST_DAMPENING_FACTOR,
    TRUSTED_ACCOUNT_AGE_DAYS,
    TRUSTED_COMPLETED_ORDERS,
    TRUSTED_DISPUTE_RATE_MAX,
    VISUAL_CONTRABAND_THRESHOLD,
    VISUAL_MISMATCH_THRESHOLD,
)
from database.repository import AuditLogRepository
from services.fraud_model_layer2 import OnlineFraudModelLayer2
from services.moderation_layer1 import ContentModerationLayer1, infer_listing_category

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
    ) -> Dict[str, Any]:
        """Execute pre-listing moderation combining Layer 1 and category-benchmarked Layer 2.

        Parameters:
            listing_id (str): Unique listing ID.
            seller_id (str): Seller identifier.
            title (str): Product listing title.
            description (str): Detailed product description.
            price_tzs (float): Price in Tanzanian Shillings.
            account_age_days (float): Seller's account age.
            seller_completed_orders (int): Past successfully delivered orders.
            seller_dispute_rate (float): Seller dispute percentage (0.0 to 1.0).
            image_data (Optional[Any]): Uploaded image (base64, bytes, or PIL Image).
            hour_of_day (int): Hour listing is submitted (0-23).
            is_repeat_buyer_same_day (int): Rapid transaction indicator.
            category (Optional[str]): Product category.

        Returns:
            Dict[str, Any]: Full moderation decision, probabilities, and audit record.
        """
        # Step 1: Execute Layer 1 Content Moderation (Context-aware disambiguation & category inference)
        layer1_result = self.layer1.evaluate_listing_content(
            title=title,
            description=description,
            image_data=image_data,
            explicit_category=category,
        )

        resolved_category = layer1_result["category"]
        mismatch_score = float(layer1_result["image_text_mismatch_score"])
        slang_score = float(layer1_result["swahili_slang_risk_score"])
        nsfw_detected = bool(layer1_result["nsfw_detected"])
        layer1_critical_flag = bool(layer1_result.get("layer1_critical_flag", False))
        flagged_keywords: List[str] = layer1_result["matched_slang_terms"]
        visual_contraband_detected = bool(layer1_result.get("visual_contraband_detected", False))
        visual_contraband_probability = float(layer1_result.get("visual_contraband_probability", 0.0))
        top_visual_label = str(layer1_result.get("top_visual_label", ""))
        top_visual_prob = float(layer1_result.get("top_visual_probability", 0.0))
        seller_claimed_concept = str(layer1_result.get("seller_claimed_concept", ""))
        visual_comparison_status = str(layer1_result.get("visual_comparison_status", "NO_IMAGE"))
        comparison_detail = str(layer1_result.get("comparison_detail", ""))
        is_severe_mismatch = bool(layer1_result.get("is_severe_mismatch", False))
        is_non_product = bool(layer1_result.get("is_non_product", False))

        # Step 2: Compute category-aware price benchmark
        median_price = float(CATEGORY_MEDIANS_TZS.get(resolved_category, CATEGORY_MEDIANS_TZS[DEFAULT_CATEGORY]))
        price_to_category_median_ratio = round(price_tzs / median_price, 4) if median_price > 0 else 1.0

        # Step 3: Assemble feature vector for Layer 2 Fraud Model
        feature_vector = {
            "category": resolved_category,
            "price_tzs": price_tzs,
            "price_to_category_median_ratio": price_to_category_median_ratio,
            "hour_of_day": hour_of_day,
            "account_age_days": account_age_days,
            "seller_completed_orders": seller_completed_orders,
            "seller_dispute_rate": seller_dispute_rate,
            "image_text_mismatch_score": mismatch_score,
            "swahili_slang_risk_score": slang_score,
            "is_repeat_buyer_same_day": is_repeat_buyer_same_day,
        }

        # Step 4: Run Layer 2 model inference
        raw_fraud_probability = self.layer2.predict_fraud_probability(feature_vector)

        # Step 5: Apply Rule C (Merchant Trust Whitelisting)
        is_trusted_merchant = (
            account_age_days >= TRUSTED_ACCOUNT_AGE_DAYS
            and seller_completed_orders >= TRUSTED_COMPLETED_ORDERS
            and seller_dispute_rate <= TRUSTED_DISPUTE_RATE_MAX
        )
        if is_trusted_merchant and not layer1_critical_flag and not visual_contraband_detected:
            raw_fraud_probability = round(raw_fraud_probability * TRUST_DAMPENING_FACTOR, 4)

        # Step 6: Apply strict production Guardrails (Rules A & B)
        final_probability, final_decision, reasons = self._reconcile_decisions(
            raw_prob=raw_fraud_probability,
            mismatch_score=mismatch_score,
            slang_score=slang_score,
            nsfw_detected=nsfw_detected,
            layer1_critical_flag=layer1_critical_flag,
            flagged_keywords=flagged_keywords,
            visual_contraband_detected=visual_contraband_detected,
            visual_contraband_probability=visual_contraband_probability,
            top_visual_label=top_visual_label,
            top_visual_probability=top_visual_prob,
            seller_claimed_concept=seller_claimed_concept,
            visual_comparison_status=visual_comparison_status,
            comparison_detail=comparison_detail,
            is_severe_mismatch=is_severe_mismatch,
            is_non_product=is_non_product,
            title=title,
            seller_dispute_rate=seller_dispute_rate,
            account_age_days=account_age_days,
            hour_of_day=hour_of_day,
            is_trusted_merchant=is_trusted_merchant,
        )

        # Step 7: Persist audit log in SQLite
        log_id = self.repository.log_listing_moderation(
            listing_id=listing_id,
            seller_id=seller_id,
            title=title,
            price_tzs=price_tzs,
            mismatch_score=mismatch_score,
            slang_score=slang_score,
            nsfw_detected=nsfw_detected,
            fraud_probability=final_probability,
            decision=final_decision,
            flagged_keywords=flagged_keywords,
        )

        return {
            "listing_id": listing_id,
            "seller_id": seller_id,
            "category": resolved_category,
            "price_to_category_median_ratio": price_to_category_median_ratio,
            "decision": final_decision,
            "fraud_probability": final_probability,
            "risk_reasons": reasons,
            "seller_claimed_concept": seller_claimed_concept,
            "top_visual_label": top_visual_label,
            "top_visual_probability": top_visual_prob,
            "visual_comparison_status": visual_comparison_status,
            "visual_comparison_detail": comparison_detail,
            "visual_contraband_detected": visual_contraband_detected,
            "layer1_details": layer1_result,
            "layer2_features": feature_vector,
            "audit_log_id": log_id,
        }

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

    def _reconcile_decisions(
        self,
        raw_prob: float,
        mismatch_score: float,
        slang_score: float,
        nsfw_detected: bool,
        layer1_critical_flag: bool,
        flagged_keywords: List[str],
        visual_contraband_detected: bool = False,
        visual_contraband_probability: float = 0.0,
        top_visual_label: str = "",
        top_visual_probability: float = 0.0,
        seller_claimed_concept: str = "",
        visual_comparison_status: str = "NO_IMAGE",
        comparison_detail: str = "",
        is_severe_mismatch: bool = False,
        is_non_product: bool = False,
        title: str = "",
        seller_dispute_rate: float = 0.0,
        account_age_days: float = 30.0,
        hour_of_day: int = 12,
        is_trusted_merchant: bool = False,
    ) -> Tuple[float, str, List[str]]:
        """Combine Layer 1 content safety findings with Layer 2 ML probability under strict guardrails.

        Rule A (Strict Hard Block):
          1. Layer 1 confirms contraband/NSFW imagery (visual_contraband_detected or layer1_critical_flag) -> BLOCK.
          2. ML prob > 0.85 AND (dispute > 0.25 OR mismatch > 0.80) -> BLOCK.
        Rule B (No Unfair Blocking on Price/Time Alone):
          Legitimate items with matching image and zero disputes capped at ALLOW or HOLD.
        Rule C (Merchant Trust Whitelisting):
          Trusted merchants protected from false friction.

        Returns:
            Tuple[float, str, List[str]]: (final_probability, final_decision, reasons)
        """
        reasons: List[str] = []
        final_prob = raw_prob

        # --- Rule A1: Confirmed contraband, hard narcotics, weapons, or illicit visual media ---
        if visual_contraband_detected:
            final_prob = max(final_prob, 0.96)
            reasons.append(
                f"Prohibited visual content detected in uploaded image: '{top_visual_label}' "
                f"({visual_contraband_probability * 100.0:.1f}% illicit confidence)."
            )
            if flagged_keywords:
                reasons.append(f"Confirmed illegal contraband / prohibited terms: {', '.join(flagged_keywords)}")
            if nsfw_detected:
                reasons.append("NSFW, adult, or abusive media detected.")
            return round(final_prob, 4), DECISION_BLOCK, reasons

        if layer1_critical_flag:
            final_prob = max(final_prob, 0.95)
            if flagged_keywords:
                reasons.append(f"Confirmed illegal contraband / prohibited terms: {', '.join(flagged_keywords)}")
            if nsfw_detected:
                reasons.append("NSFW, adult, or abusive media detected.")
            if mismatch_score >= CRITICAL_MISMATCH_THRESHOLD and slang_score >= 0.40:
                reasons.append(f"Severe vision-text mismatch ({mismatch_score:.2f}) with contraband signals.")
            return round(final_prob, 4), DECISION_BLOCK, reasons

        # --- Rule: Non-Product Media (Screenshots or Paper Invoices) ---
        if is_non_product:
            reasons.append(
                comparison_detail
                or f"Non-product image detected: Image is '{top_visual_label}' instead of physical retail merchandise."
            )
            if seller_dispute_rate > 0.15 or account_age_days < 7.0:
                final_prob = max(final_prob, 0.88)
                return round(final_prob, 4), DECISION_BLOCK, reasons
            else:
                final_prob = max(0.65, min(0.74, final_prob))
                return round(final_prob, 4), DECISION_HOLD, reasons

        # --- Rule B: Prevent unfair blocking on Price/Time alone ---
        has_matching_media = mismatch_score <= 0.35 and visual_comparison_status in ("MATCH", "NO_IMAGE")
        has_clean_record = seller_dispute_rate <= 0.05
        has_zero_slang = slang_score == 0.0

        if has_matching_media and has_clean_record and has_zero_slang:
            # Established seller or clean listing: cap at ALLOW or HOLD_FOR_REVIEW
            if is_trusted_merchant or account_age_days >= 30:
                final_prob = min(final_prob, 0.35)
                return round(final_prob, 4), DECISION_ALLOW, reasons
            else:
                # Brand new account (age < 30) with expensive item: capped at HOLD_FOR_REVIEW, never BLOCK
                final_prob = min(final_prob, 0.65)
                reasons.append("New account with high-value item: held for routine verification.")
                return round(final_prob, 4), DECISION_HOLD, reasons

        # --- Rule: Severe Contrastive Image-to-Title Mismatch (Mismatch > 0.65 prevents ALLOW) ---
        if is_severe_mismatch or visual_comparison_status == "MISMATCH" or mismatch_score > VISUAL_MISMATCH_THRESHOLD:
            reasons.append(
                comparison_detail
                or f"Visual-Text Mismatch ({mismatch_score:.2f}): Image shows '{top_visual_label}' instead of claimed '{seller_claimed_concept or title}'."
            )
            # Cannot allow; if high dispute or new account -> BLOCK, else HOLD_FOR_REVIEW
            if seller_dispute_rate > 0.20 or account_age_days < 7.0:
                final_prob = max(final_prob, 0.85)
                return round(final_prob, 4), DECISION_BLOCK, reasons
            else:
                final_prob = max(0.55, min(0.74, final_prob))
                return round(final_prob, 4), DECISION_HOLD, reasons

        # --- Rule A2: ML Fraud Probability Guardrail ---
        if final_prob > RULE_A_FRAUD_PROB_THRESHOLD:
            # Only allow BLOCK if seller has elevated dispute history OR severe vision-text mismatch
            has_corroborating_risk = (
                seller_dispute_rate > RULE_A_DISPUTE_THRESHOLD
                or mismatch_score > RULE_A_MISMATCH_THRESHOLD
            )
            if has_corroborating_risk:
                reasons.append("Critical risk: High ML fraud probability corroborated by seller dispute history or severe media mismatch.")
                return round(final_prob, 4), DECISION_BLOCK, reasons
            else:
                # Without corroborating evidence, demote BLOCK to HOLD_FOR_REVIEW
                final_prob = 0.70
                reasons.append("Elevated risk anomaly: routed to agent review for manual inspection.")
                return round(final_prob, 4), DECISION_HOLD, reasons

        final_decision = self.layer2.classify_decision(final_prob)
        return round(final_prob, 4), final_decision, reasons
