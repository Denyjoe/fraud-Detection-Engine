"""Layer 2: Self-Learning Transaction Fraud Model (Incremental / Online Learning).

Uses the River ML framework (Pipeline with StandardScaler and LogisticRegression)
to evaluate real-time fraud probability and adapt online via .learn_one().
Features:
- price_to_category_median_ratio (benchmarks price against category median to avoid raw price bias)
- hour_of_day (0-23)
- account_age_days
- seller_completed_orders
- seller_dispute_rate
- image_text_mismatch_score (0.0-1.0)
- swahili_slang_risk_score (0.0-1.0)
- is_repeat_buyer_same_day (0 or 1)
"""

import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
from river import compose, linear_model, preprocessing

from config.settings import (
    ALLOW_UPPER_BOUND,
    CATEGORY_MEDIANS_TZS,
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_REVIEW,
    DEFAULT_CATEGORY,
    DEFAULT_MODEL_FILE,
    REVIEW_UPPER_BOUND,
    LOGISTIC_L2_PENALTY,
    SUPPORTED_CATEGORIES,
)

logger = logging.getLogger(__name__)

# Canonical feature keys expected by the category-aware fraud model
FEATURE_KEYS = (
    "price_to_category_median_ratio",
    "hour_of_day",
    "account_age_days",
    "seller_completed_orders",
    "seller_dispute_rate",
    "image_text_mismatch_score",
    "swahili_slang_risk_score",
    "is_repeat_buyer_same_day",
)


def create_river_pipeline() -> compose.Pipeline:
    """Construct a new incremental River pipeline for fraud probability estimation.

    Combines an online StandardScaler with an online Logistic Regression classifier
    using stochastic gradient descent (SGD).

    Returns:
        compose.Pipeline: Configured River pipeline.
    """
    return compose.Pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(
            optimizer=None,
            l2=LOGISTIC_L2_PENALTY,
        ),
    )


class OnlineFraudModelLayer2:
    """Thread-safe online learning fraud detection service."""

    def __init__(self, model_path: str = DEFAULT_MODEL_FILE) -> None:
        """Initialize the model layer with storage path and thread lock.

        Parameters:
            model_path (str): File system path for persisting the River pipeline.
        """
        self.model_path = Path(model_path)
        self._lock = threading.Lock()
        self.pipeline: compose.Pipeline = self._load_or_initialize_model()

    def _load_or_initialize_model(self) -> compose.Pipeline:
        """Load an existing serialized model from disk, or initialize a fresh pipeline.

        Returns:
            compose.Pipeline: Ready-to-use River pipeline.
        """
        if self.model_path.exists():
            try:
                loaded = joblib.load(self.model_path)
                logger.info("Successfully loaded River model from %s", self.model_path)
                return loaded
            except Exception as exc:
                logger.warning(
                    "Failed to load serialized model from %s (%s). Reinitializing fresh pipeline.",
                    self.model_path,
                    exc,
                )
        else:
            logger.info("No existing model found at %s. Creating new pipeline.", self.model_path)

        new_pipe = create_river_pipeline()
        return new_pipe

    def save_model(self) -> None:
        """Atomically persist the active River pipeline to disk using joblib.

        Uses a temporary file in the target directory followed by an atomic rename
        to prevent file corruption during concurrent operations.

        Returns:
            None
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = self.model_path.parent
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix=".tmp") as tmp_file:
            temp_path = Path(tmp_file.name)

        try:
            joblib.dump(self.pipeline, temp_path)
            os.replace(temp_path, self.model_path)
            logger.info("Model safely persisted to %s", self.model_path)
        except Exception as exc:
            if temp_path.exists():
                temp_path.unlink()
            logger.error("Failed to save model: %s", exc)
            raise

    def sanitize_features(self, raw_features: Dict[str, Any]) -> Dict[str, float]:
        """Validate and normalize raw input features into float dictionary.

        Calculates price_to_category_median_ratio against Tanzanian category benchmarks
        to eliminate false-positive anomalies on legitimate high-value items.

        Parameters:
            raw_features (Dict[str, Any]): Input dictionary containing features.

        Returns:
            Dict[str, float]: Cleaned feature dictionary with normalized values.
        """
        cleaned: Dict[str, float] = {}

        # 1. Category-aware price ratio
        category = str(raw_features.get("category", DEFAULT_CATEGORY)).lower()
        if category not in SUPPORTED_CATEGORIES:
            category = DEFAULT_CATEGORY

        median_price = float(CATEGORY_MEDIANS_TZS.get(category, CATEGORY_MEDIANS_TZS[DEFAULT_CATEGORY]))

        if "price_to_category_median_ratio" in raw_features:
            ratio = float(raw_features["price_to_category_median_ratio"])
        else:
            price_val = float(raw_features.get("price_tzs", median_price))
            ratio = price_val / median_price if median_price > 0 else 1.0

        cleaned["price_to_category_median_ratio"] = max(0.0, round(ratio, 4))

        # 2. Hour of day (0 to 23)
        hour = int(raw_features.get("hour_of_day", 12))
        cleaned["hour_of_day"] = float(max(0, min(23, hour)))

        # 3. Account age in days
        age = float(raw_features.get("account_age_days", 0.0))
        cleaned["account_age_days"] = max(0.0, age)

        # 4. Seller completed orders
        orders = float(raw_features.get("seller_completed_orders", 0.0))
        cleaned["seller_completed_orders"] = max(0.0, orders)

        # 5. Seller dispute rate (0.0 to 1.0)
        dispute = float(raw_features.get("seller_dispute_rate", 0.0))
        cleaned["seller_dispute_rate"] = max(0.0, min(1.0, dispute))

        # 6. Vision-language mismatch score (0.0 to 1.0)
        mismatch = float(raw_features.get("image_text_mismatch_score", 0.0))
        cleaned["image_text_mismatch_score"] = max(0.0, min(1.0, mismatch))

        # 7. Swahili prohibited slang risk score (0.0 to 1.0)
        slang = float(raw_features.get("swahili_slang_risk_score", 0.0))
        cleaned["swahili_slang_risk_score"] = max(0.0, min(1.0, slang))

        # 8. Repeat buyer same day flag (0 or 1)
        repeat = 1.0 if bool(raw_features.get("is_repeat_buyer_same_day", 0)) else 0.0
        cleaned["is_repeat_buyer_same_day"] = repeat

        return cleaned

    def predict_fraud_probability(self, features: Dict[str, Any]) -> float:
        """Predict fraud probability for a given transaction feature set.

        Parameters:
            features (Dict[str, Any]): Feature mapping.

        Returns:
            float: Fraud probability between 0.0 and 1.0.
        """
        sanitized = self.sanitize_features(features)
        with self._lock:
            proba_dict = self.pipeline.predict_proba_one(sanitized)

        prob_fraud = proba_dict.get(1, proba_dict.get(True, 0.5))
        return round(float(prob_fraud), 4)

    def classify_decision(self, probability: float) -> str:
        """Map a fraud probability score to an actionable platform decision.

        ALLOW: < 0.40 (40%)
        REVIEW: 0.40 - 0.75 (40% - 75%)
        BLOCK: > 0.75 (75%)

        Parameters:
            probability (float): Fraud probability score.

        Returns:
            str: One of 'ALLOW', 'REVIEW', or 'BLOCK'.
        """
        if probability < ALLOW_UPPER_BOUND:
            return DECISION_ALLOW
        elif probability <= REVIEW_UPPER_BOUND:
            return DECISION_REVIEW
        else:
            return DECISION_BLOCK

    def learn_one(
        self, features: Dict[str, Any], actual_label: int, auto_save: bool = True
    ) -> Tuple[float, float]:
        """Perform real-time incremental online learning on a single feedback sample.

        Updates the model weights in-memory and optionally persists the updated pipeline to disk.

        Parameters:
            features (Dict[str, Any]): Feature vector of the transaction.
            actual_label (int): Ground truth (0 for Legitimate, 1 for Fraudulent).
            auto_save (bool): Whether to atomically save the model after learning.

        Returns:
            Tuple[float, float]: (previous_probability, updated_probability)
        """
        sanitized = self.sanitize_features(features)
        target = 1 if actual_label > 0 else 0

        with self._lock:
            prev_proba_dict = self.pipeline.predict_proba_one(sanitized)
            prev_prob = float(prev_proba_dict.get(1, prev_proba_dict.get(True, 0.5)))

            self.pipeline.learn_one(sanitized, target)

            new_proba_dict = self.pipeline.predict_proba_one(sanitized)
            new_prob = float(new_proba_dict.get(1, new_proba_dict.get(True, 0.5)))

            if auto_save:
                self.save_model()

        logger.info(
            "Online update applied. Label: %d | Prev Prob: %.4f -> New Prob: %.4f",
            target,
            prev_prob,
            new_prob,
        )
        return round(prev_prob, 4), round(new_prob, 4)
