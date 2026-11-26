"""Cold-Start Synthetic Dataset Generator & Pre-Training Script.

Generates 2,000 realistic Tanzanian social commerce transactions:
- Legitimate Kariakoo fashion, Mwenge electronics (35k TZS chargers, 1.8M TZS smartphones),
  and local groceries (15k TZS unga wa sembe, cooking oil, rice)
- Benchmark prices against category medians (price_to_category_median_ratio)
- Fraudulent / illegal patterns (disguised narcotics/weapons, 2:00 AM high-ticket new account drains)

Pre-trains the incremental River pipeline and serializes the warm-start model
to `models/fraud_model.joblib`.
"""

import logging
import random
from typing import Any, Dict, List, Tuple

from river import metrics

from config.settings import (
    CATEGORY_BEAUTY,
    CATEGORY_ELECTRONICS,
    CATEGORY_FASHION,
    CATEGORY_GENERAL,
    CATEGORY_GROCERIES,
    CATEGORY_HOME_GOODS,
    CATEGORY_MEDIANS_TZS,
    COLD_START_SAMPLES,
    DEFAULT_CATEGORY,
    DEFAULT_DB_FILE,
    DEFAULT_MODEL_FILE,
)
from database.db import init_db
from database.repository import AuditLogRepository
from services.fraud_model_layer2 import OnlineFraudModelLayer2, create_river_pipeline

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cold_start_trainer")

# Fix random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def generate_tanzanian_synthetic_dataset(
    total_samples: int = COLD_START_SAMPLES,
) -> List[Tuple[Dict[str, Any], int]]:
    """Generate realistic Tanzanian social commerce transaction records with category medians.

    Parameters:
        total_samples (int): Total number of records to generate (defaults to 2000).

    Returns:
        List[Tuple[Dict[str, Any], int]]: List of (features_dict, label) tuples.
    """
    dataset: List[Tuple[Dict[str, Any], int]] = []

    # Distribution: ~82% Legitimate (1640), ~18% Fraudulent / Contraband (360)
    fraud_count = int(total_samples * 0.18)
    legit_count = total_samples - fraud_count

    logger.info(
        "Generating %d synthetic records (%d legitimate, %d fraudulent)...",
        total_samples,
        legit_count,
        fraud_count,
    )

    # -------------------------------------------------------------------------
    # 1. Legitimate Transactions (Fashion, Electronics, Groceries, Home Goods)
    # -------------------------------------------------------------------------
    for _ in range(legit_count):
        archetype = random.choice([
            "fashion_garment",
            "electronics_accessory",     # e.g., 35,000 TZS Type-C fast charger
            "electronics_flagship",      # e.g., 1,800,000 TZS smartphone
            "groceries_flour_rice",      # e.g., 15,000 TZS Unga wa sembe 5kg
            "home_kitchen_utensils",     # e.g., 45,000 TZS Kisu cha jikoni set
            "beauty_cosmetics",          # e.g., 25,000 TZS perfume/hair weave
        ])

        if archetype == "fashion_garment":
            category = CATEGORY_FASHION
            price_tzs = float(random.randint(15_000, 150_000))
        elif archetype == "electronics_accessory":
            category = CATEGORY_ELECTRONICS
            price_tzs = float(random.choice([25_000, 35_000, 45_000, 60_000]))
        elif archetype == "electronics_flagship":
            category = CATEGORY_ELECTRONICS
            price_tzs = float(random.randint(850_000, 2_400_000))
        elif archetype == "groceries_flour_rice":
            category = CATEGORY_GROCERIES
            price_tzs = float(random.choice([12_000, 15_000, 18_000, 25_000, 35_000]))
        elif archetype == "home_kitchen_utensils":
            category = CATEGORY_HOME_GOODS
            price_tzs = float(random.randint(25_000, 120_000))
        else:
            category = CATEGORY_BEAUTY
            price_tzs = float(random.randint(10_000, 85_000))

        # Peak social commerce hours: daytime 08:00 - 21:00
        hour_of_day = random.choices(
            population=list(range(24)),
            weights=[1, 1, 1, 1, 2, 4, 6, 8, 10, 12, 14, 15, 14, 13, 14, 15, 16, 17, 16, 12, 8, 5, 3, 2],
            k=1,
        )[0]

        account_age_days = float(random.randint(15, 900))
        seller_completed_orders = random.randint(5, 350)
        seller_dispute_rate = round(random.uniform(0.0, 0.04), 4)
        image_text_mismatch_score = round(random.uniform(0.02, 0.20), 4)
        swahili_slang_risk_score = 0.0  # Clean listings (including unga wa sembe / kisu cha jikoni)
        is_repeat_buyer_same_day = 1 if random.random() < 0.05 else 0

        median_price = CATEGORY_MEDIANS_TZS.get(category, CATEGORY_MEDIANS_TZS[DEFAULT_CATEGORY])
        price_to_category_median_ratio = round(price_tzs / median_price, 4)

        features = {
            "category": category,
            "price_tzs": price_tzs,
            "price_to_category_median_ratio": price_to_category_median_ratio,
            "hour_of_day": hour_of_day,
            "account_age_days": account_age_days,
            "seller_completed_orders": seller_completed_orders,
            "seller_dispute_rate": seller_dispute_rate,
            "image_text_mismatch_score": image_text_mismatch_score,
            "swahili_slang_risk_score": swahili_slang_risk_score,
            "is_repeat_buyer_same_day": is_repeat_buyer_same_day,
        }
        dataset.append((features, 0))

    # -------------------------------------------------------------------------
    # 2. Fraudulent, Contraband & Complex Mismatch Scenarios
    # (Wrong-gender clothing, mobile screenshots, coded contraband, price anomalies)
    # -------------------------------------------------------------------------
    for _ in range(fraud_count):
        fraud_type = random.choice([
            "wrong_gender_clothing_mismatch", # e.g. Suti photo for Gauni listing
            "screenshot_upload_fraud",        # Mobile screenshot instead of real item
            "electronics_price_anomaly",      # Suspiciously cheap flagship or fake accessories
            "disguised_contraband",           # Coded Swahili narcotics disguised as clothing
            "late_night_cash_drain",          # 2:00 AM high-ticket new account
            "rapid_repeat_scam",              # Rapid repeat velocity drain
            "high_dispute_bad_merchant",      # High dispute merchant with counterfeit goods
        ])

        if fraud_type == "wrong_gender_clothing_mismatch":
            category = CATEGORY_FASHION
            price_tzs = float(random.randint(35_000, 180_000))
            hour_of_day = random.randint(8, 20)
            account_age_days = float(random.randint(1, 45))
            seller_completed_orders = random.randint(0, 8)
            seller_dispute_rate = round(random.uniform(0.10, 0.40), 4)
            image_text_mismatch_score = round(random.uniform(0.75, 0.92), 4) # High mismatch
            swahili_slang_risk_score = 0.0
            is_repeat_buyer_same_day = 0

        elif fraud_type == "screenshot_upload_fraud":
            category = random.choice([CATEGORY_ELECTRONICS, CATEGORY_FASHION, CATEGORY_BEAUTY])
            price_tzs = float(random.randint(50_000, 800_000))
            hour_of_day = random.randint(0, 23)
            account_age_days = float(random.randint(1, 14))
            seller_completed_orders = random.randint(0, 3)
            seller_dispute_rate = round(random.uniform(0.15, 0.50), 4)
            image_text_mismatch_score = round(random.uniform(0.85, 0.95), 4) # Screenshot mismatch
            swahili_slang_risk_score = round(random.uniform(0.0, 0.30), 4)
            is_repeat_buyer_same_day = 1 if random.random() < 0.20 else 0

        elif fraud_type == "electronics_price_anomaly":
            category = CATEGORY_ELECTRONICS
            # Suspiciously cheap fake phone e.g. 25,000 TZS or overpriced accessory
            price_tzs = float(random.choice([15_000, 25_000, 35_000, 4_500_000]))
            hour_of_day = random.randint(0, 23)
            account_age_days = float(random.randint(2, 20))
            seller_completed_orders = random.randint(0, 4)
            seller_dispute_rate = round(random.uniform(0.20, 0.60), 4)
            image_text_mismatch_score = round(random.uniform(0.65, 0.90), 4)
            swahili_slang_risk_score = round(random.uniform(0.10, 0.50), 4)
            is_repeat_buyer_same_day = 1 if random.random() < 0.30 else 0

        elif fraud_type == "disguised_contraband":
            category = random.choice([CATEGORY_FASHION, CATEGORY_GENERAL])
            price_tzs = float(random.randint(150_000, 1_500_000))
            hour_of_day = random.randint(0, 23)
            account_age_days = float(random.randint(0, 8))
            seller_completed_orders = random.randint(0, 3)
            seller_dispute_rate = round(random.uniform(0.15, 0.60), 4)
            image_text_mismatch_score = round(random.uniform(0.85, 0.98), 4)
            swahili_slang_risk_score = round(random.uniform(0.75, 1.00), 4)
            is_repeat_buyer_same_day = 0

        elif fraud_type == "late_night_cash_drain":
            category = CATEGORY_ELECTRONICS
            price_tzs = float(random.randint(2_500_000, 6_000_000))
            hour_of_day = random.randint(1, 4)
            account_age_days = float(random.randint(0, 2))
            seller_completed_orders = 0
            seller_dispute_rate = round(random.uniform(0.20, 0.50), 4)
            image_text_mismatch_score = round(random.uniform(0.60, 0.95), 4)
            swahili_slang_risk_score = round(random.uniform(0.20, 0.60), 4)
            is_repeat_buyer_same_day = 1

        elif fraud_type == "rapid_repeat_scam":
            category = random.choice([CATEGORY_ELECTRONICS, CATEGORY_FASHION])
            price_tzs = float(random.randint(400_000, 2_000_000))
            hour_of_day = random.randint(0, 23)
            account_age_days = float(random.randint(1, 15))
            seller_completed_orders = random.randint(0, 5)
            seller_dispute_rate = round(random.uniform(0.20, 0.50), 4)
            image_text_mismatch_score = round(random.uniform(0.50, 0.85), 4)
            swahili_slang_risk_score = round(random.uniform(0.40, 0.80), 4)
            is_repeat_buyer_same_day = 1

        else:
            category = random.choice([CATEGORY_FASHION, CATEGORY_HOME_GOODS])
            price_tzs = float(random.randint(250_000, 1_200_000))
            hour_of_day = random.randint(6, 22)
            account_age_days = float(random.randint(5, 40))
            seller_completed_orders = random.randint(1, 10)
            seller_dispute_rate = round(random.uniform(0.35, 0.75), 4)
            image_text_mismatch_score = round(random.uniform(0.60, 0.85), 4)
            swahili_slang_risk_score = round(random.uniform(0.50, 0.85), 4)
            is_repeat_buyer_same_day = 0

        median_price = CATEGORY_MEDIANS_TZS.get(category, CATEGORY_MEDIANS_TZS[DEFAULT_CATEGORY])
        price_to_category_median_ratio = round(price_tzs / median_price, 4)

        features = {
            "category": category,
            "price_tzs": price_tzs,
            "price_to_category_median_ratio": price_to_category_median_ratio,
            "hour_of_day": hour_of_day,
            "account_age_days": account_age_days,
            "seller_completed_orders": seller_completed_orders,
            "seller_dispute_rate": seller_dispute_rate,
            "image_text_mismatch_score": image_text_mismatch_score,
            "swahili_slang_risk_score": swahili_slang_risk_score,
            "is_repeat_buyer_same_day": is_repeat_buyer_same_day,
        }
        dataset.append((features, 1))

    # Shuffle the synthetic dataset
    random.shuffle(dataset)
    return dataset


def train_cold_start_model(
    output_model_path: str = DEFAULT_MODEL_FILE,
    total_samples: int = COLD_START_SAMPLES,
) -> OnlineFraudModelLayer2:
    """Pre-train River pipeline with category price benchmarking.

    Parameters:
        output_model_path (str): File path to save the serialized River model.
        total_samples (int): Number of synthetic training instances.

    Returns:
        OnlineFraudModelLayer2: Trained model service instance.
    """
    logger.info("Starting Cold-Start pre-training on %d records...", total_samples)
    dataset = generate_tanzanian_synthetic_dataset(total_samples)

    model_service = OnlineFraudModelLayer2(model_path=output_model_path)
    model_service.pipeline = create_river_pipeline()

    acc_metric = metrics.Accuracy()
    precision_metric = metrics.Precision()
    recall_metric = metrics.Recall()
    roc_metric = metrics.ROCAUC()
    f1_metric = metrics.F1()

    for i, (features, target) in enumerate(dataset, 1):
        sanitized = model_service.sanitize_features(features)

        proba_dict = model_service.pipeline.predict_proba_one(sanitized)
        prob_fraud = float(proba_dict.get(1, proba_dict.get(True, 0.5)))
        pred_label = 1 if prob_fraud >= 0.5 else 0

        acc_metric.update(target, pred_label)
        precision_metric.update(target, pred_label)
        recall_metric.update(target, pred_label)
        roc_metric.update(target, prob_fraud)
        f1_metric.update(target, pred_label)

        model_service.pipeline.learn_one(sanitized, target)

        if i % 500 == 0 or i == total_samples:
            logger.info(
                "Progress: %d/%d | Accuracy: %.4f | ROC-AUC: %.4f | F1: %.4f",
                i,
                total_samples,
                acc_metric.get(),
                roc_metric.get(),
                f1_metric.get(),
            )

    model_service.save_model()
    logger.info("Pre-trained model saved successfully to: %s", output_model_path)

    init_db(DEFAULT_DB_FILE)
    repo = AuditLogRepository(DEFAULT_DB_FILE)
    repo.update_model_accuracy(
        running_accuracy=round(float(acc_metric.get()), 4),
        running_f1=round(float(f1_metric.get()), 4),
    )
    logger.info("Database model metrics initialized with pre-training benchmarks.")

    return model_service


if __name__ == "__main__":
    train_cold_start_model()
