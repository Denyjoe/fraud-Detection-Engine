"""Application Configuration and Constants.

Defines all system thresholds, decision boundaries, file paths,
and environment settings for the Tanzanian Fraud Detection and Moderation Engine.
"""

import os
from pathlib import Path
from typing import Final

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
MODELS_DIR: Final[Path] = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# File Paths & Database Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_FILE: Final[str] = os.getenv(
    "FRAUD_MODEL_PATH", str(MODELS_DIR / "fraud_model.joblib")
)
DEFAULT_DB_FILE: Final[str] = os.getenv(
    "AUDIT_DB_PATH", str(BASE_DIR / "audit_logs.db")
)

# ---------------------------------------------------------------------------
# Currency and Regional Constants (Tanzanian Social Commerce)
# ---------------------------------------------------------------------------
CURRENCY_CODE: Final[str] = "TZS"
COUNTRY_CODE: Final[str] = "TZ"
MIN_TRANSACTION_TZS: Final[float] = 500.0  # Lowest typical mobile money transaction
MAX_UNVERIFIED_PRICE_TZS: Final[float] = 5_000_000.0  # High alert limit for new accounts

# ---------------------------------------------------------------------------
# Category Benchmarking & Market Medians (TZS)
# ---------------------------------------------------------------------------
CATEGORY_FASHION: Final[str] = "fashion"
CATEGORY_ELECTRONICS: Final[str] = "electronics"
CATEGORY_GROCERIES: Final[str] = "groceries"
CATEGORY_HOME_GOODS: Final[str] = "home_goods"
CATEGORY_BEAUTY: Final[str] = "beauty"
CATEGORY_GENERAL: Final[str] = "general"

SUPPORTED_CATEGORIES: Final[tuple] = (
    CATEGORY_FASHION,
    CATEGORY_ELECTRONICS,
    CATEGORY_GROCERIES,
    CATEGORY_HOME_GOODS,
    CATEGORY_BEAUTY,
    CATEGORY_GENERAL,
)

DEFAULT_CATEGORY: Final[str] = CATEGORY_GENERAL

# Realistic median prices in Tanzanian Shillings per category
CATEGORY_MEDIANS_TZS: Final[dict] = {
    CATEGORY_FASHION: 45_000.0,       # Kitenge, vitenge, gauni, suti
    CATEGORY_ELECTRONICS: 350_000.0,   # Smartphones, chargers, screens
    CATEGORY_GROCERIES: 20_000.0,      # Unga wa sembe, mchele, mafuta
    CATEGORY_HOME_GOODS: 80_000.0,     # Vyombo vya jikoni, visu, meza
    CATEGORY_BEAUTY: 30_000.0,         # Perfume, nywele, vipodozi
    CATEGORY_GENERAL: 50_000.0,        # Default benchmark
}

# ---------------------------------------------------------------------------
# Guardrail & Trust Dampening Constants
# ---------------------------------------------------------------------------
# Rule A: Hard Block requires severe signals
RULE_A_FRAUD_PROB_THRESHOLD: Final[float] = 0.85
RULE_A_DISPUTE_THRESHOLD: Final[float] = 0.25
RULE_A_MISMATCH_THRESHOLD: Final[float] = 0.80

# Rule C: Trusted merchant whitelisting criteria
TRUSTED_ACCOUNT_AGE_DAYS: Final[float] = 60.0
TRUSTED_COMPLETED_ORDERS: Final[int] = 10
TRUSTED_DISPUTE_RATE_MAX: Final[float] = 0.05
TRUST_DAMPENING_FACTOR: Final[float] = 0.70  # Reduces false friction by 30% for trusted merchants

# ---------------------------------------------------------------------------
# Decision Thresholds (Named Constants - No Magic Numbers)
# ---------------------------------------------------------------------------
DECISION_ALLOW: Final[str] = "ALLOW"
DECISION_HOLD: Final[str] = "HOLD_FOR_REVIEW"
DECISION_BLOCK: Final[str] = "BLOCK"

ALLOW_UPPER_BOUND: Final[float] = 0.40  # Scores < 0.40 are ALLOW
HOLD_UPPER_BOUND: Final[float] = 0.75   # Scores between 0.40 and 0.75 are HOLD
# Scores > 0.75 are BLOCK

# Critical overrides for Layer 1 violations
CRITICAL_SLANG_THRESHOLD: Final[float] = 0.80  # Hard drugs/weapons in Swahili/English
CRITICAL_MISMATCH_THRESHOLD: Final[float] = 0.85 # Extreme image-text discrepancy
CRITICAL_NSFW_THRESHOLD: Final[float] = 0.70    # High confidence adult or abusive media

# ---------------------------------------------------------------------------
# Vision & Language Model Configurations
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME: Final[str] = os.getenv(
    "CLIP_MODEL_NAME", "openai/clip-vit-base-patch32"
)
USE_REAL_CLIP: Final[bool] = os.getenv("USE_REAL_CLIP", "true").lower() in ("true", "1", "yes")

# Check A: Direct Contraband & NSFW Visual Scan Thresholds & Prompts
VISUAL_CONTRABAND_THRESHOLD: Final[float] = 0.35  # Combined probability of illicit concepts

# Check B: Title-to-Image Alignment Thresholds
VISUAL_MISMATCH_THRESHOLD: Final[float] = 0.65  # Mismatch > 0.65 triggers severe discrepancy flag

# ---------------------------------------------------------------------------
# Fine-Grained 36-Category Mutually Exclusive Retail & Safety Visual Taxonomy
# (Replaces vague "normal retail consumer product" with specific categories)
# ---------------------------------------------------------------------------
FINE_GRAINED_VISUAL_TAXONOMY: Final[dict] = {
    # --- Fashion: Womenswear, Menswear, Footwear, Accessories ---
    "WOMENS_DRESS": {
        "label": "a women's dress, gown, or African print kitenge dress",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"WOMENS_DRESS", "WOMENS_HEELS_SANDALS", "HANDBAG_PURSE"},
    },
    "MENS_SUIT": {
        "label": "a men's formal suit, blazer, tuxedo, or menswear",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"MENS_SUIT", "BUTTON_SHIRT", "FORMAL_SHOES"},
    },
    "TSHIRT_CASUAL": {
        "label": "a casual t-shirt, polo shirt, or sports jersey",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"TSHIRT_CASUAL", "JEANS_TROUSERS", "BUTTON_SHIRT"},
    },
    "BUTTON_SHIRT": {
        "label": "a formal button-down collared dress shirt",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"BUTTON_SHIRT", "MENS_SUIT", "JEANS_TROUSERS"},
    },
    "JEANS_TROUSERS": {
        "label": "jeans, denim pants, or casual trousers",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"JEANS_TROUSERS", "TSHIRT_CASUAL", "BUTTON_SHIRT"},
    },
    "SNEAKERS": {
        "label": "sneakers, athletic running shoes, or trainers",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"SNEAKERS", "FORMAL_SHOES"},
    },
    "FORMAL_SHOES": {
        "label": "men's leather dress shoes, loafers, or oxford shoes",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"FORMAL_SHOES", "MENS_SUIT", "SNEAKERS"},
    },
    "WOMENS_HEELS_SANDALS": {
        "label": "women's high heels, open sandals, or slippers",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"WOMENS_HEELS_SANDALS", "WOMENS_DRESS", "HANDBAG_PURSE"},
    },
    "HANDBAG_PURSE": {
        "label": "a women's handbag, purse, clutch, or tote bag",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"HANDBAG_PURSE", "BACKPACK_BAG", "WOMENS_DRESS"},
    },
    "BACKPACK_BAG": {
        "label": "a travel backpack, school bag, or duffel bag",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"BACKPACK_BAG", "HANDBAG_PURSE"},
    },
    "WRISTWATCH": {
        "label": "a luxury wristwatch, digital watch, or smart watch",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"WRISTWATCH", "JEWELRY"},
    },
    "JEWELRY": {
        "label": "gold or silver jewelry, necklace, ring, or earrings",
        "retail_domain": "fashion",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"JEWELRY", "WRISTWATCH"},
    },

    # --- Electronics: Phones, Computing, Accessories ---
    "SMARTPHONE": {
        "label": "a smartphone, mobile phone, or cellular telephone",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"SMARTPHONE", "TABLET"},
    },
    "LAPTOP": {
        "label": "a laptop computer, notebook, or macbook",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"LAPTOP", "TABLET"},
    },
    "PHONE_CHARGER": {
        "label": "a phone charger, charging cable, USB adapter, or power bank",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"PHONE_CHARGER"},
    },
    "HEADPHONES": {
        "label": "wireless earbuds, in-ear earphones, or audio headphones",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"HEADPHONES"},
    },
    "TELEVISION": {
        "label": "a flat screen television, smart TV, or display monitor",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"TELEVISION"},
    },
    "TABLET": {
        "label": "a tablet computer, iPad, or digital reading tablet",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"TABLET", "SMARTPHONE", "LAPTOP"},
    },
    "CAMERA": {
        "label": "a digital camera, DSLR camera, or photography lens",
        "retail_domain": "electronics",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"CAMERA"},
    },

    # --- Groceries, Home Goods & Beauty ---
    "FLOUR_BAG": {
        "label": "packaged maize flour, wheat flour, or grain flour bag",
        "retail_domain": "groceries",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"FLOUR_BAG", "RICE_GRAIN"},
    },
    "RICE_GRAIN": {
        "label": "a sack of rice, dried beans, or agricultural dry food grains",
        "retail_domain": "groceries",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"RICE_GRAIN", "FLOUR_BAG"},
    },
    "COOKING_OIL": {
        "label": "a bottle of vegetable cooking oil or sunflower oil container",
        "retail_domain": "groceries",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"COOKING_OIL"},
    },
    "FRESH_PRODUCE": {
        "label": "fresh raw vegetables, tomatoes, onions, or fresh fruits",
        "retail_domain": "groceries",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"FRESH_PRODUCE"},
    },
    "KITCHEN_UTENSILS": {
        "label": "kitchen knives, cooking pots, pans, or kitchenware utensils",
        "retail_domain": "home_goods",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"KITCHEN_UTENSILS"},
    },
    "FURNITURE_SOFA": {
        "label": "living room furniture, sofa couch, armchair, or table",
        "retail_domain": "home_goods",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"FURNITURE_SOFA", "BED_MATTRESS"},
    },
    "BED_MATTRESS": {
        "label": "a bedroom bed, sleeping mattress, or bedding pillows",
        "retail_domain": "home_goods",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"BED_MATTRESS", "FURNITURE_SOFA"},
    },
    "HOME_APPLIANCE": {
        "label": "electric clothes iron, kitchen blender, or small home appliance",
        "retail_domain": "home_goods",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"HOME_APPLIANCE", "KITCHEN_UTENSILS"},
    },
    "PERFUME": {
        "label": "a glass perfume bottle, cologne fragrance, or body mist",
        "retail_domain": "beauty",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"PERFUME", "COSMETICS_LOTION"},
    },
    "COSMETICS_LOTION": {
        "label": "skin lotion cream, makeup lipstick, or facial skincare cosmetics",
        "retail_domain": "beauty",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"COSMETICS_LOTION", "PERFUME"},
    },
    "HAIR_WIG": {
        "label": "a human hair wig, lace frontal, or hair extensions weave",
        "retail_domain": "beauty",
        "is_contraband": False,
        "is_non_product": False,
        "compatible_keys": {"HAIR_WIG", "COSMETICS_LOTION"},
    },

    # --- Non-Product / Fraud / Document / Screenshot ---
    "SCREENSHOT": {
        "label": "a mobile phone screenshot, digital screen capture, or social media app interface",
        "retail_domain": "non_product",
        "is_contraband": False,
        "is_non_product": True,
        "compatible_keys": set(),
    },
    "DOCUMENT_RECEIPT": {
        "label": "a printed paper document, sales receipt, payment invoice, or paper slip",
        "retail_domain": "non_product",
        "is_contraband": False,
        "is_non_product": True,
        "compatible_keys": set(),
    },

    # --- Contraband / Weapons / NSFW (Hard Block Concepts) ---
    "CANNABIS": {
        "label": "cannabis marijuana weed leaves or dried buds",
        "retail_domain": "contraband",
        "is_contraband": True,
        "is_non_product": False,
        "compatible_keys": set(),
    },
    "NARCOTICS": {
        "label": "illegal narcotics pills, drug powder, or contraband drug sachets",
        "retail_domain": "contraband",
        "is_contraband": True,
        "is_non_product": False,
        "compatible_keys": set(),
    },
    "WEAPON": {
        "label": "a handgun, pistol, military rifle firearm, or combat weapon",
        "retail_domain": "contraband",
        "is_contraband": True,
        "is_non_product": False,
        "compatible_keys": set(),
    },
    "EXPLICIT_NSFW": {
        "label": "explicit adult nudity, pornography, or naked human body parts",
        "retail_domain": "contraband",
        "is_contraband": True,
        "is_non_product": False,
        "compatible_keys": set(),
    },
}

# ---------------------------------------------------------------------------
# River Online Model Hyperparameters
# ---------------------------------------------------------------------------
LOGISTIC_LEARNING_RATE: Final[float] = 0.05
LOGISTIC_L2_PENALTY: Final[float] = 1e-4
COLD_START_SAMPLES: Final[int] = 2000
