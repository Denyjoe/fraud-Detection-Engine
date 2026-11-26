"""Application Configuration and Constants.

Defines system thresholds, decision names, file paths and the visual taxonomy for the
Tanzanian Fraud Detection and Moderation Engine.

Tunable thresholds and model names live in ``config/config.json`` and the editable
Swahili/English slang lexicon lives in ``config/lexicon.json``; both are loaded here.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Final, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Base Directories
# ---------------------------------------------------------------------------
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent
MODELS_DIR: Final[Path] = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR: Final[Path] = BASE_DIR / "data"

# ---------------------------------------------------------------------------
# Editable JSON configuration (thresholds + model names) and slang lexicon
# ---------------------------------------------------------------------------
CONFIG_PATH: Final[Path] = Path(os.getenv("FRAUD_CONFIG_PATH", str(BASE_DIR / "config" / "config.json")))
LEXICON_PATH: Final[Path] = Path(os.getenv("LEXICON_PATH", str(BASE_DIR / "config" / "lexicon.json")))
HASH_BLOCKLIST_PATH: Final[Path] = Path(
    os.getenv("HASH_BLOCKLIST_PATH", str(DATA_DIR / "hash_blocklist.json"))
)


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load the JSON configuration file and apply environment overrides.

    Environment overrides:
        MODERATION_DISABLE_MODELS: comma-separated model keys to switch off
            (e.g. ``clip,ocr``) or ``all`` to run the text rules only.
        CLIP_MODEL_NAME / USE_REAL_CLIP: kept for backward compatibility.

    Parameters:
        path (Path): Location of config.json.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    with open(path, encoding="utf-8") as handle:
        cfg: Dict[str, Any] = json.load(handle)

    models: Dict[str, Dict[str, Any]] = cfg.setdefault("models", {})
    disabled = os.getenv("MODERATION_DISABLE_MODELS", "").strip().lower()
    if disabled:
        keys = list(models) if disabled == "all" else [k.strip() for k in disabled.split(",")]
        for key in keys:
            if key in models:
                models[key]["enabled"] = False
    if os.getenv("CLIP_MODEL_NAME"):
        models.setdefault("clip", {})["name"] = os.environ["CLIP_MODEL_NAME"]
    if os.getenv("USE_REAL_CLIP", "true").lower() not in ("true", "1", "yes"):
        models.setdefault("clip", {})["enabled"] = False
    return cfg


CONFIG: Final[Dict[str, Any]] = load_config()


def enable_utf8_console() -> None:
    """Make stdout/stderr UTF-8 so Swahili text and emoji print on Windows consoles."""
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass

# ---------------------------------------------------------------------------
# File Paths & Database Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL_FILE: Final[str] = os.getenv(
    "FRAUD_MODEL_PATH", str(MODELS_DIR / "fraud_model.joblib")
)
DEFAULT_DB_FILE: Final[str] = os.getenv(
    "AUDIT_DB_PATH", str(BASE_DIR / "audit_logs.db")
)
META_MODEL_FILE: Final[str] = os.getenv(
    "META_MODEL_PATH", str(BASE_DIR / CONFIG["decision"]["meta_model_path"])
)

# ---------------------------------------------------------------------------
# Currency and Regional Constants (Tanzanian Social Commerce)
# ---------------------------------------------------------------------------
CURRENCY_CODE: Final[str] = "TZS"
COUNTRY_CODE: Final[str] = "TZ"

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
# Guardrail & Trust Dampening Constants (Layer 2 seller/price risk)
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
# Decisions
# ---------------------------------------------------------------------------
DECISION_ALLOW: Final[str] = "ALLOW"
DECISION_REVIEW: Final[str] = "REVIEW"
DECISION_BLOCK: Final[str] = "BLOCK"
DECISION_ESCALATE: Final[str] = "ESCALATE"  # urgent human review (child safety)

# Severity order used when combining decisions from several checks
DECISION_SEVERITY: Final[Dict[str, int]] = {
    DECISION_ALLOW: 0,
    DECISION_REVIEW: 1,
    DECISION_BLOCK: 2,
    DECISION_ESCALATE: 3,
}

ALLOW_UPPER_BOUND: Final[float] = 0.40   # Layer 2 scores < 0.40 are ALLOW
REVIEW_UPPER_BOUND: Final[float] = 0.75  # Layer 2 scores 0.40 - 0.75 are REVIEW, above is BLOCK
RULE_B_ALLOW_CAP: Final[float] = 0.35   # Rule B: allowed listings report at most this probability
HIGH_PRICE_RATIO: Final[float] = 3.0     # New-account listing priced >= 3x category median is reviewed
NEW_ACCOUNT_DAYS: Final[float] = 30.0

# ---------------------------------------------------------------------------
# Vision-Language (CLIP) prompts
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME: Final[str] = CONFIG["models"]["clip"]["name"]

# Prohibited visual concepts. Probabilities of the prompts in one category are summed.
CLIP_CONTRABAND_PROMPTS: Final[Dict[str, List[str]]] = {
    "DRUGS": [
        "a photo of cannabis marijuana buds or a marijuana plant",
        "a photo of rolled marijuana joints",
        "a photo of white drug powder in small plastic sachets",
    ],
    "WEAPONS": [
        "a photo of a handgun pistol",
        "a photo of a rifle or assault rifle",
        "a photo of bullets and ammunition cartridges",
    ],
    "WILDLIFE": [
        "a photo of elephant ivory tusks",
        "a photo of a leopard skin or animal pelt",
        "a photo of a pangolin or pangolin scales",
    ],
    "FINANCIAL_FRAUD": [
        "a photo of stacks of banknotes and cash money",
    ],
}

# Harmless look-alikes that absorb probability away from contraband prompts
# (leafy vegetables vs cannabis, toy guns vs real guns, flour vs powder ...).
CLIP_LOOKALIKE_PROMPTS: Final[List[str]] = [
    "a photo of fresh green leafy vegetables like spinach or sukuma wiki",
    "a photo of dried tea leaves or herbal tea",
    "a photo of potted house plants or garden plants",
    "a photo of a plastic toy gun or water gun",
    "a photo of a glue gun, drill or power tool",
    "a photo of white flour, sugar or salt in a bag",
    "a photo of medicine boxes in a pharmacy",
    "a photo of animal print fabric or clothing",
    "a photo of a cream-coloured ivory dress or fabric",
    "a photo of a person modelling clothes or swimwear",
    "a drawing, illustration or cartoon",
]

# ---------------------------------------------------------------------------
# Fine-Grained Mutually Exclusive Retail Visual Taxonomy
# Used for photo-vs-title mismatch. Contraband concepts live in CLIP_CONTRABAND_PROMPTS.
# ---------------------------------------------------------------------------


def _product(label: str, domain: str, compatible: set, non_product: bool = False) -> Dict[str, Any]:
    """Build one taxonomy entry.

    Parameters:
        label (str): CLIP prompt describing the product.
        domain (str): Retail domain.
        compatible (set): Taxonomy keys accepted as a match for this product.
        non_product (bool): True for screenshots/documents.

    Returns:
        Dict[str, Any]: Taxonomy entry.
    """
    return {
        "label": label,
        "retail_domain": domain,
        "is_non_product": non_product,
        "compatible_keys": compatible,
    }


FINE_GRAINED_VISUAL_TAXONOMY: Final[Dict[str, Dict[str, Any]]] = {
    # --- Fashion ---
    "WOMENS_DRESS": _product("a women's dress, gown, or African print kitenge dress", "fashion",
                             {"WOMENS_DRESS", "WOMENS_HEELS_SANDALS", "HANDBAG_PURSE"}),
    "MENS_SUIT": _product("a men's formal suit, blazer, tuxedo, or menswear", "fashion",
                          {"MENS_SUIT", "BUTTON_SHIRT", "FORMAL_SHOES"}),
    "TSHIRT_CASUAL": _product("a casual t-shirt, polo shirt, or sports jersey", "fashion",
                              {"TSHIRT_CASUAL", "JEANS_TROUSERS", "BUTTON_SHIRT"}),
    "BUTTON_SHIRT": _product("a formal button-down collared dress shirt", "fashion",
                             {"BUTTON_SHIRT", "MENS_SUIT", "JEANS_TROUSERS", "TSHIRT_CASUAL"}),
    "JEANS_TROUSERS": _product("jeans, denim pants, or casual trousers", "fashion",
                               {"JEANS_TROUSERS", "TSHIRT_CASUAL", "BUTTON_SHIRT"}),
    "SNEAKERS": _product("sneakers, athletic running shoes, or trainers", "fashion",
                         {"SNEAKERS", "FORMAL_SHOES", "WOMENS_HEELS_SANDALS"}),
    "FORMAL_SHOES": _product("men's leather dress shoes, loafers, or oxford shoes", "fashion",
                             {"FORMAL_SHOES", "MENS_SUIT", "SNEAKERS"}),
    "WOMENS_HEELS_SANDALS": _product("women's high heels, open sandals, or slippers", "fashion",
                                     {"WOMENS_HEELS_SANDALS", "WOMENS_DRESS", "HANDBAG_PURSE", "SNEAKERS",
                                      "FORMAL_SHOES"}),
    "HANDBAG_PURSE": _product("a women's handbag, purse, clutch, or tote bag", "fashion",
                              {"HANDBAG_PURSE", "BACKPACK_BAG", "WOMENS_DRESS"}),
    "BACKPACK_BAG": _product("a travel backpack, school bag, or duffel bag", "fashion",
                             {"BACKPACK_BAG", "HANDBAG_PURSE"}),
    "WRISTWATCH": _product("a luxury wristwatch, digital watch, or smart watch", "fashion",
                           {"WRISTWATCH", "JEWELRY"}),
    "JEWELRY": _product("gold or silver jewelry, necklace, ring, or earrings", "fashion",
                        {"JEWELRY", "WRISTWATCH"}),
    # --- Electronics ---
    "SMARTPHONE": _product("a smartphone, mobile phone, or cellular telephone", "electronics",
                           {"SMARTPHONE", "TABLET", "SCREENSHOT"}),
    "LAPTOP": _product("a laptop computer, notebook, or macbook", "electronics", {"LAPTOP", "TABLET"}),
    "PHONE_CHARGER": _product("a phone charger, charging cable, USB adapter, or power bank", "electronics",
                              {"PHONE_CHARGER", "SOLAR_PANEL"}),
    "HEADPHONES": _product("wireless earbuds, in-ear earphones, or audio headphones", "electronics",
                           {"HEADPHONES"}),
    "TELEVISION": _product("a flat screen television, smart TV, or display monitor", "electronics",
                           {"TELEVISION", "LAPTOP"}),
    "TABLET": _product("a tablet computer, iPad, or digital reading tablet", "electronics",
                       {"TABLET", "SMARTPHONE", "LAPTOP"}),
    "CAMERA": _product("a digital camera, DSLR camera, or photography lens", "electronics", {"CAMERA"}),
    # --- Groceries, Home Goods & Beauty ---
    "FLOUR_BAG": _product("packaged maize flour, wheat flour, or grain flour bag", "groceries",
                          {"FLOUR_BAG", "RICE_GRAIN"}),
    "RICE_GRAIN": _product("a sack of rice, dried beans, or agricultural dry food grains", "groceries",
                           {"RICE_GRAIN", "FLOUR_BAG"}),
    "COOKING_OIL": _product("a bottle of vegetable cooking oil or sunflower oil container", "groceries",
                            {"COOKING_OIL"}),
    "FRESH_PRODUCE": _product("fresh raw vegetables, tomatoes, onions, or fresh fruits", "groceries",
                              {"FRESH_PRODUCE", "LEAFY_GREENS"}),
    "KITCHEN_UTENSILS": _product("kitchen knives, cooking pots, pans, or kitchenware utensils", "home_goods",
                                 {"KITCHEN_UTENSILS", "HOME_APPLIANCE"}),
    "FURNITURE_SOFA": _product("living room furniture, sofa couch, armchair, or table", "home_goods",
                               {"FURNITURE_SOFA", "BED_MATTRESS"}),
    "BED_MATTRESS": _product("a bedroom bed, sleeping mattress, or bedding pillows", "home_goods",
                             {"BED_MATTRESS", "FURNITURE_SOFA"}),
    "HOME_APPLIANCE": _product("electric clothes iron, kitchen blender, or small home appliance", "home_goods",
                               {"HOME_APPLIANCE", "KITCHEN_UTENSILS"}),
    "PERFUME": _product("a glass perfume bottle, cologne fragrance, or body mist", "beauty",
                        {"PERFUME", "COSMETICS_LOTION"}),
    "COSMETICS_LOTION": _product("skin lotion cream, makeup lipstick, or facial skincare cosmetics", "beauty",
                                 {"COSMETICS_LOTION", "PERFUME", "MEDICINE_PHARMACY"}),
    "HAIR_WIG": _product("a human hair wig, lace frontal, or hair extensions weave", "beauty",
                         {"HAIR_WIG", "COSMETICS_LOTION"}),
    # --- Common Tanzanian listings outside the core categories (distractors) ---
    "BICYCLE_MOTORBIKE": _product("a bicycle, motorbike, bodaboda or bajaji", "vehicles",
                                  {"BICYCLE_MOTORBIKE", "VEHICLE_PARTS"}),
    "VEHICLE_PARTS": _product("car spare parts, engine parts, tyres or exhaust silencer", "vehicles",
                              {"VEHICLE_PARTS", "BICYCLE_MOTORBIKE", "TOOLS_HARDWARE"}),
    "TOOLS_HARDWARE": _product("hand tools, power tools, glue gun, drill or hardware", "home_goods",
                               {"TOOLS_HARDWARE", "VEHICLE_PARTS", "BUILDING_MATERIALS"}),
    "BUILDING_MATERIALS": _product("cement bags, iron roofing sheets, bricks or building materials", "home_goods",
                                   {"BUILDING_MATERIALS", "TOOLS_HARDWARE"}),
    "BOOKS_STATIONERY": _product("books, exercise books, pens or school stationery", "general",
                                 {"BOOKS_STATIONERY"}),
    "TOYS": _product("children's toys, dolls or a plastic toy", "general", {"TOYS"}),
    "LIVESTOCK": _product("live chickens, goats, cows or livestock", "groceries", {"LIVESTOCK"}),
    "LEAFY_GREENS": _product("fresh leafy green vegetables, spinach or tea leaves", "groceries",
                             {"LEAFY_GREENS", "FRESH_PRODUCE"}),
    "SOLAR_PANEL": _product("a solar panel, battery or inverter", "electronics", {"SOLAR_PANEL", "PHONE_CHARGER"}),
    "MEDICINE_PHARMACY": _product("packaged medicine, pharmacy products or supplements", "beauty",
                                  {"MEDICINE_PHARMACY", "COSMETICS_LOTION"}),
    "SPORTS_EQUIPMENT": _product("a football, gym weights or sports equipment", "general",
                                 {"SPORTS_EQUIPMENT", "SNEAKERS", "TSHIRT_CASUAL"}),
    # --- Non-Product (screenshots / documents) ---
    "SCREENSHOT": _product("a mobile phone screenshot, digital screen capture, or social media app interface",
                           "non_product", set(), non_product=True),
    "DOCUMENT_RECEIPT": _product("a printed paper document, sales receipt, payment invoice, or paper slip",
                                 "non_product", set(), non_product=True),
}

# ---------------------------------------------------------------------------
# River Online Model Hyperparameters
# ---------------------------------------------------------------------------
LOGISTIC_L2_PENALTY: Final[float] = 1e-4
COLD_START_SAMPLES: Final[int] = 2000
