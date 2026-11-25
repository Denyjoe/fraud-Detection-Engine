"""Layer 1: Illegal Goods & Content Moderation Engine.

Performs pre-listing inspection for Tanzanian Social Commerce:
1. Context-aware Swahili & English slang and prohibited terms analysis.
   - Disambiguates polysemous words ("unga wa sembe" flour vs narcotics, "kisu cha jikoni" vs weapon).
   - Zero tolerance on unambiguous contraband ("bangi", "cha arusha", "msokoto", "bastola", "upatu").
2. Zero-shot Vision-Language image-to-text mismatch detection using CLIP.
3. NSFW and abusive imagery detection.
4. Automatic category inference across Tanzanian retail categories.
"""

import base64
import io
import logging
import re
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
from PIL import Image

from config.settings import (
    CATEGORY_BEAUTY,
    CATEGORY_ELECTRONICS,
    CATEGORY_FASHION,
    CATEGORY_GENERAL,
    CATEGORY_GROCERIES,
    CATEGORY_HOME_GOODS,
    CLIP_MODEL_NAME,
    CRITICAL_MISMATCH_THRESHOLD,
    CRITICAL_NSFW_THRESHOLD,
    CRITICAL_SLANG_THRESHOLD,
    DEFAULT_CATEGORY,
    FINE_GRAINED_VISUAL_TAXONOMY,
    SUPPORTED_CATEGORIES,
    USE_REAL_CLIP,
    VISUAL_CONTRABAND_THRESHOLD,
    VISUAL_MISMATCH_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Global thread-safe singleton cache for HuggingFace CLIP model & processor
_clip_model_lock = threading.Lock()
_global_clip_processor: Optional[Any] = None
_global_clip_model: Optional[Any] = None
_global_clip_load_failed: bool = False

# ---------------------------------------------------------------------------
# Tanzanian Unambiguous Prohibited Goods & Coded Slang Lexicon
# (Zero-tolerance: always illegal in all contexts)
# ---------------------------------------------------------------------------
UNAMBIGUOUS_PROHIBITED_LEXICON: Dict[str, Dict[str, Any]] = {
    # Unambiguous Narcotics & Controlled Substances
    "bangi": {"category": "NARCOTICS", "weight": 0.95, "severity": "CRITICAL"},
    "weed": {"category": "NARCOTICS", "weight": 0.90, "severity": "HIGH"},
    "msokoto": {"category": "NARCOTICS", "weight": 0.90, "severity": "HIGH"},
    "ganja": {"category": "NARCOTICS", "weight": 0.90, "severity": "HIGH"},
    "kush": {"category": "NARCOTICS", "weight": 0.85, "severity": "HIGH"},
    "marijuana": {"category": "NARCOTICS", "weight": 0.95, "severity": "CRITICAL"},
    "cha arusha": {"category": "NARCOTICS", "weight": 0.95, "severity": "CRITICAL"},  # Famous TZ weed slang
    "ngada": {"category": "NARCOTICS", "weight": 0.95, "severity": "CRITICAL"},      # Hard drug slang
    "kete": {"category": "NARCOTICS", "weight": 0.85, "severity": "CRITICAL"},       # Drug pellets/sachets
    "madawa ya kulevya": {"category": "NARCOTICS", "weight": 1.00, "severity": "CRITICAL"},
    "tramadol": {"category": "NARCOTICS", "weight": 0.95, "severity": "CRITICAL"},
    "cocaine": {"category": "NARCOTICS", "weight": 1.00, "severity": "CRITICAL"},
    "heroin": {"category": "NARCOTICS", "weight": 1.00, "severity": "CRITICAL"},

    # Unambiguous Firearms, Weapons & Explosives
    "bastola": {"category": "WEAPONS", "weight": 1.00, "severity": "CRITICAL"},
    "bunduki": {"category": "WEAPONS", "weight": 1.00, "severity": "CRITICAL"},
    "risasi": {"category": "WEAPONS", "weight": 0.95, "severity": "CRITICAL"},
    "ammunition": {"category": "WEAPONS", "weight": 0.95, "severity": "CRITICAL"},
    "silaha": {"category": "WEAPONS", "weight": 0.90, "severity": "CRITICAL"},
    "grenade": {"category": "WEAPONS", "weight": 1.00, "severity": "CRITICAL"},
    "glock": {"category": "WEAPONS", "weight": 1.00, "severity": "CRITICAL"},
    "firearm": {"category": "WEAPONS", "weight": 1.00, "severity": "CRITICAL"},
    "kisu cha kivita": {"category": "WEAPONS", "weight": 0.90, "severity": "CRITICAL"},
    "panga la kukata": {"category": "WEAPONS", "weight": 0.75, "severity": "HIGH"},

    # Counterfeits, Fake Currencies & Cloned Documents
    "noti bandia": {"category": "FINANCIAL_FRAUD", "weight": 1.00, "severity": "CRITICAL"},
    "pesa bandia": {"category": "FINANCIAL_FRAUD", "weight": 1.00, "severity": "CRITICAL"},
    "tengeneza noti": {"category": "FINANCIAL_FRAUD", "weight": 0.95, "severity": "CRITICAL"},
    "counterfeit money": {"category": "FINANCIAL_FRAUD", "weight": 1.00, "severity": "CRITICAL"},
    "vyeti bandia": {"category": "COUNTERFEIT", "weight": 0.90, "severity": "CRITICAL"},
    "leseni feki": {"category": "COUNTERFEIT", "weight": 0.90, "severity": "CRITICAL"},
    "cloned card": {"category": "FINANCIAL_FRAUD", "weight": 0.95, "severity": "CRITICAL"},

    # Social Commerce Scams & Ponzi Schemes
    "upatu": {"category": "FINANCIAL_FRAUD", "weight": 0.85, "severity": "HIGH"},
    "mchezo wa pesa": {"category": "FINANCIAL_FRAUD", "weight": 0.80, "severity": "HIGH"},
    "tuma kwa namba hii": {"category": "FINANCIAL_FRAUD", "weight": 0.85, "severity": "HIGH"},
    "utajiri wa haraka": {"category": "FINANCIAL_FRAUD", "weight": 0.75, "severity": "HIGH"},
    "fursa ya kujiunga": {"category": "FINANCIAL_FRAUD", "weight": 0.70, "severity": "MEDIUM"},
    "pesa mara mbili": {"category": "FINANCIAL_FRAUD", "weight": 0.90, "severity": "CRITICAL"},
    "kubeti fixed matches": {"category": "FINANCIAL_FRAUD", "weight": 0.85, "severity": "HIGH"},
    "kuongeza salio": {"category": "FINANCIAL_FRAUD", "weight": 0.80, "severity": "HIGH"},
    "simu ya wizi": {"category": "COUNTERFEIT", "weight": 0.95, "severity": "CRITICAL"},
    "imei blocked": {"category": "COUNTERFEIT", "weight": 0.85, "severity": "HIGH"},
}

# ---------------------------------------------------------------------------
# Context Patterns for Ambiguous Swahili Words
# (Prevents false positives on legitimate commerce)
# ---------------------------------------------------------------------------
# 1. "Unga": Flour vs. Narcotics
UNGA_GROCERY_REGEX = re.compile(
    r"\b(unga\s+wa\s+(sembe|ngano|dona|muhogo|lishe|mahindi|ubuyu|mpunga|keki|njano|ng'ombe)|"
    r"sembe|ngano|dona|lishe|kilo|kg|kilo\s*\d+|\d+\s*kg|mchele|sukari|chumvi|mafuta|"
    r"chakula|duka\s+la\s+vyakula|kupika|ugali|chapati|bakhresa|azam)\b",
    re.IGNORECASE,
)
UNGA_DRUG_REGEX = re.compile(
    r"\b(unga\s+(mweupe|wa\s+kulevya|ngada|poda|haramu)|kunusa\s+unga|kete\s+za\s+unga)\b",
    re.IGNORECASE,
)

# 2. "Kisu": Kitchen knife vs. Weapon
KISU_KITCHEN_REGEX = re.compile(
    r"\b(kisu\s+cha\s+jikoni|kisu\s+cha\s+nyama|kisu\s+cha\s+kukatia|kisu\s+cha\s+matunda|"
    r"visu\s+vya\s+jikoni|vyombo\s+vya\s+jikoni|jiko|vyombo|mapishi|kisu\s+kipya)\b",
    re.IGNORECASE,
)
KISU_WEAPON_REGEX = re.compile(
    r"\b(kisu\s+cha\s+kivita|kisu\s+cha\s+kijeshi|kisu\s+cha\s+mapigano|kisu\s+cha\s+kuua)\b",
    re.IGNORECASE,
)

# 3. "Majani": Tea leaves vs. Cannabis
MAJANI_TEA_REGEX = re.compile(
    r"\b(majani\s+ya\s+chai|chai\s+ya\s+majani|majani\s+ya\s+kijani|chai\s+ya\s+tangawizi|"
    r"chai|kilimo|bustani|shamba)\b",
    re.IGNORECASE,
)
MAJANI_DRUG_REGEX = re.compile(
    r"\b(majani\s+ya\s+bangi|majani\s+ya\s+kush|majani\s+ya\s+kuvuta)\b",
    re.IGNORECASE,
)

# 4. "Kitunguu": Onion vs. Drug baggie
KITUNGUU_GROCERY_REGEX = re.compile(
    r"\b(kitunguu\s+maji|kitunguu\s+saumu|vitunguu|nyanya|mbogamboga|viazi|soko|chakula)\b",
    re.IGNORECASE,
)


def infer_listing_category(
    title: str, description: str, explicit_category: Optional[str] = None
) -> str:
    """Infer product category from title/description keywords if not explicitly provided.

    Parameters:
        title (str): Product listing title.
        description (str): Product description.
        explicit_category (Optional[str]): Explicit user-provided category.

    Returns:
        str: Standardized category string ('fashion', 'electronics', 'groceries', 'home_goods', 'beauty', 'general').
    """
    if explicit_category and explicit_category.lower() in SUPPORTED_CATEGORIES:
        return explicit_category.lower()

    combined_text = f"{title} {description}".lower()

    if re.search(
        r"\b(simu|chaja|type-c|earphones|earpods|laptop|tv|redio|betri|charger|"
        r"iphone|samsung|android|screen|cable|headphone|airpods|adapter|computer|tablet|kebo)\b",
        combined_text,
    ):
        return CATEGORY_ELECTRONICS

    if re.search(
        r"\b(unga|sembe|ngano|dona|mchele|maharage|sukari|chai|majani ya chai|nyama|samaki|"
        r"chakula|flour|rice|sugar|food|vitunguu|nyanya|mbogamboga|mafuta ya kula|kilo|kg)\b",
        combined_text,
    ):
        return CATEGORY_GROCERIES

    if re.search(
        r"\b(gauni|nguo|dera|suti|viatu|kitenge|vitenge|shati|suruali|kofia|mikoba|"
        r"pochi|sandals|dress|shoes|clothing|sneakers|blouse|skirt|t-shirt)\b",
        combined_text,
    ):
        return CATEGORY_FASHION

    if re.search(
        r"\b(kisu cha jikoni|jiko|meza|kitanda|godoro|pazia|feni|sofa|vyombo|sahani|furniture|blender|pasi|kikombe)\b",
        combined_text,
    ):
        return CATEGORY_HOME_GOODS

    if re.search(
        r"\b(perfume|mafuta ya kupaka|nywele|weave|wigi|lipstick|lotion|sabuni|cosmetics|makeup|wanja|shampoo)\b",
        combined_text,
    ):
        return CATEGORY_BEAUTY

    return DEFAULT_CATEGORY


# ---------------------------------------------------------------------------
# Swahili-to-English Retail Visual Mapping Dictionary
# Maps Tanzanian retail titles, slang, and keywords to precise CLIP visual prompts
# and fine-grained taxonomy keys with compatible categories.
# ---------------------------------------------------------------------------
SWAHILI_RETAIL_CONCEPT_MAPPINGS: List[Dict[str, Any]] = [
    # Fashion: Women's dresses, vitenge, dera, gowns
    {
        "keywords": ["gauni", "vitenge", "kitenge", "dera", "madera", "sketi", "blouse", "vazi la kike", "dress", "gown"],
        "concept": "a women's dress, gown, or African print kitenge dress",
        "taxonomy_key": "WOMENS_DRESS",
        "compatible_keys": {"WOMENS_DRESS", "WOMENS_HEELS_SANDALS", "HANDBAG_PURSE"},
    },
    # Fashion: Men's suits, suti, tuxedos, blazers
    {
        "keywords": ["suti", "blazer", "tuxedo", "suti ya kiume", "menswear", "suit"],
        "concept": "a men's formal suit, blazer, tuxedo, or menswear",
        "taxonomy_key": "MENS_SUIT",
        "compatible_keys": {"MENS_SUIT", "BUTTON_SHIRT", "FORMAL_SHOES"},
    },
    # Fashion: T-shirts, polo, tisheti, jersey
    {
        "keywords": ["t-shirt", "tisheti", "tshirt", "jezi", "polo", "fulana"],
        "concept": "a casual t-shirt, polo shirt, or sports jersey",
        "taxonomy_key": "TSHIRT_CASUAL",
        "compatible_keys": {"TSHIRT_CASUAL", "JEANS_TROUSERS", "BUTTON_SHIRT"},
    },
    # Fashion: Formal shirts, shati
    {
        "keywords": ["shati", "mashati", "shati la mikono", "dress shirt"],
        "concept": "a formal button-down collared dress shirt",
        "taxonomy_key": "BUTTON_SHIRT",
        "compatible_keys": {"BUTTON_SHIRT", "MENS_SUIT", "JEANS_TROUSERS"},
    },
    # Fashion: Jeans, suruali, denim, trousers
    {
        "keywords": ["suruali", "jeans", "jinzi", "kaptula", "trousers", "pants"],
        "concept": "jeans, denim pants, or casual trousers",
        "taxonomy_key": "JEANS_TROUSERS",
        "compatible_keys": {"JEANS_TROUSERS", "TSHIRT_CASUAL", "BUTTON_SHIRT"},
    },
    # Fashion: Sneakers, raba, viatu vya michezo, trainers
    {
        "keywords": ["raba", "sneakers", "viatu vya michezo", "trainers", "converse", "nike", "jordan"],
        "concept": "sneakers, athletic running shoes, or trainers",
        "taxonomy_key": "SNEAKERS",
        "compatible_keys": {"SNEAKERS", "FORMAL_SHOES"},
    },
    # Fashion: Formal shoes, viatu vya ngozi, moccasin, loafers
    {
        "keywords": ["viatu vya ngozi", "moccasin", "loafers", "oxford shoes", "viatu vya kiume"],
        "concept": "men's leather dress shoes, loafers, or oxford shoes",
        "taxonomy_key": "FORMAL_SHOES",
        "compatible_keys": {"FORMAL_SHOES", "MENS_SUIT", "SNEAKERS"},
    },
    # Fashion: General shoes, heels, viatu
    {
        "keywords": ["viatu", "sandals", "kandambili", "ndala", "slippers", "yeboyebo", "heels", "high heels"],
        "concept": "women's high heels, open sandals, or slippers",
        "taxonomy_key": "WOMENS_HEELS_SANDALS",
        "compatible_keys": {"WOMENS_HEELS_SANDALS", "WOMENS_DRESS", "SNEAKERS", "FORMAL_SHOES"},
    },
    # Fashion: Handbags, pochi, mikoba
    {
        "keywords": ["pochi", "mkoba", "mikoba", "handbag", "purse", "clutch", "tote bag"],
        "concept": "a women's handbag, purse, clutch, or tote bag",
        "taxonomy_key": "HANDBAG_PURSE",
        "compatible_keys": {"HANDBAG_PURSE", "BACKPACK_BAG", "WOMENS_DRESS"},
    },
    # Fashion: Backpack, begi la mgongo, travel bag
    {
        "keywords": ["begi la mgongo", "begi", "backpack", "duffel bag", "travel bag"],
        "concept": "a travel backpack, school bag, or duffel bag",
        "taxonomy_key": "BACKPACK_BAG",
        "compatible_keys": {"BACKPACK_BAG", "HANDBAG_PURSE"},
    },
    # Fashion: Wristwatch, saa ya mkononi
    {
        "keywords": ["saa ya mkononi", "wristwatch", "smartwatch", "smart watch", "rolex"],
        "concept": "a luxury wristwatch, digital watch, or smart watch",
        "taxonomy_key": "WRISTWATCH",
        "compatible_keys": {"WRISTWATCH", "JEWELRY"},
    },
    # Fashion: Jewelry, hereni, mkufu, pete
    {
        "keywords": ["mkufu", "hereni", "pete", "cheni", "bangili", "jewelry", "necklace", "ring"],
        "concept": "gold or silver jewelry, necklace, ring, or earrings",
        "taxonomy_key": "JEWELRY",
        "compatible_keys": {"JEWELRY", "WRISTWATCH"},
    },

    # Electronics: Smartphone, simu
    {
        "keywords": ["simu", "smartphone", "iphone", "samsung", "tecno", "infinix", "redmi", "xiaomi", "simu janja"],
        "concept": "a smartphone, mobile phone, or cellular telephone",
        "taxonomy_key": "SMARTPHONE",
        "compatible_keys": {"SMARTPHONE", "TABLET"},
    },
    # Electronics: Laptop, kompyuta
    {
        "keywords": ["laptop", "kompyuta", "macbook", "notebook", "hp", "dell", "lenovo"],
        "concept": "a laptop computer, notebook, or macbook",
        "taxonomy_key": "LAPTOP",
        "compatible_keys": {"LAPTOP", "TABLET"},
    },
    # Electronics: Charger, chaja, cable, adapter
    {
        "keywords": ["chaja", "charger", "type-c", "type c", "usb cable", "kebo", "adapter", "power bank", "fast charger"],
        "concept": "a phone charger, charging cable, USB adapter, or power bank",
        "taxonomy_key": "PHONE_CHARGER",
        "compatible_keys": {"PHONE_CHARGER"},
    },
    # Electronics: Headphones, earphones, airpods
    {
        "keywords": ["earphones", "earpods", "airpods", "headphones", "headset", "buds"],
        "concept": "wireless earbuds, in-ear earphones, or audio headphones",
        "taxonomy_key": "HEADPHONES",
        "compatible_keys": {"HEADPHONES"},
    },
    # Electronics: TV, runinga, monitor
    {
        "keywords": ["tv", "runinga", "televisheni", "flat screen", "monitor", "smart tv"],
        "concept": "a flat screen television, smart TV, or display monitor",
        "taxonomy_key": "TELEVISION",
        "compatible_keys": {"TELEVISION"},
    },
    # Electronics: Tablet, ipad
    {
        "keywords": ["tablet", "ipad"],
        "concept": "a tablet computer, iPad, or digital reading tablet",
        "taxonomy_key": "TABLET",
        "compatible_keys": {"TABLET", "SMARTPHONE", "LAPTOP"},
    },
    # Electronics: Camera, kamera
    {
        "keywords": ["kamera", "camera", "lens", "dslr"],
        "concept": "a digital camera, DSLR camera, or photography lens",
        "taxonomy_key": "CAMERA",
        "compatible_keys": {"CAMERA"},
    },

    # Groceries: Flour, unga
    {
        "keywords": ["unga wa sembe", "unga wa ngano", "unga wa dona", "unga wa lishe", "unga", "sembe", "ngano", "dona"],
        "concept": "packaged maize flour, wheat flour, or grain flour bag",
        "taxonomy_key": "FLOUR_BAG",
        "compatible_keys": {"FLOUR_BAG", "RICE_GRAIN"},
    },
    # Groceries: Rice, grains, beans, mchele
    {
        "keywords": ["mchele", "maharage", "sukari", "rice", "beans", "sugar", "nafaka"],
        "concept": "a sack of rice, dried beans, or agricultural dry food grains",
        "taxonomy_key": "RICE_GRAIN",
        "compatible_keys": {"RICE_GRAIN", "FLOUR_BAG"},
    },
    # Groceries: Cooking oil, mafuta ya kula, alizeti
    {
        "keywords": ["mafuta ya kula", "mafuta ya kupikia", "alizeti", "korie", "cooking oil", "vegetable oil"],
        "concept": "a bottle of vegetable cooking oil or sunflower oil container",
        "taxonomy_key": "COOKING_OIL",
        "compatible_keys": {"COOKING_OIL"},
    },
    # Groceries: Fresh produce, nyanya, vitunguu, mboga
    {
        "keywords": ["nyanya", "vitunguu", "kitunguu", "mbogamboga", "viazi", "matunda", "produce", "vegetables"],
        "concept": "fresh raw vegetables, tomatoes, onions, or fresh fruits",
        "taxonomy_key": "FRESH_PRODUCE",
        "compatible_keys": {"FRESH_PRODUCE"},
    },

    # Home Goods: Kitchen utensils, kisu cha jikoni, sufuria, vyombo
    {
        "keywords": ["kisu cha jikoni", "visu", "sufuria", "sahani", "vyombo vya jikoni", "cookware", "kitchen knife", "pans"],
        "concept": "kitchen knives, cooking pots, pans, or kitchenware utensils",
        "taxonomy_key": "KITCHEN_UTENSILS",
        "compatible_keys": {"KITCHEN_UTENSILS"},
    },
    # Home Goods: Sofa, meza, furniture
    {
        "keywords": ["sofa", "meza", "viti", "furniture", "kochi", "dining table"],
        "concept": "living room furniture, sofa couch, armchair, or table",
        "taxonomy_key": "FURNITURE_SOFA",
        "compatible_keys": {"FURNITURE_SOFA", "BED_MATTRESS"},
    },
    # Home Goods: Bed, mattress, godoro, kitanda
    {
        "keywords": ["kitanda", "godoro", "mto", "bed", "mattress", "bedding"],
        "concept": "a bedroom bed, sleeping mattress, or bedding pillows",
        "taxonomy_key": "BED_MATTRESS",
        "compatible_keys": {"BED_MATTRESS", "FURNITURE_SOFA"},
    },
    # Home Goods: Home appliances, pasi, blender, feni
    {
        "keywords": ["pasi", "blender", "feni", "heater", "juicer", "toaster", "microwave", "appliance"],
        "concept": "electric clothes iron, kitchen blender, or small home appliance",
        "taxonomy_key": "HOME_APPLIANCE",
        "compatible_keys": {"HOME_APPLIANCE", "KITCHEN_UTENSILS"},
    },

    # Beauty: Perfume, marashi
    {
        "keywords": ["perfume", "marashi", "cologne", "fragrance", "body spray", "attar"],
        "concept": "a glass perfume bottle, cologne fragrance, or body mist",
        "taxonomy_key": "PERFUME",
        "compatible_keys": {"PERFUME", "COSMETICS_LOTION"},
    },
    # Beauty: Cosmetics, lotion, wanja, lipstick
    {
        "keywords": ["lotion", "mafuta ya kupaka", "lipstick", "makeup", "vipodozi", "skincare", "cream", "shampoo"],
        "concept": "skin lotion cream, makeup lipstick, or facial skincare cosmetics",
        "taxonomy_key": "COSMETICS_LOTION",
        "compatible_keys": {"COSMETICS_LOTION", "PERFUME"},
    },
    # Beauty: Hair wig, weave, wigi, nywele
    {
        "keywords": ["wigi", "weave", "nywele", "wig", "lace frontal", "human hair"],
        "concept": "a human hair wig, lace frontal, or hair extensions weave",
        "taxonomy_key": "HAIR_WIG",
        "compatible_keys": {"HAIR_WIG", "COSMETICS_LOTION"},
    },
]


def translate_swahili_concept_to_english(
    title: str, description: str = "", category: Optional[str] = None
) -> Tuple[str, str, Set[str]]:
    """Translate Swahili product titles and descriptions to precise CLIP visual prompts and taxonomy keys.

    Parameters:
        title (str): Listing title in Swahili or English.
        description (str): Detailed product description.
        category (Optional[str]): Inferred or explicit retail category.

    Returns:
        Tuple[str, str, Set[str]]:
            - seller_claimed_concept: High-precision English visual concept description.
            - taxonomy_key: Primary key in FINE_GRAINED_VISUAL_TAXONOMY.
            - compatible_keys: Set of taxonomy keys considered legitimate matches.
    """
    clean_text = f"{title} {description}".lower()

    # Search defined Swahili and English retail mapping entries (prioritizing longest/most specific keyword)
    best_match: Optional[Dict[str, Any]] = None
    best_match_len: int = 0

    for entry in SWAHILI_RETAIL_CONCEPT_MAPPINGS:
        for kw in entry["keywords"]:
            pattern = rf"\b{re.escape(kw)}\b"
            if re.search(pattern, clean_text):
                if len(kw) > best_match_len:
                    best_match_len = len(kw)
                    best_match = entry

    if best_match is not None:
        return (
            best_match["concept"],
            best_match["taxonomy_key"],
            set(best_match["compatible_keys"]),
        )

    # Fallback based on high-level category domain if specific keyword was not matched
    domain_fallbacks = {
        CATEGORY_FASHION: (
            "a women's dress, gown, or African print kitenge dress",
            "WOMENS_DRESS",
            {"WOMENS_DRESS", "MENS_SUIT", "TSHIRT_CASUAL", "BUTTON_SHIRT", "JEANS_TROUSERS", "SNEAKERS", "FORMAL_SHOES", "WOMENS_HEELS_SANDALS", "HANDBAG_PURSE", "BACKPACK_BAG", "WRISTWATCH", "JEWELRY"},
        ),
        CATEGORY_ELECTRONICS: (
            "a smartphone, mobile phone, or cellular telephone",
            "SMARTPHONE",
            {"SMARTPHONE", "LAPTOP", "PHONE_CHARGER", "HEADPHONES", "TELEVISION", "TABLET", "CAMERA"},
        ),
        CATEGORY_GROCERIES: (
            "packaged maize flour, wheat flour, or grain flour bag",
            "FLOUR_BAG",
            {"FLOUR_BAG", "RICE_GRAIN", "COOKING_OIL", "FRESH_PRODUCE"},
        ),
        CATEGORY_HOME_GOODS: (
            "kitchen knives, cooking pots, pans, or kitchenware utensils",
            "KITCHEN_UTENSILS",
            {"KITCHEN_UTENSILS", "FURNITURE_SOFA", "BED_MATTRESS", "HOME_APPLIANCE"},
        ),
        CATEGORY_BEAUTY: (
            "skin lotion cream, makeup lipstick, or facial skincare cosmetics",
            "COSMETICS_LOTION",
            {"PERFUME", "COSMETICS_LOTION", "HAIR_WIG"},
        ),
    }

    if category and category.lower() in domain_fallbacks:
        return domain_fallbacks[category.lower()]

    # General fallback
    return (
        f"a photo of {title[:40].strip()}",
        "WOMENS_DRESS",
        {"WOMENS_DRESS", "MENS_SUIT", "TSHIRT_CASUAL", "BUTTON_SHIRT", "JEANS_TROUSERS", "SNEAKERS", "FORMAL_SHOES", "WOMENS_HEELS_SANDALS", "HANDBAG_PURSE", "BACKPACK_BAG", "WRISTWATCH", "JEWELRY", "SMARTPHONE", "LAPTOP", "PHONE_CHARGER", "HEADPHONES", "TELEVISION", "TABLET", "CAMERA", "FLOUR_BAG", "RICE_GRAIN", "COOKING_OIL", "FRESH_PRODUCE", "KITCHEN_UTENSILS", "FURNITURE_SOFA", "BED_MATTRESS", "HOME_APPLIANCE", "PERFUME", "COSMETICS_LOTION", "HAIR_WIG"},
    )


def _load_global_clip() -> bool:
    """Thread-safe loader for the HuggingFace CLIP model and processor.

    Returns:
        bool: True if CLIP is successfully loaded into global memory, False otherwise.
    """
    global _global_clip_processor, _global_clip_model, _global_clip_load_failed

    if _global_clip_model is not None and _global_clip_processor is not None:
        return True

    if _global_clip_load_failed:
        return False

    with _clip_model_lock:
        if _global_clip_model is not None and _global_clip_processor is not None:
            return True
        try:
            logger.info("Initializing HuggingFace CLIP model: %s", CLIP_MODEL_NAME)
            from transformers import CLIPModel, CLIPProcessor

            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
            model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            model.eval()

            _global_clip_processor = processor
            _global_clip_model = model
            logger.info("CLIP model successfully loaded and warm in memory.")
            return True
        except Exception as exc:
            _global_clip_load_failed = True
            logger.error("Failed to load HuggingFace CLIP model (%s): %s", CLIP_MODEL_NAME, exc)
            return False


class ContentModerationLayer1:
    """Production-grade Layer 1 content moderator combining vision and language verification."""

    def __init__(self, use_real_clip: bool = USE_REAL_CLIP) -> None:
        """Initialize the Layer 1 moderator and lazily prepare vision-language components.

        Parameters:
            use_real_clip (bool): Whether to load actual HuggingFace CLIP weights or heuristic fallback.
        """
        self.use_real_clip = use_real_clip
        if self.use_real_clip:
            _load_global_clip()

    @classmethod
    def warmup_clip(cls) -> bool:
        """Pre-warm CLIP model into RAM during FastAPI startup lifespan.

        Respects the USE_REAL_CLIP setting so the service can start without
        downloading vision weights (image checks then report a clear warning).

        Returns:
            bool: True if loaded and warm, False otherwise.
        """
        if not USE_REAL_CLIP:
            logger.warning("USE_REAL_CLIP is disabled: skipping vision model warm-up; image checks will be unavailable.")
            return False
        return _load_global_clip()

    def _ensure_clip_loaded(self) -> bool:
        """Ensure global CLIP model is loaded into memory.

        Returns:
            bool: True if CLIP is loaded and ready, False if unavailable.
        """
        if not self.use_real_clip:
            return False
        return _load_global_clip()

    def decode_image(self, image_data: Any) -> Optional[Image.Image]:
        """Decode an input image from base64 string, raw bytes, or PIL Image.

        Parameters:
            image_data (Any): Base64 data URI/string, bytes, or PIL.Image instance.

        Returns:
            Optional[Image.Image]: Decoded PIL RGB image, or None if decoding fails.
        """
        if image_data is None:
            return None

        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")

        try:
            if isinstance(image_data, str):
                if "," in image_data:
                    image_data = image_data.split(",", 1)[1]
                raw_bytes = base64.b64decode(image_data.strip())
            elif isinstance(image_data, bytes):
                raw_bytes = image_data
            else:
                return None

            img = Image.open(io.BytesIO(raw_bytes))
            return img.convert("RGB")
        except Exception as exc:
            logger.error("Failed to decode image: %s", exc)
            return None

    def analyze_swahili_slang_and_prohibited_terms(
        self, text: str
    ) -> Tuple[float, List[str], Dict[str, int]]:
        """Context-aware analysis of title and description for prohibited goods and slang.

        Performs semantic disambiguation to prevent false positives on legitimate Tanzanian
        household and grocery trade ("unga wa sembe", "kisu cha jikoni", "majani ya chai")
        while enforcing zero-tolerance blocks on confirmed contraband.

        Parameters:
            text (str): Combined product title and description.

        Returns:
            Tuple[float, List[str], Dict[str, int]]:
                - Normalized risk score (0.0 to 1.0).
                - List of matched illicit terms.
                - Breakdown of detected categories with frequencies.
        """
        if not text or not text.strip():
            return 0.0, [], {}

        normalized_text = text.lower()
        matched_terms: List[str] = []
        category_counts: Dict[str, int] = {}
        max_severity_weight = 0.0

        # -------------------------------------------------------------------
        # Step 1: Scan Unambiguous Contraband Lexicon (Strict Zero-Tolerance)
        # -------------------------------------------------------------------
        for keyword, meta in UNAMBIGUOUS_PROHIBITED_LEXICON.items():
            pattern = rf"\b{re.escape(keyword)}\b"
            if re.search(pattern, normalized_text):
                matched_terms.append(keyword)
                category = meta["category"]
                category_counts[category] = category_counts.get(category, 0) + 1
                weight = float(meta["weight"])
                if weight > max_severity_weight:
                    max_severity_weight = weight

        # -------------------------------------------------------------------
        # Step 2: Context-Aware Disambiguation for Polysemous Swahili Words
        # -------------------------------------------------------------------

        # A. Disambiguate "Unga" (Flour vs. Narcotics)
        if re.search(r"\bunga\b", normalized_text):
            has_grocery_context = bool(UNGA_GROCERY_REGEX.search(normalized_text))
            has_drug_context = bool(UNGA_DRUG_REGEX.search(normalized_text))

            if has_drug_context:
                matched_terms.append("unga (narcotics context)")
                category_counts["NARCOTICS"] = category_counts.get("NARCOTICS", 0) + 1
                max_severity_weight = max(max_severity_weight, 0.95)
            elif has_grocery_context:
                # Confirmed food/flour trade: completely innocent
                logger.debug("Word 'unga' disambiguated as legitimate food/flour trade.")
            else:
                # Standalone unga without grocery or drug context: low ambiguity warning
                max_severity_weight = max(max_severity_weight, 0.25)

        # B. Disambiguate "Kisu" (Kitchen knife vs. Combat weapon)
        if re.search(r"\bkisu\b", normalized_text):
            has_kitchen_context = bool(KISU_KITCHEN_REGEX.search(normalized_text))
            has_weapon_context = bool(KISU_WEAPON_REGEX.search(normalized_text))

            if has_weapon_context:
                matched_terms.append("kisu cha kivita")
                category_counts["WEAPONS"] = category_counts.get("WEAPONS", 0) + 1
                max_severity_weight = max(max_severity_weight, 0.90)
            elif has_kitchen_context:
                logger.debug("Word 'kisu' disambiguated as kitchen utensil.")
            else:
                # Standalone knife without context: modest risk
                max_severity_weight = max(max_severity_weight, 0.20)

        # C. Disambiguate "Majani" (Tea leaves vs. Cannabis)
        if re.search(r"\bmajani\b", normalized_text):
            has_tea_context = bool(MAJANI_TEA_REGEX.search(normalized_text))
            has_drug_context = bool(MAJANI_DRUG_REGEX.search(normalized_text))

            if has_drug_context:
                matched_terms.append("majani ya bangi")
                category_counts["NARCOTICS"] = category_counts.get("NARCOTICS", 0) + 1
                max_severity_weight = max(max_severity_weight, 0.90)
            elif has_tea_context:
                logger.debug("Word 'majani' disambiguated as agricultural tea leaves.")
            else:
                max_severity_weight = max(max_severity_weight, 0.15)

        # D. Disambiguate "Kitunguu" (Onion vs. Drug baggie)
        if re.search(r"\bkitunguu\b", normalized_text):
            has_grocery_context = bool(KITUNGUU_GROCERY_REGEX.search(normalized_text))
            if has_grocery_context:
                logger.debug("Word 'kitunguu' disambiguated as legitimate produce.")
            else:
                # Coded drug baggie slang check
                if "kete" in normalized_text or "msokoto" in normalized_text:
                    matched_terms.append("kitunguu (coded slang)")
                    category_counts["NARCOTICS"] = category_counts.get("NARCOTICS", 0) + 1
                    max_severity_weight = max(max_severity_weight, 0.85)

        if not matched_terms and max_severity_weight == 0.0:
            return 0.0, [], {}

        # Scale final score
        raw_score = max_severity_weight + min(0.2, (len(matched_terms) - 1) * 0.05) if matched_terms else max_severity_weight
        normalized_score = min(1.0, max(0.0, raw_score))

        return round(normalized_score, 4), matched_terms, category_counts

    def scan_and_align_image(
        self,
        image: Optional[Image.Image],
        title: str,
        description: str = "",
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Perform fine-grained 36-category zero-shot vision-language verification using CLIP.

        Translates Swahili titles to English visual concepts, classifies the image against
        36 mutually exclusive retail and safety categories, checks category compatibility,
        and computes true Claimed-vs-Detected mismatch scores.

        Parameters:
            image (Optional[Image.Image]): Decoded RGB listing photo.
            title (str): Product listing title in Swahili or English.
            description (str): Detailed product description.
            category (Optional[str]): Inferred or explicit category.

        Returns:
            Dict[str, Any]: Comprehensive vision verification telemetry.
        """
        # 1. Translate Swahili listing title into precise visual concept and taxonomy key
        claimed_concept, target_taxonomy_key, compatible_keys = translate_swahili_concept_to_english(
            title=title, description=description, category=category
        )

        if image is None:
            return {
                "seller_claimed_concept": claimed_concept,
                "top_visual_key": "NO_IMAGE",
                "top_visual_label": "No Image Uploaded",
                "top_visual_probability": 0.0,
                "claimed_title_probability": 0.0,
                "visual_contraband_detected": False,
                "visual_contraband_probability": 0.0,
                "is_non_product": False,
                "visual_comparison_status": "NO_IMAGE",
                "comparison_detail": "No product image was provided for visual inspection.",
                "image_text_mismatch_score": 0.0,
                "is_severe_mismatch": False,
                "nsfw_detected": False,
                "nsfw_confidence": 0.0,
                "nsfw_category": "SAFE",
                "warning": "No product image provided for visual verification.",
            }

        if not self._ensure_clip_loaded():
            logger.warning("CLIP model unavailable for vision-language analysis.")
            return {
                "seller_claimed_concept": claimed_concept,
                "top_visual_key": "CLIP_UNAVAILABLE",
                "top_visual_label": "Neural Network Offline",
                "top_visual_probability": 0.0,
                "claimed_title_probability": 0.0,
                "visual_contraband_detected": False,
                "visual_contraband_probability": 0.0,
                "is_non_product": False,
                "visual_comparison_status": "CLIP_UNAVAILABLE",
                "comparison_detail": "Vision neural network is currently offline.",
                "image_text_mismatch_score": 0.0,
                "is_severe_mismatch": False,
                "nsfw_detected": False,
                "nsfw_confidence": 0.0,
                "nsfw_category": "SAFE",
                "warning": "Vision neural network is unavailable.",
            }

        try:
            import torch

            # 2. Prepare 36 candidate prompts from FINE_GRAINED_VISUAL_TAXONOMY
            taxonomy_keys = list(FINE_GRAINED_VISUAL_TAXONOMY.keys())
            candidate_labels = [FINE_GRAINED_VISUAL_TAXONOMY[k]["label"] for k in taxonomy_keys]

            inputs = _global_clip_processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True,
            )

            with torch.no_grad():
                outputs = _global_clip_model(**inputs)
                logits = outputs.logits_per_image
                probs = logits.softmax(dim=1).squeeze().tolist()

            if not isinstance(probs, list):
                probs = [float(probs)]

            # 3. Identify Top Detected Visual Concept
            top_idx = int(torch.tensor(probs).argmax().item())
            top_key = taxonomy_keys[top_idx]
            top_label = candidate_labels[top_idx]
            top_prob = round(float(probs[top_idx]), 4)

            # 4. Check Contraband & Safety Probabilities
            cannabis_idx = taxonomy_keys.index("CANNABIS")
            narcotics_idx = taxonomy_keys.index("NARCOTICS")
            weapon_idx = taxonomy_keys.index("WEAPON")
            nsfw_idx = taxonomy_keys.index("EXPLICIT_NSFW")

            cannabis_prob = float(probs[cannabis_idx])
            narcotics_prob = float(probs[narcotics_idx])
            weapons_prob = float(probs[weapon_idx])
            nudity_prob = float(probs[nsfw_idx])
            illicit_prob = round(cannabis_prob + narcotics_prob + weapons_prob + nudity_prob, 4)

            contraband_keys = {"CANNABIS", "NARCOTICS", "WEAPON", "EXPLICIT_NSFW"}
            is_contraband = (
                top_key in contraband_keys
                or illicit_prob >= VISUAL_CONTRABAND_THRESHOLD
                or any(p >= 0.35 for p in (cannabis_prob, narcotics_prob, weapons_prob, nudity_prob))
            )

            # 5. Check Non-Product Fraud (Screenshots / Documents)
            non_product_keys = {"SCREENSHOT", "DOCUMENT_RECEIPT"}
            screenshot_idx = taxonomy_keys.index("SCREENSHOT")
            receipt_idx = taxonomy_keys.index("DOCUMENT_RECEIPT")
            screenshot_prob = float(probs[screenshot_idx])
            receipt_prob = float(probs[receipt_idx])

            is_non_product = (
                (top_key in non_product_keys and top_prob >= 0.25)
                or (screenshot_prob + receipt_prob >= 0.40)
            )

            # 6. Extract Claimed Category Probability
            claimed_idx = taxonomy_keys.index(target_taxonomy_key) if target_taxonomy_key in taxonomy_keys else None
            claimed_prob = round(float(probs[claimed_idx]), 4) if claimed_idx is not None else 0.0

            # 7. Evaluate True Claimed-vs-Detected Alignment & Mismatch Scoring
            if is_contraband:
                visual_comparison_status = "CONTRABAND"
                is_severe_mismatch = True
                mismatch_score = 0.95
                comparison_detail = f"Prohibited visual content detected: Image shows {top_label} ({top_prob * 100.0:.1f}%)."
            elif is_non_product:
                visual_comparison_status = "MISMATCH"
                is_severe_mismatch = True
                mismatch_score = 0.90
                comparison_detail = (
                    f"Non-product image detected: Image is {top_label} ({top_prob * 100.0:.1f}%) "
                    f"instead of physical retail merchandise."
                )
            elif top_key in compatible_keys:
                visual_comparison_status = "MATCH"
                is_severe_mismatch = False
                # Low mismatch score when visual aligns with claimed product
                mismatch_score = round(max(0.02, min(0.20, 0.22 - (top_prob * 0.15))), 4)
                comparison_detail = f"Visual match verified: Image shows {top_label} ({top_prob * 100.0:.1f}%), aligning with claimed '{title}'."
            else:
                # Contrastive Mismatch: Top detected concept does NOT match seller's claimed item
                visual_comparison_status = "MISMATCH"
                is_severe_mismatch = True
                # Penalize mismatch based on discrepancy (0.75 to 0.95)
                mismatch_score = round(max(0.75, min(0.95, 1.0 - claimed_prob)), 4)
                comparison_detail = (
                    f"Visual-Text Mismatch: Image shows {top_label} ({top_prob * 100.0:.1f}%) "
                    f"instead of claimed {claimed_concept}."
                )

            # 8. NSFW / Violence Flagging
            is_nsfw = False
            nsfw_conf = 0.0
            nsfw_cat = "SAFE"
            if nudity_prob >= CRITICAL_NSFW_THRESHOLD or (nudity_prob >= 0.35 and top_key == "EXPLICIT_NSFW"):
                is_nsfw = True
                nsfw_conf = round(nudity_prob, 4)
                nsfw_cat = "ADULT"
            elif weapons_prob >= CRITICAL_NSFW_THRESHOLD or (weapons_prob >= 0.35 and top_key == "WEAPON"):
                is_nsfw = True
                nsfw_conf = round(weapons_prob, 4)
                nsfw_cat = "VIOLENCE"
            elif cannabis_prob >= 0.35 or narcotics_prob >= 0.35:
                nsfw_conf = round(max(cannabis_prob, narcotics_prob), 4)
                nsfw_cat = "NARCOTICS"

            logger.info(
                "Fine-Grained CLIP Verification: claimed='%s' | detected='%s' (%.1f%%) | status=%s | mismatch=%.4f",
                claimed_concept,
                top_label,
                top_prob * 100.0,
                visual_comparison_status,
                mismatch_score,
            )

            return {
                "seller_claimed_concept": claimed_concept,
                "top_visual_key": top_key,
                "top_visual_label": top_label,
                "top_visual_probability": top_prob,
                "claimed_title_probability": claimed_prob,
                "visual_contraband_detected": is_contraband,
                "visual_contraband_probability": illicit_prob,
                "is_non_product": is_non_product,
                "visual_comparison_status": visual_comparison_status,
                "comparison_detail": comparison_detail,
                "image_text_mismatch_score": mismatch_score,
                "is_severe_mismatch": is_severe_mismatch,
                "nsfw_detected": is_nsfw,
                "nsfw_confidence": nsfw_conf,
                "nsfw_category": nsfw_cat,
                "warning": None,
            }

        except Exception as exc:
            logger.error("Fine-grained CLIP inference error: %s", exc)
            return {
                "seller_claimed_concept": claimed_concept,
                "top_visual_key": "INFERENCE_ERROR",
                "top_visual_label": "Inference Error",
                "top_visual_probability": 0.0,
                "claimed_title_probability": 0.0,
                "visual_contraband_detected": False,
                "visual_contraband_probability": 0.0,
                "is_non_product": False,
                "visual_comparison_status": "ERROR",
                "comparison_detail": f"Vision inference error: {exc}",
                "image_text_mismatch_score": 0.50,
                "is_severe_mismatch": False,
                "nsfw_detected": False,
                "nsfw_confidence": 0.0,
                "nsfw_category": "SAFE",
                "warning": f"Visual inference error: {exc}",
            }

    def scan_visual_contraband_and_nsfw(
        self, image: Optional[Image.Image]
    ) -> Tuple[bool, float, str, float, bool, float, str, Optional[str]]:
        """Check A: Direct Contraband and NSFW Scan (backward-compatible wrapper).

        Parameters:
            image (Optional[Image.Image]): Decoded RGB listing photo.

        Returns:
            Tuple[bool, float, str, float, bool, float, str, Optional[str]]:
                - visual_contraband_detected
                - combined_illicit_probability
                - top_visual_label
                - top_visual_probability
                - nsfw_detected
                - nsfw_confidence
                - nsfw_category
                - warning
        """
        res = self.scan_and_align_image(image=image, title="generic retail product")
        return (
            res["visual_contraband_detected"],
            res["visual_contraband_probability"],
            res["top_visual_label"],
            res["top_visual_probability"],
            res["nsfw_detected"],
            res["nsfw_confidence"],
            res["nsfw_category"],
            res["warning"],
        )

    def align_image_to_title(
        self,
        image: Optional[Image.Image],
        title: str,
        category: Optional[str] = None,
    ) -> Tuple[float, float, bool, Optional[str]]:
        """Check B: Title-to-Image Alignment (backward-compatible wrapper).

        Parameters:
            image (Optional[Image.Image]): Decoded RGB listing photo.
            title (str): Product listing title.
            category (Optional[str]): Resolved category.

        Returns:
            Tuple[float, float, bool, Optional[str]]:
                - mismatch_score
                - claimed_title_probability
                - is_severe_mismatch
                - warning
        """
        res = self.scan_and_align_image(image=image, title=title, category=category)
        return (
            res["image_text_mismatch_score"],
            res["claimed_title_probability"],
            res["is_severe_mismatch"],
            res["warning"],
        )

    def compute_image_text_mismatch(
        self, image: Optional[Image.Image], text: str
    ) -> float:
        """Backward-compatible helper to compute mismatch score.

        Parameters:
            image (Optional[Image.Image]): Listing photo.
            text (str): Product title.

        Returns:
            float: Mismatch risk score (0.0 = high correlation, 1.0 = total mismatch).
        """
        res = self.scan_and_align_image(image=image, title=text)
        return res["image_text_mismatch_score"]

    def detect_nsfw_and_abusive(
        self, image: Optional[Image.Image], text: str
    ) -> Tuple[bool, float, str]:
        """Backward-compatible helper to detect NSFW content.

        Parameters:
            image (Optional[Image.Image]): Listing image.
            text (str): Listing text.

        Returns:
            Tuple[bool, float, str]: (is_nsfw, confidence, reason)
        """
        res = self.scan_and_align_image(image=image, title=text)
        return res["nsfw_detected"], res["nsfw_confidence"], res["nsfw_category"]

    def evaluate_listing_content(
        self,
        title: str,
        description: str,
        image_data: Optional[Any] = None,
        explicit_category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Orchestrate full Layer 1 moderation with fine-grained 36-category vision-language verification.

        Parameters:
            title (str): Product listing title.
            description (str): Detailed product description.
            image_data (Optional[Any]): Image input (base64, bytes, or PIL Image).
            explicit_category (Optional[str]): Explicit category if specified.

        Returns:
            Dict[str, Any]: Detailed Layer 1 moderation report with fine-grained CLIP scores.
        """
        full_text = f"{title} {description}".strip()
        decoded_img = self.decode_image(image_data)
        category = infer_listing_category(title, description, explicit_category)

        # 1. Context-aware Swahili & English slang scan
        slang_score, matched_terms, categories = self.analyze_swahili_slang_and_prohibited_terms(full_text)

        # 2. Fine-grained 36-category vision classification and contrastive alignment
        vision_res = self.scan_and_align_image(
            image=decoded_img,
            title=title,
            description=description,
            category=category,
        )

        mismatch_score = float(vision_res["image_text_mismatch_score"])
        visual_contraband_detected = bool(vision_res["visual_contraband_detected"])
        is_nsfw = bool(vision_res["nsfw_detected"])

        # 3. Critical Flag: Triggered immediately if visual contraband, critical slang, or NSFW is detected
        layer1_critical_flag = (
            slang_score >= CRITICAL_SLANG_THRESHOLD
            or visual_contraband_detected
            or is_nsfw
            or (mismatch_score >= CRITICAL_MISMATCH_THRESHOLD and slang_score >= 0.40)
        )

        return {
            "category": category,
            "seller_claimed_concept": vision_res["seller_claimed_concept"],
            "top_visual_key": vision_res["top_visual_key"],
            "top_visual_label": vision_res["top_visual_label"],
            "top_visual_probability": vision_res["top_visual_probability"],
            "claimed_title_probability": vision_res["claimed_title_probability"],
            "visual_comparison_status": vision_res["visual_comparison_status"],
            "comparison_detail": vision_res["comparison_detail"],
            "image_text_mismatch_score": mismatch_score,
            "is_severe_mismatch": vision_res["is_severe_mismatch"],
            "is_non_product": vision_res["is_non_product"],
            "visual_contraband_detected": visual_contraband_detected,
            "visual_contraband_probability": vision_res["visual_contraband_probability"],
            "nsfw_detected": is_nsfw,
            "nsfw_confidence": vision_res["nsfw_confidence"],
            "nsfw_category": vision_res["nsfw_category"],
            "swahili_slang_risk_score": slang_score,
            "matched_slang_terms": matched_terms,
            "slang_categories": categories,
            "layer1_critical_flag": layer1_critical_flag,
            "layer1_should_block": layer1_critical_flag,
            "image_warning": vision_res["warning"],
        }
