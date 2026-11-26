"""Layer 1: Illegal Goods & Content Moderation Engine.

Pre-listing inspection for Tanzanian social commerce. For one listing (title,
description, any number of images/videos) it runs:

1. Lexicon: editable Swahili/English slang in ``config/lexicon.json`` with
   de-obfuscation (b4ng1, b.a.n.g.i, baaangi), typo matching, emoji context and allow
   lists that protect legitimate trade ("unga wa sembe", "glue gun", "Kobe Bryant").
2. Off-platform contact detection (TZ phone numbers, WhatsApp/Telegram, links).
3. Text ML: multilingual zero-shot classifier (optional NLLB translation).
4. Images: NSFW (Falconsai + NudeNet), CLIP contraband + photo-vs-title mismatch,
   OCR text checked with the same lexicon, perceptual-hash blocklist, CSAM hash hook.
5. Videos: sampled frames through the image checks, optional audio transcription.
6. Fusion into ALLOW / REVIEW / BLOCK / ESCALATE with reasons and a Swahili message.

Every model is lazy and fail-soft: a missing model switches that check off and is
reported in ``checks``; the rest keep working.
"""

import logging
import mimetypes
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from config.settings import (
    CATEGORY_BEAUTY,
    CATEGORY_ELECTRONICS,
    CATEGORY_FASHION,
    CATEGORY_GROCERIES,
    CATEGORY_HOME_GOODS,
    CONFIG,
    DECISION_BLOCK,
    DECISION_ESCALATE,
    DECISION_REVIEW,
    DEFAULT_CATEGORY,
    SUPPORTED_CATEGORIES,
)
from services import child_safety
from services.contact_detector import detect_contacts
from services.fusion import (
    CAT_CONTACT,
    CAT_FILE,
    CAT_MISMATCH,
    CAT_NON_PRODUCT,
    CAT_UNCHECKED,
    CAT_YOUTH,
    CONTENT_CATEGORIES,
    FEATURE_NAMES,
    Decision,
    Signal,
    decide,
)
from services.image_analyzer import ImageAnalysis, ImageDecodeError, analyse_image, decode_image, estimate_youth
from services.lexicon import LEVEL_WEAK, LexiconResult, get_lexicon
from services.model_registry import model_status
from services.text_classifier import NORMAL_LABEL, translate_to_english, zero_shot_scores
from services.video_analyzer import VideoAnalysis, analyse_video_bytes

logger = logging.getLogger(__name__)

TEXT_CFG: Dict[str, Any] = CONFIG["text"]
IMAGE_CFG: Dict[str, Any] = CONFIG["image"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp", ".m4v", ".wmv"}


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
) -> Tuple[str, str, Set[str], bool]:
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
            - specific: True when a product keyword matched (mismatch checks are reliable).
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
            True,
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
        concept, key, compatible = domain_fallbacks[category.lower()]
        return concept, key, compatible, False

    # General fallback: unknown product, so the photo-vs-title check is informational only
    return (
        title[:60].strip(),
        "UNKNOWN",
        set(),
        False,
    )


@dataclass
class MediaFile:
    """One uploaded file."""

    filename: str
    data: bytes
    content_type: Optional[str] = None

    @property
    def kind(self) -> str:
        """``image``, ``video`` or ``unknown`` from content type, extension or magic bytes."""
        ctype = (self.content_type or mimetypes.guess_type(self.filename)[0] or "").lower()
        ext = ("." + self.filename.rsplit(".", 1)[-1].lower()) if "." in self.filename else ""
        if ctype.startswith("image/") or ext in IMAGE_EXTENSIONS:
            return "image"
        if ctype.startswith("video/") or ext in VIDEO_EXTENSIONS:
            return "video"
        head = self.data[:16]
        if head.startswith((b"\xff\xd8\xff", b"\x89PNG", b"GIF8", b"BM")) or head[8:12] == b"WEBP":
            return "image"
        if head[4:8] in (b"ftyp", b"moov") or head.startswith(b"\x1a\x45\xdf\xa3"):
            return "video"
        return "unknown"


def _lexicon_signals(result: LexiconResult, source: str, file: Optional[str] = None) -> List[Signal]:
    """Convert lexicon categories into signals."""
    signals: List[Signal] = []
    for cat in result.categories.values():
        if cat.score <= 0:
            continue
        methods = {h.method for h in cat.hits if h.tier in ("strong", "review", "weak", "fuzzy")}
        how = " (disguised spelling)" if methods & {"deobfuscated", "fuzzy"} else ""
        where = {"text": "listing text", "ocr": "text inside image", "audio": "video audio"}.get(source, source)
        evidence = ", ".join(sorted(set(cat.terms))[:6])
        context = [h.term for h in cat.hits if h.tier in ("context", "emoji")]
        detail = f"{where} mentions {get_lexicon().label(cat.category)}: {evidence}{how}"
        if context and cat.level != LEVEL_WEAK:
            detail += f" [context: {', '.join(sorted(set(context))[:4])}]"
        signals.append(Signal(source, cat.category, cat.level, cat.score, detail, file))
    return signals


def _contact_signal(text: str, source: str, file: Optional[str] = None) -> Optional[Signal]:
    """Off-platform contact signal for ``text`` (None when clean)."""
    contacts = detect_contacts(text)
    if not contacts.found:
        return None
    where = {"text": "listing", "ocr": "image text", "audio": "video audio"}.get(source, source)
    return Signal(source, CAT_CONTACT, DECISION_REVIEW, contacts.score,
                  f"{where} pushes buyers off-platform: {contacts.summary()}", file)


def _image_signals(img: ImageAnalysis, claimed: str, source: str, file: str) -> List[Signal]:
    """Signals from one analysed image / video frame."""
    signals: List[Signal] = []
    if img.hash_match:
        signals.append(Signal("hash", str(img.hash_match.get("reason", "BANNED_IMAGE")), DECISION_BLOCK, 1.0,
                              f"image matches a previously banned image (distance {img.hash_match['distance']})",
                              file))

    nsfw = img.nsfw_probability or 0.0
    nude = img.nudenet_score or 0.0
    if (nsfw >= IMAGE_CFG["nsfw_block_threshold"] and nude >= IMAGE_CFG["nudenet_review_threshold"]) \
            or nude >= IMAGE_CFG["nudenet_block_threshold"] or nsfw >= 0.97:
        level = DECISION_BLOCK
    elif nsfw >= IMAGE_CFG["nsfw_review_threshold"] or nude >= IMAGE_CFG["nudenet_review_threshold"]:
        level = DECISION_REVIEW
    else:
        level = ""
    if level:
        labels = f", detected: {', '.join(img.nudenet_labels)}" if img.nudenet_labels else ""
        signals.append(Signal(source, "SEXUAL", level, max(nsfw, nude),
                              f"nudity/pornographic content (classifier {nsfw:.2f}, detector {nude:.2f}{labels})",
                              file))

    cat, prob = img.max_contraband
    if cat and prob >= IMAGE_CFG["clip_contraband_review_threshold"]:
        level = DECISION_BLOCK if prob >= IMAGE_CFG["clip_contraband_block_threshold"] else DECISION_REVIEW
        label = get_lexicon().label(cat)
        where = f" at {img.source.split('@', 1)[1]}" if "@" in img.source else ""
        detail = f"photo{where} appears to show {label} ({prob:.0%})"
        if claimed:
            detail += f" although the listing claims '{claimed}'"
        signals.append(Signal(source, cat, level, round(prob, 4), detail, file))
    elif img.mismatch_reliable and (img.mismatch_score or 0) >= IMAGE_CFG["clip_mismatch_threshold"]             and (img.claimed_probability or 0) <= IMAGE_CFG.get("clip_mismatch_max_claimed", 0.10):
        signals.append(Signal(source, CAT_MISMATCH, DECISION_REVIEW, img.mismatch_score or 0.0,
                              f"photo does not match the title: {img.mismatch_detail}", file))
    if img.non_product_probability >= 0.6 and not (img.mismatch_reliable and (img.mismatch_score or 0) < 0.3):
        signals.append(Signal(source, CAT_NON_PRODUCT, DECISION_REVIEW, img.non_product_probability,
                              "image is a screenshot/document rather than a product photo", file))

    if img.ocr_text:
        signals.extend(_lexicon_signals(get_lexicon().analyse(img.ocr_text), "ocr", file))
        contact = _contact_signal(img.ocr_text, "ocr", file)
        if contact:
            signals.append(contact)
    return signals


def _features(text_lex: LexiconResult, other_lex: List[LexiconResult], zero_shot: Optional[Dict[str, float]],
              contact_score: float, analyses: List[ImageAnalysis], youth_text: bool) -> Dict[str, float]:
    """Build the meta-model feature vector (names in :data:`services.fusion.FEATURE_NAMES`)."""
    feats: Dict[str, float] = {name: 0.0 for name in FEATURE_NAMES}
    for lex in [text_lex] + other_lex:
        for name, cat in lex.categories.items():
            key = f"lex_{name}"
            if key in feats:
                feats[key] = max(feats[key], cat.score)
    if zero_shot:
        for cat in CONTENT_CATEGORIES + [NORMAL_LABEL]:
            if f"zs_{cat}" in feats:
                feats[f"zs_{cat}"] = zero_shot.get(cat, 0.0)
    else:
        feats["zs_NORMAL"] = 1.0  # neutral value when the text model is off
    feats["contact"] = contact_score
    feats["obfuscation"] = 1.0 if text_lex.obfuscation_detected else 0.0
    feats["emoji_hits"] = float(sum(1 for c in text_lex.categories.values() for h in c.hits if h.tier == "emoji"))
    feats["youth_text"] = 1.0 if youth_text else 0.0
    feats["media_count"] = float(len(analyses))
    for img in analyses:
        feats["img_nsfw"] = max(feats["img_nsfw"], img.nsfw_probability or 0.0)
        feats["img_nudenet"] = max(feats["img_nudenet"], img.nudenet_score or 0.0)
        feats["img_contraband"] = max(feats["img_contraband"], img.max_contraband[1])
        if img.mismatch_reliable:
            feats["img_mismatch"] = max(feats["img_mismatch"], img.mismatch_score or 0.0)
        feats["img_non_product"] = max(feats["img_non_product"], img.non_product_probability)
        feats["hash_match"] = max(feats["hash_match"], 1.0 if img.hash_match else 0.0)
    return {k: round(v, 4) for k, v in feats.items()}


class ContentModerationLayer1:
    """Layer 1 content moderator combining text, image and video checks."""

    def moderate(
        self,
        title: str,
        description: str = "",
        files: Sequence[MediaFile] = (),
        explicit_category: Optional[str] = None,
        run_text_ml: bool = True,
    ) -> Dict[str, Any]:
        """Moderate one listing.

        Parameters:
            title (str): Listing title (Swahili, English or mixed).
            description (str): Listing description.
            files (Sequence[MediaFile]): Uploaded images and videos.
            explicit_category (Optional[str]): Seller-chosen category, inferred when omitted.
            run_text_ml (bool): Run the zero-shot text model (skipped automatically when the
                lexicon already blocks).

        Returns:
            Dict[str, Any]: Decision, reasons, Swahili seller message and per-check details.
        """
        started = time.perf_counter()
        lexicon = get_lexicon()
        text = f"{title}\n{description}".strip()
        category = infer_listing_category(title, description, explicit_category)
        claimed, _, compatible, specific = translate_swahili_concept_to_english(title, description, category)
        translation = translate_to_english(text) if not specific else None
        if translation:
            claimed = translation[:80]

        signals: List[Signal] = []
        text_lex = lexicon.analyse(text)
        signals.extend(_lexicon_signals(text_lex, "text"))
        contact = _contact_signal(text, "text")
        if contact:
            signals.append(contact)
        youth_terms = list(text_lex.youth_terms)

        # --- media -------------------------------------------------------
        analyses: List[ImageAnalysis] = []
        decoded_images: List[Tuple[ImageAnalysis, Any]] = []
        media_reports: List[Dict[str, Any]] = []
        other_lex: List[LexiconResult] = []
        csam_configured = child_safety.csam_matcher_configured()
        for media in files:
            kind = media.kind
            report: Dict[str, Any] = {"filename": media.filename, "type": kind}
            if kind in ("image", "video"):
                match = child_safety.check_media(media.data, kind)
                if match.matched:
                    signals.append(Signal("csam", "CHILD_SAFETY", DECISION_ESCALATE, 1.0,
                                          f"matched known-CSAM hash list ({match.provider})", media.filename))
                    report["checks"] = {"csam_hash": "matched"}
                    media_reports.append(report)
                    continue
                if match.provider == "error":
                    signals.append(Signal("csam", "CHILD_SAFETY", DECISION_REVIEW, 0.5,
                                          "CSAM hash service error: needs manual check", media.filename))
            if kind == "image":
                try:
                    image = decode_image(media.data)
                except ImageDecodeError as exc:
                    signals.append(Signal("file", CAT_FILE, DECISION_REVIEW, 0.4, str(exc), media.filename))
                    report["error"] = str(exc)
                    media_reports.append(report)
                    continue
                img = analyse_image(image, media.filename, claimed, compatible if specific else set())
                analyses.append(img)
                decoded_images.append((img, image))
                signals.extend(_image_signals(img, claimed, "image", media.filename))
                if img.ocr_text:
                    other_lex.append(lexicon.analyse(img.ocr_text))
                report.update(img.to_dict())
            elif kind == "video":
                video: VideoAnalysis = analyse_video_bytes(media.data, media.filename, claimed,
                                                           compatible if specific else set())
                analyses.extend(video.frames)
                for frame in video.frames:
                    signals.extend(_image_signals(frame, claimed, "video", media.filename))
                    if frame.ocr_text:
                        other_lex.append(lexicon.analyse(frame.ocr_text))
                if video.transcript:
                    audio_lex = lexicon.analyse(video.transcript)
                    other_lex.append(audio_lex)
                    signals.extend(_lexicon_signals(audio_lex, "audio", media.filename))
                    youth_terms.extend(audio_lex.youth_terms)
                    audio_contact = _contact_signal(video.transcript, "audio", media.filename)
                    if audio_contact:
                        signals.append(audio_contact)
                if not video.frames:
                    signals.append(Signal("file", CAT_FILE, DECISION_REVIEW, 0.4, "video could not be read",
                                          media.filename))
                report.update(video.to_dict())
            else:
                signals.append(Signal("file", CAT_FILE, DECISION_REVIEW, 0.4, "unsupported file type",
                                      media.filename))
            media_reports.append(report)
        if CONFIG["decision"].get("review_when_media_unchecked", True):
            unchecked = sorted({img.source.split("@", 1)[0] for img in analyses
                                if "ok" not in (img.checks.get("nsfw_classifier"), img.checks.get("nudenet"))})
            if unchecked:
                signals.append(Signal("file", CAT_UNCHECKED, DECISION_REVIEW, 0.4,
                                      "nudity models unavailable: media could not be checked automatically",
                                      ", ".join(unchecked)))
        signals = _dedupe(signals)

        # --- text ML (skipped when the rules already block) ----------------
        zero_shot: Optional[Dict[str, float]] = None
        already_blocked = any(s.level in (DECISION_BLOCK, DECISION_ESCALATE) for s in signals)
        if run_text_ml and not already_blocked:
            zero_shot = zero_shot_scores(f"{text}\n{translation}" if translation else text)
            if zero_shot:
                signals.extend(self._zero_shot_signals(zero_shot))

        # --- child safety: youth indicators only matter with a sexual signal
        sexual = any(s.category == "SEXUAL" and s.level in (DECISION_REVIEW, DECISION_BLOCK) for s in signals)
        if sexual:
            if youth_terms:
                signals.append(Signal("text", CAT_YOUTH, LEVEL_WEAK, 0.0,
                                      f"youth indicator in text: {', '.join(sorted(set(youth_terms))[:3])}"))
            sexual_files = {s.file for s in signals if s.category == "SEXUAL" and s.source in ("image", "video")}
            text_sexual = any(s.category == "SEXUAL" and s.source not in ("image", "video") for s in signals)
            for img, image in decoded_images:
                if text_sexual or img.source in sexual_files:
                    img.youth_probability = estimate_youth(image)
                    if (img.youth_probability or 0) >= IMAGE_CFG["youth_probability_threshold"]:
                        signals.append(Signal("image", CAT_YOUTH, LEVEL_WEAK, img.youth_probability or 0.0,
                                              "age estimator indicates a possible minor", img.source))

        contact_score = max((s.score for s in signals if s.category == CAT_CONTACT), default=0.0)
        features = _features(text_lex, other_lex, zero_shot, contact_score, analyses, bool(youth_terms))
        result: Decision = decide(signals, features)
        return self._report(result, category, claimed, specific, text_lex, zero_shot, analyses, media_reports,
                            files, started, csam_configured)

    # ------------------------------------------------------------------
    @staticmethod
    def _zero_shot_signals(scores: Dict[str, float]) -> List[Signal]:
        """REVIEW signal when the text model is very confident about a prohibited category."""
        normal = scores.get(NORMAL_LABEL, 0.0)
        candidates = {k: v for k, v in scores.items() if k != NORMAL_LABEL}
        if not candidates:
            return []
        cat = max(candidates, key=candidates.get)
        prob = candidates[cat]
        if prob >= float(TEXT_CFG["zero_shot_review_threshold"]) and normal <= float(TEXT_CFG["zero_shot_max_normal"]):
            return [Signal("text_ml", cat, DECISION_REVIEW, round(prob * 0.6, 4),
                           f"text model: listing looks like {get_lexicon().label(cat)} ({prob:.0%})")]
        return []

    @staticmethod
    def _report(result: Decision, category: str, claimed: str, specific: bool, text_lex: LexiconResult,
                zero_shot: Optional[Dict[str, float]], analyses: List[ImageAnalysis],
                media_reports: List[Dict[str, Any]], files: Sequence[MediaFile], started: float,
                csam_configured: bool) -> Dict[str, Any]:
        """Assemble the response dictionary (also keeps the fields the Risk Workbench UI reads)."""
        worst_text = text_lex.worst()
        contraband = [(img.max_contraband, img) for img in analyses]
        top_contra = max(contraband, key=lambda c: c[0][1], default=((None, 0.0), None))
        reasons = [r.to_dict() for r in result.reasons]
        # Confirmed mismatch only: an unflagged photo reports at most 0.30 so Layer 2 does not double count it
        flagged = {r.get("file") for r in reasons if r["category"] == CAT_MISMATCH}
        mismatches = [
            (img.mismatch_score or 0.0) if img.source in flagged else min(img.mismatch_score or 0.0, 0.30)
            for img in analyses if img.mismatch_reliable
        ]
        if not files:
            status = "NO_IMAGE"
        elif not any(img.checks.get("clip") == "ok" for img in analyses):
            status = "UNAVAILABLE"
        elif any(r["category"] in CONTENT_CATEGORIES and r["source"] in ("image", "video")
                 and r["level"] in (DECISION_BLOCK, DECISION_REVIEW) for r in reasons):
            status = "CONTRABAND"
        elif any(r["category"] == CAT_MISMATCH for r in reasons):
            status = "MISMATCH"
        elif any(r["category"] == CAT_NON_PRODUCT for r in reasons):
            status = "NON_PRODUCT"
        else:
            status = "MATCH" if specific else "NOT_VERIFIED"
        checks = {key: info["status"] for key, info in model_status().items()}
        checks["csam_hash_matching"] = "configured" if csam_configured else "NOT_CONFIGURED"
        return {
            "decision": result.decision,
            "score": result.score,
            "categories": result.categories,
            "reasons": reasons,
            "seller_message_sw": result.seller_message_sw,
            "seller_message_en": result.seller_message_en,
            "category": category,
            "seller_claimed_concept": claimed,
            "swahili_slang_risk_score": worst_text.score if worst_text else 0.0,
            "matched_slang_terms": sorted({t for c in text_lex.categories.values() for t in c.terms}),
            "obfuscation_detected": text_lex.obfuscation_detected,
            "zero_shot": zero_shot,
            "image_text_mismatch_score": round(max(mismatches, default=0.0), 4),
            "visual_comparison_status": status,
            "top_visual_label": analyses[0].top_visual_label if analyses else None,
            "top_visual_probability": analyses[0].top_visual_probability if analyses else 0.0,
            "visual_contraband_detected": status == "CONTRABAND",
            "visual_contraband_category": top_contra[0][0],
            "visual_contraband_probability": round(top_contra[0][1], 4),
            "nsfw_detected": any(r["category"] == "SEXUAL" and r["source"] in ("image", "video") for r in reasons),
            "media": media_reports,
            "meta_model_probability": result.meta_model_probability,
            "features": result.features,
            "checks": checks,
            "processing_ms": int((time.perf_counter() - started) * 1000),
        }


def _dedupe(signals: List[Signal]) -> List[Signal]:
    """Keep the strongest signal per (source, category, file); video frames collapse to one."""
    best: Dict[Tuple[str, str, Optional[str]], Signal] = {}
    for sig in signals:
        key = (sig.source, sig.category, sig.file)
        current = best.get(key)
        if current is None or (_rank(sig), sig.score) > (_rank(current), current.score):
            best[key] = sig
    return list(best.values())


def _rank(sig: Signal) -> int:
    """Severity rank of a signal level."""
    return {LEVEL_WEAK: 0, DECISION_REVIEW: 1, DECISION_BLOCK: 2, DECISION_ESCALATE: 3}.get(sig.level, 0)
