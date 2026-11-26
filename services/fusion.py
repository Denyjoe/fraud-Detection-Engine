"""Score fusion: signals -> ALLOW / REVIEW / BLOCK / ESCALATE, reasons and seller message.

Decision order (rules are authoritative; the meta-model can only make things stricter):

1. ``ESCALATE``: CSAM hash match, or any sexual signal (>= REVIEW) together with any
   youth indicator (see :mod:`services.child_safety`).
2. ``BLOCK``: any BLOCK-level signal (strong lexicon term, explicit nudity, clear
   contraband photo, banned-image hash ...).
3. Calibrated meta-model (when ``models/meta_model.joblib`` exists): probability >=
   block threshold -> BLOCK, but only when at least one non-ML signal is >= REVIEW
   (ML alone never blocks); probability >= review threshold -> REVIEW. Trained thresholds
   are never used below ``decision.meta_min_block_threshold`` / ``meta_min_review_threshold``.
4. ``REVIEW``: any REVIEW-level signal (off-platform contact, photo mismatch, ambiguous
   term with context, confident text-ML ...) or noisy-OR score >= ``decision.review_score``.
5. ``ALLOW``.
"""

import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import (
    CONFIG,
    DECISION_ALLOW,
    DECISION_BLOCK,
    DECISION_ESCALATE,
    DECISION_REVIEW,
    DECISION_SEVERITY,
    META_MODEL_FILE,
)
from services.lexicon import LEVEL_WEAK, get_lexicon

logger = logging.getLogger(__name__)

# Lexicon / zero-shot categories (order matters: it defines the meta-model features)
CONTENT_CATEGORIES: List[str] = [
    "DRUGS", "WEAPONS", "WILDLIFE", "FINANCIAL_FRAUD", "SCAM", "STOLEN_GOODS", "SEXUAL", "BANNED_TZ",
]
CAT_CONTACT = "OFF_PLATFORM_CONTACT"
CAT_MISMATCH = "IMAGE_TEXT_MISMATCH"
CAT_NON_PRODUCT = "NON_PRODUCT_IMAGE"
CAT_YOUTH = "YOUTH_INDICATOR"
CAT_CHILD_SAFETY = "CHILD_SAFETY"
CAT_FILE = "UNREADABLE_FILE"
CAT_UNCHECKED = "MEDIA_NOT_CHECKED"

ML_SOURCES = {"text_ml"}

FEATURE_NAMES: List[str] = (
    [f"lex_{c}" for c in CONTENT_CATEGORIES]
    + [f"zs_{c}" for c in CONTENT_CATEGORIES if c != "SCAM"]
    + ["zs_NORMAL", "contact", "obfuscation", "emoji_hits", "youth_text", "media_count",
       "img_nsfw", "img_nudenet", "img_contraband", "img_mismatch", "img_non_product", "hash_match"]
)

_EN_MESSAGES = {
    DECISION_ALLOW: "Your listing was approved.",
    DECISION_REVIEW: "Your listing is being checked by our team before it is published.",
    DECISION_ESCALATE: "Your listing has been suspended and is being reviewed by our safety team.",
    DECISION_BLOCK: "Your listing was blocked because it appears to sell {label_en}, which is prohibited.",
    "BLOCK_IMAGE": "Your listing was blocked because a photo or video contains prohibited content.",
    "BLOCK_GENERIC": "Your listing was blocked because it breaks the store rules.",
    "CONTACT": "Please remove phone numbers, WhatsApp/Telegram or outside links; keep chat and payment in the store.",
    "MISMATCH": "The product photo does not match the title or description. Please upload a real photo of the item.",
    "NON_PRODUCT": "The photo is not a product photo (e.g. a screenshot or receipt). Please upload a real photo.",
}


@dataclass
class Signal:
    """One piece of evidence from any check."""

    source: str  # text | ocr | audio | text_ml | contact | image | video | hash | csam | file
    category: str
    level: str  # WEAK | REVIEW | BLOCK | ESCALATE
    score: float
    detail: str
    file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class Decision:
    """Final moderation decision."""

    decision: str
    score: float
    categories: List[str]
    reasons: List[Signal]
    seller_message_sw: str
    seller_message_en: str
    meta_model_probability: Optional[float] = None
    features: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Meta-model (optional, trained by training/train_meta_model.py)
# ---------------------------------------------------------------------------
_meta_lock = threading.Lock()
_meta_bundle: Optional[Dict[str, Any]] = None
_meta_loaded = False


def load_meta_model(path: str = META_MODEL_FILE, reload: bool = False) -> Optional[Dict[str, Any]]:
    """Load the calibrated meta-model bundle (fail-soft; None when absent/invalid).

    Parameters:
        path (str): joblib file written by the training script.
        reload (bool): Force re-reading the file.

    Returns:
        Optional[Dict[str, Any]]: Bundle with ``model``, ``features``, thresholds.
    """
    global _meta_bundle, _meta_loaded
    with _meta_lock:
        if _meta_loaded and not reload:
            return _meta_bundle
        _meta_loaded = True
        _meta_bundle = None
        if not Path(path).exists():
            logger.info("No meta-model at %s: using rule-based fusion only", path)
            return None
        try:
            import joblib

            bundle = joblib.load(path)
            if list(bundle.get("features", [])) != FEATURE_NAMES:
                logger.warning("Meta-model features do not match this code version; retrain it. Ignoring.")
                return None
            _meta_bundle = bundle
            logger.info("Loaded meta-model (block>=%.2f, review>=%.2f)", bundle["block_threshold"],
                        bundle["review_threshold"])
        except Exception as exc:
            logger.warning("Could not load meta-model %s: %s", path, exc)
        return _meta_bundle


def meta_model_status() -> Dict[str, Any]:
    """Status for /health."""
    bundle = load_meta_model()
    if bundle is None:
        return {"status": "not_trained", "path": META_MODEL_FILE}
    cfg = CONFIG["decision"]
    return {"status": "ready", "trained_at": bundle.get("trained_at"), "training_rows": bundle.get("training_rows"),
            "oof_auc": bundle.get("oof_auc"),
            "block_threshold": max(bundle["block_threshold"], float(cfg.get("meta_min_block_threshold", 0.85))),
            "review_threshold": max(bundle["review_threshold"], float(cfg.get("meta_min_review_threshold", 0.50)))}


def meta_probability(features: Dict[str, float]) -> Optional[float]:
    """Violation probability from the meta-model, or None when not trained."""
    bundle = load_meta_model()
    if bundle is None:
        return None
    row = [[float(features.get(name, 0.0)) for name in FEATURE_NAMES]]
    return round(float(bundle["model"].predict_proba(row)[0][1]), 4)


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------
def _max_level(signals: List[Signal]) -> str:
    """Most severe decision level among signals (WEAK counts as ALLOW)."""
    best = DECISION_ALLOW
    for sig in signals:
        if sig.level in DECISION_SEVERITY and DECISION_SEVERITY[sig.level] > DECISION_SEVERITY[best]:
            best = sig.level
    return best


def _noisy_or(signals: List[Signal]) -> float:
    """Combine independent signal scores: 1 - prod(1 - s)."""
    remaining = 1.0
    for sig in signals:
        remaining *= 1.0 - max(0.0, min(1.0, sig.score))
    return round(1.0 - remaining, 4)


def _seller_messages(decision: str, signals: List[Signal]) -> tuple:
    """Pick the Swahili and English seller messages for a decision."""
    sw = get_lexicon().seller_messages
    lex = get_lexicon()

    def pick(key: str, **fmt: str) -> tuple:
        return sw.get(key, sw.get(decision, "")).format(**fmt), _EN_MESSAGES.get(key, "").format(**fmt)

    if decision == DECISION_ESCALATE:
        return pick(DECISION_ESCALATE)
    if decision == DECISION_BLOCK:
        blocking = sorted((s for s in signals if s.level == DECISION_BLOCK), key=lambda s: -s.score)
        rule_blocking = [s for s in blocking if s.source not in ("meta_model", "fusion")]
        top = (rule_blocking or blocking or [None])[0]
        if top is not None and top.source in ("image", "video", "hash"):
            return pick("BLOCK_IMAGE")
        if top is not None and top.category in lex.categories:
            return pick(DECISION_BLOCK, label_sw=lex.label(top.category, "sw"), label_en=lex.label(top.category))
        return pick("BLOCK_GENERIC")
    if decision == DECISION_REVIEW:
        cats = {s.category for s in signals if s.level == DECISION_REVIEW}
        if CAT_CONTACT in cats:
            return pick("CONTACT")
        if CAT_MISMATCH in cats:
            return pick("MISMATCH")
        if CAT_NON_PRODUCT in cats:
            return pick("NON_PRODUCT")
        return pick(DECISION_REVIEW)
    return pick(DECISION_ALLOW)


def decide(signals: List[Signal], features: Dict[str, float]) -> Decision:
    """Fuse all signals into the final decision.

    Parameters:
        signals (List[Signal]): Evidence from every check.
        features (Dict[str, float]): Meta-model feature vector (see FEATURE_NAMES).

    Returns:
        Decision: Final decision with reasons and seller messages.
    """
    cfg = CONFIG["decision"]
    rule_level = _max_level(signals)
    actionable = [s for s in signals if s.level != LEVEL_WEAK]
    score = _noisy_or(signals)
    meta_p = meta_probability(features)

    sexual = any(s.category == "SEXUAL" and DECISION_SEVERITY.get(s.level, 0) >= 1 for s in signals)
    youth = any(s.category == CAT_YOUTH for s in signals)
    if sexual and youth:
        signals = signals + [Signal("child_safety", CAT_CHILD_SAFETY, DECISION_ESCALATE, 1.0,
                                    "sexual signal combined with youth indicator: urgent human review")]
        rule_level = DECISION_ESCALATE

    decision = rule_level
    if decision in (DECISION_ALLOW, DECISION_REVIEW):
        if meta_p is not None:
            bundle = load_meta_model() or {}
            non_ml_evidence = any(
                s.source not in ML_SOURCES and DECISION_SEVERITY.get(s.level, 0) >= 1 for s in signals
            )
            # floors from config.json stop a small/unrepresentative training set from over-flagging
            block_t = max(bundle.get("block_threshold", 1.1), float(cfg.get("meta_min_block_threshold", 0.85)))
            review_t = max(bundle.get("review_threshold", 1.1), float(cfg.get("meta_min_review_threshold", 0.50)))
            if meta_p >= block_t and non_ml_evidence:
                decision = DECISION_BLOCK
                signals = signals + [Signal("meta_model", "COMBINED", DECISION_BLOCK, meta_p,
                                            f"combined evidence probability {meta_p:.2f}")]
            elif meta_p >= review_t and decision == DECISION_ALLOW:
                decision = DECISION_REVIEW
                signals = signals + [Signal("meta_model", "COMBINED", DECISION_REVIEW, meta_p,
                                            f"combined evidence probability {meta_p:.2f}")]
        if decision == DECISION_ALLOW and score >= float(cfg.get("review_score", 0.35)):
            decision = DECISION_REVIEW
            signals = signals + [Signal("fusion", "COMBINED", DECISION_REVIEW, score,
                                        "several weak signals together")]

    final_score = max(score, meta_p or 0.0)
    if decision == DECISION_ALLOW:
        final_score = min(final_score, float(cfg.get("review_score", 0.35)) - 0.01)
    reasons = sorted(
        (s for s in signals if s.level != LEVEL_WEAK or decision != DECISION_ALLOW),
        key=lambda s: (-DECISION_SEVERITY.get(s.level, -1), -s.score),
    )
    categories = sorted({s.category for s in actionable if DECISION_SEVERITY.get(s.level, 0) >= 1}
                        | ({CAT_CHILD_SAFETY} if decision == DECISION_ESCALATE else set()))
    sw_msg, en_msg = _seller_messages(decision, signals)
    return Decision(
        decision=decision,
        score=round(final_score, 4),
        categories=categories,
        reasons=reasons,
        seller_message_sw=sw_msg,
        seller_message_en=en_msg,
        meta_model_probability=meta_p,
        features=features,
    )
