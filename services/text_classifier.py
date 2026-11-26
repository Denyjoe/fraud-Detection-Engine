"""Text machine-learning checks.

* Zero-shot classification with a multilingual NLI model (default
  ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli``; XNLI includes Swahili, so Swahili
  listings are classified directly without translation).
* Optional Swahili -> English translation with NLLB-200 (off by default, 2.4 GB),
  used to build better English prompts for CLIP and as extra zero-shot input.

The ML scores never block on their own; they feed score fusion / the meta-model and
can send a listing to REVIEW.
"""

import logging
import re
from typing import Dict, Optional

from config.settings import CONFIG
from services.model_registry import pipeline_device, register

logger = logging.getLogger(__name__)

NORMAL_LABEL = "NORMAL"

# Category -> short English label for zero-shot NLI. Short labels scored independently
# (multi_label=True) were measured to be far more reliable on Swahili text than long
# descriptions scored with a softmax across labels.
ZERO_SHOT_LABELS: Dict[str, str] = {
    "DRUGS": "drugs",
    "WEAPONS": "guns",
    "WILDLIFE": "ivory or wild animal skins",
    "FINANCIAL_FRAUD": "fake money or fake documents",
    "STOLEN_GOODS": "stolen phones",
    "SEXUAL": "sex or prostitution",
    "BANNED_TZ": "skin bleaching or HIV cure",
    NORMAL_LABEL: "clothes, food, electronics or household goods",
}
HYPOTHESIS_TEMPLATE = "This advert is selling {}."

_SWAHILI_MARKERS = re.compile(
    r"\b(na|ya|wa|kwa|za|la|cha|vya|hii|bei|nauza|tunauza|inauzwa|mpya|safi|sana|kila|bure|simu|"
    r"nzuri|bidhaa|kutoka|pamoja|haraka|tu|mzigo|ipo|zipo|kali)\b"
)


def looks_swahili(text: str) -> bool:
    """Cheap language heuristic: True when the text contains several Swahili function words.

    Parameters:
        text (str): Listing text.

    Returns:
        bool: True when the text is probably (partly) Swahili.
    """
    words = re.findall(r"[a-z']+", text.lower())
    if not words:
        return False
    hits = sum(1 for w in words if _SWAHILI_MARKERS.fullmatch(w))
    return hits >= 2 or hits / len(words) >= 0.2


def _load_zero_shot():
    """Build the zero-shot pipeline."""
    from transformers import pipeline

    return pipeline(
        "zero-shot-classification",
        model=CONFIG["models"]["text_zero_shot"]["name"],
        device=pipeline_device(),
    )


def _load_translator():
    """Build the NLLB Swahili->English translation pipeline."""
    from transformers import pipeline

    return pipeline(
        "translation",
        model=CONFIG["models"]["translation"]["name"],
        src_lang="swh_Latn",
        tgt_lang="eng_Latn",
        device=pipeline_device(),
    )


ZERO_SHOT = register("text_zero_shot", _load_zero_shot, "multilingual zero-shot NLI")
TRANSLATOR = register("translation", _load_translator, "NLLB Swahili->English")


def translate_to_english(text: str) -> Optional[str]:
    """Translate Swahili text to English when the translator is enabled.

    Parameters:
        text (str): Swahili (or mixed) text.

    Returns:
        Optional[str]: English translation, or None when unavailable / not Swahili.
    """
    if not text.strip() or not looks_swahili(text):
        return None
    model = TRANSLATOR.get()
    if model is None:
        return None
    try:
        return str(model(text[:600], max_length=256)[0]["translation_text"])
    except Exception as exc:
        logger.warning("Translation failed: %s", exc)
        return None


def zero_shot_scores(text: str) -> Optional[Dict[str, float]]:
    """Score ``text`` against every category with the zero-shot model.

    Parameters:
        text (str): Listing text (title + description, optionally with translation).

    Returns:
        Optional[Dict[str, float]]: Independent entailment probability per category
        (incl. ``NORMAL``), or None when the model is disabled/unavailable.
    """
    if not text.strip():
        return None
    model = ZERO_SHOT.get()
    if model is None:
        return None
    labels = list(ZERO_SHOT_LABELS.values())
    try:
        out = model(text[:1000], candidate_labels=labels, hypothesis_template=HYPOTHESIS_TEMPLATE,
                    multi_label=True)
    except Exception as exc:
        logger.warning("Zero-shot inference failed: %s", exc)
        return None
    by_label = dict(zip(out["labels"], out["scores"]))
    return {cat: round(float(by_label[label]), 4) for cat, label in ZERO_SHOT_LABELS.items()}
