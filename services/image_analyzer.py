"""Image analysis: nudity, contraband, photo-vs-title mismatch, OCR and hash blocklist.

Checks (each one lazy-loaded and fail-soft, see :mod:`services.model_registry`):

* **NSFW**: ``Falconsai/nsfw_image_detection`` (ViT classifier) and **NudeNet**
  (detector of exposed body parts). Pretrained detectors only.
* **CLIP**: zero-shot comparison of the photo against contraband prompts (drugs,
  guns, ivory ...), harmless look-alikes, a product taxonomy and the seller's claimed
  product, giving contraband probabilities and a photo-vs-title mismatch score.
* **OCR**: EasyOCR reads text inside the image so it can be checked with the lexicon.
* **Perceptual hash**: pHash against ``data/hash_blocklist.json`` (images moderators
  already banned), robust to resizing/re-compression.
* **CSAM hash matching**: delegated to :mod:`services.child_safety`.
* **Age estimation**: only run when a sexual signal exists (child-safety escalation).
"""

import base64
import io
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from PIL import Image

from config.settings import (
    CLIP_CONTRABAND_PROMPTS,
    CLIP_LOOKALIKE_PROMPTS,
    CONFIG,
    FINE_GRAINED_VISUAL_TAXONOMY,
    HASH_BLOCKLIST_PATH,
)
from services.model_registry import get_device, pipeline_device, register

logger = logging.getLogger(__name__)

IMAGE_CFG: Dict[str, Any] = CONFIG["image"]
Image.MAX_IMAGE_PIXELS = int(IMAGE_CFG.get("max_image_pixels", 40_000_000))

# NudeNet classes that indicate explicit nudity (covered/clothed classes are ignored)
NUDENET_EXPLICIT = {
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "BUTTOCKS_EXPOSED",
}


class ImageDecodeError(ValueError):
    """Raised when uploaded bytes are not a readable (or are an oversized) image."""


def decode_image(data: Any) -> Image.Image:
    """Decode base64 / data-URI / bytes / PIL input into an RGB image.

    Parameters:
        data (Any): Base64 string (optionally ``data:image/...;base64,``), raw bytes or PIL image.

    Returns:
        Image.Image: RGB image.

    Raises:
        ImageDecodeError: When the data is not a valid image or exceeds the pixel limit.
    """
    if isinstance(data, Image.Image):
        return data.convert("RGB")
    try:
        if isinstance(data, str):
            raw = base64.b64decode(data.split(",", 1)[1] if "," in data else data.strip())
        elif isinstance(data, (bytes, bytearray)):
            raw = bytes(data)
        else:
            raise ImageDecodeError(f"unsupported image input type {type(data).__name__}")
        img = Image.open(io.BytesIO(raw))
        img.load()
        return img.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise ImageDecodeError(f"image too large: {exc}") from exc
    except ImageDecodeError:
        raise
    except Exception as exc:
        raise ImageDecodeError(f"not a valid image: {exc}") from exc


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def _load_nsfw():
    """Falconsai ViT NSFW classifier pipeline."""
    from transformers import pipeline

    return pipeline("image-classification", model=CONFIG["models"]["nsfw_classifier"]["name"],
                    device=pipeline_device())


def _load_nudenet():
    """NudeNet detector (ONNX; weights ship inside the pip package)."""
    from nudenet import NudeDetector

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if get_device() == "cuda" else None
    return NudeDetector(providers=providers)


def _load_age():
    """ViT age-bucket classifier pipeline."""
    from transformers import pipeline

    return pipeline("image-classification", model=CONFIG["models"]["age_estimator"]["name"],
                    device=pipeline_device())


def _load_ocr():
    """EasyOCR reader."""
    import easyocr

    return easyocr.Reader(CONFIG["models"]["ocr"].get("languages", ["en"]), gpu=get_device() == "cuda",
                          verbose=False)


class _ClipBundle:
    """CLIP model + processor with cached embeddings for the fixed prompts."""

    def __init__(self) -> None:
        """Load CLIP and pre-compute embeddings of all fixed prompts."""
        import torch
        from transformers import CLIPModel, CLIPProcessor

        name = CONFIG["models"]["clip"]["name"]
        self.torch = torch
        self.device = get_device()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.model = CLIPModel.from_pretrained(name).to(self.device).eval()
        self.fixed_prompts: List[Tuple[str, str]] = []  # (kind, key)
        texts: List[str] = []
        for category, prompts in CLIP_CONTRABAND_PROMPTS.items():
            for prompt in prompts:
                self.fixed_prompts.append(("contraband", category))
                texts.append(prompt)
        for prompt in CLIP_LOOKALIKE_PROMPTS:
            self.fixed_prompts.append(("lookalike", prompt))
            texts.append(prompt)
        for key, entry in FINE_GRAINED_VISUAL_TAXONOMY.items():
            self.fixed_prompts.append(("product", key))
            texts.append(f"a photo of {entry['label']}")
        self.fixed_texts = texts
        self.fixed_embeddings = self.encode_text(texts)

    def encode_text(self, texts: List[str]):
        """L2-normalised text embeddings."""
        with self.torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            emb = self.model.get_text_features(**inputs)
            emb = emb if isinstance(emb, self.torch.Tensor) else emb.pooler_output
        return emb / emb.norm(dim=-1, keepdim=True)

    def encode_image(self, image: Image.Image):
        """L2-normalised image embedding."""
        with self.torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            emb = self.model.get_image_features(**inputs)
            emb = emb if isinstance(emb, self.torch.Tensor) else emb.pooler_output
        return emb / emb.norm(dim=-1, keepdim=True)


NSFW_MODEL = register("nsfw_classifier", _load_nsfw, "Falconsai NSFW image classifier")
NUDENET_MODEL = register("nudenet", _load_nudenet, "NudeNet exposed-body-part detector")
AGE_MODEL = register("age_estimator", _load_age, "ViT age estimator (child-safety escalation only)")
OCR_MODEL = register("ocr", _load_ocr, "EasyOCR")
CLIP_MODEL = register("clip", _ClipBundle, "CLIP zero-shot vision-language model")


# ---------------------------------------------------------------------------
# Perceptual-hash blocklist
# ---------------------------------------------------------------------------
_hash_lock = threading.Lock()


def _load_blocklist(path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Read the pHash blocklist (empty list when the file does not exist)."""
    path = path or HASH_BLOCKLIST_PATH
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(json.load(handle).get("hashes", []))


def compute_phash(image: Image.Image) -> Optional[str]:
    """Perceptual hash (hex) of ``image``, or None when ``imagehash`` is not installed."""
    try:
        import imagehash
    except ImportError:
        return None
    return str(imagehash.phash(image))


def check_hash_blocklist(phash: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the blocklist entry within ``phash_max_distance`` of ``phash``, if any.

    Parameters:
        phash (Optional[str]): Hex perceptual hash.

    Returns:
        Optional[Dict[str, Any]]: Matching entry plus ``distance``, or None.
    """
    if not phash:
        return None
    import imagehash

    target = imagehash.hex_to_hash(phash)
    max_distance = int(IMAGE_CFG.get("phash_max_distance", 6))
    for entry in _load_blocklist():
        distance = target - imagehash.hex_to_hash(entry["hash"])
        if distance <= max_distance:
            return {**entry, "distance": int(distance)}
    return None


def add_to_blocklist(image: Image.Image, reason: str, path: Optional[Path] = None) -> str:
    """Add an image's perceptual hash to the blocklist file.

    Parameters:
        image (Image.Image): Image moderators confirmed as violating.
        reason (str): Why it is banned (e.g. "DRUGS").
        path (Optional[Path]): Blocklist JSON path (default: HASH_BLOCKLIST_PATH).

    Returns:
        str: The stored hash.
    """
    path = path or HASH_BLOCKLIST_PATH
    phash = compute_phash(image)
    if phash is None:
        raise RuntimeError("imagehash is not installed")
    with _hash_lock:
        entries = _load_blocklist(path)
        if not any(e["hash"] == phash for e in entries):
            entries.append({"hash": phash, "reason": reason, "added": datetime.now(timezone.utc).isoformat()})
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"hashes": entries}, handle, indent=2)
    return phash


# ---------------------------------------------------------------------------
# Analysis result
# ---------------------------------------------------------------------------
@dataclass
class ImageAnalysis:
    """Everything the image checks found for one image (or one video frame)."""

    source: str
    nsfw_probability: Optional[float] = None
    nudenet_score: Optional[float] = None
    nudenet_labels: List[str] = field(default_factory=list)
    contraband: Dict[str, float] = field(default_factory=dict)
    top_visual_label: Optional[str] = None
    top_visual_probability: float = 0.0
    claimed_probability: Optional[float] = None
    mismatch_score: Optional[float] = None
    mismatch_reliable: bool = False
    mismatch_detail: str = ""
    non_product_probability: float = 0.0
    ocr_text: str = ""
    phash: Optional[str] = None
    hash_match: Optional[Dict[str, Any]] = None
    youth_probability: Optional[float] = None
    checks: Dict[str, str] = field(default_factory=dict)

    @property
    def max_contraband(self) -> Tuple[Optional[str], float]:
        """(category, probability) of the strongest contraband prompt group."""
        if not self.contraband:
            return None, 0.0
        cat = max(self.contraband, key=self.contraband.get)
        return cat, self.contraband[cat]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view."""
        cat, prob = self.max_contraband
        return {
            "source": self.source,
            "nsfw_probability": self.nsfw_probability,
            "nudenet_score": self.nudenet_score,
            "nudenet_labels": self.nudenet_labels,
            "contraband": self.contraband,
            "top_contraband": cat,
            "top_contraband_probability": round(prob, 4),
            "top_visual_label": self.top_visual_label,
            "top_visual_probability": self.top_visual_probability,
            "claimed_probability": self.claimed_probability,
            "mismatch_score": self.mismatch_score,
            "mismatch_reliable": self.mismatch_reliable,
            "mismatch_detail": self.mismatch_detail,
            "non_product_probability": self.non_product_probability,
            "ocr_text": self.ocr_text,
            "phash": self.phash,
            "hash_match": self.hash_match,
            "youth_probability": self.youth_probability,
            "checks": self.checks,
        }


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------
def _run_nsfw(image: Image.Image, result: ImageAnalysis) -> None:
    """Falconsai NSFW probability."""
    model = NSFW_MODEL.get()
    if model is None:
        result.checks["nsfw_classifier"] = NSFW_MODEL.status
        return
    scores = {d["label"].lower(): float(d["score"]) for d in model(image, top_k=None)}
    result.nsfw_probability = round(scores.get("nsfw", 0.0), 4)
    result.checks["nsfw_classifier"] = "ok"


def _run_nudenet(image: Image.Image, result: ImageAnalysis) -> None:
    """NudeNet max confidence over explicit classes."""
    model = NUDENET_MODEL.get()
    if model is None:
        result.checks["nudenet"] = NUDENET_MODEL.status
        return
    detections = model.detect(np.array(image.convert("RGBA")))
    explicit = [d for d in detections if d.get("class") in NUDENET_EXPLICIT]
    result.nudenet_score = round(max((float(d["score"]) for d in explicit), default=0.0), 4)
    result.nudenet_labels = sorted({d["class"] for d in explicit})
    result.checks["nudenet"] = "ok"


def _run_ocr(image: Image.Image, result: ImageAnalysis) -> None:
    """EasyOCR text (confidence >= 0.3)."""
    model = OCR_MODEL.get()
    if model is None:
        result.checks["ocr"] = OCR_MODEL.status
        return
    small = image.copy()
    small.thumbnail((1280, 1280))
    lines = model.readtext(np.array(small))
    result.ocr_text = " ".join(text for _, text, conf in lines if conf >= 0.3).strip()
    result.checks["ocr"] = "ok"


def _run_clip(image: Image.Image, claimed_concept: Optional[str], compatible_keys: Set[str],
              result: ImageAnalysis) -> None:
    """CLIP contraband probabilities and photo-vs-claim mismatch."""
    bundle: Optional[_ClipBundle] = CLIP_MODEL.get()
    if bundle is None:
        result.checks["clip"] = CLIP_MODEL.status
        return
    torch = bundle.torch
    image_emb = bundle.encode_image(image)
    text_emb = bundle.fixed_embeddings
    kinds = list(bundle.fixed_prompts)
    labels = list(bundle.fixed_texts)
    if claimed_concept:
        text_emb = torch.cat([text_emb, bundle.encode_text([f"a photo of {claimed_concept}"])])
        kinds.append(("claimed", "CLAIMED"))
        labels.append(claimed_concept)
    scale = bundle.model.logit_scale.exp()
    probs = (scale * image_emb @ text_emb.T).softmax(dim=-1).squeeze(0).tolist()

    contraband: Dict[str, float] = {}
    product: Dict[str, float] = {}
    claimed_prob = 0.0
    for (kind, key), prob in zip(kinds, probs):
        if kind == "contraband":
            contraband[key] = contraband.get(key, 0.0) + prob
        elif kind == "product":
            product[key] = prob
        elif kind == "claimed":
            claimed_prob = prob
    top = int(np.argmax(probs))
    result.contraband = {k: round(v, 4) for k, v in contraband.items()}
    result.top_visual_label = labels[top]
    result.top_visual_probability = round(float(probs[top]), 4)
    result.non_product_probability = round(product.get("SCREENSHOT", 0.0) + product.get("DOCUMENT_RECEIPT", 0.0), 4)

    claimed_group = claimed_prob + sum(p for k, p in product.items() if k in compatible_keys)
    incompatible = {k: p for k, p in product.items()
                    if k not in compatible_keys and not FINE_GRAINED_VISUAL_TAXONOMY[k]["is_non_product"]}
    if incompatible and (claimed_group > 0 or claimed_concept):
        best_key = max(incompatible, key=incompatible.get)
        best = incompatible[best_key]
        result.claimed_probability = round(claimed_group, 4)
        result.mismatch_score = round(best / (best + claimed_group + 1e-9), 4)
        result.mismatch_reliable = bool(compatible_keys)
        result.mismatch_detail = (
            f"photo looks like '{FINE_GRAINED_VISUAL_TAXONOMY[best_key]['label']}' ({best:.0%}) "
            f"rather than the claimed '{claimed_concept}' ({claimed_group:.0%})"
        )
    result.checks["clip"] = "ok"


def estimate_youth(image: Image.Image) -> Optional[float]:
    """Probability that the person shown is a minor (sum of youth age buckets).

    Only called when a sexual signal is present; used solely to ESCALATE to humans.

    Parameters:
        image (Image.Image): RGB image.

    Returns:
        Optional[float]: Youth probability, or None when the estimator is unavailable.
    """
    model = AGE_MODEL.get()
    if model is None:
        return None
    buckets = set(IMAGE_CFG.get("youth_age_buckets", ["0-2", "3-9", "10-19"]))
    scores = model(image, top_k=None)
    return round(sum(float(d["score"]) for d in scores if d["label"] in buckets), 4)


def analyse_image(
    image: Image.Image,
    source: str,
    claimed_concept: Optional[str] = None,
    compatible_keys: Optional[Set[str]] = None,
    run_ocr: bool = True,
) -> ImageAnalysis:
    """Run every enabled image check on one image.

    Each check is isolated: an exception in one is logged and recorded in
    ``checks`` while the remaining checks still run.

    Parameters:
        image (Image.Image): RGB image.
        source (str): File name / frame label for reporting.
        claimed_concept (Optional[str]): English description of what the seller claims to sell.
        compatible_keys (Optional[Set[str]]): Taxonomy keys that count as matching the claim.
        run_ocr (bool): Whether to run OCR (skipped on most video frames for speed).

    Returns:
        ImageAnalysis: Results of all checks.
    """
    result = ImageAnalysis(source=source)
    try:
        result.phash = compute_phash(image)
        result.hash_match = check_hash_blocklist(result.phash)
        result.checks["phash"] = "ok" if result.phash else "unavailable"
    except Exception as exc:
        logger.warning("pHash check failed for %s: %s", source, exc)
        result.checks["phash"] = "error"

    steps = [
        ("nsfw_classifier", lambda: _run_nsfw(image, result)),
        ("nudenet", lambda: _run_nudenet(image, result)),
        ("clip", lambda: _run_clip(image, claimed_concept, compatible_keys or set(), result)),
    ]
    if run_ocr:
        steps.append(("ocr", lambda: _run_ocr(image, result)))
    for name, step in steps:
        try:
            step()
        except Exception as exc:
            logger.warning("Image check '%s' failed for %s: %s", name, source, exc)
            result.checks[name] = "error"
    return result
