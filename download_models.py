"""Download every enabled pretrained model listed in config/config.json.

Run once after installing requirements (and whenever you change a model name):

    python download_models.py            # enabled models only
    python download_models.py --all      # also optional ones (translation, speech)

Models are public Hugging Face / PyPI weights; they are cached in the Hugging Face
cache folder (set HF_HOME to move it). NudeNet ships its weights inside the pip
package. EasyOCR downloads its detector/recogniser on first initialisation.
"""

import argparse
import logging
import sys
from typing import List

from config.settings import CONFIG

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_models")

# Only weights + tokenizer/config files; skip TF/Flax/ONNX duplicates to save disk.
_BASE_PATTERNS: List[str] = ["*.json", "*.txt", "*.model", "*.spm", "*.tiktoken", "tokenizer*", "vocab*", "merges*"]


def download_hf(repo_id: str) -> None:
    """Download one Hugging Face repo, preferring safetensors weights.

    Parameters:
        repo_id (str): Hugging Face model id.
    """
    from huggingface_hub import list_repo_files, snapshot_download

    files = list_repo_files(repo_id)
    root_files = [f for f in files if "/" not in f]
    weights = ["*.safetensors"] if any(f.endswith(".safetensors") for f in root_files) else ["*.bin"]
    logger.info("Downloading %s (%s)", repo_id, weights[0])
    snapshot_download(repo_id, allow_patterns=_BASE_PATTERNS + weights, ignore_patterns=["*/*"])


def main() -> int:
    """Entry point.

    Returns:
        int: Process exit code (number of failed downloads).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="also download disabled/optional models")
    args = parser.parse_args()

    failures = 0
    for key, spec in CONFIG["models"].items():
        if not (spec.get("enabled") or args.all):
            logger.info("Skipping %s (disabled in config.json)", key)
            continue
        try:
            if key == "ocr":
                import easyocr

                easyocr.Reader(spec.get("languages", ["en"]), gpu=False, verbose=False)
                logger.info("EasyOCR weights ready")
            elif key == "nudenet":
                from nudenet import NudeDetector

                NudeDetector()
                logger.info("NudeNet weights ready")
            elif spec.get("name"):
                download_hf(spec["name"])
        except Exception as exc:  # keep going: each model is optional
            failures += 1
            logger.error("Could not download %s: %s", key, exc)
    logger.info("Done. %d failure(s).", failures)
    return failures


if __name__ == "__main__":
    sys.exit(main())
