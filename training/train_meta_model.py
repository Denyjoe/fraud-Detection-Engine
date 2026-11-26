"""Train and evaluate the calibrated meta-model on labelled listings.

CSV format (UTF-8, header required)::

    id,title,description,images,videos,label

* ``images`` / ``videos``: optional file paths (relative to the project root or
  absolute), several separated by ``;``.
* ``label``: ``ALLOW`` (legitimate) or ``REVIEW`` / ``BLOCK`` / ``ESCALATE`` (violating;
  the value is the decision you expect). ``0``/``1`` are accepted as ALLOW/BLOCK.

Commands::

    python -m training.train_meta_model train    data/train_listings.csv
    python -m training.train_meta_model evaluate tests/data/test_listings.csv

``train`` runs every listing through the full pipeline to get its feature vector
(features are cached in ``data/feature_cache.json``), fits a logistic regression with
sigmoid calibration, picks thresholds from out-of-fold predictions (BLOCK: precision
>= ``--block-precision``; REVIEW: recall >= ``--review-recall``) and saves
``models/meta_model.joblib``. ``evaluate`` reports precision, recall, F1, the confusion
matrix and the worst mistakes of the complete engine (rules + meta-model).
"""

import argparse
import csv
import hashlib
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config.settings import BASE_DIR, DATA_DIR, META_MODEL_FILE, enable_utf8_console

logger = logging.getLogger("train_meta_model")

DECISIONS = ["ALLOW", "REVIEW", "BLOCK", "ESCALATE"]
CACHE_PATH = DATA_DIR / "feature_cache.json"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def normalise_label(value: str) -> str:
    """Map a CSV label to ALLOW / REVIEW / BLOCK / ESCALATE."""
    value = (value or "").strip().upper()
    mapping = {"0": "ALLOW", "LEGIT": "ALLOW", "1": "BLOCK", "VIOLATION": "BLOCK", "HOLD_FOR_REVIEW": "REVIEW"}
    value = mapping.get(value, value)
    if value not in DECISIONS:
        raise ValueError(f"unknown label '{value}'")
    return value


def load_listings(path: Path) -> List[Dict[str, Any]]:
    """Read a labelled listings CSV.

    Parameters:
        path (Path): CSV file.

    Returns:
        List[Dict[str, Any]]: Rows with parsed ``media`` paths and normalised ``label``.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            media: List[Path] = []
            for column in ("images", "videos"):
                for item in (row.get(column) or "").split(";"):
                    item = item.strip()
                    if item:
                        p = Path(item)
                        media.append(p if p.is_absolute() else BASE_DIR / p)
            rows.append({
                "id": row["id"],
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "media": media,
                "label": normalise_label(row["label"]),
            })
    return rows


def _media_files(row: Dict[str, Any]) -> list:
    """Load a row's media as MediaFile objects."""
    from services.moderation_layer1 import MediaFile

    return [MediaFile(p.name, p.read_bytes()) for p in row["media"]]


def run_listing(row: Dict[str, Any]) -> Dict[str, Any]:
    """Moderate one labelled listing with the full engine.

    Parameters:
        row (Dict[str, Any]): Row from :func:`load_listings`.

    Returns:
        Dict[str, Any]: Layer 1 result.
    """
    from services.moderation_layer1 import ContentModerationLayer1

    return ContentModerationLayer1().moderate(row["title"], row["description"], _media_files(row))


def _cache_key(row: Dict[str, Any]) -> str:
    """Stable key of a row's content (text + media bytes)."""
    digest = hashlib.sha256(f"{row['title']}\n{row['description']}".encode("utf-8"))
    for p in row["media"]:
        digest.update(p.read_bytes())
    return digest.hexdigest()


def extract_features(rows: List[Dict[str, Any]], use_cache: bool = True) -> List[Dict[str, float]]:
    """Feature vectors for every row (cached on disk by content hash).

    Parameters:
        rows (List[Dict[str, Any]]): Labelled rows.
        use_cache (bool): Re-use cached feature vectors.

    Returns:
        List[Dict[str, float]]: One feature dict per row.
    """
    cache: Dict[str, Dict[str, float]] = {}
    if use_cache and CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    feats: List[Dict[str, float]] = []
    for i, row in enumerate(rows, 1):
        key = _cache_key(row)
        if key not in cache:
            cache[key] = run_listing(row)["features"]
            logger.info("[%d/%d] features for %s", i, len(rows), row["id"])
        feats.append(cache[key])
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache), encoding="utf-8")
    return feats


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def choose_thresholds(y: List[int], p: List[float], block_precision: float, review_recall: float) -> Tuple[float, float]:
    """Pick the BLOCK threshold (precision target) and REVIEW threshold (recall target).

    Parameters:
        y (List[int]): 1 = violating.
        p (List[float]): Out-of-fold probabilities.
        block_precision (float): Minimum precision for probabilities >= BLOCK threshold.
        review_recall (float): Minimum recall for probabilities >= REVIEW threshold.

    Returns:
        Tuple[float, float]: (block_threshold, review_threshold)
    """
    candidates = sorted(set(round(v, 4) for v in p))
    positives = sum(y)
    block = 1.01
    for t in candidates:
        flagged = [yi for yi, pi in zip(y, p) if pi >= t]
        if flagged and sum(flagged) / len(flagged) >= block_precision and sum(flagged) >= 1:
            block = t
            break
    review = 0.0
    for t in reversed(candidates):
        caught = sum(1 for yi, pi in zip(y, p) if yi == 1 and pi >= t)
        if positives and caught / positives >= review_recall:
            review = t
            break
    return block, min(review, block)


def train(csv_path: Path, out_path: Path, block_precision: float, review_recall: float, use_cache: bool) -> None:
    """Fit, calibrate, threshold and save the meta-model."""
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from services.fusion import FEATURE_NAMES

    rows = load_listings(csv_path)
    feats = extract_features(rows, use_cache)
    X = [[f.get(name, 0.0) for name in FEATURE_NAMES] for f in feats]
    y = [0 if r["label"] == "ALLOW" else 1 for r in rows]
    counts = Counter(y)
    if min(counts.values(), default=0) < 5:
        raise SystemExit("Need at least 5 legitimate and 5 violating listings to train.")
    folds = min(5, min(counts.values()))

    def build() -> CalibratedClassifierCV:
        base = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
        return CalibratedClassifierCV(base, method="sigmoid", cv=folds)

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    oof = cross_val_predict(build(), X, y, cv=cv, method="predict_proba")[:, 1].tolist()
    block_t, review_t = choose_thresholds(y, oof, block_precision, review_recall)
    model = build().fit(X, y)

    bundle = {
        "model": model,
        "features": FEATURE_NAMES,
        "block_threshold": block_t,
        "review_threshold": review_t,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "training_rows": len(rows),
        "oof_auc": round(float(roc_auc_score(y, oof)), 4),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)
    print(f"Trained on {len(rows)} listings ({counts[0]} legit / {counts[1]} violating), {folds}-fold CV")
    print(f"Out-of-fold ROC-AUC: {bundle['oof_auc']}")
    print(f"BLOCK threshold  : {block_t:.4f} (precision target {block_precision})")
    print(f"REVIEW threshold : {review_t:.4f} (recall target {review_recall})")
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Precision, recall, F1."""
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate(csv_path: Path, report_path: Optional[Path] = None) -> Dict[str, Any]:
    """Run the full engine on a labelled CSV and print metrics.

    Parameters:
        csv_path (Path): Labelled CSV.
        report_path (Optional[Path]): Optional JSON file for the detailed results.

    Returns:
        Dict[str, Any]: Metrics and per-row results.
    """
    rows = load_listings(csv_path)
    results = []
    for i, row in enumerate(rows, 1):
        out = run_listing(row)
        results.append({"id": row["id"], "title": row["title"], "label": row["label"], "decision": out["decision"],
                        "score": out["score"], "reasons": [r["detail"] for r in out["reasons"]][:3]})
        logger.info("[%d/%d] %s label=%s decision=%s", i, len(rows), row["id"], row["label"], out["decision"])

    def flagged(d: str) -> bool:
        return d != "ALLOW"

    def blocked(d: str) -> bool:
        return d in ("BLOCK", "ESCALATE")

    tp = sum(1 for r in results if flagged(r["label"]) and flagged(r["decision"]))
    fp = sum(1 for r in results if not flagged(r["label"]) and flagged(r["decision"]))
    fn = sum(1 for r in results if flagged(r["label"]) and not flagged(r["decision"]))
    btp = sum(1 for r in results if blocked(r["label"]) and blocked(r["decision"]))
    bfp = sum(1 for r in results if not blocked(r["label"]) and blocked(r["decision"]))
    bfn = sum(1 for r in results if blocked(r["label"]) and not blocked(r["decision"]))
    p, rc, f1 = _prf(tp, fp, fn)
    bp, brc, bf1 = _prf(btp, bfp, bfn)
    labels = ["ALLOW", "REVIEW", "BLOCK"]

    def bucket(d: str) -> str:
        return "BLOCK" if d == "ESCALATE" else d

    matrix = {a: {b: sum(1 for r in results if bucket(r["label"]) == a and bucket(r["decision"]) == b)
                  for b in labels} for a in labels}
    false_pos = sorted((r for r in results if r["label"] == "ALLOW" and flagged(r["decision"])),
                       key=lambda r: -r["score"])
    false_neg = sorted((r for r in results if flagged(r["label"]) and not flagged(r["decision"])),
                       key=lambda r: r["score"])
    under_blocked = [r for r in results if r["label"] == "BLOCK" and r["decision"] == "REVIEW"]

    print(f"\nEvaluated {len(results)} listings from {csv_path}")
    print(f"Violation detection (REVIEW+BLOCK vs ALLOW): precision {p:.3f}  recall {rc:.3f}  F1 {f1:.3f}")
    print(f"BLOCK decisions                            : precision {bp:.3f}  recall {brc:.3f}  F1 {bf1:.3f}")
    print("\nConfusion matrix (rows = label, columns = decision; ESCALATE counted as BLOCK)")
    print("            " + "".join(f"{c:>8}" for c in labels))
    for a in labels:
        print(f"  {a:<10}" + "".join(f"{matrix[a][b]:>8}" for b in labels))
    print("\nWorst mistakes:")
    for kind, items in (("FALSE POSITIVE", false_pos), ("FALSE NEGATIVE", false_neg),
                        ("BLOCK only reviewed", under_blocked)):
        for r in items[:10]:
            print(f"  {kind:<20} {r['id']:<5} score={r['score']:.2f} label={r['label']:<7} "
                  f"decision={r['decision']:<8} {r['title'][:40]!r} {r['reasons'][:1]}")
    if not (false_pos or false_neg or under_blocked):
        print("  none")

    metrics = {"violation": {"precision": p, "recall": rc, "f1": f1},
               "block": {"precision": bp, "recall": brc, "f1": bf1}, "confusion_matrix": matrix,
               "results": results}
    if report_path:
        report_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nDetailed results written to {report_path}")
    return metrics


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    enable_utf8_console()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    for noisy in ("httpx", "transformers", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="train the meta-model")
    t.add_argument("csv", type=Path)
    t.add_argument("--out", type=Path, default=Path(META_MODEL_FILE))
    t.add_argument("--block-precision", type=float, default=0.95)
    t.add_argument("--review-recall", type=float, default=0.95)
    t.add_argument("--no-cache", action="store_true", help="recompute all features")
    e = sub.add_parser("evaluate", help="evaluate the full engine")
    e.add_argument("csv", type=Path)
    e.add_argument("--report", type=Path, default=None, help="write detailed JSON results here")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.csv, args.out, args.block_precision, args.review_recall, not args.no_cache)
    else:
        evaluate(args.csv, args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
