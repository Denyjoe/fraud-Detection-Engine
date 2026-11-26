"""Labelled Tanzanian listings (tests/data/test_listings.csv) through the full engine.

Rules per row:
* label ALLOW  -> decision must be exactly ALLOW (no false positives on legit trade)
* label REVIEW -> decision must be REVIEW or stricter
* label BLOCK  -> decision must be BLOCK (or ESCALATE)
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest

from config.settings import DECISION_ALLOW, DECISION_BLOCK, DECISION_ESCALATE, DECISION_REVIEW
from training.train_meta_model import load_listings, run_listing

CSV_PATH = Path(__file__).resolve().parent / "data" / "test_listings.csv"
ROWS: List[Dict[str, Any]] = load_listings(CSV_PATH)


def test_dataset_is_balanced_and_large_enough() -> None:
    """At least 80 listings, roughly half legitimate and half violating."""
    legit = sum(1 for r in ROWS if r["label"] == DECISION_ALLOW)
    violating = len(ROWS) - legit
    assert len(ROWS) >= 80
    assert abs(legit - violating) <= 0.1 * len(ROWS)


@pytest.mark.parametrize("row", ROWS, ids=[r["id"] for r in ROWS])
def test_listing_decision(row: Dict[str, Any]) -> None:
    """Each labelled listing gets the expected decision."""
    result = run_listing(row)
    decision = result["decision"]
    reasons = [r["detail"] for r in result["reasons"]]
    if row["label"] == DECISION_ALLOW:
        assert decision == DECISION_ALLOW, f"false positive: {reasons}"
    elif row["label"] == DECISION_REVIEW:
        assert decision in (DECISION_REVIEW, DECISION_BLOCK, DECISION_ESCALATE), "violation was allowed"
    else:
        assert decision in (DECISION_BLOCK, DECISION_ESCALATE), f"expected BLOCK, got {decision}: {reasons}"
    assert result["seller_message_sw"], "seller message must always be present"
