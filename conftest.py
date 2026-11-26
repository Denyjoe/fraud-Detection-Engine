"""Pytest configuration: isolate tests from the real audit database and fraud model.

Environment variables are set before any project module is imported, so
``config.settings`` points at temporary copies for the whole test session.
"""

import os
import shutil
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="fraud-tests-"))
_REAL_MODEL = _ROOT / "models" / "fraud_model.joblib"
if _REAL_MODEL.exists():
    shutil.copy(_REAL_MODEL, _TMP / "fraud_model.joblib")

os.environ.setdefault("AUDIT_DB_PATH", str(_TMP / "audit_test.db"))
os.environ.setdefault("FRAUD_MODEL_PATH", str(_TMP / "fraud_model.joblib"))
