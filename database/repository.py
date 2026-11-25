"""Audit Log Repository.

Implements the Repository pattern for persisting and retrieving audit logs,
moderation decisions, transaction evaluations, and online learning metrics.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from database.db import get_db_connection
from config.settings import DEFAULT_DB_FILE

logger = logging.getLogger(__name__)


class AuditLogRepository:
    """Repository class encapsulating all SQLite database operations for the fraud moderation system."""

    def __init__(self, db_path: str = DEFAULT_DB_FILE) -> None:
        """Initialize the repository with the specified database path.

        Parameters:
            db_path (str): Path to SQLite audit database.
        """
        self.db_path = db_path

    def log_listing_moderation(
        self,
        listing_id: str,
        seller_id: str,
        title: str,
        price_tzs: float,
        mismatch_score: float,
        slang_score: float,
        nsfw_detected: bool,
        fraud_probability: float,
        decision: str,
        flagged_keywords: List[str],
    ) -> int:
        """Record a listing moderation inspection into the audit log.

        Parameters:
            listing_id (str): Unique listing identifier.
            seller_id (str): Seller identifier.
            title (str): Listing title.
            price_tzs (float): Listing price in Tanzanian Shillings.
            mismatch_score (float): Vision-language mismatch score (0.0 to 1.0).
            slang_score (float): Illegal goods/slang risk score (0.0 to 1.0).
            nsfw_detected (bool): Whether adult or abusive content was flagged.
            fraud_probability (float): Combined fraud risk probability (0.0 to 1.0).
            decision (str): Final system decision (ALLOW, HOLD_FOR_REVIEW, BLOCK).
            flagged_keywords (List[str]): Prohibited or slang words detected.

        Returns:
            int: Inserted row identifier.
        """
        query = """
            INSERT INTO listing_moderation_logs (
                listing_id, seller_id, title, price_tzs, mismatch_score,
                slang_score, nsfw_detected, fraud_probability, decision, flagged_keywords
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            listing_id,
            seller_id,
            title,
            price_tzs,
            mismatch_score,
            slang_score,
            1 if nsfw_detected else 0,
            fraud_probability,
            decision,
            json.dumps(flagged_keywords),
        )
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            self._increment_metric_locked(cursor, "total_inferences", 1.0)
            return cursor.lastrowid or 0

    def log_transaction_fraud(
        self,
        transaction_id: str,
        seller_id: str,
        buyer_id: Optional[str],
        price_tzs: float,
        features: Dict[str, Any],
        fraud_probability: float,
        decision: str,
    ) -> int:
        """Record an escrow payment evaluation into the audit log.

        Parameters:
            transaction_id (str): Escrow transaction identifier.
            seller_id (str): Merchant or seller identifier.
            buyer_id (Optional[str]): Buyer identifier.
            price_tzs (float): Transaction amount in TZS.
            features (Dict[str, Any]): Feature vector evaluated by the ML model.
            fraud_probability (float): Model predicted fraud probability (0.0 to 1.0).
            decision (str): Transaction risk decision (ALLOW, HOLD_FOR_REVIEW, BLOCK).

        Returns:
            int: Inserted row identifier.
        """
        query = """
            INSERT OR REPLACE INTO transaction_fraud_logs (
                transaction_id, seller_id, buyer_id, price_tzs,
                features_json, fraud_probability, decision
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        params = (
            transaction_id,
            seller_id,
            buyer_id or "anonymous",
            price_tzs,
            json.dumps(features),
            fraud_probability,
            decision,
        )
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            self._increment_metric_locked(cursor, "total_inferences", 1.0)
            return cursor.lastrowid or 0

    def log_feedback(
        self,
        transaction_id: str,
        actual_label: int,
        previous_probability: float,
        updated_probability: float,
        comment: Optional[str] = None,
    ) -> int:
        """Record an online learning feedback event and update running metrics.

        Parameters:
            transaction_id (str): Transaction identifier that received feedback.
            actual_label (int): Ground truth label (0 = Legitimate, 1 = Fraudulent).
            previous_probability (float): Probability before online model update.
            updated_probability (float): Probability after online model update.
            comment (Optional[str]): Context or administrative notes.

        Returns:
            int: Inserted row identifier.
        """
        query = """
            INSERT INTO feedback_logs (
                transaction_id, actual_label, previous_probability,
                updated_probability, comment
            ) VALUES (?, ?, ?, ?, ?)
        """
        params = (
            transaction_id,
            actual_label,
            previous_probability,
            updated_probability,
            comment or "Real-time online learning feedback update",
        )
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            self._increment_metric_locked(cursor, "total_feedbacks", 1.0)
            return cursor.lastrowid or 0

    def get_metrics_summary(self) -> Dict[str, float]:
        """Retrieve the current aggregated system metrics and log counts.

        Returns:
            Dict[str, float]: Mapping of metric names to current recorded values.
        """
        query = "SELECT metric_name, metric_value FROM model_metrics"
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            metrics = {row["metric_name"]: float(row["metric_value"]) for row in cursor.fetchall()}

            # Append live count aggregates
            cursor.execute("SELECT COUNT(*) AS total FROM listing_moderation_logs")
            metrics["total_listings_moderated"] = float(cursor.fetchone()["total"])

            cursor.execute("SELECT COUNT(*) AS total FROM transaction_fraud_logs")
            metrics["total_transactions_evaluated"] = float(cursor.fetchone()["total"])

            cursor.execute("SELECT COUNT(*) AS total FROM feedback_logs")
            metrics["total_feedback_events"] = float(cursor.fetchone()["total"])

            return metrics

    def get_recent_audit_logs(self, limit: int = 15) -> List[Dict[str, Any]]:
        """Retrieve the most recent audit logs across listings and transactions.

        Parameters:
            limit (int): Maximum number of entries to return (default: 15).

        Returns:
            List[Dict[str, Any]]: List of unified audit records ordered newest first.
        """
        query = """
            SELECT 
                'LISTING' AS event_type,
                listing_id AS reference_id,
                seller_id,
                title AS detail,
                price_tzs,
                fraud_probability,
                decision,
                created_at
            FROM listing_moderation_logs
            UNION ALL
            SELECT 
                'TRANSACTION' AS event_type,
                transaction_id AS reference_id,
                seller_id,
                'Escrow Payment' AS detail,
                price_tzs,
                fraud_probability,
                decision,
                created_at
            FROM transaction_fraud_logs
            ORDER BY created_at DESC
            LIMIT ?
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, (limit,))
            rows = cursor.fetchall()
            return [
                {
                    "event_type": row["event_type"],
                    "reference_id": row["reference_id"],
                    "seller_id": row["seller_id"],
                    "detail": row["detail"],
                    "price_tzs": float(row["price_tzs"]),
                    "fraud_probability": float(row["fraud_probability"]),
                    "decision": row["decision"],
                    "created_at": str(row["created_at"]),
                }
                for row in rows
            ]

    def update_model_accuracy(self, running_accuracy: float, running_f1: float) -> None:
        """Update running accuracy and F1 score in the metrics table.

        Parameters:
            running_accuracy (float): Current accuracy score.
            running_f1 (float): Current F1 score.

        Returns:
            None
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE model_metrics SET metric_value = ?, last_updated = CURRENT_TIMESTAMP WHERE metric_name = 'running_accuracy'",
                (running_accuracy,),
            )
            cursor.execute(
                "UPDATE model_metrics SET metric_value = ?, last_updated = CURRENT_TIMESTAMP WHERE metric_name = 'running_f1'",
                (running_f1,),
            )

    def _increment_metric_locked(self, cursor: Any, metric_name: str, delta: float) -> None:
        """Helper to increment a running metric inside an existing transaction.

        Parameters:
            cursor (Any): Active SQLite cursor.
            metric_name (str): Target metric name.
            delta (float): Value to increment by.

        Returns:
            None
        """
        cursor.execute(
            """
            UPDATE model_metrics 
            SET metric_value = metric_value + ?, last_updated = CURRENT_TIMESTAMP 
            WHERE metric_name = ?
            """,
            (delta, metric_name),
        )
