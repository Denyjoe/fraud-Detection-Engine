"""Database connection manager and schema initializer for Audit Logs.

Provides a thread-safe connection pool/context manager using SQLite in WAL mode
for real-time logging of predictions, reviews, and continuous learning feedback.
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator
import logging
from config.settings import DEFAULT_DB_FILE

logger = logging.getLogger(__name__)


def init_db(db_path: str = DEFAULT_DB_FILE) -> None:
    """Initialize the SQLite database with write-ahead logging (WAL) and required tables.

    Parameters:
        db_path (str): File system path to the SQLite database file.

    Returns:
        None
    """
    logger.info("Initializing SQLite audit database at %s", db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    try:
        cursor = conn.cursor()
        # Enable Write-Ahead Logging for high concurrency without blocking reads
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        cursor.execute("PRAGMA busy_timeout=5000;")

        # Table 1: Listing Moderation Logs (Layer 1 + Layer 2)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS listing_moderation_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                listing_id TEXT NOT NULL,
                seller_id TEXT NOT NULL,
                title TEXT NOT NULL,
                price_tzs REAL NOT NULL,
                mismatch_score REAL NOT NULL,
                slang_score REAL NOT NULL,
                nsfw_detected INTEGER NOT NULL,
                fraud_probability REAL NOT NULL,
                decision TEXT NOT NULL,
                flagged_keywords TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Table 2: Escrow Transaction Fraud Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transaction_fraud_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL UNIQUE,
                seller_id TEXT NOT NULL,
                buyer_id TEXT,
                price_tzs REAL NOT NULL,
                features_json TEXT NOT NULL,
                fraud_probability REAL NOT NULL,
                decision TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Table 3: Continuous Online Learning Feedback Logs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id TEXT NOT NULL,
                actual_label INTEGER NOT NULL,
                previous_probability REAL NOT NULL,
                updated_probability REAL NOT NULL,
                comment TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Table 4: Running System Performance & Model Metrics
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT NOT NULL UNIQUE,
                metric_value REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Seed initial metrics if not present
        cursor.execute("""
            INSERT OR IGNORE INTO model_metrics (metric_name, metric_value)
            VALUES 
                ('total_inferences', 0.0),
                ('total_feedbacks', 0.0),
                ('running_accuracy', 0.0),
                ('running_f1', 0.0);
        """)

        # Create indexes for high-frequency queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_listing_id ON listing_moderation_logs(listing_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tx_id ON transaction_fraud_logs(transaction_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_tx ON feedback_logs(transaction_id);")

        conn.commit()
    finally:
        conn.close()


@contextmanager
def get_db_connection(db_path: str = DEFAULT_DB_FILE) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for obtaining a thread-safe database connection.

    Parameters:
        db_path (str): File system path to the SQLite database.

    Yields:
        sqlite3.Connection: Active database connection with row factory enabled.
    """
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception as exc:
        conn.rollback()
        logger.error("Database transaction rolled back due to error: %s", exc)
        raise
    finally:
        conn.close()
