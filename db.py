import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import csv

DB_PATH = os.getenv('APP_DB_PATH', 'app.db')
FEEDBACK_CSV_PATH = os.getenv('FEEDBACK_CSV_PATH', 'feedback_log.csv')

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        original_text TEXT NOT NULL,
        clean_text TEXT NOT NULL,
        prediction TEXT NOT NULL,
        confidence REAL NOT NULL,
        misinfo_probability REAL,
        threshold_used REAL,
        original_language TEXT,
        ensemble_details TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER NOT NULL,
        feedback_type TEXT NOT NULL,
        correct_label TEXT,
        user_agent TEXT,
        ip_address TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
    );
    """
]

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        # Migration: ensure correct_label column exists
        cur.execute("PRAGMA table_info(feedback)")
        cols = [r[1] for r in cur.fetchall()]
        if 'correct_label' not in cols:
            try:
                cur.execute("ALTER TABLE feedback ADD COLUMN correct_label TEXT")
            except Exception:
                pass


def insert_prediction(entry: Dict[str, Any]) -> int:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO predictions
                (timestamp, original_text, clean_text, prediction, confidence, misinfo_probability, threshold_used, original_language, ensemble_details)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry['timestamp'],
                entry['original_text'],
                entry['clean_text'],
                entry['prediction'],
                entry['confidence'],
                entry.get('misinfo_probability'),
                entry.get('threshold_used'),
                entry.get('original_language','en'),
                entry.get('details_json')
            )
        )
        return cur.lastrowid


def add_feedback(prediction_timestamp: str, feedback_type: str, user_agent: Optional[str], ip: Optional[str], correct_label: Optional[str] = None) -> bool:
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id FROM predictions WHERE timestamp=?", (prediction_timestamp,))
        row = cur.fetchone()
        if not row:
            return False
        prediction_id = row[0]
        from datetime import datetime
        cur.execute(
            "INSERT INTO feedback (prediction_id, feedback_type, correct_label, user_agent, ip_address, created_at) VALUES (?,?,?,?,?,?)",
            (prediction_id, feedback_type, correct_label, user_agent, ip, datetime.utcnow().isoformat())
        )
        # Append to CSV log (auto-create with header)
        try:
            file_exists = os.path.isfile(FEEDBACK_CSV_PATH)
            with open(FEEDBACK_CSV_PATH, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['prediction_id','prediction_timestamp','feedback_type','correct_label','user_agent','ip_address','created_at'])
                writer.writerow([
                    prediction_id,
                    prediction_timestamp,
                    feedback_type,
                    correct_label or '',
                    (user_agent or '')[:250],
                    ip or '',
                    datetime.utcnow().isoformat()
                ])
            # Dedicated corrections dataset
            if correct_label:
                corr_path = os.getenv('CORRECTIONS_CSV_PATH', 'corrections.csv')
                new_file = not os.path.isfile(corr_path)
                with open(corr_path, 'a', newline='', encoding='utf-8') as cf:
                    cwriter = csv.writer(cf)
                    if new_file:
                        cwriter.writerow(['prediction_id','timestamp','original_text','predicted_label','correct_label','confidence'])
                    cur2 = conn.cursor()
                    cur2.execute("SELECT original_text, prediction, confidence FROM predictions WHERE id=?", (prediction_id,))
                    prow = cur2.fetchone()
                    if prow:
                        cwriter.writerow([prediction_id, prediction_timestamp, prow[0], prow[1], correct_label, prow[2]])
        except Exception:
            # Fail silently; DB already has the feedback
            pass
        return True


def fetch_history(limit: int = 200) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("""
            SELECT p.*, (
                SELECT json_group_array(json_object(
                    'id', f.id,
                    'feedback_type', f.feedback_type,
                    'created_at', f.created_at,
                    'correct_label', f.correct_label
                )) FROM feedback f WHERE f.prediction_id = p.id
            ) as feedback_items
            FROM predictions p
            ORDER BY p.id DESC
            LIMIT ?
        """, (limit,))
        rows = cur.fetchall()
        out = []
        import json
        for r in rows:
            details = None
            if r['ensemble_details']:
                try:
                    details = json.loads(r['ensemble_details'])
                except Exception:
                    details = {}
            feedback_items = []
            if r['feedback_items']:
                try:
                    feedback_items = json.loads(r['feedback_items'])
                except Exception:
                    feedback_items = []
            out.append({
                'timestamp': r['timestamp'],
                'original_text': r['original_text'],
                'clean_text': r['clean_text'],
                'prediction': r['prediction'],
                'confidence': r['confidence'],
                'misinfo_probability': r['misinfo_probability'],
                'threshold_used': r['threshold_used'],
                'original_language': r['original_language'],
                'details': details or {},
                'feedback_list': feedback_items
            })
        return out


def export_history_rows():
    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT timestamp, original_text, clean_text, prediction, confidence FROM predictions ORDER BY id ASC")
        return cur.fetchall()
