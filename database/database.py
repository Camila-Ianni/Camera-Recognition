import os
import sqlite3
from datetime import datetime

class DatabaseManager:
    def __init__(self, db_path="data/security_vision.db"):
        self.db_path = db_path
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        os.makedirs("data/known_faces", exist_ok=True)
        os.makedirs("data/alerts", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Table for settings
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            
            # Table for registered people
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    identifier TEXT UNIQUE NOT NULL,
                    registered_date TEXT NOT NULL,
                    status TEXT NOT NULL, -- authorized / unauthorized
                    reference_image_path TEXT NOT NULL
                )
            """)
            
            # Table for events / alerts
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    person_name TEXT,
                    object_name TEXT,
                    emotion TEXT,
                    risk_score INTEGER NOT NULL,
                    alert_level TEXT NOT NULL, -- LOW, MEDIUM, CRITICAL / HIGH
                    confidence REAL NOT NULL,
                    zone TEXT NOT NULL,
                    evidence_path TEXT
                )
            """)
            conn.commit()

    # Settings functions
    def set_setting(self, key, value):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
                (key, str(value))
            )
            conn.commit()

    def get_setting(self, key, default=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                return row["value"]
            return default

    # Registered people functions
    def register_person(self, name, identifier, status, reference_image_path):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            registered_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                INSERT OR REPLACE INTO people (name, identifier, registered_date, status, reference_image_path)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, identifier, registered_date, status, reference_image_path)
            )
            conn.commit()

    def get_registered_people(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM people ORDER BY name ASC")
            return [dict(row) for row in cursor.fetchall()]

    def delete_person(self, identifier):
        person = self.get_person(identifier)
        if not person:
            return False
            
        # Delete record
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM people WHERE identifier = ?", (identifier,))
            conn.commit()
            
        # Delete image file if it exists
        img_path = person["reference_image_path"]
        if img_path and os.path.exists(img_path):
            try:
                os.remove(img_path)
            except Exception as e:
                print(f"Error removing reference image: {e}")
        return True

    def get_person(self, identifier):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM people WHERE identifier = ?", (identifier,))
            row = cursor.fetchone()
            return dict(row) if row else None

    # Events functions
    def log_event(self, event_type, person_name=None, object_name=None, emotion=None, 
                  risk_score=0, alert_level="LOW", confidence=0.0, zone="General", evidence_path=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                INSERT INTO events (timestamp, event_type, person_name, object_name, emotion, risk_score, alert_level, confidence, zone, evidence_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (timestamp, event_type, person_name, object_name, emotion, risk_score, alert_level, confidence, zone, evidence_path)
            )
            conn.commit()
            return cursor.lastrowid

    def get_events(self, limit=100, alert_level=None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if alert_level:
                cursor.execute(
                    "SELECT * FROM events WHERE alert_level = ? ORDER BY timestamp DESC LIMIT ?",
                    (alert_level, limit)
                )
            else:
                cursor.execute(
                    "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                    (limit,)
                )
            return [dict(row) for row in cursor.fetchall()]

    def delete_old_events_and_evidence(self, days_threshold=30):
        # Find old events with evidence files
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # Calculate date threshold
            cursor.execute(
                f"SELECT id, evidence_path FROM events WHERE datetime(timestamp) < datetime('now', '-{days_threshold} days')"
            )
            old_records = cursor.fetchall()
            
            deleted_files = 0
            for record in old_records:
                evidence_path = record["evidence_path"]
                if evidence_path and os.path.exists(evidence_path):
                    try:
                        # evidence_path can be an MP4 file or folder
                        if os.path.isdir(evidence_path):
                            for file in os.listdir(evidence_path):
                                os.remove(os.path.join(evidence_path, file))
                            os.rmdir(evidence_path)
                        else:
                            os.remove(evidence_path)
                        deleted_files += 1
                    except Exception as e:
                        print(f"Error removing old evidence file {evidence_path}: {e}")
                        
            # Delete old database records
            cursor.execute(
                f"DELETE FROM events WHERE datetime(timestamp) < datetime('now', '-{days_threshold} days')"
            )
            conn.commit()
            
            return len(old_records), deleted_files
