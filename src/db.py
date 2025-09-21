from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Iterable, Tuple, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS detections(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    plate TEXT NOT NULL,
    confidence REAL,
    source TEXT NOT NULL,
    ts DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_detections_ts ON detections(ts DESC);
"""

class DB:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    def insert_detection(self, plate: str, confidence: Optional[float], source: str) -> None:
        self._conn.execute(
            "INSERT INTO detections(plate, confidence, source) VALUES (?, ?, ?)",
            (plate, confidence, source),
        )
        self._conn.commit()

    def insert_many(self, rows: Iterable[Tuple[str, Optional[float], str]]) -> None:
        self._conn.executemany(
            "INSERT INTO detections(plate, confidence, source) VALUES (?, ?, ?)",
            rows,
        )
        self._conn.commit()

    def recent(self, limit: int = 50) -> List[Tuple[int, str, Optional[float], str, str]]:
        cur = self._conn.execute(
            "SELECT id, plate, confidence, source, ts FROM detections ORDER BY ts DESC LIMIT ?",
            (limit,),
        )
        return list(cur.fetchall())

    def close(self) -> None:
        self._conn.close()