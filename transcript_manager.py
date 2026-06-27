"""
transcript_manager.py
Zoya Agent — Conversation History & Transcript Storage
Supports: In-memory, JSON file, SQLite DB
"""

import asyncio
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("transcript-manager")

# ─── Config ───────────────────────────────────────────
TRANSCRIPTS_DIR = Path(os.getenv("TRANSCRIPTS_DIR", "transcripts"))
DB_PATH = Path(os.getenv("DB_PATH", "transcripts/zoya_transcripts.db"))


# ─── Data Model ───────────────────────────────────────
class TranscriptEntry:
    def __init__(
        self,
        speaker: str,
        text: str,
        role: str = "user",          # "user" | "agent"
        session_id: Optional[str] = None,
        room_name: Optional[str] = None,
        confidence: Optional[float] = None,
    ):
        self.speaker = speaker
        self.text = text.strip()
        self.role = role
        self.session_id = session_id or "unknown"
        self.room_name = room_name or "unknown"
        self.confidence = confidence
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "session_id": self.session_id,
            "room_name": self.room_name,
            "speaker": self.speaker,
            "role": self.role,
            "text": self.text,
            "confidence": self.confidence,
        }

    def __repr__(self):
        return f"[{self.timestamp}] [{self.role.upper()}] {self.speaker}: {self.text}"


# ─── Transcript Manager ────────────────────────────────
class TranscriptManager:
    def __init__(self, session_id: str, room_name: str):
        self.session_id = session_id
        self.room_name = room_name
        self.entries: list[TranscriptEntry] = []
        self.session_start = datetime.now(timezone.utc).isoformat()

        # Setup storage
        TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info(f"📝 TranscriptManager ready | session={session_id} | room={room_name}")

    # ── SQLite Setup ────────────────────────────────────
    def _init_db(self):
        """Create DB tables if not exist"""
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    room_name       TEXT NOT NULL,
                    started_at      TEXT NOT NULL,
                    ended_at        TEXT,
                    total_messages  INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    room_name   TEXT NOT NULL,
                    timestamp   TEXT NOT NULL,
                    speaker     TEXT NOT NULL,
                    role        TEXT NOT NULL,
                    text        TEXT NOT NULL,
                    confidence  REAL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            # Session row insert
            conn.execute("""
                INSERT OR IGNORE INTO sessions (session_id, room_name, started_at)
                VALUES (?, ?, ?)
            """, (self.session_id, self.room_name, self.session_start))
            conn.commit()
        logger.info(f"🗄️  DB ready at {DB_PATH}")

    # ── Add Entry ───────────────────────────────────────
    async def add(
        self,
        speaker: str,
        text: str,
        role: str = "user",
        confidence: Optional[float] = None,
    ) -> TranscriptEntry:
        """Add a transcript entry — saves to memory + DB"""
        if not text.strip():
            return None

        entry = TranscriptEntry(
            speaker=speaker,
            text=text,
            role=role,
            session_id=self.session_id,
            room_name=self.room_name,
            confidence=confidence,
        )
        self.entries.append(entry)
        logger.info(f"💬 {entry}")

        # Save to DB async
        await asyncio.get_event_loop().run_in_executor(
            None, self._save_to_db, entry
        )
        return entry

    def _save_to_db(self, entry: TranscriptEntry):
        """Sync DB write (runs in thread executor)"""
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    INSERT INTO transcripts
                        (session_id, room_name, timestamp, speaker, role, text, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.session_id,
                    entry.room_name,
                    entry.timestamp,
                    entry.speaker,
                    entry.role,
                    entry.text,
                    entry.confidence,
                ))
                conn.execute("""
                    UPDATE sessions
                    SET total_messages = total_messages + 1
                    WHERE session_id = ?
                """, (entry.session_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ DB save failed: {e}")

    # ── Session End ─────────────────────────────────────
    async def end_session(self):
        """Call when room/session ends — saves JSON + updates DB"""
        ended_at = datetime.now(timezone.utc).isoformat()

        # Update DB session end time
        await asyncio.get_event_loop().run_in_executor(
            None, self._close_session_db, ended_at
        )

        # Save JSON file
        await self._save_json(ended_at)
        logger.info(f"✅ Session ended | {len(self.entries)} messages saved")

    def _close_session_db(self, ended_at: str):
        try:
            with sqlite3.connect(DB_PATH) as conn:
                conn.execute("""
                    UPDATE sessions SET ended_at = ? WHERE session_id = ?
                """, (ended_at, self.session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Session close DB error: {e}")

    async def _save_json(self, ended_at: str):
        """Save full session transcript as JSON file"""
        filename = TRANSCRIPTS_DIR / f"session_{self.session_id}_{self.room_name}.json"
        data = {
            "session_id": self.session_id,
            "room_name": self.room_name,
            "started_at": self.session_start,
            "ended_at": ended_at,
            "total_messages": len(self.entries),
            "transcript": [e.to_dict() for e in self.entries],
        }
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: filename.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            )
            logger.info(f"💾 JSON saved: {filename}")
        except Exception as e:
            logger.error(f"❌ JSON save failed: {e}")

    # ── Query Helpers ───────────────────────────────────
    def get_history(self, last_n: int = 20) -> list[dict]:
        """Last N messages — for LLM context injection"""
        return [e.to_dict() for e in self.entries[-last_n:]]

    def get_formatted(self, last_n: int = 10) -> str:
        """Human-readable transcript — for LLM system prompt"""
        recent = self.entries[-last_n:]
        if not recent:
            return "No conversation yet."
        lines = []
        for e in recent:
            prefix = "🤖 Zoya" if e.role == "agent" else f"👤 {e.speaker}"
            lines.append(f"{prefix}: {e.text}")
        return "\n".join(lines)

    def get_stats(self) -> dict:
        """Session stats"""
        speakers = {}
        for e in self.entries:
            speakers[e.speaker] = speakers.get(e.speaker, 0) + 1
        return {
            "session_id": self.session_id,
            "room_name": self.room_name,
            "total_messages": len(self.entries),
            "speakers": speakers,
        }


# ─── Static Query Functions ────────────────────────────
def get_all_sessions() -> list[dict]:
    """Fetch all past sessions from DB"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM sessions ORDER BY started_at DESC
            """).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"❌ get_all_sessions: {e}")
        return []


def get_session_transcript(session_id: str) -> list[dict]:
    """Fetch full transcript for a session"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT * FROM transcripts
                WHERE session_id = ?
                ORDER BY timestamp ASC
            """, (session_id,)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"❌ get_session_transcript: {e}")
        return []


def search_transcripts(query: str) -> list[dict]:
    """Full-text search across all transcripts"""
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT t.*, s.room_name as room
                FROM transcripts t
                JOIN sessions s ON t.session_id = s.session_id
                WHERE t.text LIKE ?
                ORDER BY t.timestamp DESC
                LIMIT 50
            """, (f"%{query}%",)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.error(f"❌ search_transcripts: {e}")
        return []
