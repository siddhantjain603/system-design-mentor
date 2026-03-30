"""
persistence/session_store.py
-----------------------------
JSON-based persistence for System Design Mentor sessions.
Saves progress after every chat turn, loads it back on server restart.

Designed to be swapped for SQLite/Postgres later — only this file needs changing.

Persisted fields (intentionally excludes raw chat messages to keep files small):
  - topics_completed       list[str]
  - syllabus_position      int
  - current_topic          str
  - current_topic_index    int
  - total_practice_score   float
  - questions_attempted    int
  - interview_active       bool
  - interview_stage        str | None
  - interview_system       str | None
  - interview_exchanges    int
  - interviews_completed   int
  - saved_at               ISO timestamp
"""

import json
import os
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from graph.state import MentorState, get_initial_state

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
SESSIONS_DIR = Path(os.getenv("SESSIONS_DIR", "sessions"))

# Fields we persist — everything else (messages, traces, guardrail events) is
# ephemeral and rebuilt fresh each server start.
PERSIST_FIELDS = [
    "topics_completed",
    "syllabus_position",
    "current_topic",
    "current_topic_index",
    "total_practice_score",
    "questions_attempted",
    "interview_active",
    "interview_stage",
    "interview_system",
    "interview_exchanges",
    "interviews_completed",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def _ensure_dir() -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


# ── Public API ─────────────────────────────────────────────────────────────────

def save_session(session_id: str, state: MentorState) -> None:
    """
    Persist the progress fields of a session to a JSON file.
    Safe to call after every chat turn — fast, only writes what changed.
    """
    try:
        _ensure_dir()
        payload = {field: state.get(field) for field in PERSIST_FIELDS}
        payload["session_id"] = session_id
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()

        path = _session_path(session_id)
        # Write to a temp file first, then rename — prevents corrupt files on crash
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(path)

        logger.debug(f"[persistence] saved session {session_id[:8]} — "
                     f"{len(state.get('topics_completed', []))} topics completed")
    except Exception as e:
        # Never crash the chat endpoint because of a save failure
        logger.error(f"[persistence] failed to save session {session_id}: {e}")


def load_session(session_id: str) -> Optional[MentorState]:
    """
    Load a previously saved session.
    Returns a fully initialised MentorState with persisted progress merged in,
    or None if no saved data exists for this session_id.
    """
    path = _session_path(session_id)
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text())

        # Start from a clean state so all required fields are present
        state = get_initial_state(session_id)

        # Merge persisted fields back in
        for field in PERSIST_FIELDS:
            if field in payload and payload[field] is not None:
                state[field] = payload[field]  # type: ignore[literal-required]

        saved_at = payload.get("saved_at", "unknown")
        logger.info(f"[persistence] restored session {session_id[:8]} "
                    f"(saved {saved_at}) — "
                    f"{len(state.get('topics_completed', []))} topics completed")
        return state

    except Exception as e:
        logger.error(f"[persistence] failed to load session {session_id}: {e}")
        return None


def delete_session(session_id: str) -> None:
    """Remove a session's saved data (called on /reset)."""
    path = _session_path(session_id)
    try:
        if path.exists():
            path.unlink()
            logger.debug(f"[persistence] deleted session {session_id[:8]}")
    except Exception as e:
        logger.error(f"[persistence] failed to delete session {session_id}: {e}")


def list_saved_sessions() -> list[dict]:
    """Return a summary of all saved sessions — useful for /sessions endpoint."""
    _ensure_dir()
    results = []
    for path in sorted(SESSIONS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
            results.append({
                "session_id": payload.get("session_id", path.stem),
                "topics_completed": len(payload.get("topics_completed") or []),
                "questions_attempted": payload.get("questions_attempted", 0),
                "interviews_completed": payload.get("interviews_completed", 0),
                "saved_at": payload.get("saved_at"),
            })
        except Exception:
            pass  # skip corrupt files
    return results