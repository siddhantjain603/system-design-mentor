import time
import uuid
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

from graph.supervisor import app_graph
from graph.state import get_initial_state, MentorState
from guardrails.rate_limiter import RateLimiter
from persistence.session_store import (
    save_session,
    load_session,
    delete_session,
    list_saved_sessions,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="System Design Mentor", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

sessions: dict[str, MentorState] = {}
limiter = RateLimiter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str


class ResetRequest(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_or_create_session(session_id: str) -> MentorState:
    """
    Return session from memory, or restore from disk, or create fresh.
    Priority: in-memory > saved JSON > new session
    """
    if session_id in sessions:
        return sessions[session_id]

    # Try to restore from disk
    restored = load_session(session_id)
    if restored:
        sessions[session_id] = restored
        logger.info(f"Restored session {session_id[:8]} from disk")
        return restored

    # Brand new session
    state = get_initial_state(session_id)
    sessions[session_id] = state
    return state


def _build_progress(state: MentorState) -> dict:
    """Build the progress dict sent to the frontend on every response."""
    completed_list = state.get("topics_completed", [])
    questions = state.get("questions_attempted", 0)
    total_score = state.get("total_practice_score", 0.0)
    avg_score = round(total_score / questions, 1) if questions > 0 else None
    return {
        "topics_completed": len(completed_list),        # int — for ring %
        "topics_completed_list": completed_list,         # list — for green ticks
        "total_topics": 28,                              # drives X/28 display
        "current_topic": state.get("current_topic", ""),
        "syllabus_position": state.get("syllabus_position", 0),
        "total_practice_score": total_score,
        "average_practice_score": avg_score,
        "questions_attempted": questions,
        "interviews_completed": state.get("interviews_completed", 0),
        "interview_active": state.get("interview_active", False),
    }


def _build_guardrail_summary(state: MentorState) -> dict:
    events = state.get("guardrail_events", [])
    last_input  = next((e for e in reversed(events) if e["stage"] == "input_guardrail"), None)
    last_output = next((e for e in reversed(events) if e["stage"] == "output_guardrail"), None)
    return {
        "blocked": state.get("guardrail_blocked", False),
        "violation_type": state.get("guardrail_violation"),
        "input_check": {
            "passed": not last_input["blocked"],
            "elapsed_ms": last_input["elapsed_ms"],
            "violation_detail": last_input.get("violation_detail"),
        } if last_input else None,
        "output_check": {
            "passed": not last_output["blocked"],
            "elapsed_ms": last_output["elapsed_ms"],
            "violation_detail": last_output.get("violation_detail"),
        } if last_output else None,
    }


def _build_trace_summary(state: MentorState) -> dict:
    traces     = state.get("traces", [])
    supervisor = next((t for t in reversed(traces) if t["stage"] == "supervisor_decision"), None)
    tools      = [t for t in traces if t["stage"] == "tool_call"]
    errors     = [t for t in traces if t["stage"] == "error"]
    return {
        "intent":             supervisor["intent"] if supervisor else None,
        "routed_to":          supervisor.get("routed_to") if supervisor else None,
        "tool_calls":         [{"tool": t["tool_name"], "elapsed_ms": t.get("elapsed_ms")} for t in tools[-5:]],
        "errors":             errors[-3:],
        "total_trace_events": len(traces),
    }


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.post("/chat")
async def chat(request: Request, body: ChatRequest):
    # ── Rate limiting ─────────────────────────────────────────────────────
    session_id = body.session_id or str(uuid.uuid4())
    client_ip  = request.client.host if request.client else "unknown"
    limiter.check(session_id=session_id, ip=client_ip)

    # ── Session: load from memory or disk ─────────────────────────────────
    state = _get_or_create_session(session_id)

    # ── Append user message ───────────────────────────────────────────────
    from langchain_core.messages import HumanMessage
    state["messages"] = state["messages"] + [HumanMessage(content=body.message)]

    # ── Run LangGraph ─────────────────────────────────────────────────────
    start = time.time()
    try:
        result = app_graph.invoke(state)
        sessions[session_id] = result

        # ✅ Persist progress to disk after every successful response
        save_session(session_id, result)

    except Exception as e:
        error_str = str(e)
        if "content_filter" in error_str or "ResponsibleAIPolicyViolation" in error_str:
            redirect = (
                "That message was flagged by our safety filter. I'm here to help you "
                "learn and practice **system design** — try asking me to explain a concept, "
                "give you a practice question, or start a mock interview! 🏗️"
            )
            state["messages"] = state["messages"][:-1]
            sessions[session_id] = state
            elapsed = round((time.time() - start) * 1000, 2)
            return {
                "session_id": session_id,
                "reply": redirect,
                "elapsed_ms": elapsed,
                "guardrail_summary": {
                    "blocked": True,
                    "violation_type": "azure_content_filter",
                    "input_check": {"passed": False, "elapsed_ms": 0, "violation_detail": "azure_content_filter:jailbreak"},
                    "output_check": None,
                },
                "trace_summary": {"intent": None, "routed_to": None, "tool_calls": [], "errors": [], "total_trace_events": 0},
                "rate_limit": limiter.get_stats(session_id=session_id, ip=client_ip),
                "progress": _build_progress(state),
            }
        raise HTTPException(status_code=500, detail={"error": error_str, "session_id": session_id})

    elapsed = round((time.time() - start) * 1000, 2)

    from langchain_core.messages import AIMessage
    ai_messages    = [m for m in result["messages"] if isinstance(m, AIMessage)]
    assistant_reply = ai_messages[-1].content if ai_messages else ""
    rate_stats     = limiter.get_stats(session_id=session_id, ip=client_ip)

    return {
        "session_id":       session_id,
        "reply":            assistant_reply,
        "elapsed_ms":       elapsed,
        "current_agent":    result.get("intent"),
        "intent":           result.get("intent"),
        "guardrail_summary": _build_guardrail_summary(result),
        "trace_summary":    _build_trace_summary(result),
        "rate_limit": {
            "session_requests_used": rate_stats["session_requests_in_window"],
            "session_limit":         rate_stats["session_limit"],
            "window_seconds":        rate_stats["window_seconds"],
        },
        "progress": _build_progress(result),
    }


@app.get("/guardrails/{session_id}")
async def get_guardrail_events(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state        = sessions[session_id]
    events       = state.get("guardrail_events", [])
    blocked_count = sum(1 for e in events if e.get("blocked"))
    return {
        "session_id":    session_id,
        "total_checks":  len(events),
        "blocked_count": blocked_count,
        "pass_count":    len(events) - blocked_count,
        "events":        events,
    }


@app.get("/traces/{session_id}")
async def get_traces(session_id: str, limit: int = 60):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    state  = sessions[session_id]
    traces = state.get("traces", [])

    events = []
    for t in traces[-limit:]:
        stage = t.get("stage", "unknown")
        event_type = {
            "supervisor_decision": "supervisor_decision",
            "tool_call":           "tool_call",
            "tool_result":         "tool_result",
            "agent_response":      "agent_response",
            "error":               "error",
        }.get(stage, "llm_call")

        events.append({
            "event_type": event_type,
            "timestamp":  t.get("timestamp", time.strftime("%Y-%m-%dT%H:%M:%SZ")),
            "data":       {k: v for k, v in t.items() if k not in ("stage", "timestamp")},
        })

    return {"session_id": session_id, "events": events}


@app.get("/traces/{session_id}/summary")
async def get_trace_summary(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return _build_trace_summary(sessions[session_id])


@app.get("/progress/{session_id}")
async def get_progress(session_id: str):
    """
    Returns progress for a session — works even if the server was just restarted,
    by loading from disk if not in memory.
    """
    state = _get_or_create_session(session_id)
    return _build_progress(state)


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    from langchain_core.messages import HumanMessage, AIMessage
    messages = sessions[session_id].get("messages", [])
    return {
        "session_id": session_id,
        "history": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
        ],
    }


@app.get("/sessions")
async def list_sessions():
    """Lists both in-memory sessions and saved-but-not-yet-loaded sessions."""
    in_memory = {
        sid: {
            "session_id":        sid,
            "in_memory":         True,
            "messages_count":    len(s.get("messages", [])),
            "topics_completed":  len(s.get("topics_completed", [])),
            "guardrail_blocks":  sum(1 for e in s.get("guardrail_events", []) if e.get("blocked")),
            "interview_active":  s.get("interview_active", False),
        }
        for sid, s in sessions.items()
    }

    # Merge with saved sessions not currently in memory
    saved = {
        s["session_id"]: {**s, "in_memory": False}
        for s in list_saved_sessions()
        if s["session_id"] not in in_memory
    }

    all_sessions = list({**saved, **in_memory}.values())

    return {
        "active_sessions": len(sessions),
        "saved_sessions":  len(saved),
        "sessions":        all_sessions,
    }


@app.post("/reset")
async def reset_session(body: ResetRequest, request: Request):
    session_id = body.session_id
    sessions[session_id] = get_initial_state(session_id)
    limiter.reset(session_id)
    delete_session(session_id)   # ✅ also wipe the saved JSON
    return {"status": "reset", "session_id": session_id}


@app.get("/health")
async def health():
    from persistence.session_store import SESSIONS_DIR
    return {
        "status":          "ok",
        "version":         "3.0.0",
        "active_sessions": len(sessions),
        "persistence": {
            "enabled":      True,
            "backend":      "json",
            "sessions_dir": str(SESSIONS_DIR.resolve()),
        },
        "guardrails": {
            "input_validation":        True,
            "output_validation":       True,
            "prompt_injection_detection": True,
        },
        "rate_limiter": {
            "session_limit":  limiter.session_limit,
            "ip_limit":       limiter.ip_limit,
            "window_seconds": limiter.window,
        },
    }


# ── Static frontend ────────────────────────────────────────────────────────────
try:
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse("frontend/index.html")
except Exception:
    pass
