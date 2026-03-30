"""
state.py  (updated — guardrail fields + 28-topic syllabus)
-----------------------------------------------------------
Shared state that flows through every node in the LangGraph graph.

Guardrail fields:
  - guardrail_blocked    : bool  — True if the last message was blocked
  - guardrail_violation  : str   — type of violation ("injection", "off_topic", "harmful", "output_violation")
  - guardrail_events     : list  — full log of every guardrail check (input + output)
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class MentorState(TypedDict):
    # ── Core conversation ──────────────────────────────────────────────────
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str

    # ── Routing ────────────────────────────────────────────────────────────
    intent: Optional[str]           # "learn" | "practice" | "interview" | "unclear"

    # ── Tutor / syllabus progress ──────────────────────────────────────────
    current_topic: Optional[str]
    current_topic_index: int        # 0-based index used by tutor agent & tools
    topics_completed: list[str]
    syllabus_position: int          # 0-27 index into the 28-topic syllabus

    # ── Practice tracking ──────────────────────────────────────────────────
    total_practice_score: float
    questions_attempted: int
    current_hint_level: int         # 0=no hint, 1=gentle, 2=medium, 3=full

    # ── Interview tracking ─────────────────────────────────────────────────
    interview_active: bool
    interview_stage: Optional[str]  # "requirements" | "design" | "deep_dive" | "wrap_up"
    interview_system: Optional[str] # e.g. "Design Twitter"
    interview_exchanges: int        # exchanges within current stage
    interviews_completed: int

    # ── Observability ──────────────────────────────────────────────────────
    traces: list[dict]              # supervisor + agent + tool events

    # ── Guardrails ─────────────────────────────────────────────────────────
    guardrail_blocked: bool         # True = last message was blocked by guardrails
    guardrail_violation: Optional[str]   # type of violation, None if clean
    guardrail_events: list[dict]    # full log of every guardrail check run


def get_initial_state(session_id: str) -> MentorState:
    """Factory — returns a clean state for a new session."""
    return MentorState(
        messages=[],
        session_id=session_id,
        intent=None,
        current_topic=None,
        current_topic_index=0,
        topics_completed=[],
        syllabus_position=0,
        total_practice_score=0.0,
        questions_attempted=0,
        current_hint_level=0,
        interview_active=False,
        interview_stage=None,
        interview_system=None,
        interview_exchanges=0,
        interviews_completed=0,
        traces=[],
        guardrail_blocked=False,
        guardrail_violation=None,
        guardrail_events=[],
    )