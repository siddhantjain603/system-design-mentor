"""
supervisor.py  (updated — with guardrails)
------------------------------------------
LangGraph graph flow (updated):

  User Message
       │
       ▼
  SUPERVISOR            ← detects intent, sets state["intent"]
       │
       ▼
  GUARDRAIL_NODE        ← input validation + injection detection
       │
       ├─ blocked=True  ──► END   (redirect message already in state)
       │
       └─ blocked=False
              │
              ├── "learn"     ──► TUTOR AGENT
              ├── "practice"  ──► PRACTICE AGENT
              ├── "interview" ──► INTERVIEWER AGENT
              └── "unclear"   ──► FALLBACK NODE
                                       │
                                       ▼
                               OUTPUT_GUARDRAIL_NODE  ← validates AI response
                                       │
                                       ▼
                                      END
"""

import time
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from graph.state import MentorState, get_initial_state
from agents.tutor_agent import tutor_agent
from agents.practice_agent import practice_agent
from agents.interviewer_agent import interviewer_agent
from guardrails.guardrail_node import guardrail_node, output_guardrail_node
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM for supervisor (deterministic routing)
# ---------------------------------------------------------------------------

supervisor_llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0.0,
)

SUPERVISOR_SYSTEM = """You are a routing supervisor for a System Design Mentor application.

Your ONLY job is to classify the user's intent into exactly one of these categories:
- "learn"     — user wants to learn a concept, see the syllabus, or get an explanation
- "practice"  — user wants a practice question, a hint, or to submit an answer
- "interview" — user wants to start, continue, or end a mock interview
- "unclear"   — none of the above

Respond with ONLY the single word — no punctuation, no explanation.
"""


# ---------------------------------------------------------------------------
# SUPERVISOR NODE
# ---------------------------------------------------------------------------

def supervisor_node(state: MentorState) -> MentorState:
    start = time.time()
    traces: list[dict] = list(state.get("traces", []))

    # If an interview is already active, always route there (don't interrupt)
    if state.get("interview_active"):
        traces.append({
            "stage": "supervisor_decision",
            "timestamp": time.time(),
            "intent": "interview",
            "reason": "interview_lock",
            "elapsed_ms": round((time.time() - start) * 1000, 2),
        })
        return {**state, "intent": "interview", "traces": traces}

    # Grab last human message for classification
    last_human = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    response = supervisor_llm.invoke([
        SystemMessage(content=SUPERVISOR_SYSTEM),
        HumanMessage(content=last_human),
    ])

    raw = response.content.strip().lower()
    intent = raw if raw in {"learn", "practice", "interview", "unclear"} else "unclear"

    elapsed = round((time.time() - start) * 1000, 2)
    traces.append({
        "stage": "supervisor_decision",
        "timestamp": time.time(),
        "intent": intent,
        "raw_llm_output": raw,
        "routed_to": intent,
        "elapsed_ms": elapsed,
    })

    return {**state, "intent": intent, "traces": traces}


# ---------------------------------------------------------------------------
# FALLBACK NODE
# ---------------------------------------------------------------------------

def fallback_node(state: MentorState) -> MentorState:
    msg = (
        "I'm your **System Design Mentor**! Here's what I can help with:\n\n"
        "- 📚 **Learn** — 'Explain load balancing' or 'Show me the syllabus'\n"
        "- 🏋️ **Practice** — 'Give me a caching question' or 'I need a hint'\n"
        "- 🎯 **Interview** — 'Start a mock interview'\n\n"
        "What would you like to do?"
    )
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=msg)],
    }


# ---------------------------------------------------------------------------
# ROUTING FUNCTIONS
# ---------------------------------------------------------------------------

def route_after_guardrail(state: MentorState) -> str:
    """
    Called after the guardrail node.
    If blocked → END (redirect message is already in state["messages"]).
    Otherwise → route to the correct agent based on intent.
    """
    if state.get("guardrail_blocked"):
        return END

    intent = state.get("intent", "unclear")
    route_map = {
        "learn": "tutor",
        "practice": "practice",
        "interview": "interviewer",
        "unclear": "fallback",
    }
    return route_map.get(intent, "fallback")


def route_after_agent(state: MentorState) -> str:
    """Always go to output guardrail after any agent finishes."""
    return "output_guardrail"


# ---------------------------------------------------------------------------
# AGENT WRAPPERS  (add output_guardrail routing after each)
# ---------------------------------------------------------------------------

def tutor_node(state: MentorState) -> MentorState:
    return tutor_agent(state)


def practice_node(state: MentorState) -> MentorState:
    return practice_agent(state)


def interviewer_node(state: MentorState) -> MentorState:
    return interviewer_agent(state)


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(MentorState)

    # Add all nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("guardrail", guardrail_node)          # INPUT guardrail
    graph.add_node("tutor", tutor_node)
    graph.add_node("practice", practice_node)
    graph.add_node("interviewer", interviewer_node)
    graph.add_node("fallback", fallback_node)
    graph.add_node("output_guardrail", output_guardrail_node)  # OUTPUT guardrail

    # Entry point
    graph.set_entry_point("supervisor")

    # Supervisor → guardrail (always)
    graph.add_edge("supervisor", "guardrail")

    # Guardrail → agent or END
    graph.add_conditional_edges(
        "guardrail",
        route_after_guardrail,
        {
            "tutor": "tutor",
            "practice": "practice",
            "interviewer": "interviewer",
            "fallback": "fallback",
            END: END,
        },
    )

    # Every agent → output guardrail
    graph.add_edge("tutor", "output_guardrail")
    graph.add_edge("practice", "output_guardrail")
    graph.add_edge("interviewer", "output_guardrail")
    graph.add_edge("fallback", "output_guardrail")

    # Output guardrail → END
    graph.add_edge("output_guardrail", END)

    return graph.compile()


# Singleton compiled graph
app_graph = build_graph()
