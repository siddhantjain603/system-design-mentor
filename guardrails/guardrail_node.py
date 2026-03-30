"""
guardrail_node.py
-----------------
Sits between the supervisor and all agents in the LangGraph graph.
Runs 3 checks in order:
  1. Prompt injection detection
  2. Input validation  (off-topic / harmful)
  3. Output validation (wraps agent response after execution)

On any violation → soft redirect back to system design (no hard crash).
All events are appended to state["guardrail_events"] for observability.
"""

import re
import time
from typing import Literal
from langchain_core.messages import AIMessage, HumanMessage
from graph.state import MentorState


# ---------------------------------------------------------------------------
# 1.  PROMPT INJECTION PATTERNS
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    # Classic jailbreaks
    r"ignore\b.{0,30}\binstructions?",
    r"disregard (your |all |previous )?instructions?",
    r"forget (everything|all instructions|your (role|persona|purpose))",
    r"you are now (a |an )?(?!system design)",
    r"act as (a |an )?(?!system design|mentor|tutor|interviewer|practice)",
    r"pretend (you are|to be|that)",
    r"your (new |true |real )?instructions? (are|is|say)",
    r"from now on (you|ignore|forget)",
    # Role override attempts
    r"(switch|change|override) (your )?(role|mode|persona|behavior|rules)",
    r"new (system |)prompt[:\s]",
    r"system[:\s]*\n",
    r"\[system\]",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    # Data extraction attempts
    r"(reveal|show|print|output|display|tell me) (your |the )?(system prompt|instructions|rules|config)",
    r"what (are|were) your (original |initial |system )?instructions",
    r"repeat (your |the )?instructions? (back|above|verbatim)",
]

COMPILED_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


# ---------------------------------------------------------------------------
# 2.  OFF-TOPIC DETECTION — keywords that are clearly not system design
# ---------------------------------------------------------------------------

OFF_TOPIC_KEYWORDS = [
    # Entertainment / lifestyle
    "recipe", "cook", "movie", "music", "song", "game", "sport",
    "football", "cricket", "chess", "dating", "relationship", "travel",
    "hotel", "flight", "booking", "fashion", "clothes",
    # Other tech domains (fine to catch these gently)
    "machine learning", "deep learning", "neural network", "computer vision",
    "nlp model", "train a model", "fine-tune", "pytorch", "tensorflow",
    # Finance / life advice
    "stock market", "crypto", "bitcoin", "investment", "salary negotiation",
    "resume help", "cover letter", "job application",
]

SYSTEM_DESIGN_KEYWORDS = [
    "system design", "architecture", "scalability", "load balancer",
    "database", "cache", "redis", "cdn", "api", "microservice",
    "cap theorem", "sharding", "replication", "kafka", "queue",
    "sql", "nosql", "design twitter", "design netflix", "design url",
    "distributed", "consistency", "availability", "partition",
    "latency", "throughput", "horizontal", "vertical", "scaling",
    "interview", "practice", "learn", "explain", "tutor", "concept",
    "hint", "question", "score", "debrief", "syllabus",
]

HARMFUL_PATTERNS = [
    r"\b(kill|murder|suicide|self.harm|bomb|weapon|drug)\b",
    r"\b(porn|xxx|nsfw|nude|sex)\b",
    r"\b(racist|sexist|hate speech)\b",
    r"\b(hack|exploit|malware|phishing|bypass|crack password)\b",
]
COMPILED_HARMFUL = [re.compile(p, re.IGNORECASE) for p in HARMFUL_PATTERNS]


# ---------------------------------------------------------------------------
# 3.  OUTPUT VALIDATION RULES
# ---------------------------------------------------------------------------

OUTPUT_VIOLATION_PATTERNS = [
    # Agent leaked a system prompt
    r"my (system |)prompt (is|says|reads)",
    r"i (was |am |)(told|instructed|programmed) to",
    r"as an? (openai|gpt|azure|anthropic|claude)",
    # Hallucinated off-topic content
    r"here('s| is) a (recipe|poem|song|joke|story) for you",
    # Potential PII leakage
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",   # phone numbers
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # emails
]
COMPILED_OUTPUT = [re.compile(p, re.IGNORECASE) for p in OUTPUT_VIOLATION_PATTERNS]


# ---------------------------------------------------------------------------
# SOFT REDIRECT MESSAGES
# ---------------------------------------------------------------------------

REDIRECT_MESSAGES = {
    "injection": (
        "That looks like an attempt to change my instructions — I'm here specifically "
        "to help you learn and practice **system design**. "
        "Try asking me to explain a concept, give you a practice question, or start a mock interview! 🏗️"
    ),
    "off_topic": (
        "I'm focused on **system design** topics — I'm not the best assistant for that. "
        "But I can help you with things like:\n"
        "- 📚 **Learn** — 'Explain load balancing'\n"
        "- 🏋️ **Practice** — 'Give me a question on caching'\n"
        "- 🎯 **Interview** — 'Start a mock interview'\n\n"
        "What would you like to explore?"
    ),
    "harmful": (
        "I can't help with that. I'm a system design mentor — "
        "let's keep things focused on architecture and engineering concepts. "
        "What system design topic would you like to tackle? 🏗️"
    ),
    "output_violation": (
        "I noticed my response contained something unexpected, so I've held it back. "
        "Let me try again — could you rephrase your question about system design?"
    ),
}


# ---------------------------------------------------------------------------
# CORE CHECK FUNCTIONS
# ---------------------------------------------------------------------------

def check_prompt_injection(text: str) -> tuple[bool, str]:
    """Returns (is_violation, matched_pattern)."""
    for pattern in COMPILED_INJECTION:
        m = pattern.search(text)
        if m:
            return True, m.group(0)
    return False, ""


def check_off_topic(text: str) -> tuple[bool, str]:
    """
    Returns (is_off_topic, reason).
    Logic:
      - If any harmful pattern → harmful
      - If clearly off-topic keyword AND no system-design keyword → off_topic
      - Otherwise → clean
    """
    lower = text.lower()

    # Harmful check first (highest priority)
    for pattern in COMPILED_HARMFUL:
        m = pattern.search(lower)
        if m:
            return True, "harmful"

    # Off-topic only if no system-design anchor exists
    has_sd_keyword = any(kw in lower for kw in SYSTEM_DESIGN_KEYWORDS)
    if not has_sd_keyword:
        for kw in OFF_TOPIC_KEYWORDS:
            if kw in lower:
                return True, f"off_topic:{kw}"

    return False, ""


def check_output(text: str) -> tuple[bool, str]:
    """Returns (is_violation, matched_pattern)."""
    for pattern in COMPILED_OUTPUT:
        m = pattern.search(text)
        if m:
            return True, m.group(0)
    return False, ""


# ---------------------------------------------------------------------------
# GUARDRAIL NODE  (called by LangGraph)
# ---------------------------------------------------------------------------

def guardrail_node(state: MentorState) -> MentorState:
    """
    Pre-agent guardrail: validates the latest user message.
    Sets state["guardrail_blocked"] = True and appends a redirect AIMessage
    if any check fails.  The supervisor routing function must check this flag
    before forwarding to an agent.
    """
    start = time.time()
    events: list[dict] = list(state.get("guardrail_events", []))

    # Grab the last human message
    last_human = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            last_human = msg.content
            break

    blocked = False
    violation_type = ""
    violation_detail = ""

    # --- Check 1: Prompt injection ---
    is_injection, matched = check_prompt_injection(last_human)
    if is_injection:
        blocked = True
        violation_type = "injection"
        violation_detail = matched

    # --- Check 2: Off-topic / harmful ---
    if not blocked:
        is_bad, reason = check_off_topic(last_human)
        if is_bad:
            blocked = True
            violation_type = reason.split(":")[0]   # "harmful" or "off_topic"
            violation_detail = reason

    elapsed = round((time.time() - start) * 1000, 2)

    event = {
        "stage": "input_guardrail",
        "timestamp": time.time(),
        "elapsed_ms": elapsed,
        "input_preview": last_human[:200],
        "blocked": blocked,
        "violation_type": violation_type if blocked else None,
        "violation_detail": violation_detail if blocked else None,
    }
    events.append(event)

    if blocked:
        redirect_msg = REDIRECT_MESSAGES.get(violation_type, REDIRECT_MESSAGES["off_topic"])
        return {
            **state,
            "guardrail_blocked": True,
            "guardrail_violation": violation_type,
            "guardrail_events": events,
            "messages": state["messages"] + [AIMessage(content=redirect_msg)],
        }

    return {
        **state,
        "guardrail_blocked": False,
        "guardrail_violation": None,
        "guardrail_events": events,
    }


def output_guardrail_node(state: MentorState) -> MentorState:
    """
    Post-agent guardrail: validates the latest AI response before it
    reaches the user.  Replaces violating output with a safe redirect.
    """
    start = time.time()
    events: list[dict] = list(state.get("guardrail_events", []))

    # Grab the last AI message (just appended by the agent)
    last_ai = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai = msg.content
            break

    is_violation, matched = check_output(last_ai)
    elapsed = round((time.time() - start) * 1000, 2)

    event = {
        "stage": "output_guardrail",
        "timestamp": time.time(),
        "elapsed_ms": elapsed,
        "output_preview": last_ai[:200],
        "blocked": is_violation,
        "violation_detail": matched if is_violation else None,
    }
    events.append(event)

    if is_violation:
        # Replace the last AI message with the safe redirect
        safe_messages = [
            m for m in state["messages"] if not (isinstance(m, AIMessage) and m.content == last_ai)
        ]
        safe_messages.append(AIMessage(content=REDIRECT_MESSAGES["output_violation"]))
        return {
            **state,
            "guardrail_blocked": True,
            "guardrail_violation": "output_violation",
            "guardrail_events": events,
            "messages": safe_messages,
        }

    return {
        **state,
        "guardrail_blocked": False,
        "guardrail_events": events,
    }