from langchain_core.tools import tool

# ── Interview Systems Bank ─────────────────────────────────────────────────────

INTERVIEW_SYSTEMS = [
    {
        "name": "Design a URL Shortener",
        "difficulty": "Beginner",
        "focus_areas": ["Hashing", "Databases", "Caching", "API Design"],
        "expected_duration": "30 minutes",
    },
    {
        "name": "Design Twitter Feed",
        "difficulty": "Intermediate",
        "focus_areas": ["Fanout", "Caching", "NoSQL", "Message Queues", "CDN"],
        "expected_duration": "45 minutes",
    },
    {
        "name": "Design a Rate Limiter",
        "difficulty": "Intermediate",
        "focus_areas": ["Algorithms", "Redis", "Distributed Systems", "API Gateway"],
        "expected_duration": "30 minutes",
    },
    {
        "name": "Design Netflix",
        "difficulty": "Advanced",
        "focus_areas": ["CDN", "Video Encoding", "Microservices", "Recommendations", "Storage"],
        "expected_duration": "60 minutes",
    },
    {
        "name": "Design WhatsApp",
        "difficulty": "Advanced",
        "focus_areas": ["WebSockets", "Message Storage", "Encryption", "Presence System"],
        "expected_duration": "60 minutes",
    },
]

# ── Scorecard Categories ───────────────────────────────────────────────────────

SCORECARD_CATEGORIES = [
    "Requirements Clarification",
    "High-Level Design",
    "Data Model & Storage",
    "Scalability & Performance",
    "Fault Tolerance & Reliability",
    "Trade-off Awareness",
]


# ── Tool 1 : Start Interview ───────────────────────────────────────────────────

@tool
def tool_start_interview(system_name: str = "") -> str:
    """
    Kicks off a mock system design interview session.
    Use this when the user says 'start interview', 'mock interview', or 'interview me'.

    Args:
        system_name: Optional name of a system to design. If empty, picks beginner-friendly default.
    """
    # Pick system — default to URL Shortener for beginners
    system = next(
        (s for s in INTERVIEW_SYSTEMS if s["name"].lower() == system_name.lower()),
        INTERVIEW_SYSTEMS[0],  # default: URL Shortener
    )

    focus = ", ".join(system["focus_areas"])

    return f"""
You are a senior software engineer at a top tech company conducting a system design interview.
You are interviewing a candidate who is a beginner learning system design.

System to design: **{system["name"]}**
Difficulty: {system["difficulty"]}
Key areas to probe: {focus}
Expected duration: {system["expected_duration"]}

Interview rules:
- Start by warmly welcoming the candidate
- Ask them to clarify requirements FIRST before jumping into design
- Guide with follow-up questions, don't lecture
- Be encouraging but realistic — push them to think deeper
- Move through these stages in order:
  1. Requirements Clarification (functional + non-functional)
  2. High-Level Design (main components and data flow)
  3. Deep Dive (pick 1-2 components to go deeper)
  4. Wrap-Up (bottlenecks, improvements, trade-offs)

Begin the interview now. Start with:
"Welcome! Today we'll design {system["name"]}. Before we jump in — do you have any clarifying questions, or shall we start by defining the requirements?"
""".strip()


# ── Tool 2 : Probe Deeper ──────────────────────────────────────────────────────

@tool
def tool_probe_deeper(
    system_name: str,
    interview_stage: str,
    candidate_response: str,
    topics_to_probe: list[str],
) -> str:
    """
    Generates a follow-up probing question based on the candidate's last response.
    Use this during an active interview when the candidate answers and needs to be pushed further.

    Args:
        system_name: The system being designed
        interview_stage: Current stage — 'requirements', 'high_level_design', 'deep_dive', 'wrap_up'
        candidate_response: What the candidate just said
        topics_to_probe: Areas that haven't been covered yet
    """
    missing = ", ".join(topics_to_probe) if topics_to_probe else "general depth"

    stage_guidance = {
        "requirements": "Focus on clarifying scale (users, requests/sec), consistency vs availability, read-heavy vs write-heavy.",
        "high_level_design": "Ask about specific components — databases, caches, queues. Push for a diagram description.",
        "deep_dive": "Pick the most interesting component they mentioned and ask how it works internally.",
        "wrap_up": "Ask about bottlenecks, single points of failure, and what they'd improve with more time.",
    }.get(interview_stage, "Ask a general follow-up to deepen their thinking.")

    return f"""
You are conducting a system design interview for: {system_name}
Current stage: {interview_stage}
Candidate just said: "{candidate_response}"
Areas not yet covered: {missing}

{stage_guidance}

Generate ONE sharp follow-up question that:
- Builds directly on what the candidate said
- Probes an area they glossed over or missed
- Sounds natural, like a real interviewer
- Is not a yes/no question — requires explanation

If the candidate is doing well, acknowledge briefly before asking.
If they're struggling, give a small hint embedded in the question.

Ask only ONE question. Do not explain or lecture — just ask.
""".strip()


# ── Tool 3 : Score and Debrief ─────────────────────────────────────────────────

@tool
def tool_score_and_debrief(
    system_name: str,
    interview_transcript: list[str],
) -> str:
    """
    Generates a full scorecard and debrief after the mock interview ends.
    Use this when the interview is complete or the user says 'end interview', 'score me', or 'how did I do?'.

    Args:
        system_name: The system that was designed
        interview_transcript: Full list of Q&A exchanges during the interview
    """
    transcript_text = "\n".join(interview_transcript) if interview_transcript else "No transcript available."
    categories = "\n".join(f"- {c}" for c in SCORECARD_CATEGORIES)

    return f"""
You are a senior engineer debriefing a system design interview candidate.

System designed: {system_name}
Interview transcript:
{transcript_text}

Generate a complete debrief using this exact format:

---
🎯 **Overall Score: X/10**

📋 **Scorecard:**
Rate each category from 1-10 with one sentence of feedback:
{categories}

✅ **Strengths (what you did well):**
- (3 specific things)

🔧 **Areas to Improve:**
- (3 specific gaps with actionable advice)

💡 **What a Strong Answer Would Include:**
(Write a 5-7 bullet point model answer covering the key components a senior engineer would mention)

📚 **Topics to Study Next:**
(Recommend 2-3 specific topics from the syllabus based on their gaps)

---
Be honest but constructive. End with an encouraging message motivating them to keep practicing.
""".strip()
