import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from tools.interviewer_tools import (
    tool_start_interview,
    tool_probe_deeper,
    tool_score_and_debrief,
)
from graph.state import MentorState

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────

def get_llm():
    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
        temperature=0.8,  # Slightly higher for more natural interview feel
    )

# ── Tools ─────────────────────────────────────────────────────────────────────

INTERVIEWER_TOOLS = [tool_start_interview, tool_probe_deeper, tool_score_and_debrief]

# ── System Prompt ──────────────────────────────────────────────────────────────

INTERVIEWER_SYSTEM_PROMPT = """
You are a senior software engineer at a top tech company (like Google, Meta, or Amazon)
conducting a realistic system design mock interview.

Your personality:
- Professional but approachable
- Ask ONE question at a time — never overwhelm the candidate
- Listen carefully and build on what they say
- Be realistic — don't praise weak answers, but stay constructive

Your responsibilities:
- Use tool_start_interview when the user wants to begin a mock interview
- Use tool_probe_deeper after each candidate response to ask follow-up questions
- Use tool_score_and_debrief when the interview ends or user asks for their score

Interview flow to follow strictly:
1. REQUIREMENTS → ask about functional + non-functional requirements
2. HIGH_LEVEL_DESIGN → ask them to describe main components
3. DEEP_DIVE → pick 1-2 interesting components to probe deeply
4. WRAP_UP → ask about bottlenecks, trade-offs, and improvements
5. COMPLETED → trigger score_and_debrief

Important rules:
- Ask only ONE question per turn
- Don't give away answers — guide with hints if they're stuck
- Advance the stage when the current stage feels sufficiently covered
- If the user says 'end interview', 'score me', or 'I give up' → debrief immediately
- The whole interview should feel like a real FAANG interview, not a quiz
""".strip()

# ── Stage Progression Logic ────────────────────────────────────────────────────

STAGE_ORDER = ["requirements", "high_level_design", "deep_dive", "wrap_up", "completed"]

def get_next_stage(current_stage: str) -> str:
    try:
        idx = STAGE_ORDER.index(current_stage)
        return STAGE_ORDER[min(idx + 1, len(STAGE_ORDER) - 1)]
    except ValueError:
        return "requirements"


# ── Interviewer Agent Node ─────────────────────────────────────────────────────

def interviewer_agent(state: MentorState) -> MentorState:
    """
    LangGraph node for the Interviewer Agent.
    Conducts a full multi-stage mock system design interview.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(INTERVIEWER_TOOLS)

    messages = [SystemMessage(content=INTERVIEWER_SYSTEM_PROMPT)] + state["messages"]

    interview_stage    = state.get("interview_stage", "not_started")
    interview_system   = state.get("interview_system", "")
    interview_transcript = state.get("interview_transcript", [])
    interviews_completed = state.get("interviews_completed", 0)

    # ── First LLM call ─────────────────────────────────────────────────────────
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # ── Tool execution loop ────────────────────────────────────────────────────
    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "tool_start_interview":
                system_name = tool_args.get("system_name", "")
                tool_result = tool_start_interview.invoke({"system_name": system_name})

                # Update interview state
                interview_system = system_name or "Design a URL Shortener"
                interview_stage  = "requirements"
                interview_transcript = []

            elif tool_name == "tool_probe_deeper":
                # Append last exchange to transcript
                user_msg = state.get("user_input", "")
                if user_msg:
                    interview_transcript.append(f"Candidate: {user_msg}")

                tool_result = tool_probe_deeper.invoke({
                    "system_name": tool_args.get("system_name", interview_system),
                    "interview_stage": tool_args.get("interview_stage", interview_stage),
                    "candidate_response": tool_args.get("candidate_response", user_msg),
                    "topics_to_probe": tool_args.get("topics_to_probe", []),
                })

                # Record interviewer question
                interview_transcript.append(f"Interviewer: {str(tool_result)[:200]}...")

                # Check if stage should advance (every ~3 exchanges per stage)
                stage_exchanges = sum(
                    1 for t in interview_transcript
                    if t.startswith("Candidate:")
                )
                if stage_exchanges > 0 and stage_exchanges % 3 == 0:
                    interview_stage = get_next_stage(interview_stage)

            elif tool_name == "tool_score_and_debrief":
                tool_result = tool_score_and_debrief.invoke({
                    "system_name": tool_args.get("system_name", interview_system),
                    "interview_transcript": interview_transcript,
                })
                interview_stage = "completed"
                interviews_completed += 1

            else:
                tool_result = f"Unknown tool: {tool_name}"

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    return {
        **state,
        "messages": messages,
        "current_agent": "interviewer",
        "interview_active": interview_stage not in ("not_started", "completed"),
        "interview_system": interview_system,
        "interview_stage": interview_stage,
        "interview_transcript": interview_transcript,
        "interviews_completed": interviews_completed,
    }
