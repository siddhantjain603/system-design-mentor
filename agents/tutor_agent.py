import os
import time
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage

from tools.tutor_tools import (
    tool_get_syllabus,
    tool_explain_concept,
    tool_suggest_next_topic,
    SYLLABUS,
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
        temperature=0.7,
    )

TUTOR_TOOLS = [tool_get_syllabus, tool_explain_concept, tool_suggest_next_topic]

# ── System Prompt ──────────────────────────────────────────────────────────────

TUTOR_SYSTEM_PROMPT = """
You are a friendly and expert System Design Tutor helping a complete beginner
learn system design from scratch in a structured way.

Your personality:
- Patient, encouraging, and clear
- Use simple language — avoid jargon unless you explain it
- Use emojis to make learning fun
- Celebrate progress and small wins

Your responsibilities:
- Guide the user through the 28-topic syllabus in order
- Use tool_get_syllabus when they ask what to learn or where to start
- Use tool_explain_concept when they want to learn a specific topic
- Use tool_suggest_next_topic when they finish a topic or say "next topic"

Important rules:
- Always teach one topic at a time
- Never skip ahead unless the user explicitly asks
- After explaining, always offer to practice or move to the next topic
- If the user seems confused, re-explain with a different analogy
""".strip()


# ── Tutor Agent Node ───────────────────────────────────────────────────────────

def tutor_agent(state: MentorState) -> MentorState:
    """
    LangGraph node for the Tutor Agent.
    Handles all learning-related interactions and correctly updates
    syllabus progress in state after each tool call.
    """
    start = time.time()
    llm = get_llm()
    llm_with_tools = llm.bind_tools(TUTOR_TOOLS)

    messages = [SystemMessage(content=TUTOR_SYSTEM_PROMPT)] + list(state["messages"])

    # State tracking — start from current state values
    current_topic = state.get("current_topic", "")
    syllabus_position = state.get("syllabus_position", 0)
    topics_completed = list(state.get("topics_completed", []))
    traces = list(state.get("traces", []))

    # ── First LLM call ────────────────────────────────────────────────────────
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # ── Tool execution loop ───────────────────────────────────────────────────
    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_start = time.time()

            # ── Route to correct tool ─────────────────────────────────────────
            if tool_name == "tool_get_syllabus":
                tool_result = tool_get_syllabus.invoke({
                    "topics_completed": topics_completed,
                })

            elif tool_name == "tool_explain_concept":
                # Resolve topic index — use arg if provided, else state position
                topic = tool_args.get("topic", current_topic or SYLLABUS[syllabus_position]["topic"])
                topic_index = tool_args.get("current_topic_index", syllabus_position)

                tool_result = tool_explain_concept.invoke({
                    "topic": topic,
                    "current_topic_index": topic_index,
                })

                # ✅ Update progress — mark topic as active
                if isinstance(tool_result, dict):
                    current_topic = tool_result.get("topic_name", topic)
                    syllabus_position = tool_result.get("topic_index", topic_index)

            elif tool_name == "tool_suggest_next_topic":
                topic_index = tool_args.get("current_topic_index", syllabus_position)

                tool_result = tool_suggest_next_topic.invoke({
                    "current_topic_index": topic_index,
                    "topics_completed": topics_completed,
                })

                # ✅ Update progress — mark topic as completed, advance position
                if isinstance(tool_result, dict):
                    completed_topic = tool_result.get("completed_topic")
                    if completed_topic and completed_topic not in topics_completed:
                        topics_completed.append(completed_topic)

                    next_topic = tool_result.get("next_topic")
                    next_index = tool_result.get("next_index")
                    if next_topic:
                        current_topic = next_topic
                    if next_index is not None:
                        syllabus_position = next_index

            else:
                tool_result = f"Unknown tool: {tool_name}"

            # Trace the tool call
            elapsed_tool = round((time.time() - tool_start) * 1000, 2)
            traces.append({
                "stage": "tool_call",
                "agent": "tutor",
                "tool_name": tool_name,
                "tool_args": str(tool_args)[:300],
                "elapsed_ms": elapsed_tool,
                "success": True,
            })

            # Append tool result — convert dict to string for ToolMessage
            result_content = (
                tool_result.get("display") or tool_result.get("prompt") or str(tool_result)
                if isinstance(tool_result, dict)
                else str(tool_result)
            )
            messages.append(
                ToolMessage(content=result_content, tool_call_id=tool_call["id"])
            )

        # Next LLM call with tool results
        response = llm_with_tools.invoke(messages)
        messages.append(response)

    elapsed = round((time.time() - start) * 1000, 2)
    traces.append({
        "stage": "agent_response",
        "agent": "tutor",
        "elapsed_ms": elapsed,
        "current_topic": current_topic,
        "syllabus_position": syllabus_position,
        "topics_completed_count": len(topics_completed),
    })

    return {
        **state,
        "messages": messages,
        "current_topic": current_topic,
        "syllabus_position": syllabus_position,
        "topics_completed": topics_completed,
        "traces": traces,
    }
