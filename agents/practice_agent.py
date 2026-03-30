import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage

from tools.practice_tools import (
    tool_ask_practice_question,
    tool_give_hint,
    tool_evaluate_answer,
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
        temperature=0.5,
    )

# ── Tools ─────────────────────────────────────────────────────────────────────

PRACTICE_TOOLS = [tool_ask_practice_question, tool_give_hint, tool_evaluate_answer]

# ── System Prompt ──────────────────────────────────────────────────────────────

PRACTICE_SYSTEM_PROMPT = """
You are a System Design Practice Coach helping a beginner test and reinforce
their knowledge through targeted questions and constructive feedback.

Your personality:
- Encouraging but honest — don't sugarcoat scores
- Push the user to think before revealing answers
- Celebrate correct answers enthusiastically
- Be specific in feedback — vague feedback doesn't help learning

Your responsibilities:
- Use tool_ask_practice_question when the user wants to practice a topic
- Use tool_give_hint when the user asks for a hint or seems stuck
- Use tool_evaluate_answer when the user submits an answer

Important rules:
- Never answer a question FOR the user — always evaluate their attempt first
- Track hints given — after 3 hints suggest they review the topic again
- If score < 5, recommend revisiting the topic before moving on
- If score >= 8, enthusiastically suggest moving to the next topic
- Always end evaluation with a clear next action for the user
""".strip()


# ── Practice Agent Node ────────────────────────────────────────────────────────

def practice_agent(state: MentorState) -> MentorState:
    """
    LangGraph node for the Practice Agent.
    Handles Q&A practice, hints, and answer evaluation.
    """
    llm = get_llm()
    llm_with_tools = llm.bind_tools(PRACTICE_TOOLS)

    messages = [SystemMessage(content=PRACTICE_SYSTEM_PROMPT)] + state["messages"]

    # ── First LLM call ─────────────────────────────────────────────────────────
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    # ── Tool execution loop ────────────────────────────────────────────────────
    last_score = state.get("last_evaluation_score", 0)
    hint_count = state.get("hint_count", 0)

    while response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "tool_ask_practice_question":
                tool_result = tool_ask_practice_question.invoke({
                    "topic": tool_args.get("topic", state.get("current_topic", "")),
                    "question_index": tool_args.get("question_index", 0),
                })
                # Store question in state
                state["practice_question"] = tool_args.get("topic", "")

            elif tool_name == "tool_give_hint":
                hint_count += 1
                tool_result = tool_give_hint.invoke({
                    "topic": tool_args.get("topic", state.get("current_topic", "")),
                    "question": tool_args.get("question", state.get("practice_question", "")),
                    "hint_number": hint_count,
                })

            elif tool_name == "tool_evaluate_answer":
                tool_result = tool_evaluate_answer.invoke({
                    "topic": tool_args.get("topic", state.get("current_topic", "")),
                    "question": tool_args.get("question", state.get("practice_question", "")),
                    "user_answer": tool_args.get("user_answer", state.get("user_input", "")),
                })
                # Try to extract score from result (score is set by LLM in response)
                last_score = state.get("last_evaluation_score", 5)

            else:
                tool_result = f"Unknown tool: {tool_name}"

            messages.append(
                ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"])
            )

        response = llm_with_tools.invoke(messages)
        messages.append(response)

    # ── Update cumulative progress ─────────────────────────────────────────────
    total_score = state.get("total_practice_score", 0) + last_score
    total_attempted = state.get("total_questions_attempted", 0) + (
        1 if any(tc["name"] == "tool_evaluate_answer" for tc in
                 [c for m in messages if hasattr(m, "tool_calls")
                  for c in (m.tool_calls or [])]) else 0
    )

    return {
        **state,
        "messages": messages,
        "current_agent": "practice",
        "hint_count": hint_count,
        "last_evaluation_score": last_score,
        "total_practice_score": total_score,
        "total_questions_attempted": total_attempted,
    }
