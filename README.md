# System Design Mentor

A multi-agent chat application that helps users learn system design through a structured 28-topic syllabus, practice questions, and mock interviews. Built with LangGraph, Azure OpenAI, and FastAPI.

---

## Overview

Users interact with the app via a web interface and are routed to one of three specialized agents based on their intent:

- **Tutor** — explains concepts from a 28-topic system design syllabus
- **Practice Coach** — asks questions, gives hints, and evaluates answers
- **Interviewer** — runs multi-stage mock interviews

A supervisor node classifies each message and routes it to the right agent. Input and output guardrails run on every turn.

---

## Architecture

```
User Message
     │
     ▼
SUPERVISOR          ← classifies intent: learn / practice / interview / unclear
     │
     ▼
GUARDRAIL NODE      ← prompt injection detection + off-topic / harmful check
     │
     ├─ blocked ──► END
     │
     └─ clean
          ├── learn      ──► TUTOR AGENT
          ├── practice   ──► PRACTICE AGENT
          ├── interview  ──► INTERVIEWER AGENT
          └── unclear    ──► FALLBACK NODE
                                  │
                                  ▼
                         OUTPUT GUARDRAIL NODE
                                  │
                                  ▼
                                 END
```

---

## Features

- **Three specialized agents** — tutor, practice coach, and interviewer, each with their own tools and system prompts
- **LangGraph supervisor** — classifies intent and routes messages; locks to interviewer when an interview is active
- **Guardrails** — input guardrail checks for prompt injection, off-topic content, and harmful patterns; output guardrail validates agent responses
- **Rate limiting** — per-session and per-IP limits enforced on every `/chat` request
- **Session persistence** — progress fields (topics completed, scores, interview state) saved to JSON after each turn and restored on server restart; chat messages are ephemeral
- **Observability** — traces logged for every supervisor decision, tool call, and guardrail event, accessible via `/traces/{session_id}`
- **FastAPI backend** — REST endpoints for chat, history, progress, guardrail events, traces, and session management
- **Web frontend** — single HTML file served by FastAPI

---

## Project Structure

```
project/
├── agents/
│   ├── tutor_agent.py          # Tutor agent — syllabus explanation
│   ├── practice_agent.py       # Practice coach — questions, hints, scoring
│   └── interviewer_agent.py    # Interviewer — multi-stage mock interview
├── api/
│   └── main.py                 # FastAPI app and all endpoints
├── graph/
│   ├── state.py                # MentorState TypedDict + initial state factory
│   └── supervisor.py           # LangGraph graph definition and routing logic
├── guardrails/
│   ├── guardrail_node.py       # Input and output guardrail nodes
│   └── rate_limiter.py         # Per-session and per-IP rate limiting
├── persistence/
│   └── session_store.py        # JSON-based session persistence
├── tools/
│   ├── tutor_tools.py          # Tools: get_syllabus, explain_concept, suggest_next_topic
│   ├── practice_tools.py       # Tools: ask_question, give_hint, evaluate_answer
│   └── interviewer_tools.py    # Tools: start_interview, probe_deeper, score_and_debrief
├── frontend/
│   └── index.html              # Web interface
├── research/                   # Test scripts for Azure connection, guardrails, rate limiting
├── requirements.txt
├── pyproject.toml
└── .env.example
```

---

## Prerequisites

- Python 3.11+
- Azure OpenAI resource with a GPT model deployed

---

## Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your values:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=your_gpt_deployment_name
```

Optionally set `SESSIONS_DIR` to change where session JSON files are saved (default: `sessions/`).

---

## Running the App

```bash
python -m api.main
```

The server starts at `http://localhost:8000`. Open that URL in your browser to use the web interface.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Send a message; creates or continues a session |
| `GET` | `/history/{session_id}` | Get chat history for a session |
| `GET` | `/progress/{session_id}` | Get progress (topics completed, scores, interview state) |
| `GET` | `/guardrails/{session_id}` | Get guardrail event log for a session |
| `GET` | `/traces/{session_id}` | Get trace events for a session |
| `GET` | `/traces/{session_id}/summary` | Get a summary of the latest trace |
| `GET` | `/sessions` | List all in-memory and saved sessions |
| `POST` | `/reset` | Reset a session and clear saved progress |
| `GET` | `/health` | Health check with guardrail and rate limiter config |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | ✅ | Azure OpenAI resource endpoint URL |
| `AZURE_OPENAI_API_KEY` | ✅ | Azure OpenAI API key |
| `AZURE_OPENAI_API_VERSION` | ✅ | API version (e.g., `2024-02-15-preview`) |
| `AZURE_OPENAI_CHAT_DEPLOYMENT` | ✅ | Name of your deployed GPT model |
| `SESSIONS_DIR` | ❌ | Directory for session JSON files (default: `sessions/`) |

---

## Security Notes

- Do not commit your `.env` file — it contains your Azure credentials
- The guardrail node uses regex pattern matching and keyword lists; it is not a substitute for Azure's built-in content filtering, which also runs independently