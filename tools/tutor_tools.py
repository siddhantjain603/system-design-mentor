from langchain_core.tools import tool

# ── Syllabus Definition ────────────────────────────────────────────────────────

SYLLABUS = [
    # Level 1 — Foundations
    {"index": 0,  "topic": "Client-Server Model & HTTP",        "level": "Level 1 — Foundations"},
    {"index": 1,  "topic": "APIs — REST, GraphQL, gRPC",        "level": "Level 1 — Foundations"},
    {"index": 2,  "topic": "Databases — SQL vs NoSQL",          "level": "Level 1 — Foundations"},
    {"index": 3,  "topic": "DNS & How the Internet Works",      "level": "Level 1 — Foundations"},
    {"index": 4,  "topic": "Proxies & Reverse Proxies",        "level": "Level 1 — Foundations"},

    # Level 2 — Scaling Basics
    {"index": 5,  "topic": "Caching (Redis, CDN)",              "level": "Level 2 — Scaling Basics"},
    {"index": 6,  "topic": "Load Balancing",                    "level": "Level 2 — Scaling Basics"},
    {"index": 7,  "topic": "Message Queues (Kafka, RabbitMQ)",  "level": "Level 2 — Scaling Basics"},
    {"index": 8,  "topic": "Rate Limiting & Throttling",        "level": "Level 2 — Scaling Basics"},
    {"index": 9,  "topic": "Horizontal vs Vertical Scaling",    "level": "Level 2 — Scaling Basics"},

    # Level 3 — Distributed Systems
    {"index": 10, "topic": "CAP Theorem",                       "level": "Level 3 — Distributed Systems"},
    {"index": 11, "topic": "Sharding & Partitioning",           "level": "Level 3 — Distributed Systems"},
    {"index": 12, "topic": "Replication & Consistency",         "level": "Level 3 — Distributed Systems"},
    {"index": 13, "topic": "Consistent Hashing",                "level": "Level 3 — Distributed Systems"},
    {"index": 14, "topic": "Distributed Transactions & Saga Pattern", "level": "Level 3 — Distributed Systems"},

    # Level 4 — Infrastructure & Reliability
    {"index": 15, "topic": "SQL Schema Design & Indexing",      "level": "Level 4 — Infrastructure & Reliability"},
    {"index": 16, "topic": "NoSQL Data Modelling",              "level": "Level 4 — Infrastructure & Reliability"},
    {"index": 17, "topic": "Blob Storage & Object Stores (S3)", "level": "Level 4 — Infrastructure & Reliability"},
    {"index": 18, "topic": "Content Delivery Networks (CDN)",   "level": "Level 4 — Infrastructure & Reliability"},
    {"index": 19, "topic": "Microservices vs Monolith",         "level": "Level 4 — Infrastructure & Reliability"},

    # Level 5 — Real Systems
    {"index": 20, "topic": "Design a URL Shortener",            "level": "Level 5 — Real Systems"},
    {"index": 21, "topic": "Design a Rate Limiter",             "level": "Level 5 — Real Systems"},
    {"index": 22, "topic": "Design a Notification System",      "level": "Level 5 — Real Systems"},
    {"index": 23, "topic": "Design Twitter Feed",               "level": "Level 5 — Real Systems"},
    {"index": 24, "topic": "Design Netflix",                    "level": "Level 5 — Real Systems"},
    {"index": 25, "topic": "Design WhatsApp / Chat System",     "level": "Level 5 — Real Systems"},
    {"index": 26, "topic": "Design Google Drive / Dropbox",     "level": "Level 5 — Real Systems"},
    {"index": 27, "topic": "Design a Search Autocomplete",      "level": "Level 5 — Real Systems"},
]

TOTAL_TOPICS = len(SYLLABUS)  # 28


# ── Tool 1 : Get Syllabus ──────────────────────────────────────────────────────

@tool
def tool_get_syllabus(topics_completed: list = []) -> dict:
    """
    Returns the full structured syllabus with completion status per topic.
    Use when the user asks what topics are available, wants an overview,
    or is just starting and needs to know where to begin.

    Args:
        topics_completed: List of topic names already finished (from state)
    """
    lines = ["📚 System Design Syllabus — 28 Topics\n"]
    current_level = ""
    for item in SYLLABUS:
        if item["level"] != current_level:
            current_level = item["level"]
            lines.append(f"\n🔷 {current_level}")
        done = "✅" if item["topic"] in topics_completed else "⬜"
        lines.append(f"  {done} {item['index'] + 1:>2}. {item['topic']}")

    completed = len(topics_completed)
    lines.append(f"\n📊 Progress: {completed}/{TOTAL_TOPICS} topics completed")
    lines.append("Tip: Start from Topic 1 and work your way up!")

    return {
        "display": "\n".join(lines),
        "total_topics": TOTAL_TOPICS,
        "completed_count": completed,
    }


# ── Tool 2 : Explain Concept ───────────────────────────────────────────────────

@tool
def tool_explain_concept(topic: str, current_topic_index: int) -> dict:
    """
    Returns a detailed beginner-friendly explanation prompt for a given topic.
    Also returns the topic name and index so the agent can update state.
    Use when the user wants to learn or understand a specific concept.

    Args:
        topic: The system design topic to explain (e.g. 'Caching', 'Load Balancing')
        current_topic_index: The 0-based index of this topic in the syllabus
    """
    previous_topics = [SYLLABUS[i]["topic"] for i in range(current_topic_index) if i < len(SYLLABUS)]
    prev_context = (
        f"The user has already covered: {', '.join(previous_topics)}."
        if previous_topics
        else "This is the user's first topic."
    )

    prompt = f"""
You are an expert system design tutor teaching a complete beginner.
{prev_context}

Now explain the topic: **{topic}**

Your explanation must follow this exact structure:

1. 🧠 What is it? (1-2 sentences, plain English, no jargon)
2. 🌍 Real-world analogy (relate it to something from everyday life)
3. ⚙️  How it works (step-by-step, beginner friendly, use bullet points)
4. 💡 Why it matters in system design (when and why engineers use it)
5. ✅ Key takeaways (3 bullet points the user must remember)
6. 🔗 How it connects to previously learned topics (if any)

Keep your tone friendly, encouraging, and conversational.
End with: "Ready to move on? Type 'practice' to test your knowledge or 'next topic' to continue!"
""".strip()

    return {
        "prompt": prompt,
        "topic_name": topic,
        "topic_index": current_topic_index,
    }


# ── Tool 3 : Suggest Next Topic ────────────────────────────────────────────────

@tool
def tool_suggest_next_topic(current_topic_index: int, topics_completed: list) -> dict:
    """
    Marks the current topic as completed and suggests the next one.
    Returns next topic info so the agent can update state correctly.
    Use when the user finishes a topic, says 'next topic', or asks what to study next.

    Args:
        current_topic_index: The 0-based index of the topic just completed
        topics_completed: List of topic names the user has already finished
    """
    current_topic_name = SYLLABUS[current_topic_index]["topic"] if current_topic_index < TOTAL_TOPICS else ""
    next_index = current_topic_index + 1

    if next_index >= TOTAL_TOPICS:
        return {
            "display": (
                "🎉 Congratulations! You've completed the entire syllabus!\n\n"
                "You're now ready for mock interviews. Type 'start interview' to begin! 🚀"
            ),
            "completed_topic": current_topic_name,
            "next_topic": None,
            "next_index": None,
            "syllabus_finished": True,
        }

    next_topic = SYLLABUS[next_index]
    completed_count = len(topics_completed) + 1  # +1 for the one just finished
    progress_bar = "█" * completed_count + "░" * (TOTAL_TOPICS - completed_count)

    return {
        "display": (
            f"✅ Great job completing **{current_topic_name}**!\n\n"
            f"📊 Progress: [{progress_bar}] {completed_count}/{TOTAL_TOPICS} topics done\n\n"
            f"👉 Next up: **{next_topic['topic']}** ({next_topic['level']})\n\n"
            f"Type 'learn {next_topic['topic']}' to start, or 'practice' to test what you know so far!"
        ),
        "completed_topic": current_topic_name,
        "next_topic": next_topic["topic"],
        "next_index": next_index,
        "syllabus_finished": False,
    }


# ── Helper: resolve topic by name or index ────────────────────────────────────

def get_topic_by_name(name: str) -> dict | None:
    """Find a syllabus entry by partial name match (case-insensitive)."""
    name_lower = name.lower()
    for item in SYLLABUS:
        if name_lower in item["topic"].lower():
            return item
    return None
