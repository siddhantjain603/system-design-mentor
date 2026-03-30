from langchain_core.tools import tool

# ── Question Bank ──────────────────────────────────────────────────────────────

QUESTION_BANK = {
    "Client-Server Model & HTTP": [
        "What happens step by step when you type google.com in your browser?",
        "What is the difference between HTTP and HTTPS?",
        "Explain the difference between a GET and a POST request.",
    ],
    "APIs — REST, GraphQL, gRPC": [
        "What makes an API RESTful? Name at least 3 constraints.",
        "When would you choose GraphQL over REST?",
        "What is gRPC and what problem does it solve compared to REST?",
    ],
    "Databases — SQL vs NoSQL": [
        "When would you choose a NoSQL database over a relational database?",
        "What is ACID compliance and why does it matter?",
        "Explain the difference between vertical and horizontal scaling for databases.",
    ],
    "Caching (Redis, CDN)": [
        "What is a cache eviction policy? Name two common ones and explain them.",
        "What is a CDN and how does it reduce latency?",
        "Explain cache invalidation. Why is it considered a hard problem?",
    ],
    "Load Balancing": [
        "What is the difference between Layer 4 and Layer 7 load balancing?",
        "Explain the round-robin load balancing algorithm.",
        "What is a sticky session and when would you use it?",
    ],
    "Message Queues (Kafka, RabbitMQ)": [
        "What problem do message queues solve in distributed systems?",
        "What is the difference between a queue and a topic in messaging systems?",
        "Explain at-least-once vs exactly-once delivery semantics.",
    ],
    "CAP Theorem": [
        "State the CAP theorem in your own words.",
        "Give an example of a CP system and an AP system.",
        "Why can't a distributed system be both consistent and available during a partition?",
    ],
    "Sharding & Partitioning": [
        "What is database sharding and why is it used?",
        "What is a hotspot problem in sharding and how do you avoid it?",
        "Compare range-based sharding vs hash-based sharding.",
    ],
    "Replication & Consistency": [
        "What is the difference between synchronous and asynchronous replication?",
        "Explain eventual consistency with a real-world example.",
        "What is a leader-follower replication model?",
    ],
    "Design a URL Shortener": [
        "What are the core components you need to design a URL shortener like bit.ly?",
        "How would you generate a unique short code for each URL?",
        "How would you handle 100 million URL redirects per day?",
    ],
    "Design Twitter Feed": [
        "What is a fanout problem in a social media feed and how do you solve it?",
        "How would you store and serve tweets for 500 million users?",
        "How do you handle feed ranking — should it be push or pull based?",
    ],
    "Design Netflix": [
        "How would you design a video streaming service that handles 4K content?",
        "What role does a CDN play in a service like Netflix?",
        "How would you design the recommendation engine at a high level?",
    ],
}


# ── Tool 1 : Ask Practice Question ────────────────────────────────────────────

@tool
def tool_ask_practice_question(topic: str, question_index: int = 0) -> str:
    """
    Retrieves a practice question for a given topic.
    Use this when the user wants to test their knowledge on a topic they have learned.

    Args:
        topic: The system design topic to generate a question for
        question_index: Which question to pick from the bank (0, 1, or 2)
    """
    questions = QUESTION_BANK.get(topic)

    if not questions:
        available = ", ".join(QUESTION_BANK.keys())
        return f"❌ No questions found for '{topic}'.\nAvailable topics: {available}"

    idx = question_index % len(questions)
    question = questions[idx]

    return (
        f"🧪 Practice Question ({topic})\n\n"
        f"**{question}**\n\n"
        f"Take your time and answer in your own words.\n"
        f"Type 'hint' if you're stuck, or just submit your answer!"
    )


# ── Tool 2 : Give Hint ─────────────────────────────────────────────────────────

@tool
def tool_give_hint(topic: str, question: str, hint_number: int = 1) -> str:
    """
    Provides a progressive hint for a practice question without giving away the answer.
    Use this when the user says 'hint', 'I'm stuck', or asks for help mid-question.

    Args:
        topic: The topic the question belongs to
        question: The exact question being asked
        hint_number: Which hint to give (1 = gentle nudge, 2 = stronger hint, 3 = near-answer)
    """
    hint_level = {
        1: "a gentle nudge — point them in the right direction without revealing anything",
        2: "a stronger hint — mention a key concept or component they should think about",
        3: "a near-complete hint — give the structure of the answer without the full details",
    }.get(hint_number, "a gentle nudge")

    return f"""
You are a system design tutor helping a beginner who is stuck.

Topic: {topic}
Question: {question}
Hint level requested: {hint_number}/3 — give {hint_level}

Rules:
- Do NOT give the full answer
- Be encouraging and supportive
- End with a follow-up nudge like "Does that help? Give it another try!"
- If hint_number is 3, add: "This is your last hint — give it your best shot now! 💪"
""".strip()


# ── Tool 3 : Evaluate Answer ───────────────────────────────────────────────────

@tool
def tool_evaluate_answer(topic: str, question: str, user_answer: str) -> str:
    """
    Evaluates the user's answer to a practice question and returns a score with feedback.
    Use this when the user submits an answer to a practice question.

    Args:
        topic: The topic the question belongs to
        question: The practice question that was asked
        user_answer: The answer the user provided
    """
    return f"""
You are a strict but encouraging system design evaluator.

Topic: {topic}
Question: {question}
User's Answer: {user_answer}

Evaluate the answer using this exact format:

📊 **Score: X/10**

✅ **What you got right:**
- (list correct points)

❌ **What was missing or incorrect:**
- (list gaps or mistakes)

💡 **Model Answer:**
(provide a clear, complete answer a senior engineer would give)

🎯 **Key concept to remember:**
(one sentence takeaway)

Scoring guide:
- 9-10: Complete, accurate, includes edge cases
- 7-8: Mostly correct, missing minor details
- 5-6: Core idea right but missing important parts
- 3-4: Partial understanding, significant gaps
- 1-2: Major misconceptions present
- 0: Off-topic or no attempt

End with an encouraging message and suggest: "Type 'next question' for another or 'next topic' to continue learning!"
""".strip()
