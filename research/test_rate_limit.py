"""
test_rate_limit.py
------------------
Run from project root:
    python test_rate_limit.py

Sends 6 requests with the SAME session ID to trigger the rate limit (set to 5/10s in .env).
"""

import requests
import json

URL = "http://localhost:8000/chat"
SESSION_ID = "rate-limit-test-session-001"  # hardcoded — same every request
TOTAL_REQUESTS = 4

print(f"\n── Rate Limit Test ({TOTAL_REQUESTS} requests, limit=3/30s) ────────────────")
print(f"   Session ID: {SESSION_ID}\n")

for i in range(1, TOTAL_REQUESTS + 1):
    try:
        resp = requests.post(URL, json={
            "session_id": SESSION_ID,
            "message": "what is capital of india"
        }, timeout=30)

        if resp.status_code == 429:
            detail = resp.json().get("detail", {})
            retry = detail.get("retry_after_seconds", "?")
            print(f"  Request {i}: 🚫 BLOCKED 429 — rate limit hit! retry_after={retry}s")
        elif resp.status_code == 200:
            data = resp.json()
            used = data.get("rate_limit", {}).get("session_requests_used", "?")
            limit = data.get("rate_limit", {}).get("session_limit", "?")
            print(f"  Request {i}: ✅ OK — used {used} / {limit}")
        else:
            print(f"  Request {i}: ⚠️  Unexpected status {resp.status_code}: {resp.text[:100]}")

    except requests.exceptions.ConnectionError:
        print(f"  Request {i}: ❌ Connection refused — is the server running?")
        break
    except Exception as e:
        print(f"  Request {i}: ❌ Error — {e}")

print("\n── Done ─────────────────────────────────────────────────────────────\n")