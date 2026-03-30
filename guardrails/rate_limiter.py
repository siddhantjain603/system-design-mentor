"""
rate_limiter.py
---------------
In-memory sliding-window rate limiter.
Tracks requests per session_id and per IP address.

Limits (configurable via .env):
  - Per session : 20 requests / 60 seconds  (prevents one user spamming)
  - Per IP      : 60 requests / 60 seconds  (prevents bot floods)

Usage in FastAPI:
    from guardrails.rate_limiter import RateLimiter
    limiter = RateLimiter()

    @app.post("/chat")
    async def chat(request: Request, body: ChatRequest):
        limiter.check(session_id=body.session_id, ip=request.client.host)
        ...
"""

import time
import os
from collections import defaultdict, deque
from fastapi import HTTPException


class RateLimiter:
    def __init__(
        self,
        session_limit: int = None,
        ip_limit: int = None,
        window_seconds: int = None,
    ):
        self.session_limit = session_limit or int(os.getenv("RATE_LIMIT_SESSION", 20))
        self.ip_limit = ip_limit or int(os.getenv("RATE_LIMIT_IP", 60))
        self.window = window_seconds or int(os.getenv("RATE_LIMIT_WINDOW", 60))

        # {session_id: deque of timestamps}
        self._session_windows: dict[str, deque] = defaultdict(deque)
        # {ip: deque of timestamps}
        self._ip_windows: dict[str, deque] = defaultdict(deque)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _purge_old(self, dq: deque, now: float) -> None:
        """Remove timestamps outside the sliding window."""
        while dq and now - dq[0] > self.window:
            dq.popleft()

    def _check_limit(self, dq: deque, limit: int, label: str, identifier: str) -> None:
        now = time.time()
        self._purge_old(dq, now)
        if len(dq) >= limit:
            retry_after = int(self.window - (now - dq[0])) + 1
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "rate_limit_exceeded",
                    "limit_type": label,
                    "identifier": identifier,
                    "limit": limit,
                    "window_seconds": self.window,
                    "retry_after_seconds": retry_after,
                    "message": (
                        f"Too many requests. You've hit the {label} limit "
                        f"({limit} requests per {self.window}s). "
                        f"Please wait {retry_after}s before trying again."
                    ),
                },
            )
        dq.append(now)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, session_id: str, ip: str = "unknown") -> None:
        """
        Call this at the top of every /chat request.
        Raises HTTP 429 if either limit is exceeded.
        """
        self._check_limit(self._session_windows[session_id], self.session_limit, "session", session_id)
        self._check_limit(self._ip_windows[ip], self.ip_limit, "ip", ip)

    def get_stats(self, session_id: str, ip: str = "unknown") -> dict:
        """Returns current usage stats — useful for /health or observability."""
        now = time.time()
        s_dq = self._session_windows[session_id]
        i_dq = self._ip_windows[ip]
        self._purge_old(s_dq, now)
        self._purge_old(i_dq, now)
        return {
            "session_id": session_id,
            "session_requests_in_window": len(s_dq),
            "session_limit": self.session_limit,
            "ip_requests_in_window": len(i_dq),
            "ip_limit": self.ip_limit,
            "window_seconds": self.window,
        }

    def reset(self, session_id: str) -> None:
        """Clear rate limit state for a session (call on /reset)."""
        self._session_windows.pop(session_id, None)
