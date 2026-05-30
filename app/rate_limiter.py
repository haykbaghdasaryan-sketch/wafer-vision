"""Session-based rate limiting for Streamlit app.

Implements a sliding window rate limiter that tracks request timestamps
per session. Designed to prevent resource exhaustion from automated tools.
"""

import time
from dataclasses import dataclass, field


@dataclass
class SessionRateLimiter:
    """Rate limiter using a sliding window of request timestamps.

    Tracks request timestamps and enforces a maximum number of requests
    within a rolling time window. Designed to work with Streamlit's
    session state when stored as a session attribute.

    Attributes:
        max_requests: Maximum allowed requests within the window.
        window_seconds: Duration of the sliding window in seconds.
        timestamps: List of request timestamps within the current window.
    """

    max_requests: int = 10
    window_seconds: int = 60
    timestamps: list[float] = field(default_factory=list)

    def _prune_expired(self, now: float) -> None:
        """Remove timestamps outside the current window."""
        cutoff = now - self.window_seconds
        self.timestamps = [t for t in self.timestamps if t > cutoff]

    def check(self, now: float | None = None) -> bool:
        """Check if a request is allowed under the rate limit.

        Args:
            now: Current time as a float (seconds since epoch).
                 Defaults to time.time() if not provided.

        Returns:
            True if the request is allowed, False if rate-limited.
        """
        if now is None:
            now = time.time()
        self._prune_expired(now)
        if len(self.timestamps) >= self.max_requests:
            return False
        self.timestamps.append(now)
        return True

    def wait_seconds(self, now: float | None = None) -> float:
        """Calculate how many seconds until the next request is allowed.

        Args:
            now: Current time as a float (seconds since epoch).
                 Defaults to time.time() if not provided.

        Returns:
            Seconds to wait. Returns 0.0 if a request is currently allowed.
        """
        if now is None:
            now = time.time()
        self._prune_expired(now)
        if len(self.timestamps) < self.max_requests:
            return 0.0
        # Oldest timestamp in window determines when it expires
        oldest = min(self.timestamps)
        wait = (oldest + self.window_seconds) - now
        return max(0.0, wait)

    def wait_message(self, now: float | None = None) -> str:
        """Return a human-readable message about when to retry.

        Args:
            now: Current time as a float (seconds since epoch).
                 Defaults to time.time() if not provided.

        Returns:
            A user-friendly message like "Try again in 12 seconds."
            or empty string if not rate-limited.
        """
        wait = self.wait_seconds(now)
        if wait <= 0:
            return ""
        seconds = int(wait) + 1  # Round up for user-friendliness
        if seconds == 1:
            return "Rate limit reached. Try again in 1 second."
        return f"Rate limit reached. Try again in {seconds} seconds."

    @property
    def remaining_requests(self) -> int:
        """Number of requests remaining in the current window."""
        self._prune_expired(time.time())
        return max(0, self.max_requests - len(self.timestamps))
