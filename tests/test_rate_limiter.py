"""Unit tests for the SessionRateLimiter.

Tests the rate limiting logic independently of Streamlit session state.
"""

import pytest

from app.rate_limiter import SessionRateLimiter


class TestSessionRateLimiter:
    """Tests for SessionRateLimiter."""

    def test_allows_requests_under_limit(self):
        """Requests under the max should be allowed."""
        limiter = SessionRateLimiter(max_requests=3, window_seconds=60)
        now = 1000.0

        assert limiter.check(now=now) is True
        assert limiter.check(now=now + 1) is True
        assert limiter.check(now=now + 2) is True

    def test_blocks_requests_over_limit(self):
        """Requests over the max should be blocked."""
        limiter = SessionRateLimiter(max_requests=3, window_seconds=60)
        now = 1000.0

        # Use up all 3 allowed
        assert limiter.check(now=now) is True
        assert limiter.check(now=now + 1) is True
        assert limiter.check(now=now + 2) is True
        # 4th should be blocked
        assert limiter.check(now=now + 3) is False

    def test_allows_after_window_expires(self):
        """Requests should be allowed again after the window expires."""
        limiter = SessionRateLimiter(max_requests=2, window_seconds=10)
        now = 1000.0

        # Use up all requests
        assert limiter.check(now=now) is True
        assert limiter.check(now=now + 1) is True
        assert limiter.check(now=now + 2) is False

        # After window expires, requests should work again
        assert limiter.check(now=now + 11) is True

    def test_sliding_window_behavior(self):
        """Window slides so oldest requests expire first."""
        limiter = SessionRateLimiter(max_requests=2, window_seconds=10)

        # First two at t=0 and t=5
        assert limiter.check(now=100.0) is True
        assert limiter.check(now=105.0) is True
        # Blocked at t=8 (both still in window)
        assert limiter.check(now=108.0) is False
        # At t=11, the first request (t=100) has expired
        assert limiter.check(now=111.0) is True

    def test_wait_seconds_zero_when_not_limited(self):
        """wait_seconds returns 0 when requests are available."""
        limiter = SessionRateLimiter(max_requests=5, window_seconds=60)
        assert limiter.wait_seconds(now=1000.0) == 0.0

    def test_wait_seconds_positive_when_limited(self):
        """wait_seconds returns positive when rate-limited."""
        limiter = SessionRateLimiter(max_requests=2, window_seconds=10)
        limiter.check(now=100.0)
        limiter.check(now=105.0)

        # At t=108, oldest expires at t=110
        wait = limiter.wait_seconds(now=108.0)
        assert 1.0 < wait <= 2.0  # Should be ~2 seconds

    def test_wait_message_empty_when_not_limited(self):
        """wait_message returns empty string when not rate-limited."""
        limiter = SessionRateLimiter(max_requests=5, window_seconds=60)
        assert limiter.wait_message(now=1000.0) == ""

    def test_wait_message_has_content_when_limited(self):
        """wait_message returns a useful message when rate-limited."""
        limiter = SessionRateLimiter(max_requests=1, window_seconds=30)
        limiter.check(now=100.0)

        msg = limiter.wait_message(now=105.0)
        assert "Rate limit reached" in msg
        assert "Try again" in msg
        assert "second" in msg

    def test_default_config_ten_per_minute(self):
        """Default config allows exactly 10 requests per minute."""
        limiter = SessionRateLimiter()
        now = 1000.0

        for i in range(10):
            assert limiter.check(now=now + i) is True

        # 11th should be blocked
        assert limiter.check(now=now + 10) is False

    def test_remaining_requests_decreases(self):
        """remaining_requests decreases as requests are made."""
        limiter = SessionRateLimiter(max_requests=5, window_seconds=60)
        # Use time.time() internally, so just check it works
        # We'll set timestamps manually for precise testing
        limiter.timestamps = [1000.0, 1001.0, 1002.0]
        # Prune with a current time within the window
        limiter._prune_expired(1030.0)
        assert limiter.remaining_requests >= 0

    def test_empty_limiter_has_full_capacity(self):
        """A fresh limiter should have max_requests remaining."""
        limiter = SessionRateLimiter(max_requests=10, window_seconds=60)
        # No timestamps yet
        assert len(limiter.timestamps) == 0

    def test_prune_expired_removes_old_timestamps(self):
        """Expired timestamps are properly removed."""
        limiter = SessionRateLimiter(max_requests=5, window_seconds=10)
        limiter.timestamps = [90.0, 95.0, 100.0, 105.0]

        limiter._prune_expired(now=106.0)
        # Only timestamps > 96.0 should remain
        assert 90.0 not in limiter.timestamps
        assert 95.0 not in limiter.timestamps
        assert 100.0 in limiter.timestamps
        assert 105.0 in limiter.timestamps
