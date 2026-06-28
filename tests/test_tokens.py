"""Tests for ``friday.utils.tokens.BudgetTracker``.

The tracker injects its clock + sleeper so tests can walk the rolling
window deterministically without sleeping at all. Each test owns its
own fake-clock state — no shared fixtures and no real ``time``.
"""

from __future__ import annotations

import pytest

from friday.config import RateLimitConfig
from friday.utils.tokens import (
    BudgetExceededError,
    BudgetTracker,
    estimate_tokens,
)


class FakeClock:
    """Monotonic clock + sleeper that advances on demand."""

    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        # Sleeping in test = advancing the fake clock.
        self.t += seconds

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _tracker(
    *,
    rpm: int | None = None,
    tpm: int | None = None,
    rpd: int | None = None,
    warn_at_pct: int = 80,
) -> tuple[BudgetTracker, FakeClock]:
    clock = FakeClock()
    cfg = RateLimitConfig(
        requests_per_min=rpm,
        tokens_per_min=tpm,
        requests_per_day=rpd,
        warn_at_pct=warn_at_pct,
    )
    return BudgetTracker(cfg, name="test", now=clock.now, sleep=clock.sleep), clock


# --------------------------------------------------------------------------- #
# Basics                                                                      #
# --------------------------------------------------------------------------- #


def test_record_and_summary_track_realised_tokens() -> None:
    tracker, _ = _tracker(rpm=10, tpm=1000, rpd=100)
    tracker.before_request(estimated_tokens=100)
    tracker.record(tokens=120)

    summary = tracker.usage_summary()
    assert summary["rpm_used"] == 1
    assert summary["tpm_used"] == 120
    assert summary["rpd_used"] == 1


def test_pruning_drops_minute_events_after_60s() -> None:
    tracker, clock = _tracker(rpm=10, tpm=1000)
    tracker.record(tokens=100)
    clock.advance(61.0)
    summary = tracker.usage_summary()
    assert summary["rpm_used"] == 0
    assert summary["tpm_used"] == 0


def test_daily_window_survives_minute_pruning() -> None:
    tracker, clock = _tracker(rpm=10, rpd=100)
    tracker.record(tokens=10)
    clock.advance(61.0)
    # Minute window pruned but day window still has the event.
    summary = tracker.usage_summary()
    assert summary["rpm_used"] == 0
    assert summary["rpd_used"] == 1


# --------------------------------------------------------------------------- #
# RPM gating                                                                  #
# --------------------------------------------------------------------------- #


def test_before_request_blocks_when_rpm_full() -> None:
    tracker, clock = _tracker(rpm=2)
    tracker.record(tokens=0)
    tracker.record(tokens=0)
    # Third request should sleep until the oldest minute-event ages out.
    tracker.before_request(estimated_tokens=0)
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] > 0  # we did sleep
    assert clock.sleeps[0] <= 60.1


def test_rpm_unconfigured_means_no_gating() -> None:
    tracker, clock = _tracker(rpm=None, tpm=None)
    for _ in range(100):
        tracker.before_request(estimated_tokens=10)
        tracker.record(tokens=10)
    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# TPM gating                                                                  #
# --------------------------------------------------------------------------- #


def test_before_request_blocks_when_tpm_would_overflow() -> None:
    tracker, clock = _tracker(rpm=100, tpm=1000)
    tracker.record(tokens=900)
    tracker.before_request(estimated_tokens=200)
    assert len(clock.sleeps) == 1
    assert clock.sleeps[0] > 0


def test_tpm_within_budget_does_not_sleep() -> None:
    tracker, clock = _tracker(rpm=100, tpm=1000)
    tracker.record(tokens=500)
    tracker.before_request(estimated_tokens=200)
    assert clock.sleeps == []


# --------------------------------------------------------------------------- #
# RPD gating                                                                  #
# --------------------------------------------------------------------------- #


def test_rpd_exhausted_raises_budget_exceeded() -> None:
    tracker, _ = _tracker(rpd=2)
    tracker.record(tokens=0)
    tracker.record(tokens=0)
    with pytest.raises(BudgetExceededError, match="daily"):
        tracker.before_request()


# --------------------------------------------------------------------------- #
# 429 handling                                                                #
# --------------------------------------------------------------------------- #


def test_handle_429_sleeps_for_retry_after() -> None:
    tracker, clock = _tracker(rpm=10)
    tracker.handle_429(2.5)
    assert clock.sleeps == [2.5]


def test_handle_429_none_uses_short_default() -> None:
    tracker, clock = _tracker(rpm=10)
    tracker.handle_429(None)
    assert clock.sleeps == [1.0]


def test_handle_429_caps_long_waits() -> None:
    tracker, clock = _tracker(rpm=10)
    tracker.handle_429(600.0)
    assert clock.sleeps == [30.0]


# --------------------------------------------------------------------------- #
# estimate_tokens                                                             #
# --------------------------------------------------------------------------- #


def test_estimate_tokens_zero_for_empty() -> None:
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0


def test_estimate_tokens_roughly_chars_over_four() -> None:
    # The exact heuristic is len // 4 with a floor of 1.
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("a" * 40) == 10
    assert estimate_tokens("x") == 1  # floor
