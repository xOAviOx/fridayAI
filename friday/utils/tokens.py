"""Free-tier budget tracking for cloud LLM / STT providers.

Phase 1's brief calls out three concrete bottlenecks on the Groq free
tier and a hard rule about how to handle them:

* ``requests_per_min`` — counted strictly. A new request blocks until
  the rolling window has room.
* ``tokens_per_min`` — **the actual bottleneck**. Same rolling-window
  treatment as RPM, plus a warn-at-pct trigger.
* ``requests_per_day`` — daily cap. Sleeping doesn't help here; we
  raise :class:`BudgetExceededError` so the caller can surface "out of
  budget for today" cleanly.
* 429 ``retry-after`` — honored verbatim through :meth:`handle_429`,
  not estimated.

The tracker is provider-agnostic — Groq today, Gemini tomorrow. The
config block driving it is the existing :class:`RateLimitConfig` from
``friday.config``. The clock and sleeper are injectable so tests can
walk the rolling window deterministically without actually sleeping.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, Callable

from friday.config import RateLimitConfig

log = logging.getLogger(__name__)

# Tiny extra cushion after we wake up from a rolling-window sleep — the
# server's idea of "60 seconds" doesn't always agree with ours to the
# millisecond, and waking up 100 ms early just produces a 429.
_WAKEUP_SLACK_S = 0.1


class BudgetExceededError(RuntimeError):
    """Raised when a budget has run out and waiting won't recover it.

    Currently only the daily-requests cap behaves this way. RPM and TPM
    exhaustion translate into a sleep, not an exception.
    """


class BudgetTracker:
    """Track per-provider RPM / TPM / RPD against a :class:`RateLimitConfig`.

    Construction is cheap — one per provider per process. The tracker
    is not thread-safe; the Phase 1 agent loop is synchronous so this
    is fine. If we ever go multi-threaded, swap the deques for a lock-
    protected counter.

    Parameters
    ----------
    config:
        Validated rate-limit budget. Any field left at ``None`` means
        "no cap" — handy for providers we don't bother gating.
    name:
        Provider label used in log lines (``"groq"`` etc.).
    now / sleep:
        Injection points for tests. Defaults to ``time.monotonic`` and
        ``time.sleep`` so production paths don't pay a function-call tax.
    """

    def __init__(
        self,
        config: RateLimitConfig,
        name: str = "",
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._name = name or "budget"
        self._now = now
        self._sleep = sleep
        # Each minute event records (timestamp, tokens). Empty token slot
        # for requests we don't have a token count for yet.
        self._minute: deque[tuple[float, int]] = deque()
        # Daily events are just timestamps — we cap on count, not tokens.
        self._day: deque[float] = deque()

    # ----- pre-flight ------------------------------------------------------

    def before_request(self, *, estimated_tokens: int = 0) -> None:
        """Reserve room for one request. Block (or raise) if needed.

        ``estimated_tokens`` is a cheap upper bound — we only use it to
        decide whether the TPM budget can fit this call without
        sleeping. The realised token count gets recorded via
        :meth:`record` after the call returns.
        """
        now = self._now()
        self._prune(now)

        # Hard fail on RPD — no amount of sleeping clears the daily cap.
        rpd = self._config.requests_per_day
        if rpd is not None and len(self._day) >= rpd:
            raise BudgetExceededError(
                f"{self._name}: daily request budget ({rpd}) exhausted"
            )

        # RPM — wait for the oldest minute-event to age out.
        rpm = self._config.requests_per_min
        if rpm is not None and len(self._minute) >= rpm:
            wait = 60.0 - (now - self._minute[0][0])
            if wait > 0:
                log.warning(
                    "%s: RPM exhausted (%d/%d), sleeping %.1fs",
                    self._name,
                    len(self._minute),
                    rpm,
                    wait,
                )
                self._sleep(wait + _WAKEUP_SLACK_S)
                now = self._now()
                self._prune(now)

        # TPM — the actual free-tier bottleneck. Sleep until enough
        # token-events age out that this call fits.
        tpm = self._config.tokens_per_min
        if tpm is not None and estimated_tokens > 0:
            current = sum(t for _, t in self._minute)
            if current + estimated_tokens > tpm:
                overshoot = current + estimated_tokens - tpm
                cumulative = 0
                wait = 0.0
                for ts, tokens in self._minute:
                    cumulative += tokens
                    if cumulative >= overshoot:
                        wait = 60.0 - (now - ts)
                        break
                if wait > 0:
                    log.warning(
                        "%s: TPM exhausted (%d+%d > %d), sleeping %.1fs",
                        self._name,
                        current,
                        estimated_tokens,
                        tpm,
                        wait,
                    )
                    self._sleep(wait + _WAKEUP_SLACK_S)
                    now = self._now()
                    self._prune(now)

        self._warn_if_over_threshold(estimated_tokens)

    # ----- post-flight -----------------------------------------------------

    def record(self, *, tokens: int) -> None:
        """Log a completed request's realised token count."""
        ts = self._now()
        self._minute.append((ts, max(0, tokens)))
        self._day.append(ts)
        self._prune(ts)

    def handle_429(self, retry_after_s: float | None) -> None:
        """Sleep for the server-supplied ``retry-after`` value.

        ``None`` is treated as a short conservative pause — better than
        re-firing immediately. Capped at 30s to avoid pathological
        waits during outages; the agent loop will surface anything
        longer to the user as a transient error.
        """
        delay = 1.0 if retry_after_s is None else max(0.0, retry_after_s)
        delay = min(delay, 30.0)
        log.warning(
            "%s: rate-limited by server (429); sleeping %.1fs",
            self._name,
            delay,
        )
        self._sleep(delay)

    # ----- inspection ------------------------------------------------------

    def usage_summary(self) -> dict[str, Any]:
        """Snapshot of current usage. Suitable for HUD / logging."""
        self._prune(self._now())
        rpm = self._config.requests_per_min
        tpm = self._config.tokens_per_min
        rpd = self._config.requests_per_day
        used_tokens = sum(t for _, t in self._minute)
        return {
            "rpm_used": len(self._minute),
            "rpm_limit": rpm,
            "tpm_used": used_tokens,
            "tpm_limit": tpm,
            "rpd_used": len(self._day),
            "rpd_limit": rpd,
        }

    # ----- internal --------------------------------------------------------

    def _prune(self, now: float) -> None:
        while self._minute and (now - self._minute[0][0]) > 60.0:
            self._minute.popleft()
        # Daily window is rolling 24h. Using monotonic time means the
        # window survives clock changes / DST.
        while self._day and (now - self._day[0]) > 86_400.0:
            self._day.popleft()

    def _warn_if_over_threshold(self, estimated_tokens: int) -> None:
        threshold = self._config.warn_at_pct / 100.0
        rpm = self._config.requests_per_min
        tpm = self._config.tokens_per_min
        rpd = self._config.requests_per_day

        if rpm is not None:
            after = (len(self._minute) + 1) / rpm
            if after >= threshold:
                log.warning(
                    "%s: RPM at %.0f%% of budget (%d/%d)",
                    self._name,
                    after * 100,
                    len(self._minute) + 1,
                    rpm,
                )
        if tpm is not None and estimated_tokens > 0:
            after = (sum(t for _, t in self._minute) + estimated_tokens) / tpm
            if after >= threshold:
                log.warning(
                    "%s: TPM at %.0f%% of budget (~%d/%d)",
                    self._name,
                    after * 100,
                    sum(t for _, t in self._minute) + estimated_tokens,
                    tpm,
                )
        if rpd is not None:
            after = (len(self._day) + 1) / rpd
            if after >= threshold:
                log.warning(
                    "%s: RPD at %.0f%% of budget (%d/%d)",
                    self._name,
                    after * 100,
                    len(self._day) + 1,
                    rpd,
                )


# --------------------------------------------------------------------------- #
# Token estimation                                                            #
# --------------------------------------------------------------------------- #


def estimate_tokens(text: str | None) -> int:
    """Cheap upper-bound token estimate.

    The 1-token ≈ 4-characters heuristic the OpenAI tokenizer cheatsheet
    quotes is close enough for budget gating — the realised count
    comes back from the server post-call and overrides this anyway.
    """
    if not text:
        return 0
    return max(1, len(text) // 4)


__all__ = [
    "BudgetExceededError",
    "BudgetTracker",
    "estimate_tokens",
]
