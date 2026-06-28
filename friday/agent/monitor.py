"""Proactive system monitor — FRIDAY speaks up without being asked.

Runs as a background daemon thread alongside the agent loop and fires
spoken alerts when system conditions cross configured thresholds:

  * Battery low (< ``battery_low_pct`` %)
  * Battery fully charged (≥ 100 % while plugged in)
  * CPU sustained high (> ``cpu_alert_pct`` % for ``cpu_sustained_s`` seconds)
  * RAM pressure (> ``ram_alert_pct`` %)
  * Disk almost full (> ``disk_alert_pct`` %)

Each alert type has an independent ``alert_cooldown_s`` so it won't
spam the user — the same condition fires at most once per cooldown window.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict
from typing import Callable

log = logging.getLogger(__name__)


class ProactiveMonitor:
    """Background daemon that watches system vitals and speaks alerts.

    Parameters
    ----------
    speak:
        Callable that takes a ``str`` and plays it via TTS.  Must be
        thread-safe (the loop's ``_speak_sync`` qualifies).
    cfg:
        The ``monitor`` section of :class:`~friday.config.MonitorConfig`.
    """

    def __init__(self, speak: Callable[[str], None], cfg: "MonitorConfig") -> None:  # type: ignore[name-defined]
        self._speak = speak
        self._cfg = cfg
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        # Unix timestamp of last fire per alert key.
        self._last_fired: dict[str, float] = defaultdict(float)
        # Rolling CPU high-water tracker: how many consecutive seconds above threshold.
        self._cpu_high_seconds: float = 0.0

    # ---------------------------------------------------------------------- #
    # Lifecycle                                                               #
    # ---------------------------------------------------------------------- #

    def start(self) -> None:
        if not self._cfg.enabled:
            log.info("proactive monitor disabled in config")
            return
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="friday-monitor"
        )
        self._thread.start()
        log.info(
            "proactive monitor started (interval=%ds, cooldown=%ds)",
            self._cfg.check_interval_s,
            self._cfg.alert_cooldown_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    # ---------------------------------------------------------------------- #
    # Main loop                                                               #
    # ---------------------------------------------------------------------- #

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=self._cfg.check_interval_s):
            try:
                self._check_all()
            except Exception:
                log.exception("monitor check failed")

    def _check_all(self) -> None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            log.warning("psutil not installed — proactive monitor inactive")
            self._stop_event.set()
            return

        self._check_battery(psutil)
        self._check_cpu(psutil)
        self._check_ram(psutil)
        self._check_disk(psutil)

    # ---------------------------------------------------------------------- #
    # Individual checks                                                       #
    # ---------------------------------------------------------------------- #

    def _check_battery(self, psutil) -> None:  # noqa: ANN001
        bat = psutil.sensors_battery()
        if bat is None:
            return  # desktop — no battery

        pct = bat.percent

        if not bat.power_plugged and pct <= self._cfg.battery_low_pct:
            self._alert(
                "battery_low",
                f"Heads up boss, battery's at {pct:.0f}%. "
                "Might wanna plug in soon.",
            )
        elif bat.power_plugged and pct >= 100:
            self._alert(
                "battery_full",
                "Battery's fully charged. You can unplug if you want.",
            )

    def _check_cpu(self, psutil) -> None:  # noqa: ANN001
        cpu = psutil.cpu_percent(interval=None)
        interval = self._cfg.check_interval_s

        if cpu >= self._cfg.cpu_alert_pct:
            self._cpu_high_seconds += interval
        else:
            self._cpu_high_seconds = 0.0

        if self._cpu_high_seconds >= self._cfg.cpu_sustained_s:
            # Find the top CPU consumer to name it.
            try:
                procs = sorted(
                    psutil.process_iter(["name", "cpu_percent"]),
                    key=lambda p: p.info.get("cpu_percent") or 0,
                    reverse=True,
                )
                top = next(
                    (p.info["name"] for p in procs
                     if (p.info.get("cpu_percent") or 0) > 5),
                    None,
                )
                culprit = f" Looks like {top} is the culprit." if top else ""
            except Exception:
                culprit = ""

            fired = self._alert(
                "cpu_high",
                f"Uh, CPU's been pegged at {cpu:.0f}% for a while.{culprit}",
            )
            if fired:
                self._cpu_high_seconds = 0.0  # reset after alerting

    def _check_ram(self, psutil) -> None:  # noqa: ANN001
        vm = psutil.virtual_memory()
        if vm.percent >= self._cfg.ram_alert_pct:
            free_gb = vm.available / 1_073_741_824
            self._alert(
                "ram_high",
                f"RAM's getting pretty full — only {free_gb:.1f} gigs free. "
                "You might wanna close some stuff.",
            )

    def _check_disk(self, psutil) -> None:  # noqa: ANN001
        import sys
        mount = "C:\\" if sys.platform == "win32" else "/"
        try:
            disk = psutil.disk_usage(mount)
        except Exception:
            return
        if disk.percent >= self._cfg.disk_alert_pct:
            free_gb = disk.free / 1_073_741_824
            self._alert(
                "disk_full",
                f"Heads up — disk's almost full. Only {free_gb:.1f} gigs left.",
            )

    # ---------------------------------------------------------------------- #
    # Alert helper                                                            #
    # ---------------------------------------------------------------------- #

    def _alert(self, key: str, message: str) -> bool:
        """Fire *message* via TTS if the cooldown for *key* has elapsed.

        Returns ``True`` if the alert was fired, ``False`` if suppressed.
        """
        now = time.time()
        if now - self._last_fired[key] < self._cfg.alert_cooldown_s:
            return False
        self._last_fired[key] = now
        log.info("proactive alert [%s]: %s", key, message)
        try:
            self._speak(message)
        except Exception:
            log.exception("proactive alert TTS failed for key=%s", key)
        return True


__all__ = ["ProactiveMonitor"]
