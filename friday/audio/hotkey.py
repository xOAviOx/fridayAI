"""Global push-to-talk + panic hotkey controller.

Built on ``pynput.keyboard.Listener`` directly rather than
``GlobalHotKeys`` because push-to-talk needs both press *and* release
events, not the single-shot activation that ``GlobalHotKeys`` fires.

Hotkey strings come straight from ``config.yaml`` (``hotkeys.push_to_talk``,
``hotkeys.panic``) in the form ``"ctrl+space"`` or ``"ctrl+shift+esc"``.
The parser is intentionally tiny — we don't need a full keysym table for
the only two combos we ship.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable

log = logging.getLogger(__name__)

# Tokens we recognise in the config-level hotkey strings. Anything else
# is treated as a literal character (``a``, ``space`` → space char, etc.)
# and matched against ``pynput.keyboard.KeyCode``.
_MODIFIER_NAMES = frozenset({"ctrl", "shift", "alt", "cmd", "win"})
_NAMED_KEY_ALIASES = {
    # Map user-friendly names to pynput Key attribute names.
    "esc": "esc",
    "escape": "esc",
    "space": "space",
    "tab": "tab",
    "enter": "enter",
    "return": "enter",
    "backspace": "backspace",
    "delete": "delete",
    "del": "delete",
    "home": "home",
    "end": "end",
    "pageup": "page_up",
    "pagedown": "page_down",
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
}


class HotkeyController:
    """Background listener that fires callbacks for PTT and panic combos.

    Parameters
    ----------
    ptt:
        Hotkey string for push-to-talk (e.g. ``"ctrl+space"``).
    panic:
        Hotkey string for the panic abort (e.g. ``"ctrl+shift+esc"``).
    on_ptt_press:
        Called on the audio-thread when the full PTT combo first becomes
        held. Should kick off recording. Must be cheap / non-blocking.
    on_ptt_release:
        Called when *any* key in the PTT combo is released. Should stop
        recording and hand the buffer off to STT. Must be cheap.
    on_panic:
        Called when the panic combo is pressed. Should cut TTS playback
        and abort any in-flight action.
    """

    def __init__(
        self,
        *,
        ptt: str,
        panic: str,
        on_ptt_press: Callable[[], None],
        on_ptt_release: Callable[[], None],
        on_panic: Callable[[], None],
    ) -> None:
        # Lazy import: keeps the module importable without pynput so
        # Phase 0 boot doesn't need the audio extras.
        try:
            from pynput import keyboard  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                "Hotkeys require pynput. Install the audio extras:\n"
                "    uv sync --extra audio\n"
                "or:\n"
                "    pip install -e '.[audio]'"
            ) from exc

        self._keyboard = keyboard
        self._ptt_keys = _parse_hotkey(ptt, keyboard)
        self._panic_keys = _parse_hotkey(panic, keyboard)
        self._on_ptt_press = on_ptt_press
        self._on_ptt_release = on_ptt_release
        self._on_panic = on_panic

        self._held: set[Any] = set()
        self._ptt_active = False
        # Reentrancy guard — pynput delivers events on its own thread and
        # we don't want a slow callback to block the listener.
        self._lock = threading.Lock()
        self._listener: Any = None
        log.debug("hotkeys configured: ptt=%s panic=%s", ptt, panic)

    def start(self) -> None:
        """Spin up the background listener thread."""
        if self._listener is not None:
            return
        self._listener = self._keyboard.Listener(
            on_press=self._handle_press,
            on_release=self._handle_release,
        )
        self._listener.start()
        log.debug("hotkey listener started")

    def stop(self) -> None:
        """Tear down the listener. Safe to call multiple times."""
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None
        log.debug("hotkey listener stopped")

    # ----- internal -----------------------------------------------------

    def _handle_press(self, key: Any) -> None:
        canonical = self._canonicalise(key)
        if canonical is None:
            return
        with self._lock:
            self._held.add(canonical)
            ptt_now = self._ptt_keys.issubset(self._held)
            panic_now = self._panic_keys.issubset(self._held)
            edge_ptt = ptt_now and not self._ptt_active
            if edge_ptt:
                self._ptt_active = True
        if panic_now:
            _safe_call(self._on_panic, "panic")
        elif edge_ptt:
            _safe_call(self._on_ptt_press, "ptt_press")

    def _handle_release(self, key: Any) -> None:
        canonical = self._canonicalise(key)
        if canonical is None:
            return
        with self._lock:
            self._held.discard(canonical)
            # PTT releases the moment *any* of its keys go up — feels
            # right for hold-to-talk: lifting any modifier ends the
            # utterance immediately.
            edge_ptt_release = self._ptt_active and not self._ptt_keys.issubset(
                self._held
            )
            if edge_ptt_release:
                self._ptt_active = False
        if edge_ptt_release:
            _safe_call(self._on_ptt_release, "ptt_release")

    def _canonicalise(self, key: Any) -> Any:
        """Normalise pynput's left/right modifier variants to one token.

        ``Key.ctrl_l`` and ``Key.ctrl_r`` should both satisfy ``ctrl`` in
        the combo. We collapse them onto the unsuffixed ``Key.ctrl``
        token used by ``_parse_hotkey``.
        """
        Key = self._keyboard.Key
        for base, variants in _MODIFIER_VARIANTS(Key).items():
            if key in variants:
                return base
        # Letter keys: normalise to lowercase char (KeyCode equality
        # already covers this, but be explicit).
        if isinstance(key, self._keyboard.KeyCode) and key.char is not None:
            return self._keyboard.KeyCode.from_char(key.char.lower())
        return key


# --------------------------------------------------------------------------- #
# Hotkey parsing                                                              #
# --------------------------------------------------------------------------- #


def _parse_hotkey(combo: str, keyboard: Any) -> set[Any]:
    """Parse a ``"ctrl+shift+esc"`` style string into a set of pynput keys."""
    if not combo or not combo.strip():
        raise ValueError("hotkey string cannot be empty")
    tokens = [tok.strip().lower() for tok in combo.split("+") if tok.strip()]
    if not tokens:
        raise ValueError(f"hotkey {combo!r} parsed to no tokens")

    keys: set[Any] = set()
    Key = keyboard.Key
    for tok in tokens:
        if tok in _MODIFIER_NAMES:
            keys.add(_modifier_to_key(tok, Key))
            continue
        if tok in _NAMED_KEY_ALIASES:
            attr = _NAMED_KEY_ALIASES[tok]
            keys.add(getattr(Key, attr))
            continue
        # F-keys: f1..f24
        if len(tok) > 1 and tok[0] == "f" and tok[1:].isdigit():
            keys.add(getattr(Key, tok))
            continue
        if len(tok) == 1:
            keys.add(keyboard.KeyCode.from_char(tok))
            continue
        raise ValueError(
            f"unrecognised hotkey token {tok!r} in {combo!r}. "
            f"Use modifiers (ctrl/shift/alt/cmd), named keys (space, esc, f1...), "
            f"or single characters."
        )
    return keys


def _modifier_to_key(name: str, Key: Any) -> Any:
    """Map a modifier name to the canonical unsuffixed pynput Key."""
    mapping = {
        "ctrl": Key.ctrl,
        "shift": Key.shift,
        "alt": Key.alt,
        "cmd": Key.cmd,
        "win": Key.cmd,  # ``win`` is a familiar alias for the Super/Cmd key
    }
    return mapping[name]


def _MODIFIER_VARIANTS(Key: Any) -> dict[Any, set[Any]]:
    """Group left/right modifier variants under their unsuffixed token.

    pynput exposes ``ctrl_l``/``ctrl_r``/``ctrl_gr`` etc.; users only
    care about ``ctrl``. Built as a function (not a module constant)
    because ``Key`` is only available after the lazy import.
    """
    return {
        Key.ctrl: {Key.ctrl, Key.ctrl_l, Key.ctrl_r},
        Key.shift: {Key.shift, Key.shift_l, Key.shift_r},
        Key.alt: {Key.alt, Key.alt_l, Key.alt_r, getattr(Key, "alt_gr", Key.alt)},
        Key.cmd: {Key.cmd, getattr(Key, "cmd_l", Key.cmd), getattr(Key, "cmd_r", Key.cmd)},
    }


def _safe_call(fn: Callable[[], None], label: str) -> None:
    """Run a callback on the listener thread without letting it crash the listener."""
    try:
        fn()
    except Exception:  # pragma: no cover - defensive
        log.exception("hotkey callback %s raised", label)


__all__ = ["HotkeyController"]
