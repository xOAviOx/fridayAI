"""Spotify skills.

Two capability tiers:

Tier 1 — no setup required:
    ``spotify_search`` opens the Spotify app / web player to search results
    via the ``spotify:`` URI scheme (macOS / Windows) or a web URL fallback.

Tier 2 — Spotify Developer app required:
    ``spotify_play_song`` and ``spotify_now_playing`` use the Web API via
    the ``spotipy`` library.  One-time setup:

    1. Go to https://developer.spotify.com/dashboard → Create app
    2. Set Redirect URI to:  http://localhost:8888/callback
    3. Copy Client ID and Client Secret into .env:
           SPOTIFY_CLIENT_ID=...
           SPOTIFY_CLIENT_SECRET=...
    4. First use opens a browser for OAuth → token cached in .spotify_cache
       (only once; subsequent calls reuse the cached token silently)

    Install the extra:  uv sync --extra spotify

Play / pause / next / previous are already handled by the existing
``media_control`` skill which sends OS-level media keys.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser
from urllib.parse import quote, quote_plus

from friday.skills.registry import skill

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPOTIFY_SCOPE = (
    "user-modify-playback-state "
    "user-read-playback-state "
    "user-read-currently-playing"
)
_REDIRECT_URI = "http://localhost:8888/callback"
_CACHE_PATH = ".spotify_cache"


def _spotipy_client():
    """Return an authenticated ``spotipy.Spotify`` client or raise clearly."""
    try:
        import spotipy  # type: ignore[import-not-found]
        from spotipy.oauth2 import SpotifyOAuth  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "This skill needs spotipy. Install the spotify extra:\n"
            "    uv sync --extra spotify"
        ) from exc

    client_id = os.environ.get("SPOTIFY_CLIENT_ID", "").strip()
    client_secret = os.environ.get("SPOTIFY_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        raise RuntimeError(
            "SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET must be set in .env.\n"
            "Get them from https://developer.spotify.com/dashboard"
        )

    auth = SpotifyOAuth(
        client_id=client_id,
        client_secret=client_secret,
        redirect_uri=_REDIRECT_URI,
        scope=_SPOTIFY_SCOPE,
        cache_path=_CACHE_PATH,
        open_browser=True,
    )
    return spotipy.Spotify(auth_manager=auth)


def _open_spotify_uri(uri: str) -> None:
    """Open a spotify: URI on macOS/Windows; fall back to web URL on Linux."""
    if sys.platform in ("darwin", "win32"):
        if sys.platform == "darwin":
            subprocess.Popen(["open", uri])
        else:
            subprocess.Popen(["cmd", "/c", "start", "", uri], shell=False)
    else:
        # Linux: try xdg-open, fall back to browser
        import shutil
        if shutil.which("xdg-open"):
            subprocess.Popen(["xdg-open", uri])
        else:
            webbrowser.open(uri)


# ---------------------------------------------------------------------------
# Tier 1 — no credentials needed
# ---------------------------------------------------------------------------


@skill(description="Open Spotify and search for a song, artist, or playlist.")
def spotify_search(query: str) -> str:
    """Open Spotify to search results for *query*.

    Works without any API credentials — uses the ``spotify:search:`` URI
    scheme on macOS / Windows and the Spotify web player on Linux.

    Parameters
    ----------
    query:
        Search terms, e.g. ``"Blinding Lights"``, ``"The Weeknd"``,
        ``"chill playlist"``.
    """
    q = query.strip()
    if not q:
        raise ValueError("spotify_search requires a non-empty query")

    uri = f"spotify:search:{quote(q)}"
    _open_spotify_uri(uri)
    return f"opened Spotify search for: {q}"


# ---------------------------------------------------------------------------
# Tier 2 — Spotify Web API (needs credentials + spotipy extra)
# ---------------------------------------------------------------------------


@skill(description="Search Spotify and immediately play the top matching track.")
def spotify_play_song(query: str) -> str:
    """Search for *query* via the Spotify API and play the first result.

    Falls back to opening Spotify search when API credentials aren't
    configured or no active Spotify device is found.

    Requires SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET in .env and
    the ``spotipy`` library (``uv sync --extra spotify``).

    Parameters
    ----------
    query:
        Song name, artist, or both — e.g. ``"Bohemian Rhapsody Queen"``.
    """
    q = query.strip()
    if not q:
        raise ValueError("spotify_play_song requires a non-empty query")

    try:
        sp = _spotipy_client()
    except RuntimeError:
        # Credentials not set — fall back to opening search gracefully.
        log.warning("Spotify API not configured, opening search instead")
        _open_spotify_uri(f"spotify:search:{quote(q)}")
        return f"Spotify API not configured — opened search for: {q}"

    try:
        import spotipy  # type: ignore[import-not-found]
    except ImportError:
        _open_spotify_uri(f"spotify:search:{quote(q)}")
        return f"spotipy not installed — opened search for: {q}"

    results = sp.search(q=q, type="track", limit=1)
    items = results.get("tracks", {}).get("items", [])
    if not items:
        return f"no Spotify track found for: {q}"

    track = items[0]
    uri = track["uri"]
    name = track["name"]
    artist = track["artists"][0]["name"]

    try:
        sp.start_playback(uris=[uri])
    except Exception as exc:  # SpotifyException is in the optional dep
        err = str(exc)
        if "No active device" in err or "404" in err:
            # Open Spotify desktop first, wait briefly, retry.
            _open_spotify_uri("spotify:")
            import time
            time.sleep(3)
            try:
                sp.start_playback(uris=[uri])
            except Exception:
                # Last resort: open the track URI directly.
                _open_spotify_uri(uri)
                return f"opened {name} by {artist} in Spotify"
        else:
            raise RuntimeError(f"Spotify playback error: {exc}") from exc

    return f"playing {name} by {artist}"


@skill(description="Return the currently playing Spotify track and artist.")
def spotify_now_playing() -> str:
    """Return what's currently playing in Spotify.

    Uses AppleScript on macOS (no credentials needed).  On other
    platforms, falls back to the Spotify Web API (needs credentials).
    """
    # macOS: AppleScript — fast, no API needed.
    if sys.platform == "darwin":
        script = """
        tell application "Spotify"
            if player state is playing then
                set t to name of current track
                set a to artist of current track
                return t & " by " & a
            else if player state is paused then
                set t to name of current track
                set a to artist of current track
                return "paused: " & t & " by " & a
            else
                return "Spotify is not playing"
            end if
        end tell
        """
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        log.warning("AppleScript failed: %s", result.stderr.strip())

    # Non-macOS (or AppleScript failed): use Web API.
    sp = _spotipy_client()
    current = sp.current_playback()
    if not current or not current.get("item"):
        return "nothing is playing on Spotify"

    item = current["item"]
    name = item["name"]
    artist = item["artists"][0]["name"]
    state = "playing" if current.get("is_playing") else "paused"
    return f"{state}: {name} by {artist}"


__all__ = [
    "spotify_now_playing",
    "spotify_play_song",
    "spotify_search",
]
