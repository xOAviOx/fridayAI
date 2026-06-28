"""Weather skill — powered by wttr.in (no API key, always free).

Usage examples (voice):
    "What's the weather in Mumbai?"
    "How's the weather?"         ← auto-detects location via IP
    "Is it going to rain in London?"
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request

from friday.skills.registry import skill

log = logging.getLogger(__name__)

_TIMEOUT = 8  # seconds


@skill(description="Get current weather for a city. Leave location empty to auto-detect.")
def get_weather(location: str = "") -> str:
    """Fetch current weather conditions from wttr.in.

    Parameters
    ----------
    location:
        City name, e.g. ``"Mumbai"``, ``"New York"``, ``"London"``.
        Leave empty to auto-detect from your IP address.
    """
    loc = location.strip().replace(" ", "+")
    url = f"https://wttr.in/{loc}?format=j1" if loc else "https://wttr.in/?format=j1"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "FRIDAY-AI/1.0 (voice assistant; wttr.in JSON)"},
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"weather fetch failed: HTTP {exc.code}") from exc
    except Exception as exc:
        raise RuntimeError(f"weather fetch failed: {exc}") from exc

    try:
        current = data["current_condition"][0]
        desc = current["weatherDesc"][0]["value"]
        temp_c = current["temp_C"]
        temp_f = current["temp_F"]
        feels_c = current["FeelsLikeC"]
        humidity = current["humidity"]
        wind_kmph = current["windspeedKmph"]

        # Nearest area for the report header
        area_info = data.get("nearest_area", [{}])[0]
        city = (
            area_info.get("areaName", [{}])[0].get("value", "")
            or location.strip()
            or "your location"
        )

        return (
            f"{city}: {desc}, {temp_c}°C ({temp_f}°F), "
            f"feels like {feels_c}°C. "
            f"Humidity {humidity}%, wind {wind_kmph} km/h."
        )
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"unexpected wttr.in response shape: {exc}") from exc


__all__ = ["get_weather"]
