"""Phase 6 — Screen vision skills.

Two skills backed by the Groq Llama 4 Scout vision model:

``read_screen``
    Takes a screenshot of the current display and answers a question
    about it. Useful for: "What's on my screen?", "Summarize this
    article", "What does this error say?", "Read me this text."

``analyze_image``
    Loads an image file from disk and answers a question about it.
    Useful for: "What's in this screenshot I saved?", "Read the text
    in ~/Downloads/invoice.png", "Describe this chart."

Images are resized to at most 1 280 px on the long edge before sending
so we stay within Groq's per-image token budget on free-tier accounts.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from friday.skills.registry import skill

_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
_MAX_PX = 1280   # longest-edge cap before base64-encoding


# --------------------------------------------------------------------------- #
# Image helpers                                                                #
# --------------------------------------------------------------------------- #


def _resize_png(src: str, dst: str, max_px: int = _MAX_PX) -> None:
    """Resize a PNG so its longest edge is at most *max_px*.

    Uses ``sips`` (macOS built-in) when available, then ImageMagick
    ``convert``, then skips silently — the raw image still works, just
    uses more tokens.
    """
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sips", "-Z", str(max_px), src, "--out", dst],
            capture_output=True,
        )
        if result.returncode == 0:
            return

    try:
        subprocess.run(
            ["convert", src, "-resize", f"{max_px}x{max_px}>", dst],
            capture_output=True,
            check=True,
        )
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # Fallback: copy as-is (still works, just more tokens)
    import shutil
    shutil.copy2(src, dst)


def _screenshot_base64() -> tuple[str, str]:
    """Capture the current display and return (base64_png, mime_type)."""
    with tempfile.TemporaryDirectory() as td:
        raw = os.path.join(td, "raw.png")
        small = os.path.join(td, "screen.png")

        if sys.platform == "darwin":
            subprocess.run(["screencapture", "-x", raw], check=True)
        else:
            # Linux: try scrot then import (ImageMagick)
            try:
                subprocess.run(["scrot", raw], check=True)
            except FileNotFoundError:
                subprocess.run(["import", "-window", "root", raw], check=True)

        _resize_png(raw, small)
        data = Path(small).read_bytes()

    return base64.b64encode(data).decode(), "image/png"


def _file_base64(path: str) -> tuple[str, str]:
    """Load an image file and return (base64, mime_type), resizing if PNG."""
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")

    suffix = p.suffix.lower()
    mime_map = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    mime = mime_map.get(suffix, "image/png")

    if suffix == ".png":
        with tempfile.TemporaryDirectory() as td:
            small = os.path.join(td, "img.png")
            _resize_png(str(p), small)
            data = Path(small).read_bytes()
    else:
        data = p.read_bytes()

    return base64.b64encode(data).decode(), mime


# --------------------------------------------------------------------------- #
# Vision call                                                                  #
# --------------------------------------------------------------------------- #


def _ask_vision(image_b64: str, mime: str, question: str) -> str:
    """Send an image + question to the Groq vision model."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set — vision requires the same key as the LLM."
        )

    from openai import OpenAI  # already installed

    client = OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL)
    resp = client.chat.completions.create(
        model=_VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                    },
                    {"type": "text", "text": question},
                ],
            }
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content or "(no response)"


# --------------------------------------------------------------------------- #
# Skills                                                                       #
# --------------------------------------------------------------------------- #


@skill(
    description=(
        "Take a screenshot and answer a question about what's on screen. "
        "Great for: 'What's on my screen?', 'Read me this error', "
        "'Summarize this article', 'What does this say?'."
    )
)
def read_screen(
    question: str = "What's on my screen? Describe it briefly.",
) -> str:
    """Screenshot the display and run a vision query against it."""
    b64, mime = _screenshot_base64()
    return _ask_vision(b64, mime, question)


@skill(
    description=(
        "Analyze an image file and answer a question about it. "
        "Pass the full file path (e.g. ~/Desktop/chart.png) and "
        "an optional question about what you want to know."
    )
)
def analyze_image(
    path: str,
    question: str = "Describe this image.",
) -> str:
    """Load an image file and run a vision query against it."""
    b64, mime = _file_base64(path)
    return _ask_vision(b64, mime, question)


__all__ = ["read_screen", "analyze_image"]
