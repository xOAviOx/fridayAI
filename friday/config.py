"""Configuration loader.

Two sources, merged into one validated ``Config`` object:

* ``config.yaml`` — behavior: which provider for which role, models,
  hotkeys, safety flags, allowlists, rate-limit budgets.
* ``.env`` — secrets only: API keys.

Validation rules worth knowing:

* The ``providers`` block picks one implementation per role
  (stt / llm / tts). Only the *selected* provider needs an API key —
  e.g. if ``providers.llm = groq`` we don't care that
  ``ANTHROPIC_API_KEY`` is blank.
* Anthropic is opt-in even when its key is present: ``llm.anthropic.enabled``
  must also be true. API billing is separate from any Claude Pro plan
  and we don't want a surprise bill.
* All errors raise :class:`ConfigError` with a message that names the
  offending field. ``main.py`` prints those before logging is set up.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

STTName = Literal["groq", "local_whisper"]
LLMName = Literal["groq", "gemini", "anthropic"]
TTSName = Literal["elevenlabs", "kokoro"]


class ConfigError(Exception):
    """Raised when config can't be loaded or fails validation."""


# --------------------------------------------------------------------------- #
# Per-provider settings                                                       #
# --------------------------------------------------------------------------- #


class _StrictModel(BaseModel):
    # Catch typos in config.yaml early instead of silently ignoring them.
    model_config = ConfigDict(extra="forbid")


class GroqLLMConfig(_StrictModel):
    model: str = "llama-3.3-70b-versatile"
    base_url: str = "https://api.groq.com/openai/v1"


class GeminiLLMConfig(_StrictModel):
    model: str = "gemini-2.5-flash"


class AnthropicLLMConfig(_StrictModel):
    model: str = "claude-opus-4-8"
    # Anthropic API billing is separate from Pro. Opt-in only.
    enabled: bool = False


class LLMSection(_StrictModel):
    groq: GroqLLMConfig = Field(default_factory=GroqLLMConfig)
    gemini: GeminiLLMConfig = Field(default_factory=GeminiLLMConfig)
    anthropic: AnthropicLLMConfig = Field(default_factory=AnthropicLLMConfig)


class GroqSTTConfig(_StrictModel):
    model: str = "whisper-large-v3-turbo"


class STTSection(_StrictModel):
    groq: GroqSTTConfig = Field(default_factory=GroqSTTConfig)


class ElevenLabsConfig(_StrictModel):
    voice_id: str = ""
    model: str = "eleven_turbo_v2_5"


class KokoroConfig(_StrictModel):
    # Voice ids come from the Kokoro model card. ``af_heart`` is the
    # upstream default and a safe American English pick.
    voice: str = "af_heart"
    # Single-character Kokoro language code — 'a' = American English,
    # 'b' = British, etc. Must match the voice family.
    lang_code: str = "a"
    speed: float = 1.0

    @field_validator("speed")
    @classmethod
    def _positive_speed(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("tts.kokoro.speed must be > 0")
        return v


class TTSSection(_StrictModel):
    elevenlabs: ElevenLabsConfig = Field(default_factory=ElevenLabsConfig)
    kokoro: KokoroConfig = Field(default_factory=KokoroConfig)


# --------------------------------------------------------------------------- #
# Top-level sections                                                          #
# --------------------------------------------------------------------------- #


class ProviderSelection(_StrictModel):
    stt: STTName = "groq"
    llm: LLMName = "groq"
    # Default to the local, free path so a fresh checkout boots at $0.
    # ElevenLabs is still available as an opt-in cloud alternative.
    tts: TTSName = "kokoro"


class AudioConfig(_StrictModel):
    sample_rate: int = 16000
    channels: int = 1
    # Phase 2+: voice-activity detection. When True, recording stops
    # automatically after silence rather than requiring PTT release.
    # The VAD implementation is plumbed in config here; the wiring to a
    # VAD library lands in a subsequent Phase 2 chunk.
    vad_enabled: bool = False

    @field_validator("sample_rate")
    @classmethod
    def _positive_rate(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("audio.sample_rate must be positive")
        return v


class HotkeysConfig(_StrictModel):
    push_to_talk: str = "ctrl+space"
    panic: str = "ctrl+shift+esc"


class SafetyConfig(_StrictModel):
    dry_run: bool = True
    enable_code_exec: bool = False
    enable_computer_use: bool = False
    shell_allowlist: list[str] = Field(default_factory=list)
    app_allowlist: list[str] = Field(default_factory=list)
    # When True, skills marked destructive=True are allowed to run
    # without a confirmation step.  False (the default) blocks them so
    # no accidental write/delete happens until you've opted in.
    # Only meaningful when dry_run is False.
    skip_confirm_destructive: bool = False


class RateLimitConfig(_StrictModel):
    requests_per_min: int | None = None
    tokens_per_min: int | None = None
    requests_per_day: int | None = None
    warn_at_pct: int = 80

    @field_validator("warn_at_pct")
    @classmethod
    def _pct_range(cls, v: int) -> int:
        if not 0 < v <= 100:
            raise ValueError("warn_at_pct must be in (0, 100]")
        return v


class LoggingConfig(_StrictModel):
    level: str = "INFO"
    audit_path: str | None = "friday_audit.log"


class Secrets(BaseModel):
    """API keys loaded from environment. Never logged in full."""

    model_config = ConfigDict(extra="ignore")

    groq_api_key: str | None = None
    elevenlabs_api_key: str | None = None
    gemini_api_key: str | None = None
    anthropic_api_key: str | None = None

    def has(self, name: str) -> bool:
        value = getattr(self, name, None)
        return bool(value and value.strip())


class Config(_StrictModel):
    providers: ProviderSelection = Field(default_factory=ProviderSelection)
    llm: LLMSection = Field(default_factory=LLMSection)
    stt: STTSection = Field(default_factory=STTSection)
    tts: TTSSection = Field(default_factory=TTSSection)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    hotkeys: HotkeysConfig = Field(default_factory=HotkeysConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    rate_limits: dict[str, RateLimitConfig] = Field(default_factory=dict)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Populated after env load. Excluded from `extra="forbid"` because it
    # never appears in config.yaml.
    secrets: Secrets = Field(default_factory=Secrets)


# --------------------------------------------------------------------------- #
# Loader                                                                       #
# --------------------------------------------------------------------------- #


# Which secret each (role, provider) pair requires. Used to fail loudly
# when someone selects a provider but forgot to set the matching key.
_REQUIRED_SECRETS: dict[tuple[str, str], str] = {
    ("llm", "groq"): "groq_api_key",
    ("llm", "gemini"): "gemini_api_key",
    ("llm", "anthropic"): "anthropic_api_key",
    ("stt", "groq"): "groq_api_key",
    ("tts", "elevenlabs"): "elevenlabs_api_key",
    # local_whisper and kokoro need no keys.
}


def load_config(
    *,
    config_path: Path | str = "config.yaml",
    env_path: Path | str | None = ".env",
) -> Config:
    """Load and validate config from ``config.yaml`` + ``.env``.

    Raises :class:`ConfigError` with a human-readable message on any
    parse / validation failure or missing required secret.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise ConfigError(
            f"config file not found: {config_path}. "
            "Copy the example or run from the project root."
        )

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"config.yaml is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"config.yaml must define a mapping at the top level, got {type(raw).__name__}"
        )

    # Don't let a stray `secrets:` block in yaml leak keys via the config file.
    if "secrets" in raw:
        raise ConfigError(
            "secrets must live in .env, not config.yaml — remove the `secrets:` block"
        )

    # .env is optional in CI / containers where env is set externally.
    if env_path is not None:
        env_path = Path(env_path)
        if env_path.exists():
            load_dotenv(env_path, override=False)

    secrets = Secrets(
        groq_api_key=_env("GROQ_API_KEY"),
        elevenlabs_api_key=_env("ELEVENLABS_API_KEY"),
        gemini_api_key=_env("GEMINI_API_KEY"),
        anthropic_api_key=_env("ANTHROPIC_API_KEY"),
    )

    try:
        config = Config(**raw, secrets=secrets)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc

    _check_required_secrets(config)
    _check_provider_specific_rules(config)

    return config


def _env(name: str) -> str | None:
    """Read an env var, treating empty / whitespace as missing."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _check_required_secrets(config: Config) -> None:
    """Fail if the *selected* providers don't have their API keys set."""
    selections: list[tuple[str, str]] = [
        ("llm", config.providers.llm),
        ("stt", config.providers.stt),
        ("tts", config.providers.tts),
    ]
    missing: list[str] = []
    for role, provider in selections:
        secret_name = _REQUIRED_SECRETS.get((role, provider))
        if secret_name is None:
            continue
        if not config.secrets.has(secret_name):
            env_var = secret_name.upper()
            missing.append(
                f"providers.{role} = {provider!r} requires {env_var} in .env"
            )
    if missing:
        raise ConfigError("missing required secrets:\n  - " + "\n  - ".join(missing))


def _check_provider_specific_rules(config: Config) -> None:
    """Enforce provider rules that pydantic can't express on its own.

    Note: provider-specific *runtime* settings (like an ElevenLabs voice
    id) are enforced by the provider implementation when it's actually
    constructed, not here. Phase 0 only validates what's needed to boot.
    """
    if config.providers.llm == "anthropic" and not config.llm.anthropic.enabled:
        raise ConfigError(
            "providers.llm = anthropic requires llm.anthropic.enabled = true "
            "(Anthropic API billing is separate from Pro — opt in explicitly)"
        )


def _format_validation_error(exc: ValidationError) -> str:
    """Render pydantic errors as something a human wants to read."""
    lines = ["config validation failed:"]
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)


__all__ = ["Config", "ConfigError", "Secrets", "load_config"]
