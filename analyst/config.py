"""
Central configuration, read from environment variables (and `.env`).

Every LLM provider we support speaks the OpenAI chat-completions protocol, so a
provider is just: a base URL, the env var that holds its API key, a default
model and a price table. Switching provider = setting LLM_PROVIDER.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")  # explicit path: works no matter the current directory


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str
    key_env: str          # env var holding the API key ("" = no key needed)
    default_model: str


PROVIDERS: dict[str, Provider] = {
    "gemini": Provider(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
        "GEMINI_API_KEY",
        "gemini-3.5-flash-lite",  # fast + reliable; "gemini-3.5-flash" is stronger but ~10x slower
    ),
    "groq": Provider("groq", "https://api.groq.com/openai/v1", "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    "openai": Provider("openai", "https://api.openai.com/v1", "OPENAI_API_KEY", "gpt-4o-mini"),
    "ollama": Provider("ollama", "http://localhost:11434/v1", "", "llama3.1"),
}

DEFAULT_PROVIDER = "gemini"

# USD per 1M tokens (input, output). Used only for cost *estimates* in the run
# tracker; unknown models are costed at 0 and flagged as "unpriced".
# Gemini 3.x prices are approximations based on the 2.5 tiers - check the
# provider's pricing page and edit here if you need exact numbers.
PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (0.30, 2.50),
    "gemini-3.5-flash-lite": (0.10, 0.40),
    "gemini-3.1-flash-lite": (0.10, 0.40),
    "gemini-flash-latest": (0.30, 2.50),
    "gemini-flash-lite-latest": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "gpt-4o-mini": (0.15, 0.60),
    "fake": (0.0, 0.0),
}


@dataclass(frozen=True)
class Settings:
    provider: str = DEFAULT_PROVIDER
    model: str = ""
    temperature: float = 0.1
    max_tokens: int = 8192          # thinking models spend part of this on reasoning
    request_timeout: float = 120.0  # seconds per LLM HTTP request
    max_retries: int = 5            # LLM-call retries on rate limits / 5xx / timeouts
    max_fix_attempts: int = 3       # Coder self-correction rounds after a failed execution
    max_review_rounds: int = 2      # Reviewer "revise" rounds before we accept the answer
    sandbox_timeout: int = 60       # seconds a generated script may run
    outputs_dir: Path = field(default=ROOT / "outputs")
    logs_dir: Path = field(default=ROOT / "logs")

    @property
    def provider_info(self) -> Provider:
        return PROVIDERS[self.provider]

    @property
    def api_key(self) -> str:
        env = self.provider_info.key_env
        return os.getenv(env, "") if env else "not-needed"


def get_settings(provider: str | None = None, model: str | None = None, **overrides) -> Settings:
    """Build settings from env vars, with explicit arguments taking precedence."""
    provider = (provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; choose one of {sorted(PROVIDERS)}")
    model = model or os.getenv("LLM_MODEL") or PROVIDERS[provider].default_model
    env_ints = {
        "max_fix_attempts": "MAX_FIX_ATTEMPTS",
        "max_review_rounds": "MAX_REVIEW_ROUNDS",
        "sandbox_timeout": "SANDBOX_TIMEOUT",
    }
    for key, env in env_ints.items():
        if key not in overrides and os.getenv(env):
            overrides[key] = int(os.environ[env])
    return Settings(provider=provider, model=model, **overrides)
