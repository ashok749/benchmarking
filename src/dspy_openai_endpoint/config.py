from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal

from dotenv import load_dotenv

# Load .env from the package root (two levels up from this file)
_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PACKAGE_ROOT / ".env", override=True)

StrategyName = Literal["base", "best_of_n", "gepa"]


@dataclass(frozen=True)
class ModelAlias:
    alias: str
    upstream_model: str
    strategy: StrategyName


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    endpoint_api_key: str | None = None
    upstream_api_base: str = "https://openrouter.ai/api/v1"
    upstream_api_key: str | None = None
    best_of_n_count: int = 3
    best_of_n_threshold: float = 0.85
    artifact_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[2] / "artifacts")
    model_aliases: Dict[str, ModelAlias] = field(default_factory=dict)

    @staticmethod
    def load() -> "Settings":
        settings = Settings(
            host=os.getenv("DSPY_HOST", "0.0.0.0"),
            port=int(os.getenv("DSPY_PORT", "8000")),
            endpoint_api_key=os.getenv("DSPY_ENDPOINT_API_KEY"),
            upstream_api_base=os.getenv("DSPY_UPSTREAM_API_BASE", "https://openrouter.ai/api/v1"),
            upstream_api_key=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"),
            best_of_n_count=int(os.getenv("DSPY_BEST_OF_N", "3")),
            best_of_n_threshold=float(os.getenv("DSPY_BEST_OF_N_THRESHOLD", "0.85")),
            artifact_dir=Path(os.getenv("DSPY_GEPA_ARTIFACT_DIR", str(_PACKAGE_ROOT / "artifacts"))),
        )
        object.__setattr__(settings, "model_aliases", load_model_aliases())
        return settings


def _default_aliases() -> dict[str, ModelAlias]:
    upstream_models = {
        "glm-5": "z-ai/glm-5",
        "glm-4.7": "z-ai/glm-4.7",
        "minimax-m2.7": "minimax/minimax-m2.7",
        "minimax-m2.5": "minimax/minimax-m2.5",
    }
    aliases: dict[str, ModelAlias] = {}
    for short_name, upstream_model in upstream_models.items():
        aliases[f"dspy-base-{short_name}"] = ModelAlias(
            alias=f"dspy-base-{short_name}", upstream_model=upstream_model, strategy="base"
        )
        aliases[f"dspy-bestofn-{short_name}"] = ModelAlias(
            alias=f"dspy-bestofn-{short_name}", upstream_model=upstream_model, strategy="best_of_n"
        )
        aliases[f"dspy-gepa-{short_name}"] = ModelAlias(
            alias=f"dspy-gepa-{short_name}", upstream_model=upstream_model, strategy="gepa"
        )
    return aliases


def load_model_aliases() -> dict[str, ModelAlias]:
    raw = os.getenv("DSPY_MODEL_ALIASES_JSON")
    if not raw:
        return _default_aliases()

    parsed = json.loads(raw)
    aliases: dict[str, ModelAlias] = {}
    for alias, config in parsed.items():
        aliases[alias] = ModelAlias(
            alias=alias,
            upstream_model=config["upstream_model"],
            strategy=config["strategy"],
        )
    return aliases
