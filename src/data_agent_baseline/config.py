from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


def _load_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        values[key] = value
    return values


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    framework: str = "langgraph"
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0
    enable_langsmith: bool = False
    langsmith_project: str = "kddcup2026-data-agent"


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def _to_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_app_config(config_path: Path) -> AppConfig:
    dotenv_values = _load_dotenv(ENV_FILE)
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    raw_api_base = str(agent_payload.get("api_base", agent_defaults.api_base)).strip()
    raw_api_key = str(agent_payload.get("api_key", agent_defaults.api_key)).strip()
    raw_framework = str(agent_payload.get("framework", agent_defaults.framework)).strip().lower()
    if raw_framework not in {"react", "langgraph"}:
        raise ValueError("agent.framework must be either 'react' or 'langgraph'.")
    resolved_api_base = (
        raw_api_base
        or dotenv_values.get("API_URL", "").strip()
        or dotenv_values.get("OPENAI_API_BASE", "").strip()
        or os.getenv("API_URL", "").strip()
        or os.getenv("OPENAI_API_BASE", "").strip()
        or agent_defaults.api_base
    )
    # Prefer explicit config value; fallback to environment variables for safer secret handling.
    resolved_api_key = (
        raw_api_key
        or dotenv_values.get("API_KEY", "").strip()
        or dotenv_values.get("OPENAI_API_KEY", "").strip()
        or dotenv_values.get("DASHSCOPE_API_KEY", "").strip()
        or os.getenv("API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("DASHSCOPE_API_KEY", "").strip()
    )
    if resolved_api_key and not resolved_api_key.isascii():
        raise ValueError(
            "Configured API key contains non-ASCII characters. "
            "Please check .env/API_KEY (or OPENAI_API_KEY/DASHSCOPE_API_KEY) for hidden or non-English characters."
        )

    agent_config = AgentConfig(
        framework=raw_framework or agent_defaults.framework,
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=resolved_api_base,
        api_key=resolved_api_key,
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
        enable_langsmith=_to_bool(
            agent_payload.get("enable_langsmith"),
            agent_defaults.enable_langsmith,
        ),
        langsmith_project=str(agent_payload.get("langsmith_project", agent_defaults.langsmith_project)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)
