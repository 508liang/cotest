import os
from dataclasses import dataclass
from pathlib import Path


def _load_local_env() -> None:
    """Load .env once when present, without overriding existing environment vars."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_local_env()


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _first_non_empty(*values: str) -> str:
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return ""


@dataclass(frozen=True)
class Settings:
    # Slack
    slack_bot_token: str
    slack_app_token: str
    slack_bot_id: str
    slack_legacy_mention_prefix: str

    # Search / LLM
    serpapi_key: str
    openai_api_key: str
    openai_api_base: str
    llm_model_name: str
    llm_fallback_model_name: str

    # Database
    db_host: str
    db_user: str
    db_password: str
    db_port: int
    db_name: str


def load_settings() -> Settings:
    return Settings(
        slack_bot_token=_env("SLACK_BOT_TOKEN"),
        slack_app_token=_env("SLACK_APP_TOKEN"),
        slack_bot_id=_first_non_empty(_env("SLACK_BOT_ID"), _env("BOT_ID"), "bot_id"),
        slack_legacy_mention_prefix=_env("SLACK_LEGACY_MENTION_PREFIX", "@coresearchagent_kx"),
        serpapi_key=_env("SERPAPI_KEY"),
        openai_api_key=_first_non_empty(_env("OPENAI_API_KEY"), _env("OPENAI_KEY")),
        openai_api_base=_env(
            "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ),
        llm_model_name=_env("LLM_MODEL_NAME", "qwen-plus-2025-07-28"),
        llm_fallback_model_name=_env("LLM_FALLBACK_MODEL_NAME", "qwen-turbo"),
        db_host=_env("DB_HOST", "localhost"),
        db_user=_env("DB_USER", "root"),
        db_password=_first_non_empty(_env("DB_PASSWORD"), _env("SQL_PASSWORD"), "root123456"),
        db_port=int(_env("DB_PORT", "3306")),
        db_name=_env("DB_NAME", "mysql"),
    )


settings = load_settings()


def validate_required_settings() -> None:
    missing = []
    if not settings.slack_bot_token:
        missing.append("SLACK_BOT_TOKEN")
    if not settings.slack_app_token:
        missing.append("SLACK_APP_TOKEN")
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.serpapi_key:
        missing.append("SERPAPI_KEY")
    if missing:
        raise RuntimeError("Missing required settings: " + ", ".join(missing))