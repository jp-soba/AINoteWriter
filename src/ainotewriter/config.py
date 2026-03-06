from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class AppConfig:
    x_api_key: str
    x_api_key_secret: str
    x_access_token: str
    x_access_token_secret: str
    x_bearer_token: str = ""
    x_api_base_url: str = "https://api.x.com/2"

    ai_provider: str = "xai"
    ai_api_key: str = ""
    ai_base_url: str = "https://api.x.ai/v1"
    ai_model: str = "grok-3-latest"
    ai_timeout_sec: int = 60
    claude_cli_path: str = "claude"
    claude_max_turns: int = 4
    claude_use_cli_fallback: bool = True

    discord_webhook_url: str = ""

    default_num_posts: int = 5
    default_test_mode: bool = True
    default_submit_notes: bool = False
    default_evaluate_before_submit: bool = True
    default_min_claim_opinion_score: float = -1
    default_enable_url_check: bool = False
    url_check_timeout_sec: int = 5

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            x_api_key=os.getenv("X_API_KEY", ""),
            x_api_key_secret=os.getenv("X_API_KEY_SECRET", ""),
            x_access_token=os.getenv("X_ACCESS_TOKEN", ""),
            x_access_token_secret=os.getenv("X_ACCESS_TOKEN_SECRET", ""),
            x_bearer_token=os.getenv("X_BEARER_TOKEN", ""),
            x_api_base_url=os.getenv("X_API_BASE_URL", "https://api.x.com/2"),
            ai_provider=os.getenv("AI_PROVIDER", "xai"),
            ai_api_key=os.getenv("AI_API_KEY", ""),
            ai_base_url=os.getenv("AI_BASE_URL", "https://api.x.ai/v1"),
            ai_model=os.getenv("AI_MODEL", "grok-3-latest"),
            ai_timeout_sec=int(os.getenv("AI_TIMEOUT_SEC", "60")),
            claude_cli_path=os.getenv("CLAUDE_CLI_PATH", "claude"),
            claude_max_turns=int(os.getenv("CLAUDE_MAX_TURNS", "4")),
            claude_use_cli_fallback=_as_bool(
                os.getenv("CLAUDE_USE_CLI_FALLBACK"), True
            ),
            discord_webhook_url=os.getenv("DISCORD_WEBHOOK_URL", ""),
            default_num_posts=int(os.getenv("DEFAULT_NUM_POSTS", "5")),
            default_test_mode=_as_bool(os.getenv("DEFAULT_TEST_MODE"), True),
            default_submit_notes=_as_bool(os.getenv("DEFAULT_SUBMIT_NOTES"), False),
            default_evaluate_before_submit=_as_bool(
                os.getenv("DEFAULT_EVALUATE_BEFORE_SUBMIT"), True
            ),
            default_min_claim_opinion_score=float(
                os.getenv("DEFAULT_MIN_CLAIM_OPINION_SCORE", "-1")
            ),
            default_enable_url_check=_as_bool(os.getenv("DEFAULT_ENABLE_URL_CHECK"), False),
            url_check_timeout_sec=int(os.getenv("URL_CHECK_TIMEOUT_SEC", "5")),
        )

    def validate_x_auth(self) -> None:
        missing = [
            name
            for name, value in {
                "X_API_KEY": self.x_api_key,
                "X_API_KEY_SECRET": self.x_api_key_secret,
                "X_ACCESS_TOKEN": self.x_access_token,
                "X_ACCESS_TOKEN_SECRET": self.x_access_token_secret,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required X credentials: {', '.join(missing)}")