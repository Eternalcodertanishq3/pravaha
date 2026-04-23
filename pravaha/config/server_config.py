"""Server configuration — host, port, auth, CORS, rate limits.

Defines all HTTP server settings including authentication, CORS policy,
rate limiting, and TLS configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class CORSConfig(BaseModel):
    """CORS (Cross-Origin Resource Sharing) settings.

    Attributes:
        enabled: Whether to enable CORS middleware.
        allow_origins: Allowed origin patterns. ["*"] allows all.
        allow_methods: Allowed HTTP methods.
        allow_headers: Allowed request headers.
        allow_credentials: Whether to allow credentials (cookies, auth headers).
    """

    enabled: bool = True
    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True


class AuthConfig(BaseModel):
    """Authentication configuration.

    Attributes:
        enabled: Whether API key authentication is required.
        api_keys: List of valid API keys.
        header_name: HTTP header carrying the API key.
    """

    enabled: bool = False
    api_keys: list[str] = Field(default_factory=list)
    header_name: str = "Authorization"


class RateLimitConfig(BaseModel):
    """Rate limiting configuration.

    Attributes:
        enabled: Whether rate limiting is active.
        requests_per_minute: Maximum requests per minute per client IP.
        burst_size: Allow up to this many requests in a burst before throttling.
    """

    enabled: bool = False
    requests_per_minute: int = 60
    burst_size: int = 10


class TLSConfig(BaseModel):
    """TLS/SSL configuration for HTTPS.

    Attributes:
        enabled: Whether to enable TLS.
        cert_file: Path to the TLS certificate file.
        key_file: Path to the TLS private key file.
    """

    enabled: bool = False
    cert_file: str | None = None
    key_file: str | None = None


class ServerConfig(BaseModel):
    """Full server configuration.

    Controls the FastAPI/uvicorn server settings including networking,
    security, and observability options.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_concurrent: int = 64
    request_timeout_seconds: int = 300
    cors: CORSConfig = Field(default_factory=CORSConfig)
    auth: AuthConfig = Field(default_factory=AuthConfig)
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    tls: TLSConfig = Field(default_factory=TLSConfig)
    enable_metrics: bool = True
    enable_docs: bool = True
    log_level: str = "info"

    @classmethod
    def from_yaml(cls, path: str | Path) -> ServerConfig:
        """Load server configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Server config not found: {path}, using defaults.")
            return cls()

        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        server_section = raw.get("server", raw)
        return cls.model_validate(server_section)
