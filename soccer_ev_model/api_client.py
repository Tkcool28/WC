"""Rate-limited client for the football-data.org API.

Single chokepoint for all calls. Enforces a minimum delay between requests,
backs off on HTTP 429, identifies itself with a polite User-Agent, and loads
the API token from .env in the current working directory.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

# Import urlopen as a module-level name so tests can patch it.
urlopen = urllib.request.urlopen
HTTPError = urllib.error.HTTPError


class FootballDataClient:
    BASE_URL = "https://api.football-data.org/v4"
    USER_AGENT = "Hermes-Research-Bot/1.0 (contact: todd-private)"
    MIN_DELAY_SECONDS = 6.0
    BACKOFF_429_SECONDS = 60.0
    DEFAULT_MAX_RETRIES = 3

    def __init__(
        self,
        token: str | None = None,
        min_delay: float = MIN_DELAY_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        self.token = token or self._load_token()
        if not self.token:
            raise ValueError(
                "No API token. Pass token= or set FOOTBALL_DATA_API_KEY in .env"
            )
        self.min_delay = min_delay
        self.max_retries = max_retries
        self._last_call_ts: float | None = None

    @staticmethod
    def _load_token() -> str | None:
        env_token = os.environ.get("FOOTBALL_DATA_API_KEY")
        if env_token:
            return env_token.strip()
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("FOOTBALL_DATA_API_KEY="):
                    return line.split("=", 1)[1].strip()
        return None

    def _throttle(self) -> None:
        if self._last_call_ts is None:
            return
        elapsed = time.monotonic() - self._last_call_ts
        if elapsed < self.min_delay:
            time.sleep(self.min_delay - elapsed)

    def get(self, path: str) -> dict:
        """GET a path on the API. Returns parsed JSON. Retries on 429.

        Path should be relative to /v4, e.g. "/competitions/WC" or "competitions/WC".
        Do not include "/v4" — it's added automatically.
        """
        # Strip a leading "/v4" if the caller included it, then ensure single "/".
        if path.startswith("/v4/"):
            path = path[len("/v4/"):]
        elif path == "/v4":
            path = ""
        if path and not path.startswith("/"):
            path = "/" + path
        url = self.BASE_URL + path
        self._throttle()

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(
                url,
                headers={
                    "X-Auth-Token": self.token,
                    "User-Agent": self.USER_AGENT,
                },
            )
            try:
                with urlopen(req, timeout=15) as resp:
                    self._last_call_ts = time.monotonic()
                    body = resp.read()
                    return json.loads(body)
            except HTTPError as e:
                self._last_call_ts = time.monotonic()
                if e.code == 429:
                    last_error = e
                    if attempt < self.max_retries:
                        time.sleep(self.BACKOFF_429_SECONDS)
                        continue
                    raise RuntimeError(
                        f"football-data.org rate limit hit "
                        f"({self.max_retries + 1} attempts)"
                    ) from e
                raise
        # Defensive fallback (should be unreachable due to raise above).
        raise RuntimeError("football-data.org: exhausted retries") from last_error
