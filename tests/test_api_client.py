import os
import time
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
from soccer_ev_model.api_client import FootballDataClient


def test_load_token_from_env_file(tmp_path, monkeypatch):
    """Token is read from .env in the current working directory."""
    env_file = tmp_path / ".env"
    env_file.write_text("FOOTBALL_DATA_API_KEY=test_token_abc123\n")
    monkeypatch.chdir(tmp_path)
    client = FootballDataClient()
    assert client.token == "test_token_abc123"


def test_load_token_explicit():
    """Token can be passed in directly (overrides .env)."""
    client = FootballDataClient(token="explicit_token_xyz")
    assert client.token == "explicit_token_xyz"


def test_load_token_missing_raises():
    """If no token anywhere, we get a clear error — never a silent empty key."""
    with patch.dict(os.environ, {}, clear=True):
        os.environ.pop("FOOTBALL_DATA_API_KEY", None)
        with patch("pathlib.Path.exists", return_value=False):
            try:
                FootballDataClient()
                assert False, "Expected ValueError"
            except ValueError as e:
                assert "API token" in str(e)


def test_user_agent_header():
    """The User-Agent identifies us politely. The token is sent in the header."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"name": "FIFA World Cup"}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: False
        mock_urlopen.return_value = mock_resp

        client = FootballDataClient(token="abc", min_delay=0)
        client.get("/v4/competitions/WC")

        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        # urllib normalizes header names: X-Auth-Token -> X-auth-token
        header_items = dict(request_obj.header_items())
        assert "Hermes-Research-Bot" in header_items.get("User-agent", "")
        assert header_items.get("X-auth-token") == "abc"


def test_429_triggers_backoff():
    """On 429, sleep 60s before retrying once."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen, \
         patch("soccer_ev_model.api_client.time.sleep") as mock_sleep:

        resp_429 = HTTPError(
            url=None, code=429, msg="Too Many Requests", hdrs=None, fp=None
        )

        resp_200 = MagicMock()
        resp_200.status = 200
        resp_200.read.return_value = b'{"ok": true}'
        resp_200.__enter__ = lambda s: s
        resp_200.__exit__ = lambda s, *a: False

        # First urlopen() call raises 429. Second returns 200.
        mock_urlopen.side_effect = [resp_429, resp_200]

        client = FootballDataClient(token="abc", min_delay=0, max_retries=2)
        result = client.get("/v4/test")

        # We slept for 60s on the 429
        assert any(call.args[0] == 60 for call in mock_sleep.call_args_list)
        # We got the successful 200 response
        assert result == {"ok": True}


def test_429_exhausts_retries_raises():
    """If we hit 429 more than max_retries times, raise clearly."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen, \
         patch("soccer_ev_model.api_client.time.sleep") as mock_sleep:

        resp_429 = HTTPError(
            url=None, code=429, msg="Too Many Requests", hdrs=None, fp=None
        )
        mock_urlopen.side_effect = resp_429

        client = FootballDataClient(token="abc", min_delay=0, max_retries=1)

        try:
            client.get("/v4/test")
            assert False, "Expected RuntimeError"
        except RuntimeError as e:
            assert "429" in str(e) or "rate limit" in str(e).lower()


def test_min_delay_between_calls():
    """Between two calls, we sleep at least min_delay seconds."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen, \
         patch("soccer_ev_model.api_client.time.sleep") as mock_sleep:

        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: False
        mock_urlopen.return_value = mock_resp

        client = FootballDataClient(token="abc", min_delay=6.0)
        client.get("/v4/a")
        client.get("/v4/b")

        # The second call should have slept close to 6s
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        assert any(5.99 <= s <= 6.01 for s in sleep_calls), (
            f"Expected a ~6.0s sleep, got: {sleep_calls}"
        )


def test_get_strips_v4_prefix_to_avoid_double_path():
    """The base URL already includes /v4. Paths starting with /v4 must not double up."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: False
        mock_urlopen.return_value = mock_resp

        client = FootballDataClient(token="abc", min_delay=0)
        client.get("/v4/competitions/WC")

        request_obj = mock_urlopen.call_args[0][0]
        assert request_obj.full_url == "https://api.football-data.org/v4/competitions/WC"


def test_get_returns_parsed_json():
    """A successful 200 returns the parsed JSON body."""
    with patch("soccer_ev_model.api_client.urlopen") as mock_urlopen:
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"hello": "world", "n": 42}'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = lambda s, *a: False
        mock_urlopen.return_value = mock_resp

        client = FootballDataClient(token="abc", min_delay=0)
        result = client.get("/v4/test")
        assert result == {"hello": "world", "n": 42}
