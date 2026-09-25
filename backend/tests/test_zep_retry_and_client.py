from pathlib import Path
from types import SimpleNamespace

import pytest
from zep_cloud.core.api_error import ApiError as ZepApiError

from app.utils import zep


def test_permanent_zep_errors_fail_without_retry():
    calls = []

    def operation():
        calls.append(True)
        raise ZepApiError(status_code=400, body={"message": "bad query"})

    with pytest.raises(ZepApiError):
        zep.call_zep_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_rate_limit_retry_respects_retry_after():
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise ZepApiError(
                status_code=429,
                headers={"Retry-After": "7"},
                body={"message": "slow down"},
            )
        return "ok"

    result = zep.call_zep_read_with_retry(
        operation,
        operation_name="rate limited read",
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    assert sleeps == [7.0]


def test_rate_limit_waits_through_more_windows_than_other_errors():
    calls = []
    sleeps = []

    def rate_limited():
        calls.append(True)
        if len(calls) < 5:
            raise ZepApiError(status_code=429, headers={"Retry-After": "59"}, body="limit")
        return "ok"

    assert zep.call_zep_read_with_retry(
        rate_limited, operation_name="edges page", sleep=sleeps.append
    ) == "ok"
    assert sleeps == [59.0] * 4

    server_errors = []

    def failing():
        server_errors.append(True)
        raise ZepApiError(status_code=503, body="down")

    with pytest.raises(ZepApiError):
        zep.call_zep_read_with_retry(failing, operation_name="edges page", sleep=lambda _s: None)
    assert len(server_errors) == 3


def test_rate_limit_gives_up_after_its_own_attempt_budget():
    calls = []

    def always_limited():
        calls.append(True)
        raise ZepApiError(status_code=429, headers={"Retry-After": "1"}, body="limit")

    with pytest.raises(ZepApiError):
        zep.call_zep_read_with_retry(always_limited, operation_name="edges page", sleep=lambda _s: None)
    assert len(calls) == 6


def test_zep_client_is_shared_and_uses_an_explicit_timeout(monkeypatch):
    created = []

    def fake_zep(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.delenv("ZEP_API_URL", raising=False)
    monkeypatch.setattr(zep, "Zep", fake_zep)
    zep.clear_zep_client_cache()

    first = zep.get_zep_client(" test-key ", timeout=12)
    second = zep.get_zep_client("test-key", timeout=12)

    assert first is second
    assert created == [{
        "api_key": "test-key",
        "base_url": zep.ZEP_CLOUD_BASE_URL,
        "timeout": 12.0,
    }]
    zep.clear_zep_client_cache()


def test_zep_client_rejects_self_hosted_endpoint_override(monkeypatch):
    monkeypatch.setenv("ZEP_API_URL", "https://example.invalid")

    with pytest.raises(ValueError, match="ZEP_API_URL"):
        zep.get_zep_client("test-key")


def test_zep_client_uses_internal_timeout_and_ignores_env_overrides(monkeypatch):
    created = []

    def fake_zep(**kwargs):
        created.append(kwargs)
        return SimpleNamespace(kwargs=kwargs)

    monkeypatch.delenv("ZEP_API_URL", raising=False)
    monkeypatch.setenv("ZEP_REQUEST_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("ZEP_INGESTION_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(zep, "Zep", fake_zep)
    zep.clear_zep_client_cache()

    zep.get_zep_client("test-key")

    assert created == [{
        "api_key": "test-key",
        "base_url": zep.ZEP_CLOUD_BASE_URL,
        "timeout": zep.ZEP_HTTP_REQUEST_TIMEOUT_SECONDS,
    }]
    assert zep.ZEP_HTTP_REQUEST_TIMEOUT_SECONDS == 60.0
    zep.clear_zep_client_cache()


def test_ingestion_wait_timeout_defaults_to_600_and_is_configurable(monkeypatch):
    import importlib

    try:
        monkeypatch.delenv("ZEP_INGESTION_WAIT_TIMEOUT_SECONDS", raising=False)
        assert importlib.reload(zep).ZEP_INGESTION_WAIT_TIMEOUT_SECONDS == 600

        monkeypatch.setenv("ZEP_INGESTION_WAIT_TIMEOUT_SECONDS", "3600")
        assert importlib.reload(zep).ZEP_INGESTION_WAIT_TIMEOUT_SECONDS == 3600
    finally:
        monkeypatch.undo()
        importlib.reload(zep)


def test_zep_timeout_policy_is_not_exposed_in_env_example():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    assert "ZEP_REQUEST_TIMEOUT_SECONDS" not in contents
    assert "ZEP_INGESTION_TIMEOUT_SECONDS" not in contents
