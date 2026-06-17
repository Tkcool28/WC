"""
Tests for the dashboard's safe-fallback read of the blend probabilities.

Background: the dashboard prediction-summary block now reads
    blended = result.get("blend_probs", result["pi_probs"])
so that the explicit `blend_probs` alias is preferred, with a fallback to
the historical `pi_probs` key for results from older workflow versions.

These tests don't import `dashboard.app` (it pulls in Streamlit) — they
exercise the same `.get(...)` fallback pattern directly against plain
result dicts. If the pattern is ever changed in `app.py`, mirror the
update here.
"""
import pytest


def _read_blended(result: dict) -> dict:
    """Mirror the exact pattern used in dashboard/app.py. Update both
    together if the pattern ever changes."""
    return result.get("blend_probs", result["pi_probs"])


def test_fallback_returns_pi_probs_when_blend_probs_missing():
    """When only `pi_probs` is present (old result), the fallback returns it."""
    pi_probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
    result = {"pi_probs": pi_probs}  # no blend_probs
    assert _read_blended(result) == pi_probs


def test_fallback_prefers_blend_probs_when_both_present():
    """When both keys are present, `blend_probs` wins (it's the new alias)."""
    pi_probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
    blend_probs = {"home": 0.50, "draw": 0.30, "away": 0.20}  # distinct value
    result = {"pi_probs": pi_probs, "blend_probs": blend_probs}
    assert _read_blended(result) == blend_probs
    assert _read_blended(result) is not pi_probs  # explicitly not the fallback


def test_fallback_raises_keyerror_if_neither_present():
    """If both keys are missing, the fallback pattern raises KeyError on
    the inner `result['pi_probs']` lookup. This documents the contract:
    callers must guarantee at least one of the two keys is present."""
    result = {}
    with pytest.raises(KeyError):
        _read_blended(result)


def test_fallback_handles_none_blend_probs_falls_back():
    """Defensive: if `blend_probs` is explicitly None, `.get` still returns
    None, NOT the fallback. This is a quirk of dict.get — pinning behaviour
    so it doesn't surprise us later."""
    pi_probs = {"home": 0.55, "draw": 0.25, "away": 0.20}
    result = {"blend_probs": None, "pi_probs": pi_probs}
    # .get returns the None, not the fallback
    assert _read_blended(result) is None
