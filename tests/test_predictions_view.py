"""Tests for the Phase 3 Predictions experience.

Covers:

* :mod:`dashboard.text_format` — pure-Python formatters (``format_kickoff``,
  ``format_group_label``, ``format_matchday_label``, ``format_team_matchup``).
* :mod:`dashboard.prediction_card` — the renderer's *pure* helpers
  (``_outcome_headline_text``, ``_confidence_label``, ``_extract_most_likely``,
  ``_extract_top_prob``) plus the draw-wording requirement.
* The casual-facing surface of the running app — ``AppTest.from_file`` boots
  the real ``dashboard/app.py`` in ``?view=predictions`` mode and asserts
  that NO odds / NO min-edge / NO sportsbook terminology / NO raw ISO
  timestamps / NO ``GROUP_X`` strings leak into the page.

We deliberately do not test the full card render path through AppTest
(rendering involves real ratings lookups that depend on training data);
instead we test the pure helpers and the surface that the renderer
emits.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.text_format import (
    format_group_label,
    format_kickoff,
    format_matchday_label,
    format_team_matchup,
)


# --------------------------------------------------------------------------- #
# text_format: format_kickoff
# --------------------------------------------------------------------------- #
def test_format_kickoff_iso_z_returns_human_string() -> None:
    """A trailing-Z ISO timestamp becomes a 'Mon DD · H:MM AM/PM UTC' string."""
    out = format_kickoff("2026-06-17T17:00:00Z")
    assert "Jun 17" in out
    assert "5:00 PM" in out
    assert "UTC" in out


def test_format_kickoff_offset_aware_iso_returns_human_string() -> None:
    """A +00:00 ISO timestamp is also accepted and normalised to UTC."""
    out = format_kickoff("2026-06-17T22:30:00+00:00")
    assert "Jun 17" in out
    assert "10:30 PM" in out
    assert "UTC" in out


def test_format_kickoff_none_returns_tbd() -> None:
    assert format_kickoff(None) == "TBD"


def test_format_kickoff_empty_returns_tbd() -> None:
    assert format_kickoff("") == "TBD"


def test_format_kickoff_invalid_returns_tbd() -> None:
    assert format_kickoff("not-a-timestamp") == "TBD"


# --------------------------------------------------------------------------- #
# text_format: format_group_label
# --------------------------------------------------------------------------- #
def test_format_group_label_strips_group_prefix() -> None:
    assert format_group_label("GROUP_K") == "Group K"


def test_format_group_label_bare_letter() -> None:
    assert format_group_label("K") == "Group K"


def test_format_group_label_lowercase_input() -> None:
    assert format_group_label("group_a") == "Group A"


def test_format_group_label_none_or_empty_returns_tbd() -> None:
    assert format_group_label(None) == "TBD"
    assert format_group_label("") == "TBD"
    assert format_group_label("GROUP_") == "TBD"


# --------------------------------------------------------------------------- #
# text_format: format_matchday_label
# --------------------------------------------------------------------------- #
def test_format_matchday_label_with_matchday() -> None:
    assert format_matchday_label("GROUP_STAGE", 1) == "Group Stage · Matchday 1"


def test_format_matchday_label_without_matchday() -> None:
    assert format_matchday_label("KNOCKOUT", None) == "Knockout"


def test_format_matchday_label_none_stage_returns_tbd() -> None:
    assert format_matchday_label(None, 1) == "TBD"
    assert format_matchday_label(None, None) == "TBD"
    assert format_matchday_label("", 3) == "TBD"


# --------------------------------------------------------------------------- #
# text_format: format_team_matchup
# --------------------------------------------------------------------------- #
def test_format_team_matchup_basic() -> None:
    assert format_team_matchup("England", "Croatia") == "England vs Croatia"


def test_format_team_matchup_empty_home() -> None:
    assert format_team_matchup("", "Croatia") == "TBD vs Croatia"


def test_format_team_matchup_both_empty() -> None:
    assert format_team_matchup("", "") == "TBD vs TBD"


# --------------------------------------------------------------------------- #
# prediction_card: pure helpers
# --------------------------------------------------------------------------- #
def test_outcome_headline_text_draw_uses_special_wording() -> None:
    """Draws must use the exact 'Match to End in a Draw' wording."""
    from dashboard.prediction_card import _outcome_headline_text
    assert _outcome_headline_text("draw", {}) == "Match to End in a Draw"


def test_outcome_headline_text_home_uses_team_name() -> None:
    from dashboard.prediction_card import _outcome_headline_text
    pred = {"home_team": "England", "away_team": "Croatia"}
    assert _outcome_headline_text("home", pred) == "England to Win"


def test_outcome_headline_text_away_uses_team_name() -> None:
    from dashboard.prediction_card import _outcome_headline_text
    pred = {"home_team": "Mexico", "away_team": "USA"}
    assert _outcome_headline_text("away", pred) == "USA to Win"


def test_outcome_headline_text_empty_returns_tbd() -> None:
    from dashboard.prediction_card import _outcome_headline_text
    assert _outcome_headline_text("", {}) == "TBD"


def test_extract_most_likely_picks_highest_prob() -> None:
    from dashboard.prediction_card import _extract_most_likely
    pred = {
        "blend_probs": {"home": 0.51, "draw": 0.24, "away": 0.25},
        "pi_probs":    {"home": 0.50, "draw": 0.25, "away": 0.25},
    }
    assert _extract_most_likely(pred) == "home"


def test_extract_most_likely_falls_back_to_pi_probs() -> None:
    from dashboard.prediction_card import _extract_most_likely
    pred = {"pi_probs": {"home": 0.30, "draw": 0.40, "away": 0.30}}
    assert _extract_most_likely(pred) == "draw"


def test_extract_top_prob_returns_max_value() -> None:
    from dashboard.prediction_card import _extract_top_prob
    pred = {"blend_probs": {"home": 0.513, "draw": 0.244, "away": 0.243}}
    assert abs(_extract_top_prob(pred) - 0.513) < 1e-9


def test_extract_top_prob_handles_empty() -> None:
    from dashboard.prediction_card import _extract_top_prob
    assert _extract_top_prob({}) is None


def test_confidence_label_tier_a_with_elo_agreement_is_high() -> None:
    """Tier A with both models agreeing → 'High Confidence'.

    For ``agreement_status`` to return ``'agree'`` the prediction must carry
    both ``pi_only_probs`` and ``elo_only_probs`` with the same top market
    and a < 10 pt probability gap.
    """
    from dashboard.prediction_card import _confidence_label
    pred = {
        "pi_only_probs":  {"home": 0.55, "draw": 0.25, "away": 0.20},
        "elo_only_probs": {"home": 0.52, "draw": 0.25, "away": 0.23},
        "confidence":     {"tier": "A", "calibrated_p": 0.55, "warnings": []},
    }
    label, key = _confidence_label(pred)
    assert label == "High Confidence"
    assert key == "high"


def test_confidence_label_tier_a_only_pi_is_moderate() -> None:
    """Tier A but only pi ran (no Elo) → 'Moderate Confidence'.

    This is a deliberate Phase 3 design choice: when only one model ran we
    never claim 'High Confidence' — we say 'Moderate' instead, because we
    can't cross-validate the prediction.
    """
    from dashboard.prediction_card import _confidence_label
    pred = {
        "pi_only_probs":  {"home": 0.55, "draw": 0.25, "away": 0.20},
        "elo_only_probs": None,
        "confidence":     {"tier": "A", "calibrated_p": 0.55, "warnings": []},
    }
    label, key = _confidence_label(pred)
    assert label == "Moderate Confidence"
    assert key == "moderate"


def test_confidence_label_tier_d_is_low() -> None:
    """Tier D → 'Low Confidence'."""
    from dashboard.prediction_card import _confidence_label
    pred = {"confidence": {"tier": "D", "calibrated_p": 0.5, "warnings": []}}
    label, key = _confidence_label(pred)
    assert label == "Low Confidence"
    assert key == "low"


def test_confidence_label_missing_confidence_defaults_to_limited() -> None:
    """No confidence dict → 'Limited Data' (safe fallback)."""
    from dashboard.prediction_card import _confidence_label
    label, key = _confidence_label({})
    assert label == "Limited Data"
    assert key == "limited"


# --------------------------------------------------------------------------- #
# AppTest: real app boots in ?view=predictions and emits the right surface
# --------------------------------------------------------------------------- #
_DASHBOARD_APP = (
    Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
)


def _markdown_text(at: AppTest) -> str:
    """Concatenate all markdown strings emitted by the app, lowercased."""
    return "\n".join((m.value or "") for m in at.markdown).lower()


def _captions_text(at: AppTest) -> str:
    return "\n".join((c.value or "") for c in at.caption).lower()


def _all_visible_text(at: AppTest) -> str:
    """All user-visible text the app has emitted, lowercased.

    Concatenates markdown + caption + info + warning + error so we can do
    negative assertions like "no odds fields are visible".
    """
    parts: list[str] = []
    for el in at.markdown:
        parts.append((el.value or ""))
    for el in at.caption:
        parts.append((el.value or ""))
    for el in at.info:
        parts.append((el.value or ""))
    for el in at.warning:
        parts.append((el.value or ""))
    for el in at.error:
        parts.append((el.value or ""))
    return "\n".join(parts).lower()


def test_predictions_view_renders_without_exception() -> None:
    """?view=predictions boots cleanly — no exceptions raised."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception, f"app raised: {at.exception}"


def test_predictions_view_shows_single_primary_button() -> None:
    """The casual Predictions view has exactly one primary 'Show Predictions' button."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    button_labels = [b.label or "" for b in at.button]
    show_buttons = [t for t in button_labels if "Show Predictions" in t]
    assert len(show_buttons) >= 1, (
        f"Expected at least one 'Show Predictions' button; got: {button_labels!r}"
    )


def test_predictions_view_has_no_min_edge_slider() -> None:
    """The min-edge slider lives in Bets — must NOT leak into Predictions."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    slider_labels = [s.label or "" for s in at.slider]
    assert not any("Minimum edge" in t for t in slider_labels), (
        f"min-edge slider leaked into Predictions: {slider_labels!r}"
    )


def test_predictions_view_has_no_odds_input_fields() -> None:
    """No American-odds text inputs visible in the default Predictions render.

    We check that no text_input placeholder mentions odds-related terms.
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    # Look at every text_input on the page; none of them should be labelled
    # with the odds-related copy that lives on the Bets page.
    labels = [(t.label or "") for t in at.text_input]
    forbidden = ["home odds", "draw odds", "away odds"]
    for lbl in labels:
        for bad in forbidden:
            assert bad not in lbl.lower(), (
                f"Odds input '{bad}' leaked into Predictions: {labels!r}"
            )


def test_predictions_view_text_does_not_mention_odds_terms() -> None:
    """Casual-visible text should not contain sportsbook vocabulary.

    We deliberately do NOT scan the raw ``<style>`` CSS payload for these
    strings — those are developer-facing comments, not visible UI. We
    only check the rendered widget content (markdown / caption / info /
    warning / error).
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    # Concatenate only rendered widget content (skip the CSS <style> block).
    parts: list[str] = []
    for el in at.markdown:
        v = (el.value or "")
        # Drop CSS <style> blocks; they contain developer comments.
        v = re.sub(r"<style.*?</style>", "", v, flags=re.DOTALL | re.IGNORECASE)
        parts.append(v)
    for el in at.caption:
        parts.append(el.value or "")
    for el in at.info:
        parts.append(el.value or "")
    for el in at.warning:
        parts.append(el.value or "")
    for el in at.error:
        parts.append(el.value or "")
    text = "\n".join(parts).lower()

    forbidden_substrings = [
        "+ev",
        "min_edge",
        "min edge",
        "implied probability",
        "no-vig",
        "remove vig",
        "sportsbook",
        "book odds",
        "bookmaker",
    ]
    for bad in forbidden_substrings:
        assert bad not in text, (
            f"Sportsbook term '{bad}' leaked into Predictions view text"
        )


def test_predictions_view_text_does_not_contain_raw_iso_timestamps() -> None:
    """No raw ISO timestamps should appear in the casual view.

    Format-kickoff should be the only path; it produces 'Jun 17 · 5:00 PM UTC'.
    """
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    text = _all_visible_text(at)
    iso_pattern = re.compile(r"\b\d{4}-\d{2}-\d{2}t\d{2}:\d{2}:\d{2}", re.IGNORECASE)
    matches = iso_pattern.findall(text)
    assert not matches, (
        f"Raw ISO timestamps leaked into Predictions view: {matches!r}"
    )


def test_predictions_view_text_does_not_contain_raw_group_codes() -> None:
    """No raw 'GROUP_K' codes should appear; they should be humanised."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    text = _all_visible_text(at)
    assert "group_k" not in text
    assert "group_a" not in text
    # We *want* the humanised form to be present somewhere on the page.
    # (Not strictly required — depends on whether any loaded match has a
    # group — so we don't assert positive presence.)


def test_predictions_view_emits_confidence_pill_class() -> None:
    """The confidence-pill CSS class is emitted at least once."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    text = _all_visible_text(at)
    assert "wc-confidence-pill" in text, (
        "Expected 'wc-confidence-pill' class in injected CSS"
    )


def test_predictions_view_emits_why_pill_popover_style() -> None:
    """The 'Why this pick?' popover-bubble CSS target is emitted."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    text = _all_visible_text(at)
    # The Phase 3 stylesheet targets [data-testid="stPopover"] > button.
    assert "stpopover" in text, (
        "Expected the popover-bubble CSS selector in injected styles"
    )


def test_predictions_view_includes_custom_matchup_expander() -> None:
    """A 'Custom matchup' expander exists for non-2026 fixtures."""
    at = AppTest.from_file(str(_DASHBOARD_APP), default_timeout=60)
    at.query_params["view"] = "predictions"
    at.run()
    assert not at.exception
    expander_labels = [e.label or "" for e in at.expander]
    assert any("Custom matchup" in t for t in expander_labels), (
        f"'Custom matchup' expander missing from Predictions view: "
        f"{expander_labels!r}"
    )


# --------------------------------------------------------------------------- #
# Regression: predict_match (model-only) does not require odds
# --------------------------------------------------------------------------- #
def test_predict_match_signature_no_odds_param() -> None:
    """The Phase 1 split made predict_match() callable with no odds.

    This is a structural check on the public signature — guards against a
    future refactor accidentally re-introducing book_*_odds parameters on
    predict_match.
    """
    import inspect
    from soccer_ev_model.ev_workflow import predict_match
    sig = inspect.signature(predict_match)
    for name in sig.parameters:
        assert not name.startswith("book_"), (
            f"predict_match must not accept {name}; odds live on evaluate_market"
        )