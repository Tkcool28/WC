"""
Smoke test for the Phase 4 squad-context panel renderer.

This module exists so that the "screenshot/text proof of the panel
rendered" item in the Phase 4 acceptance checklist has a concrete
artifact.  It exercises ``_render_squad_context`` with a synthetic
``result`` dict and prints the exact stream of streamlit calls the
panel would emit, so the human reviewer can compare the panel's
structure against the spec.

Run: ``python tests/test_panel_render_proof.py``  (no pytest needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make repo root importable.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dashboard import app as dashboard_app  # noqa: E402


class _StreamRecorder:
    """Capture streamlit calls so we can print what the panel emits.

    Streamlit raises ``NoSessionContext`` / ``ScriptRunContext`` errors
    in bare mode, so we replace the functions the panel calls with
    no-op shims that record the call for later inspection.
    """

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple, dict]] = []

    def subheader(self, *a, **kw): self.events.append(("subheader", a, kw))
    def caption(self, *a, **kw):   self.events.append(("caption",   a, kw))
    def markdown(self, *a, **kw):  self.events.append(("markdown",  a, kw))
    def columns(self, n):
        rec = self
        class _Col:
            def __enter__(self2): return self2
            def __exit__(self2, *e): pass
            def markdown(self3, *a, **kw): rec.events.append(("markdown[col]", a, kw))
            def caption(self3, *a, **kw):   rec.events.append(("caption[col]",   a, kw))
        return [_Col() for _ in range(n if isinstance(n, int) else len(n))]
    def success(self, *a, **kw):   self.events.append(("success",   a, kw))
    def info(self, *a, **kw):      self.events.append(("info",      a, kw))
    def warning(self, *a, **kw):   self.events.append(("warning",   a, kw))
    def error(self, *a, **kw):     self.events.append(("error",     a, kw))


def render_proof(home_id: str, away_id: str) -> list[tuple[str, tuple, dict]]:
    """Run the squad-context panel against a fake result and return events."""
    rec = _StreamRecorder()
    # Patch the streamlit functions used inside the panel.
    orig = {
        "subheader": dashboard_app.st.subheader,
        "caption":   dashboard_app.st.caption,
        "markdown":  dashboard_app.st.markdown,
        "columns":   dashboard_app.st.columns,
        "success":   dashboard_app.st.success,
        "info":      dashboard_app.st.info,
        "warning":   dashboard_app.st.warning,
        "error":     dashboard_app.st.error,
    }
    try:
        dashboard_app.st.subheader = rec.subheader
        dashboard_app.st.caption   = rec.caption
        dashboard_app.st.markdown  = rec.markdown
        dashboard_app.st.columns   = rec.columns
        dashboard_app.st.success   = rec.success
        dashboard_app.st.info      = rec.info
        dashboard_app.st.warning   = rec.warning
        dashboard_app.st.error     = rec.error

        fake_result = {
            "home_team": "Argentina",
            "away_team": "Brazil",
            "canonical_home_id": home_id,
            "canonical_away_id": away_id,
        }
        dashboard_app._render_squad_context(
            fake_result, home_id, away_id,
        )
    finally:
        for k, v in orig.items():
            setattr(dashboard_app.st, k, v)
    return rec.events


def _format_event(ev: tuple) -> str:
    name, args, kwargs = ev
    parts: list[str] = []
    for a in args:
        s = str(a).replace("\n", " | ")
        if len(s) > 120:
            s = s[:117] + "..."
        parts.append(s)
    if kwargs:
        parts.append(f"kwargs={ {k: v for k, v in kwargs.items()} }")
    return f"  {name}({', '.join(parts)})"


def main() -> int:
    print("=" * 72)
    print("Phase 4 panel — render proof (ARG vs BRA — both known)")
    print("=" * 72)
    events = render_proof("ARG", "BRA")
    for ev in events:
        print(_format_event(ev))

    print()
    print("=" * 72)
    print("Phase 4 panel — render proof (ARG vs ZZZ — home known, away unknown)")
    print("=" * 72)
    events = render_proof("ARG", "ZZZ")
    for ev in events:
        print(_format_event(ev))

    print()
    print("=" * 72)
    print("Phase 4 panel — render proof (XXX vs YYY — both unknown)")
    print("=" * 72)
    events = render_proof("XXX", "YYY")
    for ev in events:
        print(_format_event(ev))

    # Sanity-check: the spec-mandated exact label MUST be present.
    print()
    print("=" * 72)
    print("Spec-mandated label check")
    print("=" * 72)
    events = render_proof("ARG", "BRA")
    label = "Context only — not included in the probability model yet."
    found = any(
        ev[0] == "caption" and any(label in str(a) for a in ev[1])
        for ev in events
    )
    assert found, f"Spec label not emitted! {label!r}"
    print(f"  ✅ Spec label found: {label!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
