"""Dashboard CSS / visual layer.

Phase 6 will land the actual mobile-first stylesheet here. For Phase 2 we
just ensure the import surface exists so ``app.py`` can wire it up.
"""
from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    """No-op placeholder. Real styles land in Phase 6.

    Kept as an explicit function (rather than nothing) so the call site in
    :func:`dashboard.app.main` is stable across phases.
    """
    return None
