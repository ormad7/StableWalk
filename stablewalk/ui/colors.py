"""
Shared visualization palette (matplotlib / 3D skeleton / charts).

Semantic colors — use SIDE_LEFT, SIDE_RIGHT, COM, METRIC_GLOBAL, STABILITY,
WARNING, and CRITICAL everywhere; do not hardcode alternate hex values in viewers.

Chrome colors are imported from design_system.tokens so Tk and Matplotlib share
one brand. Limb L/R greens/reds stay distinct from status OK/fail.
"""

from __future__ import annotations

import stablewalk.ui.design_system.tokens as _T

# --- Semantic limb / metric (canonical for series) ---
SIDE_LEFT = "#22c55e"  # Green — left body side / left limb traces
SIDE_RIGHT = "#ef4444"  # Red — right body side / right limb traces
METRIC_GLOBAL = "#f0f4f8"  # White — aggregate / total / global metrics
COM = "#4dabf7"  # Blue — center of mass
# Stability classification hues aligned with clinical status tokens.
STABILITY = _T.STATUS_WARN
WARNING = "#ffc857"  # Orange — warning / reduced stability (chart wash)
CRITICAL = _T.STATUS_FAIL

# Chart playhead (synced time cursor on dark panels)
PLAYHEAD = METRIC_GLOBAL

# --- UI / panel chrome (matplotlib faces, axes) — single source: tokens ---
ACCENT = _T.ACCENT
ACCENT_ALT = COM
BORDER = _T.BORDER
ELEVATED = _T.ELEVATED
INFO = _T.ACCENT
PANEL = _T.PANEL
TEXT = _T.TEXT
TEXT_SECONDARY = _T.TEXT_SECONDARY
MUTED = _T.MUTED
VIZ_JOINT = "#c9a227"

# Biomechanical stability classification (gait analysis)
STABILITY_STABLE = STABILITY
STABILITY_REDUCED = WARNING
STABILITY_UNSTABLE = CRITICAL

# Base-of-support floor polygon — high-contrast green / yellow / red
# (limb convention: left green / right red — not status-badge greens)
BOS_FILL_STABLE = SIDE_LEFT
BOS_FILL_REDUCED = "#eab308"
BOS_FILL_UNSTABLE = SIDE_RIGHT
BOS_EDGE_STABLE = "#16a34a"
BOS_EDGE_REDUCED = "#ca8a04"
BOS_EDGE_UNSTABLE = "#dc2626"

# COM overlay reuses BoS stability palette for sphere, trail, and projection.
COM_FILL_STABLE = BOS_FILL_STABLE
COM_FILL_REDUCED = BOS_FILL_REDUCED
COM_FILL_UNSTABLE = BOS_FILL_UNSTABLE
COM_EDGE_STABLE = BOS_EDGE_STABLE
COM_EDGE_REDUCED = BOS_EDGE_REDUCED
COM_EDGE_UNSTABLE = BOS_EDGE_UNSTABLE
