"""
Shared visualization palette (matplotlib / 3D skeleton / charts).

Semantic colors — use SIDE_LEFT, SIDE_RIGHT, COM, METRIC_GLOBAL, STABILITY,
WARNING, and CRITICAL everywhere; do not hardcode alternate hex values in viewers.

Kept separate from ``theme`` so analysis and 3D plotting do not depend on Tk.
"""

# --- Semantic (canonical) ---
SIDE_LEFT = "#22c55e"  # Green — left body side / left limb traces
SIDE_RIGHT = "#ef4444"  # Red — right body side / right limb traces
METRIC_GLOBAL = "#f0f4f8"  # White — aggregate / total / global metrics
COM = "#4dabf7"  # Blue — center of mass
STABILITY = "#ffd43b"  # Yellow — stability (stable classification)
WARNING = "#ffc857"  # Orange — warning / reduced stability
CRITICAL = "#ef4444"  # Red — critical / unstable

# Chart playhead (synced time cursor on dark panels)
PLAYHEAD = METRIC_GLOBAL

# --- UI / panel chrome (matplotlib faces, axes) ---
# Brand accent must NOT equal SIDE_LEFT — keep L/R limb colors distinct.
ACCENT = "#2ec99a"
ACCENT_ALT = COM
BORDER = "#354257"
ELEVATED = "#222d40"
INFO = "#74c0fc"
PANEL = "#1a2332"
TEXT = "#eef3f8"
MUTED = "#8494ab"
VIZ_JOINT = "#c9a227"

# Biomechanical stability classification (gait analysis)
STABILITY_STABLE = STABILITY
STABILITY_REDUCED = WARNING
STABILITY_UNSTABLE = CRITICAL

# Base-of-support floor polygon — high-contrast green / yellow / red
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
