"""StableWalk design tokens — spacing, type, color, control sizes.

Presentation only. Analysis formulas and metric values are unchanged.

Every dashboard page should import spacing / type / color from here (or via
``stablewalk.ui.theme`` aliases) so chrome stays consistent.
"""

from __future__ import annotations

# ── Spacing (4 / 8 / 16 / 24 / 32 primary scale) ──────────────────────────
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 16
SPACE_4 = 24
SPACE_5 = 32
SPACE_6 = 32
SPACE_8 = 32  # 8×4 rhythm alias

PAD_XS = SPACE_1  # 4 — tight control / chip gap
PAD_SM = SPACE_2  # 8 — control / card gap
PAD_MD = SPACE_3  # 16 — section / card interior
PAD_LG = SPACE_4  # 24 — major section separation
PAD_XL = SPACE_5  # 32

# Semantic layout aliases — prefer these over raw PAD_* at call sites.
INLINE_GAP = PAD_XS  # 4 — between sibling controls in one toolbar
CONTROL_GAP = PAD_SM  # 8 — between related controls
SECTION_GAP = PAD_MD  # 16 — between sections / cards
PAGE_GUTTER = PAD_MD  # 16 — page edge / column gutter
CARD_GAP = PAD_SM  # 8 — gap between cards in a row
KPI_GAP = PAD_SM

PANEL_PAD = (PAD_MD, PAD_SM)  # interior padding for main panels
CARD_PAD = (PAD_MD, PAD_SM)
CARD_PAD_TIGHT = (PAD_SM, PAD_SM)
DASHBOARD_TAB_PAD = (PAD_MD, PAD_SM)
PAGE_HEADER_PADY = (0, PAD_MD)  # space under every page title
PAGE_HEADER_SUBTITLE_PADY = (PAD_SM, 0)
SECTION_TITLE_PADY = (PAD_MD, PAD_XS)  # above / below section captions

# ── Surfaces (lighter clinical dark — less “instrument rack”) ─────────────
BG = "#10151c"
SURFACE = "#161c26"
PANEL = "#1a222e"
PANEL_SECONDARY = "#1e2734"
ELEVATED = "#222b3a"
CONTROL = "#2a3444"
BORDER = "#314052"
BORDER_SUBTLE = "#283242"
BORDER_FOCUS = "#6aa8d8"

TEXT = "#e8eef6"
TEXT_SECONDARY = "#b8c4d4"
MUTED = "#8a9aad"
# Slightly brighter than prior dim for WCAG-friendly meta on dark panels.
MUTED_DIM = "#7a8b9f"

# ── Semantic status (validated meaning only) ──────────────────────────────
STATUS_OK = "#3d9a6a"  # green — acceptable / completed
STATUS_WARN = "#d4a84b"  # yellow — limited / uncertain / partial
STATUS_FAIL = "#d95555"  # red — failed / missing / clinically concerning
STATUS_INFO = "#5b9fd4"  # blue — neutral action / selected
ACCENT = STATUS_INFO
ACCENT_FG = "#071018"

# ── Typography hierarchy (min 9 pt body / metadata) ───────────────────────
# Page → Section → Card → Body → Meta. Prefer these over ad-hoc Segoe tuples.
FONT_PAGE_TITLE = ("Segoe UI Semibold", 15)
FONT_SECTION_TITLE = ("Segoe UI Semibold", 11)
FONT_CARD_TITLE = ("Segoe UI Semibold", 10)
FONT_METRIC = ("Segoe UI Semibold", 18)  # hero KPI / MetricCard value
FONT_METRIC_MD = ("Segoe UI Semibold", 16)  # secondary KPI value
FONT_METRIC_SM = ("Segoe UI Semibold", 14)  # panel metric values
FONT_BODY = ("Segoe UI", 10)
FONT_BODY_SEMIBOLD = ("Segoe UI Semibold", 10)
FONT_SUPPORTING = ("Segoe UI", 9)
FONT_SUPPORTING_SEMIBOLD = ("Segoe UI Semibold", 9)
FONT_META = ("Segoe UI", 9)
FONT_MONO = ("Cascadia Mono", 10)
FONT_MONO_SM = ("Cascadia Mono", 9)
FONT_BUTTON = FONT_BODY_SEMIBOLD
FONT_BUTTON_SM = FONT_SUPPORTING_SEMIBOLD

# ── Controls ──────────────────────────────────────────────────────────────
CONTROL_HEIGHT = 32
BUTTON_HEIGHT = 32
INPUT_HEIGHT = 32
COMBO_HEIGHT = 32
RADIUS_PX = 4  # documented intent; Tk approximates via flat relief
BTN_PAD = (12, 7)  # primary / export — targets ~32 px height
BTN_PAD_COMPACT = (10, 5)  # secondary / toolbar
BTN_PAD_TRANSPORT = (10, 5)

# ── Card chrome ───────────────────────────────────────────────────────────
# Prefer surface contrast over borders; accent strip optional (often hidden).
CARD_BORDER_WIDTH = 0
CARD_ACCENT_HEIGHT = 2
SEPARATOR_HEIGHT = 1

# LabelFrame / section title treatment (Title Case, never ALL CAPS in chrome).
SECTION_TITLE_CASE = "title"  # informational; format_labelframe_title enforces strip

# ── Subtle motion (fade / micro-slide; see ui_motion.py) ───────────────────
MOTION_MS = 180  # target duration band: 150–200 ms
MOTION_STEPS = 5
MOTION_SLIDE_PX = 6
