"""Visual difference chrome for Compare Mode (display only).

Communicates Δ = B − A with horizontal bars, tone indicators, % change,
and mini sparklines before numeric readouts.
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Any, Sequence

import numpy as np

from stablewalk.analysis.session_compare import MetricDiff, SessionComparisonResult
from stablewalk.ui.scientific_labels import UNIT_CADENCE
from stablewalk.ui.theme import (
    BORDER,
    DANGER,
    FONT_HEADING,
    FONT_UI_SEMIBOLD,
    FONT_UI_SM,
    FONT_UI_XS,
    MUTED,
    MUTED_DIM,
    PAD_MD,
    PAD_SM,
    PAD_XS,
    SUCCESS,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    WARNING,
)
from stablewalk.ui.tk.compare_state import (
    SESSION_B_METRICS_PLACEHOLDER_BODY,
    SESSION_B_METRICS_PLACEHOLDER_TITLE,
    SESSION_B_WAITING_TITLE,
)

# Session A / B stroke colors (shared instrument palette).
from stablewalk.ui.colors import COM as _COLOR_A
from stablewalk.ui.theme import ACCENT_ALT as _COLOR_B


def pct_change(left: float | None, right: float | None) -> float | None:
    """Percent change of B relative to A: 100 * (B − A) / |A|."""
    if left is None or right is None:
        return None
    try:
        a = float(left)
        b = float(right)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b) or abs(a) < 1e-12:
        return None
    return 100.0 * (b - a) / abs(a)


def format_delta(val: float | None, unit: str) -> str:
    if val is None or not math.isfinite(float(val)):
        return "—"
    v = float(val)
    if abs(v) >= 100:
        body = f"{v:.0f}"
    elif abs(v) >= 10:
        body = f"{v:.1f}"
    else:
        body = f"{v:.2f}"
    sign = "+" if v > 0 else ""
    return f"{sign}{body}{unit}"


def format_pct(pct: float | None) -> str:
    if pct is None or not math.isfinite(pct):
        return ""
    sign = "+" if pct > 0 else ""
    return f"{sign}{pct:.0f}%"


def tone_color(tone: str) -> str:
    if tone == "better":
        return SUCCESS
    if tone == "worse":
        return DANGER
    if tone == "na":
        return MUTED
    return TEXT


def tone_glyph(tone: str, delta: float | None) -> str:
    if tone == "better":
        return "▲"
    if tone == "worse":
        return "▼"
    if delta is None:
        return "●"
    if abs(float(delta)) < 1e-9:
        return "●"
    return "◆"


def _resample_series(values: Sequence[float] | np.ndarray | None, n: int = 48) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(
        [np.nan if v is None else float(v) for v in values],
        dtype=float,
    )
    if arr.size < 2 or not np.isfinite(arr).any():
        return None
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return None
    # Linear resample onto a fixed grid for sparkline overlay.
    src_t = np.linspace(0.0, 1.0, finite.size)
    dst_t = np.linspace(0.0, 1.0, max(8, int(n)))
    return np.interp(dst_t, src_t, finite)


def draw_dual_sparkline(
    canvas: tk.Canvas,
    series_a: Sequence[float] | np.ndarray | None,
    series_b: Sequence[float] | np.ndarray | None,
    *,
    color_a: str = _COLOR_A,
    color_b: str = _COLOR_B,
) -> None:
    """Paint A/B mini trajectories into an existing canvas (visual-first cue)."""
    if getattr(canvas, "_spark_drawing", False):
        return
    setattr(canvas, "_spark_drawing", True)
    try:
        canvas.delete("all")
        w = max(int(canvas.winfo_width()), 120)
        h = max(int(canvas.winfo_height()), 36)
        # Only resize when needed — configure(width/height) re-fires <Configure>
        # and would recurse through the spark paint handler.
        try:
            cur_w = int(float(canvas.cget("width") or 0))
            cur_h = int(float(canvas.cget("height") or 0))
        except (tk.TclError, TypeError, ValueError):
            cur_w, cur_h = 0, 0
        if cur_w != w or cur_h != h:
            canvas.configure(width=w, height=h)
        pad_x, pad_y = 4, 4
        a = _resample_series(series_a)
        b = _resample_series(series_b)
        if a is None and b is None:
            canvas.create_text(
                w // 2,
                h // 2,
                text="Load A & B for trajectory overlay",
                fill=MUTED,
                font=FONT_UI_XS,
            )
            return

        stacks = [s for s in (a, b) if s is not None]
        lo = float(min(float(np.nanmin(s)) for s in stacks))
        hi = float(max(float(np.nanmax(s)) for s in stacks))
        if not math.isfinite(lo) or not math.isfinite(hi) or abs(hi - lo) < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        span = hi - lo

        def _poly(series: np.ndarray, color: str, width: int = 2) -> None:
            n = series.size
            pts: list[float] = []
            for i, y in enumerate(series):
                x = pad_x + (w - 2 * pad_x) * (i / max(n - 1, 1))
                yy = pad_y + (h - 2 * pad_y) * (1.0 - (float(y) - lo) / span)
                pts.extend((x, yy))
            if len(pts) >= 4:
                canvas.create_line(*pts, fill=color, width=width, smooth=True)

        if a is not None:
            _poly(a, color_a, 2)
        if b is not None:
            _poly(b, color_b, 2)
    finally:
        setattr(canvas, "_spark_drawing", False)


def _horizontal_delta_bar(
    parent: tk.Misc,
    *,
    delta: float | None,
    scale: float,
    color: str,
    bg: str = SURFACE,
) -> tk.Frame:
    """Centered horizontal bar: negative ← | → positive."""
    host = tk.Frame(parent, bg=bg, height=12, highlightthickness=0)
    host.pack(fill=tk.X)
    host.pack_propagate(False)
    track = tk.Frame(host, bg=BORDER, height=8, highlightthickness=0)
    track.place(relx=0, rely=0.5, relwidth=1.0, height=8, anchor="w")
    spine = tk.Frame(host, bg=MUTED_DIM, width=2, highlightthickness=0)
    spine.place(relx=0.5, rely=0.5, relheight=1.0, anchor="center")

    if delta is None or scale <= 1e-12 or not math.isfinite(float(delta)):
        return host

    frac = max(-1.0, min(1.0, float(delta) / scale))
    fill = tk.Frame(host, bg=color, height=8, highlightthickness=0)
    if frac >= 0:
        fill.place(relx=0.5, rely=0.5, relwidth=0.5 * frac, relheight=0.65, anchor="w")
    else:
        fill.place(relx=0.5, rely=0.5, relwidth=0.5 * abs(frac), relheight=0.65, anchor="e")
    return host


def install_visual_diff_chrome(
    frame: tk.Misc,
    *,
    bg: str = SURFACE,
) -> dict[str, Any]:
    """
    Replace text-table chrome with visual-first difference panel widgets.

    Returns handles: spark_canvas, legend, diff_host
    """
    for child in frame.winfo_children():
        child.destroy()

    tk.Label(
        frame,
        text="Differences (Δ = B − A)",
        bg=bg,
        fg=TEXT,
        font=FONT_HEADING,
        anchor="w",
    ).pack(fill=tk.X, padx=PAD_SM, pady=(PAD_SM, PAD_XS // 2))

    tk.Label(
        frame,
        text="Green: B higher · Red: B lower",
        bg=bg,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_XS))

    spark_wrap = tk.Frame(frame, bg=bg, highlightthickness=0)
    spark_wrap.pack(fill=tk.X, padx=PAD_SM, pady=(0, PAD_XS))
    tk.Label(
        spark_wrap,
        text="Synchronized knee trajectories",
        bg=bg,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="w",
    ).pack(fill=tk.X)
    spark = tk.Canvas(
        spark_wrap,
        bg=bg,
        highlightthickness=1,
        highlightbackground=BORDER,
        height=44,
    )
    spark.pack(fill=tk.X, pady=(2, 0))

    legend = tk.Frame(frame, bg=bg, highlightthickness=0)
    legend.pack(fill=tk.X, padx=PAD_SM, pady=(2, PAD_XS))
    for color, label in (
        (_COLOR_A, "Session A"),
        (_COLOR_B, "Session B"),
        (SUCCESS, "B better"),
        (DANGER, "B worse"),
    ):
        chip = tk.Frame(legend, bg=bg, highlightthickness=0)
        chip.pack(side=tk.LEFT, padx=(0, PAD_SM))
        swatch = tk.Frame(chip, bg=color, width=10, height=10, highlightthickness=0)
        swatch.pack(side=tk.LEFT, padx=(0, PAD_XS))
        swatch.pack_propagate(False)
        tk.Label(chip, text=label, bg=bg, fg=MUTED, font=FONT_UI_XS).pack(side=tk.LEFT)

    diff_host = tk.Frame(frame, bg=bg, highlightthickness=0)
    diff_host.pack(fill=tk.BOTH, expand=True, padx=PAD_SM, pady=(0, PAD_SM))

    return {
        "spark_canvas": spark,
        "diff_host": diff_host,
        "legend": legend,
    }


def fill_visual_diffs(
    host: tk.Misc,
    result: SessionComparisonResult | None,
    *,
    identical: bool = False,
    priority: Sequence[str] = (),
    max_rows: int = 8,
    series_a: Sequence[float] | np.ndarray | None = None,
    series_b: Sequence[float] | np.ndarray | None = None,
    spark_canvas: tk.Canvas | None = None,
    bg: str = SURFACE,
) -> None:
    """Populate the difference host with bar / % / indicator rows."""
    # Skip destroy/rebuild when the visible Δ payload is unchanged (Compare reopen).
    fp = (
        identical,
        None
        if result is None
        else tuple(
            (d.name, d.delta, d.left_value, d.right_value, d.tone_right)
            for d in result.diffs[: max(0, int(max_rows) + 4)]
        ),
        id(series_a) if series_a is not None else None,
        id(series_b) if series_b is not None else None,
    )
    if getattr(host, "_visual_diff_fp", None) == fp and host.winfo_children():
        return
    setattr(host, "_visual_diff_fp", fp)

    for child in host.winfo_children():
        child.destroy()

    if spark_canvas is not None:
        # Capture series on the canvas so one durable Configure handler can
        # redraw without stacking bind("+") handlers on every refresh.
        spark_canvas._spark_series = (series_a, series_b)  # type: ignore[attr-defined]

        def _paint(_event: object | None = None) -> None:
            if getattr(spark_canvas, "_spark_drawing", False):
                return
            stored = getattr(spark_canvas, "_spark_series", (None, None))
            draw_dual_sparkline(spark_canvas, stored[0], stored[1])

        if not getattr(spark_canvas, "_spark_configure_bound", False):
            spark_canvas.bind("<Configure>", _paint, add="+")
            setattr(spark_canvas, "_spark_configure_bound", True)
            setattr(spark_canvas, "_spark_paint", _paint)
        else:
            setattr(spark_canvas, "_spark_paint", _paint)
        try:
            spark_canvas.after_idle(_paint)
        except tk.TclError:
            _paint()

    if identical:
        tk.Label(
            host,
            text="Both comparison sessions are identical.",
            bg=bg,
            fg=WARNING,
            font=FONT_UI_SM,
            wraplength=320,
            justify=tk.LEFT,
        ).pack(anchor="w")
        return
    if result is None:
        tk.Label(
            host,
            text=SESSION_B_METRICS_PLACEHOLDER_TITLE,
            bg=bg,
            fg=TEXT,
            font=FONT_UI_SEMIBOLD,
            wraplength=320,
            justify=tk.LEFT,
            anchor="w",
        ).pack(anchor="w")
        tk.Label(
            host,
            text=(
                "Analyze or load Session B to show Difference Summary, "
                "percent differences, and green/red improvement indicators."
            ),
            bg=bg,
            fg=MUTED,
            font=FONT_UI_SM,
            wraplength=320,
            justify=tk.LEFT,
            anchor="w",
        ).pack(anchor="w", pady=(PAD_XS, 0))
        return

    pri = set(priority)
    ordered = [d for d in result.diffs if d.name in priority]
    ordered.extend(d for d in result.diffs if d.name not in pri)

    rows: list[MetricDiff] = []
    for d in ordered:
        if d.name == "Stability" and d.delta is None:
            continue
        if d.delta is None and d.name not in ("Stability",):
            continue
        rows.append(d)
        if len(rows) >= max_rows:
            break

    scale = max((abs(float(d.delta)) for d in rows if d.delta is not None), default=1.0)
    scale = max(scale, 1e-6)

    for d in rows:
        color = tone_color(d.tone_right)
        glyph = tone_glyph(d.tone_right, d.delta)
        pct = pct_change(d.left_value, d.right_value)
        unit = d.unit or ""

        block = tk.Frame(host, bg=bg, highlightthickness=0)
        block.pack(fill=tk.X, pady=(0, PAD_SM))

        top = tk.Frame(block, bg=bg, highlightthickness=0)
        top.pack(fill=tk.X)
        tk.Label(
            top,
            text=glyph,
            bg=bg,
            fg=color,
            font=FONT_UI_SM,
            width=2,
            anchor="w",
        ).pack(side=tk.LEFT)
        short = d.name.replace(" difference", "").replace("Difference", "").strip()
        tk.Label(
            top,
            text=short,
            bg=bg,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        pct_txt = format_pct(pct)
        if pct_txt:
            tk.Label(
                top,
                text=pct_txt,
                bg=bg,
                fg=color,
                font=FONT_UI_SM,
                anchor="e",
            ).pack(side=tk.RIGHT)

        bar_row = tk.Frame(block, bg=bg, highlightthickness=0)
        bar_row.pack(fill=tk.X, pady=(2, 0))
        bar_host = tk.Frame(bar_row, bg=bg, height=14, highlightthickness=0)
        bar_host.pack(fill=tk.X)
        _horizontal_delta_bar(
            bar_host,
            delta=d.delta,
            scale=scale,
            color=color,
            bg=bg,
        )

        nums = tk.Frame(block, bg=bg, highlightthickness=0)
        nums.pack(fill=tk.X, pady=(2, 0))
        a_txt = format_delta(d.left_value, unit).lstrip("+")
        b_txt = format_delta(d.right_value, unit).lstrip("+")
        d_txt = format_delta(d.delta, unit)
        tk.Label(
            nums,
            text=f"A  {a_txt}    B  {b_txt}    Δ  {d_txt}",
            bg=bg,
            fg=TEXT_SECONDARY,
            font=FONT_UI_XS,
            anchor="w",
        ).pack(side=tk.LEFT)


# Always-visible Difference Summary (Δ = B − A). Educational directionality:
# higher Session B values are treated as improvements for these five metrics.
DIFFERENCE_SUMMARY_SPECS: tuple[tuple[str, str, str, str, str], ...] = (
    # title, CompareMetrics attr, unit, format, directionality
    (
        "Cadence difference",
        "cadence_spm",
        f" {UNIT_CADENCE}",
        "{:.0f}",
        "higher_better",
    ),
    ("Speed difference", "walking_speed_m_s", " m/s", "{:.2f}", "higher_better"),
    ("ROM difference", "knee_rom_deg", "°", "{:.1f}", "higher_better"),
    ("Symmetry difference", "symmetry_pct", "%", "{:.0f}", "higher_better"),
    ("Stability difference", "stability_margin_m", " m", "{:.3f}", "higher_better"),
)


def _summary_tone(delta: float | None, *, directionality: str) -> str:
    if delta is None or not math.isfinite(float(delta)) or abs(float(delta)) < 1e-9:
        return "neutral"
    if directionality == "higher_better":
        return "better" if float(delta) > 0 else "worse"
    if directionality == "lower_better":
        return "better" if float(delta) < 0 else "worse"
    return "neutral"


def build_session_b_metrics_placeholder(
    parent: tk.Misc,
    *,
    bg: str = SURFACE,
    title: str = SESSION_B_METRICS_PLACEHOLDER_TITLE,
    body: str = SESSION_B_METRICS_PLACEHOLDER_BODY,
) -> tk.Frame:
    """Single explanatory card used instead of blank Δ / metric values."""
    card = tk.Frame(
        parent,
        bg=SURFACE,
        highlightthickness=1,
        highlightbackground=BORDER,
    )
    tk.Label(
        card,
        text=title,
        bg=SURFACE,
        fg=TEXT,
        font=FONT_UI_SEMIBOLD,
        anchor="center",
        justify=tk.CENTER,
    ).pack(fill=tk.X, padx=PAD_MD, pady=(PAD_MD, PAD_XS))
    tk.Label(
        card,
        text=body,
        bg=SURFACE,
        fg=MUTED,
        font=FONT_UI_SM,
        anchor="center",
        justify=tk.LEFT,
        wraplength=720,
    ).pack(fill=tk.X, padx=PAD_MD, pady=(0, PAD_MD))
    del bg
    return card


def _set_difference_summary_mode(
    handles: dict[str, Any],
    *,
    show_metrics: bool,
) -> None:
    """Toggle Difference Summary metric grid vs Session B placeholder card."""
    grid = handles.get("grid")
    placeholder = handles.get("placeholder")
    legend = handles.get("legend")
    handles["mode"] = "metrics" if show_metrics else "placeholder"
    if show_metrics:
        if placeholder is not None:
            try:
                placeholder.pack_forget()
            except tk.TclError:
                pass
        if grid is not None:
            try:
                grid.pack(fill=tk.X)
            except tk.TclError:
                pass
        if legend is not None:
            try:
                legend.configure(
                    text="Δ = B − A  ·  % vs A  ·  Green = improvement  ·  Red = deterioration"
                )
            except tk.TclError:
                pass
        return
    if grid is not None:
        try:
            grid.pack_forget()
        except tk.TclError:
            pass
    if placeholder is not None:
        try:
            placeholder.pack(fill=tk.X)
        except tk.TclError:
            pass
    if legend is not None:
        try:
                legend.configure(text=SESSION_B_WAITING_TITLE)
        except tk.TclError:
            pass


def difference_summary_column_count(width: int, n: int = 5) -> int:
    """How many summary cards fit in one row without crushing labels."""
    n = max(1, int(n))
    w = max(0, int(width))
    if w < 640:
        return min(2, n)
    if w < 980:
        return min(3, n)
    return n


def install_difference_summary_panel(
    parent: tk.Misc,
    *,
    bg: str = SURFACE,
) -> dict[str, Any]:
    """Difference Summary strip — five Δ metrics that wrap instead of compressing."""
    shell = tk.Frame(parent, bg=bg, highlightthickness=0)
    shell.pack(fill=tk.X)

    header = tk.Frame(shell, bg=bg, highlightthickness=0)
    header.pack(fill=tk.X, pady=(0, PAD_SM))
    tk.Label(
        header,
        text="Difference Summary",
        bg=bg,
        fg=TEXT,
        font=FONT_HEADING,
        anchor="w",
    ).pack(side=tk.LEFT)
    legend = tk.Label(
        header,
        text=SESSION_B_WAITING_TITLE,
        bg=bg,
        fg=MUTED,
        font=FONT_UI_XS,
        anchor="e",
    )
    legend.pack(side=tk.RIGHT)

    placeholder = build_session_b_metrics_placeholder(shell, bg=bg)
    placeholder.pack(fill=tk.X)

    grid = tk.Frame(shell, bg=bg, highlightthickness=0)
    cells: dict[str, dict[str, tk.Label]] = {}
    cell_frames: list[tk.Frame] = []
    for col, (title, _attr, _unit, _fmt, _dir) in enumerate(DIFFERENCE_SUMMARY_SPECS):
        grid.columnconfigure(col, weight=1, minsize=148)
        cell = tk.Frame(
            grid,
            bg=SURFACE,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        cell.grid(
            row=0,
            column=col,
            sticky="nsew",
            padx=(0 if col == 0 else PAD_SM, 0),
            pady=0,
        )
        cell_frames.append(cell)
        title_lbl = tk.Label(
            cell,
            text=title,
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
            wraplength=160,
            justify=tk.LEFT,
        )
        title_lbl.pack(anchor="w", padx=PAD_SM, pady=(PAD_SM, 0))
        delta_lbl = tk.Label(
            cell,
            text="",
            bg=SURFACE,
            fg=TEXT,
            font=FONT_UI_SM,
            anchor="w",
        )
        delta_lbl.pack(anchor="w", padx=PAD_SM, pady=(4, 0))
        pct_lbl = tk.Label(
            cell,
            text="",
            bg=SURFACE,
            fg=MUTED,
            font=FONT_UI_XS,
            anchor="w",
        )
        pct_lbl.pack(anchor="w", padx=PAD_SM, pady=(2, 0))
        detail_lbl = tk.Label(
            cell,
            text="",
            bg=SURFACE,
            fg=MUTED_DIM,
            font=FONT_UI_XS,
            anchor="w",
            wraplength=160,
            justify=tk.LEFT,
        )
        detail_lbl.pack(anchor="w", padx=PAD_SM, pady=(0, PAD_SM))
        cells[title] = {
            "delta": delta_lbl,
            "pct": pct_lbl,
            "detail": detail_lbl,
            "title": title_lbl,
        }

    handles = {
        "shell": shell,
        "grid": grid,
        "placeholder": placeholder,
        "legend": legend,
        "cells": cells,
        "cell_frames": cell_frames,
        "mode": "placeholder",
        "_wrap_cols": len(cell_frames),
    }

    def _flow(_event: object | None = None) -> None:
        try:
            width = int(grid.winfo_width() or 0)
        except tk.TclError:
            return
        n = len(cell_frames)
        if n == 0 or width < 40:
            return
        cols = difference_summary_column_count(width, n)
        if cols == handles.get("_wrap_cols"):
            return
        handles["_wrap_cols"] = cols
        wrap = max(96, (width // cols) - 28)
        for c in range(n):
            grid.columnconfigure(c, weight=1 if c < cols else 0, minsize=148 if c < cols else 0)
        for index, cell in enumerate(cell_frames):
            r, c = divmod(index, cols)
            cell.grid(
                row=r,
                column=c,
                sticky="nsew",
                padx=(0 if c == 0 else PAD_SM, 0),
                pady=(0 if r == 0 else PAD_SM, 0),
            )
        for pair in cells.values():
            title_lbl = pair.get("title")
            detail_lbl = pair.get("detail")
            if title_lbl is not None:
                try:
                    title_lbl.configure(wraplength=wrap)
                except tk.TclError:
                    pass
            if detail_lbl is not None:
                try:
                    detail_lbl.configure(wraplength=wrap)
                except tk.TclError:
                    pass

    grid.bind("<Configure>", _flow, add="+")
    return handles


def fill_difference_summary(
    handles: dict[str, Any] | None,
    left: Any | None,
    right: Any | None,
    *,
    identical: bool = False,
    ready: bool | None = None,
) -> None:
    """Refresh Difference Summary cells from Session A/B metrics (auto Δ = B − A)."""
    if not handles:
        return
    cells: dict[str, dict[str, tk.Label]] = handles.get("cells") or {}
    if not cells:
        return

    both_present = left is not None and right is not None
    if ready is None:
        ready = both_present and not identical

    if identical:
        _set_difference_summary_mode(handles, show_metrics=True)
        for title, _attr, _unit, _fmt, _dir in DIFFERENCE_SUMMARY_SPECS:
            pair = cells.get(title)
            if pair is None:
                continue
            pair["delta"].configure(text="Δ 0 (identical)", fg=MUTED)
            pct_lbl = pair.get("pct")
            if pct_lbl is not None:
                pct_lbl.configure(text="0%", fg=MUTED)
            pair["detail"].configure(text="Sessions are identical")
        return

    if not ready:
        # Replace empty Δ cells with one clear requirement card.
        _set_difference_summary_mode(handles, show_metrics=False)
        return

    _set_difference_summary_mode(handles, show_metrics=True)

    for title, attr, unit, _fmt, directionality in DIFFERENCE_SUMMARY_SPECS:
        pair = cells.get(title)
        if pair is None:
            continue
        delta_lbl = pair["delta"]
        pct_lbl = pair.get("pct")
        detail_lbl = pair["detail"]
        a_raw = getattr(left, attr, None) if left is not None else None
        b_raw = getattr(right, attr, None) if right is not None else None
        try:
            a_val = float(a_raw) if a_raw is not None else None
        except (TypeError, ValueError):
            a_val = None
        try:
            b_val = float(b_raw) if b_raw is not None else None
        except (TypeError, ValueError):
            b_val = None
        if a_val is None or b_val is None or not (
            math.isfinite(a_val) and math.isfinite(b_val)
        ):
            from stablewalk.ui.unavailable_labels import compare_unavailable_label

            left_ok = a_val is not None and math.isfinite(float(a_val))
            right_ok = b_val is not None and math.isfinite(float(b_val))
            label = compare_unavailable_label(
                title,
                left_available=left_ok,
                right_available=right_ok,
                left_session_ready=left is not None,
                right_session_ready=right is not None,
            )
            delta_lbl.configure(text=label, fg=MUTED)
            if pct_lbl is not None:
                pct_lbl.configure(text="", fg=MUTED)
            # Detail explains which side is missing without repeating the primary label.
            if not left_ok and not right_ok:
                detail_lbl.configure(text="Not present in either session")
            elif not left_ok:
                detail_lbl.configure(
                    text=(
                        "Metric unavailable for this recording"
                        if left is not None
                        else "Not present in Session A"
                    )
                )
            else:
                detail_lbl.configure(
                    text=(
                        "Metric unavailable for this recording"
                        if right is not None
                        else "Not present in Session B"
                    )
                )
            continue
        delta = b_val - a_val
        tone = _summary_tone(delta, directionality=directionality)
        color = tone_color(tone)
        glyph = tone_glyph(tone, delta)
        pct = pct_change(a_val, b_val)
        pct_txt = format_pct(pct)
        delta_lbl.configure(
            text=f"{glyph} {format_delta(delta, unit)}",
            fg=color,
        )
        if pct_lbl is not None:
            pct_lbl.configure(
                text=pct_txt if pct_txt else "n/a vs A",
                fg=color if pct_txt else MUTED,
            )
        a_txt = format_delta(a_val, unit).lstrip("+")
        b_txt = format_delta(b_val, unit).lstrip("+")
        detail_lbl.configure(text=f"A {a_txt} → B {b_txt}")


__all__ = [
    "DIFFERENCE_SUMMARY_SPECS",
    "build_session_b_metrics_placeholder",
    "draw_dual_sparkline",
    "difference_summary_column_count",
    "fill_difference_summary",
    "fill_visual_diffs",
    "format_delta",
    "format_pct",
    "install_difference_summary_panel",
    "install_visual_diff_chrome",
    "pct_change",
    "tone_color",
    "tone_glyph",
]
