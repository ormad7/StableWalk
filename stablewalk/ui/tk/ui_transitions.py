"""Smooth UI transitions after analysis without affecting playback sync."""

from __future__ import annotations

import re
import tkinter as tk
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from stablewalk.ui.theme import ACCENT, BG, BORDER, ELEVATED, PANEL, SURFACE

_ANIM_JOB = "_sw_anim_job"
_ANIM_BAR_JOB = "_sw_anim_bar_job"
_REVEAL_STASH = "_sw_reveal_stash"

_FADE_SHADES = (BG, SURFACE, PANEL, PANEL)
_NUMERIC_RE = re.compile(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?")


def should_animate_analysis_ui(gui) -> bool:
    """True only during the one-shot post-analysis reveal, never while playing."""
    if getattr(gui, "playing", False):
        return False
    return bool(getattr(gui, "_analysis_reveal_active", False))


def abort_analysis_reveal(gui) -> None:
    """Cancel in-flight reveal animations and restore hidden widgets."""
    gui._analysis_reveal_active = False
    gui._analysis_reveal_token = int(getattr(gui, "_analysis_reveal_token", 0)) + 1
    restore_reveal_cards(collect_analysis_reveal_cards(gui))


def _cancel_after(widget: tk.Misc, attr: str) -> None:
    job = getattr(widget, attr, None)
    if job is None:
        return
    try:
        widget.after_cancel(job)
    except tk.TclError:
        pass
    setattr(widget, attr, None)


def extract_leading_number(text: str) -> tuple[float | None, str, str]:
    """Return ``(value, prefix, suffix)`` for the first number in *text*."""
    match = _NUMERIC_RE.search(text)
    if match is None:
        return None, text, ""
    prefix = text[: match.start()]
    suffix = text[match.end() :]
    try:
        return float(match.group()), prefix, suffix
    except ValueError:
        return None, text, ""


def _infer_decimals(target_text: str) -> int:
    match = re.search(r"\.(\d+)", target_text)
    if match is not None:
        return len(match.group(1))
    if "%" in target_text or "/min" in target_text:
        return 0
    if " s" in target_text or "m/s" in target_text:
        return 2
    return 1


def format_interpolated_number(
    prefix: str,
    value: float,
    suffix: str,
    *,
    decimals: int,
) -> str:
    if decimals <= 0:
        shown = f"{round(value)}"
    else:
        shown = f"{value:.{decimals}f}"
    return f"{prefix}{shown}{suffix}"


def animate_label_text(
    label: tk.Label,
    target_text: str,
    *,
    animate: bool,
    apply_fn: Callable[[str], None] | None = None,
    duration_ms: int = 200,
    steps: int = 8,
) -> None:
    """Update label text, optionally tweening the first numeric portion."""
    _cancel_after(label, _ANIM_JOB)

    def apply(text: str) -> None:
        if apply_fn is not None:
            apply_fn(text)
        else:
            label.configure(text=text)

    current = label.cget("text")
    if not animate or current == target_text:
        apply(target_text)
        return

    old_val, old_pre, old_suf = extract_leading_number(current)
    new_val, new_pre, new_suf = extract_leading_number(target_text)
    if (
        old_val is None
        or new_val is None
        or old_pre != new_pre
        or old_suf != new_suf
        or abs(old_val - new_val) < 1e-9
    ):
        apply(target_text)
        return

    step_ms = max(16, duration_ms // steps)
    decimals = _infer_decimals(target_text)
    state = {"step": 0}

    def tick() -> None:
        state["step"] += 1
        t = min(1.0, state["step"] / steps)
        eased = 1.0 - (1.0 - t) ** 2
        value = old_val + (new_val - old_val) * eased
        if state["step"] >= steps:
            apply(target_text)
            setattr(label, _ANIM_JOB, None)
            return
        apply(
            format_interpolated_number(
                new_pre,
                value,
                new_suf,
                decimals=decimals,
            )
        )
        setattr(label, _ANIM_JOB, label.after(step_ms, tick))

    tick()


def animate_bar_width(
    fill: tk.Frame,
    target_fraction: float,
    *,
    animate: bool,
    available: bool,
    color: str | None = None,
    duration_ms: int = 220,
    steps: int = 8,
) -> None:
    """Smoothly update a progress bar ``relwidth``."""
    _cancel_after(fill, _ANIM_BAR_JOB)
    frac = max(0.0, min(1.0, target_fraction))
    bar_color = color if color is not None else (ACCENT if available else BORDER)
    fill.configure(bg=bar_color)

    try:
        current = float(fill.place_info().get("relwidth", 0.0) or 0.0)
    except tk.TclError:
        current = 0.0

    if not animate or abs(current - frac) < 1e-4:
        fill.place_configure(relwidth=frac)
        return

    step_ms = max(16, duration_ms // steps)
    state = {"step": 0}

    def tick() -> None:
        state["step"] += 1
        t = min(1.0, state["step"] / steps)
        eased = 1.0 - (1.0 - t) ** 2
        width = current + (frac - current) * eased
        if state["step"] >= steps:
            fill.place_configure(relwidth=frac)
            setattr(fill, _ANIM_BAR_JOB, None)
            return
        fill.place_configure(relwidth=width)
        setattr(fill, _ANIM_BAR_JOB, fill.after(step_ms, tick))

    tick()


def _widget_bg(widget: tk.Misc) -> str | None:
    try:
        return str(widget.cget("bg"))
    except tk.TclError:
        return None


def _collect_fade_targets(frame: tk.Misc, *, max_depth: int = 3) -> list[tuple[tk.Misc, str]]:
    targets: list[tuple[tk.Misc, str]] = []

    def walk(widget: tk.Misc, depth: int) -> None:
        if depth > max_depth:
            return
        bg = _widget_bg(widget)
        if bg in {PANEL, ELEVATED, SURFACE}:
            targets.append((widget, bg))
        if depth < max_depth:
            for child in widget.winfo_children():
                walk(child, depth + 1)

    walk(frame, 0)
    return targets


def fade_in_frame(
    frame: tk.Misc,
    *,
    root: tk.Misc,
    on_done: Callable[[], None] | None = None,
    step_ms: int = 40,
) -> None:
    """Simulate a fade-in by stepping panel backgrounds from darker to normal."""
    targets = _collect_fade_targets(frame)

    if not targets:
        if on_done is not None:
            on_done()
        return

    shades = list(_FADE_SHADES)
    state = {"step": 0}

    def apply_shade(index: int) -> None:
        shade = shades[min(index, len(shades) - 1)]
        for widget, final_bg in targets:
            widget.configure(bg=shade if index < len(shades) - 1 else final_bg)

    def step() -> None:
        apply_shade(state["step"])
        state["step"] += 1
        if state["step"] >= len(shades):
            for widget, final_bg in targets:
                widget.configure(bg=final_bg)
            if on_done is not None:
                on_done()
            return
        root.after(step_ms, step)

    step()


def _stash_widget(widget: tk.Misc) -> dict[str, Any] | None:
    try:
        manager = widget.winfo_manager()
    except tk.TclError:
        return None
    if manager == "grid":
        info = widget.grid_info()
        widget.grid_remove()
        return {"manager": "grid", "info": info}
    if manager == "pack":
        info = widget.pack_info()
        widget.pack_forget()
        return {"manager": "pack", "info": info}
    return None


def _restore_widget(widget: tk.Misc, stash: dict[str, Any] | None) -> None:
    if not stash:
        return
    try:
        if stash["manager"] == "grid":
            widget.grid(**stash["info"])
        elif stash["manager"] == "pack":
            widget.pack(**stash["info"])
    except tk.TclError:
        pass


def hide_for_reveal(cards: Iterable[tk.Misc]) -> None:
    for card in cards:
        stash = _stash_widget(card)
        if stash is not None:
            setattr(card, _REVEAL_STASH, stash)


def restore_reveal_cards(cards: Iterable[tk.Misc]) -> None:
    for card in cards:
        stash = getattr(card, _REVEAL_STASH, None)
        _restore_widget(card, stash)
        if hasattr(card, _REVEAL_STASH):
            delattr(card, _REVEAL_STASH)


def _pulse_card(card: tk.Misc, *, root: tk.Misc) -> None:
    try:
        original_thickness = int(card.cget("highlightthickness") or 0)
        original_border = str(card.cget("highlightbackground") or BORDER)
    except tk.TclError:
        return

    pulse = [ACCENT, original_border, original_border]
    state = {"step": 0}

    def step() -> None:
        if state["step"] < len(pulse):
            try:
                card.configure(
                    highlightbackground=pulse[state["step"]],
                    highlightthickness=max(original_thickness, 2),
                )
            except tk.TclError:
                return
            state["step"] += 1
            root.after(55, step)
        else:
            try:
                card.configure(
                    highlightbackground=original_border,
                    highlightthickness=original_thickness,
                )
            except tk.TclError:
                pass

    step()


def reveal_cards_staggered(
    cards: Sequence[tk.Misc],
    *,
    root: tk.Misc,
    delay_ms: int = 50,
    on_done: Callable[[], None] | None = None,
) -> None:
    """Restore hidden cards one-by-one with a subtle highlight pulse."""
    visible = [card for card in cards if getattr(card, _REVEAL_STASH, None) is not None]
    if not visible:
        if on_done is not None:
            on_done()
        return

    index = {"i": 0}

    def reveal_next() -> None:
        i = index["i"]
        if i >= len(visible):
            if on_done is not None:
                on_done()
            return
        card = visible[i]
        stash = getattr(card, _REVEAL_STASH, None)
        _restore_widget(card, stash)
        if hasattr(card, _REVEAL_STASH):
            delattr(card, _REVEAL_STASH)
        _pulse_card(card, root=root)
        index["i"] += 1
        root.after(delay_ms, reveal_next)

    reveal_next()


def collect_analysis_reveal_cards(gui) -> list[tk.Misc]:
    cards: list[tk.Misc] = []
    cards.extend(getattr(gui, "_summary_metric_cards", {}).values())
    cards.extend(getattr(gui, "_summary_event_cards", {}).values())
    cards.extend(getattr(gui, "_overview_score_cards", []))
    cards.extend(getattr(gui, "_motion_metric_rows", {}).values())
    return [card for card in cards if card is not None]


def analysis_tab_frames(gui) -> list[tk.Misc]:
    frames = [
        getattr(gui, "_tab_overview", None),
        getattr(gui, "_tab_motion", None),
        getattr(gui, "_tab_biomechanics", None),
        getattr(gui, "_tab_results_summary", None),
        getattr(gui, "_tab_compare", None),
    ]
    return [frame for frame in frames if frame is not None]


def fade_analysis_tab_if_needed(gui, tab: tk.Misc) -> None:
    """Fade a tab the first time it is shown during post-analysis reveal."""
    if not should_animate_analysis_ui(gui):
        return
    faded = getattr(gui, "_analysis_faded_tabs", None)
    if faded is None:
        faded = set()
        gui._analysis_faded_tabs = faded
    key = str(tab)
    if key in faded:
        return
    faded.add(key)
    fade_in_frame(tab, root=gui.root)


def _refresh_panels_with_animation(gui) -> None:
    update_motion = getattr(gui, "_update_motion_temporal_metrics_panel", None)
    update_summary = getattr(gui, "_update_analysis_summary_panel", None)
    update_stability = getattr(gui, "_update_stability_panel", None)
    if update_motion is not None:
        update_motion(animate=True)
    if update_summary is not None:
        update_summary(animate=True)
    if update_stability is not None:
        biomech = getattr(gui, "_biomech", None)
        if biomech is not None:
            update_stability(biomech, animate=True)


def _finish_post_analysis_reveal(gui, token: int) -> None:
    if token != getattr(gui, "_analysis_reveal_token", 0):
        return
    _refresh_panels_with_animation(gui)
    gui._analysis_reveal_active = False


def _run_post_analysis_reveal(gui, token: int) -> None:
    if token != getattr(gui, "_analysis_reveal_token", 0):
        return
    if not getattr(gui, "sequence", None):
        gui._analysis_reveal_active = False
        return

    cards = collect_analysis_reveal_cards(gui)
    hide_for_reveal(cards)

    notebook = getattr(gui, "_dashboard_notebook", None)
    active_tab = None
    if notebook is not None:
        try:
            active_tab = notebook.nametowidget(notebook.select())
        except tk.TclError:
            active_tab = None

    def after_cards() -> None:
        if token != getattr(gui, "_analysis_reveal_token", 0):
            restore_reveal_cards(cards)
            return
        _finish_post_analysis_reveal(gui, token)

    def after_fade() -> None:
        if token != getattr(gui, "_analysis_reveal_token", 0):
            restore_reveal_cards(cards)
            return
        reveal_cards_staggered(cards, root=gui.root, on_done=after_cards)

    if active_tab is not None:
        fade_in_frame(active_tab, root=gui.root, on_done=after_fade)
        gui._analysis_faded_tabs = {str(active_tab)}
    else:
        after_fade()


def schedule_post_analysis_reveal(gui) -> None:
    """Run one-shot entrance animations after analysis data is loaded."""
    gui._analysis_reveal_active = True
    gui._analysis_reveal_token = int(getattr(gui, "_analysis_reveal_token", 0)) + 1
    token = gui._analysis_reveal_token
    gui._analysis_faded_tabs = set()
    gui.root.after_idle(lambda: _run_post_analysis_reveal(gui, token))


__all__ = [
    "abort_analysis_reveal",
    "animate_bar_width",
    "animate_label_text",
    "collect_analysis_reveal_cards",
    "extract_leading_number",
    "fade_analysis_tab_if_needed",
    "fade_in_frame",
    "format_interpolated_number",
    "reveal_cards_staggered",
    "schedule_post_analysis_reveal",
    "should_animate_analysis_ui",
]
