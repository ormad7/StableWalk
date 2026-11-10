"""Overview primary viewers vs secondary gait-info splitter."""

from __future__ import annotations

import tkinter as tk
from types import SimpleNamespace

import pytest

from stablewalk.ui.tk.analysis_split import MIN_H_PRIMARY_VIZ, MIN_H_SECONDARY_INFO
from stablewalk.ui.tk.dashboard_overview_panes import install_overview_workspace_split
from stablewalk.ui.tk.dashboard_sections import OVERVIEW_HOST_VIZ_ROW


def _tk_root() -> tk.Tk:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")
    root.withdraw()
    return root


def test_workspace_split_puts_viewers_above_info() -> None:
    root = _tk_root()
    try:
        host = tk.Frame(root, width=1100, height=700)
        host.grid(row=0, column=0, sticky="nsew")
        host.columnconfigure(0, weight=1)
        host.rowconfigure(OVERVIEW_HOST_VIZ_ROW, weight=1)
        gui = SimpleNamespace()
        primary, inner = install_overview_workspace_split(gui, host)
        root.update_idletasks()
        split = gui._overview_main_split
        assert str(split.cget("orient")) == str(tk.VERTICAL)
        panes = [str(p) for p in split.panes()]
        assert len(panes) == 2
        assert str(primary) in panes
        assert str(gui._overview_info_outer) in panes
        assert inner is gui._overview_info_inner
        assert int(host.grid_rowconfigure(OVERVIEW_HOST_VIZ_ROW)["weight"]) == 1
        assert gui._section_visual is primary
        assert gui._primary_viz_host is primary
        assert primary.master is split
        assert gui._overview_info_outer.master is split
        assert inner.master is gui._overview_info_outer
        assert str(inner) not in panes
    finally:
        root.destroy()


def test_primary_minsize_is_larger_than_secondary() -> None:
    assert MIN_H_PRIMARY_VIZ >= 280
    assert MIN_H_SECONDARY_INFO >= 80
    assert MIN_H_PRIMARY_VIZ > MIN_H_SECONDARY_INFO


def test_tight_splitter_gives_viewers_the_leftover_height() -> None:
    from stablewalk.ui.tk.dashboard_overview_panes import (
        COMPACT_SECONDARY_INFO_H,
        DEFAULT_SECONDARY_INFO_H,
        _auto_overview_primary_height,
    )

    tight = MIN_H_PRIMARY_VIZ + COMPACT_SECONDARY_INFO_H
    primary_tight = _auto_overview_primary_height(tight + 8)
    assert primary_tight >= MIN_H_PRIMARY_VIZ
    assert (tight + 8) - primary_tight <= COMPACT_SECONDARY_INFO_H + 8

    mid = MIN_H_PRIMARY_VIZ + COMPACT_SECONDARY_INFO_H + 80
    primary_mid = _auto_overview_primary_height(mid)
    assert mid - primary_mid == COMPACT_SECONDARY_INFO_H
    assert primary_mid > 300

    roomy = MIN_H_PRIMARY_VIZ + DEFAULT_SECONDARY_INFO_H + 160
    primary_roomy = _auto_overview_primary_height(roomy)
    assert roomy - primary_roomy == DEFAULT_SECONDARY_INFO_H

    large = 800
    primary_large = _auto_overview_primary_height(large)
    assert primary_large == large - DEFAULT_SECONDARY_INFO_H
    assert primary_large >= 500


def test_available_viz_h_uses_sash_leftover_not_current_primary() -> None:
    from stablewalk.ui.tk.dashboard_overview_panes import DEFAULT_SECONDARY_INFO_H
    from stablewalk.ui.tk.dashboard_sections import overview_available_viz_h

    class _Sized:
        def __init__(self, height: int) -> None:
            self._height = height

        def winfo_height(self) -> int:
            return self._height

    primary = _Sized(316)
    gui = SimpleNamespace(
        _overview_main_split=_Sized(800),
        _primary_viz_host=primary,
        _section_visual=primary,
        _overview_v_sash_user=False,
        _overview_v_sash_frac=None,
    )
    available = overview_available_viz_h(gui)
    assert available == 800 - DEFAULT_SECONDARY_INFO_H
    assert available > primary.winfo_height()
