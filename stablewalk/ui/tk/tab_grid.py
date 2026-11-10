"""Shared grid helpers for dashboard tabs (one manager per parent)."""

from __future__ import annotations

import tkinter as tk


def stretch_columns(host: tk.Misc, *columns: int, minsize: int | None = None) -> None:
    """Give listed columns leftover width (default: column 0).

    Only the stretch weight is updated. Existing minsize / uniform values stay
    unless ``minsize`` is passed explicitly.
    """
    cols = columns or (0,)
    for col in cols:
        if minsize is None:
            host.columnconfigure(col, weight=1)
        else:
            host.columnconfigure(col, weight=1, minsize=minsize)


def stretch_rows(host: tk.Misc, *rows: int, minsize: int | None = None) -> None:
    """Give listed rows leftover height (default: row 0).

    Only the stretch weight is updated. Existing minsize / uniform values stay
    unless ``minsize`` is passed explicitly.
    """
    rws = rows or (0,)
    for row in rws:
        if minsize is None:
            host.rowconfigure(row, weight=1)
        else:
            host.rowconfigure(row, weight=1, minsize=minsize)


def compact_rows(host: tk.Misc, *rows: int) -> None:
    """Keep listed rows at their natural (header / toolbar / footer) size."""
    for row in rows:
        host.rowconfigure(row, weight=0)


def compact_columns(host: tk.Misc, *columns: int) -> None:
    """Keep listed columns at their natural size."""
    for col in columns:
        host.columnconfigure(col, weight=0)


def grid_fill(
    widget: tk.Misc,
    *,
    row: int = 0,
    column: int = 0,
    rowspan: int = 1,
    columnspan: int = 1,
    padx: int | tuple[int, int] = 0,
    pady: int | tuple[int, int] = 0,
    sticky: str = "nsew",
) -> None:
    """Place ``widget`` with grid so it participates in parent weights."""
    widget.grid(
        row=row,
        column=column,
        rowspan=rowspan,
        columnspan=columnspan,
        sticky=sticky,
        padx=padx,
        pady=pady,
    )


def grid_ew(
    widget: tk.Misc,
    *,
    row: int = 0,
    column: int = 0,
    columnspan: int = 1,
    padx: int | tuple[int, int] = 0,
    pady: int | tuple[int, int] = 0,
) -> None:
    """Place a chrome row that stretches horizontally only."""
    grid_fill(
        widget,
        row=row,
        column=column,
        columnspan=columnspan,
        padx=padx,
        pady=pady,
        sticky="ew",
    )
