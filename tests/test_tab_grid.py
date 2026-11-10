"""Tab grid helper — expanding rows/columns without mixed pack+grid."""

from __future__ import annotations

import tkinter as tk

from stablewalk.ui.tk.tab_grid import (
    compact_rows,
    grid_ew,
    grid_fill,
    stretch_columns,
    stretch_rows,
)


def test_grid_fill_uses_nsew_and_weights():
    root = tk.Tk()
    root.withdraw()
    host = tk.Frame(root)
    host.pack()
    stretch_columns(host)
    stretch_rows(host)
    child = tk.Frame(host, width=10, height=10)
    grid_fill(child)
    sticky = str(child.grid_info()["sticky"])
    assert "n" in sticky and "s" in sticky and "e" in sticky and "w" in sticky
    assert int(host.grid_columnconfigure(0)["weight"]) == 1
    assert int(host.grid_rowconfigure(0)["weight"]) == 1
    bar = tk.Frame(host)
    grid_ew(bar, row=1)
    compact_rows(host, 1)
    assert "e" in str(bar.grid_info()["sticky"]) and "w" in str(bar.grid_info()["sticky"])
    assert int(host.grid_rowconfigure(1)["weight"]) == 0
    host.rowconfigure(0, minsize=96, uniform="keep")
    stretch_rows(host, 0)
    assert int(host.grid_rowconfigure(0)["minsize"]) == 96
    assert host.grid_rowconfigure(0)["uniform"] == "keep"
    root.destroy()
