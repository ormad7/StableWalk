"""
StableWalk desktop GUI — backward-compatible entry point.

The implementation lives in ``stablewalk.ui.tk.app`` (Play toolbar, larger
Positions panel, playback fixes). Use either:

  python main.py --gui
  python gui.py
  python -m stablewalk.gui_app
"""

from __future__ import annotations

from stablewalk.ui.tk.app import StableWalkGUI, launch_gui, main

__all__ = ["StableWalkGUI", "launch_gui", "main"]
