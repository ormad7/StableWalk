"""
Backward compatibility shim — import from ``stablewalk.ui.motion_display`` instead.

This module re-exports the public API from the reorganized package layout.
"""

from stablewalk.ui.motion_display import *  # noqa: F403

try:
    from stablewalk.ui.motion_display import __all__ as __all__
except ImportError:
    __all__ = [n for n in dir() if not n.startswith("_")]
