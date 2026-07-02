"""
Backward compatibility shim — import from ``stablewalk.ui.demo`` instead.

This module re-exports the public API from the reorganized package layout.
"""

from stablewalk.ui.demo import *  # noqa: F403

try:
    from stablewalk.ui.demo import __all__ as __all__
except ImportError:
    __all__ = [n for n in dir() if not n.startswith("_")]
