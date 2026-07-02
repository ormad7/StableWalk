"""
Backward compatibility shim — import from ``stablewalk.pose.video_source`` instead.

This module re-exports the public API from the reorganized package layout.
"""

from stablewalk.pose.video_source import *  # noqa: F403

try:
    from stablewalk.pose.video_source import __all__ as __all__
except ImportError:
    __all__ = [n for n in dir() if not n.startswith("_")]
