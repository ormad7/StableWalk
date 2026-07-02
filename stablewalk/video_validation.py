"""
Backward compatibility shim — import from ``stablewalk.pose.video_validation`` instead.

This module re-exports the public API from the reorganized package layout.
"""

from stablewalk.pose.video_validation import *  # noqa: F403

try:
    from stablewalk.pose.video_validation import __all__ as __all__
except ImportError:
    __all__ = [n for n in dir() if not n.startswith("_")]
