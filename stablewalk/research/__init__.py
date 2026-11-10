"""Research gait-video catalog and independent A/B comparison."""

from __future__ import annotations

from stablewalk.research.compare import ResearchComparison, compare_independent_clips
from stablewalk.research.dataset import ClipMetadata, ResearchCatalog, import_local_video, load_catalog
from stablewalk.research.pipeline import (
    IndependentClipResult,
    analyze_independent_clip,
    slots_are_independent,
    slots_share_source_file,
)

__all__ = [
    "ClipMetadata",
    "IndependentClipResult",
    "ResearchCatalog",
    "ResearchComparison",
    "analyze_independent_clip",
    "compare_independent_clips",
    "import_local_video",
    "load_catalog",
    "slots_are_independent",
    "slots_share_source_file",
]
