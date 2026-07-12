"""
GAVD metadata filtering for StableWalk abnormal demo selection.

GAVD provides clinical annotations only — videos must be retrieved independently
from public YouTube URLs referenced in the annotation files.

Repository: https://github.com/Rahmyyy/GAVD
"""

from __future__ import annotations

import ast
import csv
import json
import logging
import re
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from stablewalk import config

logger = logging.getLogger(__name__)

GAVD_REPO_DATA_URL = "https://raw.githubusercontent.com/Rahmyyy/GAVD/main/data"
GAVD_ANNOTATION_PARTS = tuple(
    f"GAVD_Clinical_Annotations_{i}.csv" for i in range(1, 6)
)

SUPPORT_DEVICE_PATTERNS = re.compile(
    r"walker|walking frame|rollator|cane|crutch|stick|zimmer|assistive|"
    r"walking aid|w\/ walker|with walker",
    re.IGNORECASE,
)

NON_CLINICAL_GAIT_PATTERNS = re.compile(
    r"^exercise$|^normal$|^healthy$|^control$|^demo$",
    re.IGNORECASE,
)

ACCEPTABLE_VIEWS = frozenset(
    {
        "front",
        "back",
        "left side",
        "right side",
        "left",
        "right",
        "sagittal",
        "frontal",
    }
)

OBVIOUS_BAD_VIEWS = re.compile(
    r"oblique|diagonal|overhead|top|aerial|birds?.?eye",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GavdSelectionThresholds:
    min_seq_frames: int = 90
    min_bbox_height_ratio: float = 0.35
    min_bbox_width_ratio: float = 0.08
    max_bbox_top_ratio: float = 0.45
    min_dominant_view_fraction: float = 0.85
    min_estimated_cycles: int = 3
    frames_per_cycle_estimate: int = 30


@dataclass
class GavdSequenceCandidate:
    seq: str
    youtube_id: str
    url: str
    dataset: str
    gait_pat: str
    cam_view: str
    frame_count: int
    dominant_view: str
    dominant_view_fraction: float
    mean_bbox_height_ratio: float
    mean_bbox_width_ratio: float
    estimated_cycles: int
    video_height: int
    video_width: int
    excluded_reason: str | None = None
    metadata_score: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "youtube_id": self.youtube_id,
            "url": self.url,
            "dataset": self.dataset,
            "gait_pat": self.gait_pat,
            "cam_view": self.cam_view,
            "frame_count": self.frame_count,
            "dominant_view": self.dominant_view,
            "dominant_view_fraction": round(self.dominant_view_fraction, 3),
            "mean_bbox_height_ratio": round(self.mean_bbox_height_ratio, 3),
            "mean_bbox_width_ratio": round(self.mean_bbox_width_ratio, 3),
            "estimated_cycles": self.estimated_cycles,
            "video_height": self.video_height,
            "video_width": self.video_width,
            "excluded_reason": self.excluded_reason,
            "metadata_score": round(self.metadata_score, 3),
            "notes": list(self.notes),
        }


def gavd_cache_dir() -> Path:
    return config.DEMO_VIDEOS_DIR / "_gavd_cache"


def download_gavd_annotations(
    *,
    cache_dir: Path | None = None,
    parts: Iterable[str] = GAVD_ANNOTATION_PARTS,
    force: bool = False,
) -> list[Path]:
    """Download GAVD annotation CSV parts into the demo cache."""
    out = cache_dir or gavd_cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name in parts:
        target = out / name
        if target.is_file() and not force:
            paths.append(target)
            continue
        url = f"{GAVD_REPO_DATA_URL}/{name}"
        logger.info("Downloading %s", url)
        urllib.request.urlretrieve(url, target)
        paths.append(target)
    return paths


def _parse_dict_field(value: str) -> dict[str, Any]:
    value = (value or "").strip()
    if not value:
        return {}
    try:
        parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else {}
    except (SyntaxError, ValueError):
        return {}


def _normalize_view(view: str) -> str:
    return re.sub(r"\s+", " ", (view or "").strip().lower())


def _view_ok(view: str) -> bool:
    v = _normalize_view(view)
    if not v:
        return False
    if OBVIOUS_BAD_VIEWS.search(v):
        return False
    return any(token in v for token in ACCEPTABLE_VIEWS)


def _support_device_in_text(*parts: str) -> bool:
    text = " ".join(p for p in parts if p)
    return bool(SUPPORT_DEVICE_PATTERNS.search(text))


def _iter_annotation_rows(paths: Iterable[Path]) -> Iterable[dict[str, str]]:
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield row


def build_gavd_abnormal_candidates(
    annotation_paths: Iterable[Path],
    *,
    thresholds: GavdSelectionThresholds | None = None,
) -> list[GavdSequenceCandidate]:
    """
    Filter GAVD abnormal sequences by metadata quality gates.

    Does not download YouTube videos — produces ranked candidates for manual or
    scripted retrieval followed by ``validate_demo_candidate``.
    """
    cfg = thresholds or GavdSelectionThresholds()
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _iter_annotation_rows(annotation_paths):
        grouped[row["seq"]].append(row)

    candidates: list[GavdSequenceCandidate] = []

    for seq, rows in grouped.items():
        first = rows[0]
        dataset = (first.get("dataset") or "").strip()
        gait_pat = (first.get("gait_pat") or "").strip()
        youtube_id = (first.get("id") or "").strip()
        url = (first.get("url") or "").strip()

        if "abnormal" not in dataset.lower():
            continue

        if _support_device_in_text(gait_pat, dataset, seq):
            candidates.append(
                GavdSequenceCandidate(
                    seq=seq,
                    youtube_id=youtube_id,
                    url=url,
                    dataset=dataset,
                    gait_pat=gait_pat,
                    cam_view=first.get("cam_view", ""),
                    frame_count=len(rows),
                    dominant_view="",
                    dominant_view_fraction=0.0,
                    mean_bbox_height_ratio=0.0,
                    mean_bbox_width_ratio=0.0,
                    estimated_cycles=0,
                    video_height=0,
                    video_width=0,
                    excluded_reason="assistive device or support keyword in metadata",
                )
            )
            continue

        if NON_CLINICAL_GAIT_PATTERNS.match(gait_pat.strip()):
            continue

        views = [_normalize_view(r.get("cam_view", "")) for r in rows]
        view_counts: dict[str, int] = defaultdict(int)
        for v in views:
            if v:
                view_counts[v] += 1
        if not view_counts:
            continue
        dominant_view, dominant_count = max(view_counts.items(), key=lambda x: x[1])
        dominant_fraction = dominant_count / max(len(rows), 1)

        if dominant_fraction < cfg.min_dominant_view_fraction:
            continue
        if not _view_ok(dominant_view):
            continue
        if len(rows) < cfg.min_seq_frames:
            continue

        bbox_heights: list[float] = []
        bbox_widths: list[float] = []
        bbox_tops: list[float] = []
        vid_h = vid_w = 0
        for row in rows:
            bbox = _parse_dict_field(row.get("bbox", ""))
            vid = _parse_dict_field(row.get("vid_info", ""))
            vid_h = int(vid.get("height") or vid_h or 0)
            vid_w = int(vid.get("width") or vid_w or 0)
            height = float(bbox.get("height") or 0.0)
            width = float(bbox.get("width") or 0.0)
            top = float(bbox.get("top") or 0.0)
            if vid_h > 0 and height > 0:
                bbox_heights.append(min(height / vid_h, 1.0))
            if vid_w > 0 and width > 0:
                bbox_widths.append(width / vid_w)
            if vid_h > 0 and top >= 0:
                bbox_tops.append(top / vid_h)

        if not bbox_heights:
            continue

        mean_h = sum(bbox_heights) / len(bbox_heights)
        mean_w = sum(bbox_widths) / len(bbox_widths) if bbox_widths else 0.0
        mean_top = sum(bbox_tops) / len(bbox_tops) if bbox_tops else 1.0
        estimated_cycles = max(1, len(rows) // cfg.frames_per_cycle_estimate)

        excluded = None
        notes: list[str] = []
        if mean_h < cfg.min_bbox_height_ratio:
            excluded = "subject too small in frame (bbox height ratio)"
        elif mean_w < cfg.min_bbox_width_ratio:
            excluded = "subject too narrow in frame"
        elif mean_top > cfg.max_bbox_top_ratio:
            excluded = "subject cropped toward bottom of frame"
        elif estimated_cycles < cfg.min_estimated_cycles:
            excluded = (
                f"estimated cycles {estimated_cycles} < {cfg.min_estimated_cycles}"
            )

        score = (
            mean_h * 2.0
            + dominant_fraction
            + min(estimated_cycles, 6) * 0.15
            + (0.2 if _view_ok(dominant_view) else 0.0)
        )
        if excluded:
            score *= 0.35
            notes.append(excluded)

        candidates.append(
            GavdSequenceCandidate(
                seq=seq,
                youtube_id=youtube_id,
                url=url,
                dataset=dataset,
                gait_pat=gait_pat,
                cam_view=dominant_view,
                frame_count=len(rows),
                dominant_view=dominant_view,
                dominant_view_fraction=dominant_fraction,
                mean_bbox_height_ratio=mean_h,
                mean_bbox_width_ratio=mean_w,
                estimated_cycles=estimated_cycles,
                video_height=vid_h,
                video_width=vid_w,
                excluded_reason=excluded,
                metadata_score=score,
                notes=notes,
            )
        )

    candidates.sort(
        key=lambda c: (c.excluded_reason is not None, -c.metadata_score),
    )
    return candidates


def select_gavd_abnormal_candidates(
    *,
    cache_dir: Path | None = None,
    top_n: int = 20,
    download: bool = True,
) -> dict[str, Any]:
    """Download annotations (if needed), filter, and return ranked abnormal candidates."""
    paths = (
        download_gavd_annotations(cache_dir=cache_dir)
        if download
        else list((cache_dir or gavd_cache_dir()).glob("GAVD_Clinical_Annotations_*.csv"))
    )
    candidates = build_gavd_abnormal_candidates(paths)
    eligible = [c for c in candidates if c.excluded_reason is None]
    return {
        "source": "GAVD",
        "source_url": "https://github.com/Rahmyyy/GAVD",
        "annotation_files": [str(p) for p in paths],
        "total_abnormal_sequences": sum(
            1 for c in candidates if "abnormal" in c.dataset.lower()
        ),
        "eligible_count": len(eligible),
        "top_candidates": [c.to_dict() for c in eligible[:top_n]],
        "review_rejects": [
            c.to_dict() for c in candidates if c.excluded_reason is not None
        ][:top_n],
        "next_steps": [
            "Retrieve top candidate video from the public YouTube URL.",
            "Trim to the annotated seq frame range if needed.",
            "Run: python scripts/validate_demo_candidate.py --video <path>",
            "Only install as abnormal_gait.mp4 after ACCEPT or ACCEPT_WITH_LIMITATIONS.",
        ],
    }


def write_gavd_selection_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = [
    "GAVD_REPO_DATA_URL",
    "GavdSequenceCandidate",
    "GavdSelectionThresholds",
    "build_gavd_abnormal_candidates",
    "download_gavd_annotations",
    "select_gavd_abnormal_candidates",
    "write_gavd_selection_report",
]
