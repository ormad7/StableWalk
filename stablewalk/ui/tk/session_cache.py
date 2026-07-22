"""
In-memory cache of analyzed sessions for Comparison Mode.

Pinning a session does not re-run the pipeline — it snapshots the current
analysis so Compare can load two sessions side-by-side instantly.

Session A and Session B are independent slots. Never alias one slot's key
onto the other when a requested session is missing.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    com_path_xyz,
    extract_compare_metrics,
    knee_angle_series,
)
from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.skeleton_3d import Skeleton3D

# Demo session keys that Compare Mode should offer even before they are pinned.
DEMO_COMPARE_KEYS: tuple[str, ...] = ("normal", "abnormal", "athletic")

_DEMO_POSE_STEMS: dict[str, tuple[str, ...]] = {
    "normal": ("normal_gait", "normal", "validation_normal"),
    "abnormal": ("abnormal_gait", "abnormal", "validation_abnormal"),
    "athletic": ("athletic_walking", "athletic", "validation_athletic", "performance"),
}


@dataclass
class AnalyzedSessionSnapshot:
    """Frozen view of one fully analyzed gait session."""

    key: str
    label: str
    source: str = ""
    poses_path: Path | None = None
    frames_dir: Path | None = None
    video_path: str | None = None
    fps: float = 30.0
    n_frames: int = 0
    pose_indices: list[int] = field(default_factory=list)
    sequence: PoseSequence | None = None
    skeletons: dict[int, Skeleton3D] = field(default_factory=dict)
    metrics: CompareMetrics = field(default_factory=CompareMetrics)
    knee_t: list[float] = field(default_factory=list)
    knee_y: list[float | None] = field(default_factory=list)
    path_xyz: np.ndarray | None = None
    frame_paths: list[Path] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return self.n_frames > 0 and (
            self.sequence is not None or bool(self.frame_paths) or bool(self.skeletons)
        )


class SessionCompareCache:
    """Keyed session store (demo keys or custom stems)."""

    def __init__(self) -> None:
        self._sessions: dict[str, AnalyzedSessionSnapshot] = {}
        self.left_key: str = "normal"
        self.right_key: str = "abnormal"

    def pin(self, snapshot: AnalyzedSessionSnapshot) -> None:
        """Store *snapshot* under its own key only — never overwrites the other slot."""
        if not snapshot.key:
            return
        self._sessions[snapshot.key] = snapshot

    def get(self, key: str | None) -> AnalyzedSessionSnapshot | None:
        if not key:
            return None
        return self._sessions.get(key)

    def left(self) -> AnalyzedSessionSnapshot | None:
        return self.get(self.left_key)

    def right(self) -> AnalyzedSessionSnapshot | None:
        return self.get(self.right_key)

    def keys(self) -> list[str]:
        return sorted(self._sessions.keys())

    def labels(self) -> dict[str, str]:
        return {k: s.label for k, s in self._sessions.items()}

    def choice_keys(self) -> list[str]:
        """Pinned sessions plus standard demo keys for the Session A/B selectors."""
        ordered: list[str] = []
        for key in DEMO_COMPARE_KEYS:
            if key not in ordered:
                ordered.append(key)
        for key in self.keys():
            if key not in ordered:
                ordered.append(key)
        return ordered

    def set_slot(self, slot: str, key: str) -> None:
        """Assign Session A (left) or Session B (right) without coalescing slots."""
        key = str(key or "").strip()
        if not key:
            return
        if slot == "right":
            self.right_key = key
        else:
            self.left_key = key

    def clear(self) -> None:
        self._sessions.clear()


def sessions_are_identical(
    left: AnalyzedSessionSnapshot | None,
    right: AnalyzedSessionSnapshot | None,
    *,
    left_key: str | None = None,
    right_key: str | None = None,
) -> bool:
    """True when both slots resolve to the same session identity."""
    if left_key is not None and right_key is not None and left_key == right_key:
        return True
    if left is None or right is None:
        return False
    if left is right:
        return True
    if left.key and right.key and left.key == right.key:
        return True
    if left.source and right.source and left.source == right.source:
        if left.n_frames == right.n_frames and left.n_frames > 0:
            return True
    return False


def _frame_paths_from_dir(frames_dir: Path | None) -> list[Path]:
    if frames_dir is None or not Path(frames_dir).is_dir():
        return []
    root = Path(frames_dir)
    paths = sorted(root.glob("frame_*.jpg"))
    if not paths:
        paths = sorted(root.glob("*.jpg"))
    return paths


def _demo_label(key: str) -> str:
    try:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES

        for ex in DEMO_GAIT_EXAMPLES:
            if ex.key == key:
                return ex.display_name
    except Exception:
        pass
    return key.replace("_", " ").title()


def resolve_demo_poses_path(key: str) -> Path | None:
    """Locate a cached poses JSON for a demo Compare key."""
    from stablewalk import config

    stems = _DEMO_POSE_STEMS.get(key, (key,))
    for stem in stems:
        candidate = config.POSES_DIR / f"{stem}_poses.json"
        if candidate.is_file():
            return candidate
    return None


def resolve_demo_frames_dir(key: str) -> Path | None:
    from stablewalk import config

    stems = _DEMO_POSE_STEMS.get(key, (key,))
    for stem in stems:
        candidate = config.FRAMES_DIR / stem
        if candidate.is_dir():
            return candidate
    return None


def resolve_demo_video_path(key: str) -> str | None:
    try:
        from stablewalk.ui.media.demo_gait import DEMO_GAIT_EXAMPLES, demo_path

        for ex in DEMO_GAIT_EXAMPLES:
            if ex.key == key:
                path = demo_path(ex)
                return str(path) if path.is_file() else str(path)
    except Exception:
        pass
    return None


def build_snapshot_from_sequence(
    sequence: PoseSequence,
    *,
    key: str,
    label: str,
    source: str = "",
    poses_path: Path | None = None,
    frames_dir: Path | None = None,
    video_path: str | None = None,
) -> AnalyzedSessionSnapshot:
    """Build an independent compare snapshot from a pose sequence (no GUI state)."""
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.biomechanical import run_biomechanical_analysis
    from stablewalk.analysis.estimated_vgrf_analysis import analyze_estimated_vgrf
    from stablewalk.analysis.foot_contact_analysis import analyze_foot_contact
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.io.pose_loader import sequence_needs_enrichment
    from stablewalk.pose.enrichment import enrich_pose_sequence

    if sequence_needs_enrichment(sequence):
        try:
            enrich_pose_sequence(sequence)
        except Exception:
            pass

    fps = float(getattr(sequence, "fps", None) or 30.0)
    pose_indices = [
        i for i, fr in enumerate(sequence.frames) if getattr(fr, "detected", True)
    ]
    if not pose_indices and sequence.frames:
        pose_indices = list(range(len(sequence.frames)))

    gait_motion = None
    cycles = None
    contact = None
    estimated_vgrf = None
    biomech = None
    try:
        gait_motion = pose_sequence_to_gait_motion(sequence)
    except Exception:
        gait_motion = None
    if gait_motion is not None:
        try:
            cycles = analyze_gait_cycles(gait_motion)
        except Exception:
            cycles = None
        try:
            contact = analyze_foot_contact(gait_motion, cycles=cycles)
        except Exception:
            contact = None
        try:
            estimated_vgrf = analyze_estimated_vgrf(
                gait_motion, contact, sequence=sequence
            )
        except Exception:
            estimated_vgrf = None
        try:
            biomech = run_biomechanical_analysis(
                gait_motion,
                sequence,
                cycles=cycles,
                contact=contact,
            )
        except Exception:
            biomech = None

    knee_t, knee_y = knee_angle_series(sequence)
    knee_vals = [float(v) for v in knee_y if v is not None]
    knee_rom = (
        float(max(knee_vals) - min(knee_vals)) if len(knee_vals) >= 2 else None
    )
    n_frames = len(pose_indices) or len(sequence.frames)
    duration_s = (n_frames - 1) / fps if n_frames > 1 and fps > 1e-6 else None

    metrics = extract_compare_metrics(
        label=label,
        session_key=key,
        biomechanical=biomech,
        estimated_vgrf=estimated_vgrf,
        cycles=cycles,
        sequence=sequence,
        knee_rom_deg=knee_rom,
        duration_s=duration_s,
    )
    # Detach metrics from any shared GUI object graph.
    metrics = deepcopy(metrics)

    path = com_path_xyz(biomech)
    if path is not None:
        path = np.array(path, dtype=float, copy=True)

    return AnalyzedSessionSnapshot(
        key=key,
        label=label,
        source=str(video_path or source or ""),
        poses_path=Path(poses_path) if poses_path else None,
        frames_dir=Path(frames_dir) if frames_dir else None,
        video_path=str(video_path) if video_path else None,
        fps=fps,
        n_frames=n_frames,
        pose_indices=list(pose_indices),
        sequence=sequence,
        skeletons={},
        metrics=metrics,
        knee_t=list(knee_t),
        knee_y=list(knee_y),
        path_xyz=path,
        frame_paths=_frame_paths_from_dir(Path(frames_dir) if frames_dir else None),
    )


def try_build_demo_snapshot(key: str) -> AnalyzedSessionSnapshot | None:
    """Load an independent demo snapshot from cached poses (never from Session A)."""
    from stablewalk.io.pose_loader import load_pose_sequence

    poses_path = resolve_demo_poses_path(key)
    if poses_path is None:
        return None
    try:
        sequence = load_pose_sequence(poses_path)
    except Exception:
        return None
    frames_dir = resolve_demo_frames_dir(key)
    video_path = resolve_demo_video_path(key)
    return build_snapshot_from_sequence(
        sequence,
        key=key,
        label=_demo_label(key),
        source=str(video_path or poses_path),
        poses_path=poses_path,
        frames_dir=frames_dir,
        video_path=video_path,
    )


def ensure_demo_sessions_pinned(cache: SessionCompareCache) -> list[str]:
    """Pin available demo sessions from disk without overwriting existing pins."""
    loaded: list[str] = []
    for key in DEMO_COMPARE_KEYS:
        if cache.get(key) is not None:
            continue
        snap = try_build_demo_snapshot(key)
        if snap is not None:
            cache.pin(snap)
            loaded.append(key)
    return loaded


def build_snapshot_from_gui(gui: Any) -> AnalyzedSessionSnapshot | None:
    """Capture the GUI's current analysis into a compare-ready snapshot."""
    sequence = getattr(gui, "sequence", None)
    if sequence is None:
        return None

    demo = getattr(gui, "_active_demo_gait", None)
    if demo is not None:
        key = str(demo.key)
        label = str(demo.display_name)
    else:
        source = getattr(gui, "_session_display_src", None) or getattr(
            gui, "_current_source", ""
        )
        stem = Path(str(source)).stem if source else "session"
        key = stem
        label = stem.replace("_", " ").title() or "Session"

    meta = getattr(gui, "_run_metadata", None)
    poses_path = getattr(gui, "_poses_path", None)
    frames_dir = getattr(meta, "frames_dir", None) if meta is not None else None
    video_path = getattr(meta, "source", None) if meta is not None else None
    if video_path is None:
        video_path = getattr(gui, "_current_source", None)

    fps = float(getattr(sequence, "fps", None) or getattr(meta, "fps", None) or 30.0)
    pose_indices = list(getattr(gui, "pose_indices", None) or [])
    if not pose_indices and sequence.frames:
        pose_indices = list(range(len(sequence.frames)))

    skeletons = dict(getattr(gui, "_skeleton_cache", None) or {})

    usable = detected = None
    resolver = getattr(gui, "_resolved_gait_cycle_count", None)
    if callable(resolver):
        try:
            usable, detected = resolver()
        except Exception:
            usable, detected = None, None

    ba = getattr(gui, "_biomech_analysis", None)
    summary = getattr(gui, "_analysis_summary_cache", None)
    if summary is None:
        builder = getattr(gui, "_build_analysis_summary", None)
        if callable(builder):
            try:
                summary = builder()
            except Exception:
                summary = None

    knee_t, knee_y = knee_angle_series(sequence)
    knee_vals = [float(v) for v in knee_y if v is not None]
    knee_rom = (
        float(max(knee_vals) - min(knee_vals)) if len(knee_vals) >= 2 else None
    )
    n_frames = len(pose_indices) or len(sequence.frames)
    duration_s = (n_frames - 1) / fps if n_frames > 1 and fps > 1e-6 else None

    metrics = extract_compare_metrics(
        label=label,
        session_key=key,
        biomechanical=ba,
        estimated_vgrf=getattr(gui, "_estimated_vgrf", None),
        cycles=getattr(gui, "_gait_cycle", None),
        usable_gait_cycles=usable,
        detected_gait_cycles=detected,
        summary=summary,
        sequence=sequence,
        knee_rom_deg=knee_rom,
        duration_s=duration_s,
    )
    metrics = deepcopy(metrics)

    path = com_path_xyz(ba)
    if path is not None:
        path = np.array(path, dtype=float, copy=True)

    return AnalyzedSessionSnapshot(
        key=key,
        label=label,
        source=str(video_path or ""),
        poses_path=Path(poses_path) if poses_path else None,
        frames_dir=Path(frames_dir) if frames_dir else None,
        video_path=str(video_path) if video_path else None,
        fps=fps,
        n_frames=n_frames,
        pose_indices=pose_indices,
        sequence=sequence,
        skeletons=skeletons,
        metrics=metrics,
        knee_t=list(knee_t),
        knee_y=list(knee_y),
        path_xyz=path,
        frame_paths=_frame_paths_from_dir(Path(frames_dir) if frames_dir else None),
    )


def ensure_compare_cache(gui: Any) -> SessionCompareCache:
    cache = getattr(gui, "_compare_session_cache", None)
    if cache is None:
        cache = SessionCompareCache()
        gui._compare_session_cache = cache
    return cache


def pin_current_session(gui: Any) -> AnalyzedSessionSnapshot | None:
    """Pin the active analysis so Comparison Mode can reuse it without re-analysis."""
    snap = build_snapshot_from_gui(gui)
    if snap is None:
        return None
    ensure_compare_cache(gui).pin(snap)
    refresh = getattr(gui, "_refresh_comparison_mode", None)
    if callable(refresh):
        try:
            refresh()
        except Exception:
            pass
    return snap


__all__ = [
    "DEMO_COMPARE_KEYS",
    "AnalyzedSessionSnapshot",
    "SessionCompareCache",
    "build_snapshot_from_gui",
    "build_snapshot_from_sequence",
    "ensure_compare_cache",
    "ensure_demo_sessions_pinned",
    "pin_current_session",
    "resolve_demo_poses_path",
    "sessions_are_identical",
    "try_build_demo_snapshot",
]
