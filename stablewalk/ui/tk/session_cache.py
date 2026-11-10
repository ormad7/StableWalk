"""
In-memory cache of analyzed sessions for Comparison Mode.

Pinning a session does not re-run the pipeline — it snapshots the current
analysis so Compare can load two sessions side-by-side instantly.

Session A and Session B are independent slots. Never alias one slot's key
onto the other when a requested session is missing.
"""

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from stablewalk.analysis.session_compare import (
    CompareMetrics,
    com_path_xyz,
    extract_compare_metrics,
    knee_angle_series,
)
from stablewalk.models.pose_data import PoseSequence
from stablewalk.pose.skeleton_3d import Skeleton3D

logger = logging.getLogger(__name__)

# Demo session keys that Compare Mode should offer even before they are pinned.
DEMO_COMPARE_KEYS: tuple[str, ...] = ("normal", "abnormal", "athletic")

_DEMO_POSE_STEMS: dict[str, tuple[str, ...]] = {
    "normal": ("normal_gait_v2",),
    "abnormal": ("abnormal_gait_v2",),
    "athletic": ("fast_walking_v2",),
}


def _heel_strike_times_s(cycles: Any | None) -> list[float]:
    """Extract sorted heel-strike times for Compare gait-cycle axis labeling."""
    if cycles is None:
        return []
    events = list(getattr(cycles, "events", None) or [])
    left = sorted(
        float(e.time_s)
        for e in events
        if getattr(e, "event_type", "") == "left_heel_strike"
        and np.isfinite(float(getattr(e, "time_s", float("nan"))))
    )
    if len(left) >= 2:
        return left
    any_hs = sorted(
        float(e.time_s)
        for e in events
        if str(getattr(e, "event_type", "")).endswith("_heel_strike")
        and np.isfinite(float(getattr(e, "time_s", float("nan"))))
    )
    return any_hs


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
    # Heel-strike times (s) for Compare gait-cycle % axis — display metadata only.
    heel_strike_times_s: list[float] = field(default_factory=list)
    # Independent copies of Session B pipeline outputs (never shared with A).
    gait_cycle: Any | None = None
    gait_motion: Any | None = None
    foot_contact: Any | None = None
    biomech_analysis: Any | None = None
    # Picker / provenance metadata (optional).
    backend: str = ""
    confidence_label: str = ""
    analysis_date: str = ""
    valid_frame_count: int = 0
    bundle_path: str = ""
    pipeline_complete: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    analysis_mode: str = ""
    max_frames: int | None = None

    @property
    def ready(self) -> bool:
        return self.n_frames > 0 and (
            self.sequence is not None or bool(self.frame_paths) or bool(self.skeletons)
        )


class SessionCompareCache:
    """Keyed session store (demo keys or custom stems).

    Caps non-demo pins so repeated Analyze / Load does not retain every
    historical ``PoseSequence`` for the process lifetime.
    """

    _MAX_CUSTOM_SESSIONS = 6

    def __init__(self) -> None:
        self._sessions: dict[str, AnalyzedSessionSnapshot] = {}
        self._pin_order: list[str] = []
        self.left_key: str = "normal"
        self.right_key: str = "abnormal"

    def pin(self, snapshot: AnalyzedSessionSnapshot) -> None:
        """Store *snapshot* under its own key only — never overwrites the other slot."""
        if not snapshot.key:
            return
        key = snapshot.key
        self._sessions[key] = snapshot
        if key in self._pin_order:
            self._pin_order.remove(key)
        self._pin_order.append(key)
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        """Drop oldest custom pins; never evict demos or active A/B slots."""
        protected = set(DEMO_COMPARE_KEYS)
        protected.add(self.left_key)
        protected.add(self.right_key)
        custom = [k for k in self._pin_order if k not in DEMO_COMPARE_KEYS]
        while len(custom) > self._MAX_CUSTOM_SESSIONS:
            victim = None
            for k in custom:
                if k not in protected:
                    victim = k
                    break
            if victim is None:
                break
            self._sessions.pop(victim, None)
            if victim in self._pin_order:
                self._pin_order.remove(victim)
            custom = [k for k in self._pin_order if k not in DEMO_COMPARE_KEYS]

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

    def rekey(self, old_key: str, new_key: str) -> AnalyzedSessionSnapshot | None:
        """Move a pinned snapshot to a new key without sharing the other slot."""
        old_key = str(old_key or "").strip()
        new_key = str(new_key or "").strip()
        snap = self._sessions.get(old_key)
        if snap is None or not new_key or old_key == new_key:
            return snap
        self._sessions.pop(old_key, None)
        if old_key in self._pin_order:
            self._pin_order.remove(old_key)
        snap.key = new_key
        self.pin(snap)
        if self.left_key == old_key:
            self.left_key = new_key
        if self.right_key == old_key:
            self.right_key = new_key
        return snap

    def clear(self) -> None:
        self._sessions.clear()
        self._pin_order.clear()

    def retain_pinned_slots(
        self, *, also_keep: set[str] | Iterable[str] | None = None
    ) -> int:
        """Drop sessions that are not A/B slots (or *also_keep*).

        Used on new video load so only the current session and explicitly
        pinned comparison slots remain in memory. Demo keys that are not
        currently selected as A or B are released and can be reloaded later.
        """
        keep: set[str] = set()
        if self.left_key:
            keep.add(self.left_key)
        if self.right_key:
            keep.add(self.right_key)
        if also_keep:
            keep.update(str(k) for k in also_keep if k)
        removed = 0
        for key in list(self._sessions.keys()):
            if key in keep:
                continue
            self._sessions.pop(key, None)
            if key in self._pin_order:
                self._pin_order.remove(key)
            removed += 1
        return removed


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
    left_sources = {
        _recording_identity(value)
        for value in (left.source, left.video_path)
        if value
    }
    right_sources = {
        _recording_identity(value)
        for value in (right.source, right.video_path)
        if value
    }
    if left_sources and right_sources and left_sources.intersection(right_sources):
        return True
    return False


def _recording_identity(value: str | Path) -> str:
    """Canonical recording identity for accidental A/B duplicate detection."""
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return Path(raw).expanduser().resolve(strict=False).as_posix().casefold()
    except (OSError, RuntimeError, ValueError):
        return raw.replace("\\", "/").casefold()


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
    import json

    from stablewalk import config
    from stablewalk.pose.subject_lock import (
        pose_dump_matches_selection,
        selection_for_source,
    )
    from stablewalk.ui.media.demo_gait import example_by_key

    stems = _DEMO_POSE_STEMS.get(key, (key,))
    example = example_by_key(key)
    filename = example.filename if example is not None else ""
    wanted = selection_for_source(filename)
    for stem in stems:
        candidate = config.POSES_DIR / f"{stem}_poses.json"
        if candidate.is_file() and _pose_file_usable(candidate, wanted):
            return candidate
    matches: list[Path] = []
    for path in config.POSES_DIR.glob("*_poses.json"):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = str(raw.get("source_video") or "")
        if Path(source).name != filename:
            continue
        if pose_dump_matches_selection(raw, wanted):
            matches.append(path)
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def _pose_file_usable(path: Path, selection) -> bool:
    import json

    from stablewalk.pose.subject_lock import pose_dump_matches_selection

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return pose_dump_matches_selection(raw, selection)


def resolve_demo_frames_dir(key: str) -> Path | None:
    from stablewalk import config

    stems = _DEMO_POSE_STEMS.get(key, (key,))
    for stem in stems:
        candidate = config.FRAMES_DIR / stem
        if candidate.is_dir():
            return candidate
    return None


def _demo_candidate_names(key: str) -> list[str]:
    """Official v2 filenames only — never athletic_walking.mp4 or other legacy names."""
    from stablewalk.ui.media.demo_gait import example_by_key, is_legacy_demo_filename

    ex = example_by_key(key)
    if ex is None or is_legacy_demo_filename(ex.filename):
        return []
    return [ex.filename]


def _demo_path_belongs_to_key(key: str, path: Path) -> bool:
    """True when *path* is this demo key's recording, not another demo's file."""
    stem = path.stem.lower()
    allowed = {Path(name).stem.lower() for name in _demo_candidate_names(key)}
    if stem not in allowed:
        return False
    if key == "abnormal" and "abnormal" not in stem:
        return False
    if key == "normal" and ("abnormal" in stem or "athletic" in stem):
        return False
    return True


def resolve_demo_video_path(key: str) -> str | None:
    """Locate the official v2 demo recording only; never a legacy filename."""
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key, is_legacy_demo_filename

    ex = example_by_key(key)
    if ex is None:
        return None
    candidate = demo_path(ex)
    if is_legacy_demo_filename(candidate.name):
        return None
    if candidate.is_file() and _demo_path_belongs_to_key(key, candidate):
        return str(candidate)
    return None


def log_session_b_debug(field: str, value: object) -> None:
    """Temporary Compare Session B pipeline breadcrumbs (exact prefix required)."""
    logger.info("[COMPARE B] %s=%s", field, value)


def log_compare_debug(event: str, **fields: object) -> None:
    """Defensive Compare lifecycle breadcrumbs (load / analyze / generate / refresh)."""
    extra = " ".join(f"{key}={value}" for key, value in fields.items())
    if extra:
        logger.info("[COMPARE] %s %s", event, extra)
    else:
        logger.info("[COMPARE] %s", event)


def log_compare_step(message: str) -> None:
    """Temporary console breadcrumbs for the Load demo comparison flow."""
    line = f"[COMPARE] {message}"
    print(line, flush=True)
    logger.info("%s", line)


def hydrate_demo_snapshot(
    snap: AnalyzedSessionSnapshot | None, key: str
) -> AnalyzedSessionSnapshot | None:
    """Attach real demo video/frame paths when missing. Never invents metrics."""
    if snap is None:
        return None
    from dataclasses import replace

    key = str(key or snap.key or "").strip()
    video = str(snap.video_path or "").strip() or None
    if not video or not Path(video).is_file():
        resolved = resolve_demo_video_path(key)
        if resolved:
            video = resolved
    frames_dir = snap.frames_dir
    frame_paths = list(snap.frame_paths or [])
    if not frame_paths:
        if frames_dir is None:
            frames_dir = resolve_demo_frames_dir(key)
        frame_paths = _frame_paths_from_dir(frames_dir)
    source = str(snap.source or "").strip()
    if not source:
        source = str(video or "")
    unchanged = (
        video == (str(snap.video_path).strip() if snap.video_path else None)
        and list(snap.frame_paths or []) == frame_paths
        and (frames_dir is None or snap.frames_dir == frames_dir)
        and (not source or source == str(snap.source or ""))
    )
    if unchanged:
        return snap
    return replace(
        snap,
        video_path=video,
        frames_dir=frames_dir if frames_dir is not None else snap.frames_dir,
        frame_paths=frame_paths,
        source=source or snap.source,
    )


def demo_snapshot_is_compare_ready(
    snap: AnalyzedSessionSnapshot | None,
    *,
    other: AnalyzedSessionSnapshot | None = None,
) -> bool:
    """True when a demo slot can render video, skeleton, and real metrics.

    Knee-ROM plus a frame count is not enough: Session B must own pose data
    and a renderable video/frame source so Compare does not show placeholders.
    """
    if snap is None:
        return False
    if not bool(getattr(snap, "ready", False)):
        return False
    if int(getattr(snap, "n_frames", 0) or 0) <= 0:
        return False
    has_skel = bool(getattr(snap, "skeletons", None))
    if not has_skel:
        seq = getattr(snap, "sequence", None)
        frames = getattr(seq, "frames", None) if seq is not None else None
        has_skel = bool(frames)
    if not has_skel:
        return False
    frame_paths = getattr(snap, "frame_paths", None) or []
    video = str(getattr(snap, "video_path", None) or "").strip()
    video_ok = bool(video and Path(video).is_file())
    if not video_ok and not frame_paths:
        return False
    metrics = getattr(snap, "metrics", None)
    if metrics is None:
        return False
    has_kpi = any(
        getattr(metrics, attr, None) is not None
        for attr in (
            "gait_quality",
            "cadence_spm",
            "walking_speed_m_s",
            "step_length_m",
            "symmetry_pct",
            "stability_margin_m",
            "knee_rom_deg",
            "com_excursion_m",
            "hip_rom_deg",
        )
    )
    if not has_kpi:
        return False
    if other is not None:
        if snap is other:
            return False
        seq = getattr(snap, "sequence", None)
        other_seq = getattr(other, "sequence", None)
        if seq is not None and other_seq is not None and seq is other_seq:
            return False
    return True


def isolate_analyzed_snapshot(
    snap: AnalyzedSessionSnapshot,
    *,
    key: str | None = None,
) -> AnalyzedSessionSnapshot:
    """Return a Session B-owned copy that shares no arrays with Session A."""
    from dataclasses import replace

    seq = snap.sequence
    if seq is not None:
        try:
            seq = deepcopy(seq)
        except Exception:
            logger.debug("Session B sequence deepcopy failed; keeping loaded copy")
    path = snap.path_xyz
    if path is not None:
        path = np.array(path, dtype=float, copy=True)
    try:
        metrics = deepcopy(snap.metrics)
    except Exception:
        metrics = snap.metrics
    gait_cycle = snap.gait_cycle
    foot_contact = snap.foot_contact
    biomech = snap.biomech_analysis
    try:
        gait_cycle = deepcopy(gait_cycle) if gait_cycle is not None else None
        foot_contact = deepcopy(foot_contact) if foot_contact is not None else None
        biomech = deepcopy(biomech) if biomech is not None else None
    except Exception:
        logger.debug("Session B analysis deepcopy failed; keeping pipeline-owned objects")
    try:
        metadata = dict(snap.metadata or {})
    except Exception:
        metadata = {}
    return replace(
        snap,
        key=str(key or snap.key),
        sequence=seq,
        path_xyz=path,
        metrics=metrics,
        pose_indices=list(snap.pose_indices),
        knee_t=list(snap.knee_t),
        knee_y=list(snap.knee_y),
        frame_paths=list(snap.frame_paths),
        heel_strike_times_s=list(snap.heel_strike_times_s),
        skeletons=dict(snap.skeletons),
        gait_cycle=gait_cycle,
        foot_contact=foot_contact,
        biomech_analysis=biomech,
        metadata=metadata,
    )


def _count_pose_and_3d_frames(sequence: PoseSequence | None) -> tuple[int, int]:
    pose_n = 0
    d3 = 0
    if sequence is None:
        return 0, 0
    for fr in getattr(sequence, "frames", []) or []:
        kps = getattr(fr, "keypoints", None)
        if getattr(fr, "detected", False) and kps:
            pose_n += 1
        if getattr(fr, "skeleton_3d", None) or getattr(fr, "positions", None):
            d3 += 1
    return pose_n, d3


def resolve_session_b_video_path(key: str, *, hint: str | None = None) -> str | None:
    """Resolve the selected Session B name to a real recording path.

    The Compare combobox value (e.g. ``abnormal``) is an identifier, not a
    filename. Never return the primary/Session A recording for a different key.
    ``hint`` is only used when it belongs to *key*.
    """
    key = str(key or "").strip()
    if not key:
        return None

    def _hint_belongs(path: Path) -> bool:
        if not path.is_file():
            return False
        if key.startswith("video:"):
            try:
                return path.resolve() == Path(key.removeprefix("video:")).resolve()
            except (OSError, ValueError):
                return str(path) == key.removeprefix("video:")
        if key in DEMO_COMPARE_KEYS:
            return _demo_path_belongs_to_key(key, path)
        stem = Path(key).stem.lower()
        return path.stem.lower() == stem or path.name.lower() == Path(key).name.lower()

    if key.startswith("video:"):
        candidate = Path(key.removeprefix("video:"))
        return str(candidate) if candidate.is_file() else None

    if key in DEMO_COMPARE_KEYS:
        resolved = resolve_demo_video_path(key)
        if resolved:
            return resolved
        if hint:
            hinted = Path(str(hint))
            if _hint_belongs(hinted):
                return str(hinted)
        return None

    if hint:
        hinted = Path(str(hint))
        if _hint_belongs(hinted):
            return str(hinted)

    from stablewalk import config

    stem = Path(key).stem
    for folder in (config.DEMO_VIDEOS_DIR, config.VIDEOS_DIR, config.LEGACY_VIDEOS_DIR):
        for candidate in (
            folder / f"{stem}.mp4",
            folder / Path(key).name,
        ):
            if candidate.is_file() and (
                key not in DEMO_COMPARE_KEYS or _demo_path_belongs_to_key(key, candidate)
            ):
                return str(candidate)
    return None


def analyze_video(
    path: str,
    *,
    run_name: str | None = None,
    pose_backend: str | None = None,
    max_frames: int | None = None,
    force_reprocess: bool = True,
):
    """Same gait pipeline used by the main Analyze button (no GUI session writes)."""
    from stablewalk.core.pipeline import run_gait_pipeline
    from stablewalk.pose.video_source import derive_run_name

    source = str(path)
    if not run_name:
        run_name = derive_run_name(source, unique_session=True)
    return run_gait_pipeline(
        source,
        run_name=run_name,
        force_reprocess=force_reprocess,
        pose_backend=pose_backend,
        max_frames=max_frames,
    )


def run_independent_session_b_analysis(
    key: str,
    *,
    video_path: str | None = None,
    pose_backend: str | None = None,
    max_frames: int | None = None,
    analysis_mode: str | None = None,
    left_video_identity: str | None = None,
) -> tuple[AnalyzedSessionSnapshot | None, str | None]:
    """Run the main Analyze pipeline for Session B only (never reads Session A)."""
    import traceback
    from datetime import datetime, timezone

    key = str(key or "").strip()
    log_session_b_debug("selected_name", key)
    resolved = resolve_session_b_video_path(key, hint=video_path)
    log_session_b_debug("resolved_path", resolved)
    exists = bool(resolved and Path(resolved).is_file())
    log_session_b_debug("path_exists", exists)
    logger.info(
        "[COMPARE] Resolved Session B path=%s exists=%s",
        resolved,
        exists,
    )

    if not exists:
        if key in DEMO_COMPARE_KEYS:
            try:
                from stablewalk.ui.media.demo_download import download_demo_video

                download_demo_video(key, force=False)
            except Exception as exc:
                tb = traceback.format_exc().strip()
                logger.exception("Session B demo download failed for %s", key)
                return (
                    None,
                    f"Could not download Session B ('{key}') video: {exc}\n{tb}",
                )
            resolved = resolve_session_b_video_path(key, hint=video_path)
            exists = bool(resolved and Path(resolved).is_file())
            log_session_b_debug("resolved_path", resolved)
            log_session_b_debug("path_exists", exists)
        if not exists:
            return (
                None,
                f"Session B ('{key}') video was not found. "
                "Resolve 'abnormal' to data/demo_videos/abnormal_gait_v2.mp4 "
                "(or use Load recording… for a file different from Session A).",
            )

    video = Path(resolved)
    if left_video_identity and _recording_identity(video) == left_video_identity:
        return (
            None,
            f"Session B ('{key}') resolved to Session A's recording "
            f"({video.name}) and was rejected.",
        )

    log_session_b_debug("analysis_started", True)
    try:
        from stablewalk.io.pose_loader import load_pose_sequence
        from stablewalk.pose.video_source import derive_run_name
        from stablewalk.ui.analysis_mode import (
            MODE_FULL,
            normalize_analysis_mode,
            resolve_gui_max_frames,
        )

        requested_mode = normalize_analysis_mode(
            analysis_mode, max_frames=max_frames
        ) or MODE_FULL
        max_frames, requested_mode = resolve_gui_max_frames(
            mode=requested_mode,
            is_demo=key in DEMO_COMPARE_KEYS,
            explicit_max_frames=(
                max_frames if requested_mode != MODE_FULL else None
            ),
        )
        run_name = derive_run_name(str(video), unique_session=True)
        result = analyze_video(
            str(video),
            run_name=run_name,
            pose_backend=pose_backend,
            max_frames=max_frames,
            force_reprocess=True,
        )
        sequence = load_pose_sequence(result.poses_path)
        try:
            sequence = deepcopy(sequence)
        except Exception:
            pass
        snap = build_snapshot_from_sequence(
            sequence,
            key=key,
            label=_demo_label(key) if key in DEMO_COMPARE_KEYS else (
                video.stem.replace("_", " ").title() or video.name
            ),
            source=str(video),
            poses_path=result.poses_path,
            frames_dir=result.frames_dir,
            video_path=str(video),
            analysis_mode=requested_mode,
            max_frames=max_frames,
        )
        snap = isolate_analyzed_snapshot(snap, key=key)
        pose_n, d3_n = _count_pose_and_3d_frames(snap.sequence)
        metrics = snap.metrics
        metric_keys = [
            name
            for name in (
                "gait_quality",
                "cadence_spm",
                "symmetry_pct",
                "stability_margin_m",
                "knee_rom_deg",
                "hip_rom_deg",
                "ankle_rom_deg",
                "com_excursion_m",
                "foot_clearance_m",
                "usable_gait_cycles",
                "detected_gait_cycles",
            )
            if getattr(metrics, name, None) is not None
        ]
        snap.valid_frame_count = int(snap.n_frames)
        snap.backend = str(
            getattr(result, "pose_backend_used", None) or pose_backend or ""
        )
        snap.analysis_date = datetime.now(timezone.utc).isoformat()
        snap.pipeline_complete = True
        snap.metadata = {
            "source_video_path": str(video),
            "frame_count": int(snap.n_frames),
            "fps": float(snap.fps),
            "decoded_frames": len(snap.frame_paths),
            "pose_landmarks": pose_n,
            "smoothed_landmarks": True,
            "3d_coordinates": d3_n,
            "joint_angles": True,
            "velocity": True,
            "acceleration": True,
            "rom": True,
            "com": snap.path_xyz is not None,
            "stability_metrics": metrics.stability_margin_m is not None,
            "cadence": metrics.cadence_spm is not None,
            "symmetry": metrics.symmetry_pct is not None,
            "gait_phases": snap.gait_cycle is not None,
            "foot_clearance": getattr(metrics, "foot_clearance_m", None) is not None,
            "gait_quality": metrics.gait_quality is not None,
            "metrics_keys": metric_keys,
            "pipeline": "run_gait_pipeline",
            "run_name": str(getattr(result, "run_name", "") or ""),
            "analysis_mode": requested_mode,
            "max_frames": max_frames,
        }
        log_session_b_debug("analysis_finished", True)
        log_session_b_debug("result_type", type(snap).__name__)
        log_session_b_debug("frame_count", snap.n_frames)
        log_session_b_debug("fps", snap.fps)
        log_session_b_debug("landmarks_count", pose_n)
        log_session_b_debug("pose_frames", pose_n)
        log_session_b_debug("3d_frames", d3_n)
        log_session_b_debug("metrics_keys", metric_keys)
        return snap, None
    except Exception as exc:
        tb = traceback.format_exc().strip()
        logger.exception("Independent Session B analysis failed for %s", key)
        log_session_b_debug("analysis_finished", False)
        log_session_b_debug("result_type", type(exc).__name__)
        return None, f"{exc}\n{tb}"


def build_snapshot_from_sequence(
    sequence: PoseSequence,
    *,
    key: str,
    label: str,
    source: str = "",
    poses_path: Path | None = None,
    frames_dir: Path | None = None,
    video_path: str | None = None,
    analysis_mode: str = "",
    max_frames: int | None = None,
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
    from stablewalk.io.pose_loader import playback_frame_indices

    pose_indices = playback_frame_indices(sequence)

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
        path = np.asarray(path, dtype=float)
        if path.base is not None or not getattr(path.flags, "owndata", True):
            path = path.copy()

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
        heel_strike_times_s=_heel_strike_times_s(cycles),
        gait_cycle=cycles,
        gait_motion=gait_motion,
        foot_contact=contact,
        biomech_analysis=biomech,
        analysis_mode=str(analysis_mode or ""),
        max_frames=max_frames,
        metadata={"analysis_mode": str(analysis_mode or ""), "max_frames": max_frames},
    )


def pin_normal_vs_abnormal(cache: SessionCompareCache) -> list[str]:
    """Pin Normal (A) and Abnormal (B) demo snapshots. Return missing keys."""
    missing: list[str] = []
    for key in ("normal", "abnormal"):
        if cache.get(key) is not None:
            continue
        snap = try_build_demo_snapshot(key)
        if snap is None:
            missing.append(key)
        else:
            cache.pin(snap)
    cache.set_slot("left", "normal")
    cache.set_slot("right", "abnormal")
    return missing


def _demo_snapshot_from_poses(key: str) -> AnalyzedSessionSnapshot:
    """Build a demo snapshot from cached poses. Raises on load/analysis failure."""
    from stablewalk.io.pose_loader import load_pose_sequence

    poses_path = resolve_demo_poses_path(key)
    if poses_path is None:
        raise FileNotFoundError(f"No cached poses for '{key}'")
    sequence = load_pose_sequence(poses_path)
    frames_dir = resolve_demo_frames_dir(key)
    video_path = resolve_demo_video_path(key)
    from stablewalk.ui.analysis_mode import MODE_FULL

    return build_snapshot_from_sequence(
        sequence,
        key=key,
        label=_demo_label(key),
        source=str(video_path or poses_path),
        poses_path=poses_path,
        frames_dir=frames_dir,
        video_path=video_path,
        analysis_mode=MODE_FULL,
        max_frames=None,
    )


def try_build_demo_snapshot(key: str) -> AnalyzedSessionSnapshot | None:
    """Load an independent demo snapshot from cached poses (never from Session A)."""
    if resolve_demo_poses_path(key) is None:
        return None
    try:
        return _demo_snapshot_from_poses(key)
    except Exception:
        return None


def load_or_analyze_demo_session(
    key: str,
    *,
    video_path: str | None = None,
    pose_backend: str | None = None,
    max_frames: int | None = None,
    analysis_mode: str | None = None,
    left_video_identity: str | None = None,
) -> tuple[AnalyzedSessionSnapshot | None, str | None]:
    """Load cached demo analysis, or run the gait pipeline. Never reads Session A."""
    from stablewalk.ui.analysis_mode import MODE_FULL, MODE_PREVIEW, normalize_analysis_mode

    key = str(key or "").strip()
    requested = normalize_analysis_mode(analysis_mode, max_frames=max_frames) or MODE_FULL
    snap = try_build_demo_snapshot(key)
    if snap is not None:
        snap = hydrate_demo_snapshot(snap, key) or snap
        try:
            snap = isolate_analyzed_snapshot(snap, key=key)
        except Exception:
            pass
        if not getattr(snap, "analysis_mode", ""):
            snap.analysis_mode = MODE_FULL
            snap.max_frames = None
        src = str(
            getattr(snap, "video_path", None) or getattr(snap, "source", "") or ""
        )
        if left_video_identity and src and _recording_identity(src) == left_video_identity:
            snap = None
        elif requested == MODE_PREVIEW:
            # Cached dumps are full-source; do not silently use them as preview B.
            snap = None
        elif demo_snapshot_is_compare_ready(snap):
            return snap, None
    return run_independent_session_b_analysis(
        key,
        video_path=video_path,
        pose_backend=pose_backend,
        max_frames=max_frames,
        analysis_mode=requested,
        left_video_identity=left_video_identity,
    )


def build_demo_snapshot_with_reason(
    key: str,
) -> tuple[AnalyzedSessionSnapshot | None, str | None]:
    """Build Session B independently from demo poses, or the demo video.

    Never reads ``gui.sequence`` / Session A. Keeps the demo key (e.g. ``abnormal``)
    so Session B stays a separate slot. Returns ``(snapshot, None)`` on success
    or ``(None, technical_reason)`` on failure.
    """
    import traceback

    key = str(key or "").strip()
    if not key:
        return None, "No Session B recording is selected."

    poses_error: str | None = None
    poses_path = resolve_demo_poses_path(key)
    if poses_path is not None:
        try:
            return _demo_snapshot_from_poses(key), None
        except Exception as exc:
            poses_error = (
                f"Could not use cached poses for '{key}': {exc}\n"
                f"{traceback.format_exc()}".strip()
            )
            logger.exception("Session B cached poses failed for %s", key)

    video_raw = resolve_demo_video_path(key)
    video = Path(video_raw) if video_raw else None
    download_error: str | None = None
    if video is None or not video.is_file():
        try:
            from stablewalk.ui.media.demo_download import download_demo_video

            downloaded = bool(download_demo_video(key, force=False))
            if not downloaded:
                download_error = (
                    f"Demo download for '{key}' did not produce a video file."
                )
        except Exception as exc:
            download_error = (
                f"Demo download for '{key}' failed: {exc}\n"
                f"{traceback.format_exc()}".strip()
            )
            logger.exception("Session B demo download failed for %s", key)
        video_raw = resolve_demo_video_path(key)
        video = Path(video_raw) if video_raw else None
    if video is not None and video.is_file():
        try:
            from stablewalk.pose.backends.pipeline import (
                process_video_with_configured_backend,
            )

            sequence = process_video_with_configured_backend(video)
            snap = build_snapshot_from_sequence(
                sequence,
                key=key,
                label=_demo_label(key),
                source=str(video),
                poses_path=poses_path,
                frames_dir=resolve_demo_frames_dir(key),
                video_path=str(video),
            )
            return snap, None
        except Exception as exc:
            extra = f" Also: {poses_error}" if poses_error else ""
            tb = traceback.format_exc().strip()
            logger.exception("Session B video analysis failed for %s", key)
            return (
                None,
                f"Could not analyze Session B video ({video.name}): {exc}.{extra}\n{tb}".strip(),
            )

    parts = [
        f"No cached poses or video found for Session B ('{key}'). "
        "Place the recording in data/demo_videos (abnormal_gait_v2.mp4, "
        "normal_gait_v2.mp4, or fast_walking_v2.mp4) "
        "or use Load recording… to choose a file different from Session A."
    ]
    if poses_error:
        parts.append(poses_error)
    if download_error:
        parts.append(download_error)
    return None, "\n".join(parts)


def try_build_bundle_snapshot(key: str) -> AnalyzedSessionSnapshot | None:
    """Build one cached bundle snapshot off the Tk thread."""
    if not str(key or "").startswith("bundle:"):
        return None
    bundle_path = Path(str(key).removeprefix("bundle:"))
    if not bundle_path.is_dir():
        return None

    from stablewalk.io.pose_loader import load_pose_sequence
    from stablewalk.io.session_bundle import load_session_bundle

    loaded = load_session_bundle(bundle_path)
    meta = getattr(loaded, "metadata", None) or {}
    poses_path = bundle_path / "poses.json"
    if not poses_path.is_file():
        configured = meta.get("poses_json_path")
        poses_path = Path(str(configured)) if configured else poses_path
    if not poses_path.is_file():
        return None

    sequence = load_pose_sequence(poses_path)
    label = str(
        meta.get("display_name")
        or meta.get("bundle_name")
        or bundle_path.name
    )
    video_path = str(meta.get("video_source") or "")
    snap = build_snapshot_from_sequence(
        sequence,
        key=key,
        label=label,
        source=str(bundle_path),
        poses_path=poses_path,
        video_path=video_path,
    )
    snap.bundle_path = str(bundle_path)
    snap.analysis_date = str(meta.get("exported_at") or meta.get("saved_at") or "")
    snap.backend = str(
        (meta.get("workspace") or {}).get("pose_backend")
        or meta.get("backend")
        or "pose"
    )
    snap.valid_frame_count = int(meta.get("frame_count") or snap.n_frames)
    snap.pipeline_complete = True
    return snap


def try_build_video_snapshot(key: str) -> AnalyzedSessionSnapshot | None:
    """Analyze a separately selected recording for Compare in a worker."""
    if not str(key or "").startswith("video:"):
        return None
    video_path = Path(str(key).removeprefix("video:"))
    if not video_path.is_file():
        return None

    from stablewalk.pose.backends.pipeline import process_video_with_configured_backend

    sequence = process_video_with_configured_backend(video_path)
    return build_snapshot_from_sequence(
        sequence,
        key=key,
        label=video_path.stem.replace("_", " ").title() or video_path.name,
        source=str(video_path),
        video_path=str(video_path),
    )


def ensure_demo_sessions_pinned(cache: SessionCompareCache) -> list[str]:
    """Return demo keys already pinned — never analyze on the caller thread.

    Demo snapshots are prepared by Comparison Mode's background worker.
    """
    return [key for key in DEMO_COMPARE_KEYS if cache.get(key) is not None]


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
    # Never rebuild the analysis summary here — that runs on the Tk thread when
    # Compare pins the current session and can freeze the UI for seconds.

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
        path = np.asarray(path, dtype=float)
        if path.base is not None or not getattr(path.flags, "owndata", True):
            path = path.copy()

    # Keep at most a sparse skeleton sample for Compare — full per-frame caches
    # duplicate the live GUI store and dominate RSS after repeated Analyze runs.
    # Compare rebuilds missing frames lazily from ``sequence``.
    sparse_skeletons: dict[int, Skeleton3D] = {}
    if skeletons and pose_indices:
        step = max(1, len(pose_indices) // 24)
        for i, idx in enumerate(pose_indices):
            if i % step != 0 and i != len(pose_indices) - 1:
                continue
            sk = skeletons.get(idx)
            if sk is not None:
                sparse_skeletons[idx] = sk

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
        skeletons=sparse_skeletons,
        metrics=metrics,
        knee_t=list(knee_t),
        knee_y=list(knee_y),
        path_xyz=path,
        frame_paths=_frame_paths_from_dir(Path(frames_dir) if frames_dir else None),
        heel_strike_times_s=_heel_strike_times_s(getattr(gui, "_gait_cycle", None)),
        gait_cycle=getattr(gui, "_gait_cycle", None),
        gait_motion=getattr(gui, "gait_motion", None),
        foot_contact=getattr(gui, "_foot_contact", None),
        biomech_analysis=getattr(gui, "_biomech_analysis", None),
        analysis_mode=str(getattr(gui, "_analysis_mode", "") or ""),
        max_frames=getattr(gui, "_analysis_max_frames", None),
    )


def ensure_compare_cache(gui: Any) -> SessionCompareCache:
    cache = getattr(gui, "_compare_session_cache", None)
    if cache is None:
        cache = SessionCompareCache()
        gui._compare_session_cache = cache
    return cache


def pin_current_session(
    gui: Any, *, refresh: bool = True, slot: str | None = None
) -> AnalyzedSessionSnapshot | None:
    """Pin the active analysis so Comparison Mode can reuse it without re-analysis."""
    snap = build_snapshot_from_gui(gui)
    if snap is None:
        return None
    cache = ensure_compare_cache(gui)
    cache.pin(snap)
    if slot in ("left", "right"):
        cache.set_slot(slot, snap.key)
        other_key = cache.right_key if slot == "left" else cache.left_key
        if other_key == snap.key:
            if slot == "left":
                cache.right_key = ""
            else:
                cache.left_key = ""
    if refresh:
        refresh_fn = getattr(gui, "_refresh_comparison_mode", None)
        if callable(refresh_fn):
            try:
                refresh_fn()
            except Exception:
                pass
    return snap


__all__ = [
    "DEMO_COMPARE_KEYS",
    "AnalyzedSessionSnapshot",
    "SessionCompareCache",
    "analyze_video",
    "build_demo_snapshot_with_reason",
    "build_snapshot_from_gui",
    "build_snapshot_from_sequence",
    "demo_snapshot_is_compare_ready",
    "ensure_compare_cache",
    "ensure_demo_sessions_pinned",
    "hydrate_demo_snapshot",
    "isolate_analyzed_snapshot",
    "load_or_analyze_demo_session",
    "log_compare_debug",
    "log_compare_step",
    "log_session_b_debug",
    "pin_current_session",
    "pin_normal_vs_abnormal",
    "resolve_demo_poses_path",
    "resolve_session_b_video_path",
    "run_independent_session_b_analysis",
    "sessions_are_identical",
    "try_build_bundle_snapshot",
    "try_build_demo_snapshot",
    "try_build_video_snapshot",
]
