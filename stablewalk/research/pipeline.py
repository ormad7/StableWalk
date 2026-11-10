"""Independent pose + analysis pipeline for one research clip.

Each call owns its PoseSequence and GaitMotionRecording. The live GUI
session is never mutated. Video A and Video B must each be analyzed by a
separate invocation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stablewalk.research.dataset import ClipMetadata
from stablewalk.research.probe import is_remote_url

KIND_DERIVED = "derived"


@dataclass
class IndependentClipResult:
    """One video's analysis products — never aliased onto another slot."""

    video_id: str
    label: str
    metadata: ClipMetadata | None
    sequence: Any
    gait_motion: Any
    cycles: Any
    spectral: Any
    asymmetry: Any
    warnings: list[str] = field(default_factory=list)
    pose_backend: str = ""
    n_frames: int = 0
    fps: float = 0.0
    quality: dict[str, Any] = field(default_factory=dict)

    @property
    def sequence_id(self) -> int:
        return id(self.sequence)

    @property
    def gait_motion_id(self) -> int:
        return id(self.gait_motion)

    @property
    def ready(self) -> bool:
        return self.gait_motion is not None and self.n_frames > 0

    def identity(self) -> dict[str, Any]:
        path = None
        if self.metadata is not None and self.metadata.resolved_path is not None:
            path = str(self.metadata.resolved_path)
        return {
            "video_id": self.video_id,
            "sequence_id": self.sequence_id,
            "gait_motion_id": self.gait_motion_id,
            "cycles_id": id(self.cycles) if self.cycles is not None else None,
            "spectral_id": id(self.spectral) if self.spectral is not None else None,
            "asymmetry_id": id(self.asymmetry) if self.asymmetry is not None else None,
            "path": path,
            "n_frames": self.n_frames,
            "fps": self.fps,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "label": self.label,
            "metadata": None if self.metadata is None else self.metadata.to_dict(),
            "pose_backend": self.pose_backend,
            "n_frames": self.n_frames,
            "fps": self.fps,
            "quality": dict(self.quality),
            "warnings": list(self.warnings),
            "identity": self.identity(),
            "kind": KIND_DERIVED,
            "spectral_available": bool(
                self.spectral is not None and getattr(self.spectral, "available", False)
            ),
            "asymmetry_available": bool(
                self.asymmetry is not None and getattr(self.asymmetry, "available", False)
            ),
        }


def slots_are_independent(left: IndependentClipResult, right: IndependentClipResult) -> bool:
    """True when A and B do not share pose / analysis objects.

    Same source file with two separate pose runs is allowed (distinct objects).
    Shared sequence, gait_motion, or cycles objects are never independent.
    """
    if left is right:
        return False
    if left.sequence is not None and left.sequence is right.sequence:
        return False
    if left.gait_motion is not None and left.gait_motion is right.gait_motion:
        return False
    if left.cycles is not None and left.cycles is right.cycles:
        return False
    if left.spectral is not None and left.spectral is right.spectral:
        return False
    if left.asymmetry is not None and left.asymmetry is right.asymmetry:
        return False
    return True


def slots_share_source_file(
    left: IndependentClipResult, right: IndependentClipResult
) -> bool:
    """True when both slots resolve to the same video path (case-insensitive)."""
    if left.video_id and right.video_id and left.video_id == right.video_id:
        return True
    path_a = left.identity().get("path")
    path_b = right.identity().get("path")
    if not path_a or not path_b:
        return False
    try:
        return Path(path_a).expanduser().resolve(strict=False) == Path(
            path_b
        ).expanduser().resolve(strict=False)
    except (OSError, RuntimeError, ValueError):
        return str(path_a).replace("\\", "/").casefold() == str(path_b).replace(
            "\\", "/"
        ).casefold()


def _quality(sequence, recording, cycles) -> dict[str, Any]:
    n = 0
    detected = 0
    if sequence is not None:
        frames = list(getattr(sequence, "frames", None) or [])
        n = len(frames)
        detected = sum(1 for f in frames if getattr(f, "detected", False))
    elif recording is not None:
        n = int(getattr(recording, "frame_count", 0) or 0)
        detected = n
    n_cycles = len(getattr(cycles, "cycles", None) or []) if cycles is not None else 0
    return {
        "n_frames": n,
        "detected_frames": detected,
        "detected_fraction": (detected / n) if n else 0.0,
        "gait_cycle_count": n_cycles,
        "kind": KIND_DERIVED,
    }


def analyze_independent_clip(
    *,
    video_path: str | Path | None = None,
    sequence=None,
    recording=None,
    metadata: ClipMetadata | None = None,
    video_id: str = "",
    label: str = "",
    run_pose: bool = False,
    max_frames: int | None = None,
) -> IndependentClipResult:
    """Run pose (optional) and analysis for one clip only.

    Pass distinct ``sequence`` / ``recording`` objects for Video A and Video B.
    ``run_pose=True`` requires a local video path; remote URLs are rejected.
    """
    from stablewalk.adapters.pose_adapter import pose_sequence_to_gait_motion
    from stablewalk.analysis.asymmetry3d.report import analyze_asymmetry3d
    from stablewalk.analysis.gait_cycle_analysis import analyze_gait_cycles
    from stablewalk.analysis.spectral.report import analyze_spectral

    warnings: list[str] = []
    backend = ""
    vid = video_id or (metadata.video_id if metadata else "") or "clip"
    display = label or (metadata.category_label if metadata else vid)

    if run_pose:
        if video_path is None:
            raise ValueError("run_pose=True requires video_path")
        if is_remote_url(video_path):
            raise ValueError("Pose estimation does not download remote videos.")
        path = Path(video_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video not found: {path}")
        from stablewalk.pose.backends.pipeline_runner import extract_pose_from_video

        extraction = extract_pose_from_video(path, max_frames=max_frames, enrich=True)
        sequence = extraction.sequence
        backend = str(getattr(extraction.resolution, "used", "") or "")
        if recording is not None:
            warnings.append("Provided recording ignored because pose was re-extracted.")
            recording = None

    if recording is None and sequence is not None:
        recording = pose_sequence_to_gait_motion(sequence)
    if recording is None:
        raise ValueError("No pose sequence or gait recording for this clip.")

    n_frames = int(getattr(recording, "frame_count", 0) or 0)
    fps = float(getattr(recording, "fps", 0.0) or 0.0)
    cycles = None
    try:
        cycles = analyze_gait_cycles(recording)
    except Exception as exc:
        warnings.append(f"Gait-cycle analysis failed: {exc}")

    spectral = None
    try:
        spectral = analyze_spectral(recording, sequence=sequence, cycles=cycles)
    except Exception as exc:
        warnings.append(f"Spectral analysis failed: {exc}")

    asymmetry = None
    try:
        asymmetry = analyze_asymmetry3d(
            recording,
            cycles=cycles,
            sequence=sequence,
            detect_if_missing=False,
        )
    except Exception as exc:
        warnings.append(f"Asymmetry analysis failed: {exc}")

    return IndependentClipResult(
        video_id=vid,
        label=display,
        metadata=metadata,
        sequence=sequence,
        gait_motion=recording,
        cycles=cycles,
        spectral=spectral,
        asymmetry=asymmetry,
        warnings=warnings,
        pose_backend=backend,
        n_frames=n_frames,
        fps=fps,
        quality=_quality(sequence, recording, cycles),
    )
