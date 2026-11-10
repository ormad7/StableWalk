"""Analyzed-span vs valid-duration metrics from original timestamps.

Interval-counting convention (do not use 1/fps as a substitute):
  A contiguous run of n valid frames contributes ``t[n-1] − t[0]``
  (n−1 adjacent intervals). An isolated valid frame contributes 0 s.
  Analyzed span is last_analyzed_timestamp − first_analyzed_timestamp
  and includes internal gaps. Cumulative valid duration sums run lengths.
  Coverage is cumulative_valid_duration / analyzed_span (0 when span is 0).
  Longest valid segment is the maximum run length.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

INTERVAL_COUNTING_CONVENTION = (
    "Durations use original source timestamps, not 1/fps. "
    "A contiguous run of n valid frames contributes t[n-1] − t[0] "
    "(n−1 adjacent intervals). An isolated valid frame contributes 0 s. "
    "Analyzed span is last_analyzed_timestamp − first_analyzed_timestamp "
    "and includes internal gaps. Cumulative valid duration is the sum of "
    "those run lengths. Coverage = cumulative_valid_duration / analyzed_span. "
    "Longest valid segment is the maximum run length."
)


@dataclass(frozen=True)
class TimelineDurationMetrics:
    analyzed_span_s: float
    cumulative_valid_duration_s: float
    coverage: float
    longest_valid_segment_s: float
    n_analyzed_frames: int
    n_valid_frames: int
    n_valid_runs: int
    first_timestamp_s: float | None
    last_timestamp_s: float | None
    convention: str = INTERVAL_COUNTING_CONVENTION

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["coverage_pct"] = round(self.coverage * 100.0, 3)
        return payload


def _is_connected(prev_t: float, next_t: float, fps: float) -> bool:
    """True when two timestamps belong to the same valid run."""
    if fps <= 1e-6:
        return next_t > prev_t
    gap = next_t - prev_t
    if gap <= 0.0:
        return False
    # Allow one missed sample; larger jumps start a new run.
    return gap <= (2.5 / fps)


def compute_timeline_duration_metrics(
    timestamps_s: Sequence[float],
    valid_mask: Sequence[bool],
    *,
    fps: float = 0.0,
) -> TimelineDurationMetrics:
    """Compute analyzed span / cumulative valid duration from original times."""
    n = min(len(timestamps_s), len(valid_mask))
    times = [float(timestamps_s[i]) for i in range(n)]
    valid = [bool(valid_mask[i]) for i in range(n)]
    if not times:
        return TimelineDurationMetrics(
            analyzed_span_s=0.0,
            cumulative_valid_duration_s=0.0,
            coverage=0.0,
            longest_valid_segment_s=0.0,
            n_analyzed_frames=0,
            n_valid_frames=0,
            n_valid_runs=0,
            first_timestamp_s=None,
            last_timestamp_s=None,
        )
    analyzed_span = max(times) - min(times)
    n_valid = sum(1 for flag in valid if flag)
    runs: list[tuple[float, float]] = []
    run_start: float | None = None
    run_last: float | None = None
    for t, is_valid in zip(times, valid):
        if not is_valid:
            if run_start is not None and run_last is not None:
                runs.append((run_start, run_last))
            run_start = None
            run_last = None
            continue
        if run_start is None:
            run_start = t
            run_last = t
            continue
        assert run_last is not None
        if _is_connected(run_last, t, fps):
            run_last = t
        else:
            runs.append((run_start, run_last))
            run_start = t
            run_last = t
    if run_start is not None and run_last is not None:
        runs.append((run_start, run_last))
    lengths = [max(0.0, end - start) for start, end in runs]
    cumulative = float(sum(lengths))
    longest = float(max(lengths)) if lengths else 0.0
    coverage = (cumulative / analyzed_span) if analyzed_span > 1e-12 else 0.0
    return TimelineDurationMetrics(
        analyzed_span_s=float(analyzed_span),
        cumulative_valid_duration_s=cumulative,
        coverage=min(max(coverage, 0.0), 1.0),
        longest_valid_segment_s=longest,
        n_analyzed_frames=n,
        n_valid_frames=n_valid,
        n_valid_runs=len(runs),
        first_timestamp_s=times[0],
        last_timestamp_s=times[-1],
    )


def valid_pose_mask(frames: Iterable[object]) -> list[bool]:
    """Valid gait evidence: detected and not skipped."""
    out: list[bool] = []
    for frame in frames:
        detected = bool(getattr(frame, "detected", False))
        skipped = bool(getattr(frame, "skipped", False))
        out.append(detected and not skipped)
    return out


@dataclass(frozen=True)
class IntervalPoseEvidence:
    """Pose support inside one HS–HS interval. Gaps are not bridged."""

    n_pose_frames: int
    n_valid_pose_frames: int
    valid_pose_coverage: float
    longest_internal_gap_s: float
    n_unsupported_gaps: int


def cycle_interval_pose_evidence(
    start_time_s: float,
    end_time_s: float,
    timestamps_s: Sequence[float],
    valid_mask: Sequence[bool],
    *,
    fps: float,
) -> IntervalPoseEvidence:
    """Coverage and longest unsupported pose gap inside ``[start, end]``.

    Consecutive valid timestamps farther apart than one missed sample
    (``2.5 / fps``) are an unsupported gap. Interval endpoints that are not
    connected to the first/last valid pose also count. Callers must reject
    those intervals instead of interpolating across the hole.
    """
    duration = max(0.0, float(end_time_s) - float(start_time_s))
    n = min(len(timestamps_s), len(valid_mask))
    times: list[float] = []
    valid: list[bool] = []
    for i in range(n):
        t = float(timestamps_s[i])
        if start_time_s - 1e-9 <= t <= end_time_s + 1e-9:
            times.append(t)
            valid.append(bool(valid_mask[i]))
    n_pose = len(times)
    n_ok = sum(1 for flag in valid if flag)
    if duration <= 1e-12:
        return IntervalPoseEvidence(n_pose, n_ok, 0.0, 0.0, 0)

    metrics = compute_timeline_duration_metrics(times, valid, fps=fps)
    coverage = (
        metrics.cumulative_valid_duration_s / duration if duration > 1e-12 else 0.0
    )
    coverage = min(max(coverage, 0.0), 1.0)

    def _same_or_connected(prev: float, nxt: float) -> bool:
        if abs(nxt - prev) <= 1e-9:
            return True
        return _is_connected(prev, nxt, fps)

    valid_times = [t for t, flag in zip(times, valid) if flag]
    gaps: list[float] = []
    if not valid_times:
        gaps.append(duration)
    else:
        if not _same_or_connected(float(start_time_s), valid_times[0]):
            gaps.append(valid_times[0] - float(start_time_s))
        for prev, nxt in zip(valid_times, valid_times[1:]):
            if not _same_or_connected(prev, nxt):
                gaps.append(nxt - prev)
        if not _same_or_connected(valid_times[-1], float(end_time_s)):
            gaps.append(float(end_time_s) - valid_times[-1])
    longest = max(gaps) if gaps else 0.0
    return IntervalPoseEvidence(
        n_pose_frames=n_pose,
        n_valid_pose_frames=n_ok,
        valid_pose_coverage=coverage,
        longest_internal_gap_s=float(max(0.0, longest)),
        n_unsupported_gaps=len(gaps),
    )
