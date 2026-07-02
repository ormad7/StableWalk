"""
Gait cycle analysis: phases, heel strike, toe-off, stability hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from stablewalk.gait_events import GaitEvent, analyze_gait_sequence
from stablewalk.models.pose_data import PoseSequence


class GaitLabel(Enum):
    UNKNOWN = "unknown"
    STABLE = "stable"
    UNSTABLE = "unstable"


@dataclass
class GaitComparisonResult:
    """Result of comparing a sequence to a reference (future use)."""

    label: GaitLabel
    stability_score: float | None = None
    notes: str = ""


class GaitCycleAnalyzer:
    """Detect gait phases and key events (heel strike, toe-off) from pose sequence."""

    def analyze_events(self, sequence: PoseSequence) -> list[GaitEvent]:
        events, _ = analyze_gait_sequence(sequence.frames)
        return events

    def event_summary(self, sequence: PoseSequence) -> str:
        if sequence.gait_events_timeline:
            lines = [
                f"  frame {e['frame']}: {e['side']} {e['event']}"
                for e in sequence.gait_events_timeline
            ]
            return "Gait events:\n" + "\n".join(lines[:20])
        events = self.analyze_events(sequence)
        if not events:
            return "No gait events detected."
        return "Gait events:\n" + "\n".join(
            f"  frame {e.frame_index}: {e.side} {e.event_type}" for e in events[:20]
        )


def build_robot_walk_simulation(sequence: PoseSequence):
    """Create a robotic walk simulation from a processed pose sequence."""
    from stablewalk.walk_simulator import WalkSimulator

    return WalkSimulator().from_pose_sequence(sequence)


class GaitStabilityAnalyzer:
    """
    Placeholder for Step 7+ / bonus: compare walking patterns.

    Will use joint-angle variability, symmetry, and COM proxy metrics.
    """

    def analyze(self, sequence: PoseSequence) -> GaitComparisonResult:
        return GaitComparisonResult(
            label=GaitLabel.UNKNOWN,
            notes="Stability analysis not implemented yet.",
        )

    def compare(
        self,
        reference: PoseSequence,
        sample: PoseSequence,
    ) -> GaitComparisonResult:
        return GaitComparisonResult(
            label=GaitLabel.UNKNOWN,
            notes="Reference comparison not implemented yet.",
        )
