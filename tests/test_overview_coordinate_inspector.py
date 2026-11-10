"""Unit tests for Overview 3D Coordinate Inspector formatting."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
from stablewalk.models.pose_data import Keypoint, PoseFrame, PoseSequence
from stablewalk.ui.overview_coordinate_inspector import (
    COORDINATE_INSPECTOR_DETAILS_KEYS,
    COORDINATE_INSPECTOR_FIELD_FAMILY,
    COORDINATE_INSPECTOR_KEYS,
    COORDINATE_INSPECTOR_PRIMARY_KEYS,
    COORDINATE_INSPECTOR_SECTIONS,
    COORDINATE_INSPECTOR_SELECTED_KEYS,
    INSPECTOR_UNAVAILABLE_DISPLAY,
    PATH_EXTENT_KEYS,
    apply_coordinate_inspector_labels,
    build_coordinate_inspector_fields,
    build_path_extent_fields,
    empty_coordinate_inspector_fields,
    empty_path_extent_fields,
    inspector_value_color,
    normalize_inspector_display,
    tip_xyz_cm_from_cache,
)
from stablewalk.ui.theme import SELECTED_JOINT_ACCENT, SELECTED_JOINT_ACCENT_BGR
from stablewalk.ui.value_provenance import NA_DISPLAY


def _snap(
    *,
    frame: int = 52,
    time_s: float = 2.12,
    joint_id: str = "right_knee",
    pos: tuple[float, float, float] = (0.124, -0.047, 0.918),
    vel: tuple[float, float, float] | None = (0.3, 0.2, 0.2),
    angle: float | None = 37.2,
) -> SkeletonSnapshot:
    vv = Vec3(vel[0], vel[1], vel[2]) if vel is not None else None
    speed = None
    if vv is not None:
        speed = (vv.x**2 + vv.y**2 + vv.z**2) ** 0.5
    joint = JointSample(
        joint_id=joint_id,
        position=Vec3(pos[0], pos[1], pos[2]),
        angle_deg=angle,
        velocity=speed,
        velocity_vector=vv,
    )
    return SkeletonSnapshot(
        frame_index=frame,
        time_s=time_s,
        joints={joint_id: joint},
    )


class OverviewCoordinateInspectorTests(unittest.TestCase):
    def test_empty_fields_are_na(self) -> None:
        fields = empty_coordinate_inspector_fields()
        self.assertEqual(set(fields), set(COORDINATE_INSPECTOR_KEYS))
        self.assertTrue(all(v == NA_DISPLAY for v in fields.values()))

    def test_unavailable_display_is_em_dash(self) -> None:
        self.assertEqual(INSPECTOR_UNAVAILABLE_DISPLAY, "—")
        self.assertEqual(normalize_inspector_display(""), "—")
        self.assertEqual(normalize_inspector_display("   "), "—")
        self.assertEqual(normalize_inspector_display(NA_DISPLAY), "—")
        self.assertEqual(normalize_inspector_display(f"{NA_DISPLAY} — missing"), "—")
        self.assertEqual(normalize_inspector_display("12.4 cm"), "12.4 cm")
        self.assertEqual(normalize_inspector_display("Low (20%)"), "Low (20%)")

    def test_sections_group_fields(self) -> None:
        self.assertEqual(len(COORDINATE_INSPECTOR_SECTIONS), 4)
        ids = [s[0] for s in COORDINATE_INSPECTOR_SECTIONS]
        self.assertEqual(ids, ["identity", "position", "motion", "quality"])
        self.assertEqual(
            COORDINATE_INSPECTOR_SECTIONS[0][2],
            ("joint_name", "frame", "time"),
        )
        self.assertEqual(COORDINATE_INSPECTOR_SECTIONS[1][2], ("x", "y", "z"))
        self.assertEqual(
            COORDINATE_INSPECTOR_SECTIONS[2][2],
            ("velocity", "acceleration", "angle", "rom"),
        )
        self.assertEqual(
            COORDINATE_INSPECTOR_SECTIONS[3][2],
            ("tracking_confidence", "pose_confidence", "visibility"),
        )
        self.assertEqual(COORDINATE_INSPECTOR_FIELD_FAMILY["x"], "position")
        self.assertEqual(COORDINATE_INSPECTOR_FIELD_FAMILY["velocity"], "motion")
        self.assertEqual(
            COORDINATE_INSPECTOR_FIELD_FAMILY["pose_confidence"], "quality"
        )
        self.assertEqual(
            COORDINATE_INSPECTOR_PRIMARY_KEYS,
            (
                "joint_name",
                "frame",
                "time",
                "x",
                "y",
                "z",
                "velocity",
                "acceleration",
                "angle",
                "tracking_confidence",
            ),
        )
        self.assertEqual(
            frozenset(COORDINATE_INSPECTOR_PRIMARY_KEYS)
            | frozenset(COORDINATE_INSPECTOR_DETAILS_KEYS),
            frozenset(COORDINATE_INSPECTOR_KEYS),
        )

    def test_value_color_by_family(self) -> None:
        colors = {
            "identity": "#fff",
            "position": "#0af",
            "motion": "#fa0",
            "quality": "#0f0",
        }
        self.assertEqual(
            inspector_value_color(
                "x", text="12.4 cm", family_colors=colors, muted_fg="#888", text_fg="#fff"
            ),
            "#0af",
        )
        self.assertEqual(
            inspector_value_color(
                "visibility",
                text="Low (30%)",
                family_colors=colors,
                muted_fg="#888",
                text_fg="#fff",
            ),
            "#888",
        )

    def test_tip_cache_reader(self) -> None:
        ax = SimpleNamespace(_stablewalk_tip_xyz_cm=(12.4, -4.7, 91.8))
        self.assertEqual(tip_xyz_cm_from_cache(ax), (12.4, -4.7, 91.8))
        self.assertIsNone(tip_xyz_cm_from_cache(None))
        self.assertIsNone(
            tip_xyz_cm_from_cache(SimpleNamespace(_stablewalk_tip_xyz_cm=None))
        )

    def test_build_uses_tip_and_velocity_units(self) -> None:
        snap = _snap()
        prev = _snap(
            frame=51,
            time_s=2.08,
            vel=(0.1, 0.0, 0.0),
        )
        recording = SimpleNamespace(
            frame_count=120,
            fps=25.0,
            snapshot_at=lambda i: prev if i == 51 else snap if i == 52 else None,
        )
        state = SimpleNamespace(
            total_frames=120,
            display_frame_number=53,
            current_frame_index=52,
            current_time_seconds=2.12,
            duration_seconds=4.76,
        )
        fields = build_coordinate_inspector_fields(
            "right_knee",
            snapshot=snap,
            recording=recording,
            sequence=None,
            playback_state=state,
            tip_xyz_cm=(12.4, -4.7, 91.8),
            session_rom_display="40.0°",
            list_pos=52,
        )
        self.assertIn("knee", fields["joint_name"].lower())
        self.assertTrue(fields["joint_name"].lower().startswith("right"))
        self.assertEqual(fields["frame"], "53 / 120")
        self.assertEqual(fields["time"], "2.12 s / 4.76 s")
        self.assertEqual(fields["x"].strip(), "12.4 cm")
        self.assertEqual(fields["y"].strip(), "-4.7 cm")
        self.assertEqual(fields["z"].strip(), "91.8 cm")
        self.assertTrue(fields["velocity"].endswith(" m/s"))
        self.assertFalse(fields["velocity"].lstrip().startswith(NA_DISPLAY))
        self.assertTrue(fields["acceleration"].endswith(" m/s²"))
        self.assertFalse(fields["acceleration"].lstrip().startswith(NA_DISPLAY))
        self.assertEqual(fields["angle"].strip(), "37.2°")
        self.assertEqual(fields["rom"], "40.0°")
        self.assertIn("tracking_confidence", fields)
        self.assertIn("pose_confidence", fields)
        self.assertIn("visibility", fields)

    def test_missing_tip_and_velocity_are_na(self) -> None:
        snap = _snap(vel=None, angle=None)
        fields = build_coordinate_inspector_fields(
            "right_knee",
            snapshot=snap,
            tip_xyz_cm=None,
            session_rom_display=None,
            list_pos=0,
        )
        self.assertTrue(fields["x"].startswith(NA_DISPLAY))
        self.assertTrue(fields["velocity"].startswith(NA_DISPLAY))
        self.assertTrue(fields["acceleration"].startswith(NA_DISPLAY))
        self.assertTrue(fields["rom"].startswith(NA_DISPLAY))
        self.assertTrue(fields["angle"].startswith(NA_DISPLAY))

    def test_low_confidence_label(self) -> None:
        snap = _snap()
        kp = Keypoint(name="right_knee", x=0.0, y=0.0, z=0.0, visibility=0.2)
        frames = [
            PoseFrame(
                frame_index=i,
                image_path="",
                timestamp_s=i / 25.0,
                keypoints=[kp],
            )
            for i in range(53)
        ]
        sequence = PoseSequence(source_video="", fps=25.0, frames=frames)
        fields = build_coordinate_inspector_fields(
            "right_knee",
            snapshot=snap,
            sequence=sequence,
            tip_xyz_cm=(1.0, 2.0, 3.0),
            session_rom_display="12°",
            list_pos=52,
        )
        self.assertEqual(fields["pose_confidence"].strip(), "20%")
        self.assertTrue(fields["visibility"].startswith("Low"))
        self.assertIn("20%", fields["visibility"])

    def test_tracking_visibility_maps_playback_position_to_source_frame(self) -> None:
        frames = [
            PoseFrame(
                frame_index=i,
                image_path="",
                timestamp_s=i / 25.0,
                keypoints=[
                    Keypoint(
                        name="right_knee",
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        visibility=visibility,
                    )
                ],
            )
            for i, visibility in enumerate((0.1, 0.2, 0.9))
        ]
        sequence = PoseSequence(source_video="", fps=25.0, frames=frames)
        gui = SimpleNamespace(_biomech_analysis=None, pose_indices=[0, 2])

        fields = build_coordinate_inspector_fields(
            "right_knee",
            sequence=sequence,
            list_pos=1,
            gui=gui,
        )

        self.assertIn("90%", fields["tracking_confidence"])

    def test_no_hardcoded_placeholder_numbers(self) -> None:
        fields = build_coordinate_inspector_fields(None)
        for key, value in fields.items():
            self.assertNotIn("12.4", value)
            self.assertNotIn("0.42", value)
            self.assertTrue(
                value == NA_DISPLAY or value.startswith(NA_DISPLAY),
                msg=f"{key}={value!r}",
            )

    def test_selected_joint_accent_is_shared(self) -> None:
        self.assertEqual(SELECTED_JOINT_ACCENT, "#ffd666")
        self.assertEqual(SELECTED_JOINT_ACCENT_BGR, (102, 214, 255))
        self.assertIn("joint_name", COORDINATE_INSPECTOR_SELECTED_KEYS)
        self.assertIn("x", COORDINATE_INSPECTOR_SELECTED_KEYS)
        colors = {
            "identity": "#fff",
            "position": "#0af",
            "motion": "#fa0",
            "quality": "#0f0",
        }
        self.assertEqual(
            inspector_value_color(
                "x",
                text="12.4 cm",
                family_colors=colors,
                muted_fg="#888",
                text_fg="#fff",
                joint_selected=True,
                selected_accent=SELECTED_JOINT_ACCENT,
            ),
            SELECTED_JOINT_ACCENT,
        )
        # Live playback highlight overrides the selected-joint accent.
        self.assertEqual(
            inspector_value_color(
                "x",
                text="12.5 cm",
                family_colors=colors,
                muted_fg="#888",
                text_fg="#fff",
                changed=True,
                highlight_fg="#ffffff",
                joint_selected=True,
                selected_accent=SELECTED_JOINT_ACCENT,
            ),
            "#ffffff",
        )

    def test_apply_labels_uses_selected_accent(self) -> None:
        class _Lbl:
            def __init__(self) -> None:
                self.text = "N/A"
                self.fg = "#fff"

            def configure(self, **kwargs) -> None:
                if "text" in kwargs:
                    self.text = kwargs["text"]
                if "fg" in kwargs:
                    self.fg = kwargs["fg"]

        labels = {k: _Lbl() for k in ("joint_name", "x", "frame")}
        fields = {
            "joint_name": "Right Knee",
            "x": "12.4 cm",
            "frame": "1 / 10",
        }
        apply_coordinate_inspector_labels(
            labels,
            fields,
            text_fg="#fff",
            muted_fg="#888",
            last_fields=None,
            family_colors={"identity": "#fff", "position": "#0af"},
            joint_selected=True,
            selected_accent=SELECTED_JOINT_ACCENT,
            force_recolor=True,
        )
        self.assertEqual(labels["joint_name"].fg, SELECTED_JOINT_ACCENT)
        self.assertEqual(labels["x"].fg, SELECTED_JOINT_ACCENT)
        self.assertEqual(labels["frame"].fg, "#fff")

    def test_apply_labels_paints_em_dash_for_unavailable(self) -> None:
        class _Lbl:
            def __init__(self) -> None:
                self.text = ""
                self.fg = "#fff"

            def configure(self, **kwargs) -> None:
                if "text" in kwargs:
                    self.text = kwargs["text"]
                if "fg" in kwargs:
                    self.fg = kwargs["fg"]

        labels = {k: _Lbl() for k in ("x", "velocity", "rom")}
        apply_coordinate_inspector_labels(
            labels,
            {"x": "", "velocity": NA_DISPLAY, "rom": f"{NA_DISPLAY} — missing"},
            text_fg="#fff",
            muted_fg="#888",
            last_fields=None,
            family_colors={"position": "#0af", "motion": "#fa0"},
            force_recolor=True,
        )
        self.assertEqual(labels["x"].text, "—")
        self.assertEqual(labels["velocity"].text, "—")
        self.assertEqual(labels["rom"].text, "—")
        self.assertEqual(labels["x"].fg, "#888")

    def test_path_extent_fields_from_bounds(self) -> None:
        empty = empty_path_extent_fields()
        self.assertEqual(set(empty), set(PATH_EXTENT_KEYS))
        self.assertTrue(all(v == NA_DISPLAY for v in empty.values()))

        ax = SimpleNamespace(
            _stablewalk_path_bounds_m=(
                (-0.10, 0.20),
                (0.0, 0.15),
                (-0.05, 0.25),
            )
        )
        fields = build_path_extent_fields(ax)
        self.assertIn("×", fields["bounding_box"])
        self.assertTrue(fields["bounding_box"].endswith("cm"))
        self.assertIn("…", fields["x_range"])
        self.assertIn("-10.0", fields["x_range"])
        self.assertIn("20.0", fields["x_range"])
        self.assertIn("…", fields["y_range"])
        self.assertIn("…", fields["z_range"])

        # Fallback from full path XYZ cache when bounds attr missing.
        ax2 = SimpleNamespace(
            _stablewalk_path_bounds_m=None,
            _stablewalk_full_path_xyz=(
                "key",
                [0.0, 0.1],
                [0.0, 0.2],
                [0.0, 0.3],
                [],
            ),
        )
        fields2 = build_path_extent_fields(ax2)
        self.assertIn("10.0", fields2["x_range"])
        self.assertIn("30.0", fields2["z_range"])


if __name__ == "__main__":
    unittest.main()
