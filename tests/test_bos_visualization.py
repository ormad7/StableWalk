"""Tests for Base of Support floor visualization helpers."""

from __future__ import annotations

from stablewalk.ui.viewers.gait_skeleton_renderer import (
    COM_TRAIL_FRAME_COUNT,
    _line_to_floor_strip,
    _stability_overlay_color,
    _stability_overlay_edge_color,
    _stability_polygon_style,
    _support_edge_accent,
)


def test_stability_polygon_style_maps_green_yellow_red() -> None:
    stable = _stability_polygon_style("Stable")
    reduced = _stability_polygon_style("Reduced Stability")
    unstable = _stability_polygon_style("Unstable")
    assert stable[0] == "#22c55e"
    assert reduced[0] == "#eab308"
    assert unstable[0] == "#ef4444"
    # BoS remains visible but subordinate to the articulated body.
    assert 0.08 <= stable[2] <= 0.18
    assert unstable[3] >= stable[3]


def test_com_stability_colors_match_bos_palette() -> None:
    assert _stability_overlay_color("Stable") == "#22c55e"
    assert _stability_overlay_color("Reduced Stability") == "#eab308"
    assert _stability_overlay_color("Unstable") == "#ef4444"
    assert _stability_overlay_edge_color("Stable") == "#16a34a"
    assert _stability_overlay_edge_color("Reduced Stability") == "#ca8a04"
    assert _stability_overlay_edge_color("Unstable") == "#dc2626"


def test_com_trail_frame_count_reasonable() -> None:
    assert COM_TRAIL_FRAME_COUNT >= 10


def test_support_edge_accent_single_and_double_stance() -> None:
    assert _support_edge_accent("left_stance") is not None
    assert _support_edge_accent("right_stance") is not None
    assert _support_edge_accent("double_support") is not None
    assert _support_edge_accent("swing") is None


def test_line_to_floor_strip_expands_degenerate_and_segment() -> None:
    quad = _line_to_floor_strip((0.0, 0.0), (0.0, 0.0), height=1.0)
    assert len(quad) == 4
    strip = _line_to_floor_strip((0.0, 0.0), (0.2, 0.0), height=1.0)
    assert len(strip) == 4
    assert strip[0][1] == strip[1][1]
