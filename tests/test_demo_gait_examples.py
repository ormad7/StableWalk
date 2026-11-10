"""Example menu labels, About metadata, v2 paths, and no legacy fallback."""

from __future__ import annotations

from pathlib import Path

from stablewalk.ui.media.demo_gait import (
    DEMO_GAIT_EXAMPLES,
    demo_gait_interpretation,
    demo_path,
    example_by_key,
    format_about_active_example,
    is_demo_video_path,
    is_legacy_demo_filename,
    missing_file_message,
)


def test_example_menu_labels_use_source_conditions():
    names = {ex.key: ex.display_name for ex in DEMO_GAIT_EXAMPLES}
    assert names["abnormal"] == "Abnormal Gait"
    assert names["normal"] == "Normal Gait"
    assert names["athletic"] == "Fast Walking"
    assert example_by_key("athletic").button_label == "Fast Walking"


def test_official_filenames_are_v2_not_legacy():
    files = {ex.key: ex.filename for ex in DEMO_GAIT_EXAMPLES}
    assert files["abnormal"] == "abnormal_gait_v2.mp4"
    assert files["normal"] == "normal_gait_v2.mp4"
    assert files["athletic"] == "fast_walking_v2.mp4"
    for name in files.values():
        assert not is_legacy_demo_filename(name)


def test_about_text_lists_source_fields_not_scores():
    ex = example_by_key("normal")
    text = format_about_active_example(ex)
    for needle in (
        "Source URL:",
        "Dataset:",
        "Documented condition / activity:",
        "License / access:",
        "Original FPS",
        "Duration",
        "Selected time interval:",
        "gait_pat=normal",
        "YouTube",
    ):
        assert needle in text
    assert "usable cycle" not in text.lower()
    assert "/100" not in text


def test_abnormal_about_preserves_parkinsons_not_dementia_or_neuropathic():
    text = format_about_active_example(example_by_key("abnormal"))
    assert "parkinsons" in text.lower() or "Parkinson" in text
    assert "Not dementia" in text
    assert "Neuropathic" not in text


def test_fast_walking_about_uses_twente_walk_54():
    text = format_about_active_example(example_by_key("athletic"))
    assert "Fast Walking" in text
    assert "6644593" in text
    assert "walk_54" in text
    assert "5.4" in text
    assert "athletic_walking.mp4 is not used" in text
    assert "no validated rgb" not in text.lower()


def test_interpretation_uses_computed_cycles_not_hardcoded_expectation():
    rich = demo_gait_interpretation(
        "abnormal", usable_cycles=5, detected_cycles=6, completeness_pct=70.0
    )
    assert rich is not None
    assert "5 usable" in rich.gait_quality
    assert "0 usable cycles" not in rich.teacher_compare
    assert "expected" not in rich.gait_quality.lower()

    thin = demo_gait_interpretation(
        "athletic", usable_cycles=1, detected_cycles=1, completeness_pct=20.0
    )
    assert thin is not None
    assert "Insufficient data" in thin.gait_quality
    assert "faster" not in thin.gait_quality.lower()


def test_legacy_athletic_is_not_treated_as_an_example():
    from stablewalk import config

    assert is_legacy_demo_filename("athletic_walking.mp4")
    legacy = config.DEMO_VIDEOS_DIR / "athletic_walking.mp4"
    if legacy.is_file():
        assert is_demo_video_path(legacy) is None
    old_abnormal = config.DEMO_VIDEOS_DIR / "abnormal_gait.mp4"
    if old_abnormal.is_file():
        assert is_demo_video_path(old_abnormal) is None


def test_resolve_demo_video_path_ignores_athletic_walking(tmp_path, monkeypatch):
    import stablewalk.config as cfg
    from stablewalk.ui.tk.session_cache import resolve_demo_video_path

    (tmp_path / "athletic_walking.mp4").write_bytes(b"legacy-stock")
    (tmp_path / "abnormal_gait.mp4").write_bytes(b"old-abnormal")
    (tmp_path / "normal_gait.mp4").write_bytes(b"old-normal")
    (tmp_path / "abnormal_gait_v2.mp4").write_bytes(b"new-abnormal")
    (tmp_path / "normal_gait_v2.mp4").write_bytes(b"new-normal")
    monkeypatch.setattr(cfg, "DEMO_VIDEOS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "VIDEOS_DIR", tmp_path)
    monkeypatch.setattr(cfg, "LEGACY_VIDEOS_DIR", tmp_path)

    assert Path(resolve_demo_video_path("abnormal")).name == "abnormal_gait_v2.mp4"
    assert Path(resolve_demo_video_path("normal")).name == "normal_gait_v2.mp4"
    assert resolve_demo_video_path("athletic") is None


def test_missing_fast_walking_message_names_v2_and_forbids_legacy():
    text = missing_file_message(example_by_key("athletic"))
    assert "Fast Walking is not available" in text
    assert "fast_walking_v2.mp4" in text
    assert "athletic_walking.mp4" in text
    assert "will not load" in text.lower()
    official = demo_path(example_by_key("athletic"))
    assert official.name == "fast_walking_v2.mp4"
