"""Local research catalog: metadata, no auto-download, clinical-label policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from stablewalk.research.categories import (
    display_clinical_diagnosis,
    dementia_label_allowed,
    normalize_category,
)
from stablewalk.research.dataset import ClipMetadata, import_local_video, load_catalog, save_catalog
from stablewalk.research.probe import is_remote_url


def test_category_aliases() -> None:
    assert normalize_category("Normal") == "normal_gait"
    assert normalize_category("hemiplegic") == "hemiplegic_gait"
    assert normalize_category("dementia") == "dementia_research"
    assert normalize_category("???") == "unknown_or_unverified"


def test_clinical_diagnosis_requires_verified_source() -> None:
    assert (
        display_clinical_diagnosis(
            diagnosis="dementia",
            verified=False,
            source="hospital notes",
        )
        is None
    )
    assert (
        display_clinical_diagnosis(
            diagnosis="dementia",
            verified=True,
            source="",
        )
        is None
    )
    shown = display_clinical_diagnosis(
        diagnosis="Alzheimer disease",
        verified=True,
        source="DSM-5 clinician report (study XYZ)",
    )
    assert shown == "Alzheimer disease"
    assert dementia_label_allowed(
        diagnosis="dementia",
        verified=True,
        source="DSM-5 clinician report (study XYZ)",
    )
    assert not dementia_label_allowed(
        diagnosis="dementia",
        verified=False,
        source="internet forum",
    )


def test_dementia_category_alone_is_not_a_diagnosis() -> None:
    clip = ClipMetadata(
        video_id="x",
        gait_category="dementia_research",
        clinical_diagnosis=None,
        clinical_verified=False,
        clinical_source="",
    )
    assert clip.clinical_display is None
    assert "Dementia-related gait" in clip.category_label


def test_remote_url_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stablewalk.research.dataset.config.RESEARCH_DATASET_DIR", tmp_path
    )
    monkeypatch.setattr(
        "stablewalk.research.dataset.config.RESEARCH_VIDEOS_DIR", tmp_path / "videos"
    )
    monkeypatch.setattr(
        "stablewalk.research.dataset.config.RESEARCH_CATALOG_PATH", tmp_path / "catalog.json"
    )
    with pytest.raises(ValueError, match="download"):
        import_local_video("https://example.com/clinical.mp4")
    assert is_remote_url("https://example.com/x.mp4")
    assert not is_remote_url(str(tmp_path / "local.mp4"))


def test_import_copies_local_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "stablewalk.research.dataset.config.RESEARCH_DATASET_DIR", tmp_path
    )
    monkeypatch.setattr(
        "stablewalk.research.dataset.config.RESEARCH_VIDEOS_DIR", tmp_path / "videos"
    )
    monkeypatch.setattr(
        "stablewalk.research.dataset.config.RESEARCH_CATALOG_PATH", tmp_path / "catalog.json"
    )
    src = tmp_path / "walk.mp4"
    src.write_bytes(b"not-a-real-mp4-but-local")
    clip = import_local_video(
        src,
        gait_category="limping_gait",
        authenticity="experimentally_induced",
        license="CC-BY-4.0",
        original_url="https://example.org/page-not-downloaded",
    )
    assert clip.gait_category == "limping_gait"
    assert clip.authenticity == "experimentally_induced"
    assert clip.resolved_path is not None
    assert clip.resolved_path.is_file()
    assert clip.resolved_path != src
    assert clip.original_url.startswith("https://")
    cat = load_catalog()
    assert cat.get(clip.video_id) is not None
    assert cat.auto_download is False
