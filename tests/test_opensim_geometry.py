"""OpenSim .vtp search path and model mesh inventory — no SDK required."""

from __future__ import annotations

from pathlib import Path

from stablewalk.opensim_geometry import (
    GAIT2392_MESH_FILES,
    discover_geometry_dirs,
    download_geometry_file,
    geometry_file_present,
    mesh_files_from_osim,
    missing_geometry_files,
    register_geometry_search_paths,
)


def test_gait2392_mesh_list_includes_logged_missing_files() -> None:
    for name in (
        "treadmill.vtp",
        "femur_r.vtp",
        "tibia_r.vtp",
        "hat_ribs.vtp",
        "l_pelvis.vtp",
    ):
        assert name in GAIT2392_MESH_FILES


def test_mesh_files_from_osim_reads_mesh_file_tags(tmp_path: Path) -> None:
    osim = tmp_path / "tiny.osim"
    osim.write_text(
        """
        <OpenSimDocument>
          <mesh_file>femur_r.vtp</mesh_file>
          <mesh_file> Geometry/tibia_r.vtp </mesh_file>
          <mesh_file>femur_r.vtp</mesh_file>
        </OpenSimDocument>
        """,
        encoding="utf-8",
    )
    assert mesh_files_from_osim(osim) == ("femur_r.vtp", "tibia_r.vtp")


def test_missing_geometry_reports_absent_files(tmp_path: Path) -> None:
    present = tmp_path / "Geometry"
    present.mkdir()
    (present / "femur_r.vtp").write_bytes(b"mesh")
    dirs = [present]
    assert geometry_file_present("femur_r.vtp", dirs) == present / "femur_r.vtp"
    assert missing_geometry_files(("femur_r.vtp", "missing.vtp"), dirs) == ["missing.vtp"]


def test_download_rejects_unsafe_names(tmp_path: Path) -> None:
    assert download_geometry_file("../secret.vtp", tmp_path) is None
    assert download_geometry_file("dir/femur_r.vtp", tmp_path) is None
    assert list(tmp_path.iterdir()) == []


def test_register_geometry_search_paths_is_safe_without_sdk(tmp_path: Path) -> None:
    folder = tmp_path / "Geometry"
    folder.mkdir()
    registered = register_geometry_search_paths(None, [folder])
    assert registered == [folder]


def test_discover_geometry_dirs_includes_existing_folder(tmp_path: Path, monkeypatch) -> None:
    geo = tmp_path / "Geometry"
    geo.mkdir()
    monkeypatch.setenv("OPENSIM_HOME", str(tmp_path))
    found = discover_geometry_dirs()
    assert geo.resolve() in found
