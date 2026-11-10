"""OpenSim bone-mesh (.vtp) discovery for Gait2392 and other bundled models.

OpenSim prints ``Couldn't find file 'femur_r.vtp'`` when DisplayGeometry is
present and no Geometry search path contains that mesh. IK does not need the
meshes, but the warnings look like a failed run. This module:

* finds Geometry folders (project, next to the .osim, OpenSim install)
* downloads only the meshes a model actually names, from opensim-models
* registers those folders with ``ModelVisualizer.addDirToGeometrySearchPaths``

Missing meshes stay missing if download is unavailable — IK still runs.
"""

from __future__ import annotations

import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from stablewalk import config
from stablewalk.opensim_models import OPENSIM_MODELS_RAW

logger = logging.getLogger(__name__)

GEOMETRY_DIR = config.OPENSIM_MODELS_DIR / "Geometry"
OPENSIM_GEOMETRY_RAW = f"{OPENSIM_MODELS_RAW}/Geometry"

_MESH_FILE_RE = re.compile(r"<mesh_file>\s*([^<\s]+)\s*</mesh_file>", re.IGNORECASE)

# Meshes named by Gait2392_Pipeline/subject01_simbody.osim (and siblings).
GAIT2392_MESH_FILES: tuple[str, ...] = (
    "treadmill.vtp",
    "sacrum.vtp",
    "pelvis.vtp",
    "l_pelvis.vtp",
    "femur_r.vtp",
    "tibia_r.vtp",
    "fibula.vtp",
    "talus.vtp",
    "foot.vtp",
    "bofoot.vtp",
    "femur_l.vtp",
    "tibia_l.vtp",
    "l_fibula.vtp",
    "l_talus.vtp",
    "l_foot.vtp",
    "l_bofoot.vtp",
    "hat_spine.vtp",
    "hat_jaw.vtp",
    "hat_skull.vtp",
    "hat_ribs.vtp",
)

_REGISTERED_DIRS: set[str] = set()


def mesh_files_from_osim(model_path: Path | str | None) -> tuple[str, ...]:
    """Return ``<mesh_file>`` names from an .osim, or the Gait2392 default list."""
    if model_path is None:
        return GAIT2392_MESH_FILES
    path = Path(model_path)
    if not path.is_file():
        return GAIT2392_MESH_FILES
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return GAIT2392_MESH_FILES
    names = [name.strip() for name in _MESH_FILE_RE.findall(text) if name.strip()]
    unique: list[str] = []
    seen: set[str] = set()
    for name in names:
        key = Path(name).name
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return tuple(unique) if unique else GAIT2392_MESH_FILES


def discover_geometry_dirs(
    *,
    model_path: Path | str | None = None,
    extra: Path | str | None = None,
) -> list[Path]:
    """Existing folders that may contain OpenSim .vtp meshes."""
    candidates: list[Path] = []
    if extra is not None:
        candidates.append(Path(extra))
    candidates.append(GEOMETRY_DIR)
    if model_path is not None:
        model = Path(model_path)
        candidates.append(model.parent / "Geometry")
        candidates.append(model.parent)
    env_home = os.environ.get("OPENSIM_HOME", "").strip()
    if env_home:
        candidates.append(Path(env_home) / "Geometry")
    prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if prefix:
        root = Path(prefix)
        candidates.extend(
            (
                root / "Geometry",
                root / "Library" / "Geometry",
                root / "share" / "doc" / "OpenSim" / "Geometry",
            )
        )
    for version in ("4.5", "4.4", "4.3", "4.2"):
        candidates.append(Path(rf"C:\OpenSim {version}\Geometry"))

    found: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen or not resolved.is_dir():
            continue
        seen.add(resolved)
        found.append(resolved)
    return found


def geometry_file_present(name: str, search_dirs: list[Path] | None = None) -> Path | None:
    """Return the first existing mesh path, or None."""
    filename = Path(name).name
    for folder in search_dirs or discover_geometry_dirs():
        candidate = folder / filename
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def missing_geometry_files(
    mesh_names: tuple[str, ...] | list[str],
    search_dirs: list[Path] | None = None,
) -> list[str]:
    dirs = search_dirs if search_dirs is not None else discover_geometry_dirs()
    return [name for name in mesh_names if geometry_file_present(name, dirs) is None]


def download_geometry_file(name: str, dest_dir: Path | None = None) -> Path | None:
    """Download one official mesh into ``models/opensim/Geometry``. None on failure."""
    filename = Path(name).name
    if filename != name or ".." in filename or "/" in filename or "\\" in filename:
        logger.warning("Refusing to download geometry with unsafe name: %s", name)
        return None
    target_dir = dest_dir or GEOMETRY_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / filename
    if dest.is_file() and dest.stat().st_size > 0:
        return dest
    url = f"{OPENSIM_GEOMETRY_RAW}/{filename}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "StableWalk/1.0 (OpenSim geometry downloader)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = response.read()
    except (OSError, urllib.error.URLError) as exc:
        logger.warning("OpenSim geometry download failed for %s: %s", filename, exc)
        return None
    if not data:
        logger.warning("OpenSim geometry download empty: %s", filename)
        return None
    dest.write_bytes(data)
    logger.info("OpenSim geometry saved: %s (%d bytes)", dest, len(data))
    return dest


def ensure_model_geometry(
    model_path: Path | str | None = None,
    *,
    download: bool = True,
) -> list[Path]:
    """Download missing meshes named by the model, then return search folders."""
    meshes = mesh_files_from_osim(model_path)
    dirs = discover_geometry_dirs(model_path=model_path)
    missing = missing_geometry_files(meshes, dirs)
    if missing and download:
        logger.info(
            "OpenSim geometry: downloading %d missing mesh(es) into %s",
            len(missing),
            GEOMETRY_DIR,
        )
        for name in missing:
            download_geometry_file(name, GEOMETRY_DIR)
        dirs = discover_geometry_dirs(model_path=model_path)
        still = missing_geometry_files(meshes, dirs)
        if still:
            logger.warning(
                "OpenSim geometry still missing after download: %s",
                ", ".join(still),
            )
        else:
            logger.info("OpenSim geometry: all %d model mesh(es) available", len(meshes))
    elif missing:
        logger.warning(
            "OpenSim geometry missing (download disabled): %s",
            ", ".join(missing),
        )
    return dirs


def register_geometry_search_paths(
    osim_module: object | None,
    search_dirs: list[Path] | None = None,
) -> list[Path]:
    """Tell the OpenSim SDK where to find .vtp files. Safe if the API is absent."""
    dirs = search_dirs if search_dirs is not None else discover_geometry_dirs()
    if osim_module is None or not dirs:
        return dirs
    visualizer = getattr(osim_module, "ModelVisualizer", None)
    adder = getattr(visualizer, "addDirToGeometrySearchPaths", None) if visualizer else None
    if adder is None:
        logger.debug("OpenSim ModelVisualizer.addDirToGeometrySearchPaths is unavailable")
        return dirs
    for folder in dirs:
        key = str(folder)
        if key in _REGISTERED_DIRS:
            continue
        try:
            adder(key)
            _REGISTERED_DIRS.add(key)
            logger.info("OpenSim geometry search path: %s", folder)
        except Exception as exc:
            logger.warning("Could not register OpenSim geometry path %s: %s", folder, exc)
    return dirs


def prepare_opensim_geometry(
    osim_module: object | None,
    model_path: Path | str | None = None,
    *,
    download: bool = True,
) -> list[Path]:
    """Ensure meshes for ``model_path`` exist and are on the SDK search path."""
    dirs = ensure_model_geometry(model_path, download=download)
    return register_geometry_search_paths(osim_module, dirs)


__all__ = [
    "GAIT2392_MESH_FILES",
    "GEOMETRY_DIR",
    "OPENSIM_GEOMETRY_RAW",
    "discover_geometry_dirs",
    "download_geometry_file",
    "ensure_model_geometry",
    "geometry_file_present",
    "mesh_files_from_osim",
    "missing_geometry_files",
    "prepare_opensim_geometry",
    "register_geometry_search_paths",
]
