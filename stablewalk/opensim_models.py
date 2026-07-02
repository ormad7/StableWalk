"""
OpenSim musculoskeletal model discovery for StableWalk.

Searches ``models/opensim/`` for ``.osim`` files (including subfolders such as
``Gait2392_Simbody/`` and ``Gait2354_Simbody/`` from the official
`opensim-org/opensim-models` repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

from stablewalk import config

logger = logging.getLogger(__name__)

# Official repo: https://github.com/opensim-org/opensim-models
OPENSIM_MODELS_REPO = "https://github.com/opensim-org/opensim-models"
OPENSIM_MODELS_RAW = (
    "https://raw.githubusercontent.com/opensim-org/opensim-models/master"
)

# Preferred filenames for *manual* selection only (never auto-load gait2354).
PREFERRED_MODEL_FILENAMES: tuple[str, ...] = (
    "subject01_simbody.osim",
    "subject01_simbody_adjusted.osim",
    "subject01_adjusted.osim",
    "subject01.osim",
    "gait2392_millard2012muscle.osim",
    "gait2392_thelen2003muscle.osim",
)

# Never auto-select these (unscaled templates with no MarkerSet).
AUTOLOAD_BLOCKLIST: tuple[str, ...] = (
    "gait2354_simbody.osim",
    "gait2392_simbody.osim",
)

# Official demo pipeline model (has markers; used for reference only, not auto-loaded).
GAIT2392_PIPELINE_MODEL = (
    config.OPENSIM_MODELS_DIR / "Gait2392_Pipeline" / "subject01_simbody.osim"
)

# Bundles available from opensim-org/opensim-models (Models/ subfolder).
OFFICIAL_MODEL_BUNDLES: dict[str, dict[str, object]] = {
    "gait2392": {
        "folder": "Gait2392_Simbody",
        "files": [
            "gait2392_millard2012muscle.osim",
            "gait2392_thelen2003muscle.osim",
            "subject01.osim",
            "subject01_adjusted.osim",
            "subject01_simbody_adjusted.osim",
        ],
        "note": (
            "The official repo does not include gait2392_simbody.osim; "
            "gait2392_millard2012muscle.osim is the unscaled Simbody model."
        ),
    },
    "gait2354": {
        "folder": "Gait2354_Simbody",
        "files": [
            "gait2354_simbody.osim",
            "subject01_simbody.osim",
        ],
        "note": "Includes gait2354_simbody.osim (unscaled Simbody model).",
    },
}


def discover_opensim_models(base_dir: Path | None = None) -> list[Path]:
    """
    Recursively find all ``.osim`` files under ``models/opensim/``.

    Returns paths sorted by relative depth then name (shallow preferred).
    """
    root = base_dir or config.OPENSIM_MODELS_DIR
    if not root.is_dir():
        return []
    models = [p.resolve() for p in root.rglob("*.osim") if p.is_file()]
    models.sort(key=lambda p: (len(p.relative_to(root).parts), str(p).lower()))
    return models


def _probe_model_marker_count(path: Path) -> int | None:
    """Return marker count for an ``.osim`` without noisy logging (discovery only)."""
    try:
        from stablewalk.opensim_sdk import OPENSIM_AVAILABLE, check_opensim_available

        check_opensim_available()
        if not OPENSIM_AVAILABLE:
            return None
        import opensim as osim  # type: ignore[import-untyped]

        model = osim.Model(str(path.resolve()))
        model.initSystem()
        return int(model.getMarkerSet().getSize())
    except Exception:
        return None


def _is_autoload_blocked(path: Path) -> bool:
    """True for unscaled template models that must not be auto-loaded."""
    name = path.name.lower()
    if name in {b.lower() for b in AUTOLOAD_BLOCKLIST}:
        return True
    if "gait2354_simbody" in name:
        return True
    return False


def find_usable_opensim_model(
    base_dir: Path | None = None,
    *,
    require_markers: bool = True,
    min_markers: int = 1,
) -> Path | None:
    """
    Return the first ``.osim`` suitable for marker-based IK, or ``None``.

    Never returns blocklisted models (e.g. ``gait2354_simbody.osim`` with 0 markers).
    Does **not** auto-load — callers use this only when explicitly resolving a model.
    """
    root = base_dir or config.OPENSIM_MODELS_DIR

    # Prefer the verified Gait2392 demo pipeline model when present.
    candidates: list[Path] = []
    if GAIT2392_PIPELINE_MODEL.is_file():
        candidates.append(GAIT2392_PIPELINE_MODEL.resolve())

    by_name = {p.name.lower(): p.resolve() for p in discover_opensim_models(base_dir)}
    for preferred in PREFERRED_MODEL_FILENAMES:
        hit = by_name.get(preferred.lower())
        if hit is not None and hit not in candidates:
            candidates.append(hit)

    for path in discover_opensim_models(base_dir):
        if path.resolve() not in candidates:
            candidates.append(path.resolve())

    for path in candidates:
        if _is_autoload_blocked(path):
            continue
        if not require_markers:
            return path
        marker_count = _probe_model_marker_count(path)
        if marker_count is not None and marker_count >= min_markers:
            return path
    return None


def find_preferred_opensim_model(base_dir: Path | None = None) -> Path | None:
    """
    Return a usable ``.osim`` with markers, excluding blocklisted templates.

    Returns ``None`` if no suitable model exists (no silent fallback to gait2354).
    """
    return find_usable_opensim_model(base_dir, require_markers=True, min_markers=1)


def list_opensim_model_choices(
    base_dir: Path | None = None,
    *,
    include_blocked: bool = False,
) -> list[tuple[str, Path]]:
    """
    Human-readable labels and paths for GUI / CLI selection.

    Label format: ``Gait2354_Simbody/gait2354_simbody.osim``
    """
    root = base_dir or config.OPENSIM_MODELS_DIR
    choices: list[tuple[str, Path]] = []
    for path in discover_opensim_models(base_dir):
        if not include_blocked and _is_autoload_blocked(path):
            continue
        try:
            label = str(path.relative_to(root))
        except ValueError:
            label = path.name
        choices.append((label.replace("\\", "/"), path))
    return choices


def model_display_name(path: Path, base_dir: Path | None = None) -> str:
    """Short display name for a discovered model."""
    root = base_dir or config.OPENSIM_MODELS_DIR
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return path.name


def ensure_models_dir() -> Path:
    """Create ``models/opensim`` if missing."""
    config.OPENSIM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return config.OPENSIM_MODELS_DIR


__all__ = [
    "OPENSIM_MODELS_REPO",
    "OPENSIM_MODELS_RAW",
    "PREFERRED_MODEL_FILENAMES",
    "OFFICIAL_MODEL_BUNDLES",
    "AUTOLOAD_BLOCKLIST",
    "GAIT2392_PIPELINE_MODEL",
    "discover_opensim_models",
    "find_usable_opensim_model",
    "find_preferred_opensim_model",
    "list_opensim_model_choices",
    "model_display_name",
    "ensure_models_dir",
]
