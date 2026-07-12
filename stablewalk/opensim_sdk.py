"""
Real OpenSim SDK integration layer for StableWalk.
==================================================

Pipeline roles (keep separate):
-------------------------------
* **MediaPipe** (upstream, ``stablewalk/pose/``) extracts body landmarks from the
  walking video. That is the primary motion source for the dashboard.
* **OpenSim** (this module) represents the same motion *biomechanically* when the
  real OpenSim Python SDK and a musculoskeletal ``.osim`` model are available.

This module is the boundary to the **actual OpenSim Python SDK**. Unlike
``stablewalk.opensim_integration`` (which only *writes OpenSim-compatible files*
and never needs OpenSim), this module performs **real OpenSim operations** when
the SDK is installed:

* loading and validating a musculoskeletal ``.osim`` model (``opensim.Model``),
* preparing Inverse Kinematics (IK) inputs from an exported ``.trc`` marker file,
* running OpenSim's ``InverseKinematicsTool`` to produce a real ``.mot``.

Real OpenSim requires:
----------------------
1. The OpenSim Python SDK installed in the same Python environment.
2. A valid musculoskeletal ``.osim`` model whose MarkerSet matches exported TRC
   marker names (e.g. ``L_KNEE``, ``R_HIP``).

Design rules (honest, non-faking):
----------------------------------
* The SDK is imported through a **safe optional import**. If it is missing, the
  module still imports and every function returns a clear status instead of
  crashing.
* **No OpenSim execution is faked.** If the SDK is not installed, or a model /
  marker file / setup is missing, the functions say exactly what is missing and
  perform no pretend computation.
* The MediaPipe pipeline, the dashboard, and the ``.trc``/``.mot``/JSON exports
  are unaffected and keep working with or without OpenSim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Module handle populated by :func:`check_opensim_available` (fresh probe).
osim = None  # type: ignore
OPENSIM_AVAILABLE = False


def check_opensim_available() -> tuple[bool, Any]:
    """
    Fresh probe for the OpenSim Python SDK.

    Returns ``(True, opensim_module)`` on success or ``(False, error_message)``.
    Never raises. Updates module-level :data:`OPENSIM_AVAILABLE` and :data:`osim`.
    """
    global osim, OPENSIM_AVAILABLE
    try:
        import opensim as _osim  # type: ignore

        osim = _osim
        OPENSIM_AVAILABLE = True
        return True, _osim
    except ImportError as exc:
        osim = None  # type: ignore
        OPENSIM_AVAILABLE = False
        return False, str(exc)
    except Exception as exc:  # DLL / partial install on some platforms
        osim = None  # type: ignore
        OPENSIM_AVAILABLE = False
        return False, f"{type(exc).__name__}: {exc}"


# Initial probe at import time (refreshed again on startup / GUI refresh).
check_opensim_available()

# Exact messages shown in the GUI when prerequisites are missing.
SDK_NOT_INSTALLED_MESSAGE = (
    "OpenSim SDK not installed — compatible export only"
)
SELECT_MODEL_MESSAGE = (
    "OpenSim SDK is installed, but no .osim model is selected. "
    "Please select an OpenSim musculoskeletal model (.osim) to run inverse kinematics."
)
NO_MODEL_LOADED_MESSAGE = (
    "OpenSim SDK installed, but no .osim model loaded yet. "
    "IK cannot run until a model is loaded."
)
IK_SETUP_NOT_CONFIGURED = (
    "OpenSim model loaded and TRC markers exported. "
    "IK setup file is not configured yet."
)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class SdkStatus:
    """Whether the real OpenSim SDK is importable."""

    available: bool
    version: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "version": self.version, "message": self.message}


@dataclass
class ModelInfo:
    """Result of loading / validating an ``.osim`` model."""

    path: str
    valid: bool
    name: str | None = None
    num_bodies: int | None = None
    num_markers: int | None = None
    num_coordinates: int | None = None
    marker_names: list[str] = field(default_factory=list)
    coordinate_names: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "valid": self.valid,
            "name": self.name,
            "num_bodies": self.num_bodies,
            "num_markers": self.num_markers,
            "num_coordinates": self.num_coordinates,
            "marker_names": self.marker_names,
            "coordinate_names": self.coordinate_names,
            "message": self.message,
        }


@dataclass
class IkInputs:
    """Everything required before OpenSim Inverse Kinematics can run."""

    marker_file: str | None
    model_path: str | None
    setup_path: str | None
    time_range: tuple[float, float] | None
    ready: bool
    missing: list[str] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker_file": self.marker_file,
            "model_path": self.model_path,
            "setup_path": self.setup_path,
            "time_range": list(self.time_range) if self.time_range else None,
            "ready": self.ready,
            "missing": self.missing,
            "message": self.message,
        }


@dataclass
class IkReadiness:
    """Whether real OpenSim IK can run (SDK + model + TRC + marker name overlap)."""

    can_run: bool
    message: str
    matched_markers: list[str] = field(default_factory=list)
    missing_in_model: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "can_run": self.can_run,
            "message": self.message,
            "matched_markers": self.matched_markers,
            "missing_in_model": self.missing_in_model,
        }


@dataclass
class IkResult:
    """Outcome of attempting to run Inverse Kinematics."""

    ran: bool
    output_motion_path: str | None
    message: str
    time_range: tuple[float, float] | None = None
    setup_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "output_motion_path": self.output_motion_path,
            "message": self.message,
            "time_range": list(self.time_range) if self.time_range else None,
            "setup_path": self.setup_path,
        }


# ---------------------------------------------------------------------------
# 1. SDK availability
# ---------------------------------------------------------------------------
def check_opensim_sdk(*, refresh: bool = True) -> SdkStatus:
    """
    Report whether the real OpenSim Python SDK is importable.

    Never raises (even on a broken/partial install).

    Args:
        refresh: When ``True`` (default), re-probe ``import opensim`` instead of
            relying on a stale value from an earlier failed import.
    """
    if refresh:
        check_opensim_available()
    if not OPENSIM_AVAILABLE:
        return SdkStatus(available=False, version=None, message=SDK_NOT_INSTALLED_MESSAGE)
    version: str | None
    try:  # pragma: no cover - requires the SDK
        version = str(osim.GetVersion())  # type: ignore[attr-defined]
    except Exception:
        version = "unknown"
    return SdkStatus(
        available=True,
        version=version,
        message=f"OpenSim SDK available (version {version})",
    )


def log_opensim_startup_status(
    log: logging.Logger | None = None,
    *,
    model_path: str | Path | None = None,
) -> dict[str, Any]:
    """
    Log a clear OpenSim status block at application startup.

    Returns a small dict suitable for tests / OPENSIM_STATUS.md updates.
    """
    log = log or logger
    available, result = check_opensim_available()
    status = check_opensim_sdk(refresh=False)

    info: dict[str, Any] = {
        "detected": available,
        "import_success": available,
        "module_path": None,
        "version": status.version,
        "model_loaded": False,
        "model_name": None,
    }

    log.info("OpenSim SDK detected: %s", available)
    if available:
        osim_mod = result
        mod_path = getattr(osim_mod, "__file__", None)
        info["module_path"] = mod_path
        log.info("OpenSim import: success")
        if mod_path:
            log.info("OpenSim module path: %s", mod_path)
        if status.version:
            log.info("OpenSim version: %s", status.version)
        log.info("OpenSim SDK: Installed")
        log.info("Mode: Real OpenSim SDK enabled")
    else:
        log.info("OpenSim import: failure — %s", result)
        log.info("OpenSim SDK: Not installed")
        log.info("Mode: Compatible export only")

    resolved_model = Path(model_path) if model_path else None
    if resolved_model and resolved_model.is_file():
        from stablewalk.opensim_models import _is_autoload_blocked, find_usable_opensim_model

        if _is_autoload_blocked(resolved_model):
            resolved_model = find_usable_opensim_model()
        if resolved_model:
            model_info = validate_opensim_model(resolved_model)
            if model_info.valid and (model_info.num_markers or 0) > 0:
                info["model_loaded"] = True
                info["model_name"] = model_info.name
                log.info("OpenSim reference model available: %s", model_info.name)
            elif model_info.valid:
                log.info(
                    "OpenSim model skipped (no markers): %s",
                    resolved_model.name,
                )
    if available:
        from stablewalk.opensim_marker_mapping import demo_ik_setup_available

        if demo_ik_setup_available():
            log.info(
                "OpenSim Demo IK ready: %s",
                "models/opensim/Gait2392_Pipeline/subject01_Setup_IK.xml",
            )
        log.info("StableWalk IK: Not ready (marker mapping required)")

    return info


def log_post_analysis_opensim_status(
    export_paths: dict[str, Path] | None,
    log: logging.Logger | None = None,
    *,
    model_path: str | Path | None = None,
    ik_ran: bool = False,
    ik_output: str | Path | None = None,
) -> None:
    """Log honest post-analysis OpenSim export / IK status."""
    log = log or logger
    status = check_opensim_sdk(refresh=True)

    if export_paths:
        log.info("OpenSim-compatible files prepared (MediaPipe source, not OpenSim IK):")
        for kind, path in export_paths.items():
            log.info("  OpenSim %s -> %s", kind, path)
    else:
        log.info("OpenSim export: skipped (no pose data or export failed)")

    if status.available:
        from stablewalk.opensim_marker_mapping import demo_ik_setup_available

        if demo_ik_setup_available():
            log.info(
                "OpenSim Demo IK ready: %s (use --run-opensim-demo-ik or GUI button)",
                "models/opensim/Gait2392_Pipeline/subject01_Setup_IK.xml",
            )
        log.info("StableWalk IK: not executed (marker mapping required)")
    else:
        log.info("OpenSim IK: not executed (SDK not installed)")


# ---------------------------------------------------------------------------
# 2 & 3. Load / validate a musculoskeletal model
# ---------------------------------------------------------------------------
def load_opensim_model(model_path: str | Path) -> ModelInfo:
    """
    Load an OpenSim ``.osim`` model (real ``opensim.Model``) and summarize it.

    Returns a :class:`ModelInfo`. If the SDK is unavailable or the file does not
    load, ``valid`` is ``False`` and ``message`` explains why (no crash).
    """
    model_path = Path(model_path)
    if not OPENSIM_AVAILABLE:
        return ModelInfo(
            path=str(model_path), valid=False, message=SDK_NOT_INSTALLED_MESSAGE
        )
    if not model_path.is_file():
        return ModelInfo(
            path=str(model_path), valid=False,
            message=f"Model file not found: {model_path}",
        )

    try:  # pragma: no cover - requires the SDK
        logger.info("Loading OpenSim model...")
        logger.info("  Path: %s", model_path)
        model = osim.Model(str(model_path.resolve()))  # type: ignore[union-attr]
        model.initSystem()  # validates the model can be realized
        model_name = model.getName()
        logger.info("Model loaded successfully: %s", model_name)
        logger.info(
            "  Markers: %d | Coordinates: %d | Bodies: %d",
            model.getMarkerSet().getSize(),
            model.getCoordinateSet().getSize(),
            model.getBodySet().getSize(),
        )

        marker_set = model.getMarkerSet()
        marker_names = [marker_set.get(i).getName() for i in range(marker_set.getSize())]
        coord_set = model.getCoordinateSet()
        coord_names = [coord_set.get(i).getName() for i in range(coord_set.getSize())]

        return ModelInfo(
            path=str(model_path),
            valid=True,
            name=model.getName(),
            num_bodies=model.getBodySet().getSize(),
            num_markers=marker_set.getSize(),
            num_coordinates=coord_set.getSize(),
            marker_names=marker_names,
            coordinate_names=coord_names,
            message="Model loaded and initialized successfully.",
        )
    except Exception as exc:  # pragma: no cover - requires the SDK
        return ModelInfo(
            path=str(model_path), valid=False,
            message=f"Failed to load/initialize model: {exc}",
        )


def validate_opensim_model(model_path: str | Path) -> ModelInfo:
    """
    Validate that a model loads and contains usable markers/coordinates.

    Thin wrapper over :func:`load_opensim_model` that also flags a model with no
    markers (which cannot be used for marker-based IK).
    """
    info = load_opensim_model(model_path)
    if info.valid and (info.num_markers or 0) == 0:
        info.valid = False
        info.message = (
            "Model loaded but contains no markers. Marker-based inverse kinematics "
            "requires a MarkerSet whose marker names match the exported .trc markers."
        )
    return info


# ---------------------------------------------------------------------------
# 4. Prepare Inverse Kinematics inputs
# ---------------------------------------------------------------------------
def _read_trc_time_range(trc_path: Path) -> tuple[float, float] | None:
    """
    Read first/last time stamps from a ``.trc`` file (pure Python, no SDK).

    Works on the files written by ``stablewalk.opensim_integration.export_trc_file``.
    """
    try:
        lines = trc_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    times: list[float] = []
    for line in lines:
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        # Data rows start with an integer frame number then a float time.
        if not parts[0].strip().isdigit():
            continue
        try:
            times.append(float(parts[1]))
        except ValueError:
            continue
    if not times:
        return None
    return (min(times), max(times))


def _read_trc_marker_names(trc_path: Path) -> list[str]:
    """
    Parse marker names from a StableWalk / OpenSim ``.trc`` header (line 4).

    Format written by ``export_trc_file``: ``Frame#``, ``Time``, then each marker
    name followed by two blank columns before the next marker name.
    """
    try:
        lines = trc_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    if len(lines) < 4:
        return []
    parts = lines[3].split("\t")
    names: list[str] = []
    i = 2  # skip Frame# and Time
    while i < len(parts):
        token = parts[i].strip()
        if token:
            names.append(token)
        i += 3
    return names


def _format_marker_mismatch_message(
    trc_markers: list[str],
    model_markers: list[str],
    matched: list[str],
    missing: list[str],
    *,
    min_required: int = 3,
) -> str:
    """Human-readable marker compatibility report for IK failures."""
    lines = [
        (
            f"Marker mismatch: {len(matched)} of {len(trc_markers)} TRC markers "
            f"match the model (need at least {min_required})."
        ),
        "",
        "TRC markers missing from model MarkerSet:",
        ", ".join(missing) if missing else "(none parsed from TRC)",
        "",
        "Matched markers:",
        ", ".join(matched) if matched else "(none)",
    ]
    if model_markers:
        sample = model_markers[:24]
        lines.extend(["", "Model MarkerSet (sample):", ", ".join(sample)])
        if len(model_markers) > len(sample):
            lines.append(f"... and {len(model_markers) - len(sample)} more")
    lines.extend(
        [
            "",
            "Tip: choose a model whose MarkerSet includes exported names "
            "such as L_KNEE, R_HIP, L_HEEL, R_ANKLE.",
        ]
    )
    return "\n".join(lines)


def check_ik_readiness(
    model_path: str | Path | None,
    trc_path: str | Path | None,
    *,
    min_matched_markers: int = 3,
) -> IkReadiness:
    """
    Check whether real OpenSim IK can run without faking results.

    Requires: SDK installed, valid model with markers, exported TRC, and at
    least ``min_matched_markers`` marker names present in both the TRC and the
    model MarkerSet.
    """
    if not OPENSIM_AVAILABLE:
        return IkReadiness(can_run=False, message=SDK_NOT_INSTALLED_MESSAGE)

    resolved_model = Path(model_path) if model_path else _default_model()
    resolved_trc = Path(trc_path) if trc_path else None

    if resolved_model is None or not resolved_model.is_file():
        return IkReadiness(can_run=False, message=SELECT_MODEL_MESSAGE)
    if resolved_trc is None or not resolved_trc.is_file():
        return IkReadiness(
            can_run=False,
            message="Export a TRC marker file for the current session first.",
        )

    model_info = validate_opensim_model(resolved_model)
    if not model_info.valid:
        return IkReadiness(can_run=False, message=model_info.message)

    trc_markers = _read_trc_marker_names(resolved_trc)
    if not trc_markers:
        return IkReadiness(
            can_run=False,
            message="TRC file has no readable marker names.",
        )

    model_marker_set = set(model_info.marker_names)
    matched = [m for m in trc_markers if m in model_marker_set]
    missing = [m for m in trc_markers if m not in model_marker_set]

    if len(matched) < min_matched_markers:
        return IkReadiness(
            can_run=False,
            message=_format_marker_mismatch_message(
                trc_markers,
                model_info.marker_names,
                matched,
                missing,
                min_required=min_matched_markers,
            ),
            matched_markers=matched,
            missing_in_model=missing,
        )

    return IkReadiness(
        can_run=True,
        message=f"IK ready ({len(matched)} shared markers).",
        matched_markers=matched,
        missing_in_model=missing,
    )


def prepare_inverse_kinematics_inputs(
    marker_file: str | Path,
    model_path: str | Path | None = None,
    *,
    setup_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    write_setup: bool = True,
) -> IkInputs:
    """
    Gather and validate everything OpenSim IK needs, before running it.

    Steps:
      1. Confirm the ``.trc`` marker file exists and read its time range.
      2. Resolve the model (explicit path, else a bundled default).
      3. Validate the model loads (if the SDK is available).
      4. Optionally write a real IK **setup** ``.xml`` via the SDK
         (``InverseKinematicsTool.printToXML``), so IK is fully reproducible.

    Returns an :class:`IkInputs` describing readiness and anything missing.
    No IK is executed here.
    """
    marker_file = Path(marker_file)
    missing: list[str] = []

    if not marker_file.is_file():
        missing.append(f"marker file (.trc): {marker_file}")
    time_range = _read_trc_time_range(marker_file) if marker_file.is_file() else None

    # Resolve model
    resolved_model: Path | None = Path(model_path) if model_path else _default_model()
    if resolved_model is None:
        missing.append("OpenSim model (.osim) — none provided and no default found")
    elif not resolved_model.is_file():
        missing.append(f"OpenSim model (.osim): {resolved_model}")

    # Marker-name overlap between exported TRC and model MarkerSet.
    ik_ready = check_ik_readiness(resolved_model, marker_file if marker_file.is_file() else None)
    if marker_file.is_file() and resolved_model and resolved_model.is_file() and not ik_ready.can_run:
        missing.append(ik_ready.message)

    # If SDK present and we have a model, validate it loads.
    model_msg = ""
    if OPENSIM_AVAILABLE and resolved_model and resolved_model.is_file():
        info = validate_opensim_model(resolved_model)
        if not info.valid:
            missing.append(f"valid model — {info.message}")
        model_msg = info.message

    # Write IK setup XML only when markers are compatible.
    written_setup: Path | None = Path(setup_path) if setup_path else None
    output_mot = marker_file.parent / f"{marker_file.stem}_ik.mot"
    if (
        write_setup
        and OPENSIM_AVAILABLE
        and ik_ready.can_run
        and resolved_model
        and resolved_model.is_file()
        and marker_file.is_file()
        and not [m for m in missing if "valid model" in m]
    ):
        out_dir = Path(output_dir) if output_dir else marker_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        written_setup = Path(setup_path) if setup_path else out_dir / f"{marker_file.stem}_ik_setup.xml"
        try:  # pragma: no cover - requires the SDK
            write_ik_setup_xml(
                marker_file,
                resolved_model,
                written_setup,
                output_mot,
                time_range,
            )
        except Exception as exc:  # pragma: no cover - requires the SDK
            missing.append(f"IK setup generation failed: {exc}")
            written_setup = None

    ready = not missing
    if not OPENSIM_AVAILABLE:
        message = SDK_NOT_INSTALLED_MESSAGE
    elif resolved_model is None:
        message = SELECT_MODEL_MESSAGE
    elif not ik_ready.can_run and marker_file.is_file() and resolved_model and resolved_model.is_file():
        message = ik_ready.message
    elif ready:
        message = "Inverse kinematics inputs are ready."
    else:
        message = "Missing inputs: " + "; ".join(missing)
    if model_msg and ready:
        message += f" ({model_msg})"

    return IkInputs(
        marker_file=str(marker_file) if marker_file.is_file() else None,
        model_path=str(resolved_model) if resolved_model else None,
        setup_path=str(written_setup) if written_setup else None,
        time_range=time_range,
        ready=ready,
        missing=missing,
        message=message,
    )


# ---------------------------------------------------------------------------
# 5. IK setup XML + run Inverse Kinematics
# ---------------------------------------------------------------------------
def write_ik_setup_xml(
    marker_file: str | Path,
    model_path: str | Path,
    setup_path: str | Path,
    output_mot: str | Path,
    time_range: tuple[float, float] | None = None,
    *,
    results_dir: str | Path | None = None,
    trc_marker_names: list[str] | None = None,
    synthetic_markers: set[str] | None = None,
    derived_markers: set[str] | None = None,
    temporal_markers: set[str] | None = None,
) -> Path:
    """
    Write an OpenSim Inverse Kinematics setup ``.xml`` with **absolute paths**.

    OpenSim resolves model/TRC/MOT paths from the setup file; relative paths often fail.
    """
    if not OPENSIM_AVAILABLE:
        raise RuntimeError(SDK_NOT_INSTALLED_MESSAGE)

    marker_file = Path(marker_file).resolve()
    model_path = Path(model_path).resolve()
    setup_path = Path(setup_path).resolve()
    output_mot = Path(output_mot).resolve()
    setup_path.parent.mkdir(parents=True, exist_ok=True)

    tr = time_range or _read_trc_time_range(marker_file)
    logger.info("Preparing IK setup XML...")
    ik_tool = osim.InverseKinematicsTool()  # type: ignore[union-attr]
    ik_tool.setName(marker_file.stem)
    ik_tool.set_model_file(str(model_path))
    ik_tool.setMarkerDataFileName(str(marker_file))
    ik_tool.setOutputMotionFileName(str(output_mot))
    if results_dir is not None:
        results_path = Path(results_dir).resolve()
        results_path.mkdir(parents=True, exist_ok=True)
        if hasattr(ik_tool, "setResultsDir"):
            ik_tool.setResultsDir(str(results_path))  # type: ignore[attr-defined]
    if tr is not None:
        ik_tool.setStartTime(tr[0])
        ik_tool.setEndTime(tr[1])

    marker_names = trc_marker_names or _read_trc_marker_names(marker_file)
    derived = derived_markers or synthetic_markers or set()
    temporal = temporal_markers or set()
    if marker_names:
        from stablewalk.opensim_marker_mapping import get_marker_weight

        task_set = osim.IKTaskSet()  # type: ignore[union-attr]
        for name in marker_names:
            task = osim.IKMarkerTask()  # type: ignore[union-attr]
            task.setName(name)
            task.setApply(True)
            task.setWeight(
                get_marker_weight(
                    name,
                    is_derived_anatomical=name in derived,
                    is_temporal_estimated=name in temporal,
                    is_synthetic=name in derived,
                )
            )
            task_set.adoptAndAppend(task)
        ik_tool.set_IKTaskSet(task_set)
        logger.info(
            "  IK marker tasks: %d (derived: %d, temporal: %d)",
            len(marker_names),
            len(derived & set(marker_names)),
            len(temporal & set(marker_names)),
        )

    ik_tool.printToXML(str(setup_path))
    logger.info("  IK setup saved: %s", setup_path)
    return setup_path


def run_inverse_kinematics_if_available(
    marker_file: str | Path,
    model_path: str | Path | None = None,
    *,
    output_mot: str | Path | None = None,
    setup_path: str | Path | None = None,
    time_range: tuple[float, float] | None = None,
    trc_marker_names: list[str] | None = None,
    synthetic_markers: set[str] | None = None,
    derived_markers: set[str] | None = None,
    temporal_markers: set[str] | None = None,
) -> IkResult:
    """
    Run OpenSim Inverse Kinematics **only if** the SDK, model, and marker file
    are all present. Otherwise return a clear, honest status (no fake results).

    Creates ``<session>_ik_setup.xml`` then runs
    ``InverseKinematicsTool(setup_xml).run()``.

    On success, writes a real ``<session>_ik.mot`` and verifies it exists on disk.
    """
    from stablewalk.opensim_marker_mapping import (
        STABLEWALK_IK_MAPPING_IMPLEMENTED,
        STABLEWALK_IK_NOT_READY_MSG,
    )

    if not STABLEWALK_IK_MAPPING_IMPLEMENTED:
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=STABLEWALK_IK_NOT_READY_MSG,
        )

    check_opensim_available()
    if not OPENSIM_AVAILABLE:
        return IkResult(ran=False, output_motion_path=None, message=SDK_NOT_INSTALLED_MESSAGE)

    marker_file = Path(marker_file)
    if not marker_file.is_file():
        return IkResult(
            ran=False, output_motion_path=None,
            message=f"Marker file (.trc) not found: {marker_file}",
        )
    logger.info("TRC file validated: %s", marker_file.resolve())

    resolved_model: Path | None = Path(model_path) if model_path else None
    if resolved_model is None or not resolved_model.is_file():
        return IkResult(ran=False, output_motion_path=None, message=SELECT_MODEL_MESSAGE)

    readiness = check_ik_readiness(resolved_model, marker_file)
    if not readiness.can_run:
        logger.error("OpenSim IK blocked — marker/model mismatch:\n%s", readiness.message)
        return IkResult(ran=False, output_motion_path=None, message=readiness.message)

    model_info = validate_opensim_model(resolved_model)
    if not model_info.valid:
        return IkResult(
            ran=False, output_motion_path=None,
            message=f"Model not usable for IK: {model_info.message}",
        )

    tr = time_range or _read_trc_time_range(marker_file)
    if output_mot is None:
        output_mot = marker_file.with_name(f"{marker_file.stem}_ik.mot")
    output_mot = Path(output_mot)
    output_mot.parent.mkdir(parents=True, exist_ok=True)

    if setup_path is None:
        setup_path = marker_file.with_name(f"{marker_file.stem}_ik_setup.xml")
    setup_path = Path(setup_path)

    try:  # pragma: no cover - requires the SDK
        marker_names = trc_marker_names or _read_trc_marker_names(marker_file)
        write_ik_setup_xml(
            marker_file,
            resolved_model,
            setup_path,
            output_mot,
            tr,
            results_dir=marker_file.parent,
            trc_marker_names=marker_names,
            synthetic_markers=synthetic_markers,
            derived_markers=derived_markers,
            temporal_markers=temporal_markers,
        )
        logger.info("Running OpenSim IK...")
        logger.info("  Setup XML: %s", setup_path.resolve())
        ik_tool = osim.InverseKinematicsTool(str(setup_path.resolve()))  # type: ignore[union-attr]
        ran_ok = bool(ik_tool.run())
        if not output_mot.is_file() or "_ik" not in output_mot.name.lower():
            detail = "OpenSim run() returned False" if not ran_ok else "output file missing"
            return IkResult(
                ran=False,
                output_motion_path=None,
                setup_path=str(setup_path),
                message=f"OpenSim IK did not create output file: {output_mot} ({detail})",
                time_range=tr,
            )
    except Exception as exc:  # pragma: no cover - requires the SDK
        logger.error("OpenSim IK failed: %s", exc)
        return IkResult(
            ran=False,
            output_motion_path=None,
            setup_path=str(setup_path) if setup_path.is_file() else None,
            message=f"OpenSim IK failed: {exc}",
            time_range=tr,
        )

    logger.info("IK completed successfully")
    logger.info("IK output saved: %s", output_mot.resolve())
    return IkResult(
        ran=True,
        output_motion_path=str(output_mot.resolve()),
        setup_path=str(setup_path.resolve()),
        message=f"IK completed successfully. Output: {output_mot.name}",
        time_range=tr,
    )


def run_stablewalk_ik_experimental(
    stablewalk_trc: str | Path,
    *,
    model_path: str | Path | None = None,
    run_name: str | None = None,
) -> IkResult:
    """
    Run OpenSim IK on a **mapped** StableWalk TRC (experimental / partial markers).

    Accepts the raw ``<run_name>.trc`` or ``<run_name>_mapped_for_opensim.trc``.
    Writes ``stablewalk_setup_ik.xml`` and ``<run_name>_ik.mot`` under the session
    output folder. Never validates raw MediaPipe TRC names against the OpenSim model.

    Success requires a real ``<run_name>_ik.mot`` on disk.
    """
    from stablewalk.opensim_marker_mapping import (
        DEMO_IK_MODEL,
        MAPPED_TRC_SUFFIX,
        MIN_EXPERIMENTAL_IK_MARKERS,
        STABLEWALK_IK_EXPERIMENTAL_WARNING,
        STABLEWALK_IK_SETUP_FILENAME,
        compare_stablewalk_trc_to_opensim,
        demo_ik_setup_available,
        log_ik_validation_summary,
        mapped_trc_path_for,
        run_name_from_trc_path,
    )

    def _ik_log(fmt: str, *args: object) -> None:
        text = fmt % args if args else fmt
        logger.info("%s", text)
        print(text, flush=True)

    check_opensim_available()
    if not OPENSIM_AVAILABLE:
        return IkResult(ran=False, output_motion_path=None, message=SDK_NOT_INSTALLED_MESSAGE)

    trc_input = Path(stablewalk_trc)
    if not trc_input.is_file():
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=f"StableWalk TRC not found: {trc_input}",
        )

    name = run_name or run_name_from_trc_path(trc_input)
    if trc_input.stem.endswith(MAPPED_TRC_SUFFIX):
        mapped_trc = trc_input
        source_trc = trc_input.with_name(f"{name}.trc")
    else:
        source_trc = trc_input
        mapped_trc = mapped_trc_path_for(source_trc)

    resolved_model = Path(model_path) if model_path else DEMO_IK_MODEL
    if not resolved_model.is_file():
        if not demo_ik_setup_available() and model_path is None:
            return IkResult(
                ran=False,
                output_motion_path=None,
                message=(
                    "OpenSim Gait2392 model not found. Expected "
                    f"{DEMO_IK_MODEL} or pass --opensim-model PATH."
                ),
            )
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=f"OpenSim model not found: {resolved_model}",
        )

    out_dir = source_trc.parent
    output_mot = out_dir / f"{name}_ik.mot"
    setup_path = out_dir / STABLEWALK_IK_SETUP_FILENAME

    if not source_trc.is_file():
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=(
                f"StableWalk export TRC not found: {source_trc}\n"
                "Please click Export OpenSim Files first."
            ),
        )

    if not mapped_trc.is_file():
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=(
                f"Mapped TRC not found: {mapped_trc}\n"
                "Please click Export OpenSim Files first."
            ),
        )

    try:
        comparison = compare_stablewalk_trc_to_opensim(source_trc)
        log_ik_validation_summary(comparison)
    except (OSError, ValueError) as exc:
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=f"Failed to validate mapped TRC: {exc}",
        )

    if not comparison.mapped_trc_path or not Path(comparison.mapped_trc_path).is_file():
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=comparison.message or "Mapped TRC could not be created.",
        )

    if comparison.mapping_status not in ("improved", "partial", "experimental"):
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=(
                f"Marker mapping not ready for experimental IK "
                f"(status: {comparison.mapping_status}).\n{comparison.message}"
            ),
        )

    if not comparison.ik_experimental_ready:
        return IkResult(
            ran=False,
            output_motion_path=None,
            message=(
                f"Not enough mapped markers for experimental IK "
                f"({len(comparison.mapped_matching_markers)} matched, "
                f"need {MIN_EXPERIMENTAL_IK_MARKERS}).\n{comparison.message}"
            ),
        )

    _ik_log("Run StableWalk IK Experimental")
    _ik_log("Current run/session: %s", name)
    _ik_log("Using mapped TRC: %s", mapped_trc.resolve())
    _ik_log("Using OpenSim model: %s", resolved_model.resolve())
    _ik_log("IK setup XML: %s", setup_path.resolve())
    _ik_log("IK output path: %s", output_mot.resolve())
    _ik_log("Marker mapping status: %s", comparison.mapping_status)
    _ik_log("Coverage: %s%%", comparison.coverage_percent)
    _ik_log("Warning: this is experimental and not clinical-grade")
    _ik_log("Running OpenSim InverseKinematicsTool...")

    result = run_inverse_kinematics_if_available(
        mapped_trc,
        resolved_model,
        output_mot=output_mot,
        setup_path=setup_path,
        trc_marker_names=comparison.mapped_opensim_markers,
        synthetic_markers=set(comparison.synthetic_markers),
        derived_markers=set(comparison.derived_anatomical_markers),
        temporal_markers=set(comparison.temporal_estimated_markers),
    )

    if result.ran and result.output_motion_path and Path(result.output_motion_path).is_file():
        _ik_log("StableWalk IK completed successfully")
        _ik_log("  IK output: %s", Path(result.output_motion_path).resolve())
        result.message = (
            f"StableWalk IK completed successfully. Output: {Path(result.output_motion_path).name}. "
            f"{STABLEWALK_IK_EXPERIMENTAL_WARNING}"
        )
    else:
        _ik_log("StableWalk IK failed")
        _ik_log("  Error: %s", result.message)
        logger.error("StableWalk IK failed: %s", result.message)
    return result

@dataclass
class DemoIkResult:
    """Outcome of running the bundled Gait2392 demo IK setup XML."""

    ran: bool
    setup_path: str | None
    output_motion_path: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ran": self.ran,
            "setup_path": self.setup_path,
            "output_motion_path": self.output_motion_path,
            "message": self.message,
        }


def run_opensim_demo_ik(
    setup_xml: str | Path | None = None,
) -> DemoIkResult:
    """
    Run the official OpenSim Gait2392 demo IK setup XML.

    Changes working directory to ``Gait2392_Pipeline/`` before calling
    ``InverseKinematicsTool(setup_xml.name).run()`` because the setup file
    uses relative paths.

    Success is reported only when a real ``*ik*.mot`` file exists on disk.
    """
    import glob
    import os

    from stablewalk import config
    from stablewalk.opensim_marker_mapping import demo_ik_setup_available

    check_opensim_available()
    if not OPENSIM_AVAILABLE:
        return DemoIkResult(
            ran=False,
            setup_path=None,
            output_motion_path=None,
            message=SDK_NOT_INSTALLED_MESSAGE,
        )

    if setup_xml:
        setup_path = Path(setup_xml).resolve()
    else:
        setup_path = (config.OPENSIM_MODELS_DIR / "Gait2392_Pipeline" / "subject01_Setup_IK.xml").resolve()
    pipeline_dir = setup_path.parent.resolve()

    if not setup_path.is_file():
        return DemoIkResult(
            ran=False,
            setup_path=str(setup_path),
            output_motion_path=None,
            message=f"Demo IK setup XML not found: {setup_path}",
        )

    if not demo_ik_setup_available():
        return DemoIkResult(
            ran=False,
            setup_path=str(setup_path),
            output_motion_path=None,
            message=(
                "OpenSim demo bundle incomplete. Expected subject01_Setup_IK.xml, "
                "subject01_walk1.trc, and subject01_simbody.osim in Gait2392_Pipeline/"
            ),
        )

    logger.info("Running OpenSim Demo IK...")
    logger.info("Demo IK working directory: %s", pipeline_dir)
    logger.info("Demo IK setup XML: %s", setup_path.name)

    old_cwd = os.getcwd()
    output_path: str | None = None
    error_msg: str | None = None
    try:
        os.chdir(pipeline_dir)
        before = set(glob.glob("*ik*.mot"))
        ik_tool = osim.InverseKinematicsTool(setup_path.name)  # type: ignore[union-attr]
        ik_tool.run()
        after = set(glob.glob("*ik*.mot"))
        created = sorted(after - before)
        if created:
            output_path = str((pipeline_dir / created[0]).resolve())
        else:
            existing = sorted(glob.glob("*ik*.mot"))
            if existing:
                output_path = str((pipeline_dir / existing[0]).resolve())
            else:
                error_msg = "IK ran but no *ik*.mot output file was found."
    except Exception as exc:  # pragma: no cover - requires SDK
        error_msg = str(exc)
        logger.error("OpenSim Demo IK failed: %s", exc)
    finally:
        os.chdir(old_cwd)

    if (
        output_path
        and Path(output_path).is_file()
        and "_ik" in Path(output_path).name.lower()
    ):
        logger.info("OpenSim Demo IK completed successfully")
        logger.info("Demo IK output file: %s", output_path)
        return DemoIkResult(
            ran=True,
            setup_path=str(setup_path),
            output_motion_path=output_path,
            message=f"OpenSim Demo IK completed successfully. Output: {Path(output_path).name}",
        )

    message = error_msg or "IK ran but no *ik*.mot output file was found."
    return DemoIkResult(
        ran=False,
        setup_path=str(setup_path),
        output_motion_path=None,
        message=message,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _default_model() -> Path | None:
    """Usable ``.osim`` with markers (never gait2354 template). Returns None if none."""
    try:
        from stablewalk.opensim_models import find_usable_opensim_model

        return find_usable_opensim_model(require_markers=True, min_markers=1)
    except Exception:
        return None


def opensim_environment_summary() -> dict[str, Any]:
    """One-call summary for the GUI/CLI: SDK status + default model presence."""
    status = check_opensim_sdk()
    default_model = _default_model()
    summary: dict[str, Any] = {
        "sdk": status.to_dict(),
        "default_model": str(default_model) if default_model else None,
    }
    if status.available and default_model:
        summary["model"] = validate_opensim_model(default_model).to_dict()
    return summary


def update_opensim_status_md(
    *,
    demo_ik_result: DemoIkResult | None = None,
    status_path: str | Path | None = None,
) -> Path:
    """
    Write ``OPENSIM_STATUS.md`` with honest runtime verification results.

    Called after ``--run-opensim-demo-ik`` and optionally from the GUI.
    """
    from stablewalk import config
    from stablewalk.opensim_marker_mapping import (
        DEMO_IK_SETUP_XML,
        MEDIAPIPE_LIMITATION_EXPLANATION,
        compare_stablewalk_trc_to_opensim,
        demo_ik_setup_available,
        mapping_status_label,
        reliability_label,
        stablewalk_ik_status_label,
    )

    md_path = Path(status_path) if status_path else config.PROJECT_ROOT / "OPENSIM_STATUS.md"
    sdk = check_opensim_sdk(refresh=True)

    stablewalk_trc = config.OPENSIM_DIR / "walk_stream" / "walk_stream.trc"
    trc_exported = stablewalk_trc.is_file()
    mapping = compare_stablewalk_trc_to_opensim(stablewalk_trc if trc_exported else None)

    if demo_ik_result and demo_ik_result.ran and demo_ik_result.output_motion_path:
        demo_status = "completed"
        demo_output = demo_ik_result.output_motion_path
    elif demo_ik_result:
        demo_status = "failed"
        demo_output = demo_ik_result.message
    else:
        expected_mot = DEMO_IK_SETUP_XML.parent / "subject01_walk1_ik.mot"
        if expected_mot.is_file():
            demo_status = "completed (output on disk)"
            demo_output = str(expected_mot.resolve())
        elif demo_ik_setup_available():
            demo_status = "ready (not run yet)"
            demo_output = str(expected_mot)
        else:
            demo_status = "unavailable (pipeline files missing)"
            demo_output = "(none)"

    mapping_label = mapping_status_label(mapping)
    sw_ik_mot = stablewalk_trc.parent / "walk_stream_ik.mot" if trc_exported else None
    sw_ik_ran = sw_ik_mot.is_file() if sw_ik_mot else False
    if sw_ik_ran:
        sw_ik_status = "completed (experimental — not fully validated)"
        sw_ik_output = str(sw_ik_mot.resolve())
    elif mapping.ik_experimental_ready:
        sw_ik_status = "experimental / not fully validated"
        sw_ik_output = "(not run yet)"
    else:
        sw_ik_status = "not ready"
        sw_ik_output = "(none)"

    lines = [
        "# OpenSim Status — Demo IK + StableWalk IK",
        "",
        f"**Last updated:** {__import__('datetime').date.today().isoformat()} (auto-generated)",
        "",
        "---",
        "",
        "## Runtime verification",
        "",
        "| Check | Status |",
        "| --- | --- |",
        f"| OpenSim SDK detected | {'yes' if sdk.available else 'no'} |",
        f"| OpenSim Demo IK | {demo_status} |",
        f"| Demo IK output file | `{demo_output}` |",
        f"| StableWalk TRC export | {'completed' if trc_exported else 'missing'} |",
        f"| Mapped TRC | `{mapping.mapped_trc_path or '(not created)'}` |",
        f"| Marker mapping | {mapping_label.lower()} |",
        f"| Mapped markers matching OpenSim | {len(mapping.mapped_matching_markers)} / {len(mapping.opensim_reference_markers)} ({mapping.coverage_percent}%) |",
        f"| Direct mapped markers | {len(mapping.direct_mapped_markers)} |",
        f"| Synthetic markers | {len(mapping.synthetic_markers)} |",
        f"| IK readiness tier | {mapping.ik_readiness_tier} |",
        f"| Reliability | {mapping.reliability} (not clinical-grade) |",
        f"| StableWalk IK | {sw_ik_status} |",
        f"| StableWalk IK output file | `{sw_ik_output}` |",
        "",
        "StableWalk IK note:",
        "",
        f"> {MEDIAPIPE_LIMITATION_EXPLANATION}",
        "",
        "---",
        "",
        "## Two separate OpenSim workflows",
        "",
        "### Part A — OpenSim Demo IK (proves SDK integration)",
        "",
        "Uses the **official OpenSim Gait2392 sample files** (not MediaPipe):",
        "",
        "| File | Path |",
        "| --- | --- |",
        "| IK setup XML | `models/opensim/Gait2392_Pipeline/subject01_Setup_IK.xml` |",
        "| Demo TRC | `models/opensim/Gait2392_Pipeline/subject01_walk1.trc` |",
        "| Static TRC | `models/opensim/Gait2392_Pipeline/subject01_static.trc` |",
        "| Model | `models/opensim/Gait2392_Pipeline/subject01_simbody.osim` |",
        "| IK output | `models/opensim/Gait2392_Pipeline/subject01_walk1_ik.mot` |",
        "",
        "**GUI:** Click **Run OpenSim Demo IK**",
        "",
        "**CLI:** `python main.py --run-opensim-demo-ik`",
        "",
        "**Code:** `run_opensim_demo_ik()` → `InverseKinematicsTool(setup_xml).run()`",
        "",
        "Success is reported **only** when a real `*ik*.mot` file exists on disk.",
        "",
        "### Part B — StableWalk IK (MediaPipe → OpenSim)",
        "",
        "Uses **StableWalk-exported** files from video analysis:",
        "",
        "| File | Path |",
        "| --- | --- |",
        "| StableWalk TRC | `data/output/opensim/walk_stream/walk_stream.trc` |",
        "| Mapped TRC | `data/output/opensim/walk_stream/walk_stream_mapped_for_opensim.trc` |",
        "| StableWalk MOT | `data/output/opensim/walk_stream/walk_stream.mot` (MediaPipe angles) |",
        "| JSON | `data/output/opensim/walk_stream/walk_stream_opensim.json` |",
        "| Marker mapping | `models/opensim/marker_mapping.json` |",
        "| StableWalk IK output | `data/output/opensim/walk_stream/walk_stream_ik.mot` (only if IK runs) |",
        "",
        "**GUI:** **Run StableWalk IK Experimental** (mapped + synthetic markers — experimental, not clinical-grade)",
        "",
        "StableWalk exports 17 MediaPipe landmarks (`L_KNEE`, `R_HIP`, …). The mapped TRC renames "
        "direct matches (`R.ASIS`, `R.Heel`, …) and adds **synthetic** thigh, shank, sacral, "
        "midfoot, and sternum markers interpolated from hips/knees/ankles/feet. This improves "
        "coverage versus the previous 9-marker mapping, but MediaPipe is still not equivalent "
        "to full optical motion capture.",
        "",
        "---",
        "",
        "## Quick verification",
        "",
        "```bash",
        "# Run demo IK immediately (requires OpenSim SDK in conda env)",
        "python main.py --run-opensim-demo-ik",
        "",
        "# Or from Python",
        "python -c \"from stablewalk.opensim_sdk import run_opensim_demo_ik; r=run_opensim_demo_ik(); print(r)\"",
        "```",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


__all__ = [
    "OPENSIM_AVAILABLE",
    "SDK_NOT_INSTALLED_MESSAGE",
    "SELECT_MODEL_MESSAGE",
    "NO_MODEL_LOADED_MESSAGE",
    "IK_SETUP_NOT_CONFIGURED",
    "SdkStatus",
    "ModelInfo",
    "IkInputs",
    "IkReadiness",
    "IkResult",
    "check_opensim_available",
    "check_opensim_sdk",
    "log_opensim_startup_status",
    "log_post_analysis_opensim_status",
    "load_opensim_model",
    "validate_opensim_model",
    "check_ik_readiness",
    "prepare_inverse_kinematics_inputs",
    "write_ik_setup_xml",
    "run_inverse_kinematics_if_available",
    "run_stablewalk_ik_experimental",
    "DemoIkResult",
    "run_opensim_demo_ik",
    "opensim_environment_summary",
    "update_opensim_status_md",
]
