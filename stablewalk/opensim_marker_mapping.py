"""
OpenSim marker mapping and StableWalk IK readiness validation.

Maps MediaPipe-exported TRC marker names (``L_KNEE``, ``R_HIP``, …) to OpenSim
Gait2392 marker names (``L.Shank.Upper``, ``R.ASIS``, …), reconstructs missing
markers using subject-scaled anatomical segment frames, applies temporal
filtering, and compares against the OpenSim demo model/setup markers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from stablewalk import config
from stablewalk.biomechanics.marker_reconstruction import (
    DEFAULT_RECONSTRUCTION_CONFIG,
    GAIT2392_MARKER_NAMES,
    CATALOG_BY_NAME,
    IkReadinessLevel,
    MarkerReconstructionConfig,
    MarkerReconstructionResult,
    format_mapping_catalog_table,
    mapping_catalog_csv,
    reconstruct_markers_from_trc_frames,
    write_mapped_trc_from_reconstruction,
)
from stablewalk.opensim_integration import MEDIAPIPE_TO_OPENSIM_MARKERS
from stablewalk.opensim_sdk import _read_trc_marker_names

logger = logging.getLogger(__name__)

MARKER_MAPPING_JSON = config.OPENSIM_MODELS_DIR / "marker_mapping.json"
MARKER_MAPPING_REPORT = config.OPENSIM_MODELS_DIR / "marker_mapping_report.txt"
MAPPED_TRC_SUFFIX = "_mapped_for_opensim"

STABLEWALK_IK_MAPPING_IMPLEMENTED = True

GAIT2392_PIPELINE_DIR = config.OPENSIM_MODELS_DIR / "Gait2392_Pipeline"
DEMO_IK_SETUP_XML = GAIT2392_PIPELINE_DIR / "subject01_Setup_IK.xml"
DEMO_IK_DEMO_TRC = GAIT2392_PIPELINE_DIR / "subject01_walk1.trc"
DEMO_IK_MODEL = GAIT2392_PIPELINE_DIR / "subject01_simbody.osim"
DEMO_IK_OUTPUT_MOT = GAIT2392_PIPELINE_DIR / "subject01_walk1_ik.mot"

STABLEWALK_IK_NOT_READY_MSG = (
    "StableWalk IK is not ready yet — export a TRC and ensure marker mapping is configured."
)

STABLEWALK_IK_EXPERIMENTAL_WARNING = (
    "Experimental only: MediaPipe-based markers are approximated and are not equivalent "
    "to full optical motion capture. Biomechanical reliability remains limited."
)

STABLEWALK_IK_SETUP_FILENAME = "stablewalk_setup_ik.xml"

MEDIAPIPE_LIMITATION_EXPLANATION = (
    "MediaPipe pose landmarks are approximated for OpenSim IK. Reliable landmarks "
    "are mapped directly; additional Gait2392 markers are reconstructed using "
    "subject-scaled anatomical segment frames (pelvis, thigh, shank, foot, torso) "
    "with temporal filtering and confidence scoring. This is not clinical-grade mocap."
)

# Direct StableWalk → OpenSim mappings (names must exist on Gait2392).
# Note: Gait2392 has no R.Knee.Lat / R.Ankle.Lat — knee/ankle feed synthetic shank/thigh.
DEFAULT_STABLEWALK_TO_OPENSIM: dict[str, str] = {
    "R_SHOULDER": "R.Acromium",
    "L_SHOULDER": "L.Acromium",
    "HEAD": "Top.Head",
    "R_HIP": "R.ASIS",
    "L_HIP": "L.ASIS",
    "R_HEEL": "R.Heel",
    "L_HEEL": "L.Heel",
    "R_TOE": "R.Toe.Tip",
    "L_TOE": "L.Toe.Tip",
    "R_ANKLE": "R.Shank.Front",
    "L_ANKLE": "L.Shank.Front",
}

PROPOSED_MEDIAPIPE_TO_GAIT2392: dict[str, str] = {
    **DEFAULT_STABLEWALK_TO_OPENSIM,
    "L_KNEE": "(derived anatomical → L.Shank.Upper / L.Thigh.*)",
    "R_KNEE": "(derived anatomical → R.Shank.Upper / R.Thigh.*)",
    "L_ELBOW": "(optional — not on Gait2392 lower-body IK set)",
    "R_ELBOW": "(optional — not on Gait2392 lower-body IK set)",
    "L_WRIST": "(optional — not on Gait2392 lower-body IK set)",
    "R_WRIST": "(optional — not on Gait2392 lower-body IK set)",
}

# Full Gait2392 IK marker order for mapped TRC export.
PREFERRED_MAPPED_MARKER_ORDER: list[str] = list(GAIT2392_MARKER_NAMES)

MIN_EXPERIMENTAL_IK_MARKERS = 5
MIN_REASONABLY_USABLE_MARKERS = 12
MIN_RELIABLE_IK_MARKERS = 20
MIN_DIRECT_MATCHES_FOR_STABLEWALK_IK = MIN_EXPERIMENTAL_IK_MARKERS

# Base IK weights (aligned with subject01_Setup_IK.xml); synthetic markers scaled down.
MARKER_WEIGHT_BASE: dict[str, float] = {
    "R.ASIS": 10.0,
    "L.ASIS": 10.0,
    "V.Sacral": 10.0,
    "R.Heel": 10.0,
    "L.Heel": 10.0,
    "R.Toe.Tip": 10.0,
    "L.Toe.Tip": 10.0,
    "R.Acromium": 0.5,
    "L.Acromium": 0.5,
    "Top.Head": 0.1,
    "Sternum": 1.0,
}
MARKER_WEIGHT_DEFAULT = 1.0
DERIVED_ANATOMICAL_WEIGHT_SCALE = 0.55
TEMPORAL_ESTIMATED_WEIGHT_SCALE = 0.35
# Backward-compatible alias
SYNTHETIC_WEIGHT_SCALE = DERIVED_ANATOMICAL_WEIGHT_SCALE


@dataclass
class SyntheticMarkerSpec:
    """Legacy rule type — retained for backward compatibility only."""

    opensim_name: str
    description: str
    required_landmarks: list[str]
    compute: Callable[[dict[str, tuple[float, float, float]]], tuple[float, float, float] | None]


# Legacy naive synthetic specs (superseded by marker_reconstruction.MARKER_CATALOG).
SYNTHETIC_MARKER_SPECS: list[SyntheticMarkerSpec] = []


@dataclass
class MarkerComparison:
    """Result of comparing StableWalk TRC markers with OpenSim reference markers."""

    stablewalk_trc_markers: list[str] = field(default_factory=list)
    opensim_reference_markers: list[str] = field(default_factory=list)
    opensim_demo_trc_markers: list[str] = field(default_factory=list)
    matching_markers: list[str] = field(default_factory=list)
    missing_in_opensim: list[str] = field(default_factory=list)
    extra_in_opensim: list[str] = field(default_factory=list)
    mapped_opensim_markers: list[str] = field(default_factory=list)
    direct_mapped_markers: list[str] = field(default_factory=list)
    derived_anatomical_markers: list[str] = field(default_factory=list)
    temporal_estimated_markers: list[str] = field(default_factory=list)
    unavailable_markers: list[str] = field(default_factory=list)
    synthetic_markers: list[str] = field(default_factory=list)
    synthetic_marker_details: dict[str, str] = field(default_factory=dict)
    mapped_matching_markers: list[str] = field(default_factory=list)
    unmapped_stablewalk_markers: list[str] = field(default_factory=list)
    missing_for_ik: list[str] = field(default_factory=list)
    mapping_status: str = "missing"  # missing | experimental | partial | improved
    ik_readiness_tier: str = "not ready"
    ik_readiness_level: str = "NOT READY"
    ik_readiness_score: float = 0.0
    reliability: str = "limited"
    coverage_percent: float = 0.0
    raw_coverage_percent: float = 0.0
    high_confidence_coverage_percent: float = 0.0
    low_confidence_markers: list[str] = field(default_factory=list)
    reconstruction: MarkerReconstructionResult | None = None
    stablewalk_ik_ready: bool = False
    ik_experimental_ready: bool = False
    ik_reasonably_usable: bool = False
    ik_reliable_ready: bool = False
    stablewalk_trc_path: str | None = None
    mapped_trc_path: str | None = None
    opensim_reference_source: str | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stablewalk_trc_markers": self.stablewalk_trc_markers,
            "opensim_demo_ik_markers": self.opensim_reference_markers,
            "opensim_demo_trc_markers": self.opensim_demo_trc_markers,
            "matching_markers_direct": self.matching_markers,
            "mapped_opensim_markers": self.mapped_opensim_markers,
            "direct_mapped_markers": self.direct_mapped_markers,
            "derived_anatomical_markers": self.derived_anatomical_markers,
            "temporal_estimated_markers": self.temporal_estimated_markers,
            "unavailable_markers": self.unavailable_markers,
            "synthetic_markers": self.synthetic_markers,
            "synthetic_marker_details": self.synthetic_marker_details,
            "mapped_matching_markers": self.mapped_matching_markers,
            "unmapped_stablewalk_markers": self.unmapped_stablewalk_markers,
            "missing_for_ik": self.missing_for_ik,
            "missing_in_opensim_model": self.missing_in_opensim,
            "extra_in_opensim": self.extra_in_opensim,
            "stablewalk_to_opensim_mapping": load_stablewalk_to_opensim_mapping(),
            "proposed_mapping": PROPOSED_MEDIAPIPE_TO_GAIT2392,
            "mediapipe_export_markers": sorted(set(MEDIAPIPE_TO_OPENSIM_MARKERS.values())),
            "mapping_status": self.mapping_status,
            "ik_readiness_tier": self.ik_readiness_tier,
            "ik_readiness_level": self.ik_readiness_level,
            "ik_readiness_score": self.ik_readiness_score,
            "reliability": self.reliability,
            "coverage_percent": self.coverage_percent,
            "raw_coverage_percent": self.raw_coverage_percent,
            "high_confidence_coverage_percent": self.high_confidence_coverage_percent,
            "low_confidence_markers": self.low_confidence_markers,
            "stablewalk_ik_ready": self.stablewalk_ik_ready,
            "ik_experimental_ready": self.ik_experimental_ready,
            "ik_reasonably_usable": self.ik_reasonably_usable,
            "ik_reliable_ready": self.ik_reliable_ready,
            "min_experimental_markers": MIN_EXPERIMENTAL_IK_MARKERS,
            "min_reasonably_usable_markers": MIN_REASONABLY_USABLE_MARKERS,
            "min_reliable_markers": MIN_RELIABLE_IK_MARKERS,
            "stablewalk_trc_path": self.stablewalk_trc_path,
            "mapped_trc_path": self.mapped_trc_path,
            "opensim_reference_source": self.opensim_reference_source,
            "message": self.message,
        }


def get_marker_weight(
    opensim_name: str,
    *,
    is_synthetic: bool = False,
    is_derived_anatomical: bool = False,
    is_temporal_estimated: bool = False,
) -> float:
    """Return IK task weight for a marker (derived/temporal markers weighted lower)."""
    base = MARKER_WEIGHT_BASE.get(opensim_name, MARKER_WEIGHT_DEFAULT)
    if is_temporal_estimated:
        return max(base * TEMPORAL_ESTIMATED_WEIGHT_SCALE, 0.03)
    if is_synthetic or is_derived_anatomical:
        return max(base * DERIVED_ANATOMICAL_WEIGHT_SCALE, 0.05)
    return base


def ensure_default_marker_mapping_json() -> Path:
    """Write or upgrade ``marker_mapping.json`` with the default StableWalk → OpenSim mapping."""
    MARKER_MAPPING_JSON.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if MARKER_MAPPING_JSON.is_file():
        try:
            data = json.loads(MARKER_MAPPING_JSON.read_text(encoding="utf-8"))
            if isinstance(data.get("stablewalk_to_opensim"), dict):
                existing = {
                    str(k): str(v)
                    for k, v in data["stablewalk_to_opensim"].items()
                    if isinstance(v, str)
                }
            elif isinstance(data, dict):
                existing = {
                    str(k): str(v)
                    for k, v in data.items()
                    if isinstance(k, str) and isinstance(v, str) and ("_" in k or k.isupper())
                }
        except (OSError, json.JSONDecodeError):
            pass

    merged = dict(DEFAULT_STABLEWALK_TO_OPENSIM)
    merged.update(existing)
    if merged != existing or not MARKER_MAPPING_JSON.is_file():
        MARKER_MAPPING_JSON.write_text(
            json.dumps(merged, indent=2) + "\n",
            encoding="utf-8",
        )
    return MARKER_MAPPING_JSON


def load_stablewalk_to_opensim_mapping() -> dict[str, str]:
    """Load StableWalk → OpenSim marker name mapping from ``marker_mapping.json``."""
    ensure_default_marker_mapping_json()
    try:
        data = json.loads(MARKER_MAPPING_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(DEFAULT_STABLEWALK_TO_OPENSIM)

    if isinstance(data.get("stablewalk_to_opensim"), dict):
        loaded = {
            str(k): str(v)
            for k, v in data["stablewalk_to_opensim"].items()
            if isinstance(v, str)
        }
    else:
        loaded = {
            str(k): str(v)
            for k, v in data.items()
            if isinstance(k, str) and isinstance(v, str) and ("_" in k or k.isupper())
        }

    merged = dict(DEFAULT_STABLEWALK_TO_OPENSIM)
    merged.update(loaded)
    return merged


def mapped_trc_path_for(source_trc: Path) -> Path:
    """Path for the mapped TRC alongside the source export."""
    source_trc = Path(source_trc)
    return source_trc.with_name(f"{source_trc.stem}{MAPPED_TRC_SUFFIX}.trc")


def run_name_from_trc_path(trc_path: Path) -> str:
    """Session run name from raw or mapped StableWalk TRC filename."""
    stem = Path(trc_path).stem
    if stem.endswith(MAPPED_TRC_SUFFIX):
        return stem[: -len(MAPPED_TRC_SUFFIX)]
    return stem


def _read_trc_frame_data(
    source_trc: Path,
) -> tuple[list[str], list[tuple[int, float, dict[str, tuple[float, float, float]]]], str, str]:
    """
    Parse TRC into per-frame landmark positions.

    Returns ``(marker_names, frames, data_rate, units)`` where each frame is
    ``(frame_num, time, {landmark: (x,y,z)})``.
    """
    lines = source_trc.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"Invalid TRC file (too few header lines): {source_trc}")

    marker_names = _read_trc_marker_names(source_trc)
    meta_parts = lines[2].split("\t")
    rate = meta_parts[0] if meta_parts else "30.0"
    units = meta_parts[4] if len(meta_parts) > 4 else "mm"

    frames: list[tuple[int, float, dict[str, tuple[float, float, float]]]] = []
    for line in lines[6:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].strip().isdigit():
            continue
        frame_num = int(parts[0])
        time_val = float(parts[1])
        positions: dict[str, tuple[float, float, float]] = {}
        for idx, name in enumerate(marker_names):
            base = 2 + idx * 3
            if base + 2 < len(parts):
                try:
                    positions[name] = (
                        float(parts[base]),
                        float(parts[base + 1]),
                        float(parts[base + 2]),
                    )
                except ValueError:
                    pass
        frames.append((frame_num, time_val, positions))
    return marker_names, frames, rate, units


def _order_mapped_markers(names: set[str]) -> list[str]:
    ordered = [n for n in PREFERRED_MAPPED_MARKER_ORDER if n in names]
    for name in sorted(names):
        if name not in ordered:
            ordered.append(name)
    return ordered


def create_mapped_trc(
    source_trc: Path,
    output_trc: Path,
    mapping: dict[str, str] | None = None,
    *,
    reconstruction_config: MarkerReconstructionConfig | None = None,
) -> tuple[Path, list[str], list[str], dict[str, str], MarkerReconstructionResult]:
    """
    Create a mapped TRC with OpenSim marker names (direct + anatomical reconstruction).

    Returns ``(path, direct_names, derived_names, derived_details, reconstruction)``.
    The third return value is kept as ``derived_names`` for backward compatibility
    (formerly ``synthetic_marker_names``).
    """
    source_trc = Path(source_trc)
    output_trc = Path(output_trc)
    _ = mapping or load_stablewalk_to_opensim_mapping()

    _, frames, rate, _units = _read_trc_frame_data(source_trc)
    if not frames:
        raise ValueError(f"No frame data in TRC: {source_trc}")

    fps = float(rate) if rate else 30.0
    reconstruction = reconstruct_markers_from_trc_frames(
        frames,
        fps=fps,
        config=reconstruction_config or DEFAULT_RECONSTRUCTION_CONFIG,
    )
    write_mapped_trc_from_reconstruction(
        reconstruction,
        output_trc,
        data_rate=str(rate),
    )

    direct_names = [
        n for n in reconstruction.marker_names
        if CATALOG_BY_NAME.get(n)
        and CATALOG_BY_NAME[n].mapping_type == "DIRECT"
        and any(
            reconstruction.frames[i][2][n].position is not None
            for i in range(len(reconstruction.frames))
        )
    ]
    derived_names = [
        n for n in reconstruction.marker_names
        if n not in direct_names
        and any(
            reconstruction.frames[i][2][n].position is not None
            for i in range(len(reconstruction.frames))
        )
    ]
    derived_details = {
        n: next(
            (
                row["Calculation"]
                for row in reconstruction.catalog_rows
                if row.get("OpenSim Marker") == n
            ),
            "Anatomical segment-frame reconstruction",
        )
        for n in derived_names
    }

    logger.info("Wrote mapped TRC for OpenSim IK -> %s", output_trc)
    logger.info(
        "  %d direct + %d derived anatomical marker(s); IK readiness %s (%.1f)",
        len(direct_names),
        len(derived_names),
        reconstruction.ik_readiness.value,
        reconstruction.ik_readiness_score,
    )
    return output_trc, direct_names, derived_names, derived_details, reconstruction


def configure_stablewalk_ik_setup(
    setup_path: Path,
    trc_marker_names: list[str],
    *,
    synthetic_markers: set[str] | None = None,
    derived_markers: set[str] | None = None,
    temporal_markers: set[str] | None = None,
) -> None:
    """
    Post-process IK setup XML: disable missing marker tasks, apply weights.

    OpenSim ``printToXML`` includes all model markers; this enables only markers
    present in the mapped TRC and sets reasonable weights.
    """
    derived = derived_markers or synthetic_markers or set()
    temporal = temporal_markers or set()
    trc_set = set(trc_marker_names)
    text = setup_path.read_text(encoding="utf-8")

    def _patch_task(match: re.Match[str]) -> str:
        block = match.group(0)
        name_match = re.search(r'name="([^"]+)"', block)
        if not name_match:
            return block
        name = name_match.group(1)
        if name not in trc_set:
            return re.sub(r"<apply>true</apply>", "<apply>false</apply>", block)
        weight = get_marker_weight(
            name,
            is_derived_anatomical=name in derived,
            is_temporal_estimated=name in temporal,
            is_synthetic=name in derived,
        )
        return re.sub(
            r"<weight>[^<]*</weight>",
            f"<weight>{weight:g}</weight>",
            block,
        )

    patched = re.sub(
        r"<IKMarkerTask\s+name=\"[^\"]+\">.*?</IKMarkerTask>",
        _patch_task,
        text,
        flags=re.DOTALL,
    )
    setup_path.write_text(patched, encoding="utf-8")


def read_ik_setup_marker_tasks(setup_xml: Path) -> list[str]:
    """Parse ``IKMarkerTask`` names from an OpenSim IK setup XML."""
    try:
        text = setup_xml.read_text(encoding="utf-8")
    except OSError:
        return []
    names = re.findall(r'<IKMarkerTask\s+name="([^"]+)"', text)
    seen: set[str] = set()
    ordered: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered


def read_ik_setup_output_mot(setup_xml: Path) -> Path:
    """Read ``output_motion_file`` from an IK setup XML (relative to setup dir)."""
    try:
        text = setup_xml.read_text(encoding="utf-8")
    except OSError:
        return setup_xml.parent / "subject01_walk1_ik.mot"
    match = re.search(r"<output_motion_file>([^<]+)</output_motion_file>", text)
    name = match.group(1).strip() if match else "subject01_walk1_ik.mot"
    return (setup_xml.parent / name).resolve()


def demo_ik_setup_available() -> bool:
    """True when the official Gait2392 demo IK bundle is present."""
    return (
        DEMO_IK_SETUP_XML.is_file()
        and DEMO_IK_DEMO_TRC.is_file()
        and DEMO_IK_MODEL.is_file()
    )


def _compute_readiness(comparison: MarkerComparison) -> None:
    """Fill readiness tiers, coverage, reliability, and mapping status on *comparison*."""
    ref_count = len(comparison.opensim_reference_markers) or 1
    n_match = len(comparison.mapped_matching_markers)
    comparison.coverage_percent = round(100.0 * n_match / ref_count, 1)

    recon = comparison.reconstruction
    if recon is not None:
        comparison.raw_coverage_percent = recon.raw_coverage_percent
        comparison.high_confidence_coverage_percent = recon.high_confidence_coverage_percent
        comparison.ik_readiness_level = recon.ik_readiness.value
        comparison.ik_readiness_score = recon.ik_readiness_score
        comparison.low_confidence_markers = list(recon.low_confidence_markers)
    else:
        comparison.raw_coverage_percent = comparison.coverage_percent
        comparison.high_confidence_coverage_percent = comparison.coverage_percent * 0.85

    comparison.ik_experimental_ready = n_match >= MIN_EXPERIMENTAL_IK_MARKERS
    comparison.ik_reasonably_usable = n_match >= MIN_REASONABLY_USABLE_MARKERS
    comparison.ik_reliable_ready = n_match >= MIN_RELIABLE_IK_MARKERS
    comparison.stablewalk_ik_ready = comparison.ik_reasonably_usable

    score = comparison.ik_readiness_score
    if score >= DEFAULT_RECONSTRUCTION_CONFIG.ik_readiness_high:
        comparison.ik_readiness_tier = "high quality"
    elif score >= DEFAULT_RECONSTRUCTION_CONFIG.ik_readiness_moderate:
        comparison.ik_readiness_tier = "moderate quality"
    elif comparison.ik_experimental_ready:
        comparison.ik_readiness_tier = "experimental"
    else:
        comparison.ik_readiness_tier = "not ready"

    derived_ratio = (
        len(comparison.derived_anatomical_markers)
        / max(len(comparison.mapped_opensim_markers), 1)
    )
    if comparison.high_confidence_coverage_percent >= 55 and derived_ratio <= 0.65:
        comparison.reliability = "high"
    elif comparison.raw_coverage_percent >= 40:
        comparison.reliability = "moderate"
    else:
        comparison.reliability = "limited"

    if not comparison.stablewalk_trc_markers:
        comparison.mapping_status = "missing"
    elif comparison.raw_coverage_percent >= 90 and score >= 60:
        comparison.mapping_status = "improved"
    elif comparison.ik_experimental_ready:
        comparison.mapping_status = (
            "partial" if comparison.ik_reasonably_usable else "experimental"
        )
    else:
        comparison.mapping_status = "experimental"

    if comparison.mapping_status == "improved":
        comparison.message = (
            f"Anatomical marker reconstruction: {n_match}/{ref_count} OpenSim markers "
            f"({comparison.raw_coverage_percent}% raw, "
            f"{comparison.high_confidence_coverage_percent}% high-confidence). "
            f"IK readiness: {comparison.ik_readiness_level}. "
            f"{MEDIAPIPE_LIMITATION_EXPLANATION}"
        )
    elif comparison.ik_experimental_ready:
        comparison.message = (
            f"Partial marker mapping: {n_match}/{ref_count} OpenSim markers "
            f"(raw {comparison.raw_coverage_percent}%, "
            f"high-confidence {comparison.high_confidence_coverage_percent}%). "
            f"IK readiness: {comparison.ik_readiness_level}. "
            f"{MEDIAPIPE_LIMITATION_EXPLANATION}"
        )
    elif comparison.stablewalk_trc_markers:
        comparison.message = (
            f"Only {n_match} mapped markers match OpenSim "
            f"(need {MIN_EXPERIMENTAL_IK_MARKERS} for experimental IK)."
        )
    else:
        comparison.message = "StableWalk TRC not exported yet."


def compare_stablewalk_trc_to_opensim(
    stablewalk_trc: Path | None,
    *,
    reference_setup: Path | None = None,
    reference_trc: Path | None = None,
    min_matches: int = MIN_DIRECT_MATCHES_FOR_STABLEWALK_IK,
) -> MarkerComparison:
    """
    Map StableWalk TRC markers, write mapped TRC, compare with OpenSim demo markers.

    Updates ``marker_mapping_report.txt`` on disk.
    """
    setup = reference_setup or (DEMO_IK_SETUP_XML if DEMO_IK_SETUP_XML.is_file() else None)
    demo_trc = reference_trc or (DEMO_IK_DEMO_TRC if DEMO_IK_DEMO_TRC.is_file() else None)
    mapping = load_stablewalk_to_opensim_mapping()

    result = MarkerComparison()

    if stablewalk_trc and stablewalk_trc.is_file():
        result.stablewalk_trc_markers = _read_trc_marker_names(stablewalk_trc)
        result.stablewalk_trc_path = str(stablewalk_trc.resolve())
    else:
        result.mapping_status = "missing"
        result.message = "StableWalk TRC not exported yet."
        _write_marker_mapping_report(result)
        return result

    if setup and setup.is_file():
        result.opensim_reference_markers = read_ik_setup_marker_tasks(setup)
        result.opensim_reference_source = str(setup.resolve())
    elif demo_trc and demo_trc.is_file():
        result.opensim_reference_markers = _read_trc_marker_names(demo_trc)
        result.opensim_reference_source = str(demo_trc.resolve())

    if demo_trc and demo_trc.is_file():
        result.opensim_demo_trc_markers = _read_trc_marker_names(demo_trc)

    if not result.opensim_reference_markers:
        result.mapping_status = "missing"
        result.message = (
            "OpenSim demo reference not found. Expected "
            f"{DEMO_IK_SETUP_XML} or demo TRC."
        )
        _write_marker_mapping_report(result)
        return result

    ref_set = set(result.opensim_reference_markers)
    sw_set = set(result.stablewalk_trc_markers)
    result.matching_markers = sorted(sw_set & ref_set)
    result.missing_in_opensim = sorted(sw_set - ref_set)
    result.extra_in_opensim = sorted(ref_set - sw_set)

    mapped_path = mapped_trc_path_for(stablewalk_trc)
    try:
        _, direct_names, derived_names, derived_details, reconstruction = create_mapped_trc(
            stablewalk_trc, mapped_path, mapping
        )
        result.mapped_trc_path = str(mapped_path.resolve())
        result.mapped_opensim_markers = _read_trc_marker_names(mapped_path)
        result.direct_mapped_markers = direct_names
        result.derived_anatomical_markers = derived_names
        result.synthetic_markers = derived_names
        result.synthetic_marker_details = derived_details
        result.reconstruction = reconstruction
        result.temporal_estimated_markers = [
            n for n in reconstruction.marker_names
            if any(
                reconstruction.frames[i][2][n].mapping_type == "TEMPORAL_ESTIMATED"
                for i in range(len(reconstruction.frames))
            )
        ]
        ref_set_full = set(GAIT2392_MARKER_NAMES)
        present = set(result.mapped_opensim_markers)
        result.unavailable_markers = sorted(ref_set_full - present)
    except (OSError, ValueError) as exc:
        result.mapping_status = "missing"
        result.message = f"Failed to create mapped TRC: {exc}"
        _write_marker_mapping_report(result)
        log_marker_comparison(result)
        return result

    mapped_set = set(result.mapped_opensim_markers)
    result.mapped_matching_markers = sorted(mapped_set & ref_set)
    result.missing_for_ik = sorted(ref_set - mapped_set)
    result.unmapped_stablewalk_markers = sorted(
        sw for sw in result.stablewalk_trc_markers if sw not in mapping
    )

    _compute_readiness(result)
    _write_marker_mapping_report(result)
    log_marker_comparison(result)
    return result


def log_ik_validation_summary(comparison: MarkerComparison) -> None:
    """Print pre-IK validation block (counts, coverage, readiness tier)."""
    ref_n = len(comparison.opensim_reference_markers)
    logger.info("--- StableWalk IK validation (pre-run) ---")
    logger.info("Original MediaPipe markers: %d", len(comparison.stablewalk_trc_markers))
    logger.info("Mapped OpenSim markers (total): %d", len(comparison.mapped_opensim_markers))
    logger.info("  Direct markers: %d", len(comparison.direct_mapped_markers))
    logger.info(
        "  Derived anatomical markers: %d",
        len(comparison.derived_anatomical_markers),
    )
    logger.info(
        "  Temporal estimated markers: %d",
        len(comparison.temporal_estimated_markers),
    )
    logger.info("  Unavailable markers: %d", len(comparison.unavailable_markers))
    logger.info(
        "Raw coverage: %.1f%% | High-confidence coverage: %.1f%%",
        comparison.raw_coverage_percent,
        comparison.high_confidence_coverage_percent,
    )
    logger.info(
        "Markers matching OpenSim model: %d / %d (%.1f%% name overlap)",
        len(comparison.mapped_matching_markers),
        ref_n,
        comparison.coverage_percent,
    )
    logger.info(
        "IK readiness: %s (score %.1f)",
        comparison.ik_readiness_level,
        comparison.ik_readiness_score,
    )
    if comparison.low_confidence_markers:
        logger.warning(
            "Low-confidence markers before IK (not blocking): %s",
            ", ".join(comparison.low_confidence_markers),
        )
    logger.info("IK readiness tier: %s", comparison.ik_readiness_tier)
    logger.info("Marker mapping status: %s", comparison.mapping_status)
    logger.info("Reliability: %s (not clinical-grade)", comparison.reliability)
    logger.info("%s", MEDIAPIPE_LIMITATION_EXPLANATION)
    logger.info("--- end IK validation ---")


def log_marker_comparison(comparison: MarkerComparison) -> None:
    """Print a clear marker comparison block to the console."""
    logger.info("--- StableWalk ↔ OpenSim marker comparison ---")
    logger.info(
        "Original StableWalk TRC markers (%d): %s",
        len(comparison.stablewalk_trc_markers),
        ", ".join(comparison.stablewalk_trc_markers) or "(none)",
    )
    logger.info(
        "Mapped OpenSim marker names (%d): %s",
        len(comparison.mapped_opensim_markers),
        ", ".join(comparison.mapped_opensim_markers) or "(none)",
    )
    logger.info(
        "  Direct (%d): %s",
        len(comparison.direct_mapped_markers),
        ", ".join(comparison.direct_mapped_markers) or "(none)",
    )
    logger.info(
        "  Derived anatomical (%d): %s",
        len(comparison.derived_anatomical_markers),
        ", ".join(comparison.derived_anatomical_markers) or "(none)",
    )
    logger.info(
        "  Temporal estimated (%d): %s",
        len(comparison.temporal_estimated_markers),
        ", ".join(comparison.temporal_estimated_markers) or "(none)",
    )
    logger.info(
        "OpenSim demo/model markers (%d): %s",
        len(comparison.opensim_reference_markers),
        ", ".join(comparison.opensim_reference_markers[:20])
        + (" ..." if len(comparison.opensim_reference_markers) > 20 else ""),
    )
    logger.info(
        "Matching markers after mapping (%d): %s",
        len(comparison.mapped_matching_markers),
        ", ".join(comparison.mapped_matching_markers) or "(none)",
    )
    logger.info("Coverage: %.1f%%", comparison.coverage_percent)
    logger.info(
        "Raw coverage: %.1f%% | High-confidence: %.1f%%",
        comparison.raw_coverage_percent,
        comparison.high_confidence_coverage_percent,
    )
    logger.info(
        "IK readiness: %s (score %.1f)",
        comparison.ik_readiness_level,
        comparison.ik_readiness_score,
    )
    logger.info("Reliability: %s", comparison.reliability)
    logger.info("IK experimental possible: %s", comparison.ik_experimental_ready)
    logger.info("IK reasonably usable: %s", comparison.ik_reasonably_usable)
    logger.info("Marker mapping status: %s", comparison.mapping_status)
    if comparison.mapped_trc_path:
        logger.info("Mapped TRC: %s", comparison.mapped_trc_path)
    logger.info("Mapping JSON: %s", MARKER_MAPPING_JSON)
    logger.info("Mapping report: %s", MARKER_MAPPING_REPORT)
    if comparison.message:
        logger.info("%s", comparison.message)
    logger.info("--- end marker comparison ---")


def _write_marker_mapping_report(comparison: MarkerComparison) -> Path:
    """Write human-readable marker comparison to marker_mapping_report.txt."""
    mapping = load_stablewalk_to_opensim_mapping()
    catalog = format_mapping_catalog_table()
    lines = [
        "StableWalk ↔ OpenSim Marker Mapping Report",
        "=" * 50,
        "",
        f"StableWalk TRC path: {comparison.stablewalk_trc_path or '(not exported)'}",
        f"Mapped TRC path: {comparison.mapped_trc_path or '(not created)'}",
        f"OpenSim reference: {comparison.opensim_reference_source or '(not found)'}",
        "",
        "Reconstruction summary",
        "-" * 30,
        f"Direct markers: {len(comparison.direct_mapped_markers)}",
        f"Derived anatomical markers: {len(comparison.derived_anatomical_markers)}",
        f"Temporal estimated markers: {len(comparison.temporal_estimated_markers)}",
        f"Unavailable markers: {len(comparison.unavailable_markers)}",
        "",
        f"Raw coverage: {comparison.raw_coverage_percent}%",
        f"High-confidence coverage: {comparison.high_confidence_coverage_percent}%",
        f"IK readiness: {comparison.ik_readiness_level} (score {comparison.ik_readiness_score:.1f})",
        "",
        f"Original MediaPipe markers ({len(comparison.stablewalk_trc_markers)}):",
        ", ".join(comparison.stablewalk_trc_markers) or "(none)",
        "",
        f"Mapped OpenSim markers — total ({len(comparison.mapped_opensim_markers)}):",
        ", ".join(comparison.mapped_opensim_markers) or "(none)",
        "",
        f"Direct mappings ({len(comparison.direct_mapped_markers)}):",
        ", ".join(comparison.direct_mapped_markers) or "(none)",
        "",
        f"Derived anatomical markers ({len(comparison.derived_anatomical_markers)}):",
    ]
    for name in comparison.derived_anatomical_markers:
        detail = comparison.synthetic_marker_details.get(name, "")
        lines.append(f"  [DERIVED_ANATOMICAL] {name} — {detail}")
    if not comparison.derived_anatomical_markers:
        lines.append("(none)")
    if comparison.temporal_estimated_markers:
        lines.extend([
            "",
            f"Temporal estimated markers ({len(comparison.temporal_estimated_markers)}):",
            ", ".join(comparison.temporal_estimated_markers),
        ])
    if comparison.low_confidence_markers:
        lines.extend([
            "",
            f"Low-confidence markers ({len(comparison.low_confidence_markers)}):",
            ", ".join(comparison.low_confidence_markers),
        ])
    lines.extend([
        "",
        f"OpenSim demo/model markers ({len(comparison.opensim_reference_markers)}):",
        ", ".join(comparison.opensim_reference_markers) or "(none)",
        "",
        f"Matching markers after mapping ({len(comparison.mapped_matching_markers)}):",
        ", ".join(comparison.mapped_matching_markers) or "(none)",
        "",
        f"Name overlap coverage: {comparison.coverage_percent}%",
        f"IK readiness tier: {comparison.ik_readiness_tier}",
        f"Reliability: {comparison.reliability} (not clinical-grade)",
        "",
        f"Unmapped StableWalk markers ({len(comparison.unmapped_stablewalk_markers)}):",
        ", ".join(comparison.unmapped_stablewalk_markers) or "(none)",
        "",
        f"Missing for IK in OpenSim model ({len(comparison.missing_for_ik)}):",
        ", ".join(comparison.missing_for_ik) or "(none)",
        "",
        f"Mapping status: {comparison.mapping_status}",
        f"IK experimental ready: {comparison.ik_experimental_ready}",
        f"IK reasonably usable: {comparison.ik_reasonably_usable}",
        f"IK high coverage: {comparison.ik_reliable_ready}",
        "",
        comparison.message,
        "",
        MEDIAPIPE_LIMITATION_EXPLANATION,
        "",
        "Complete mapping catalog",
        "-" * 30,
    ])
    header = ["OpenSim Marker", "Mapping Type", "Source MediaPipe Landmarks", "Calculation", "Confidence", "Biomechanical Risk"]
    lines.append("\t".join(header))
    for row in catalog:
        lines.append("\t".join(str(row.get(h, "")) for h in header))
    lines.extend([
        "",
        "StableWalk → OpenSim direct mapping (marker_mapping.json):",
    ])
    for sw, osim_name in mapping.items():
        lines.append(f"  {sw} -> {osim_name}")
    lines.extend([
        "",
        "Landmarks used for derived anatomical reconstruction:",
    ])
    for sw, note in PROPOSED_MEDIAPIPE_TO_GAIT2392.items():
        if sw not in mapping:
            lines.append(f"  {sw} -> {note}")

    MARKER_MAPPING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MARKER_MAPPING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    catalog_path = MARKER_MAPPING_REPORT.with_name("marker_mapping_catalog.csv")
    catalog_path.write_text(mapping_catalog_csv(catalog), encoding="utf-8")
    return MARKER_MAPPING_REPORT


def load_marker_mapping_json() -> dict[str, Any] | None:
    if not MARKER_MAPPING_JSON.is_file():
        return None
    try:
        return json.loads(MARKER_MAPPING_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def mapping_status_label(comparison: MarkerComparison | None) -> str:
    if comparison is None:
        return "Missing"
    status = comparison.mapping_status
    if status == "improved":
        return "Improved"
    if status == "partial":
        return "Partial"
    if status == "experimental":
        return "Experimental"
    return "Missing"


def reliability_label(comparison: MarkerComparison | None) -> str:
    if comparison is None:
        return "limited"
    return comparison.reliability


def stablewalk_ik_status_label(comparison: MarkerComparison | None, *, run_state: str | None = None) -> str:
    """GUI label for StableWalk IK readiness."""
    if run_state == "Running":
        return "Running"
    if run_state == "Completed":
        return "Completed"
    if run_state == "Failed":
        return "Failed"
    if comparison is None:
        return "Not ready"
    if comparison.ik_experimental_ready:
        if comparison.ik_reasonably_usable:
            return "Experimental but runnable"
        return "Experimental"
    return "Not ready"


# Backward-compatible alias
STABLEWALK_IK_PARTIAL_MSG = MEDIAPIPE_LIMITATION_EXPLANATION


__all__ = [
    "MARKER_MAPPING_JSON",
    "MARKER_MAPPING_REPORT",
    "MAPPED_TRC_SUFFIX",
    "STABLEWALK_IK_MAPPING_IMPLEMENTED",
    "STABLEWALK_IK_NOT_READY_MSG",
    "STABLEWALK_IK_PARTIAL_MSG",
    "STABLEWALK_IK_EXPERIMENTAL_WARNING",
    "STABLEWALK_IK_SETUP_FILENAME",
    "MEDIAPIPE_LIMITATION_EXPLANATION",
    "DEFAULT_STABLEWALK_TO_OPENSIM",
    "GAIT2392_PIPELINE_DIR",
    "DEMO_IK_SETUP_XML",
    "DEMO_IK_DEMO_TRC",
    "DEMO_IK_MODEL",
    "DEMO_IK_OUTPUT_MOT",
    "PROPOSED_MEDIAPIPE_TO_GAIT2392",
    "GAIT2392_MARKER_NAMES",
    "DERIVED_ANATOMICAL_WEIGHT_SCALE",
    "TEMPORAL_ESTIMATED_WEIGHT_SCALE",
    "SYNTHETIC_WEIGHT_SCALE",
    "PREFERRED_MAPPED_MARKER_ORDER",
    "SYNTHETIC_MARKER_SPECS",
    "MarkerReconstructionConfig",
    "MarkerReconstructionResult",
    "MIN_REASONABLY_USABLE_MARKERS",
    "MIN_RELIABLE_IK_MARKERS",
    "get_marker_weight",
    "configure_stablewalk_ik_setup",
    "MarkerComparison",
    "SyntheticMarkerSpec",
    "ensure_default_marker_mapping_json",
    "load_stablewalk_to_opensim_mapping",
    "mapped_trc_path_for",
    "run_name_from_trc_path",
    "create_mapped_trc",
    "read_ik_setup_marker_tasks",
    "read_ik_setup_output_mot",
    "demo_ik_setup_available",
    "compare_stablewalk_trc_to_opensim",
    "log_marker_comparison",
    "log_ik_validation_summary",
    "load_marker_mapping_json",
    "mapping_status_label",
    "reliability_label",
    "stablewalk_ik_status_label",
]
