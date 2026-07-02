"""
OpenSim marker mapping and StableWalk IK readiness validation.

Maps MediaPipe-exported TRC marker names (``L_KNEE``, ``R_HIP``, …) to OpenSim
Gait2392 marker names (``L.Shank.Upper``, ``R.ASIS``, …), generates synthetic
markers where MediaPipe landmarks are insufficient, writes a mapped TRC, and
compares against the OpenSim demo model/setup markers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from stablewalk import config
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
    "MediaPipe pose landmarks are approximated for OpenSim IK. Direct mappings use "
    "exported landmarks; synthetic markers are interpolated from hips, knees, ankles, "
    "and feet. This improves coverage versus raw export but is not clinical-grade mocap."
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
    "L_KNEE": "(synthetic → L.Shank.Upper / L.Thigh.Upper)",
    "R_KNEE": "(synthetic → R.Shank.Upper / R.Thigh.Upper)",
    "L_ELBOW": "(optional — not on Gait2392 lower-body IK set)",
    "R_ELBOW": "(optional — not on Gait2392 lower-body IK set)",
    "L_WRIST": "(optional — not on Gait2392 lower-body IK set)",
    "R_WRIST": "(optional — not on Gait2392 lower-body IK set)",
}

# Preferred column order in mapped TRC (matches Gait2392 demo layout where possible).
PREFERRED_MAPPED_MARKER_ORDER: list[str] = [
    "Sternum",
    "R.Acromium",
    "L.Acromium",
    "Top.Head",
    "R.ASIS",
    "L.ASIS",
    "V.Sacral",
    "R.Thigh.Upper",
    "L.Thigh.Upper",
    "R.Shank.Upper",
    "L.Shank.Upper",
    "R.Shank.Front",
    "L.Shank.Front",
    "R.Heel",
    "L.Heel",
    "R.Midfoot.Sup",
    "L.Midfoot.Sup",
    "R.Midfoot.Lat",
    "L.Midfoot.Lat",
    "R.Toe.Tip",
    "L.Toe.Tip",
]

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
SYNTHETIC_WEIGHT_SCALE = 0.5


@dataclass
class SyntheticMarkerSpec:
    """Rule for generating one OpenSim marker from MediaPipe landmarks."""

    opensim_name: str
    description: str
    required_landmarks: list[str]
    compute: Callable[[dict[str, tuple[float, float, float]]], tuple[float, float, float] | None]


def _lerp3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    return (
        a[0] + t * (b[0] - a[0]),
        a[1] + t * (b[1] - a[1]),
        a[2] + t * (b[2] - a[2]),
    )


def _centroid3(
    *points: tuple[float, float, float],
) -> tuple[float, float, float]:
    n = len(points)
    return (
        sum(p[0] for p in points) / n,
        sum(p[1] for p in points) / n,
        sum(p[2] for p in points) / n,
    )


def _compute_v_sacral(
    lm: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    left, right = lm.get("L_HIP"), lm.get("R_HIP")
    if left is None or right is None:
        return None
    mid = _lerp3(left, right, 0.5)
    return (mid[0], mid[1] * 0.98, mid[2])


def _compute_sternum(
    lm: dict[str, tuple[float, float, float]],
) -> tuple[float, float, float] | None:
    left, right, head = lm.get("L_SHOULDER"), lm.get("R_SHOULDER"), lm.get("HEAD")
    if left is None or right is None:
        return None
    shoulder_mid = _lerp3(left, right, 0.5)
    if head is not None:
        return _lerp3(shoulder_mid, head, 0.25)
    return shoulder_mid


def _compute_thigh_upper(side: str):
    def _fn(lm: dict[str, tuple[float, float, float]]) -> tuple[float, float, float] | None:
        hip, knee = lm.get(f"{side}_HIP"), lm.get(f"{side}_KNEE")
        if hip is None or knee is None:
            return None
        return _lerp3(hip, knee, 0.35)

    return _fn


def _compute_shank_upper(side: str):
    def _fn(lm: dict[str, tuple[float, float, float]]) -> tuple[float, float, float] | None:
        knee, ankle = lm.get(f"{side}_KNEE"), lm.get(f"{side}_ANKLE")
        if knee is None or ankle is None:
            return None
        return _lerp3(knee, ankle, 0.35)

    return _fn


def _compute_midfoot_sup(side: str):
    def _fn(lm: dict[str, tuple[float, float, float]]) -> tuple[float, float, float] | None:
        ankle, heel, toe = (
            lm.get(f"{side}_ANKLE"),
            lm.get(f"{side}_HEEL"),
            lm.get(f"{side}_TOE"),
        )
        pts = [p for p in (ankle, heel, toe) if p is not None]
        if len(pts) < 2:
            return None
        return _centroid3(*pts)

    return _fn


def _compute_midfoot_lat(side: str):
    def _fn(lm: dict[str, tuple[float, float, float]]) -> tuple[float, float, float] | None:
        ankle, heel = lm.get(f"{side}_ANKLE"), lm.get(f"{side}_HEEL")
        if ankle is None or heel is None:
            return None
        return _lerp3(ankle, heel, 0.5)

    return _fn


SYNTHETIC_MARKER_SPECS: list[SyntheticMarkerSpec] = [
    SyntheticMarkerSpec(
        "V.Sacral",
        "Midpoint of L_HIP and R_HIP (approximate sacral marker)",
        ["L_HIP", "R_HIP"],
        _compute_v_sacral,
    ),
    SyntheticMarkerSpec(
        "Sternum",
        "Between shoulder midpoint and HEAD (approximate sternum)",
        ["L_SHOULDER", "R_SHOULDER"],
        _compute_sternum,
    ),
    SyntheticMarkerSpec(
        "R.Thigh.Upper",
        "35% from R_HIP toward R_KNEE",
        ["R_HIP", "R_KNEE"],
        _compute_thigh_upper("R"),
    ),
    SyntheticMarkerSpec(
        "L.Thigh.Upper",
        "35% from L_HIP toward L_KNEE",
        ["L_HIP", "L_KNEE"],
        _compute_thigh_upper("L"),
    ),
    SyntheticMarkerSpec(
        "R.Shank.Upper",
        "35% from R_KNEE toward R_ANKLE",
        ["R_KNEE", "R_ANKLE"],
        _compute_shank_upper("R"),
    ),
    SyntheticMarkerSpec(
        "L.Shank.Upper",
        "35% from L_KNEE toward L_ANKLE",
        ["L_KNEE", "L_ANKLE"],
        _compute_shank_upper("L"),
    ),
    SyntheticMarkerSpec(
        "R.Midfoot.Sup",
        "Centroid of R_ANKLE, R_HEEL, R_TOE",
        ["R_ANKLE", "R_HEEL", "R_TOE"],
        _compute_midfoot_sup("R"),
    ),
    SyntheticMarkerSpec(
        "L.Midfoot.Sup",
        "Centroid of L_ANKLE, L_HEEL, L_TOE",
        ["L_ANKLE", "L_HEEL", "L_TOE"],
        _compute_midfoot_sup("L"),
    ),
    SyntheticMarkerSpec(
        "R.Midfoot.Lat",
        "Midpoint of R_ANKLE and R_HEEL",
        ["R_ANKLE", "R_HEEL"],
        _compute_midfoot_lat("R"),
    ),
    SyntheticMarkerSpec(
        "L.Midfoot.Lat",
        "Midpoint of L_ANKLE and L_HEEL",
        ["L_ANKLE", "L_HEEL"],
        _compute_midfoot_lat("L"),
    ),
]


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
    synthetic_markers: list[str] = field(default_factory=list)
    synthetic_marker_details: dict[str, str] = field(default_factory=dict)
    mapped_matching_markers: list[str] = field(default_factory=list)
    unmapped_stablewalk_markers: list[str] = field(default_factory=list)
    missing_for_ik: list[str] = field(default_factory=list)
    mapping_status: str = "missing"  # missing | experimental | partial | improved
    ik_readiness_tier: str = "not ready"
    reliability: str = "limited"
    coverage_percent: float = 0.0
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
            "reliability": self.reliability,
            "coverage_percent": self.coverage_percent,
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


def get_marker_weight(opensim_name: str, *, is_synthetic: bool = False) -> float:
    """Return IK task weight for a marker (synthetic markers weighted lower)."""
    base = MARKER_WEIGHT_BASE.get(opensim_name, MARKER_WEIGHT_DEFAULT)
    if is_synthetic:
        return max(base * SYNTHETIC_WEIGHT_SCALE, 0.05)
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
) -> tuple[Path, list[str], list[str], dict[str, str]]:
    """
    Create a mapped TRC with OpenSim marker names (direct + synthetic).

    Returns ``(path, direct_marker_names, synthetic_marker_names, synthetic_details)``.
    """
    source_trc = Path(source_trc)
    output_trc = Path(output_trc)
    mapping = mapping or load_stablewalk_to_opensim_mapping()

    source_markers, frames, rate, units = _read_trc_frame_data(source_trc)
    if not frames:
        raise ValueError(f"No frame data in TRC: {source_trc}")

    # Direct mappings: StableWalk name → OpenSim name (skip duplicate OpenSim targets).
    direct_pairs: list[tuple[str, str]] = []
    used_opensim: set[str] = set()
    for sw_name in source_markers:
        osim_name = mapping.get(sw_name)
        if osim_name and osim_name not in used_opensim:
            direct_pairs.append((sw_name, osim_name))
            used_opensim.add(osim_name)

    if not direct_pairs and not SYNTHETIC_MARKER_SPECS:
        raise ValueError("No StableWalk markers could be mapped to OpenSim names.")

    direct_opensim_names = [osim for _, osim in direct_pairs]
    synthetic_details: dict[str, str] = {}
    synthetic_names: list[str] = []

    for spec in SYNTHETIC_MARKER_SPECS:
        if spec.opensim_name in used_opensim:
            continue
        synthetic_names.append(spec.opensim_name)
        synthetic_details[spec.opensim_name] = spec.description
        used_opensim.add(spec.opensim_name)

    all_marker_names = _order_mapped_markers(set(direct_opensim_names) | set(synthetic_names))
    n_frames = len(frames)

    out_lines: list[str] = []
    out_lines.append(f"PathFileType\t4\t(X/Y/Z)\t{output_trc.name}")
    out_lines.append(
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
        "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames"
    )
    out_lines.append(
        f"{rate}\t{rate}\t{n_frames}\t{len(all_marker_names)}\t{units}\t"
        f"{rate}\t1\t{n_frames}"
    )
    out_lines.append("Frame#\tTime\t" + "\t\t\t".join(all_marker_names) + "\t\t")
    out_lines.append(
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, len(all_marker_names) + 1))
    )
    out_lines.append("")

    spec_by_name = {s.opensim_name: s for s in SYNTHETIC_MARKER_SPECS}

    for frame_num, time_val, landmarks in frames:
        cells = [str(frame_num), f"{time_val:.6f}"]
        computed: dict[str, tuple[float, float, float]] = {}

        for sw_name, osim_name in direct_pairs:
            pos = landmarks.get(sw_name)
            if pos is not None:
                computed[osim_name] = pos

        for osim_name in synthetic_names:
            spec = spec_by_name[osim_name]
            pos = spec.compute(landmarks)
            if pos is not None:
                computed[osim_name] = pos

        for osim_name in all_marker_names:
            pos = computed.get(osim_name)
            if pos is not None:
                cells.extend([f"{pos[0]:.6f}", f"{pos[1]:.6f}", f"{pos[2]:.6f}"])
            else:
                cells.extend(["", "", ""])

        out_lines.append("\t".join(cells))

    output_trc.parent.mkdir(parents=True, exist_ok=True)
    output_trc.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    # Filter synthetic list to those actually present in output (computed at least once).
    actual_synthetic = [
        n for n in synthetic_names
        if n in _read_trc_marker_names(output_trc)
    ]
    actual_direct = [osim for _, osim in direct_pairs]

    logger.info("Wrote mapped TRC for OpenSim IK -> %s", output_trc)
    logger.info(
        "  %d direct + %d synthetic marker(s): %s",
        len(actual_direct),
        len(actual_synthetic),
        ", ".join(all_marker_names),
    )
    return output_trc, actual_direct, actual_synthetic, synthetic_details


def configure_stablewalk_ik_setup(
    setup_path: Path,
    trc_marker_names: list[str],
    *,
    synthetic_markers: set[str] | None = None,
) -> None:
    """
    Post-process IK setup XML: disable missing marker tasks, apply weights.

    OpenSim ``printToXML`` includes all model markers; this enables only markers
    present in the mapped TRC and sets reasonable weights.
    """
    synthetic = synthetic_markers or set()
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
        weight = get_marker_weight(name, is_synthetic=name in synthetic)
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

    comparison.ik_experimental_ready = n_match >= MIN_EXPERIMENTAL_IK_MARKERS
    comparison.ik_reasonably_usable = n_match >= MIN_REASONABLY_USABLE_MARKERS
    comparison.ik_reliable_ready = n_match >= MIN_RELIABLE_IK_MARKERS
    comparison.stablewalk_ik_ready = comparison.ik_reasonably_usable

    if not comparison.ik_experimental_ready:
        comparison.ik_readiness_tier = "not ready"
    elif comparison.ik_reliable_ready:
        comparison.ik_readiness_tier = "reliable"
    elif comparison.ik_reasonably_usable:
        comparison.ik_readiness_tier = "reasonably usable"
    else:
        comparison.ik_readiness_tier = "experimental"

    synth_ratio = (
        len(comparison.synthetic_markers) / max(len(comparison.mapped_opensim_markers), 1)
    )
    if comparison.coverage_percent >= 65 and synth_ratio <= 0.55:
        comparison.reliability = "high"
    elif comparison.coverage_percent >= 40:
        comparison.reliability = "moderate"
    else:
        comparison.reliability = "limited"

    if not comparison.stablewalk_trc_markers:
        comparison.mapping_status = "missing"
    elif comparison.ik_reasonably_usable and n_match >= 18:
        comparison.mapping_status = "improved"
    elif comparison.ik_experimental_ready:
        comparison.mapping_status = "partial" if comparison.ik_reasonably_usable else "experimental"
    else:
        comparison.mapping_status = "experimental"

    if comparison.mapping_status == "improved":
        comparison.message = (
            f"Improved marker mapping: {n_match}/{ref_count} OpenSim markers covered "
            f"({comparison.coverage_percent}%) via direct + synthetic markers. "
            f"{MEDIAPIPE_LIMITATION_EXPLANATION}"
        )
    elif comparison.ik_experimental_ready:
        comparison.message = (
            f"Partial marker mapping: {n_match}/{ref_count} OpenSim markers "
            f"({comparison.coverage_percent}%). {MEDIAPIPE_LIMITATION_EXPLANATION}"
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
        _, direct_names, synthetic_names, synthetic_details = create_mapped_trc(
            stablewalk_trc, mapped_path, mapping
        )
        result.mapped_trc_path = str(mapped_path.resolve())
        result.mapped_opensim_markers = _read_trc_marker_names(mapped_path)
        result.direct_mapped_markers = direct_names
        result.synthetic_markers = synthetic_names
        result.synthetic_marker_details = synthetic_details
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
    logger.info("  Direct mappings: %d", len(comparison.direct_mapped_markers))
    logger.info("  Synthetic markers: %d", len(comparison.synthetic_markers))
    for name in comparison.synthetic_markers:
        detail = comparison.synthetic_marker_details.get(name, "")
        logger.info("    [synthetic] %s — %s", name, detail)
    logger.info(
        "Markers matching OpenSim model: %d / %d (%.1f%% coverage)",
        len(comparison.mapped_matching_markers),
        ref_n,
        comparison.coverage_percent,
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
        "  Synthetic (%d): %s",
        len(comparison.synthetic_markers),
        ", ".join(comparison.synthetic_markers) or "(none)",
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
    logger.info("IK readiness tier: %s", comparison.ik_readiness_tier)
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
    lines = [
        "StableWalk ↔ OpenSim Marker Mapping Report",
        "=" * 50,
        "",
        f"StableWalk TRC path: {comparison.stablewalk_trc_path or '(not exported)'}",
        f"Mapped TRC path: {comparison.mapped_trc_path or '(not created)'}",
        f"OpenSim reference: {comparison.opensim_reference_source or '(not found)'}",
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
        f"Synthetic markers ({len(comparison.synthetic_markers)}):",
    ]
    for name in comparison.synthetic_markers:
        detail = comparison.synthetic_marker_details.get(name, "")
        lines.append(f"  [SYNTHETIC] {name} — {detail}")
    if not comparison.synthetic_markers:
        lines.append("(none)")
    lines.extend([
        "",
        f"OpenSim demo/model markers ({len(comparison.opensim_reference_markers)}):",
        ", ".join(comparison.opensim_reference_markers) or "(none)",
        "",
        f"Matching markers after mapping ({len(comparison.mapped_matching_markers)}):",
        ", ".join(comparison.mapped_matching_markers) or "(none)",
        "",
        f"Coverage: {comparison.coverage_percent}%",
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
        "StableWalk → OpenSim direct mapping (marker_mapping.json):",
    ])
    for sw, osim_name in mapping.items():
        lines.append(f"  {sw} -> {osim_name}")
    lines.extend([
        "",
        "Landmarks used only for synthetic generation:",
    ])
    for sw, note in PROPOSED_MEDIAPIPE_TO_GAIT2392.items():
        if sw not in mapping:
            lines.append(f"  {sw} -> {note}")

    MARKER_MAPPING_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MARKER_MAPPING_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
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
    "PREFERRED_MAPPED_MARKER_ORDER",
    "SYNTHETIC_MARKER_SPECS",
    "MIN_EXPERIMENTAL_IK_MARKERS",
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
