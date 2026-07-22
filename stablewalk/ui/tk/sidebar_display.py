"""
Compact sidebar label text for OpenSim panels (display only).
"""

from __future__ import annotations

OPENSIM_RELIABILITY_WARNING = "Experimental, not clinical-grade"
RELIABILITY_TOOLTIP = (
    "MediaPipe-based IK is not clinical-grade motion capture."
)

PRESENTATION_MODE_TITLE = "Presentation Demo Mode"
PRESENTATION_MODE_SUBTITLE = "Synthetic demo — no video export required"
PRESENTATION_IK_STATUS = "StableWalk IK: Requires analyzed video"
PRESENTATION_MAPPING_STATUS = "Mapping: Available after video export"
PRESENTATION_EXPORT_SUMMARY = "Export: Load video to export"
PRESENTATION_EXPORT_FILE = "Demo mode"
PRESENTATION_LAST_EXPORT = "Last Export: Demo mode"
PRESENTATION_LAST_IK = "Last IK: Requires analyzed video"
PRESENTATION_RELIABILITY = "Reliability: After real-video export"
PRESENTATION_WORKFLOW_NOTE = (
    "To run OpenSim IK on a real walk, load/analyze a video, "
    "click Export OpenSim Files, then Run StableWalk IK."
)
PRESENTATION_IK_BUTTON_TOOLTIP = (
    "Run Export OpenSim Files on an analyzed video first."
)
PRESENTATION_EXPORT_BUTTON_TOOLTIP = (
    "Load and analyze a video, then click Export OpenSim Files."
)

OPENSIM_SUGGESTED_MODEL_SECTION = (
    "Suggested OpenSim model (click Select to load)"
)
OPENSIM_MODEL_DROPDOWN_HINT = (
    "The dropdown only selects a candidate model. "
    "Click Select to validate and load it."
)


def compact_sdk_line(*, installed: bool, version: str | None = None) -> str:
    del version  # version shown in tooltip / OpenSim log only
    return "SDK: Installed" if installed else "SDK: Not installed"


def sdk_tooltip_line(*, version: str | None) -> str:
    ver = (version or "").strip()
    if ver and ver != "unknown":
        return f"OpenSim SDK version: {ver}"
    return "OpenSim SDK installed"


def compact_mode_line(*, sdk: bool) -> str:
    return "Mode: Real SDK" if sdk else "Mode: Export only"


def compact_opensim_ready_line(*, sdk: bool, presentation: bool = False) -> str:
    if presentation:
        return "OpenSim: Demo mode"
    return "OpenSim: Ready" if sdk else "OpenSim: Not installed"


def compact_model_line(*, model_valid: bool) -> str:
    return "Model: Loaded" if model_valid else "Model: Not loaded"


def compact_export_summary_line(
    *,
    export_complete: bool,
    has_session: bool,
    presentation: bool = False,
) -> str:
    if presentation:
        return "Export: Demo mode"
    if export_complete:
        return "Export: Complete"
    if has_session:
        return "Export: Pending"
    return "Export: —"


SUGGESTED_MODEL_NOTE = "Suggested only — click Select to load manually"
LOADED_MODEL_NONE_HINT = "None — click Select or Load .osim Model"
OPENSIM_STATUS_PATH_MAX_LEN = 52


def opensim_status_path_display(
    rel_path: str | None,
    *,
    max_len: int = OPENSIM_STATUS_PATH_MAX_LEN,
) -> tuple[str, str]:
    """Return (short_display, full_path_for_tooltip)."""
    full = (rel_path or "").strip()
    if not full:
        return "—", ""
    short = truncate_line(full, max_len=max_len)
    return short, full


def ik_pipeline_model_status(
    *,
    rel_path: str | None,
    file_available: bool,
) -> tuple[str, str]:
    """OpenSim Status: default model used by Demo / StableWalk IK internally."""
    if not (rel_path or "").strip():
        return "IK pipeline model: —", ""
    short, tip = opensim_status_path_display(rel_path)
    text = f"IK pipeline model: {short}"
    if not file_available:
        tip = (
            f"{tip}\n\n(File not found on disk)"
            if tip
            else "(File not found on disk)"
        )
    return text, tip


def suggested_model_status(
    *,
    rel_path: str | None,
) -> tuple[str, str]:
    """OpenSim Status: dropdown candidate (not loaded until Select)."""
    short, tip = opensim_status_path_display(rel_path)
    return f"Suggested model: {short}", tip


def loaded_model_status(
    *,
    rel_path: str | None,
    loaded: bool,
) -> tuple[str, str]:
    """OpenSim Status: model validated via Select / Load .osim Model."""
    if not loaded:
        return f"Loaded model: {LOADED_MODEL_NONE_HINT}", ""
    short, tip = opensim_status_path_display(rel_path)
    return f"Loaded model: {short}", tip


def opensim_model_load_success_message(*, num_markers: int | None) -> str:
    """Popup body after a model passes validation and is loaded."""
    lines = ["OpenSim model loaded successfully."]
    if num_markers is not None:
        lines.append(f"Markers: {num_markers}")
    return "\n\n".join(lines)


def opensim_model_ik_error_message(
    *,
    blocked_template: bool = False,
    validation_message: str = "",
    num_markers: int | None = None,
) -> str:
    """Popup body when a selected model cannot be used for marker-based IK."""
    if blocked_template:
        return (
            "This .osim model could not be used for IK because it is an unscaled "
            "template without usable markers.\n\n"
            "Use a scaled model with a MarkerSet (for example "
            "subject01_simbody.osim from Gait2392_Pipeline/)."
        )
    if num_markers == 0 or "no markers" in validation_message.lower():
        return (
            "This .osim model could not be used for IK because it has no usable "
            "markers."
        )
    detail = validation_message.strip()
    if detail:
        return (
            "This .osim model could not be used for IK.\n\n"
            f"{detail}"
        )
    return "This .osim model could not be used for IK."


def mapping_compact_line(*, label: str, coverage_percent: float | None) -> str:
    if label in ("—", "Missing"):
        return "Mapping: —"
    if coverage_percent is not None:
        return f"Mapping: {label} {coverage_percent:.1f}%"
    return f"Mapping: {label}"


def export_file_line(
    label: str,
    *,
    exported: bool,
    has_session: bool,
    has_pose: bool,
) -> str:
    if exported:
        return f"{label}: Exported"
    if has_session and has_pose:
        return f"{label}: Missing"
    return f"{label}: Missing"


def demo_ik_line(*, state: str | None, sdk_ready: bool) -> str:
    if state == "Running":
        return "Demo IK: Running"
    if state == "Completed":
        return "Demo IK: Completed"
    if state == "Failed":
        return "Demo IK: Failed"
    if sdk_ready:
        return "Demo IK: Ready"
    return "Demo IK: —"


def stablewalk_ik_line(
    *,
    run_state: str | None,
    status_label: str,
) -> str:
    if run_state == "Running":
        return "StableWalk IK: Running"
    if run_state == "Completed":
        return "StableWalk IK: Completed"
    if run_state == "Failed":
        return "StableWalk IK: Failed"
    if status_label == "Ready":
        return "StableWalk IK: Ready"
    if status_label in ("Experimental but runnable", "Experimental"):
        return "StableWalk IK: Experimental"
    return "StableWalk IK: Not ready"


def mapping_label_line(*, label: str) -> str:
    return f"Mapping: {label}"


def markers_count_line(
    *,
    matched: int | None = None,
    reference: int | None = None,
    mapping_percent: float | None = None,
    mapped: int | None = None,
    total: int | None = None,
) -> str:
    """Sidebar marker count / mapping percentage line."""
    n_matched = matched if matched is not None else mapped
    n_ref = reference if reference is not None else total
    if n_matched is None or n_ref is None or n_ref <= 0:
        if mapping_percent is not None:
            return f"Markers: mapping {mapping_percent:.0f}%"
        return "Markers: —"
    if mapping_percent is not None:
        return f"Markers: {n_matched}/{n_ref} ({mapping_percent:.0f}%)"
    return f"Markers: {n_matched}/{n_ref}"


def coverage_line(*, percent: float | None) -> str:
    if percent is None:
        return "Coverage: —"
    return f"Coverage: {percent:.1f}%"


def reliability_line(*, reliability: str | None) -> str:
    """Compact sidebar reliability (short line; see RELIABILITY_TOOLTIP for detail)."""
    if not reliability:
        return "Reliability: —"
    if reliability.lower() in ("experimental", "moderate", "limited"):
        return "Reliability: Experimental"
    return f"Reliability: {reliability.capitalize()}"


def last_export_line(*, completed: bool) -> str:
    return "Last Export: Completed" if completed else "Last Export: Not yet exported"


def last_ik_line(*, run_state: str | None, ik_output_on_disk: bool) -> str:
    if run_state == "Failed":
        return "Last IK: Failed"
    if run_state == "Completed" or ik_output_on_disk:
        return "Last IK: Completed"
    return "Last IK: Not yet run"


def truncate_line(text: str, max_len: int = 52) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"
