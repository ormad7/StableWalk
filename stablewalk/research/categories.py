"""Gait categories, authenticity, and clinical-label policy for research clips.

Clinical diagnoses (including dementia) are displayed only when the catalog
records a verified source. Category tags alone are never treated as a
diagnosis.
"""

from __future__ import annotations

from enum import Enum

GAIT_CATEGORY_VALUES = (
    "normal_gait",
    "hemiplegic_gait",
    "limping_gait",
    "ataxic_gait",
    "parkinsonian_gait",
    "shuffling_gait",
    "other_abnormal_gait",
    "unknown_or_unverified",
    "dementia_research",
)

GAIT_CATEGORY_LABELS: dict[str, str] = {
    "normal_gait": "Normal gait",
    "hemiplegic_gait": "Hemiplegic gait",
    "limping_gait": "Limping gait",
    "ataxic_gait": "Ataxic gait",
    "parkinsonian_gait": "Parkinsonian gait",
    "shuffling_gait": "Shuffling gait",
    "other_abnormal_gait": "Other abnormal gait",
    "unknown_or_unverified": "Unknown or unverified",
    "dementia_research": "Dementia-related gait (research)",
}

AUTHENTICITY_VALUES = (
    "authentic",
    "experimentally_induced",
    "simulated",
)

AUTHENTICITY_LABELS: dict[str, str] = {
    "authentic": "Authentic (observed walking)",
    "experimentally_induced": "Experimentally induced",
    "simulated": "Simulated / animated",
}

CAMERA_VIEW_VALUES = (
    "unknown",
    "frontal",
    "sagittal_left",
    "sagittal_right",
    "oblique",
)

DEMENTIA_CATEGORY = "dementia_research"

CLINICAL_POLICY = (
    "Do not label a recording as dementia-related (or any clinical diagnosis) "
    "unless catalog metadata includes a verified diagnosis and a reliable source. "
    "Gait category tags are research organization only."
)


class GaitCategory(str, Enum):
    NORMAL = "normal_gait"
    HEMIPLEGIC = "hemiplegic_gait"
    LIMPING = "limping_gait"
    ATAXIC = "ataxic_gait"
    PARKINSONIAN = "parkinsonian_gait"
    SHUFFLING = "shuffling_gait"
    OTHER_ABNORMAL = "other_abnormal_gait"
    UNKNOWN = "unknown_or_unverified"
    DEMENTIA_RESEARCH = "dementia_research"


def normalize_category(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "normal": GaitCategory.NORMAL.value,
        "healthy": GaitCategory.NORMAL.value,
        "hemiplegia": GaitCategory.HEMIPLEGIC.value,
        "hemiplegic": GaitCategory.HEMIPLEGIC.value,
        "limp": GaitCategory.LIMPING.value,
        "limping": GaitCategory.LIMPING.value,
        "ataxia": GaitCategory.ATAXIC.value,
        "ataxic": GaitCategory.ATAXIC.value,
        "parkinson": GaitCategory.PARKINSONIAN.value,
        "parkinsons": GaitCategory.PARKINSONIAN.value,
        "parkinsonian": GaitCategory.PARKINSONIAN.value,
        "shuffle": GaitCategory.SHUFFLING.value,
        "shuffling": GaitCategory.SHUFFLING.value,
        "abnormal": GaitCategory.OTHER_ABNORMAL.value,
        "other": GaitCategory.OTHER_ABNORMAL.value,
        "unknown": GaitCategory.UNKNOWN.value,
        "unverified": GaitCategory.UNKNOWN.value,
        "dementia": GaitCategory.DEMENTIA_RESEARCH.value,
        "dementia_related": GaitCategory.DEMENTIA_RESEARCH.value,
        "dementia_research": GaitCategory.DEMENTIA_RESEARCH.value,
    }
    mapped = aliases.get(raw, raw)
    if mapped in GAIT_CATEGORY_VALUES:
        return mapped
    return GaitCategory.UNKNOWN.value


def normalize_authenticity(value: str | None) -> str:
    raw = str(value or "").strip().lower().replace(" ", "_")
    if raw in AUTHENTICITY_VALUES:
        return raw
    if raw in ("real", "observed", "clinical_observation"):
        return "authentic"
    if raw in ("induced", "experimental"):
        return "experimentally_induced"
    if raw in ("synthetic", "animation", "avatar"):
        return "simulated"
    return "authentic"


def display_clinical_diagnosis(
    *,
    diagnosis: str | None,
    verified: bool,
    source: str | None,
    category: str | None = None,
) -> str | None:
    """Return a diagnosis string only when a reliable source verified it."""
    text = str(diagnosis or "").strip()
    src = str(source or "").strip()
    if text and verified and src:
        return text
    return None


def dementia_label_allowed(
    *,
    diagnosis: str | None,
    verified: bool,
    source: str | None,
) -> bool:
    shown = display_clinical_diagnosis(
        diagnosis=diagnosis, verified=verified, source=source
    )
    if not shown:
        return False
    return "dement" in shown.lower()
