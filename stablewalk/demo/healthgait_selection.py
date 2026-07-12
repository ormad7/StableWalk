"""
Health&Gait metadata selection for StableWalk normal and performance demos.

Official source: https://zenodo.org/records/14039922

Important: the public Health&Gait release does **not** redistribute raw RGB walking
videos. The Zenodo archive provides silhouettes, pose JSON, optical flow, semantic
segmentation, and CSV metadata. StableWalk GUI demos require candidate MP4 files
that must be obtained separately (controlled re-recording) or validated after
authorized access — see DEMO_VIDEO_SOURCES.md.
"""

from __future__ import annotations

import csv
import json
import logging
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import ContentTooShortError, URLError
from urllib.request import urlretrieve

from stablewalk import config

logger = logging.getLogger(__name__)

ZENODO_RECORD = "14039922"
ZENODO_SAMPLES_URL = (
    f"https://zenodo.org/api/records/{ZENODO_RECORD}/files/dataset_samples.zip/content"
)
EXPECTED_SAMPLES_ZIP_BYTES = 62_369_710
DOWNLOAD_MAX_ATTEMPTS = 3

SAMPLE_STRUCTURE_DOC = """
Health&Gait dataset_samples.zip structure (inspected):
  dataset_samples/gait_parameters.csv
  dataset_samples/gait_parameters_estimation.csv
  dataset_samples/participants_measures.csv
  dataset_samples/silhouette/PAxxx/{UGS|FGS}/WoJ_1_YOLOV8/*.jpg
  dataset_samples/pose/... (when present in full archive)
  dataset_samples/semantic_segmentation/...
  dataset_samples/optical_flow/...

Full archive (Health_Gait.z01–z25 + Health_Gait.zip) is ~26.8 GB and follows:
  <participant_id>/<gait_speed>/<modality>/...

Per Scientific Data (2025): raw RGB recordings are NOT included in the public release.
""".strip()


@dataclass
class HealthGaitTrialCandidate:
    participant_id: str
    gait_speed: str
    velocity_m_s: float | None
    cadence_spm: float | None
    step_cm: float | None
    stride_cm: float | None
    selection_score: float
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "participant_id": self.participant_id,
            "gait_speed": self.gait_speed,
            "velocity_m_s": self.velocity_m_s,
            "cadence_spm": self.cadence_spm,
            "step_cm": self.step_cm,
            "stride_cm": self.stride_cm,
            "selection_score": round(self.selection_score, 3),
            "notes": list(self.notes),
        }


def healthgait_cache_dir() -> Path:
    return config.DEMO_VIDEOS_DIR / "_healthgait_cache"


def healthgait_samples_csv_dir() -> Path:
    """Directory with extracted CSVs from dataset_samples.zip."""
    for candidate in (
        config.DEMO_VIDEOS_DIR / "_healthgait_samples",
        healthgait_cache_dir() / "dataset_samples_csv",
    ):
        if any(candidate.glob("*.csv")):
            return candidate
    return healthgait_cache_dir() / "dataset_samples_csv"


def has_cached_healthgait_csvs() -> bool:
    return any(healthgait_samples_csv_dir().glob("*.csv"))


def _zip_is_usable(zip_path: Path) -> bool:
    if not zip_path.is_file():
        return False
    if not zipfile.is_zipfile(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.testzip()
        return True
    except (zipfile.BadZipFile, OSError):
        return False


def download_healthgait_samples(
    *,
    cache_dir: Path | None = None,
    force: bool = False,
) -> Path | None:
    """
    Download dataset_samples.zip (~62 MB) for metadata/schema inspection.

    Returns the zip path when valid, or None when download failed but callers
    may still use previously extracted CSV caches.
    """
    out = cache_dir or healthgait_cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / "dataset_samples.zip"
    if zip_path.is_file() and not force:
        if _zip_is_usable(zip_path):
            return zip_path
        logger.warning("Removing corrupt cached Health&Gait zip: %s", zip_path)
        zip_path.unlink(missing_ok=True)

    tmp_path = zip_path.with_suffix(".zip.part")
    last_error: Exception | None = None
    for attempt in range(1, DOWNLOAD_MAX_ATTEMPTS + 1):
        try:
            tmp_path.unlink(missing_ok=True)
            urlretrieve(ZENODO_SAMPLES_URL, tmp_path)
            size = tmp_path.stat().st_size
            if size < int(EXPECTED_SAMPLES_ZIP_BYTES * 0.95):
                raise ContentTooShortError(
                    f"retrieval incomplete: got only {size} bytes",
                    tmp_path,
                )
            if not zipfile.is_zipfile(tmp_path):
                raise zipfile.BadZipFile("downloaded file is not a zip archive")
            tmp_path.replace(zip_path)
            return zip_path
        except (ContentTooShortError, URLError, OSError, zipfile.BadZipFile) as exc:
            last_error = exc
            logger.warning(
                "Health&Gait samples download attempt %s/%s failed: %s",
                attempt,
                DOWNLOAD_MAX_ATTEMPTS,
                exc,
            )
            tmp_path.unlink(missing_ok=True)
            if attempt < DOWNLOAD_MAX_ATTEMPTS:
                time.sleep(2.0 * attempt)

    if has_cached_healthgait_csvs():
        logger.warning(
            "Health&Gait zip download failed (%s); using cached extracted CSVs.",
            last_error,
        )
        return None
    if last_error is not None:
        raise last_error
    return None


def extract_healthgait_csvs(
    zip_path: Path,
    *,
    extract_dir: Path | None = None,
) -> dict[str, Path]:
    """Extract CSV metadata files from dataset_samples.zip."""
    target = extract_dir or (zip_path.parent / "dataset_samples_csv")
    target.mkdir(parents=True, exist_ok=True)
    found: dict[str, Path] = {}
    if not zipfile.is_zipfile(zip_path):
        # Fall back to previously extracted CSVs (e.g. partial download in progress).
        for path in sorted(target.glob("*.csv")):
            key = path.stem.replace("dataset_samples_", "")
            found[key] = path
        return found
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.endswith(".csv"):
                continue
            data = zf.read(name)
            local = target / Path(name).name
            local.write_bytes(data)
            key = local.stem.replace("dataset_samples_", "")
            found[key] = local
    return found


def _float_or_none(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def rank_healthgait_trials(
    gait_parameters_csv: Path,
    *,
    gait_speed: str,
    target_velocity: float | None = None,
) -> list[HealthGaitTrialCandidate]:
    """
    Rank participants for a gait-speed condition using sensor gait_parameters.csv.

    gait_speed: ``UGS`` (normal) or ``FGS`` (performance / fast gait).
    """
    speed = gait_speed.upper()
    vel_key = f"Velocity_{speed}"
    cadence_key = f"Cadence_{speed}"
    step_key = f"Step_{speed}"
    stride_key = f"Stride_{speed}"

    rows = _read_csv_rows(gait_parameters_csv)
    candidates: list[HealthGaitTrialCandidate] = []

    velocities = [
        _float_or_none(r.get(vel_key))
        for r in rows
        if _float_or_none(r.get(vel_key)) is not None
    ]
    median_velocity = (
        sorted(velocities)[len(velocities) // 2] if velocities else None
    )

    for row in rows:
        pid = (row.get("ID") or "").strip()
        if not pid:
            continue
        velocity = _float_or_none(row.get(vel_key))
        cadence = _float_or_none(row.get(cadence_key))
        step = _float_or_none(row.get(step_key))
        stride = _float_or_none(row.get(stride_key))
        notes: list[str] = []
        score = 0.0

        if velocity is None:
            notes.append(f"missing {vel_key}")
            score = -1.0
        else:
            score += velocity
            if speed == "UGS" and median_velocity is not None:
                # Prefer moderate usual gait speed near cohort median.
                score -= abs(velocity - median_velocity) * 0.5
                notes.append("ranked near cohort median usual gait speed")
            if speed == "FGS":
                notes.append("fast gait speed (FGS) trial — performance demo candidate")

        if cadence is not None and 90 <= cadence <= 140:
            score += 0.05
        if step is not None and stride is not None and stride > step:
            score += 0.02

        candidates.append(
            HealthGaitTrialCandidate(
                participant_id=pid,
                gait_speed=speed,
                velocity_m_s=velocity,
                cadence_spm=cadence,
                step_cm=step,
                stride_cm=stride,
                selection_score=score,
                notes=notes,
            )
        )

    candidates.sort(key=lambda c: c.selection_score, reverse=True)
    return candidates


def inspect_healthgait_samples(
    *,
    cache_dir: Path | None = None,
    download: bool = True,
) -> dict[str, Any]:
    """Inspect dataset_samples.zip and summarize availability for demo selection."""
    cache = cache_dir or healthgait_cache_dir()
    zip_path: Path | None = cache / "dataset_samples.zip"
    download_warning: str | None = None

    if download and not has_cached_healthgait_csvs():
        try:
            zip_path = download_healthgait_samples(cache_dir=cache, force=False)
        except Exception as exc:
            download_warning = f"dataset_samples.zip download failed: {exc}"
            zip_path = cache / "dataset_samples.zip"
    elif download and has_cached_healthgait_csvs():
        zip_path = (
            download_healthgait_samples(cache_dir=cache, force=False)
            if _zip_is_usable(cache / "dataset_samples.zip")
            else None
        )
        if zip_path is None:
            download_warning = (
                "Using cached extracted Health&Gait CSVs; zip download skipped or unavailable."
            )
    elif not download:
        zip_path = cache / "dataset_samples.zip" if (cache / "dataset_samples.zip").is_file() else None

    if zip_path is None and not has_cached_healthgait_csvs():
        return {
            "error": download_warning or "dataset_samples.zip not found",
            "download_url": ZENODO_SAMPLES_URL,
        }

    zip_valid = zip_path is not None and _zip_is_usable(zip_path)
    if zip_valid and zip_path is not None:
        csv_paths = extract_healthgait_csvs(zip_path)
    else:
        csv_paths = {
            p.stem.replace("dataset_samples_", ""): p
            for p in healthgait_samples_csv_dir().glob("*.csv")
        }
    mp4_files: list[str] = []
    silhouette_dirs: list[str] = []
    if zip_valid and zip_path is not None:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            mp4_files = [n for n in names if n.lower().endswith((".mp4", ".avi", ".mov"))]
            silhouette_dirs = sorted(
                {n.split("/")[2] for n in names if "/silhouette/" in n and n.count("/") >= 3}
            )
    else:
        names = []

    gait_params = csv_paths.get("gait_parameters")
    normal_ranked: list[HealthGaitTrialCandidate] = []
    performance_ranked: list[HealthGaitTrialCandidate] = []
    if gait_params is not None:
        normal_ranked = rank_healthgait_trials(gait_params, gait_speed="UGS")
        performance_ranked = rank_healthgait_trials(gait_params, gait_speed="FGS")

    result = {
        "source": "Health&Gait",
        "source_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "samples_zip": str(zip_path) if zip_path is not None else None,
        "samples_zip_bytes": zip_path.stat().st_size if zip_valid and zip_path else None,
        "raw_video_in_samples": len(mp4_files) > 0,
        "raw_video_files_in_samples": mp4_files[:20],
        "sample_participants_in_silhouettes": silhouette_dirs,
        "csv_files": {k: str(v) for k, v in csv_paths.items()},
        "structure_notes": SAMPLE_STRUCTURE_DOC,
        "public_release_limitation": (
            "Scientific Data (2025) states raw RGB walking videos are not provided "
            "in the public Health&Gait release. Silhouette/pose/optical-flow exports "
            "are available; StableWalk GUI demos still require validated MP4 candidates."
        ),
        "normal_candidates_ugs": [c.to_dict() for c in normal_ranked[:10]],
        "performance_candidates_fgs": [c.to_dict() for c in performance_ranked[:10]],
        "recommended_workflow": [
            "1. Download dataset_samples.zip (done or cached).",
            "2. Download full CSV metadata from the 26.8 GB archive when selecting finals.",
            "3. Rank UGS trials for Normal and FGS trials for Performance.",
            "4. Obtain RGB MP4 via controlled re-capture or authorized source.",
            "5. Run validate_demo_candidate.py before installing demo files.",
        ],
    }
    if download_warning:
        result["download_warning"] = download_warning
    return result


def select_healthgait_demos(
    gait_parameters_csv: Path,
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """Select Normal (UGS) and Performance (FGS) trial candidates from metadata."""
    normal = rank_healthgait_trials(gait_parameters_csv, gait_speed="UGS")
    performance = rank_healthgait_trials(gait_parameters_csv, gait_speed="FGS")
    return {
        "source": "Health&Gait",
        "source_url": f"https://zenodo.org/records/{ZENODO_RECORD}",
        "normal_category": "Normal",
        "performance_category": "Performance",
        "normal_trials_ugs": [c.to_dict() for c in normal[:top_n]],
        "performance_trials_fgs": [c.to_dict() for c in performance[:top_n]],
        "selection_rules": {
            "normal": "UGS (usual gait speed), moderate velocity near cohort median",
            "performance": "FGS (fast gait speed), highest validated Velocity_FGS",
        },
        "video_acquisition_note": (
            "Health&Gait public Zenodo release does not ship raw MP4. "
            "Use metadata to choose participant/speed, then provide a validated MP4."
        ),
    }


def write_healthgait_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


__all__ = [
    "HealthGaitTrialCandidate",
    "ZENODO_SAMPLES_URL",
    "download_healthgait_samples",
    "extract_healthgait_csvs",
    "has_cached_healthgait_csvs",
    "inspect_healthgait_samples",
    "rank_healthgait_trials",
    "select_healthgait_demos",
    "write_healthgait_report",
]
