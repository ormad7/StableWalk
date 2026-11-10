"""Local research gait-video catalog.

Clips are **user-provided local files**. This module copies them into
``data/research_dataset/`` and stores metadata. It never downloads videos
and never redistributes copyrighted clinical recordings.
"""

from __future__ import annotations

import json
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from stablewalk import config
from stablewalk.research.categories import (
    AUTHENTICITY_VALUES,
    CAMERA_VIEW_VALUES,
    CLINICAL_POLICY,
    GAIT_CATEGORY_LABELS,
    display_clinical_diagnosis,
    normalize_authenticity,
    normalize_category,
)
from stablewalk.research.probe import is_remote_url, probe_local_video

CATALOG_SCHEMA = "stablewalk-research-dataset"
CATALOG_VERSION = "1.0"

_ID_SAFE = re.compile(r"[^a-zA-Z0-9._-]+")


@dataclass
class ClipMetadata:
    video_id: str
    source: str = "user_import"
    original_url: str = ""
    original_filename: str = ""
    local_path: str = ""
    gait_category: str = "unknown_or_unverified"
    camera_view: str = "unknown"
    fps: float | None = None
    width: int | None = None
    height: int | None = None
    frame_count: int | None = None
    duration_s: float | None = None
    clinical_diagnosis: str | None = None
    clinical_verified: bool = False
    clinical_source: str = ""
    license: str = "user_provided_unknown"
    usage_restrictions: str = ""
    authenticity: str = "authentic"
    notes: str = ""
    added_at_utc: str = ""

    @property
    def resolved_path(self) -> Path | None:
        raw = str(self.local_path or "").strip()
        if not raw:
            return None
        path = Path(raw)
        if not path.is_absolute():
            path = config.RESEARCH_DATASET_DIR / path
        return path

    @property
    def clinical_display(self) -> str | None:
        return display_clinical_diagnosis(
            diagnosis=self.clinical_diagnosis,
            verified=self.clinical_verified,
            source=self.clinical_source,
            category=self.gait_category,
        )

    @property
    def category_label(self) -> str:
        return GAIT_CATEGORY_LABELS.get(self.gait_category, self.gait_category)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["clinical_display"] = self.clinical_display
        data["category_label"] = self.category_label
        data["resolution"] = (
            f"{self.width}×{self.height}"
            if self.width and self.height
            else None
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClipMetadata:
        allowed = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in allowed}
        kwargs.setdefault("video_id", "unknown")
        clip = cls(**kwargs)
        clip.gait_category = normalize_category(clip.gait_category)
        clip.authenticity = normalize_authenticity(clip.authenticity)
        view = str(clip.camera_view or "unknown").strip().lower().replace(" ", "_")
        clip.camera_view = view if view in CAMERA_VIEW_VALUES else "unknown"
        if clip.authenticity not in AUTHENTICITY_VALUES:
            clip.authenticity = "authentic"
        return clip


@dataclass
class ResearchCatalog:
    clips: list[ClipMetadata] = field(default_factory=list)
    schema: str = CATALOG_SCHEMA
    schema_version: str = CATALOG_VERSION
    policy: str = CLINICAL_POLICY
    auto_download: bool = False

    def get(self, video_id: str) -> ClipMetadata | None:
        for clip in self.clips:
            if clip.video_id == video_id:
                return clip
        return None

    def ids(self) -> list[str]:
        return [c.video_id for c in self.clips]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "auto_download": False,
            "policy": self.policy,
            "clips": [c.to_dict() for c in self.clips],
        }


def _ensure_dirs() -> None:
    config.RESEARCH_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    config.RESEARCH_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)


def catalog_path() -> Path:
    return config.RESEARCH_CATALOG_PATH


def load_catalog(path: Path | None = None) -> ResearchCatalog:
    dest = path or catalog_path()
    if not dest.is_file():
        return ResearchCatalog()
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ResearchCatalog()
    clips = [ClipMetadata.from_dict(row) for row in payload.get("clips") or []]
    return ResearchCatalog(
        clips=clips,
        schema=str(payload.get("schema") or CATALOG_SCHEMA),
        schema_version=str(payload.get("schema_version") or CATALOG_VERSION),
        policy=str(payload.get("policy") or CLINICAL_POLICY),
        auto_download=False,
    )


def save_catalog(catalog: ResearchCatalog, path: Path | None = None) -> Path:
    _ensure_dirs()
    dest = path or catalog_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
    return dest


def _make_video_id(filename: str) -> str:
    stem = _ID_SAFE.sub("_", Path(filename).stem).strip("._") or "clip"
    stem = stem[:40]
    return f"{stem}_{uuid.uuid4().hex[:8]}"


def import_local_video(
    source_path: str | Path,
    *,
    gait_category: str = "unknown_or_unverified",
    camera_view: str = "unknown",
    authenticity: str = "authentic",
    original_url: str = "",
    clinical_diagnosis: str | None = None,
    clinical_verified: bool = False,
    clinical_source: str = "",
    license: str = "user_provided_unknown",
    usage_restrictions: str = "",
    notes: str = "",
    source: str = "user_import",
    copy_into_dataset: bool = True,
    catalog: ResearchCatalog | None = None,
) -> ClipMetadata:
    """Register a local walking video. Remote URLs are rejected."""
    if is_remote_url(source_path):
        raise ValueError(
            "Automatic download of remote videos is disabled. Save the file "
            "locally (with an appropriate license) and import that path."
        )
    src = Path(source_path).expanduser()
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {src}")
    if src.suffix.lower() not in config.VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported video type: {src.suffix}")

    _ensure_dirs()
    cat = catalog if catalog is not None else load_catalog()
    video_id = _make_video_id(src.name)
    dest = config.RESEARCH_VIDEOS_DIR / f"{video_id}{src.suffix.lower()}"
    if copy_into_dataset:
        shutil.copy2(src, dest)
        stored = dest
        rel = stored.relative_to(config.RESEARCH_DATASET_DIR).as_posix()
    else:
        stored = src.resolve()
        rel = str(stored)

    probe = probe_local_video(stored)
    clip = ClipMetadata(
        video_id=video_id,
        source=source,
        original_url=str(original_url or "").strip(),
        original_filename=src.name,
        local_path=rel,
        gait_category=normalize_category(gait_category),
        camera_view=camera_view if camera_view in CAMERA_VIEW_VALUES else "unknown",
        fps=probe.get("fps"),
        width=probe.get("width"),
        height=probe.get("height"),
        frame_count=probe.get("frame_count"),
        duration_s=probe.get("duration_s"),
        clinical_diagnosis=(str(clinical_diagnosis).strip() or None)
        if clinical_diagnosis
        else None,
        clinical_verified=bool(clinical_verified) and bool(str(clinical_source or "").strip()),
        clinical_source=str(clinical_source or "").strip(),
        license=str(license or "user_provided_unknown"),
        usage_restrictions=str(usage_restrictions or ""),
        authenticity=normalize_authenticity(authenticity),
        notes=str(notes or ""),
        added_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    cat.clips.append(clip)
    save_catalog(cat)
    return clip


def iter_clips(catalog: ResearchCatalog | None = None) -> Iterable[ClipMetadata]:
    cat = catalog if catalog is not None else load_catalog()
    return list(cat.clips)
