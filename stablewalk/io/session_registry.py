"""Recent sessions registry and autosave preferences for StableWalk."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from stablewalk import config

logger = logging.getLogger(__name__)

REGISTRY_SCHEMA = "stablewalk-session-registry"
REGISTRY_VERSION = "1.0"
DEFAULT_AUTOSAVE_INTERVAL_S = 120
MAX_RECENT = 12


def registry_path() -> Path:
    return config.SESSION_EXPORT_DIR / "session_registry.json"


def autosave_root() -> Path:
    return config.SESSION_EXPORT_DIR / "autosave"


@dataclass
class RecentSessionEntry:
    path: str
    display_name: str
    video_source: str = ""
    saved_at: str = ""
    frame_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "display_name": self.display_name,
            "video_source": self.video_source,
            "saved_at": self.saved_at,
            "frame_count": self.frame_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RecentSessionEntry:
        return cls(
            path=str(data.get("path") or ""),
            display_name=str(data.get("display_name") or Path(str(data.get("path") or "")).name),
            video_source=str(data.get("video_source") or ""),
            saved_at=str(data.get("saved_at") or ""),
            frame_count=int(data.get("frame_count") or 0),
        )


def _default_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "autosave_enabled": True,
        "autosave_interval_s": DEFAULT_AUTOSAVE_INTERVAL_S,
        "recent": [],
    }


def load_registry() -> dict[str, Any]:
    path = registry_path()
    if not path.is_file():
        return _default_registry()
    try:
        with path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            return _default_registry()
        base = _default_registry()
        base.update(data)
        if not isinstance(base.get("recent"), list):
            base["recent"] = []
        return base
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read session registry: %s", exc)
        return _default_registry()


def save_registry(data: dict[str, Any]) -> None:
    config.ensure_output_dirs()
    path = registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["schema"] = REGISTRY_SCHEMA
    payload["version"] = REGISTRY_VERSION
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def list_recent_sessions(*, limit: int = MAX_RECENT) -> list[RecentSessionEntry]:
    data = load_registry()
    entries: list[RecentSessionEntry] = []
    for raw in data.get("recent") or []:
        if not isinstance(raw, dict):
            continue
        entry = RecentSessionEntry.from_dict(raw)
        if not entry.path:
            continue
        if Path(entry.path).is_dir() or (Path(entry.path) / "session_metadata.json").is_file():
            entries.append(entry)
        if len(entries) >= limit:
            break
    return entries


def remember_session(
    path: str | Path,
    *,
    display_name: str | None = None,
    video_source: str = "",
    frame_count: int = 0,
) -> None:
    bundle = Path(path).expanduser().resolve()
    data = load_registry()
    recent = [
        item
        for item in (data.get("recent") or [])
        if isinstance(item, dict) and Path(str(item.get("path") or "")).resolve() != bundle
    ]
    entry = RecentSessionEntry(
        path=str(bundle),
        display_name=display_name or bundle.name,
        video_source=video_source,
        saved_at=datetime.now().isoformat(timespec="seconds"),
        frame_count=frame_count,
    )
    recent.insert(0, entry.to_dict())
    data["recent"] = recent[:MAX_RECENT]
    save_registry(data)


def is_autosave_enabled() -> bool:
    return bool(load_registry().get("autosave_enabled", True))


def set_autosave_enabled(enabled: bool) -> None:
    data = load_registry()
    data["autosave_enabled"] = bool(enabled)
    save_registry(data)


def autosave_interval_s() -> int:
    try:
        value = int(load_registry().get("autosave_interval_s") or DEFAULT_AUTOSAVE_INTERVAL_S)
    except (TypeError, ValueError):
        value = DEFAULT_AUTOSAVE_INTERVAL_S
    return max(30, min(3600, value))


def set_autosave_interval_s(seconds: int) -> None:
    data = load_registry()
    data["autosave_interval_s"] = max(30, min(3600, int(seconds)))
    save_registry(data)


__all__ = [
    "DEFAULT_AUTOSAVE_INTERVAL_S",
    "MAX_RECENT",
    "RecentSessionEntry",
    "autosave_interval_s",
    "autosave_root",
    "is_autosave_enabled",
    "list_recent_sessions",
    "load_registry",
    "remember_session",
    "registry_path",
    "save_registry",
    "set_autosave_enabled",
    "set_autosave_interval_s",
]
