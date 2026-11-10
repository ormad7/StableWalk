#!/usr/bin/env python3
"""
Download a small set of GAVD RGB walking candidates from documented YouTube URLs.

GAVD ships annotations only. This script:
  1. Reads cached GAVD clinical CSVs
  2. Picks Parkinson (pathological) and normal sequences
  3. Retrieves those public YouTube videos with yt-dlp
  4. Trims to the annotated continuous walking span without changing speed

Dementia (Dem@Care) and Health&Gait FGS RGB are not public; this script does not
substitute simulated gait or a different diagnosis.

Usage:
  python scripts/download_gavd_rgb_candidates.py --dry-run
  python scripts/download_gavd_rgb_candidates.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stablewalk import config
from stablewalk.demo.gavd_selection import (
    GavdSelectionThresholds,
    _iter_annotation_rows,
    _normalize_view,
    _parse_dict_field,
    _support_device_in_text,
    _view_ok,
    download_gavd_annotations,
    gavd_cache_dir,
)

PREFERRED_VIEWS = frozenset({"left side", "right side", "front", "back"})
TARGET_MIN_S = 15.0
TARGET_MAX_S = 30.0
ASSUMED_FPS_FOR_RANKING = 30.0


def _ffmpeg_exe() -> str | None:
    which = shutil.which("ffmpeg")
    if which:
        return which
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _yt_dlp_exe() -> str | None:
    return shutil.which("yt-dlp")


def _collect_sequences(paths: list[Path]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _iter_annotation_rows(paths):
        grouped[row["seq"]].append(row)

    cfg = GavdSelectionThresholds()
    out: list[dict[str, Any]] = []
    for seq, rows in grouped.items():
        first = rows[0]
        gait_pat = (first.get("gait_pat") or "").strip()
        dataset = (first.get("dataset") or "").strip()
        youtube_id = (first.get("id") or "").strip()
        url = (first.get("url") or "").strip()
        if not youtube_id or not url:
            continue
        if _support_device_in_text(gait_pat, dataset, seq):
            continue

        frames = []
        for row in rows:
            try:
                frames.append(int(row.get("frame_num") or 0))
            except ValueError:
                continue
        if not frames:
            continue
        frame_start, frame_end = min(frames), max(frames)
        frame_count = frame_end - frame_start + 1
        if frame_count < cfg.min_seq_frames:
            continue

        views = [_normalize_view(r.get("cam_view", "")) for r in rows]
        view_counts: dict[str, int] = defaultdict(int)
        for v in views:
            if v:
                view_counts[v] += 1
        if not view_counts:
            continue
        dominant_view, dominant_count = max(view_counts.items(), key=lambda x: x[1])
        dominant_fraction = dominant_count / max(len(rows), 1)
        if dominant_fraction < cfg.min_dominant_view_fraction:
            continue
        if not _view_ok(dominant_view) or dominant_view not in PREFERRED_VIEWS:
            continue

        bbox_heights: list[float] = []
        bbox_bottoms: list[float] = []
        vid_h = vid_w = 0
        for row in rows:
            bbox = _parse_dict_field(row.get("bbox", ""))
            vid = _parse_dict_field(row.get("vid_info", ""))
            vid_h = int(vid.get("height") or vid_h or 0)
            vid_w = int(vid.get("width") or vid_w or 0)
            height = float(bbox.get("height") or 0.0)
            top = float(bbox.get("top") or 0.0)
            if vid_h > 0 and height > 0:
                bbox_heights.append(min(height / vid_h, 1.0))
                bbox_bottoms.append(min((top + height) / vid_h, 1.0))
        if not bbox_heights:
            continue
        mean_h = sum(bbox_heights) / len(bbox_heights)
        mean_bottom = sum(bbox_bottoms) / len(bbox_bottoms)
        if mean_h < cfg.min_bbox_height_ratio:
            continue

        est_s = frame_count / ASSUMED_FPS_FOR_RANKING
        duration_score = 1.0
        if TARGET_MIN_S <= est_s <= TARGET_MAX_S:
            duration_score = 2.0
        elif est_s > TARGET_MAX_S:
            duration_score = 1.2
        elif est_s >= 10.0:
            duration_score = 0.8
        else:
            duration_score = 0.3

        side_bonus = 0.25 if "side" in dominant_view else 0.0
        score = mean_h * 2.0 + mean_bottom + duration_score + side_bonus + dominant_fraction
        out.append(
            {
                "seq": seq,
                "youtube_id": youtube_id,
                "url": url,
                "dataset": dataset,
                "gait_pat": gait_pat,
                "cam_view": dominant_view,
                "dominant_view_fraction": round(dominant_fraction, 3),
                "frame_start": frame_start,
                "frame_end": frame_end,
                "frame_count": frame_count,
                "estimated_duration_s_at_30fps": round(est_s, 2),
                "mean_bbox_height_ratio": round(mean_h, 3),
                "mean_bbox_bottom_ratio": round(mean_bottom, 3),
                "video_height": vid_h,
                "video_width": vid_w,
                "metadata_score": round(score, 3),
            }
        )
    out.sort(key=lambda c: -c["metadata_score"])
    return out


def _pick(
    sequences: list[dict[str, Any]],
    gait_pat: str,
    n: int,
    *,
    dataset_contains: str | None = None,
) -> list[dict[str, Any]]:
    want = gait_pat.strip().lower()
    needle = (dataset_contains or "").strip().lower()
    picked: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in sequences:
        if (item["gait_pat"] or "").strip().lower() != want:
            continue
        if needle and (item["dataset"] or "").strip().lower() != needle:
            continue
        yid = item["youtube_id"]
        if yid in seen_ids:
            continue
        seen_ids.add(yid)
        picked.append(item)
        if len(picked) >= n:
            break
    return picked


def _run(
    cmd: list[str],
    *,
    timeout: int,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
    )


def _yt_env(ffmpeg: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(Path(ffmpeg).parent) + os.pathsep + env.get("PATH", "")
    return env


def _node_runtime_arg() -> list[str]:
    node = shutil.which("node")
    if not node:
        return []
    return ["--js-runtimes", f"node:{node}"]


def _yt_meta(url: str, yt_dlp: str, env: dict[str, str]) -> dict[str, Any]:
    cmd = [yt_dlp, "-j", "--no-playlist", "--skip-download", *(_node_runtime_arg()), url]
    proc = _run(cmd, timeout=90, env=env)
    if proc.returncode != 0 or not (proc.stdout or "").strip():
        return {"ok": False, "error": (proc.stderr or proc.stdout or "")[-800:]}
    try:
        data = json.loads(proc.stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "error": "yt-dlp metadata was not JSON"}
    fps = float(data.get("fps") or 0.0)
    duration = float(data.get("duration") or 0.0)
    return {
        "ok": True,
        "fps": fps,
        "duration": duration,
        "width": data.get("width"),
        "height": data.get("height"),
        "title": data.get("title"),
        "availability": data.get("availability"),
    }


def _section_arg(start_frame: int, frame_count: int, fps: float, video_duration: float) -> str | None:
    if fps <= 1:
        return None
    start_s = max(0.0, start_frame / fps - 0.5)
    end_s = (start_frame + frame_count) / fps + 0.5
    if TARGET_MAX_S > 0:
        end_s = min(end_s, start_s + TARGET_MAX_S + 1.0)
    if video_duration > 0:
        end_s = min(end_s, video_duration)
    if end_s <= start_s + 1.0:
        return None
    return f"*{start_s:.3f}-{end_s:.3f}"


def _collect_produced(dest: Path) -> list[Path]:
    return [
        p
        for p in dest.parent.glob(dest.stem + ".*")
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}
    ]


def _download_youtube(
    url: str,
    dest: Path,
    ffmpeg: str,
    yt_dlp: str,
    *,
    start_frame: int,
    frame_count: int,
) -> tuple[bool, str, dict[str, Any]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = _yt_env(ffmpeg)
    meta = _yt_meta(url, yt_dlp, env)
    if not meta.get("ok"):
        return False, meta.get("error") or "yt-dlp metadata failed", meta

    title = (meta.get("title") or "").lower()
    if any(token in title for token in ("slo mo", "slow mo", "slow-mo", "slow motion")):
        return False, f"rejected: source title documents altered speed ({meta.get('title')})", meta
    if any(token in title for token in ("incremental", "increasing speed", "0.5 m/s")):
        return False, f"rejected: source title documents incremental-speed protocol ({meta.get('title')})", meta

    duration = float(meta.get("duration") or 0.0)
    section = None
    if duration > 90:
        section = _section_arg(
            start_frame,
            frame_count,
            float(meta.get("fps") or 0.0),
            duration,
        )
    out_tmpl = str(dest.with_suffix(".%(ext)s"))
    attempts: list[tuple[str | None, str | None]] = [
        (None, section),
        (None, None),
        ("android,ios,web", None),
    ]
    last_err = ""
    for client, sect in attempts:
        cmd = [
            yt_dlp,
            "--no-playlist",
            "--no-mtime",
            "--ffmpeg-location",
            str(Path(ffmpeg).parent),
            *(_node_runtime_arg()),
            "-f",
            "bv*[height<=720][ext=mp4]/b[height<=720][ext=mp4]/bv*[ext=mp4]/b[ext=mp4]/bv*[height<=720]/b",
            "-o",
            out_tmpl,
            "--merge-output-format",
            "mp4",
        ]
        if sect:
            cmd.extend(["--download-sections", sect, "--force-keyframes-at-cuts"])
        if client:
            cmd.extend(["--extractor-args", f"youtube:player_client={client}"])
        cmd.append(url)
        proc = _run(cmd, timeout=360, env=env)
        produced = _collect_produced(dest)
        if proc.returncode == 0 and produced:
            src = produced[0]
            if src != dest:
                dest.unlink(missing_ok=True)
                src.replace(dest)
            detail = "downloaded via yt-dlp"
            if sect:
                detail += f" section={sect}"
            if client:
                detail += f" client={client}"
            return True, detail, meta
        last_err = (proc.stderr or proc.stdout or "").strip()[-800:]
        for leftover in produced:
            leftover.unlink(missing_ok=True)
    return False, last_err or "yt-dlp failed", meta


def _probe(path: Path) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    info = {
        "opened": bool(cap.isOpened()),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
    }
    cap.release()
    fps = info["fps"] if info["fps"] > 1 else 30.0
    info["duration_s"] = round(info["frame_count"] / fps, 3) if info["frame_count"] else 0.0
    return info


def _trim_segment(
    source: Path,
    dest: Path,
    *,
    start_frame: int,
    frame_count: int,
    ffmpeg: str,
    source_fps: float,
) -> tuple[bool, str]:
    fps = source_fps if source_fps > 1 else 30.0
    start_s = max(0.0, start_frame / fps)
    duration_s = frame_count / fps
    if duration_s > TARGET_MAX_S:
        duration_s = TARGET_MAX_S
        frame_count = int(round(duration_s * fps))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-an",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    proc = _run(cmd, timeout=180)
    if proc.returncode != 0 or not dest.is_file() or dest.stat().st_size < 10_000:
        err = (proc.stderr or "")[-500:]
        return False, err or "ffmpeg trim failed"
    return True, (
        f"trimmed start_frame={start_frame} frames={frame_count} "
        f"start_s={start_s:.3f} duration_s={duration_s:.3f} source_fps={fps:.3f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download GAVD RGB walking candidates")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--per-class", type=int, default=2, help="Videos per accessible class")
    parser.add_argument(
        "--exclude-ids",
        default="",
        help="Comma-separated YouTube IDs to skip (already downloaded or rejected)",
    )
    parser.add_argument(
        "--only-ids",
        default="",
        help="Comma-separated YouTube IDs to download (overrides ranking)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=config.DEMO_VIDEOS_DIR / "candidates" / "gavd_rgb",
    )
    args = parser.parse_args()

    paths = download_gavd_annotations()
    sequences = _collect_sequences(paths)
    exclude_ids = {
        tok.strip()
        for tok in (args.exclude_ids or "").split(",")
        if tok.strip()
    }
    if exclude_ids:
        sequences = [s for s in sequences if s["youtube_id"] not in exclude_ids]
    only_ids = {
        tok.strip()
        for tok in (getattr(args, "only_ids", "") or "").split(",")
        if tok.strip()
    }
    fallback_n = max(args.per_class * 3, 6)
    if only_ids:
        chosen = [s for s in sequences if s["youtube_id"] in only_ids]
        seen: set[str] = set()
        parkinson = []
        normal = []
        for item in chosen:
            yid = item["youtube_id"]
            if yid in seen:
                continue
            pat = (item["gait_pat"] or "").strip().lower()
            if pat == "parkinsons":
                seen.add(yid)
                parkinson.append(item)
            elif pat == "normal":
                seen.add(yid)
                normal.append(item)
    else:
        parkinson = _pick(sequences, "parkinsons", fallback_n)
        normal = _pick(sequences, "normal", fallback_n, dataset_contains="normal gait")
    if len(normal) < args.per_class:
        extra = _pick(sequences, "normal", fallback_n)
        seen = {item["youtube_id"] for item in normal}
        for item in extra:
            if item["youtube_id"] not in seen:
                item = dict(item)
                item["notes"] = list(item.get("notes") or []) + [
                    "gait_pat=normal but dataset is not 'Normal Gait'; kept as fallback only"
                ]
                normal.append(item)
                seen.add(item["youtube_id"])
            if len(normal) >= fallback_n:
                break

    blockers = {
        "dementia": {
            "requested_label": "documented dementia gait",
            "accessible": False,
            "blocker": (
                "Dem@Care Ds1/Ds2/Ds6/Ds8 RGB-D videos are not publicly downloadable. "
                "Access requires an institutional-email request to info@demcare.eu, "
                "agreement to non-commercial Terms of Use, no redistribution, "
                "no re-identification, and no showing of patients' faces. "
                "GAVD annotations contain no dementia/Alzheimer/MCI gait_pat."
            ),
            "substituted": False,
        },
        "fast_walking": {
            "requested_label": "fast/sport walking, not running",
            "accessible": False,
            "blocker": (
                "Health&Gait FGS is the matching source metadata (usual vs fast gait speed) "
                "but the public Zenodo release (10.5281/zenodo.14039922) does not include "
                "raw RGB video. GAVD has no fast/sport-walking class; gait_pat=exercise "
                "is a different label and was not used. Simulated WeightGait clips were "
                "not downloaded as a substitute."
            ),
            "substituted": False,
        },
    }

    report: dict[str, Any] = {
        "source": "GAVD",
        "source_url": "https://github.com/Rahmyyy/GAVD",
        "download_method": "yt-dlp retrieval of public YouTube URLs listed in GAVD CSVs",
        "annotation_dir": str(gavd_cache_dir()),
        "blockers": blockers,
        "selected": {
            "parkinsons": parkinson,
            "normal": normal,
        },
        "downloads": [],
    }

    print("=== GAVD RGB candidate download ===")
    print("Dementia: BLOCKED (Dem@Care access request; not substituted)")
    print("Fast walking RGB: BLOCKED (Health&Gait FGS has no public RGB; not substituted)")
    print(f"Parkinson sequences selected: {len(parkinson)}")
    for item in parkinson:
        print(
            f"  {item['youtube_id']} {item['cam_view']} "
            f"frames {item['frame_start']}-{item['frame_end']} "
            f"bbox_h={item['mean_bbox_height_ratio']} {item['url']}"
        )
    print(f"Normal sequences selected: {len(normal)}")
    for item in normal:
        print(
            f"  {item['youtube_id']} {item['cam_view']} "
            f"frames {item['frame_start']}-{item['frame_end']} "
            f"bbox_h={item['mean_bbox_height_ratio']} {item['url']}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "download_report.json"
    if args.dry_run:
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Dry run report: {report_path}")
        return 0

    ffmpeg = _ffmpeg_exe()
    yt_dlp = _yt_dlp_exe()
    if not ffmpeg:
        print("ffmpeg not found (system or imageio_ffmpeg)")
        return 1
    if not yt_dlp:
        print("yt-dlp not found on PATH")
        return 1

    raw_dir = args.out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = [("parkinsons", item) for item in parkinson] + [("normal", item) for item in normal]
    success_by_label = {"parkinsons": 0, "normal": 0}
    for label, item in jobs:
        if success_by_label[label] >= args.per_class:
            continue
        yid = item["youtube_id"]
        raw = raw_dir / f"{yid}.mp4"
        trim_name = f"gavd_{label}_{yid}_{item['seq'][:10]}.mp4"
        trimmed = args.out_dir / trim_name
        rec: dict[str, Any] = {
            "category": "abnormal" if label == "parkinsons" else "normal",
            "documented_label": item["gait_pat"],
            "not_dementia": label == "parkinsons",
            **item,
            "raw_path": str(raw),
            "trimmed_path": str(trimmed),
        }
        print(f"\nDownloading {label} {yid} ...")
        ok, detail, meta = _download_youtube(
            item["url"],
            raw,
            ffmpeg,
            yt_dlp,
            start_frame=int(item["frame_start"]),
            frame_count=int(item["frame_count"]),
        )
        rec["download_ok"] = ok
        rec["download_detail"] = detail
        rec["youtube_meta"] = {
            k: meta.get(k) for k in ("fps", "duration", "width", "height", "title", "ok")
        }
        if not ok:
            print(f"  FAILED: {detail[:300]}")
            report["downloads"].append(rec)
            continue
        probe = _probe(raw)
        rec["raw_probe"] = probe
        print(
            f"  raw {probe.get('width')}x{probe.get('height')} "
            f"{probe.get('fps'):.3f} fps {probe.get('duration_s'):.1f}s"
        )
        # Section downloads already cover the annotated span; keep original
        # timeline. If the file is still longer than 30 s, trim using source FPS.
        source_fps = float(meta.get("fps") or probe.get("fps") or 30.0)
        section_used = "section=" in detail
        if section_used and probe.get("duration_s", 0) <= TARGET_MAX_S + 2:
            shutil.copy2(raw, trimmed)
            trim_ok, trim_detail = True, (
                "kept yt-dlp section (original fps/speed; no extra trim)"
            )
        else:
            trim_start = 0 if section_used else int(item["frame_start"])
            trim_ok, trim_detail = _trim_segment(
                raw,
                trimmed,
                start_frame=trim_start,
                frame_count=int(item["frame_count"]),
                ffmpeg=ffmpeg,
                source_fps=source_fps,
            )
        rec["trim_ok"] = trim_ok
        rec["trim_detail"] = trim_detail
        if trim_ok:
            rec["trimmed_probe"] = _probe(trimmed)
            success_by_label[label] += 1
            print(f"  trimmed -> {trimmed.name} {trim_detail}")
        else:
            print(f"  trim FAILED: {trim_detail[:300]}")
        report["downloads"].append(rec)

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0 if any(d.get("trim_ok") for d in report["downloads"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
