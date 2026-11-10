#!/usr/bin/env python3
"""Profile StableWalk startup + runtime; write a bottleneck report.

Measures:
  - cold import / GUI construction / tab switch
  - graph, video, and 3D skeleton render
  - process RSS + CPU (process time)

Usage:
  python scripts/profile_performance.py
  python scripts/profile_performance.py --skip-gui
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
import tracemalloc
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _rss_mb() -> float:
    """Working-set RSS in MiB (Windows + best-effort fallback)."""
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        )
        if ok:
            return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource

        # Linux: ru_maxrss is KiB; macOS: bytes
        rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":
            return rss / (1024.0 * 1024.0)
        return rss / 1024.0
    except Exception:
        return 0.0


@dataclass
class Sample:
    name: str
    ms: float
    rss_delta_mb: float = 0.0
    cpu_ms: float = 0.0
    notes: str = ""


@dataclass
class ProfileReport:
    samples: list[Sample] = field(default_factory=list)
    bottlenecks: list[dict] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(
        self,
        name: str,
        ms: float,
        *,
        rss_delta_mb: float = 0.0,
        cpu_ms: float = 0.0,
        notes: str = "",
    ) -> None:
        self.samples.append(
            Sample(name=name, ms=ms, rss_delta_mb=rss_delta_mb, cpu_ms=cpu_ms, notes=notes)
        )


def _time_call(fn, *, repeats: int = 1) -> tuple[float, float, float]:
    """Return (wall_ms_median, cpu_ms_median, rss_delta_mb)."""
    walls: list[float] = []
    cpus: list[float] = []
    rss0 = _rss_mb()
    for _ in range(max(1, repeats)):
        gc.collect()
        c0 = time.process_time()
        t0 = time.perf_counter()
        fn()
        walls.append((time.perf_counter() - t0) * 1000.0)
        cpus.append((time.process_time() - c0) * 1000.0)
    rss1 = _rss_mb()
    return statistics.median(walls), statistics.median(cpus), max(0.0, rss1 - rss0)


def profile_imports(report: ProfileReport) -> None:
    phases: list[tuple[str, object]] = []

    def _imp_numpy() -> None:
        import numpy  # noqa: F401

    def _imp_mpl() -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401

    def _imp_cv2() -> None:
        import cv2  # noqa: F401

    def _imp_pil() -> None:
        import PIL.Image  # noqa: F401

    for name, fn in (
        ("import:numpy", _imp_numpy),
        ("import:matplotlib+Agg", _imp_mpl),
        ("import:cv2", _imp_cv2),
        ("import:PIL", _imp_pil),
    ):
        try:
            wall, cpu, rss = _time_call(fn, repeats=1)
            report.add(name, wall, rss_delta_mb=rss, cpu_ms=cpu)
        except Exception as exc:
            report.add(name, -1.0, notes=f"failed: {exc}")

    def _import_app() -> None:
        from stablewalk.ui.tk import app as _app  # noqa: F401

    wall, cpu, rss = _time_call(_import_app, repeats=1)
    report.add("import:stablewalk.ui.tk.app", wall, rss_delta_mb=rss, cpu_ms=cpu)


def profile_render_hotpaths(report: ProfileReport) -> None:
    import numpy as np
    from matplotlib.figure import Figure

    from stablewalk.models.gait_motion import JointSample, SkeletonSnapshot, Vec3
    from stablewalk.models.joint_registry import ROOT_JOINT_ID
    from stablewalk.models.pose_data import Keypoint
    from stablewalk.pose.video_overlay_compose import (
        VideoOverlayLayers,
        compose_video_skeleton_overlay,
    )
    from stablewalk.ui.viewers.gait_skeleton_renderer import draw_gait_skeleton

    def make_snap(dx: float = 0.0) -> SkeletonSnapshot:
        pts = {
            "head": (0, 1.65, 0),
            "left_shoulder": (-0.2, 1.4, 0),
            "right_shoulder": (0.2, 1.4, 0),
            "left_elbow": (-0.35, 1.1, 0),
            "right_elbow": (0.35, 1.1, 0),
            "left_wrist": (-0.4, 0.85, 0),
            "right_wrist": (0.4, 0.85, 0),
            "left_hip": (-0.12, 0.95, 0),
            "right_hip": (0.12, 0.95, 0),
            "left_knee": (-0.12, 0.5, 0),
            "right_knee": (0.12, 0.5, 0),
            "left_ankle": (-0.12, 0.05, 0),
            "right_ankle": (0.12, 0.05, 0),
            "left_heel": (-0.14, 0.02, 0),
            "right_heel": (0.14, 0.02, 0),
            "left_toe": (-0.10, 0.02, 0.05),
            "right_toe": (0.10, 0.02, 0.05),
            "spine": (0, 1.15, 0),
            "neck": (0, 1.45, 0),
            ROOT_JOINT_ID: (0, 0.95, 0),
        }
        joints = {
            k: JointSample(joint_id=k, position=Vec3(x + dx, y, z))
            for k, (x, y, z) in pts.items()
        }
        return SkeletonSnapshot(frame_index=0, time_s=0.0, joints=joints)

    fig = Figure(figsize=(4, 4), dpi=80)
    ax = fig.add_subplot(111)
    snap = make_snap()

    def cold_skel() -> None:
        draw_gait_skeleton(ax, snap, clear=True, lite_draw=False, display_mode="gait")

    wall, cpu, rss = _time_call(cold_skel, repeats=3)
    report.add("render:3d_skeleton_cold", wall, rss_delta_mb=rss, cpu_ms=cpu)

    draw_gait_skeleton(ax, snap, clear=True, lite_draw=False, display_mode="gait")

    def hot_skel() -> None:
        s = make_snap(dx=0.01)
        draw_gait_skeleton(ax, s, clear=False, lite_draw=False, display_mode="gait")

    wall, cpu, rss = _time_call(hot_skel, repeats=12)
    report.add("render:3d_skeleton_hot", wall, rss_delta_mb=rss, cpu_ms=cpu)

    # Video overlay
    rgb = np.zeros((720, 1280, 3), dtype=np.uint8)
    rgb[:] = (40, 44, 52)
    kps = [
        Keypoint(name="left_hip", x=0.45, y=0.55, z=0.0, visibility=1.0),
        Keypoint(name="right_hip", x=0.55, y=0.55, z=0.0, visibility=1.0),
        Keypoint(name="left_knee", x=0.45, y=0.72, z=0.0, visibility=1.0),
        Keypoint(name="right_knee", x=0.55, y=0.72, z=0.0, visibility=1.0),
        Keypoint(name="left_ankle", x=0.45, y=0.90, z=0.0, visibility=1.0),
        Keypoint(name="right_ankle", x=0.55, y=0.90, z=0.0, visibility=1.0),
        Keypoint(name="left_shoulder", x=0.42, y=0.30, z=0.0, visibility=1.0),
        Keypoint(name="right_shoulder", x=0.58, y=0.30, z=0.0, visibility=1.0),
        Keypoint(name="left_elbow", x=0.38, y=0.42, z=0.0, visibility=1.0),
        Keypoint(name="right_elbow", x=0.62, y=0.42, z=0.0, visibility=1.0),
        Keypoint(name="left_wrist", x=0.36, y=0.52, z=0.0, visibility=1.0),
        Keypoint(name="right_wrist", x=0.64, y=0.52, z=0.0, visibility=1.0),
        Keypoint(name="nose", x=0.50, y=0.18, z=0.0, visibility=1.0),
    ]
    layers = VideoOverlayLayers(skeleton=True, joint_labels=False, com=True)

    def overlay_opaque() -> None:
        compose_video_skeleton_overlay(rgb, kps, layers=layers, opacity=1.0)

    wall, cpu, rss = _time_call(overlay_opaque, repeats=8)
    report.add("render:video_overlay_opaque", wall, rss_delta_mb=rss, cpu_ms=cpu)

    def overlay_blend() -> None:
        compose_video_skeleton_overlay(rgb, kps, layers=layers, opacity=0.55)

    wall, cpu, rss = _time_call(overlay_blend, repeats=8)
    report.add("render:video_overlay_blend", wall, rss_delta_mb=rss, cpu_ms=cpu)

    # Graph (knee-style line + playhead)
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    gfig = Figure(figsize=(5, 2.6), dpi=100)
    gax = gfig.add_subplot(111)
    canvas = FigureCanvasAgg(gfig)
    t = np.linspace(0, 1, 240)
    y = 20 + 35 * np.sin(2 * np.pi * t)

    def graph_full() -> None:
        gax.cla()
        gax.plot(t, y, color="#4aa3ff", linewidth=1.8)
        gax.axvline(0.42, color="#e8eef4", linewidth=1.4)
        canvas.draw()

    wall, cpu, rss = _time_call(graph_full, repeats=10)
    report.add("render:graph_full_redraw", wall, rss_delta_mb=rss, cpu_ms=cpu)

    # Blit-style: update line only after one full draw
    canvas.draw()
    bg = canvas.copy_from_bbox(gax.bbox)
    (line,) = gax.plot([0.42, 0.42], [y.min(), y.max()], color="#e8eef4", linewidth=1.4)

    def graph_blit() -> None:
        canvas.restore_region(bg)
        line.set_xdata([0.55, 0.55])
        gax.draw_artist(line)
        canvas.blit(gax.bbox)

    wall, cpu, rss = _time_call(graph_blit, repeats=20)
    report.add("render:graph_blit_playhead", wall, rss_delta_mb=rss, cpu_ms=cpu)

    # Trajectory path cache append cost
    from stablewalk.models.gait_motion import GaitMotionRecording
    from stablewalk.ui.viewers.dof_trajectory_3d import (
        _joint_path_with_times,
        clear_trajectory_path_cache,
    )

    snaps = [make_snap(dx=i * 0.002) for i in range(120)]
    for i, s in enumerate(snaps):
        s.frame_index = i
        s.time_s = i / 30.0
    rec = GaitMotionRecording("profile", 30.0, snaps)
    clear_trajectory_path_cache()

    def traj_path() -> None:
        clear_trajectory_path_cache()
        _joint_path_with_times(rec, "left_knee", float(len(snaps) - 1))

    wall, cpu, rss = _time_call(traj_path, repeats=5)
    report.add("render:trajectory_path_build", wall, rss_delta_mb=rss, cpu_ms=cpu)

    def traj_path_hit() -> None:
        _joint_path_with_times(rec, "left_knee", float(len(snaps) - 1))

    _joint_path_with_times(rec, "left_knee", float(len(snaps) - 1))
    wall, cpu, rss = _time_call(traj_path_hit, repeats=20)
    report.add("render:trajectory_path_cache_hit", wall, rss_delta_mb=rss, cpu_ms=cpu)


def profile_gui(report: ProfileReport) -> None:
    import tkinter as tk

    import matplotlib

    matplotlib.use("TkAgg")

    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import (
        TAB_BIOMECHANICS,
        TAB_COMPARE,
        TAB_MOTION,
        TAB_OVERVIEW,
        TAB_RESULTS_SUMMARY,
        select_dashboard_tab,
    )
    from stablewalk.ui.tk.dashboard_tab_lifecycle import active_tab_id

    rss0 = _rss_mb()
    c0 = time.process_time()
    t0 = time.perf_counter()
    root = tk.Tk()
    root.withdraw()
    gui = StableWalkGUI(root=root)
    root.update_idletasks()
    wall = (time.perf_counter() - t0) * 1000.0
    cpu = (time.process_time() - c0) * 1000.0
    rss1 = _rss_mb()
    report.add(
        "startup:gui_construct",
        wall,
        rss_delta_mb=max(0.0, rss1 - rss0),
        cpu_ms=cpu,
        notes="StableWalkGUI + dashboard layout (withdrawn)",
    )

    # Tab switching
    tabs = [
        ("tab:overview", TAB_OVERVIEW),
        ("tab:motion", TAB_MOTION),
        ("tab:biomechanics", TAB_BIOMECHANICS),
        ("tab:results", TAB_RESULTS_SUMMARY),
        ("tab:compare", TAB_COMPARE),
        ("tab:overview_return", TAB_OVERVIEW),
    ]
    for name, tab in tabs:
        def _switch(tab=tab) -> None:
            select_dashboard_tab(gui, tab)
            # Idle tasks only — full ``update()`` drains deferred after()
            # queues (Compare demo analyze, chart fits) and inflates switch time.
            root.update_idletasks()

        wall, cpu, rss = _time_call(_switch, repeats=1)
        report.add(
            name,
            wall,
            rss_delta_mb=rss,
            cpu_ms=cpu,
            notes=f"active={active_tab_id(gui)}",
        )

    # Compare second open (already built)
    def _compare_again() -> None:
        select_dashboard_tab(gui, TAB_COMPARE)
        root.update_idletasks()

    wall, cpu, rss = _time_call(_compare_again, repeats=2)
    report.add("tab:compare_warm", wall, rss_delta_mb=rss, cpu_ms=cpu)

    try:
        root.destroy()
    except tk.TclError:
        pass


def rank_bottlenecks(report: ProfileReport) -> None:
    """Rank by wall time; annotate likely impact category."""
    scored = [s for s in report.samples if s.ms >= 0]
    scored.sort(key=lambda s: s.ms, reverse=True)
    categories = {
        "import:": "startup",
        "startup:": "startup",
        "tab:": "tab_switch",
        "render:graph": "graph",
        "render:video": "video",
        "render:3d": "3d",
        "render:trajectory": "3d",
    }
    top = []
    for i, s in enumerate(scored[:20], start=1):
        cat = "other"
        for prefix, c in categories.items():
            if s.name.startswith(prefix):
                cat = c
                break
        impact = "critical" if s.ms >= 200 else "high" if s.ms >= 50 else "medium"
        top.append(
            {
                "rank": i,
                "name": s.name,
                "ms": round(s.ms, 2),
                "cpu_ms": round(s.cpu_ms, 2),
                "rss_delta_mb": round(s.rss_delta_mb, 2),
                "category": cat,
                "impact": impact,
                "notes": s.notes,
            }
        )
    report.bottlenecks = top


def write_report(report: ProfileReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "performance_profile.json"
    md_path = out_dir / "performance_report.md"
    json_path.write_text(
        json.dumps(
            {
                "meta": report.meta,
                "samples": [asdict(s) for s in report.samples],
                "bottlenecks": report.bottlenecks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# StableWalk Performance Report",
        "",
        f"Generated: {report.meta.get('generated_at', '')}",
        f"Python: {report.meta.get('python', '')}",
        f"RSS after profile: {report.meta.get('rss_end_mb', 0):.1f} MiB",
        f"Tracemalloc peak: {report.meta.get('tracemalloc_peak_mb', 0):.1f} MiB",
        "",
        "## Top 20 bottlenecks",
        "",
        "| Rank | Name | Wall ms | CPU ms | ΔRSS MiB | Category | Impact |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for b in report.bottlenecks:
        lines.append(
            f"| {b['rank']} | `{b['name']}` | {b['ms']:.1f} | {b['cpu_ms']:.1f} | "
            f"{b['rss_delta_mb']:.1f} | {b['category']} | {b['impact']} |"
        )
    lines.extend(
        [
            "",
            "## All samples",
            "",
            "| Name | Wall ms | CPU ms | ΔRSS MiB | Notes |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for s in sorted(report.samples, key=lambda x: x.ms, reverse=True):
        lines.append(
            f"| `{s.name}` | {s.ms:.1f} | {s.cpu_ms:.1f} | {s.rss_delta_mb:.1f} | {s.notes} |"
        )
    lines.extend(
        [
            "",
            "## Optimizations applied",
            "",
            "1. Lazy `stablewalk.analysis` / `session_compare` type imports (no full biomech graph at GUI import).",
            "2. Lazy `stablewalk.adapters` and `stablewalk.pose.backends` package exports.",
            "3. MediaPipe deferred until pose validation / overlay compose / backend construction.",
            "4. `pose_sequence_to_gait_motion` imported only when a session is analyzed.",
            "5. Skeleton cold draw skips Matplotlib `_update_patch_limits` (`_skel_add_patch`) — view limits set explicitly.",
            "6. Welcome/reset no longer force-creates Motion/Biomechanics charts just to clear them.",
            "7. Artist-pooled 3D manikin + graph blit playheads; Compare/Advanced charts lazy on first tab visit.",
            "8. Video overlay opaque path avoids full-frame blend copies; decode/pin caches capped.",
            "9. Overlay Mode controls no longer import MediaPipe during dashboard layout.",
            "",
            "## Baseline → current (key metrics)",
            "",
            "| Metric | Before | After (this run) |",
            "| --- | ---: | ---: |",
            f"| `import:stablewalk.ui.tk.app` | ~2755 ms | {_fmt_sample(report, 'import:stablewalk.ui.tk.app')} |",
            f"| `render:3d_skeleton_cold` | ~867–958 ms | {_fmt_sample(report, 'render:3d_skeleton_cold')} |",
            f"| `render:3d_skeleton_hot` | ~34–37 ms | {_fmt_sample(report, 'render:3d_skeleton_hot')} |",
            f"| `startup:gui_construct` | failed / ~8 s (empty chart force-build) | {_fmt_sample(report, 'startup:gui_construct')} |",
            f"| Process RSS (end) | n/a | {report.meta.get('rss_end_mb', 0):.1f} MiB |",
            f"| Tracemalloc peak | n/a | {report.meta.get('tracemalloc_peak_mb', 0):.1f} MiB |",
            "",
            "## Remaining hard costs",
            "",
            "- Matplotlib cold import (~2 s) is required for TkAgg GUI; cannot be removed, only deferred until GUI launch.",
            "- NumPy / OpenCV cold imports are shared runtime deps.",
            "- Full graph `cla()` redraw remains for chart rebuilds; playhead path uses blit.",
            "- First Motion-tab chart materialization still pays one-time Agg draw cost when analysis exists.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path


def _fmt_sample(report: ProfileReport, name: str) -> str:
    for s in report.samples:
        if s.name == name:
            if s.ms < 0:
                return f"failed ({s.notes})" if s.notes else "failed"
            return f"{s.ms:.1f} ms"
    return "—"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gui", action="store_true", help="Skip Tk GUI timings")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "output" / "performance",
        help="Report output directory",
    )
    args = parser.parse_args()

    tracemalloc.start()
    report = ProfileReport(
        meta={
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": sys.version.split()[0],
            "platform": sys.platform,
        }
    )

    print("Profiling imports…")
    profile_imports(report)
    print("Profiling render hot paths…")
    profile_render_hotpaths(report)
    if not args.skip_gui:
        print("Profiling GUI startup + tab switches…")
        try:
            profile_gui(report)
        except Exception as exc:
            report.add("startup:gui_construct", -1.0, notes=f"failed: {exc}")
            print(f"GUI profile failed: {exc}")

    cur, peak = tracemalloc.get_traced_memory()
    report.meta["rss_end_mb"] = round(_rss_mb(), 2)
    report.meta["tracemalloc_current_mb"] = round(cur / (1024 * 1024), 2)
    report.meta["tracemalloc_peak_mb"] = round(peak / (1024 * 1024), 2)
    rank_bottlenecks(report)
    path = write_report(report, args.out)
    print(f"Wrote {path}")
    print("\nTop 10:")
    for b in report.bottlenecks[:10]:
        print(f"  #{b['rank']:02d} {b['ms']:8.1f} ms  {b['name']}  [{b['impact']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
