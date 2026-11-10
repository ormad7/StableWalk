"""Load normal_gait, verify joint-data export and 3D Coordinate Inspector."""

from __future__ import annotations

import contextlib
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@contextlib.contextmanager
def _suppress_messageboxes():
    import tkinter.messagebox as mb

    originals = (mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel)
    mb.showinfo = lambda *a, **k: True  # type: ignore[assignment]
    mb.showwarning = lambda *a, **k: None  # type: ignore[assignment]
    mb.showerror = lambda *a, **k: None  # type: ignore[assignment]
    mb.askokcancel = lambda *a, **k: True  # type: ignore[assignment]
    try:
        yield
    finally:
        mb.showinfo, mb.showwarning, mb.showerror, mb.askokcancel = originals


def _pump(root, *, seconds: float = 0.25) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            root.update()
        except Exception:
            break
        time.sleep(0.01)


def _label_text(labels: dict, key: str) -> str:
    lbl = labels.get(key)
    if lbl is None:
        return ""
    try:
        return str(lbl.cget("text") or "")
    except Exception:
        return ""


def main() -> int:
    import tkinter as tk

    from stablewalk import config
    from stablewalk.ui.media.demo_gait import demo_path, example_by_key
    from stablewalk.ui.tk.app import StableWalkGUI
    from stablewalk.ui.tk.dashboard_notebook import TAB_OVERVIEW, select_dashboard_tab

    poses_path = config.POSES_DIR / "normal_gait_poses.json"
    ex = example_by_key("normal")
    video_path = demo_path(ex) if ex else config.DEMO_VIDEOS_DIR / "normal_gait.mp4"
    if not Path(video_path).is_file() and not poses_path.is_file():
        print(f"FAILED: missing {video_path} and {poses_path}", file=sys.stderr)
        return 1

    failures: list[str] = []
    with _suppress_messageboxes():
        gui = StableWalkGUI()
        root = gui.root
        root.geometry("1400x860+20+20")
        _pump(root, seconds=0.4)
        select_dashboard_tab(gui, TAB_OVERVIEW)
        _pump(root, seconds=0.2)

        source = str(video_path)
        gui.url_var.set(source)
        try:
            gui.url_entry.delete(0, "end")
            gui.url_entry.insert(0, source)
        except Exception:
            pass
        gui._current_source = source

        loaded = False
        if poses_path.is_file():
            try:
                loaded = bool(
                    gui.load_poses(
                        poses_path,
                        fresh=True,
                        expected_source=source,
                        wait_for_analysis=True,
                    )
                )
            except TypeError:
                loaded = bool(gui.load_poses(poses_path, fresh=True))
        if not loaded and Path(video_path).is_file():
            try:
                gui._load_source(source)
                loaded = True
            except Exception as exc:
                failures.append(f"load video failed: {exc}")
        if not loaded:
            print("FAILED: could not load normal_gait", file=sys.stderr)
            try:
                root.destroy()
            except Exception:
                pass
            return 1

        _pump(root, seconds=1.2)
        gui._activate_dof_item("right_knee", add_if_missing=True)
        _pump(root, seconds=0.6)
        if gui.skeleton_player is not None and gui.skeleton_player.frame_count > 1:
            gui._go_to(min(12, gui.skeleton_player.frame_count - 1))
        _pump(root, seconds=0.5)

        recording = gui._analysis_motion_recording()
        frame_count = int(getattr(recording, "frame_count", 0) or 0)
        print(f"recording_frames={frame_count}")
        if frame_count <= 0:
            failures.append("no gait_motion recording after load")

        rows = gui._collect_joint_data_rows()
        print(f"joint_rows={len(rows)}")
        if not rows:
            failures.append("joint data rows empty")
        else:
            sample = rows[min(12, len(rows) - 1)]
            print(
                "row_sample",
                sample.frame,
                sample.timestamp,
                sample.joint,
                sample.x,
                sample.y,
                sample.z,
                sample.angle,
                sample.velocity,
                sample.acceleration,
                sample.confidence,
                sample.gait_phase,
            )
            if sample.x is None and sample.y is None and sample.z is None:
                failures.append("XYZ missing on sample row")

        gui._open_collected_data_dialog()
        _pump(root, seconds=0.4)
        dlg = getattr(gui, "_collected_data_dialog", None)
        tree = getattr(gui, "_joint_data_dialog_tree", None)
        dlg_ok = False
        try:
            dlg_ok = bool(dlg is not None and dlg.winfo_exists())
        except tk.TclError:
            dlg_ok = False
        tree_count = 0
        if tree is not None:
            try:
                tree_count = len(tree.get_children())
            except tk.TclError:
                tree_count = 0
        print(f"joint_dialog_open={int(dlg_ok)} tree_rows={tree_count}")
        if not dlg_ok:
            failures.append("Joint Data dialog did not open")
        if tree_count <= 0:
            failures.append("Joint Data tree empty")

        tmp = Path(tempfile.mkdtemp(prefix="sw_joint_export_"))
        from stablewalk.ui.joint_data_table import (
            write_joint_data_csv,
            write_joint_data_json,
        )

        if rows:
            csv_path = write_joint_data_csv(rows, tmp / "joint.csv")
            json_path = write_joint_data_json(
                rows,
                tmp / "joint.json",
                video_source=source,
                selected_joints=["right_knee"],
                fps=float(getattr(recording, "fps", 0) or 0) or None,
            )
            with csv_path.open(encoding="utf-8", newline="") as handle:
                header = next(csv.reader(handle))
            required = {
                "Frame",
                "Time",
                "Joint",
                "X",
                "Y",
                "Z",
                "Angle",
                "Velocity",
                "Acceleration",
                "Confidence",
                "Gait Phase",
            }
            missing = required - set(header)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            json_row = (payload.get("rows") or [{}])[0]
            print(f"csv_header={header}")
            print(f"csv_rows={sum(1 for _ in csv_path.read_text(encoding='utf-8').splitlines()) - 1}")
            print(f"json_time={json_row.get('time')} json_gait_phase={json_row.get('gait_phase')}")
            if missing:
                failures.append(f"CSV missing columns: {sorted(missing)}")
            if "time" not in json_row:
                failures.append("JSON row missing time")

        overview = getattr(gui, "overview_traj_readout_values", None) or {}
        ov_joint = _label_text(overview, "joint_name")
        ov_frame = _label_text(overview, "frame")
        ov_x = _label_text(overview, "x")
        print(f"overview_inspector joint={ov_joint!r} frame={ov_frame!r} x={ov_x!r}")
        if ov_joint in {"", "—", "N/A"}:
            failures.append("Overview inspector joint empty")
        if ov_x in {"", "—", "N/A"}:
            failures.append("Overview inspector X empty")

        gui._open_coordinate_inspector_window()
        _pump(root, seconds=0.5)
        win = getattr(gui, "_coord_inspector_window", None)
        win_ok = False
        try:
            win_ok = bool(win is not None and win.winfo_exists())
        except tk.TclError:
            win_ok = False
        win_vals = getattr(gui, "_coord_inspector_window_values", None) or {}
        before = _label_text(win_vals, "frame")
        print(f"inspector_window_open={int(win_ok)} frame={before!r}")
        if not win_ok:
            failures.append("3D Coordinate Inspector window did not open")

        n = int(getattr(gui.skeleton_player, "frame_count", 0) or 0)
        if n > 2:
            target = min(n - 1, 40)
            gui._go_to(target)
            _pump(root, seconds=0.5)
            after = _label_text(win_vals, "frame")
            after_x = _label_text(win_vals, "x")
            print(f"inspector_after_seek frame={after!r} x={after_x!r}")
            if after == before and n > 20:
                failures.append("inspector frame did not update after seek")
            if after_x in {"", "—", "N/A"}:
                failures.append("inspector X empty after seek")

        try:
            root.destroy()
        except Exception:
            pass

    if failures:
        print("FAILED:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
