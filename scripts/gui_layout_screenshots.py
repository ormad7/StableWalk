"""Capture live StableWalk tab screenshots for layout validation.

Forces the Tk window above other apps and prefers PrintWindow so the grab
is the application frame, not whatever sits at the same screen rectangle.
"""
from __future__ import annotations

import ctypes
import sys
import time
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tkinter as tk
from PIL import Image

from stablewalk.ui.tk.app import StableWalkGUI
from stablewalk.ui.tk.dashboard_notebook import (
    TAB_ADVANCED,
    TAB_BIOMECHANICS,
    TAB_COMPARE,
    TAB_MOTION,
    TAB_OVERVIEW,
    TAB_RESULTS_SUMMARY,
    select_dashboard_tab,
)
from stablewalk.ui.tk.layout_scale import apply_process_dpi_awareness, dpi_scale_factor

OUT = _ROOT / "docs" / "gui_layout_screenshots"
PW_RENDERFULLCONTENT = 0x00000002
GA_ROOT = 2


def _print_window(root: tk.Misc) -> Image.Image | None:
    """Capture the Tk toplevel via PrintWindow (Windows)."""
    if sys.platform != "win32":
        return None
    try:
        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hwnd = int(root.winfo_id())
        if hwnd:
            hwnd = int(user32.GetAncestor(hwnd, GA_ROOT) or hwnd)
        if not hwnd:
            return None
        left = ctypes.c_long()
        top = ctypes.c_long()
        right = ctypes.c_long()
        bottom = ctypes.c_long()

        class RECT(ctypes.Structure):
            _fields_ = [
                ("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long),
            ]

        rect = RECT()
        if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        w = int(rect.right - rect.left)
        h = int(rect.bottom - rect.top)
        if w < 80 or h < 80:
            return None
        hdc = user32.GetDC(hwnd)
        mem = gdi32.CreateCompatibleDC(hdc)
        bmp = gdi32.CreateCompatibleBitmap(hdc, w, h)
        gdi32.SelectObject(mem, bmp)
        ok = bool(user32.PrintWindow(hwnd, mem, PW_RENDERFULLCONTENT))
        if not ok:
            ok = bool(user32.PrintWindow(hwnd, mem, 0))
        if not ok:
            gdi32.DeleteObject(bmp)
            gdi32.DeleteDC(mem)
            user32.ReleaseDC(hwnd, hdc)
            return None
        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [
                ("biSize", ctypes.c_uint32),
                ("biWidth", ctypes.c_int32),
                ("biHeight", ctypes.c_int32),
                ("biPlanes", ctypes.c_uint16),
                ("biBitCount", ctypes.c_uint16),
                ("biCompression", ctypes.c_uint32),
                ("biSizeImage", ctypes.c_uint32),
                ("biXPelsPerMeter", ctypes.c_int32),
                ("biYPelsPerMeter", ctypes.c_int32),
                ("biClrUsed", ctypes.c_uint32),
                ("biClrImportant", ctypes.c_uint32),
            ]

        header = BITMAPINFOHEADER()
        header.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        header.biWidth = w
        header.biHeight = -h
        header.biPlanes = 1
        header.biBitCount = 32
        buf = (ctypes.c_ubyte * (w * h * 4))()
        gdi32.GetDIBits(mem, bmp, 0, h, buf, ctypes.byref(header), 0)
        img = Image.frombuffer("RGB", (w, h), bytes(buf), "raw", "BGRX", 0, 1)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem)
        user32.ReleaseDC(hwnd, hdc)
        return img
    except Exception:
        return None


def _grab_screen(root: tk.Misc) -> Image.Image | None:
    try:
        from PIL import ImageGrab
    except Exception:
        return None
    x = int(root.winfo_rootx())
    y = int(root.winfo_rooty())
    w = int(root.winfo_width())
    h = int(root.winfo_height())
    if w < 80 or h < 80:
        return None
    return ImageGrab.grab(bbox=(x, y, x + w, y + h))


def shot(root: tk.Misc, name: str) -> tuple[int, int]:
    try:
        root.attributes("-topmost", True)
        root.deiconify()
        root.lift()
        root.focus_force()
    except tk.TclError:
        pass
    root.update_idletasks()
    root.update()
    time.sleep(0.2)
    root.update()
    img = _print_window(root) or _grab_screen(root)
    w, h = int(root.winfo_width() or 0), int(root.winfo_height() or 0)
    if img is not None:
        OUT.mkdir(parents=True, exist_ok=True)
        img.save(OUT / name)
        print(f"saved {name} {w} {h} {img.size}", flush=True)
    else:
        print(f"FAILED {name} {w} {h}", flush=True)
    return w, h


def _find_notebook(widget: tk.Misc, titles: tuple[str, ...], depth: int = 0):
    if depth > 6:
        return None
    try:
        if widget.winfo_class() == "TNotebook":
            texts = [str(widget.tab(t, "text")) for t in widget.tabs()]
            if any(title in texts for title in titles):
                return widget
        for child in widget.winfo_children():
            found = _find_notebook(child, titles, depth + 1)
            if found is not None:
                return found
    except tk.TclError:
        return None
    return None


def _select_inner(nb, title: str) -> None:
    if nb is None:
        return
    for tab_id in nb.tabs():
        if str(nb.tab(tab_id, "text")) == title:
            nb.select(tab_id)
            return


def _capture_compare(app: StableWalkGUI, root: tk.Misc, prefix: str) -> None:
    """Map Compare without running activate_tab (that path can freeze Tk)."""
    try:
        nb = getattr(app, "_dashboard_notebook", None)
        page = getattr(app, "_tab_compare", None)
        if nb is None or page is None:
            print("compare tab missing", flush=True)
            return
        print(f"{prefix} compare select", flush=True)
        # Pretend Compare is already active so the skip handler does not
        # call handle_notebook_tab_changed / activate_tab.
        app._skip_next_notebook_tab_changed = True
        app._active_dashboard_tab_id = TAB_COMPARE
        nb.select(page)
        root.update_idletasks()
        root.update()
        time.sleep(0.12)
        shot(root, f"{prefix}_compare.png")
        inner = _find_notebook(page, ("Research dataset", "Laboratory"))
        if inner is not None:
            _select_inner(inner, "Research dataset")
            root.update()
            time.sleep(0.08)
            shot(root, f"{prefix}_compare_research.png")
        print(f"{prefix} compare done", flush=True)
    except Exception as exc:
        print(f"compare tab failed: {exc}", flush=True)


def _set_geometry(root: tk.Misc, spec: str) -> tuple[int, int]:
    try:
        root.state("normal")
    except tk.TclError:
        pass
    root.geometry(spec)
    root.update_idletasks()
    root.update()
    time.sleep(0.25)
    root.update()
    return int(root.winfo_width() or 0), int(root.winfo_height() or 0)


def main() -> int:
    errs: list[str] = []

    def hook(exc, val, tb) -> None:
        errs.append("".join(traceback.format_exception(exc, val, tb)))

    tk.Tk.report_callback_exception = staticmethod(hook)
    apply_process_dpi_awareness()
    root = tk.Tk()
    root.geometry("1366x768+20+20")
    app = StableWalkGUI(root=root)
    root.update()
    root.deiconify()
    try:
        root.attributes("-topmost", True)
        root.lift()
    except tk.TclError:
        pass
    root.update()

    scale = dpi_scale_factor(root)
    screen = (int(root.winfo_screenwidth()), int(root.winfo_screenheight()))
    print(
        f"screen={screen} dpi_scale={scale:.3f} tk={root.winfo_width()}x{root.winfo_height()}",
        flush=True,
    )

    poses = _ROOT / "data" / "output" / "poses" / "normal_gait_poses.json"
    if poses.is_file():
        try:
            app.load_poses(poses, fresh=True, wait_for_analysis=True)
        except TypeError:
            app.load_poses(poses, fresh=True)
        except Exception as exc:
            print(f"load_poses failed: {exc}", flush=True)
        for _ in range(20):
            root.update()
            time.sleep(0.03)

    sizes = (
        ("1366", "1366x768+20+20", True),
        ("1600", "1600x900+20+20", False),
        ("1920", "1920x1080+20+20", False),
        ("2560", "2560x1440+20+20", False),
    )
    major = (
        (TAB_OVERVIEW, "overview"),
        (TAB_MOTION, "motion"),
        (TAB_BIOMECHANICS, "biomech"),
        (TAB_RESULTS_SUMMARY, "results"),
        (TAB_ADVANCED, "advanced"),
    )

    for prefix, spec, extras in sizes:
        actual_w, actual_h = _set_geometry(root, spec)
        print(f"geometry {spec} actual={actual_w}x{actual_h}", flush=True)
        for tab, name in major:
            try:
                select_dashboard_tab(app, tab)
                shot(root, f"{prefix}_{name}.png")
            except Exception as exc:
                print(f"{prefix} {name} failed: {exc}", flush=True)
        _capture_compare(app, root, prefix)
        if extras:
            inner = _find_notebook(
                getattr(app, "_tab_motion"),
                ("Spectral Analysis", "Gait Correlation & Spectral Analysis", "Kinematics"),
            )
            if inner is not None:
                select_dashboard_tab(app, TAB_MOTION)
                for title in (
                    "Kinematics",
                    "Gait Correlation & Spectral Analysis",
                    "Spectral Analysis",
                    "Spectral Features",
                ):
                    _select_inner(inner, title)
                    root.update()
                    time.sleep(0.12)
                    shot(root, f"{prefix}_motion_{title.replace(' ', '_')}.png")
            bnb = _find_notebook(
                getattr(app, "_tab_biomechanics"),
                ("COM", "Balance", "Contact"),
            )
            if bnb is not None:
                select_dashboard_tab(app, TAB_BIOMECHANICS)
                for title in (
                    "COM",
                    "Ground Reaction Force",
                    "Balance",
                    "Contact",
                    "Gait Phase",
                ):
                    _select_inner(bnb, title)
                    root.update()
                    time.sleep(0.12)
                    shot(root, f"{prefix}_biomech_{title.replace(' ', '_')}.png")

    try:
        root.state("zoomed")
        root.update_idletasks()
        root.update()
        time.sleep(0.35)
        print(
            f"maximized actual={root.winfo_width()}x{root.winfo_height()}",
            flush=True,
        )
        for tab, name in major:
            try:
                select_dashboard_tab(app, tab)
                shot(root, f"max_{name}.png")
            except Exception as exc:
                print(f"max {name} failed: {exc}", flush=True)
        _capture_compare(app, root, "max")
    except tk.TclError as exc:
        print(f"maximized failed: {exc}", flush=True)

    print(f"errs {len(errs)}", flush=True)
    if errs:
        print(errs[0][:800], flush=True)
    ws = getattr(app, "_advanced_workspace", None)
    if ws is not None:
        print(
            "advanced",
            ws.winfo_width(),
            ws.winfo_height(),
            ws.winfo_viewable(),
            flush=True,
        )
    root.destroy()
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
