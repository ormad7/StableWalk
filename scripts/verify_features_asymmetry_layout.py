"""Render Spectral Features and Asymmetry workspaces at 1366×768 and 1920×1080."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

OUT = ROOT / "docs" / "spectral_layout_verify"
OUT.mkdir(parents=True, exist_ok=True)


def _pair(name: str = "knee_flexion"):
    pct = np.linspace(0.0, 100.0, 101)
    left = 20.0 + 15.0 * np.sin(2 * np.pi * pct / 100.0)
    right = 22.0 + 14.0 * np.sin(2 * np.pi * pct / 100.0 + 0.15)
    return SimpleNamespace(
        name=name,
        aligned=True,
        phase_pct=tuple(float(x) for x in pct),
        left_mean=left,
        right_mean=right,
        left_std=np.ones_like(left) * 2.0,
        right_std=np.ones_like(right) * 2.0,
        units="deg",
        peak_left=float(left.max()),
        peak_right=float(right.max()),
        n_left_cycles=4,
        n_right_cycles=4,
    )


def _asym_report():
    def scalar(name, display, units, left, right, **extra):
        return SimpleNamespace(
            name=name,
            display=display,
            units=units,
            left=left,
            right=right,
            difference=left - right if left is not None and right is not None else None,
            difference_units=units,
            comparison_kind=extra.get("comparison_kind", "scalar"),
            si_percent=extra.get("si_percent", 4.0),
            asi_percent=extra.get("asi_percent", 4.0),
            si_norm=extra.get("si_norm", 0.96),
            quality=extra.get("quality", "derived"),
            reliable=True,
            note=extra.get("note", "Ipsilateral HS–HS cycles."),
            formula=extra.get("formula", "ROM = P95−P5"),
            n_left_cycles=4,
            n_right_cycles=4,
        )

    scalars = [
        scalar("knee_flexion_rom", "Knee flexion ROM", "deg", 32.0, 30.0),
        scalar("hip_flexion_rom", "Hip flexion ROM", "deg", 28.0, 27.0),
        scalar("ankle_flexion_rom", "Ankle flexion ROM", "deg", 18.0, 17.0),
        scalar("stance_duration", "Stance duration", "s", 0.62, 0.60),
        scalar("swing_duration", "Swing duration", "s", 0.40, 0.42),
        scalar(
            "knee_flexion_peak_phase",
            "Knee peak timing",
            "% gait cycle",
            42.0,
            46.0,
            comparison_kind="timing",
            si_percent=None,
            asi_percent=None,
            si_norm=None,
        ),
    ]
    by_name = {s.name: s for s in scalars}
    return SimpleNamespace(
        available=True,
        failed=False,
        scalars=scalars,
        warnings=[],
        kinematics=SimpleNamespace(provenance="inferred", coordinate_system="pelvis", units="m"),
        events=SimpleNamespace(
            view_type="side",
            reliability="moderate",
            reliability_reason="ok",
            left_cycles=[object()] * 4,
            right_cycles=[object()] * 4,
            heel_strikes=[object()] * 8,
            toe_offs=[object()] * 8,
        ),
        moments=None,
        trajectory_by_name=lambda name: _pair(name),
        scalar_by_name=lambda name, _b=by_name: _b.get(name),
    )


def main() -> int:
    import tkinter as tk

    from matplotlib.figure import Figure

    from stablewalk.analysis.spectral.report import analyze_spectral
    from stablewalk.ui.tk.dashboard_asymmetry import (
        build_asymmetry_workspace,
        render_asymmetry_workspace,
    )
    from stablewalk.ui.tk.dashboard_spectral_features import (
        build_spectral_features_workspace,
        render_spectral_features_workspace,
    )
    from stablewalk.ui.tk.layout_scale import dpi_scale_factor
    from test_spectral_analysis import _sine_recording

    root = tk.Tk()
    root.title("Features / Asymmetry layout verify")
    root.attributes("-topmost", True)
    scale = dpi_scale_factor(root)
    gui = SimpleNamespace(root=root, _motion_page_notebook=None, _layout_dpi_scale=1.0)
    parent_f = tk.Frame(root)
    parent_a = tk.Frame(root)
    build_spectral_features_workspace(gui, parent_f)
    build_asymmetry_workspace(gui, parent_a)
    gui._spectral_report = analyze_spectral(_sine_recording())
    render_spectral_features_workspace(gui)
    gui._asymmetry_report = _asym_report()
    render_asymmetry_workspace(gui)

    payload = {"dpi_scale": scale, "sizes": {}}
    for width, height in ((1366, 768), (1920, 1080)):
        row = {}
        for key, parent, host_attr, place_name, sash_attr in (
            ("features", parent_f, "_spectral_features_host", "_place_spectral_features_sash", "_spectral_features_sash_for_width"),
            ("asymmetry", parent_a, "_asymmetry_host", "_place_asymmetry_sash", "_asymmetry_sash_for_width"),
        ):
            parent_f.place_forget()
            parent_a.place_forget()
            parent.place(x=0, y=0, width=width, height=height)
            root.geometry(f"{width}x{height}+40+40")
            root.deiconify()
            root.lift()
            root.update_idletasks()
            root.update()
            place = getattr(gui, place_name, None)
            if callable(place):
                setattr(gui, sash_attr, None)
                place()
            root.update()
            host = getattr(gui, host_attr)
            fig: Figure = gui.fig_spectral_features if key == "features" else gui.fig_asymmetry
            dest = OUT / f"{key}_{width}x{height}.png"
            fig.savefig(dest, dpi=100, facecolor=fig.get_facecolor())
            row[key] = {
                "host": [int(host.winfo_width()), int(host.winfo_height())],
                "figure": dest.name,
                "fig_face": str(fig.get_facecolor()),
            }
        payload["sizes"][f"{width}x{height}"] = row
        print(json.dumps({f"{width}x{height}": row}, indent=2), flush=True)

    (OUT / "features_asymmetry_sizes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    root.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
