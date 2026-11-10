"""Display-only layout helpers for Spectral Features and Asymmetry."""

from __future__ import annotations

from types import SimpleNamespace


def test_feature_value_and_units_are_separated() -> None:
    from stablewalk.ui.tk.dashboard_spectral_features import (
        feature_numeric_display,
        feature_preprocess_copy,
        feature_quality_reason,
        feature_units_display,
        feature_uses_zscore,
    )

    rec = SimpleNamespace(
        name="spectral_entropy",
        available=True,
        value=0.42,
        units="1",
        display_value=lambda: "0.42 1",
        quality="moderate",
        notes=[],
    )
    assert feature_numeric_display(rec) == "0.42"
    assert feature_units_display(rec) == "dimensionless"
    assert feature_uses_zscore(rec)
    assert "z-score" in feature_preprocess_copy(rec).lower()
    assert "dimensionless" in feature_preprocess_copy(rec).lower()
    assert feature_quality_reason(rec).startswith("moderate")

    rms = SimpleNamespace(
        name="rms_amplitude",
        available=True,
        value=12.5,
        units="deg",
        display_value=lambda: "12.5 deg",
        quality="low",
        notes=["gap fraction high"],
    )
    assert feature_numeric_display(rms) == "12.5"
    assert feature_units_display(rms) == "deg"
    assert not feature_uses_zscore(rms)
    assert "detrend" in feature_preprocess_copy(rms).lower()
    assert "not z-scored" in feature_preprocess_copy(rms).lower()
    assert "gap fraction" in feature_quality_reason(rms)
    assert "Numeric result" in feature_quality_reason(rms) or "retained" in feature_quality_reason(rms).lower()
    assert "clinical" in feature_quality_reason(rms).lower()

    missing = SimpleNamespace(
        name="stride_duration_s",
        available=False,
        value=None,
        units="s",
        display_value=lambda: "insufficient cycles",
        quality="insufficient",
        notes=["insufficient cycles: 1 detected"],
    )
    assert feature_numeric_display(missing) == "insufficient cycles"
    assert "insufficient" in feature_quality_reason(missing)
    assert "N/A" in feature_quality_reason(missing) or "not a value of 0" in feature_quality_reason(missing).lower()

    blank = SimpleNamespace(
        name="peak_frequency_hz",
        available=False,
        value=None,
        units="Hz",
        display_value=lambda: "unavailable",
        quality="insufficient",
        notes=[],
    )
    assert feature_numeric_display(blank) == "N/A"


def test_asymmetry_row_maps_to_the_matching_plot() -> None:
    from stablewalk.ui.tk.dashboard_asymmetry import (
        _metric_units,
        _quality_explanation,
        asymmetry_plot_id_for_row,
    )

    knee = SimpleNamespace(name="knee_flexion_peak", units="deg", comparison_kind="scalar")
    rom = SimpleNamespace(name="hip_flexion_rom", units="deg", comparison_kind="scalar")
    heel = SimpleNamespace(name="step_length", units="m", comparison_kind="scalar")
    phase = SimpleNamespace(name="stance_duration", units="s", comparison_kind="scalar")
    timing = SimpleNamespace(
        name="knee_flexion_peak_phase",
        units="% gait cycle",
        comparison_kind="timing",
    )
    assert asymmetry_plot_id_for_row(knee) == "knee"
    assert asymmetry_plot_id_for_row(rom) == "rom"
    assert asymmetry_plot_id_for_row(heel) == "foot"
    assert asymmetry_plot_id_for_row(phase) == "phase"
    assert asymmetry_plot_id_for_row(timing) == "knee"
    assert _metric_units(timing) == "% gait cycle"

    poor = SimpleNamespace(
        name="knee_flexion_rom",
        units="deg",
        comparison_kind="scalar",
        quality="insufficient",
        reliable=False,
        left=None,
        right=12.0,
        n_left_cycles=0,
        n_right_cycles=2,
        note="Need usable flexion series.",
    )
    expl = _quality_explanation(poor)
    assert "insufficient" in expl.lower()
    assert "HS–HS" in expl or "cycle" in expl.lower()
    assert "Need usable" in expl


def test_features_and_asymmetry_build_at_common_sizes() -> None:
    import tkinter as tk

    from stablewalk.ui.tk.dashboard_asymmetry import build_asymmetry_workspace
    from stablewalk.ui.tk.dashboard_spectral_features import (
        build_spectral_features_workspace,
        spectral_features_split_minsizes,
    )

    try:
        root = tk.Tk()
    except tk.TclError:
        return
    root.withdraw()
    try:
        gui = SimpleNamespace(root=root, _motion_page_notebook=None, _layout_dpi_scale=1.0)
        parent_f = tk.Frame(root)
        parent_a = tk.Frame(root)
        feat = build_spectral_features_workspace(gui, parent_f)
        asym = build_asymmetry_workspace(gui, parent_a)
        assert gui._spectral_features_split is not None
        assert gui._asymmetry_split is not None
        assert "units" in gui._spectral_features_tree["columns"]
        assert "units" in gui._asymmetry_tree["columns"]
        assert str(gui.fig_spectral_features.get_facecolor())
        assert str(gui.fig_asymmetry.get_facecolor())
        table_min, viz_min = spectral_features_split_minsizes(1.0)
        assert table_min >= 280
        assert viz_min >= 220
        feat_tree = gui._spectral_features_tree
        assert str(feat_tree.cget("xscrollcommand"))
        assert str(feat_tree.column("channel", "stretch")).lower() in {"1", "true"}
        assert str(feat_tree.column("value", "stretch")).lower() in {"1", "true"}
        assert str(gui.btn_spectral_feature_formula.cget("text")) == "Show formula"
        assert str(gui.lbl_spectral_feature_def.winfo_manager() or "") == ""
        asym_tree = gui._asymmetry_tree
        assert str(asym_tree.cget("xscrollcommand"))
        assert str(asym_tree.column("metric", "stretch")).lower() in {"0", "false", ""}
        assert str(gui.btn_asymmetry_formula.cget("text")) == "Show formulas"
        assert str(gui.lbl_asymmetry_legend.winfo_manager() or "") == ""
        assert "AI / SI" in str(gui.lbl_asymmetry_index_strip.cget("text"))
        assert len(gui.ax_asymmetry) == 2
        assert gui.fig_asymmetry.axes[0].get_gridspec().get_height_ratios()[0] > 1.2

        measurements = {}
        root.deiconify()
        pages = (
            (
                "features",
                parent_f,
                feat,
                "_place_spectral_features_sash",
                "_spectral_features_sash_for_width",
                "_spectral_features_table_host",
                "_spectral_features_viz",
            ),
            (
                "asymmetry",
                parent_a,
                asym,
                "_place_asymmetry_sash",
                "_asymmetry_sash_for_width",
                "_asymmetry_table_host",
                "_asymmetry_graph_host",
            ),
        )
        for width, height in ((1366, 768), (1920, 1080)):
            row = {}
            for key, parent, host, place_name, sash_attr, left_attr, right_attr in pages:
                parent_f.place_forget()
                parent_a.place_forget()
                parent.place(x=0, y=0, width=width, height=height)
                root.geometry(f"{width}x{height}+20+20")
                root.update_idletasks()
                root.update()
                place = getattr(gui, place_name, None)
                if callable(place):
                    setattr(gui, sash_attr, None)
                    place()
                root.update_idletasks()
                root.update()
                left_w = int(getattr(gui, left_attr).winfo_width() or 0)
                right_w = int(getattr(gui, right_attr).winfo_width() or 0)
                row[key] = {
                    "host": int(host.winfo_width() or 0),
                    "left": left_w,
                    "right": right_w,
                }
                assert int(host.winfo_width() or 0) >= min(width - 24, 800)
                assert left_w >= 160
                assert right_w >= 160
            measurements[f"{width}x{height}"] = row
        dest = __import__("pathlib").Path(__file__).resolve().parents[1] / "docs" / "spectral_layout_verify"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "features_asymmetry_sizes.json").write_text(
            __import__("json").dumps(measurements, indent=2), encoding="utf-8"
        )
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass
