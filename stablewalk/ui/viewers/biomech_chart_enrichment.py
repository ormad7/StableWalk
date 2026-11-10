"""Shared enrichment for Biomechanics workspace graphs (display only).

Provides per-series statistics, peak markers, automatic interpretation text,
and export naming. Does not invent clinical thresholds beyond existing
chart_reference display bands.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from stablewalk.ui.colors import MUTED, PANEL, TEXT
from stablewalk.ui.viewers.chart_reference import (
    COM_VERTICAL_PELVIS_RELATIVE_ILLUSTRATIVE_BH,
    VGRF_BW_NORMAL,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# Export stem per Biomechanics workspace tab (active graph only).
BIOMECH_EXPORT_NAMES: dict[str, str] = {
    "com": "biomech_com",
    "grf": "biomech_grf",
    "balance": "biomech_balance",
    "contact": "biomech_contact",
    "stability": "biomech_phase",
}


@dataclass(frozen=True)
class SeriesStats:
    """Basic descriptive stats for one plotted series."""

    maximum: float | None
    minimum: float | None
    mean: float | None
    std: float | None
    n: int
    unit: str = ""
    peak_count: int = 0

    def format_line(self) -> str:
        if self.n <= 0 or self.mean is None:
            return "No statistics yet."
        u = f" {self.unit}" if self.unit else ""
        parts = [
            f"Max {self.maximum:.3g}{u}",
            f"Min {self.minimum:.3g}{u}",
            f"Avg {self.mean:.3g}{u}",
            f"SD {self.std:.3g}{u}" if self.std is not None else "SD —",
        ]
        if self.peak_count > 0:
            parts.append(f"Peaks {self.peak_count}")
        return "   ·   ".join(parts)


@dataclass(frozen=True)
class BiomechTabEnrichment:
    """Footer + export payload for one workspace tab."""

    tab_id: str
    interpretation: str
    stats_line: str
    export_name: str
    stats: SeriesStats | None = None


# Educational display labels for the interpretation card.
CLINICAL_CARD_SECTION_KEYS: tuple[str, ...] = (
    "clinical_interpretation",
    "normal_range",
    "current_observation",
    "possible_implication",
    "overall_assessment",
)

CLINICAL_CARD_SECTION_TITLES: dict[str, str] = {
    "clinical_interpretation": "Interpretation",
    "normal_range": "Normal range",
    "current_observation": "Current observation",
    "possible_implication": "Possible implication",
    "overall_assessment": "Overall assessment",
}

CLINICAL_CARD_DISCLAIMER = (
    "Educational reference only — not a medical diagnosis or clinical recommendation."
)

_TAB_TITLES: dict[str, str] = {
    "com": "Center of Mass",
    "grf": "Ground Reaction Force",
    "balance": "Balance",
    "contact": "Contact",
    "stability": "Stability",
}


@dataclass(frozen=True)
class ClinicalInterpretationCard:
    """Structured educational copy for one Biomechanics graph tab."""

    tab_id: str
    title: str
    clinical_interpretation: str
    normal_range: str
    current_observation: str
    possible_implication: str
    overall_assessment: str
    disclaimer: str = CLINICAL_CARD_DISCLAIMER

    def as_sections(self) -> dict[str, str]:
        return {
            "clinical_interpretation": self.clinical_interpretation,
            "normal_range": self.normal_range,
            "current_observation": self.current_observation,
            "possible_implication": self.possible_implication,
            "overall_assessment": self.overall_assessment,
        }


_SERIES_STATS_CACHE: dict[tuple, SeriesStats] = {}
_SERIES_STATS_CACHE_MAX = 64


def clear_series_stats_cache() -> None:
    _SERIES_STATS_CACHE.clear()


def compute_series_stats(
    values: Sequence[float] | np.ndarray,
    *,
    unit: str = "",
    peak_count: int = 0,
) -> SeriesStats:
    """Descriptive stats with a small content-keyed cache (scrub-safe)."""
    arr = np.asarray(values, dtype=float)
    # Key before filtering so identical source arrays reuse the same entry.
    try:
        data_ptr = int(arr.__array_interface__["data"][0])  # type: ignore[index]
    except Exception:
        data_ptr = id(arr)
    cache_key = (
        data_ptr,
        int(arr.size),
        str(unit),
        int(peak_count),
        float(arr[0]) if arr.size else 0.0,
        float(arr[-1]) if arr.size else 0.0,
    )
    cached = _SERIES_STATS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        stats = SeriesStats(None, None, None, None, 0, unit=unit, peak_count=0)
    else:
        stats = SeriesStats(
            maximum=float(np.nanmax(finite)),
            minimum=float(np.nanmin(finite)),
            mean=float(np.nanmean(finite)),
            std=float(np.nanstd(finite)) if finite.size > 1 else 0.0,
            n=int(finite.size),
            unit=unit,
            peak_count=int(peak_count),
        )
    if len(_SERIES_STATS_CACHE) >= _SERIES_STATS_CACHE_MAX:
        try:
            _SERIES_STATS_CACHE.pop(next(iter(_SERIES_STATS_CACHE)))
        except Exception:
            _SERIES_STATS_CACHE.clear()
    _SERIES_STATS_CACHE[cache_key] = stats
    return stats


def local_peak_indices(values: np.ndarray, *, min_sep: int = 3) -> list[int]:
    """Local-maxima indices for display markers (not clinical peak detection)."""
    n = len(values)
    if n < 3:
        return []
    peaks: list[int] = []
    last = -min_sep
    for i in range(1, n - 1):
        v = values[i]
        if not np.isfinite(v):
            continue
        a, b = values[i - 1], values[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if v >= a and v >= b and (i - last) >= min_sep:
            peaks.append(i)
            last = i
    return peaks


def annotate_peaks(
    ax: Axes,
    times: np.ndarray | Sequence[float],
    values: np.ndarray | Sequence[float],
    *,
    color: str,
    label: str = "Peaks",
) -> int:
    """Scatter local maxima; returns peak count drawn."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.size < 3 or y.size != t.size:
        return 0
    idxs = local_peak_indices(y, min_sep=max(3, len(y) // 40))
    if not idxs:
        return 0
    xs = [float(t[i]) for i in idxs if np.isfinite(t[i]) and np.isfinite(y[i])]
    ys = [float(y[i]) for i in idxs if np.isfinite(t[i]) and np.isfinite(y[i])]
    if not xs:
        return 0
    ax.scatter(
        xs,
        ys,
        s=36,
        color=color,
        edgecolors=PANEL,
        linewidths=0.6,
        zorder=7,
        marker="o",
        label=label,
    )
    return len(xs)


def annotate_stats_box(ax: Axes, stats: SeriesStats, *, loc: str = "upper left") -> None:
    """Compact Max / Min / Avg / SD annotation inside the axes."""
    if stats.n <= 0 or stats.mean is None:
        return
    from stablewalk.ui.viewers.chart_style import ANNOTATION_SIZE, mark_secondary_annotation

    u = f" {stats.unit}" if stats.unit else ""
    text = (
        f"Max {stats.maximum:.3g}{u}\n"
        f"Min {stats.minimum:.3g}{u}\n"
        f"Avg {stats.mean:.3g}{u}\n"
        f"SD  {stats.std:.3g}{u}"
    )
    ha = "left" if "left" in loc else "right"
    x = 0.02 if "left" in loc else 0.98
    y = 0.98 if "upper" in loc else 0.02
    va = "top" if "upper" in loc else "bottom"
    mark_secondary_annotation(
        ax.text(
            x,
            y,
            text,
            transform=ax.transAxes,
            ha=ha,
            va=va,
            color=TEXT,
            fontsize=ANNOTATION_SIZE - 1.0,
            family="monospace",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc=PANEL, ec=MUTED, alpha=0.92),
        )
    )


def enrich_numeric_series(
    ax: Axes,
    times: Sequence[float] | np.ndarray,
    values: Sequence[float] | np.ndarray,
    *,
    unit: str,
    peak_color: str,
    draw_peaks: bool = True,
    stats_loc: str = "upper left",
    peak_label: str = "Peaks",
) -> SeriesStats:
    """Peaks + stats box for a continuous series already plotted on ``ax``."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    n_peaks = 0
    if draw_peaks:
        n_peaks = annotate_peaks(ax, t, y, color=peak_color, label=peak_label)
    stats = compute_series_stats(y, unit=unit, peak_count=n_peaks)
    annotate_stats_box(ax, stats, loc=stats_loc)
    return stats


def _frac_inside_band(values: np.ndarray, band: tuple[float, float]) -> float | None:
    arr = values[np.isfinite(values)]
    if arr.size == 0:
        return None
    lo, hi = float(band[0]), float(band[1])
    return float(np.mean((arr >= lo) & (arr <= hi)))


def interpret_com(stats: SeriesStats, values: np.ndarray) -> str:
    band = COM_VERTICAL_PELVIS_RELATIVE_ILLUSTRATIVE_BH
    frac = _frac_inside_band(values, band)
    base = (
        "Segment-weighted COM vertical position relative to the per-frame pelvis origin "
        "(approximately body-height normalized). This is not floor-absolute COM height "
        "or global vertical COM excursion."
    )
    if stats.mean is None:
        return base
    bits = [base]
    if frac is not None:
        bits.append(
            f"About {frac * 100:.0f}% of samples fall in the illustrative display band "
            f"({band[0]:.2f}–{band[1]:.2f} approximate BH)."
        )
    if stats.std is not None and stats.std > 0.04:
        bits.append("Vertical excursion (SD) is relatively large — review for compensatory motion or tracking noise.")
    elif stats.std is not None and stats.std < 0.005:
        bits.append("Very flat COM height — check tracking completeness or limited walking dynamics.")
    return " ".join(bits)


def interpret_balance(
    stats: SeriesStats,
    stable_pct: float | None,
    *,
    reportable: bool = True,
    analysis: object | None = None,
) -> str:
    contract = None
    try:
        from stablewalk.analysis.biomechanical.stability_margin import (
            balance_display_contract,
        )

        if analysis is not None:
            contract = balance_display_contract(analysis)
    except Exception:
        contract = None
    unvalidated = (
        not reportable
        or (contract is not None and contract.validity != "valid")
        or (contract is not None and not contract.classification_supported)
    )
    if unvalidated:
        return (
            "Monocular COM versus base of support is experimental and is not a "
            "physical stability score. The plotted series is an unvalidated "
            "preview of signed COM–BoS distance. Normal/Stable/Unstable bands "
            "and the 0.04 m threshold have no documented basis for this method "
            "and are not used as classifications."
        )
    base = (
        "Stability margin of the COM relative to the base of support. "
        "Positive signed distance means the estimated COM projection is inside "
        "the estimated support polygon."
    )
    bits = [base]
    if stable_pct is not None:
        bits.append(f"Internal diagnostic stable-frame share ≈ {float(stable_pct):.0f}%.")
    if stats.mean is not None:
        bits.append(f"Mean signed distance {stats.mean:.3f} m.")
    return " ".join(bits)


def interpret_grf(stats: SeriesStats, left_peak: float | None, right_peak: float | None) -> str:
    base = (
        "Estimated vertical ground reaction force (virtual GRF in body weights). "
        "Not a force-plate measurement — use for relative left/right loading patterns only."
    )
    bits = [base]
    if stats.mean is not None:
        lo, hi = VGRF_BW_NORMAL
        frac = _frac_inside_band(
            np.asarray(
                [stats.mean] if stats.mean is not None else [],
                dtype=float,
            ),
            (lo, hi),
        )
        bits.append(
            f"Total/primary series avg {stats.mean:.2f} BW "
            f"(display normal ≈ {lo:.2f}–{hi:.2f} BW)."
        )
        _ = frac
    if left_peak is not None and right_peak is not None and left_peak > 0 and right_peak > 0:
        asym = abs(left_peak - right_peak) / max(left_peak, right_peak)
        bits.append(
            f"Peak L {left_peak:.2f} vs R {right_peak:.2f} BW "
            f"(relative |Δ| ≈ {asym * 100:.0f}%)."
        )
    return " ".join(bits)


def interpret_contact(
    *,
    stance_pct: float | None,
    swing_pct: float | None,
    hs_count: int | None,
) -> str:
    base = (
        "Left/right foot contact probability and stance events over time. "
        "Clear alternating contact with brief double support is expected in walking."
    )
    bits = [base]
    if stance_pct is not None and swing_pct is not None:
        bits.append(f"Stance ≈ {stance_pct:.0f}% · Swing ≈ {swing_pct:.0f}% of the cycle.")
        if stance_pct > 70:
            bits.append("Stance share is high vs a typical ~60% walk duty cycle — review cadence and double support.")
    if hs_count is not None and hs_count > 0:
        bits.append(f"Detected heel-strike events: {hs_count}.")
    return " ".join(bits)


def interpret_phase(
    *,
    double_support_pct: float | None,
    stable_pct: float | None,
) -> str:
    base = (
        "Per-foot gait phase timelines (Unknown, Swing, Stance). "
        "Unknown means the foot was not a usable contact observation — not confirmed "
        "swing and not a clinical grade. Bilateral double support is not interpolated "
        "as an in-between category."
    )
    bits = [base]
    if double_support_pct is not None:
        bits.append(f"Double-support share ≈ {double_support_pct:.0f}% of frames.")
        if double_support_pct > 35:
            bits.append("Elevated double support vs typical walking — review balance confidence.")
    if stable_pct is not None:
        bits.append(f"Linked stability: {float(stable_pct):.0f}% stable frames.")
    return " ".join(bits)


def phase_duration_stats(contact: Any) -> SeriesStats:
    """Per-foot phase shares: Max=stance%, Min=swing%, Avg=unknown% (L/R mean)."""
    from stablewalk.analysis.foot_contact_analysis import (
        FOOT_PHASE_STANCE,
        FOOT_PHASE_SWING,
        FOOT_PHASE_UNKNOWN,
        per_foot_phase_codes,
    )

    if contact is None or not getattr(contact, "per_frame", None):
        return SeriesStats(None, None, None, None, 0, unit="")
    left, right = per_foot_phase_codes(contact)
    n = len(left)
    if n == 0:
        return SeriesStats(None, None, None, None, 0, unit="%")
    stance = float(
        0.5 * (np.mean(left == FOOT_PHASE_STANCE) + np.mean(right == FOOT_PHASE_STANCE)) * 100.0
    )
    swing = float(
        0.5 * (np.mean(left == FOOT_PHASE_SWING) + np.mean(right == FOOT_PHASE_SWING)) * 100.0
    )
    unknown = float(
        0.5
        * (np.mean(left == FOOT_PHASE_UNKNOWN) + np.mean(right == FOOT_PHASE_UNKNOWN))
        * 100.0
    )
    return SeriesStats(
        maximum=stance,
        minimum=swing,
        mean=unknown,
        std=0.0,
        n=n,
        unit="%",
        peak_count=0,
    )


def format_phase_stats_line(stats: SeriesStats) -> str:
    if stats.n <= 0:
        return "No statistics yet."
    return (
        f"Stance {stats.maximum:.0f}%   ·   "
        f"Swing {stats.minimum:.0f}%   ·   "
        f"Unknown {stats.mean:.0f}%"
    )


def annotate_phase_stats_box(ax: Axes, contact: Any) -> SeriesStats:
    """Stats box for gait-phase charts with an explicit key under the box."""
    from stablewalk.ui.viewers.chart_style import ANNOTATION_SIZE, mark_secondary_annotation

    stats = phase_duration_stats(contact)
    if stats.n <= 0:
        return stats
    text = (
        f"Stance  {stats.maximum:.0f}%\n"
        f"Swing   {stats.minimum:.0f}%\n"
        f"Unknown {stats.mean:.0f}%"
    )
    mark_secondary_annotation(
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=TEXT,
            fontsize=ANNOTATION_SIZE - 1.0,
            family="monospace",
            zorder=8,
            bbox=dict(boxstyle="round,pad=0.28", fc=PANEL, ec=MUTED, alpha=0.92),
        )
    )
    return stats


def _empty_clinical_card(tab_id: str) -> ClinicalInterpretationCard:
    title = _TAB_TITLES.get(tab_id, tab_id.upper())
    return ClinicalInterpretationCard(
        tab_id=tab_id,
        title=title,
        clinical_interpretation=(
            f"{title} is a pose-estimated educational view of walking mechanics. "
            "Load and analyze a walking video to generate session-specific observations."
        ),
        normal_range="Display reference band appears after analysis is available.",
        current_observation="No observation yet — waiting for biomechanics analysis.",
        possible_implication=(
            "Without session data, implications cannot be summarized. "
            "This panel never provides a medical diagnosis."
        ),
        overall_assessment=(
            "Overall assessment unavailable until estimates are computed. "
            "Use this card for learning about typical gait patterns only."
        ),
    )


def _fmt_stat(value: float | None, *, unit: str = "", places: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "—"
    if unit:
        return f"{float(value):.{places}f} {unit}".rstrip()
    return f"{float(value):.{places}f}"


def build_clinical_interpretation(
    gui: Any, tab_id: str, *, enrichment: BiomechTabEnrichment | None = None
) -> ClinicalInterpretationCard:
    """Educational Clinical Interpretation sections for the active Biomechanics graph.

    Uses display-only reference bands. Never invents a medical diagnosis.
    """
    tid = str(tab_id or "com")
    enr = enrichment if enrichment is not None else build_tab_enrichment(gui, tid)
    stats = enr.stats
    title = _TAB_TITLES.get(tid, tid.upper())
    ba = getattr(gui, "_biomech_analysis", None)

    if tid == "com":
        lo, hi = COM_VERTICAL_PELVIS_RELATIVE_ILLUSTRATIVE_BH
        normal = (
            f"Illustrative pelvis-relative display band: {lo:.2f}–{hi:.2f} approximate "
            "body-height units. This is not a clinical normal range and is not comparable "
            "to floor-absolute COM-height references."
        )
        if stats is None or stats.n <= 0:
            return _empty_clinical_card(tid)
        frac = None
        if ba is not None and ba.center_of_mass is not None:
            try:
                ys = np.asarray(
                    [float(p[1]) for p in ba.center_of_mass.positions if p is not None],
                    dtype=float,
                )
                frac = _frac_inside_band(
                    ys, COM_VERTICAL_PELVIS_RELATIVE_ILLUSTRATIVE_BH
                )
            except Exception:
                frac = None
        obs_bits = [
            f"Mean {_fmt_stat(stats.mean, unit='BH')}",
            f"range {_fmt_stat(stats.minimum, unit='BH')}–{_fmt_stat(stats.maximum, unit='BH')}",
            f"SD {_fmt_stat(stats.std, unit='BH')}",
        ]
        if frac is not None:
            obs_bits.append(f"{frac * 100:.0f}% of samples inside the display band")
        implication = (
            "Larger vertical COM excursion can reflect energetic compensation, "
            "soft-tissue artifact, or tracking noise in monocular estimates."
        )
        if stats.std is not None and stats.std > 0.04:
            implication = (
                "Relatively large COM height variability — useful for reviewing "
                "compensatory motion patterns or pose-tracking limits (educational)."
            )
        elif stats.std is not None and stats.std < 0.005:
            implication = (
                "Very flat COM height may indicate limited dynamics or incomplete tracking — "
                "review video coverage and landmark confidence."
            )
        assessment = (
            "Educational summary: pelvis-relative COM estimates are for pattern review. "
            "Band occupancy is descriptive only; this is not a diagnosis."
        )
        return ClinicalInterpretationCard(
            tid,
            title,
            enr.interpretation,
            normal,
            "Current observation: " + "; ".join(obs_bits) + ".",
            implication,
            assessment,
        )

    if tid == "grf":
        lo, hi = VGRF_BW_NORMAL
        normal = (
            f"Display normal band for estimated vertical GRF ≈ {lo:.2f}–{hi:.2f} body weights (BW). "
            "These are virtual GRF estimates, not force-plate measurements."
        )
        if stats is None or stats.n <= 0:
            return _empty_clinical_card(tid)
        obs = (
            f"Current observation: mean {_fmt_stat(stats.mean, unit='BW', places=2)}, "
            f"peak {_fmt_stat(stats.maximum, unit='BW', places=2)} "
            f"({stats.peak_count} local peaks marked)."
        )
        implication = (
            "Left/right peak differences are teaching cues for loading symmetry. "
            "They do not confirm joint pathology or prescribe treatment."
        )
        assessment = (
            "Educational summary: use relative GRF shape and side-to-side comparison only. "
            "Not a medical device output or diagnosis."
        )
        return ClinicalInterpretationCard(
            tid, title, enr.interpretation, normal, obs, implication, assessment
        )

    if tid == "balance":
        sm = getattr(ba, "stability_margin", None) if ba is not None else None
        reportable = bool(getattr(sm, "reportable", True)) if sm is not None else True
        classification_supported = bool(
            getattr(sm, "classification_supported", False)
        ) if sm is not None else False
        if not reportable or not classification_supported:
            normal = (
                "No documented Normal/Stable/Unstable range applies to this "
                "monocular COM–BoS method. The 0.04 m threshold is an internal "
                "diagnostic bin, not Hof margin of stability. Physical COM–BoS "
                "requires measured support and scale."
            )
            if stats is None or stats.n <= 0:
                return _empty_clinical_card(tid)
            return ClinicalInterpretationCard(
                tid,
                title,
                interpret_balance(stats, None, reportable=False, analysis=sm),
                normal,
                "Current observation: unvalidated signed-distance preview; not a session KPI.",
                "Do not conclude Reduced or Unstable balance from this plot.",
                "Unavailable as a physical stability metric. Not a clinical diagnosis.",
            )
        normal = (
            "No population Normal band is documented for this COM–BoS proxy. "
            "Positive signed distance means the estimated COM projection is "
            "inside the estimated support polygon."
        )
        if stats is None or stats.n <= 0:
            return _empty_clinical_card(tid)
        stable_pct = None
        if sm is not None:
            stable_pct = getattr(sm, "stable_pct", None)
        obs_bits = [
            f"mean margin {_fmt_stat(stats.mean, unit='m')}",
            f"min {_fmt_stat(stats.minimum, unit='m')}",
        ]
        if stable_pct is not None:
            obs_bits.append(f"stable-frame share ≈ {float(stable_pct):.0f}%")
        implication = (
            "Signed-distance trends can highlight intervals to review in the video "
            "for learning — not clinical labeling, and not a Normal/Stable/Unstable score."
        )
        assessment = (
            "Educational summary: signed COM–BoS distance is a pose-estimated geometric "
            "proxy only. Not a clinical diagnosis."
        )
        return ClinicalInterpretationCard(
            tid,
            title,
            enr.interpretation,
            normal,
            "Current observation: " + "; ".join(obs_bits) + ".",
            implication,
            assessment,
        )

    if tid == "contact":
        normal = (
            "Typical overground walking reference (educational): stance ≈ 60% and swing ≈ 40% "
            "of the gait cycle, with brief double support. Contact here is pose-estimated "
            "probability, not a pressure-mat measurement."
        )
        stance = swing = None
        gm = getattr(ba, "gait_metrics", None) if ba is not None else None
        if gm is not None:
            if gm.stance_pct is not None:
                stance = gm.stance_pct.value
            if gm.swing_pct is not None:
                swing = gm.swing_pct.value
        if stats is None or (stats.n <= 0 and stance is None):
            return _empty_clinical_card(tid)
        obs_bits: list[str] = []
        if stance is not None:
            obs_bits.append(f"stance ≈ {float(stance):.0f}%")
        if swing is not None:
            obs_bits.append(f"swing ≈ {float(swing):.0f}%")
        if stats is not None and stats.n > 0:
            obs_bits.append(f"left-contact probability mean {_fmt_stat(stats.mean, places=2)}")
        implication = (
            "High stance share or irregular alternating contact can illustrate cautious "
            "timing or asymmetry for classroom discussion — not a clinical finding."
        )
        if stance is not None and float(stance) > 70:
            implication = (
                "Stance share is high vs a typical ~60% walking duty cycle — review cadence "
                "and double-support intervals as educational cues only."
            )
        assessment = (
            "Educational summary: contact timing teaches gait-cycle structure. "
            "It is not a diagnosis of limp, paresis, or injury."
        )
        return ClinicalInterpretationCard(
            tid,
            title,
            enr.interpretation,
            normal,
            "Current observation: " + ("; ".join(obs_bits) if obs_bits else "limited contact stats.") + ".",
            implication,
            assessment,
        )

    # stability / phase
    normal = (
        "Educational phase reference for walking: regular stance–swing alternation with "
        "modest double-support share (often near ~10–20% of the cycle in literature "
        "examples). Values here are pose-derived labels."
    )
    if stats is None or stats.n <= 0:
        return _empty_clinical_card(tid)
    obs = (
        f"Current observation: stance {_fmt_stat(stats.maximum, unit='%', places=0)}, "
        f"swing {_fmt_stat(stats.minimum, unit='%', places=0)}, "
        f"double support {_fmt_stat(stats.mean, unit='%', places=0)}, "
        f"uncertain {_fmt_stat(stats.std, unit='%', places=0)}."
    )
    implication = (
        "Prolonged double support or irregular phase switching can be discussed as "
        "caution or asymmetry cues in training contexts — never as a medical label."
    )
    if stats.mean is not None and float(stats.mean) > 35:
        implication = (
            "Elevated double-support share vs typical walking examples — useful for "
            "reviewing balance confidence in the recording (educational only)."
        )
    assessment = (
        "Educational summary: phase structure supports learning temporal stability. "
        "Overall assessment here is descriptive, not a clinical diagnosis."
    )
    return ClinicalInterpretationCard(
        tid, title, enr.interpretation, normal, obs, implication, assessment
    )


def build_tab_enrichment(gui: Any, tab_id: str) -> BiomechTabEnrichment:
    """Build interpretation + Max/Min/Avg/SD line for the active Biomechanics tab."""
    tid = str(tab_id or "com")
    export_name = BIOMECH_EXPORT_NAMES.get(tid, f"biomech_{tid}")
    ba = getattr(gui, "_biomech_analysis", None)
    contact = getattr(gui, "_foot_contact", None)
    vgrf = getattr(gui, "_estimated_vgrf", None)
    gait = getattr(gui, "_gait_cycle", None)

    if tid == "com":
        ys: list[float] = []
        if ba is not None and ba.center_of_mass is not None:
            try:
                ys = [float(p[1]) for p in ba.center_of_mass.positions if p is not None]
            except Exception:
                ys = []
        arr = np.asarray(ys, dtype=float)
        peaks = len(local_peak_indices(arr, min_sep=max(3, len(arr) // 40))) if arr.size else 0
        stats = compute_series_stats(arr, unit="BH", peak_count=peaks)
        return BiomechTabEnrichment(
            tid,
            interpret_com(stats, arr),
            stats.format_line(),
            export_name,
            stats,
        )

    if tid == "grf":
        series = None
        left_peak = right_peak = None
        if vgrf is not None and getattr(vgrf, "available", False):
            try:
                bw = float(getattr(vgrf, "body_weight_n", 0.0) or 0.0)
                total = np.asarray(vgrf.total_vgrf_vertical, dtype=float)
                series = total / max(bw, 1e-6) if bw > 0 else np.asarray(vgrf.left_vgrf_bw, dtype=float)
            except Exception:
                series = None
            m = getattr(vgrf, "metrics", None)
            if m is not None:
                left_peak = getattr(m, "left_peak_force_bw", None)
                right_peak = getattr(m, "right_peak_force_bw", None)
        arr = np.asarray(series if series is not None else [], dtype=float)
        peaks = len(local_peak_indices(arr, min_sep=max(3, len(arr) // 40))) if arr.size else 0
        stats = compute_series_stats(arr, unit="BW", peak_count=peaks)
        return BiomechTabEnrichment(
            tid,
            interpret_grf(
                stats,
                float(left_peak) if left_peak is not None else None,
                float(right_peak) if right_peak is not None else None,
            ),
            stats.format_line(),
            export_name,
            stats,
        )

    if tid == "balance":
        margins: list[float] = []
        stable_pct = None
        reportable = True
        sm = None
        if ba is not None and ba.stability_margin is not None:
            sm = ba.stability_margin
            reportable = bool(getattr(sm, "reportable", True))
            stable_pct = getattr(sm, "stable_pct", None) if reportable else None
            margins = [
                float(f.stability_margin_m)
                for f in sm.per_frame
                if getattr(f, "stability_margin_m", None) is not None
            ]
        arr = np.asarray(margins, dtype=float)
        peaks = len(local_peak_indices(arr, min_sep=max(3, len(arr) // 40))) if arr.size else 0
        stats = compute_series_stats(arr, unit="m", peak_count=peaks)
        return BiomechTabEnrichment(
            tid,
            interpret_balance(
                stats,
                float(stable_pct) if stable_pct is not None else None,
                reportable=reportable,
                analysis=sm,
            ),
            (
                stats.format_line()
                if reportable
                else "Unvalidated preview — not reportable as a physical stability score."
            ),
            export_name,
            stats,
        )

    if tid == "contact":
        # Prefer left contact probability as the primary continuous series for stats.
        arr = np.asarray([], dtype=float)
        src = contact if contact is not None else (getattr(ba, "contact", None) if ba else None)
        if src is not None and getattr(src, "left_contact_probability", None) is not None:
            try:
                arr = np.asarray(src.left_contact_probability, dtype=float)
            except Exception:
                arr = np.asarray([], dtype=float)
        peaks = len(local_peak_indices(arr, min_sep=max(3, len(arr) // 40))) if arr.size else 0
        stats = compute_series_stats(arr, unit="", peak_count=peaks)
        stance = swing = None
        gm = getattr(ba, "gait_metrics", None) if ba is not None else None
        if gm is not None:
            if gm.stance_pct is not None:
                stance = gm.stance_pct.value
            if gm.swing_pct is not None:
                swing = gm.swing_pct.value
        hs_count = None
        metrics = getattr(gait, "metrics", None) if gait is not None else None
        if metrics is not None:
            hs_count = int(
                (metrics.left_heel_strike_count or 0)
                + (metrics.right_heel_strike_count or 0)
            )
        # Richer stats line: Max/Min/Avg/SD of L contact + temporal cues.
        extra: list[str] = []
        if stance is not None:
            extra.append(f"Stance {stance:.0f}%")
        if swing is not None:
            extra.append(f"Swing {swing:.0f}%")
        if hs_count:
            extra.append(f"HS {hs_count}")
        stats_line = stats.format_line()
        if extra:
            stats_line = f"{stats_line}   ·   " + "   ·   ".join(extra)
        return BiomechTabEnrichment(
            tid,
            interpret_contact(stance_pct=stance, swing_pct=swing, hs_count=hs_count),
            stats_line,
            export_name,
            stats,
        )

    # stability / phase
    src_c = contact if contact is not None else (getattr(ba, "contact", None) if ba else None)
    stats = phase_duration_stats(src_c)
    ds_pct = stats.mean
    stable_pct = None
    sm = getattr(ba, "stability_margin", None) if ba is not None else None
    if sm is not None and getattr(sm, "stable_pct", None) is not None:
        stable_pct = float(sm.stable_pct)
    return BiomechTabEnrichment(
        tid,
        interpret_phase(double_support_pct=ds_pct, stable_pct=stable_pct),
        format_phase_stats_line(stats),
        export_name,
        stats,
    )


def draw_gait_events_from_contact(ax: Axes, contact: Any, *, show_legend: bool = False) -> None:
    """Heel-strike / toe-off markers from a FootContactAnalysisResult-like object."""
    if contact is None or not getattr(contact, "per_frame", None):
        return
    from stablewalk.ui.viewers.chart_reference import draw_gait_event_markers

    t = contact.timestamps
    left_hs, right_hs, left_to, right_to = [], [], [], []
    for i, frame in enumerate(contact.per_frame):
        if getattr(frame, "left_heel_strike", False):
            left_hs.append(float(t[i]))
        if getattr(frame, "right_heel_strike", False):
            right_hs.append(float(t[i]))
        if getattr(frame, "left_toe_off", False):
            left_to.append(float(t[i]))
        if getattr(frame, "right_toe_off", False):
            right_to.append(float(t[i]))
    draw_gait_event_markers(
        ax,
        left_hs=left_hs,
        right_hs=right_hs,
        left_to=left_to,
        right_to=right_to,
        show_legend=show_legend,
    )


__all__ = [
    "BIOMECH_EXPORT_NAMES",
    "CLINICAL_CARD_DISCLAIMER",
    "CLINICAL_CARD_SECTION_KEYS",
    "CLINICAL_CARD_SECTION_TITLES",
    "BiomechTabEnrichment",
    "ClinicalInterpretationCard",
    "SeriesStats",
    "annotate_peaks",
    "annotate_stats_box",
    "build_clinical_interpretation",
    "build_tab_enrichment",
    "compute_series_stats",
    "draw_gait_events_from_contact",
    "annotate_phase_stats_box",
    "enrich_numeric_series",
    "format_phase_stats_line",
    "local_peak_indices",
    "phase_duration_stats",
]
