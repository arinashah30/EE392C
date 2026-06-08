#!/usr/bin/env python3
"""Generate milestone comparison plots from consolidated sweep + M4 JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "plots"

LLAMA_SWEEP = REPO / "data/traces/memory_sweeps"
QWEN_SWEEP = REPO / "data/traces/memory_sweeps_qwen"

FRACTIONS = [10, 25, 50, 75]

M4_CONFIGS = [
    ("50sbuf_25hbm", "Primary\n50/25"),
    ("25sbuf_25hbm", "Balanced\n25/25"),
    ("50sbuf_10hbm", "Near-core\n50/10"),
    ("25sbuf_50hbm", "Capacity\n25/50"),
]

# Single-line x labels for presentation bar charts (avoids cramped multi-line ticks)
M4_PRESENTATION_LABELS = [
    "Primary 50/25",
    "Balanced 25/25",
    "Near-core 50/10",
    "Capacity 25/50",
]

LLAMA_COLOR = "#2563eb"
QWEN_COLOR = "#dc2626"
TECH_COLORS = {"1t1c": "#059669", "3t": "#7c3aed", "rram": "#0891b2", "feram": "#ea580c"}

HBM_REDUCTION_YLABEL = "HBM traffic reduction from baseline (%)"
LATENCY_IMPROVEMENT_YLABEL = "Latency improvement from baseline (%)"
ENERGY_REDUCTION_YLABEL = "Energy reduction from baseline (%)"
ENERGY_NOTE = "negative = higher energy (e.g. StRAM refresh)"


def improvement_from_baseline(pct_change: float) -> float:
    """Positive = better (faster, or lower energy)."""
    return -pct_change


def hbm_reduction_from_baseline(pct_change: float) -> float:
    """Convert simulator pct_change (negative = less traffic) to positive reduction %."""
    return -pct_change


def load_sweep(path: Path, *, spill: str = "best_case") -> dict[int, dict]:
    with path.open() as f:
        data = json.load(f)
    out: dict[int, dict] = {}
    for run in data["runs"]:
        if run["spill_victim_order"] != spill:
            continue
        pct = run["replacement_pct"]
        c = run["comparison"]
        hops = run["candidate"]["transfers_by_hop"]
        hbm_pct = c["hbm_traffic_bytes"]["pct_change"]
        out[pct] = {
            "hbm_pct": hbm_pct,
            "hbm_reduction_pct": hbm_reduction_from_baseline(hbm_pct),
            "time_pct": c["time_ns"]["pct_change"],
            "latency_improvement_pct": improvement_from_baseline(c["time_ns"]["pct_change"]),
            "energy_pct": c["energy_pJ"]["pct_change"],
            "energy_reduction_pct": improvement_from_baseline(c["energy_pJ"]["pct_change"]),
            "hbm_mb": c["hbm_traffic_bytes"]["candidate"] / 1e6,
            "baseline_hbm_mb": c["hbm_traffic_bytes"]["baseline"] / 1e6,
            "baseline_time_ms": c["time_ns"]["baseline"] / 1e6,
            "ltram_sbuf": hops.get("ltram->sbuf", 0),
            "stram_sbuf": hops.get("stram->sbuf", 0),
            "hbm_sbuf": hops.get("hbm->sbuf", 0),
        }
    return out


def load_m4(path: Path) -> dict:
    with path.open() as f:
        data = json.load(f)
    c = data["comparison"]
    hops = data["candidate"]["results"]["transfers_by_hop"]
    hbm_pct = c["hbm_traffic_bytes"]["pct_change"]
    return {
        "hbm_pct": hbm_pct,
        "hbm_reduction_pct": hbm_reduction_from_baseline(hbm_pct),
        "time_pct": c["time_ns"]["pct_change"],
        "latency_improvement_pct": improvement_from_baseline(c["time_ns"]["pct_change"]),
        "energy_pct": c["energy_pJ"]["pct_change"],
        "energy_reduction_pct": improvement_from_baseline(c["energy_pJ"]["pct_change"]),
        "hbm_mb": c["hbm_traffic_bytes"]["candidate"] / 1e6,
        "baseline_hbm_mb": c["hbm_traffic_bytes"]["baseline"] / 1e6,
        "ltram_sbuf": hops.get("ltram->sbuf", 0),
        "stram_sbuf": hops.get("stram->sbuf", 0),
        "hbm_sbuf": hops.get("hbm->sbuf", 0),
    }


def _style_axes(ax, *, ylabel: str, title: str, reduction: bool = False, signed: bool = False) -> None:
    if signed:
        ax.axhline(0, color="0.45", linewidth=0.9, linestyle="--", zorder=0)
    elif reduction:
        ax.set_ylim(bottom=0)
    else:
        ax.axhline(0, color="0.45", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, pad=10)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_sweep_lines(
    ax,
    series: dict[str, dict[int, dict]],
    *,
    metric: str,
    xlabel: str,
    title: str,
    ylabel: str,
    signed: bool = False,
) -> None:
    x = np.arange(len(FRACTIONS))
    for label, data in series.items():
        ax.plot(
            x,
            [data[p][metric] for p in FRACTIONS],
            marker="o",
            linewidth=2,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}%" for p in FRACTIONS])
    ax.set_xlabel(xlabel)
    reduction = metric in ("hbm_reduction_pct", "latency_improvement_pct") and not signed
    _style_axes(ax, ylabel=ylabel, title=title, reduction=reduction, signed=signed)
    ax.legend(frameon=False, fontsize=8, loc="upper left" if reduction else "best")


def _format_bar_label(val: float) -> str:
    if abs(val) < 2:
        return f"{val:.2f}%"
    return f"{val:.1f}%"


def _grouped_bars(
    ax,
    labels: list[str],
    llama_vals: list[float],
    qwen_vals: list[float],
    *,
    ylabel: str,
    title: str,
    signed: bool = False,
    annotate: bool = True,
    show_legend: bool = True,
    xtick_rotation: float = 0,
) -> None:
    x = np.arange(len(labels))
    width = 0.36
    ax.bar(x - width / 2, llama_vals, width, label="Llama 3.2-1B", color=LLAMA_COLOR)
    ax.bar(x + width / 2, qwen_vals, width, label="Qwen1.5-MoE-A2.7B", color=QWEN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=xtick_rotation,
        ha="right" if xtick_rotation else "center",
    )
    reduction = not signed
    _style_axes(ax, ylabel=ylabel, title=title, reduction=reduction, signed=signed)
    if show_legend:
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    if annotate:
        ax.relim()
        ax.autoscale()
        ymin, ymax = ax.get_ylim()
        span = max(ymax - ymin, 1e-6)
        dy_pos = span * 0.06
        dy_neg = -span * 0.06
        for i, (lv, qv) in enumerate(zip(llama_vals, qwen_vals)):
            for offset, val in [(-width / 2, lv), (width / 2, qv)]:
                va = "bottom" if val >= 0 else "top"
                dy = dy_pos if val >= 0 else dy_neg
                ax.text(
                    i + offset,
                    val + dy,
                    _format_bar_label(val),
                    ha="center",
                    va=va,
                    fontsize=7,
                )
        # Headroom so bar labels are not clipped
        ymin, ymax = ax.get_ylim()
        span = ymax - ymin
        if signed:
            ax.set_ylim(ymin - span * 0.08, ymax + span * 0.12)
        else:
            ax.set_ylim(0, ymax + span * 0.15)


def _sweep_row(
    fig,
    grid_row: int,
    *,
    title: str,
    xlabel: str,
    llama_path: Path,
    qwen_path: Path,
) -> None:
    series = {
        "Llama 3.2-1B": load_sweep(llama_path),
        "Qwen1.5-MoE-A2.7B": load_sweep(qwen_path),
    }
    panels = [
        ("hbm_reduction_pct", HBM_REDUCTION_YLABEL, False),
        ("latency_improvement_pct", LATENCY_IMPROVEMENT_YLABEL, False),
        ("energy_reduction_pct", ENERGY_REDUCTION_YLABEL, True),
    ]
    for col, (metric, ylabel, signed) in enumerate(panels):
        ax = fig.add_subplot(2, 3, grid_row * 3 + col + 1)
        _plot_sweep_lines(
            ax,
            series,
            metric=metric,
            xlabel=xlabel if col == 1 else "",
            title=f"{title} — {ylabel.split('(')[0].strip()}",
            ylabel=ylabel,
            signed=signed,
        )
        if signed and col == 2:
            ax.text(
                0.02,
                0.02,
                ENERGY_NOTE,
                transform=ax.transAxes,
                fontsize=7,
                color="0.35",
                va="bottom",
            )


def plot_presentation_sweep_dashboard() -> Path:
    """M2 + M3 sweeps: HBM, latency, energy in one 2×3 figure."""
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        "M2 & M3 — Area sweeps (best_case): HBM, latency, and energy vs baseline",
        fontsize=13,
        y=0.98,
    )
    _sweep_row(
        fig,
        0,
        title="M2 StRAM-only (1T1C)",
        xlabel="SBUF area → StRAM",
        llama_path=LLAMA_SWEEP / "consolidated_stram_edram_1t1c.json",
        qwen_path=QWEN_SWEEP / "consolidated_stram_edram_1t1c.json",
    )
    _sweep_row(
        fig,
        1,
        title="M3 LtRAM-only (RRAM)",
        xlabel="HBM area → LtRAM",
        llama_path=LLAMA_SWEEP / "consolidated_ltram_rram.json",
        qwen_path=QWEN_SWEEP / "consolidated_ltram_rram.json",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "presentation_m2_m3_sweep_dashboard.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_presentation_m4_dashboard() -> Path:
    """M4: HBM, latency, energy for all hierarchy configs."""
    metrics = [
        ("hbm_reduction_pct", HBM_REDUCTION_YLABEL, False),
        ("latency_improvement_pct", LATENCY_IMPROVEMENT_YLABEL, False),
        ("energy_reduction_pct", ENERGY_REDUCTION_YLABEL, True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    fig.suptitle(
        "M4 — Full hierarchy configs (`decode_tiered`, best_case)",
        fontsize=13,
        y=0.98,
    )

    for ax, (key, ylabel, signed) in zip(axes, metrics):
        llama_vals = [
            load_m4(REPO / "results" / f"m4_llama_{slug}_tiered.json")[key]
            for slug, _ in M4_CONFIGS
        ]
        qwen_vals = [
            load_m4(REPO / "results" / "m5_qwen" / f"m4_{slug}_tiered.json")[key]
            for slug, _ in M4_CONFIGS
        ]
        short = ylabel.split("(")[0].strip()
        _grouped_bars(
            ax,
            M4_PRESENTATION_LABELS,
            llama_vals,
            qwen_vals,
            ylabel=ylabel,
            title=short,
            signed=signed,
            show_legend=False,
            xtick_rotation=18,
        )
        if signed:
            ax.text(
                0.02,
                0.98,
                ENERGY_NOTE,
                transform=ax.transAxes,
                fontsize=7,
                color="0.35",
                va="top",
            )

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=LLAMA_COLOR),
        plt.Rectangle((0, 0), 1, 1, color=QWEN_COLOR),
    ]
    fig.legend(
        handles,
        ["Llama 3.2-1B", "Qwen1.5-MoE-A2.7B"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=9,
    )

    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.22, top=0.86, wspace=0.28)
    out = OUT_DIR / "presentation_m4_dashboard.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_presentation_tech_dashboard() -> Path:
    """StRAM + LtRAM technology comparisons with three metrics each."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Technology comparison — StRAM (1T1C vs 3T) and LtRAM (RRAM vs FeRAM), Llama",
        fontsize=13,
        y=0.98,
    )

    rows = [
        (
            "M2 StRAM (Llama)",
            LLAMA_SWEEP / "consolidated_stram_edram_1t1c.json",
            LLAMA_SWEEP / "consolidated_stram_edram_3t.json",
            "1T1C eDRAM",
            "3T eDRAM",
            "SBUF area → StRAM",
        ),
        (
            "M3 LtRAM (Llama)",
            LLAMA_SWEEP / "consolidated_ltram_rram.json",
            LLAMA_SWEEP / "consolidated_ltram_feram.json",
            "RRAM",
            "FeRAM",
            "HBM area → LtRAM",
        ),
    ]
    panels = [
        ("hbm_reduction_pct", HBM_REDUCTION_YLABEL, False),
        ("latency_improvement_pct", LATENCY_IMPROVEMENT_YLABEL, False),
        ("energy_reduction_pct", ENERGY_REDUCTION_YLABEL, True),
    ]

    for row_idx, (row_title, path_a, path_b, label_a, label_b, xlabel) in enumerate(rows):
        series = {label_a: load_sweep(path_a), label_b: load_sweep(path_b)}
        for col_idx, (metric, ylabel, signed) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            _plot_sweep_lines(
                ax,
                series,
                metric=metric,
                xlabel=xlabel if col_idx == 1 else "",
                title=f"{row_title}",
                ylabel=ylabel.split("(")[0].strip(),
                signed=signed,
            )
            if signed:
                ax.text(0.02, 0.02, ENERGY_NOTE, transform=ax.transAxes, fontsize=6, color="0.35")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = OUT_DIR / "presentation_tech_dashboard.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_presentation_cross_milestone_dashboard() -> Path:
    """Best headline per milestone: HBM, latency, energy side by side."""

    def milestone_vals(key: str) -> tuple[list[float], list[float]]:
        llama = [
            load_sweep(LLAMA_SWEEP / "consolidated_stram_edram_1t1c.json")[50][key],
            load_sweep(LLAMA_SWEEP / "consolidated_ltram_rram.json")[25][key],
            max(
                load_m4(REPO / "results" / f"m4_llama_{slug}_tiered.json")[key]
                for slug, _ in M4_CONFIGS
            ),
        ]
        qwen = [
            load_sweep(QWEN_SWEEP / "consolidated_stram_edram_1t1c.json")[75][key],
            load_sweep(QWEN_SWEEP / "consolidated_ltram_rram.json")[75][key],
            max(
                load_m4(REPO / "results" / "m5_qwen" / f"m4_{slug}_tiered.json")[key]
                for slug, _ in M4_CONFIGS
            ),
        ]
        return llama, qwen

    milestones = ["M2\nStRAM", "M3\nLtRAM", "M4\nFull"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.suptitle(
        "Cross-milestone headlines (best_case) — best config per tier",
        fontsize=13,
        y=1.02,
    )

    for ax, (key, ylabel, signed) in zip(
        axes,
        [
            ("hbm_reduction_pct", HBM_REDUCTION_YLABEL, False),
            ("latency_improvement_pct", LATENCY_IMPROVEMENT_YLABEL, False),
            ("energy_reduction_pct", ENERGY_REDUCTION_YLABEL, True),
        ],
    ):
        llama, qwen = milestone_vals(key)
        _grouped_bars(
            ax,
            milestones,
            llama,
            qwen,
            ylabel=ylabel,
            title=ylabel.split("(")[0].strip(),
            signed=signed,
            annotate=True,
        )
        if signed:
            ax.text(0.02, 0.02, ENERGY_NOTE, transform=ax.transAxes, fontsize=7, color="0.35")

    fig.tight_layout()
    out = OUT_DIR / "presentation_cross_milestone_dashboard.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m1_baseline() -> Path:
    llama = load_sweep(LLAMA_SWEEP / "consolidated_ltram_rram.json")
    qwen = load_sweep(QWEN_SWEEP / "consolidated_ltram_rram.json")
    l10 = llama[FRACTIONS[0]]
    q10 = qwen[FRACTIONS[0]]
    with (LLAMA_SWEEP / "consolidated_ltram_rram.json").open() as f:
        bl_energy_llama = json.load(f)["runs"][0]["comparison"]["energy_pJ"]["baseline"] / 1e12
    with (QWEN_SWEEP / "consolidated_ltram_rram.json").open() as f:
        bl_energy_qwen = json.load(f)["runs"][0]["comparison"]["energy_pJ"]["baseline"] / 1e12

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8))
    models = ["Llama 3.2-1B", "Qwen1.5-MoE-A2.7B"]
    colors = [LLAMA_COLOR, QWEN_COLOR]

    for ax, vals, ylabel in [
        (axes[0], [l10["baseline_time_ms"], q10["baseline_time_ms"]], "Worst-core time (ms)"),
        (axes[1], [l10["baseline_hbm_mb"], q10["baseline_hbm_mb"]], "HBM traffic (MB)"),
        (axes[2], [bl_energy_llama, bl_energy_qwen], "Total energy (TJ)"),
        (axes[3], [4070, 8409], "hbm→sbuf hop count"),
    ]:
        ax.bar(models, vals, color=colors, width=0.55)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for i, v in enumerate(vals):
            fmt = f"{v:.1f}" if isinstance(v, float) and v > 100 else f"{v:.2f}" if isinstance(v, float) else str(v)
            ax.text(i, v, fmt, ha="center", va="bottom", fontsize=8)

    fig.suptitle("Milestone 1 — Baseline Trainium2 (trn2.3xlarge, 4 cores)", fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "m1_baseline_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m2_stram_model_compare() -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _plot_sweep_lines(
        ax,
        {
            "Llama 3.2-1B": load_sweep(LLAMA_SWEEP / "consolidated_stram_edram_1t1c.json"),
            "Qwen1.5-MoE-A2.7B": load_sweep(QWEN_SWEEP / "consolidated_stram_edram_1t1c.json"),
        },
        metric="hbm_reduction_pct",
        xlabel="SBUF die area traded to StRAM (1T1C eDRAM)",
        title="M2 — StRAM-only: Llama vs Qwen (best_case)",
        ylabel=HBM_REDUCTION_YLABEL,
    )
    fig.tight_layout()
    out = OUT_DIR / "m2_stram_hbm_reduction.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m2_stram_tech_compare() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, sweep_dir, model in [
        (axes[0], LLAMA_SWEEP, "Llama 3.2-1B"),
        (axes[1], QWEN_SWEEP, "Qwen1.5-MoE-A2.7B"),
    ]:
        _plot_sweep_lines(
            ax,
            {
                "1T1C eDRAM": load_sweep(sweep_dir / "consolidated_stram_edram_1t1c.json"),
                "3T eDRAM": load_sweep(sweep_dir / "consolidated_stram_edram_3t.json"),
            },
            metric="hbm_reduction_pct",
            xlabel="SBUF area → StRAM",
            title=f"M2 — StRAM tech compare ({model})",
            ylabel=HBM_REDUCTION_YLABEL,
        )
    fig.tight_layout()
    out = OUT_DIR / "m2_stram_tech_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m3_ltram_model_compare() -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    _plot_sweep_lines(
        ax,
        {
            "Llama 3.2-1B": load_sweep(LLAMA_SWEEP / "consolidated_ltram_rram.json"),
            "Qwen1.5-MoE-A2.7B": load_sweep(QWEN_SWEEP / "consolidated_ltram_rram.json"),
        },
        metric="hbm_reduction_pct",
        xlabel="HBM die area traded to LtRAM (RRAM)",
        title="M3 — LtRAM-only: Llama vs Qwen (best_case)",
        ylabel=HBM_REDUCTION_YLABEL,
    )
    fig.tight_layout()
    out = OUT_DIR / "m3_ltram_hbm_reduction.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m3_ltram_tech_compare() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, sweep_dir, model in [
        (axes[0], LLAMA_SWEEP, "Llama 3.2-1B"),
        (axes[1], QWEN_SWEEP, "Qwen1.5-MoE-A2.7B"),
    ]:
        _plot_sweep_lines(
            ax,
            {
                "RRAM": load_sweep(sweep_dir / "consolidated_ltram_rram.json"),
                "FeRAM": load_sweep(sweep_dir / "consolidated_ltram_feram.json"),
            },
            metric="hbm_reduction_pct",
            xlabel="HBM area → LtRAM",
            title=f"M3 — LtRAM tech compare ({model})",
            ylabel=HBM_REDUCTION_YLABEL,
        )
    fig.tight_layout()
    out = OUT_DIR / "m3_ltram_tech_comparison.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m4_hierarchy() -> Path:
    llama_vals = []
    qwen_vals = []
    labels = []

    for slug, label in M4_CONFIGS:
        labels.append(label)
        llama_vals.append(
            load_m4(REPO / "results" / f"m4_llama_{slug}_tiered.json")["hbm_reduction_pct"]
        )
        qwen_vals.append(
            load_m4(REPO / "results" / "m5_qwen" / f"m4_{slug}_tiered.json")["hbm_reduction_pct"]
        )

    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, llama_vals, width, label="Llama 3.2-1B", color=LLAMA_COLOR)
    ax.bar(x + width / 2, qwen_vals, width, label="Qwen1.5-MoE-A2.7B", color=QWEN_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Full hierarchy design (StRAM fraction / LtRAM fraction)")
    _style_axes(
        ax,
        ylabel=HBM_REDUCTION_YLABEL,
        title="M4 — Full hierarchy configs (`decode_tiered`, best_case)",
        reduction=True,
    )
    ax.legend(frameon=False, loc="upper right")

    for i, (lv, qv) in enumerate(zip(llama_vals, qwen_vals)):
        ax.text(i - width / 2, lv + 0.4, f"{lv:.1f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, qv + 0.4, f"{qv:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = OUT_DIR / "m4_hierarchy_hbm_reduction.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_m4_ltram_fills() -> Path:
    """M4 evidence: ltram→sbuf fills by config."""
    labels = []
    llama_fills = []
    qwen_fills = []

    for slug, label in M4_CONFIGS:
        labels.append(label.replace("\n", " "))
        llama_fills.append(load_m4(REPO / "results" / f"m4_llama_{slug}_tiered.json")["ltram_sbuf"])
        qwen_fills.append(load_m4(REPO / "results" / "m5_qwen" / f"m4_{slug}_tiered.json")["ltram_sbuf"])

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - width / 2, llama_fills, width, label="Llama", color=LLAMA_COLOR)
    ax.bar(x + width / 2, qwen_fills, width, label="Qwen", color=QWEN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("`ltram→sbuf` transfer count")
    ax.set_title("M4 — Weight offload evidence (`ltram→sbuf` fills)")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "m4_ltram_fills_by_config.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cross_milestone_headlines() -> Path:
    """Synthesis: best HBM reduction per milestone per model."""
    llama_m2 = load_sweep(LLAMA_SWEEP / "consolidated_stram_edram_1t1c.json")[50]["hbm_reduction_pct"]
    llama_m3 = load_sweep(LLAMA_SWEEP / "consolidated_ltram_rram.json")[25]["hbm_reduction_pct"]
    llama_m4 = max(
        load_m4(REPO / "results" / f"m4_llama_{slug}_tiered.json")["hbm_reduction_pct"]
        for slug, _ in M4_CONFIGS
    )

    qwen_m2 = load_sweep(QWEN_SWEEP / "consolidated_stram_edram_1t1c.json")[75]["hbm_reduction_pct"]
    qwen_m3 = load_sweep(QWEN_SWEEP / "consolidated_ltram_rram.json")[75]["hbm_reduction_pct"]
    qwen_m4 = max(
        load_m4(REPO / "results" / "m5_qwen" / f"m4_{slug}_tiered.json")["hbm_reduction_pct"]
        for slug, _ in M4_CONFIGS
    )

    milestones = ["M2 StRAM", "M3 LtRAM", "M4 Full"]
    llama = [llama_m2, llama_m3, llama_m4]
    qwen = [qwen_m2, qwen_m3, qwen_m4]

    x = np.arange(len(milestones))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - width / 2, llama, width, label="Llama", color=LLAMA_COLOR)
    ax.bar(x + width / 2, qwen, width, label="Qwen", color=QWEN_COLOR)
    ax.set_xticks(x)
    ax.set_xticklabels(milestones)
    _style_axes(
        ax,
        ylabel=HBM_REDUCTION_YLABEL,
        title="Cross-milestone headline (best_case)",
        reduction=True,
    )
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    out = OUT_DIR / "cross_milestone_best_hbm_reduction.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots = [
        # Presentation dashboards (recommended for slides)
        plot_presentation_sweep_dashboard(),
        plot_presentation_m4_dashboard(),
        plot_presentation_tech_dashboard(),
        plot_presentation_cross_milestone_dashboard(),
        # Individual figures (optional detail)
        plot_m1_baseline(),
        plot_m2_stram_model_compare(),
        plot_m2_stram_tech_compare(),
        plot_m3_ltram_model_compare(),
        plot_m3_ltram_tech_compare(),
        plot_m4_hierarchy(),
        plot_m4_ltram_fills(),
        plot_cross_milestone_headlines(),
    ]
    for p in plots:
        print(f"Wrote {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
