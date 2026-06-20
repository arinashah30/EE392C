#!/usr/bin/env python3
"""Session-lifetime box plots from lifetime analysis JSON (Llama vs Qwen style)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from dmsim.trace.lifetime_analysis import analyze_trace_lifetimes, result_to_dict
from dmsim.trace.schema import AccessEvent, load_trace

OUT_DIR = REPO / "results" / "plots"
DEFAULT_LLAMA = REPO / "results/lifetime_llama.json"
DEFAULT_QWEN = REPO / "results/lifetime_qwen.json"
DEFAULT_LLAMA_TRACE = REPO / "data/traces/llama32_1b_decode_4core_dge_kv.json"
DEFAULT_QWEN_TRACE = REPO / "data/traces/qwen1_5_moe_decode_4core_dge_v2.json"

CATEGORY_ORDER = ["weight", "kv_cache", "activation", "other", "hidden", "unknown"]
CATEGORY_COLORS = {
    "weight": "#3b82f6",
    "kv_cache": "#ef4444",
    "activation": "#10b981",
    "other": "#9ca3af",
    "hidden": "#a855f7",
    "unknown": "#d1d5db",
}
MODEL_TITLE_COLORS = {
    "llama": "#2563eb",
    "qwen": "#dc2626",
}


def _format_lifetime_ms(value: float) -> str:
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    if value >= 1:
        return f"{value:.3f}"
    return f"{value:.3g}"


def load_lifetime_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def lifetime_json_from_trace(trace_path: Path, cache_path: Path | None) -> dict:
    if cache_path and cache_path.exists():
        return load_lifetime_json(cache_path)
    trace = load_trace(trace_path)
    payload = result_to_dict(analyze_trace_lifetimes(trace))
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as handle:
            json.dump(payload, handle, indent=2)
    return payload


def filter_tensors(
    tensors: list[dict],
    *,
    exclude_zero_lifetime: bool,
    min_lifetime_ms: float,
    multi_access_only: bool,
) -> list[dict]:
    kept: list[dict] = []
    for tensor in tensors:
        lifetime_ms = float(tensor["lifetime_ms"])
        if multi_access_only and int(tensor["access_count"]) <= 1:
            continue
        if exclude_zero_lifetime and lifetime_ms <= 0:
            continue
        if lifetime_ms < min_lifetime_ms:
            continue
        kept.append(tensor)
    return kept


def lifetimes_by_category(tensors: list[dict]) -> dict[str, list[float]]:
    by_category: dict[str, list[float]] = {cat: [] for cat in CATEGORY_ORDER}
    for tensor in tensors:
        category = tensor.get("category", "unknown")
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(float(tensor["lifetime_ms"]))
    return by_category


def access_traffic_by_category(trace_path: Path) -> tuple[dict[str, float], int]:
    """Return per-category access traffic in GiB and total access-event count."""
    trace = load_trace(trace_path)
    tensor_category = {tensor.id: tensor.category.value for tensor in trace.tensors}
    bytes_by_category: dict[str, int] = {cat: 0 for cat in CATEGORY_ORDER}
    access_events = 0
    for event in trace.parsed_events():
        if not isinstance(event, AccessEvent):
            continue
        access_events += 1
        category = tensor_category.get(event.tensor_id, "unknown")
        if category not in bytes_by_category:
            bytes_by_category[category] = 0
        bytes_by_category[category] += event.bytes
    gib_by_category = {
        cat: value / (1024**3) for cat, value in bytes_by_category.items() if value > 0
    }
    return gib_by_category, access_events


def plot_access_traffic_panel(
    ax: plt.Axes,
    traffic_gib: dict[str, float],
    *,
    title: str,
    title_color: str,
    access_events: int,
) -> None:
    categories = [cat for cat in CATEGORY_ORDER if traffic_gib.get(cat, 0) > 0]
    categories.reverse()
    if not categories:
        ax.text(0.5, 0.5, "No access traffic", ha="center", va="center")
        ax.set_title(title, color=title_color, fontweight="bold")
        return

    values = [traffic_gib[cat] for cat in categories]
    colors = [CATEGORY_COLORS.get(cat, "#cbd5e1") for cat in categories]
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, values, color=colors, alpha=0.9, edgecolor="#374151", height=0.55)
    ax.set_yticks(y_pos, categories)
    ax.set_xlabel("Access traffic (GiB)")
    ax.set_title(
        f"{title}\n(n={access_events:,} access events)",
        color=title_color,
        fontweight="bold",
        pad=10,
    )
    ax.grid(True, axis="x", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="y", labelsize=10)
    xmax = max(values) * 1.08 if values else 1.0
    ax.set_xlim(0, xmax)


def plot_session_lifetime_panel(
    ax: plt.Axes,
    by_category: dict[str, list[float]],
    *,
    title: str,
    title_color: str,
    log_scale: bool,
) -> None:
    categories = [cat for cat in CATEGORY_ORDER if by_category.get(cat)]
    categories.reverse()
    if not categories:
        ax.text(0.5, 0.5, "No tensors after filter", ha="center", va="center")
        ax.set_title(title, color=title_color, fontweight="bold")
        return

    data = [by_category[cat] for cat in categories]
    labels = [f"{cat} ({len(vals)})" for cat, vals in zip(categories, data)]

    bp = ax.boxplot(
        data,
        orientation="horizontal",
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        whis=(0, 100),
        medianprops={"color": "black", "linewidth": 1.5},
        whiskerprops={"color": "#374151", "linewidth": 1.0},
        capprops={"color": "#374151", "linewidth": 1.0},
    )

    for patch, cat in zip(bp["boxes"], categories):
        patch.set_facecolor(CATEGORY_COLORS.get(cat, "#cbd5e1"))
        patch.set_alpha(0.85)
        patch.set_edgecolor("#374151")

    if log_scale:
        all_vals = [v for vals in data for v in vals]
        xmin = min(all_vals)
        xmax = max(all_vals)
        lower = 10 ** np.floor(np.log10(max(xmin * 0.8, 1e-4)))
        upper = 10 ** np.ceil(np.log10(max(xmax * 1.2, 1.0)))
        ax.set_xscale("log")
        ax.set_xlim(lower, min(upper, 1e4))
        ax.set_xlabel("Session lifetime (ms, log scale)")
    else:
        ax.set_xlabel("Session lifetime (ms)")

    ax.set_title(title, color=title_color, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.25, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="y", labelsize=10)

    whiskers = bp["whiskers"]
    for i, vals in enumerate(data):
        lo_x = min(whiskers[2 * i].get_xdata())
        hi_x = max(whiskers[2 * i + 1].get_xdata())
        y = whiskers[2 * i].get_ydata()[0]
        lo = min(vals)
        hi = max(vals)
        ax.annotate(
            _format_lifetime_ms(lo),
            xy=(lo_x, y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4b5563",
        )
        ax.annotate(
            _format_lifetime_ms(hi),
            xy=(hi_x, y),
            xytext=(0, 11),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#4b5563",
        )


def build_figure(
    panels: list[tuple[str, dict[str, list[float]], str]],
    *,
    log_scale: bool,
) -> plt.Figure:
    fig, axes = plt.subplots(
        1,
        len(panels),
        figsize=(7.5 * len(panels), 4.8),
        constrained_layout=True,
        squeeze=False,
    )
    for ax, (title, by_category, model_key) in zip(axes[0], panels):
        plot_session_lifetime_panel(
            ax,
            by_category,
            title=title,
            title_color=MODEL_TITLE_COLORS.get(model_key, "#111827"),
            log_scale=log_scale,
        )
    return fig


def build_dashboard_figure(
    panels: list[dict],
    *,
    log_scale: bool,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        len(panels),
        figsize=(7.5 * len(panels), 9.2),
        constrained_layout=True,
        squeeze=False,
    )
    if len(panels) == 1:
        axes = np.array([[axes[0, 0]], [axes[1, 0]]])

    for col, panel in enumerate(panels):
        plot_access_traffic_panel(
            axes[0, col],
            panel["traffic_gib"],
            title=panel["traffic_title"],
            title_color=MODEL_TITLE_COLORS.get(panel["model_key"], "#111827"),
            access_events=panel["access_events"],
        )
        plot_session_lifetime_panel(
            axes[1, col],
            panel["lifetimes"],
            title=panel["lifetime_title"],
            title_color=MODEL_TITLE_COLORS.get(panel["model_key"], "#111827"),
            log_scale=log_scale,
        )

    fig.suptitle(
        "Trace access traffic & session lifetimes",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    return fig


def resolve_panel(
    label: str,
    json_path: Path | None,
    trace_path: Path | None,
    cache_path: Path | None,
    title: str,
    *,
    exclude_zero_lifetime: bool,
    min_lifetime_ms: float,
    multi_access_only: bool,
) -> tuple[str, dict[str, list[float]], str]:
    if json_path:
        payload = load_lifetime_json(json_path)
    elif trace_path:
        payload = lifetime_json_from_trace(trace_path, cache_path)
    else:
        raise SystemExit(f"No input provided for panel {label!r} (--{label}-lifetime-json or --{label}-trace)")

    tensors = filter_tensors(
        payload["tensors"],
        exclude_zero_lifetime=exclude_zero_lifetime,
        min_lifetime_ms=min_lifetime_ms,
        multi_access_only=multi_access_only,
    )
    return title, lifetimes_by_category(tensors), label


def resolve_dashboard_panel(
    label: str,
    json_path: Path | None,
    trace_path: Path | None,
    cache_path: Path | None,
    traffic_title: str,
    lifetime_title: str,
    default_trace: Path,
    *,
    exclude_zero_lifetime: bool,
    min_lifetime_ms: float,
    multi_access_only: bool,
) -> dict:
    if json_path:
        payload = load_lifetime_json(json_path)
    elif trace_path:
        payload = lifetime_json_from_trace(trace_path, cache_path)
    else:
        raise SystemExit(
            f"No input provided for panel {label!r} (--{label}-lifetime-json or --{label}-trace)"
        )

    resolved_trace = trace_path or default_trace
    if not resolved_trace.exists():
        raise SystemExit(f"Trace not found for {label} traffic panel: {resolved_trace}")

    tensors = filter_tensors(
        payload["tensors"],
        exclude_zero_lifetime=exclude_zero_lifetime,
        min_lifetime_ms=min_lifetime_ms,
        multi_access_only=multi_access_only,
    )
    traffic_gib, access_events = access_traffic_by_category(resolved_trace)
    return {
        "model_key": label,
        "traffic_title": traffic_title,
        "lifetime_title": lifetime_title,
        "traffic_gib": traffic_gib,
        "access_events": access_events,
        "lifetimes": lifetimes_by_category(tensors),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot session lifetime box charts from lifetime analysis JSON"
    )
    parser.add_argument("--llama-lifetime-json", type=Path, default=DEFAULT_LLAMA)
    parser.add_argument("--qwen-lifetime-json", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--llama-trace", type=Path, default=None)
    parser.add_argument("--qwen-trace", type=Path, default=None)
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="2x2 layout: access traffic (top) + session lifetimes (bottom)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path",
    )
    parser.add_argument(
        "--exclude-zero-lifetime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop tensors whose first/last touch share the same timestamp (default: true)",
    )
    parser.add_argument(
        "--min-lifetime-ms",
        type=float,
        default=0.0,
        help="Additional lower bound after zero-lifetime filtering",
    )
    parser.add_argument(
        "--multi-access-only",
        action="store_true",
        help="Only include tensors with more than one access event",
    )
    parser.add_argument(
        "--linear-scale",
        action="store_true",
        help="Use linear x-axis instead of log scale",
    )
    parser.add_argument(
        "--llama-title",
        default="Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--qwen-title",
        default="Qwen1.5-MoE-A2.7B",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = (
            OUT_DIR / "trace_traffic_lifetime_dashboard.png"
            if args.dashboard
            else OUT_DIR / "session_lifetime_comparison.png"
        )

    filter_kwargs = {
        "exclude_zero_lifetime": args.exclude_zero_lifetime,
        "min_lifetime_ms": args.min_lifetime_ms,
        "multi_access_only": args.multi_access_only,
    }

    if args.dashboard:
        dashboard_panels: list[dict] = []
        if args.llama_lifetime_json.exists() or args.llama_trace:
            dashboard_panels.append(
                resolve_dashboard_panel(
                    "llama",
                    args.llama_lifetime_json if args.llama_lifetime_json.exists() else None,
                    args.llama_trace,
                    args.llama_lifetime_json,
                    f"Access traffic by category — {args.llama_title}",
                    f"Session lifetime — {args.llama_title}",
                    DEFAULT_LLAMA_TRACE,
                    **filter_kwargs,
                )
            )
        if args.qwen_lifetime_json.exists() or args.qwen_trace:
            dashboard_panels.append(
                resolve_dashboard_panel(
                    "qwen",
                    args.qwen_lifetime_json if args.qwen_lifetime_json.exists() else None,
                    args.qwen_trace,
                    args.qwen_lifetime_json,
                    f"Access traffic by category — {args.qwen_title}",
                    f"Session lifetime — {args.qwen_title}",
                    DEFAULT_QWEN_TRACE,
                    **filter_kwargs,
                )
            )
        if not dashboard_panels:
            raise SystemExit(
                "No lifetime inputs found. Provide JSON under results/ or pass --llama-trace / --qwen-trace."
            )
        fig = build_dashboard_figure(dashboard_panels, log_scale=not args.linear_scale)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.output}")
        return

    lifetime_panels: list[tuple[str, dict[str, list[float]], str]] = []
    if args.llama_lifetime_json.exists() or args.llama_trace:
        lifetime_panels.append(
            resolve_panel(
                "llama",
                args.llama_lifetime_json if args.llama_lifetime_json.exists() else None,
                args.llama_trace,
                args.llama_lifetime_json,
                f"Session lifetime — {args.llama_title}",
                **filter_kwargs,
            )
        )
    if args.qwen_lifetime_json.exists() or args.qwen_trace:
        lifetime_panels.append(
            resolve_panel(
                "qwen",
                args.qwen_lifetime_json if args.qwen_lifetime_json.exists() else None,
                args.qwen_trace,
                args.qwen_lifetime_json,
                f"Session lifetime — {args.qwen_title}",
                **filter_kwargs,
            )
        )

    if not lifetime_panels:
        raise SystemExit(
            "No lifetime inputs found. Provide JSON under results/ or pass --llama-trace / --qwen-trace."
        )

    fig = build_figure(lifetime_panels, log_scale=not args.linear_scale)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
