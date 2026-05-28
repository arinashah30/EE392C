#!/usr/bin/env python3
"""
Visualize a dmsim ingested trace (JSON from `dmsim ingest`).

Produces PNG figures under an output directory:
  - trace_summary.png — tensor counts/bytes by category, optional access bytes
  - trace_top_tensors.png — largest tensors by static size (colored by category)

Requires: matplotlib (and numpy, pulled in by matplotlib)

  pip install matplotlib

  python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core.json
  python profiler/visualize_trace.py data/traces/llama32_1b_decode_4core.json -o profiler/out/llama32_viz

  # Ingest without synthetic hbm_traffic_* from unknown DMA, then visualize:
  python -m dmsim.cli ingest --profile-dir ... --model-key 446048307616134 \\
    --skip-unattributed-dma --output data/traces/llama32_decode_no_unknown.json
  python profiler/visualize_trace.py data/traces/llama32_decode_no_unknown.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _human_bytes(n: int) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n} B"


def _load_trace(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _aggregate_tensors(data: dict) -> tuple[dict[str, int], dict[str, int]]:
    """category -> (count, total_bytes)"""
    counts: dict[str, int] = defaultdict(int)
    bytes_: dict[str, int] = defaultdict(int)
    for t in data.get("tensors") or []:
        cat = str(t.get("category") or "other")
        counts[cat] += 1
        bytes_[cat] += int(t.get("bytes") or 0)
    return dict(counts), dict(bytes_)


def _aggregate_accesses(data: dict) -> tuple[dict[str, int], int]:
    """category -> total access bytes; second return = number of access events."""
    tensors = {t["id"]: t for t in data.get("tensors") or []}
    by_cat: dict[str, int] = defaultdict(int)
    n_access = 0
    for ev in data.get("events") or []:
        if ev.get("type") != "access":
            continue
        n_access += 1
        tid = ev.get("tensor_id") or ""
        rec = tensors.get(tid) or {}
        cat = str(rec.get("category") or "other")
        by_cat[cat] += int(ev.get("bytes") or 0)
    return dict(by_cat), n_access


def _top_tensors(data: dict, limit: int) -> list[dict]:
    rows = list(data.get("tensors") or [])
    rows.sort(key=lambda t: int(t.get("bytes") or 0), reverse=True)
    return rows[:limit]


def _plot_summary(
    out_path: Path,
    counts: dict[str, int],
    bytes_by_cat: dict[str, int],
    access_bytes: dict[str, int],
    n_access: int,
    workload: str,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    cats = sorted(set(counts) | set(bytes_by_cat) | set(access_bytes))
    if not cats:
        cats = ["other"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(f"dmsim trace — {workload or '(no workload name)'}", fontsize=12)

    # Counts
    ax = axes[0]
    vals = [counts.get(c, 0) for c in cats]
    ax.barh(cats, vals, color="#3498db")
    ax.set_xlabel("tensor count")
    ax.set_title("Tensors by category")

    # Static bytes
    ax = axes[1]
    mb = [bytes_by_cat.get(c, 0) / (1 << 20) for c in cats]
    ax.barh(cats, mb, color="#2ecc71")
    ax.set_xlabel("MiB (sum of tensor.bytes)")
    ax.set_title("Declared tensor size by category")

    # Access bytes
    ax = axes[2]
    if n_access == 0:
        ax.text(0.5, 0.5, "No access events\n(kernel boundaries only)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        ab = [access_bytes.get(c, 0) / (1 << 20) for c in cats]
        ax.barh(cats, ab, color="#e67e22")
        ax.set_xlabel("MiB (sum of access.bytes)")
        ax.set_title(f"Access traffic by category (n={n_access})")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _category_color(cat: str) -> str:
    return {
        "weight": "#9b59b6",
        "kv_cache": "#e74c3c",
        "activation": "#3498db",
        "hidden": "#1abc9c",
        "other": "#95a5a6",
    }.get(cat, "#7f8c8d")


def _plot_top_tensors(out_path: Path, top: list[dict], dpi: int) -> None:
    import matplotlib.pyplot as plt

    if not top:
        return
    labels = [t.get("name") or t.get("id", "?") for t in top]
    # shorten for display
    labels = [s[:28] + "…" if len(s) > 30 else s for s in labels]
    cats = [str(t.get("category") or "other") for t in top]
    sizes = [int(t.get("bytes") or 0) for t in top]
    mb = [s / (1 << 20) for s in sizes]
    colors = [_category_color(c) for c in cats]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
    ax.barh(labels[::-1], mb[::-1], color=colors[::-1])
    ax.set_xlabel("MiB (tensor.bytes)")
    ax.set_title(f"Top {len(top)} tensors by static size")
    from matplotlib.patches import Patch

    legend_cats = sorted(set(cats))
    ax.legend(
        handles=[Patch(color=_category_color(c), label=c) for c in legend_cats],
        loc="lower right",
        fontsize=8,
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot dmsim ingested trace JSON")
    parser.add_argument("trace", type=Path, help="Path to ingested trace .json")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNGs (default: profiler/out/<trace_stem>_viz/)",
    )
    parser.add_argument("--top", type=int, default=20, help="How many tensors in top-tensors chart")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    trace_path = args.trace.resolve()
    if not trace_path.is_file():
        print(f"error: not a file: {trace_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output_dir
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / "out" / f"{trace_path.stem}_viz"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib  # noqa: F401

        matplotlib.use("Agg")
    except ImportError:
        print("error: matplotlib is required. Run: pip install matplotlib", file=sys.stderr)
        sys.exit(1)

    data = _load_trace(trace_path)
    meta = data.get("metadata") or {}
    workload = str(meta.get("workload") or trace_path.stem)

    counts, bytes_by_cat = _aggregate_tensors(data)
    access_bytes, n_access = _aggregate_accesses(data)

    summary_png = out_dir / "trace_summary.png"
    _plot_summary(summary_png, counts, bytes_by_cat, access_bytes, n_access, workload, args.dpi)

    top_png = out_dir / "trace_top_tensors.png"
    _plot_top_tensors(top_png, _top_tensors(data, args.top), args.dpi)

    print(f"Wrote {summary_png}")
    print(f"Wrote {top_png}")
    print("  totals:", {k: _human_bytes(v) for k, v in sorted(bytes_by_cat.items())})
    print(f"  access events: {n_access}")


if __name__ == "__main__":
    main()
