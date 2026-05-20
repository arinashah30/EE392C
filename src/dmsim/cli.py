from __future__ import annotations

import argparse
import json
from pathlib import Path

from dmsim.config.loader import _repo_root, load_hierarchy, load_policy
from dmsim.metrics.report import compare_results, format_report, write_report_json
from dmsim.sim.engine import run_simulation
from dmsim.trace.schema import load_trace
from dmsim.trace.neuron_json_ingest import IngestOptions, ingest_and_write, list_neuron_cores


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differentiated memory hierarchy simulator (Trainium2)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run_parser = sub.add_parser("run", help="Run a single simulation")
    run_parser.add_argument("--hierarchy", type=Path, required=True)
    run_parser.add_argument("--policy", type=Path, required=True)
    run_parser.add_argument("--trace", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, default=None)

    cmp_parser = sub.add_parser("compare", help="Baseline vs candidate on same trace")
    cmp_parser.add_argument("--baseline-hierarchy", type=Path, required=True)
    cmp_parser.add_argument("--candidate-hierarchy", type=Path, required=True)
    cmp_parser.add_argument("--baseline-policy", type=Path, required=True)
    cmp_parser.add_argument("--candidate-policy", type=Path, required=True)
    cmp_parser.add_argument("--trace", type=Path, required=True)
    cmp_parser.add_argument("--output", type=Path, default=None)

    ingest_parser = sub.add_parser(
        "ingest", help="Convert Neuron Explorer JSON profile to a dmsim trace"
    )
    ingest_parser.add_argument(
        "--profile-dir",
        type=Path,
        required=True,
        help="Directory containing profile.json and per-NeuronCore JSON files",
    )
    ingest_parser.add_argument("--output", type=Path, required=True)
    ingest_parser.add_argument("--nc", type=int, default=0, help="NeuronCore id")
    ingest_parser.add_argument(
        "--model-key",
        type=str,
        default=None,
        help="Substring to select among multiple NEFF captures (e.g. model hash)",
    )
    ingest_parser.add_argument(
        "--min-transfer-bytes",
        type=int,
        default=64,
        help="Ignore DMA transfers smaller than this",
    )
    ingest_parser.add_argument(
        "--no-aggregate-dma",
        action="store_true",
        help="Emit one access per DMA record (very large traces)",
    )
    ingest_parser.add_argument(
        "--max-access-events",
        type=int,
        default=50_000,
        help="Cap access events after ingest (0 = no cap)",
    )

    pipe_parser = sub.add_parser(
        "pipeline", help="Ingest Neuron JSON profile then run simulation"
    )
    pipe_parser.add_argument("--profile-dir", type=Path, required=True)
    pipe_parser.add_argument("--hierarchy", type=Path, required=True)
    pipe_parser.add_argument("--policy", type=Path, required=True)
    pipe_parser.add_argument("--nc", type=int, default=0)
    pipe_parser.add_argument("--model-key", type=str, default=None)
    pipe_parser.add_argument("--trace-cache", type=Path, default=None)
    pipe_parser.add_argument("--min-transfer-bytes", type=int, default=64)

    args = parser.parse_args()
    root = _repo_root()

    if args.command == "ingest":
        opts = IngestOptions(
            neuron_core_id=args.nc,
            model_key=args.model_key,
            min_transfer_bytes=args.min_transfer_bytes,
            aggregate_dma=not args.no_aggregate_dma,
            max_access_events=args.max_access_events or None,
        )
        trace = ingest_and_write(
            _resolve(root, args.profile_dir),
            _resolve(root, args.output),
            options=opts,
        )
        print(
            f"Wrote {args.output} — {len(trace.tensors)} tensors, "
            f"{len(trace.events)} events (nc={args.nc})"
        )
        return

    if args.command == "pipeline":
        profile_dir = _resolve(root, args.profile_dir)
        cache = args.trace_cache or (
            root / "data/traces" / f"ingested_nc{args.nc}.json"
        )
        opts = IngestOptions(
            neuron_core_id=args.nc,
            model_key=args.model_key,
            min_transfer_bytes=args.min_transfer_bytes,
        )
        trace = ingest_and_write(profile_dir, cache, options=opts)
        hierarchy = load_hierarchy(_resolve(root, args.hierarchy), repo_root=root)
        policy = load_policy(_resolve(root, args.policy))
        result = run_simulation(trace, hierarchy, policy)
        print(format_report(result))
        return

    if args.command == "run":
        trace = load_trace(args.trace)
        hierarchy = load_hierarchy(_resolve(root, args.hierarchy), repo_root=root)
        policy = load_policy(_resolve(root, args.policy))
        result = run_simulation(trace, hierarchy, policy)
        text = format_report(result)
        print(text)
        if args.output:
            write_report_json(
                args.output,
                {**result.__dict__, "transfers_by_hop": result.transfers_by_hop},
            )
        return

    if args.command == "compare":
        trace = load_trace(args.trace)
        baseline_h = load_hierarchy(_resolve(root, args.baseline_hierarchy), repo_root=root)
        candidate_h = load_hierarchy(_resolve(root, args.candidate_hierarchy), repo_root=root)
        baseline_p = load_policy(_resolve(root, args.baseline_policy))
        candidate_p = load_policy(_resolve(root, args.candidate_policy))

        baseline_r = run_simulation(trace, baseline_h, baseline_p)
        candidate_r = run_simulation(trace, candidate_h, candidate_p)

        print(format_report(baseline_r))
        print()
        print(format_report(candidate_r))
        print()
        comparison = compare_results(baseline_r, candidate_r)
        print("=== comparison (candidate vs baseline) ===")
        print(json.dumps(comparison, indent=2))

        if args.output:
            write_report_json(
                args.output,
                {
                    "baseline": baseline_r.__dict__,
                    "candidate": candidate_r.__dict__,
                    "comparison": comparison,
                },
            )


def _resolve(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


if __name__ == "__main__":
    main()
