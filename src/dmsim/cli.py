from __future__ import annotations

import argparse
import json
from pathlib import Path

from dmsim.config.loader import _repo_root, load_hierarchy, load_policy
from dmsim.config.snapshot import build_compare_report, build_run_report
from dmsim.metrics.report import compare_results, format_report, write_report_json
from dmsim.sim.engine import run_simulation
from dmsim.trace.schema import Trace, load_trace
from dmsim.trace.neuron_json_ingest import (
    IngestOptions,
    ingest_and_write,
    list_neuron_cores,
)


def _resolve(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _load_hierarchy_for_trace(root: Path, hierarchy_path: Path, trace: Trace):
    hierarchy = load_hierarchy(
        _resolve(root, hierarchy_path),
        repo_root=root,
        num_cores=trace.metadata.num_neuron_cores,
    )
    return hierarchy


def _print_area_budget(hierarchy) -> None:
    if not hierarchy.area_budget_notes:
        return
    print("=== area budget (constant die area) ===")
    for key, value in sorted(hierarchy.area_budget_notes.items()):
        print(f"  {key}: {value}")
    for level in hierarchy.enabled_levels:
        if level.id in ("sbuf", "hbm", "stram", "ltram"):
            print(f"  {level.id}_capacity_bytes: {level.capacity_bytes}")
    print()


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
    ingest_parser.add_argument(
        "--nc",
        type=int,
        default=None,
        help="Single NeuronCore id (default: ingest all cores in profile)",
    )
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
        default=200_000,
        help="Cap access events after ingest (0 = no cap)",
    )
    ingest_parser.add_argument(
        "--skip-unattributed-dma",
        action="store_true",
        help="Drop unknown dynamic DMA; do not synthesize hbm_traffic_* tensors",
    )

    pipe_parser = sub.add_parser(
        "pipeline", help="Ingest all NeuronCores then run or compare simulation"
    )
    pipe_parser.add_argument("--profile-dir", type=Path, required=True)
    pipe_parser.add_argument("--trace-cache", type=Path, default=None)
    pipe_parser.add_argument("--model-key", type=str, default=None)
    pipe_parser.add_argument("--min-transfer-bytes", type=int, default=64)
    pipe_parser.add_argument("--hierarchy", type=Path, default=None)
    pipe_parser.add_argument("--policy", type=Path, default=None)
    pipe_parser.add_argument(
        "--baseline-hierarchy",
        type=Path,
        default=None,
        help="If set with --candidate-hierarchy, run compare after ingest",
    )
    pipe_parser.add_argument("--candidate-hierarchy", type=Path, default=None)
    pipe_parser.add_argument(
        "--baseline-policy",
        type=Path,
        default="configs/policies/baseline_hbm.yaml",
    )
    pipe_parser.add_argument(
        "--candidate-policy",
        type=Path,
        default="configs/policies/decode_tiered.yaml",
    )
    pipe_parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()
    root = _repo_root()

    if args.command == "ingest":
        all_cores = args.nc is None
        opts = IngestOptions(
            neuron_core_id=args.nc if args.nc is not None else 0,
            all_cores=all_cores,
            model_key=args.model_key,
            min_transfer_bytes=args.min_transfer_bytes,
            aggregate_dma=not args.no_aggregate_dma,
            max_access_events=args.max_access_events or None,
            skip_unattributed_dma=args.skip_unattributed_dma,
        )
        trace = ingest_and_write(
            _resolve(root, args.profile_dir),
            _resolve(root, args.output),
            options=opts,
        )
        cores = trace.metadata.neuron_core_ids or [trace.metadata.neuron_core_id]
        print(
            f"Wrote {args.output} — {len(trace.tensors)} tensors, "
            f"{len(trace.events)} events, cores={cores}"
        )
        return

    if args.command == "pipeline":
        profile_dir = _resolve(root, args.profile_dir)
        cache = args.trace_cache or (root / "data/traces/ingested_all_cores.json")
        opts = IngestOptions(
            all_cores=True,
            model_key=args.model_key,
            min_transfer_bytes=args.min_transfer_bytes,
        )
        trace = ingest_and_write(profile_dir, cache, options=opts)
        print(
            f"Ingested {cache} — {len(trace.tensors)} tensors, "
            f"{len(trace.events)} events, cores={trace.metadata.neuron_core_ids}"
        )

        if args.baseline_hierarchy and args.candidate_hierarchy:
            baseline_h = _load_hierarchy_for_trace(
                root, args.baseline_hierarchy, trace
            )
            candidate_h = _load_hierarchy_for_trace(
                root, args.candidate_hierarchy, trace
            )
            baseline_p = load_policy(_resolve(root, args.baseline_policy))
            candidate_p = load_policy(_resolve(root, args.candidate_policy))
            _print_area_budget(candidate_h)
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
                    _resolve(root, args.output),
                    build_compare_report(
                        baseline_hierarchy=baseline_h,
                        candidate_hierarchy=candidate_h,
                        baseline_policy=baseline_p,
                        candidate_policy=candidate_p,
                        trace=trace,
                        baseline_result=baseline_r,
                        candidate_result=candidate_r,
                        comparison=comparison,
                        paths={
                            "baseline_hierarchy": _resolve(
                                root, args.baseline_hierarchy
                            ),
                            "candidate_hierarchy": _resolve(
                                root, args.candidate_hierarchy
                            ),
                            "baseline_policy": _resolve(root, args.baseline_policy),
                            "candidate_policy": _resolve(root, args.candidate_policy),
                        },
                    ),
                )
            return

        if not args.hierarchy or not args.policy:
            raise SystemExit(
                "pipeline requires --hierarchy and --policy, or "
                "--baseline-hierarchy and --candidate-hierarchy"
            )
        hierarchy = _load_hierarchy_for_trace(root, args.hierarchy, trace)
        policy = load_policy(_resolve(root, args.policy))
        _print_area_budget(hierarchy)
        result = run_simulation(trace, hierarchy, policy)
        print(format_report(result))
        return

    if args.command == "run":
        trace = load_trace(_resolve(root, args.trace))
        hierarchy = _load_hierarchy_for_trace(root, args.hierarchy, trace)
        policy = load_policy(_resolve(root, args.policy))
        _print_area_budget(hierarchy)
        result = run_simulation(trace, hierarchy, policy)
        print(format_report(result))
        if args.output:
            write_report_json(
                _resolve(root, args.output),
                build_run_report(
                    hierarchy=hierarchy,
                    policy=policy,
                    trace=trace,
                    result=result,
                    hierarchy_path=_resolve(root, args.hierarchy),
                    policy_path=_resolve(root, args.policy),
                ),
            )
        return

    if args.command == "compare":
        trace = load_trace(_resolve(root, args.trace))
        baseline_h = _load_hierarchy_for_trace(
            root, args.baseline_hierarchy, trace
        )
        candidate_h = _load_hierarchy_for_trace(
            root, args.candidate_hierarchy, trace
        )
        baseline_p = load_policy(_resolve(root, args.baseline_policy))
        candidate_p = load_policy(_resolve(root, args.candidate_policy))

        _print_area_budget(candidate_h)
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
                _resolve(root, args.output),
                build_compare_report(
                    baseline_hierarchy=baseline_h,
                    candidate_hierarchy=candidate_h,
                    baseline_policy=baseline_p,
                    candidate_policy=candidate_p,
                    trace=trace,
                    baseline_result=baseline_r,
                    candidate_result=candidate_r,
                    comparison=comparison,
                    paths={
                        "baseline_hierarchy": _resolve(root, args.baseline_hierarchy),
                        "candidate_hierarchy": _resolve(root, args.candidate_hierarchy),
                        "baseline_policy": _resolve(root, args.baseline_policy),
                        "candidate_policy": _resolve(root, args.candidate_policy),
                    },
                ),
            )


if __name__ == "__main__":
    main()
