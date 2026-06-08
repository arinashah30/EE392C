#!/usr/bin/env python3
"""Run baseline vs candidate memory sweeps and write consolidated comparison JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from dmsim.config.loader import _repo_root, load_hierarchy, load_policy  # noqa: E402
from dmsim.config.models import PolicyConfig  # noqa: E402
from dmsim.config.snapshot import build_compare_report  # noqa: E402
from dmsim.metrics.report import compare_results, write_report_json  # noqa: E402
from dmsim.sim.engine import run_simulation  # noqa: E402
from dmsim.trace.schema import Trace, load_trace  # noqa: E402

FRACTIONS_PCT = (10, 25, 50, 75)
SPILL_ORDERS = ("best_case", "worst_case")

TRACE_DEFAULT = "data/traces/llama32_1b_decode_4core_min1_no_unknown.json"
BASELINE_HIERARCHY = "configs/hierarchy/trainium2_baseline.yaml"
BASELINE_POLICY = "configs/policies/baseline_hbm.yaml"
GENERATED_DIR = REPO_ROOT / "configs/hierarchy/generated"

LTRAM_POLICY = "configs/policies/decode_ltram_only.yaml"
STRAM_POLICY = "configs/policies/decode_stram_only.yaml"

LTRAM_TECHS = ("rram", "feram")
STRAM_TECHS = ("edram_1t1c", "edram_3t")


def _hierarchy_yaml_ltram(pct: int, tech: str) -> str:
    frac = pct / 100.0
    return f"""name: trainium2_ltram_{pct}pct_hbm_{tech}
description: LtRAM ({tech}) takes {pct}% of nominal HBM die area; full SBUF; StRAM off
instance: configs/instances/trn2_3xlarge.yaml

interconnect:
  dma_bandwidth_GBs: 368
  on_chip_bandwidth_GBs: 10000
  level_domain:
    psum: on_chip
    sbuf: on_chip
    stram: on_chip
    ltram: off_chip
    hbm: off_chip

levels:
  - id: psum
    enabled: true
    tech: psum_trainium2
    scope: per_core
    capacity_bytes: 2097152

  - id: sbuf
    enabled: true
    tech: tsmc_n22_sram
    scope: per_core
    capacity_bytes: 29360128

  - id: stram
    enabled: false
    tech: edram_1t1c
    scope: per_core
    capacity_bytes: 0

  - id: ltram
    enabled: true
    tech: {tech}
    scope: per_chip
    capacity_bytes: 0

  - id: hbm
    enabled: true
    tech: hbm_trainium2
    scope: per_chip

kernel:
  wipe_levels_on_boundary: [psum, sbuf]

area_budget:
  enabled: true
  nominal_sbuf_bytes_per_core: 29360128
  nominal_hbm_gib_per_chip: 96
  stram_replaces_sbuf_fraction: 0.0
  ltram_replaces_hbm_fraction: {frac}
"""


def _hierarchy_yaml_stram(pct: int, tech: str) -> str:
    frac = pct / 100.0
    return f"""name: trainium2_stram_{pct}pct_sbuf_{tech}
description: StRAM ({tech}) takes {pct}% of nominal SBUF die area per core; full HBM; LtRAM off
instance: configs/instances/trn2_3xlarge.yaml

interconnect:
  dma_bandwidth_GBs: 368
  on_chip_bandwidth_GBs: 10000
  level_domain:
    psum: on_chip
    sbuf: on_chip
    stram: on_chip
    ltram: off_chip
    hbm: off_chip

levels:
  - id: psum
    enabled: true
    tech: psum_trainium2
    scope: per_core
    capacity_bytes: 2097152

  - id: sbuf
    enabled: true
    tech: tsmc_n22_sram
    scope: per_core
    capacity_bytes: 29360128

  - id: stram
    enabled: true
    tech: {tech}
    scope: per_core
    capacity_bytes: 0

  - id: ltram
    enabled: false
    tech: rram
    scope: per_chip
    capacity_bytes: 0

  - id: hbm
    enabled: true
    tech: hbm_trainium2
    scope: per_chip

kernel:
  wipe_levels_on_boundary: [psum, sbuf]

area_budget:
  enabled: true
  nominal_sbuf_bytes_per_core: 29360128
  nominal_hbm_gib_per_chip: 96
  stram_replaces_sbuf_fraction: {frac}
  ltram_replaces_hbm_fraction: 0.0
"""


def write_hierarchy(tier: str, pct: int, tech: str, *, write_dir: Path) -> Path:
    write_dir.mkdir(parents=True, exist_ok=True)
    if tier == "ltram":
        text = _hierarchy_yaml_ltram(pct, tech)
        path = write_dir / f"trainium2_ltram_{pct}pct_hbm_{tech}.yaml"
    elif tier == "stram":
        text = _hierarchy_yaml_stram(pct, tech)
        path = write_dir / f"trainium2_stram_{pct}pct_sbuf_{tech}.yaml"
    else:
        raise ValueError(f"unknown tier: {tier}")
    path.write_text(text)
    return path


def run_one_compare(
    trace: Trace,
    baseline_h,
    candidate_h,
    baseline_p: PolicyConfig,
    candidate_p: PolicyConfig,
    *,
    root: Path,
    baseline_h_path: Path,
    candidate_h_path: Path,
    baseline_p_path: Path,
    candidate_p_path: Path,
) -> dict:
    baseline_r = run_simulation(trace, baseline_h, baseline_p)
    candidate_r = run_simulation(trace, candidate_h, candidate_p)
    comparison = compare_results(baseline_r, candidate_r)
    return build_compare_report(
        baseline_hierarchy=baseline_h,
        candidate_hierarchy=candidate_h,
        baseline_policy=baseline_p,
        candidate_policy=candidate_p,
        trace=trace,
        baseline_result=baseline_r,
        candidate_result=candidate_r,
        comparison=comparison,
        paths={
            "baseline_hierarchy": baseline_h_path,
            "candidate_hierarchy": candidate_h_path,
            "baseline_policy": baseline_p_path,
            "candidate_policy": candidate_p_path,
        },
    )


def run_tier_tech_sweep(
    tier: str,
    tech: str,
    trace: Trace,
    *,
    root: Path,
    fractions_pct: tuple[int, ...] = FRACTIONS_PCT,
    spill_orders: tuple[str, ...] = SPILL_ORDERS,
    write_hierarchies: bool = True,
    per_run_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    if tier == "ltram":
        candidate_policy_path = root / LTRAM_POLICY
    elif tier == "stram":
        candidate_policy_path = root / STRAM_POLICY
    else:
        raise ValueError(tier)

    baseline_h_path = root / BASELINE_HIERARCHY
    baseline_p_path = root / BASELINE_POLICY
    baseline_h = load_hierarchy(baseline_h_path, repo_root=root, num_cores=trace.metadata.num_neuron_cores)
    baseline_p = load_policy(baseline_p_path)
    base_candidate_p = load_policy(candidate_policy_path)

    hier_dir = GENERATED_DIR if write_hierarchies else None
    runs: list[dict] = []

    for pct in fractions_pct:
        if hier_dir is not None:
            candidate_h_path = write_hierarchy(tier, pct, tech, write_dir=hier_dir)
        else:
            import tempfile

            tmp = Path(tempfile.mkdtemp())
            candidate_h_path = write_hierarchy(tier, pct, tech, write_dir=tmp)

        candidate_h = load_hierarchy(
            candidate_h_path, repo_root=root, num_cores=trace.metadata.num_neuron_cores
        )

        for spill in spill_orders:
            candidate_p = base_candidate_p.model_copy(update={"spill_victim_order": spill})
            report = run_one_compare(
                trace,
                baseline_h,
                candidate_h,
                baseline_p,
                candidate_p,
                root=root,
                baseline_h_path=baseline_h_path,
                candidate_h_path=candidate_h_path,
                baseline_p_path=baseline_p_path,
                candidate_p_path=candidate_policy_path,
            )
            cand_res = report["candidate"]["results"]
            entry = {
                "replacement_pct": pct,
                "replacement_target": "hbm" if tier == "ltram" else "sbuf",
                "spill_victim_order": spill,
                "comparison": report["comparison"],
                "candidate": {
                    "hierarchy_name": candidate_h.name,
                    "total_time_ns": cand_res["total_time_ns"],
                    "total_energy_pJ": cand_res["total_energy_pJ"],
                    "hbm_traffic_bytes": (
                        cand_res["hbm_read_bytes"] + cand_res["hbm_write_bytes"]
                    ),
                    "cross_domain_traffic_bytes": (
                        cand_res.get("cross_domain_read_bytes", 0)
                        + cand_res.get("cross_domain_write_bytes", 0)
                    ),
                    "transfers_by_hop": cand_res["transfers_by_hop"],
                },
            }
            runs.append(entry)

            if per_run_dir is not None:
                slug = f"{tier}_{tech}_{pct}pct_{spill}"
                out_path = per_run_dir / f"compare_{slug}.json"
                write_report_json(out_path, report)
                entry["per_run_json"] = str(out_path.relative_to(root))

            if verbose:
                c = report["comparison"]
                print(
                    f"  {tier} {tech} {pct}% {spill}: "
                    f"time {c['time_ns']['pct_change']:+.2f}%  "
                    f"energy {c['energy_pJ']['pct_change']:+.2f}%  "
                    f"off_chip_if {c['cross_domain_traffic_bytes']['pct_change']:+.2f}%"
                )

    return {
        "sweep_kind": tier,
        "technology": tech,
        "trace": str(trace.metadata.workload),
        "trace_path": TRACE_DEFAULT,
        "baseline": {
            "hierarchy": BASELINE_HIERARCHY,
            "policy": BASELINE_POLICY,
        },
        "candidate": {
            "policy": str(candidate_policy_path.relative_to(root)),
            "hierarchy_template": f"trainium2_{tier}_{{pct}}pct_{'hbm' if tier == 'ltram' else 'sbuf'}_{tech}.yaml",
        },
        "fractions_pct": list(fractions_pct),
        "spill_orders": list(spill_orders),
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tier",
        choices=("ltram", "stram", "all-tech"),
        required=True,
        help="ltram or stram HBM/SBUF replacement sweep, or all-tech (both tiers × all techs)",
    )
    parser.add_argument("--ltram-tech", choices=LTRAM_TECHS, default="rram")
    parser.add_argument("--stram-tech", choices=STRAM_TECHS, default="edram_1t1c")
    parser.add_argument("--trace", type=Path, default=Path(TRACE_DEFAULT))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/traces/memory_sweeps"),
        help="Directory for consolidated JSON (and optional per-run files)",
    )
    parser.add_argument(
        "--no-per-run-json",
        action="store_true",
        help="Skip writing individual compare_*.json files",
    )
    parser.add_argument(
        "--no-write-hierarchies",
        action="store_true",
        help="Do not persist generated hierarchy YAML under configs/hierarchy/generated/",
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="10,25,50,75",
        help="Comma-separated replacement percentages (default 10,25,50,75)",
    )
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    fractions_pct = tuple(int(x.strip()) for x in args.fractions.split(",") if x.strip())

    root = _repo_root()
    trace_path = args.trace if args.trace.is_absolute() else root / args.trace
    trace = load_trace(trace_path)
    out_dir = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
    per_run_dir = None if args.no_per_run_json else out_dir / "runs"
    write_hiers = not args.no_write_hierarchies

    jobs: list[tuple[str, str]] = []
    if args.tier == "all-tech":
        for tech in LTRAM_TECHS:
            jobs.append(("ltram", tech))
        for tech in STRAM_TECHS:
            jobs.append(("stram", tech))
    elif args.tier == "ltram":
        jobs.append(("ltram", args.ltram_tech))
    else:
        jobs.append(("stram", args.stram_tech))

    for tier, tech in jobs:
        if not args.quiet:
            print(f"\n=== {tier.upper()} sweep — tech {tech} ===")
        consolidated = run_tier_tech_sweep(
            tier,
            tech,
            trace,
            root=root,
            fractions_pct=fractions_pct,
            write_hierarchies=write_hiers,
            per_run_dir=per_run_dir,
            verbose=not args.quiet,
        )
        out_name = f"consolidated_{tier}_{tech}.json"
        out_path = out_dir / out_name
        write_report_json(out_path, consolidated)
        if not args.quiet:
            print(f"Wrote {out_path.relative_to(root)}")

    if not args.quiet:
        print(f"\nDone. Consolidated results in {out_dir.relative_to(root)}/")


if __name__ == "__main__":
    main()
