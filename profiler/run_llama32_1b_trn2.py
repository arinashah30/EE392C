"""Compile + run Llama-3.2-1B-Instruct on trn2.3xlarge via NXDI.

Uses ``neuronx_distributed_inference.inference_demo`` (``NeuronLlamaForCausalLM``).
The legacy ``llama2/LlamaRunner`` stack compiles but hits NRT_EXEC_OOB on decode
for Llama 3.2 (llama3 RoPE / KV layout).

    python run_llama32_1b_trn2.py compile
    python run_llama32_1b_trn2.py run --prompt "I believe the meaning of life is"
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def _bootstrap_lnc_from_argv() -> None:
    lnc = os.environ.get("NEURON_LOGICAL_NC_CONFIG")
    if "--lnc" in sys.argv:
        idx = sys.argv.index("--lnc")
        if idx + 1 < len(sys.argv):
            lnc = sys.argv[idx + 1]
    if lnc is None:
        lnc = "1"
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)


_bootstrap_lnc_from_argv()

SHM_BASE = "/dev/shm"


def _ensure_shm_env() -> None:
    os.environ.setdefault("BASE_COMPILE_WORK_DIR", f"{SHM_BASE}/nxd_model_compile")
    os.environ.setdefault("TMPDIR", f"{SHM_BASE}/tmp")
    os.environ.setdefault("HF_HOME", f"{SHM_BASE}/huggingface")
    os.environ.setdefault("TORCH_HOME", f"{SHM_BASE}/torch")
    os.environ.setdefault("XLA_CACHE_DIR", f"{SHM_BASE}/xla_cache")
    for key in ("BASE_COMPILE_WORK_DIR", "TMPDIR", "HF_HOME", "TORCH_HOME", "XLA_CACHE_DIR"):
        os.makedirs(os.environ[key], exist_ok=True)


def _assert_traced_on_shm(path: str) -> None:
    if os.environ.get("ALLOW_DISK_ARTIFACTS", "").lower() in ("1", "true", "yes"):
        return
    abs_path = os.path.abspath(path.rstrip(os.sep))
    if not abs_path.startswith(SHM_BASE + os.sep):
        raise SystemExit(
            f"compiled-model-path must be under {SHM_BASE}/ (got {abs_path}). "
            "Set ALLOW_DISK_ARTIFACTS=1 to override."
        )


_ensure_shm_env()

DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_TRACED_BASE = "/dev/shm/traced_model"
DEFAULT_LNC = 1
DEFAULT_TP_DEGREE = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_PROMPT_LENGTH = 128
DEFAULT_SEQUENCE_LENGTH = 256
DEFAULT_MAX_NEW_TOKENS = 64
# Llama 3.2 primary EOS (config lists 128001, 128008, 128009)
DEFAULT_PAD_TOKEN_ID = 128001


def resolve_model_path(model_path: str) -> str:
    if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json")):
        return os.path.abspath(model_path)
    from huggingface_hub import snapshot_download

    print(f"[resolve] {model_path!r} is not a local checkpoint dir; resolving from HF cache ...")
    cache_dir = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    kwargs = {"local_files_only": True}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    snapshot_path = snapshot_download(repo_id=model_path, **kwargs)
    print(f"[resolve] using snapshot at {snapshot_path}")
    return snapshot_path


def default_compiled_path(args) -> str:
    return os.path.join(
        DEFAULT_TRACED_BASE,
        f"Llama-3.2-1B-Instruct-nxdi-lnc{args.lnc}-tp{args.tp_degree}"
        f"-b{args.batch_size}-ctx{args.max_prompt_length}-seq{args.sequence_length}",
    )


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def apply_lnc(lnc: int) -> None:
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)


def _max_new_tokens(args) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    if args.max_length is not None:
        # Legacy flag: treat as cap on new tokens (not HF total length).
        return max(1, args.max_length - args.max_prompt_length)
    return DEFAULT_MAX_NEW_TOKENS


def _nxdi_base_argv(args) -> list[str]:
    compiled = _ensure_trailing_sep(args.compiled_model_path)
    argv = [
        sys.executable,
        "-m",
        "neuronx_distributed_inference.inference_demo",
        "--model-type",
        "llama",
        "--task-type",
        "causal-lm",
        "run",
        "--model-path",
        args.model_path,
        "--compiled-model-path",
        compiled,
        "--logical-nc-config",
        str(args.lnc),
        "--tp-degree",
        str(args.tp_degree),
        "--batch-size",
        str(args.batch_size),
        "--seq-len",
        str(args.sequence_length),
        "--max-context-length",
        str(args.max_prompt_length),
        "--max-new-tokens",
        str(_max_new_tokens(args)),
        "--torch-dtype",
        "bfloat16",
        "--pad-token-id",
        str(args.pad_token_id),
        "--top-k",
        "1",
    ]
    for prompt in args.prompts:
        argv.extend(["--prompt", prompt])
    return argv


def _maybe_clean_compile_cache() -> None:
    nxd_tmp = os.environ.get("BASE_COMPILE_WORK_DIR", f"{SHM_BASE}/nxd_model_compile")
    if os.path.isdir(nxd_tmp):
        print(f"[clean] removing stale compile cache {nxd_tmp}")
        shutil.rmtree(nxd_tmp, ignore_errors=True)


def _run_nxdi(argv: list[str]) -> None:
    env = os.environ.copy()
    env["NEURON_LOGICAL_NC_CONFIG"] = str(
        env.get("NEURON_LOGICAL_NC_CONFIG", str(DEFAULT_LNC))
    )
    print("[nxdi]", " ".join(argv[2:]), flush=True)
    subprocess.run(argv, check=True, env=env)


def cmd_compile(args) -> None:
    apply_lnc(args.lnc)
    args.model_path = resolve_model_path(args.model_path)
    if args.compiled_model_path is None:
        args.compiled_model_path = default_compiled_path(args)
    args.compiled_model_path = _ensure_trailing_sep(args.compiled_model_path)
    _assert_traced_on_shm(args.compiled_model_path)
    os.makedirs(args.compiled_model_path, exist_ok=True)

    print("\n=== compile (NXDI) ===")
    print(f"  model_path          = {args.model_path}")
    print(f"  compiled_model_path = {args.compiled_model_path}")
    print(f"  lnc                 = {args.lnc}")
    print(f"  tp_degree           = {args.tp_degree}")

    if args.clean_cache:
        _maybe_clean_compile_cache()

    # NXDI requires --prompt even with --compile-only (not used during compile).
    if not args.prompts:
        args.prompts = ["The capital of France is"]

    argv = _nxdi_base_argv(args)
    argv.append("--compile-only")
    _run_nxdi(argv)
    print(f"\n[compile] artifacts at {args.compiled_model_path}")


def _resolve_compiled_path(args) -> str:
    if args.compiled_model_path is not None:
        return args.compiled_model_path
    candidate = default_compiled_path(args)
    if not os.path.isdir(candidate):
        raise SystemExit(
            f"No --compiled-model-path given and {candidate!r} does not exist. "
            "Run `compile` first."
        )
    return candidate


def cmd_run(args) -> None:
    apply_lnc(args.lnc)
    args.model_path = resolve_model_path(args.model_path)
    args.compiled_model_path = _ensure_trailing_sep(_resolve_compiled_path(args))
    _assert_traced_on_shm(args.compiled_model_path)

    if not args.prompts:
        raise SystemExit("Pass at least one --prompt.")
    if len(args.prompts) != args.batch_size:
        args.prompts = (args.prompts * args.batch_size)[: args.batch_size]

    print("\n=== run (NXDI) ===")
    print(f"  compiled_model_path = {args.compiled_model_path}")
    print(f"  lnc                 = {args.lnc}")
    print(f"  tp_degree           = {args.tp_degree}")
    print(f"  prompts             = {args.prompts}")
    print(f"  max_new_tokens      = {_max_new_tokens(args)}")

    argv = _nxdi_base_argv(args)
    argv.append("--skip-compile")
    _run_nxdi(argv)


def cmd_compile_and_run(args) -> None:
    cmd_compile(args)
    cmd_run(args)


def _add_common_args(p: argparse.ArgumentParser, *, with_prompts: bool) -> None:
    p.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_ID,
        help=f"HF checkpoint dir or model id (default {DEFAULT_MODEL_ID})",
    )
    p.add_argument(
        "--compiled-model-path",
        "--traced-model-path",
        dest="compiled_model_path",
        default=None,
        help="NXDI compiled model directory (trailing / recommended).",
    )
    p.add_argument(
        "--lnc",
        type=int,
        choices=(1, 2),
        default=DEFAULT_LNC,
        help="Logical NeuronCore config. LNC=1 => 4 logical cores on trn2.3xlarge.",
    )
    p.add_argument("--tp-degree", type=int, default=DEFAULT_TP_DEGREE)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH)
    p.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    p.add_argument("--pad-token-id", type=int, default=DEFAULT_PAD_TOKEN_ID)
    p.add_argument(
        "--clean-cache",
        action="store_true",
        help="Remove /tmp/nxd_model before compile (recommended after LNC/TP changes).",
    )
    if with_prompts:
        p.add_argument("--prompt", dest="prompts", action="append", default=[])
        p.add_argument("--max-new-tokens", type=int, default=None)
        p.add_argument(
            "--max-length",
            type=int,
            default=None,
            help="Legacy alias: approximate new tokens as max_length - max_prompt_length.",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="run_llama32_1b_trn2",
        description="Llama-3.2-1B on trn2 via neuronx_distributed_inference",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compile = sub.add_parser("compile")
    _add_common_args(p_compile, with_prompts=False)
    p_compile.set_defaults(func=cmd_compile, clean_cache=True)

    p_run = sub.add_parser("run")
    _add_common_args(p_run, with_prompts=True)
    p_run.set_defaults(func=cmd_run, clean_cache=False)

    p_both = sub.add_parser("compile-and-run")
    _add_common_args(p_both, with_prompts=True)
    p_both.set_defaults(func=cmd_compile_and_run, clean_cache=True)

    args = parser.parse_args()
    if args.tp_degree > 4 // args.lnc:
        raise SystemExit(
            f"--tp-degree {args.tp_degree} exceeds available logical cores "
            f"({4 // args.lnc}) for LNC={args.lnc} on trn2.3xlarge."
        )
    if not hasattr(args, "prompts"):
        args.prompts = []
    if not hasattr(args, "max_new_tokens"):
        args.max_new_tokens = None
    if not hasattr(args, "max_length"):
        args.max_length = None
    args.func(args)


if __name__ == "__main__":
    main()
