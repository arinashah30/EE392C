"""Compile + run Qwen1.5-MoE-A2.7B on trn2.3xlarge for dmsim profiling.

    python profiler/run_qwen_moe_trn2.py compile
    python profiler/run_qwen_moe_trn2.py run --prompt "The capital of France is"

Or use ``profiler/capture_and_export_qwen.sh`` (Neuron profiler + Explorer JSON).

Defaults match ``capture_and_export.sh`` for Llama: LNC=1, TP=4 (all 4 NeuronCores),
batch=1, max_prompt_length=128, sequence_length=256, prompt "The capital of France is".
MoE weights are large — compile may OOM on small hosts; use COMPILE=1 once on trn2.3xlarge.

Run from anywhere (cwd doesn't matter). Vendored under ``profiler/nxd_inference/``:
``modules/`` and ``runner.py`` from AWS neuronx-distributed inference examples.
Also shims ``transformers.modeling_utils.shard_checkpoint`` for transformers>=4.45.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time


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
    """Keep NxD compile scratch + torch temp files off root disk."""
    os.environ.setdefault("TMPDIR", f"{SHM_BASE}/tmp")
    os.environ.setdefault("HF_HOME", f"{SHM_BASE}/huggingface")
    os.environ.setdefault("TORCH_HOME", f"{SHM_BASE}/torch")
    os.environ.setdefault("XLA_CACHE_DIR", f"{SHM_BASE}/xla_cache")
    for key in ("TMPDIR", "HF_HOME", "TORCH_HOME", "XLA_CACHE_DIR"):
        os.makedirs(os.environ[key], exist_ok=True)


def _compile_work_dir(lnc: int, tp_degree: int) -> str:
    return os.path.join(SHM_BASE, f"nxd_model_compile-lnc{lnc}-tp{tp_degree}")


def _prepare_compile_work_dir(lnc: int, tp_degree: int, *, clean: bool = True) -> str:
    """NEFF paths are absolute under BASE_COMPILE_WORK_DIR — isolate by LNC/TP."""
    work_dir = _compile_work_dir(lnc, tp_degree)
    os.environ["BASE_COMPILE_WORK_DIR"] = work_dir
    if clean and os.path.isdir(work_dir):
        print(f"[clean] removing stale compile cache {work_dir}")
        shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir, exist_ok=True)
    return work_dir


def _assert_traced_on_shm(path: str) -> None:
    if os.environ.get("ALLOW_DISK_ARTIFACTS", "").lower() in ("1", "true", "yes"):
        return
    abs_path = os.path.abspath(path.rstrip(os.sep))
    if not abs_path.startswith(SHM_BASE + os.sep):
        raise SystemExit(
            f"traced-model-path must be under {SHM_BASE}/ (got {abs_path}).\n"
            "Large compile/profile artifacts belong on tmpfs, not root disk.\n"
            f"Example: {SHM_BASE}/traced_model/Qwen1.5-MoE-A2.7B-...\n"
            "Set ALLOW_DISK_ARTIFACTS=1 to override."
        )


_ensure_shm_env()

# ---------------------------------------------------------------------------
# 1. Vendored NxD inference example (modules/, runner.py) + local qwen_moe/.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_NXD_INFERENCE = os.path.join(_SCRIPT_DIR, "nxd_inference")
if not os.path.isdir(os.path.join(_NXD_INFERENCE, "modules")):
    raise SystemExit(
        f"Missing vendored inference tree at {_NXD_INFERENCE}/modules. "
        "Expected profiler/nxd_inference/ from neuronx-distributed examples/inference."
    )
sys.path.insert(0, _NXD_INFERENCE)


# ---------------------------------------------------------------------------
# 2. Shim `shard_checkpoint`, removed in transformers >= 4.45.
#    The example pins transformers==4.40.0; the Neuron inference venv
#    ships 4.57.6, so we patch in a minimal compatible function so
#    `examples/inference/modules/checkpoint.py` imports. We never call
#    `save_state_dict_safetensors` in this flow (it's only used for
#    quantized-checkpoint save), so the shim's correctness only needs
#    to be import-time correct.
# ---------------------------------------------------------------------------
import transformers.modeling_utils as _hf_modeling_utils  # noqa: E402

if not hasattr(_hf_modeling_utils, "shard_checkpoint"):
    def _shard_checkpoint(state_dict, max_shard_size="10GB", weights_name="pytorch_model.bin"):
        """Drop-in stub for the transformers<4.45 helper. Returns a single
        shard so the caller can still write a flat safetensors file.

        We intentionally don't try to honor max_shard_size; this shim only
        runs when someone calls save_state_dict_safetensors(), which our
        compile/run flow never does.
        """
        return {weights_name: state_dict}, None

    _hf_modeling_utils.shard_checkpoint = _shard_checkpoint


# ---------------------------------------------------------------------------
# 3. Now we can import the example.
# ---------------------------------------------------------------------------
import gc  # noqa: E402

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from huggingface_hub.errors import LocalEntryNotFoundError  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import GenerationConfig  # noqa: E402

from qwen_moe.qwen_moe_runner import QwenMoeRunner  # noqa: E402

# ---------------------------------------------------------------------------
# 4. Patch ModelBuilder.shard_weights_with_cache to be memory-frugal.
#
#    Default behaviour: the per-rank shard dict is *returned* and the caller
#    accumulates one entry per rank in a list. For Qwen1.5-MoE at TP=2 each
#    shard is ~14 GB so the default isn't catastrophic, but the frugal path
#    only helps and is harmless to leave on. See run_mixtral_trn2.py for the
#    Mixtral motivation (94 GB total weights).
#
#    We also avoid passing ``{k: v.contiguous() for ...}`` into ``save_file``:
#    that dict comprehension duplicates the entire shard in RAM and can OOM
#    a 124 GiB host right after slicing (~68 GiB RSS → ~82+ GiB peak).
# ---------------------------------------------------------------------------
from neuronx_distributed.trace.model_builder import ModelBuilder  # noqa: E402

from neuronx_distributed.trace.trace import get_sharded_checkpoint  # noqa: E402


def _rss_gib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return -1.0


def _frugal_shard_weights_with_cache(self, rank, model, checkpoint, serialize_path=None):
    """Write each TP rank to disk and return None (no list accumulation).

    Rank 0 shards *in place* on ``checkpoint`` (no ``checkpoint.copy()`` — a
    shallow copy still leaves the full tensors in the parent dict and peaks
  ~72+ GiB RSS on trn2.3xlarge). Rank 1+ reloads a fresh state dict because
    rank 0 replaced tensors with slices.
    """
    from neuronx_distributed.trace.trace import preprocess_checkpoint

    print(f"[shard] rank {rank} start  RSS={_rss_gib():.1f} GiB", flush=True)
    if rank > self.start_rank_id:
        print(f"[shard] rank {rank} reloading full checkpoint ...", flush=True)
        fresh = self.checkpoint_loader()
        checkpoint.clear()
        checkpoint.update(fresh)
        del fresh
        gc.collect()
        preprocess_checkpoint(model, checkpoint)
        self.cast_weights(checkpoint, model, "")
        is_cached = True
    else:
        is_cached = True

    get_sharded_checkpoint(checkpoint, model, rank, self.tp_degree, is_cached=is_cached)
    print(f"[shard] rank {rank} sliced RSS={_rss_gib():.1f} GiB", flush=True)
    if serialize_path is not None:
        out_path = os.path.join(serialize_path, f"tp{rank}_sharded_checkpoint.safetensors")
        for _k, _v in list(checkpoint.items()):
            if not _v.is_contiguous():
                checkpoint[_k] = _v.contiguous()
        save_file(checkpoint, out_path)
    print(f"[shard] rank {rank} saved  RSS={_rss_gib():.1f} GiB", flush=True)
    checkpoint.clear()
    gc.collect()
    print(f"[shard] rank {rank} freed  RSS={_rss_gib():.1f} GiB", flush=True)
    return None


ModelBuilder.shard_weights_with_cache = _frugal_shard_weights_with_cache


# ---------------------------------------------------------------------------
# Defaults sized for trn2.3xlarge (4 NeuronCores, LNC=1 => 4 logical, TP=4).
# MoE compile must pass --lnc=1 to neuron-cc (see get_compiler_args in
# qwen_moe/neuron_modeling_qwen_moe.py).
# ---------------------------------------------------------------------------
DEFAULT_HF_MODEL_ID = "Qwen/Qwen1.5-MoE-A2.7B"
_LOCAL_CHECKPOINT = f"{SHM_BASE}/Qwen1.5-MoE-A2.7B"


def _looks_like_hf_repo_id(model_path: str) -> bool:
    if not model_path or model_path.startswith(("/", "~", ".")):
        return False
    if "\\" in model_path:
        return False
    parts = model_path.split("/")
    return len(parts) == 2 and all(parts)


def _dir_has_hf_weights(model_dir: str) -> bool:
    if not os.path.isdir(model_dir):
        return False
    names = os.listdir(model_dir)
    if "model.safetensors" in names or "pytorch_model.bin" in names:
        return True
    return any(n.endswith(".safetensors") and n.startswith("model-") for n in names)


def _looks_like_runtime_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and os.path.isfile(os.path.join(path, "config.json"))
        and os.path.isfile(os.path.join(path, "tokenizer.json"))
    )


def _discover_local_checkpoint() -> str | None:
    explicit = os.environ.get("QWEN_MODEL_PATH", "").strip()
    if explicit and _dir_has_hf_weights(explicit):
        return os.path.abspath(explicit)
    if _dir_has_hf_weights(_LOCAL_CHECKPOINT):
        return _LOCAL_CHECKPOINT
    return None


def _default_model_path() -> str:
    return _discover_local_checkpoint() or DEFAULT_HF_MODEL_ID


DEFAULT_TRACED_BASE = "/dev/shm/traced_model"
DEFAULT_LNC = 1
DEFAULT_TP_DEGREE = 4
DEFAULT_BATCH_SIZE = 1
DEFAULT_MAX_PROMPT_LENGTH = 128
DEFAULT_SEQUENCE_LENGTH = 256


# Names match qwen_moe/routers/__init__.py::ROUTER_REGISTRY. Kept inline so
# the CLI help message is informative without an import-time dependency
# on neuronx-distributed (which not every dev machine has installed).
_ROUTER_CHOICES = (
    # Group A: accuracy-preserving (drop-in for Qwen's pretrained TopK)
    "topk_softmax",
    "topk_softmax_over_topk",
    "topk_sigmoid",
    "group_limited_topk",
    "sparsemax_topk",
    # Group B: SOTA but disruptive (need T1 fine-tune for quality)
    "hierarchical",
    "sinkhorn_balanced",
    "hash_routing",
    "expert_choice",
    "base_layer",
    "soft_moe",
)


def _parse_router_kwargs(items):
    """Convert ``--router-kwarg key=value`` repeats into a dict.

    Values are parsed as int if they're a valid integer, else float, else
    left as string. Empty / None input returns an empty dict.
    """
    out = {}
    for item in (items or []):
        if "=" not in item:
            raise SystemExit(
                f"--router-kwarg expects key=value, got {item!r}"
            )
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def resolve_model_path(model_path: str, *, fallback_dir: str | None = None) -> str:
    """Resolve either a local checkpoint dir or an HF model id to an on-disk path.

    For `run`, callers pass ``fallback_dir=<traced_model_path>``. The traced
    dir already carries config.json + tokenizer files (copied at compile
    time), so if the HF hub cache is empty / pruned we transparently fall
    back to it instead of failing with LocalEntryNotFoundError.
    """
    if os.path.isdir(model_path) and os.path.isfile(os.path.join(model_path, "config.json")):
        return os.path.abspath(model_path)

    local_ckpt = _discover_local_checkpoint()
    if local_ckpt is not None:
        print(f"[resolve] using local checkpoint {local_ckpt} "
              f"(instead of {model_path!r})")
        return local_ckpt

    # Absolute / relative filesystem path that does not exist — never pass to HF hub.
    if _looks_like_hf_repo_id(model_path):
        repo_id = model_path
    elif model_path.startswith(("/", "~", ".")) or os.path.isabs(model_path):
        raise SystemExit(
            f"Local checkpoint {model_path!r} not found "
            f"(need config.json and weight shard files).\n"
            "Download once to tmpfs, e.g.:\n"
            f"  huggingface-cli download {DEFAULT_HF_MODEL_ID} "
            f"--local-dir {_LOCAL_CHECKPOINT}\n"
            "Or set MODEL_PATH to an existing checkpoint dir."
        )
    else:
        repo_id = DEFAULT_HF_MODEL_ID

    print(f"[resolve] {model_path!r} is not a local checkpoint dir; "
          f"resolving HF id {repo_id!r} from cache ...")
    local_only = os.environ.get("ALLOW_HF_DOWNLOAD", "").lower() not in (
        "1", "true", "yes",
    )
    try:
        snapshot_path = snapshot_download(
            repo_id=repo_id, local_files_only=local_only,
        )
        print(f"[resolve] using snapshot at {snapshot_path}")
        return snapshot_path
    except LocalEntryNotFoundError as exc:
        if fallback_dir is not None and _looks_like_runtime_dir(fallback_dir):
            print(f"[resolve] HF cache miss for {repo_id!r}; "
                  f"falling back to traced-model dir {fallback_dir} "
                  f"(config + tokenizer were copied there at compile time).")
            return os.path.abspath(fallback_dir)
        raise SystemExit(
            f"Could not resolve model_path={model_path!r}: {exc}\n"
            "Fix options:\n"
            f"  * huggingface-cli download {DEFAULT_HF_MODEL_ID} "
            f"--local-dir {_LOCAL_CHECKPOINT}\n"
            "  * export ALLOW_HF_DOWNLOAD=1 to pull from the hub, or\n"
            "  * for `run` only: reuse a traced dir with config.json + tokenizer.json"
        )


def default_traced_path(args) -> str:
    """Generate a per-shape default for --traced-model-path.

    Trailing slash is REQUIRED: the upstream compile() in
    examples/inference/modules/model_base.py uses string concatenation
    `serialize_base_path + "model.pt"` (not os.path.join). Without the slash,
    the JIT graph ends up at `<dir>model.pt` next to the directory.

    The router kernel name is folded into the cache key so each of the
    10 router variants gets its own NEFF cache directory. NKI fast-path
    is also distinguished (different traced graph than torch path).
    """
    router_tag = getattr(args, "router_kernel", "topk_softmax")
    nki_tag = "-nki" if getattr(args, "use_nki_router", False) else ""
    ft_tag = "-ft" if getattr(args, "router_ft_checkpoint", None) else ""
    eft_tag = "-expertft" if getattr(args, "expert_ft_checkpoint", None) else ""
    base = os.path.join(
        DEFAULT_TRACED_BASE,
        f"Qwen1.5-MoE-A2.7B-lnc{args.lnc}-tp{args.tp_degree}"
        f"-b{args.batch_size}-p{args.max_prompt_length}-s{args.sequence_length}"
        f"-r{router_tag}{nki_tag}{ft_tag}{eft_tag}",
    )
    return base + os.sep


def _ensure_trailing_sep(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def apply_lnc(lnc: int) -> None:
    """Set NEURON_LOGICAL_NC_CONFIG so NxD picks up the requested LNC.

    Must be called before any neuronx-distributed / torch_neuronx code that
    queries get_platform_lnc().
    """
    os.environ["NEURON_LOGICAL_NC_CONFIG"] = str(lnc)


def make_runner(model_path: str) -> QwenMoeRunner:
    # The traced-model dir we fall back to in resolve_model_path() doesn't
    # carry generation_config.json (we never copy it at compile time), so
    # tolerate its absence and synthesize a minimal config.
    try:
        generation_config = GenerationConfig.from_pretrained(model_path)
    except (OSError, EnvironmentError) as exc:
        print(f"[runner] no generation_config.json under {model_path!r} ({exc}); "
              "using a default GenerationConfig().")
        generation_config = GenerationConfig()
    generation_config.top_k = 1
    generation_config.do_sample = True
    generation_config.pad_token_id = 0
    return QwenMoeRunner(
        model_path=model_path,
        tokenizer_path=model_path,
        generation_config=generation_config,
    )


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------
def cmd_shard_only(args) -> None:
    """Shard HF weights into an existing traced dir (skip trace + neuron-cc)."""
    apply_lnc(args.lnc)
    args.model_path = resolve_model_path(args.model_path)
    if args.traced_model_path is None:
        args.traced_model_path = default_traced_path(args)
    args.traced_model_path = _ensure_trailing_sep(args.traced_model_path)
    model_pt = os.path.join(args.traced_model_path, "model.pt")
    if not os.path.isfile(model_pt):
        raise SystemExit(
            f"Missing {model_pt}. Run `compile` first, or use full compile not shard-only."
        )
    print(f"\n=== shard-only ===")
    print(f"  traced_model_path = {args.traced_model_path}")
    print(f"  (reusing existing model.pt; only rewriting weights/)")
    runner = make_runner(args.model_path)
    t0 = time.time()
    runner.trace_shard_only(
        traced_model_path=args.traced_model_path,
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        sequence_length=args.sequence_length,
        router_kernel=args.router_kernel,
        use_nki_router=args.use_nki_router,
        router_ft_checkpoint=args.router_ft_checkpoint,
        expert_ft_checkpoint=args.expert_ft_checkpoint,
        router_kwargs=_parse_router_kwargs(args.router_kwarg),
    )
    print(f"\n[shard-only] done in {time.time() - t0:.1f}s")


def cmd_compile(args) -> None:
    """Trace + compile + serialize the Qwen1.5-MoE model. Slow."""
    apply_lnc(args.lnc)
    work_dir = _prepare_compile_work_dir(args.lnc, args.tp_degree, clean=True)
    args.model_path = resolve_model_path(args.model_path)
    if args.traced_model_path is None:
        args.traced_model_path = default_traced_path(args)
    args.traced_model_path = _ensure_trailing_sep(args.traced_model_path)
    _assert_traced_on_shm(args.traced_model_path)

    print(f"\n=== compile ===")
    print(f"  model_path        = {args.model_path}")
    print(f"  traced_model_path = {args.traced_model_path}")
    print(f"  lnc               = {args.lnc}  (NEURON_LOGICAL_NC_CONFIG)")
    print(f"  compile_work_dir  = {work_dir}")
    print(f"  tp_degree         = {args.tp_degree}")
    print(f"  batch_size        = {args.batch_size}")
    print(f"  max_prompt_length = {args.max_prompt_length}")
    print(f"  sequence_length   = {args.sequence_length}")
    print(f"  router_kernel     = {args.router_kernel}")
    print(f"  use_nki_router    = {args.use_nki_router}")
    if args.router_ft_checkpoint:
        print(f"  router_ft_ckpt    = {args.router_ft_checkpoint}")
    if args.expert_ft_checkpoint:
        print(f"  expert_ft_ckpt    = {args.expert_ft_checkpoint}")
    if args.router_kwarg:
        print(f"  router_kwargs     = {args.router_kwarg}")

    os.makedirs(args.traced_model_path, exist_ok=True)
    torch.manual_seed(0)

    runner = make_runner(args.model_path)
    t0 = time.time()
    runner.trace(
        traced_model_path=args.traced_model_path,
        tp_degree=args.tp_degree,
        batch_size=args.batch_size,
        max_prompt_length=args.max_prompt_length,
        sequence_length=args.sequence_length,
        # Router kernel selection flows through MoENeuronConfig and is
        # consumed by NeuronQwenMoeDecoderLayer's RouterFactory.create()
        # call (see qwen_moe/neuron_modeling_qwen_moe.py).
        router_kernel=args.router_kernel,
        use_nki_router=args.use_nki_router,
        router_ft_checkpoint=args.router_ft_checkpoint,
        expert_ft_checkpoint=args.expert_ft_checkpoint,
        router_kwargs=_parse_router_kwargs(args.router_kwarg),
    )
    print(f"\n[compile] trace + compile + shard took {time.time() - t0:.1f}s")
    print(f"[compile] artifacts at {args.traced_model_path}")
    print(f"[compile] now run with:")
    print(f"  python {sys.argv[0]} run "
          f"--traced-model-path {args.traced_model_path} "
          f"--prompt 'your prompt here'")


def _resolve_traced_path(args) -> str:
    """For `run`: if --traced-model-path wasn't given, fall back to the
    default we'd have used at compile time."""
    if args.traced_model_path is not None:
        return args.traced_model_path
    candidate = default_traced_path(args)
    if not os.path.isdir(candidate):
        raise SystemExit(
            f"No --traced-model-path given and the default candidate "
            f"{candidate!r} doesn't exist. Run `compile` first, or pass "
            f"--traced-model-path explicitly."
        )
    return candidate


def cmd_run(args) -> None:
    """Load a previously compiled Qwen model and generate on Neuron."""
    apply_lnc(args.lnc)
    _prepare_compile_work_dir(args.lnc, args.tp_degree, clean=False)
    # Resolve the traced path FIRST so we can use it as a fallback when the
    # HF hub cache for --model-path is empty (the traced dir carries
    # config.json + tokenizer.json from compile time).
    args.traced_model_path = _ensure_trailing_sep(_resolve_traced_path(args))
    _assert_traced_on_shm(args.traced_model_path)
    args.model_path = resolve_model_path(
        args.model_path, fallback_dir=args.traced_model_path,
    )

    if not args.prompts:
        raise SystemExit("Pass at least one --prompt.")
    # Pad / truncate to the compiled batch size.
    if len(args.prompts) != args.batch_size:
        args.prompts = (args.prompts * args.batch_size)[: args.batch_size]

    print(f"\n=== run ===")
    print(f"  traced_model_path = {args.traced_model_path}")
    print(f"  batch_size        = {args.batch_size}")
    print(f"  prompts           = {args.prompts}")

    torch.manual_seed(0)
    runner = make_runner(args.model_path)

    print("\n[load] loading sharded weights to Neuron device ...")
    t0 = time.time()
    neuron_model = runner.load_neuron_model(args.traced_model_path)
    print(f"[load] took {time.time() - t0:.1f}s")

    # The Neuron model's auto-built generation_config (from
    # PreTrainedModel.__init__) lacks `transformers_version`, which the
    # transformers>=4.50 generate() path dereferences unconditionally.
    # Attach the runner's loaded generation_config (parsed from the
    # checkpoint's generation_config.json, which does carry the field).
    neuron_model.generation_config = runner.generation_config

    if args.check_accuracy:
        print("\n[accuracy] checking logits vs HF CPU golden (slow, fp32 on CPU) ...")
        runner.check_accuracy(neuron_model, args.batch_size, args.sequence_length)

    print("\n[generate] running ...")
    t0 = time.time()
    _, outputs = runner.generate_on_neuron(args.prompts, neuron_model)
    print(f"[generate] took {time.time() - t0:.1f}s")
    print("\n[generate] outputs:")
    for idx, output in enumerate(outputs):
        print(f"  output[{idx}]: {output}")

    if args.benchmark:
        print("\n[benchmark] sampling ...")
        runner.benchmark_sampling(neuron_model)


def cmd_compile_and_run(args) -> None:
    cmd_compile(args)
    cmd_run(args)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _add_common_args(p: argparse.ArgumentParser, *, with_prompts: bool) -> None:
    p.add_argument("--model-path", default=_default_model_path(),
                   help="Local Qwen1.5-MoE checkpoint dir, or an HF model id. "
                        "Default: /dev/shm/Qwen1.5-MoE-A2.7B when present, "
                        f"else {DEFAULT_HF_MODEL_ID}. Override with QWEN_MODEL_PATH.")
    p.add_argument("--traced-model-path", default=None,
                   help="Where to read/write the compiled NEFF + sharded "
                        "weights. Defaults to "
                        f"{DEFAULT_TRACED_BASE}/Qwen1.5-MoE-A2.7B-lnc{{LNC}}"
                        "-tp{TP}-b{BS}-p{P}-s{S}/")
    p.add_argument("--lnc", type=int, choices=(1, 2), default=DEFAULT_LNC,
                   help="Logical NeuronCore config. LNC=1 => 4 logical cores on "
                        f"trn2.3xlarge (profiler default LNC={DEFAULT_LNC}).")
    p.add_argument("--tp-degree", type=int, default=DEFAULT_TP_DEGREE,
                   help=f"Tensor-parallel degree. Default {DEFAULT_TP_DEGREE} "
                        f"(matches default --lnc {DEFAULT_LNC}).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--max-prompt-length", type=int, default=DEFAULT_MAX_PROMPT_LENGTH,
                   help=f"Max input prompt length. Default {DEFAULT_MAX_PROMPT_LENGTH}.")
    p.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH,
                   help="Max total sequence length (prompt + generated). "
                        f"Default {DEFAULT_SEQUENCE_LENGTH}.")
    # ------------------------------------------------------------------
    # Router-kernel selection (see moe_demo/qwen_moe/routers/README.md)
    # ------------------------------------------------------------------
    p.add_argument("--router-kernel", default="topk_softmax",
                   choices=_ROUTER_CHOICES,
                   help="MoE router kernel to use. Default 'topk_softmax' "
                        "reproduces the upstream RouterTopK behavior. The "
                        "other 9 choices implement alternative routing "
                        "strategies (see moe_demo/qwen_moe/routers/README.md). "
                        "Each value gets its own NEFF cache (folded into "
                        "the default --traced-model-path).")
    p.add_argument("--use-nki-router", action="store_true",
                   help="Route through the fused-NKI shell in "
                        "qwen_moe/routers/fused_router_kernel.py instead of "
                        "the torch reference path. Only affects Group-A "
                        "routers (1-5). Falls back to torch if NKI imports "
                        "fail. Default off.")
    p.add_argument("--router-ft-checkpoint", default=None,
                   help="Optional path to a router_weights.safetensors "
                        "produced by moe_demo/bench/finetune_router.py "
                        "(Tier-T1 router-only fine-tune). Merged into the "
                        "router's state_dict after construction. No-op for "
                        "routers with no learnable parameters "
                        "(hash_routing). Default: T0 (no fine-tune).")
    p.add_argument("--expert-ft-checkpoint", default=None,
                   help="Optional path to expert_weights_merged.safetensors "
                        "from bench/finetune_expert_lora_hash.py (Tier-T2). "
                        "Overrides routed expert MLP weights before sharding.")
    p.add_argument("--router-kwarg", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="Per-router escape-hatch kwarg, e.g. "
                        "--router-kwarg n_group=12 --router-kwarg topk_group=4 "
                        "for group_limited_topk, or --router-kwarg num_slots=4 "
                        "for soft_moe. Repeat for multiple kwargs.")
    if with_prompts:
        p.add_argument("--prompt", dest="prompts", action="append", default=[],
                       help="Prompt to generate from. Repeat --prompt N times "
                            "to fill a batch_size=N model.")
        p.add_argument("--check-accuracy", action="store_true",
                       help="After loading, validate Neuron logits vs the HF "
                            "CPU golden (heavy: runs full Qwen in fp32).")
        p.add_argument("--benchmark", action="store_true",
                       help="After generating, run benchmark_sampling().")


def main() -> None:
    parser = argparse.ArgumentParser(prog="run_qwen_moe_trn2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compile = sub.add_parser("compile", help="Trace + compile + serialize.")
    _add_common_args(p_compile, with_prompts=False)
    p_compile.set_defaults(func=cmd_compile)

    p_shard = sub.add_parser(
        "shard-only",
        help="Rewrite weights/ only; reuse existing model.pt (after OOM at shard).",
    )
    _add_common_args(p_shard, with_prompts=False)
    p_shard.set_defaults(func=cmd_shard_only)

    p_run = sub.add_parser("run", help="Load compiled model and generate.")
    _add_common_args(p_run, with_prompts=True)
    p_run.set_defaults(func=cmd_run)

    p_both = sub.add_parser("compile-and-run", help="compile then run.")
    _add_common_args(p_both, with_prompts=True)
    p_both.set_defaults(func=cmd_compile_and_run)

    args = parser.parse_args()
    if args.tp_degree > 4 // args.lnc:
        raise SystemExit(
            f"--tp-degree {args.tp_degree} exceeds available logical cores "
            f"({4 // args.lnc}) for LNC={args.lnc} on trn2.3xlarge."
        )
    args.func(args)


if __name__ == "__main__":
    main()
