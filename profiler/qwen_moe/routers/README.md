# Qwen-MoE Router Kernels

A swappable collection of 10 MoE routing strategies that can be
plugged into the Qwen1.5-MoE-A2.7B demo (`moe_demo/qwen_moe/`) via the
`--router-kernel <name>` CLI flag on `run_qwen_moe_trn2.py`.

The router is the small module that takes a flattened token tensor of
shape `(T, H)`, computes a `(T, E)` score matrix, and returns the
chosen experts plus the per-expert affinities the MoE block uses to
weight the expert outputs. Qwen ships with `RouterTopK` (a
`linear → softmax → torch.topk` stack); this directory expands that
into a research-comparison harness.

The 10 routers are split into two groups:

* **Group A — accuracy-preserving** (drop-in replacements that keep
  Qwen's pretrained expert specialization within numerical tolerance):
  `topk_softmax`, `topk_softmax_over_topk`, `topk_sigmoid`,
  `group_limited_topk`, `sparsemax_topk`.

* **Group B — SOTA but disruptive** (change the routing distribution
  meaningfully; without a router-only fine-tune they will degrade
  Qwen's MMLU / HellaSwag / ARC scores):
  `sinkhorn_balanced`, `hash_routing`, `expert_choice`, `base_layer`,
  `soft_moe`.

See [`../../bench/TEST_PLAN.md`](../../bench/TEST_PLAN.md) for the full
benchmarking pipeline. See [`../../bench/finetune_router.py`](../../bench/finetune_router.py)
for the optional Tier-T1 router-only fine-tune that recovers most of
the Group B accuracy gap.

## Contract

Every router is a subclass of `qwen_moe.routers.base.SwappableRouter`
*and* (in practice) of `neuronx_distributed.modules.moe.routing.RouterBase`.
The `forward(hidden_states)` method returns a 3-tuple matching the
existing `RouterTopK` contract so the stock `MoE` block can consume
any router without modification:

```python
router_logits: torch.Tensor      # (T, E)  raw scores
expert_affinities: torch.Tensor  # (T, E)  normalized weights (zeros
                                 #         outside the chosen subset)
expert_index: torch.Tensor       # (T, top_k)  chosen expert ids, long
```

For routers that change the dispatch shape (`expert_choice`,
`soft_moe`) we ship a paired `MoE` subclass (`MoEExpertChoice`,
`MoESoftMix`) in the same file. `RouterFactory.build_moe(router, ...)`
hands back the right wrapper class.

## The 10 kernels

### Group A — drop-in score-function swaps

| # | Name | Score function | Citation |
|---|---|---|---|
| 1 | `topk_softmax` | `softmax(logits, fp64)` then `topk(k)` — Qwen baseline. | Switch Transformer (Fedus et al. 2021) |
| 2 | `topk_softmax_over_topk` | `topk(logits, k)` then `softmax` over only the chosen `k` values. Identical chosen experts to #1, no fp64 cast. | Already supported as `apply_act_fn_over_topk=True` in NxD's `RouterTopK`. |
| 3 | `topk_sigmoid` | Per-expert `sigmoid(logits)`, no row normalization. `topk(k)` selects the `k` largest scores. | DeepSeek-V2 / V3 scoring (Liu et al. 2024). |
| 4 | `group_limited_topk` | Partition experts into `n_group` blocks (Qwen: 12 groups of 5). Select `topk_group` highest-scoring groups (4) and run topk within them. With `e_score_correction_bias` for auxiliary-loss-free balancing. | DeepSeek-V3 (Liu et al. 2024) — wraps NxD's `GroupLimitedRouter`. |
| 5 | `sparsemax_topk` | Sparsemax projection of logits onto the probability simplex (exact zeros), then `topk(k)`. Cheaper than sparse softmax in the tail. | Martins & Astudillo, 2016 ("Sparsemax"); DSelect-K (Hazimeh et al. 2021). |

### Group B — SOTA but disruptive

| # | Name | Selection rule | Citation |
|---|---|---|---|
| 6 | `sinkhorn_balanced` | Sinkhorn-Knopp row/column normalization of `exp(logits)` over `~30` iterations enforces uniform expert load. Top-K selection runs on the *balanced* scores. | Megatron-LM (Korthikanti et al. 2022); Switch-Sinkhorn (Clark et al. 2022). Extends NxD's `RouterSinkhorn`. |
| 7 | `hash_routing` | `expert_index = MurmurHash(token_id ⊕ layer_idx) mod E` per token, drawn `top_k` distinct experts. **Zero learnable parameters.** | HASH Layer (Roller et al. 2021). |
| 8 | `expert_choice` | Each *expert* picks the top-`T` tokens (where `T = ⌈top_k · #tokens / E⌉`) by score. Inverts the optimization direction. | Zhou et al. 2022 ("Mixture-of-Experts with Expert Choice Routing"). |
| 9 | `base_layer` | Balanced assignment via one iteration of an auction-style algorithm (drop-in version of the linear-assignment routing from BASE Layer). | Lewis et al. 2021 ("BASE Layers"). |
| 10 | `soft_moe` | Replaces hard `top_k` with `S` "slots" per expert; each slot is a learned soft mixture over tokens. Output is a dense weighted sum of all experts. No sparse selection. | Puigcerver et al. 2023 ("From Sparse to Soft Mixtures of Experts"). |

## Fused NKI router shell

`fused_router_kernel.py` provides a single NKI kernel template that
fuses the `linear_router` GEMM + score function + `topk(E, K)` for the
Group-A routers. It's enabled by `--use-nki-router` on the run script.
The kernel takes a compile-time-constant `score_mode` argument
(`"softmax" | "softmax_over_topk" | "sigmoid" | "sparsemax" |
"group_limited"`) that NKI specializes away at compile time.

The Group-B routers stay on torch. NKI versions of Sinkhorn / BASE /
soft-MoE / expert-choice are doable but cost multiple engineer-weeks
each (Sinkhorn iterates 30x, BASE has data-dependent control flow,
SoftMoE is two GEMMs not one); they're documented as a follow-up
track rather than shipped here.

## Loading a Tier-T1 fine-tune checkpoint

```bash
python run_qwen_moe_trn2.py compile \
    --router-kernel expert_choice \
    --router-ft-checkpoint bench/finetuned/expert_choice/router_weights.safetensors
```

The factory transparently loads a `safetensors` (or `torch.save`)
state dict into the router after construction. Keys may be either
fully-qualified (matching `convert_qwen_moe_to_neuron_state_dict`'s
emitted layout) or local to the router module — the factory strips
the `layers.{l}.mlp.router.` prefix if present so a single checkpoint
file can carry weights for all 24 layers.

Routers that have no learnable parameters (`hash_routing`) silently
ignore the flag.

## Follow-up tracks (out of scope for the current project)

Both are documented for the next engineer who picks this up if T1
fine-tuning isn't sufficient for some Group B router (most likely
candidate: `soft_moe`, which changes the layer output shape
distribution).

* **T2 — LoRA on expert MLPs.** Add low-rank adapters to
  `gate_up_proj` and `down_proj` per routed expert. Trainable: ~14M
  params (vs. ~3M for routers alone). Wall-clock: ~3–5 days on a
  trn2.48xlarge per router. Expected to recover the remaining ~1–3
  MMLU points after T1.

* **T3 — Mid-training upcycle.** Full router + expert fine-tune on
  10–50B tokens. Recovers near-baseline quality on all Group B
  routers (including SoftMoE) but costs 1–3 weeks of trn2.48xlarge
  compute. Only worth it for SOTA paper claims.
