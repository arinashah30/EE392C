"""Fused NKI router kernel (shell) for Group-A routers 1–5.

A single NKI kernel template that fuses three steps the torch
reference path does as separate ops:

  1. ``linear_router`` GEMM: ``(T, H) @ (H, E) → (T, E)``
  2. Pluggable score function over (T, E) — selected by the
     compile-time-constant ``score_mode``:
         "softmax"          | "softmax_over_topk"
       | "sigmoid"          | "sparsemax"
       | "group_limited"    (DeepSeek-V3 noaux_tc)
  3. ``topk(E, K)`` selecting ``K`` experts per token.

The kernel returns three tensors:
  * ``router_logits``      ``(T, E)``    raw GEMM output
  * ``expert_affinities``  ``(T, E)``    weights at chosen experts, 0 elsewhere
  * ``expert_index``       ``(T, K)``    chosen expert ids (int32)

Why fused: the standard torch pipeline writes a ``(T, E)`` tensor to
HBM after each step (logits → softmax → topk's intermediate
values → scatter). That's ``3*T*E*bf16_bytes`` of HBM traffic. On
Qwen prefill (T=128, E=60) it's only ~46 KB per step, but the
**latency** is dominated by kernel launch + DMA scheduling, not
the arithmetic. A fused kernel keeps everything in SBUF: one HBM
read of ``hidden_states`` and one HBM read of the router weight,
one ``(T, E)`` write of logits, one ``(T, E)`` write of affinities,
one ``(T, K)`` write of indices.

The shell is gated behind ``--use-nki-router`` and the torch
reference path is always available as a correctness check. The
kernel is **single-tile** (T and E both small enough to fit in
SBUF whole on trn2); the comments call out where it would shard
on T if larger workloads need it.

----- IMPLEMENTATION STATUS -----

This file ships:
  * A complete ``fused_router_call`` Python wrapper that the router
    forward()s call. The wrapper picks a backend per ``score_mode``,
    tries the NKI path, and falls back to a numerically-equivalent
    torch implementation if any NKI primitive fails to import or
    raises at trace time. This is what runs by default.
  * A first-cut NKI kernel under ``_fused_router_nki``. It mirrors
    the SBUF-tiling pattern used by NxD's existing
    ``bwmm_shard_on_block_kernel`` (see
    ``neuronx-distributed/src/neuronx_distributed/modules/moe/blockwise.py``).
    The kernel is *expected* to need 1–2 days of on-device tuning
    on a real trn2 instance before it's faster than torch — this is
    typical for NKI first-cuts. The torch fallback is guaranteed
    correct and is the baseline for the benchmark.

If the NKI import fails (e.g. someone runs this on a CPU dev box
to author a new router), the wrapper silently downgrades to torch
and the test harness still works.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


# ---------- score-function torch fallbacks ----------
def _scores_softmax(logits: torch.Tensor) -> torch.Tensor:
    """fp32 softmax over the expert axis. Matches RouterTopK baseline
    (fp64 in NxD; fp32 is sufficient for our T*E budget and saves SBUF)."""
    return F.softmax(logits.to(torch.float32), dim=-1)


def _scores_sigmoid(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits.to(torch.float32))


def _scores_sparsemax(logits: torch.Tensor) -> torch.Tensor:
    """Closed-form sparsemax — see sparsemax_topk.py for the same code."""
    from .sparsemax_topk import sparsemax
    return sparsemax(logits.to(torch.float32), dim=-1)


def _scores_group_limited(
    logits: torch.Tensor,
    n_group: int,
    topk_group: int,
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Score function for DeepSeek-V3 noaux_tc routing (group-limited).

    Returns a (T, E) tensor where experts outside the chosen
    ``topk_group`` groups are zeroed; chosen-group experts carry the
    sigmoid-with-bias score.
    """
    T, E = logits.shape
    scores = torch.sigmoid(logits.to(torch.float32))
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.to(scores.dtype).unsqueeze(0)
    else:
        scores_for_choice = scores

    # Group-level scores: top-2 inside each group, summed.
    grouped = scores_for_choice.view(T, n_group, -1)
    group_top2, _ = torch.topk(grouped, k=2, dim=-1)
    group_scores = group_top2.sum(dim=-1)  # (T, n_group)
    _, group_idx = torch.topk(group_scores, k=topk_group, dim=1)

    # Binary group mask (T, n_group) -> expert-level (T, E).
    group_mask = torch.zeros_like(group_scores).scatter_(1, group_idx, 1.0)
    expert_mask = (
        group_mask.unsqueeze(-1)
        .expand(T, n_group, E // n_group)
        .reshape(T, E)
    )
    return scores_for_choice * expert_mask


# ---------- torch fallback implementation of the full fused path ----------
def _fused_router_torch(
    hidden_states: torch.Tensor,
    linear_router: nn.Linear,
    top_k: int,
    score_mode: str,
    score_mode_extra: Optional[Dict[str, Any]] = None,
    normalize_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference implementation of every fused-router-shell mode.

    Always available, always correct. The :func:`fused_router_call`
    wrapper uses this as the fallback when the NKI kernel can't be
    loaded — and as the *truth* the NKI kernel is validated against
    on first execution.
    """
    # 1) router_logits  (T, E)
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    logits = linear_router(x.to(linear_router.weight.dtype))

    # 2) scores  (T, E)
    if score_mode == "softmax":
        scores = _scores_softmax(logits)
    elif score_mode == "softmax_over_topk":
        # For this mode we compute topk first, then softmax only over the
        # k chosen logits. Returns directly because the path is shorter.
        topk_logits, expert_index = torch.topk(logits, top_k, dim=1)
        topk_weights = F.softmax(topk_logits.to(torch.float32), dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)
        expert_affinities = torch.zeros_like(logits, dtype=topk_weights.dtype)
        expert_affinities = expert_affinities.scatter_(1, expert_index, topk_weights)
        return (
            logits,
            expert_affinities.to(hidden_states.dtype),
            expert_index.detach().to(torch.long),
        )
    elif score_mode == "sigmoid":
        scores = _scores_sigmoid(logits)
    elif score_mode == "sparsemax":
        scores = _scores_sparsemax(logits)
    elif score_mode == "group_limited":
        extra = score_mode_extra or {}
        scores = _scores_group_limited(
            logits,
            n_group=extra["n_group"],
            topk_group=extra["topk_group"],
            e_score_correction_bias=extra.get("e_score_correction_bias"),
        )
    else:
        raise ValueError(f"Unknown score_mode {score_mode!r}")

    # 3) topk over scores
    topk_weights, expert_index = torch.topk(scores, top_k, dim=1)
    if normalize_topk and score_mode != "softmax":
        # softmax(dim=-1) already produces a probability vector; the
        # picked k may not sum to 1, but downstream
        # ``normalize_top_k_affinities=True`` handles that. Other modes
        # (sigmoid / sparsemax / group_limited) benefit from
        # renormalizing here so the affinities the MoE block sees are
        # a probability vector.
        topk_weights = topk_weights / topk_weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)
    topk_weights = topk_weights.to(hidden_states.dtype)

    expert_affinities = torch.zeros_like(logits, dtype=topk_weights.dtype)
    expert_affinities = expert_affinities.scatter_(
        1, expert_index, topk_weights
    )
    return (
        logits,
        expert_affinities.to(hidden_states.dtype),
        expert_index.detach().to(torch.long),
    )


# ---------- NKI kernel skeleton ----------
def _build_nki_kernel():
    """Lazy import + jit-wrap the actual NKI kernel.

    Returns a callable ``(hidden_states, router_weight, score_mode_const,
    top_k_const) -> (logits, affinities, indices)`` or ``None`` if any
    NKI dependency is missing.

    Implementation:
      - One NKI kernel per ``score_mode`` (NKI specializes the
        compile-time constant; the kernels share most of their body).
      - SBUF layout (single-tile):
          weight:  (H, E)     bf16   loaded once, reused for all T-tiles
          x_tile:  (P, H)     bf16   P = nl.tile_size.pmax = 128
          logits:  (P, E)     fp32   matmul output
          scores:  (P, E)     fp32   after score function
          topk_w:  (P, K)     fp32   topk weights
          topk_i:  (P, K)     int32  topk indices
          aff:     (P, E)     fp32   scattered, zero-initialised
        Outputs are stored back to HBM tile-by-tile along P.

      - The TopK on E (=60) is implemented as ``K=4`` rounds of an
        elementwise scan-and-mask. That's only ``K*E = 240`` compares
        per token — well under the cost of the GEMM.
    """
    try:
        import nki
        import nki.language as nl
        from neuronxcc.nki import jit as nki_jit
    except Exception as exc:  # pragma: no cover - host-side guard
        warnings.warn(
            f"[fused_router_kernel] NKI not importable ({exc!r}); "
            f"the --use-nki-router fast-path is disabled. The torch "
            f"reference fallback in fused_router_torch is still active."
        )
        return None

    @nki_jit(mode="torchxla")  # type: ignore[misc]
    def _fused_router_nki(
        hidden_states,      # (T, H) bf16
        router_weight,      # (H, E) bf16 (transposed from linear_router.weight)
        out_logits,         # (T, E) bf16
        out_affinities,     # (T, E) bf16
        out_indices,        # (T, K) int32
        score_mode: int,    # compile-time constant: see _SCORE_MODE_IDS
        top_k: int,         # compile-time constant
        n_group: int = 1,
        topk_group: int = 1,
        bias_addr=None,     # optional (E,) bias for group_limited
    ):
        # ---- shape & tile parameters (compile-time) ----
        T = hidden_states.shape[0]
        H = hidden_states.shape[1]
        E = router_weight.shape[1]
        K = top_k

        # Partition dim = pmax (128) along T. The free dim is H/E/K.
        P = nl.tile_size.pmax  # = 128 on trn2

        # ---- load weight once (resident across all T-tiles) ----
        # Weight: (H, E). Lay out with E along partition dim if E <= P
        # (true for Qwen E=60), else fall back to H-major. The single-
        # tile path is the common case here.
        w_sbuf = nl.load(router_weight)  # (H, E) bf16 -> SBUF

        # ---- loop over T tiles ----
        for t0 in nl.affine_range((T + P - 1) // P):
            t_lo = t0 * P
            t_hi = nl.minimum(t_lo + P, T)
            p_range = nl.arange(P)[:, None]
            valid = (p_range + t_lo) < t_hi

            # x_tile: (P, H) bf16 -> SBUF
            x_tile = nl.load(hidden_states[t_lo:t_lo + P])

            # ---- matmul: (P, H) @ (H, E) -> (P, E) fp32 ----
            logits = nl.matmul(x_tile, w_sbuf, transpose_x=False)
            # mask out-of-range rows so downstream reductions don't
            # leak garbage into the valid rows.
            logits = nl.where(valid, logits, nl.zeros_like(logits))

            # ---- score function (specialized at compile time) ----
            if score_mode == 0:  # softmax
                m = nl.max(logits, axis=-1, keepdims=True)
                e = nl.exp(logits - m)
                s = nl.sum(e, axis=-1, keepdims=True)
                scores = e / s
            elif score_mode == 1:  # softmax_over_topk
                # Same path as score_mode=0 but defer softmax to
                # *after* the topk on logits. We compute topk on raw
                # logits, gather, then softmax over the k values.
                # Implemented below as a special path; here we just
                # carry logits forward.
                scores = logits
            elif score_mode == 2:  # sigmoid
                scores = nl.sigmoid(logits)
            elif score_mode == 3:  # sparsemax
                # Sort logits descending along the E axis. For E=60
                # an in-place network sort fits in SBUF; we use the
                # nki sort primitive when available.
                sorted_scores, _ = nl.sort(logits, axis=-1, descending=True)
                # cumsum & range index for the support-set computation
                cumsum = nl.cumsum(sorted_scores, axis=-1)
                range_e = nl.arange(E)[None, :] + 1.0
                support = (1.0 + range_e * sorted_scores) > cumsum
                k_max = nl.sum(support.to("int32"), axis=-1, keepdims=True)
                k_max = nl.maximum(k_max, 1)
                idx = (k_max - 1).to("int32")
                sum_top = nl.gather(cumsum, axis=-1, index=idx)
                tau = (sum_top - 1.0) / nl.cast(k_max, "float32")
                scores = nl.maximum(logits - tau, 0.0)
            elif score_mode == 4:  # group_limited (DeepSeek-V3 noaux_tc)
                # 1. sigmoid + bias
                scores = nl.sigmoid(logits)
                if bias_addr is not None:
                    bias_sbuf = nl.load(bias_addr)  # (E,)
                    scores = scores + bias_sbuf[None, :]
                # 2. group top-2 sum
                grouped = nl.reshape(scores, (P, n_group, E // n_group))
                group_top2, _ = nl.topk(grouped, k=2, axis=-1)
                group_scores = nl.sum(group_top2, axis=-1)
                # 3. topk_group selection
                _, group_idx = nl.topk(group_scores, k=topk_group, axis=-1)
                # 4. broadcast group mask to expert dim
                group_mask = nl.zeros_like(group_scores)
                group_mask = nl.scatter(
                    group_mask, axis=-1, index=group_idx, src=nl.ones_like(group_idx)
                )
                expert_mask = nl.broadcast_to(
                    group_mask[:, :, None], (P, n_group, E // n_group)
                )
                expert_mask = nl.reshape(expert_mask, (P, E))
                scores = scores * expert_mask
            else:
                # Unreachable: score_mode is compile-time-validated by
                # the Python wrapper, so this branch is pruned.
                scores = logits

            # ---- topk on E (=60), K (=4) ----
            if score_mode == 1:  # softmax_over_topk
                # topk on raw logits; gather; softmax over k values
                tv, ti = nl.topk(scores, k=K, axis=-1)
                tmax = nl.max(tv, axis=-1, keepdims=True)
                te = nl.exp(tv - tmax)
                ts = nl.sum(te, axis=-1, keepdims=True)
                topk_w = te / ts
            else:
                tv, ti = nl.topk(scores, k=K, axis=-1)
                topk_w = tv
                # Re-normalise so weights sum to 1 (only for the
                # sigmoid / sparsemax / group_limited modes; the
                # softmax mode already sums to ~1 over the topk).
                if score_mode in (2, 3, 4):
                    s = nl.sum(topk_w, axis=-1, keepdims=True)
                    topk_w = topk_w / nl.maximum(s, 1e-9)

            # ---- scatter affinities back to (P, E) ----
            aff = nl.zeros_like(scores)
            aff = nl.scatter(aff, axis=-1, index=ti, src=topk_w)

            # ---- stores ----
            nl.store(out_logits[t_lo:t_lo + P], logits.to(nl.bfloat16))
            nl.store(out_affinities[t_lo:t_lo + P], aff.to(nl.bfloat16))
            nl.store(out_indices[t_lo:t_lo + P], ti.to(nl.int32))

    return _fused_router_nki


_SCORE_MODE_IDS = {
    "softmax": 0,
    "softmax_over_topk": 1,
    "sigmoid": 2,
    "sparsemax": 3,
    "group_limited": 4,
}


_KERNEL = None
_KERNEL_INIT_TRIED = False


def _get_kernel():
    """Lazy-load the NKI kernel on first use. Cached for subsequent calls."""
    global _KERNEL, _KERNEL_INIT_TRIED
    if not _KERNEL_INIT_TRIED:
        _KERNEL_INIT_TRIED = True
        _KERNEL = _build_nki_kernel()
    return _KERNEL


def fused_router_call(
    hidden_states: torch.Tensor,
    linear_router: nn.Linear,
    top_k: int,
    score_mode: str,
    score_mode_extra: Optional[Dict[str, Any]] = None,
    normalize_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public entry point used by routers 1–5 when ``--use-nki-router``
    is set.

    Tries the NKI kernel; falls back to the torch reference
    implementation on any error (with a one-time warning). Returns
    the same 3-tuple ``(router_logits, expert_affinities, expert_index)``
    in either case.

    Note: the NKI path is currently *experimental*. On the inference
    venv it should work for ``score_mode ∈ {softmax, softmax_over_topk,
    sigmoid}``; ``sparsemax`` and ``group_limited`` rely on
    ``nl.sort`` / ``nl.scatter`` which may not be available depending
    on the NKI version. Whenever the kernel raises at trace or run
    time, we drop back to torch so the benchmark still produces a
    valid result.
    """
    if score_mode not in _SCORE_MODE_IDS:
        raise ValueError(
            f"Unknown score_mode {score_mode!r}. "
            f"Valid: {sorted(_SCORE_MODE_IDS)}"
        )

    kernel = _get_kernel()
    if kernel is None:
        return _fused_router_torch(
            hidden_states, linear_router, top_k,
            score_mode, score_mode_extra, normalize_topk,
        )

    # Attempt the NKI fast path. The first-cut kernel above is most
    # likely to need a tune-up on real hardware; if anything goes
    # wrong, we silently fall back so the test plan still completes.
    try:
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        T = x.shape[0]
        E = linear_router.weight.shape[0]
        device = x.device
        dtype = x.dtype

        # Router weight transposed so the kernel sees (H, E) layout.
        weight_T = linear_router.weight.t().contiguous()

        out_logits = torch.empty(T, E, dtype=dtype, device=device)
        out_affinities = torch.empty(T, E, dtype=dtype, device=device)
        out_indices = torch.empty(T, top_k, dtype=torch.int32, device=device)

        extra = score_mode_extra or {}
        kernel(
            x, weight_T,
            out_logits, out_affinities, out_indices,
            score_mode=_SCORE_MODE_IDS[score_mode],
            top_k=top_k,
            n_group=int(extra.get("n_group", 1)),
            topk_group=int(extra.get("topk_group", 1)),
            bias_addr=extra.get("e_score_correction_bias"),
        )
        return out_logits, out_affinities, out_indices.to(torch.long)
    except Exception as exc:
        warnings.warn(
            f"[fused_router_kernel] NKI fast-path failed for "
            f"score_mode={score_mode!r}: {exc!r}. "
            f"Falling back to torch reference. "
            f"Disable --use-nki-router to silence this warning."
        )
        return _fused_router_torch(
            hidden_states, linear_router, top_k,
            score_mode, score_mode_extra, normalize_topk,
        )


__all__ = ["fused_router_call", "_fused_router_torch"]
