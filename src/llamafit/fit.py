"""Fit simulation and recommendations.

``simulate_fit`` mirrors the order of llama.cpp's built-in ``--fit``
(enabled by default there, target = leave ``--fit-target`` MiB free):

1. keep everything on GPU at the requested context if it fits;
2. otherwise shrink the context (never below ``--fit-ctx``, and never when
   the user pinned ``-c`` explicitly);
3. otherwise keep layers on GPU but push MoE expert tensors to the CPU
   (dense-first priority, like ``--n-cpu-moe``);
4. otherwise reduce the number of offloaded layers (``-ngl``).

``build_recommendations`` goes one step further than llama.cpp: instead of
only shrinking the context, it also evaluates KV-cache quantization
(``-ctk/-ctv q8_0 / q4_0``) so you can keep your target context.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import GIB, MIB, MemEstimate, RunConfig, estimate
from .model import ModelInfo

CTX_STEP = 256  # context granularity used when searching

# Symmetric K/V pairs keep llama.cpp's fused flash-attention kernels on the
# fast path; mismatched pairs silently fall back to a slower implementation.
KV_PRESETS = [
    ("f16", "f16"),
    ("q8_0", "q8_0"),
    ("q4_0", "q4_0"),
]

KV_QUALITY_NOTES = {
    "f16": "lossless baseline",
    "q8_0": "KV ×0.53, ≈lossless in practice",
    "q4_0": "KV ×0.28, some quality loss; slow at long ctx",
}


@dataclass
class FitResult:
    fits: bool
    cfg: RunConfig
    est: MemEstimate
    budget: int
    actions: list[str] = field(default_factory=list)

    @property
    def headroom(self) -> int:
        return self.budget - self.est.gpu_total


@dataclass
class KVScenario:
    """One -ctk/-ctv choice evaluated against the budget."""

    ctk: str
    ctv: str
    kv_bytes_at_target: int
    fits_full_at_target: bool
    max_ctx_full_offload: int      # 0 if even fit_ctx doesn't fit
    max_ngl_at_target: int
    est_at_target: MemEstimate


@dataclass
class Recommendation:
    title: str
    cfg: RunConfig
    est: MemEstimate
    reason: str


def _full_ngl(model: ModelInfo) -> int:
    return model.n_layer + 1


def _cfg(model: ModelInfo, base: RunConfig, **overrides) -> RunConfig:
    d = dict(
        n_ctx=base.n_ctx,
        n_gpu_layers=base.n_gpu_layers,
        n_cpu_moe=base.n_cpu_moe,
        cache_type_k=base.cache_type_k,
        cache_type_v=base.cache_type_v,
        flash_attn=base.flash_attn,
        n_ubatch=base.n_ubatch,
        n_batch=base.n_batch,
    )
    d.update(overrides)
    return RunConfig(**d)


def _gpu_total(model: ModelInfo, cfg: RunConfig, overhead: int) -> int:
    return estimate(model, cfg, overhead).gpu_total


def _search_max_ctx(
    model: ModelInfo, base: RunConfig, budget: int, overhead: int, lo: int, hi: int
) -> int:
    """Largest ctx in [lo, hi] (multiples of CTX_STEP) that fits, else 0."""
    lo_s, hi_s = max(1, lo // CTX_STEP), hi // CTX_STEP
    if lo_s > hi_s:
        return 0
    if _gpu_total(model, _cfg(model, base, n_ctx=lo_s * CTX_STEP), overhead) > budget:
        return 0
    while lo_s < hi_s:
        mid = (lo_s + hi_s + 1) // 2
        if _gpu_total(model, _cfg(model, base, n_ctx=mid * CTX_STEP), overhead) <= budget:
            lo_s = mid
        else:
            hi_s = mid - 1
    return lo_s * CTX_STEP


def _search_max_ngl(model: ModelInfo, base: RunConfig, budget: int, overhead: int) -> int:
    """Largest ngl in [0, n_layer+1] that fits (monotone in ngl)."""
    lo, hi = 0, _full_ngl(model)
    if _gpu_total(model, _cfg(model, base, n_gpu_layers=lo), overhead) > budget:
        return -1  # not even ngl=0 fits (weights aren't on GPU then — means budget < overhead)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _gpu_total(model, _cfg(model, base, n_gpu_layers=mid), overhead) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _search_min_cpu_moe(model: ModelInfo, base: RunConfig, budget: int, overhead: int) -> int:
    """Smallest --n-cpu-moe that fits with full offload, else -1."""
    full = _cfg(model, base, n_gpu_layers=_full_ngl(model))
    lo, hi = 0, model.n_layer
    if _gpu_total(model, _cfg(model, full, n_cpu_moe=hi), overhead) > budget:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if _gpu_total(model, _cfg(model, full, n_cpu_moe=mid), overhead) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return lo


def simulate_fit(
    model: ModelInfo,
    budget: int,
    base: RunConfig,
    overhead: int,
    fit_ctx: int = 4096,
    ctx_pinned: bool = False,
) -> FitResult:
    """Predict what llama.cpp --fit would settle on for this budget."""
    full = _full_ngl(model)
    want = _cfg(model, base, n_gpu_layers=full)
    actions: list[str] = []

    def done(cfg: RunConfig) -> FitResult:
        return FitResult(True, cfg, estimate(model, cfg, overhead), budget, actions)

    if _gpu_total(model, want, overhead) <= budget:
        return done(want)

    # step 1: shrink context (unless pinned by the user)
    if not ctx_pinned and base.n_ctx > fit_ctx:
        best = _search_max_ctx(model, want, budget, overhead, fit_ctx, base.n_ctx)
        if best:
            actions.append(f"context reduced {base.n_ctx} → {best}")
            return done(_cfg(model, want, n_ctx=best))
        actions.append(f"context reduced {base.n_ctx} → {fit_ctx} (fit-ctx floor)")
        want = _cfg(model, want, n_ctx=fit_ctx)

    # step 2: MoE experts to CPU, dense stays on GPU
    if model.is_moe:
        k = _search_min_cpu_moe(model, want, budget, overhead)
        if k >= 0:
            actions.append(f"expert tensors of first {k} layers moved to CPU (--n-cpu-moe {k})")
            return done(_cfg(model, want, n_cpu_moe=k))
        actions.append("all expert tensors to CPU — still not enough")
        want = _cfg(model, want, n_cpu_moe=model.n_layer)

    # step 3: reduce offloaded layers
    ngl = _search_max_ngl(model, want, budget, overhead)
    if ngl >= 0:
        actions.append(f"offloaded layers reduced {full} → {ngl}")
        cfg = _cfg(model, want, n_gpu_layers=ngl)
        return done(cfg)

    cfg = _cfg(model, want, n_gpu_layers=0)
    actions.append("does not fit at all — GPU budget below runtime overhead")
    return FitResult(False, cfg, estimate(model, cfg, overhead), budget, actions)


def evaluate_kv_scenarios(
    model: ModelInfo,
    budget: int | None,
    base: RunConfig,
    overhead: int,
    max_ctx_cap: int,
) -> list[KVScenario]:
    """Evaluate each KV-cache quantization preset at the target context."""
    out = []
    full = _full_ngl(model)
    for ctk, ctv in KV_PRESETS:
        cfg = _cfg(model, base, n_gpu_layers=full, cache_type_k=ctk, cache_type_v=ctv)
        est = estimate(model, cfg, overhead)
        if budget is not None:
            fits = est.gpu_total <= budget
            max_ctx = _search_max_ctx(model, cfg, budget, overhead, CTX_STEP, max_ctx_cap)
            ngl = _search_max_ngl(model, cfg, budget, overhead)
        else:
            fits, max_ctx, ngl = False, 0, full
        out.append(
            KVScenario(
                ctk=ctk,
                ctv=ctv,
                kv_bytes_at_target=est.gpu_kv + est.cpu_kv,
                fits_full_at_target=fits,
                max_ctx_full_offload=max_ctx,
                max_ngl_at_target=max(0, ngl),
                est_at_target=est,
            )
        )
    return out


def build_recommendations(
    model: ModelInfo,
    budget: int,
    base: RunConfig,
    overhead: int,
    scenarios: list[KVScenario],
) -> list[Recommendation]:
    """Ranked, concrete parameter sets that keep the target context."""
    recs: list[Recommendation] = []
    full = _full_ngl(model)
    by_kv = {(s.ctk, s.ctv): s for s in scenarios}

    f16 = by_kv[("f16", "f16")]
    q8 = by_kv[("q8_0", "q8_0")]
    q4 = by_kv[("q4_0", "q4_0")]

    if f16.fits_full_at_target:
        cfg = _cfg(model, base, n_gpu_layers=full)
        recs.append(
            Recommendation(
                "full offload, f16 KV",
                cfg,
                f16.est_at_target,
                "everything fits without compromises — no KV quantization needed",
            )
        )
        if q8.max_ctx_full_offload > f16.max_ctx_full_offload:
            cfg8 = _cfg(
                model, base, n_gpu_layers=full, cache_type_k="q8_0", cache_type_v="q8_0",
                n_ctx=min(q8.max_ctx_full_offload, model.n_ctx_train or q8.max_ctx_full_offload),
            )
            recs.append(
                Recommendation(
                    "more context with q8_0 KV",
                    cfg8,
                    estimate(model, cfg8, overhead),
                    f"-ctk q8_0 -ctv q8_0 stretches full-offload context to ~{cfg8.n_ctx:,}",
                )
            )
        return recs

    if q8.fits_full_at_target:
        cfg = _cfg(model, base, n_gpu_layers=full, cache_type_k="q8_0", cache_type_v="q8_0")
        recs.append(
            Recommendation(
                "full offload with q8_0 KV cache",
                cfg,
                q8.est_at_target,
                "f16 KV does not fit at this context; q8_0 halves the KV cache with "
                "negligible quality loss (keep K/V types symmetric for the fast FA path)",
            )
        )
    elif q4.fits_full_at_target:
        cfg = _cfg(model, base, n_gpu_layers=full, cache_type_k="q4_0", cache_type_v="q4_0")
        recs.append(
            Recommendation(
                "full offload with q4_0 KV cache",
                cfg,
                q4.est_at_target,
                "only q4_0 KV fits at this context — expect some quality loss and slower "
                "long-context decoding; consider a smaller context with q8_0 instead",
            )
        )

    # MoE: keep dense layers + KV on GPU, spill experts
    if model.is_moe:
        moe_base = _cfg(model, base, cache_type_k="q8_0", cache_type_v="q8_0")
        k = _search_min_cpu_moe(model, moe_base, budget, overhead)
        if k >= 0 and not q8.fits_full_at_target:
            cfg = _cfg(model, moe_base, n_gpu_layers=full, n_cpu_moe=k)
            recs.append(
                Recommendation(
                    f"MoE experts on CPU (--n-cpu-moe {k}) + q8_0 KV",
                    cfg,
                    estimate(model, cfg, overhead),
                    "for MoE models, spilling expert tensors is much faster than dropping "
                    "whole layers — attention & KV stay on GPU",
                )
            )

    # dense fallback: partial offload at target ctx
    if not model.is_moe and not q8.fits_full_at_target:
        if q8.max_ngl_at_target > 0:
            cfg = _cfg(
                model, base, n_gpu_layers=q8.max_ngl_at_target,
                cache_type_k="q8_0", cache_type_v="q8_0",
            )
            recs.append(
                Recommendation(
                    f"partial offload (-ngl {q8.max_ngl_at_target}) + q8_0 KV",
                    cfg,
                    estimate(model, cfg, overhead),
                    f"keeps your context; ~{q8.max_ngl_at_target}/{model.n_layer + 1} layers "
                    "on GPU, the rest run on CPU (slower prompt & token rate)",
                )
            )

    # alternative: keep quality, shrink context
    if q8.max_ctx_full_offload >= CTX_STEP and not q8.fits_full_at_target:
        cfg = _cfg(
            model, base, n_gpu_layers=full, n_ctx=q8.max_ctx_full_offload,
            cache_type_k="q8_0", cache_type_v="q8_0",
        )
        recs.append(
            Recommendation(
                f"or: full offload at ctx {q8.max_ctx_full_offload:,}",
                cfg,
                estimate(model, cfg, overhead),
                "if you can live with less context, full offload is the fastest option "
                "(this is what llama.cpp --fit chooses by default)",
            )
        )

    if not recs:
        ngl = max(0, _search_max_ngl(
            model, _cfg(model, base, cache_type_k="q8_0", cache_type_v="q8_0"), budget, overhead
        ))
        cfg = _cfg(
            model, base, n_gpu_layers=ngl, n_cpu_moe=model.n_layer if model.is_moe else 0,
            cache_type_k="q8_0", cache_type_v="q8_0",
        )
        recs.append(
            Recommendation(
                "model larger than this GPU",
                cfg,
                estimate(model, cfg, overhead),
                "even q4_0 KV with maximum spilling does not reach your target — "
                "consider a smaller quant, a smaller context, or more VRAM",
            )
        )
    return recs


def command_line(model_path: str, cfg: RunConfig, server: bool = True) -> str:
    """Suggested llama.cpp invocation for a config."""
    binary = "llama-server" if server else "llama-cli"
    parts = [binary, "-m", model_path, "-c", str(cfg.n_ctx), "-ngl", str(cfg.n_gpu_layers)]
    if cfg.flash_attn:
        parts += ["-fa", "on"]
    if (cfg.cache_type_k, cfg.cache_type_v) != ("f16", "f16"):
        parts += ["-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v]
    if cfg.n_cpu_moe > 0:
        parts += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
    if cfg.n_ubatch != 512:
        parts += ["-ub", str(cfg.n_ubatch)]
    return " ".join(parts)
