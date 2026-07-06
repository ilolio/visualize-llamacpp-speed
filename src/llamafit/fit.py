"""Fit simulation and recommendations.

``simulate_fit`` mirrors the order of llama.cpp's built-in ``--fit``
(enabled by default there, target = leave ``--fit-target`` MiB free):

1. keep everything on GPU at the requested context if it fits;
2. otherwise shrink the context (never below ``--fit-ctx``);
3. otherwise keep layers on GPU but push MoE expert tensors to the CPU
   (dense-first priority, like ``--n-cpu-moe``);
4. otherwise reduce the number of offloaded layers (``-ngl``).

Parameters the user pinned explicitly (``Pins``) are treated as
constraints: the pinned dimension is held fixed and the remaining free
dimensions are searched.  Pin everything and the simulation degenerates
into a plain "does this exact config fit?" check.

``build_recommendations`` goes one step further than llama.cpp: instead of
only shrinking the context, it also evaluates KV-cache quantization
(``-ctk/-ctv q8_0 / q4_0``) so you can keep your target context — unless
the KV types are pinned, in which case it keeps them and recommends the
free parameters instead (e.g. the largest context that still fits).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import MIB, MemEstimate, RunConfig, estimate
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
    "f32": "2× f16 memory — debugging only",
    "f16": "lossless baseline",
    "bf16": "same size as f16",
    "q8_0": "KV ×0.53, ≈lossless in practice",
    "q5_1": "KV ×0.38, minor quality loss",
    "q5_0": "KV ×0.34, minor quality loss",
    "q4_1": "KV ×0.31, some quality loss",
    "q4_0": "KV ×0.28, some quality loss; slow at long ctx",
    "iq4_nl": "KV ×0.28, some quality loss",
}


@dataclass(frozen=True)
class Pins:
    """Which parameters the user fixed explicitly (they are never changed)."""

    ctx: bool = False   # -c
    ngl: bool = False   # --ngl
    kv: bool = False    # --ctk / --ctv
    moe: bool = False   # --n-cpu-moe

    def describe(self, cfg: RunConfig) -> str:
        parts = []
        if self.ctx:
            parts.append(f"-c {cfg.n_ctx}")
        if self.ngl:
            parts.append(f"-ngl {cfg.n_gpu_layers}")
        if self.kv:
            parts.append(f"-ctk/-ctv {cfg.cache_type_k}/{cfg.cache_type_v}")
        if self.moe:
            parts.append(f"--n-cpu-moe {cfg.n_cpu_moe}")
        return ", ".join(parts)


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
    """Smallest --n-cpu-moe that fits at base's -ngl, else -1."""
    lo, hi = 0, model.n_layer
    if _gpu_total(model, _cfg(model, base, n_cpu_moe=hi), overhead) > budget:
        return -1
    while lo < hi:
        mid = (lo + hi) // 2
        if _gpu_total(model, _cfg(model, base, n_cpu_moe=mid), overhead) <= budget:
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
    pins: Pins = Pins(),
) -> FitResult:
    """Predict what llama.cpp --fit would settle on for this budget.

    Dimensions listed in ``pins`` are held fixed; only the free ones are
    adjusted.
    """
    full = _full_ngl(model)
    want = base if pins.ngl else _cfg(model, base, n_gpu_layers=full)
    actions: list[str] = []
    if pinned := pins.describe(base):
        actions.append(f"pinned by you, kept fixed: {pinned}")

    def done(cfg: RunConfig) -> FitResult:
        return FitResult(True, cfg, estimate(model, cfg, overhead), budget, actions)

    if _gpu_total(model, want, overhead) <= budget:
        return done(want)

    # step 1: shrink context (unless pinned by the user)
    if not pins.ctx and base.n_ctx > fit_ctx:
        best = _search_max_ctx(model, want, budget, overhead, fit_ctx, base.n_ctx)
        if best:
            actions.append(f"context reduced {base.n_ctx} → {best}")
            return done(_cfg(model, want, n_ctx=best))
        actions.append(f"context reduced {base.n_ctx} → {fit_ctx} (fit-ctx floor)")
        want = _cfg(model, want, n_ctx=fit_ctx)

    # step 2: MoE experts to CPU, dense stays on GPU
    if model.is_moe and not pins.moe:
        k = _search_min_cpu_moe(model, want, budget, overhead)
        if k >= 0:
            actions.append(f"expert tensors of first {k} layers moved to CPU (--n-cpu-moe {k})")
            return done(_cfg(model, want, n_cpu_moe=k))
        actions.append("all expert tensors to CPU — still not enough")
        want = _cfg(model, want, n_cpu_moe=model.n_layer)

    # step 3: reduce offloaded layers
    if not pins.ngl:
        ngl = _search_max_ngl(model, want, budget, overhead)
        if ngl >= 0:
            actions.append(f"offloaded layers reduced {full} → {ngl}")
            cfg = _cfg(model, want, n_gpu_layers=ngl)
            return done(cfg)
        want = _cfg(model, want, n_gpu_layers=0)
        actions.append("does not fit at all — GPU budget below runtime overhead")
    else:
        over = _gpu_total(model, want, overhead) - budget
        actions.append(
            f"over budget by {over / MIB:.0f} MiB — pinned parameters leave nothing to adjust"
        )
    return FitResult(False, want, estimate(model, want, overhead), budget, actions)


def evaluate_kv_scenarios(
    model: ModelInfo,
    budget: int | None,
    base: RunConfig,
    overhead: int,
    max_ctx_cap: int,
) -> list[KVScenario]:
    """Evaluate each KV-cache quantization preset at the target context.

    The pair from ``base`` (e.g. a user-pinned ``--ctk/--ctv``) is always
    included, so exotic types like ``q5_1`` or asymmetric K/V pairs show up
    in the comparison too.
    """
    out = []
    full = _full_ngl(model)
    presets = list(KV_PRESETS)
    if (base.cache_type_k, base.cache_type_v) not in presets:
        presets.insert(0, (base.cache_type_k, base.cache_type_v))
    for ctk, ctv in presets:
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
    pins: Pins = Pins(),
    max_ctx_cap: int | None = None,
) -> list[Recommendation]:
    """Ranked, concrete parameter sets that keep the target context.

    Pinned parameters are constraints: a pinned KV type is never swapped for
    a different quantization, a pinned ``-ngl``/``--n-cpu-moe`` is never
    changed — the free dimensions carry the adjustment instead.
    """
    recs: list[Recommendation] = []
    full = _full_ngl(model)
    ngl = base.n_gpu_layers if pins.ngl else full
    offload = "full offload" if ngl >= full else f"-ngl {ngl}"
    if max_ctx_cap is None:
        max_ctx_cap = max(base.n_ctx, model.n_ctx_train or base.n_ctx)
    by_kv = {(s.ctk, s.ctv): s for s in scenarios}
    pinned_pair = (base.cache_type_k, base.cache_type_v)
    # quality ladder walked when looking for a KV type that fits
    ladder = [pinned_pair] if pins.kv else list(KV_PRESETS)

    def kv_label(pair: tuple[str, str]) -> str:
        return pair[0] if pair[0] == pair[1] else f"{pair[0]}/{pair[1]}"

    def eval_pair(pair: tuple[str, str]) -> tuple[bool, int, MemEstimate]:
        """(fits at target ctx, max ctx, estimate) for a KV pair at our ngl."""
        s = by_kv.get(pair)
        if s is not None and not pins.ngl:
            return s.fits_full_at_target, s.max_ctx_full_offload, s.est_at_target
        cfg = _cfg(model, base, n_gpu_layers=ngl,
                   cache_type_k=pair[0], cache_type_v=pair[1])
        est = estimate(model, cfg, overhead)
        max_ctx = _search_max_ctx(model, cfg, budget, overhead, CTX_STEP, max_ctx_cap)
        return est.gpu_total <= budget, max_ctx, est

    first_fits, first_max_ctx, first_est = eval_pair(ladder[0])

    if first_fits:
        cfg = _cfg(model, base, n_gpu_layers=ngl,
                   cache_type_k=ladder[0][0], cache_type_v=ladder[0][1])
        recs.append(
            Recommendation(
                f"{offload}, {kv_label(ladder[0])} KV",
                cfg,
                first_est,
                "your pinned KV type fits at the target context — no compromises needed"
                if pins.kv else
                "everything fits without compromises — no KV quantization needed",
            )
        )
        if not pins.ctx:
            if pins.kv or pins.ngl:
                # headroom left: same settings, larger context
                grow = min(first_max_ctx, model.n_ctx_train or first_max_ctx)
                if grow > base.n_ctx:
                    cfg_more = _cfg(model, cfg, n_ctx=grow)
                    recs.append(
                        Recommendation(
                            f"more context at {kv_label(ladder[0])} KV",
                            cfg_more,
                            estimate(model, cfg_more, overhead),
                            f"headroom left — -c can grow to ~{grow:,} with your KV type "
                            f"({offload})",
                        )
                    )
            elif not pins.ngl:
                q8 = by_kv[("q8_0", "q8_0")]
                if q8.max_ctx_full_offload > first_max_ctx:
                    cfg8 = _cfg(
                        model, base, n_gpu_layers=full,
                        cache_type_k="q8_0", cache_type_v="q8_0",
                        n_ctx=min(q8.max_ctx_full_offload,
                                  model.n_ctx_train or q8.max_ctx_full_offload),
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

    # first choice doesn't fit — walk down the quality ladder (KV not pinned)
    for pair in ladder[1:]:
        fits, _max_ctx, est = eval_pair(pair)
        if not fits:
            continue
        cfg = _cfg(model, base, n_gpu_layers=ngl,
                   cache_type_k=pair[0], cache_type_v=pair[1])
        if pair == ("q8_0", "q8_0"):
            reason = ("f16 KV does not fit at this context; q8_0 halves the KV cache with "
                      "negligible quality loss (keep K/V types symmetric for the fast FA path)")
        else:
            reason = ("only q4_0 KV fits at this context — expect some quality loss and slower "
                      "long-context decoding; consider a smaller context with q8_0 instead")
        recs.append(Recommendation(f"{offload} with {kv_label(pair)} KV cache", cfg, est, reason))
        break

    # remaining fallbacks use the pinned KV type, or q8_0 when KV is free
    fb = pinned_pair if pins.kv else ("q8_0", "q8_0")
    fb_fits, fb_max_ctx, _fb_est = eval_pair(fb)
    fb_flag = "" if pins.kv else " + q8_0 KV"

    if not fb_fits:
        # MoE: keep dense layers + KV on GPU, spill experts
        if model.is_moe and not pins.moe:
            moe_base = _cfg(model, base, n_gpu_layers=ngl,
                            cache_type_k=fb[0], cache_type_v=fb[1])
            k = _search_min_cpu_moe(model, moe_base, budget, overhead)
            if k >= 0:
                cfg = _cfg(model, moe_base, n_cpu_moe=k)
                recs.append(
                    Recommendation(
                        f"MoE experts on CPU (--n-cpu-moe {k}){fb_flag}",
                        cfg,
                        estimate(model, cfg, overhead),
                        "for MoE models, spilling expert tensors is much faster than dropping "
                        "whole layers — attention & KV stay on GPU",
                    )
                )

        # dense fallback: partial offload at target ctx
        if not model.is_moe and not pins.ngl:
            fbs = by_kv.get(fb)
            best_ngl = fbs.max_ngl_at_target if fbs else 0
            if best_ngl > 0:
                cfg = _cfg(model, base, n_gpu_layers=best_ngl,
                           cache_type_k=fb[0], cache_type_v=fb[1])
                recs.append(
                    Recommendation(
                        f"partial offload (-ngl {best_ngl}){fb_flag}",
                        cfg,
                        estimate(model, cfg, overhead),
                        f"keeps your context; ~{best_ngl}/{model.n_layer + 1} layers "
                        "on GPU, the rest run on CPU (slower prompt & token rate)",
                    )
                )

        # alternative: keep quality, shrink context
        if not pins.ctx and fb_max_ctx >= CTX_STEP:
            cfg = _cfg(model, base, n_gpu_layers=ngl, n_ctx=fb_max_ctx,
                       cache_type_k=fb[0], cache_type_v=fb[1])
            recs.append(
                Recommendation(
                    ("or: " if recs else "") + f"{offload} at ctx {fb_max_ctx:,}",
                    cfg,
                    estimate(model, cfg, overhead),
                    "if you can live with less context, full GPU offload is the fastest option "
                    "(this is what llama.cpp --fit chooses by default)",
                )
            )

    if not recs:
        ngl_fb = ngl if pins.ngl else max(0, _search_max_ngl(
            model, _cfg(model, base, cache_type_k=fb[0], cache_type_v=fb[1]), budget, overhead
        ))
        cfg = _cfg(
            model, base, n_gpu_layers=ngl_fb,
            n_cpu_moe=base.n_cpu_moe if (pins.moe or not model.is_moe) else model.n_layer,
            cache_type_k=fb[0], cache_type_v=fb[1],
        )
        recs.append(
            Recommendation(
                "model larger than this GPU",
                cfg,
                estimate(model, cfg, overhead),
                "your pinned parameters cannot reach the target — free a pinned flag, or "
                "consider a smaller quant, a smaller context, or more VRAM"
                if (pins.kv or pins.ngl or pins.moe) else
                "even q4_0 KV with maximum spilling does not reach your target — "
                "consider a smaller quant, a smaller context, or more VRAM",
            )
        )
    return recs


def _model_args(source: str) -> list[str]:
    """Map llamafit's model argument to the matching llama.cpp flag.

    Local files use ``-m``; ``-hf`` style specs (``org/repo[:QUANT]``,
    ``hf:...``, ``hf.co/...``) use ``-hf`` (plus ``--hf-file`` for direct-file
    specs); http(s) URLs use ``-mu``.  Mirrors ``load_gguf``'s precedence:
    an existing local path always wins.
    """
    from pathlib import Path

    from .gguf import parse_hf_spec

    if Path(source).expanduser().exists():
        return ["-m", source]
    if source.startswith(("http://", "https://")):
        return ["-mu", source]
    if source.startswith("hf:") and source.lower().endswith(".gguf"):
        parts = source[3:].lstrip("/").split("/")
        if len(parts) >= 3:
            return ["-hf", f"{parts[0]}/{parts[1]}", "--hf-file", "/".join(parts[2:])]
    spec = parse_hf_spec(source)
    if spec is not None:
        repo, tag = spec
        return ["-hf", f"{repo}:{tag}" if tag else repo]
    return ["-m", source]


def command_line(model_path: str, cfg: RunConfig, server: bool = True) -> str:
    """Suggested llama.cpp invocation for a config."""
    binary = "llama-server" if server else "llama-cli"
    parts = [binary, *_model_args(model_path),
             "-c", str(cfg.n_ctx), "-ngl", str(cfg.n_gpu_layers)]
    if cfg.flash_attn:
        parts += ["-fa", "on"]
    if (cfg.cache_type_k, cfg.cache_type_v) != ("f16", "f16"):
        parts += ["-ctk", cfg.cache_type_k, "-ctv", cfg.cache_type_v]
    if cfg.n_cpu_moe > 0:
        parts += ["--n-cpu-moe", str(cfg.n_cpu_moe)]
    if cfg.n_ubatch != 512:
        parts += ["-ub", str(cfg.n_ubatch)]
    return " ".join(parts)
