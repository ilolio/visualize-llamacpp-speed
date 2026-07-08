"""llamafit command-line interface."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

from rich.console import Console
from rich.text import Text

from . import __version__
from .fit import Pins, build_recommendations, evaluate_kv_scenarios, simulate_fit
from .gguf import GGUFError, discover_mmproj, load_gguf, load_mmproj
from .gpu import detect_gpus, system_ram
from .memory import GIB, MIB, RunConfig, estimate, total_kv_bytes
from .model import extract_model_info
from .render import (
    budget_section,
    fit_panel,
    fmt_bytes,
    kv_table,
    layer_strip,
    model_panel,
    ram_section,
    recommendation_panels,
    requirement_section,
    vram_section,
)

DEFAULT_CTX_CAP = 32768
MAX_SEARCH_CTX = 1 << 22  # 4M — upper bound for "max context" searches


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="llamafit",
        description=(
            "Visualize how a GGUF model fits into VRAM (weights + KV cache) and get "
            "llama.cpp parameter recommendations (-ngl, -c, -ctk/-ctv, --n-cpu-moe). "
            "Only the GGUF header is read — works on local files, split shards, "
            "llama.cpp -hf style specs (org/repo[:QUANT]), hf:org/repo/file.gguf "
            "and https:// URLs."
        ),
    )
    p.add_argument(
        "model",
        help="GGUF path, llama.cpp -hf style spec (org/repo[:QUANT], e.g. "
             "unsloth/Qwen3.5-9B-GGUF:Q4_K_M), hf:org/repo/file.gguf, or https:// URL",
    )
    p.add_argument(
        "-c", "--ctx", type=int, default=None,
        help="target context size (default: min(training ctx, 32768))",
    )
    p.add_argument("--ngl", type=int, default=None,
                   help="pin -ngl (fit & recommendations keep it and adjust the rest)")
    p.add_argument("--ctk", "--cache-type-k", dest="ctk", default=None,
                   help="pin KV cache K type (f16, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, ...); "
                        "recommendations keep it and adjust context/offload instead")
    p.add_argument("--ctv", "--cache-type-v", dest="ctv", default=None,
                   help="pin KV cache V type (defaults to --ctk when only that is given)")
    p.add_argument("--n-cpu-moe", type=int, default=None,
                   help="pin --n-cpu-moe (expert tensors of first N layers on CPU; default: 0)")
    p.add_argument("--mmproj", default=None, metavar="SRC",
                   help="multimodal projector GGUF (path, https:// URL, or hf:org/repo/file.gguf) "
                        "to include in the VRAM math; auto-detected otherwise")
    p.add_argument("--no-mmproj", action="store_true",
                   help="do not auto-detect or include a multimodal projector")
    p.add_argument("--no-mmproj-offload", action="store_true",
                   help="keep the multimodal projector on CPU (llama.cpp --no-mmproj-offload)")
    p.add_argument("--vram", type=float, default=None, metavar="GIB",
                   help="total VRAM budget in GiB (default: auto-detect via nvidia-smi/rocm-smi/Metal)")
    p.add_argument("--fit-target", type=int, default=1024, metavar="MIB",
                   help="free VRAM to keep in reserve, like llama.cpp --fit-target (default: 1024)")
    p.add_argument("--fit-ctx", type=int, default=4096, metavar="N",
                   help="minimum context the fit simulation may shrink to (default: 4096)")
    p.add_argument("--overhead", type=int, default=500, metavar="MIB",
                   help="estimated runtime overhead per GPU, e.g. CUDA context (default: 500)")
    p.add_argument("-ub", "--ubatch", type=int, default=512, help="micro-batch size (default: 512)")
    p.add_argument("-b", "--batch", type=int, default=2048, help="logical batch size (default: 2048)")
    p.add_argument("--fa", choices=["on", "off"], default="on",
                   help="assume flash attention on/off (default: on)")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--version", action="version", version=f"llamafit {__version__}")
    return p


def _est_dict(est) -> dict:
    d = dataclasses.asdict(est)
    d["gpu_total"] = est.gpu_total
    d["cpu_total"] = est.cpu_total
    return d


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    console = Console(stderr=False)
    err = Console(stderr=True)

    try:
        gguf = load_gguf(args.model)
        model = extract_model_info(gguf)
    except GGUFError as e:
        err.print(f"[red]error:[/red] {e}")
        return 1

    if model.n_layer <= 0 or model.n_embd <= 0:
        err.print(f"[red]error:[/red] {args.model} has no usable hyperparameters "
                  f"(arch={model.arch!r}) — is this an mmproj/adapter file?")
        return 1

    # --- multimodal projector (mmproj): a separate GGUF loaded alongside the
    # model.  Explicit --mmproj wins; otherwise auto-detect a sibling / repo file. ---
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    mmproj_source = args.mmproj
    if mmproj_source is None and not args.no_mmproj:
        mmproj_source = discover_mmproj(args.model, token)
    if mmproj_source is not None:
        try:
            model.mmproj_name, model.mmproj_bytes, _ = load_mmproj(mmproj_source)
            model.mmproj_source = mmproj_source
            model.mmproj_on_gpu = not args.no_mmproj_offload
        except GGUFError as e:
            err.print(f"[yellow]warning:[/yellow] could not read mmproj {mmproj_source!r}: {e}")

    # --- resolve budget ---
    gpus = detect_gpus()
    overhead = args.overhead * MIB
    if args.vram is not None:
        budget_total = int(args.vram * GIB)
        budget_note = f"budget = --vram {args.vram:g} GiB − fit-target {args.fit_target} MiB"
    elif gpus:
        budget_total = sum(g.free for g in gpus)
        names = ", ".join(f"{g.name} ({fmt_bytes(g.total)}, {fmt_bytes(g.free)} free)" for g in gpus)
        budget_note = f"budget = free VRAM on {names} − fit-target {args.fit_target} MiB"
        if len(gpus) > 1:
            budget_note += " — multiple GPUs pooled; actual split depends on --tensor-split"
    else:
        budget_total = None
        budget_note = "no GPU detected — pass --vram to enable fit simulation"
    budget = (budget_total - args.fit_target * MIB) if budget_total is not None else None

    # --- validate pinned parameters ---
    from .memory import KV_CACHE_TYPES

    for flag, val in (("--ctk", args.ctk), ("--ctv", args.ctv)):
        if val is not None and val not in KV_CACHE_TYPES:
            err.print(f"[red]error:[/red] {flag} {val!r} is not a valid KV cache type "
                      f"(choose from {', '.join(KV_CACHE_TYPES)})")
            return 1
    if args.ngl is not None:
        args.ngl = max(0, min(args.ngl, model.n_layer + 1))

    # --- target context ---
    pins = Pins(
        ctx=args.ctx is not None,
        ngl=args.ngl is not None,
        kv=args.ctk is not None or args.ctv is not None,
        moe=args.n_cpu_moe is not None,
    )
    train_ctx = model.n_ctx_train or DEFAULT_CTX_CAP
    target_ctx = args.ctx if pins.ctx else min(train_ctx, DEFAULT_CTX_CAP)

    base = RunConfig(
        n_ctx=target_ctx,
        n_gpu_layers=args.ngl if args.ngl is not None else model.n_layer + 1,
        n_cpu_moe=args.n_cpu_moe or 0,
        cache_type_k=args.ctk or args.ctv or "f16",
        cache_type_v=args.ctv or args.ctk or "f16",
        flash_attn=args.fa == "on",
        n_ubatch=args.ubatch,
        n_batch=args.batch,
    )

    max_ctx_cap = min(max(train_ctx, target_ctx), MAX_SEARCH_CTX)
    scenarios = evaluate_kv_scenarios(model, budget, base, overhead, max_ctx_cap)

    fit = None
    recs = []
    if budget is not None:
        # pinned parameters are held fixed; the free dimensions are fitted
        fit = simulate_fit(model, budget, base, overhead,
                           fit_ctx=min(args.fit_ctx, target_ctx), pins=pins)
        recs = build_recommendations(model, budget, base, overhead, scenarios,
                                     pins=pins, max_ctx_cap=max_ctx_cap)

    # --- JSON mode ---
    if args.json:
        out = {
            "model": {
                "name": model.name, "arch": model.arch, "quant": model.quant,
                "n_params": model.n_params, "file_size": model.file_size,
                "n_layer": model.n_layer, "n_embd": model.n_embd,
                "n_head": model.n_head, "n_head_kv": model.n_head_kv,
                "n_ctx_train": model.n_ctx_train, "n_vocab": model.n_vocab,
                "n_expert": model.n_expert, "moe": model.is_moe,
                "swa_window": model.swa_window, "mla": model.mla,
                "weight_bytes_total": model.weight_bytes_total,
                "mmproj": None if not model.mmproj_bytes else {
                    "name": model.mmproj_name,
                    "source": model.mmproj_source,
                    "weight_bytes": model.mmproj_bytes,
                    "on_gpu": model.mmproj_on_gpu,
                },
            },
            "budget": {
                "bytes": budget,
                "note": budget_note,
                "fit_target_bytes": args.fit_target * MIB,
                "vram_total_bytes": sum(g.total for g in gpus) if gpus else None,
                "vram_used_bytes": sum(g.total - g.free for g in gpus) if gpus else None,
                "vram_free_bytes": sum(g.free for g in gpus) if gpus else None,
                "gpus": [dataclasses.asdict(g) for g in gpus],
            },
            "target_ctx": target_ctx,
            "requirement": {
                "weights_bytes": model.weight_bytes_total,
                "mmproj_bytes": model.mmproj_bytes,
                "kv_bytes": total_kv_bytes(model, base),
                "total_bytes": (model.weight_bytes_total + model.mmproj_bytes
                                + total_kv_bytes(model, base)),
                "ctx": base.n_ctx,
                "cache_type_k": base.cache_type_k,
                "cache_type_v": base.cache_type_v,
            },
            "kv_scenarios": [
                {**dataclasses.asdict(s), "est_at_target": _est_dict(s.est_at_target)}
                for s in scenarios
            ],
            "fit": None if fit is None else {
                "fits": fit.fits, "actions": fit.actions,
                "config": dataclasses.asdict(fit.cfg), "estimate": _est_dict(fit.est),
            },
            "recommendations": [
                {"title": r.title, "reason": r.reason,
                 "config": dataclasses.asdict(r.cfg), "estimate": _est_dict(r.est)}
                for r in recs
            ],
        }
        print(json.dumps(out, indent=2))
        return 0

    # --- render ---
    console.print()
    console.print(model_panel(model))

    # VRAM budget formula + total / used / free VRAM (multiple GPUs pooled)
    console.print()
    console.print(budget_section(gpus, budget, args.fit_target, args.vram))

    display_cfg = fit.cfg if fit is not None else base
    display_est = fit.est if fit is not None else estimate(model, base, overhead)

    # 1) raw demand of the requested config (weights + KV, before any offload)
    console.print()
    console.print(requirement_section(model, base, budget))

    # 2) actual GPU usage once the fit/offload decisions are applied
    if budget is not None and fit is not None:
        offloaded = (
            display_cfg.n_gpu_layers <= model.n_layer
            or display_cfg.n_cpu_moe > 0
        )
        console.print()
        console.print(vram_section(display_est, budget, display_cfg, budget_note, offloaded))
        # 3) what lands in host RAM after those offload decisions
        console.print()
        console.print(ram_section(display_est, system_ram()))
        console.print()
        console.print(layer_strip(model, display_cfg))

    console.print()
    console.print(kv_table(scenarios, target_ctx, budget is not None, model))
    console.print(Text(
        "*compute buffer is an estimate (±15%); weights & KV are exact from GGUF metadata",
        style="dim",
    ))
    for note in display_est.notes:
        console.print(Text(f"note: {note}", style="yellow"))

    if fit is not None:
        console.print()
        console.print(fit_panel(fit))

    if recs:
        console.print()
        console.print(Text("Recommendations", style="bold underline"))
        for panel in recommendation_panels(recs, args.model, budget,
                                            model.mmproj_source or None, model.mmproj_on_gpu):
            console.print(panel)

    if budget is None:
        console.print()
        console.print(Text(
            "tip: pass --vram <GiB> (or run on the GPU machine) to unlock fit simulation "
            "and recommendations",
            style="dim",
        ))
    console.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
