"""Terminal rendering with rich.

Colors follow a fixed role → hue assignment (CVD-safe categorical set):
weights = blue, KV cache = aqua, compute = yellow, overhead = grey.
Every colored mark is accompanied by a text label — identity is never
carried by color alone.
"""

from __future__ import annotations

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .fit import FitResult, KVScenario, Recommendation, command_line
from .gpu import GPU
from .memory import GIB, MIB, MemEstimate, RunConfig, total_kv_bytes
from .model import ModelInfo

C_WEIGHTS = "#3987e5"   # blue
C_KV = "#199e70"        # aqua
C_COMPUTE = "#c98500"   # yellow
C_OVERHEAD = "#8a8983"  # grey
C_FREE = "grey30"
C_OVER = "#e66767"      # red (status: over budget)
C_CPU = "grey42"

BAR_WIDTH = 60


def fmt_bytes(n: float) -> str:
    if n >= GIB:
        return f"{n / GIB:.2f} GiB"
    if n >= MIB:
        return f"{n / MIB:.0f} MiB"
    return f"{n / 1024:.0f} KiB"


def fmt_params(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    return f"{n / 1e6:.0f}M"


def _bar(segments: list[tuple[str, int, str]], budget: int | None, width: int = BAR_WIDTH) -> Text:
    """One-line stacked bar. segments = [(label, bytes, color)]."""
    total = sum(s[1] for s in segments)
    scale_to = max(total, budget or 0, 1)
    text = Text()
    drawn = 0
    for _label, size, color in segments:
        if size <= 0:
            continue
        cells = max(1, round(size / scale_to * width))
        cells = min(cells, width - drawn)
        text.append("█" * cells, style=color)
        drawn += cells
    if budget is not None and total < budget:
        free_cells = max(0, round(budget / scale_to * width) - drawn)
        text.append("░" * free_cells, style=C_FREE)
        drawn += free_cells
    return text


def _legend(entries: list[tuple[str, int, str]]) -> Text:
    text = Text()
    for i, (label, size, color) in enumerate(entries):
        if i:
            text.append("   ")
        text.append("■ ", style=color)
        text.append(f"{label} ", style="bold")
        text.append(fmt_bytes(size), style="default")
    return text


def model_panel(model: ModelInfo) -> Panel:
    t = Table.grid(padding=(0, 3))
    t.add_column(style="dim")
    t.add_column()
    t.add_column(style="dim")
    t.add_column()

    attn = f"{model.n_head} heads"
    kv_heads = sorted(set(model.n_head_kv))
    if kv_heads and kv_heads != [model.n_head]:
        attn += f" / {'-'.join(str(h) for h in kv_heads)} KV heads (GQA)"
    if model.mla:
        attn += " · MLA"
    extras = []
    if model.is_moe:
        extras.append(f"MoE {model.n_expert} experts ({model.n_expert_used} active)")
    if model.swa_window:
        pat = "unknown pattern" if model.swa_unknown else (
            "per-layer map" if model.swa_pattern == -1 else f"1 full : {model.swa_pattern - 1} SWA"
        )
        extras.append(f"SWA window {model.swa_window:,} ({pat})")

    t.add_row("arch", model.arch, "params", fmt_params(model.n_params))
    t.add_row("quant", model.quant or "?", "file size", fmt_bytes(model.file_size)
              + (f" ({model.n_shards} shards)" if model.n_shards > 1 else ""))
    t.add_row("layers", str(model.n_layer), "attention", attn)
    t.add_row("embd / ffn", f"{model.n_embd} / {model.n_ff}", "train ctx", f"{model.n_ctx_train:,}")
    if extras:
        t.add_row("features", " · ".join(extras), "", "")
    return Panel(t, title=f"[bold]{model.name}[/bold]", border_style="dim", expand=False)


def budget_section(
    gpus: list[GPU],
    budget: int | None,
    fit_target_mib: int,
    vram_override: float | None,
) -> Group:
    """How the VRAM budget is derived, with the total / used / free VRAM it comes from.

    Sits near the top so the numbers every later bar is measured against are
    visible up front. Multiple GPUs are pooled (sum of total / free).
    """
    lines: list[Text | str] = []
    header = Text()
    header.append("VRAM budget  ", style="bold")

    if gpus:
        total = sum(g.total for g in gpus)
        free = sum(g.free for g in gpus)
        used = total - free
        # budget = free VRAM − fit-target reserve
        header.append(fmt_bytes(free), style=f"bold {C_KV}")
        header.append(" free − ")
        header.append(f"{fit_target_mib} MiB", style=C_OVERHEAD)
        header.append(" fit-target = ")
        header.append(fmt_bytes(budget if budget is not None else free), style="bold")
        header.append(" budget")
        lines.append(header)

        vram = Text()
        vram.append("  VRAM  ", style="dim")
        vram.append("total ", style="dim")
        vram.append(fmt_bytes(total), style="bold")
        vram.append("   used ", style="dim")
        vram.append(fmt_bytes(used), style=f"bold {C_OVER}")
        vram.append("   free ", style="dim")
        vram.append(fmt_bytes(free), style=f"bold {C_KV}")
        if len(gpus) > 1:
            vram.append(f"   ·  {len(gpus)} GPUs pooled", style="dim")
        lines.append(vram)

        if len(gpus) > 1:
            for g in gpus:
                row = Text("    ")
                row.append(f"{g.name}", style="default")
                row.append(
                    f"  total {fmt_bytes(g.total)}, used {fmt_bytes(g.total - g.free)}, "
                    f"free {fmt_bytes(g.free)}",
                    style="dim",
                )
                lines.append(row)
            lines.append(Text("  actual split depends on --tensor-split", style="dim"))
    elif vram_override is not None:
        header.append(f"{vram_override:g} GiB", style="bold")
        header.append(" --vram − ")
        header.append(f"{fit_target_mib} MiB", style=C_OVERHEAD)
        header.append(" fit-target = ")
        header.append(fmt_bytes(budget if budget is not None else 0), style="bold")
        header.append(" budget")
        lines.append(header)
        lines.append(Text("  no GPU probed — using the --vram value as total VRAM", style="dim"))
    else:
        header.append("no GPU detected", style=f"bold {C_OVER}")
        header.append(" — pass --vram <GiB> to enable fit simulation", style="dim")
        lines.append(header)

    return Group(*lines)


def requirement_section(model: ModelInfo, cfg: RunConfig, budget: int | None) -> Group:
    """Raw demand of the model itself: weights + KV cache, before any offload."""
    weights = model.weight_bytes_total
    kv = total_kv_bytes(model, cfg)
    total = weights + kv
    segments = [
        ("weights", weights, C_WEIGHTS),
        (f"KV {cfg.cache_type_k}", kv, C_KV),
    ]
    header = Text()
    header.append("Model memory needed  ", style="bold")
    # Lead with the context premise: KV size below is meaningless without it.
    header.append("@ ctx ")
    header.append(f"{cfg.n_ctx:,}", style=f"bold {C_KV}")
    header.append("   ")
    header.append(
        f"weights {fmt_bytes(weights)} + KV {cfg.cache_type_k} {fmt_bytes(kv)} "
        f"= {fmt_bytes(total)}"
    )
    lines: list[Text | str] = [header]
    lines.append(_bar(segments, budget))
    if budget is not None and total > budget:
        marker_pos = round(budget / total * BAR_WIDTH)
        lines.append(Text(" " * max(0, marker_pos - 1) + f"▲ budget {fmt_bytes(budget)}",
                          style=C_OVER))
    legend = _legend(segments)
    if budget is not None and total <= budget:
        legend.append("   ")
        legend.append("░ ", style=C_FREE)
        legend.append("free ", style="bold")
        legend.append(fmt_bytes(budget - total))
    lines.append(legend)
    return Group(*lines)


def vram_section(
    est: MemEstimate, budget: int, cfg: RunConfig, budget_note: str, offloaded: bool
) -> Group:
    segments = [
        ("weights", est.gpu_weights, C_WEIGHTS),
        (f"KV {cfg.cache_type_k}", est.gpu_kv, C_KV),
        ("compute*", est.gpu_compute, C_COMPUTE),
        ("overhead", est.gpu_overhead, C_OVERHEAD),
    ]
    lines: list[Text | str] = []
    header = Text()
    title = "GPU usage after CPU offload  " if offloaded else "GPU usage (nothing offloaded)  "
    header.append(title, style="bold")
    header.append(f"{fmt_bytes(est.gpu_total)} used / {fmt_bytes(budget)} budget", style="default")
    if est.gpu_total <= budget:
        header.append(f"   ✓ fits, {fmt_bytes(budget - est.gpu_total)} headroom", style="green")
    else:
        header.append(f"   ✗ over by {fmt_bytes(est.gpu_total - budget)}", style=f"bold {C_OVER}")
    lines.append(header)
    lines.append(_bar(segments, budget))
    if est.gpu_total > budget:
        marker_pos = round(budget / est.gpu_total * BAR_WIDTH)
        lines.append(Text(" " * max(0, marker_pos - 1) + "▲ budget", style=C_OVER))
    legend = _legend([s for s in segments if s[1] > 0])
    if est.gpu_total <= budget:
        legend.append("   ")
        legend.append("░ ", style=C_FREE)
        legend.append("free ", style="bold")
        legend.append(fmt_bytes(budget - est.gpu_total))
    lines.append(legend)
    note = Text(budget_note, style="dim")
    lines.append(note)
    return Group(*lines)


def ram_section(est: MemEstimate, ram: tuple[int, int] | None) -> Group:
    """Host RAM taken by the parts that stay on the CPU after offload."""
    segments = [
        ("weights", est.cpu_weights, C_WEIGHTS),
        ("KV", est.cpu_kv, C_KV),
        ("buffers", est.cpu_compute, C_COMPUTE),
    ]
    avail = ram[1] if ram else None
    header = Text()
    header.append("CPU RAM needed  ", style="bold")
    header.append(fmt_bytes(est.cpu_total))
    if ram is not None:
        total, avail_b = ram
        header.append(f" / {fmt_bytes(avail_b)} available ({fmt_bytes(total)} total)")
        if est.cpu_total <= avail_b:
            header.append(f"   ✓ fits, {fmt_bytes(avail_b - est.cpu_total)} headroom",
                          style="green")
        else:
            header.append(
                f"   ✗ over by {fmt_bytes(est.cpu_total - avail_b)} — expect swapping",
                style=f"bold {C_OVER}",
            )
    lines: list[Text | str] = [header]
    lines.append(_bar(segments, avail))
    if avail is not None and est.cpu_total > avail:
        marker_pos = round(avail / est.cpu_total * BAR_WIDTH)
        lines.append(Text(" " * max(0, marker_pos - 1) + f"▲ available {fmt_bytes(avail)}",
                          style=C_OVER))
    legend = _legend([s for s in segments if s[1] > 0])
    if avail is not None and est.cpu_total <= avail:
        legend.append("   ")
        legend.append("░ ", style=C_FREE)
        legend.append("free ", style="bold")
        legend.append(fmt_bytes(avail - est.cpu_total))
    lines.append(legend)
    return Group(*lines)


def layer_strip(model: ModelInfo, cfg: RunConfig) -> Group:
    n_cells = model.n_layer + 1  # +1 = output layer
    width = min(n_cells, BAR_WIDTH)
    text = Text()
    ngl_blocks = min(cfg.n_gpu_layers, model.n_layer)
    first_gpu = model.n_layer - ngl_blocks
    for cell in range(width):
        # map cell -> layer index range
        i = round(cell * n_cells / width)
        if i >= model.n_layer:  # output layer
            on_gpu = cfg.n_gpu_layers > model.n_layer
            text.append("█" if on_gpu else "░", style=C_WEIGHTS if on_gpu else C_CPU)
        elif i >= first_gpu:
            if model.is_moe and i < cfg.n_cpu_moe:
                text.append("▓", style=C_COMPUTE)
            else:
                text.append("█", style=C_WEIGHTS)
        else:
            text.append("░", style=C_CPU)
    label = Text()
    label.append("layers  ", style="bold")
    label.append(f"-ngl {cfg.n_gpu_layers}: {min(cfg.n_gpu_layers, model.n_layer)}/{model.n_layer} blocks on GPU")
    label.append(" +output" if cfg.n_gpu_layers > model.n_layer else " (output on CPU)",
                 style="default" if cfg.n_gpu_layers > model.n_layer else "dim")
    legend = Text()
    legend.append("█ GPU", style=C_WEIGHTS)
    if model.is_moe and cfg.n_cpu_moe:
        legend.append("   ")
        legend.append("▓ dense on GPU, experts on CPU", style=C_COMPUTE)
    legend.append("   ")
    legend.append("░ CPU", style=C_CPU)
    return Group(label, text, legend)


def kv_table(
    scenarios: list[KVScenario], target_ctx: int, budget_known: bool, model: ModelInfo
) -> Table:
    t = Table(
        title=f"KV cache options @ ctx {target_ctx:,}",
        title_justify="left",
        border_style="dim",
        header_style="bold",
    )
    t.add_column("-ctk/-ctv")
    t.add_column("KV size", justify="right")
    t.add_column("GPU total", justify="right")
    if budget_known:
        t.add_column("fits (full offload)", justify="center")
        t.add_column("max ctx (full offload)", justify="right")
        t.add_column("max -ngl @ ctx", justify="right")
    t.add_column("quality")
    from .fit import KV_QUALITY_NOTES

    for s in scenarios:
        row = [
            s.ctk if s.ctk == s.ctv else f"{s.ctk}/{s.ctv}",
            fmt_bytes(s.kv_bytes_at_target),
            fmt_bytes(s.est_at_target.gpu_total),
        ]
        if budget_known:
            row.append(Text("✓ yes", style="green") if s.fits_full_at_target
                       else Text("✗ no", style=C_OVER))
            row.append(f"{s.max_ctx_full_offload:,}" if s.max_ctx_full_offload else "—")
            row.append(f"{s.max_ngl_at_target}/{model.n_layer + 1}")
        row.append(Text(KV_QUALITY_NOTES.get(s.ctk, ""), style="dim"))
        t.add_row(*row)
    return t


def fit_panel(fit: FitResult) -> Panel:
    lines: list[Text | str] = []
    if fit.actions:
        for a in fit.actions:
            lines.append(Text(f"· {a}", style="yellow"))
    else:
        lines.append(Text("· nothing to do — fits as requested", style="green"))
    cfg = fit.cfg
    t = Table.grid(padding=(0, 2))
    t.add_column(style="dim")
    t.add_column(style="bold")
    t.add_row("-c (context)", f"{cfg.n_ctx:,}")
    t.add_row("-ngl (GPU layers)", str(cfg.n_gpu_layers))
    if cfg.n_cpu_moe:
        t.add_row("--n-cpu-moe", str(cfg.n_cpu_moe))
    t.add_row("KV cache", f"{cfg.cache_type_k}/{cfg.cache_type_v}")
    t.add_row("GPU total", fmt_bytes(fit.est.gpu_total)
              + (f"  (headroom {fmt_bytes(fit.headroom)})" if fit.fits else ""))
    lines.append(t)
    return Panel(
        Group(*lines),
        title="[bold]llama.cpp --fit simulation[/bold] (what auto-fit would pick)",
        border_style="dim",
        expand=False,
    )


def recommendation_panels(
    recs: list[Recommendation], model_path: str, budget: int
) -> list[Panel]:
    panels = []
    for i, rec in enumerate(recs):
        body: list[Text | str] = []
        body.append(Text(rec.reason, style="dim"))
        est_line = Text()
        est_line.append("GPU ", style="bold")
        est_line.append(fmt_bytes(rec.est.gpu_total))
        est_line.append(f" / {fmt_bytes(budget)}")
        if rec.est.cpu_total > 512 * MIB:
            est_line.append("    CPU RAM ", style="bold")
            est_line.append(fmt_bytes(rec.est.cpu_total))
        body.append(est_line)
        cmd = command_line(model_path, rec.cfg)
        body.append(Text(f"$ {cmd}", style="bold cyan"))
        title = f"★ {rec.title}" if i == 0 else rec.title
        panels.append(
            Panel(
                Group(*body),
                title=f"[bold]{title}[/bold]",
                border_style="green" if i == 0 else "dim",
                expand=False,
            )
        )
    return panels
