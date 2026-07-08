"""Memory math: weights placement, KV cache size, compute-buffer estimate.

All sizes are bytes.  The weight numbers are exact (taken from the GGUF
tensor table); KV cache is exact given the hyperparameters; the compute
buffer is an estimate (llama.cpp sizes it from the actual graph) — expect
roughly ±10-15% there, which is why a safety margin (``--fit-target``) is
kept, exactly like llama.cpp's own ``--fit`` does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import ModelInfo

MIB = 1024 * 1024
GIB = 1024 * MIB

# ggml quantization: type name -> (block size in elements, bytes per block).
# Only types accepted by llama.cpp --cache-type-k/--cache-type-v.
KV_CACHE_TYPES: dict[str, tuple[int, int]] = {
    "f32": (1, 4),
    "f16": (1, 2),
    "bf16": (1, 2),
    "q8_0": (32, 34),
    "q5_1": (32, 24),
    "q5_0": (32, 22),
    "q4_1": (32, 20),
    "q4_0": (32, 18),
    "iq4_nl": (32, 18),
}

# ggml_type id -> name, for labeling quantization when general.file_type is
# missing (ids from ggml/include/ggml.h).
GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
    9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
    20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16",
    26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16", 34: "TQ1_0",
    35: "TQ2_0", 39: "MXFP4",
}


def kv_bytes_per_element(cache_type: str) -> float:
    try:
        block, size = KV_CACHE_TYPES[cache_type]
    except KeyError:
        raise ValueError(
            f"unknown KV cache type {cache_type!r} (choose from {', '.join(KV_CACHE_TYPES)})"
        ) from None
    return size / block


@dataclass
class RunConfig:
    """Parameters that mirror llama.cpp flags."""

    n_ctx: int
    n_gpu_layers: int          # 0..n_layer+1 (n_layer+1 = output layer too)
    n_cpu_moe: int = 0         # --n-cpu-moe: expert tensors of first N layers on CPU
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    flash_attn: bool = True
    n_ubatch: int = 512
    n_batch: int = 2048
    n_seq: int = 1


@dataclass
class MemEstimate:
    gpu_weights: int = 0
    gpu_mmproj: int = 0        # multimodal projector weights (offloaded to GPU)
    gpu_kv: int = 0
    gpu_compute: int = 0
    gpu_overhead: int = 0      # runtime/context allocations (CUDA context etc.)
    cpu_weights: int = 0
    cpu_kv: int = 0
    cpu_compute: int = 0
    kv_per_layer_full: int = 0  # bytes of one full-context layer, for reference
    notes: list[str] = field(default_factory=list)

    @property
    def gpu_total(self) -> int:
        return self.gpu_weights + self.gpu_mmproj + self.gpu_kv + self.gpu_compute + self.gpu_overhead

    @property
    def cpu_total(self) -> int:
        return self.cpu_weights + self.cpu_kv + self.cpu_compute


def layer_kv_bytes(
    model: ModelInfo, layer: int, n_ctx: int, ctk: str, ctv: str, n_ubatch: int
) -> int:
    """KV cache bytes of one layer.

    SWA layers only keep ``sliding_window + n_ubatch`` cells (llama.cpp pads
    the window by a micro-batch).  MLA models store the compressed latent in
    the K cache and nothing in V.
    """
    k_dim, v_dim = model.kv_dims(layer)
    tokens = n_ctx
    if model.swa_window and layer < len(model.is_swa_layer) and model.is_swa_layer[layer]:
        tokens = min(n_ctx, model.swa_window + n_ubatch)
    return round(tokens * (k_dim * kv_bytes_per_element(ctk) + v_dim * kv_bytes_per_element(ctv)))


def total_kv_bytes(model: ModelInfo, cfg: RunConfig, layers: range | None = None) -> int:
    layers = layers if layers is not None else range(model.n_layer)
    return sum(
        layer_kv_bytes(model, i, cfg.n_ctx, cfg.cache_type_k, cfg.cache_type_v, cfg.n_ubatch)
        for i in layers
    )


def compute_buffer_bytes(model: ModelInfo, cfg: RunConfig, output_on_gpu: bool) -> int:
    """Estimate of the GPU compute (graph) buffer.

    Dominant contributors:
    - output logits: ``n_vocab × n_ubatch × f32`` when the output layer is
      offloaded (this is usually the largest single tensor);
    - without flash attention: the materialized ``KQ`` matrix
      ``n_kv × n_ubatch × n_head × f32`` — this is what makes long contexts
      explode without ``-fa``;
    - FFN / attention activations: ``n_ubatch × max(n_ff, 4·n_embd) × f32``
      for a couple of live tensors.
    """
    ub = cfg.n_ubatch
    logits = model.n_vocab * ub * 4 if output_on_gpu else 0
    if cfg.flash_attn:
        attn = ub * model.n_embd_head_k * model.n_head * 4 * 2
    else:
        attn = cfg.n_ctx * ub * model.n_head * 4
    ffn = ub * max(model.n_ff, 4 * model.n_embd) * 4 * 2
    misc = ub * model.n_embd * 4 * 8
    return logits + max(attn, ffn, misc)


def offloaded_layers(model: ModelInfo, n_gpu_layers: int) -> range:
    """llama.cpp offloads the *last* ngl blocks."""
    ngl_blocks = min(n_gpu_layers, model.n_layer)
    return range(model.n_layer - ngl_blocks, model.n_layer)


def estimate(model: ModelInfo, cfg: RunConfig, overhead_bytes: int = 0) -> MemEstimate:
    """Full memory placement estimate for a given configuration."""
    est = MemEstimate(gpu_overhead=overhead_bytes)

    gpu_layers = offloaded_layers(model, cfg.n_gpu_layers)
    output_on_gpu = cfg.n_gpu_layers > model.n_layer

    # weights
    est.cpu_weights += model.embd_bytes
    for i, lw in enumerate(model.layer_weights):
        on_gpu = i in gpu_layers
        expert_on_cpu = model.is_moe and i < cfg.n_cpu_moe
        if on_gpu:
            est.gpu_weights += lw.dense
            if expert_on_cpu:
                est.cpu_weights += lw.expert
            else:
                est.gpu_weights += lw.expert
        else:
            est.cpu_weights += lw.total
    if output_on_gpu:
        est.gpu_weights += model.output_bytes
    else:
        est.cpu_weights += model.output_bytes

    # multimodal projector: a fixed allocation, independent of -ngl / ctx / KV.
    # llama.cpp offloads the vision/audio encoder to GPU by default.
    if model.mmproj_bytes:
        if model.mmproj_on_gpu:
            est.gpu_mmproj += model.mmproj_bytes
        else:
            est.cpu_weights += model.mmproj_bytes

    # KV cache follows its layer's device
    for i in range(model.n_layer):
        b = layer_kv_bytes(model, i, cfg.n_ctx, cfg.cache_type_k, cfg.cache_type_v, cfg.n_ubatch)
        if i in gpu_layers:
            est.gpu_kv += b
        else:
            est.cpu_kv += b

    est.kv_per_layer_full = round(
        cfg.n_ctx
        * (
            model.kv_dims(model.n_layer - 1)[0] * kv_bytes_per_element(cfg.cache_type_k)
            + model.kv_dims(model.n_layer - 1)[1] * kv_bytes_per_element(cfg.cache_type_v)
        )
    )

    # compute buffers
    if cfg.n_gpu_layers > 0:
        est.gpu_compute = compute_buffer_bytes(model, cfg, output_on_gpu)
    host_logits = 0 if output_on_gpu else model.n_vocab * cfg.n_ubatch * 4
    est.cpu_compute = host_logits + cfg.n_batch * model.n_embd * 4 * 2

    if model.swa_unknown:
        est.notes.append(
            "sliding-window pattern unknown for this architecture — KV sized as full "
            "context on every layer (over-estimate)"
        )
    if model.mla:
        est.notes.append("MLA latent KV cache (deepseek2-style): V cache is folded into K")
    if model.mmproj_bytes:
        where = "GPU" if model.mmproj_on_gpu else "CPU"
        est.notes.append(
            f"multimodal projector weights included on {where}; the vision/audio encoder "
            "also uses a compute buffer (image-size dependent) not counted here"
        )
    return est
