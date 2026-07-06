"""Extract inference-relevant model information from GGUF metadata.

Handles the hyperparameters that drive memory use in llama.cpp:

- GQA (grouped-query attention): ``head_count_kv`` may be a scalar or a
  per-layer array (some layers may even have 0 KV heads — recurrent blocks).
- SWA (sliding-window attention): gemma2/3, gpt-oss, cohere2, … only keep a
  window of KV per SWA layer; the full/SWA layer pattern is per-architecture.
- MoE: expert tensors (``*_exps``) can be kept on CPU with ``--n-cpu-moe``.
- MLA (deepseek2 etc.): KV cache stores the compressed latent, not full K/V.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .gguf import GGUFFile

# LLAMA_FTYPE → display name (llama.cpp include/llama.h)
FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S",
    15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS",
    20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
    25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M",
    30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0",
    38: "MXFP4",
}

# Which layers use sliding-window attention, per architecture.
# value = (pattern, full_layer_predicate) where predicate(i, pattern) is True
# for layers that keep the FULL context KV. Derived from llama.cpp
# src/llama-model.cpp (hparams.set_swa_pattern calls).
_SWA_PATTERNS: dict[str, int] = {
    "gemma2": 2,     # alternating, every 2nd layer full
    "gemma3": 6,     # 5 SWA : 1 full
    "gemma3n": 6,
    "gpt-oss": 2,
    "cohere2": 4,
    "llama4": 4,     # 3 chunked : 1 full
    "exaone4": 4,
}

_EXPERT_TENSOR_RE = re.compile(r"\bffn_(?:gate|down|up)_exps\b")
_BLK_RE = re.compile(r"^blk\.(\d+)\.")


@dataclass
class LayerWeights:
    dense: int = 0   # bytes of always-active tensors in this block
    expert: int = 0  # bytes of conditional expert tensors (``*_exps``)

    @property
    def total(self) -> int:
        return self.dense + self.expert


@dataclass
class ModelInfo:
    name: str
    arch: str
    quant: str
    n_params: int
    file_size: int
    n_shards: int

    n_layer: int
    n_embd: int
    n_head: int
    n_head_kv: list[int]          # per layer
    n_embd_head_k: int
    n_embd_head_v: int
    n_ctx_train: int
    n_vocab: int
    n_ff: int

    n_expert: int = 0
    n_expert_used: int = 0

    swa_window: int = 0           # 0 = no sliding-window attention
    swa_pattern: int = 0
    is_swa_layer: list[bool] = field(default_factory=list)
    swa_unknown: bool = False     # SWA advertised but pattern unknown → we
                                  # over-estimate by treating layers as full

    mla: bool = False             # deepseek2-style latent KV cache
    mla_kv_dim: int = 0           # per-token per-layer cache width (elements)

    # weights, classified for offload simulation
    layer_weights: list[LayerWeights] = field(default_factory=list)
    embd_bytes: int = 0           # token_embd — always on CPU in llama.cpp
    output_bytes: int = 0         # output head + final norm — offloaded when ngl > n_layer

    @property
    def is_moe(self) -> bool:
        return self.n_expert > 1

    @property
    def weight_bytes_total(self) -> int:
        return self.embd_bytes + self.output_bytes + sum(w.total for w in self.layer_weights)

    def kv_row_bytes_f16(self, layer: int) -> int:
        """Bytes of one token's K+V at f16 for a given layer (for quick view)."""
        k, v = self.kv_dims(layer)
        return (k + v) * 2

    def kv_dims(self, layer: int) -> tuple[int, int]:
        """(K elements, V elements) stored per token for a layer."""
        if self.mla:
            return self.mla_kv_dim, 0
        heads = self.n_head_kv[layer] if layer < len(self.n_head_kv) else 0
        return self.n_embd_head_k * heads, self.n_embd_head_v * heads


def _as_list(value, n: int, default: int) -> list[int]:
    if value is None:
        return [default] * n
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value]
        if len(vals) < n:
            vals += [default] * (n - len(vals))
        return vals[:n]
    return [int(value)] * n


def _max_of(value, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        return max((int(v) for v in value), default=default)
    return int(value)


def extract_model_info(g: GGUFFile) -> ModelInfo:
    md = g.metadata
    arch = md.get("general.architecture", "unknown")

    def key(suffix: str):
        return md.get(f"{arch}.{suffix}")

    n_layer = int(key("block_count") or 0)
    n_embd = int(key("embedding_length") or 0)
    n_head = _max_of(key("attention.head_count"), 1)
    n_head_kv = _as_list(key("attention.head_count_kv"), n_layer, n_head)
    head_dim_default = n_embd // n_head if n_head else 0
    n_embd_head_k = int(key("attention.key_length") or head_dim_default)
    n_embd_head_v = int(key("attention.value_length") or head_dim_default)
    n_ctx_train = int(key("context_length") or 0)
    n_ff = _max_of(key("feed_forward_length"), 0)

    n_vocab = key("vocab_size")
    if n_vocab is None:
        n_vocab = g.array_len("tokenizer.ggml.tokens") or 0
    n_vocab = int(n_vocab)

    n_expert = int(key("expert_count") or 0)
    n_expert_used = int(key("expert_used_count") or 0)

    # --- MLA (deepseek2 family): the cache stores kv_lora_rank + rope dims ---
    mla = False
    mla_kv_dim = 0
    kv_lora_rank = key("attention.kv_lora_rank")
    if kv_lora_rank:
        mla = True
        rope_dim = int(key("rope.dimension_count") or 64)
        mla_kv_dim = int(kv_lora_rank) + rope_dim

    # --- SWA pattern ---
    swa_window = int(key("attention.sliding_window") or 0)
    swa_pattern = 0
    is_swa_layer = [False] * n_layer
    swa_unknown = False
    if swa_window and swa_window < n_ctx_train:
        pattern = key("attention.sliding_window_pattern") or _SWA_PATTERNS.get(arch, 0)
        if isinstance(pattern, (list, tuple)):
            # some conversions store a per-layer bool array (True = SWA layer)
            is_swa_layer = [bool(v) for v in pattern][:n_layer]
            is_swa_layer += [False] * (n_layer - len(is_swa_layer))
            swa_pattern = -1
        elif pattern and int(pattern) > 1:
            swa_pattern = int(pattern)
            # llama.cpp set_swa_pattern: layer i is full iff (i+1) % pattern == 0
            is_swa_layer = [(i + 1) % swa_pattern != 0 for i in range(n_layer)]
        else:
            swa_unknown = True  # keep all layers "full" → safe over-estimate
    else:
        swa_window = 0

    # --- classify weights for offload simulation ---
    layer_weights = [LayerWeights() for _ in range(n_layer)]
    embd_bytes = 0
    output_bytes = 0
    n_params = 0
    type_bytes: dict[int, int] = {}
    for t in g.tensors:
        n_params += t.n_elements
        m = _BLK_RE.match(t.name)
        if m:
            i = int(m.group(1))
            if i < n_layer:
                if _EXPERT_TENSOR_RE.search(t.name):
                    layer_weights[i].expert += t.nbytes
                else:
                    layer_weights[i].dense += t.nbytes
                type_bytes[t.ggml_type] = type_bytes.get(t.ggml_type, 0) + t.nbytes
            else:
                output_bytes += t.nbytes
        elif t.name.startswith("token_embd"):
            embd_bytes += t.nbytes
        else:
            output_bytes += t.nbytes

    quant = FILE_TYPE_NAMES.get(md.get("general.file_type", -1), "")
    if not quant and type_bytes:
        # fall back to the dominant block-tensor type
        from .memory import GGML_TYPE_NAMES

        dominant = max(type_bytes, key=type_bytes.get)  # type: ignore[arg-type]
        quant = GGML_TYPE_NAMES.get(dominant, f"type{dominant}")

    name = md.get("general.name") or ""
    size_label = md.get("general.size_label") or ""
    if size_label and size_label not in name:
        name = f"{name} {size_label}".strip()

    return ModelInfo(
        name=name or arch,
        arch=arch,
        quant=quant,
        n_params=n_params,
        file_size=g.file_size,
        n_shards=g.n_shards,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_head_kv=n_head_kv,
        n_embd_head_k=n_embd_head_k,
        n_embd_head_v=n_embd_head_v,
        n_ctx_train=n_ctx_train,
        n_vocab=n_vocab,
        n_ff=n_ff,
        n_expert=n_expert,
        n_expert_used=n_expert_used,
        swa_window=swa_window,
        swa_pattern=swa_pattern,
        is_swa_layer=is_swa_layer,
        swa_unknown=swa_unknown,
        mla=mla,
        mla_kv_dim=mla_kv_dim,
        layer_weights=layer_weights,
        embd_bytes=embd_bytes,
        output_bytes=output_bytes,
    )
