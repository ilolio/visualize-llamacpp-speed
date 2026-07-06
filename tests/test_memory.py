from llamafit.gguf import load_gguf
from llamafit.memory import (
    GIB,
    MIB,
    RunConfig,
    estimate,
    kv_bytes_per_element,
    layer_kv_bytes,
    total_kv_bytes,
)
from llamafit.model import extract_model_info

from conftest import llama_metadata, llama_tensors, write_gguf


def _model(tmp_path, md=None, tensors=None):
    p = tmp_path / "m.gguf"
    write_gguf(p, md or llama_metadata(), tensors or llama_tensors())
    return extract_model_info(load_gguf(str(p)))


def test_kv_bytes_per_element():
    assert kv_bytes_per_element("f16") == 2
    assert kv_bytes_per_element("f32") == 4
    assert kv_bytes_per_element("q8_0") == 34 / 32
    assert kv_bytes_per_element("q4_0") == 18 / 32


def test_kv_cache_llama3_8b_shape(tmp_path):
    """Llama-3-8B numbers: 32 layers, 8 KV heads x 128 dim → 16 MiB/layer/4k @ f16."""
    md = llama_metadata(n_layer=32, n_embd=4096, n_head=32, n_head_kv=8, n_ctx_train=131072)
    m = _model(tmp_path, md, llama_tensors(n_layer=32, n_embd=4096))
    cfg = RunConfig(n_ctx=4096, n_gpu_layers=33)
    # per layer: 4096 tokens * (1024 K + 1024 V) * 2 bytes = 16 MiB
    assert layer_kv_bytes(m, 0, 4096, "f16", "f16", 512) == 16 * MIB
    assert total_kv_bytes(m, cfg) == 512 * MIB
    # q8_0 = 34/32 bytes per element → 17/32 of f16
    cfg8 = RunConfig(n_ctx=4096, n_gpu_layers=33, cache_type_k="q8_0", cache_type_v="q8_0")
    assert total_kv_bytes(m, cfg8) == 512 * MIB * 17 // 32


def test_swa_reduces_kv(tmp_path):
    md = llama_metadata(n_layer=6, n_ctx_train=131072)
    md["general.architecture"] = "gemma3"
    md = {k.replace("llama.", "gemma3."): v for k, v in md.items()}
    md["gemma3.attention.sliding_window"] = 1024
    m = _model(tmp_path, md)
    assert m.swa_pattern == 6
    # layers 0..4 are SWA, layer 5 is full ((i+1) % 6 == 0)
    assert m.is_swa_layer == [True] * 5 + [False]
    full = layer_kv_bytes(m, 5, 8192, "f16", "f16", 512)
    swa = layer_kv_bytes(m, 0, 8192, "f16", "f16", 512)
    assert swa < full
    # SWA layer holds window + ubatch tokens
    assert swa == full * (1024 + 512) // 8192


def test_swa_unknown_pattern_is_conservative(tmp_path):
    md = llama_metadata(n_layer=4, n_ctx_train=131072)
    md["general.architecture"] = "mystery"
    md = {k.replace("llama.", "mystery."): v for k, v in md.items()}
    md["mystery.attention.sliding_window"] = 1024
    m = _model(tmp_path, md)
    assert m.swa_unknown
    # treated as full-context KV on all layers (over-estimate, never under)
    assert layer_kv_bytes(m, 0, 8192, "f16", "f16", 512) == layer_kv_bytes(
        m, 3, 8192, "f16", "f16", 512
    )


def test_mla_cache(tmp_path):
    md = llama_metadata(n_layer=4)
    md["general.architecture"] = "deepseek2"
    md = {k.replace("llama.", "deepseek2."): v for k, v in md.items()}
    md["deepseek2.attention.kv_lora_rank"] = 512
    md["deepseek2.rope.dimension_count"] = 64
    m = _model(tmp_path, md)
    assert m.mla
    k, v = m.kv_dims(0)
    assert (k, v) == (576, 0)
    assert layer_kv_bytes(m, 0, 4096, "f16", "f16", 512) == 4096 * 576 * 2


def test_estimate_placement_full_offload(tmp_path):
    m = _model(tmp_path)
    cfg = RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1)
    est = estimate(m, cfg, overhead_bytes=100 * MIB)
    # token_embd always on CPU
    assert est.cpu_weights == m.embd_bytes
    assert est.gpu_weights == m.weight_bytes_total - m.embd_bytes
    assert est.cpu_kv == 0
    assert est.gpu_kv == total_kv_bytes(m, cfg)
    assert est.gpu_overhead == 100 * MIB
    assert est.gpu_total == est.gpu_weights + est.gpu_kv + est.gpu_compute + est.gpu_overhead


def test_estimate_partial_offload(tmp_path):
    m = _model(tmp_path)  # 4 layers
    cfg = RunConfig(n_ctx=4096, n_gpu_layers=2)
    est = estimate(m, cfg)
    # last 2 blocks on GPU, first 2 + output + embd on CPU
    expected_gpu_w = sum(w.total for w in m.layer_weights[2:])
    assert est.gpu_weights == expected_gpu_w
    assert est.cpu_weights == m.embd_bytes + m.output_bytes + sum(
        w.total for w in m.layer_weights[:2]
    )
    # KV split follows layers
    assert est.gpu_kv == total_kv_bytes(m, cfg, range(2, 4))
    assert est.cpu_kv == total_kv_bytes(m, cfg, range(0, 2))


def test_estimate_ngl_zero_uses_no_vram(tmp_path):
    m = _model(tmp_path)
    est = estimate(m, RunConfig(n_ctx=4096, n_gpu_layers=0))
    assert est.gpu_weights == 0
    assert est.gpu_kv == 0
    assert est.gpu_compute == 0


def test_moe_expert_split(tmp_path):
    md = llama_metadata()
    md["llama.expert_count"] = 8
    md["llama.expert_used_count"] = 2
    tensors = llama_tensors()
    for i in range(4):
        tensors += [
            (f"blk.{i}.ffn_gate_exps.weight", (256, 512, 8)),
            (f"blk.{i}.ffn_down_exps.weight", (512, 256, 8)),
            (f"blk.{i}.ffn_up_exps.weight", (256, 512, 8)),
        ]
    m = _model(tmp_path, md, tensors)
    assert m.is_moe
    assert all(w.expert > 0 for w in m.layer_weights)
    full = m.n_layer + 1
    base = estimate(m, RunConfig(n_ctx=4096, n_gpu_layers=full))
    spill = estimate(m, RunConfig(n_ctx=4096, n_gpu_layers=full, n_cpu_moe=2))
    moved = sum(w.expert for w in m.layer_weights[:2])
    assert base.gpu_weights - spill.gpu_weights == moved
    assert spill.cpu_weights - base.cpu_weights == moved
    # KV stays fully on GPU when only experts spill
    assert spill.gpu_kv == base.gpu_kv


def test_flash_attention_shrinks_compute(tmp_path):
    m = _model(tmp_path)
    fa_on = estimate(m, RunConfig(n_ctx=32768, n_gpu_layers=5, flash_attn=True))
    fa_off = estimate(m, RunConfig(n_ctx=32768, n_gpu_layers=5, flash_attn=False))
    assert fa_on.gpu_compute < fa_off.gpu_compute
