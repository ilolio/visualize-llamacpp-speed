from llamafit.fit import (
    build_recommendations,
    command_line,
    evaluate_kv_scenarios,
    simulate_fit,
)
from llamafit.gguf import load_gguf
from llamafit.memory import GIB, MIB, RunConfig, estimate
from llamafit.model import extract_model_info

from conftest import llama_metadata, llama_tensors, write_gguf


def _model(tmp_path, md=None, tensors=None):
    p = tmp_path / "m.gguf"
    write_gguf(p, md or llama_metadata(), tensors or llama_tensors())
    return extract_model_info(load_gguf(str(p)))


def _big_model(tmp_path):
    """~1 GiB of weights, hefty KV: 16 layers x 1024 embd."""
    md = llama_metadata(n_layer=16, n_embd=1024, n_head=16, n_head_kv=16,
                        n_ff=8192, n_ctx_train=131072, n_vocab=4000)
    return _model(tmp_path, md, llama_tensors(n_layer=16, n_embd=1024, n_ff=8192, n_vocab=4000))


def test_fit_no_action_needed(tmp_path):
    m = _model(tmp_path)
    base = RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1)
    fit = simulate_fit(m, budget=8 * GIB, base=base, overhead=0)
    assert fit.fits
    assert fit.actions == []
    assert fit.cfg.n_gpu_layers == m.n_layer + 1
    assert fit.cfg.n_ctx == 4096


def test_fit_shrinks_context_first(tmp_path):
    m = _big_model(tmp_path)
    base = RunConfig(n_ctx=131072, n_gpu_layers=m.n_layer + 1)
    full_est = estimate(m, base)
    budget = full_est.gpu_total - 200 * MIB  # just under full-context needs
    fit = simulate_fit(m, budget, base, overhead=0, fit_ctx=4096)
    assert fit.fits
    assert fit.cfg.n_gpu_layers == m.n_layer + 1  # layers untouched
    assert 4096 <= fit.cfg.n_ctx < 131072
    assert fit.cfg.n_ctx % 256 == 0
    assert any("context reduced" in a for a in fit.actions)
    # maximality: one step more context must not fit
    bigger = RunConfig(**{**fit.cfg.__dict__, "n_ctx": fit.cfg.n_ctx + 256})
    assert estimate(m, bigger).gpu_total > budget


def test_fit_pinned_ctx_drops_layers_instead(tmp_path):
    m = _big_model(tmp_path)
    base = RunConfig(n_ctx=131072, n_gpu_layers=m.n_layer + 1)
    budget = estimate(m, base).gpu_total - 200 * MIB
    fit = simulate_fit(m, budget, base, overhead=0, fit_ctx=4096, ctx_pinned=True)
    assert fit.fits
    assert fit.cfg.n_ctx == 131072
    assert fit.cfg.n_gpu_layers < m.n_layer + 1


def test_fit_moe_spills_experts_before_layers(tmp_path):
    md = llama_metadata(n_layer=8, n_embd=512, n_ff=2048, n_ctx_train=32768)
    md["llama.expert_count"] = 8
    md["llama.expert_used_count"] = 2
    tensors = llama_tensors(n_layer=8, n_embd=512, n_ff=2048)
    for i in range(8):
        tensors += [
            (f"blk.{i}.ffn_gate_exps.weight", (512, 2048, 8)),
            (f"blk.{i}.ffn_down_exps.weight", (2048, 512, 8)),
            (f"blk.{i}.ffn_up_exps.weight", (512, 2048, 8)),
        ]
    m = _model(tmp_path, md, tensors)
    full = RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1)
    need = estimate(m, full).gpu_total
    expert_total = sum(w.expert for w in m.layer_weights)
    budget = need - expert_total // 2  # forces roughly half the experts off GPU
    fit = simulate_fit(m, budget, full, overhead=0, fit_ctx=4096, ctx_pinned=True)
    assert fit.fits
    assert fit.cfg.n_gpu_layers == m.n_layer + 1  # all layers stay on GPU
    assert 0 < fit.cfg.n_cpu_moe <= m.n_layer
    # minimality: one fewer spilled layer must not fit
    cfg = RunConfig(**{**fit.cfg.__dict__, "n_cpu_moe": fit.cfg.n_cpu_moe - 1})
    assert estimate(m, cfg).gpu_total > budget


def test_fit_impossible_budget(tmp_path):
    m = _model(tmp_path)
    base = RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1)
    fit = simulate_fit(m, budget=-1, base=base, overhead=10 * MIB, ctx_pinned=True)
    assert not fit.fits
    assert fit.cfg.n_gpu_layers == 0


def test_kv_scenarios_ordering(tmp_path):
    m = _big_model(tmp_path)
    base = RunConfig(n_ctx=32768, n_gpu_layers=m.n_layer + 1)
    scen = evaluate_kv_scenarios(m, budget=4 * GIB, base=base, overhead=0, max_ctx_cap=131072)
    assert [s.ctk for s in scen] == ["f16", "q8_0", "q4_0"]
    # heavier quantization → smaller KV, never larger max ctx budget-wise
    assert scen[0].kv_bytes_at_target > scen[1].kv_bytes_at_target > scen[2].kv_bytes_at_target
    assert scen[0].max_ctx_full_offload <= scen[1].max_ctx_full_offload <= scen[2].max_ctx_full_offload


def test_recommendations_easy_case(tmp_path):
    m = _model(tmp_path)
    base = RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1)
    scen = evaluate_kv_scenarios(m, budget=8 * GIB, base=base, overhead=0, max_ctx_cap=8192)
    recs = build_recommendations(m, 8 * GIB, base, 0, scen)
    assert recs
    assert "f16" in recs[0].title
    assert recs[0].cfg.cache_type_k == "f16"


def test_recommendations_suggest_q8_when_f16_does_not_fit(tmp_path):
    m = _big_model(tmp_path)
    base = RunConfig(n_ctx=65536, n_gpu_layers=m.n_layer + 1)
    f16_need = estimate(m, base).gpu_total
    cfg8 = RunConfig(n_ctx=65536, n_gpu_layers=m.n_layer + 1,
                     cache_type_k="q8_0", cache_type_v="q8_0")
    q8_need = estimate(m, cfg8).gpu_total
    budget = (f16_need + q8_need) // 2  # between the two
    scen = evaluate_kv_scenarios(m, budget, base, 0, max_ctx_cap=131072)
    recs = build_recommendations(m, budget, base, 0, scen)
    assert "q8_0" in recs[0].title
    assert recs[0].cfg.n_ctx == 65536  # target context preserved


def test_command_line():
    cfg = RunConfig(n_ctx=16384, n_gpu_layers=33, cache_type_k="q8_0",
                    cache_type_v="q8_0", n_cpu_moe=5)
    cmd = command_line("model.gguf", cfg)
    assert cmd.startswith("llama-server -m model.gguf -c 16384 -ngl 33")
    assert "-ctk q8_0 -ctv q8_0" in cmd
    assert "--n-cpu-moe 5" in cmd
    plain = command_line("m.gguf", RunConfig(n_ctx=4096, n_gpu_layers=10))
    assert "-ctk" not in plain
