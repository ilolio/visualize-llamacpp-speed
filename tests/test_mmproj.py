"""Tests for multimodal projector (mmproj) discovery and memory accounting."""

import json

import llamafit.gguf as gguf_mod
from llamafit.cli import main
from llamafit.gguf import discover_mmproj, load_gguf, load_mmproj, pick_hf_mmproj
from llamafit.memory import MIB, RunConfig, estimate
from llamafit.model import extract_model_info

from conftest import llama_metadata, llama_tensors, write_gguf


def _write_mmproj(path, nbytes=200 * MIB, name="test-vit"):
    """A projector GGUF; one fat tensor stands in for the whole encoder."""
    write_gguf(
        path,
        {"general.architecture": "clip", "general.name": name},
        [("mm.patch_embd.weight", (1, 1), nbytes)],
    )
    return path


def _model(tmp_path, name="vlm-Q4_K_M.gguf"):
    p = tmp_path / name
    write_gguf(p, llama_metadata(), llama_tensors())
    return p


# --- projector picking (mirrors llama.cpp -hf auto-selection) ---

def test_pick_hf_mmproj_prefers_full_precision():
    files = [
        "model-Q4_K_M.gguf",
        "mmproj-model-Q8_0.gguf",
        "mmproj-model-f16.gguf",
        "README.md",
    ]
    assert pick_hf_mmproj(files) == "mmproj-model-f16.gguf"


def test_pick_hf_mmproj_quantized_fallback():
    assert pick_hf_mmproj(["mmproj-model-Q8_0.gguf"]) == "mmproj-model-Q8_0.gguf"


def test_pick_hf_mmproj_none_when_absent():
    assert pick_hf_mmproj(["model-Q4_K_M.gguf", "README.md"]) is None


# --- discovery ---

def test_discover_local_sibling(tmp_path):
    model = _model(tmp_path)
    proj = _write_mmproj(tmp_path / "mmproj-vlm-f16.gguf")
    assert discover_mmproj(str(model)) == str(proj)


def test_discover_no_sibling(tmp_path):
    assert discover_mmproj(str(_model(tmp_path, "plain-Q4_K_M.gguf"))) is None


def test_discover_hf_spec(monkeypatch):
    monkeypatch.setattr(
        gguf_mod, "_hf_list_repo_files",
        lambda repo, token: ["model-Q4_K_M.gguf", "mmproj-model-f16.gguf"],
    )
    assert discover_mmproj("org/repo:Q4_K_M") == (
        "https://huggingface.co/org/repo/resolve/main/mmproj-model-f16.gguf"
    )


def test_load_mmproj_sums_tensor_bytes(tmp_path):
    proj = _write_mmproj(tmp_path / "mmproj-vlm-f16.gguf", 128 * MIB, name="my-vit")
    name, nbytes, n_tensors = load_mmproj(str(proj))
    assert name == "my-vit"
    assert nbytes == 128 * MIB
    assert n_tensors == 1


# --- memory accounting ---

def test_estimate_mmproj_placement(tmp_path):
    m = extract_model_info(load_gguf(str(_model(tmp_path))))
    cfg = RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1)
    base = estimate(m, cfg)

    m.mmproj_bytes = 200 * MIB
    on_gpu = estimate(m, cfg)
    assert on_gpu.gpu_mmproj == 200 * MIB
    assert on_gpu.gpu_total == base.gpu_total + 200 * MIB
    assert on_gpu.cpu_weights == base.cpu_weights

    m.mmproj_on_gpu = False
    on_cpu = estimate(m, cfg)
    assert on_cpu.gpu_mmproj == 0
    assert on_cpu.gpu_total == base.gpu_total
    assert on_cpu.cpu_weights == base.cpu_weights + 200 * MIB


def test_mmproj_is_fixed_across_offload(tmp_path):
    """The projector allocation does not change with -ngl."""
    m = extract_model_info(load_gguf(str(_model(tmp_path))))
    m.mmproj_bytes = 100 * MIB
    full = estimate(m, RunConfig(n_ctx=4096, n_gpu_layers=m.n_layer + 1))
    partial = estimate(m, RunConfig(n_ctx=4096, n_gpu_layers=1))
    assert full.gpu_mmproj == partial.gpu_mmproj == 100 * MIB


# --- CLI end-to-end ---

def test_cli_auto_detects_sibling(tmp_path, capsys):
    model = _model(tmp_path)
    _write_mmproj(tmp_path / "mmproj-vlm-f16.gguf", 300 * MIB)
    assert main([str(model), "--vram", "8", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["mmproj"]["weight_bytes"] == 300 * MIB
    assert out["model"]["mmproj"]["on_gpu"] is True
    assert out["requirement"]["mmproj_bytes"] == 300 * MIB
    assert out["requirement"]["total_bytes"] == (
        out["requirement"]["weights_bytes"] + 300 * MIB + out["requirement"]["kv_bytes"]
    )
    assert out["fit"]["estimate"]["gpu_mmproj"] == 300 * MIB


def test_cli_no_mmproj_opts_out(tmp_path, capsys):
    model = _model(tmp_path)
    _write_mmproj(tmp_path / "mmproj-vlm-f16.gguf")
    assert main([str(model), "--vram", "8", "--no-mmproj", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["mmproj"] is None
    assert out["requirement"]["mmproj_bytes"] == 0


def test_cli_explicit_mmproj_and_no_offload(tmp_path, capsys):
    model = _model(tmp_path)
    # non-"mmproj" name → not auto-detected, must be passed explicitly
    proj = _write_mmproj(tmp_path / "projector-f16.gguf", 150 * MIB)
    assert main([str(model), "--vram", "8", "--mmproj", str(proj),
                 "--no-mmproj-offload", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["mmproj"]["weight_bytes"] == 150 * MIB
    assert out["model"]["mmproj"]["on_gpu"] is False
    assert out["fit"]["estimate"]["gpu_mmproj"] == 0
    assert out["fit"]["estimate"]["cpu_weights"] >= 150 * MIB


def test_command_line_emits_mmproj_flags(tmp_path):
    from llamafit.fit import command_line

    proj = str(_write_mmproj(tmp_path / "mmproj-vlm-f16.gguf"))
    cfg = RunConfig(n_ctx=4096, n_gpu_layers=9)
    on_gpu = command_line(str(_model(tmp_path)), cfg, mmproj=proj)
    assert f"--mmproj {proj}" in on_gpu
    assert "--no-mmproj-offload" not in on_gpu
    on_cpu = command_line(str(_model(tmp_path)), cfg, mmproj=proj, mmproj_offload=False)
    assert "--no-mmproj-offload" in on_cpu
    # -hf specs download the projector automatically → no --mmproj emitted
    remote = command_line("org/repo:Q4_K_M", cfg, mmproj=proj)
    assert "--mmproj" in remote  # local projector path still exists → emitted


def test_cli_terminal_shows_mmproj(tmp_path, capsys):
    model = _model(tmp_path)
    _write_mmproj(tmp_path / "mmproj-vlm-f16.gguf", 300 * MIB)
    assert main([str(model), "--vram", "8"]) == 0
    out = capsys.readouterr().out
    assert "mmproj" in out
    assert "--mmproj" in out  # suggested command wires up the local projector
