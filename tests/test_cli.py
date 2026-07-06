import json

import pytest

from llamafit.cli import main


def test_json_output(tiny_llama, capsys):
    rc = main([str(tiny_llama), "--vram", "8", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"]["arch"] == "llama"
    assert out["model"]["n_layer"] == 4
    assert out["target_ctx"] == 8192  # min(train ctx, 32768)
    assert len(out["kv_scenarios"]) == 3
    assert out["fit"]["fits"] is True
    assert out["fit"]["config"]["n_gpu_layers"] == 5
    assert out["recommendations"]
    assert out["budget"]["bytes"] == 8 * 1024**3 - 1024 * 1024**2


def test_terminal_output(tiny_llama, capsys):
    rc = main([str(tiny_llama), "--vram", "8"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "TestLlama" in out
    assert "Model memory needed" in out  # raw weights+KV demand comes first
    assert "GPU usage" in out            # then the post-offload view
    assert "CPU RAM needed" in out       # and the host-RAM side of the split
    assert out.index("Model memory needed") < out.index("GPU usage") < out.index("CPU RAM needed")
    assert "KV cache options" in out
    assert "llama-server" in out


def test_no_budget_still_works(tiny_llama, capsys, monkeypatch):
    monkeypatch.setattr("llamafit.cli.detect_gpus", lambda: [])
    rc = main([str(tiny_llama)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Model memory needed" in out  # requirement bar works without a budget
    assert "KV cache options" in out
    assert "--vram" in out  # tip shown


def test_pinned_params(tiny_llama, capsys):
    rc = main([str(tiny_llama), "--vram", "8", "--ngl", "2", "--ctk", "q8_0", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["fit"]["config"]["n_gpu_layers"] == 2
    assert out["fit"]["config"]["cache_type_k"] == "q8_0"
    assert out["fit"]["config"]["cache_type_v"] == "q8_0"
    assert "pinned" in out["fit"]["actions"][0]


def test_ctx_flag(tiny_llama, capsys):
    rc = main([str(tiny_llama), "--vram", "8", "-c", "2048", "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["target_ctx"] == 2048


def test_bad_file(tmp_path, capsys):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"garbage!")
    assert main([str(p)]) == 1


def test_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
