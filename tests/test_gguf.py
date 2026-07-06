from pathlib import Path

import pytest

from llamafit.gguf import ArrayInfo, GGUFError, load_gguf
from llamafit.model import extract_model_info

from conftest import llama_metadata, llama_tensors, write_gguf


def test_parse_metadata(tiny_llama):
    g = load_gguf(str(tiny_llama))
    assert g.version == 3
    assert g.get("general.architecture") == "llama"
    assert g.get("llama.block_count") == 4
    assert g.get("llama.attention.head_count_kv") == 4
    assert g.array_len("tokenizer.ggml.tokens") == 1000
    assert g.file_size == tiny_llama.stat().st_size


def test_tensor_sizes_from_offsets(tiny_llama):
    g = load_gguf(str(tiny_llama))
    by_name = {t.name: t for t in g.tensors}
    embd = by_name["token_embd.weight"]
    assert embd.shape == (256, 1000)
    # F16, alignment-padded: exact size since 256*1000*2 is a multiple of 32
    assert embd.nbytes == 256 * 1000 * 2
    q = by_name["blk.0.attn_q.weight"]
    assert q.nbytes == 256 * 256 * 2
    # every tensor got a positive size
    assert all(t.nbytes > 0 for t in g.tensors)


def test_model_info_extraction(tiny_llama):
    g = load_gguf(str(tiny_llama))
    m = extract_model_info(g)
    assert m.arch == "llama"
    assert m.quant == "F16"
    assert m.n_layer == 4
    assert m.n_head_kv == [4, 4, 4, 4]
    assert m.n_embd_head_k == 256 // 8
    assert m.n_vocab == 1000
    assert m.embd_bytes == 256 * 1000 * 2
    assert m.output_bytes >= 256 * 1000 * 2  # output.weight + output_norm
    assert len(m.layer_weights) == 4
    assert all(w.dense > 0 and w.expert == 0 for w in m.layer_weights)
    # total classification covers every tensor byte
    assert m.weight_bytes_total == sum(t.nbytes for t in g.tensors)


def test_large_arrays_are_summarized(tiny_llama):
    g = load_gguf(str(tiny_llama))
    # vocab of 1000 is below the materialize limit; force a big one
    assert isinstance(g.metadata["tokenizer.ggml.tokens"], list)


def test_big_vocab_array_skipped(tmp_path):
    md = llama_metadata(n_vocab=6000)
    write_gguf(tmp_path / "m.gguf", md, llama_tensors(n_vocab=6000))
    g = load_gguf(str(tmp_path / "m.gguf"))
    tokens = g.metadata["tokenizer.ggml.tokens"]
    assert isinstance(tokens, ArrayInfo)
    assert tokens.length == 6000
    assert g.array_len("tokenizer.ggml.tokens") == 6000
    # vocab_size key still wins for the model info
    m = extract_model_info(g)
    assert m.n_vocab == 6000


def test_shards(tmp_path):
    md = llama_metadata()
    tensors = llama_tensors()
    half = len(tensors) // 2
    p1 = tmp_path / "model-00001-of-00002.gguf"
    p2 = tmp_path / "model-00002-of-00002.gguf"
    write_gguf(p1, md, tensors[:half])
    write_gguf(p2, {"general.architecture": "llama"}, tensors[half:])
    g = load_gguf(str(p1))
    assert g.n_shards == 2
    assert len(g.tensors) == len(tensors)
    assert g.file_size == p1.stat().st_size + p2.stat().st_size
    # metadata comes from shard 1
    assert g.get("general.name") == "TestLlama"


def test_missing_shard_raises(tmp_path):
    p1 = tmp_path / "model-00001-of-00002.gguf"
    write_gguf(p1, llama_metadata(), llama_tensors()[:3])
    with pytest.raises(GGUFError, match="missing shard"):
        load_gguf(str(p1))


def test_not_gguf(tmp_path):
    p = tmp_path / "bogus.gguf"
    p.write_bytes(b"NOPE" + b"\0" * 100)
    with pytest.raises(GGUFError, match="bad magic"):
        load_gguf(str(p))


def test_file_not_found():
    with pytest.raises(GGUFError, match="not found"):
        load_gguf("/does/not/exist.gguf")
