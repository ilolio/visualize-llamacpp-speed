"""Test helpers: a minimal GGUF writer to build synthetic models."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

ALIGN = 32

T_UINT32, T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64 = 4, 6, 7, 8, 9, 10

GGML_F16 = 1
F16_SIZE = 2


def _pack_string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def _pack_value(v) -> bytes:
    if isinstance(v, bool):
        return struct.pack("<I", T_BOOL) + struct.pack("<B", int(v))
    if isinstance(v, int):
        if v < 1 << 32:
            return struct.pack("<I", T_UINT32) + struct.pack("<I", v)
        return struct.pack("<I", T_UINT64) + struct.pack("<Q", v)
    if isinstance(v, float):
        return struct.pack("<I", T_FLOAT32) + struct.pack("<f", v)
    if isinstance(v, str):
        return struct.pack("<I", T_STRING) + _pack_string(v)
    if isinstance(v, (list, tuple)):
        first = v[0] if v else 0
        if isinstance(first, str):
            elem_t, packer = T_STRING, _pack_string
        elif isinstance(first, bool):
            elem_t, packer = T_BOOL, lambda x: struct.pack("<B", int(x))
        elif isinstance(first, float):
            elem_t, packer = T_FLOAT32, lambda x: struct.pack("<f", x)
        else:
            elem_t, packer = T_UINT32, lambda x: struct.pack("<I", x)
        body = b"".join(packer(x) for x in v)
        return struct.pack("<I", T_ARRAY) + struct.pack("<IQ", elem_t, len(v)) + body
    raise TypeError(f"cannot encode {type(v)}")


def write_gguf(path: Path, metadata: dict, tensors: list) -> Path:
    """Write a GGUF v3 file with sparse (zero) tensor data.

    Each tensor entry is ``(name, shape)`` for F16 sizing, or
    ``(name, shape, nbytes)`` to emulate an arbitrary quantization's storage
    size (the parser derives sizes from offsets, so any value works).
    """
    header = bytearray()
    header += b"GGUF"
    header += struct.pack("<I", 3)
    header += struct.pack("<QQ", len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _pack_string(key)
        header += _pack_value(value)

    entries = []
    offset = 0
    for t in tensors:
        name, shape = t[0], t[1]
        if len(t) > 2:
            nbytes = t[2]
        else:
            nbytes = F16_SIZE
            for d in shape:
                nbytes *= d
        entries.append((name, shape, offset))
        offset += (nbytes + ALIGN - 1) // ALIGN * ALIGN

    for name, shape, off in entries:
        header += _pack_string(name)
        header += struct.pack("<I", len(shape))
        header += struct.pack(f"<{len(shape)}Q", *shape)
        header += struct.pack("<I", GGML_F16)
        header += struct.pack("<Q", off)

    pad = (-len(header)) % ALIGN
    with open(path, "wb") as f:
        f.write(header)
        f.write(b"\0" * pad)
        # tensor data is never read — a sparse file of the right size is enough
        if offset:
            f.seek(offset - 1, 1)
            f.write(b"\0")
    return path


def llama_metadata(
    n_layer=4,
    n_embd=256,
    n_head=8,
    n_head_kv=4,
    n_ff=512,
    n_ctx_train=8192,
    n_vocab=1000,
    **extra,
):
    md = {
        "general.architecture": "llama",
        "general.name": "TestLlama",
        "general.file_type": 1,  # F16
        "llama.block_count": n_layer,
        "llama.embedding_length": n_embd,
        "llama.attention.head_count": n_head,
        "llama.attention.head_count_kv": n_head_kv,
        "llama.feed_forward_length": n_ff,
        "llama.context_length": n_ctx_train,
        "llama.vocab_size": n_vocab,
        "tokenizer.ggml.tokens": [f"tok{i}" for i in range(n_vocab)],
    }
    md.update(extra)
    return md


def llama_tensors(n_layer=4, n_embd=256, n_ff=512, n_vocab=1000):
    tensors = [("token_embd.weight", (n_embd, n_vocab))]
    for i in range(n_layer):
        tensors += [
            (f"blk.{i}.attn_q.weight", (n_embd, n_embd)),
            (f"blk.{i}.attn_k.weight", (n_embd, n_embd // 2)),
            (f"blk.{i}.attn_v.weight", (n_embd, n_embd // 2)),
            (f"blk.{i}.attn_output.weight", (n_embd, n_embd)),
            (f"blk.{i}.ffn_gate.weight", (n_embd, n_ff)),
            (f"blk.{i}.ffn_down.weight", (n_ff, n_embd)),
            (f"blk.{i}.ffn_up.weight", (n_embd, n_ff)),
        ]
    tensors += [("output_norm.weight", (n_embd,)), ("output.weight", (n_embd, n_vocab))]
    return tensors


@pytest.fixture
def tiny_llama(tmp_path):
    path = tmp_path / "tiny-llama-F16.gguf"
    write_gguf(path, llama_metadata(), llama_tensors())
    return path
