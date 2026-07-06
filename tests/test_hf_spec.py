"""Tests for llama.cpp -hf style spec resolution (org/repo[:QUANT])."""

import os

import pytest

import llamafit.gguf as gguf_mod
from llamafit.gguf import GGUFError, load_gguf, parse_hf_spec, pick_hf_file


# --- spec parsing ---

def test_parse_plain_spec():
    assert parse_hf_spec("unsloth/Qwen3.5-9B-GGUF:Q4_K_M") == ("unsloth/Qwen3.5-9B-GGUF", "Q4_K_M")
    assert parse_hf_spec("unsloth/Qwen3.5-9B-GGUF") == ("unsloth/Qwen3.5-9B-GGUF", None)


def test_parse_prefixed_specs():
    assert parse_hf_spec("hf:org/repo:Q8_0") == ("org/repo", "Q8_0")
    assert parse_hf_spec("hf:org/repo") == ("org/repo", None)
    assert parse_hf_spec("hf.co/org/repo:IQ4_XS") == ("org/repo", "IQ4_XS")


def test_parse_rejects_file_forms():
    # direct-file form is handled by the URL path, not the spec resolver
    assert parse_hf_spec("hf:org/repo/file.gguf") is None
    assert parse_hf_spec("models/local-file.gguf") is None
    assert parse_hf_spec("model.gguf") is None
    assert parse_hf_spec("./relative/path") is None
    assert parse_hf_spec("no-slash") is None
    assert parse_hf_spec("https://huggingface.co/org/repo") is None


# --- file picking (mirrors llama.cpp download.cpp) ---

FILES = [
    "README.md",
    "model-Q2_K.gguf",
    "model-Q4_K_M.gguf",
    "model-Q4_K_S.gguf",
    "model-Q8_0.gguf",
    "mmproj-model-f16.gguf",
    "imatrix-stuff-Q4_K_M.gguf",
]


def test_pick_exact_tag():
    assert pick_hf_file(FILES, "Q4_K_M") == "model-Q4_K_M.gguf"
    assert pick_hf_file(FILES, "Q2_K") == "model-Q2_K.gguf"


def test_pick_is_case_insensitive():
    assert pick_hf_file(FILES, "q4_k_m") == "model-Q4_K_M.gguf"


def test_tag_boundary_prevents_prefix_match():
    # Q4_K must not match Q4_K_M / Q4_K_S (llama.cpp uses "{tag}[.-]")
    with pytest.raises(GGUFError, match="no GGUF matches"):
        pick_hf_file(["model-Q4_K_M.gguf", "model-Q4_K_S.gguf"], "Q4_K")
    assert pick_hf_file(FILES + ["model-Q4_K.gguf"], "Q4_K") == "model-Q4_K.gguf"


def test_default_tags_prefer_q4_k_m_then_q8_0():
    assert pick_hf_file(FILES, None) == "model-Q4_K_M.gguf"
    assert pick_hf_file(["m-Q8_0.gguf", "m-Q5_K_M.gguf"], None) == "m-Q8_0.gguf"


def test_default_falls_back_to_single_file():
    assert pick_hf_file(["only-model-IQ4_XS.gguf", "README.md"], None) == "only-model-IQ4_XS.gguf"


def test_mmproj_imatrix_mtp_excluded():
    files = ["mmproj-Q4_K_M.gguf", "imatrix-Q4_K_M.gguf", "mtp-model-Q4_K_M.gguf"]
    with pytest.raises(GGUFError, match="no GGUF model files"):
        pick_hf_file(files, "Q4_K_M")


def test_shards_pick_first_only():
    files = [
        "big-Q4_K_M-00002-of-00003.gguf",
        "big-Q4_K_M-00001-of-00003.gguf",
        "big-Q4_K_M-00003-of-00003.gguf",
    ]
    assert pick_hf_file(files, "Q4_K_M") == "big-Q4_K_M-00001-of-00003.gguf"


def test_unmatched_tag_lists_available():
    with pytest.raises(GGUFError, match="available files.*model-Q8_0.gguf"):
        pick_hf_file(FILES, "IQ1_S")


# --- end-to-end dispatch with mocked network ---

class FakeRangeFile:
    """Stands in for HTTPRangeFile, serving a local file."""

    registry: dict[str, str] = {}

    def __init__(self, url, token=None):
        self.path = self.registry[url]
        self.fp = open(self.path, "rb")
        self.size = os.path.getsize(self.path)

    def read(self, n):
        return self.fp.read(n)

    def seek(self, off, whence=0):
        return self.fp.seek(off, whence)

    def tell(self):
        return self.fp.tell()

    def close(self):
        self.fp.close()


def test_load_gguf_resolves_hf_spec(tiny_llama, monkeypatch):
    url = "https://huggingface.co/unsloth/Tiny-GGUF/resolve/main/tiny-Q4_K_M.gguf"
    FakeRangeFile.registry = {url: str(tiny_llama)}
    monkeypatch.setattr(gguf_mod, "HTTPRangeFile", FakeRangeFile)
    monkeypatch.setattr(
        gguf_mod, "_hf_list_repo_files",
        lambda repo, token: ["README.md", "tiny-Q4_K_M.gguf", "tiny-Q8_0.gguf"],
    )
    g = load_gguf("unsloth/Tiny-GGUF:Q4_K_M")
    assert g.get("general.architecture") == "llama"
    assert g.path == url
    # default tag resolution also lands on Q4_K_M
    assert load_gguf("unsloth/Tiny-GGUF").path == url
    # hf:-prefixed spec forms must go through the same resolver
    assert load_gguf("hf:unsloth/Tiny-GGUF:Q4_K_M").path == url
    assert load_gguf("hf:unsloth/Tiny-GGUF").path == url
    assert load_gguf("hf.co/unsloth/Tiny-GGUF:Q4_K_M").path == url
    # ...while the direct-file form keeps working
    FakeRangeFile.registry[
        "https://huggingface.co/unsloth/Tiny-GGUF/resolve/main/tiny-Q8_0.gguf"
    ] = str(tiny_llama)
    g2 = load_gguf("hf:unsloth/Tiny-GGUF/tiny-Q8_0.gguf")
    assert g2.path.endswith("tiny-Q8_0.gguf")


def test_local_path_wins_over_spec(tmp_path, monkeypatch):
    # a relative path that also matches the spec grammar must stay local
    def boom(repo, token):
        raise AssertionError("network resolver must not be called")

    monkeypatch.setattr(gguf_mod, "_hf_list_repo_files", boom)
    d = tmp_path / "org"
    d.mkdir()
    from conftest import llama_metadata, llama_tensors, write_gguf

    write_gguf(d / "repo", llama_metadata(), llama_tensors())
    monkeypatch.chdir(tmp_path)
    g = load_gguf("org/repo")
    assert g.get("general.architecture") == "llama"
