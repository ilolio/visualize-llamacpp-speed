"""Minimal, dependency-free GGUF metadata reader.

Reads only the header (metadata key/values + tensor infos) — tensor data is
never loaded.  Works on local files, split shards (``-00001-of-00003.gguf``)
and remote URLs via HTTP range requests, so a 100 GB model can be inspected
by downloading a few MB.

Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import json
import os
import re
import struct
import urllib.error
import urllib.request
from urllib.parse import quote
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
T_UINT8, T_INT8, T_UINT16, T_INT16 = 0, 1, 2, 3
T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 4, 5, 6, 7
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 8, 9, 10, 11, 12

_SCALAR_FMT = {
    T_UINT8: ("<B", 1),
    T_INT8: ("<b", 1),
    T_UINT16: ("<H", 2),
    T_INT16: ("<h", 2),
    T_UINT32: ("<I", 4),
    T_INT32: ("<i", 4),
    T_FLOAT32: ("<f", 4),
    T_BOOL: ("<B", 1),
    T_UINT64: ("<Q", 8),
    T_INT64: ("<q", 8),
    T_FLOAT64: ("<d", 8),
}

# Arrays longer than this keep only their length (tokenizer vocabularies can
# be hundreds of thousands of entries — we never need their contents).
ARRAY_MATERIALIZE_LIMIT = 4096

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


class GGUFError(Exception):
    """Raised when a file cannot be parsed as GGUF."""


@dataclass
class ArrayInfo:
    """Placeholder for a large array whose contents were skipped."""

    elem_type: int
    length: int


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    offset: int
    nbytes: int = 0  # filled in from offset deltas

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass
class GGUFFile:
    """Parsed header of one GGUF file (or several merged shards)."""

    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorInfo]
    file_size: int  # total bytes across all shards
    n_shards: int = 1

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def array_len(self, key: str) -> int | None:
        v = self.metadata.get(key)
        if isinstance(v, ArrayInfo):
            return v.length
        if isinstance(v, (list, tuple)):
            return len(v)
        return None


class _Reader:
    """Buffered reader over a binary source with mandatory exact reads."""

    def __init__(self, fp: BinaryIO, size: int, name: str):
        self.fp = fp
        self.size = size
        self.name = name

    def read(self, n: int) -> bytes:
        data = self.fp.read(n)
        if len(data) != n:
            raise GGUFError(f"{self.name}: truncated file (wanted {n} bytes, got {len(data)})")
        return data

    def skip(self, n: int) -> None:
        self.fp.seek(n, os.SEEK_CUR)

    def tell(self) -> int:
        return self.fp.tell()


class HTTPRangeFile:
    """File-like object over HTTP using Range requests, with chunk caching."""

    CHUNK = 1 << 20  # 1 MiB

    def __init__(self, url: str, token: str | None = None):
        self.url = url
        self.headers = {"User-Agent": "llamafit/0.1"}
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
        self.pos = 0
        self._chunks: dict[int, bytes] = {}
        self.size = self._probe_size()

    def _request(self, start: int, end: int) -> bytes:
        req = urllib.request.Request(
            self.url, headers={**self.headers, "Range": f"bytes={start}-{end}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read()
        except urllib.error.URLError as e:
            raise GGUFError(f"network error fetching {self.url}: {e}") from e

    def _probe_size(self) -> int:
        req = urllib.request.Request(self.url, headers={**self.headers, "Range": "bytes=0-0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                content_range = resp.headers.get("Content-Range", "")
                m = re.search(r"/(\d+)$", content_range)
                if m:
                    return int(m.group(1))
                length = resp.headers.get("Content-Length")
                if resp.status == 200 and length:
                    return int(length)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise GGUFError(
                    f"access denied for {self.url} — for gated Hugging Face repos, "
                    "set the HF_TOKEN environment variable"
                ) from e
            raise GGUFError(f"HTTP {e.code} fetching {self.url}") from e
        except urllib.error.URLError as e:
            raise GGUFError(f"network error fetching {self.url}: {e}") from e
        raise GGUFError(f"server for {self.url} does not support range requests")

    def _chunk(self, idx: int) -> bytes:
        if idx not in self._chunks:
            start = idx * self.CHUNK
            end = min(start + self.CHUNK, self.size) - 1
            self._chunks[idx] = self._request(start, end)
        return self._chunks[idx]

    def read(self, n: int) -> bytes:
        n = min(n, self.size - self.pos)
        out = bytearray()
        while n > 0:
            idx, off = divmod(self.pos, self.CHUNK)
            piece = self._chunk(idx)[off : off + n]
            out += piece
            self.pos += len(piece)
            n -= len(piece)
        return bytes(out)

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_SET:
            self.pos = offset
        elif whence == os.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def tell(self) -> int:
        return self.pos

    def close(self) -> None:
        self._chunks.clear()


def _read_string(r: _Reader) -> str:
    (n,) = struct.unpack("<Q", r.read(8))
    if n > 1 << 32:
        raise GGUFError(f"{r.name}: implausible string length {n}")
    return r.read(n).decode("utf-8", errors="replace")


def _read_value(r: _Reader, vtype: int) -> Any:
    if vtype in _SCALAR_FMT:
        fmt, sz = _SCALAR_FMT[vtype]
        (v,) = struct.unpack(fmt, r.read(sz))
        return bool(v) if vtype == T_BOOL else v
    if vtype == T_STRING:
        return _read_string(r)
    if vtype == T_ARRAY:
        (elem_type,) = struct.unpack("<I", r.read(4))
        (count,) = struct.unpack("<Q", r.read(8))
        if count > ARRAY_MATERIALIZE_LIMIT:
            _skip_array(r, elem_type, count)
            return ArrayInfo(elem_type, count)
        return [_read_value(r, elem_type) for _ in range(count)]
    raise GGUFError(f"{r.name}: unknown metadata value type {vtype}")


def _skip_array(r: _Reader, elem_type: int, count: int) -> None:
    if elem_type in _SCALAR_FMT:
        r.skip(count * _SCALAR_FMT[elem_type][1])
        return
    if elem_type == T_STRING:
        for _ in range(count):
            (n,) = struct.unpack("<Q", r.read(8))
            r.skip(n)
        return
    if elem_type == T_ARRAY:
        for _ in range(count):
            (et,) = struct.unpack("<I", r.read(4))
            (c,) = struct.unpack("<Q", r.read(8))
            _skip_array(r, et, c)
        return
    raise GGUFError(f"{r.name}: unknown array element type {elem_type}")


def _parse_one(fp: BinaryIO, size: int, name: str) -> tuple[int, dict[str, Any], list[TensorInfo]]:
    r = _Reader(fp, size, name)
    if r.read(4) != GGUF_MAGIC:
        raise GGUFError(f"{name}: not a GGUF file (bad magic)")
    (version,) = struct.unpack("<I", r.read(4))
    if version not in (2, 3):
        raise GGUFError(f"{name}: unsupported GGUF version {version} (only v2/v3)")
    tensor_count, kv_count = struct.unpack("<QQ", r.read(16))
    if tensor_count > 1 << 24 or kv_count > 1 << 24:
        raise GGUFError(f"{name}: implausible header counts")

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = _read_string(r)
        (vtype,) = struct.unpack("<I", r.read(4))
        metadata[key] = _read_value(r, vtype)

    tensors: list[TensorInfo] = []
    for _ in range(tensor_count):
        tname = _read_string(r)
        (n_dims,) = struct.unpack("<I", r.read(4))
        if n_dims > 8:
            raise GGUFError(f"{name}: tensor {tname} has {n_dims} dims")
        dims = struct.unpack(f"<{n_dims}Q", r.read(8 * n_dims))
        ggml_type, = struct.unpack("<I", r.read(4))
        (offset,) = struct.unpack("<Q", r.read(8))
        tensors.append(TensorInfo(tname, tuple(dims), ggml_type, offset))

    # Tensor byte sizes from offset deltas within the data section.  This is
    # exact for what the file actually stores (padding included) and needs no
    # per-quant-type size table, so it never goes stale.
    alignment = metadata.get("general.alignment", DEFAULT_ALIGNMENT)
    if not isinstance(alignment, int) or alignment <= 0:
        alignment = DEFAULT_ALIGNMENT
    header_end = r.tell()
    data_start = (header_end + alignment - 1) // alignment * alignment
    data_size = size - data_start
    ordered = sorted(tensors, key=lambda t: t.offset)
    for i, t in enumerate(ordered):
        end = ordered[i + 1].offset if i + 1 < len(ordered) else data_size
        t.nbytes = max(0, end - t.offset)

    return version, metadata, tensors


def _shard_paths(path: Path) -> list[Path]:
    m = SHARD_RE.search(path.name)
    if not m:
        return [path]
    total = int(m.group(2))
    prefix = path.name[: m.start()]
    shards = [path.parent / f"{prefix}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]
    missing = [p.name for p in shards if not p.exists()]
    if missing:
        raise GGUFError(f"missing shard(s): {', '.join(missing)}")
    return shards


def _shard_urls(url: str) -> list[str]:
    m = SHARD_RE.search(url)
    if not m:
        return [url]
    total = int(m.group(2))
    prefix = url[: m.start()]
    return [f"{prefix}-{i:05d}-of-{total:05d}.gguf" for i in range(1, total + 1)]


# -hf style spec: "org/repo" or "org/repo:QUANT" (exactly one slash, no .gguf)
HF_SPEC_RE = re.compile(r"^([A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*?)(?::([\w.-]+))?$")

# llama.cpp's default tag preference when -hf has no :quant
HF_DEFAULT_TAGS = ["Q4_K_M", "Q8_0"]


def parse_hf_spec(source: str) -> tuple[str, str | None] | None:
    """Parse a llama.cpp ``-hf`` style spec into (repo, tag).

    Accepts ``org/repo``, ``org/repo:Q4_K_M`` and the same with an ``hf:`` or
    ``hf.co/`` prefix.  Returns None for anything that looks like a file path
    or a direct-file spec (``hf:org/repo/file.gguf``).
    """
    s = source
    if s.startswith("hf:"):
        s = s[3:].lstrip("/")
    elif s.startswith("hf.co/"):
        s = s[6:]
    if s.lower().endswith(".gguf"):
        return None
    m = HF_SPEC_RE.match(s)
    if not m:
        return None
    return m.group(1), m.group(2)


def _gguf_is_model_file(name: str) -> bool:
    """Mirror llama.cpp: skip multimodal projectors, imatrix data, MTP files."""
    low = name.lower()
    return "mmproj" not in low and "imatrix" not in low and "mtp-" not in low


def _first_shard_or_single(name: str) -> bool:
    m = SHARD_RE.search(name)
    return m is None or int(m.group(1)) == 1


def pick_hf_file(files: list[str], tag: str | None) -> str:
    """Pick the GGUF filename for a quant tag, matching llama.cpp's logic:

    try each tag as a case-insensitive ``{tag}[.-]`` search over model GGUFs
    (skipping non-first shards); default tags are Q4_K_M then Q8_0.
    """
    candidates = [
        f for f in files
        if f.lower().endswith(".gguf") and _gguf_is_model_file(f) and _first_shard_or_single(f)
    ]
    if not candidates:
        raise GGUFError("no GGUF model files found in this repo")
    tags = [tag] if tag else HF_DEFAULT_TAGS
    for t in tags:
        pattern = re.compile(re.escape(t) + r"[.-]", re.IGNORECASE)
        for f in candidates:
            if pattern.search(f):
                return f
    if tag is None and len(candidates) == 1:
        return candidates[0]
    available = ", ".join(sorted(candidates))
    wanted = tag if tag is not None else " / ".join(HF_DEFAULT_TAGS)
    raise GGUFError(f"no GGUF matches tag {wanted!r} — available files: {available}")


def _hf_list_repo_files(repo: str, token: str | None) -> list[str]:
    """List a Hugging Face repo's files via the hub API."""
    url = f"https://huggingface.co/api/models/{repo}"
    headers = {"User-Agent": "llamafit/0.1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise GGUFError(
                f"access denied for repo {repo} — for gated repos, set HF_TOKEN"
            ) from e
        if e.code == 404:
            raise GGUFError(f"Hugging Face repo not found: {repo}") from e
        raise GGUFError(f"HTTP {e.code} listing {repo}") from e
    except urllib.error.URLError as e:
        raise GGUFError(f"network error listing {repo}: {e}") from e
    return [s["rfilename"] for s in data.get("siblings", []) if "rfilename" in s]


def resolve_hf_spec(repo: str, tag: str | None, token: str | None) -> str:
    """Resolve a ``-hf`` spec to a resolve/main URL for the chosen file."""
    files = _hf_list_repo_files(repo, token)
    chosen = pick_hf_file(files, tag)
    return f"https://huggingface.co/{repo}/resolve/main/{quote(chosen)}"


def _normalize_url(source: str) -> str | None:
    if source.startswith(("http://", "https://")):
        return source
    if source.startswith("hf:"):
        spec = source[3:].lstrip("/")
        parts = spec.split("/")
        if len(parts) < 3:
            raise GGUFError(
                "hf: source must look like hf:org/repo/file.gguf or hf:org/repo[:QUANT]"
            )
        org, repo, filename = parts[0], parts[1], "/".join(parts[2:])
        return f"https://huggingface.co/{org}/{repo}/resolve/main/{filename}"
    return None


def load_gguf(source: str) -> GGUFFile:
    """Parse a GGUF header from a local path, HF spec, or http(s) URL.

    Accepted Hugging Face forms (checked only when no local file matches):
    ``org/repo[:QUANT]`` (llama.cpp ``-hf`` style, e.g. ``unsloth/X-GGUF:Q4_K_M``),
    ``hf:org/repo[:QUANT]``, ``hf:org/repo/file.gguf``.

    For split models, pass any shard — siblings are discovered automatically
    and their tensor lists merged (metadata comes from the first shard).
    """
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    # -hf style specs (org/repo[:QUANT], possibly hf:-prefixed) are checked
    # before the direct-file forms, but a matching local path always wins.
    spec = parse_hf_spec(source)
    if spec is not None and not Path(source).expanduser().exists():
        url = resolve_hf_spec(spec[0], spec[1], token)
    else:
        url = _normalize_url(source)
    if url is not None:
        version = 0
        metadata: dict[str, Any] = {}
        tensors: list[TensorInfo] = []
        total_size = 0
        urls = _shard_urls(url)
        for i, u in enumerate(urls):
            f = HTTPRangeFile(u, token=token)
            try:
                v, md, ts = _parse_one(f, f.size, u)
            finally:
                f.close()
            total_size += f.size
            if i == 0:
                version, metadata = v, md
            tensors.extend(ts)
        return GGUFFile(url, version, metadata, tensors, total_size, n_shards=len(urls))

    path = Path(source).expanduser()
    if not path.exists():
        raise GGUFError(f"file not found: {path}")
    shards = _shard_paths(path)
    version = 0
    metadata = {}
    tensors = []
    total_size = 0
    for i, p in enumerate(shards):
        size = p.stat().st_size
        total_size += size
        with open(p, "rb") as fp:
            v, md, ts = _parse_one(fp, size, p.name)
        if i == 0:
            version, metadata = v, md
        tensors.extend(ts)
    return GGUFFile(str(shards[0]), version, metadata, tensors, total_size, n_shards=len(shards))
