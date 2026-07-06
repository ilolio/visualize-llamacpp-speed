"""Detect available GPU VRAM and system RAM without extra dependencies."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from dataclasses import dataclass

from .memory import MIB


@dataclass
class GPU:
    name: str
    total: int  # bytes
    free: int   # bytes
    backend: str = "cuda"


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _detect_nvidia() -> list[GPU]:
    if not shutil.which("nvidia-smi"):
        return []
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            try:
                gpus.append(GPU(parts[0], int(parts[1]) * MIB, int(parts[2]) * MIB, "cuda"))
            except ValueError:
                continue
    return gpus


def _detect_amd() -> list[GPU]:
    if not shutil.which("rocm-smi"):
        return []
    out = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    gpus = []
    try:
        data = json.loads(out)
        for card, info in sorted(data.items()):
            if not isinstance(info, dict):
                continue
            total = info.get("VRAM Total Memory (B)")
            used = info.get("VRAM Total Used Memory (B)")
            if total is not None:
                total_b = int(total)
                free_b = total_b - int(used or 0)
                gpus.append(GPU(str(card), total_b, free_b, "rocm"))
    except (ValueError, TypeError):
        return []
    return gpus


def _detect_metal() -> list[GPU]:
    if platform.system() != "Darwin":
        return []
    out = _run(["sysctl", "-n", "hw.memsize"])
    if not out:
        return []
    try:
        memsize = int(out.strip())
    except ValueError:
        return []
    # Metal caps a process at recommendedMaxWorkingSetSize, ~75% of unified
    # memory by default — treat that as the "VRAM" budget.
    budget = int(memsize * 0.75)
    return [GPU("Apple unified memory (75% Metal budget)", budget, budget, "metal")]


def detect_gpus() -> list[GPU]:
    for probe in (_detect_nvidia, _detect_amd, _detect_metal):
        gpus = probe()
        if gpus:
            return gpus
    return []


def system_ram() -> tuple[int, int] | None:
    """(total, available) bytes of host RAM, or None if unknown."""
    if platform.system() == "Linux":
        try:
            fields = {}
            with open("/proc/meminfo") as f:
                for line in f:
                    key, _, rest = line.partition(":")
                    fields[key.strip()] = rest
            total = int(fields["MemTotal"].split()[0]) * 1024
            avail = int(fields["MemAvailable"].split()[0]) * 1024
            return total, avail
        except (OSError, KeyError, ValueError, IndexError):
            return None
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                total = int(out.strip())
                return total, total  # "available" not cheaply known on macOS
            except ValueError:
                return None
    return None
