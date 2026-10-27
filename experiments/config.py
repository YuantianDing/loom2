"""Static inputs for the loom2 experiment sweep."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

LOOM2 = Path(__file__).resolve().parents[1]
BENCHMARK = LOOM2 / "benchmark"

# Benchmark families under ``benchmark/``; ids come from each benchmark.json.
BENCH_SETS = {
    "reg": BENCHMARK / "Reg",
    "reg-plus": BENCHMARK / "Reg+",
    "fo": BENCHMARK / "FO",
    "ltl": BENCHMARK / "LTL",
    "ctl": BENCHMARK / "CTL",
}


def available_memory() -> int:
    """``MemAvailable`` from ``/proc/meminfo``, in bytes."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")


def pool_bytes(setting: object) -> int:
    """Resolve ``max_pool_bytes``: an integer, or ``"3/4 available"`` for
    three quarters of the memory available when the instance starts."""
    if isinstance(setting, int):
        return setting
    if setting == "3/4 available":
        return available_memory() * 3 // 4
    raise ValueError(f"unknown max_pool_bytes setting {setting!r}")


# A method's configuration is method-specific and becomes part of the SQLite
# cache key. ``max_pool_bytes`` is resolved per instance by ``pool_bytes``;
# the resolved byte count is stored in each result.
LOOM2_OPENMP_CONFIG = {
    "max_size": 20,
    "timeout_seconds": 600.0,
    "opt_level": 2,
    "max_pool_bytes": "3/4 available",
    "initial_hash_capacity": 1 << 29,
    "omp": True,
    "omp_threads": None,
}

LOOM2_CPU1_CONFIG = {
    "max_size": 20,
    "timeout_seconds": 600.0,
    "opt_level": 2,
    "max_pool_bytes": "3/4 available",
    "initial_hash_capacity": 1 << 29,
    "omp": False,
    "omp_threads": None,
}

METHOD_CONFIGS = {
    "loom2.openmp": LOOM2_OPENMP_CONFIG,
    "loom2.cpu.1": LOOM2_CPU1_CONFIG,
}


def benchmark_ids(bench_set: str) -> list[str]:
    return sorted(json.loads((BENCH_SETS[bench_set] / "benchmark.json").read_text()))


def instances() -> Iterator[tuple[str, str, str, dict[str, object]]]:
    """Yield every benchmark and configured method for ``run.py``."""
    for bench_set in BENCH_SETS:
        for bench_id in benchmark_ids(bench_set):
            for method, method_configs in METHOD_CONFIGS.items():
                yield bench_set, bench_id, method, method_configs
