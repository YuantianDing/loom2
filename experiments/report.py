#!/usr/bin/env python3
"""Summarize an experiment database: status counts, times and value bits per set."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

import config
from func import canonical_json, value_bits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", nargs="?", default="results.db")
    parser.add_argument("--all-configs", action="store_true",
                        help="include results of configurations no longer in config.py")
    parser.add_argument("--bits", action="store_true",
                        help="also list the bits of one value per nonterminal for every benchmark")
    arguments = parser.parse_args()
    current = {(method, canonical_json(configs)) for method, configs in config.METHOD_CONFIGS.items()}
    with sqlite3.connect(arguments.database) as connection:
        rows = [
            (bench_set, bench_id, method, results)
            for bench_set, bench_id, method, configs, results in connection.execute(
                "SELECT bench_set, bench_id, method, method_configs, results FROM instances"
            )
            if arguments.all_configs or (method, configs) in current
        ]
    statuses: dict = defaultdict(Counter)
    seconds: dict = defaultdict(list)
    # Results cached before ``value_bits`` was recorded get it recomputed.
    bits: dict = {}
    for bench_set, bench_id, method, results in rows:
        key = (bench_set, method)
        if results is None:
            statuses[key]["running"] += 1
            continue
        result = json.loads(results)
        if (bench_set, bench_id) not in bits:
            bits[bench_set, bench_id] = result.get("value_bits") or value_bits(bench_set, bench_id)
        statuses[key][result["status"]] += 1
        if result["status"] == "done":
            seconds[key].append(result["seconds"])
    for key in sorted(statuses):
        times = sorted(seconds[key])
        median = times[len(times) // 2] if times else float("nan")
        counts = ", ".join(f"{status}={count}" for status, count in sorted(statuses[key].items()))
        # Bits of one start-symbol value, over the set's benchmarks.
        widths = sorted(next(iter(entry.values())) for (bench_set, _), entry in bits.items() if bench_set == key[0])
        width = f"min {widths[0]} / median {widths[len(widths) // 2]} / max {widths[-1]}" if widths else "n/a"
        print(f"{key[0]:>9} {key[1]}: {counts}; solved median {median:.1f}s; value bits {width}")
    if arguments.bits:
        for (bench_set, bench_id), entry in sorted(bits.items()):
            start, width = next(iter(entry.items()))
            print(f"{bench_set}/{bench_id}: {start}={width}, widest of {len(entry)} nonterminals={max(entry.values())}")


if __name__ == "__main__":
    main()
