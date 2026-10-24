#!/usr/bin/env python3
"""Summarize an experiment database: status counts and times per set."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict

import config
from func import canonical_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", nargs="?", default="results.db")
    parser.add_argument("--all-configs", action="store_true",
                        help="include results of configurations no longer in config.py")
    arguments = parser.parse_args()
    current = {(method, canonical_json(configs)) for method, configs in config.METHOD_CONFIGS.items()}
    with sqlite3.connect(arguments.database) as connection:
        rows = [
            (bench_set, method, results)
            for bench_set, method, configs, results in connection.execute(
                "SELECT bench_set, method, method_configs, results FROM instances"
            )
            if arguments.all_configs or (method, configs) in current
        ]
    statuses: dict = defaultdict(Counter)
    seconds: dict = defaultdict(list)
    for bench_set, method, results in rows:
        key = (bench_set, method)
        if results is None:
            statuses[key]["running"] += 1
            continue
        result = json.loads(results)
        statuses[key][result["status"]] += 1
        if result["status"] == "done":
            seconds[key].append(result["seconds"])
    for key in sorted(statuses):
        times = sorted(seconds[key])
        median = times[len(times) // 2] if times else float("nan")
        counts = ", ".join(f"{status}={count}" for status, count in sorted(statuses[key].items()))
        print(f"{key[0]:>9} {key[1]}: {counts}; solved median {median:.1f}s")


if __name__ == "__main__":
    main()
