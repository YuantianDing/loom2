#!/usr/bin/env python3
"""Populate an experiment database by calling ``func.instance`` for every input.

    nohup python experiments/run.py experiments/results.db > experiments/run.log 2>&1 &

Finished instances are cached, so rerunning resumes where a run stopped.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import config
from func import instance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", nargs="?", default="results.db")
    parser.add_argument("--set", action="append", choices=sorted(config.BENCH_SETS),
                        help="limit to these benchmark sets (repeatable)")
    arguments = parser.parse_args()

    with sqlite3.connect(Path(arguments.database)) as connection:
        for bench_set, bench_id, method, method_configs in config.instances():
            if arguments.set and bench_set not in arguments.set:
                continue
            result = instance(connection, bench_set, bench_id, method, method_configs)
            detail = result.get("solution") or result.get("error") or ""
            print(f"{bench_set}/{bench_id} {method}: {result['status']} "
                  f"{result.get('seconds', 0):.1f}s {detail}", flush=True)


if __name__ == "__main__":
    main()
