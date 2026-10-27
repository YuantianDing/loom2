"""Execute one benchmark instance, with SQLite's ``instances`` table as cache."""

from __future__ import annotations

import importlib
import json
import os
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS instances (
    bench_set TEXT NOT NULL,
    bench_id TEXT NOT NULL,
    method TEXT NOT NULL,
    method_configs JSONB NOT NULL,
    results JSONB,
    PRIMARY KEY (bench_set, bench_id, method, method_configs)
)
"""

if str(config.LOOM2) not in sys.path:
    sys.path.insert(0, str(config.LOOM2))


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value or ""


def kill_group(process: subprocess.Popen) -> None:
    """SIGKILL the process group of `process` (its own: see `run`), so no
    child it started outlives the run; a group that is gone is ignored."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def matching_processes(marker: str) -> list[tuple[int, str]]:
    """`(pid, command line)` of every live process with `marker` in its name
    or arguments, except this process and its ancestors."""
    excluded = set()
    pid = os.getpid()
    while pid > 1:
        excluded.add(pid)
        try:
            pid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            state = (entry / "stat").read_text().rsplit(")", 1)[1].split()[0]
            name = (entry / "comm").read_text().strip()
            arguments = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            continue  # exited while being read
        if state != "Z" and (marker in name or marker in arguments):
            found.append((int(entry.name), arguments or name))
    return found


def wait_for_marker(process: subprocess.Popen, marker: str) -> None:
    """Spin until no process has `marker` in its name or arguments,
    SIGKILLing `process`'s group again on every pass. Processes outside the
    group are waited for, not killed; they are reported every 10 seconds."""
    started = reported = time.monotonic()
    while True:
        remaining = matching_processes(marker)
        if not remaining:
            return
        kill_group(process)
        if time.monotonic() - reported >= 10:
            reported = time.monotonic()
            waited = reported - started
            print(f"waiting {waited:.0f}s for processes with {marker!r}: {remaining}", file=sys.stderr, flush=True)
        time.sleep(0.01)


def run(command: list[str], timeout: float, environment: dict[str, str], marker: str) -> dict[str, Any]:
    """Run `command` in a new process group, which is SIGKILLed afterwards;
    then wait until no process has `marker` (the benchmark id) in its name
    or arguments (see `wait_for_marker`).

    Output goes to temporary files rather than pipes, so the run ends when
    `command` exits even if a child it left behind still holds them open.
    """
    started = time.monotonic()
    with tempfile.TemporaryFile("w+") as stdout, tempfile.TemporaryFile("w+") as stderr:
        process = subprocess.Popen(
            command,
            cwd=config.LOOM2,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            returncode = None
        finally:
            kill_group(process)
            process.wait()
            wait_for_marker(process, marker)
        stdout.seek(0)
        stderr.seek(0)
        output = {"stdout": stdout.read(), "stderr": stderr.read()}
    if returncode is None:
        status = "timeout"
    else:
        status = "done" if returncode == 0 else "error"
    return {"status": status, "seconds": time.monotonic() - started, "returncode": returncode, **output}


def family_directory(bench_set: str, bench_id: str) -> Path:
    try:
        directory = config.BENCH_SETS[bench_set]
    except KeyError as error:
        raise ValueError(f"unknown benchmark set: {bench_set}") from error
    if bench_id not in config.benchmark_ids(bench_set):
        raise ValueError(f"unknown benchmark: {bench_set}/{bench_id}")
    return directory


def parse_expression(source: str):
    tokens = re.findall(r"[^\s(),]+|[(),]", source)
    position = 0

    def expression():
        nonlocal position
        name = tokens[position]
        position += 1
        arguments = []
        if position < len(tokens) and tokens[position] == "(":
            position += 1
            while True:
                arguments.append(expression())
                position += 1
                if tokens[position - 1] == ")":
                    break
        return name, arguments

    tree = expression()
    if position != len(tokens):
        raise ValueError(f"trailing tokens in {source!r}")
    return tree


def load_theory(directory: Path, bench_id: str):
    """Instantiate a benchmark's theory with the Python benchmark code.
    Returns ``(theory, positives, negatives)``, the examples as labels."""
    metadata = json.loads((directory / "benchmark.json").read_text())[bench_id]
    models = {}
    for file in metadata.get("models", [metadata.get("model")]):
        models.update(json.loads((directory / f"models/{file}.json").read_text()))
    family = directory / f"{directory.name.lower()}.py"
    if family.is_file():
        # FO/LTL/CTL: JSON theory over models; check the start bit per model.
        module = importlib.import_module(f"benchmark.{directory.name}.{family.stem}")
        definition = json.loads((directory / f"theories/{metadata['theory']}.json").read_text())
        names = list(dict.fromkeys(metadata["positive"] + metadata["negative"]))
        theory = module.theory(definition, {name: models[name] for name in names})
        positives = {theory.label(name) for name in metadata["positive"]}
        negatives = {theory.label(name) for name in metadata["negative"]}
    else:
        module = importlib.import_module(f"benchmark.{directory.name}.theories.{metadata['theory']}")
        positives = {models[name] for name in metadata["positive"]}
        negatives = {models[name] for name in metadata["negative"]}
        theory = module.theory(positives, negatives)
    return theory, positives, negatives


def value_bits(bench_set: str, bench_id: str) -> dict[str, int]:
    """Bits in one value of each grammar nonterminal, as ``synth_fun``
    computes them; the first entry is the start symbol."""
    theory, _, _ = load_theory(family_directory(bench_set, bench_id), bench_id)
    languages = getattr(theory, "languages", {})
    return {symbol: languages.get(symbol, theory.language).bit_width() for symbol in theory.grammar}


def verify(directory: Path, bench_id: str, output: str) -> dict[str, Any]:
    """Re-evaluate the printed expression with the Python theory and check
    every example. Returns ``solution``, ``size`` and ``verified``."""
    match = re.match(r"\S+ = (.+)$", output.strip())
    if not match:
        return {"solution": None, "size": None, "verified": None}
    theory, positives, negatives = load_theory(directory, bench_id)

    def evaluate(tree):
        name, arguments = tree
        return theory.constructors[name](*(evaluate(argument) for argument in arguments))

    tree = parse_expression(match.group(1))
    value = evaluate(tree)
    verified = all(value[s] is True for s in positives) and all(value[s] is False for s in negatives)
    size = len(re.findall(r"[^\s(),]+", match.group(1)))
    return {"solution": match.group(1), "size": size, "verified": verified}


def compute_loom2_openmp(directory: Path, bench_id: str, method_configs: Mapping[str, Any]) -> dict[str, Any]:
    """Compile with ``benchmark/run.py`` in a subprocess, run the
    synthesizer, and verify its answer."""
    environment = dict(os.environ)
    threads = method_configs.get("omp_threads")
    if threads is not None:
        environment["OMP_NUM_THREADS"] = str(threads)
    pool = config.pool_bytes(method_configs["max_pool_bytes"])
    timeout = float(method_configs["timeout_seconds"])
    with tempfile.TemporaryDirectory(prefix="loom2-experiment-") as temporary:
        command = [
            sys.executable, str(config.BENCHMARK / "run.py"), directory.name, bench_id,
            "--stages", temporary,
            "--max-size", str(method_configs["max_size"]),
            "--max-pool-bytes", str(pool),
            "--initial-hash-capacity", str(method_configs["initial_hash_capacity"]),
            "-O", str(method_configs["opt_level"]),
            "--openmp" if method_configs.get("omp", True) else "--no-openmp",
        ]
        compiled = run(command, timeout, environment, bench_id)
        executables = sorted(Path(temporary).glob("08-*"))
        executed = (
            run([str(executables[0])], timeout, environment, bench_id)
            if compiled["status"] == "done" and executables
            else None
        )
        sizes = {path.name: path.stat().st_size for path in Path(temporary).iterdir() if path.is_file()}
    outcome = executed or compiled
    result = {
        "status": outcome["status"],
        "seconds": compiled["seconds"] + (executed["seconds"] if executed else 0.0),
        "returncode": outcome["returncode"],
        "compile_stdout": compiled["stdout"],
        "compile_stderr": compiled["stderr"],
        "stdout": outcome["stdout"],
        "stderr": outcome["stderr"],
        "compile_seconds": compiled["seconds"],
        "execute_seconds": executed["seconds"] if executed else None,
        "max_pool_bytes": pool,
        "artifact_bytes": sizes,
    }
    if executed and executed["status"] == "done":
        result.update(verify(directory, bench_id, executed["stdout"]))
        if "no solution" in executed["stdout"]:
            result["status"] = "unsolved"
        elif result["verified"] is False:
            result["status"] = "wrong"
    return result


def compute(bench_set: str, bench_id: str, method: str, method_configs: Mapping[str, Any]) -> dict[str, Any]:
    """Map an uncached method-specific instance description to its result."""
    directory = family_directory(bench_set, bench_id)
    if method == "loom2.openmp":
        return {
            **compute_loom2_openmp(directory, bench_id, method_configs),
            "value_bits": value_bits(bench_set, bench_id),
        }
    raise ValueError(f"unknown method: {method}")


def instance(
    connection: sqlite3.Connection,
    bench_set: str,
    bench_id: str,
    method: str,
    method_configs: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a cached result or run and cache this exact experiment instance.

    The only cache key is ``(bench_set, bench_id, method, method_configs)``.
    An inserted row has ``results = NULL`` until this function finishes it.
    """
    connection.execute(SCHEMA)
    configs = canonical_json(method_configs)
    key = (bench_set, bench_id, method, configs)
    connection.execute(
        "INSERT OR IGNORE INTO instances (bench_set, bench_id, method, method_configs) VALUES (?, ?, ?, ?)",
        key,
    )
    connection.commit()
    row = connection.execute(
        "SELECT results FROM instances WHERE bench_set = ? AND bench_id = ? AND method = ? AND method_configs = ?",
        key,
    ).fetchone()
    if row is not None and row[0] is not None:
        return json.loads(row[0])

    try:
        result = compute(bench_set, bench_id, method, method_configs)
    except Exception as error:
        result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    connection.execute(
        "UPDATE instances SET results = ? WHERE bench_set = ? AND bench_id = ? AND method = ? AND method_configs = ?",
        (canonical_json(result), *key),
    )
    connection.commit()
    return result
