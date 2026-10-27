# Loom 2

Loom 2 is a Python/Rust compiler for syntax-guided synthesis over finite
Boolean semantics. Constructors are ordinary Python functions over `Vec` bit
tensors. They are traced into a Rust tensor IR, lowered to packed `i64` word
programs, emitted as MLIR with [high-mlir](../high-mlir) using only upstream dialects, and
compiled through LLVM into a parallel native enumerator.

See [the architecture guide](docs/architecture.md) for each stage.

## Example

```python
import loom2

Strings = loom2.Vec(**{"": bool, "a": bool, "aa": bool})

@loom2.compile
def re_union(a: Strings, b: Strings):
    return a | b
```

`loom2.synth_fun(output, language=..., grammar=..., constraints=...)` builds
an executable that prints the smallest expression satisfying the constraints.

Calling a compiled function on runtime values runs it through the LLVM JIT.
The first call traces it, lowers it with the synthesizer's own pipeline and
compiles it in-process; later calls reuse the code.

```python
value = Strings({"": True, "a": False, "aa": True})
re_union(value, value)        # JIT-compiled on first call
```

`loom2.compile(..., jit=False)` (or `LOOM2_JIT=0`) keeps the Python body;
`jit=True` raises instead of falling back when a function cannot be JIT-ed.

## Building

The default Rust build needs no LLVM:

```console
cargo test
```

Native code generation needs LLVM/MLIR 23. `mlir-sys` locates it through
`$MLIR_SYS_230_PREFIX/bin/llvm-config`; `TABLEGEN_230_PREFIX` must point at
the same prefix. `.cargo/config.toml` sets both to `.llvm` in this
directory, a gitignored prefix: make it a symlink to your installation, or,
if that keeps `llvm-config` elsewhere (vcpkg uses `tools/llvm/`), a
directory whose `bin/` contains a symlink to it. Exported variables take
precedence.

```console
ln -s /path/to/llvm-23 .llvm
cargo test --features llvm
maturin develop --release --features llvm,pyo3/extension-module
python -m unittest discover -s tests
```

Linking the generated executable uses `CC`/`CXX` (default `cc`/`c++`) and
`-lomp`; set `LOOM2_OPENMP_LIBRARY_DIR` at build time if `libomp` is outside
the default search path.

### CUDA

The `cuda` feature adds a GPU backend: every production filter becomes a
CUDA kernel, and the per-nonterminal hash sets are
[cuco](https://github.com/NVIDIA/cuCollections) `static_set`s. It needs the
CUDA toolkit (`CUDA_HOME`, default `/usr/local/cuda`) and the cuco submodule;
`LOOM2_CUDA_ARCH` (such as `75`) picks the GPU architecture, defaulting to
the one `nvidia-smi` reports.

```console
git submodule update --init
cargo test --features cuda
maturin develop --release --features cuda,pyo3/extension-module
python benchmark/run.py Reg+ 0030 --device cuda --run
```

`loom2.synth_fun(..., device="cuda")` selects it from Python; see
[cpp/cuda](cpp/cuda/README.md) for the runtime.

Regenerate the native type stub after changing a PyO3 API:

```console
cargo run --bin generate_stubs
```

## Benchmarks

`benchmark/Reg` (57 regular expressions over `{0,1}`), `benchmark/Reg+`
(322 Regel DeepRegex and StackOverflow regular expressions), and
`benchmark/FO` (306 first-order invariants from 28 Ivy protocol theories),
`benchmark/LTL` (802 samples2LTL lasso-trace problems), and `benchmark/CTL`
(2107 LearnCTL Kripke-structure problems) hold Loom's suites as Python and
JSON data:

| Path                     | Contents                                                  |
| ------------------------ | --------------------------------------------------------- |
| `benchmark.json`         | per benchmark: `theory`, `model`, `positive`, `negative`  |
| `models/<model>.json`    | Reg: strings; FO: structures; LTL: lassos; CTL: Kripke    |
| `theories/<theory>.py`   | Reg/Reg+: grammar, character predicates, `synth_fun`      |
| `theories/<theory>.json` | FO: signature and per-nonterminal productions by kind     |
| `theories/<theory>.json` | LTL/CTL: constructors by kind (`G`, `U`, `AF`, `EU`, ...) |
| `fo.py`, `ltl.py`, `ctl.py` | interpret a theory JSON with the `Vec` API             |
| `regex.py`               | shared regex theory builder (one substring `Literal`)     |

`benchmark/run.py` builds one benchmark of any family:

```console
python benchmark/run.py Reg+ 0030 --run
```

builds the executable `benchmark/build/reg-plus-0030` (with debug info;
`--no-debug-info` drops it), runs it, and prints the expression. The
intermediate stages are discarded unless `--stages DIR` keeps them:
`01-tensor-fn.ir`, `02-fused.ir`, the MLIR/LLVM stages `03-frontend.mlir`
through `07-program.o`, the executable `08-reg-plus-0030`, and
`09-result.txt`. The generated code's line tables point at the Rust code
in `src/lowering/exec` that emitted each operation, so a debugger shows
which emitter line produced the instruction being stepped.
