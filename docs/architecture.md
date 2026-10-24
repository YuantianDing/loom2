# Architecture

Loom 2 traces Python `Vec` programs into a Rust tensor IR and compiles a
syntax-guided synthesis problem over them into a native, parallel
enumerator. Files under `src/.old/` are historical and are not built.

## Pipeline

```text
Python Vec callbacks           loom2/lang, loom2/tracing
    -> TensorFn graphs         src/compile          01-tensor-fn.ir
    -> fused graphs            src/compile/pass     02-fused.ir
    -> vector loop programs    src/lowering/loops   03-loops.ir
    -> synthesis module        src/lowering/exec    04-frontend.mlir   (Melior)
    -> CPU MLIR                src/lowering/llvm    05-cpu.mlir
    -> LLVM-dialect MLIR                            06-llvm.mlir
    -> LLVM IR                                      07-llvm.ll
    -> native object                                08-program.o
    -> executable              + cpp/cpu runtime
```

With `device="cuda"`, `05-gpu.mlir` replaces `05-cpu.mlir`, the LLVM-dialect
module carries one PTX `gpu.binary` per kernel, and the executable links the
`cpp/cuda` runtime (see [CUDA backend](#cuda-backend)).

`loom2.synth_fun` drives the whole pipeline through one native call,
`build_synthesizer`, and every stage is written to the artifact directory.
Only upstream MLIR dialects (`func`, `arith`, `scf`, `memref`, `vector`, and
`omp`/`llvm` during lowering) and upstream passes are used.

## Source map

```text
loom2/
├── lang/vec.py            Vec schemas, pytree indices, runtime values
├── tracing/               trace callbacks into a native TensorFnBuilder
└── compile/               loom2.compile and loom2.synth_fun
src/
├── lib.rs, bindings.rs    PyO3 module (build_synthesizer with `llvm`)
├── grammar.rs             validated grammar and constructor tokens
├── problem.rs             SynthFun: functions, grammar, traced constraint
├── compile/
│   ├── tensor.rs          flat dense/jagged tensor storage
│   ├── ir.rs              TensorType and TensorFn
│   ├── expr.rs            tensor-expression graph
│   ├── typing.rs          layout and support inference
│   ├── pass/fusion.rs     permute/reduce fusion into gather and gather2
│   ├── builder.rs         TensorFnBuilder used by tracing
│   └── display.rs         stable textual TensorFn form
└── lowering/
    ├── circuit/           permutation circuits (currently unused)
    ├── packed.rs          row/lane reference evaluator
    ├── loops.rs           TensorFn -> LoopProgram (vector loops)
    ├── exec/              Melior emission (`mlir` feature)
    │   ├── functions.rs   define_fun_* and the constraint
    │   ├── hash_filter.rs compute_hash and per-production filters
    │   └── synthesis.rs   synthesize, main, result printer
    ├── llvm.rs            pass pipelines, LLVM codegen, linking (`llvm`)
    ├── jit.rs             in-process LLVM JIT of one TensorFn (`llvm`)
    └── pipeline.rs        end-to-end build with every stage (`llvm`)
cpp/cpu/                   hash set, memref copy, and printer runtime
cpp/cuda/                  cuco hash set, managed allocation, mgpu* runtime
benchmark/<family>/        benchmark.json, models/, theories/, regex.py, run.py
```

## Frontend

`Vec` describes regular, jagged, and labeled bit tensors. While tracing, one
`Vec(bool)` is one packed expression; Python keeps only opaque handles.

| Python                                 | TensorExpr                              |
| -------------------------------------- | --------------------------------------- |
| parameter                              | `Input`                                 |
| `Vec(...)` literal                     | `Constant`                              |
| `a & b`, `a \| b`, `a ^ b`, `~a`        | `BVLogical` (nested, shared, never flattened) |
| `a[k]`, `a[Vec.pytree(...)]`           | `BVJaggedPermute`                       |
| `.forall()`, `.exists()`, `.xorsum()`  | `expand_reduce.op %a [rows, reduce(n)]` |
| `a[index].exists()`                    | `BVGather` (after fusion)               |
| `(a[p] & b[q]).exists()`               | `BVGather2`, `gather2.or.and` (after fusion) |
| `a[k] = v`, `a[""] = True`             | `a ^ ((a ^ v) & mask)`                  |

A selection's rows are the groups of its last dimension and each row's
elements are lanes; rank-1 selections have one lane per row, which is the flat
layout. Pytree rows may be empty: an empty row has no lanes and reduces to the
operator identity. Recursion is written as explicit length-staged loops (see
`re.*` in `benchmark/Reg+/regex.py`), so traced graphs are acyclic. Handles share
their `Rc` node and cache their type, so tracing is linear in graph size.

## Types

A `TensorType` has a row count and a support mask with one `u64` per
`(word group, lane)`: set bits may be one, all others are zero. Packed values
use the same lane-major layout. Support follows the logic, so constants narrow
or widen it:

```text
a & c   support(a) ∩ c        a | c, a ^ c   support(a) ∪ c
permute support of each source bit       reduce  OR/XOR: any lane; AND: all rows
gather  exact, using row lengths (an empty AND row is one)
```

The textual form is `tensor<GROUPSxLANESxu64, WIDTHbits/REST[, N live]>`.

## Fusion

`compile::pass::fuse` runs before lowering. It composes a permute of a
single-lane permute into one permute, turns `reduce(op, permute(x))` into
`BVGather`, and `reduce(op, inner(permute(a), permute(b)))` into `BVGather2`,
where `reduce` is a `BTExpandReduce`.

`BTExpandReduce` gives every dimension an extent, e.g.
`expand_reduce.or %a [4, expand(4), reduce(4), 4] : bits[64]`: the input is
read row-major with its batch and reduce extents, the output with its batch and
expand extents, and reduced dimensions fold with the node's operator. On a flat
input this is a dense reshape, which fusion turns into a `BVGather`; on a
jagged input, `[rows, reduce(lanes)]` folds each row's lanes (missing lanes are
the identity).
Tracing itself emits only permutes, logic, and reductions.

## Loop programs

`loops::lower_loop_program` turns a fused `TensorFn` into a `LoopProgram`:
vector variables (inputs, temps, outputs, each with a word count and a live
range) and whole-vector operations (`Load`, `Store`, `Broadcast`, `Constant`,
`Splat`, `And`/`Or`/`Xor`, `Rotl`, `Copy`, `ForLoop`). A `ForLoop` body is
the live range of its induction variable. Programs use 32-bit words when all
values fit in 32 bits and 64-bit words otherwise.

Values are lane-major: `lanes x groups` words. Bit movement is table driven:
a permute loops over steps, broadcasting one source word, masking it, and
rotating every destination word by its own amount; `BVGather`/`BVGather2`
loop over lanes and fold each lane into a `groups`-word accumulator while it
is produced. Code size therefore does not depend on the number of bits, which
keeps LLVM compilation fast; tests check this and compare every program with
the row/lane reference evaluator in `packed.rs`.

## Executable model

Pools live in one preallocated arena. A row is
`[semantic words | preorder token bits]`. For every size and nonterminal,
`synthesize` calls one `hash_filter_*` per rule and child-size composition.
Each filter is an `scf.parallel` loop that evaluates the constructor, checks
the per-nonterminal hash set before claiming a row, and records the first
candidate whose constraint holds in `%solution`/`%found` for early exit.
`main` prints the solution with a generated, schema-aware printer.

## Backend

Constructors take and return `vector<Nxi64>` (32-bit programs bitcast to
`vector<2Nxi32>` inside); filters read child values with
`vector.transfer_read` and store results with `vector.transfer_write`. Step
tables live in private `memref.global` constants read with `vector.load`.

`lower_to_cpu_mlir` inlines constructors into filters (filters stay
outlined), canonicalizes, and maps `scf.parallel` to OpenMP.
`lower_to_llvm_mlir` lowers memrefs before `scf-to-cf`, so the
`memref.alloca_scope` created by the OpenMP conversion is still a single
block. LLVM IR comes from `mlirTranslateModuleToLLVMIR`; the object is
optimized and emitted through the LLVM C API, then linked with `cpp/cpu` and
`libomp`.

## CUDA backend

`BackendOptions::device = Device::Cuda { chip, block_size }` reuses the
frontend module unchanged and only swaps the lowering, again with upstream
passes alone:

```text
inline, canonicalize, cse
func.func(scf-parallel-loop-tiling{block_size}, gpu-map-parallel-loops,
          convert-parallel-loops-to-gpu, gpu-launch-sink-index-computations)
gpu-kernel-outlining                                          05-gpu.mlir
nvvm-attach-target{chip}, gpu.module(convert-gpu-to-nvvm), host *-to-llvm,
finalize-memref-to-llvm{use-generic-functions}, gpu-to-llvm,
gpu-module-to-binary{format=isa}                              06-llvm.mlir
```

Each filter's `scf.parallel` becomes one kernel: the tile loop maps to
blocks and the intra-tile loop to threads, so one thread evaluates one child
combination. Everything the body touches is visible to the device: the
filter's row cursor is a `memref<1xi64>` argument shared by all filter calls
(not a stack slot), and `memref.alloc` lowers to
`_mlir_memref_to_llvm_alloc`, which `cpp/cuda` serves from managed memory, so
the arena, `%solution`, `%found`, and the cursor need no copies. The kernel
calls `mlir_loom_hashset_i64_insert_one` as an external device function;
`mgpuModuleLoadJIT` links each kernel's PTX with an embedded relocatable
fatbin of the cuco insert when the executable starts. The inliner folds
single-iteration loops, so nullary filters stay host code and use the
runtime's host-side insert, a one-thread kernel. Kernel launches are
synchronous, so the host reads the cursor and `%found` between filters as on
the CPU.

Symbols are named with `symbol_name`, which keeps only `[A-Za-z0-9_]`, since
kernel names must be valid PTX identifiers.

## JIT

`loom2.compile` functions called on `VecValue` arguments run through
`lowering::jit`. The traced `TensorFn` is fused, lowered to a loop program,
and emitted as `define_fun_<name>` exactly as in the synthesizer, plus a
public `loom_jit_entry(%out: !llvm.ptr, %in: !llvm.ptr)` that loads each
argument's `vector<Nxi64>` from one concatenated word buffer, calls the
constructor, and stores the result. After `lower_to_cpu_mlir` and
`lower_to_llvm_mlir` (without OpenMP), MLIR's ExecutionEngine compiles the
module with LLVM ORC for the host CPU. Calls made while tracing run the
Python body, so constructors still inline into their callers' graphs.

`tests/test_jit.py` checks every operator and every constructor of one
benchmark per theory against the Python body; with `llvm`, the loop-program
tests also JIT each function at -O0 and -O3.

## Build boundaries

| Feature   | Adds                                         | Needs           |
| --------- | -------------------------------------------- | --------------- |
| (default) | IR, tracing builder, fusion, loop programs | Rust only       |
| `mlir`    | Melior emission                              | LLVM/MLIR 23    |
| `llvm`    | codegen, linking, `build_synthesizer`, `JitFunction`, `cpp/cpu` runtime | LLVM/MLIR 23, C/C++ compiler, libomp |
| `cuda`    | `Device::Cuda`, `cpp/cuda` runtime          | `llvm` + CUDA toolkit (nvcc), cuco submodule, an NVIDIA GPU to run |

`mlir-sys` finds LLVM through `MLIR_SYS_230_PREFIX/bin/llvm-config`.
