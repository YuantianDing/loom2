# CPU runtime

These sources provide the native symbols referenced by an object produced by
the `mlir-loom` LLVM backend:

- `i64hashset.cpp` implements the concurrent hash-set C ABI as a preallocated,
  fixed-capacity, lock-free linear-probing table;
- `memref_copy.cpp` implements the helper used by lowered `memref.copy`;
- `main.c` implements FNV hashing and the generic packed-token result printer.

The LLVM backend inserts the executable's `main` function, so the runtime must
not define a second one. Cargo builds the C and C++ sources as separate static
archives when the `llvm` feature is enabled. `lowering::llvm::link_cpu_object`
links both archives with the generated object using `CXX` (or `c++`) and adds
the thread/OpenMP libraries selected by `BackendOptions`.

The vendored `libcuckoo/` headers are no longer used: a growing cuckoo table
can double long before it is full, which made memory use unbounded.

The result printer intentionally prints numeric prefix tokens. Translating
those tokens back to constructor names belongs to a later schema-aware runtime
layer.
