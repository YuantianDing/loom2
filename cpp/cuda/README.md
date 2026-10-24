# CUDA runtime

These sources provide the native symbols referenced by an object produced
for `Device::Cuda` (`loom2.synth_fun(..., device="cuda")`):

- `i64hashset.cuh` defines the per-nonterminal hash set: a
  [cuco](https://github.com/NVIDIA/cuCollections) `static_set<int64_t>` with
  scalar linear probing, so each GPU thread inserts its own candidate. Key 0
  is the empty sentinel and is stored as 1, as in the CPU table;
- `i64hashset_device.cu` is the device half of the hash-set C ABI
  (`_mlir_ciface_mlir_loom_hashset_i64_insert_one`). It is compiled as
  relocatable device code into a fatbin that build.rs embeds in the runtime;
- `runtime.cu` creates the sets (a handle is a device pointer to the set's
  insert ref), provides a host-side insert for filters the inliner folded
  into host code (nullary productions), serves `memref.alloc` from managed
  memory through `_mlir_memref_to_llvm_alloc`, and implements the `mgpu*`
  functions emitted by `gpu-to-llvm`. `mgpuModuleLoadJIT` links each kernel's
  PTX with the embedded fatbin through `cuLink*`, so the kernel's `.extern`
  hash-set calls resolve when the executable starts.

`cuCollections/` is a git submodule pinned to a tested commit; run
`git submodule update --init` (or set `LOOM2_CUCO_INCLUDE_DIR`) before
building with `--features cuda`. cuco uses the CCCL shipped with the CUDA
toolkit (CUDA 13.2 ships CCCL 3.2).

The executable links `main.c` and `memref_copy.cpp` from `cpp/cpu`, this
runtime, `libcudart_static`, and the driver's `libcuda`. The architecture
(`LOOM2_CUDA_ARCH`, such as `75`) is fixed when loom2 is built and is also
the default NVVM target chip of generated kernels.
