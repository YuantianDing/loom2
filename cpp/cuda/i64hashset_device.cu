// Device half of the hash-set C ABI, compiled as relocatable device code
// (`nvcc -rdc=true -fatbin`) and embedded into the runtime. Generated kernels
// are PTX that declares these functions `.extern`; `mgpuModuleLoadJIT` in
// `runtime.cu` links every kernel against this fatbin.

#include "i64hashset.cuh"

extern "C" {
// `convert-gpu-to-nvvm` calls external functions through the
// `_mlir_ciface_` name; the plain name is kept for hand-written kernels.
__device__ bool _mlir_ciface_mlir_loom_hashset_i64_insert_one(std::uint64_t set,
                                                              std::int64_t key) {
  return mlir_loom::gpu::insert(set, key);
}

__device__ bool mlir_loom_hashset_i64_insert_one(std::uint64_t set,
                                                 std::int64_t key) {
  return mlir_loom::gpu::insert(set, key);
}
}
