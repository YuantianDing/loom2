#ifndef MLIRLOOM_CUDA_I64HASHSET_CUH
#define MLIRLOOM_CUDA_I64HASHSET_CUH

#include <cuco/static_set.cuh>

#include <cstdint>

namespace mlir_loom::gpu {

/// Fixed-capacity concurrent set of 64-bit hashes: a `cuco::static_set` with
/// scalar (non-cooperative) linear probing, so every GPU thread inserts its
/// own candidate. Key 0 is the empty sentinel and is stored as 1, exactly as
/// in the CPU table.
using Set = cuco::static_set<std::int64_t, cuco::extent<std::size_t>,
                             cuda::thread_scope_device,
                             cuda::std::equal_to<std::int64_t>,
                             cuco::linear_probing<1, cuco::default_hash_function<std::int64_t>>>;

/// Device-side insert view of a `Set`. Hash-set handles point to one of these
/// in device memory.
using InsertRef = decltype(std::declval<Set &>().ref(cuco::insert));

inline constexpr std::int64_t empty_key = 0;

__device__ inline bool insert(std::uint64_t handle, std::int64_t key) {
  auto *ref = reinterpret_cast<InsertRef *>(static_cast<std::uintptr_t>(handle));
  return ref->insert(key == empty_key ? 1 : key);
}

} // namespace mlir_loom::gpu

#endif
