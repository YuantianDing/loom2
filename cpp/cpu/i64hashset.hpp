

#ifndef MLIRLOOM_COMPILER_RUNTIME_H
#define MLIRLOOM_COMPILER_RUNTIME_H

#include <cstddef>
#include <cstdint>

namespace mlir_loom {

/// Thread-safe, fixed-capacity set of 64-bit hashes used by the generated
/// production filters (`mlir_loom_hashset_i64_*`).
class Int64HashSet {
public:
  explicit Int64HashSet(std::size_t initialCapacity = 1024);
  ~Int64HashSet();
  Int64HashSet(const Int64HashSet &) = delete;
  Int64HashSet &operator=(const Int64HashSet &) = delete;

  bool insert(std::int64_t key);
  bool contains(std::int64_t key) const;
  void reserve(std::size_t capacity);
  std::size_t size() const;

private:
  struct Impl;
  Impl *impl_;
};

} // namespace mlir_loom

extern "C" {
std::uint64_t mlir_loom_hashset_i64_create_handle(std::uint64_t capacity);
void mlir_loom_hashset_i64_destroy_handle(std::uint64_t set);
bool mlir_loom_hashset_i64_insert_one(std::uint64_t set, std::int64_t key);
bool mlir_loom_hashset_i64_contains_one(std::uint64_t set, std::int64_t key);
void mlir_loom_hashset_i64_reserve(std::uint64_t set, std::uint64_t capacity);

// MLIR's `llvm.emit_c_interface` convention prefixes calls with
// `_mlir_ciface_`.  Keep the plain C ABI above useful to embedders and expose
// these forwarding symbols for objects produced by the MLIR-Loom backend.
std::uint64_t
_mlir_ciface_mlir_loom_hashset_i64_create_handle(std::uint64_t capacity);
void _mlir_ciface_mlir_loom_hashset_i64_destroy_handle(std::uint64_t set);
bool _mlir_ciface_mlir_loom_hashset_i64_insert_one(std::uint64_t set,
                                                   std::int64_t key);
bool _mlir_ciface_mlir_loom_hashset_i64_contains_one(std::uint64_t set,
                                                     std::int64_t key);
void _mlir_ciface_mlir_loom_hashset_i64_reserve(std::uint64_t set,
                                                std::uint64_t capacity);
}

#endif
