#include "i64hashset.hpp"

#include <atomic>
#include <cstdlib>
#include <new>

namespace mlir_loom {

/// Fixed-capacity, lock-free linear-probing set of 64-bit keys.
///
/// The table is allocated once (zero pages are mapped lazily by the OS) and
/// never grows, so the synthesizer's memory use is fixed when the set is
/// created. Slot value 0 means empty; key 0 is stored as 1. Keys are already
/// uniformly mixed FNV hashes, so their low bits index the table directly.
/// When every slot is taken, `insert` reports the key as new without storing
/// it; the caller bounds insertions so this only happens on overshoot.
struct Int64HashSet::Impl {
  explicit Impl(std::size_t requested) {
    std::size_t capacity = 64;
    while (capacity < requested)
      capacity <<= 1;
    mask = capacity - 1;
    slots = static_cast<std::atomic<std::uint64_t> *>(
        std::calloc(capacity, sizeof(std::atomic<std::uint64_t>)));
    if (!slots)
      throw std::bad_alloc();
  }
  ~Impl() { std::free(slots); }

  std::atomic<std::uint64_t> *slots;
  std::size_t mask;
  std::atomic<std::size_t> count{0};
};

Int64HashSet::Int64HashSet(std::size_t initialCapacity)
    : impl_(new Impl(initialCapacity)) {}

Int64HashSet::~Int64HashSet() { delete impl_; }

bool Int64HashSet::insert(std::int64_t key) {
  std::uint64_t value = static_cast<std::uint64_t>(key);
  if (value == 0)
    value = 1;
  std::size_t index = static_cast<std::size_t>(value) & impl_->mask;
  for (std::size_t probe = 0; probe <= impl_->mask; ++probe) {
    std::atomic<std::uint64_t> &slot = impl_->slots[index];
    std::uint64_t current = slot.load(std::memory_order_acquire);
    if (current == value)
      return false;
    if (current == 0) {
      if (slot.compare_exchange_strong(current, value, std::memory_order_acq_rel)) {
        impl_->count.fetch_add(1, std::memory_order_relaxed);
        return true;
      }
      if (current == value)
        return false;
    }
    index = (index + 1) & impl_->mask;
  }
  return true;
}

bool Int64HashSet::contains(std::int64_t key) const {
  std::uint64_t value = static_cast<std::uint64_t>(key);
  if (value == 0)
    value = 1;
  std::size_t index = static_cast<std::size_t>(value) & impl_->mask;
  for (std::size_t probe = 0; probe <= impl_->mask; ++probe) {
    std::uint64_t current = impl_->slots[index].load(std::memory_order_acquire);
    if (current == value)
      return true;
    if (current == 0)
      return false;
    index = (index + 1) & impl_->mask;
  }
  return false;
}

void Int64HashSet::reserve(std::size_t) {
  // Capacity is fixed at creation.
}

std::size_t Int64HashSet::size() const {
  return impl_->count.load(std::memory_order_relaxed);
}
} // namespace mlir_loom

namespace {
mlir_loom::Int64HashSet *decode(std::uint64_t handle) {
  return reinterpret_cast<mlir_loom::Int64HashSet *>(
      static_cast<std::uintptr_t>(handle));
}
} // namespace

extern "C" {
std::uint64_t mlir_loom_hashset_i64_create_handle(std::uint64_t capacity) {
  auto *set = new mlir_loom::Int64HashSet(static_cast<std::size_t>(capacity));
  return static_cast<std::uint64_t>(reinterpret_cast<std::uintptr_t>(set));
}

void mlir_loom_hashset_i64_destroy_handle(std::uint64_t set) {
  delete decode(set);
}

bool mlir_loom_hashset_i64_insert_one(std::uint64_t set, std::int64_t key) {
  return decode(set)->insert(key);
}

bool mlir_loom_hashset_i64_contains_one(std::uint64_t set, std::int64_t key) {
  return decode(set)->contains(key);
}

void mlir_loom_hashset_i64_reserve(std::uint64_t set, std::uint64_t capacity) {
  decode(set)->reserve(static_cast<std::size_t>(capacity));
}

std::uint64_t
_mlir_ciface_mlir_loom_hashset_i64_create_handle(std::uint64_t capacity) {
  return mlir_loom_hashset_i64_create_handle(capacity);
}

void _mlir_ciface_mlir_loom_hashset_i64_destroy_handle(std::uint64_t set) {
  mlir_loom_hashset_i64_destroy_handle(set);
}

bool _mlir_ciface_mlir_loom_hashset_i64_insert_one(std::uint64_t set,
                                                   std::int64_t key) {
  return mlir_loom_hashset_i64_insert_one(set, key);
}

bool _mlir_ciface_mlir_loom_hashset_i64_contains_one(std::uint64_t set,
                                                     std::int64_t key) {
  return mlir_loom_hashset_i64_contains_one(set, key);
}

void _mlir_ciface_mlir_loom_hashset_i64_reserve(std::uint64_t set,
                                                std::uint64_t capacity) {
  mlir_loom_hashset_i64_reserve(set, capacity);
}
}
