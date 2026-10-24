// Runtime implementation used when MLIR lowers `memref.copy` to `memrefCopy`.

#include <cstdint>
#include <cstring>
#include <vector>

namespace {
struct UnrankedMemRefDescriptor {
  std::int64_t rank;
  void *descriptor;
};

struct StridedMemRefDescriptor {
  void *allocated;
  char *aligned;
  std::int64_t offset;
  std::int64_t sizes[1];
};

struct DynamicMemRef {
  std::int64_t rank;
  char *aligned;
  std::int64_t offset;
  const std::int64_t *sizes;
  const std::int64_t *strides;
};

DynamicMemRef unpack(const UnrankedMemRefDescriptor &unranked) {
  const auto *descriptor =
      static_cast<const StridedMemRefDescriptor *>(unranked.descriptor);
  const std::int64_t *sizes = unranked.rank == 0 ? nullptr : descriptor->sizes;
  return {unranked.rank, descriptor->aligned, descriptor->offset, sizes,
          sizes == nullptr ? nullptr : sizes + unranked.rank};
}
} // namespace

extern "C" void memrefCopy(std::int64_t elementSize, void *source,
                            void *destination) {
  if (elementSize <= 0 || source == nullptr || destination == nullptr)
    return;

  const auto &sourceDescriptor =
      *static_cast<const UnrankedMemRefDescriptor *>(source);
  const auto &destinationDescriptor =
      *static_cast<const UnrankedMemRefDescriptor *>(destination);
  DynamicMemRef src = unpack(sourceDescriptor);
  DynamicMemRef dst = unpack(destinationDescriptor);
  if (src.rank != dst.rank)
    return;
  for (std::int64_t dimension = 0; dimension < src.rank; ++dimension) {
    if (src.sizes[dimension] == 0 ||
        src.sizes[dimension] != dst.sizes[dimension])
      return;
  }

  char *sourceBytes = src.aligned + src.offset * elementSize;
  char *destinationBytes = dst.aligned + dst.offset * elementSize;
  if (src.rank == 0) {
    std::memcpy(destinationBytes, sourceBytes,
                static_cast<std::size_t>(elementSize));
    return;
  }

  std::vector<std::int64_t> indices(src.rank, 0);
  std::int64_t sourceOffset = 0;
  std::int64_t destinationOffset = 0;
  for (;;) {
    std::memcpy(destinationBytes + destinationOffset,
                sourceBytes + sourceOffset,
                static_cast<std::size_t>(elementSize));
    for (std::int64_t axis = src.rank - 1; axis >= 0; --axis) {
      ++indices[axis];
      sourceOffset += src.strides[axis] * elementSize;
      destinationOffset += dst.strides[axis] * elementSize;
      if (indices[axis] != src.sizes[axis])
        break;
      if (axis == 0)
        return;
      indices[axis] = 0;
      sourceOffset -= src.sizes[axis] * src.strides[axis] * elementSize;
      destinationOffset -= dst.sizes[axis] * dst.strides[axis] * elementSize;
    }
  }
}
