#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

// Runtime hooks referenced by the object emitted from frontend MLIR. The
// backend inserts `main`; this file deliberately does not define another one.

uint64_t compute_fnv_hash(const uint64_t *data, uint64_t length) {
    const uint64_t FNV_OFFSET_BASIS = UINT64_C(0xcbf29ce484222325);
    const uint64_t FNV_PRIME = UINT64_C(0x100000001b3);

    uint64_t hash = FNV_OFFSET_BASIS;
    for (uint64_t i = 0; i < length; ++i) {
        hash ^= data[i];
        hash *= FNV_PRIME;
    }
    return hash;
}

void mlir_loom_print_result(uint64_t *allocated, uint64_t *aligned,
                            intptr_t offset, intptr_t size, intptr_t stride,
                            intptr_t token_bits) {
    (void)allocated;
    fputs("result:", stdout);
    if (token_bits <= 0 || token_bits > 64 || size <= 0) {
        fputs(" <malformed>\n", stdout);
        return;
    }
    int printed = 0;
    for (intptr_t bit = 0; bit + token_bits <= size * 64; bit += token_bits) {
        intptr_t word = bit / 64;
        intptr_t shift = bit % 64;
        uint64_t token = aligned[offset + word * stride] >> shift;
        if (shift != 0 && token_bits > 64 - shift && word + 1 < size)
            token |= aligned[offset + (word + 1) * stride] << (64 - shift);
        if (token_bits != 64)
            token &= (UINT64_C(1) << token_bits) - 1;
        if (token == 0)
            break;
        printf(" %" PRIu64, token);
        printed = 1;
    }
    if (!printed)
        fputs(" <none>", stdout);
    fputc('\n', stdout);
}
