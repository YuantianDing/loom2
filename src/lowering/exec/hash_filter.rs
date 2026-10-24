//! Fill the hash-related slots of the synthesis module.
//!
//! ```text
//! func.func private @compute_hash_w<N>(%w0: i64, ..., %wN-1: i64) -> i64
//!   // FNV-1a over the canonical packed semantic words
//!
//! func.func private @hash_filter_<nonterminal>_<constructor>_rule<rule>(
//!     %arena: memref<?xi64>, %output_offset: index,
//!     %output_capacity: index, %output_row_words: index, %start: index,
//!     %cursor: memref<1xi64>,          // shared by every filter call
//!     %child0: memref<?x?xi64, strided<[?, 1], offset: ?>>, ...,
//!     %child0_expression_bits: index, ...,
//!     %seen: i64,                      // cpp/cpu/i64hashset handle
//!     %solution: memref<?xi64>, %found: memref<1xi8>
//! ) -> memref<?x?xi64, strided<[?, 1], offset: ?>> {
//!   %output = memref.reinterpret_cast %arena at %output_offset
//!   %cursor[0] = %start
//!   scf.parallel (%linear) = (0) to (product(rows(%child_i))) step (1) {
//!     scf.if atomic_load(%found) == 0 {
//!       %children = mixed_radix_select(%linear, %child0, ...)
//!       %value = func.call @define_fun_<constructor>(child value words...)
//!       %hash = func.call @compute_hash_w<N>(%value)
//!       scf.if func.call @mlir_loom_hashset_i64_insert_one(%seen, %hash) {
//!         %slot = atomic_fetch_add(%cursor, 1)
//!         scf.if %slot < %output_capacity {
//!           // row = [semantic words | token bits | child expression bits...]
//!           store_value_and_expression(%output[%slot], %value, %children...)
//!           record_first_solution_if_constraints_match(%solution, %found)
//!         }
//!       }
//!     }
//!   }
//!   return view(%output, min(%cursor, %output_capacity))
//! }
//! ```
//!
//! Pool rows always have the layout `[semantic words | preorder token bits]`.
//! Expression words beyond the used bits are zero, so child expressions can be
//! appended by shifting whole words.

use melior::ir::block::BlockLike;
use melior::ir::{Block, Operation, Value};

use super::functions::define_fun_symbol;
use super::{arguments, symbol_name, Emitter, POOL};

/// External C ABI provided by `cpp/cpu/i64hashset.cpp`.
pub const HASHSET_CREATE: &str = "mlir_loom_hashset_i64_create_handle";
pub const HASHSET_INSERT: &str = "mlir_loom_hashset_i64_insert_one";

/// Everything needed to emit one production filter.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HashFilterConfig {
    /// Result nonterminal.
    pub nonterminal: String,
    /// Constructor symbol and its stable nonzero token.
    pub constructor: String,
    pub token: usize,
    /// Declaration-order rule number.
    pub rule: usize,
    /// Bits per constructor token.
    pub token_bits: usize,
    /// Semantic words of the result nonterminal.
    pub value_words: usize,
    /// Semantic words of each child nonterminal, in argument order.
    pub child_value_words: Vec<usize>,
    /// Constraint checked for candidates of the start nonterminal.
    pub constraint: Option<String>,
}

impl HashFilterConfig {
    pub fn symbol(&self) -> String {
        format!(
            "hash_filter_{}_{}_rule{}",
            symbol_name(&self.nonterminal),
            symbol_name(&self.constructor),
            self.rule
        )
    }
}

/// Name of the FNV helper for `words` semantic words.
pub fn compute_hash_symbol(words: usize) -> String {
    format!("compute_hash_w{words}")
}

/// Emit `@compute_hash_w<words>`: unrolled FNV-1a over packed words.
pub fn lower_compute_hash<'c>(emitter: &Emitter<'c>, words: usize) -> Operation<'c> {
    let i64_type = emitter.ty("i64");
    let vector = emitter.ty(&format!("vector<{words}xi64>"));
    emitter.function(
        &compute_hash_symbol(words),
        &[vector],
        &[i64_type],
        true,
        &[],
        Some(&|block: &Block<'c>| {
            let packed = arguments(block)[0];
            let prime = emitter.i64(block, 0x100000001b3);
            let mut hash = emitter.i64(block, 0xcbf29ce484222325);
            for position in 0..words {
                let attribute = emitter.attr(&format!("array<i64: {position}>"));
                let word = emitter.value(block, "vector.extract", &[packed], i64_type, &[("static_position", attribute)]);
                let mixed = emitter.binary(block, "arith.xori", hash, word);
                hash = emitter.binary(block, "arith.muli", mixed, prime);
            }
            emitter.op(block, "func.return", &[hash], &[], &[], Vec::new());
        }),
    )
}

/// OR `bits` bits from `source[source_row, source_column..]` into
/// `output[slot, column_base..]` starting at bit `bit_offset`.
#[allow(clippy::too_many_arguments)]
fn append_bits<'c>(
    emitter: &Emitter<'c>,
    block: &Block<'c>,
    output: Value<'c, '_>,
    slot: Value<'c, '_>,
    column_base: Value<'c, '_>,
    expression_words: Value<'c, '_>,
    source: Value<'c, '_>,
    source_row: Value<'c, '_>,
    source_column: Value<'c, '_>,
    bits: Value<'c, '_>,
    bit_offset: Value<'c, '_>,
) {
    let zero = emitter.index(block, 0);
    let one = emitter.index(block, 1);
    let sixty_three = emitter.index(block, 63);
    let sixty_four = emitter.index(block, 64);
    let rounded = emitter.binary(block, "arith.addi", bits, sixty_three);
    let words = emitter.binary(block, "arith.divui", rounded, sixty_four);
    let first_word = emitter.binary(block, "arith.divui", bit_offset, sixty_four);
    let shift = emitter.binary(block, "arith.remui", bit_offset, sixty_four);
    let shift_i64 = emitter.index_cast(block, shift, "i64");
    let sixty_four_i64 = emitter.i64(block, 64);
    let back_shift = emitter.binary(block, "arith.subi", sixty_four_i64, shift_i64);
    let unaligned = emitter.cmpi(block, "ne", shift, zero);
    emitter.for_loop(block, zero, words, |body, word| {
        let column = emitter.binary(body, "arith.addi", source_column, word);
        let value = emitter.load(body, source, &[source_row, column], "i64");
        let target = emitter.binary(body, "arith.addi", first_word, word);
        let output_column = emitter.binary(body, "arith.addi", column_base, target);
        let previous = emitter.load(body, output, &[slot, output_column], "i64");
        let low = emitter.binary(body, "arith.shli", value, shift_i64);
        let merged = emitter.binary(body, "arith.ori", previous, low);
        emitter.store(body, merged, output, &[slot, output_column]);
        let next = emitter.binary(body, "arith.addi", target, one);
        let fits = emitter.cmpi(body, "ult", next, expression_words);
        let spill = emitter.binary(body, "arith.andi", unaligned, fits);
        emitter.if_then(body, spill, |inner| {
            let next_column = emitter.binary(inner, "arith.addi", column_base, next);
            let previous = emitter.load(inner, output, &[slot, next_column], "i64");
            let high = emitter.binary(inner, "arith.shrui", value, back_shift);
            let merged = emitter.binary(inner, "arith.ori", previous, high);
            emitter.store(inner, merged, output, &[slot, next_column]);
        });
    });
}

/// Emit one production-specific parallel filter.
pub fn lower_hash_filter<'c>(emitter: &Emitter<'c>, config: &HashFilterConfig) -> Operation<'c> {
    let index = emitter.ty("index");
    let i64_type = emitter.ty("i64");
    let pool = emitter.ty(POOL);
    let arity = config.child_value_words.len();
    let mut signature = vec![emitter.ty("memref<?xi64>"), index, index, index, index, emitter.ty("memref<1xi64>")];
    signature.extend(std::iter::repeat_n(pool, arity));
    signature.extend(std::iter::repeat_n(index, arity));
    signature.extend([i64_type, emitter.ty("memref<?xi64>"), emitter.ty("memref<1xi8>")]);
    let attributes = [
        ("loom.constructor", emitter.attr(&format!("{:?}", config.constructor))),
        ("loom.token", emitter.attr(&format!("{} : i64", config.token))),
        ("loom.token_bits", emitter.attr(&format!("{} : i64", config.token_bits))),
        ("loom.semantic_words", emitter.attr(&format!("{} : i64", config.value_words))),
    ];
    emitter.function(
        &config.symbol(),
        &signature,
        &[pool],
        true,
        &attributes,
        Some(&|entry: &Block<'c>| {
            let arguments = arguments(entry);
            let (arena, offset, capacity, row_words, start, cursor) =
                (arguments[0], arguments[1], arguments[2], arguments[3], arguments[4], arguments[5]);
            let children = &arguments[6..6 + arity];
            let child_bits = &arguments[6 + arity..6 + 2 * arity];
            let (seen, solution, found) = (
                arguments[6 + 2 * arity],
                arguments[7 + 2 * arity],
                arguments[8 + 2 * arity],
            );

            let zero = emitter.index(entry, 0);
            let one = emitter.index(entry, 1);
            let output = emitter.pool_view(entry, arena, offset, capacity, row_words);
            // The cursor is caller-provided memory rather than a stack slot,
            // so the loop body may run on a device.
            let start_i64 = emitter.index_cast(entry, start, "i64");
            emitter.store(entry, start_i64, cursor, &[zero]);
            let mut counts = Vec::with_capacity(arity);
            let mut batch = one;
            for &child in children {
                let rows = emitter.dim(entry, child, 0);
                batch = emitter.binary(entry, "arith.muli", batch, rows);
                counts.push(rows);
            }

            let body = Block::new(&[(index, emitter.location)]);
            let linear: Value = body.argument(0).expect("induction variable").into();
            let zero_i8 = emitter.i8(&body, 0);
            let state = emitter.atomic_rmw(&body, "ori", zero_i8, found, zero);
            let searching = emitter.cmpi(&body, "eq", state, zero_i8);
            emitter.if_then(&body, searching, |block| {
                // Mixed-radix decode: the last child varies fastest.
                let mut rows = vec![zero; arity];
                let mut rest = linear;
                for child in (0..arity).rev() {
                    rows[child] = emitter.binary(block, "arith.remui", rest, counts[child]);
                    rest = emitter.binary(block, "arith.divui", rest, counts[child]);
                }
                // One vector.transfer_read per child value.
                let words: Vec<Value> = (0..arity)
                    .map(|child| emitter.transfer_read(block, children[child], rows[child], config.child_value_words[child]))
                    .collect();
                let packed = emitter.ty(&format!("vector<{}xi64>", config.value_words));
                let value = emitter.call(block, &define_fun_symbol(&config.constructor), &words, &[packed]);
                // Insert and store only while the pool has room. A full pool
                // stops growing the hash set; its candidates are still checked.
                let used = emitter.atomic_rmw(block, "addi", emitter.i64(block, 0), cursor, zero);
                let used = emitter.index_cast(block, used, "index");
                let room = emitter.cmpi(block, "ult", used, capacity);
                let i1 = emitter.ty("i1");
                let duplicate = emitter.if_else(
                    block,
                    room,
                    &[i1],
                    |block| {
                        let hash = emitter.call(block, &compute_hash_symbol(config.value_words), &value, &[i64_type]);
                        let fresh = emitter.call(block, HASHSET_INSERT, &[seen, hash[0]], &[i1]);
                        emitter.if_then(block, fresh[0], |block| {
                            let one_i64 = emitter.i64(block, 1);
                            let slot = emitter.atomic_rmw(block, "addi", one_i64, cursor, zero);
                            let slot = emitter.index_cast(block, slot, "index");
                            let fits = emitter.cmpi(block, "ult", slot, capacity);
                            emitter.if_then(block, fits, |block| {
                                store_row(emitter, block, config, output, slot, row_words, &value, children, &rows, child_bits);
                            });
                        });
                        let truth = emitter.value(block, "arith.constant", &[], i1, &[("value", emitter.attr("true"))]);
                        vec![emitter.binary(block, "arith.xori", fresh[0], truth)]
                    },
                    |block| vec![emitter.value(block, "arith.constant", &[], i1, &[("value", emitter.attr("false"))])],
                )[0];
                if let Some(constraint) = &config.constraint {
                    // A value seen before was already checked when first inserted.
                    let truth = emitter.value(block, "arith.constant", &[], i1, &[("value", emitter.attr("true"))]);
                    let new = emitter.binary(block, "arith.xori", duplicate, truth);
                    emitter.if_then(block, new, |block| {
                        let holds = emitter.call(block, constraint, &value, &[i1]);
                        emitter.if_then(block, holds[0], |block| {
                            let zero = emitter.index(block, 0);
                            let one_i8 = emitter.i8(block, 1);
                            let zero_i8 = emitter.i8(block, 0);
                            let previous = emitter.atomic_rmw(block, "ori", one_i8, found, zero);
                            let first = emitter.cmpi(block, "eq", previous, zero_i8);
                            emitter.if_then(block, first, |block| {
                                // Write the expression straight into %solution.
                                let zero = emitter.index(block, 0);
                                let one = emitter.index(block, 1);
                                let words = emitter.dim(block, solution, 0);
                                let target = emitter.pool_view(block, solution, zero, one, words);
                                write_expression(
                                    emitter, block, config, target, zero, zero, words, children, &rows, child_bits,
                                );
                            });
                        });
                    });
                }
            });
            emitter.op(&body, "scf.reduce", &[], &[], &[], Vec::new());
            emitter.op(
                entry,
                "scf.parallel",
                &[zero, batch, one],
                &[],
                &[("operandSegmentSizes", emitter.attr("array<i32: 1, 1, 1, 0>"))],
                vec![super::region(body)],
            );

            let used = emitter.load(entry, cursor, &[zero], "i64");
            let used = emitter.index_cast(entry, used, "index");
            let used = emitter.binary(entry, "arith.minui", used, capacity);
            let result = emitter.pool_view(entry, arena, offset, used, row_words);
            emitter.op(entry, "func.return", &[result], &[], &[], Vec::new());
        }),
    )
}

/// Write `[value | token | child expressions...]` into `output[slot]`.
#[allow(clippy::too_many_arguments)]
fn store_row<'c>(
    emitter: &Emitter<'c>,
    block: &Block<'c>,
    config: &HashFilterConfig,
    output: Value<'c, '_>,
    slot: Value<'c, '_>,
    row_words: Value<'c, '_>,
    value: &[Value<'c, '_>],
    children: &[Value<'c, '_>],
    rows: &[Value<'c, '_>],
    child_bits: &[Value<'c, '_>],
) {
    emitter.transfer_write(block, value[0], output, slot);
    let base = emitter.index(block, config.value_words);
    let expression_words = emitter.binary(block, "arith.subi", row_words, base);
    write_expression(emitter, block, config, output, slot, base, expression_words, children, rows, child_bits);
}

/// Write `[token | child expressions...]` into `target[row, base..base + words]`.
#[allow(clippy::too_many_arguments)]
fn write_expression<'c>(
    emitter: &Emitter<'c>,
    block: &Block<'c>,
    config: &HashFilterConfig,
    target: Value<'c, '_>,
    row: Value<'c, '_>,
    base: Value<'c, '_>,
    expression_words: Value<'c, '_>,
    children: &[Value<'c, '_>],
    rows: &[Value<'c, '_>],
    child_bits: &[Value<'c, '_>],
) {
    // The token occupies the low bits of the first expression word.
    let token = emitter.i64(block, config.token as u64);
    emitter.store(block, token, target, &[row, base]);
    let zero_i64 = emitter.i64(block, 0);
    let one = emitter.index(block, 1);
    emitter.for_loop(block, one, expression_words, |body, word| {
        let column = emitter.binary(body, "arith.addi", base, word);
        emitter.store(body, zero_i64, target, &[row, column]);
    });
    let mut offset = emitter.index(block, config.token_bits);
    for (child, (&pool, &bits)) in children.iter().zip(child_bits).enumerate() {
        let source_column = emitter.index(block, config.child_value_words[child]);
        append_bits(
            emitter, block, target, row, base, expression_words, pool, rows[child], source_column, bits, offset,
        );
        offset = emitter.binary(block, "arith.addi", offset, bits);
    }
}
