//! Fill the hash-related slots of the synthesis module.
//!
//! ```text
//! func.func private @compute_hash_w<N>(%packed: vector<Nxi64>) -> i64
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
//! For a bit-sliced constructor (one with a gather, see `sliced`), an
//! iteration instead covers a tuple of 64-row child blocks and of shifts:
//!
//! ```text
//!   scf.parallel (%linear) = (0) to (blocks(child0) * ... * 64^(arity-1)) step (1) {
//!     %sliced_i = transpose64(block_i(%child_i)) rotated right by shift_i
//!     func.call @define_fun_sliced_<constructor>(%sliced..., %out)
//!     %values = transpose64(%out)              // 64 packed candidates
//!     for every lane whose child rows exist: hash, store, and check as above
//!   }
//! ```
//!
//! Lane `k` pairs row `k` of the first block with row `(k + shift_i) % 64` of
//! block `i`, so the 64 shifts of each later child cover the cartesian
//! product of the blocks.
//!
//! Pool rows always have the layout `[semantic words | preorder token bits]`.
//! Expression words beyond the used bits are zero, so child expressions can be
//! appended by shifting whole words.

use high_mlir::arith::{self, CmpIPredicate, Constant};
use high_mlir::func::Signature;
use high_mlir::memref::{self, AtomicRMWKind};
use high_mlir::vector::{self, Transfer};
use high_mlir::{scf, Attr, Ty, Val, I1, I64, I8, INDEX};

use super::functions::{define_fun_symbol, packed_type};
use super::sliced::{lane_mask, rotate_right, slice_rows, sliced_signature, sliced_type, to_i64, unslice_rows};
use super::{index, pool_type, pool_view, symbol_name, to_index, words_type};

/// External C ABI provided by `cpp/cpu/i64hashset.cpp`.
pub const HASHSET_CREATE: &str = "mlir_loom_hashset_i64_create_handle";
pub const HASHSET_INSERT: &str = "mlir_loom_hashset_i64_insert_one";

/// `@mlir_loom_hashset_i64_create_handle(capacity) -> handle`.
pub fn hashset_create() -> Signature {
    Signature::new(HASHSET_CREATE, [I64.into()], [I64.into()])
}

/// `@mlir_loom_hashset_i64_insert_one(handle, hash) -> inserted`.
pub fn hashset_insert() -> Signature {
    Signature::new(HASHSET_INSERT, [I64.into(), I64.into()], [I1.into()])
}

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
    /// Logical bits of the result and of each child, for a bit-sliced constructor.
    pub value_bits: usize,
    pub child_bits: Vec<usize>,
    /// Build 64 candidates at once with the bit-sliced constructor.
    pub sliced: bool,
    /// Build one orientation of a commutative constructor: the candidates
    /// whose first child is stored no later than the second (see
    /// `Grammar::symmetric`).
    pub commutative: bool,
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

    /// The filter's signature (see the module documentation).
    pub fn signature(&self) -> Signature {
        let arity = self.child_value_words.len();
        let mut inputs = vec![words_type(), INDEX.into(), INDEX.into(), INDEX.into(), INDEX.into(), Ty::memref([Some(1)], I64)];
        inputs.extend(std::iter::repeat_n(pool_type(), arity));
        inputs.extend(std::iter::repeat_n(Ty::from(INDEX), arity));
        inputs.extend([I64.into(), words_type(), Ty::memref([Some(1)], I8)]);
        Signature::new(self.symbol(), inputs, [pool_type()])
    }

    /// `@define_fun_<constructor>` as called by this filter.
    fn constructor_signature(&self) -> Signature {
        let words = |words: usize| packed_type(words * 64);
        Signature::new(
            define_fun_symbol(&self.constructor),
            self.child_value_words.iter().map(|&child| words(child)),
            [words(self.value_words)],
        )
    }
}

/// Name of the FNV helper for `words` semantic words.
pub fn compute_hash_symbol(words: usize) -> String {
    format!("compute_hash_w{words}")
}

fn compute_hash_signature(words: usize) -> Signature {
    Signature::new(compute_hash_symbol(words), [packed_type(words * 64)], [I64.into()])
}

/// Emit `@compute_hash_w<words>`: unrolled FNV-1a over packed words.
pub fn lower_compute_hash(words: usize) -> Val {
    let function = compute_hash_signature(words).define(|arguments| {
        let prime = Constant::int(0x100000001b3, I64);
        let mut hash: Val = Constant::int(0xcbf29ce484222325u64 as i64, I64).into();
        for position in 0..words {
            let word = vector::extract(&arguments[0], position as i64);
            hash = arith::muli(arith::xori(hash, word), &prime, ());
        }
        hash.into()
    });
    function.annotate("sym_visibility", Attr::str("private"));
    function
}

fn in_bounds() -> Transfer {
    Transfer { in_bounds: Some(vec![true]), ..().into() }
}

/// `a < b` as unsigned `index` values.
fn ult(a: &Val, b: &Val) -> Val {
    arith::cmpi(CmpIPredicate::Ult, a, b)
}

/// OR `bits` bits from `source[source_row, source_column..]` into
/// `output[slot, column_base..]` starting at bit `bit_offset`.
#[allow(clippy::too_many_arguments)]
fn append_bits(
    output: &Val,
    slot: &Val,
    column_base: &Val,
    expression_words: &Val,
    source: &Val,
    source_row: &Val,
    source_column: &Val,
    bits: &Val,
    bit_offset: &Val,
) {
    let words = arith::divui(arith::addi(bits, index(63), ()), index(64), ());
    let first_word = arith::divui(bit_offset, index(64), ());
    let shift = arith::remui(bit_offset, index(64));
    let shift_i64: Val = arith::index_cast(&shift, I64).into();
    let back_shift = arith::subi(Constant::int(64, I64), &shift_i64, ());
    let unaligned = arith::cmpi(CmpIPredicate::Ne, &shift, index(0));
    scf! {for 0 to words step 1}.then(|word| {
        let column = arith::addi(source_column, &word, ());
        let value = memref::load(source, (source_row, &column), ());
        let target = arith::addi(&first_word, &word, ());
        let output_column = arith::addi(column_base, &target, ());
        let previous = memref::load(output, (slot, &output_column), ());
        let merged = arith::ori(previous, arith::shli(&value, &shift_i64, ()));
        memref::store(merged, output, (slot, &output_column), ());
        let next = arith::addi(&target, index(1), ());
        let spill = arith::andi(&unaligned, ult(&next, expression_words));
        scf! {if spill}
            .then(|| {
                let next_column = arith::addi(column_base, &next, ());
                let previous = memref::load(output, (slot, &next_column), ());
                let merged = arith::ori(previous, arith::shrui(&value, &back_shift, ()));
                memref::store(merged, output, (slot, &next_column), ());
                ().into()
            })
            .end();
        ().into()
    });
}

/// Pool and expression arguments of one filter call.
struct Children<'a> {
    pools: &'a [Val],
    rows: &'a [Val],
    bits: &'a [Val],
}

/// The arguments and runtime functions of the filter being built.
struct Filter<'a> {
    config: &'a HashFilterConfig,
    output: Val,
    capacity: &'a Val,
    row_words: &'a Val,
    cursor: &'a Val,
    seen: &'a Val,
    solution: &'a Val,
    found: &'a Val,
    pools: &'a [Val],
    child_bits: &'a [Val],
    hash: Signature,
    insert: Signature,
    constraint: Option<Signature>,
    /// Per child pool: its `(offset, row stride)` in the arena, when the
    /// constructor is commutative.
    positions: Vec<(Val, Val)>,
}

impl Filter<'_> {
    /// Hash, store, and check one candidate `value` built from child rows `rows`.
    fn candidate(&self, value: &Val, rows: &[Val]) {
        let (config, capacity, cursor) = (self.config, self.capacity, self.cursor);
        // Insert and store only while the pool has room. A full pool stops
        // growing the hash set; its candidates are still checked.
        let used = to_index(&memref::atomic_rmw(AtomicRMWKind::AddI, 0i64, cursor, 0));
        let children = Children { pools: self.pools, rows, bits: self.child_bits };
        let [duplicate] = scf! {if ult(&used, capacity)}
            .then(|| {
                let hashed = self.hash.call([value])[0].clone();
                let fresh = self.insert.call((self.seen, &hashed))[0].clone();
                scf! {if &fresh}
                    .then(|| {
                        let slot = to_index(&memref::atomic_rmw(AtomicRMWKind::AddI, 1i64, cursor, 0));
                        scf! {if ult(&slot, capacity)}
                            .then(|| {
                                store_row(config, &self.output, &slot, self.row_words, value, &children);
                                ().into()
                            })
                            .end();
                        ().into()
                    })
                    .end();
                arith::xori(&fresh, true).into()
            })
            .else_(|| Constant::bool(false).into())
            .try_into()
            .unwrap();
        if let Some(constraint) = &self.constraint {
            // A value seen before was already checked when first inserted.
            scf! {if arith::xori(&duplicate, true)}
                .then(|| {
                    let holds = constraint.call([value])[0].clone();
                    scf! {if holds}
                        .then(|| {
                            let previous = memref::atomic_rmw(AtomicRMWKind::OrI, 1i8, self.found, 0);
                            scf! {if arith::cmpi(CmpIPredicate::Eq, previous, 0)}
                                .then(|| {
                                    // Write the expression straight into %solution.
                                    let words: Val = memref::dim(self.solution, 0).into();
                                    let target = pool_view(self.solution, &index(0), &index(1), &words);
                                    write_expression(config, &target, &index(0), &index(0), &words, &children);
                                    ().into()
                                })
                                .end();
                            ().into()
                        })
                        .end();
                    ().into()
                })
                .end();
        }
    }

    /// The arena position of row `row` of child `child`'s pool.
    fn position(&self, child: usize, row: &Val) -> Val {
        let (offset, stride) = &self.positions[child];
        arith::addi(offset, arith::muli(row, stride, ()), ())
    }

    /// Whether the candidate of child rows `rows` is the kept orientation
    /// of a commutative constructor: its first child is stored no later.
    fn oriented(&self, rows: &[Val]) -> Option<Val> {
        self.config.commutative.then(|| {
            arith::cmpi(CmpIPredicate::Ule, self.position(0, &rows[0]), self.position(1, &rows[1]))
        })
    }

    /// Run `body` when `condition` (if any) holds.
    fn when(condition: Option<Val>, body: impl FnOnce()) {
        match condition {
            None => body(),
            Some(condition) => {
                scf! {if condition}
                    .then(|| {
                        body();
                        ().into()
                    })
                    .end();
            }
        }
    }

    /// Whether no solution has been found yet (an atomic read of `%found`).
    fn searching(&self) -> Val {
        let state = memref::atomic_rmw(AtomicRMWKind::OrI, 0i8, self.found, 0);
        arith::cmpi(CmpIPredicate::Eq, state, 0)
    }
}

/// Emit one production-specific parallel filter.
pub fn lower_hash_filter(config: &HashFilterConfig) -> Val {
    let arity = config.child_value_words.len();
    let function = config.signature().define(|arguments| {
        let (arena, offset, capacity, row_words, start, cursor) =
            (&arguments[0], &arguments[1], &arguments[2], &arguments[3], &arguments[4], &arguments[5]);
        let filter = Filter {
            config,
            output: pool_view(arena, offset, capacity, row_words),
            capacity,
            row_words,
            cursor,
            seen: &arguments[6 + 2 * arity],
            solution: &arguments[7 + 2 * arity],
            found: &arguments[8 + 2 * arity],
            pools: &arguments[6..6 + arity],
            child_bits: &arguments[6 + arity..6 + 2 * arity],
            hash: compute_hash_signature(config.value_words),
            insert: hashset_insert(),
            constraint: config.constraint.as_ref().map(|symbol| {
                Signature::new(symbol.clone(), [packed_type(config.value_words * 64)], [I1.into()])
            }),
            positions: if config.commutative {
                arguments[6..6 + arity]
                    .iter()
                    .map(|pool| {
                        let (_, offset, _, strides) = memref::extract_strided_metadata(pool);
                        (offset.into(), strides[0].clone().into())
                    })
                    .collect()
            } else {
                Vec::new()
            },
        };
        // The cursor is caller-provided memory rather than a stack slot, so
        // the loop body may run on a device.
        memref::store(arith::index_cast(start, I64), cursor, 0, ());
        let counts: Vec<Val> = filter.pools.iter().map(|pool| memref::dim(pool, 0).into()).collect();
        if config.sliced {
            sliced_body(&filter, &counts);
        } else {
            packed_body(&filter, &counts);
        }
        let used = to_index(&memref::load(cursor, 0, ()));
        let used = arith::minui(used, capacity);
        pool_view(arena, offset, &used, row_words).into()
    });
    function
        .annotate("sym_visibility", Attr::str("private"))
        .annotate("loom.constructor", Attr::str(&config.constructor))
        .annotate("loom.token", Attr::Int(config.token as i64, I64.into()))
        .annotate("loom.token_bits", Attr::Int(config.token_bits as i64, I64.into()))
        .annotate("loom.semantic_words", Attr::Int(config.value_words as i64, I64.into()))
        .annotate("loom.sliced", Attr::Bool(config.sliced));
    function
}

/// One parallel iteration per candidate: read the child values, call the
/// packed constructor, and check the result.
fn packed_body(filter: &Filter, counts: &[Val]) {
    let config = filter.config;
    let arity = counts.len();
    let constructor = config.constructor_signature();
    let batch = counts.iter().fold(index(1), |batch, rows| arith::muli(batch, rows, ()));
    scf! {parallel (0) to (batch) step (1)}.then(|linear| {
        scf! {if filter.searching()}
            .then(|| {
                // Mixed-radix decode: the last child varies fastest.
                let mut rows = vec![index(0); arity];
                let mut rest: Val = linear.clone().into();
                for child in (0..arity).rev() {
                    rows[child] = arith::remui(&rest, &counts[child]);
                    rest = arith::divui(&rest, &counts[child], ());
                }
                Filter::when(filter.oriented(&rows), || {
                    // One vector.transfer_read per child value.
                    let words: Vec<Val> = (0..arity)
                        .map(|child| {
                            let ty = packed_type(config.child_value_words[child] * 64);
                            vector::transfer_read(&filter.pools[child], (&rows[child], 0), 0i64, ty, in_bounds()).into()
                        })
                        .collect();
                    let value = constructor.call(words)[0].clone();
                    filter.candidate(&value, &rows);
                });
                ().into()
            })
            .end();
        ().into()
    });
}

/// One parallel iteration per tuple of 64-row child blocks and of shifts:
/// each block is transposed into bit-sliced form, every child after the
/// first rotated by its shift, so lane `k` pairs row `k` of the first block
/// with row `(k + shift) % 64` of each other block; the 64 shifts of a child
/// cover the cartesian product of its block with the others. The bit-sliced
/// constructor builds 64 candidates at once, which are transposed back and
/// checked one by one.
fn sliced_body(filter: &Filter, counts: &[Val]) {
    let config = filter.config;
    let arity = counts.len();
    let constructor = sliced_signature(&config.constructor, &config.child_bits, config.value_bits);
    let blocks: Vec<Val> = counts.iter().map(|rows| arith::divui(arith::addi(rows, index(63), ()), index(64), ())).collect();
    let shifts = 64usize.pow(arity as u32 - 1);
    let batch = blocks.iter().fold(index(shifts), |batch, blocks| arith::muli(batch, blocks, ()));
    scf! {parallel (0) to (batch) step (1)}.then(|linear| {
        scf! {if filter.searching()}
            .then(|| {
                // Mixed-radix decode: shifts vary fastest, then the last child's block.
                let mut rest: Val = linear.clone().into();
                let mut shift = vec![Val::from(Constant::int(0, I64)); arity];
                for child in (1..arity).rev() {
                    shift[child] = to_i64(&arith::remui(&rest, index(64)));
                    rest = arith::divui(&rest, index(64), ());
                }
                let mut first = vec![index(0); arity];
                for child in (0..arity).rev() {
                    first[child] = arith::muli(arith::remui(&rest, &blocks[child]), index(64), ());
                    rest = arith::divui(&rest, &blocks[child], ());
                }
                let present: Vec<Val> =
                    (0..arity).map(|child| arith::minui(index(64), arith::subi(&counts[child], &first[child], ()))).collect();
                // Lane `k` is a candidate when every child has its row.
                let mask = (1..arity).fold(lane_mask(&present[0]), |mask, child| {
                    arith::andi(mask, rotate_right(&lane_mask(&present[child]), &shift[child]))
                });
                let mut run = arith::cmpi(CmpIPredicate::Ne, &mask, 0);
                if config.commutative {
                    // Skip blocks whose every first-child row is stored after
                    // every second-child row: all their candidates are swaps.
                    let last = arith::subi(arith::addi(&first[1], &present[1], ()), index(1), ());
                    let ordered = arith::cmpi(CmpIPredicate::Ule, filter.position(0, &first[0]), filter.position(1, &last));
                    run = arith::andi(run, ordered);
                }
                scf! {if run}
                    .then(|| {
                        let scratch: Val = memref::alloca(Ty::memref([Some(64)], I64), (), ()).into();
                        let mut arguments: Vec<Val> = (0..arity)
                            .map(|child| {
                                let sliced: Val = memref::alloca(sliced_type(config.child_bits[child]), (), ()).into();
                                let pool = &filter.pools[child];
                                let row = |lane: &Val, word: &Val| {
                                    memref::load(pool, (&arith::addi(&first[child], lane, ()), word), ())
                                };
                                let words = config.child_value_words[child];
                                slice_rows(&row, &present[child], words, config.child_bits[child], &shift[child], &scratch, &sliced);
                                sliced
                            })
                            .collect();
                        let out: Val = memref::alloca(sliced_type(config.value_bits), (), ()).into();
                        arguments.push(out.clone());
                        constructor.call(arguments);
                        let values: Val =
                            memref::alloca(Ty::memref([Some(64), Some(config.value_words as u64)], I64), (), ()).into();
                        unslice_rows(&out, config.value_bits, config.value_words, &scratch, &|lane, word, value| {
                            memref::store(value, &values, (lane, word), ());
                        });
                        scf! {for 0 to 64 step 1}.then(|lane| {
                            let lane: Val = lane.into();
                            let bit = arith::andi(arith::shrui(&mask, to_i64(&lane), ()), Constant::int(1, I64));
                            scf! {if arith::cmpi(CmpIPredicate::Ne, bit, 0)}
                                .then(|| {
                                    let ty = packed_type(config.value_words * 64);
                                    let value: Val = vector::load(&values, (&lane, index(0)), ty, ()).into();
                                    let rows: Vec<Val> = (0..arity)
                                        .map(|child| {
                                            let offset = if child == 0 {
                                                lane.clone()
                                            } else {
                                                let shifted = arith::addi(&lane, to_index(&shift[child]), ());
                                                arith::andi(shifted, index(63))
                                            };
                                            arith::addi(&first[child], offset, ())
                                        })
                                        .collect();
                                    Filter::when(filter.oriented(&rows), || filter.candidate(&value, &rows));
                                    ().into()
                                })
                                .end();
                            ().into()
                        });
                        ().into()
                    })
                    .end();
                ().into()
            })
            .end();
        ().into()
    });
}

/// Write `[value | token | child expressions...]` into `output[slot]`.
fn store_row(config: &HashFilterConfig, output: &Val, slot: &Val, row_words: &Val, value: &Val, children: &Children) {
    vector::transfer_write(value, output, (slot, 0), in_bounds());
    let base = index(config.value_words);
    let expression_words = arith::subi(row_words, &base, ());
    write_expression(config, output, slot, &base, &expression_words, children);
}

/// Write `[token | child expressions...]` into `target[row, base..base + words]`.
fn write_expression(
    config: &HashFilterConfig,
    target: &Val,
    row: &Val,
    base: &Val,
    expression_words: &Val,
    children: &Children,
) {
    // The token occupies the low bits of the first expression word.
    memref::store(Constant::int(config.token as i64, I64), target, (row, base), ());
    scf! {for 1 to expression_words step 1}.then(|word| {
        let column = arith::addi(base, &word, ());
        memref::store(0i64, target, (row, &column), ());
        ().into()
    });
    let mut offset = index(config.token_bits);
    for (child, (pool, bits)) in children.pools.iter().zip(children.bits).enumerate() {
        let source_column = index(config.child_value_words[child]);
        append_bits(target, row, base, expression_words, pool, &children.rows[child], &source_column, bits, &offset);
        offset = arith::addi(&offset, bits, ());
    }
}
