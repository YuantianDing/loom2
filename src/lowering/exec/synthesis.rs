//! Assemble the synthesis module and its `synthesize` entry point.
//!
//! ```text
//! func.func @synthesize() -> memref<?xi64> attributes {
//!     loom.synth_fun = "<name>", loom.start_sort = "<start>",
//!     loom.tokens = "<token-map>", loom.token_bits = <token-bits>, ...
//! } {
//!   // max_pool_bytes = hash tables (a quarter, power-of-two slots per
//!   // nonterminal) + the value/expression arena (the rest).
//!   %arena = memref.alloc(<arena words>) : memref<?xi64>
//!   %solution = zeroed_memref(<ceil(max_size * token_bits / 64)>)
//!   %found = 0
//!   %cursor = memref<1xi64>            // row counter shared by the filters
//!   %seen_<group> = mlir_loom_hashset_i64_create_handle(<table slots>)
//!
//!   %pools = memref<nonterminals x (max_size + 1) x 2xi64>   // (offset, rows)
//!   scf.for %size = 1 to max_expression_size + 1 {
//!     <for nonterminal> {
//!       %row_words = ceil(width / 64) + ceil(%size * token_bits / 64)
//!       // Tables stay at most half full, so their memory is fixed.
//!       %capacity = min(total_candidates, remaining_arena / %row_words,
//!                       table_slots / 2 - rows_stored_<group>)
//!       <for rule> scf.for <every positive split of %size - 1> {
//!         scf.if %found == 0 && <child pools are not empty> {
//!           %pool = func.call @hash_filter_<nonterminal>_<constructor>_rule<rule>(
//!               %arena, <offset/capacity/start>, %cursor, <child pools>,
//!               %seen_<group>, %solution, %found)
//!         }
//!       }
//!       %pools[nonterminal, %size] = (offset, rows(%pool))
//!       advance_arena_cursor(rows(%pool) * %row_words)
//!     }
//!   }
//!   return %solution
//! }
//!
//! func.func @main() -> i32 {
//!   mlir_loom_print_result(synthesize(), <token-bits>)   // cpp runtime
//! }
//! ```

use std::collections::{BTreeSet, HashMap};

use high_mlir::arith::{self, CmpIPredicate, Constant};
use high_mlir::func::Signature;
use high_mlir::{builtin, memref, scf, Attr, Root, Ty, Val, I32, I64, I8, INDEX};

use super::functions::{lower_constraint, lower_tensor_fn, result_bits};
use super::hash_filter::{hashset_create, lower_compute_hash, lower_hash_filter, HashFilterConfig};
use super::sliced::{is_sliced, lower_sliced_tensor_fn, lower_transpose};
use super::sliced::to_i64;
use super::{index, invalid, pool_view, to_index, words_type, Result};
use crate::compile::pass::fuse;
use crate::compile::TensorFn;
use crate::grammar::token_width;
use crate::lowering::word_count;
use crate::problem::SynthFun;

/// Runtime result printer provided by the generated or generic C runtime.
pub const PRINT_RESULT: &str = "mlir_loom_print_result";
/// Symbol of the start-nonterminal constraint.
pub const CONSTRAINT: &str = "sygus_constraints";

/// Bounds and resource policy for syntax-guided enumeration.
///
/// `max_pool_bytes` is a hard bound shared by every generated pool, not a
/// separate allowance per nonterminal or expression size.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SynthesisOptions {
    /// Largest candidate, measured in constructor tokens.
    pub max_expression_size: usize,
    /// Maximum byte capacity of the preallocated value/expression arena.
    pub max_pool_bytes: usize,
    /// Initial reservation of every per-nonterminal hash set.
    pub initial_hash_capacity: usize,
}

impl Default for SynthesisOptions {
    fn default() -> Self {
        Self {
            max_expression_size: 20,
            max_pool_bytes: 256 * 1024 * 1024,
            initial_hash_capacity: 1 << 20,
        }
    }
}

/// Fused graphs and grammar facts computed before MLIR emission.
#[derive(Clone, Debug)]
pub struct SynthesisPlan {
    /// One fused graph per constructor, in token order.
    pub constructors: Vec<TensorFn>,
    /// Whether each constructor is built bit-sliced (see [`super::sliced`]).
    pub sliced: Vec<bool>,
    /// The fused constraint over the start nonterminal.
    pub constraint: TensorFn,
    pub filters: Vec<HashFilterConfig>,
    pub token_bits: usize,
    /// `(token, constructor, arity)` in token order.
    pub tokens: Vec<(usize, String, usize)>,
}

/// Validate a problem and fuse every traced function. With `bit_sliced`,
/// constructors with a gather are built bit-sliced (CPU only: on a GPU every
/// thread would keep the sliced buffers in local memory).
pub fn plan_synthesis(problem: &SynthFun, bit_sliced: bool) -> Result<SynthesisPlan> {
    const STAGE: &str = "synthesis lowering";
    let grammar = problem.grammar();
    let widths: HashMap<&str, usize> = grammar
        .nonterminals()
        .iter()
        .map(|nonterminal| (nonterminal.symbol.as_str(), nonterminal.width))
        .collect();
    let token_bits = token_width(grammar.constructor_count()).map_err(|error| invalid(STAGE, error.to_string()))?;
    if token_bits > 64 {
        return Err(invalid(STAGE, "constructor tokens must fit in one word"));
    }

    let mut tokens: Vec<(usize, String, usize)> = Vec::new();
    let mut constructors = Vec::new();
    let mut sliced = Vec::new();
    let mut filters = Vec::new();
    for nonterminal in grammar.nonterminals() {
        for production in &nonterminal.productions {
            let function = problem.functions().get(&production.constructor).ok_or_else(|| {
                invalid(STAGE, format!("no traced function for constructor {:?}", production.constructor))
            })?;
            let arity = production.children.len();
            if function.inputs.len() != arity {
                return Err(invalid(
                    STAGE,
                    format!("{} takes {} arguments, the grammar gives {arity}", function.name, function.inputs.len()),
                ));
            }
            if !tokens.iter().any(|(token, _, _)| *token == production.token()) {
                let graph = fuse(function);
                let bits = result_bits(&graph)?;
                if bits != nonterminal.width {
                    return Err(invalid(
                        STAGE,
                        format!(
                            "{} returns {bits} bits, but {} has width {}",
                            function.name, nonterminal.symbol, nonterminal.width
                        ),
                    ));
                }
                for ((name, tensor_type), child) in function.inputs.iter().zip(&production.children) {
                    let width = tensor_type.width();
                    if width != widths[child.as_str()] {
                        return Err(invalid(
                            STAGE,
                            format!("{}: argument {name} has {width} bits but {child} has {}", function.name, widths[child.as_str()]),
                        ));
                    }
                }
                tokens.push((production.token(), production.constructor.clone(), arity));
                sliced.push(bit_sliced && is_sliced(&graph));
                constructors.push(graph);
            }
            let position = tokens.iter().position(|(token, _, _)| *token == production.token()).expect("planned token");
            filters.push(HashFilterConfig {
                nonterminal: nonterminal.symbol.clone(),
                constructor: production.constructor.clone(),
                token: production.token(),
                rule: production.rule(),
                token_bits,
                value_words: word_count(nonterminal.width),
                child_value_words: production.children.iter().map(|child| word_count(widths[child.as_str()])).collect(),
                constraint: nonterminal.solution.then(|| CONSTRAINT.to_string()),
                value_bits: nonterminal.width,
                child_bits: production.children.iter().map(|child| widths[child.as_str()]).collect(),
                sliced: sliced[position],
                commutative: grammar.symmetric(nonterminal, production),
            });
        }
    }
    let constraint = fuse(problem.constraint());
    if constraint.inputs.len() != 1 || constraint.inputs[0].1.width() != widths[grammar.start()] {
        return Err(invalid(STAGE, "the constraint must take one value of the start nonterminal"));
    }
    Ok(SynthesisPlan {
        constructors,
        sliced,
        constraint,
        filters,
        token_bits,
        tokens,
    })
}

/// Emit the complete synthesis module.
///
/// On failure no partially assembled module is returned.
pub fn lower_synthesis(problem: &SynthFun, plan: &SynthesisPlan, options: SynthesisOptions) -> Result<Root> {
    const STAGE: &str = "synthesis lowering";
    if options.max_expression_size == 0 {
        return Err(invalid(STAGE, "the maximum expression size must be positive"));
    }
    let groups = problem.grammar().groups().len();
    let slots = table_slots(options, groups);
    let tables = 8 * slots * groups;
    let arena_words = options.max_pool_bytes.saturating_sub(tables) / 8;
    if arena_words == 0 {
        return Err(invalid(STAGE, "the value/expression arena must hold at least one word"));
    }
    let grammar = problem.grammar();
    let int = |value: usize| Attr::Int(value as i64, I64.into());
    let print_result = Signature::new(PRINT_RESULT, [words_type(), INDEX.into()], []);

    let token_map = plan
        .tokens
        .iter()
        .map(|(token, name, _)| format!("{token}={name}"))
        .collect::<Vec<_>>()
        .join(";");
    let solution_words = word_count(options.max_expression_size * plan.token_bits);
    let synthesize = Signature::new("synthesize", [], [words_type()]);
    let mut lowered = Ok(());
    let module = builtin::module(|| {
        lowered = (|| {
            // Runtime declarations: cpp/cpu hash set and result printer.
            hashset_create().declare();
            super::hash_filter::hashset_insert().declare();
            print_result.declare();
            if plan.sliced.contains(&true) {
                lower_transpose();
            }
            for (constructor, &sliced) in plan.constructors.iter().zip(&plan.sliced) {
                if sliced {
                    lower_sliced_tensor_fn(constructor)?;
                } else {
                    lower_tensor_fn(constructor)?;
                }
            }
            lower_constraint(CONSTRAINT, &plan.constraint)?;
            let hash_widths: BTreeSet<usize> = plan.filters.iter().map(|filter| filter.value_words).collect();
            hash_widths.into_iter().for_each(|words| drop(lower_compute_hash(words)));
            plan.filters.iter().for_each(|filter| drop(lower_hash_filter(filter)));

            let body = synthesize.define(|_| {
                synthesize_body(problem, plan, slots, options, arena_words, solution_words).into()
            });
            body.annotate("loom.synth_fun", Attr::str(problem.name()))
                .annotate("loom.start_sort", Attr::str(grammar.start()))
                .annotate("loom.tokens", Attr::str(&token_map))
                .annotate("loom.token_bits", int(plan.token_bits))
                .annotate("loom.max_expression_size", int(options.max_expression_size))
                .annotate("loom.max_pool_bytes", int(options.max_pool_bytes));

            Signature::new("main", [], [I32.into()]).define(|_| {
                let result = synthesize.call(Vec::<Val>::new())[0].clone();
                result.annotate("no_inline", Attr::Unit);
                print_result.call((&result, index(plan.token_bits)));
                Constant::int(0, I32).into()
            });
            Ok(())
        })();
    });
    lowered?;
    module
        .val()
        .annotate("loom.frontend", Attr::str("loom2-tensor-fn"))
        .annotate("loom.storage", Attr::str("two-dimensional-ve-pools"))
        .annotate("loom.semantic_encoding", Attr::str("packed-i64"))
        .annotate("loom.max_expression_size", int(options.max_expression_size))
        .annotate("loom.max_pool_bytes", int(options.max_pool_bytes))
        .annotate("loom.initial_hash_capacity", int(options.initial_hash_capacity));
    Ok(module)
}

/// Slots of each per-nonterminal hash table: a quarter of the budget shared
/// evenly, rounded down to a power of two, and at least
/// `initial_hash_capacity` (rounded up) and 64.
pub fn table_slots(options: SynthesisOptions, nonterminals: usize) -> usize {
    let share = options.max_pool_bytes / 4 / 8 / nonterminals.max(1);
    let budgeted = if share == 0 { 0 } else { 1usize << (usize::BITS - 1 - share.leading_zeros()) };
    budgeted
        .max(options.initial_hash_capacity.next_power_of_two())
        .max(64)
}

/// Body of `@synthesize`; returns `%solution`.
fn synthesize_body(
    problem: &SynthFun,
    plan: &SynthesisPlan,
    slots: usize,
    options: SynthesisOptions,
    arena_words: usize,
    solution_words: usize,
) -> Val {
    let grammar = problem.grammar();
    let arena_size = index(arena_words);
    let arena: Val = memref::alloc(words_type(), (&arena_size,), ()).into();
    let solution_size = index(solution_words);
    let solution: Val = memref::alloc(words_type(), (&solution_size,), ()).into();
    scf! {for 0 to solution_size step 1}.then(|word| {
        memref::store(0i64, &solution, (&word,), ());
        ().into()
    });
    let found: Val = memref::alloc(Ty::memref([Some(1)], I8), (), ()).into();
    let filter_cursor: Val = memref::alloc(Ty::memref([Some(1)], I64), (), ()).into();
    memref::store(0i8, &found, 0, ());
    let create = hashset_create();
    let capacity = Constant::int(slots as i64, I64);
    // One hash set per group: its nonterminals deduplicate against each other.
    let seen: HashMap<&str, Val> =
        grammar.groups().into_iter().map(|group| (group, create.call((&capacity,))[0].clone())).collect();

    // Every stored row has one table entry; keeping each table at most half
    // full bounds probing and makes the table memory fixed.
    let half_table = index(slots / 2);
    let groups = grammar.groups();
    let nonterminals = grammar.nonterminals();
    let number: HashMap<&str, usize> =
        nonterminals.iter().enumerate().map(|(number, nonterminal)| (nonterminal.symbol.as_str(), number)).collect();
    // Runtime state: rows stored per group, the arena cursor, and per
    // nonterminal and size the pool `(offset, rows)`. Sizes are a runtime
    // loop, so the code grows with the productions, not with the sizes and
    // their splits between children.
    let stored: Val = memref::alloc(Ty::memref([Some(groups.len() as u64)], I64), (), ()).into();
    for group in 0..groups.len() {
        memref::store(0i64, &stored, (index(group),), ());
    }
    let cursor: Val = memref::alloc(Ty::memref([Some(1)], I64), (), ()).into();
    memref::store(0i64, &cursor, 0, ());
    let sizes = options.max_expression_size + 1;
    let pools: Val =
        memref::alloc(Ty::memref([Some(nonterminals.len() as u64), Some(sizes as u64), Some(2)], I64), (), ()).into();
    let start: Val = memref::alloc(Ty::memref([Some(1)], I64), (), ()).into();
    let total: Val = memref::alloc(Ty::memref([Some(1)], I64), (), ()).into();
    let words = |bits: Val| arith::divui(arith::addi(bits, index(63), ()), index(64), ());
    let row_words = |symbol: &str, size: &Val| {
        let width = nonterminals[number[symbol]].width;
        arith::addi(index(word_count(width)), words(arith::muli(size, index(plan.token_bits), ())), ())
    };
    // The pool of `symbol` at `size`.
    let pool = |symbol: &str, size: &Val| {
        let at = |field: usize| to_index(&memref::load(&pools, (index(number[symbol]), size, index(field)), ()));
        pool_view(&arena, &at(0), &at(1), &row_words(symbol, size))
    };

    scf! {for 1 to (sizes) step 1}.then(|size| {
        let size: Val = size.into();
        for nonterminal in nonterminals {
            let symbol = nonterminal.symbol.as_str();
            let group = index(groups.iter().position(|group| *group == nonterminal.group).expect("a group"));
            let row_words = row_words(symbol, &size);
            let offset = to_index(&memref::load(&cursor, 0, ()));
            // Candidates: every rule applied to every positive split of `size - 1`.
            memref::store(0i64, &total, 0, ());
            for production in &nonterminal.productions {
                for_each_split(&size, production.children.len(), &mut |split| {
                    let rows = production
                        .children
                        .iter()
                        .zip(split)
                        .fold(index(1), |rows, (child, child_size)| arith::muli(rows, memref::dim(&pool(child, child_size), 0), ()));
                    let sum = arith::addi(memref::load(&total, 0, ()), to_i64(&rows), ());
                    memref::store(sum, &total, 0, ());
                });
            }
            let remaining = arith::subi(&arena_size, &offset, ());
            let room = arith::divui(remaining, &row_words, ());
            let table_room = arith::subi(&half_table, to_index(&memref::load(&stored, (&group,), ())), ());
            let room = arith::minui(room, table_room);
            let capacity = arith::minui(to_index(&memref::load(&total, 0, ())), room);

            memref::store(0i64, &start, 0, ());
            for production in &nonterminal.productions {
                let filter = plan.filters.iter().find(|filter| filter.rule == production.rule()).expect("planned filter");
                for_each_split(&size, production.children.len(), &mut |split| {
                    let children: Vec<Val> = production.children.iter().zip(split).map(|(child, child_size)| pool(child, child_size)).collect();
                    let rows = children.iter().fold(index(1), |rows, child| arith::muli(rows, memref::dim(child, 0), ()));
                    let searching = arith::cmpi(CmpIPredicate::Eq, memref::load(&found, 0, ()), 0);
                    let mut run = arith::andi(searching, arith::cmpi(CmpIPredicate::Ne, &rows, index(0)));
                    if filter.commutative {
                        // Rows are stored in size order, so with a larger first
                        // child every candidate is the swap of a kept one.
                        run = arith::andi(run, arith::cmpi(CmpIPredicate::Ule, &split[0], &split[1]));
                    }
                    scf! {if run}
                        .then(|| {
                            let mut operands = vec![
                                arena.clone(),
                                offset.clone(),
                                capacity.clone(),
                                row_words.clone(),
                                to_index(&memref::load(&start, 0, ())),
                                filter_cursor.clone(),
                            ];
                            operands.extend(children.iter().cloned());
                            operands.extend(split.iter().map(|child_size| arith::muli(child_size, index(plan.token_bits), ())));
                            operands.extend([seen[nonterminal.group.as_str()].clone(), solution.clone(), found.clone()]);
                            let pool = filter.signature().call(operands)[0].clone();
                            pool.annotate("no_inline", Attr::Unit);
                            memref::store(to_i64(&memref::dim(&pool, 0).into()), &start, 0, ());
                            ().into()
                        })
                        .end();
                });
            }
            let rows = memref::load(&start, 0, ());
            memref::store(to_i64(&offset), &pools, (index(number[symbol]), &size, index(0)), ());
            memref::store(&rows, &pools, (index(number[symbol]), &size, index(1)), ());
            let used = arith::muli(&rows, to_i64(&row_words), ());
            memref::store(arith::addi(to_i64(&offset), used, ()), &cursor, 0, ());
            memref::store(arith::addi(memref::load(&stored, (&group,), ()), &rows, ()), &stored, (&group,), ());
        }
        ().into()
    });
    solution
}

/// Run `body` on every split of `size - 1` into `arity` positive child sizes
/// (`index` values; the empty split when `arity` is zero and `size` is 1).
fn for_each_split(size: &Val, arity: usize, body: &mut dyn FnMut(&[Val])) {
    let rest = arith::subi(size, index(1), ());
    if arity == 0 {
        scf! {if arith::cmpi(CmpIPredicate::Eq, &rest, index(0))}
            .then(|| {
                body(&[]);
                ().into()
            })
            .end();
        return;
    }
    split_rest(&rest, arity, &mut Vec::new(), body);
}

/// Splits of `rest` into `parts` positive sizes after `prefix`.
fn split_rest(rest: &Val, parts: usize, prefix: &mut Vec<Val>, body: &mut dyn FnMut(&[Val])) {
    if parts == 1 {
        scf! {if arith::cmpi(CmpIPredicate::Uge, rest, index(1))}
            .then(|| {
                prefix.push(rest.clone());
                body(prefix);
                prefix.pop();
                ().into()
            })
            .end();
        return;
    }
    // The first part leaves at least one for each later part.
    let last = arith::subi(rest, index(parts - 1), ());
    let upper = arith::addi(&last, index(1), ());
    let bounded = arith::select(arith::cmpi(CmpIPredicate::Sge, &last, index(1)), &upper, index(1));
    scf! {for 1 to bounded step 1}.then(|head| {
        let head: Val = head.into();
        prefix.push(head.clone());
        split_rest(&arith::subi(rest, &head, ()), parts - 1, prefix, body);
        prefix.pop();
        ().into()
    });
}

/// C source of a result printer that decodes tokens to constructor names.
pub fn result_printer_source(problem: &SynthFun, plan: &SynthesisPlan, max_size: usize) -> String {
    let mut table = vec!["    {0, 0}".to_string(); plan.tokens.iter().map(|(token, _, _)| token + 1).max().unwrap_or(1)];
    for (token, name, arity) in &plan.tokens {
        table[*token] = format!("    {{{name:?}, {arity}}}");
    }
    PRINTER
        .replace("@TABLE@", &table.join(",\n"))
        .replace("@FUNCTION@", problem.name())
        .replace("@TOKEN_BITS@", &plan.token_bits.to_string())
        .replace("@MAX_SIZE@", &max_size.to_string())
}

const PRINTER: &str = r#"#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

struct token_info { const char *name; uint32_t arity; };
static const struct token_info tokens[] = {
@TABLE@
};
struct reader {
  const uint64_t *data; intptr_t offset, size, stride, token_bits, cursor, nodes;
  int malformed;
};
static int read_token(const struct reader *r, intptr_t position, uint64_t *token) {
  intptr_t bit = position * r->token_bits, word = bit / 64, shift = bit % 64;
  if (word >= r->size) return 0;
  uint64_t value = r->data[r->offset + word * r->stride] >> shift;
  if (shift && r->token_bits > 64 - shift && word + 1 < r->size)
    value |= r->data[r->offset + (word + 1) * r->stride] << (64 - shift);
  if (r->token_bits != 64) value &= (UINT64_C(1) << r->token_bits) - 1;
  *token = value;
  return 1;
}
static void print_expression(struct reader *r) {
  uint64_t id = 0;
  if (!read_token(r, r->cursor++, &id) || id == 0 ||
      id >= sizeof(tokens) / sizeof(tokens[0]) || tokens[id].name == 0) {
    fputs("<malformed>", stdout); r->malformed = 1; return;
  }
  ++r->nodes;
  fputs(tokens[id].name, stdout);
  if (!tokens[id].arity) return;
  putchar('(');
  for (uint32_t child = 0; child < tokens[id].arity; ++child) {
    if (child) fputs(", ", stdout);
    print_expression(r);
  }
  putchar(')');
}
void mlir_loom_print_result(uint64_t *allocated, uint64_t *aligned, intptr_t offset,
                            intptr_t size, intptr_t stride, intptr_t token_bits) {
  (void)allocated;
  struct reader r = {aligned, offset, size, stride, token_bits, 0, 0, 0};
  uint64_t first = 0;
  if (token_bits != @TOKEN_BITS@ || !read_token(&r, 0, &first) || first == 0) {
    printf("@FUNCTION@: no solution within %d constructors\n", @MAX_SIZE@);
    return;
  }
  fputs("@FUNCTION@ = ", stdout);
  print_expression(&r);
  putchar('\n');
}
"#;
