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
//!   %seen_<nonterminal> = mlir_loom_hashset_i64_create_handle(<table slots>)
//!
//!   <for size = 1 .. max_expression_size> {
//!     <for nonterminal> {
//!       %row_words = ceil(width / 64) + ceil(size * token_bits / 64)
//!       // Tables stay at most half full, so their memory is fixed.
//!       %capacity = min(total_candidates, remaining_arena / %row_words,
//!                       table_slots / 2 - rows_stored_<nonterminal>)
//!       // Generated for every rule and positive composition of size - 1.
//!       scf.if %found == 0 {
//!         %pool = func.call @hash_filter_<nonterminal>_<constructor>_rule<rule>(
//!             %arena, <offset/capacity/start>, %cursor, <child pools>,
//!             %seen_<nonterminal>, %solution, %found)
//!       }
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

use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationLike, OperationMutLike};
use melior::ir::{Block, Module, Value};

use super::functions::{lower_constraint, lower_tensor_fn};
use super::hash_filter::{
    lower_compute_hash, lower_hash_filter, HashFilterConfig, HASHSET_CREATE, HASHSET_INSERT,
};
use super::{invalid, Emitter, Result, POOL};
use crate::grammar::{positive_compositions, token_width};
use crate::compile::pass::fuse;
use crate::compile::TensorFn;
use crate::lowering::loops::{lower_loop_program, word_count, LoopProgram};
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
            max_expression_size: 16,
            max_pool_bytes: 256 * 1024 * 1024,
            initial_hash_capacity: 1 << 20,
        }
    }
}

/// Word programs and grammar facts computed before MLIR emission.
#[derive(Clone, Debug)]
pub struct SynthesisPlan {
    /// One packed program per constructor, in token order.
    pub constructors: Vec<LoopProgram>,
    pub constraint: LoopProgram,
    /// Fused graphs of the constructors (token order) and the constraint.
    pub fused: Vec<TensorFn>,
    pub filters: Vec<HashFilterConfig>,
    pub token_bits: usize,
    /// `(token, constructor, arity)` in token order.
    pub tokens: Vec<(usize, String, usize)>,
}

/// Validate a problem and lower every traced function to words.
pub fn plan_synthesis(problem: &SynthFun) -> Result<SynthesisPlan> {
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
    let mut fused = Vec::new();
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
                let program = lower_loop_program(&graph)?;
                fused.push(graph);
                if program.result_bits != nonterminal.width {
                    return Err(invalid(
                        STAGE,
                        format!(
                            "{} returns {} bits, but {} has width {}",
                            function.name, program.result_bits, nonterminal.symbol, nonterminal.width
                        ),
                    ));
                }
                for (((name, _), width), child) in function.inputs.iter().zip(&program.input_bits).zip(&production.children) {
                    if *width != widths[child.as_str()] {
                        return Err(invalid(
                            STAGE,
                            format!("{}: argument {name} has {width} bits but {child} has {}", function.name, widths[child.as_str()]),
                        ));
                    }
                }
                tokens.push((production.token(), production.constructor.clone(), arity));
                constructors.push(program);
            }
            filters.push(HashFilterConfig {
                nonterminal: nonterminal.symbol.clone(),
                constructor: production.constructor.clone(),
                token: production.token(),
                rule: production.rule(),
                token_bits,
                value_words: word_count(nonterminal.width),
                child_value_words: production.children.iter().map(|child| word_count(widths[child.as_str()])).collect(),
                constraint: (nonterminal.symbol == grammar.start()).then(|| CONSTRAINT.to_string()),
            });
        }
    }
    let constraint_graph = fuse(problem.constraint());
    let constraint = lower_loop_program(&constraint_graph)?;
    fused.push(constraint_graph);
    if constraint.input_bits.len() != 1 || constraint.input_bits[0] != widths[grammar.start()] {
        return Err(invalid(STAGE, "the constraint must take one value of the start nonterminal"));
    }
    Ok(SynthesisPlan {
        constructors,
        constraint,
        fused,
        filters,
        token_bits,
        tokens,
    })
}

/// Emit the complete synthesis module.
///
/// On failure no partially assembled module is returned.
pub fn lower_synthesis<'c>(
    emitter: &Emitter<'c>,
    problem: &SynthFun,
    plan: &SynthesisPlan,
    options: SynthesisOptions,
) -> Result<Module<'c>> {
    const STAGE: &str = "synthesis lowering";
    if options.max_expression_size == 0 {
        return Err(invalid(STAGE, "the maximum expression size must be positive"));
    }
    let slots = table_slots(options, problem.grammar().nonterminals().len());
    let tables = 8 * slots * problem.grammar().nonterminals().len();
    let arena_words = options.max_pool_bytes.saturating_sub(tables) / 8;
    if arena_words == 0 {
        return Err(invalid(STAGE, "the value/expression arena must hold at least one word"));
    }
    let grammar = problem.grammar();
    let mut module = Module::new(emitter.location);
    {
        let mut operation = module.as_operation_mut();
        for (name, value) in [
            ("loom.frontend", "\"loom2-tensor-fn\"".to_string()),
            ("loom.storage", "\"two-dimensional-ve-pools\"".to_string()),
            ("loom.semantic_encoding", "\"packed-i64\"".to_string()),
            ("loom.max_expression_size", format!("{} : i64", options.max_expression_size)),
            ("loom.max_pool_bytes", format!("{} : i64", options.max_pool_bytes)),
            ("loom.initial_hash_capacity", format!("{} : i64", options.initial_hash_capacity)),
        ] {
            operation.set_attribute(name, emitter.attr(&value));
        }
    }
    let body = module.body();
    let i64_type = emitter.ty("i64");

    // Runtime declarations: cpp/cpu hash set and result printer.
    body.append_operation(emitter.function(HASHSET_CREATE, &[i64_type], &[i64_type], true, &[], None));
    body.append_operation(emitter.function(HASHSET_INSERT, &[i64_type, i64_type], &[emitter.ty("i1")], true, &[], None));
    body.append_operation(emitter.function(
        PRINT_RESULT,
        &[emitter.ty("memref<?xi64>"), emitter.ty("index")],
        &[],
        true,
        &[],
        None,
    ));

    for program in &plan.constructors {
        let (function, globals) = lower_tensor_fn(emitter, program);
        for global in globals {
            body.append_operation(global);
        }
        body.append_operation(function);
    }
    let (constraint, globals) = lower_constraint(emitter, CONSTRAINT, &plan.constraint)?;
    for global in globals {
        body.append_operation(global);
    }
    body.append_operation(constraint);
    let hash_widths: BTreeSet<usize> = plan.filters.iter().map(|filter| filter.value_words).collect();
    for words in hash_widths {
        body.append_operation(lower_compute_hash(emitter, words));
    }
    for filter in &plan.filters {
        body.append_operation(lower_hash_filter(emitter, filter));
    }

    let token_map = plan
        .tokens
        .iter()
        .map(|(token, name, _)| format!("{token}={name}"))
        .collect::<Vec<_>>()
        .join(";");
    let attributes = [
        ("loom.synth_fun", emitter.attr(&format!("{:?}", problem.name()))),
        ("loom.start_sort", emitter.attr(&format!("{:?}", grammar.start()))),
        ("loom.tokens", emitter.attr(&format!("{token_map:?}"))),
        ("loom.token_bits", emitter.attr(&format!("{} : i64", plan.token_bits))),
        ("loom.max_expression_size", emitter.attr(&format!("{} : i64", options.max_expression_size))),
        ("loom.max_pool_bytes", emitter.attr(&format!("{} : i64", options.max_pool_bytes))),
    ];
    let solution_words = word_count(options.max_expression_size * plan.token_bits);
    body.append_operation(emitter.function(
        "synthesize",
        &[],
        &[emitter.ty("memref<?xi64>")],
        false,
        &attributes,
        Some(&|block: &Block<'c>| {
            emit_synthesize_body(emitter, block, problem, plan, slots, options, arena_words, solution_words)
        }),
    ));

    body.append_operation(emitter.function(
        "main",
        &[],
        &[emitter.ty("i32")],
        false,
        &[],
        Some(&|block: &Block<'c>| {
            let result = emitter.call_outlined(block, "synthesize", &[], &[emitter.ty("memref<?xi64>")]);
            let token_bits = emitter.index(block, plan.token_bits);
            emitter.call(block, PRINT_RESULT, &[result[0], token_bits], &[]);
            let zero = emitter.value(block, "arith.constant", &[], emitter.ty("i32"), &[(
                "value",
                emitter.attr("0 : i32"),
            )]);
            emitter.op(block, "func.return", &[zero], &[], &[], Vec::new());
        }),
    ));

    if !module.as_operation().verify() {
        return Err(invalid(STAGE, "the generated module failed MLIR verification"));
    }
    Ok(module)
}

fn alloc<'c, 'b>(emitter: &Emitter<'c>, block: &'b Block<'c>, ty: &str, size: Option<Value<'c, '_>>) -> Value<'c, 'b> {
    let operands: Vec<Value> = size.into_iter().collect();
    let segments = emitter.attr(&format!("array<i32: {}, 0>", operands.len()));
    emitter.value(block, "memref.alloc", &operands, emitter.ty(ty), &[("operandSegmentSizes", segments)])
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

#[allow(clippy::too_many_arguments)]
fn emit_synthesize_body<'c>(
    emitter: &Emitter<'c>,
    block: &Block<'c>,
    problem: &SynthFun,
    plan: &SynthesisPlan,
    slots: usize,
    options: SynthesisOptions,
    arena_words: usize,
    solution_words: usize,
) {
    let grammar = problem.grammar();
    let pool_type = emitter.ty(POOL);
    let zero = emitter.index(block, 0);
    let arena_size = emitter.index(block, arena_words);
    let arena = alloc(emitter, block, "memref<?xi64>", Some(arena_size));
    let solution_size = emitter.index(block, solution_words);
    let solution = alloc(emitter, block, "memref<?xi64>", Some(solution_size));
    let zero_i64 = emitter.i64(block, 0);
    emitter.for_loop(block, zero, solution_size, |body, word| {
        emitter.store(body, zero_i64, solution, &[word]);
    });
    let found = alloc(emitter, block, "memref<1xi8>", None);
    let filter_cursor = alloc(emitter, block, "memref<1xi64>", None);
    let zero_i8 = emitter.i8(block, 0);
    emitter.store(block, zero_i8, found, &[zero]);
    let capacity = emitter.i64(block, slots as u64);
    let seen: HashMap<&str, Value> = grammar
        .nonterminals()
        .iter()
        .map(|nonterminal| {
            let handle = emitter.call(block, HASHSET_CREATE, &[capacity], &[emitter.ty("i64")])[0];
            (nonterminal.symbol.as_str(), handle)
        })
        .collect();

    // Every stored row has one table entry; keeping each table at most half
    // full bounds probing and makes the table memory fixed.
    let half_table = emitter.index(block, slots / 2);
    let mut stored: HashMap<&str, Value> = grammar
        .nonterminals()
        .iter()
        .map(|nonterminal| (nonterminal.symbol.as_str(), zero))
        .collect();
    let mut cursor = zero;
    let mut pools: HashMap<(&str, usize), Value> = HashMap::new();
    for size in 1..=options.max_expression_size {
        for nonterminal in grammar.nonterminals() {
            let value_words = word_count(nonterminal.width);
            let row_words = emitter.index(block, value_words + word_count(size * plan.token_bits));
            // Every rule applied to every positive split of `size - 1`.
            let mut calls = Vec::new();
            for production in &nonterminal.productions {
                for split in positive_compositions(size - 1, production.children.len()) {
                    let filter = plan
                        .filters
                        .iter()
                        .find(|filter| filter.rule == production.rule())
                        .expect("planned filter");
                    calls.push((filter, production, split));
                }
            }
            let empty = emitter.pool_view(block, arena, cursor, zero, row_words);
            if calls.is_empty() {
                pools.insert((nonterminal.symbol.as_str(), size), empty);
                continue;
            }
            let mut total = zero;
            for (_, production, split) in &calls {
                let mut product = emitter.index(block, 1);
                for (child, &child_size) in production.children.iter().zip(split) {
                    let rows = emitter.dim(block, pools[&(child.as_str(), child_size)], 0);
                    product = emitter.binary(block, "arith.muli", product, rows);
                }
                total = emitter.binary(block, "arith.addi", total, product);
            }
            let remaining = emitter.binary(block, "arith.subi", arena_size, cursor);
            let room = emitter.binary(block, "arith.divui", remaining, row_words);
            let table_room =
                emitter.binary(block, "arith.subi", half_table, stored[nonterminal.symbol.as_str()]);
            let room = emitter.binary(block, "arith.minui", room, table_room);
            let output_capacity = emitter.binary(block, "arith.minui", total, room);

            let mut pool = empty;
            let mut start = zero;
            for (filter, production, split) in &calls {
                let state = emitter.load(block, found, &[zero], "i8");
                let searching = emitter.cmpi(block, "eq", state, zero_i8);
                let mut operands = vec![arena, cursor, output_capacity, row_words, start, filter_cursor];
                for (child, &child_size) in production.children.iter().zip(split) {
                    operands.push(pools[&(child.as_str(), child_size)]);
                }
                let bits: Vec<Value> = split
                    .iter()
                    .map(|&child_size| emitter.index(block, child_size * plan.token_bits))
                    .collect();
                operands.extend(bits);
                operands.extend([seen[nonterminal.symbol.as_str()], solution, found]);
                let previous = pool;
                pool = emitter.if_else(
                    block,
                    searching,
                    &[pool_type],
                    |then| emitter.call_outlined(then, &filter.symbol(), &operands, &[pool_type]),
                    |_| vec![previous],
                )[0];
                start = emitter.dim(block, pool, 0);
            }
            let used = emitter.binary(block, "arith.muli", start, row_words);
            cursor = emitter.binary(block, "arith.addi", cursor, used);
            let rows = emitter.binary(block, "arith.addi", stored[nonterminal.symbol.as_str()], start);
            stored.insert(nonterminal.symbol.as_str(), rows);
            pools.insert((nonterminal.symbol.as_str(), size), pool);
        }
    }
    emitter.op(block, "func.return", &[solution], &[], &[], Vec::new());
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
