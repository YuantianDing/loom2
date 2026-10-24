//! Emit the executable synthesis module with Melior.
//!
//! Only upstream MLIR dialects are used (`func`, `arith`, `scf`, `memref`,
//! `vector`). The concurrent hash set is reached through the plain C ABI in
//! `cpp/cpu/i64hashset.hpp`, and the module defines its own `main`.
//!
//! ```text
//! module attributes {loom.semantic_encoding = "packed-i64", ...} {
//!   <functions::lower_tensor_fn for each constructor>      @define_fun_*
//!   <functions::lower_constraint for the start symbol>      @sygus_constraints
//!   <hash_filter::lower_compute_hash per word count>        @compute_hash_w*
//!   <hash_filter::lower_hash_filter per production>         @hash_filter_*
//!   <synthesis::lower_synthesis entry point>                @synthesize, @main
//! }
//! ```
//!
//! Packed values are passed as `vector<Nxi64>` rather than memrefs, read from
//! and written to pools with `vector.transfer_read`/`transfer_write`, so the
//! inliner turns every constructor into vector code inside its parallel
//! filter. There is no `tensor<...xi1>` value at any stage.

pub mod functions;
pub mod hash_filter;
pub mod synthesis;

pub use functions::{lower_constraint, lower_tensor_fn};
pub use hash_filter::{lower_compute_hash, lower_hash_filter, HashFilterConfig};
pub use synthesis::{lower_synthesis, SynthesisOptions};

use melior::ir::attribute::Attribute;
use melior::ir::block::BlockLike;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::ValueLike;
use melior::ir::{Block, Identifier, Location, Operation, OperationRef, Region, RegionLike, Type, Value};
use melior::Context;

use super::{LoweringError, LoweringResult};

/// Shared Melior construction helpers.
pub struct Emitter<'c> {
    pub context: &'c Context,
    pub location: Location<'c>,
}

/// Memref type of one value/expression pool.
pub(crate) const POOL: &str = "memref<?x?xi64, strided<[?, 1], offset: ?>>";

impl<'c> Emitter<'c> {
    pub fn new(context: &'c Context) -> Self {
        Self {
            context,
            location: Location::unknown(context),
        }
    }

    pub fn ty(&self, source: &str) -> Type<'c> {
        Type::parse(self.context, source).unwrap_or_else(|| panic!("invalid MLIR type {source}"))
    }

    pub fn attr(&self, source: &str) -> Attribute<'c> {
        Attribute::parse(self.context, source)
            .unwrap_or_else(|| panic!("invalid MLIR attribute {source}"))
    }

    /// Build a detached operation.
    pub fn build(
        &self,
        name: &str,
        operands: &[Value<'c, '_>],
        results: &[Type<'c>],
        attributes: &[(&str, Attribute<'c>)],
        regions: Vec<Region<'c>>,
    ) -> Operation<'c> {
        let attributes: Vec<_> = attributes
            .iter()
            .map(|(name, value)| (Identifier::new(self.context, name), *value))
            .collect();
        OperationBuilder::new(name, self.location)
            .add_operands(operands)
            .add_results(results)
            .add_attributes(&attributes)
            .add_regions_vec(regions)
            .build()
            .unwrap_or_else(|error| panic!("failed to build {name}: {error}"))
    }

    /// Append an operation to `block`.
    pub fn op<'b>(
        &self,
        block: &'b Block<'c>,
        name: &str,
        operands: &[Value<'c, '_>],
        results: &[Type<'c>],
        attributes: &[(&str, Attribute<'c>)],
        regions: Vec<Region<'c>>,
    ) -> OperationRef<'c, 'b> {
        block.append_operation(self.build(name, operands, results, attributes, regions))
    }

    /// Append a single-result operation and return its value.
    pub fn value<'b>(
        &self,
        block: &'b Block<'c>,
        name: &str,
        operands: &[Value<'c, '_>],
        result: Type<'c>,
        attributes: &[(&str, Attribute<'c>)],
    ) -> Value<'c, 'b> {
        self.op(block, name, operands, &[result], attributes, Vec::new())
            .result(0)
            .expect("single result")
            .into()
    }

    pub fn results<'b>(operation: OperationRef<'c, 'b>) -> Vec<Value<'c, 'b>> {
        (0..operation.result_count())
            .map(|index| operation.result(index).expect("result").into())
            .collect()
    }

    pub fn i64<'b>(&self, block: &'b Block<'c>, value: u64) -> Value<'c, 'b> {
        let attribute = self.attr(&format!("{} : i64", value as i64));
        self.value(block, "arith.constant", &[], self.ty("i64"), &[("value", attribute)])
    }

    pub fn i8<'b>(&self, block: &'b Block<'c>, value: i8) -> Value<'c, 'b> {
        let attribute = self.attr(&format!("{value} : i8"));
        self.value(block, "arith.constant", &[], self.ty("i8"), &[("value", attribute)])
    }

    pub fn index<'b>(&self, block: &'b Block<'c>, value: usize) -> Value<'c, 'b> {
        let attribute = self.attr(&format!("{value} : index"));
        self.value(block, "arith.constant", &[], self.ty("index"), &[("value", attribute)])
    }

    /// Binary arithmetic whose result type equals the operand type.
    pub fn binary<'b>(
        &self,
        block: &'b Block<'c>,
        name: &str,
        left: Value<'c, '_>,
        right: Value<'c, '_>,
    ) -> Value<'c, 'b> {
        self.value(block, name, &[left, right], left.r#type(), &[])
    }

    pub fn cmpi<'b>(
        &self,
        block: &'b Block<'c>,
        predicate: &str,
        left: Value<'c, '_>,
        right: Value<'c, '_>,
    ) -> Value<'c, 'b> {
        let code = match predicate {
            "eq" => 0,
            "ne" => 1,
            "ult" => 6,
            other => panic!("unsupported predicate {other}"),
        };
        let attribute = self.attr(&format!("{code} : i64"));
        self.value(block, "arith.cmpi", &[left, right], self.ty("i1"), &[("predicate", attribute)])
    }

    pub fn index_cast<'b>(&self, block: &'b Block<'c>, value: Value<'c, '_>, to: &str) -> Value<'c, 'b> {
        self.value(block, "arith.index_cast", &[value], self.ty(to), &[])
    }

    pub fn call<'b>(
        &self,
        block: &'b Block<'c>,
        callee: &str,
        operands: &[Value<'c, '_>],
        results: &[Type<'c>],
    ) -> Vec<Value<'c, 'b>> {
        let callee = self.attr(&format!("@{}", quote_symbol(callee)));
        Self::results(self.op(block, "func.call", operands, results, &[("callee", callee)], Vec::new()))
    }

    /// A call the inliner must keep, preserving the callee's allocation scope.
    pub fn call_outlined<'b>(
        &self,
        block: &'b Block<'c>,
        callee: &str,
        operands: &[Value<'c, '_>],
        results: &[Type<'c>],
    ) -> Vec<Value<'c, 'b>> {
        let attributes = [
            ("callee", self.attr(&format!("@{}", quote_symbol(callee)))),
            ("no_inline", self.attr("unit")),
        ];
        Self::results(self.op(block, "func.call", operands, results, &attributes, Vec::new()))
    }

    pub fn load<'b>(&self, block: &'b Block<'c>, memref: Value<'c, '_>, indices: &[Value<'c, '_>], element: &str) -> Value<'c, 'b> {
        let mut operands = vec![memref];
        operands.extend_from_slice(indices);
        self.value(block, "memref.load", &operands, self.ty(element), &[])
    }

    pub fn store(&self, block: &Block<'c>, value: Value<'c, '_>, memref: Value<'c, '_>, indices: &[Value<'c, '_>]) {
        let mut operands = vec![value, memref];
        operands.extend_from_slice(indices);
        self.op(block, "memref.store", &operands, &[], &[], Vec::new());
    }

    /// `memref.atomic_rmw <kind> %value, %memref[%index]`; returns the old value.
    pub fn atomic_rmw<'b>(
        &self,
        block: &'b Block<'c>,
        kind: &str,
        value: Value<'c, '_>,
        memref: Value<'c, '_>,
        index: Value<'c, '_>,
    ) -> Value<'c, 'b> {
        let code = match kind {
            "addi" => 1,
            "ori" => 14,
            other => panic!("unsupported atomic kind {other}"),
        };
        let attribute = self.attr(&format!("{code} : i64"));
        self.value(block, "memref.atomic_rmw", &[value, memref, index], value.r#type(), &[("kind", attribute)])
    }

    /// `vector.transfer_read %pool[%row, 0] : vector<words x i64>`.
    pub fn transfer_read<'b>(
        &self,
        block: &'b Block<'c>,
        pool: Value<'c, '_>,
        row: Value<'c, '_>,
        words: usize,
    ) -> Value<'c, 'b> {
        let zero = self.index(block, 0);
        let padding = self.i64(block, 0);
        let attributes = [
            ("permutation_map", self.attr("affine_map<(d0, d1) -> (d1)>")),
            ("in_bounds", self.attr("[true]")),
            ("operandSegmentSizes", self.attr("array<i32: 1, 2, 1, 0>")),
        ];
        self.value(
            block,
            "vector.transfer_read",
            &[pool, row, zero, padding],
            self.ty(&format!("vector<{words}xi64>")),
            &attributes,
        )
    }

    /// `vector.transfer_write %value, %pool[%row, 0]`.
    pub fn transfer_write(&self, block: &Block<'c>, value: Value<'c, '_>, pool: Value<'c, '_>, row: Value<'c, '_>) {
        let zero = self.index(block, 0);
        let attributes = [
            ("permutation_map", self.attr("affine_map<(d0, d1) -> (d1)>")),
            ("in_bounds", self.attr("[true]")),
            ("operandSegmentSizes", self.attr("array<i32: 1, 1, 2, 0>")),
        ];
        self.op(block, "vector.transfer_write", &[value, pool, row, zero], &[], &attributes, Vec::new());
    }

    pub fn dim<'b>(&self, block: &'b Block<'c>, memref: Value<'c, '_>, dimension: usize) -> Value<'c, 'b> {
        let dimension = self.index(block, dimension);
        self.value(block, "memref.dim", &[memref, dimension], self.ty("index"), &[])
    }

    /// View `rows x row_words` words of `arena` starting at word `offset`.
    pub fn pool_view<'b>(
        &self,
        block: &'b Block<'c>,
        arena: Value<'c, '_>,
        offset: Value<'c, '_>,
        rows: Value<'c, '_>,
        row_words: Value<'c, '_>,
    ) -> Value<'c, 'b> {
        let dynamic = i64::MIN;
        let attributes = [
            ("static_offsets", self.attr(&format!("array<i64: {dynamic}>"))),
            ("static_sizes", self.attr(&format!("array<i64: {dynamic}, {dynamic}>"))),
            ("static_strides", self.attr(&format!("array<i64: {dynamic}, 1>"))),
            ("operandSegmentSizes", self.attr("array<i32: 1, 1, 2, 1>")),
        ];
        self.value(
            block,
            "memref.reinterpret_cast",
            &[arena, offset, rows, row_words, row_words],
            self.ty(POOL),
            &attributes,
        )
    }

    /// `scf.if` without results whose then-region is built by `then`.
    pub fn if_then(&self, block: &Block<'c>, condition: Value<'c, '_>, then: impl FnOnce(&Block<'c>)) {
        let body = Block::new(&[]);
        then(&body);
        self.op(&body, "scf.yield", &[], &[], &[], Vec::new());
        self.op(block, "scf.if", &[condition], &[], &[], vec![region(body), Region::new()]);
    }

    /// `scf.if` with results; each closure returns the values to yield.
    pub fn if_else<'b>(
        &self,
        block: &'b Block<'c>,
        condition: Value<'c, '_>,
        results: &[Type<'c>],
        then: impl for<'x> FnOnce(&'x Block<'c>) -> Vec<Value<'c, 'x>>,
        otherwise: impl for<'x> FnOnce(&'x Block<'c>) -> Vec<Value<'c, 'x>>,
    ) -> Vec<Value<'c, 'b>> {
        let then_block = Block::new(&[]);
        let yielded = then(&then_block);
        self.op(&then_block, "scf.yield", &yielded, &[], &[], Vec::new());
        let else_block = Block::new(&[]);
        let yielded = otherwise(&else_block);
        self.op(&else_block, "scf.yield", &yielded, &[], &[], Vec::new());
        Self::results(self.op(block, "scf.if", &[condition], results, &[], vec![region(then_block), region(else_block)]))
    }

    /// `scf.for %i = lower to upper step 1` without loop-carried values.
    pub fn for_loop(
        &self,
        block: &Block<'c>,
        lower: Value<'c, '_>,
        upper: Value<'c, '_>,
        body: impl FnOnce(&Block<'c>, Value<'c, '_>),
    ) {
        let step = self.index(block, 1);
        let inner = Block::new(&[(self.ty("index"), self.location)]);
        let induction: Value = inner.argument(0).expect("induction variable").into();
        body(&inner, induction);
        self.op(&inner, "scf.yield", &[], &[], &[], Vec::new());
        self.op(block, "scf.for", &[lower, upper, step], &[], &[], vec![region(inner)]);
    }

    /// A `func.func` with the given signature; `body` fills the entry block.
    pub fn function(
        &self,
        name: &str,
        arguments: &[Type<'c>],
        results: &[Type<'c>],
        private: bool,
        attributes: &[(&str, Attribute<'c>)],
        body: Option<&dyn Fn(&Block<'c>)>,
    ) -> Operation<'c> {
        let signature = format!(
            "({}) -> ({})",
            arguments.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
            results.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
        );
        let mut all_attributes = vec![
            ("sym_name", self.attr(&format!("\"{name}\""))),
            ("function_type", self.attr(&signature)),
        ];
        if private {
            all_attributes.push(("sym_visibility", self.attr("\"private\"")));
        }
        all_attributes.extend_from_slice(attributes);
        let region = Region::new();
        if let Some(body) = body {
            let entry = Block::new(&arguments.iter().map(|&ty| (ty, self.location)).collect::<Vec<_>>());
            body(&entry);
            region.append_block(entry);
        }
        self.build("func.func", &[], &[], &all_attributes, vec![region])
    }
}

pub(crate) fn region<'c>(block: Block<'c>) -> Region<'c> {
    let region = Region::new();
    region.append_block(block);
    region
}

/// Collect a block's arguments.
pub fn arguments<'c, 'b>(block: &'b Block<'c>) -> Vec<Value<'c, 'b>> {
    (0..block.argument_count())
        .map(|index| block.argument(index).expect("argument").into())
        .collect()
}

/// Make a constructor name safe for an MLIR symbol: `re.++` -> `re._2b__2b_`.
pub fn symbol_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '_' | '.') {
                character.to_string()
            } else {
                format!("_{:x}_", character as u32)
            }
        })
        .collect()
}

fn quote_symbol(name: &str) -> String {
    if name.chars().all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '.' | '$')) {
        name.to_string()
    } else {
        format!("\"{name}\"")
    }
}

pub(crate) fn invalid(stage: &'static str, message: impl Into<String>) -> LoweringError {
    LoweringError::InvalidInput {
        stage,
        message: message.into(),
    }
}

/// Load every upstream dialect and translation into a fresh context.
pub fn new_context() -> Context {
    let registry = melior::dialect::DialectRegistry::new();
    melior::utility::register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    melior::utility::register_all_llvm_translations(&context);
    context
}

pub(crate) type Result<T> = LoweringResult<T>;
