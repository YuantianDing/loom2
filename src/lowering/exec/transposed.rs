//! Transposed constructors: 64 candidates at once.
//!
//! A constructor whose fused graph has a `gather` or `gather2` is also built
//! transposed: every logical bit of a value is one `i64` word whose bit `k`
//! belongs to candidate `k`, so each step of a gather combines one bit of 64
//! candidates with a single word operation:
//!
//! ```text
//! func.func private @define_fun_transposed_<name>(
//!     %arg0: memref<<arg0_bits>xi64>, ..., %out: memref<<result_bits>xi64>)
//!   <input>          the argument buffer
//!   <const>          a table of 0 / -1 words
//!   <and, or, xor>   word by word over all bits
//!   <gather(2)>      acc[O[k]] = op(acc[O[k]], inner(a[I1[k]], b[I2[k]]))
//!   <flat permute>   a gather.or of one element per row
//! ```
//!
//! Temporaries are stack buffers, reused once their last consumer is built.
//! A production filter (see `hash_filter`) turns blocks of 64 pool rows into
//! transposed form with 64x64 bit transposes ([`transpose_rows`]), and the result
//! back into 64 packed values ([`untranspose_rows`]).

use std::collections::HashMap;

use high_mlir::arith::{self, CmpIPredicate, Constant};
use high_mlir::func::Signature;
use high_mlir::{memref, scf, Attr, Ty, Tuple, Val, I32, I64};

use super::functions::{define_with_tables, result_bits, Tables};
use super::{index, invalid, symbol_name, to_index, Result};
use crate::compile::typing::{expand_reduce_form, ExpandReduceForm, TypeCache};
use crate::compile::{LogicalOp, TensorExpr, TensorFn};

const STAGE: &str = "transposed lowering";

/// Name of the transposed function of `constructor`.
pub fn transposed_symbol(constructor: &str) -> String {
    format!("define_fun_transposed_{}", symbol_name(constructor))
}

/// `memref<<bits>xi64>`, a transposed value.
pub fn transposed_type(bits: usize) -> Ty {
    Ty::memref([Some(bits as u64)], I64)
}

/// `@define_fun_transposed_<constructor>` for arguments and a result of these widths.
pub fn transposed_signature(constructor: &str, argument_bits: &[usize], result_bits: usize) -> Signature {
    let inputs = argument_bits.iter().chain([&result_bits]).map(|&bits| transposed_type(bits));
    Signature::new(transposed_symbol(constructor), inputs, [])
}

/// The operands of a fused node.
fn children(expression: &TensorExpr) -> Vec<&TensorExpr> {
    match expression {
        TensorExpr::Input(_) | TensorExpr::Constant(_) | TensorExpr::ScalarConstant(_) => Vec::new(),
        TensorExpr::BVLogical(logical) => logical.operands.iter().map(|operand| &*operand.expr).collect(),
        TensorExpr::BVGather(gather) => vec![&*gather.expr],
        TensorExpr::BVGather2(gather) => vec![&*gather.a, &*gather.b],
        TensorExpr::BVJaggedPermute(permute) => vec![&*permute.expr],
        TensorExpr::BTShift(shift) => vec![&*shift.expr],
        TensorExpr::BTExpandReduce(change) => vec![&*change.expr],
    }
}

/// Whether a fused node can be built transposed, and whether it is a gather.
fn supported(expression: &TensorExpr, types: &mut TypeCache) -> Option<bool> {
    if !types.infer(expression).ok()?.is_flat() {
        return None;
    }
    match expression {
        TensorExpr::Input(_) | TensorExpr::Constant(_) | TensorExpr::ScalarConstant(_) => Some(false),
        TensorExpr::BVLogical(logical) => logical.operands.iter().all(|operand| operand.reduce_dims.is_empty()).then_some(false),
        TensorExpr::BVGather(_) | TensorExpr::BVGather2(_) => Some(true),
        TensorExpr::BVJaggedPermute(permute) => {
            (0..permute.indices.row_count()).all(|row| permute.indices.row(row).len() <= 1).then_some(false)
        }
        TensorExpr::BTExpandReduce(change) => {
            let input = types.infer(&change.expr).ok()?;
            matches!(expand_reduce_form(change, &input).ok()?, ExpandReduceForm::Dense(_)).then_some(true)
        }
        TensorExpr::BTShift(_) => None,
    }
}

/// Whether the (fused) constructor `function` is built transposed: it takes
/// arguments, has a gather, and every node is flat and supported.
pub fn is_transposed(function: &TensorFn) -> bool {
    if function.inputs.is_empty() {
        return false;
    }
    let mut types = TypeCache::default();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![&function.expression];
    let mut gathers = false;
    while let Some(expression) = stack.pop() {
        if !seen.insert(expression as *const TensorExpr) {
            continue;
        }
        match supported(expression, &mut types) {
            None => return false,
            Some(gather) => gathers |= gather,
        }
        stack.extend(children(expression));
    }
    gathers
}

/// One gather, as flat element lists: `out[k] = op(out[k], inner(a[in_1[k]], b[in_2[k]]))`.
struct Elements<'e> {
    op: LogicalOp,
    inner: LogicalOp,
    sources: Vec<&'e TensorExpr>,
    indices: Vec<&'e [usize]>,
    out: &'e [usize],
    width: usize,
}

struct TransposedLowerer<'a, 't> {
    inputs: &'a [(String, crate::compile::TensorType)],
    parameters: Vec<Val>,
    types: TypeCache,
    tables: &'a mut Tables<'t>,
    /// Buffer of every node built, and whether it is a temporary.
    memo: HashMap<*const TensorExpr, (Val, bool)>,
    /// Consumers of every node not built yet.
    uses: HashMap<*const TensorExpr, usize>,
    /// Released temporaries, by bits.
    free: HashMap<usize, Vec<Val>>,
}

impl TransposedLowerer<'_, '_> {
    fn width(&mut self, expression: &TensorExpr) -> Result<usize> {
        Ok(self.types.infer(expression).map_err(|error| invalid(STAGE, error))?.width())
    }

    /// A temporary of `bits` words.
    fn buffer(&mut self, bits: usize) -> Val {
        match self.free.get_mut(&bits).and_then(Vec::pop) {
            Some(buffer) => buffer,
            None => memref::alloca(transposed_type(bits), (), ()).into(),
        }
    }

    /// A read-only table of `words`.
    fn words(&mut self, words: Vec<i64>) -> Val {
        self.tables.get(I64, words)
    }

    /// `buffer[i] = value` for every word.
    fn fill(buffer: &Val, bits: usize, value: i64) {
        let value = Constant::int(value, I64);
        scf! {for 0 to (bits) step 1}.then(|i| {
            memref::store(&value, buffer, (&i,), ());
            ().into()
        });
    }

    /// The buffer of `expression`, building it (and its operands) if needed.
    fn expression(&mut self, expression: &TensorExpr) -> Result<Val> {
        let key = expression as *const TensorExpr;
        if let Some((buffer, _)) = self.memo.get(&key) {
            return Ok(buffer.clone());
        }
        let support = self.types.infer(expression).map_err(|error| invalid(STAGE, error))?;
        let built = if support.support() == 0 && !matches!(expression, TensorExpr::Input(_)) {
            let words = vec![0; support.width()];
            (self.words(words), false)
        } else {
            self.lower(expression)?
        };
        for child in children(expression) {
            self.release(child);
        }
        self.memo.insert(key, built.clone());
        Ok(built.0)
    }

    /// One consumer of `expression` is built; free its buffer after the last.
    fn release(&mut self, expression: &TensorExpr) {
        let key = expression as *const TensorExpr;
        let Some(uses) = self.uses.get_mut(&key) else { return };
        *uses -= 1;
        if *uses == 0 {
            if let Some((buffer, true)) = self.memo.get(&key) {
                let bits = buffer.ty().shape().and_then(|shape| shape[0]).expect("static") as usize;
                self.free.entry(bits).or_default().push(buffer.clone());
            }
        }
    }

    fn lower(&mut self, expression: &TensorExpr) -> Result<(Val, bool)> {
        let fail = |message: String| invalid(STAGE, message);
        Ok(match expression {
            TensorExpr::Input(input) => {
                let argument = self
                    .inputs
                    .iter()
                    .position(|(name, _)| name == &input.name)
                    .ok_or_else(|| fail(format!("unknown input {:?}", input.name)))?;
                (self.parameters[argument].clone(), false)
            }
            TensorExpr::Constant(constant) => {
                let data = constant.value.data();
                let words = (0..constant.tensor_type.width()).map(|bit| -(((data[bit / 64] >> (bit % 64)) & 1) as i64)).collect();
                (self.words(words), false)
            }
            TensorExpr::ScalarConstant(constant) => {
                let words = (0..64).map(|bit| -(((constant.value >> bit) & 1) as i64)).collect();
                (self.words(words), false)
            }
            TensorExpr::BVLogical(logical) => {
                let operands = logical
                    .operands
                    .iter()
                    .map(|operand| self.expression(&operand.expr))
                    .collect::<Result<Vec<_>>>()?;
                let bits = self.width(expression)?;
                let out = self.buffer(bits);
                scf! {for 0 to (bits) step 1}.then(|i| {
                    let mut value = memref::load(&operands[0], (&i,), ());
                    for operand in &operands[1..] {
                        value = logic(logical.op, &value, &memref::load(operand, (&i,), ()));
                    }
                    memref::store(value, &out, (&i,), ());
                    ().into()
                });
                (out, true)
            }
            TensorExpr::BVGather(gather) => {
                gather.validate().map_err(fail)?;
                let elements = Elements {
                    op: gather.op,
                    inner: LogicalOp::Or,
                    sources: vec![&gather.expr],
                    indices: vec![gather.indices_in.data()],
                    out: gather.indices_out.data(),
                    width: gather.width,
                };
                (self.gather(&elements)?, true)
            }
            TensorExpr::BVGather2(gather) => {
                gather.validate().map_err(fail)?;
                let elements = Elements {
                    op: gather.op,
                    inner: gather.inner,
                    sources: vec![&gather.a, &gather.b],
                    indices: vec![gather.indices_1.data(), gather.indices_2.data()],
                    out: gather.indices_out.data(),
                    width: gather.width,
                };
                (self.gather(&elements)?, true)
            }
            TensorExpr::BVJaggedPermute(permute) => {
                // Rows have at most one lane: a gather.or of one element per row.
                let (mut out, mut indices) = (Vec::new(), Vec::new());
                for row in 0..permute.indices.row_count() {
                    if let Some(&index) = permute.indices.row(row).first() {
                        out.push(row);
                        indices.push(index);
                    }
                }
                let elements = Elements {
                    op: LogicalOp::Or,
                    inner: LogicalOp::Or,
                    sources: vec![&permute.expr],
                    indices: vec![&indices],
                    out: &out,
                    width: permute.indices.row_count(),
                };
                (self.gather(&elements)?, true)
            }
            TensorExpr::BTExpandReduce(change) => {
                let input = self.types.infer(&change.expr).map_err(fail)?;
                let ExpandReduceForm::Dense(gather) = expand_reduce_form(change, &input).map_err(fail)? else {
                    return Err(fail("a lane fold is not transposed".into()));
                };
                let elements = Elements {
                    op: gather.op,
                    inner: LogicalOp::Or,
                    sources: vec![&change.expr],
                    indices: vec![gather.indices_in.data()],
                    out: gather.indices_out.data(),
                    width: gather.width,
                };
                (self.gather(&elements)?, true)
            }
            TensorExpr::BTShift(_) => return Err(fail("shifts are not transposed".into())),
        })
    }

    /// A gather as one loop over its elements, each updating one word.
    /// Elements whose value is known zero (from the sources' support) take
    /// no step: they leave an OR/XOR accumulator unchanged and clear the
    /// initial word of an AND.
    fn gather(&mut self, elements: &Elements) -> Result<Val> {
        let mut lives = Vec::new();
        let mut buffers = Vec::new();
        for source in &elements.sources {
            let support = self.types.infer(source).map_err(|error| invalid(STAGE, error))?;
            lives.push(support.mask().data().to_vec());
            buffers.push(self.expression(source)?);
        }
        let live = |side: usize, index: usize| lives[side][index / 64] & (1u64 << (index % 64)) != 0;
        let identity = if elements.op == LogicalOp::And { -1 } else { 0 };
        let mut init = vec![identity; elements.width];
        let mut indices = vec![Vec::new(); elements.sources.len()];
        let mut destinations = Vec::new();
        for (element, &row) in elements.out.iter().enumerate() {
            let alive: Vec<bool> = (0..elements.sources.len()).map(|side| live(side, elements.indices[side][element])).collect();
            let always_zero = match elements.inner {
                _ if alive.len() == 1 => !alive[0],
                LogicalOp::And => alive.iter().any(|alive| !alive),
                LogicalOp::Or | LogicalOp::Xor => alive.iter().all(|alive| !alive),
            };
            if always_zero {
                if elements.op == LogicalOp::And {
                    init[row] = 0;
                }
                continue;
            }
            for (side, side_indices) in indices.iter_mut().enumerate() {
                side_indices.push(elements.indices[side][element] as i64);
            }
            destinations.push(row as i64);
        }

        let out = self.buffer(elements.width);
        if init.iter().all(|&word| word == identity) {
            Self::fill(&out, elements.width, identity);
        } else {
            let table = self.words(init);
            scf! {for 0 to (elements.width) step 1}.then(|i| {
                memref::store(memref::load(&table, (&i,), ()), &out, (&i,), ());
                ().into()
            });
        }
        if destinations.is_empty() {
            return Ok(out);
        }
        let steps = destinations.len();
        let tables: Vec<Val> = indices.into_iter().map(|data| self.tables.get(I32, data)).collect();
        let destinations = self.tables.get(I32, destinations);
        let step_body = |step: &Val| {
            let mut value: Option<Val> = None;
            for (table, buffer) in tables.iter().zip(&buffers) {
                let word = memref::load(buffer, (&to_index(&memref::load(table, (step,), ())),), ());
                value = Some(match value {
                    None => word,
                    Some(value) => logic(elements.inner, &value, &word),
                });
            }
            let destination = to_index(&memref::load(&destinations, (step,), ()));
            let previous = memref::load(&out, (&destination,), ());
            memref::store(logic(elements.op, &previous, &value.expect("a gather has a source")), &out, (&destination,), ());
        };
        if steps == 1 {
            step_body(&index(0));
        } else {
            let unrolling_factor : usize = 4;
            let total_in_loop : usize = steps / unrolling_factor * unrolling_factor;
            scf! {for 0 to (total_in_loop) step (unrolling_factor)}.then(|step| {
                let mut step: Val = step.into();
                for _ in 0..unrolling_factor {
                    step_body(&step);
                    step = arith::addi(&step, &index(1), ());
                }
                ().into()
            });

            if total_in_loop < steps {
                scf! {for (total_in_loop) to (steps) step 1}.then(|step| {
                    step_body(&step.into());
                    ().into()
                });
            }
        }
        Ok(out)
    }
}

fn logic(op: LogicalOp, a: &Val, b: &Val) -> Val {
    match op {
        LogicalOp::And => arith::andi(a, b),
        LogicalOp::Or => arith::ori(a, b),
        LogicalOp::Xor => arith::xori(a, b),
    }
}

/// The nodes of a fused graph that consume each node.
fn uses(root: &TensorExpr) -> HashMap<*const TensorExpr, usize> {
    let mut uses = HashMap::new();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(expression) = stack.pop() {
        if !seen.insert(expression as *const TensorExpr) {
            continue;
        }
        for child in children(expression) {
            *uses.entry(child as *const TensorExpr).or_insert(0) += 1;
            stack.push(child);
        }
    }
    uses
}

/// Emit `@define_fun_transposed_<name>` (see the module documentation) into the
/// module being built.
pub fn lower_transposed_tensor_fn(function: &TensorFn) -> Result<Val> {
    let result_bits = result_bits(function)?;
    let argument_bits: Vec<usize> = function.inputs.iter().map(|(_, tensor_type)| tensor_type.width()).collect();
    let signature = transposed_signature(&function.name, &argument_bits, result_bits);
    let define = |tables: &mut Tables| -> Result<Val> {
        let mut failure = None;
        let defined = signature.try_define(|parameters| {
            let mut lowerer = TransposedLowerer {
                inputs: &function.inputs,
                parameters: parameters[..argument_bits.len()].to_vec(),
                types: TypeCache::default(),
                tables: &mut *tables,
                memo: HashMap::new(),
                uses: uses(&function.expression),
                free: HashMap::new(),
            };
            match lowerer.expression(&function.expression) {
                Ok(result) => {
                    let out = &parameters[argument_bits.len()];
                    scf! {for 0 to (result_bits) step 1}.then(|i| {
                        memref::store(memref::load(&result, (&i,), ()), out, (&i,), ());
                        ().into()
                    });
                }
                Err(error) => failure = Some(error),
            }
            Tuple(Vec::new())
        });
        if let Some(error) = failure {
            return Err(error);
        }
        defined.map_err(|error| invalid(STAGE, error.to_string()))
    };
    let function_val = define_with_tables(&symbol_name(&format!("transposed_{}", function.name)), define)?;
    function_val.annotate("sym_visibility", Attr::str("private")).annotate("loom.constructor", Attr::str(&function.name));
    Ok(function_val)
}

/// Symbol of the 64x64 bit transpose.
pub const TRANSPOSE: &str = "loom_transpose64";

fn transpose_signature() -> Signature {
    Signature::new(TRANSPOSE, [Ty::memref([Some(64)], I64)], [])
}

/// Emit `@loom_transpose64(%block: memref<64xi64>)`: in place, bit `k` of
/// word `j` becomes bit `j` of word `k` (six stages of masked swaps).
pub fn lower_transpose() -> Val {
    let function = transpose_signature().define(|arguments| {
        let block = &arguments[0];
        let stages: [(u64, u64); 6] = [
            (32, 0x0000_0000_ffff_ffff),
            (16, 0x0000_ffff_0000_ffff),
            (8, 0x00ff_00ff_00ff_00ff),
            (4, 0x0f0f_0f0f_0f0f_0f0f),
            (2, 0x3333_3333_3333_3333),
            (1, 0x5555_5555_5555_5555),
        ];
        for (width, mask) in stages {
            let shift = width.trailing_zeros() as usize;
            scf! {for 0 to 32 step 1}.then(|i| {
                // The `i`-th index with bit `width` clear, and its partner.
                let low = arith::andi(&i, index(width as usize - 1));
                let high = arith::shli(arith::shrui(&i, index(shift), ()), index(shift + 1), ());
                let k = arith::ori(high, low);
                let partner = arith::addi(&k, index(width as usize), ());
                let a = memref::load(block, (&k,), ());
                let b = memref::load(block, (&partner,), ());
                let width = Constant::int(width as i64, I64);
                let swapped = arith::andi(arith::xori(arith::shrui(&a, &width, ()), &b), Constant::int(mask as i64, I64));
                memref::store(arith::xori(&b, &swapped), block, (&partner,), ());
                memref::store(arith::xori(&a, arith::shli(&swapped, &width, ())), block, (&k,), ());
                ().into()
            });
        }
        Tuple(Vec::new())
    });
    function.annotate("sym_visibility", Attr::str("private"));
    function
}

/// `x` rotated right by `shift` (an `i64` in `0..64`): bit `k` is bit `k + shift`.
pub fn rotate_right(x: &Val, shift: &Val) -> Val {
    let back = arith::andi(arith::subi(Constant::int(64, I64), shift, ()), Constant::int(63, I64));
    arith::ori(arith::shrui(x, shift, ()), arith::shli(x, back, ()))
}

/// Lanes `0..count` (an `index` in `0..=64`) as an `i64` mask.
pub fn lane_mask(count: &Val) -> Val {
    let count = arith::index_cast(count, I64);
    let partial = arith::subi(arith::shli(Constant::int(1, I64), &count, ()), Constant::int(1, I64), ());
    arith::select(arith::cmpi(CmpIPredicate::Uge, &count, 64), Constant::int(-1, I64), partial).into()
}

/// Turn `count` (at most 64) rows of `words`-word values, read by `row(lane,
/// word)`, into the transposed `out` of `bits` words, every word rotated right
/// by `shift`: lane `k` of `out` holds row `(k + shift) % 64`. Lanes from
/// `count` on are zero. `scratch` is a `memref<64xi64>`.
pub fn transpose_rows(row: &dyn Fn(&Val, &Val) -> Val, count: &Val, words: usize, bits: usize, shift: &Val, scratch: &Val, out: &Val) {
    let transpose = transpose_signature();
    scf! {for 0 to (words) step 1}.then(|word| {
        let word: Val = word.into();
        scf! {for 0 to 64 step 1}.then(|lane| {
            let lane: Val = lane.into();
            let [value] = scf! {if arith::cmpi(CmpIPredicate::Ult, &lane, count)}
                .then(|| row(&lane, &word).into())
                .else_(|| Constant::int(0, I64).into())
                .try_into()
                .unwrap();
            memref::store(value, scratch, (&lane,), ());
            ().into()
        });
        transpose.call((scratch,));
        let first = arith::muli(&word, index(64), ());
        let rows = arith::minui(index(64), arith::subi(index(bits), &first, ()));
        scf! {for 0 to rows step 1}.then(|bit| {
            let value = memref::load(scratch, (&bit,), ());
            memref::store(rotate_right(&value, shift), out, (&arith::addi(&first, &bit, ()),), ());
            ().into()
        });
        ().into()
    });
}

/// Turn the transposed `transposed` of `bits` words into 64 values of `words` words
/// each, stored by `store(lane, word, value)`. `scratch` is a `memref<64xi64>`.
pub fn untranspose_rows(transposed: &Val, bits: usize, words: usize, scratch: &Val, store: &dyn Fn(&Val, &Val, &Val)) {
    let transpose = transpose_signature();
    scf! {for 0 to (words) step 1}.then(|word| {
        let word: Val = word.into();
        let first = arith::muli(&word, index(64), ());
        scf! {for 0 to 64 step 1}.then(|j| {
            let bit = arith::addi(&first, &j, ());
            let [value] = scf! {if arith::cmpi(CmpIPredicate::Ult, &bit, index(bits))}
                .then(|| memref::load(transposed, (&bit,), ()).into())
                .else_(|| Constant::int(0, I64).into())
                .try_into()
                .unwrap();
            memref::store(value, scratch, (&j,), ());
            ().into()
        });
        transpose.call((scratch,));
        scf! {for 0 to 64 step 1}.then(|lane| {
            let lane: Val = lane.into();
            store(&lane, &word, &memref::load(scratch, (&lane,), ()));
            ().into()
        });
        ().into()
    });
}

/// An `index` as `i64`.
pub fn to_i64(value: &Val) -> Val {
    arith::index_cast(value, I64).into()
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use super::*;
    use crate::compile::pass::fuse;
    use crate::compile::{
        BTExpandReduce, BVJaggedPermute, BVLogical, Constant as TensorConstant, ExpandReduceDimType as D, Input,
        JTReduced, JaggedTensor, TensorType,
    };
    use crate::lowering::jit::JitFunction;
    use crate::lowering::packed::evaluate_tensor_fn;
    use crate::lowering::word_count;

    fn random_words(seed: &mut u64, width: usize) -> Vec<u64> {
        (0..word_count(width))
            .map(|word| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                let bits = width - word * 64;
                if bits >= 64 { *seed } else { *seed & ((1u64 << bits) - 1) }
            })
            .collect()
    }

    /// The transposed JIT of `function` (at -O0 and -O3) against the
    /// reference evaluator, on 64 random candidates at once.
    fn check(function: &TensorFn) {
        assert!(is_transposed(&fuse(function)), "{} is transposed", function.name);
        let jits: Vec<_> = [0, 3]
            .into_iter()
            .map(|level| JitFunction::compile_transposed(function, level).unwrap_or_else(|error| panic!("{error}")))
            .collect();
        let mut seed = 0x2545f4914f6cdd1d;
        for _ in 0..3 {
            let lanes: Vec<Vec<Vec<u64>>> = (0..64)
                .map(|_| function.inputs.iter().map(|(_, tensor_type)| random_words(&mut seed, tensor_type.width())).collect())
                .collect();
            let inputs: Vec<u64> = lanes.iter().flat_map(|arguments| arguments.concat()).collect();
            for jit in &jits {
                let output = jit.call(&inputs).unwrap();
                let words = jit.result_words();
                for (lane, arguments) in lanes.iter().enumerate() {
                    let expected = evaluate_tensor_fn(function, arguments).unwrap();
                    assert_eq!(&output[lane * words..(lane + 1) * words], expected.words(), "lane {lane} of {}", function.name);
                }
            }
        }
    }

    fn jagged(rows: &[Vec<usize>]) -> JaggedTensor<usize> {
        let mut offsets = vec![0];
        let mut data = Vec::new();
        for row in rows {
            data.extend(row);
            offsets.push(data.len());
        }
        JaggedTensor::try_new(data, vec![rows.len()], vec![offsets]).unwrap()
    }

    fn input(name: &str, width: usize) -> (TensorExpr, (String, TensorType)) {
        let tensor_type = TensorType::bitvec(width).unwrap();
        (Input::new(name, tensor_type.clone()).into(), (name.into(), tensor_type))
    }

    fn split_rows(width: usize, seed: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
        let prefixes: Vec<Vec<usize>> = (0..width)
            .map(|row| (0..(row * 7 + seed) % 5).map(|lane| (row + lane * 13 + seed) % width).collect())
            .collect();
        let suffixes = prefixes
            .iter()
            .enumerate()
            .map(|(row, lanes)| (0..lanes.len()).map(|lane| (row * 3 + lane + seed) % width).collect())
            .collect();
        (prefixes, suffixes)
    }

    #[test]
    fn gathers_and_gather2_match_the_reference_on_64_lanes() {
        for width in [15, 100, 150] {
            let (a, first) = input("a", width);
            let (b, second) = input("b", width);
            let (prefixes, suffixes) = split_rows(width, width);
            for inner in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
                for outer in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
                    let pointwise = BVLogical::new(
                        inner,
                        [
                            JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), []),
                            JTReduced::new(BVJaggedPermute::new(b.clone(), jagged(&suffixes)), []),
                        ],
                    );
                    let reduced = BVLogical::new(outer, [JTReduced::new(pointwise, [1])]);
                    check(&TensorFn::new("split", vec![first.clone(), second.clone()], reduced));
                }
                let gather = BVLogical::new(inner, [JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), [1])]);
                check(&TensorFn::new("gather", vec![first.clone()], gather));
            }
        }
    }

    #[test]
    fn dense_expand_reduce_constants_and_flat_permutes_are_transposed() {
        let (a, argument) = input("a", 64);
        for op in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
            let node = BTExpandReduce::new(a.clone(), [D::Batch(4), D::Expand(4), D::Reduce(4), D::Batch(4)], op);
            check(&TensorFn::new("dense", vec![argument.clone()], node));
        }
        // or(gather(a), const) and gather.or over and(flat permute(a), a).
        let (b, small) = input("b", 70);
        let (prefixes, _) = split_rows(70, 2);
        let gather = BVLogical::new(LogicalOp::And, [JTReduced::new(BVJaggedPermute::new(b.clone(), jagged(&prefixes)), [1])]);
        let constant = TensorConstant::new(JaggedTensor::from_flat(vec![0x0123_4567_89ab_cdef, 0x15]), TensorType::bitvec(70).unwrap());
        let mixed = BVLogical::new(LogicalOp::Or, [JTReduced::new(gather, []), JTReduced::new(constant, [])]);
        check(&TensorFn::new("mixed", vec![small.clone()], mixed));
        let flat: Vec<Vec<usize>> = (0..70).map(|row| if row % 3 == 0 { vec![] } else { vec![(row * 11) % 70] }).collect();
        let permuted = BVLogical::new(
            LogicalOp::And,
            [JTReduced::new(BVJaggedPermute::new(b.clone(), jagged(&flat)), []), JTReduced::new(b.clone(), [])],
        );
        let exists = BTExpandReduce::new(permuted, [D::Reduce(70)], LogicalOp::Or);
        check(&TensorFn::new("exists", vec![small], exists));
    }

    #[test]
    fn only_gathers_with_arguments_are_transposed() {
        let (a, argument) = input("a", 40);
        let or = TensorFn::new("or", vec![argument.clone()], BVLogical::new(LogicalOp::Or, [JTReduced::new(a.clone(), [])]));
        assert!(!is_transposed(&fuse(&or)));
        let constant = TensorConstant::new(JaggedTensor::from_flat(vec![5]), TensorType::bitvec(40).unwrap());
        assert!(!is_transposed(&fuse(&TensorFn::new("constant", Vec::new(), constant))));
        // A lane fold over a jagged permute that does not fuse is not transposed.
        let (prefixes, suffixes) = split_rows(40, 3);
        let triple = BVLogical::new(
            LogicalOp::And,
            [
                JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), []),
                JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&suffixes)), []),
                JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&suffixes)), []),
            ],
        );
        let folded = BVLogical::new(LogicalOp::Or, [JTReduced::new(triple, [1])]);
        assert!(!is_transposed(&fuse(&TensorFn::new("folded", vec![argument], folded))));
    }
}
