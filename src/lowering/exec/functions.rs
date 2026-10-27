//! Emit the packed `define_fun_*` and constraint functions straight from
//! fused `TensorFn` graphs.
//!
//! Every value is a vector of `W`-bit words, where `W` is 32 when every value
//! of the function fits in 32 bits and 64 otherwise. A value with `L` lanes
//! and `G` words per lane is lane-major: word `lane * G + group` holds rows
//! `group * W ..` of that lane. Bit movement is driven by constant step
//! tables, so code size depends on the expression structure, not on the
//! number of bits, and LLVM compiles it quickly:
//!
//! ```text
//! BVJaggedPermute   for step: out |= rotl(splat(src[S[step]]) & M[step], R[step])
//! BVGather          for element: acc[O] = op(acc[O], a[I1])           (one bit)
//! BVGather2         for element: acc[O] = op(acc[O], inner(a[I1], b[I2]))
//! lane fold         for lane: acc = op(acc, value[lane])
//! BVLogical         and / or / xor on whole vectors
//! ```
//!
//! `S` names a source word, `M` selects its bits for every destination word,
//! and `R` rotates each destination word by its own amount. Gathers update one
//! accumulator bit per step: `I1`/`I2`/`O` are `i32` bit-index tables, and the
//! accumulator and the sources are stack `memref.alloca`s, so a step loads
//! and stores single words. Values are zero
//! wherever a lane does not exist; AND folds first OR in the complement of
//! the lane's existence mask.
//!
//! ```text
//! memref.global "private" constant @table_<fn>_t<k> : memref<Nxi<W> or Nxi32> = dense<...>
//!
//! func.func private @define_fun_<TensorFn.name>(
//!     %arg0: vector<<ceil(arg0_bits / 64)>xi64>, ...
//! ) -> vector<<ceil(result_bits / 64)>xi64>
//!     attributes {loom.constructor = "<TensorFn.name>"} {
//!   <32-bit functions: vector.bitcast to vector<...xi32>>
//!   <step tables>       memref.get_global + vector.load / memref.load per step
//!   <loops>             scf.for carrying one accumulator vector
//!   <logic, rotations>  arith on whole vectors
//!   return %packed_result
//! }
//! ```
//!
//! The functions are pure, so production filters may call them concurrently.

use std::collections::HashMap;

use high_mlir::arith::{self, CmpIPredicate, Constant};
use high_mlir::func::Signature;
use high_mlir::{builtin, memref, scf, vector, Attr, IntType, Ty, Tuple, Val, I1, I32, I64, INDEX};

use super::{invalid, symbol_name, Result};
use crate::compile::typing::{expand_reduce_form, ExpandReduceForm, TypeCache};
use crate::compile::{JaggedTensor, LogicalOp, ShiftOp, TensorExpr, TensorFn, TensorType};
use crate::lowering::word_count;

const STAGE: &str = "constructor lowering";

/// A lowered value: `lanes x groups` words, lane-major.
#[derive(Clone)]
struct Packed {
    value: Val,
    lanes: usize,
    groups: usize,
    /// `exists[lane][group]`: rows that have this lane.
    exists: Vec<Vec<u64>>,
    rows: usize,
}

/// Per output word `(lane, group)`: `(destination bit, source bit index)`.
type Moves = Vec<Vec<Vec<(u32, usize)>>>;

struct Lowerer<'a, 't> {
    width: u32,
    mask: u64,
    element: IntType,
    inputs: &'a [(String, TensorType)],
    arguments: Vec<Val>,
    memo: HashMap<*const TensorExpr, Packed>,
    types: TypeCache,
    tables: &'a mut Tables<'t>,
}

/// A read-only table: a private constant `memref.global`.
struct TableSpec {
    name: String,
    ty: Ty,
    words: Vec<i64>,
}

/// The tables of one function. They must be module-level globals, but are
/// only known while its body is built, so the body is built twice (see
/// [`define_with_tables`]): a dry run in a scratch module records them, then
/// the globals are built before the function. Lowering is deterministic, so
/// the `k`-th table requested by the second run is the `k`-th one recorded.
enum Globals {
    Record(Vec<TableSpec>),
    Built(Vec<Val>),
}

/// The read-only tables of the function being built, by contents.
pub(super) struct Tables<'a> {
    memo: HashMap<(u32, Vec<i64>), Val>,
    globals: &'a mut Globals,
    /// Symbol prefix of this function's tables.
    prefix: String,
}

impl Tables<'_> {
    /// The `memref.get_global` of a `memref<Nx<element>>` holding `words`.
    /// Tables are built before the loops that index them.
    pub(super) fn get(&mut self, element: IntType, words: Vec<i64>) -> Val {
        let key = (element.0, words);
        if let Some(table) = self.memo.get(&key) {
            return table.clone();
        }
        let index = self.memo.len();
        let ty = Ty::memref([Some(key.1.len() as u64)], element);
        let table: Val = match &mut *self.globals {
            Globals::Record(specs) => {
                let name = format!("{}_t{index}", self.prefix);
                specs.push(TableSpec { name, ty: ty.clone(), words: key.1.clone() });
                // A stand-in of the table's type, for the dry run.
                memref::alloca(ty, (), ()).into()
            }
            Globals::Built(globals) => memref::get_global(&globals[index]).into(),
        };
        self.memo.insert(key, table.clone());
        table
    }
}

/// Build a function with `define` twice, first in a scratch module to record
/// its tables, then into the module being built, after the globals of those
/// tables. `define` must build the same function both times.
pub(super) fn define_with_tables(prefix: &str, define: impl Fn(&mut Tables) -> Result<Val>) -> Result<Val> {
    let mut recorded = Globals::Record(Vec::new());
    let mut dry_run = Ok(());
    builtin::module(|| {
        let mut tables = Tables { memo: HashMap::new(), globals: &mut recorded, prefix: prefix.to_string() };
        dry_run = define(&mut tables).map(drop);
    });
    dry_run?;
    let Globals::Record(specs) = recorded else { unreachable!("the dry run records") };
    let mut globals = Globals::Built(
        specs.into_iter().map(|spec| memref::global(spec.name, spec.ty).constant().ints(spec.words).build()).collect(),
    );
    define(&mut Tables { memo: HashMap::new(), globals: &mut globals, prefix: prefix.to_string() })
}

impl Lowerer<'_, '_> {
    /// Word groups for `rows` rows (even in 32-bit mode, so values bitcast to `i64`).
    fn groups(&self, rows: usize) -> usize {
        word_count(rows) * (64 / self.width as usize)
    }

    fn vector_type(&self, words: usize) -> Ty {
        Ty::vector([words as u64], self.element)
    }

    fn signed(&self, word: u64) -> i64 {
        if self.width == 32 {
            word as u32 as i32 as i64
        } else {
            word as i64
        }
    }

    /// `arith.constant dense<words>`.
    fn constant(&self, words: &[u64]) -> Val {
        let words: Vec<i64> = words.iter().map(|&word| self.signed(word & self.mask)).collect();
        Constant::ints(words.iter().copied(), self.vector_type(words.len()))
    }

    fn broadcast(&self, value: u64, count: usize) -> Val {
        self.constant(&vec![value; count])
    }

    /// A read-only step table in a `memref.global`, as a `memref<Nxi<W>>`.
    fn table(&mut self, data: Vec<u64>) -> Val {
        let words = data.iter().map(|&word| self.signed(word)).collect();
        self.tables.get(self.element, words)
    }

    fn index(value: usize) -> Val {
        Constant::index(value as i64).into()
    }

    /// `row * words + word` as an `index`.
    fn offset(row: &Val, words: usize, word: usize) -> Val {
        let start = arith::muli(row, Self::index(words), ());
        if word == 0 {
            start
        } else {
            arith::addi(start, Self::index(word), ())
        }
    }

    /// Word `S[step]` of a table, as an `index`.
    fn table_index(&self, table: &Val, step: &Val) -> Val {
        arith::index_cast(memref::load(table, (step,), ()), INDEX).into()
    }

    /// Row `row` of `words` words of a table.
    fn table_row(&self, table: &Val, row: &Val, words: usize) -> Val {
        vector::load(table, (Self::offset(row, words, 0),), self.vector_type(words), ()).into()
    }

    /// Row `row` of `words` words of a vector value.
    fn row(&self, source: &Val, row: &Val, words: usize, source_words: usize) -> Val {
        if words == source_words {
            return source.clone();
        }
        let elements: Vec<Val> =
            (0..words).map(|word| vector::extract(source, (&Self::offset(row, words, word),))).collect();
        vector::from_elements(elements, self.vector_type(words)).into()
    }

    /// `rotl(x, amounts)` per word, amounts in `0..W`.
    fn rotl(&self, x: &Val, amounts: &Val, words: usize) -> Val {
        let width = u64::from(self.width);
        let full = self.broadcast(width, words);
        let low = self.broadcast(width - 1, words);
        let high = arith::shli(x, amounts, ());
        let back = arith::andi(arith::subi(full, amounts, ()), low);
        let wrapped = arith::shrui(x, back, ());
        arith::ori(high, wrapped)
    }

    fn logic(op: LogicalOp, a: &Val, b: &Val) -> Val {
        match op {
            LogicalOp::And => arith::andi(a, b),
            LogicalOp::Or => arith::ori(a, b),
            LogicalOp::Xor => arith::xori(a, b),
        }
    }

    /// `body(iv, acc)` for `iv` in `0..trips`, carrying `acc`. One trip binds
    /// `iv` to constant zero.
    fn for_loop(&mut self, trips: usize, init: Val, body: impl FnOnce(&mut Self, Val, Val) -> Val) -> Val {
        match trips {
            0 => init,
            1 => body(self, Self::index(0), init),
            _ => scf! {for 0 to (trips) step 1 iter_args(init)}.then(|iv, acc| body(self, iv.into(), acc).into()),
        }
    }

    fn identity(op: LogicalOp) -> u64 {
        if op == LogicalOp::And { u64::MAX } else { 0 }
    }

    /// Step tables for words built from `moves` out of a flat source.
    /// Returns `(steps, S, M, R)` with `M`/`R` holding `words` per step.
    fn steps(&self, moves: &[Vec<(u32, usize)>], source_live: &[u64]) -> (usize, Vec<u64>, Vec<u64>, Vec<u64>) {
        let width = self.width as usize;
        // For every source word: per destination word, rotation -> source mask.
        let mut by_source: HashMap<usize, Vec<HashMap<u32, u64>>> = HashMap::new();
        for (word, pairs) in moves.iter().enumerate() {
            for &(destination, index) in pairs {
                let (source_word, bit) = (index / width, (index % width) as u32);
                if source_live[source_word] & (1u64 << bit) == 0 {
                    continue; // the source bit is known zero
                }
                let rotation = (destination + self.width - bit) % self.width;
                let words = by_source.entry(source_word).or_insert_with(|| vec![HashMap::new(); moves.len()]);
                *words[word].entry(rotation).or_insert(0) |= 1u64 << bit;
            }
        }
        let mut sources: Vec<_> = by_source.into_iter().collect();
        sources.sort_by_key(|(source, _)| *source);
        let (mut s, mut m, mut r) = (Vec::new(), Vec::new(), Vec::new());
        let mut steps = 0;
        for (source, words) in sources {
            let lists: Vec<Vec<(u32, u64)>> = words
                .into_iter()
                .map(|rotations| {
                    let mut list: Vec<_> = rotations.into_iter().collect();
                    list.sort();
                    list
                })
                .collect();
            let count = lists.iter().map(Vec::len).max().unwrap_or(0);
            for step in 0..count {
                s.push(source as u64);
                for list in &lists {
                    let (rotation, mask) = list.get(step).copied().unwrap_or((0, 0));
                    m.push(mask);
                    r.push(u64::from(rotation));
                }
                steps += 1;
            }
        }
        (steps, s, m, r)
    }

    /// `rotl(splat(source[S[step]]) & M[step], R[step])` over `words` words.
    fn moved(&self, source: &Val, tables: &(Val, Val, Val), step: &Val, words: usize) -> Val {
        let (s, m, r) = tables;
        let index = self.table_index(s, step);
        let word = vector::extract(source, (&index,));
        let splat: Val = vector::broadcast(word, self.vector_type(words)).into();
        let selected = arith::andi(splat, self.table_row(m, step, words));
        self.rotl(&selected, &self.table_row(r, step, words), words)
    }

    /// Moves of a permutation, by output word `(lane, group)`.
    fn moves(&self, indices: &JaggedTensor<usize>) -> (Moves, Vec<Vec<u64>>, usize) {
        let width = self.width as usize;
        let rows = indices.row_count();
        let groups = self.groups(rows);
        let lanes = (0..rows).map(|row| indices.row(row).len()).max().unwrap_or(0);
        let mut moves = vec![vec![Vec::new(); groups]; lanes];
        let mut exists = vec![vec![0u64; groups]; lanes];
        for row in 0..rows {
            for (lane, &index) in indices.row(row).iter().enumerate() {
                moves[lane][row / width].push(((row % width) as u32, index));
                exists[lane][row / width] |= 1u64 << (row % width);
            }
        }
        (moves, exists, lanes)
    }

    fn source_live(&mut self, expression: &TensorExpr, groups: usize) -> Result<Vec<u64>> {
        let support = self.types.infer(expression).map_err(|error| invalid(STAGE, error))?;
        let words: Vec<u64> = support.mask().data().to_vec();
        Ok(split(&words, self.width).into_iter().chain(std::iter::repeat(0)).take(groups).collect())
    }

    fn flat(&self, value: &Packed) -> Result<Val> {
        if value.lanes == 1 {
            Ok(value.value.clone())
        } else {
            Err(invalid(STAGE, "expected a flat bit-vector, found a jagged value"))
        }
    }

    fn valid_mask(&self, rows: usize) -> Vec<u64> {
        let width = self.width as usize;
        (0..self.groups(rows))
            .map(|group| {
                let start = group * width;
                if start >= rows {
                    0
                } else if rows - start >= width {
                    self.mask
                } else {
                    (1u64 << (rows - start)) - 1
                }
            })
            .collect()
    }

    fn permute(&mut self, source: Val, live: &[u64], indices: &JaggedTensor<usize>) -> Packed {
        let (moves, exists, lanes) = self.moves(indices);
        let rows = indices.row_count();
        let groups = self.groups(rows);
        let words = lanes * groups;
        let flat: Vec<Vec<(u32, usize)>> = moves.into_iter().flatten().collect();
        let (steps, s, m, r) = self.steps(&flat, live);
        let mut out = self.broadcast(0, words.max(1));
        if steps > 0 {
            let tables = (self.table(s), self.table(m), self.table(r));
            out = self.for_loop(steps, out, |this, step, acc| {
                let moved = this.moved(&source, &tables, &step, words);
                arith::ori(acc, moved)
            });
        }
        Packed { value: out, lanes, groups, exists, rows }
    }

    /// `BVGather` (one source) and `BVGather2` (two sources) as one loop with
    /// one step per element, built directly from the flat element indices.
    ///
    /// Step `k` reads bit `I_side[k]` of every source, combines the bits with
    /// `inner`, and folds the result into the single accumulator bit `O[k]`:
    /// OR/XOR set or flip it, AND clears it when the value is 0. Elements whose
    /// value is known zero (from the sources' support) take no step: they
    /// leave an OR/XOR accumulator unchanged and are cleared in the initial
    /// accumulator of an AND.
    fn gather_flat(
        &mut self,
        op: LogicalOp,
        inner: LogicalOp,
        sources: &[(Val, Vec<u64>, &[usize])],
        out: &[usize],
        width: usize,
    ) -> Packed {
        let bits = self.width as usize;
        let valid = self.valid_mask(width);
        let mut init = if op == LogicalOp::And { valid } else { vec![0; self.groups(width)] };
        let mut indices = vec![Vec::new(); sources.len()];
        let mut destinations = Vec::new();
        for (element, &row) in out.iter().enumerate() {
            let lives: Vec<bool> = sources
                .iter()
                .map(|(_, live, element_indices)| {
                    let index = element_indices[element];
                    live[index / bits] & (1u64 << (index % bits)) != 0
                })
                .collect();
            let always_zero = match inner {
                _ if sources.len() == 1 => !lives[0],
                LogicalOp::And => lives.iter().any(|live| !live),
                LogicalOp::Or | LogicalOp::Xor => lives.iter().all(|live| !live),
            };
            if always_zero {
                if op == LogicalOp::And {
                    init[row / bits] &= !(1u64 << (row % bits));
                }
                continue;
            }
            for (side, (_, _, element_indices)) in sources.iter().enumerate() {
                indices[side].push(element_indices[element] as u64);
            }
            destinations.push(row as u64);
        }

        let groups = init.len();
        let mut acc = self.constant(&init);
        if !destinations.is_empty() {
            let steps = destinations.len();
            let tables: Vec<Val> = indices.into_iter().map(|data| self.index_table(data)).collect();
            let destinations = self.index_table(destinations);
            // The accumulator and the sources live in stack buffers, so a
            // step loads single words: a dynamic `vector.extract` would spill
            // the whole source vector on every step.
            let buffer = self.spill(&acc);
            let mut spilled: Vec<(Val, Val)> = Vec::new();
            let buffers: Vec<Val> = sources
                .iter()
                .map(|(source, _, _)| match spilled.iter().find(|(value, _)| value == source) {
                    Some((_, buffer)) => buffer.clone(),
                    None => {
                        let buffer = self.spill(source);
                        spilled.push((source.clone(), buffer.clone()));
                        buffer
                    }
                })
                .collect();
            let this = &*self;
            let step_body = |step: Val| {
                let mut value: Option<Val> = None;
                for (table, source) in tables.iter().zip(&buffers) {
                    let bit = this.bit(source, &this.table_index(table, &step));
                    value = Some(match value {
                        None => bit,
                        Some(value) => Self::logic(inner, &value, &bit),
                    });
                }
                let destination = this.table_index(&destinations, &step);
                this.update_bit(op, &buffer, &destination, &value.expect("a gather has a source"));
            };
            if steps == 1 {
                step_body(Self::index(0));
            } else {
                scf! {for 0 to (steps) step 1}.then(|step| {
                    step_body(step.into());
                    ().into()
                });
            }
            acc = vector::load(&buffer, (Self::index(0),), self.vector_type(groups), ()).into();
        }
        self.flat_value(acc, width)
    }

    /// A read-only table of bit indices, as a `memref<Nxi32>`.
    fn index_table(&mut self, data: Vec<u64>) -> Val {
        self.tables.get(I32, data.iter().map(|&index| index as i64).collect())
    }

    /// `(word, bit)` of a bit index: word as an `index`, bit as a `W`-bit integer.
    fn locate(&self, index: &Val) -> (Val, Val) {
        let word = arith::shrui(index, Self::index(self.width.trailing_zeros() as usize), ());
        let bit = arith::index_cast(arith::andi(index, Self::index(self.width as usize - 1)), self.element);
        (word, bit.into())
    }

    /// A stack `memref<Nxi<W>>` holding the words of the vector `value`.
    fn spill(&self, value: &Val) -> Val {
        let words = value.ty().as_vector().expect("packed values are vectors").dims()[0];
        let buffer: Val = memref::alloca(Ty::memref([Some(words)], self.element), (), ()).into();
        vector::store(value, &buffer, (Self::index(0),), ());
        buffer
    }

    /// Bit `index` of a flat value spilled to `source`, as a `W`-bit 0 or 1.
    fn bit(&self, source: &Val, index: &Val) -> Val {
        let (word, bit) = self.locate(index);
        let word = memref::load(source, (&word,), ());
        arith::andi(arith::shrui(word, bit, ()), Constant::int(1, self.element))
    }

    /// Fold `value` (0 or 1) into bit `index` of the accumulator `buffer` by `op`.
    fn update_bit(&self, op: LogicalOp, buffer: &Val, index: &Val, value: &Val) {
        let (position, bit) = self.locate(index);
        let word = memref::load(buffer, (&position,), ());
        let updated = match op {
            LogicalOp::Or => arith::ori(word, arith::shli(value, &bit, ())),
            LogicalOp::Xor => arith::xori(word, arith::shli(value, &bit, ())),
            LogicalOp::And => {
                // Clear the bit when `value` is 0: word & ~((value ^ 1) << bit).
                let one = Constant::int(1, self.element);
                let ones = Constant::int(-1, self.element);
                let clear = arith::shli(arith::xori(value, one), &bit, ());
                arith::andi(word, arith::xori(clear, ones))
            }
        };
        memref::store(updated, buffer, (&position,), ());
    }

    fn flat_value(&self, value: Val, rows: usize) -> Packed {
        let exists = vec![self.valid_mask(rows)];
        Packed { value, lanes: 1, groups: self.groups(rows), exists, rows }
    }

    fn reduce(&mut self, op: LogicalOp, value: &Packed) -> Packed {
        let groups = value.groups;
        let words = value.lanes * groups;
        let absent: Vec<u64> = value
            .exists
            .iter()
            .flat_map(|lane| lane.iter().map(|bits| !bits & self.mask).collect::<Vec<_>>())
            .collect();
        let absent = (op == LogicalOp::And && value.lanes > 0).then(|| self.table(absent));
        let source = value.value.clone();
        let init = self.broadcast(Self::identity(op), groups);
        let acc = self.for_loop(value.lanes, init, |this, lane, acc| {
            let mut row = this.row(&source, &lane, groups, words);
            if let Some(absent) = &absent {
                row = arith::ori(row, this.table_row(absent, &lane, groups));
            }
            Self::logic(op, &acc, &row)
        });
        let result = if op == LogicalOp::And {
            arith::andi(acc, self.constant(&self.valid_mask(value.rows)))
        } else {
            acc
        };
        self.flat_value(result, value.rows)
    }

    /// Pad `value` to `lanes` lanes (missing lanes are zero).
    fn pad(&mut self, value: &Packed, lanes: usize) -> Packed {
        if value.lanes == lanes {
            return value.clone();
        }
        let groups = value.groups;
        let words = value.lanes * groups;
        let source = value.value.clone();
        let zeros = self.broadcast(0, lanes * groups);
        let out = self.for_loop(value.lanes, zeros, |this, lane, out| {
            let row = this.row(&source, &lane, groups, words);
            (0..groups).fold(out, |out, word| {
                let element = vector::extract(&row, word as i64);
                vector::insert(element, out, (&Self::offset(&lane, groups, word),)).into()
            })
        });
        let mut exists = value.exists.clone();
        exists.resize(lanes, vec![0; groups]);
        Packed { value: out, lanes, groups, exists, rows: value.rows }
    }

    fn expression(&mut self, expression: &TensorExpr) -> Result<Packed> {
        let key = expression as *const TensorExpr;
        if let Some(value) = self.memo.get(&key) {
            return Ok(value.clone());
        }
        let support = self.types.infer(expression).map_err(|error| invalid(STAGE, error))?;
        let value = if support.support() == 0 && support.is_flat() {
            let rows = support.width();
            let zero = self.broadcast(0, self.groups(rows));
            self.flat_value(zero, rows)
        } else {
            self.lower(expression)?
        };
        self.memo.insert(key, value.clone());
        Ok(value)
    }

    fn lower(&mut self, expression: &TensorExpr) -> Result<Packed> {
        let fail = |message: String| invalid(STAGE, message);
        Ok(match expression {
            TensorExpr::Input(input) => {
                let arg = self
                    .inputs
                    .iter()
                    .position(|(name, _)| name == &input.name)
                    .ok_or_else(|| fail(format!("unknown input {:?}", input.name)))?;
                self.flat_value(self.arguments[arg].clone(), input.tensor_type.width())
            }
            TensorExpr::Constant(constant) => {
                let rows = constant.tensor_type.width();
                let mut words = split(constant.value.data(), self.width);
                words.resize(self.groups(rows), 0);
                let valid = self.valid_mask(rows);
                let words: Vec<u64> = words.iter().zip(&valid).map(|(word, valid)| word & valid).collect();
                self.flat_value(self.constant(&words), rows)
            }
            TensorExpr::ScalarConstant(constant) => {
                let words = split(&[constant.value], self.width);
                self.flat_value(self.constant(&words), 64)
            }
            TensorExpr::BVJaggedPermute(permute) => {
                let source = self.expression(&permute.expr)?;
                let value = self.flat(&source)?;
                let live = self.source_live(&permute.expr, source.groups)?;
                self.permute(value, &live, &permute.indices)
            }
            TensorExpr::BVGather(gather) => {
                let source = self.expression(&gather.expr)?;
                let value = self.flat(&source)?;
                let live = self.source_live(&gather.expr, source.groups)?;
                gather.validate().map_err(fail)?;
                self.gather_flat(
                    gather.op,
                    LogicalOp::Or,
                    &[(value, live, gather.indices_in.data())],
                    gather.indices_out.data(),
                    gather.width,
                )
            }
            TensorExpr::BVGather2(gather) => {
                let a = self.expression(&gather.a)?;
                let b = self.expression(&gather.b)?;
                let (a_value, b_value) = (self.flat(&a)?, self.flat(&b)?);
                let a_live = self.source_live(&gather.a, a.groups)?;
                let b_live = self.source_live(&gather.b, b.groups)?;
                gather.validate().map_err(fail)?;
                self.gather_flat(
                    gather.op,
                    gather.inner,
                    &[(a_value, a_live, gather.indices_1.data()), (b_value, b_live, gather.indices_2.data())],
                    gather.indices_out.data(),
                    gather.width,
                )
            }
            TensorExpr::BVLogical(logical) => {
                let mut values = Vec::with_capacity(logical.operands.len());
                for operand in &logical.operands {
                    let value = self.expression(&operand.expr)?;
                    values.push(if operand.reduce_dims.is_empty() {
                        value
                    } else {
                        self.reduce(logical.op, &value)
                    });
                }
                let first = values.first().ok_or_else(|| fail("logic needs an operand".into()))?;
                let (rows, groups) = (first.rows, first.groups);
                if values.iter().any(|value| value.rows != rows) {
                    return Err(fail("logical operands have different widths".into()));
                }
                let lanes = values.iter().map(|value| value.lanes).max().unwrap_or(1);
                let values: Vec<Packed> = values.iter().map(|value| self.pad(value, lanes)).collect();
                let mut result = values[0].value.clone();
                for value in &values[1..] {
                    result = Self::logic(logical.op, &result, &value.value);
                }
                let mut exists = vec![vec![0u64; groups]; lanes];
                for value in &values {
                    for (lane, words) in value.exists.iter().enumerate() {
                        for (word, bits) in words.iter().enumerate() {
                            exists[lane][word] |= bits;
                        }
                    }
                }
                Packed { value: result, lanes, groups, exists, rows }
            }
            TensorExpr::BTShift(shift) => {
                let source = self.expression(&shift.expr)?;
                let value = self.flat(&source)?;
                let rows = source.rows;
                let indices: Vec<Vec<usize>> = (0..rows)
                    .map(|destination| {
                        let index = destination.checked_add(shift.shift_amounts).filter(|&index| index < rows);
                        match shift.shift_op {
                            ShiftOp::Logical => index.into_iter().collect(),
                            ShiftOp::Arithmetic => vec![index.unwrap_or(rows - 1)],
                            ShiftOp::Rotate => vec![(destination + shift.shift_amounts % rows) % rows],
                        }
                    })
                    .collect();
                let mut offsets = vec![0];
                let mut data = Vec::new();
                for row in &indices {
                    data.extend(row);
                    offsets.push(data.len());
                }
                let indices = JaggedTensor::new(data, vec![rows], vec![offsets]);
                let live = self.source_live(&shift.expr, source.groups)?;
                let permuted = self.permute(value, &live, &indices);
                self.pad(&permuted, 1)
            }
            TensorExpr::BTExpandReduce(change) => {
                let input = self.types.infer(&change.expr).map_err(fail)?;
                let value = self.expression(&change.expr)?;
                match expand_reduce_form(change, &input).map_err(fail)? {
                    ExpandReduceForm::Dense(gather) => {
                        let flat = self.flat(&value)?;
                        let live = self.source_live(&change.expr, value.groups)?;
                        self.gather_flat(
                            gather.op,
                            LogicalOp::Or,
                            &[(flat, live, gather.indices_in.data())],
                            gather.indices_out.data(),
                            gather.width,
                        )
                    }
                    ExpandReduceForm::LaneFold => self.reduce(change.op, &value),
                }
            }
        })
    }
}

fn split(words: &[u64], width: u32) -> Vec<u64> {
    if width == 64 {
        return words.to_vec();
    }
    words.iter().flat_map(|&word| [word & 0xffff_ffff, word >> 32]).collect()
}

/// Word width for a function: 32 when every value fits in 32 bits.
fn choose_width(function: &TensorFn, result_bits: usize) -> u32 {
    let widest = function
        .inputs
        .iter()
        .map(|(_, tensor_type)| tensor_type.width())
        .chain([result_bits])
        .max()
        .unwrap_or(64);
    if widest <= 32 { 32 } else { 64 }
}

/// Name of the packed constructor function for `constructor`.
pub fn define_fun_symbol(constructor: &str) -> String {
    format!("define_fun_{}", symbol_name(constructor))
}

/// The `vector<Nxi64>` type of a value of `bits` bits.
pub fn packed_type(bits: usize) -> Ty {
    Ty::vector([word_count(bits) as u64], I64)
}

/// Logical bit width of the result of a (fused) function.
pub fn result_bits(function: &TensorFn) -> Result<usize> {
    let tensor_type = function.expression.infer_type().map_err(|error| invalid(STAGE, error))?;
    Ok(tensor_type.width())
}

/// Lower `function` to the private `@<symbol>`, whose results `finish`
/// builds from the one-lane packed value of `W`-bit words, preceded by the
/// globals of its tables. Must be called while a module is being built.
fn emit(
    function: &TensorFn,
    symbol: &str,
    results: Vec<Ty>,
    finish: impl Fn(&Val, u32) -> Vec<Val>,
) -> Result<Val> {
    let result_bits = result_bits(function)?;
    let width = choose_width(function, result_bits);
    let element = if width == 32 { I32 } else { I64 };
    let arguments: Vec<Ty> = function.inputs.iter().map(|(_, tensor_type)| packed_type(tensor_type.width())).collect();
    let signature = Signature::new(symbol, arguments, results);
    let define = |tables: &mut Tables| -> Result<Val> {
        let mut failure = None;
        let defined = signature.try_define(|parameters| {
            let mut lowerer = Lowerer {
                width,
                mask: if width == 64 { u64::MAX } else { (1u64 << width) - 1 },
                element,
                inputs: &function.inputs,
                arguments: Vec::new(),
                memo: HashMap::new(),
                types: TypeCache::default(),
                tables: &mut *tables,
            };
            lowerer.arguments = parameters
                .iter()
                .zip(&function.inputs)
                .map(|(parameter, (_, tensor_type))| {
                    if width == 32 {
                        vector::bitcast(parameter, lowerer.vector_type(lowerer.groups(tensor_type.width()))).into()
                    } else {
                        parameter.clone()
                    }
                })
                .collect();
            match lowerer.expression(&function.expression) {
                Ok(result) => {
                    let result = lowerer.pad(&result, 1);
                    Tuple(finish(&result.value, width))
                }
                Err(error) => {
                    failure = Some(error);
                    Tuple(Vec::new())
                }
            }
        });
        if let Some(error) = failure {
            return Err(error);
        }
        defined.map_err(|error| invalid(STAGE, error.to_string()))
    };

    let function = define_with_tables(&symbol_name(&format!("table_{}", function.name)), define)?;
    function.annotate("sym_visibility", Attr::str("private"));
    Ok(function)
}

/// Emit one packed constructor, `@define_fun_<name>`, into the module being
/// built.
pub fn lower_tensor_fn(function: &TensorFn) -> Result<Val> {
    let result = packed_type(result_bits(function)?);
    let lowered = emit(function, &define_fun_symbol(&function.name), vec![result.clone()], |output, width| {
        vec![if width == 32 { vector::bitcast(output, result.clone()).into() } else { output.clone() }]
    })?;
    lowered.annotate("loom.constructor", Attr::str(&function.name));
    Ok(lowered)
}

/// Emit `@<symbol>(candidate) -> i1` from a one-bit constraint into the
/// module being built.
pub fn lower_constraint(symbol: &str, function: &TensorFn) -> Result<Val> {
    if result_bits(function)? != 1 || function.inputs.len() != 1 {
        return Err(invalid(STAGE, "a constraint must map one candidate to one bit"));
    }
    emit(function, symbol, vec![I1.into()], |output, _| {
        vec![arith::cmpi(CmpIPredicate::Ne, vector::extract(output, 0), 0)]
    })
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use super::*;
    use crate::compile::pass::fuse;
    use crate::compile::{
        BTShift, BVJaggedPermute, BVLogical, Constant as TensorConstant, Input, JTReduced, JaggedTensor,
        TensorType,
    };
    use crate::lowering::jit::JitFunction;
    use crate::lowering::packed::evaluate_tensor_fn;

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

    /// Check the fused graph and the JIT at -O0 and -O3 against the
    /// reference evaluator; returns the printed constructor.
    fn check(function: &TensorFn) -> String {
        let fused = fuse(function);
        let mut lowered = Ok(());
        let printed = builtin::module(|| lowered = lower_tensor_fn(&fused).map(drop)).to_string();
        lowered.unwrap();
        let jits: Vec<_> = [0, 3]
            .into_iter()
            .map(|level| JitFunction::compile(function, level).unwrap_or_else(|error| panic!("{error}")))
            .collect();
        let mut seed = 0x9e3779b97f4a7c15;
        for _ in 0..48 {
            let arguments: Vec<_> = function
                .inputs
                .iter()
                .map(|(_, tensor_type)| random_words(&mut seed, tensor_type.width()))
                .collect();
            let expected = evaluate_tensor_fn(function, &arguments).unwrap();
            assert_eq!(evaluate_tensor_fn(&fused, &arguments).unwrap(), expected, "fusion changed semantics");
            for jit in &jits {
                assert_eq!(jit.call(&arguments.concat()).unwrap(), expected.words(), "JIT of {}", jit.llvm_mlir);
            }
        }
        printed
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
    fn gathers_and_gather2_match_the_reference_in_both_word_widths() {
        for width in [15, 100, 150] {
            let (a, first) = input("a", width);
            let (b, second) = input("b", width);
            let (prefixes, suffixes) = split_rows(width, width);
            for inner in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
                for outer in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
                    // (a[prefixes] inner b[suffixes]).reduce(outer) -> gather2
                    let pointwise = BVLogical::new(
                        inner,
                        [
                            JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), []),
                            JTReduced::new(BVJaggedPermute::new(b.clone(), jagged(&suffixes)), []),
                        ],
                    );
                    let reduced = BVLogical::new(outer, [JTReduced::new(pointwise, [1])]);
                    let text = check(&TensorFn::new("split", vec![first.clone(), second.clone()], reduced));
                    // 32-bit words show up as `vector<Nxi32>` (index tables are always i32).
                    let words32 = text.split("vector<").skip(1).any(|rest| {
                        rest.split('>').next().is_some_and(|ty| ty.ends_with("xi32"))
                    });
                    assert_eq!(words32, width <= 32, "{text}");
                }
                // a[prefixes].reduce(inner) -> gather
                let gather = BVLogical::new(inner, [JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), [1])]);
                check(&TensorFn::new("gather", vec![first.clone()], gather));
            }
        }
    }

    #[test]
    fn unfused_permutes_reductions_and_logic_match_the_reference() {
        let width = 90;
        let (a, argument) = input("a", width);
        let (prefixes, suffixes) = split_rows(width, 3);
        // Three operands block gather2 fusion: lane-major permutes + reduce.
        let pointwise = BVLogical::new(
            LogicalOp::And,
            [
                JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), []),
                JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&suffixes)), []),
                JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&suffixes)), []),
            ],
        );
        let reduced = BVLogical::new(
            LogicalOp::Or,
            [JTReduced::new(pointwise, [1]), JTReduced::new(a.clone(), [])],
        );
        check(&TensorFn::new("unfused", vec![argument.clone()], reduced));
    }

    #[test]
    fn expand_reduce_folds_match_the_reference_fused_and_unfused() {
        use crate::compile::{BTExpandReduce, ExpandReduceDimType as D};
        let width = 70;
        let (a, first) = input("a", width);
        let (b, second) = input("b", width);
        let (prefixes, suffixes) = split_rows(width, 5);
        for op in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
            let lanes = prefixes.iter().map(Vec::len).max().unwrap();
            let dims = [D::Batch(width), D::Reduce(lanes)];
            // exists over a permute -> gather; over and(permute, permute) -> gather2.
            let gather = BTExpandReduce::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), dims, op);
            let pair = BVLogical::new(
                LogicalOp::And,
                [
                    JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), []),
                    JTReduced::new(BVJaggedPermute::new(b.clone(), jagged(&suffixes)), []),
                ],
            );
            let gather2 = BTExpandReduce::new(pair, dims, op);
            // Three permutes do not fuse: lane-major permutes + lane fold.
            let triple = BVLogical::new(
                LogicalOp::Or,
                [
                    JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&prefixes)), []),
                    JTReduced::new(BVJaggedPermute::new(b.clone(), jagged(&suffixes)), []),
                    JTReduced::new(BVJaggedPermute::new(a.clone(), jagged(&suffixes)), []),
                ],
            );
            let unfused = BTExpandReduce::new(triple, dims, op);
            let inputs = vec![first.clone(), second.clone()];
            check(&TensorFn::new("gather", inputs.clone(), gather));
            let fused = fuse(&TensorFn::new("gather2", inputs.clone(), gather2.clone()));
            assert!(matches!(fused.expression, TensorExpr::BVGather2(_)), "fused into gather2");
            check(&TensorFn::new("gather2", inputs.clone(), gather2));
            check(&TensorFn::new("unfused", inputs, unfused));
        }
    }

    #[test]
    fn dense_expand_reduce_matches_the_reference() {
        use crate::compile::{BTExpandReduce, ExpandReduceDimType as D};
        // %a : bits[64];  expand_reduce.op %a [4, expand(4), reduce(4), 4] : bits[64]
        let (a, argument) = input("a", 64);
        for op in [LogicalOp::And, LogicalOp::Or, LogicalOp::Xor] {
            let node = BTExpandReduce::new(a.clone(), [D::Batch(4), D::Expand(4), D::Reduce(4), D::Batch(4)], op);
            assert_eq!((node.input_width(), node.output_width()), (64, 64));
            check(&TensorFn::new("dense", vec![argument.clone()], node));
            // Reduce everything, expand only, and a reduction over a 2x3 view.
            check(&TensorFn::new("all", vec![argument.clone()], BTExpandReduce::new(a.clone(), [D::Reduce(64)], op)));
            let (b, small) = input("b", 6);
            check(&TensorFn::new("view", vec![small.clone()], BTExpandReduce::new(b.clone(), [D::Reduce(2), D::Batch(3)], op)));
            check(&TensorFn::new("expand", vec![small], BTExpandReduce::new(b, [D::Expand(5), D::Batch(6)], op)));
        }
    }

    #[test]
    fn masked_update_and_shifts_match_the_reference() {
        let width = 150;
        let (a, argument) = input("a", width);
        let constant = |words: Vec<u64>| {
            TensorExpr::from(TensorConstant::new(JaggedTensor::from_flat(words), TensorType::bitvec(width).unwrap()))
        };
        // a[""] = true  as  a ^ ((a ^ ones) & mask)
        let flipped = BVLogical::new(
            LogicalOp::Xor,
            [JTReduced::new(a.clone(), []), JTReduced::new(constant(vec![u64::MAX, u64::MAX, (1 << 22) - 1]), [])],
        );
        let selected = BVLogical::new(
            LogicalOp::And,
            [JTReduced::new(flipped, []), JTReduced::new(constant(vec![1, 0, 0]), [])],
        );
        let update = BVLogical::new(LogicalOp::Xor, [JTReduced::new(a.clone(), []), JTReduced::new(selected, [])]);
        check(&TensorFn::new("set", vec![argument.clone()], update));
        for (op, amount) in [(ShiftOp::Logical, 70), (ShiftOp::Arithmetic, 5), (ShiftOp::Rotate, 81)] {
            check(&TensorFn::new("shift", vec![argument.clone()], BTShift::new(a.clone(), op, amount)));
        }
    }

    #[test]
    fn code_size_does_not_grow_with_the_bit_width() {
        let count = |width: usize| {
            let (a, first) = input("a", width);
            let (b, second) = input("b", width);
            let (prefixes, suffixes) = split_rows(width, 1);
            let pointwise = BVLogical::new(
                LogicalOp::And,
                [
                    JTReduced::new(BVJaggedPermute::new(a, jagged(&prefixes)), []),
                    JTReduced::new(BVJaggedPermute::new(b, jagged(&suffixes)), []),
                ],
            );
            let reduced = BVLogical::new(LogicalOp::Or, [JTReduced::new(pointwise, [1])]);
            check(&TensorFn::new("split", vec![first, second], reduced)).lines().count()
        };
        assert_eq!(count(100), count(400));
    }
}
