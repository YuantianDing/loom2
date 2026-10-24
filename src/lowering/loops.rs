//! Lower a [`TensorFn`] to a small vector loop program.
//!
//! Every variable is a vector of `word_width`-bit words. A value with `L`
//! lanes and `G` words per lane is stored lane-major: word `lane * G + group`
//! holds rows `group * word_width ..` of that lane. Bit movement is driven by
//! constant tables, so code size depends on the expression structure, not on
//! the number of bits, and LLVM compiles it quickly:
//!
//! ```text
//! BVJaggedPermute   for step: out |= rotl(splat(src[S[step]]) & M[step], R[step])
//! BVGather          for step: acc = op(acc, rotl(splat(a[S1]) & M1, R1))
//! BVGather2         for step: acc = op(acc, inner(rotl(splat(a[S1]) & M1, R1),
//!                                                 rotl(splat(b[S2]) & M2, R2)))
//! reduction         for lane: acc = op(acc, value[lane])
//! BVLogical         and / or / xor on whole vectors
//! ```
//!
//! `S` names a source word, `M` selects its bits for every destination word,
//! and `R` rotates each destination word by its own amount. Values are zero
//! wherever a lane does not exist; AND folds first OR in the complement of
//! the lane's existence mask.

use std::collections::HashMap;
use std::fmt::{self, Write as _};
use std::ops::Range;

use crate::compile::typing::TypeCache;
use crate::compile::{JaggedTensor, LogicalOp, ShiftOp, TensorExpr, TensorFn};

use super::{LoweringError, LoweringResult};

/// A variable: an input, a temporary, or an output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum VarId {
    Input(usize),
    Temp(usize),
    Output(usize),
}

/// One vector operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Operation {
    /// `dst = src[index]`, reading `src` as rows of `words(dst)` words.
    Load(VarId, VarId, VarId),
    /// `dst[index] = src`, writing row `index` of `words(src)` words.
    Store(VarId, VarId, VarId),
    /// `dst = [value; count]`.
    Broadcast(VarId, u64, usize),
    /// `dst = table` (constant data).
    Constant(VarId, Vec<u64>),
    /// `dst = [src[0]; words(dst)]`.
    Splat(VarId, VarId),
    /// Loop `iv` over `0..trip`; the body is every operation inside the live
    /// range of `iv`. Loops may nest.
    ForLoop(VarId, VarId),
    And(VarId, VarId, VarId),
    Or(VarId, VarId, VarId),
    Xor(VarId, VarId, VarId),
    /// `dst = rotl(a, amounts)` per word, amounts in `0..word_width`.
    Rotl(VarId, VarId, VarId),
    /// `dst = src`.
    Copy(VarId, VarId),
}

impl Operation {
    /// Variables written by this operation.
    pub fn writes(&self) -> Option<VarId> {
        match *self {
            Operation::Load(dst, ..)
            | Operation::Store(dst, ..)
            | Operation::Broadcast(dst, ..)
            | Operation::Constant(dst, _)
            | Operation::Splat(dst, _)
            | Operation::And(dst, ..)
            | Operation::Or(dst, ..)
            | Operation::Xor(dst, ..)
            | Operation::Rotl(dst, ..)
            | Operation::Copy(dst, _) => Some(dst),
            Operation::ForLoop(iv, _) => Some(iv),
        }
    }

    /// Variables read by this operation.
    pub fn reads(&self) -> Vec<VarId> {
        match *self {
            Operation::Load(_, src, index) => vec![src, index],
            Operation::Store(dst, index, src) => vec![dst, index, src],
            Operation::Broadcast(..) | Operation::Constant(..) => Vec::new(),
            Operation::Splat(_, src) | Operation::Copy(_, src) => vec![src],
            Operation::ForLoop(_, trip) => vec![trip],
            Operation::And(_, a, b) | Operation::Or(_, a, b) | Operation::Xor(_, a, b) | Operation::Rotl(_, a, b) => {
                vec![a, b]
            }
        }
    }
}

/// A packed function as vector loops.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopProgram {
    pub name: String,
    /// 32 or 64.
    pub word_width: u32,
    /// `(words, live range)` per variable.
    pub inputs: Vec<(usize, Range<usize>)>,
    pub temps: Vec<(usize, Range<usize>)>,
    pub outputs: Vec<(usize, Range<usize>)>,
    pub ops: Vec<Operation>,
    /// Logical bit widths of the inputs and the result.
    pub input_bits: Vec<usize>,
    pub result_bits: usize,
}

/// `i64` words needed for `bits` bits.
pub fn word_count(bits: usize) -> usize {
    bits.div_ceil(64)
}

impl LoopProgram {
    pub fn words(&self, var: VarId) -> usize {
        match var {
            VarId::Input(index) => self.inputs[index].0,
            VarId::Temp(index) => self.temps[index].0,
            VarId::Output(index) => self.outputs[index].0,
        }
    }

    pub fn live(&self, var: VarId) -> Range<usize> {
        match var {
            VarId::Input(index) => self.inputs[index].1.clone(),
            VarId::Temp(index) => self.temps[index].1.clone(),
            VarId::Output(index) => self.outputs[index].1.clone(),
        }
    }

    /// Words of `word_width` bits per 64-bit word.
    pub fn ratio(&self) -> usize {
        64 / self.word_width as usize
    }

    /// Number of operations outside of constant tables.
    pub fn operation_count(&self) -> usize {
        self.ops.iter().filter(|op| !matches!(op, Operation::Constant(..))).count()
    }

    /// Run the program on packed `u64` argument words (reference semantics).
    pub fn evaluate(&self, arguments: &[Vec<u64>]) -> Vec<u64> {
        let width = self.word_width;
        let mask = if width == 64 { u64::MAX } else { (1u64 << width) - 1 };
        let mut state: HashMap<VarId, Vec<u64>> = HashMap::new();
        for (index, words) in arguments.iter().enumerate() {
            state.insert(VarId::Input(index), split(words, width));
        }
        self.run(0..self.ops.len(), &mut state, width, mask);
        join(&state[&VarId::Output(0)], width)
    }

    fn run(&self, range: Range<usize>, state: &mut HashMap<VarId, Vec<u64>>, width: u32, mask: u64) {
        let mut position = range.start;
        while position < range.end {
            let op = &self.ops[position];
            let get = |state: &HashMap<VarId, Vec<u64>>, var: VarId| state[&var].clone();
            match op {
                Operation::Load(dst, src, index) => {
                    let n = self.words(*dst);
                    let row = get(state, *index)[0] as usize;
                    let source = get(state, *src);
                    state.insert(*dst, source[row * n..row * n + n].to_vec());
                }
                Operation::Store(dst, index, src) => {
                    let value = get(state, *src);
                    let row = get(state, *index)[0] as usize;
                    let target = state.get_mut(dst).unwrap();
                    target[row * value.len()..(row + 1) * value.len()].copy_from_slice(&value);
                }
                Operation::Broadcast(dst, value, count) => {
                    state.insert(*dst, vec![*value & mask; *count]);
                }
                Operation::Constant(dst, data) => {
                    state.insert(*dst, data.iter().map(|word| word & mask).collect());
                }
                Operation::Splat(dst, src) => {
                    let word = get(state, *src)[0];
                    state.insert(*dst, vec![word; self.words(*dst)]);
                }
                Operation::ForLoop(iv, trip) => {
                    let trips = get(state, *trip)[0];
                    let body = position + 1..self.live(*iv).end;
                    for iteration in 0..trips {
                        state.insert(*iv, vec![iteration]);
                        self.run(body.clone(), state, width, mask);
                    }
                    position = body.end;
                    continue;
                }
                Operation::And(dst, a, b) | Operation::Or(dst, a, b) | Operation::Xor(dst, a, b) | Operation::Rotl(dst, a, b) => {
                    let (x, y) = (get(state, *a), get(state, *b));
                    let result = x
                        .iter()
                        .zip(&y)
                        .map(|(&x, &y)| match op {
                            Operation::And(..) => x & y,
                            Operation::Or(..) => x | y,
                            Operation::Xor(..) => x ^ y,
                            _ => {
                                let r = (y as u32) % width;
                                if r == 0 {
                                    x
                                } else {
                                    ((x << r) | (x >> (width - r))) & mask
                                }
                            }
                        })
                        .collect();
                    state.insert(*dst, result);
                }
                Operation::Copy(dst, src) => {
                    let value = get(state, *src);
                    state.insert(*dst, value);
                }
            }
            position += 1;
        }
    }
}

fn split(words: &[u64], width: u32) -> Vec<u64> {
    if width == 64 {
        return words.to_vec();
    }
    words.iter().flat_map(|&word| [word & 0xffff_ffff, word >> 32]).collect()
}

fn join(words: &[u64], width: u32) -> Vec<u64> {
    if width == 64 {
        return words.to_vec();
    }
    words.chunks(2).map(|pair| pair[0] | (pair.get(1).copied().unwrap_or(0) << 32)).collect()
}

/// A lowered value: `lanes x groups` words, lane-major.
#[derive(Clone, Debug)]
struct Value {
    var: VarId,
    lanes: usize,
    groups: usize,
    /// `exists[lane][group]`: rows that have this lane.
    exists: Vec<Vec<u64>>,
    rows: usize,
}

/// Per output word `(lane, group)`: `(destination bit, source bit index)`.
type Moves = Vec<Vec<Vec<(u32, usize)>>>;

struct Lowerer<'a> {
    width: u32,
    mask: u64,
    ops: Vec<Operation>,
    temps: Vec<usize>,
    inputs: &'a [(String, crate::compile::TensorType)],
    memo: HashMap<*const TensorExpr, Value>,
    types: TypeCache,
    constants: HashMap<Vec<u64>, VarId>,
    loops: Vec<(VarId, usize)>,
    loop_ends: HashMap<VarId, usize>,
}

impl Lowerer<'_> {
    /// Word groups for `rows` rows (even in 32-bit mode, so values bitcast to `i64`).
    fn groups(&self, rows: usize) -> usize {
        word_count(rows) * (64 / self.width as usize)
    }

    fn temp(&mut self, words: usize) -> VarId {
        self.temps.push(words);
        VarId::Temp(self.temps.len() - 1)
    }

    fn broadcast(&mut self, value: u64, count: usize) -> VarId {
        let var = self.temp(count);
        self.ops.push(Operation::Broadcast(var, value & self.mask, count));
        var
    }

    fn constant(&mut self, data: Vec<u64>) -> VarId {
        if let Some(&var) = self.constants.get(&data) {
            if self.loops.is_empty() {
                return var;
            }
        }
        let var = self.temp(data.len());
        self.ops.push(Operation::Constant(var, data.clone()));
        if self.loops.is_empty() {
            self.constants.insert(data, var);
        }
        var
    }

    fn binary(&mut self, op: LogicalOp, a: VarId, b: VarId, words: usize) -> VarId {
        let dst = self.temp(words);
        self.ops.push(match op {
            LogicalOp::And => Operation::And(dst, a, b),
            LogicalOp::Or => Operation::Or(dst, a, b),
            LogicalOp::Xor => Operation::Xor(dst, a, b),
        });
        dst
    }

    fn load(&mut self, src: VarId, index: VarId, words: usize) -> VarId {
        let dst = self.temp(words);
        self.ops.push(Operation::Load(dst, src, index));
        dst
    }

    /// Emit `body(iv)` `trips` times. One trip binds `iv` to constant zero.
    fn for_loop(&mut self, trips: usize, body: impl FnOnce(&mut Self, VarId)) {
        match trips {
            0 => {}
            1 => {
                let iv = self.broadcast(0, 1);
                body(self, iv);
            }
            _ => {
                let trip = self.broadcast(trips as u64, 1);
                let iv = self.temp(1);
                self.ops.push(Operation::ForLoop(iv, trip));
                self.loops.push((iv, self.ops.len()));
                body(self, iv);
                self.loops.pop();
                self.loop_ends.insert(iv, self.ops.len());
            }
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

    /// `out |= rotl(splat(src[S[i]]) & M[i], R[i])` for every step, over
    /// `words` destination words. `tables` are `(S, M, R)` variables and
    /// rows are selected by `row` when the tables hold several lanes.
    fn emit_steps(&mut self, out: VarId, source: VarId, tables: (VarId, VarId, VarId), steps: usize, words: usize) {
        let (s_table, m_table, r_table) = tables;
        self.for_loop(steps, |this, step| {
            let s = this.load(s_table, step, 1);
            let word = this.load(source, s, 1);
            let splat = this.temp(words);
            this.ops.push(Operation::Splat(splat, word));
            let m = this.load(m_table, step, words);
            let r = this.load(r_table, step, words);
            let selected = this.binary(LogicalOp::And, splat, m, words);
            let rotated = this.temp(words);
            this.ops.push(Operation::Rotl(rotated, selected, r));
            this.ops.push(Operation::Or(out, out, rotated));
        });
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

    fn source_live(&mut self, expression: &TensorExpr, groups: usize) -> LoweringResult<Vec<u64>> {
        let support = self.types.infer(expression).map_err(invalid)?;
        let words: Vec<u64> = support.mask().data().to_vec();
        Ok(split(&words, self.width).into_iter().chain(std::iter::repeat(0)).take(groups).collect())
    }

    fn flat(&self, value: &Value) -> LoweringResult<VarId> {
        if value.lanes == 1 {
            Ok(value.var)
        } else {
            Err(invalid("expected a flat bit-vector, found a jagged value"))
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

    fn permute(&mut self, source: VarId, live: &[u64], indices: &JaggedTensor<usize>) -> Value {
        let (moves, exists, lanes) = self.moves(indices);
        let rows = indices.row_count();
        let groups = self.groups(rows);
        let words = lanes * groups;
        let flat: Vec<Vec<(u32, usize)>> = moves.into_iter().flatten().collect();
        let (steps, s, m, r) = self.steps(&flat, live);
        let out = self.broadcast(0, words.max(1));
        if steps > 0 {
            let tables = (self.constant(s), self.constant(m), self.constant(r));
            self.emit_steps(out, source, tables, steps, words);
        }
        Value { var: out, lanes, groups, exists, rows }
    }

    /// `BVGather` (one source) and `BVGather2` (two sources) as one loop over
    /// steps, built directly from the flat element indices.
    ///
    /// A step reads one source word per side, broadcast to every destination
    /// word, and gives each destination word one rotation per side. Within a
    /// step every destination bit holds at most one element, so elements never
    /// mix before the fold; unused destination bits are zero (and are set for
    /// an AND fold), so they leave the accumulator unchanged.
    fn gather_flat(
        &mut self,
        op: LogicalOp,
        inner: LogicalOp,
        sources: &[(VarId, Vec<u64>, &[usize])],
        out: &[usize],
        width: usize,
    ) -> Value {
        let bits = self.width as usize;
        let groups = self.groups(width);
        // Group elements by their source words, then per destination word by
        // their rotations; repeated destination bits go to further steps.
        type Bucket = HashMap<Vec<u32>, Vec<(u32, Vec<u32>)>>;
        let mut by_sources: HashMap<Vec<usize>, Vec<Bucket>> = HashMap::new();
        for (element, &row) in out.iter().enumerate() {
            let (word, destination) = (row / bits, (row % bits) as u32);
            let mut words = Vec::with_capacity(sources.len());
            let mut rotations = Vec::with_capacity(sources.len());
            let mut source_bits = Vec::with_capacity(sources.len());
            let mut lives = Vec::with_capacity(sources.len());
            for (_, live, indices) in sources {
                let index = indices[element];
                let (source_word, bit) = (index / bits, (index % bits) as u32);
                words.push(source_word);
                rotations.push((destination + self.width - bit) % self.width);
                source_bits.push(bit);
                lives.push(live[source_word] & (1u64 << bit) != 0);
            }
            let always_zero = match inner {
                _ if sources.len() == 1 => !lives[0],
                LogicalOp::And => lives.iter().any(|live| !live),
                LogicalOp::Or | LogicalOp::Xor => lives.iter().all(|live| !live),
            };
            if always_zero && op != LogicalOp::And {
                continue; // contributes the OR/XOR identity
            }
            let buckets = by_sources.entry(words).or_insert_with(|| vec![HashMap::new(); groups]);
            buckets[word].entry(rotations).or_default().push((destination, source_bits));
        }
        let mut groups_sorted: Vec<_> = by_sources.into_iter().collect();
        groups_sorted.sort_by(|left, right| left.0.cmp(&right.0));

        let count = sources.len();
        let mut s_tables = vec![Vec::new(); count];
        let mut m_tables = vec![Vec::new(); count];
        let mut r_tables = vec![Vec::new(); count];
        let mut unused = Vec::new();
        let mut steps = 0;
        for (source_words, buckets) in groups_sorted {
            // Assign every element a step offset within this source group.
            let mut slots: Vec<Vec<(Vec<u32>, u64, Vec<u64>)>> = Vec::new(); // [offset][word]
            for (word, bucket) in buckets.into_iter().enumerate() {
                let mut rotations: Vec<_> = bucket.into_iter().collect();
                rotations.sort();
                let mut offset = 0;
                for (rotation, elements) in rotations {
                    let mut layers: Vec<(u64, Vec<u64>)> = Vec::new();
                    for (destination, source_bits) in elements {
                        let layer = layers
                            .iter()
                            .position(|(used, _)| used & (1u64 << destination) == 0)
                            .unwrap_or_else(|| {
                                layers.push((0, vec![0; count]));
                                layers.len() - 1
                            });
                        layers[layer].0 |= 1u64 << destination;
                        for (side, bit) in source_bits.iter().enumerate() {
                            layers[layer].1[side] |= 1u64 << bit;
                        }
                    }
                    for (used, masks) in layers {
                        while slots.len() <= offset {
                            slots.push(vec![(vec![0; count], 0, vec![0; count]); groups]);
                        }
                        slots[offset][word] = (rotation.clone(), used, masks);
                        offset += 1;
                    }
                }
            }
            for slot in slots {
                for side in 0..count {
                    s_tables[side].push(source_words[side] as u64);
                }
                for (rotation, used, masks) in slot {
                    for side in 0..count {
                        m_tables[side].push(masks[side]);
                        r_tables[side].push(u64::from(rotation[side]));
                    }
                    unused.push(!used & self.mask);
                }
                steps += 1;
            }
        }

        let acc = self.broadcast(Self::identity(op), groups);
        if steps > 0 {
            let tables: Vec<(VarId, VarId, VarId)> = (0..count)
                .map(|side| {
                    (
                        self.constant(std::mem::take(&mut s_tables[side])),
                        self.constant(std::mem::take(&mut m_tables[side])),
                        self.constant(std::mem::take(&mut r_tables[side])),
                    )
                })
                .collect();
            let unused = (op == LogicalOp::And).then(|| self.constant(unused));
            let vars: Vec<VarId> = sources.iter().map(|(var, _, _)| *var).collect();
            self.for_loop(steps, |this, step| {
                let mut parts = Vec::with_capacity(count);
                for (side, &(s, m, r)) in tables.iter().enumerate() {
                    let index = this.load(s, step, 1);
                    let word = this.load(vars[side], index, 1);
                    let splat = this.temp(groups);
                    this.ops.push(Operation::Splat(splat, word));
                    let mask = this.load(m, step, groups);
                    let rotation = this.load(r, step, groups);
                    let selected = this.binary(LogicalOp::And, splat, mask, groups);
                    let rotated = this.temp(groups);
                    this.ops.push(Operation::Rotl(rotated, selected, rotation));
                    parts.push(rotated);
                }
                let mut value = parts[0];
                for &part in &parts[1..] {
                    value = this.binary(inner, value, part, groups);
                }
                if let Some(unused) = unused {
                    let missing = this.load(unused, step, groups);
                    value = this.binary(LogicalOp::Or, value, missing, groups);
                }
                let folded = this.binary(op, acc, value, groups);
                this.ops.push(Operation::Copy(acc, folded));
            });
        }
        let result = if op == LogicalOp::And {
            let valid = self.valid_mask(width);
            let valid = self.constant(valid);
            self.binary(LogicalOp::And, acc, valid, groups)
        } else {
            acc
        };
        self.flat_value(result, width)
    }

    fn flat_value(&self, var: VarId, rows: usize) -> Value {
        let exists = vec![self.valid_mask(rows)];
        Value { var, lanes: 1, groups: self.groups(rows), exists, rows }
    }

    fn reduce(&mut self, op: LogicalOp, value: &Value) -> Value {
        let groups = value.groups;
        let acc = self.broadcast(Self::identity(op), groups);
        let absent: Vec<u64> = value
            .exists
            .iter()
            .flat_map(|lane| lane.iter().map(|bits| !bits & self.mask).collect::<Vec<_>>())
            .collect();
        let absent = (op == LogicalOp::And && value.lanes > 0).then(|| self.constant(absent));
        let source = value.var;
        self.for_loop(value.lanes, |this, lane| {
            let mut row = this.load(source, lane, groups);
            if let Some(absent) = absent {
                let missing = this.load(absent, lane, groups);
                row = this.binary(LogicalOp::Or, row, missing, groups);
            }
            let folded = this.binary(op, acc, row, groups);
            this.ops.push(Operation::Copy(acc, folded));
        });
        let result = if op == LogicalOp::And {
            let valid = self.valid_mask(value.rows);
            let valid = self.constant(valid);
            self.binary(LogicalOp::And, acc, valid, groups)
        } else {
            acc
        };
        self.flat_value(result, value.rows)
    }

    /// Pad `value` to `lanes` lanes (missing lanes are zero).
    fn pad(&mut self, value: &Value, lanes: usize) -> Value {
        if value.lanes == lanes {
            return value.clone();
        }
        let groups = value.groups;
        let out = self.broadcast(0, lanes * groups);
        let source = value.var;
        self.for_loop(value.lanes, |this, lane| {
            let row = this.load(source, lane, groups);
            this.ops.push(Operation::Store(out, lane, row));
        });
        let mut exists = value.exists.clone();
        exists.resize(lanes, vec![0; groups]);
        Value { var: out, lanes, groups, exists, rows: value.rows }
    }

    fn expression(&mut self, expression: &TensorExpr) -> LoweringResult<Value> {
        let key = expression as *const TensorExpr;
        if let Some(value) = self.memo.get(&key) {
            return Ok(value.clone());
        }
        let support = self.types.infer(expression).map_err(invalid)?;
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

    fn lower(&mut self, expression: &TensorExpr) -> LoweringResult<Value> {
        Ok(match expression {
            TensorExpr::Input(input) => {
                let arg = self
                    .inputs
                    .iter()
                    .position(|(name, _)| name == &input.name)
                    .ok_or_else(|| invalid(format!("unknown input {:?}", input.name)))?;
                self.flat_value(VarId::Input(arg), input.tensor_type.width())
            }
            TensorExpr::Constant(constant) => {
                let rows = constant.tensor_type.width();
                let mut words = split(constant.value.data(), self.width);
                words.resize(self.groups(rows), 0);
                let valid = self.valid_mask(rows);
                let words = words.iter().zip(&valid).map(|(word, valid)| word & valid).collect();
                let var = self.constant(words);
                self.flat_value(var, rows)
            }
            TensorExpr::ScalarConstant(constant) => {
                let words = split(&[constant.value], self.width);
                let var = self.constant(words);
                self.flat_value(var, 64)
            }
            TensorExpr::BVJaggedPermute(permute) => {
                let source = self.expression(&permute.expr)?;
                let var = self.flat(&source)?;
                let live = self.source_live(&permute.expr, source.groups)?;
                self.permute(var, &live, &permute.indices)
            }
            TensorExpr::BVGather(gather) => {
                let source = self.expression(&gather.expr)?;
                let var = self.flat(&source)?;
                let live = self.source_live(&gather.expr, source.groups)?;
                gather.validate().map_err(invalid)?;
                self.gather_flat(
                    gather.op,
                    LogicalOp::Or,
                    &[(var, live, gather.indices_in.data())],
                    gather.indices_out.data(),
                    gather.width,
                )
            }
            TensorExpr::BVGather2(gather) => {
                let a = self.expression(&gather.a)?;
                let b = self.expression(&gather.b)?;
                let (a_var, b_var) = (self.flat(&a)?, self.flat(&b)?);
                let a_live = self.source_live(&gather.a, a.groups)?;
                let b_live = self.source_live(&gather.b, b.groups)?;
                gather.validate().map_err(invalid)?;
                self.gather_flat(
                    gather.op,
                    gather.inner,
                    &[(a_var, a_live, gather.indices_1.data()), (b_var, b_live, gather.indices_2.data())],
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
                let first = values.first().ok_or_else(|| invalid("logic needs an operand"))?;
                let (rows, groups) = (first.rows, first.groups);
                if values.iter().any(|value| value.rows != rows) {
                    return Err(invalid("logical operands have different widths"));
                }
                let lanes = values.iter().map(|value| value.lanes).max().unwrap_or(1);
                let values: Vec<Value> = values.iter().map(|value| self.pad(value, lanes)).collect();
                let words = lanes * groups;
                let mut result = values[0].var;
                for value in &values[1..] {
                    result = self.binary(logical.op, result, value.var, words);
                }
                let mut exists = vec![vec![0u64; groups]; lanes];
                for value in &values {
                    for (lane, words) in value.exists.iter().enumerate() {
                        for (word, bits) in words.iter().enumerate() {
                            exists[lane][word] |= bits;
                        }
                    }
                }
                Value { var: result, lanes, groups, exists, rows }
            }
            TensorExpr::BTShift(shift) => {
                let source = self.expression(&shift.expr)?;
                let var = self.flat(&source)?;
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
                let permuted = self.permute(var, &live, &indices);
                self.pad(&permuted, 1)
            }
            TensorExpr::BTExpandReduce(change) => {
                let input = self.types.infer(&change.expr).map_err(invalid)?;
                let value = self.expression(&change.expr)?;
                match crate::compile::typing::expand_reduce_form(change, &input).map_err(invalid)? {
                    crate::compile::typing::ExpandReduceForm::Dense(gather) => {
                        let var = self.flat(&value)?;
                        let live = self.source_live(&change.expr, value.groups)?;
                        self.gather_flat(
                            gather.op,
                            LogicalOp::Or,
                            &[(var, live, gather.indices_in.data())],
                            gather.indices_out.data(),
                            gather.width,
                        )
                    }
                    crate::compile::typing::ExpandReduceForm::LaneFold => self.reduce(change.op, &value),
                }
            }
        })
    }
}

fn invalid(message: impl Into<String>) -> LoweringError {
    LoweringError::InvalidInput {
        stage: "loop-program lowering",
        message: message.into(),
    }
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

/// Lower one (already fused) traced function to a loop program.
pub fn lower_loop_program(function: &TensorFn) -> LoweringResult<LoopProgram> {
    let result_type = function.expression.infer_type().map_err(invalid)?;
    let result_bits = result_type.width();
    let width = choose_width(function, result_bits);
    let mut lowerer = Lowerer {
        width,
        mask: if width == 64 { u64::MAX } else { (1u64 << width) - 1 },
        ops: Vec::new(),
        temps: Vec::new(),
        inputs: &function.inputs,
        memo: HashMap::new(),
        types: TypeCache::default(),
        constants: HashMap::new(),
        loops: Vec::new(),
        loop_ends: HashMap::new(),
    };
    let result = lowerer.expression(&function.expression)?;
    let result = lowerer.pad(&result, 1);
    let words = result.groups;
    lowerer.ops.push(Operation::Copy(VarId::Output(0), result.var));

    let ratio = (64 / width) as usize;
    let input_bits: Vec<usize> = function.inputs.iter().map(|(_, t)| t.width()).collect();
    let mut program = LoopProgram {
        name: function.name.clone(),
        word_width: width,
        inputs: input_bits.iter().map(|&bits| (word_count(bits) * ratio, 0..0)).collect(),
        temps: lowerer.temps.iter().map(|&words| (words, 0..0)).collect(),
        outputs: vec![(words, 0..0)],
        ops: lowerer.ops,
        input_bits,
        result_bits,
    };
    compute_liveness(&mut program, &lowerer.loop_ends);
    Ok(program)
}

/// Live ranges: from the first to the last operation touching a variable,
/// extended over every loop that uses a variable defined before it. A loop's
/// induction variable is live exactly over the loop body.
fn compute_liveness(program: &mut LoopProgram, loop_ends: &HashMap<VarId, usize>) {
    let mut ranges: HashMap<VarId, Range<usize>> = HashMap::new();
    let mut touch = |var: VarId, position: usize| {
        let range = ranges.entry(var).or_insert(position..position + 1);
        range.start = range.start.min(position);
        range.end = range.end.max(position + 1);
    };
    for (position, op) in program.ops.iter().enumerate() {
        for var in op.reads().into_iter().chain(op.writes()) {
            touch(var, position);
        }
    }
    for index in 0..program.inputs.len() {
        let range = ranges.entry(VarId::Input(index)).or_insert(0..1);
        range.start = 0;
    }
    let loops: Vec<(usize, usize)> = program
        .ops
        .iter()
        .enumerate()
        .filter_map(|(position, op)| match op {
            Operation::ForLoop(iv, _) => Some((position, loop_ends[iv])),
            _ => None,
        })
        .collect();
    loop {
        let mut changed = false;
        for range in ranges.values_mut() {
            for &(start, end) in &loops {
                let inside = range.end > start + 1 && range.start < end;
                if inside && range.start <= start && range.end < end {
                    range.end = end;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for (position, op) in program.ops.iter().enumerate() {
        if let Operation::ForLoop(iv, _) = op {
            ranges.insert(*iv, position..loop_ends[iv]);
        }
    }
    let get = |var: VarId| ranges.get(&var).cloned().unwrap_or(0..0);
    for (index, entry) in program.inputs.iter_mut().enumerate() {
        entry.1 = get(VarId::Input(index));
    }
    for (index, entry) in program.temps.iter_mut().enumerate() {
        entry.1 = get(VarId::Temp(index));
    }
    for (index, entry) in program.outputs.iter_mut().enumerate() {
        entry.1 = get(VarId::Output(index));
    }
}

fn name(var: VarId) -> String {
    match var {
        VarId::Input(index) => format!("%i{index}"),
        VarId::Temp(index) => format!("%t{index}"),
        VarId::Output(index) => format!("%o{index}"),
    }
}

impl fmt::Display for LoopProgram {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let arguments: Vec<String> = self
            .inputs
            .iter()
            .enumerate()
            .map(|(index, (words, _))| format!("%i{index}: i{}x{words}", self.word_width))
            .collect();
        writeln!(
            out,
            "loops {}({}) -> %o0: i{}x{} {{  // {} operations, {} temps",
            self.name,
            arguments.join(", "),
            self.word_width,
            self.outputs[0].0,
            self.operation_count(),
            self.temps.len()
        )?;
        let mut ends: Vec<usize> = Vec::new();
        for (position, op) in self.ops.iter().enumerate() {
            while ends.last() == Some(&position) {
                ends.pop();
                writeln!(out, "{}}}", "  ".repeat(ends.len() + 1))?;
            }
            let indent = "  ".repeat(ends.len() + 1);
            let mut line = String::new();
            match op {
                Operation::Load(dst, src, index) => write!(line, "{} = load {}[{}]", name(*dst), name(*src), name(*index))?,
                Operation::Store(dst, index, src) => write!(line, "{}[{}] = store {}", name(*dst), name(*index), name(*src))?,
                Operation::Broadcast(dst, value, count) => write!(line, "{} = broadcast {value:#x} x {count}", name(*dst))?,
                Operation::Constant(dst, data) => {
                    let shown: Vec<String> = data.iter().take(6).map(|word| format!("{word:#x}")).collect();
                    let more = if data.len() > 6 { ", ..." } else { "" };
                    write!(line, "{} = const <{}> [{}{more}]", name(*dst), data.len(), shown.join(", "))?
                }
                Operation::Splat(dst, src) => write!(line, "{} = splat {}", name(*dst), name(*src))?,
                Operation::ForLoop(iv, trip) => {
                    writeln!(out, "{indent}for {} in 0..{} {{", name(*iv), name(*trip))?;
                    ends.push(self.live(*iv).end);
                    continue;
                }
                Operation::And(dst, a, b) => write!(line, "{} = and {}, {}", name(*dst), name(*a), name(*b))?,
                Operation::Or(dst, a, b) => write!(line, "{} = or {}, {}", name(*dst), name(*a), name(*b))?,
                Operation::Xor(dst, a, b) => write!(line, "{} = xor {}, {}", name(*dst), name(*a), name(*b))?,
                Operation::Rotl(dst, a, b) => write!(line, "{} = rotl {}, {}", name(*dst), name(*a), name(*b))?,
                Operation::Copy(dst, src) => write!(line, "{} = {}", name(*dst), name(*src))?,
            }
            writeln!(out, "{indent}{line}")?;
        }
        while let Some(_) = ends.pop() {
            writeln!(out, "{}}}", "  ".repeat(ends.len() + 1))?;
        }
        writeln!(out, "  return %o0\n}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::pass::fuse;
    use crate::compile::{
        BTShift, BVJaggedPermute, BVLogical, Constant, Input, JTReduced, JaggedTensor, TensorType,
    };
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

    /// Check traced and fused programs, the loop interpreter, and (with
    /// `llvm`) the JIT at -O0 and -O3 against the reference evaluator.
    fn check(function: &TensorFn) -> LoopProgram {
        let fused = fuse(function);
        let program = lower_loop_program(&fused).unwrap();
        #[cfg(feature = "llvm")]
        let jits: Vec<_> = [0, 3]
            .into_iter()
            .map(|level| crate::lowering::jit::JitFunction::compile(function, level).unwrap_or_else(|error| panic!("{error}")))
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
            assert_eq!(program.evaluate(&arguments), expected.words(), "{program}");
            #[cfg(feature = "llvm")]
            for jit in &jits {
                assert_eq!(jit.call(&arguments.concat()).unwrap(), expected.words(), "JIT of {program}");
            }
        }
        program
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
                    let program = check(&TensorFn::new("split", vec![first.clone(), second.clone()], reduced));
                    assert_eq!(program.word_width, if width <= 32 { 32 } else { 64 });
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
            TensorExpr::from(Constant::new(JaggedTensor::from_flat(words), TensorType::bitvec(width).unwrap()))
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
            check(&TensorFn::new("split", vec![first, second], reduced)).operation_count()
        };
        assert_eq!(count(100), count(400));
    }
}
