//! Type and support inference.
//!
//! A [`TensorType`] mask over-approximates the bits that may be one; all other
//! bits are zero. Logic with constants therefore changes the mask:
//!
//! ```text
//! and: support = intersection      a & mask  keeps only `mask`
//! or:  support = union             a | c     adds `c`
//! xor: support = union
//! ```
//!
//! Lowering emits nothing for a lane word whose support is empty.
//!
//! Layout rules. `BVJaggedPermute` produces one row per index row and one lane
//! per row element; reductions fold the lane dimension back to a flat
//! bit-vector; pointwise logic pads missing lanes with zeros. `BVGather` is a
//! permute fused with its reduction, and knows each row's length, so an
//! AND-gather of an empty row is correctly supported (it folds to one).

use std::collections::HashMap;

use std::rc::Rc;

use super::{JaggedTensor, LogicalOp, TensorExpr, TensorType};

fn group_tail(width: usize, group: usize) -> u64 {
    let bits = width - group * 64;
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

fn source_support(source: &TensorType) -> Result<Vec<u64>, String> {
    if !source.is_flat() {
        return Err("a permutation source must be a flat bit-vector".into());
    }
    Ok(source.mask().data().to_vec())
}

fn rows(indices: &JaggedTensor<usize>) -> Vec<&[usize]> {
    (0..indices.row_count()).map(|row| indices.row(row)).collect()
}

fn check_indices(source: &TensorType, indices: &JaggedTensor<usize>) -> Result<(), String> {
    if indices.jagged_dims().len() > 1 {
        return Err("nested jagged permutation indices are not supported".into());
    }
    if indices.row_count() == 0 {
        return Err("a permutation must produce at least one row".into());
    }
    match indices.data().iter().find(|&&index| index >= source.width()) {
        Some(index) => Err(format!(
            "permutation index {index} is out of bounds for width {}",
            source.width()
        )),
        None => Ok(()),
    }
}

fn live(support: &[u64], index: usize) -> bool {
    support[index / 64] & (1u64 << (index % 64)) != 0
}

/// Type of `BVJaggedPermute(source, indices)`.
pub fn permute_type(source: &TensorType, indices: &JaggedTensor<usize>) -> Result<TensorType, String> {
    check_indices(source, indices)?;
    let support = source_support(source)?;
    let rows = rows(indices);
    let width = rows.len();
    let mut lanes: Vec<Vec<u64>> = (0..width.div_ceil(64))
        .map(|group| {
            let longest = rows[group * 64..width.min(group * 64 + 64)]
                .iter()
                .map(|row| row.len())
                .max()
                .unwrap_or(0);
            vec![0; longest]
        })
        .collect();
    for (row, sources) in rows.iter().enumerate() {
        for (lane, &index) in sources.iter().enumerate() {
            if live(&support, index) {
                lanes[row / 64][lane] |= 1u64 << (row % 64);
            }
        }
    }
    Ok(TensorType::jagged(width, lanes))
}

/// Type of `BVGather`: exact, using which rows each element targets.
pub fn gather_type(gather: &crate::compile::BVGather, source: &TensorType) -> Result<TensorType, String> {
    gather.validate()?;
    let support = source_support(source)?;
    if let Some(index) = gather.indices_in.data().iter().find(|&&index| index >= source.width()) {
        return Err(format!("gather index {index} is out of bounds for width {}", source.width()));
    }
    let elements = gather.indices_in.data().iter().map(|&index| live(&support, index));
    Ok(fold_rows(gather.width, gather.indices_out.data(), elements, gather.op))
}

/// Support of folding element bits (`true` = may be one) into their rows.
fn fold_rows(width: usize, rows: &[usize], elements: impl Iterator<Item = bool>, op: LogicalOp) -> TensorType {
    let mut any = vec![false; width];
    let mut all = vec![true; width];
    for (&row, element) in rows.iter().zip(elements) {
        any[row] |= element;
        all[row] &= element;
    }
    let mut words = vec![0u64; width.div_ceil(64)];
    for row in 0..width {
        let may_be_one = match op {
            LogicalOp::And => all[row],
            LogicalOp::Or | LogicalOp::Xor => any[row],
        };
        if may_be_one {
            words[row / 64] |= 1u64 << (row % 64);
        }
    }
    TensorType::flat(width, words)
}

/// Type of `BVGather2`: exact, using which rows each element targets.
pub fn gather2_type(gather: &crate::compile::BVGather2, a: &TensorType, b: &TensorType) -> Result<TensorType, String> {
    gather.validate()?;
    if !a.is_flat() || !b.is_flat() {
        return Err("gather2 sources must be flat bit-vectors".into());
    }
    for (indices, source) in [(&gather.indices_1, a), (&gather.indices_2, b)] {
        if let Some(index) = indices.data().iter().find(|&&index| index >= source.width()) {
            return Err(format!("gather2 index {index} is out of bounds for width {}", source.width()));
        }
    }
    let (a_support, b_support) = (a.mask().data(), b.mask().data());
    let elements = gather.indices_1.data().iter().zip(gather.indices_2.data()).map(|(&left, &right)| {
        let (x, y) = (live(a_support, left), live(b_support, right));
        match gather.inner {
            LogicalOp::And => x && y,
            LogicalOp::Or | LogicalOp::Xor => x || y,
        }
    });
    Ok(fold_rows(gather.width, gather.indices_out.data(), elements, gather.op))
}

/// How a `BTExpandReduce` applies to an input of type `input`.
pub enum ExpandReduceForm {
    /// Dense row-major reshape of a flat input: a gather over these elements.
    Dense(crate::compile::BVGather),
    /// Fold the lanes of each row of a jagged input.
    LaneFold,
}

/// Decide the form of `node` over `input`, checking its extents.
pub fn expand_reduce_form(
    node: &crate::compile::BTExpandReduce,
    input: &TensorType,
) -> Result<ExpandReduceForm, String> {
    if input.is_flat() && input.width() == node.input_width() {
        let (out, input_bits) = node.elements();
        let gather = crate::compile::BVGather::new(
            Rc::clone(&node.expr),
            JaggedTensor::from_flat(out),
            JaggedTensor::from_flat(input_bits),
            node.output_width(),
            node.op,
        );
        return Ok(ExpandReduceForm::Dense(gather));
    }
    match node.lane_fold() {
        Some((rows, lanes)) if rows == input.width() && input.lanes().iter().all(|group| group.len() <= lanes) => {
            Ok(ExpandReduceForm::LaneFold)
        }
        _ => Err(format!(
            "expand_reduce over {} input bits needs {} (batch x reduce) bits, or [rows, reduce(lanes)] over a jagged input",
            input.width(),
            node.input_width()
        )),
    }
}

/// Type of `BTExpandReduce`.
pub fn expand_reduce_type(node: &crate::compile::BTExpandReduce, input: &TensorType) -> Result<TensorType, String> {
    if node.output_width() == 0 {
        return Err("expand_reduce must produce at least one bit".into());
    }
    match expand_reduce_form(node, input)? {
        ExpandReduceForm::Dense(gather) => gather_type(&gather, input),
        ExpandReduceForm::LaneFold => Ok(reduce_type(node.op, input)),
    }
}

/// Type of folding the lane dimension of `operand` with `op`.
pub fn reduce_type(op: LogicalOp, operand: &TensorType) -> TensorType {
    let width = operand.width();
    let words = operand
        .lanes()
        .iter()
        .enumerate()
        .map(|(group, lanes)| match op {
            // Row lengths are not part of the type, and an empty row folds to
            // one, so an AND reduction is conservatively fully supported.
            LogicalOp::And => group_tail(width, group),
            LogicalOp::Or | LogicalOp::Xor => lanes.iter().fold(0, |all, &word| all | word),
        })
        .collect();
    TensorType::flat(width, words)
}

/// Type of pointwise logic over already-reduced operand types.
pub fn logical_type(op: LogicalOp, operands: &[TensorType]) -> Result<TensorType, String> {
    let first = operands.first().ok_or("logical expressions require at least one operand")?;
    let width = first.width();
    if let Some(other) = operands.iter().find(|operand| operand.width() != width) {
        return Err(format!(
            "logical operand has width {}; expected {width}",
            other.width()
        ));
    }
    let all: Vec<Vec<Vec<u64>>> = operands.iter().map(TensorType::lanes).collect();
    let lanes = (0..width.div_ceil(64))
        .map(|group| {
            let count = all.iter().map(|lanes| lanes[group].len()).max().unwrap_or(0);
            (0..count)
                .map(|lane| {
                    let words = all.iter().map(|lanes| lanes[group].get(lane).copied().unwrap_or(0));
                    match op {
                        LogicalOp::And => words.fold(u64::MAX, |all, word| all & word),
                        LogicalOp::Or | LogicalOp::Xor => words.fold(0, |all, word| all | word),
                    }
                })
                .collect()
        })
        .collect();
    Ok(TensorType::jagged(width, lanes))
}

/// Memoized type inference over a shared expression DAG.
#[derive(Default)]
pub struct TypeCache {
    types: HashMap<*const TensorExpr, TensorType>,
}

impl TypeCache {
    pub fn infer(&mut self, expression: &TensorExpr) -> Result<TensorType, String> {
        let key = expression as *const TensorExpr;
        if let Some(tensor_type) = self.types.get(&key) {
            return Ok(tensor_type.clone());
        }
        let tensor_type = match expression {
            TensorExpr::Input(input) => input.tensor_type.clone(),
            TensorExpr::Constant(constant) => {
                let width = constant.tensor_type.width();
                if constant.value.data().len() != width.div_ceil(64) {
                    return Err("constant word count does not match its width".into());
                }
                // A constant's support is exactly its one bits.
                TensorType::flat(width, constant.value.data().to_vec())
            }
            TensorExpr::ScalarConstant(constant) => TensorType::flat(64, vec![constant.value]),
            TensorExpr::BVJaggedPermute(permute) => {
                let source = self.infer(&permute.expr)?;
                permute_type(&source, &permute.indices)?
            }
            TensorExpr::BVGather(gather) => {
                let source = self.infer(&gather.expr)?;
                gather_type(gather, &source)?
            }
            TensorExpr::BVGather2(gather) => {
                let a = self.infer(&gather.a)?;
                let b = self.infer(&gather.b)?;
                gather2_type(gather, &a, &b)?
            }
            TensorExpr::BVLogical(logical) => {
                let operands = logical
                    .operands
                    .iter()
                    .map(|operand| {
                        let tensor_type = self.infer(&operand.expr)?;
                        Ok(if operand.reduce_dims.is_empty() {
                            tensor_type
                        } else {
                            reduce_type(logical.op, &tensor_type)
                        })
                    })
                    .collect::<Result<Vec<_>, String>>()?;
                logical_type(logical.op, &operands)?
            }
            TensorExpr::BTShift(shift) => {
                let source = self.infer(&shift.expr)?;
                if !source.is_flat() {
                    return Err("shifts require a flat bit-vector".into());
                }
                TensorType::bitvec(source.width())?
            }
            TensorExpr::BTExpandReduce(change) => {
                let input = self.infer(&change.expr)?;
                expand_reduce_type(change, &input)?
            }
        };
        self.types.insert(key, tensor_type.clone());
        Ok(tensor_type)
    }
}
