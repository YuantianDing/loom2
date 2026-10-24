//! Direct packed-`u64` execution of native tensor functions.
//!
//! This evaluator is the semantic reference for MLIR generation. Pointwise
//! logic operates word-wise; gathers and reductions touch individual bits only
//! where their compile-time index tables require it. No `tensor<...xi1>` value
//! is materialized.

use std::collections::HashMap;

use crate::compile::{LogicalOp, ShiftOp, TensorExpr, TensorFn};

use super::{LoweringError, LoweringResult};

/// Canonically padded packed bit-vector.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedValue {
    words: Vec<u64>,
    width: usize,
}

impl PackedValue {
    pub fn new(words: Vec<u64>, width: usize) -> LoweringResult<Self> {
        if width == 0 || words.len() != width.div_ceil(64) {
            return Err(invalid(format!(
                "width {width} requires {} words, got {}",
                width.div_ceil(64),
                words.len()
            )));
        }
        let mut value = Self { words, width };
        value.clear_padding();
        Ok(value)
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn width(&self) -> usize {
        self.width
    }

    fn set(&mut self, index: usize, bit: bool) {
        if bit {
            self.words[index / 64] |= 1u64 << (index % 64);
        }
    }

    fn clear_padding(&mut self) {
        let tail = self.width % 64;
        if tail != 0 {
            *self.words.last_mut().unwrap() &= (1u64 << tail) - 1;
        }
    }
}

/// A logical value: `rows[d]` lists the lanes of row `d`. Flat bit-vectors
/// have exactly one lane per row.
#[derive(Clone)]
struct Rows(Vec<Vec<bool>>);

impl Rows {
    fn flat(words: &[u64], width: usize) -> Self {
        Self((0..width).map(|bit| vec![words[bit / 64] & (1u64 << (bit % 64)) != 0]).collect())
    }

    fn leaf(&self, index: usize) -> LoweringResult<bool> {
        match self.0.get(index).map(Vec::as_slice) {
            Some([bit]) => Ok(*bit),
            Some([]) => Ok(false),
            Some(_) => Err(invalid("permutation and shift sources must be flat")),
            None => Err(invalid(format!("bit index {index} is out of bounds"))),
        }
    }

    fn reduce(&self, op: LogicalOp) -> Self {
        Self(
            self.0
                .iter()
                .map(|lanes| {
                    vec![lanes.iter().fold(op == LogicalOp::And, |all, &bit| match op {
                        LogicalOp::And => all & bit,
                        LogicalOp::Or => all | bit,
                        LogicalOp::Xor => all ^ bit,
                    })]
                })
                .collect(),
        )
    }

    fn permute(&self, indices: &crate::compile::JaggedTensor<usize>) -> LoweringResult<Self> {
        (0..indices.row_count())
            .map(|row| indices.row(row).iter().map(|&index| self.leaf(index)).collect())
            .collect::<LoweringResult<Vec<_>>>()
            .map(Self)
    }
}

struct Evaluator<'a> {
    inputs: HashMap<&'a str, PackedValue>,
    cache: HashMap<*const TensorExpr, Rows>,
}

impl Evaluator<'_> {
    fn expression(&mut self, expression: &TensorExpr) -> LoweringResult<Rows> {
        let key = expression as *const TensorExpr;
        if let Some(value) = self.cache.get(&key) {
            return Ok(value.clone());
        }
        let value = match expression {
            TensorExpr::Input(input) => {
                let value = self
                    .inputs
                    .get(input.name.as_str())
                    .ok_or_else(|| invalid(format!("missing input {:?}", input.name)))?;
                Rows::flat(&value.words, value.width)
            }
            TensorExpr::Constant(constant) => {
                Rows::flat(constant.value.data(), constant.tensor_type.width())
            }
            TensorExpr::ScalarConstant(constant) => Rows::flat(&[constant.value], 64),
            TensorExpr::BVJaggedPermute(permute) => self.expression(&permute.expr)?.permute(&permute.indices)?,
            TensorExpr::BVGather(gather) => {
                gather.validate().map_err(invalid)?;
                let source = self.expression(&gather.expr)?;
                let mut rows = vec![Vec::new(); gather.width];
                for (&row, &index) in gather.indices_out.data().iter().zip(gather.indices_in.data()) {
                    rows[row].push(source.leaf(index)?);
                }
                Rows(rows).reduce(gather.op)
            }
            TensorExpr::BVGather2(gather) => {
                gather.validate().map_err(invalid)?;
                let (a, b) = (self.expression(&gather.a)?, self.expression(&gather.b)?);
                let mut rows = vec![Vec::new(); gather.width];
                for ((&row, &left), &right) in gather
                    .indices_out
                    .data()
                    .iter()
                    .zip(gather.indices_1.data())
                    .zip(gather.indices_2.data())
                {
                    let (x, y) = (a.leaf(left)?, b.leaf(right)?);
                    rows[row].push(match gather.inner {
                        LogicalOp::And => x & y,
                        LogicalOp::Or => x | y,
                        LogicalOp::Xor => x ^ y,
                    });
                }
                Rows(rows).reduce(gather.op)
            }
            TensorExpr::BVLogical(logical) => {
                let mut operands = Vec::with_capacity(logical.operands.len());
                for operand in &logical.operands {
                    let value = self.expression(&operand.expr)?;
                    operands.push(if operand.reduce_dims.is_empty() {
                        value
                    } else {
                        value.reduce(logical.op)
                    });
                }
                let width = operands
                    .first()
                    .map(|value| value.0.len())
                    .ok_or_else(|| invalid("logical expression has no operands"))?;
                if operands.iter().any(|value| value.0.len() != width) {
                    return Err(invalid("logical operands have different widths"));
                }
                // Missing lanes are zero; a row has as many lanes as its longest operand.
                Rows(
                    (0..width)
                        .map(|row| {
                            let lanes = operands.iter().map(|value| value.0[row].len()).max().unwrap_or(0);
                            (0..lanes)
                                .map(|lane| {
                                    let mut bits = operands
                                        .iter()
                                        .map(|value| value.0[row].get(lane).copied().unwrap_or(false));
                                    let first = bits.next().unwrap_or(false);
                                    bits.fold(first, |all, bit| match logical.op {
                                        LogicalOp::And => all & bit,
                                        LogicalOp::Or => all | bit,
                                        LogicalOp::Xor => all ^ bit,
                                    })
                                })
                                .collect()
                        })
                        .collect(),
                )
            }
            TensorExpr::BTShift(shift) => {
                let source = self.expression(&shift.expr)?;
                let width = source.0.len();
                let bits = (0..width)
                    .map(|destination| {
                        let index = destination.checked_add(shift.shift_amounts).filter(|&index| index < width);
                        Ok(vec![match shift.shift_op {
                            ShiftOp::Logical => index.map(|index| source.leaf(index)).transpose()?.unwrap_or(false),
                            ShiftOp::Arithmetic => {
                                source.leaf(index.unwrap_or(width - 1))?
                            }
                            ShiftOp::Rotate => source.leaf((destination + shift.shift_amounts % width) % width)?,
                        }])
                    })
                    .collect::<LoweringResult<Vec<_>>>()?;
                Rows(bits)
            }
            TensorExpr::BTExpandReduce(change) => {
                let value = self.expression(&change.expr)?;
                let flat = value.0.iter().all(|lanes| lanes.len() <= 1);
                if flat && value.0.len() == change.input_width() {
                    let (out, input) = change.elements();
                    let mut rows = vec![Vec::new(); change.output_width()];
                    for (&row, &index) in out.iter().zip(&input) {
                        rows[row].push(value.leaf(index)?);
                    }
                    Rows(rows).reduce(change.op)
                } else if change.lane_fold().is_some_and(|(rows, _)| rows == value.0.len()) {
                    value.reduce(change.op)
                } else {
                    return Err(invalid("expand_reduce extents do not match its input"));
                }
            }
        };
        self.cache.insert(key, value.clone());
        Ok(value)
    }
}

/// Evaluate a TensorFn using packed words for every argument and result.
pub fn evaluate_tensor_fn(
    function: &TensorFn,
    arguments: &[Vec<u64>],
) -> LoweringResult<PackedValue> {
    if arguments.len() != function.inputs.len() {
        return Err(invalid(format!(
            "function {:?} expects {} arguments, got {}",
            function.name,
            function.inputs.len(),
            arguments.len()
        )));
    }
    let inputs = function
        .inputs
        .iter()
        .zip(arguments)
        .map(|((name, tensor_type), words)| {
            Ok((
                name.as_str(),
                PackedValue::new(words.clone(), tensor_type.bit_width().map_err(invalid)?)?,
            ))
        })
        .collect::<LoweringResult<HashMap<_, _>>>()?;
    let rows = Evaluator {
        inputs,
        cache: HashMap::new(),
    }
    .expression(&function.expression)?;
    let width = rows.0.len();
    let mut result = PackedValue::new(vec![0; width.div_ceil(64)], width)?;
    for bit in 0..width {
        result.set(bit, rows.leaf(bit)?);
    }
    Ok(result)
}

fn invalid(message: impl Into<String>) -> LoweringError {
    LoweringError::InvalidInput {
        stage: "packed TensorFn execution",
        message: message.into(),
    }
}

/// FNV-1a over canonical packed semantic words.
pub fn hash_words(words: &[u64]) -> u64 {
    words.iter().fold(0xcbf29ce484222325, |hash, word| {
        (hash ^ word).wrapping_mul(0x100000001b3)
    })
}

#[cfg(test)]
mod tests {
    use crate::compile::{BVJaggedPermute, Input, JaggedTensor, TensorFn, TensorType};

    use super::evaluate_tensor_fn;

    #[test]
    fn permutes_packed_bits_without_i1_tensors() {
        let tensor_type = TensorType::bitvec(5).unwrap();
        let input = Input::new("bits", tensor_type.clone());
        let gather = BVJaggedPermute::new(input, JaggedTensor::from_flat(vec![4, 0, 2]));
        let function = TensorFn::new("gather", vec![("bits".into(), tensor_type)], gather);
        let result = evaluate_tensor_fn(&function, &[vec![0b10101]]).unwrap();
        assert_eq!(result.words(), &[0b111]);
        assert_eq!(result.width(), 3);
    }
}
