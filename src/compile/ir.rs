//! Top-level typed tensor IR.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyclass;

use super::expr::TensorExpr;
use super::tensor::JaggedTensor;

/// The logical type and support of a packed, possibly jagged bit-vector.
///
/// A value has `width` rows. Rows are packed 64 to a word group, and each word
/// group holds one or more lanes (the jagged dimension produced by
/// `BVJaggedPermute`). `mask` has one word per `(word group, lane)`: its set
/// bits may be one, and every other bit equals `rest`. Lowering never emits an
/// instruction for a lane word whose mask is zero.
///
/// The mask stores `tensor_dims = [word groups]` and, for jagged values, one
/// layer of lane offsets per word group. Flat bit-vectors have exactly one lane
/// per word group and no jagged layer.
///
/// Displayed as `tensor<GROUPSxLANESxu64, WIDTHbits/REST>`, for example
/// `tensor<3x[0-3]xu64, 150bits/0>`.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorType {
    width: usize,
    mask: JaggedTensor<u64>,
    rest: Option<bool>,
}

fn tail(width: usize, group: usize) -> u64 {
    let bits = width - group * 64;
    if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits) - 1
    }
}

impl TensorType {
    /// Construct a type from its width, support mask, and outside value.
    pub fn new(width: usize, mask: JaggedTensor<u64>, rest: Option<bool>) -> Self {
        Self { width, mask, rest }
    }

    /// Fully supported flat bit-vector.
    pub fn bitvec(width: usize) -> Result<Self, String> {
        if width == 0 {
            return Err("bit-vector widths must be positive".into());
        }
        let words = (0..width.div_ceil(64)).map(|group| tail(width, group)).collect();
        Ok(Self::flat(width, words))
    }

    /// Flat bit-vector whose support is `support` (one word per group).
    pub fn flat(width: usize, support: Vec<u64>) -> Self {
        let support = support
            .into_iter()
            .enumerate()
            .map(|(group, word)| word & tail(width, group))
            .collect();
        Self::new(width, JaggedTensor::from_flat(support), Some(false))
    }

    /// Jagged bit-vector with `lanes[group]` support words. Normalized to the
    /// flat form when every word group has exactly one lane.
    pub fn jagged(width: usize, lanes: Vec<Vec<u64>>) -> Self {
        assert_eq!(lanes.len(), width.div_ceil(64), "one lane list per word group");
        if lanes.iter().all(|group| group.len() == 1) {
            return Self::flat(width, lanes.into_iter().map(|group| group[0]).collect());
        }
        let mut offsets = vec![0];
        let mut data = Vec::new();
        for (group, words) in lanes.into_iter().enumerate() {
            data.extend(words.into_iter().map(|word| word & tail(width, group)));
            offsets.push(data.len());
        }
        let groups = offsets.len() - 1;
        Self::new(width, JaggedTensor::new(data, vec![groups], vec![offsets]), Some(false))
    }

    /// Flat type from a contiguous low-bit mask, as sent by Python tracing.
    pub fn from_bitvec_mask(mask: JaggedTensor<u64>) -> Result<Self, String> {
        if !mask.jagged_dims().is_empty() {
            return Err("bit-vector masks cannot have jagged packed-word dimensions".into());
        }
        let Some((&last, prefix)) = mask.data().split_last() else {
            return Err("bit-vector masks must contain at least one packed word".into());
        };
        if prefix.iter().any(|&word| word != u64::MAX) {
            return Err("all non-final bit-vector mask words must be full".into());
        }
        if last == 0 || (last != u64::MAX && last.wrapping_add(1).count_ones() != 1) {
            return Err("the final bit-vector mask word must contain contiguous low bits".into());
        }
        Self::bitvec(prefix.len() * 64 + last.count_ones() as usize)
    }

    /// Support words and their layout.
    pub fn mask(&self) -> &JaggedTensor<u64> {
        &self.mask
    }

    /// Value of every bit outside the support, or None when it is unknown.
    pub fn rest(&self) -> Option<bool> {
        self.rest
    }

    /// Number of logical rows.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Number of logical rows; kept for callers that expect a fallible width.
    pub fn bit_width(&self) -> Result<usize, String> {
        Ok(self.width)
    }

    /// Whether every word group has exactly one lane.
    pub fn is_flat(&self) -> bool {
        self.mask.jagged_dims().is_empty()
    }

    /// Whether this is a fully supported flat bit-vector.
    pub fn is_bitvec(&self) -> bool {
        Self::bitvec(self.width).is_ok_and(|full| &full == self)
    }

    /// Support words of each word group, one per lane.
    pub fn lanes(&self) -> Vec<Vec<u64>> {
        match self.mask.jagged_dims().first() {
            None => self.mask.data().iter().map(|&word| vec![word]).collect(),
            Some(offsets) => offsets
                .windows(2)
                .map(|pair| self.mask.data()[pair[0]..pair[1]].to_vec())
                .collect(),
        }
    }

    /// Number of bits that may be one.
    pub fn support(&self) -> usize {
        self.mask.data().iter().map(|word| word.count_ones() as usize).sum()
    }
}

impl std::fmt::Display for TensorType {
    fn fmt(&self, out: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let lanes = self.lanes();
        let counts: Vec<usize> = lanes.iter().map(Vec::len).collect();
        let (low, high) = (
            counts.iter().copied().min().unwrap_or(0),
            counts.iter().copied().max().unwrap_or(0),
        );
        let lane_text = if low == high { low.to_string() } else { format!("[{low}-{high}]") };
        let rest = match self.rest {
            Some(true) => "1",
            Some(false) => "0",
            None => "?",
        };
        write!(out, "tensor<{}x{lane_text}xu64, {}bits/{rest}", lanes.len(), self.width)?;
        let full: usize = lanes
            .iter()
            .enumerate()
            .map(|(group, words)| words.len() * tail(self.width, group).count_ones() as usize)
            .sum();
        if self.support() != full {
            write!(out, ", {} live", self.support())?;
        }
        out.write_str(">")
    }
}

#[pymethods]
impl TensorType {
    /// Number of logical bits represented by this packed type.
    #[pyo3(name = "bit_width")]
    fn py_bit_width(&self) -> PyResult<usize> {
        self.bit_width().map_err(PyValueError::new_err)
    }

    fn __repr__(&self) -> String {
        self.to_string()
    }
}

/// A traced function expressed as one tensor-expression graph.
///
/// Inputs retain their logical TensorType. The expression is the function
/// result; intermediate computations are represented by shared expression
/// nodes rather than a separate operation for every output bit.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen, unsendable)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TensorFn {
    /// Stable function name used by diagnostics and lowering.
    pub name: String,
    /// Ordered input names and types.
    pub inputs: Vec<(String, TensorType)>,
    /// Root of the result expression graph.
    pub expression: TensorExpr,
}

impl TensorFn {
    /// Construct a traced tensor function.
    pub fn new(
        name: impl Into<String>,
        inputs: Vec<(String, TensorType)>,
        expression: impl Into<TensorExpr>,
    ) -> Self {
        Self {
            name: name.into(),
            inputs,
            expression: expression.into(),
        }
    }
}

#[pymethods]
impl TensorFn {
    #[getter]
    fn py_name(&self) -> &str {
        &self.name
    }

    /// Infer the flattened result width.
    fn result_width(&self) -> PyResult<usize> {
        self.expression
            .infer_type()
            .and_then(|tensor_type| tensor_type.bit_width())
            .map_err(PyValueError::new_err)
    }

    /// Stable textual form of the traced graph.
    fn __str__(&self) -> String {
        self.to_string()
    }

    /// The fused graph and its vector loop program, as used for MLIR emission.
    fn loop_program(&self) -> PyResult<String> {
        let fused = crate::compile::pass::fuse(self);
        crate::lowering::loops::lower_loop_program(&fused)
            .map(|program| program.to_string())
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorFn(name={:?}, inputs={}, result_width={})",
            self.name,
            self.inputs.len(),
            self.result_width()
                .map(|width| width.to_string())
                .unwrap_or_else(|_| "invalid".into())
        )
    }
}
