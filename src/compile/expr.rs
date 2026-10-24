//! Expression nodes for the typed tensor IR.
//!
//! Expressions describe whole tensor operations. Index selections and
//! reductions carry JaggedTensor metadata, avoiding a separate node for every
//! bit produced by tracing.
//!
//! ## Terminology
//!
//! `BT` means **bit-tensor**: the logical tensor contained inside a bit-vector
//! value. A bit-tensor is part of the IR's value semantics. It is distinct from
//! [`JaggedTensor`], which is a Rust storage container used for irregular
//! metadata such as extraction indices.
//!
//! `BV` and `BT` operations assume their input is a bit-vector represented as
//! a one-dimensional `JaggedTensor<u64>`.
//!
//! Every expression evaluates to a bit-vector together with the logical
//! bit-tensor layout described by its [`TensorType`]. The operations below are
//! defined on that logical value; packed `u64` words are a lowering detail.

use std::rc::Rc;

use enum_dispatch::enum_dispatch;
use pyo3_stub_gen::derive::gen_stub_pyclass;
use smallvec::SmallVec;

use super::ir::TensorType;
use super::tensor::JaggedTensor;

/// Common inspection interface for expression nodes. Types are inferred for
/// a whole graph at once by [`TensorExpr::infer_type`].
#[enum_dispatch]
pub trait TExpr {
    /// Direct child expressions, in evaluation order.
    fn children(&self) -> Vec<Rc<TensorExpr>>;

}

/// Any expression in a TensorFn.
#[enum_dispatch(TExpr)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TensorExpr {
    /// Read one named function argument without changing its type or layout.
    Input,
    /// Produce a compile-time value with an explicitly declared type.
    Constant,
    ScalarConstant,
    /// Apply associative Boolean logic, including logical reductions.
    BVLogical,
    /// A jagged permutation fused with the reduction of its lanes.
    BVGather,
    /// Two jagged permutations, a binary logic, and a reduction, fused.
    BVGather2,
    /// Permute source bits into jagged lanes of packed output words.
    BVJaggedPermute,
    /// Reclassify bit-tensor dimensions as batch, expanded, or reduced.
    BTExpandReduce,
    /// Shift or rotate bits within the bit-tensor value.
    BTShift,
}

/// Read a named input to a traced function.
///
/// Evaluating this node returns the argument value identified by [`Input::name`]
/// with exactly the declared [`Input::tensor_type`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Input {
    /// Name used to resolve the argument in the enclosing TensorFn.
    pub name: String,
    /// Logical bit-tensor type required for the argument.
    pub tensor_type: TensorType,
}

impl Input {
    pub fn new(name: impl Into<String>, tensor_type: TensorType) -> Self {
        Self {
            name: name.into(),
            tensor_type,
        }
    }
}

impl TExpr for Input {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        Vec::new()
    }
}

/// Associative Boolean operation used for pointwise logic and reductions.
///
/// The same operator is used both to reduce each [`JTReduced`] operand and
/// to combine the resulting operands pointwise. The identity values are true
/// for `And` and false for `Or` and `Xor`; these also define empty reductions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LogicalOp {
    /// Boolean conjunction: a result bit is set only if every input bit is set.
    And,
    /// Boolean disjunction: a result bit is set if any input bit is set.
    Or,
    /// Boolean parity: a result bit is set if an odd number of inputs are set.
    Xor,
}

/// A pointwise Boolean operation, optionally reducing dimensions per operand.
///
/// Evaluation has two steps:
///
/// 1. Each operand is reduced over its [`JTReduced::reduce_dims`] using
///    [`BVLogical::op`]. An operand with no reduced dimensions is unchanged.
/// 2. The remaining, shape-compatible operands are combined elementwise using
///    the same operator.
///
/// Consequently `Or`, `And`, and `Xor` reductions represent `exists`,
/// `forall`, and `xorsum`, respectively. Operands may themselves be
/// `BVLogical` nodes with the same operator; they are not flattened, so a
/// chain such as `a |= x` stays a chain of shared nodes rather than
/// re-listing every earlier operand. Boolean NOT has no separate
/// node and can be represented by XOR with a constant-one value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BVLogical {
    /// Operator used for both reductions and pointwise combination.
    pub op: LogicalOp,
    /// Inputs and the bit-tensor dimensions reduced from each input.
    pub operands: SmallVec<[JTReduced; 2]>,
}

impl BVLogical {
    pub fn new(op: LogicalOp, operands: impl IntoIterator<Item = JTReduced>) -> Self {
        Self {
            op,
            operands: operands.into_iter().collect(),
        }
    }
}

impl TExpr for BVLogical {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        self.operands
            .iter()
            .map(|operand| Rc::clone(&operand.expr))
            .collect()
    }
}

/// One logical operand, optionally reduced over its jagged dimension.
///
/// A reduced operand folds the lanes of each row (the jagged dimension of the
/// operand's [`TensorType`]) with the enclosing [`BVLogical::op`], producing a
/// flat bit-vector with one bit per row; empty rows fold to the identity.
/// `reduce_dims` names the reduced logical axis (always the last one, since
/// quantifiers only apply to the last dimension). Reduction happens before the
/// operand is combined pointwise with its siblings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JTReduced {
    /// Expression evaluated before reduction.
    pub expr: Rc<TensorExpr>,
    /// Logical bit-tensor axes removed from the expression; empty for none.
    pub reduce_dims: SmallVec<[usize; 1]>,
}

impl JTReduced {
    pub fn new(expr: impl Into<TensorExpr>, reduce_dims: impl IntoIterator<Item = usize>) -> Self {
        Self::shared(Rc::new(expr.into()), reduce_dims)
    }

    pub fn shared(expr: Rc<TensorExpr>, reduce_dims: impl IntoIterator<Item = usize>) -> Self {
        Self {
            expr,
            reduce_dims: reduce_dims.into_iter().collect(),
        }
    }
}

/// Permute a bit-vector into jagged lanes of packed 64-bit words.
///
/// This operation assumes that `expr` evaluates to a bit-vector represented by
/// a one-dimensional `JaggedTensor<u64>`. Source bit `s` is bit `s % 64` of
/// source word `s / 64`.
///
/// Each outer logical position `d` in `indices` denotes destination bit `d`.
/// The jagged list at that position contains one source bit for each output
/// lane. For every `s = indices[d][lane]`, evaluation performs:
///
/// ```text
/// output[d / 64][lane][d % 64] = expr[s]
/// ```
///
/// Thus the outer index position selects a destination word and bit, while the
/// position inside its jagged list selects an output lane. Entries with the
/// same lane number are transposed into the same packed output value. Output
/// bits without a source entry are zero.
///
/// The logical output width is the number of outer destination positions in
/// `indices`; no separate width is stored. The output contains
/// `ceil(width / 64)` word groups. The lane count of word group `w` is the
/// maximum jagged-list length among the destination positions assigned to that
/// word, so different word groups may have different lane counts.
///
/// Source indices must be in bounds for `expr`. There is no implicit wrapping.
/// Repeated source indices are allowed and copy the source bit into each named
/// destination. Empty jagged lists leave that destination bit zero in every
/// lane and still count toward the logical output width.
///
/// # Example
///
/// Given `a = 0b11010110` and 65 outer destination positions:
///
/// ```text
/// indices = [[1, 2, 3], [4, 5], [6]] + [[]] * 61 + [[2, 3]]
/// ```
///
/// Positions 0, 1, and 2 belong to packed word 0. Position 64 belongs to
/// packed word 1. The result is:
///
/// ```text
/// [
///     [
///         (a[1] << 0) | (a[4] << 1) | (a[6] << 2),
///         (a[2] << 0) | (a[5] << 1),
///          a[3] << 0,
///     ],
///     [
///         a[2] << 0,
///         a[3] << 0,
///     ],
/// ]
/// ```
///
/// Lowering can realize this mapping with masks, shifts, and the packed
/// permutation circuits in `lowering::circuit`; it does not need to materialize
/// a separate IR operation for every bit.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BVJaggedPermute {
    /// Source bit-vector, packed as a one-dimensional sequence of `u64` words.
    pub expr: Rc<TensorExpr>,
    /// One jagged list of source bit positions per logical destination bit.
    ///
    /// A value at `indices[d][lane]` supplies destination bit `d` in output
    /// lane `lane`. The number of outer positions determines the output width.
    pub indices: JaggedTensor<usize>,
}

impl BVJaggedPermute {
    pub fn new(expr: impl Into<TensorExpr>, indices: JaggedTensor<usize>) -> Self {
        Self {
            expr: Rc::new(expr.into()),
            indices,
        }
    }
}

impl TExpr for BVJaggedPermute {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        vec![Rc::clone(&self.expr)]
    }
}

/// A [`BVJaggedPermute`] fused with the reduction of its lanes.
///
/// Like [`BVGather2`], the indices are flat with one entry per element `e`:
///
/// ```text
/// output[indices_out[e]] = op over e of expr[indices_in[e]]
/// ```
///
/// This is what `a[index].exists()` (or `forall`/`xorsum`) traces to after
/// fusion. The output has `width` rows; a row that no element targets yields
/// the identity of `op`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BVGather {
    /// Flat source bit-vector.
    pub expr: Rc<TensorExpr>,
    /// Destination row of each element.
    pub indices_out: JaggedTensor<usize>,
    /// Source bit read by each element.
    pub indices_in: JaggedTensor<usize>,
    /// Number of output rows.
    pub width: usize,
    /// Fold operator.
    pub op: LogicalOp,
}

impl BVGather {
    pub fn new(
        expr: impl Into<Rc<TensorExpr>>,
        indices_out: JaggedTensor<usize>,
        indices_in: JaggedTensor<usize>,
        width: usize,
        op: LogicalOp,
    ) -> Self {
        Self {
            expr: expr.into(),
            indices_out,
            indices_in,
            width,
            op,
        }
    }

    /// The gather equivalent to reducing `permute(expr, rows)` with `op`.
    pub fn from_rows(expr: impl Into<Rc<TensorExpr>>, rows: &JaggedTensor<usize>, op: LogicalOp) -> Self {
        let width = rows.row_count();
        let out = (0..width).flat_map(|row| std::iter::repeat_n(row, rows.row(row).len())).collect();
        Self::new(expr, JaggedTensor::from_flat(out), JaggedTensor::from_flat(rows.data().to_vec()), width, op)
    }

    /// Check that both index tensors are flat and equally long, and that
    /// every destination row is below `width`.
    pub fn validate(&self) -> Result<(), String> {
        validate_elements(&[&self.indices_out, &self.indices_in], self.width)
    }
}

/// Shared checks for flat gather indices; `indices[0]` holds destination rows.
pub(crate) fn validate_elements(indices: &[&JaggedTensor<usize>], width: usize) -> Result<(), String> {
    if indices.iter().any(|tensor| !tensor.jagged_dims().is_empty()) {
        return Err("gather indices must be one-dimensional".into());
    }
    let count = indices[0].len();
    if indices.iter().any(|tensor| tensor.len() != count) {
        return Err("gather index tensors must have the same length".into());
    }
    if width == 0 {
        return Err("a gather must produce at least one row".into());
    }
    match indices[0].data().iter().find(|&&row| row >= width) {
        Some(row) => Err(format!("gather destination row {row} is outside width {width}")),
        None => Ok(()),
    }
}

impl TExpr for BVGather {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        vec![Rc::clone(&self.expr)]
    }
}

/// Two [`BVJaggedPermute`]s combined pointwise and reduced, fused.
///
/// The three index tensors are flat and have one entry per element `e`:
///
/// ```text
/// output[indices_out[e]] = op over e of inner(a[indices_1[e]], b[indices_2[e]])
/// ```
///
/// This is `(a[p] & b[q]).exists()`, displayed as `gather2.or.and`. The
/// output has `width` rows; a row that no element targets yields the identity
/// of `op`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BVGather2 {
    pub a: Rc<TensorExpr>,
    pub b: Rc<TensorExpr>,
    /// Destination row of each element.
    pub indices_out: JaggedTensor<usize>,
    /// Bit of `a` read by each element.
    pub indices_1: JaggedTensor<usize>,
    /// Bit of `b` read by each element.
    pub indices_2: JaggedTensor<usize>,
    /// Number of output rows.
    pub width: usize,
    /// Pointwise operator combining the two bits of an element.
    pub inner: LogicalOp,
    /// Reduction operator over the elements of a row.
    pub op: LogicalOp,
}

impl BVGather2 {
    /// Check that the three index tensors are flat, equally long, and that
    /// every destination row is below `width`.
    pub fn validate(&self) -> Result<(), String> {
        validate_elements(&[&self.indices_out, &self.indices_1, &self.indices_2], self.width)
    }
}

impl TExpr for BVGather2 {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        vec![Rc::clone(&self.a), Rc::clone(&self.b)]
    }
}

impl TensorExpr {
    /// Infer the type (layout and support) of this expression graph.
    pub fn infer_type(&self) -> Result<TensorType, String> {
        super::typing::TypeCache::default().infer(self)
    }
}

/// Reshape a bit-tensor by batching, expanding, and reducing dimensions.
///
/// `BT` means bit-tensor—the tensor contained inside a bit-vector value—not a
/// [`JaggedTensor`]. Every dimension carries its extent:
///
/// ```text
/// %a : bits[64]
/// expand_reduce.or %a [4, expand(4), reduce(4), 4] : bits[64]
/// ```
///
/// The input is read row-major with the shape of its batch and reduce
/// dimensions (`[4, 4, 4]` above); the output is row-major with the shape of
/// its batch and expand dimensions (`[4, 4, 4]`). An expanded dimension
/// broadcasts, and every reduced dimension is folded with [`BTExpandReduce::op`]
/// (empty reductions give its identity).
///
/// Over a jagged (lane-layout) input, `[rows, reduce(lanes)]` folds the lanes
/// of each row; lanes a row does not have count as the identity. This is what
/// `forall`, `exists`, and `xorsum` trace to.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BTExpandReduce {
    /// Bit-tensor value whose dimensions are transformed.
    pub expr: Rc<TensorExpr>,
    /// Role and extent of each dimension, outermost first.
    pub bt_dims: SmallVec<[ExpandReduceDimType; 4]>,
    /// Fold operator of the reduced dimensions.
    pub op: LogicalOp,
}

impl BTExpandReduce {
    pub fn new(
        expr: impl Into<TensorExpr>,
        bt_dims: impl IntoIterator<Item = ExpandReduceDimType>,
        op: LogicalOp,
    ) -> Self {
        Self {
            expr: Rc::new(expr.into()),
            bt_dims: bt_dims.into_iter().collect(),
            op,
        }
    }

    fn product(&self, keep: impl Fn(&ExpandReduceDimType) -> bool) -> usize {
        self.bt_dims.iter().filter(|dim| keep(dim)).map(ExpandReduceDimType::extent).product()
    }

    /// Bits of the dense input: the product of batch and reduce extents.
    pub fn input_width(&self) -> usize {
        self.product(|dim| !matches!(dim, ExpandReduceDimType::Expand(_)))
    }

    /// Bits of the output: the product of batch and expand extents.
    pub fn output_width(&self) -> usize {
        self.product(|dim| !matches!(dim, ExpandReduceDimType::Reduce(_)))
    }

    /// Whether some dimension is reduced.
    pub fn reduces(&self) -> bool {
        self.bt_dims.iter().any(|dim| matches!(dim, ExpandReduceDimType::Reduce(_)))
    }

    /// `Some((rows, lanes))` for the lane-fold form `[rows, reduce(lanes)]`
    /// (or `[reduce(lanes)]`, one row).
    pub fn lane_fold(&self) -> Option<(usize, usize)> {
        match self.bt_dims.as_slice() {
            [ExpandReduceDimType::Reduce(lanes)] => Some((1, *lanes)),
            [ExpandReduceDimType::Batch(rows), ExpandReduceDimType::Reduce(lanes)] => Some((*rows, *lanes)),
            _ => None,
        }
    }

    /// The dense semantics as gather elements `(output bit, input bit)`.
    pub fn elements(&self) -> (Vec<usize>, Vec<usize>) {
        let extents: Vec<usize> = self.bt_dims.iter().map(ExpandReduceDimType::extent).collect();
        let total: usize = extents.iter().product();
        let (mut out, mut input) = (Vec::with_capacity(total), Vec::with_capacity(total));
        for linear in 0..total {
            // Decode the multi-index, outermost dimension first.
            let mut rest = linear;
            let mut index = vec![0; extents.len()];
            for (position, &extent) in extents.iter().enumerate().rev() {
                index[position] = rest % extent;
                rest /= extent;
            }
            let (mut o, mut i) = (0, 0);
            for (dim, &value) in self.bt_dims.iter().zip(&index) {
                match dim {
                    ExpandReduceDimType::Batch(extent) => {
                        o = o * extent + value;
                        i = i * extent + value;
                    }
                    ExpandReduceDimType::Expand(extent) => o = o * extent + value,
                    ExpandReduceDimType::Reduce(extent) => i = i * extent + value,
                }
            }
            out.push(o);
            input.push(i);
        }
        (out, input)
    }
}

impl TExpr for BTExpandReduce {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        vec![Rc::clone(&self.expr)]
    }
}

/// Role and extent of one bit-tensor dimension in [`BTExpandReduce`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpandReduceDimType {
    /// An input dimension preserved in the result.
    Batch(usize),
    /// A result dimension that broadcasts the input across it.
    Expand(usize),
    /// An input dimension folded with the node's operator.
    Reduce(usize),
}

impl ExpandReduceDimType {
    pub fn extent(&self) -> usize {
        match *self {
            ExpandReduceDimType::Batch(extent) | ExpandReduceDimType::Expand(extent) | ExpandReduceDimType::Reduce(extent) => extent,
        }
    }
}

/// Shift or rotate the bit dimension of a bit-tensor.
///
/// `BT` means bit-tensor—the tensor contained inside a bit-vector value—not a
/// [`JaggedTensor`].
///
/// The operation moves bits toward lower logical bit indices by
/// [`BTShift::shift_amounts`]. It preserves the bit-tensor shape. Amounts at
/// least as large as the bit width follow the selected [`ShiftOp`] semantics;
/// rotation reduces the amount modulo the width.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BTShift {
    /// Bit-tensor value to shift.
    pub expr: Rc<TensorExpr>,
    /// Filling or wrapping behavior at the vacated high-bit positions.
    pub shift_op: ShiftOp,
    /// Unsigned number of logical bit positions to move.
    pub shift_amounts: usize,
}

impl BTShift {
    pub fn new(expr: impl Into<TensorExpr>, shift_op: ShiftOp, shift_amounts: usize) -> Self {
        Self {
            expr: Rc::new(expr.into()),
            shift_op,
            shift_amounts,
        }
    }
}

impl TExpr for BTShift {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        vec![Rc::clone(&self.expr)]
    }
}

/// Bit-shift behavior for [`BTShift`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShiftOp {
    /// Right shift, discarding low bits and inserting zero high bits.
    Logical,
    /// Right shift, discarding low bits and repeating the original sign bit.
    Arithmetic,
    /// Right rotation, wrapping discarded low bits into the high positions.
    Rotate,
}

/// Produce a tensor-shaped compile-time constant.
///
/// [`Constant::value`] supplies the packed constant data and its logical
/// grouping; [`Constant::tensor_type`] determines valid bits and the value of
/// bits outside the explicit mask. Evaluation performs no runtime reads.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Constant {
    /// Compile-time packed words and their tensor layout.
    pub value: JaggedTensor<u64>,
    /// Declared logical type of the constant.
    pub tensor_type: TensorType,
}

impl Constant {
    pub fn new(value: JaggedTensor<u64>, tensor_type: TensorType) -> Self {
        Self { value, tensor_type }
    }
}

impl TExpr for Constant {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        Vec::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ScalarConstant {
    pub value: u64,
}

impl TExpr for ScalarConstant {
    fn children(&self) -> Vec<Rc<TensorExpr>> {
        Vec::new()
    }
}
