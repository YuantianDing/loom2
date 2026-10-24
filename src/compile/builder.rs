//! PyO3 builder used by the tensor-wide Python tracer.

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;
use smallvec::SmallVec;

use std::rc::Rc;

use super::typing::{expand_reduce_type, logical_type, permute_type, reduce_type};
use super::{
    BTExpandReduce, BTShift, ShiftOp, BVJaggedPermute, BVLogical, Constant, ExpandReduceDimType, Input, JTReduced, JaggedTensor, LogicalOp,
    TensorExpr, TensorFn, TensorType,
};

/// Opaque expression owned by Python while one function is being traced.
///
/// The node is shared (`Rc`) so every Python use of one value refers to the
/// same graph node, and its type is cached so each new node is validated
/// against its operands without re-walking the whole graph.
#[pyclass(module = "loom2._loom2", frozen, unsendable)]
#[derive(Clone)]
pub struct TensorExprHandle {
    pub(crate) expression: Rc<TensorExpr>,
    pub(crate) tensor_type: TensorType,
}

impl TensorExprHandle {
    fn new(expression: impl Into<TensorExpr>, tensor_type: TensorType) -> Self {
        Self {
            expression: Rc::new(expression.into()),
            tensor_type,
        }
    }
}

/// Incrementally constructs one native [`TensorFn`].
#[pyclass(module = "loom2._loom2", unsendable)]
pub struct TensorFnBuilder {
    name: String,
    inputs: Vec<(String, TensorType)>,
    finished: bool,
}

fn parts<T>(value: &Bound<'_, PyAny>) -> PyResult<JaggedTensor<T>>
where
    for<'a, 'py> T: FromPyObject<'a, 'py> + Clone,
{
    let data = value.getattr("data")?.extract::<Vec<T>>()?;
    let jagged_dims = value.getattr("jagged_dims")?.extract::<Vec<Vec<usize>>>()?;
    let tensor_dims = match value.getattr("tensor_dims") {
        Ok(dimensions) => dimensions.extract::<Vec<usize>>()?,
        Err(_) if jagged_dims.is_empty() => vec![data.len()],
        Err(_) => vec![jagged_dims[0].len().saturating_sub(1)],
    };
    JaggedTensor::try_new(data, tensor_dims, jagged_dims).map_err(PyValueError::new_err)
}

fn handle(value: &Bound<'_, PyAny>) -> PyResult<TensorExprHandle> {
    Ok(value.extract::<PyRef<'_, TensorExprHandle>>()?.clone())
}

fn logical_op(operation: &str) -> PyResult<LogicalOp> {
    match operation {
        "and" => Ok(LogicalOp::And),
        "or" => Ok(LogicalOp::Or),
        "xor" => Ok(LogicalOp::Xor),
        _ => Err(PyValueError::new_err(format!(
            "unknown Boolean operation {operation:?}"
        ))),
    }
}

#[pymethods]
impl TensorFnBuilder {
    #[new]
    fn new(name: String) -> PyResult<Self> {
        if name.is_empty() {
            return Err(PyValueError::new_err("TensorFn names must not be empty"));
        }
        Ok(Self {
            name,
            inputs: Vec::new(),
            finished: false,
        })
    }

    fn input_bit_vector(
        &mut self,
        name: String,
        mask: &Bound<'_, PyAny>,
    ) -> PyResult<TensorExprHandle> {
        if self.finished {
            return Err(PyRuntimeError::new_err(
                "TensorFnBuilder is already finished",
            ));
        }
        if self.inputs.iter().any(|(existing, _)| existing == &name) {
            return Err(PyValueError::new_err(format!(
                "duplicate TensorFn input {name:?}"
            )));
        }
        let tensor_type =
            TensorType::from_bitvec_mask(parts::<u64>(mask)?).map_err(PyValueError::new_err)?;
        self.inputs.push((name.clone(), tensor_type.clone()));
        Ok(TensorExprHandle::new(Input::new(name, tensor_type.clone()), tensor_type))
    }

    fn constant_bit_vector(
        &self,
        value: &Bound<'_, PyAny>,
        mask: &Bound<'_, PyAny>,
    ) -> PyResult<TensorExprHandle> {
        let value = parts::<u64>(value)?;
        let declared =
            TensorType::from_bitvec_mask(parts::<u64>(mask)?).map_err(PyValueError::new_err)?;
        let width = declared.width();
        if value.data().len() != width.div_ceil(64) {
            return Err(PyValueError::new_err(
                "constant word count does not match its TensorType mask",
            ));
        }
        // The support of a constant is exactly its one bits.
        let support = TensorType::flat(width, value.data().to_vec());
        Ok(TensorExprHandle::new(Constant::new(value, declared), support))
    }

    /// Pointwise logic over operands, each optionally reduced over its
    /// jagged dimension. Fusion happens later, in `compile::pass::fusion`.
    fn bv_logical(
        &self,
        operation: &str,
        operands: &Bound<'_, PyAny>,
    ) -> PyResult<TensorExprHandle> {
        let op = logical_op(operation)?;
        let mut reduced = SmallVec::<[JTReduced; 2]>::new();
        let mut types = Vec::new();
        for operand in operands.try_iter()? {
            let operand = operand?;
            let value = handle(&operand.getattr("expression")?)?;
            let reduce_dims = operand.getattr("reduce_dims")?.extract::<Vec<usize>>()?;
            if !reduce_dims.is_empty() {
                types.push(reduce_type(op, &value.tensor_type));
                reduced.push(JTReduced::shared(value.expression, reduce_dims));
                continue;
            }
            // A nested BVLogical stays one shared operand (no flattening).
            types.push(value.tensor_type.clone());
            reduced.push(JTReduced::shared(value.expression, []));
        }
        let tensor_type = logical_type(op, &types).map_err(PyValueError::new_err)?;
        if let [single] = reduced.as_slice() {
            if single.reduce_dims.is_empty() {
                return Ok(TensorExprHandle {
                    expression: Rc::clone(&single.expr),
                    tensor_type,
                });
            }
        }
        Ok(TensorExprHandle::new(BVLogical::new(op, reduced), tensor_type))
    }

    /// `BTExpandReduce`: `dims` holds `(kind, extent)` pairs, with `kind`
    /// one of `"batch"`, `"expand"`, `"reduce"`; `op` folds reduced dimensions.
    fn bt_expand_reduce(
        &self,
        source: &Bound<'_, PyAny>,
        dims: Vec<(String, usize)>,
        op: &str,
    ) -> PyResult<TensorExprHandle> {
        let source = handle(source)?;
        let dims = dims
            .into_iter()
            .map(|(kind, extent)| match kind.as_str() {
                "batch" => Ok(ExpandReduceDimType::Batch(extent)),
                "expand" => Ok(ExpandReduceDimType::Expand(extent)),
                "reduce" => Ok(ExpandReduceDimType::Reduce(extent)),
                other => Err(PyValueError::new_err(format!("unknown dimension kind {other:?}"))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let node = BTExpandReduce {
            expr: source.expression,
            bt_dims: dims.into_iter().collect(),
            op: logical_op(op)?,
        };
        let tensor_type = expand_reduce_type(&node, &source.tensor_type).map_err(PyValueError::new_err)?;
        Ok(TensorExprHandle::new(node, tensor_type))
    }

    /// `BTShift`: move bits toward lower indices by `amount`
    /// (`output[d] = source[d + amount]`); `operation` is `"logical"`
    /// (zero fill), `"arithmetic"` (sign fill), or `"rotate"`.
    fn bt_shift(&self, source: &Bound<'_, PyAny>, operation: &str, amount: usize) -> PyResult<TensorExprHandle> {
        let source = handle(source)?;
        if !source.tensor_type.is_flat() {
            return Err(PyValueError::new_err("shifts require a flat bit-vector"));
        }
        let shift_op = match operation {
            "logical" => ShiftOp::Logical,
            "arithmetic" => ShiftOp::Arithmetic,
            "rotate" => ShiftOp::Rotate,
            other => return Err(PyValueError::new_err(format!("unknown shift {other:?}"))),
        };
        let tensor_type = TensorType::bitvec(source.tensor_type.width()).map_err(PyValueError::new_err)?;
        Ok(TensorExprHandle::new(
            BTShift {
                expr: source.expression,
                shift_op,
                shift_amounts: amount,
            },
            tensor_type,
        ))
    }

    /// `BVJaggedPermute`: row `d` has one lane per source index in `indices[d]`.
    fn bv_jagged_permute(
        &self,
        source: &Bound<'_, PyAny>,
        indices: &Bound<'_, PyAny>,
    ) -> PyResult<TensorExprHandle> {
        let source = handle(source)?;
        let indices = parts::<usize>(indices)?;
        let tensor_type =
            permute_type(&source.tensor_type, &indices).map_err(PyValueError::new_err)?;
        Ok(TensorExprHandle::new(
            BVJaggedPermute {
                expr: source.expression,
                indices,
            },
            tensor_type,
        ))
    }

    fn finish(&mut self, result: &Bound<'_, PyAny>) -> PyResult<TensorFn> {
        if self.finished {
            return Err(PyRuntimeError::new_err(
                "TensorFnBuilder is already finished",
            ));
        }
        self.finished = true;
        let result = handle(result)?;
        if !result.tensor_type.is_flat() {
            return Err(PyValueError::new_err(
                "a traced function must return a flat bit-vector; reduce its jagged dimension first",
            ));
        }
        Ok(TensorFn::new(
            self.name.clone(),
            self.inputs.clone(),
            Rc::unwrap_or_clone(result.expression),
        ))
    }
}
