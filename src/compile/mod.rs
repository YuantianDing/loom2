//! Typed IR produced by Python tracing.
//!
//! The active design represents whole tensor operations. It is intentionally
//! independent of MLIR so tracing and transformation passes can be tested on
//! systems without an LLVM installation.

mod builder;
mod display;
pub mod expr;
pub mod pass;
pub mod ir;
pub mod tensor;
pub mod typing;

pub use builder::{TensorExprHandle, TensorFnBuilder};
pub use expr::{
    BTExpandReduce, BTShift, BVGather, BVGather2, BVJaggedPermute, BVLogical, Constant, ExpandReduceDimType, Input,
    JTReduced, LogicalOp, ShiftOp, TExpr, TensorExpr,
};
pub use ir::{TensorFn, TensorType};
pub use tensor::JaggedTensor;
