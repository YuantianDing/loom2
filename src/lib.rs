//! Native IR and lowering support for Loom 2.
//!
//! The crate is split into four layers:
//!
//! - compile defines the tensor-expression graph produced by Python tracing.
//! - grammar and problem describe syntax-guided synthesis inputs.
//! - lowering::circuit implements packed-i64 permutation circuits.
//! - lowering::words lowers each TensorFn to packed i64 word operations.
//! - lowering::exec emits the synthesis module with Melior (upstream dialects).
//! - lowering::llvm runs MLIR/LLVM lowering and links the CPU runtime.
//!
//! MLIR support is feature-gated so the IR and circuit tests do not require a
//! matching LLVM installation. Enable `mlir` for the Melior emitters and `llvm`
//! for native code generation and the `cpp/cpu` runtime.

mod bindings;
pub mod compile;
pub mod grammar;
pub mod lowering;
pub mod problem;

use pyo3::prelude::*;
use pyo3_stub_gen::derive::gen_stub_pyfunction;

/// Return the native extension version.
#[gen_stub_pyfunction]
#[pyfunction]
pub fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Native support module for the Python package.
#[pymodule]
fn _loom2(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(native_version, module)?)?;
    bindings::register(module)
}

pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
