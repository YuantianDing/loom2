//! Emit the executable synthesis module with high-mlir.
//!
//! Only upstream MLIR dialects are used (`func`, `arith`, `scf`, `memref`,
//! `vector`). The concurrent hash set is reached through the plain C ABI in
//! `cpp/cpu/i64hashset.hpp`, and the module defines its own `main`.
//!
//! ```text
//! module attributes {loom.semantic_encoding = "packed-i64", ...} {
//!   <functions::lower_tensor_fn for each constructor>      @define_fun_*
//!   <functions::lower_constraint for the start symbol>      @sygus_constraints
//!   <hash_filter::lower_compute_hash per word count>        @compute_hash_w*
//!   <hash_filter::lower_hash_filter per production>         @hash_filter_*
//!   <synthesis::lower_synthesis entry point>                @synthesize, @main
//! }
//! ```
//!
//! Packed values are passed as `vector<Nxi64>` rather than memrefs, read from
//! and written to pools with `vector.transfer_read`/`transfer_write`, so the
//! inliner turns every constructor into vector code inside its parallel
//! filter. There is no `tensor<...xi1>` value at any stage.
//!
//! The module is built as a high-mlir expression and converted to a Melior
//! module ([`to_melior`]) for the MLIR pass pipelines.

pub mod functions;
pub mod hash_filter;
pub mod sliced;
pub mod synthesis;

pub use functions::{lower_constraint, lower_tensor_fn};
pub use hash_filter::{lower_compute_hash, lower_hash_filter, HashFilterConfig};
pub use synthesis::{lower_synthesis, SynthesisOptions};

use std::ffi::c_void;
use std::path::Path;

use high_mlir::{MemRefType, Root, Ty, Val, I64, INDEX};
use melior::ir::operation::OperationLike;
use melior::ir::Module;
use melior::{Context, StringRef};
use mlir_sys::{
    mlirIdentifierStr, mlirLocationFileLineColGet, mlirLocationFileLineColRangeGetFilename,
    mlirLocationFileLineColRangeGetStartColumn, mlirLocationFileLineColRangeGetStartLine,
    mlirLocationIsAFileLineColRange, mlirOperationGetContext, mlirOperationGetLocation, mlirOperationSetLocation,
    mlirOperationWalk, MlirOperation, MlirWalkOrder_MlirWalkPreOrder, MlirWalkResult,
    MlirWalkResult_MlirWalkResultAdvance,
};

use super::{LoweringError, LoweringResult};

/// Memref type of one value/expression pool,
/// `memref<?x?xi64, strided<[?, 1], offset: ?>>`.
pub(crate) fn pool_type() -> Ty {
    Ty::new(MemRefType::new([None, None], I64).with_strides([None, Some(1)], None)).erase()
}

/// `memref<?xi64>`, the arena and the solution.
pub(crate) fn words_type() -> Ty {
    Ty::memref([None], I64)
}

/// View `rows x row_words` words of `arena` starting at word `offset`.
pub(crate) fn pool_view(arena: &Val, offset: &Val, rows: &Val, row_words: &Val) -> Val {
    high_mlir::memref::reinterpret_cast(arena, (offset,), (rows, row_words), (row_words, 1)).into()
}

/// `arith.constant <value> : index`.
pub(crate) fn index(value: usize) -> Val {
    high_mlir::arith::Constant::index(value as i64).into()
}

/// `arith.index_cast` to `index`.
pub(crate) fn to_index(value: &Val) -> Val {
    high_mlir::arith::index_cast(value, INDEX).into()
}

/// Make a constructor name safe for an MLIR symbol that is also a valid PTX
/// identifier (kernels are named after filters): `re.++` -> `re_2e__2b__2b_`.
pub fn symbol_name(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || character == '_' {
                character.to_string()
            } else {
                format!("_{:x}_", character as u32)
            }
        })
        .collect()
}

pub(crate) fn invalid(stage: &'static str, message: impl Into<String>) -> LoweringError {
    LoweringError::InvalidInput {
        stage,
        message: message.into(),
    }
}

/// Load every upstream dialect and translation into a fresh context.
pub fn new_context() -> Context {
    let context = high_mlir::context();
    melior::utility::register_all_llvm_translations(&context);
    context
}

/// Convert a high-mlir module to a verified Melior module.
///
/// Every operation is located at the Rust code that built it (high-mlir's
/// `ADD_RUST_LOC`), so the executable's debug info points into this crate.
/// Paths of this crate's sources are relative to its root; they are made
/// absolute so debuggers find them from any directory.
pub fn to_melior<'c>(context: &'c Context, module: &Root, stage: &'static str) -> Result<Module<'c>> {
    let mut built = module.to_melior(context).map_err(|error| invalid(stage, error.to_string()))?;
    if !built.as_operation().verify() {
        return Err(invalid(stage, "the generated module failed MLIR verification"));
    }
    resolve_source_paths(&mut built, Path::new(env!("CARGO_MANIFEST_DIR")));
    Ok(built)
}

/// Prefix every relative `file:line:col` location in `module` with `root`.
fn resolve_source_paths(module: &mut Module, root: &Path) {
    unsafe extern "C" fn resolve(operation: MlirOperation, root: *mut c_void) -> MlirWalkResult {
        // SAFETY: `root` is the `&Path` passed to the walk below, and
        // `operation` is alive while it is visited.
        unsafe {
            let root = &*(root as *const &Path);
            let location = mlirOperationGetLocation(operation);
            if !mlirLocationIsAFileLineColRange(location) {
                return MlirWalkResult_MlirWalkResultAdvance;
            }
            let file = StringRef::from_raw(mlirIdentifierStr(mlirLocationFileLineColRangeGetFilename(location)));
            let Ok(file) = file.as_str() else { return MlirWalkResult_MlirWalkResultAdvance };
            if Path::new(file).is_absolute() || file.is_empty() {
                return MlirWalkResult_MlirWalkResultAdvance;
            }
            let path = root.join(file).to_string_lossy().into_owned();
            let line = mlirLocationFileLineColRangeGetStartLine(location) as u32;
            let column = mlirLocationFileLineColRangeGetStartColumn(location) as u32;
            let context = mlirOperationGetContext(operation);
            let resolved = mlirLocationFileLineColGet(context, StringRef::new(&path).to_raw(), line, column);
            mlirOperationSetLocation(operation, resolved);
            MlirWalkResult_MlirWalkResultAdvance
        }
    }
    let root: *const &Path = &root;
    // SAFETY: the callback only reads `root` and rewrites locations.
    unsafe {
        mlirOperationWalk(
            module.as_operation_mut().to_raw(),
            Some(resolve),
            root as *mut c_void,
            MlirWalkOrder_MlirWalkPreOrder,
        )
    };
}

pub(crate) type Result<T> = LoweringResult<T>;
