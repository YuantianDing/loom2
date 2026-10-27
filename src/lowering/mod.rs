//! Lower traced tensor functions to a native synthesizer.
//!
//! ```text
//! TensorFn --pass::fuse--> fused TensorFn --exec (high-mlir)--> synthesis module
//!          --llvm--> CPU MLIR -> LLVM MLIR -> LLVM IR -> object --link--> executable
//! ```
//!
//! The module uses upstream dialects only. Names in angle brackets are
//! compile-time substitutions; semantic values are packed `i64` words.
//!
//! ```text
//! module attributes {loom.semantic_encoding = "packed-i64", ...} {
//!   // cpp/cpu runtime: concurrent hash set and result printer
//!   func.func private @mlir_loom_hashset_i64_create_handle(i64) -> i64
//!   func.func private @mlir_loom_hashset_i64_insert_one(i64, i64) -> i1
//!   func.func private @mlir_loom_print_result(memref<?xi64>, index)
//!
//!   // Read-only step tables of the constructors.
//!   memref.global "private" constant @table_<constructor>_t<k> : memref<...>
//!
//!   // One pure function for every constructor; vectors in, vector out.
//!   func.func private @define_fun_<constructor>(vector<Nxi64>, ...) -> vector<Nxi64>
//!   func.func private @sygus_constraints(vector<Nxi64>) -> i1
//!   func.func private @compute_hash_w<words>(vector<Nxi64>) -> i64
//!
//!   // One parallel evaluator for every grammar production.
//!   func.func private @hash_filter_<nonterminal>_<constructor>_rule<rule>(
//!       %arena: memref<?xi64>, %output_offset: index,
//!       %output_capacity: index, %output_row_words: index, %start: index,
//!       %cursor: memref<1xi64>,
//!       %child0: memref<?x?xi64, strided<[?, 1], offset: ?>>, ...,
//!       %child0_expression_bits: index, ...,
//!       %seen: i64, %solution: memref<?xi64>, %found: memref<1xi8>
//!   ) -> memref<?x?xi64, strided<[?, 1], offset: ?>>
//!
//!   // Allocate shared state and call the production filters by exact size.
//!   func.func @synthesize() -> memref<?xi64>
//!   func.func @main() -> i32
//! }
//! ```
//!
//! [`packed`] (the reference evaluator) needs no LLVM. [`circuit`] (verified
//! permutation circuits) is currently unused.
//! The `mlir` feature enables [`exec`]; `llvm` adds [`llvm`], [`pipeline`],
//! and [`jit`] (in-process LLVM JIT of one constructor).

use std::error::Error;
use std::fmt;

pub mod circuit;
#[cfg(feature = "mlir")]
pub mod exec;
#[cfg(feature = "llvm")]
pub mod jit;
#[cfg(feature = "llvm")]
pub mod llvm;
#[cfg(feature = "llvm")]
pub mod pipeline;
pub mod packed;

/// `i64` words needed for `bits` bits.
pub fn word_count(bits: usize) -> usize {
    bits.div_ceil(64)
}

/// Result type shared by lowering stages.
pub type LoweringResult<T> = Result<T, LoweringError>;

/// Failure reported by a lowering stage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LoweringError {
    /// The API boundary exists, but its code generator is not implemented.
    Pending { stage: &'static str },
    /// The input IR violates a requirement of the selected lowering.
    InvalidInput {
        stage: &'static str,
        message: String,
    },
}

impl LoweringError {
    #[allow(dead_code)]
    pub(crate) fn pending(stage: &'static str) -> Self {
        Self::Pending { stage }
    }
}

impl fmt::Display for LoweringError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending { stage } => write!(out, "{stage} is not implemented"),
            Self::InvalidInput { stage, message } => write!(out, "{stage}: {message}"),
        }
    }
}

impl Error for LoweringError {}
