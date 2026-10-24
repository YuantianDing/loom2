//! Complete input to syntax-guided synthesis.

use std::collections::HashMap;

use pyo3_stub_gen::derive::gen_stub_pyclass;

use crate::compile::TensorFn;
use crate::grammar::Grammar;

/// A synthesis problem built from traced tensor functions.
///
/// Grammar constructor names resolve through the function map. The
/// constraint is itself a traced function from one start-symbol value to a
/// single bit; a candidate is a solution when that bit is set.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen, unsendable)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SynthFun {
    name: String,
    functions: HashMap<String, TensorFn>,
    grammar: Grammar,
    constraint: TensorFn,
}

impl SynthFun {
    /// Construct a synthesis problem without invoking lowering.
    pub fn new(
        name: impl Into<String>,
        functions: HashMap<String, TensorFn>,
        grammar: Grammar,
        constraint: TensorFn,
    ) -> Self {
        Self {
            name: name.into(),
            functions,
            grammar,
            constraint,
        }
    }

    /// Name of the synthesized function.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Traced functions addressable by grammar constructor name.
    pub fn functions(&self) -> &HashMap<String, TensorFn> {
        &self.functions
    }

    /// Grammar that enumerates candidate programs.
    pub fn grammar(&self) -> &Grammar {
        &self.grammar
    }

    /// One-bit constraint over values of the start nonterminal.
    pub fn constraint(&self) -> &TensorFn {
        &self.constraint
    }
}
