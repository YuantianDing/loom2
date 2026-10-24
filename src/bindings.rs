//! Registration for the native Python module.

use pyo3::prelude::*;

use crate::compile::{Constant, TensorExprHandle, TensorFn, TensorFnBuilder, TensorType};
use crate::grammar::{Grammar, Nonterminal, Production};
use crate::problem::SynthFun;

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<TensorType>()?;
    module.add_class::<TensorFn>()?;
    module.add_class::<Constant>()?;
    module.add_class::<TensorExprHandle>()?;
    module.add_class::<TensorFnBuilder>()?;
    module.add_class::<Grammar>()?;
    module.add_class::<Nonterminal>()?;
    module.add_class::<Production>()?;
    module.add_class::<SynthFun>()?;
    #[cfg(feature = "llvm")]
    module.add_function(wrap_pyfunction!(native::build_synthesizer, module)?)?;
    #[cfg(feature = "llvm")]
    module.add_class::<native::JitFunction>()?;
    Ok(())
}

#[cfg(feature = "llvm")]
mod native {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;

    use crate::compile::TensorFn;
    use crate::grammar::{Grammar, Nonterminal, Production};
    use crate::lowering::exec::SynthesisOptions;
    use crate::lowering::jit;
    use crate::lowering::llvm::BackendOptions;
    use crate::lowering::pipeline;
    use crate::problem::SynthFun;

    /// One `(symbol, width, [(constructor, [child symbols])])` grammar entry.
    type NonterminalSpec = (String, usize, Vec<(String, Vec<String>)>);

    /// Lower traced functions and a grammar to a native synthesizer.
    ///
    /// The first nonterminal is the start symbol. Every stage is written to
    /// `artifact_dir`; the returned paths are in pipeline order.
    #[pyfunction]
    #[pyo3(signature = (
        name, functions, grammar, constraint, artifact_dir, executable, *,
        max_size, max_pool_bytes, initial_hash_capacity, optimization_level, openmp
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn build_synthesizer(
        name: String,
        functions: Vec<PyRef<'_, TensorFn>>,
        grammar: Vec<NonterminalSpec>,
        constraint: PyRef<'_, TensorFn>,
        artifact_dir: PathBuf,
        executable: PathBuf,
        max_size: usize,
        max_pool_bytes: usize,
        initial_hash_capacity: usize,
        optimization_level: u8,
        openmp: bool,
    ) -> PyResult<Vec<PathBuf>> {
        let start = grammar
            .first()
            .map(|(symbol, _, _)| symbol.clone())
            .ok_or_else(|| PyValueError::new_err("the synthesis grammar must not be empty"))?;
        let nonterminals = grammar
            .into_iter()
            .map(|(symbol, width, productions)| {
                let productions = productions
                    .into_iter()
                    .map(|(constructor, children)| Production::new(constructor, children))
                    .collect();
                Nonterminal::new(symbol, width, productions)
            })
            .collect();
        let grammar = Grammar::new(start, nonterminals).map_err(|error| PyValueError::new_err(error.to_string()))?;
        let functions: HashMap<String, TensorFn> = functions
            .iter()
            .map(|function| (function.name.clone(), (**function).clone()))
            .collect();
        let problem = SynthFun::new(name, functions, grammar, (*constraint).clone());
        let options = SynthesisOptions {
            max_expression_size: max_size,
            max_pool_bytes,
            initial_hash_capacity,
        };
        let backend = BackendOptions {
            optimization_level,
            openmp,
            ..BackendOptions::default()
        };
        pipeline::build_synthesizer(&problem, options, &backend, &artifact_dir, &executable)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// A traced `TensorFn` compiled in-process with the LLVM JIT.
    ///
    /// Arguments and the result are packed `u64` words, least significant
    /// bit first; all arguments are passed concatenated.
    #[pyclass(module = "loom2._loom2", unsendable)]
    pub struct JitFunction {
        inner: jit::JitFunction,
    }

    #[pymethods]
    impl JitFunction {
        #[new]
        #[pyo3(signature = (function, *, optimization_level = 2))]
        fn new(function: PyRef<'_, TensorFn>, optimization_level: u8) -> PyResult<Self> {
            if optimization_level > 3 {
                return Err(PyValueError::new_err("optimization_level must be 0 through 3"));
            }
            jit::JitFunction::compile(&function, optimization_level)
                .map(|inner| Self { inner })
                .map_err(|error| PyRuntimeError::new_err(error.to_string()))
        }

        /// Run on concatenated packed argument words; returns the result words.
        fn __call__(&self, words: Vec<u64>) -> PyResult<Vec<u64>> {
            self.inner.call(&words).map_err(|error| PyValueError::new_err(error.to_string()))
        }

        #[getter]
        fn input_bits(&self) -> Vec<usize> {
            self.inner.input_bits.clone()
        }

        #[getter]
        fn result_bits(&self) -> usize {
            self.inner.result_bits
        }

        /// The LLVM-dialect MLIR handed to the execution engine.
        #[getter]
        fn llvm_mlir(&self) -> String {
            self.inner.llvm_mlir.clone()
        }
    }
}
