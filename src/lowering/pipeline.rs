//! End-to-end synthesis build that retains every intermediate stage.
//!
//! | File                     | Stage                                          |
//! | ------------------------ | ---------------------------------------------- |
//! | `01-tensor-fn.ir`        | traced `TensorFn` graphs (constructors, constraint) |
//! | `02-fused.ir`            | after permute/reduce fusion (`gather`, `gather2`) |
//! | `03-loops.ir`            | vector loop programs                           |
//! | `04-frontend.mlir`       | Melior-built synthesis module                  |
//! | `05-cpu.mlir`            | inlined, canonicalized, OpenMP CPU MLIR        |
//! | `05-gpu.mlir`            | instead, for CUDA: outlined `gpu.launch_func` kernels |
//! | `06-llvm.mlir`           | LLVM-dialect MLIR                              |
//! | `07-llvm.ll`             | translated LLVM IR                             |
//! | `08-program.o`           | optimized native object                        |
//! | `result-printer.c`       | token-to-constructor result printer            |
//! | `result-schema.tsv`      | numeric constructor-token decoder              |
//! | executable               | object linked with `cpp/cpu` (or `cpp/cuda`)   |

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};


use super::exec::synthesis::{plan_synthesis, result_printer_source};
use super::exec::{lower_synthesis, new_context, Emitter, SynthesisOptions};
use super::llvm::{
    compile_c_object, link_cpu_object, lower_to_llvm_mlir, lower_to_target_mlir, BackendError, BackendOptions,
    Device, LlvmModule,
};
use crate::problem::SynthFun;

fn write(directory: &Path, name: &str, contents: impl AsRef<[u8]>, written: &mut Vec<PathBuf>) -> Result<(), BackendError> {
    let path = directory.join(name);
    fs::write(&path, contents)?;
    written.push(path);
    Ok(())
}

/// Build a native synthesizer and write every stage into `artifacts`.
/// Returns the written paths in pipeline order, executable last.
pub fn build_synthesizer(
    problem: &SynthFun,
    options: SynthesisOptions,
    backend: &BackendOptions,
    artifacts: &Path,
    executable: &Path,
) -> Result<Vec<PathBuf>, BackendError> {
    let compile = |error: super::LoweringError| BackendError::Compile(error.to_string());
    fs::create_dir_all(artifacts)?;
    let mut written = Vec::new();

    let mut traced = String::new();
    let plan = plan_synthesis(problem).map_err(compile)?;
    for (_, name, _) in &plan.tokens {
        writeln!(traced, "{}", problem.functions()[name]).unwrap();
    }
    writeln!(traced, "// constraint over the start nonterminal\n{}", problem.constraint()).unwrap();
    write(artifacts, "01-tensor-fn.ir", traced, &mut written)?;

    let mut fused = String::new();
    for graph in &plan.fused {
        writeln!(fused, "{graph}").unwrap();
    }
    write(artifacts, "02-fused.ir", fused, &mut written)?;

    let mut loops = String::new();
    for program in plan.constructors.iter().chain([&plan.constraint]) {
        writeln!(loops, "{program}").unwrap();
    }
    write(artifacts, "03-loops.ir", loops, &mut written)?;

    let context = new_context();
    let emitter = Emitter::new(&context);
    let mut module = lower_synthesis(&emitter, problem, &plan, options).map_err(compile)?;
    write(artifacts, "04-frontend.mlir", module.as_operation().to_string(), &mut written)?;

    lower_to_target_mlir(&context, &mut module, backend)?;
    let target = if backend.device == Device::Cpu { "05-cpu.mlir" } else { "05-gpu.mlir" };
    write(artifacts, target, module.as_operation().to_string(), &mut written)?;
    lower_to_llvm_mlir(&context, &mut module, backend)?;
    write(artifacts, "06-llvm.mlir", module.as_operation().to_string(), &mut written)?;

    let llvm = LlvmModule::translate(&module)?;
    write(artifacts, "07-llvm.ll", llvm.to_ir(), &mut written)?;
    let object = llvm.emit_object(backend)?;
    write(artifacts, "08-program.o", object, &mut written)?;

    let printer = result_printer_source(problem, &plan, options.max_expression_size);
    write(artifacts, "result-printer.c", printer, &mut written)?;
    let mut schema = String::from("token\tconstructor\tarity\n");
    for (token, name, arity) in &plan.tokens {
        writeln!(schema, "{token}\t{name}\t{arity}").unwrap();
    }
    write(artifacts, "result-schema.tsv", schema, &mut written)?;

    let printer_object = artifacts.join("result-printer.o");
    compile_c_object(&artifacts.join("result-printer.c"), &printer_object, backend)?;
    let linked = link_cpu_object(artifacts.join("08-program.o"), std::slice::from_ref(&printer_object), executable, backend);
    let _ = fs::remove_file(&printer_object);
    linked?;
    written.push(executable.to_path_buf());
    Ok(written)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::process::Command;

    use super::*;
    use crate::compile::{
        BVGather, BVLogical, Constant, Input, JTReduced, JaggedTensor, LogicalOp, TensorExpr,
        TensorFn, TensorType,
    };
    use crate::grammar::{Grammar, Nonterminal, Production};

    fn constant(name: &str, bits: u64) -> TensorFn {
        let value = Constant::new(JaggedTensor::from_flat(vec![bits]), TensorType::bitvec(3).unwrap());
        TensorFn::new(name, Vec::new(), value)
    }

    fn three_bit_problem() -> SynthFun {
        let bits = TensorType::bitvec(3).unwrap();
        let union = BVLogical::new(
            LogicalOp::Or,
            [
                JTReduced::new(Input::new("a", bits.clone()), []),
                JTReduced::new(Input::new("b", bits.clone()), []),
            ],
        );
        let functions = HashMap::from([
            ("c1".to_string(), constant("c1", 0b001)),
            ("c2".to_string(), constant("c2", 0b010)),
            ("c4".to_string(), constant("c4", 0b100)),
            (
                "or".to_string(),
                TensorFn::new("or", vec![("a".into(), bits.clone()), ("b".into(), bits.clone())], union),
            ),
        ]);
        // Accept exactly 0b011: bits 0 and 1 set, bit 2 clear.
        let candidate = TensorExpr::from(Input::new("x", bits.clone()));
        let row = |indices: Vec<usize>| {
            let length = indices.len();
            JaggedTensor::try_new(indices, vec![1], vec![vec![0, length]]).unwrap()
        };
        let positive = BVGather::from_rows(candidate.clone(), &row(vec![0, 1]), LogicalOp::And);
        let negative = BVGather::from_rows(candidate, &row(vec![2]), LogicalOp::Or);
        let one = Constant::new(JaggedTensor::from_flat(vec![1]), TensorType::bitvec(1).unwrap());
        let constraint = BVLogical::new(
            LogicalOp::And,
            [
                JTReduced::new(positive, []),
                JTReduced::new(
                    BVLogical::new(LogicalOp::Xor, [JTReduced::new(negative, []), JTReduced::new(one, [])]),
                    [],
                ),
            ],
        );
        let grammar = Grammar::new(
            "S",
            vec![Nonterminal::new(
                "S",
                3,
                vec![
                    Production::new("c1", Vec::<String>::new()),
                    Production::new("c2", Vec::<String>::new()),
                    Production::new("c4", Vec::<String>::new()),
                    Production::new("or", ["S", "S"]),
                ],
            )],
        )
        .unwrap();
        SynthFun::new(
            "r",
            functions,
            grammar,
            TensorFn::new("constraint", vec![("x".into(), bits)], constraint),
        )
    }

    fn synthesize(max_pool_bytes: usize) -> String {
        synthesize_on(max_pool_bytes, BackendOptions::default())
    }

    fn synthesize_on(max_pool_bytes: usize, backend: BackendOptions) -> String {
        let problem = three_bit_problem();
        let device = if backend.device == Device::Cpu { "cpu" } else { "cuda" };
        let directory = std::env::temp_dir().join(format!(
            "loom2-pipeline-{}-{device}-{max_pool_bytes}",
            std::process::id()
        ));
        let executable = directory.join("synth");
        let options = SynthesisOptions {
            max_expression_size: 4,
            max_pool_bytes,
            initial_hash_capacity: 64,
        };
        let stages = build_synthesizer(&problem, options, &backend, &directory, &executable)
            .unwrap_or_else(|error| panic!("{error}"));
        assert_eq!(stages.last(), Some(&executable));
        let output = Command::new(&executable).output().unwrap();
        assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
        let text = String::from_utf8(output.stdout).unwrap();
        let _ = fs::remove_dir_all(&directory);
        text
    }

    #[test]
    fn builds_and_runs_a_three_bit_synthesizer() {
        let text = synthesize(1 << 20);
        assert!(["r = or(c1, c2)\n", "r = or(c2, c1)\n"].contains(&text.as_str()), "{text}");
    }

    #[test]
    fn a_full_pool_still_checks_candidates_and_records_the_solution() {
        // 544 bytes = one 64-slot table (512 bytes) + a 4-word arena: two
        // size-1 rows (c1, c2) and no size-3 row, so or(c1, c2) is found
        // without being stored.
        let text = synthesize(544);
        assert!(["r = or(c1, c2)\n", "r = or(c2, c1)\n"].contains(&text.as_str()), "{text}");
    }

    #[cfg(feature = "cuda")]
    fn cuda() -> BackendOptions {
        BackendOptions {
            device: Device::Cuda(crate::lowering::llvm::CudaOptions::default()),
            ..BackendOptions::default()
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn builds_and_runs_a_three_bit_synthesizer_on_cuda() {
        let text = synthesize_on(1 << 20, cuda());
        assert!(["r = or(c1, c2)\n", "r = or(c2, c1)\n"].contains(&text.as_str()), "{text}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn a_full_pool_still_checks_candidates_on_cuda() {
        let text = synthesize_on(544, cuda());
        assert!(["r = or(c1, c2)\n", "r = or(c2, c1)\n"].contains(&text.as_str()), "{text}");
    }
}
