//! End-to-end synthesis build that retains every intermediate stage.
//!
//! | File                     | Stage                                          |
//! | ------------------------ | ---------------------------------------------- |
//! | `01-tensor-fn.ir`        | traced `TensorFn` graphs (constructors, constraint) |
//! | `02-fused.ir`            | after permute/reduce fusion (`gather`, `gather2`) |
//! | `03-frontend.mlir`       | high-mlir-built synthesis module               |
//! | `04-cpu.mlir`            | inlined, canonicalized, OpenMP CPU MLIR        |
//! | `04-gpu.mlir`            | instead, for CUDA: outlined `gpu.launch_func` kernels |
//! | `05-llvm.mlir`           | LLVM-dialect MLIR                              |
//! | `06-llvm.ll`             | translated LLVM IR                             |
//! | `07-program.o`           | optimized native object                        |
//! | `result-printer.c`       | token-to-constructor result printer            |
//! | `result-schema.tsv`      | numeric constructor-token decoder              |
//! | executable               | object linked with `cpp/cpu` (or `cpp/cuda`)   |

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};


use super::exec::synthesis::{plan_synthesis, result_printer_source};
use super::exec::{lower_synthesis, new_context, to_melior, SynthesisOptions};
use super::llvm::{
    attach_debug_info, compile_c_object, link_cpu_object, lower_to_llvm_mlir, lower_to_target_mlir, BackendError, BackendOptions,
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
    let artifacts = &fs::canonicalize(artifacts)?;
    let mut written = Vec::new();

    let mut traced = String::new();
    let plan = plan_synthesis(problem, backend.device == Device::Cpu).map_err(compile)?;
    for (_, name, _) in &plan.tokens {
        writeln!(traced, "{}", problem.functions()[name]).unwrap();
    }
    writeln!(traced, "// constraint over the start nonterminal\n{}", problem.constraint()).unwrap();
    write(artifacts, "01-tensor-fn.ir", traced, &mut written)?;

    let mut fused = String::new();
    for graph in plan.constructors.iter().chain([&plan.constraint]) {
        writeln!(fused, "{graph}").unwrap();
    }
    write(artifacts, "02-fused.ir", fused, &mut written)?;

    let context = new_context();
    let frontend = lower_synthesis(problem, &plan, options).map_err(compile)?;
    let mut module = to_melior(&context, &frontend, "synthesis lowering").map_err(compile)?;
    drop(frontend);
    write(artifacts, "03-frontend.mlir", module.as_operation().to_string(), &mut written)?;

    lower_to_target_mlir(&context, &mut module, backend)?;
    let target = if backend.device == Device::Cpu { "04-cpu.mlir" } else { "04-gpu.mlir" };
    write(artifacts, target, module.as_operation().to_string(), &mut written)?;
    lower_to_llvm_mlir(&context, &mut module, backend)?;
    write(artifacts, "05-llvm.mlir", module.as_operation().to_string(), &mut written)?;
    if backend.debug_info {
        // Line tables point at the Rust code that built each operation.
        attach_debug_info(&context, &mut module)?;
    }

    let llvm = LlvmModule::translate(&module)?;
    write(artifacts, "06-llvm.ll", llvm.to_ir(), &mut written)?;
    let object = llvm.emit_object(backend)?;
    write(artifacts, "07-program.o", object, &mut written)?;

    let printer = result_printer_source(problem, &plan, options.max_expression_size);
    write(artifacts, "result-printer.c", printer, &mut written)?;
    let mut schema = String::from("token\tconstructor\tarity\n");
    for (token, name, arity) in &plan.tokens {
        writeln!(schema, "{token}\t{name}\t{arity}").unwrap();
    }
    write(artifacts, "result-schema.tsv", schema, &mut written)?;

    let printer_object = artifacts.join("result-printer.o");
    compile_c_object(&artifacts.join("result-printer.c"), &printer_object, backend)?;
    let linked = link_cpu_object(artifacts.join("07-program.o"), std::slice::from_ref(&printer_object), executable, backend);
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
    use std::rc::Rc;

    use crate::compile::{
        BVGather, BVGather2, BVLogical, Constant, Input, JTReduced, JaggedTensor, LogicalOp, TensorExpr,
        TensorFn, TensorType,
    };
    use crate::lowering::exec::synthesis::plan_synthesis;
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

    /// Sets of exponents below 12 with `or` (union) and `mul` (all sums,
    /// a commutative `gather2`, so it is built bit-sliced). The target
    /// `{1, 2, 4}` needs 7 constructors, e.g. `or(c1, mul(c1, mul(c1, c02)))`.
    fn exponent_problem(commutative: bool) -> SynthFun {
        const WIDTH: usize = 12;
        let bits = TensorType::bitvec(WIDTH).unwrap();
        let input = |name: &str| TensorExpr::from(Input::new(name, bits.clone()));
        let pairs: Vec<(usize, usize)> = (0..WIDTH).flat_map(|i| (0..WIDTH - i).map(move |j| (i, j))).collect();
        let mul = BVGather2 {
            a: Rc::new(input("a")),
            b: Rc::new(input("b")),
            indices_out: JaggedTensor::from_flat(pairs.iter().map(|(i, j)| i + j).collect()),
            indices_1: JaggedTensor::from_flat(pairs.iter().map(|(i, _)| *i).collect()),
            indices_2: JaggedTensor::from_flat(pairs.iter().map(|(_, j)| *j).collect()),
            width: WIDTH,
            inner: LogicalOp::And,
            op: LogicalOp::Or,
        };
        let union = BVLogical::new(LogicalOp::Or, [JTReduced::new(input("a"), []), JTReduced::new(input("b"), [])]);
        let leaf = |name: &str, set: u64| {
            TensorFn::new(name, Vec::new(), Constant::new(JaggedTensor::from_flat(vec![set]), bits.clone()))
        };
        let arguments = vec![("a".to_string(), bits.clone()), ("b".to_string(), bits.clone())];
        let functions = HashMap::from([
            ("c1".to_string(), leaf("c1", 0b10)),
            ("c02".to_string(), leaf("c02", 0b101)),
            ("c3".to_string(), leaf("c3", 0b1000)),
            ("mul".to_string(), TensorFn::new("mul", arguments.clone(), mul)),
            ("or".to_string(), TensorFn::new("or", arguments, union)),
        ]);
        let target = 0b10110u64;
        let row = |indices: Vec<usize>| {
            let length = indices.len();
            JaggedTensor::try_new(indices, vec![1], vec![vec![0, length]]).unwrap()
        };
        let (positives, negatives): (Vec<usize>, Vec<usize>) = (0..WIDTH).partition(|bit| target >> bit & 1 == 1);
        let candidate = TensorExpr::from(Input::new("x", bits.clone()));
        let positive = BVGather::from_rows(candidate.clone(), &row(positives), LogicalOp::And);
        let negative = BVGather::from_rows(candidate, &row(negatives), LogicalOp::Or);
        let one = Constant::new(JaggedTensor::from_flat(vec![1]), TensorType::bitvec(1).unwrap());
        let constraint = BVLogical::new(
            LogicalOp::And,
            [
                JTReduced::new(positive, []),
                JTReduced::new(BVLogical::new(LogicalOp::Xor, [JTReduced::new(negative, []), JTReduced::new(one, [])]), []),
            ],
        );
        let binary = |name: &str| Production::new(name, ["S", "S"]).with_commutative(commutative);
        let grammar = Grammar::new(
            "S",
            vec![Nonterminal::new(
                "S",
                WIDTH,
                vec![
                    Production::new("c1", Vec::<String>::new()),
                    Production::new("c02", Vec::<String>::new()),
                    Production::new("c3", Vec::<String>::new()),
                    binary("mul"),
                    binary("or"),
                ],
            )],
        )
        .unwrap();
        SynthFun::new("r", functions, grammar, TensorFn::new("constraint", vec![("x".into(), bits)], constraint))
    }

    /// The exponent set of a printed `exponent_problem` expression, and its size.
    fn exponents(text: &str) -> (u64, usize) {
        fn parse(tokens: &mut std::iter::Peekable<std::vec::IntoIter<String>>) -> (u64, usize) {
            let name = tokens.next().unwrap();
            let leaf = |set| (set, 1);
            match name.as_str() {
                "c1" => leaf(0b10),
                "c02" => leaf(0b101),
                "c3" => leaf(0b1000),
                _ => {
                    assert_eq!(tokens.next().as_deref(), Some("("));
                    let (a, sa) = parse(tokens);
                    assert_eq!(tokens.next().as_deref(), Some(","));
                    let (b, sb) = parse(tokens);
                    assert_eq!(tokens.next().as_deref(), Some(")"));
                    let set = if name == "or" {
                        a | b
                    } else {
                        (0..12).filter(|i| a >> i & 1 == 1).fold(0, |set, i| set | (b << i)) & 0xfff
                    };
                    (set, 1 + sa + sb)
                }
            }
        }
        let body = text.trim().strip_prefix("r = ").unwrap_or_else(|| panic!("{text}"));
        let tokens: Vec<String> = body
            .replace('(', " ( ")
            .replace(')', " ) ")
            .replace(',', " , ")
            .split_whitespace()
            .map(str::to_string)
            .collect();
        parse(&mut tokens.into_iter().peekable())
    }

    #[test]
    fn commutative_bit_sliced_constructors_keep_the_smallest_solution() {
        for commutative in [false, true] {
            let problem = exponent_problem(commutative);
            let plan = plan_synthesis(&problem, true).unwrap();
            assert!(plan.sliced.iter().any(|&sliced| sliced), "mul is bit-sliced");
            assert_eq!(plan.filters.iter().filter(|filter| filter.commutative).count(), if commutative { 2 } else { 0 });
            let directory = std::env::temp_dir().join(format!("loom2-commutative-{}-{commutative}", std::process::id()));
            let executable = directory.join("synth");
            let options = SynthesisOptions { max_expression_size: 7, max_pool_bytes: 1 << 22, initial_hash_capacity: 64 };
            build_synthesizer(&problem, options, &BackendOptions::default(), &directory, &executable)
                .unwrap_or_else(|error| panic!("{error}"));
            let output = Command::new(&executable).output().unwrap();
            let text = String::from_utf8(output.stdout).unwrap();
            let _ = fs::remove_dir_all(&directory);
            assert_eq!(exponents(&text), (0b10110, 7), "commutative={commutative}: {text}");
        }
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

