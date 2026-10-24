//! In-process LLVM JIT for one traced `TensorFn`.
//!
//! ```text
//! TensorFn --fuse--> --loops--> define_fun_<name>(vector<Nxi64>...) -> vector<Mxi64>
//!   + func.func @loom_jit_entry(%out: !llvm.ptr, %in: !llvm.ptr)
//!       load every input vector from the packed word buffer `%in`,
//!       call define_fun_<name>, store the result to `%out`
//!   --(CPU MLIR, LLVM dialect)--> MLIR ExecutionEngine (LLVM ORC)
//! ```
//!
//! Inputs are concatenated `u64` words (bit `d` of an argument is word
//! `d / 64`, bit `d % 64`); the result uses the same packing. The constructor
//! is the same code the synthesizer runs, so the JIT matches it bit for bit.

use std::ffi::c_void;

use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
use melior::ir::{Module, Value};
use mlir_sys::{mlirExecutionEngineCreate, mlirExecutionEngineDestroy, mlirExecutionEngineLookup, MlirExecutionEngine};

use super::exec::functions::{define_fun_symbol, lower_tensor_fn, packed_type};
use super::exec::{arguments, new_context, Emitter};
use super::llvm::{lower_to_cpu_mlir, lower_to_llvm_mlir, BackendError, BackendOptions};
use super::loops::{lower_loop_program, word_count};
use crate::compile::pass::fuse;
use crate::compile::TensorFn;

/// Public symbol of the generated entry point.
pub const ENTRY: &str = "loom_jit_entry";

type Entry = unsafe extern "C" fn(*mut u64, *const u64);

/// A JIT-compiled constructor.
pub struct JitFunction {
    engine: MlirExecutionEngine,
    entry: Entry,
    /// Logical bit width of every input, in order.
    pub input_bits: Vec<usize>,
    /// Logical bit width of the result.
    pub result_bits: usize,
    /// The LLVM-dialect module handed to the engine.
    pub llvm_mlir: String,
}

impl JitFunction {
    /// Lower, JIT-compile, and look up the entry point of `function`.
    pub fn compile(function: &TensorFn, optimization_level: u8) -> Result<Self, BackendError> {
        let lowering = |error: super::LoweringError| BackendError::Compile(error.to_string());
        let program = lower_loop_program(&fuse(function)).map_err(lowering)?;
        let backend = BackendOptions {
            optimization_level,
            openmp: false,
            ..BackendOptions::default()
        };

        let context = new_context();
        let emitter = Emitter::new(&context);
        let mut module = Module::new(emitter.location);
        let body = module.body();
        let (constructor, globals) = lower_tensor_fn(&emitter, &program);
        for global in globals {
            body.append_operation(global);
        }
        body.append_operation(constructor);

        let pointer = emitter.ty("!llvm.ptr");
        let argument_types: Vec<_> = program.input_bits.iter().map(|&bits| packed_type(&emitter, bits)).collect();
        let result_type = packed_type(&emitter, program.result_bits);
        let alignment = emitter.attr("8 : i64");
        let element = emitter.attr("i64");
        let entry_body = |block: &melior::ir::Block<'_>| {
            let parameters = arguments(block);
            let (output, input) = (parameters[0], parameters[1]);
            let mut offset = 0usize;
            let mut values: Vec<Value> = Vec::new();
            for (&bits, &ty) in program.input_bits.iter().zip(&argument_types) {
                let indices = emitter.attr(&format!("array<i32: {offset}>"));
                let address = emitter.value(
                    block,
                    "llvm.getelementptr",
                    &[input],
                    pointer,
                    &[("rawConstantIndices", indices), ("elem_type", element)],
                );
                values.push(emitter.value(block, "llvm.load", &[address], ty, &[("alignment", alignment)]));
                offset += word_count(bits);
            }
            let result = emitter.call(block, &define_fun_symbol(&program.name), &values, &[result_type]);
            emitter.op(block, "llvm.store", &[result[0], output], &[], &[("alignment", alignment)], Vec::new());
            emitter.op(block, "func.return", &[], &[], &[], Vec::new());
        };
        body.append_operation(emitter.function(ENTRY, &[pointer, pointer], &[], false, &[], Some(&entry_body)));
        if !module.as_operation().verify() {
            return Err(BackendError::Compile(format!(
                "invalid JIT module for {}:\n{}",
                function.name,
                module.as_operation()
            )));
        }

        lower_to_cpu_mlir(&context, &mut module, &backend)?;
        lower_to_llvm_mlir(&context, &mut module, &backend)?;
        let llvm_mlir = module.as_operation().to_string();

        // SAFETY: the module is a valid LLVM-dialect module; the engine owns
        // its own LLVM context, so the MLIR context may be dropped afterwards.
        let engine = unsafe { mlirExecutionEngineCreate(module.to_raw(), i32::from(optimization_level), 0, std::ptr::null(), false, false) };
        if engine.ptr.is_null() {
            return Err(BackendError::Compile(format!("LLVM JIT rejected {}", function.name)));
        }
        let symbol = melior::StringRef::new(ENTRY);
        let address = unsafe { mlirExecutionEngineLookup(engine, symbol.to_raw()) };
        if address.is_null() {
            unsafe { mlirExecutionEngineDestroy(engine) };
            return Err(BackendError::Compile(format!("JIT entry {ENTRY} not found")));
        }
        // SAFETY: `loom_jit_entry` was emitted with exactly this signature.
        let entry = unsafe { std::mem::transmute::<*mut c_void, Entry>(address) };
        Ok(Self {
            engine,
            entry,
            input_bits: program.input_bits.clone(),
            result_bits: program.result_bits,
            llvm_mlir,
        })
    }

    /// `u64` words of every input, concatenated.
    pub fn input_words(&self) -> usize {
        self.input_bits.iter().map(|&bits| word_count(bits)).sum()
    }

    /// `u64` words of the result.
    pub fn result_words(&self) -> usize {
        word_count(self.result_bits)
    }

    /// Run on packed inputs; bits beyond each input's width must be zero.
    pub fn call(&self, inputs: &[u64]) -> Result<Vec<u64>, BackendError> {
        if inputs.len() != self.input_words() {
            return Err(BackendError::InvalidOptions(format!(
                "expected {} input words, got {}",
                self.input_words(),
                inputs.len()
            )));
        }
        let mut output = vec![0u64; self.result_words()];
        // SAFETY: both buffers have exactly the word counts the entry reads/writes.
        unsafe { (self.entry)(output.as_mut_ptr(), inputs.as_ptr()) };
        Ok(output)
    }
}

impl Drop for JitFunction {
    fn drop(&mut self) {
        unsafe { mlirExecutionEngineDestroy(self.engine) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{
        BTShift, BVJaggedPermute, BVLogical, Constant, Input, JTReduced, JaggedTensor, LogicalOp, ShiftOp,
        TensorExpr, TensorType,
    };
    use crate::lowering::packed::evaluate_tensor_fn;
    use std::rc::Rc;

    fn input(name: &str, width: usize) -> (TensorExpr, (String, TensorType)) {
        let tensor_type = TensorType::bitvec(width).unwrap();
        (Input::new(name, tensor_type.clone()).into(), (name.into(), tensor_type))
    }

    fn random_words(seed: &mut u64, width: usize) -> Vec<u64> {
        (0..word_count(width))
            .map(|word| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                let bits = width - word * 64;
                if bits >= 64 { *seed } else { *seed & ((1u64 << bits) - 1) }
            })
            .collect()
    }

    /// JIT `function` and compare it with the packed reference evaluator.
    fn check(function: &TensorFn) -> JitFunction {
        let jit = JitFunction::compile(function, 2).unwrap_or_else(|error| panic!("{error}"));
        let mut seed = 0x2545f4914f6cdd1d;
        for sample in 0..32 {
            let arguments: Vec<Vec<u64>> = function
                .inputs
                .iter()
                .map(|(_, tensor_type)| {
                    let width = tensor_type.width();
                    match sample {
                        0 => vec![0; word_count(width)],
                        1 => random_words(&mut 1, width).iter().enumerate().map(|(word, _)| {
                            let bits = width - word * 64;
                            if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 }
                        }).collect(),
                        _ => random_words(&mut seed, width),
                    }
                })
                .collect();
            let expected = evaluate_tensor_fn(function, &arguments).unwrap();
            assert_eq!(jit.call(&arguments.concat()).unwrap(), expected.words(), "{}", jit.llvm_mlir);
        }
        jit
    }

    fn or(parts: Vec<TensorExpr>) -> TensorExpr {
        BVLogical::new(LogicalOp::Or, parts.into_iter().map(|part| JTReduced::new(part, []))).into()
    }

    #[test]
    fn jit_runs_a_shifted_or_over_two_words() {
        let (x, first) = input("x", 100);
        let (y, second) = input("y", 100);
        let shifted = BTShift {
            expr: Rc::new(x),
            shift_op: ShiftOp::Logical,
            shift_amounts: 3,
        };
        let function = TensorFn::new("f", vec![first, second], or(vec![shifted.into(), y]));
        let jit = check(&function);
        let x = [0b1000u64 | (1 << 63), 1 << 35];
        let y = [1u64, 0];
        // (x >> 3) | y over 100 bits.
        let expected = vec![(x[0] >> 3) | (x[1] << 61) | y[0], x[1] >> 3];
        assert_eq!(jit.call(&[x[0], x[1], y[0], y[1]]).unwrap(), expected);
    }

    #[test]
    fn widths_around_word_boundaries_in_both_word_widths() {
        // <= 32 bits lowers with 32-bit words, wider with 64-bit words.
        for width in [1, 5, 31, 32, 33, 63, 64, 65, 127, 128, 129, 300] {
            let (a, first) = input("a", width);
            let (b, second) = input("b", width);
            let xor = BVLogical::new(LogicalOp::Xor, [JTReduced::new(a.clone(), []), JTReduced::new(b, [])]);
            let shifted = BTShift::new(a.clone(), ShiftOp::Logical, width / 3 + 1);
            let rotated = BTShift::new(a, ShiftOp::Rotate, width / 2);
            let function = TensorFn::new("edge", vec![first, second], or(vec![xor.into(), shifted.into(), rotated.into()]));
            check(&function);
        }
    }

    #[test]
    fn nullary_constants_and_one_bit_results() {
        let words = vec![0xdead_beef_0123_4567, 0x5];
        let constant = Constant::new(JaggedTensor::from_flat(words.clone()), TensorType::bitvec(67).unwrap());
        let jit = check(&TensorFn::new("constant", Vec::new(), constant));
        assert_eq!(jit.input_words(), 0);
        assert_eq!(jit.call(&[]).unwrap(), words);

        // One-bit result: a[0] & a[69] (a permute with one row of two lanes).
        let (a, argument) = input("a", 70);
        let rows = JaggedTensor::try_new(vec![0, 69], vec![1], vec![vec![0, 2]]).unwrap();
        let both = BVLogical::new(LogicalOp::And, [JTReduced::new(BVJaggedPermute::new(a, rows), [1])]);
        let jit = check(&TensorFn::new("bit", vec![argument], both));
        assert_eq!(jit.result_bits, 1);
        assert_eq!(jit.call(&[1, 1 << 5]).unwrap(), vec![1]);
        assert_eq!(jit.call(&[1, 0]).unwrap(), vec![0]);
    }

    #[test]
    fn many_inputs_are_read_at_their_own_offsets() {
        let widths = [3, 64, 65, 1, 130];
        let inputs: Vec<_> = widths.iter().enumerate().map(|(index, &width)| input(&format!("x{index}"), width)).collect();
        // Every input permuted onto 130 bits, then xor-ed together.
        let parts: Vec<JTReduced> = inputs
            .iter()
            .zip(widths)
            .map(|((expr, _), width)| {
                let rows = JaggedTensor::try_new((0..130).map(|bit| (bit * 7) % width).collect(), vec![130], vec![(0..=130).collect()]).unwrap();
                JTReduced::new(BVLogical::new(LogicalOp::Or, [JTReduced::new(BVJaggedPermute::new(expr.clone(), rows), [1])]), [])
            })
            .collect();
        let function = TensorFn::new("many", inputs.into_iter().map(|(_, argument)| argument).collect(), BVLogical::new(LogicalOp::Xor, parts));
        let jit = check(&function);
        assert_eq!(jit.input_words(), 1 + 1 + 2 + 1 + 3);
    }

    #[test]
    fn rejects_a_wrong_number_of_input_words() {
        let (a, argument) = input("a", 65);
        let jit = JitFunction::compile(&TensorFn::new("id", vec![argument], or(vec![a])), 0).unwrap();
        assert!(jit.call(&[0]).is_err());
        assert!(jit.call(&[0, 0, 0]).is_err());
    }
}
