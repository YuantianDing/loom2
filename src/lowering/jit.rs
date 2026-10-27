//! In-process LLVM JIT for one traced `TensorFn`.
//!
//! ```text
//! TensorFn --fuse--> define_fun_<name>(vector<Nxi64>...) -> vector<Mxi64>
//!   + func.func @loom_jit_entry(%out: memref<Mxi64>, %in: memref<<total>xi64>)
//!       vector.load every input from the packed word buffer `%in`,
//!       call define_fun_<name>, vector.store the result to `%out`
//!   --(CPU MLIR, LLVM dialect)--> MLIR ExecutionEngine (LLVM ORC)
//! ```
//!
//! Inputs are concatenated `u64` words (bit `d` of an argument is word
//! `d / 64`, bit `d % 64`); the result uses the same packing. The constructor
//! is the same code the synthesizer runs, so the JIT matches it bit for bit.
//! Each memref argument is passed as the expanded descriptor
//! `(allocated, aligned, offset, size, stride)`.

use std::ffi::c_void;

use high_mlir::func::Signature;
use high_mlir::arith::{self, Constant};
use high_mlir::{builtin, memref, vector, Root, Ty, Tuple, Val, I64};
use mlir_sys::{mlirExecutionEngineCreate, mlirExecutionEngineDestroy, mlirExecutionEngineLookup, MlirExecutionEngine};

use super::exec::functions::{define_fun_symbol, lower_tensor_fn, packed_type, result_bits};
use super::exec::sliced::{
    is_sliced, lower_sliced_tensor_fn, lower_transpose, slice_rows, sliced_signature, sliced_type, unslice_rows,
};
use super::exec::{index, new_context, to_melior};
use super::llvm::{lower_to_cpu_mlir, lower_to_llvm_mlir, BackendError, BackendOptions};
use super::word_count;
use crate::compile::pass::fuse;
use crate::compile::TensorFn;

/// Public symbol of the generated entry point.
pub const ENTRY: &str = "loom_jit_entry";

type Entry = unsafe extern "C" fn(*mut u64, *mut u64, i64, i64, i64, *const u64, *const u64, i64, i64, i64);

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
    /// Candidates per call: 1, or 64 for a bit-sliced constructor.
    pub lanes: usize,
}

impl JitFunction {
    /// Lower, JIT-compile, and look up the entry point of `function`.
    pub fn compile(function: &TensorFn, optimization_level: u8) -> Result<Self, BackendError> {
        let lowering = |error: super::LoweringError| BackendError::Compile(error.to_string());
        let fused = fuse(function);
        let input_bits: Vec<usize> = fused.inputs.iter().map(|(_, tensor_type)| tensor_type.width()).collect();
        let result_bits = result_bits(&fused).map_err(lowering)?;
        let callee = Signature::new(
            define_fun_symbol(&fused.name),
            input_bits.iter().map(|&bits| packed_type(bits)),
            [packed_type(result_bits)],
        );
        let input_words: usize = input_bits.iter().map(|&bits| word_count(bits)).sum();
        let buffer = |words: usize| Ty::memref([Some(words as u64)], I64);
        let mut lowered = Ok(());
        let module = builtin::module(|| {
            lowered = lower_tensor_fn(&fused).map(drop);
            Signature::new(ENTRY, [buffer(word_count(result_bits)), buffer(input_words)], []).define(|buffers| {
                let (output, input) = (&buffers[0], &buffers[1]);
                let mut offset = 0usize;
                let mut values: Vec<Val> = Vec::new();
                for &bits in &input_bits {
                    values.push(vector::load(input, (index(offset),), packed_type(bits), ()).into());
                    offset += word_count(bits);
                }
                let result = callee.call(values)[0].clone();
                vector::store(result, output, (index(0),), ());
                Tuple(Vec::new())
            });
        });
        lowered.map_err(lowering)?;
        Self::load(&function.name, &module, optimization_level, input_bits, result_bits, 1)
    }

    /// JIT-compile the bit-sliced constructor of `function` (see
    /// [`sliced`](super::exec::sliced)) behind the same transposes as the
    /// synthesizer. [`call`](Self::call) then takes 64 candidates: lane `k`'s
    /// inputs at word `k * input_words()`, its result at `k * result_words()`.
    pub fn compile_sliced(function: &TensorFn, optimization_level: u8) -> Result<Self, BackendError> {
        let lowering = |error: super::LoweringError| BackendError::Compile(error.to_string());
        let fused = fuse(function);
        if !is_sliced(&fused) {
            return Err(BackendError::InvalidOptions(format!("{} is not bit-sliced", function.name)));
        }
        let input_bits: Vec<usize> = fused.inputs.iter().map(|(_, tensor_type)| tensor_type.width()).collect();
        let result_bits = result_bits(&fused).map_err(lowering)?;
        let input_words: usize = input_bits.iter().map(|&bits| word_count(bits)).sum();
        let result_words = word_count(result_bits);
        let callee = sliced_signature(&fused.name, &input_bits, result_bits);
        let buffer = |words: usize| Ty::memref([Some((64 * words) as u64)], I64);
        let mut lowered = Ok(());
        let module = builtin::module(|| {
            lower_transpose();
            lowered = lower_sliced_tensor_fn(&fused).map(drop);
            Signature::new(ENTRY, [buffer(result_words), buffer(input_words)], []).define(|buffers| {
                let (output, input) = (&buffers[0], &buffers[1]);
                let scratch: Val = memref::alloca(Ty::memref([Some(64)], I64), (), ()).into();
                let mut offset = 0usize;
                let mut arguments = Vec::new();
                for &bits in &input_bits {
                    let words = word_count(bits);
                    let sliced: Val = memref::alloca(sliced_type(bits), (), ()).into();
                    let row = |lane: &Val, word: &Val| {
                        let at = arith::addi(arith::muli(lane, index(input_words), ()), arith::addi(word, index(offset), ()), ());
                        memref::load(input, (&at,), ())
                    };
                    slice_rows(&row, &index(64), words, bits, &Constant::int(0, I64).into(), &scratch, &sliced);
                    arguments.push(sliced);
                    offset += words;
                }
                let out: Val = memref::alloca(sliced_type(result_bits), (), ()).into();
                arguments.push(out.clone());
                callee.call(arguments);
                unslice_rows(&out, result_bits, result_words, &scratch, &|lane, word, value| {
                    let at = arith::addi(arith::muli(lane, index(result_words), ()), word, ());
                    memref::store(value, output, (&at,), ());
                });
                Tuple(Vec::new())
            });
        });
        lowered.map_err(lowering)?;
        Self::load(&function.name, &module, optimization_level, input_bits, result_bits, 64)
    }

    /// Lower `module` to the LLVM dialect and load it into an execution engine.
    fn load(
        name: &str,
        module: &Root,
        optimization_level: u8,
        input_bits: Vec<usize>,
        result_bits: usize,
        lanes: usize,
    ) -> Result<Self, BackendError> {
        let lowering = |error: super::LoweringError| BackendError::Compile(error.to_string());
        let backend = BackendOptions {
            optimization_level,
            openmp: false,
            ..BackendOptions::default()
        };
        let context = new_context();
        let mut module = to_melior(&context, module, "JIT lowering").map_err(lowering)?;

        lower_to_cpu_mlir(&context, &mut module, &backend)?;
        lower_to_llvm_mlir(&context, &mut module, &backend)?;
        let llvm_mlir = module.as_operation().to_string();

        // SAFETY: the module is a valid LLVM-dialect module; the engine owns
        // its own LLVM context, so the MLIR context may be dropped afterwards.
        let engine = unsafe { mlirExecutionEngineCreate(module.to_raw(), i32::from(optimization_level), 0, std::ptr::null(), false, false) };
        if engine.ptr.is_null() {
            return Err(BackendError::Compile(format!("LLVM JIT rejected {name}")));
        }
        let symbol = melior::StringRef::new(ENTRY);
        let address = unsafe { mlirExecutionEngineLookup(engine, symbol.to_raw()) };
        if address.is_null() {
            unsafe { mlirExecutionEngineDestroy(engine) };
            return Err(BackendError::Compile(format!("JIT entry {ENTRY} not found")));
        }
        // SAFETY: `loom_jit_entry` takes two expanded rank-1 memref descriptors.
        let entry = unsafe { std::mem::transmute::<*mut c_void, Entry>(address) };
        Ok(Self {
            engine,
            entry,
            input_bits,
            result_bits,
            llvm_mlir,
            lanes,
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

    /// Run on packed inputs (of every lane); bits beyond each input's width
    /// must be zero.
    pub fn call(&self, inputs: &[u64]) -> Result<Vec<u64>, BackendError> {
        if inputs.len() != self.lanes * self.input_words() {
            return Err(BackendError::InvalidOptions(format!(
                "expected {} input words, got {}",
                self.lanes * self.input_words(),
                inputs.len()
            )));
        }
        let mut output = vec![0u64; self.lanes * self.result_words()];
        let (output_words, input_words) = (output.len() as i64, inputs.len() as i64);
        let (output_data, input_data) = (output.as_mut_ptr(), inputs.as_ptr());
        // SAFETY: both buffers have exactly the word counts the entry reads/writes.
        unsafe {
            (self.entry)(output_data, output_data, 0, output_words, 1, input_data, input_data, 0, input_words, 1)
        };
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
