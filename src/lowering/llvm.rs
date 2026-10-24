//! Compile the synthesis module through upstream MLIR and LLVM, then link it
//! with the CPU runtime under `cpp/cpu`.
//!
//! ```text
//! frontend MLIR --(inline, canonicalize, cse, scf-to-openmp)--> CPU MLIR
//! CPU MLIR      --(standard *-to-llvm conversions)-----------> LLVM-dialect MLIR
//! LLVM dialect  --(mlirTranslateModuleToLLVMIR)--------------> LLVM IR
//! LLVM IR       --(LLVM new pass manager + target machine)---> object
//! object + result printer + cpp/cpu runtime ------------------> executable
//! ```
//!
//! Every stage is a separate public function so benchmark drivers can retain
//! each intermediate artifact. No out-of-tree MLIR dialect or pass is used.

use std::env;
use std::ffi::{c_char, c_void, CStr, CString, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Once;

use melior::ir::Module;
use melior::pass::PassManager;
use melior::Context;

const CPU_CPP_RUNTIME: &str = env!("LOOM2_CPU_CPP_RUNTIME");
const CPU_C_RUNTIME: &str = env!("LOOM2_CPU_C_RUNTIME");
const OPENMP_LIBRARY_DIRECTORY: Option<&str> = option_env!("LOOM2_OPENMP_LIBRARY_DIR");

/// Native-code generation and linking policy.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BackendOptions {
    /// LLVM optimization level in the inclusive range `0..=3`.
    pub optimization_level: u8,
    /// Lower `scf.parallel` through OpenMP and link `libomp`.
    pub openmp: bool,
    /// C++ linker. `None` uses `CXX`, falling back to `c++`.
    pub linker: Option<PathBuf>,
    /// Extra arguments appended immediately before `-o <executable>`.
    pub linker_arguments: Vec<OsString>,
}

impl Default for BackendOptions {
    fn default() -> Self {
        Self {
            optimization_level: 3,
            openmp: true,
            linker: None,
            linker_arguments: Vec::new(),
        }
    }
}

/// Failure from MLIR passes, LLVM code generation, I/O, or linking.
#[derive(Debug)]
pub enum BackendError {
    InvalidOptions(String),
    Compile(String),
    Io(io::Error),
    Link { status: String, stderr: String },
}

impl fmt::Display for BackendError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions(message) | Self::Compile(message) => formatter.write_str(message),
            Self::Io(error) => error.fmt(formatter),
            Self::Link { status, stderr } => {
                write!(formatter, "native linker exited with {status}")?;
                if !stderr.trim().is_empty() {
                    write!(formatter, ":\n{}", stderr.trim_end())?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for BackendError {}

impl From<io::Error> for BackendError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

fn check_options(options: &BackendOptions) -> Result<(), BackendError> {
    if options.optimization_level > 3 {
        return Err(BackendError::InvalidOptions(format!(
            "invalid LLVM optimization level {}; expected 0 through 3",
            options.optimization_level
        )));
    }
    Ok(())
}

fn run_pipeline(context: &Context, module: &mut Module, pipeline: &str) -> Result<(), BackendError> {
    melior::utility::register_all_passes();
    let manager = PassManager::new(context);
    melior::utility::parse_pass_pipeline(manager.as_operation_pass_manager(), pipeline)
        .map_err(|error| BackendError::Compile(format!("invalid pass pipeline {pipeline}: {error}")))?;
    manager
        .run(module)
        .map_err(|error| BackendError::Compile(format!("MLIR pipeline {pipeline} failed: {error}")))
}

/// Inline packed constructors and map `scf.parallel` to OpenMP (or serial
/// loops). The result is still standard CPU-level MLIR.
pub fn lower_to_cpu_mlir(context: &Context, module: &mut Module, options: &BackendOptions) -> Result<(), BackendError> {
    check_options(options)?;
    let mut passes = vec!["inline", "symbol-dce"];
    if options.optimization_level > 0 {
        passes.extend(["canonicalize", "cse"]);
    }
    passes.push(if options.openmp {
        "convert-scf-to-openmp"
    } else {
        "scf-parallel-for-to-nested-fors"
    });
    if options.optimization_level > 0 {
        passes.push("canonicalize");
    }
    run_pipeline(context, module, &format!("builtin.module({})", passes.join(",")))
}

/// Convert CPU MLIR to the LLVM dialect.
pub fn lower_to_llvm_mlir(context: &Context, module: &mut Module, options: &BackendOptions) -> Result<(), BackendError> {
    // `convert-scf-to-openmp` wraps each parallel body in `memref.alloca_scope`.
    // Lower memrefs (and with them the scope, via stack save/restore) while
    // that region is still a single block, i.e. before `convert-scf-to-cf`.
    let mut passes = vec![
        "convert-vector-to-scf",
        "expand-strided-metadata",
        "lower-affine",
        "convert-vector-to-llvm",
        "finalize-memref-to-llvm",
        "convert-scf-to-cf",
    ];
    if options.openmp {
        passes.push("convert-openmp-to-llvm");
    }
    passes.extend([
        "convert-math-to-llvm",
        "convert-arith-to-llvm",
        "convert-index-to-llvm",
        "convert-func-to-llvm",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
    ]);
    run_pipeline(context, module, &format!("builtin.module({})", passes.join(",")))
}

// LLVM and MLIR C APIs not covered by `mlir-sys`; both are linked statically.
type LLVMContextRef = *mut c_void;
type LLVMModuleRef = *mut c_void;
type LLVMTargetRef = *mut c_void;
type LLVMTargetMachineRef = *mut c_void;
type LLVMMemoryBufferRef = *mut c_void;
type LLVMPassBuilderOptionsRef = *mut c_void;
type LLVMErrorRef = *mut c_void;

unsafe extern "C" {
    fn mlirTranslateModuleToLLVMIR(module: mlir_sys::MlirOperation, context: LLVMContextRef) -> LLVMModuleRef;
    fn LLVMContextCreate() -> LLVMContextRef;
    fn LLVMContextDispose(context: LLVMContextRef);
    fn LLVMDisposeModule(module: LLVMModuleRef);
    fn LLVMPrintModuleToString(module: LLVMModuleRef) -> *mut c_char;
    fn LLVMDisposeMessage(message: *mut c_char);
    fn LLVMGetDefaultTargetTriple() -> *mut c_char;
    fn LLVMGetHostCPUName() -> *mut c_char;
    fn LLVMGetHostCPUFeatures() -> *mut c_char;
    fn LLVMGetTargetFromTriple(triple: *const c_char, target: *mut LLVMTargetRef, error: *mut *mut c_char) -> i32;
    fn LLVMCreateTargetMachine(
        target: LLVMTargetRef,
        triple: *const c_char,
        cpu: *const c_char,
        features: *const c_char,
        level: u32,
        relocation: u32,
        code_model: u32,
    ) -> LLVMTargetMachineRef;
    fn LLVMDisposeTargetMachine(machine: LLVMTargetMachineRef);
    fn LLVMSetTarget(module: LLVMModuleRef, triple: *const c_char);
    fn LLVMCreateTargetDataLayout(machine: LLVMTargetMachineRef) -> *mut c_void;
    fn LLVMSetModuleDataLayout(module: LLVMModuleRef, layout: *mut c_void);
    fn LLVMDisposeTargetData(layout: *mut c_void);
    fn LLVMTargetMachineEmitToMemoryBuffer(
        machine: LLVMTargetMachineRef,
        module: LLVMModuleRef,
        file_type: u32,
        error: *mut *mut c_char,
        buffer: *mut LLVMMemoryBufferRef,
    ) -> i32;
    fn LLVMGetBufferStart(buffer: LLVMMemoryBufferRef) -> *const c_char;
    fn LLVMGetBufferSize(buffer: LLVMMemoryBufferRef) -> usize;
    fn LLVMDisposeMemoryBuffer(buffer: LLVMMemoryBufferRef);
    fn LLVMCreatePassBuilderOptions() -> LLVMPassBuilderOptionsRef;
    fn LLVMDisposePassBuilderOptions(options: LLVMPassBuilderOptionsRef);
    fn LLVMRunPasses(
        module: LLVMModuleRef,
        passes: *const c_char,
        machine: LLVMTargetMachineRef,
        options: LLVMPassBuilderOptionsRef,
    ) -> LLVMErrorRef;
    fn LLVMGetErrorMessage(error: LLVMErrorRef) -> *mut c_char;
    fn LLVMDisposeErrorMessage(message: *mut c_char);
    fn LLVMInitializeX86TargetInfo();
    fn LLVMInitializeX86Target();
    fn LLVMInitializeX86TargetMC();
    fn LLVMInitializeX86AsmPrinter();
}

fn take_message(message: *mut c_char) -> String {
    if message.is_null() {
        return String::new();
    }
    // SAFETY: LLVM returns a NUL-terminated string owned by the caller.
    let text = unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned();
    unsafe { LLVMDisposeMessage(message) };
    text
}

/// An LLVM module translated from the LLVM dialect.
pub struct LlvmModule {
    context: LLVMContextRef,
    module: LLVMModuleRef,
}

impl LlvmModule {
    /// Translate a module that only contains LLVM-dialect operations.
    pub fn translate(module: &Module) -> Result<Self, BackendError> {
        let context = unsafe { LLVMContextCreate() };
        // SAFETY: the operation handle is valid while `module` is borrowed.
        let translated = unsafe { mlirTranslateModuleToLLVMIR(module.as_operation().to_raw(), context) };
        if translated.is_null() {
            unsafe { LLVMContextDispose(context) };
            return Err(BackendError::Compile("failed to translate the LLVM dialect to LLVM IR".into()));
        }
        Ok(Self {
            context,
            module: translated,
        })
    }

    /// Textual LLVM IR.
    pub fn to_ir(&self) -> String {
        take_message(unsafe { LLVMPrintModuleToString(self.module) })
    }

    /// Optimize with the new pass manager and emit a native object.
    pub fn emit_object(&self, options: &BackendOptions) -> Result<Vec<u8>, BackendError> {
        check_options(options)?;
        static INITIALIZE: Once = Once::new();
        INITIALIZE.call_once(|| unsafe {
            LLVMInitializeX86TargetInfo();
            LLVMInitializeX86Target();
            LLVMInitializeX86TargetMC();
            LLVMInitializeX86AsmPrinter();
        });
        unsafe {
            let triple = LLVMGetDefaultTargetTriple();
            let mut target = std::ptr::null_mut();
            let mut error = std::ptr::null_mut();
            if LLVMGetTargetFromTriple(triple, &mut target, &mut error) != 0 {
                LLVMDisposeMessage(triple);
                return Err(BackendError::Compile(format!("no native LLVM target: {}", take_message(error))));
            }
            let cpu = LLVMGetHostCPUName();
            let features = LLVMGetHostCPUFeatures();
            // Relocation model PIC (2) so the object links into a default PIE.
            let machine = LLVMCreateTargetMachine(
                target,
                triple,
                cpu,
                features,
                u32::from(options.optimization_level),
                2,
                0,
            );
            LLVMDisposeMessage(cpu);
            LLVMDisposeMessage(features);
            LLVMSetTarget(self.module, triple);
            LLVMDisposeMessage(triple);
            let layout = LLVMCreateTargetDataLayout(machine);
            LLVMSetModuleDataLayout(self.module, layout);
            LLVMDisposeTargetData(layout);

            let pipeline = CString::new(format!("default<O{}>", options.optimization_level)).unwrap();
            let pass_options = LLVMCreatePassBuilderOptions();
            let failure = LLVMRunPasses(self.module, pipeline.as_ptr(), machine, pass_options);
            LLVMDisposePassBuilderOptions(pass_options);
            if !failure.is_null() {
                let message = LLVMGetErrorMessage(failure);
                let text = CStr::from_ptr(message).to_string_lossy().into_owned();
                LLVMDisposeErrorMessage(message);
                LLVMDisposeTargetMachine(machine);
                return Err(BackendError::Compile(format!("LLVM optimization failed: {text}")));
            }

            let mut buffer = std::ptr::null_mut();
            // File type 1 is an object file.
            let status = LLVMTargetMachineEmitToMemoryBuffer(machine, self.module, 1, &mut error, &mut buffer);
            LLVMDisposeTargetMachine(machine);
            if status != 0 {
                return Err(BackendError::Compile(format!("object emission failed: {}", take_message(error))));
            }
            let bytes = std::slice::from_raw_parts(LLVMGetBufferStart(buffer).cast::<u8>(), LLVMGetBufferSize(buffer)).to_vec();
            LLVMDisposeMemoryBuffer(buffer);
            Ok(bytes)
        }
    }
}

impl Drop for LlvmModule {
    fn drop(&mut self) {
        unsafe {
            LLVMDisposeModule(self.module);
            LLVMContextDispose(self.context);
        }
    }
}

/// Link `object` and optional extra objects with the `cpp/cpu` runtime.
///
/// Extra objects precede the runtime archives, so a schema-aware result
/// printer overrides the generic numeric printer in `main.c`.
pub fn link_cpu_object(
    object: impl AsRef<Path>,
    extra_objects: &[PathBuf],
    executable: impl AsRef<Path>,
    options: &BackendOptions,
) -> Result<(), BackendError> {
    let executable = executable.as_ref();
    if let Some(parent) = executable.parent().filter(|path| !path.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let linker = options
        .linker
        .clone()
        .or_else(|| env::var_os("CXX").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("c++"));
    let mut command = Command::new(linker);
    command
        .arg(object.as_ref())
        .args(extra_objects)
        .arg(CPU_C_RUNTIME)
        .arg(CPU_CPP_RUNTIME)
        .arg("-pthread");
    if options.openmp {
        if let Some(directory) = OPENMP_LIBRARY_DIRECTORY {
            command.arg(format!("-L{directory}"));
            command.arg(format!("-Wl,-rpath,{directory}"));
        }
        command.arg("-lomp");
    }
    command.args(&options.linker_arguments).arg("-o").arg(executable);
    let output = command.output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(BackendError::Link {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

/// Compile a C source file (such as the generated result printer) to an object.
pub fn compile_c_object(source: &Path, object: &Path, options: &BackendOptions) -> Result<(), BackendError> {
    let compiler = env::var_os("CC").unwrap_or_else(|| "cc".into());
    let output = Command::new(compiler)
        .arg(format!("-O{}", options.optimization_level))
        .args(["-std=c11", "-fPIC", "-c"])
        .arg(source)
        .arg("-o")
        .arg(object)
        .output()?;
    if output.status.success() {
        Ok(())
    } else {
        Err(BackendError::Link {
            status: output.status.to_string(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_optimization_level_before_running_passes() {
        let options = BackendOptions {
            optimization_level: 4,
            ..BackendOptions::default()
        };
        assert!(matches!(check_options(&options), Err(BackendError::InvalidOptions(_))));
    }
}
