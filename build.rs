//! Build the optional CPU and CUDA runtime archives without imposing LLVM or
//! CUDA on default builds.
//!
//! LLVM and MLIR themselves are located by `mlir-sys` (through
//! `MLIR_SYS_230_PREFIX`) when the `mlir` feature is enabled. The `cuda`
//! feature finds `nvcc` under `CUDA_HOME` (default `/usr/local/cuda`) and the
//! cuco headers in the `cpp/cuda/cuCollections` submodule (or
//! `LOOM2_CUCO_INCLUDE_DIR`), and targets `LOOM2_CUDA_ARCH` (such as `75`;
//! default: the compute capability `nvidia-smi` reports for GPU 0, else 75).

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=LOOM2_OPENMP_LIBRARY_DIR");
    for variable in ["CUDA_HOME", "LOOM2_CUCO_INCLUDE_DIR", "LOOM2_CUDA_ARCH"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }

    if env::var_os("CARGO_FEATURE_LLVM").is_none() {
        return;
    }

    let manifest = PathBuf::from(required_env("CARGO_MANIFEST_DIR"));
    let output = PathBuf::from(required_env("OUT_DIR"));
    build_cpu_runtime(&manifest, &output);
    if env::var_os("CARGO_FEATURE_CUDA").is_some() {
        build_cuda_runtime(&manifest, &output);
    }
    if let Some(directory) = env::var_os("LOOM2_OPENMP_LIBRARY_DIR") {
        println!(
            "cargo:rustc-env=LOOM2_OPENMP_LIBRARY_DIR={}",
            PathBuf::from(directory).display()
        );
    }
}

fn build_cpu_runtime(manifest: &Path, output: &Path) {
    let cpu = manifest.join("cpp/cpu");
    for source in [
        "i64hashset.cpp",
        "i64hashset.hpp",
        "memref_copy.cpp",
        "main.c",
    ] {
        println!("cargo:rerun-if-changed={}", cpu.join(source).display());
    }
    emit_source_dependencies(&cpu.join("libcuckoo"));

    cc::Build::new()
        .cargo_metadata(false)
        .cpp(true)
        .std("c++17")
        .include(&cpu)
        .file(cpu.join("i64hashset.cpp"))
        .file(cpu.join("memref_copy.cpp"))
        .compile("loom2_cpu_cpp");
    cc::Build::new()
        .cargo_metadata(false)
        .file(cpu.join("main.c"))
        .compile("loom2_cpu_c");

    println!(
        "cargo:rustc-env=LOOM2_CPU_CPP_RUNTIME={}",
        output.join(static_library_name("loom2_cpu_cpp")).display()
    );
    println!(
        "cargo:rustc-env=LOOM2_CPU_C_RUNTIME={}",
        output.join(static_library_name("loom2_cpu_c")).display()
    );
}

fn build_cuda_runtime(manifest: &Path, output: &Path) {
    let cuda = manifest.join("cpp/cuda");
    for source in ["i64hashset.cuh", "i64hashset_device.cu", "runtime.cu"] {
        println!("cargo:rerun-if-changed={}", cuda.join(source).display());
    }
    let home = env::var_os("CUDA_HOME").map(PathBuf::from).unwrap_or_else(|| "/usr/local/cuda".into());
    let nvcc = home.join("bin/nvcc");
    let cuco = env::var_os("LOOM2_CUCO_INCLUDE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| cuda.join("cuCollections/include"));
    if !cuco.join("cuco/static_set.cuh").exists() {
        panic!(
            "cuco headers not found in {}; run `git submodule update --init` or set LOOM2_CUCO_INCLUDE_DIR",
            cuco.display()
        );
    }
    let arch = env::var("LOOM2_CUDA_ARCH").ok().or_else(detect_cuda_arch).unwrap_or_else(|| "75".into());
    let gencode = format!("arch=compute_{arch},code=[sm_{arch},compute_{arch}]");
    let common = ["-std=c++17", "--expt-relaxed-constexpr", "--expt-extended-lambda", "-O3"];

    // Relocatable device library linked into every generated kernel at load
    // time, embedded as a byte array.
    let fatbin = output.join("i64hashset_device.fatbin");
    let status = Command::new(&nvcc)
        .args(common)
        .args(["-rdc=true", "-fatbin", "-gencode", &gencode])
        .arg("-I")
        .arg(&cuco)
        .arg(cuda.join("i64hashset_device.cu"))
        .arg("-o")
        .arg(&fatbin)
        .status()
        .unwrap_or_else(|error| panic!("failed to run {}: {error}", nvcc.display()));
    assert!(status.success(), "nvcc failed to build the hash-set device library");
    let bytes = fs::read(&fatbin).expect("read the device fatbin");
    let mut embedded = String::from("#include <stddef.h>\n__attribute__((aligned(8))) const unsigned char loom2_hashset_device_fatbin[] = {");
    for (index, byte) in bytes.iter().enumerate() {
        if index % 24 == 0 {
            embedded.push('\n');
        }
        embedded.push_str(&format!("{byte},"));
    }
    embedded.push_str(&format!("\n}};\nconst size_t loom2_hashset_device_fatbin_size = {};\n", bytes.len()));
    let embedded_path = output.join("i64hashset_device_fatbin.c");
    fs::write(&embedded_path, embedded).expect("write the embedded fatbin");

    // nvcc is invoked directly: cc-rs passes host-compiler flags to an
    // explicitly chosen nvcc without `-Xcompiler`.
    let mut objects = Vec::new();
    for source in [cuda.join("runtime.cu"), manifest.join("cpp/cpu/memref_copy.cpp")] {
        let object = output.join(format!("{}.o", source.file_stem().unwrap().to_string_lossy()));
        let status = Command::new(&nvcc)
            .args(common)
            .args(["-Xcompiler", "-fPIC", "-gencode", &gencode, "-I"])
            .arg(&cuco)
            .arg("-I")
            .arg(&cuda)
            .arg("-c")
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .status()
            .unwrap_or_else(|error| panic!("failed to run {}: {error}", nvcc.display()));
        assert!(status.success(), "nvcc failed to compile {}", source.display());
        objects.push(object);
    }
    let archive = output.join(static_library_name("loom2_cuda_cu"));
    let _ = fs::remove_file(&archive);
    let status = Command::new(env::var_os("AR").unwrap_or_else(|| "ar".into()))
        .arg("crs")
        .arg(&archive)
        .args(&objects)
        .status()
        .expect("failed to run ar");
    assert!(status.success(), "ar failed to archive the CUDA runtime");
    cc::Build::new().cargo_metadata(false).file(&embedded_path).compile("loom2_cuda_fatbin");

    println!(
        "cargo:rustc-env=LOOM2_CUDA_RUNTIME={}",
        output.join(static_library_name("loom2_cuda_cu")).display()
    );
    println!(
        "cargo:rustc-env=LOOM2_CUDA_FATBIN={}",
        output.join(static_library_name("loom2_cuda_fatbin")).display()
    );
    println!("cargo:rustc-env=LOOM2_CUDA_LIBRARY_DIR={}", home.join("lib64").display());
    println!("cargo:rustc-env=LOOM2_CUDA_ARCH={arch}");
}

/// `sm_XY` digits of GPU 0, from `nvidia-smi`.
fn detect_cuda_arch() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    let text = String::from_utf8(output.stdout).ok()?;
    let digits: String = text.lines().next()?.chars().filter(char::is_ascii_digit).collect();
    (output.status.success() && !digits.is_empty()).then_some(digits)
}

fn required_env(name: &str) -> OsString {
    env::var_os(name).unwrap_or_else(|| panic!("Cargo did not set {name}"))
}

fn static_library_name(stem: &str) -> String {
    if env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        format!("{stem}.lib")
    } else {
        format!("lib{stem}.a")
    }
}

fn emit_source_dependencies(root: &Path) {
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            emit_source_dependencies(&path);
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("c" | "cc" | "cpp" | "h" | "hpp" | "td" | "txt")
        ) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

