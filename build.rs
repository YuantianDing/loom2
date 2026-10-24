//! Build the optional CPU runtime archives without imposing LLVM on default builds.
//!
//! LLVM and MLIR themselves are located by `mlir-sys` (through
//! `MLIR_SYS_230_PREFIX`) when the `mlir` feature is enabled.

use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=LOOM2_OPENMP_LIBRARY_DIR");

    if env::var_os("CARGO_FEATURE_LLVM").is_none() {
        return;
    }

    let manifest = PathBuf::from(required_env("CARGO_MANIFEST_DIR"));
    let output = PathBuf::from(required_env("OUT_DIR"));
    build_cpu_runtime(&manifest, &output);
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

