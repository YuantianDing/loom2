//! Regenerate the Python type stubs collected from PyO3 annotations.

use std::fs;
use std::path::Path;

fn main() -> pyo3_stub_gen::Result<()> {
    _loom2::stub_info()?.generate()?;

    // pyo3-stub-gen models a dotted native module as a package in a mixed
    // project. Maturin installs this project as a sibling extension module, so
    // normalize the generated package stub to `loom2/_loom2.pyi`.
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let generated_dir = root.join("loom2/_loom2");
    let generated = generated_dir.join("__init__.pyi");
    let destination = root.join("loom2/_loom2.pyi");
    if generated.exists() {
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(generated, &destination)?;
        fs::remove_dir(generated_dir)?;
    }

    // Keep generated files friendly to `git diff --check` even when Rust
    // docstrings contain Markdown paragraph separators.
    let contents = fs::read_to_string(&destination)?;
    let contents = contents
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    fs::write(destination, format!("{}\n", contents.trim_end()))?;

    Ok(())
}
