"""Python entry point for Loom's typed tensor IR.

The active frontend traces ``Vec(bool)`` callbacks directly into the Rust
``TensorFn`` graph. Static ``Vec(int)`` selections become jagged permutation
metadata; no Python-side expression AST is retained.
"""

from .lang import Vec, VecIndex, VecValue
from .compile import DEFAULT_MAX_SIZE, Cfg, CompiledFunction, TreeAutomaton, SynthesisError, SynthesisProgram, compile, synth_fun
from .tracing import trace

try:
    from ._loom2 import native_version
except ImportError:  # The pure-Python surface is usable before maturin builds.
    def native_version():
        """Return ``None`` when the native extension has not been built."""
        return None


__all__ = [
    "DEFAULT_MAX_SIZE",
    "Cfg",
    "TreeAutomaton",
    "Vec",
    "VecIndex",
    "VecValue",
    "CompiledFunction",
    "SynthesisError",
    "SynthesisProgram",
    "compile",
    "native_version",
    "trace",
    "synth_fun",
]
