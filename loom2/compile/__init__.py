"""Compilation and synthesis entry points."""

from .automaton import TreeAutomaton
from .cfg import Cfg
from .function import CompiledFunction, compile
from .synthesis import DEFAULT_MAX_SIZE, SynthesisError, SynthesisProgram, synth_fun

__all__ = [
    "Cfg",
    "TreeAutomaton",
    "DEFAULT_MAX_SIZE",
    "CompiledFunction",
    "SynthesisError",
    "SynthesisProgram",
    "compile",
    "synth_fun",
]
