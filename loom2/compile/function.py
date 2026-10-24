"""Python callable wrapper for functions traced into native ``TensorFn`` IR.

Calling a :class:`CompiledFunction` on runtime :class:`~loom2.VecValue`
arguments runs it through the LLVM JIT: the function is traced once, lowered
by the same pipeline the synthesizer uses (fusion, loop program, MLIR, LLVM),
and compiled in-process by MLIR's ExecutionEngine. Later calls reuse the
compiled code. Calls made while tracing another function, or with values the
JIT cannot take, run the Python body.
"""

from functools import update_wrapper
from inspect import signature
import os
from typing import Any, Callable, Optional, Union, get_type_hints

from loom2.lang import Vec, VecValue
from loom2.tracing import is_tracing, trace

#: ``"auto"`` JIT when possible; ``True`` require the JIT; ``False`` Python only.
JitMode = Union[bool, str]


def _native_jit() -> Optional[type]:
    try:
        from loom2._loom2 import JitFunction
    except ImportError:  # extension missing or built without the ``llvm`` feature
        return None
    return JitFunction


def _default_mode() -> JitMode:
    setting = os.environ.get("LOOM2_JIT", "auto").lower()
    return {"0": False, "false": False, "off": False, "1": True, "true": True, "on": True}.get(setting, "auto")


def _pack(values: tuple, width: int) -> list:
    number = 0
    for index, value in enumerate(values):
        if value:
            number |= 1 << index
    mask = (1 << 64) - 1
    return [(number >> (64 * word)) & mask for word in range((width + 63) // 64)]


def _unpack(words: list, width: int) -> tuple:
    number = 0
    for index, word in enumerate(words):
        number |= word << (64 * index)
    return tuple(bool((number >> index) & 1) for index in range(width))


class CompiledFunction:
    """A normal Python function with a stable synthesis constructor name."""

    def __init__(
        self,
        function: Callable[..., Any],
        name: Optional[str] = None,
        *,
        jit: Optional[JitMode] = None,
        optimization_level: int = 2,
    ) -> None:
        self.function = function
        update_wrapper(self, function)
        if name is not None:
            self.__name__ = name
        self.jit = _default_mode() if jit is None else jit
        self.optimization_level = optimization_level
        self._compiled: Any = None  # JitFunction, or False once known unusable
        self._parameters: Optional[list] = None
        self._return: Any = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.jit is False or is_tracing():
            return self.function(*args, **kwargs)
        bound = signature(self.function).bind(*args, **kwargs)
        values = list(bound.arguments.values())
        if not all(isinstance(value, VecValue) for value in values):
            if self.jit is True:
                raise TypeError(f"{self.__name__}: the JIT takes VecValue arguments only")
            return self.function(*args, **kwargs)
        compiled = self.compiled()
        if compiled is None:
            return self.function(*args, **kwargs)
        return self._run(compiled, values)

    def compiled(self) -> Any:
        """The native ``JitFunction``, compiled on first use; ``None`` if unavailable."""
        if self._compiled is None:
            self._compiled = self._compile()
        return self._compiled or None

    def _compile(self) -> Any:
        native = _native_jit()
        if native is None:
            if self.jit is True:
                raise RuntimeError("the loom2 extension was built without the `llvm` feature")
            return False
        try:
            traced = self.trace()
        except (TypeError, NotImplementedError, ValueError):
            if self.jit is True:
                raise
            return False  # not traceable: keep the Python body
        try:
            hints = get_type_hints(self.function)
        except (NameError, TypeError):
            hints = dict(getattr(self.function, "__annotations__", {}))
        self._parameters = [hints.get(name) for name in signature(self.function).parameters]
        self._return = hints.get("return")
        return native(traced, optimization_level=self.optimization_level)

    def _run(self, compiled: Any, values: list) -> Any:
        words: list = []
        for value, schema, bits in zip(values, self._parameters, compiled.input_bits):
            if len(value.values) != bits or (
                isinstance(schema, Vec) and value.schema is not schema and value.schema != schema
            ):
                raise TypeError(f"{self.__name__}: argument does not match its Vec annotation")
            words.extend(_pack(value.values, bits))
        result = _unpack(compiled(words), compiled.result_bits)
        if isinstance(self._return, Vec):
            return VecValue(self._return, result)
        if self._return is bool or (self._return is None and compiled.result_bits == 1):
            return result[0]
        return VecValue(Vec(*(bool for _ in result)), result)

    @property
    def arity(self) -> int:
        return len(signature(self.function).parameters)

    def trace(self, *, builder_factory=None) -> object:
        """Execute the callback once and return its tensor-wide function IR."""
        original_name = self.function.__name__
        self.function.__name__ = self.__name__
        try:
            if builder_factory is None:
                return trace(self.function)
            return trace(self.function, builder_factory=builder_factory)
        finally:
            self.function.__name__ = original_name


def compile(
    function: Optional[Callable[..., Any]] = None,
    *,
    name: Optional[str] = None,
    jit: Optional[JitMode] = None,
    optimization_level: int = 2,
) -> Any:
    """Mark a typed Python function as a Loom tensor constructor.

    Tracing is lazy: :func:`loom2.synth_fun` traces for synthesis, and the
    first call on ``VecValue`` arguments traces and JIT-compiles it with LLVM.
    ``jit`` is ``"auto"`` (default; fall back to Python when the function
    cannot be traced or the extension lacks LLVM), ``True`` (raise instead),
    or ``False`` (always run the Python body). ``LOOM2_JIT=0`` disables the
    JIT globally.
    """

    def decorate(callback: Callable[..., Any]) -> CompiledFunction:
        return CompiledFunction(callback, name, jit=jit, optimization_level=optimization_level)

    return decorate if function is None else decorate(function)


__all__ = ["CompiledFunction", "compile"]
