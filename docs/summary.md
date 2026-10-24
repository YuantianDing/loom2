# Python Reimplementation of `../loom2mlir`

* `maturin` Project, implemented using Python + Rust + C++ (MLIR)
* Remember to add `derive_more` crate. Derive automatic `Debug`, `Display`, `From`, `Into`, etc.
* Use `indenter` to format large code.

## Source structure

```text
loom2/
  lang/             # Input types and functions.
  bit_arrangement/  # Assign bits for each value.
  tracing/          # Trace Python into a tensor AST.
  compile/          # Python compilation orchestration.
src/
  compile/
    expr.rs
    ir.rs
    mod.rs
  lowering/
    circuit/
      i6.rs
      mod.rs
      permute.rs
      sorting.rs
      u64_splitting.rs
      verify.rs
    exec/
      constraint.rs
      definefun.rs
      hash_filter.rs
      mod.rs
      synthesis.rs
    error.rs
    grammar.rs
    mlir.rs
    mod.rs
```

## Python Representation of Loom's input format

`Vec` is used to represent an bit array.

E.g. `Vec[2, 3](bool)` has the logical shape of `tensor<2x3xi1>` during tracing; MLIR lowering packs the flattened bits directly into `i64` words.

`Vec` allow jagged tensor schma, using the `*args` syntax:

`Vec(Vec[3](bool), Vec[3](bool))` and `Vec[2](bool, bool, bool)`  is equivalent as `Vec[2, 3](bool)`. Assume everything under `*args` have ths same rank/dimensions.

`Vec` allow labels to denote each subtensor, using the `**kwarg` syntax:

`Vec(a=Vec[3](bool), b=Vec[3](bool))` and `Vec[2](a=bool, b=bool, c=bool)`  is equivalent as `Vec[2, 3](bool)`. Assume everything under `*args` have ths same rank/dimensions. Such label will be used as indices. However, they will be removed after some passes.

Any type can replace `bool` at tracing, however, only runtime `bool` and constant `int` will be used in the subsequent IR.

`Vec.pytree` uses to convert a Python structure into `Vec`.

```py
Strings = {"": bool, "0": bool, "1": bool, "01": bool, "10": bool, "010": bool]


@loom2.compile
def re_and(a: Vec(**Strings), b: Vec(**Strings)):
    return a & b

@loom2.compile
def re_concat(a: Vec(**Strings), b: Vec(**Strings)):
    a_indices = { k: [k[:i] for i in range(len(k)+1)] for k in Strings.keys()}
    b_indices = { k: [k[i:] for i in range(len(k)+1)] for k in Strings.keys()}
    a_indices, b_indices = Vec.pytree(a_indices), Vec.pytree(b_indices)
    return (a[a_indices] & b[b_indices]).exists()
```

```py
@loom2.compile
def re_star(a: StringBV):
    a[""] = true

    max_len = max(len(s) for s in Substrings)

    for i in range(1, max_len+1):
        indices1 = Vec.pytree({
            k: [k[:i] for i in range(len(k)+1)] if len(k) == i else []
            for k in Substrings
        })
        indices2 = Vec.pytree({
            k: [k[i:] for i in range(len(k)+1)] if len(k) == i else []
            for k in Substrings
        })

        a |= (a[indices1] & a[indices2]).exists()

    return a
```

Assumes:

1. Quantifiers (`exists`, etc) should only be used on the last dimension.
