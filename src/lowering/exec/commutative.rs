//! Filters for commutative binary constructors applied to one child pool.
//!
//! Enumerate the upper triangle of pairs directly. Packed filters pair rows;
//! transposed filters pair 64-row blocks and keep all 64 shifts, including
//! every lane of diagonal blocks. There is no within-block symmetry check.

use high_mlir::arith::{self, CmpIPredicate};
use high_mlir::Val;

use super::hash_filter::{lower_filter, HashFilterConfig};
use super::index;

/// Emit `@commutative_hash_filter_*`. Both child arguments must describe
/// the same pool, with the same expression size.
pub fn lower_commutative_hash_filter(config: &HashFilterConfig) -> Val {
    assert!(config.commutative && config.child_value_words.len() == 2);
    lower_filter(config, true)
}

/// Number of unordered pairs, including the diagonal.
pub(super) fn pair_count(count: &Val) -> Val {
    arith::divui(arith::muli(count, arith::addi(count, index(1), ()), ()), index(2), ())
}

/// Decode `linear` in `0..count * (count + 1) / 2` to `(a, b)`, `a <= b`.
/// Pair triangle rows `r` and `count - 1 - r`: together they have `count + 1`
/// entries. For odd `count`, the final, middle row fills only half a strip.
/// This needs integer division and selects, with no square root or search.
pub(super) fn pair(linear: &Val, count: &Val) -> [Val; 2] {
    let width = arith::addi(count, index(1), ());
    let row = arith::divui(linear, &width, ());
    let column = arith::remui(linear, &width);
    let length = arith::subi(count, &row, ());
    let first = arith::cmpi(CmpIPredicate::Ult, &column, &length);
    let a = arith::select(&first, &row, arith::subi(&length, index(1), ()));
    let b = arith::select(&first, arith::addi(&row, &column, ()), arith::subi(&column, index(1), ()));
    [a.into(), b.into()]
}

#[cfg(all(test, feature = "llvm"))]
mod tests {
    use std::collections::BTreeSet;

    use high_mlir::{builtin, func::Signature, INDEX};
    use melior::ExecutionEngine;

    use super::*;
    use crate::lowering::exec::{new_context, to_melior};
    use crate::lowering::llvm::{lower_to_cpu_mlir, lower_to_llvm_mlir, BackendOptions};

    #[test]
    fn triangular_pairs_cover_each_unordered_pair_once() {
        let root = builtin::module(|| {
            Signature::new("count", [INDEX.into()], [INDEX.into()])
                .define(|args| pair_count(&args[0]).into());
            Signature::new("pair", [INDEX.into(), INDEX.into()], [INDEX.into()]).define(|args| {
                let [a, b] = pair(&args[1], &args[0]);
                arith::addi(arith::muli(a, &args[0], ()), b, ()).into()
            });
        });
        let context = new_context();
        let mut module = to_melior(&context, &root, "commutative pairs test").unwrap();
        let backend = BackendOptions { openmp: false, ..BackendOptions::default() };
        lower_to_cpu_mlir(&context, &mut module, &backend).unwrap();
        lower_to_llvm_mlir(&context, &mut module, &backend).unwrap();
        let engine = ExecutionEngine::new(&module, 2, &[], false, false);
        let count = engine.lookup("count");
        let pair = engine.lookup("pair");
        assert!(!count.is_null() && !pair.is_null());
        // SAFETY: these entry points take and return native-width index values;
        // the engine stays alive for every call.
        let count: unsafe extern "C" fn(usize) -> usize = unsafe { std::mem::transmute(count) };
        let pair: unsafe extern "C" fn(usize, usize) -> usize = unsafe { std::mem::transmute(pair) };
        for n in (0..=129).chain([255, 256, 257]) {
            let length = unsafe { count(n) };
            assert_eq!(length, n * (n + 1) / 2);
            let actual: BTreeSet<usize> = (0..length).map(|i| unsafe { pair(n, i) }).collect();
            let expected: BTreeSet<usize> = (0..n).flat_map(|a| (a..n).map(move |b| a * n + b)).collect();
            assert_eq!(actual.len(), length, "duplicate pairs for {n} rows");
            assert_eq!(actual, expected, "missing pairs for {n} rows");
        }
    }
}
