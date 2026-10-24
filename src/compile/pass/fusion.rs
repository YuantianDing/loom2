//! Fuse permutations with the reductions that consume them.
//!
//! ```text
//! permute(permute(x, p), q)                 => permute(x, p . q)   (p single-lane)
//! reduce(op, permute(x, p))                 => gather.op x p
//!   (`reduce` is `BTExpandReduce [batch.., reduce.op]` or a reduced operand)
//! reduce(op, inner(permute(a, p), permute(b, q)))
//!                                           => gather2.op.inner a p, b q
//! ```
//!
//! Fused nodes reduce each row while its lanes are produced, so lowering never
//! materializes the lane-major intermediate. Shared subgraphs stay shared, and
//! nested logic with the same operator is not flattened.

use std::collections::HashMap;
use std::rc::Rc;

use crate::compile::typing::{expand_reduce_form, ExpandReduceForm, TypeCache};

use crate::compile::{
    BTExpandReduce, BTShift, BVGather, BVGather2, BVJaggedPermute, BVLogical, JTReduced, JaggedTensor,
    TensorExpr, TensorFn,
};

#[derive(Default)]
struct Fuser {
    memo: HashMap<*const TensorExpr, Rc<TensorExpr>>,
    types: TypeCache,
}

fn single_lane(indices: &JaggedTensor<usize>) -> bool {
    (0..indices.row_count()).all(|row| indices.row(row).len() == 1)
}

impl Fuser {
    fn rewrite(&mut self, expression: &Rc<TensorExpr>) -> Rc<TensorExpr> {
        let key = Rc::as_ptr(expression);
        if let Some(done) = self.memo.get(&key) {
            return Rc::clone(done);
        }
        let result = self.rewrite_node(expression);
        self.memo.insert(key, Rc::clone(&result));
        result
    }

    fn rewrite_node(&mut self, expression: &Rc<TensorExpr>) -> Rc<TensorExpr> {
        match expression.as_ref() {
            TensorExpr::Input(_) | TensorExpr::Constant(_) | TensorExpr::ScalarConstant(_) => {
                Rc::clone(expression)
            }
            TensorExpr::BVJaggedPermute(permute) => {
                let source = self.rewrite(&permute.expr);
                Rc::new(compose(source, permute.indices.clone()).into())
            }
            TensorExpr::BVGather(gather) => {
                let source = self.rewrite(&gather.expr);
                let BVJaggedPermute { expr, indices } = compose(source, gather.indices_in.clone());
                Rc::new(BVGather { expr, indices_in: indices, ..gather.clone() }.into())
            }
            TensorExpr::BVGather2(gather) => Rc::new(
                BVGather2 {
                    a: self.rewrite(&gather.a),
                    b: self.rewrite(&gather.b),
                    ..gather.clone()
                }
                .into(),
            ),
            TensorExpr::BTShift(shift) => Rc::new(
                BTShift {
                    expr: self.rewrite(&shift.expr),
                    ..shift.clone()
                }
                .into(),
            ),
            TensorExpr::BTExpandReduce(change) => {
                let expr = self.rewrite(&change.expr);
                let node = BTExpandReduce { expr: Rc::clone(&expr), ..change.clone() };
                let form = self.types.infer(&expr).ok().and_then(|input| expand_reduce_form(&node, &input).ok());
                match form {
                    // A dense reshape is a gather; compose it through permutes.
                    Some(ExpandReduceForm::Dense(gather)) => {
                        let BVJaggedPermute { expr, indices } = compose(expr, gather.indices_in.clone());
                        Rc::new(BVGather { expr, indices_in: indices, ..gather }.into())
                    }
                    Some(ExpandReduceForm::LaneFold) => {
                        fuse_reduction(change.op, &expr).unwrap_or_else(|| Rc::new(node.into()))
                    }
                    None => Rc::new(node.into()),
                }
            }
            TensorExpr::BVLogical(logical) => {
                let mut operands = Vec::with_capacity(logical.operands.len());
                for operand in &logical.operands {
                    let expr = self.rewrite(&operand.expr);
                    if operand.reduce_dims.is_empty() {
                        // Nested logic is kept as a shared operand, not flattened.
                        operands.push(JTReduced::shared(expr, []));
                        continue;
                    }
                    match fuse_reduction(logical.op, &expr) {
                        Some(fused) => operands.push(JTReduced::shared(fused, [])),
                        None => operands.push(JTReduced::shared(expr, operand.reduce_dims.iter().copied())),
                    }
                }
                if let [single] = operands.as_slice() {
                    if single.reduce_dims.is_empty() {
                        return Rc::clone(&single.expr);
                    }
                }
                Rc::new(BVLogical::new(logical.op, operands).into())
            }
        }
    }
}

/// `permute(source, indices)`, composed through a single-lane permutation.
fn compose(source: Rc<TensorExpr>, indices: JaggedTensor<usize>) -> BVJaggedPermute {
    if let TensorExpr::BVJaggedPermute(inner) = source.as_ref() {
        let rows = inner.indices.row_count();
        if single_lane(&inner.indices) && indices.data().iter().all(|&index| index < rows) {
            let data = indices.data().iter().map(|&index| inner.indices.row(index)[0]).collect();
            let (_, tensor_dims, jagged_dims) = indices.into_parts();
            return BVJaggedPermute {
                expr: Rc::clone(&inner.expr),
                indices: JaggedTensor::new(data, tensor_dims, jagged_dims),
            };
        }
    }
    BVJaggedPermute { expr: source, indices }
}

/// Whether two permutations have the same row count and row lengths, so each
/// lane pairs one bit of each source.
fn same_rows(a: &JaggedTensor<usize>, b: &JaggedTensor<usize>) -> bool {
    a.row_count() == b.row_count() && (0..a.row_count()).all(|row| a.row(row).len() == b.row(row).len())
}

/// The fused node for `reduce(op, expression)`, if one applies.
fn fuse_reduction(op: crate::compile::LogicalOp, expression: &Rc<TensorExpr>) -> Option<Rc<TensorExpr>> {
    match expression.as_ref() {
        TensorExpr::BVJaggedPermute(permute) => Some(Rc::new(
            BVGather::from_rows(Rc::clone(&permute.expr), &permute.indices, op).into(),
        )),
        TensorExpr::BVLogical(inner) => match inner.operands.as_slice() {
            [left, right] if left.reduce_dims.is_empty() && right.reduce_dims.is_empty() => {
                match (left.expr.as_ref(), right.expr.as_ref()) {
                    (TensorExpr::BVJaggedPermute(a), TensorExpr::BVJaggedPermute(b)) if same_rows(&a.indices, &b.indices) => {
                        // Flatten to one element per (row, lane): destination
                        // row, bit of `a`, bit of `b`.
                        let rows = a.indices.row_count();
                        let mut out = Vec::new();
                        for row in 0..rows {
                            out.extend(std::iter::repeat_n(row, a.indices.row(row).len()));
                        }
                        Some(Rc::new(
                            BVGather2 {
                                a: Rc::clone(&a.expr),
                                b: Rc::clone(&b.expr),
                                indices_out: JaggedTensor::from_flat(out),
                                indices_1: JaggedTensor::from_flat(a.indices.data().to_vec()),
                                indices_2: JaggedTensor::from_flat(b.indices.data().to_vec()),
                                width: rows,
                                inner: inner.op,
                                op,
                            }
                            .into(),
                        ))
                    }
                    _ => None,
                }
            }
            _ => None,
        },
        _ => None,
    }
}

/// Apply every fusion rule to a traced function.
pub fn fuse(function: &TensorFn) -> TensorFn {
    let mut fuser = Fuser::default();
    let root = fuser.rewrite(&Rc::new(function.expression.clone()));
    TensorFn::new(function.name.clone(), function.inputs.clone(), Rc::unwrap_or_clone(root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::{Input, LogicalOp, TensorType};

    #[test]
    fn nested_logic_stays_a_chain_of_shared_nodes() {
        // ((a | b) | c), as produced by `x = a | b; x |= c`.
        let bits = TensorType::bitvec(8).unwrap();
        let input = |name: &str| JTReduced::new(Input::new(name, bits.clone()), []);
        let inner = Rc::new(TensorExpr::from(BVLogical::new(LogicalOp::Or, [input("a"), input("b")])));
        let outer = BVLogical::new(LogicalOp::Or, [JTReduced::shared(inner, []), input("c")]);
        let inputs = ["a", "b", "c"].map(|name| (name.to_string(), bits.clone())).to_vec();
        let fused = fuse(&TensorFn::new("chain", inputs, outer));
        let TensorExpr::BVLogical(root) = &fused.expression else { panic!("expected logic") };
        assert_eq!(root.operands.len(), 2, "not flattened into or(a, b, c)");
        let TensorExpr::BVLogical(nested) = root.operands[0].expr.as_ref() else {
            panic!("the first operand stays a nested logic node")
        };
        assert_eq!(nested.operands.len(), 2);
    }
}
