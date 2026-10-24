//! Packed-word circuit utilities.
//!
//! The permutation synthesizer moves selected bits between i64 registers by
//! composing masks, rotations, merges, and XOR operations. Generated circuits
//! are checked symbolically before they are returned.

use std::collections::HashMap;
use std::rc::Rc;

#[allow(dead_code)]
mod i6;
#[allow(dead_code)]
mod permute;
#[allow(dead_code)]
mod sorting;
#[allow(dead_code)]
pub mod verify;

use i6::{Binary, BinaryBackward, I6ShiftEncoding, Negabinary};
use permute::{permute_circuit, Expr, RegisterExpr};

/// One node of a flattened permutation circuit over a single `i64` word.
///
/// Operands are indices of earlier nodes. `Output` marks the register of one
/// permutation order; the orders write disjoint destination bits, so their
/// union is the permuted word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitExpr {
    Input,
    Empty,
    Masked(u64, usize),
    /// Rotate left by `1..=63` bits.
    Rotl(u32, usize),
    Merge(usize, usize),
    Xor(usize, usize),
    MaskedXor(u64, usize, usize),
    Output(usize),
}

fn flatten<E: I6ShiftEncoding>(registers: &[RegisterExpr]) -> Vec<CircuitExpr> {
    fn visit<E: I6ShiftEncoding>(
        expression: &Rc<Expr>,
        nodes: &mut Vec<CircuitExpr>,
        ids: &mut HashMap<*const Expr, usize>,
    ) -> usize {
        if let Some(&id) = ids.get(&Rc::as_ptr(expression)) {
            return id;
        }
        let node = match expression.as_ref() {
            Expr::Input => CircuitExpr::Input,
            Expr::Empty => CircuitExpr::Empty,
            Expr::Masked(mask, child) => CircuitExpr::Masked(*mask, visit::<E>(child, nodes, ids)),
            Expr::ShiftLevel(level, child) => {
                let amount: u8 = E::base(*level).into();
                CircuitExpr::Rotl(u32::from(amount) % 64, visit::<E>(child, nodes, ids))
            }
            Expr::Merge(left, right) => {
                CircuitExpr::Merge(visit::<E>(left, nodes, ids), visit::<E>(right, nodes, ids))
            }
            Expr::Xor(left, right) => {
                CircuitExpr::Xor(visit::<E>(left, nodes, ids), visit::<E>(right, nodes, ids))
            }
            Expr::MaskedXor(mask, left, right) => CircuitExpr::MaskedXor(
                *mask,
                visit::<E>(left, nodes, ids),
                visit::<E>(right, nodes, ids),
            ),
        };
        nodes.push(node);
        ids.insert(Rc::as_ptr(expression), nodes.len() - 1);
        nodes.len() - 1
    }

    let mut nodes = Vec::new();
    let mut ids = HashMap::new();
    for register in registers {
        let root = visit::<E>(register.shared_expression(), &mut nodes, &mut ids);
        nodes.push(CircuitExpr::Output(root));
    }
    nodes
}

/// Synthesize the smallest verified circuit for `(source, destination, order)`
/// bit moves within one word, over every supported shift encoding.
///
/// Panics if the generator rejects the mapping; callers that accept arbitrary
/// mappings should catch the panic and fall back to another lowering.
pub fn permute_words(mapping: Vec<(u8, u8, u32)>) -> Vec<CircuitExpr> {
    let candidates = [
        flatten::<Binary>(&permute_circuit::<Binary>(mapping.clone())),
        flatten::<BinaryBackward>(&permute_circuit::<BinaryBackward>(mapping.clone())),
        flatten::<Negabinary>(&permute_circuit::<Negabinary>(mapping)),
    ];
    candidates
        .into_iter()
        .min_by_key(Vec::len)
        .expect("at least one encoding")
}
