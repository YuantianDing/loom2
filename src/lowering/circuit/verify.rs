//! Symbolic verification for expressions produced by `permute`.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use super::i6::I6ShiftEncoding;

use super::permute::{Expr, RegisterExpr};

type SymbolicWord = [u64; 64];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerificationError {
    MaskedXorWithoutImmediateXor,
    OverlappingMerge {
        bit: u8,
    },
    OutputCount {
        expected: usize,
        actual: usize,
    },
    DuplicateDestination {
        order: u32,
        bit: u8,
    },
    IncorrectBit {
        order: u32,
        bit: u8,
        expected_source: Option<u8>,
        actual_sources: u64,
    },
}

impl fmt::Display for VerificationError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaskedXorWithoutImmediateXor => {
                out.write_str("MaskedXor must be a direct operand of Xor")
            }
            Self::OverlappingMerge { bit } => {
                write!(out, "Merge has two non-empty operands at output bit {bit}")
            }
            Self::OutputCount { expected, actual } => {
                write!(
                    out,
                    "expected {expected} output expressions, found {actual}"
                )
            }
            Self::DuplicateDestination { order, bit } => {
                write!(out, "order {order} assigns output bit {bit} more than once")
            }
            Self::IncorrectBit {
                order,
                bit,
                expected_source,
                actual_sources,
            } => {
                write!(out, "order {order}, output bit {bit}: expected ")?;
                match expected_source {
                    Some(source) => write!(out, "input bit {source}")?,
                    None => out.write_str("zero")?,
                }
                write!(out, ", found XOR source mask {actual_sources:#018x}")
            }
        }
    }
}

impl Error for VerificationError {}

/// Verify all output expressions against `(source, destination, order)` mappings.
///
/// Expressions are interpreted symbolically over GF(2). This proves the result
/// for every possible input word instead of testing a collection of input values.
pub(crate) fn verify_permutation<E: I6ShiftEncoding>(
    expressions: &[RegisterExpr],
    mapping: &[(u8, u8, u32)],
) -> Result<(), VerificationError> {
    let expected_count = mapping
        .iter()
        .map(|(_, _, order)| *order as usize + 1)
        .max()
        .unwrap_or(0);
    if expressions.len() != expected_count {
        return Err(VerificationError::OutputCount {
            expected: expected_count,
            actual: expressions.len(),
        });
    }

    let mut expected = vec![[None; 64]; expected_count];
    for &(source, destination, order) in mapping {
        let slot = &mut expected[order as usize][destination as usize];
        if slot.replace(source).is_some() {
            return Err(VerificationError::DuplicateDestination {
                order,
                bit: destination,
            });
        }
    }

    for (order, (expression, expected)) in expressions.iter().zip(expected).enumerate() {
        verify_expression::<E>(expression.expression(), order as u32, &expected)?;
    }
    Ok(())
}

pub(crate) fn verify_expression<E: I6ShiftEncoding>(
    expression: &Expr,
    order: u32,
    expected: &[Option<u8>; 64],
) -> Result<(), VerificationError> {
    validate_masked_xor_placement(expression)?;
    let actual = Evaluator::default().evaluate::<E>(expression)?;

    for bit in 0_u8..64 {
        let expected_source = expected[bit as usize];
        let expected_sources = expected_source.map_or(0, |source| 1_u64 << source);
        let actual_sources = actual[bit as usize];
        if actual_sources != expected_sources {
            return Err(VerificationError::IncorrectBit {
                order,
                bit,
                expected_source,
                actual_sources,
            });
        }
    }
    Ok(())
}

fn validate_masked_xor_placement(expression: &Expr) -> Result<(), VerificationError> {
    fn visit(
        expression: &Expr,
        is_direct_xor_operand: bool,
        seen: &mut HashSet<(*const Expr, bool)>,
    ) -> Result<(), VerificationError> {
        if !seen.insert((expression, is_direct_xor_operand)) {
            return Ok(());
        }
        match expression {
            Expr::Input | Expr::Empty => Ok(()),
            Expr::Masked(_, child) | Expr::ShiftLevel(_, child) => visit(child, false, seen),
            Expr::Merge(left, right) => {
                visit(left, false, seen)?;
                visit(right, false, seen)
            }
            Expr::Xor(left, right) => {
                visit(left, true, seen)?;
                visit(right, true, seen)
            }
            Expr::MaskedXor(_, left, right) => {
                if !is_direct_xor_operand {
                    return Err(VerificationError::MaskedXorWithoutImmediateXor);
                }
                visit(left, false, seen)?;
                visit(right, false, seen)
            }
        }
    }

    visit(expression, false, &mut HashSet::new())
}

#[derive(Default)]
struct Evaluator {
    cache: HashMap<*const Expr, SymbolicWord>,
}

impl Evaluator {
    fn evaluate<E: I6ShiftEncoding>(
        &mut self,
        expression: &Expr,
    ) -> Result<SymbolicWord, VerificationError> {
        let pointer = expression as *const Expr;
        if let Some(word) = self.cache.get(&pointer) {
            return Ok(*word);
        }

        let result = match expression {
            Expr::Input => std::array::from_fn(|bit| 1_u64 << bit),
            Expr::Empty => [0; 64],
            Expr::Masked(mask, child) => {
                let mut word = self.evaluate::<E>(child)?;
                for bit in 0..64 {
                    if mask & (1_u64 << bit) == 0 {
                        word[bit] = 0;
                    }
                }
                word
            }
            Expr::ShiftLevel(level, child) => {
                let word = self.evaluate::<E>(child)?;
                let offset: i8 = E::base(*level).into();
                let mut shifted = [0; 64];
                for bit in 0..64 {
                    let destination = (bit as i8 + offset).rem_euclid(64) as usize;
                    shifted[destination] = word[bit];
                }
                shifted
            }
            Expr::Merge(left, right) => {
                let left = self.evaluate::<E>(left)?;
                let right = self.evaluate::<E>(right)?;
                let mut merged = [0; 64];
                for bit in 0..64 {
                    if left[bit] != 0 && right[bit] != 0 {
                        return Err(VerificationError::OverlappingMerge { bit: bit as u8 });
                    }
                    merged[bit] = left[bit] | right[bit];
                }
                merged
            }
            Expr::Xor(left, right) => {
                let left = self.evaluate::<E>(left)?;
                let right = self.evaluate::<E>(right)?;
                std::array::from_fn(|bit| left[bit] ^ right[bit])
            }
            Expr::MaskedXor(mask, left, right) => {
                let left = self.evaluate::<E>(left)?;
                let right = self.evaluate::<E>(right)?;
                std::array::from_fn(|bit| {
                    if mask & (1_u64 << bit) != 0 {
                        left[bit] ^ right[bit]
                    } else {
                        0
                    }
                })
            }
        };

        self.cache.insert(pointer, result);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use super::super::i6::Negabinary;
    use super::*;

    #[test]
    fn verifies_the_immediate_xor_masked_xor_swap_pattern() {
        let left = Rc::new(Expr::Input);
        let right = Rc::new(Expr::ShiftLevel(0, left.clone()));
        let swap = Rc::new(Expr::MaskedXor(1, left.clone(), right));
        let expression = Expr::Xor(left, swap);
        let mut expected = std::array::from_fn(|bit| Some(bit as u8));
        expected[0] = Some(63);

        assert_eq!(
            verify_expression::<Negabinary>(&expression, 0, &expected),
            Ok(())
        );
    }

    #[test]
    fn rejects_masked_xor_without_an_immediate_xor() {
        let input = Rc::new(Expr::Input);
        let expression = Expr::MaskedXor(1, input.clone(), input);

        assert_eq!(
            validate_masked_xor_placement(&expression),
            Err(VerificationError::MaskedXorWithoutImmediateXor)
        );
    }

    #[test]
    fn rejects_overlapping_merge_operands() {
        let input = Rc::new(Expr::Input);
        let expression = Expr::Merge(input.clone(), input);

        assert_eq!(
            Evaluator::default().evaluate::<Negabinary>(&expression),
            Err(VerificationError::OverlappingMerge { bit: 0 })
        );
    }
}
