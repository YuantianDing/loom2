//! Typed grammar definitions for syntax-guided synthesis.
//!
//! A grammar names one start nonterminal and a collection of typed production
//! rules. Construction validates references and assigns stable numeric tokens
//! for the eventual packed synthesis representation.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;

use pyo3_stub_gen::derive::gen_stub_pyclass;

/// An invalid synthesis grammar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GrammarError(String);

impl GrammarError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for GrammarError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        out.write_str(&self.0)
    }
}

impl Error for GrammarError {}

/// Typed synthesis grammar consumed by executable lowering.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Grammar {
    start: String,
    nonterminals: Vec<Nonterminal>,
}

/// One grammar nonterminal and the bit width of values it denotes.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Nonterminal {
    pub symbol: String,
    pub width: usize,
    pub productions: Vec<Production>,
}

/// A constructor application whose children name grammar nonterminals.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Production {
    pub constructor: String,
    pub children: Vec<String>,
    token: usize,
    rule: usize,
}

impl Grammar {
    /// Validate and construct a grammar.
    pub fn new(
        start: impl Into<String>,
        mut nonterminals: Vec<Nonterminal>,
    ) -> Result<Self, GrammarError> {
        let start = start.into();
        if nonterminals.is_empty() {
            return Err(GrammarError::new("synthesis grammar is empty"));
        }

        let symbols = nonterminals
            .iter()
            .map(|nonterminal| nonterminal.symbol.clone())
            .collect::<HashSet<_>>();
        if symbols.len() != nonterminals.len() {
            return Err(GrammarError::new(
                "synthesis grammar contains duplicate nonterminal names",
            ));
        }
        if !symbols.contains(&start) {
            return Err(GrammarError::new(format!(
                "synthesis grammar has no start nonterminal {start:?}"
            )));
        }

        let mut tokens = HashMap::<String, usize>::new();
        let mut next_token = 1usize;
        let mut next_rule = 0usize;
        for nonterminal in &mut nonterminals {
            validate_nonterminal(nonterminal, &symbols)?;
            for production in &mut nonterminal.productions {
                production.token =
                    *tokens
                        .entry(production.constructor.clone())
                        .or_insert_with(|| {
                            let token = next_token;
                            next_token += 1;
                            token
                        });
                production.rule = next_rule;
                next_rule += 1;
            }
        }

        Ok(Self {
            start,
            nonterminals,
        })
    }

    /// Name of the grammar entry point.
    pub fn start(&self) -> &str {
        &self.start
    }

    /// Nonterminals in declaration order.
    pub fn nonterminals(&self) -> &[Nonterminal] {
        &self.nonterminals
    }

    /// Number of distinct constructor tokens.
    pub fn constructor_count(&self) -> usize {
        self.nonterminals
            .iter()
            .flat_map(|nonterminal| &nonterminal.productions)
            .map(|production| production.token)
            .collect::<HashSet<_>>()
            .len()
    }
}

fn validate_nonterminal(
    nonterminal: &Nonterminal,
    symbols: &HashSet<String>,
) -> Result<(), GrammarError> {
    if nonterminal.symbol.is_empty() {
        return Err(GrammarError::new(
            "grammar nonterminal names must not be empty",
        ));
    }
    if nonterminal.width == 0 {
        return Err(GrammarError::new(format!(
            "grammar nonterminal {:?} has zero width",
            nonterminal.symbol
        )));
    }
    if nonterminal.productions.is_empty() {
        return Err(GrammarError::new(format!(
            "grammar nonterminal {:?} has no productions",
            nonterminal.symbol
        )));
    }

    for production in &nonterminal.productions {
        if production.constructor.is_empty() {
            return Err(GrammarError::new(
                "grammar constructor names must not be empty",
            ));
        }
        for child in &production.children {
            if !symbols.contains(child) {
                return Err(GrammarError::new(format!(
                    "grammar constructor {:?} references unknown nonterminal {child:?}",
                    production.constructor
                )));
            }
        }
    }
    Ok(())
}

impl Nonterminal {
    pub fn new(symbol: impl Into<String>, width: usize, productions: Vec<Production>) -> Self {
        Self {
            symbol: symbol.into(),
            width,
            productions,
        }
    }
}

impl Production {
    pub fn new(
        constructor: impl Into<String>,
        children: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            constructor: constructor.into(),
            children: children.into_iter().map(Into::into).collect(),
            token: 0,
            rule: 0,
        }
    }

    /// Nonzero token used by packed expression encodings.
    pub fn token(&self) -> usize {
        self.token
    }

    /// Stable rule index in declaration order.
    pub fn rule(&self) -> usize {
        self.rule
    }
}

/// Enumerate positive child sizes that sum to one expression size.
///
/// This helper is used by size-ordered synthesis enumeration.
pub fn positive_compositions(total: usize, parts: usize) -> Vec<Vec<usize>> {
    if parts == 0 {
        return if total == 0 {
            vec![Vec::new()]
        } else {
            Vec::new()
        };
    }
    if total < parts {
        return Vec::new();
    }
    let mut result = Vec::new();
    compose(total, parts, &mut Vec::new(), &mut result);
    result
}

fn compose(total: usize, parts: usize, prefix: &mut Vec<usize>, result: &mut Vec<Vec<usize>>) {
    if parts == 1 {
        prefix.push(total);
        result.push(prefix.clone());
        prefix.pop();
        return;
    }
    for head in 1..=(total - (parts - 1)) {
        prefix.push(head);
        compose(total - head, parts - 1, prefix, result);
        prefix.pop();
    }
}

/// Bits required for a nonzero constructor token plus zero padding.
pub fn token_width(constructor_count: usize) -> Result<usize, GrammarError> {
    if constructor_count == 0 {
        return Err(GrammarError::new("synthesis grammar has no productions"));
    }
    Ok((usize::BITS - constructor_count.leading_zeros()) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_numbers_a_typed_grammar() {
        let grammar = Grammar::new(
            "Start",
            vec![Nonterminal::new(
                "Start",
                4,
                vec![
                    Production::new("constant", Vec::<String>::new()),
                    Production::new("pair", ["Start", "Start"]),
                ],
            )],
        )
        .unwrap();

        assert_eq!(grammar.start(), "Start");
        assert_eq!(grammar.nonterminals()[0].productions[0].token(), 1);
        assert_eq!(grammar.nonterminals()[0].productions[1].token(), 2);
    }

    #[test]
    fn partitions_children_by_exact_expression_size() {
        assert_eq!(positive_compositions(0, 0), vec![Vec::<usize>::new()]);
        assert!(positive_compositions(1, 2).is_empty());
        assert_eq!(
            positive_compositions(4, 2),
            vec![vec![1, 3], vec![2, 2], vec![3, 1]]
        );
    }

    #[test]
    fn encodes_all_nonzero_constructor_tokens() {
        assert_eq!(token_width(1).unwrap(), 1);
        assert_eq!(token_width(2).unwrap(), 2);
        assert_eq!(token_width(3).unwrap(), 2);
        assert_eq!(token_width(4).unwrap(), 3);
    }
}
