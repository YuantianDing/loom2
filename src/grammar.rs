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
///
/// Nonterminals of one `group` share a hash set: a value is kept only the
/// first time any of them produces it, so a group must only gather
/// nonterminals that denote the same values (such as the copies of one
/// nonterminal that pattern subtraction makes). By default every
/// nonterminal is its own group.
///
/// Candidates of a `solution` nonterminal are checked against the
/// constraint. When no nonterminal is marked, those in the start symbol's
/// group are.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Nonterminal {
    pub symbol: String,
    pub width: usize,
    pub group: String,
    pub solution: bool,
    pub productions: Vec<Production>,
}

/// A constructor application whose children name grammar nonterminals.
#[gen_stub_pyclass]
#[pyo3::pyclass(module = "loom2._loom2", frozen)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Production {
    pub constructor: String,
    pub children: Vec<String>,
    /// The constructor is commutative: `f(x, y) = f(y, x)` (binary only).
    pub commutative: bool,
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

        let mut group_widths = HashMap::<&str, (&str, usize)>::new();
        for nonterminal in &nonterminals {
            let (first, width) = *group_widths
                .entry(nonterminal.group.as_str())
                .or_insert((nonterminal.symbol.as_str(), nonterminal.width));
            if width != nonterminal.width {
                return Err(GrammarError::new(format!(
                    "nonterminals {first:?} ({width} bits) and {:?} ({} bits) share hash group {:?} but differ in width",
                    nonterminal.symbol, nonterminal.width, nonterminal.group
                )));
            }
        }

        if !nonterminals.iter().any(|nonterminal| nonterminal.solution) {
            let group = nonterminals.iter().find(|nonterminal| nonterminal.symbol == start).map(|start| start.group.clone());
            for nonterminal in &mut nonterminals {
                nonterminal.solution = Some(&nonterminal.group) == group.as_ref();
            }
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

    /// Whether only one orientation of a commutative `production` of
    /// `nonterminal` needs building: its children share a hash group, and
    /// `nonterminal` has the swapped production, which builds the other.
    pub fn symmetric(&self, nonterminal: &Nonterminal, production: &Production) -> bool {
        if !production.commutative {
            return false;
        }
        let group = |symbol: &str| self.nonterminals.iter().find(|other| other.symbol == symbol).map(|other| other.group.as_str());
        let (a, b) = (&production.children[0], &production.children[1]);
        group(a) == group(b)
            && nonterminal.productions.iter().any(|other| {
                other.constructor == production.constructor && other.children[..] == [b.clone(), a.clone()]
            })
    }

    /// Hash groups in order of first appearance.
    pub fn groups(&self) -> Vec<&str> {
        let mut groups: Vec<&str> = Vec::new();
        for nonterminal in &self.nonterminals {
            if !groups.contains(&nonterminal.group.as_str()) {
                groups.push(&nonterminal.group);
            }
        }
        groups
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
        if production.commutative && production.children.len() != 2 {
            return Err(GrammarError::new(format!(
                "commutative constructor {:?} has {} children, not 2",
                production.constructor,
                production.children.len()
            )));
        }
    }
    Ok(())
}

impl Nonterminal {
    pub fn new(symbol: impl Into<String>, width: usize, productions: Vec<Production>) -> Self {
        let symbol = symbol.into();
        Self {
            group: symbol.clone(),
            symbol,
            width,
            solution: false,
            productions,
        }
    }

    /// The same nonterminal, whose candidates are (or are not) checked
    /// against the constraint.
    pub fn with_solution(self, solution: bool) -> Self {
        Self { solution, ..self }
    }

    /// The same nonterminal in hash group `group`.
    pub fn with_group(self, group: impl Into<String>) -> Self {
        Self { group: group.into(), ..self }
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
            commutative: false,
            token: 0,
            rule: 0,
        }
    }

    /// The same production, with a commutative (or not) constructor.
    pub fn with_commutative(self, commutative: bool) -> Self {
        Self { commutative, ..self }
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
    fn hash_groups_share_one_width_and_solutions_default_to_the_start_group() {
        let leaf = || vec![Production::new("constant", Vec::<String>::new())];
        let grammar = Grammar::new(
            "S",
            vec![
                Nonterminal::new("S", 4, leaf()),
                Nonterminal::new("S#1", 4, leaf()).with_group("S"),
                Nonterminal::new("T", 4, leaf()),
            ],
        )
        .unwrap();
        assert_eq!(grammar.groups(), ["S", "T"]);
        let solutions: Vec<bool> = grammar.nonterminals().iter().map(|nonterminal| nonterminal.solution).collect();
        assert_eq!(solutions, [true, true, false]);
        let marked = Grammar::new(
            "S",
            vec![Nonterminal::new("S", 4, leaf()), Nonterminal::new("S#1", 4, leaf()).with_solution(true)],
        )
        .unwrap();
        let solutions: Vec<bool> = marked.nonterminals().iter().map(|nonterminal| nonterminal.solution).collect();
        assert_eq!(solutions, [false, true]);
        let mixed = Grammar::new(
            "S",
            vec![Nonterminal::new("S", 4, leaf()), Nonterminal::new("U", 5, leaf()).with_group("S")],
        );
        assert!(mixed.unwrap_err().to_string().contains("differ in width"));
    }

    #[test]
    fn commutativity_needs_both_orientations_in_one_group() {
        let leaf = || Production::new("constant", Vec::<String>::new());
        let or = |a: &str, b: &str| Production::new("or", [a, b]).with_commutative(true);
        let grammar = Grammar::new(
            "S",
            vec![
                Nonterminal::new("S", 4, vec![leaf(), or("S", "S"), or("S", "T"), or("T", "S"), or("S", "U")]),
                Nonterminal::new("T", 4, vec![leaf()]).with_group("S"),
                Nonterminal::new("U", 4, vec![leaf()]),
            ],
        )
        .unwrap();
        let start = &grammar.nonterminals()[0];
        let symmetric: Vec<bool> = start.productions.iter().map(|production| grammar.symmetric(start, production)).collect();
        assert_eq!(symmetric, [false, true, true, true, false]);
        let unary = Grammar::new("S", vec![Nonterminal::new("S", 4, vec![leaf(), Production::new("not", ["S"]).with_commutative(true)])]);
        assert!(unary.unwrap_err().to_string().contains("commutative"));
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
