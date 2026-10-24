use std::{collections::HashMap, rc::Rc};

use smallvec::smallvec;
use smallvec::SmallVec;

use super::i6::{i6, I6ShiftEncoding};
use super::sorting::sorting_circuits;
use super::verify::verify_expression;

#[derive(Debug)]
pub enum Expr {
    Input,
    Empty,
    Masked(u64, Rc<Expr>),
    ShiftLevel(u8, Rc<Expr>),
    Merge(Rc<Expr>, Rc<Expr>),
    Xor(Rc<Expr>, Rc<Expr>),
    MaskedXor(u64, Rc<Expr>, Rc<Expr>),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct IndexRecord {
    cur: i6,
    src: i6,
    action: i6,
    order: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Register(HashMap<i6, (i6, SmallVec<[(i6, u32); 1]>)>);

impl FromIterator<IndexRecord> for Register {
    fn from_iter<T: IntoIterator<Item = IndexRecord>>(iter: T) -> Self {
        let mut map = HashMap::new();
        for record in iter {
            let (src, li) = map
                .entry(record.cur)
                .or_insert_with(|| (record.src, SmallVec::new()));
            assert_eq!(*src, record.src);
            li.push((record.action, record.order));
        }
        Register(map)
    }
}

impl Register {
    fn new() -> Self {
        Register(HashMap::new())
    }
    fn into_iter(self) -> impl Iterator<Item = IndexRecord> {
        self.0.into_iter().flat_map(|(cur, (src, li))| {
            li.into_iter().map(move |(action, order)| IndexRecord {
                cur,
                src,
                action,
                order,
            })
        })
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    fn add(&mut self, record: IndexRecord) {
        assert!(self.try_add(record));
    }
    fn try_add(&mut self, record: IndexRecord) -> bool {
        if let Some((src, li)) = self.0.get_mut(&record.cur) {
            if *src == record.src {
                li.push((record.action, record.order));
                true
            } else {
                false
            }
        } else {
            self.0.insert(
                record.cur,
                (record.src, smallvec![(record.action, record.order)]),
            );
            true
        }
    }
    fn is_disjoint(&self, other: &Register) -> bool {
        self.0.keys().all(|k| !other.0.contains_key(k))
    }
    fn retain(&mut self, mut f: impl FnMut(&IndexRecord) -> bool) {
        self.0.retain(|&cur, (src, li)| {
            li.retain(|&mut (action, order)| {
                f(&IndexRecord {
                    cur,
                    src: *src,
                    action,
                    order,
                })
            });
            !li.is_empty()
        });
    }
    fn iter<'a>(&'a self) -> impl Iterator<Item = IndexRecord> + 'a {
        self.0.iter().flat_map(|(&cur, (src, li))| {
            li.iter().map(move |&(action, order)| IndexRecord {
                cur,
                src: *src,
                action,
                order,
            })
        })
    }
    fn try_shift_level<E: I6ShiftEncoding>(self, level: u8) -> (Register, Register) {
        let mut orig = Register::new();
        let mut shifted = Register::new();
        for record in self.into_iter() {
            let (should_shift, new_cur) = E::level_perform(level, record.action, record.cur);
            if should_shift {
                shifted.add(IndexRecord {
                    cur: new_cur,
                    ..record
                });
            } else {
                orig.add(record);
            }
        }
        (orig, shifted)
    }

    fn get_mask(&self) -> u64 {
        let mut mask = 0;
        for record in self.iter() {
            mask |= 1_u64 << Into::<u8>::into(record.cur);
        }
        mask
    }

    fn stack(&mut self, other: &mut Register) -> u64 {
        let mut mask = 0;
        other.retain(|record| {
            if self.try_add(record.clone()) {
                mask |= 1_u64 << Into::<u8>::into(record.cur);
                false
            } else {
                true
            }
        });
        mask
    }

    fn sort2(&self, other: &Register) -> (Register, Register, u64) {
        let mut final_self = Register::new();
        let mut final_other = Register::new();

        let mut swapped: u64 = 0;
        for i in 0_u8..64 {
            let cur = i.into();
            let a = self.0.get(&cur).map(|(src, li)| {
                assert!(li.len() == 1);
                (src, li[0].0, li[0].1)
            });

            let b = other.0.get(&cur).map(|(src, li)| {
                assert!(li.len() == 1);
                (src, li[0].0, li[0].1)
            });

            match (a, b) {
                (Some((src_a, dst_a, order_a)), Some((src_b, dst_b, order_b))) => {
                    if order_b < order_a {
                        final_self.add(IndexRecord {
                            cur,
                            src: *src_b,
                            action: dst_b,
                            order: order_b,
                        });
                        final_other.add(IndexRecord {
                            cur,
                            src: *src_a,
                            action: dst_a,
                            order: order_a,
                        });
                        swapped |= 1_u64 << i;
                    } else {
                        final_other.add(IndexRecord {
                            cur,
                            src: *src_b,
                            action: dst_b,
                            order: order_b,
                        });
                        final_self.add(IndexRecord {
                            cur,
                            src: *src_a,
                            action: dst_a,
                            order: order_a,
                        });
                    }
                }
                (None, Some((src_b, dst_b, order_b))) => {
                    final_self.add(IndexRecord {
                        cur,
                        src: *src_b,
                        action: dst_b,
                        order: order_b,
                    });
                    swapped |= 1_u64 << i;
                }
                (Some((src_a, dst_a, order_a)), None) => {
                    final_self.add(IndexRecord {
                        cur,
                        src: *src_a,
                        action: dst_a,
                        order: order_a,
                    });
                }
                _ => {}
            }
        }
        (final_self, final_other, swapped)
    }

    fn dedup(&self) -> Vec<Register> {
        let mut result: Vec<Register> = Vec::new();
        for record in self.iter() {
            let mut flag = false;
            for reg in result.iter_mut() {
                if !reg.0.contains_key(&record.cur) {
                    reg.add(record.clone());
                    flag = true;
                    break;
                }
            }
            if !flag {
                let mut new_reg = Register(HashMap::new());
                new_reg.add(record);
                result.push(new_reg);
            }
        }
        result
    }
}

#[derive(Debug, Clone)]
pub struct RegisterExpr {
    reg: Register,
    expr: Rc<Expr>,
    unmasked: bool,
}

impl RegisterExpr {
    pub(crate) fn expression(&self) -> &Expr {
        &self.expr
    }
    pub(crate) fn shared_expression(&self) -> &Rc<Expr> {
        &self.expr
    }
    fn input(reg: Register) -> Self {
        Self {
            reg,
            expr: Rc::new(Expr::Input),
            unmasked: false,
        }
    }
    fn verify<E: I6ShiftEncoding>(&self) -> bool {
        let mut self0 = self.clone();
        self0.make_masked();
        let mut semantics = [None; 64];
        for record in self0.reg.iter() {
            let dst: u8 = record.cur.into();
            if semantics[dst as usize]
                .map(|src| src != Into::<u8>::into(record.src))
                .unwrap_or(false)
            {
                return false;
            }
            semantics[dst as usize] = Some(record.src.into());
        }
        verify_expression::<E>(self0.expr.as_ref(), 0, &semantics).is_ok()
    }

    fn try_shift_level<E: I6ShiftEncoding>(self, level: u8) -> (Option<Self>, Option<Self>) {
        let (orig, shifted) = self.reg.try_shift_level::<E>(level);

        if orig.is_empty() {
            (
                None,
                Some(Self {
                    reg: shifted,
                    expr: Expr::ShiftLevel(level, self.expr).into(),
                    unmasked: self.unmasked,
                }),
            )
        } else if shifted.is_empty() {
            (
                Some(Self {
                    reg: orig,
                    expr: self.expr,
                    unmasked: self.unmasked,
                }),
                None,
            )
        } else {
            (
                Some(Self {
                    expr: self.expr.clone(),
                    reg: orig,
                    unmasked: true,
                }),
                Some(Self {
                    expr: Expr::ShiftLevel(level, self.expr).into(),
                    reg: shifted,
                    unmasked: true,
                }),
            )
        }
    }

    fn make_masked(&mut self) {
        if self.unmasked {
            let mask = self.reg.get_mask();
            self.expr = Rc::new(Expr::Masked(mask, self.expr.clone()));
            self.unmasked = false;
        }
    }

    fn sort2(&mut self, other: &mut Self) -> u64 {
        if self.reg.0.len() < other.reg.0.len() {
            std::mem::swap(self, other);
        }

        let (new_self, new_other, swapped) = self.reg.sort2(&other.reg);
        if swapped == 0 {
            return 0;
        }
        if other.reg.sort2(&self.reg).2 == 0 {
            std::mem::swap(self, other);
            return 0;
        }

        self.reg = new_self;
        other.reg = new_other;

        let self_orig = self.expr.clone();
        let other_orig = other.expr.clone();

        let xor = Rc::new(Expr::MaskedXor(
            swapped,
            self_orig.clone(),
            other_orig.clone(),
        ));
        self.expr = Rc::new(Expr::Xor(self_orig, xor.clone()));
        other.expr = Rc::new(Expr::Xor(other_orig, xor));

        swapped
    }
    fn stack(&mut self, other: &mut Self) -> bool {
        if self.reg.0.len() < other.reg.0.len() {
            std::mem::swap(self, other);
        }
        let self_orig = self.expr.clone();
        let other_orig = other.expr.clone();
        let mask = self.reg.stack(&mut other.reg);
        if mask == 0 {
            return false;
        } else if other.reg.is_empty() {
            self.expr = Rc::new(Expr::Merge(self_orig, other_orig));
            other.expr = Rc::new(Expr::Empty);
            return true;
        } else {
            let other_masked = Rc::new(Expr::Masked(mask, other_orig.clone()));
            self.expr = Rc::new(Expr::Merge(self_orig, other_masked));
            other.expr = Rc::new(Expr::Masked(!mask, other_orig));
            return true;
        }
    }

    fn dedup(&self) -> Vec<Self> {
        let initial_mask = self.reg.get_mask();
        let regs = self.reg.dedup();
        let mut result = Vec::new();
        for reg in regs {
            if reg.get_mask() == initial_mask {
                result.push(Self {
                    expr: self.expr.clone(),
                    reg,
                    unmasked: self.unmasked,
                });
            } else {
                result.push(Self {
                    expr: self.expr.clone(),
                    reg,
                    unmasked: true,
                });
            }
        }
        result
    }

    fn merge(&mut self, other: &mut Self) -> bool {
        if self.reg.is_disjoint(&other.reg) {
            self.make_masked();
            other.make_masked();
            let self_orig = self.expr.clone();
            let other_orig = other.expr.clone();
            self.reg.0.extend(other.reg.0.drain());
            self.expr = Rc::new(Expr::Merge(self_orig, other_orig));
            other.expr = Rc::new(Expr::Empty);
            return true;
        }
        false
    }
}

fn step<E: I6ShiftEncoding>(level: u8, result: Vec<RegisterExpr>) -> Vec<RegisterExpr> {
    let mut splitted_result = result
        .into_iter()
        .flat_map(|re| {
            let (orig, shifted) = re.try_shift_level::<E>(level);
            orig.into_iter().chain(shifted.into_iter())
        })
        .collect::<Vec<_>>();
    assert!(
        splitted_result.iter().all(|re| re.verify::<E>()),
        "invalid expression after shift level {level}"
    );

    splitted_result.sort_by_key(|re| usize::MAX - re.reg.0.len());
    try_all_merge(&mut splitted_result);
    splitted_result.sort_by_key(|re| usize::MAX - re.reg.0.len());

    while splitted_result.last().map_or(false, |a| a.reg.is_empty()) {
        splitted_result.pop();
    }

    splitted_result
}

fn expected_result<E: I6ShiftEncoding>(mapping: &[(u8, u8, u32)]) -> Vec<Register> {
    let mut result = Vec::new();
    for (src, dst, order) in mapping {
        while result.len() <= *order as usize {
            result.push(Register::new());
        }
        let action = E::encode(i6::from(*dst) - i6::from(*src));
        let record = IndexRecord {
            cur: (*dst).into(),
            src: (*src).into(),
            action,
            order: *order,
        };
        result[*order as usize].add(record);
    }
    result
}

fn try_all_merge(registers: &mut Vec<RegisterExpr>) {
    let mut result = Vec::new();
    while let Some(mut a) = registers.pop() {
        for b in registers.iter_mut() {
            if b.merge(&mut a) {
                break;
            }
        }
        if !a.reg.is_empty() {
            result.push(a);
        }
    }
    *registers = result;
}

fn permute_circuit_impl<E: I6ShiftEncoding>(
    mapping: Vec<(u8, u8, u32)>,
    mut trace: impl FnMut(&str, &[RegisterExpr]),
) -> Vec<RegisterExpr> {
    let expected = expected_result::<E>(&mapping);
    let reg = mapping
        .iter()
        .copied()
        .map(|(src, dst, order)| {
            let action = E::encode(i6::from(dst) - i6::from(src));
            IndexRecord {
                cur: src.into(),
                src: src.into(),
                action,
                order,
            }
        })
        .collect();

    let reg_expr = RegisterExpr::input(reg);

    let mut result: Vec<_> = vec![reg_expr];
    trace("initial", &result);

    for level in 0..6 {
        result = step::<E>(level, result);
        trace(&format!("shift level {level}"), &result);
        assert!(
            result.iter().all(|re| re.verify::<E>()),
            "invalid expression after shift level {level}"
        );
    }

    result = result.into_iter().flat_map(|re| re.dedup()).collect();
    result.sort_by_key(|re| re.reg.iter().map(|record| record.order).sum::<u32>());
    try_all_merge(&mut result);

    for re in result.iter_mut() {
        re.make_masked();
    }

    trace("deduplicated", &result);
    result.sort_by_key(|re| !re.reg.0.len());
    for (i, j) in sorting_circuits(result.len()) {
        let [a, b] = result.get_disjoint_mut([i, j]).expect("invalid indices");
        a.sort2(b);
    }

    while result.last().map_or(false, |a| a.reg.is_empty()) {
        result.pop();
    }

    trace("sorted", &result);

    let result_reg = result.iter().map(|re| re.reg.clone()).collect::<Vec<_>>();

    assert_eq!(result_reg, expected);
    super::verify::verify_permutation::<E>(&result, &mapping)
        .unwrap_or_else(|error| panic!("generated an invalid permutation expression: {error}"));

    result
}

pub fn permute_circuit<E: I6ShiftEncoding>(mapping: Vec<(u8, u8, u32)>) -> Vec<RegisterExpr> {
    permute_circuit_impl::<E>(mapping, |_, _| {})
}

#[cfg(test)]
mod tests {
    use super::super::i6::{Binary, BinaryBackward, Negabinary};
    use super::*;
    use std::collections::HashSet;

    fn i6_as_u8(value: i6) -> u8 {
        value.into()
    }

    fn expr_summary(expr: &Rc<Expr>) -> String {
        fn visit(expr: &Rc<Expr>, seen: &mut HashSet<*const Expr>, depth: usize) -> usize {
            if !seen.insert(Rc::as_ptr(expr)) {
                return depth;
            }
            match expr.as_ref() {
                Expr::Input | Expr::Empty => depth,
                Expr::Masked(_, child) | Expr::ShiftLevel(_, child) => {
                    visit(child, seen, depth + 1)
                }
                Expr::Merge(left, right)
                | Expr::Xor(left, right)
                | Expr::MaskedXor(_, left, right) => {
                    visit(left, seen, depth + 1).max(visit(right, seen, depth + 1))
                }
            }
        }

        let root = match expr.as_ref() {
            Expr::Input => "Input".to_owned(),
            Expr::Empty => "Empty".to_owned(),
            Expr::Masked(mask, _) => format!("Masked({mask:#018x})"),
            Expr::ShiftLevel(level, _) => format!("ShiftLevel({level})"),
            Expr::Merge(_, _) => "Merge".to_owned(),
            Expr::Xor(_, _) => "Xor".to_owned(),
            Expr::MaskedXor(mask, _, _) => format!("MaskedXor({mask:#018x})"),
        };
        let mut seen = HashSet::new();
        let depth = visit(expr, &mut seen, 1);
        format!("{root}, {} unique node(s), depth {depth}", seen.len())
    }

    fn print_step(label: &str, registers: &[RegisterExpr]) {
        println!("\n=== {label} ({} register(s)) ===", registers.len());
        for (index, register) in registers.iter().enumerate() {
            let mut records = register.reg.iter().collect::<Vec<_>>();
            records.sort_by_key(|record| {
                (
                    record.order,
                    i6_as_u8(record.cur),
                    i6_as_u8(record.src),
                    i6_as_u8(record.action),
                )
            });
            println!("register {index}: mask = {:#018x}", register.reg.get_mask());
            for record in records {
                println!(
                    "  order {}: src {:02} -> cur {:02}, action {:#08b}",
                    record.order,
                    i6_as_u8(record.src),
                    i6_as_u8(record.cur),
                    i6_as_u8(record.action),
                );
            }
            println!("  expr: {}", expr_summary(&register.expr));
        }
    }

    fn eval(expr: &Expr, input: u64) -> u64 {
        match expr {
            Expr::Input => input,
            Expr::Empty => 0,
            Expr::Masked(mask, value) => mask & eval(value, input),
            Expr::ShiftLevel(level, value) => {
                let offset: i8 = Negabinary::base(*level).into();
                let value = eval(value, input);
                if offset >= 0 {
                    value.rotate_left(offset as u32)
                } else {
                    value.rotate_right((-offset) as u32)
                }
            }
            Expr::Merge(left, right) => eval(left, input) | eval(right, input),
            Expr::Xor(left, right) => eval(left, input) ^ eval(right, input),
            Expr::MaskedXor(mask, left, right) => mask & (eval(left, input) ^ eval(right, input)),
        }
    }

    fn expected_output(mapping: &[(u8, u8, u32)], order: u32, input: u64) -> u64 {
        mapping
            .iter()
            .filter(|(_, _, candidate_order)| *candidate_order == order)
            .fold(0, |output, (src, dst, _)| {
                output | (((input >> src) & 1) << dst)
            })
    }

    #[test]
    fn permute_circuit_traces_and_evaluates_every_stage() {
        // Three independent permutations exercise duplicated source bits,
        // collisions between intermediate positions, wraparound, every shift
        // level, deduplication, and the final sorting network.
        let positions = [0, 1, 2, 31, 32, 63];
        let mut mapping = Vec::new();
        for (order, destinations) in [positions, [1, 2, 31, 32, 63, 0], [63, 32, 31, 2, 1, 0]]
            .into_iter()
            .enumerate()
        {
            mapping.extend(
                positions
                    .into_iter()
                    .zip(destinations)
                    .map(|(src, dst)| (src, dst, order as u32)),
            );
        }

        println!("mapping (src -> dst @ order): {mapping:?}");
        let result = permute_circuit_impl::<Negabinary>(mapping.clone(), print_step);

        assert_eq!(result.len(), 3, "one output expression per order");
        for input in [0, u64::MAX, 0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210]
            .into_iter()
            .chain((0..64).map(|bit| 1_u64 << bit))
        {
            for (order, register) in result.iter().enumerate() {
                assert_eq!(
                    eval(&register.expr, input),
                    expected_output(&mapping, order as u32, input),
                    "incorrect output for order {order} and input {input:#018x}",
                );
            }
        }
    }

    #[test]
    fn binary_encodings_generate_verified_permutations() {
        let mapping = (0_u8..64)
            .map(|source| (source, source.wrapping_add(37) % 64, 0))
            .collect::<Vec<_>>();

        assert_eq!(permute_circuit::<Binary>(mapping.clone()).len(), 1);
        assert_eq!(permute_circuit::<BinaryBackward>(mapping).len(), 1);
    }

    #[test]
    fn permute_circuit_symbolically_verifies_test_mapping() {
        fn operation_counts(registers: &[RegisterExpr]) -> [usize; 5] {
            fn visit(
                expression: &Rc<Expr>,
                seen: &mut HashSet<*const Expr>,
                counts: &mut [usize; 5],
            ) {
                if !seen.insert(Rc::as_ptr(expression)) {
                    return;
                }

                match expression.as_ref() {
                    Expr::Input | Expr::Empty => {}
                    Expr::Masked(_, child) => {
                        counts[0] += 1;
                        visit(child, seen, counts);
                    }
                    Expr::ShiftLevel(_, child) => {
                        counts[1] += 1;
                        visit(child, seen, counts);
                    }
                    Expr::Merge(left, right) => {
                        counts[2] += 1;
                        visit(left, seen, counts);
                        visit(right, seen, counts);
                    }
                    Expr::Xor(left, right) => {
                        counts[3] += 1;
                        visit(left, seen, counts);
                        visit(right, seen, counts);
                    }
                    Expr::MaskedXor(_, left, right) => {
                        counts[4] += 2;
                        visit(left, seen, counts);
                        visit(right, seen, counts);
                    }
                }
            }

            let mut seen = HashSet::new();
            let mut counts = [0; 5];
            for register in registers {
                visit(&register.expr, &mut seen, &mut counts);
            }
            counts
        }

        let expected_orders = TEST
            .iter()
            .map(|(_, _, order)| *order as usize + 1)
            .max()
            .unwrap_or(0);

        fn generate<E: I6ShiftEncoding>(encoding: &str) -> Vec<RegisterExpr> {
            permute_circuit_impl::<E>(TEST.to_vec(), |stage, registers| {
                println!(
                    "{encoding:>15} | {stage:>13}: {} registers, {} operations",
                    registers.len(),
                    operation_counts(registers).iter().sum::<usize>(),
                );
            })
        }

        for (encoding, result) in [
            ("negabinary", generate::<Negabinary>("negabinary")),
            ("binary", generate::<Binary>("binary")),
            (
                "binary backward",
                generate::<BinaryBackward>("binary backward"),
            ),
        ] {
            assert_eq!(
                result.len(),
                expected_orders,
                "{encoding} produced an unexpected number of output expressions",
            );
            let operation_counts = operation_counts(&result);
            println!(
                "{encoding:>15} | total operations: {}",
                operation_counts.iter().sum::<usize>()
            );
            if encoding == "binary" {
                println!(
                    "         binary | Masked: {}, ShiftLevel: {}, Merge: {}, Xor: {}, MaskedXor: {}",
                    operation_counts[0],
                    operation_counts[1],
                    operation_counts[2],
                    operation_counts[3],
                    operation_counts[4],
                );
            }
        }
    }

    const TEST: [(u8, u8, u32); 692] = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 1, 1),
        (0, 2, 0),
        (1, 2, 1),
        (2, 2, 2),
        (0, 3, 0),
        (1, 3, 1),
        (2, 3, 2),
        (3, 3, 3),
        (0, 4, 0),
        (1, 4, 1),
        (2, 4, 2),
        (3, 4, 3),
        (4, 4, 4),
        (0, 5, 0),
        (1, 5, 1),
        (2, 5, 2),
        (3, 5, 3),
        (4, 5, 4),
        (5, 5, 5),
        (0, 6, 0),
        (1, 6, 1),
        (2, 6, 2),
        (3, 6, 3),
        (4, 6, 4),
        (5, 6, 5),
        (6, 6, 6),
        (0, 7, 0),
        (1, 7, 1),
        (2, 7, 2),
        (3, 7, 3),
        (4, 7, 4),
        (5, 7, 5),
        (6, 7, 6),
        (7, 7, 7),
        (0, 8, 0),
        (1, 8, 1),
        (2, 8, 2),
        (3, 8, 3),
        (4, 8, 4),
        (5, 8, 5),
        (6, 8, 6),
        (7, 8, 7),
        (8, 8, 8),
        (0, 9, 0),
        (1, 9, 1),
        (2, 9, 2),
        (3, 9, 3),
        (4, 9, 4),
        (5, 9, 5),
        (6, 9, 6),
        (7, 9, 7),
        (8, 9, 8),
        (9, 9, 9),
        (0, 10, 0),
        (1, 10, 1),
        (2, 10, 2),
        (3, 10, 3),
        (4, 10, 4),
        (5, 10, 5),
        (6, 10, 6),
        (10, 10, 7),
        (0, 11, 0),
        (1, 11, 1),
        (2, 11, 2),
        (3, 11, 3),
        (4, 11, 4),
        (5, 11, 5),
        (6, 11, 6),
        (10, 11, 7),
        (11, 11, 8),
        (0, 12, 0),
        (1, 12, 1),
        (2, 12, 2),
        (3, 12, 3),
        (4, 12, 4),
        (5, 12, 5),
        (6, 12, 6),
        (10, 12, 7),
        (11, 12, 8),
        (12, 12, 9),
        (0, 13, 0),
        (1, 13, 1),
        (2, 13, 2),
        (3, 13, 3),
        (4, 13, 4),
        (5, 13, 5),
        (6, 13, 6),
        (10, 13, 7),
        (11, 13, 8),
        (12, 13, 9),
        (13, 13, 10),
        (0, 14, 0),
        (1, 14, 1),
        (2, 14, 2),
        (3, 14, 3),
        (4, 14, 4),
        (5, 14, 5),
        (14, 14, 6),
        (0, 15, 0),
        (1, 15, 1),
        (2, 15, 2),
        (3, 15, 3),
        (4, 15, 4),
        (5, 15, 5),
        (14, 15, 6),
        (15, 15, 7),
        (0, 16, 0),
        (1, 16, 1),
        (2, 16, 2),
        (3, 16, 3),
        (4, 16, 4),
        (5, 16, 5),
        (14, 16, 6),
        (15, 16, 7),
        (16, 16, 8),
        (0, 17, 0),
        (1, 17, 1),
        (2, 17, 2),
        (3, 17, 3),
        (4, 17, 4),
        (5, 17, 5),
        (14, 17, 6),
        (15, 17, 7),
        (16, 17, 8),
        (17, 17, 9),
        (0, 18, 0),
        (1, 18, 1),
        (2, 18, 2),
        (3, 18, 3),
        (4, 18, 4),
        (18, 18, 5),
        (0, 19, 0),
        (1, 19, 1),
        (2, 19, 2),
        (3, 19, 3),
        (4, 19, 4),
        (18, 19, 5),
        (19, 19, 6),
        (0, 20, 0),
        (1, 20, 1),
        (2, 20, 2),
        (3, 20, 3),
        (4, 20, 4),
        (18, 20, 5),
        (19, 20, 6),
        (20, 20, 7),
        (0, 21, 0),
        (1, 21, 1),
        (2, 21, 2),
        (3, 21, 3),
        (4, 21, 4),
        (18, 21, 5),
        (19, 21, 6),
        (20, 21, 7),
        (21, 21, 8),
        (0, 22, 0),
        (1, 22, 1),
        (2, 22, 2),
        (3, 22, 3),
        (22, 22, 4),
        (0, 23, 0),
        (1, 23, 1),
        (2, 23, 2),
        (3, 23, 3),
        (22, 23, 4),
        (23, 23, 5),
        (0, 24, 0),
        (1, 24, 1),
        (2, 24, 2),
        (3, 24, 3),
        (22, 24, 4),
        (23, 24, 5),
        (24, 24, 6),
        (0, 25, 0),
        (1, 25, 1),
        (2, 25, 2),
        (3, 25, 3),
        (22, 25, 4),
        (23, 25, 5),
        (24, 25, 6),
        (25, 25, 7),
        (0, 26, 0),
        (1, 26, 1),
        (2, 26, 2),
        (26, 26, 3),
        (0, 27, 0),
        (1, 27, 1),
        (2, 27, 2),
        (26, 27, 3),
        (27, 27, 4),
        (0, 28, 0),
        (1, 28, 1),
        (2, 28, 2),
        (26, 28, 3),
        (27, 28, 4),
        (28, 28, 5),
        (0, 29, 0),
        (1, 29, 1),
        (2, 29, 2),
        (26, 29, 3),
        (27, 29, 4),
        (28, 29, 5),
        (29, 29, 6),
        (0, 30, 0),
        (1, 30, 1),
        (30, 30, 2),
        (0, 31, 0),
        (1, 31, 1),
        (30, 31, 2),
        (31, 31, 3),
        (0, 32, 0),
        (1, 32, 1),
        (30, 32, 2),
        (31, 32, 3),
        (32, 32, 4),
        (0, 33, 0),
        (1, 33, 1),
        (30, 33, 2),
        (31, 33, 3),
        (32, 33, 4),
        (33, 33, 5),
        (0, 34, 0),
        (34, 34, 1),
        (0, 35, 0),
        (34, 35, 1),
        (35, 35, 2),
        (0, 36, 0),
        (34, 36, 1),
        (35, 36, 2),
        (36, 36, 3),
        (0, 37, 0),
        (34, 37, 1),
        (35, 37, 2),
        (36, 37, 3),
        (37, 37, 4),
        (0, 38, 0),
        (38, 38, 1),
        (0, 39, 0),
        (38, 39, 1),
        (39, 39, 2),
        (0, 40, 0),
        (38, 40, 1),
        (39, 40, 2),
        (40, 40, 3),
        (0, 41, 0),
        (1, 41, 1),
        (2, 41, 2),
        (3, 41, 3),
        (4, 41, 4),
        (5, 41, 5),
        (6, 41, 6),
        (7, 41, 7),
        (8, 41, 8),
        (9, 41, 9),
        (41, 41, 10),
        (0, 42, 0),
        (1, 42, 1),
        (2, 42, 2),
        (3, 42, 3),
        (4, 42, 4),
        (5, 42, 5),
        (6, 42, 6),
        (7, 42, 7),
        (8, 42, 8),
        (9, 42, 9),
        (41, 42, 10),
        (42, 42, 11),
        (0, 43, 0),
        (1, 43, 1),
        (2, 43, 2),
        (3, 43, 3),
        (4, 43, 4),
        (5, 43, 5),
        (6, 43, 6),
        (7, 43, 7),
        (8, 43, 8),
        (9, 43, 9),
        (41, 43, 10),
        (42, 43, 11),
        (43, 43, 12),
        (0, 44, 0),
        (1, 44, 1),
        (2, 44, 2),
        (3, 44, 3),
        (4, 44, 4),
        (5, 44, 5),
        (6, 44, 6),
        (7, 44, 7),
        (8, 44, 8),
        (9, 44, 9),
        (41, 44, 10),
        (42, 44, 11),
        (43, 44, 12),
        (44, 44, 13),
        (0, 45, 0),
        (1, 45, 1),
        (2, 45, 2),
        (3, 45, 3),
        (4, 45, 4),
        (5, 45, 5),
        (6, 45, 6),
        (7, 45, 7),
        (8, 45, 8),
        (9, 45, 9),
        (41, 45, 10),
        (42, 45, 11),
        (43, 45, 12),
        (44, 45, 13),
        (45, 45, 14),
        (0, 46, 0),
        (1, 46, 1),
        (2, 46, 2),
        (3, 46, 3),
        (4, 46, 4),
        (5, 46, 5),
        (6, 46, 6),
        (7, 46, 7),
        (8, 46, 8),
        (9, 46, 9),
        (41, 46, 10),
        (42, 46, 11),
        (43, 46, 12),
        (44, 46, 13),
        (45, 46, 14),
        (46, 46, 15),
        (0, 47, 0),
        (1, 47, 1),
        (2, 47, 2),
        (3, 47, 3),
        (4, 47, 4),
        (5, 47, 5),
        (6, 47, 6),
        (7, 47, 7),
        (8, 47, 8),
        (9, 47, 9),
        (41, 47, 10),
        (42, 47, 11),
        (43, 47, 12),
        (44, 47, 13),
        (45, 47, 14),
        (46, 47, 15),
        (47, 47, 16),
        (0, 48, 0),
        (1, 48, 1),
        (2, 48, 2),
        (3, 48, 3),
        (4, 48, 4),
        (5, 48, 5),
        (6, 48, 6),
        (7, 48, 7),
        (8, 48, 8),
        (9, 48, 9),
        (41, 48, 10),
        (42, 48, 11),
        (43, 48, 12),
        (44, 48, 13),
        (45, 48, 14),
        (46, 48, 15),
        (47, 48, 16),
        (48, 48, 17),
        (0, 49, 0),
        (1, 49, 1),
        (2, 49, 2),
        (3, 49, 3),
        (4, 49, 4),
        (5, 49, 5),
        (6, 49, 6),
        (7, 49, 7),
        (8, 49, 8),
        (9, 49, 9),
        (41, 49, 10),
        (42, 49, 11),
        (43, 49, 12),
        (44, 49, 13),
        (45, 49, 14),
        (46, 49, 15),
        (47, 49, 16),
        (48, 49, 17),
        (49, 49, 18),
        (0, 50, 0),
        (1, 50, 1),
        (2, 50, 2),
        (3, 50, 3),
        (4, 50, 4),
        (5, 50, 5),
        (6, 50, 6),
        (7, 50, 7),
        (8, 50, 8),
        (9, 50, 9),
        (41, 50, 10),
        (42, 50, 11),
        (43, 50, 12),
        (44, 50, 13),
        (45, 50, 14),
        (46, 50, 15),
        (47, 50, 16),
        (48, 50, 17),
        (49, 50, 18),
        (50, 50, 19),
        (0, 51, 0),
        (1, 51, 1),
        (2, 51, 2),
        (3, 51, 3),
        (4, 51, 4),
        (5, 51, 5),
        (6, 51, 6),
        (7, 51, 7),
        (8, 51, 8),
        (9, 51, 9),
        (41, 51, 10),
        (42, 51, 11),
        (43, 51, 12),
        (44, 51, 13),
        (45, 51, 14),
        (46, 51, 15),
        (47, 51, 16),
        (48, 51, 17),
        (49, 51, 18),
        (50, 51, 19),
        (51, 51, 20),
        (0, 52, 0),
        (1, 52, 1),
        (2, 52, 2),
        (3, 52, 3),
        (4, 52, 4),
        (5, 52, 5),
        (6, 52, 6),
        (7, 52, 7),
        (8, 52, 8),
        (9, 52, 9),
        (41, 52, 10),
        (42, 52, 11),
        (43, 52, 12),
        (44, 52, 13),
        (45, 52, 14),
        (46, 52, 15),
        (47, 52, 16),
        (48, 52, 17),
        (49, 52, 18),
        (50, 52, 19),
        (51, 52, 20),
        (52, 52, 21),
        (0, 53, 0),
        (1, 53, 1),
        (2, 53, 2),
        (3, 53, 3),
        (4, 53, 4),
        (5, 53, 5),
        (6, 53, 6),
        (7, 53, 7),
        (8, 53, 8),
        (9, 53, 9),
        (41, 53, 10),
        (42, 53, 11),
        (43, 53, 12),
        (44, 53, 13),
        (45, 53, 14),
        (46, 53, 15),
        (47, 53, 16),
        (48, 53, 17),
        (49, 53, 18),
        (50, 53, 19),
        (51, 53, 20),
        (52, 53, 21),
        (53, 53, 22),
        (0, 54, 0),
        (1, 54, 1),
        (2, 54, 2),
        (3, 54, 3),
        (4, 54, 4),
        (5, 54, 5),
        (6, 54, 6),
        (7, 54, 7),
        (8, 54, 8),
        (9, 54, 9),
        (41, 54, 10),
        (42, 54, 11),
        (43, 54, 12),
        (44, 54, 13),
        (45, 54, 14),
        (46, 54, 15),
        (47, 54, 16),
        (48, 54, 17),
        (49, 54, 18),
        (50, 54, 19),
        (51, 54, 20),
        (52, 54, 21),
        (53, 54, 22),
        (54, 54, 23),
        (0, 55, 0),
        (1, 55, 1),
        (2, 55, 2),
        (3, 55, 3),
        (4, 55, 4),
        (5, 55, 5),
        (6, 55, 6),
        (7, 55, 7),
        (8, 55, 8),
        (9, 55, 9),
        (41, 55, 10),
        (42, 55, 11),
        (43, 55, 12),
        (44, 55, 13),
        (45, 55, 14),
        (46, 55, 15),
        (47, 55, 16),
        (48, 55, 17),
        (49, 55, 18),
        (50, 55, 19),
        (51, 55, 20),
        (52, 55, 21),
        (53, 55, 22),
        (54, 55, 23),
        (55, 55, 24),
        (0, 56, 0),
        (1, 56, 1),
        (2, 56, 2),
        (3, 56, 3),
        (4, 56, 4),
        (5, 56, 5),
        (6, 56, 6),
        (7, 56, 7),
        (8, 56, 8),
        (9, 56, 9),
        (41, 56, 10),
        (42, 56, 11),
        (43, 56, 12),
        (44, 56, 13),
        (45, 56, 14),
        (46, 56, 15),
        (47, 56, 16),
        (48, 56, 17),
        (49, 56, 18),
        (50, 56, 19),
        (51, 56, 20),
        (52, 56, 21),
        (53, 56, 22),
        (54, 56, 23),
        (55, 56, 24),
        (56, 56, 25),
        (0, 57, 0),
        (1, 57, 1),
        (2, 57, 2),
        (3, 57, 3),
        (4, 57, 4),
        (5, 57, 5),
        (6, 57, 6),
        (7, 57, 7),
        (8, 57, 8),
        (9, 57, 9),
        (41, 57, 10),
        (42, 57, 11),
        (43, 57, 12),
        (44, 57, 13),
        (45, 57, 14),
        (46, 57, 15),
        (47, 57, 16),
        (48, 57, 17),
        (49, 57, 18),
        (50, 57, 19),
        (51, 57, 20),
        (52, 57, 21),
        (53, 57, 22),
        (54, 57, 23),
        (55, 57, 24),
        (56, 57, 25),
        (57, 57, 26),
        (0, 58, 0),
        (1, 58, 1),
        (2, 58, 2),
        (3, 58, 3),
        (4, 58, 4),
        (5, 58, 5),
        (6, 58, 6),
        (7, 58, 7),
        (8, 58, 8),
        (9, 58, 9),
        (41, 58, 10),
        (42, 58, 11),
        (43, 58, 12),
        (44, 58, 13),
        (45, 58, 14),
        (46, 58, 15),
        (47, 58, 16),
        (48, 58, 17),
        (49, 58, 18),
        (50, 58, 19),
        (51, 58, 20),
        (52, 58, 21),
        (53, 58, 22),
        (54, 58, 23),
        (55, 58, 24),
        (56, 58, 25),
        (57, 58, 26),
        (58, 58, 27),
        (0, 59, 0),
        (1, 59, 1),
        (2, 59, 2),
        (3, 59, 3),
        (4, 59, 4),
        (5, 59, 5),
        (6, 59, 6),
        (7, 59, 7),
        (8, 59, 8),
        (9, 59, 9),
        (41, 59, 10),
        (42, 59, 11),
        (43, 59, 12),
        (44, 59, 13),
        (45, 59, 14),
        (46, 59, 15),
        (47, 59, 16),
        (48, 59, 17),
        (49, 59, 18),
        (50, 59, 19),
        (51, 59, 20),
        (52, 59, 21),
        (53, 59, 22),
        (54, 59, 23),
        (55, 59, 24),
        (56, 59, 25),
        (57, 59, 26),
        (58, 59, 27),
        (59, 59, 28),
        (0, 60, 0),
        (1, 60, 1),
        (2, 60, 2),
        (3, 60, 3),
        (4, 60, 4),
        (5, 60, 5),
        (6, 60, 6),
        (7, 60, 7),
        (8, 60, 8),
        (9, 60, 9),
        (41, 60, 10),
        (42, 60, 11),
        (43, 60, 12),
        (44, 60, 13),
        (60, 60, 14),
        (0, 61, 0),
        (1, 61, 1),
        (2, 61, 2),
        (3, 61, 3),
        (4, 61, 4),
        (5, 61, 5),
        (6, 61, 6),
        (7, 61, 7),
        (8, 61, 8),
        (9, 61, 9),
        (41, 61, 10),
        (42, 61, 11),
        (43, 61, 12),
        (44, 61, 13),
        (60, 61, 14),
        (61, 61, 15),
        (0, 62, 0),
        (1, 62, 1),
        (2, 62, 2),
        (3, 62, 3),
        (4, 62, 4),
        (5, 62, 5),
        (6, 62, 6),
        (7, 62, 7),
        (8, 62, 8),
        (9, 62, 9),
        (41, 62, 10),
        (42, 62, 11),
        (43, 62, 12),
        (44, 62, 13),
        (60, 62, 14),
        (61, 62, 15),
        (62, 62, 16),
        (0, 63, 0),
        (1, 63, 1),
        (2, 63, 2),
        (3, 63, 3),
        (4, 63, 4),
        (5, 63, 5),
        (6, 63, 6),
        (7, 63, 7),
        (8, 63, 8),
        (9, 63, 9),
        (41, 63, 10),
        (42, 63, 11),
        (43, 63, 12),
        (44, 63, 13),
        (60, 63, 14),
        (61, 63, 15),
        (62, 63, 16),
        (63, 63, 17),
    ];
}
