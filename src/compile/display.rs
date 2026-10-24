//! Stable textual form of traced tensor functions.
//!
//! Shared subexpressions are printed once as SSA-style `%n` bindings, in
//! dependency order, so unrolled recursion stays linear in the output.
//!
//! Every binding shows its inferred type, including the support mask when it
//! is narrower than the full layout:
//!
//! ```text
//! fn re.opt(a: bits[15]) -> bits[15] {
//!   %0 = input a : tensor<1x1xu64, 15bits/0>
//!   %1 = const [0x0000000000000001] : tensor<1x1xu64, 15bits/0, 1 live>
//!   %2 = or %0, %1 : tensor<1x1xu64, 15bits/0>
//!   return %2
//! }
//! ```

use std::collections::HashMap;
use std::fmt::{self, Write as _};

use super::typing::TypeCache;
use super::{ExpandReduceDimType, JaggedTensor, LogicalOp, ShiftOp, TensorExpr, TensorFn};

/// Tensors with at most this many elements are printed inline.
const INLINE_ELEMENTS: usize = 16;
/// Elements shown for a tensor printed on its own line.
const SHOWN_ELEMENTS: usize = 16;

struct Printer {
    lines: Vec<String>,
    /// Tensor lines of the instruction being printed.
    pending: Vec<String>,
    /// Number of out-of-line tensors so far (`#0`, `#1`, ...).
    tensors: usize,
    ids: HashMap<*const TensorExpr, usize>,
    types: TypeCache,
}

fn op_name(op: LogicalOp) -> &'static str {
    match op {
        LogicalOp::And => "and",
        LogicalOp::Or => "or",
        LogicalOp::Xor => "xor",
    }
}

fn list<T: fmt::Display>(values: impl IntoIterator<Item = T>) -> String {
    let items: Vec<String> = values.into_iter().map(|value| value.to_string()).collect();
    format!("[{}]", items.join(", "))
}

/// Elements of a tensor, row by row (one row for a flat tensor).
fn tensor_rows<T: fmt::Display + Copy>(data: &[T], offsets: Option<&[usize]>) -> Vec<Vec<String>> {
    match offsets {
        None => vec![data.iter().map(ToString::to_string).collect()],
        Some(offsets) => offsets
            .windows(2)
            .map(|pair| data[pair[0]..pair[1]].iter().map(ToString::to_string).collect())
            .collect(),
    }
}

impl Printer {
    /// Render a tensor: inline when small, otherwise as `#k` with the
    /// (truncated) tensor on its own line below the instruction.
    fn tensor(&mut self, rows: Vec<Vec<String>>, jagged: bool) -> String {
        let total: usize = rows.iter().map(Vec::len).sum::<usize>() + if jagged { rows.len() } else { 0 };
        let render = |limit: usize| {
            let mut shown = 0;
            let mut parts = Vec::new();
            let mut truncated = false;
            for row in &rows {
                if shown >= limit {
                    truncated = true;
                    break;
                }
                let take = row.len().min(limit - shown).max(usize::from(!jagged));
                let mut items: Vec<String> = row.iter().take(take).cloned().collect();
                if take < row.len() {
                    items.push("...".into());
                    truncated = true;
                }
                shown += take.max(1);
                if jagged {
                    parts.push(format!("[{}]", items.join(", ")));
                } else {
                    parts.extend(items);
                }
            }
            if truncated && (jagged || parts.last().map(String::as_str) != Some("...")) {
                parts.push("...".into());
            }
            format!("[{}]", parts.join(", "))
        };
        if total <= INLINE_ELEMENTS {
            return render(usize::MAX);
        }
        let name = format!("#{}", self.tensors);
        self.tensors += 1;
        let elements: usize = rows.iter().map(Vec::len).sum();
        let shape = if jagged { format!("{} rows, {elements} elements", rows.len()) } else { format!("{elements} elements") };
        self.pending.push(format!("    {name} = {}  // {shape}", render(SHOWN_ELEMENTS)));
        name
    }

    fn indices(&mut self, indices: &JaggedTensor<usize>) -> String {
        match indices.jagged_dims() {
            [] => self.tensor(tensor_rows(indices.data(), None), false),
            [offsets] => self.tensor(tensor_rows(indices.data(), Some(offsets)), true),
            _ => "<nested jagged indices>".into(),
        }
    }
}

impl Printer {
    /// Print `expression` (and its unprinted children); return its id.
    fn visit(&mut self, expression: &TensorExpr) -> usize {
        let key = expression as *const TensorExpr;
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let text = match expression {
            TensorExpr::Input(input) => format!("input {}", input.name),
            TensorExpr::Constant(constant) => {
                // Runs of equal words, `first..last: word`; zero runs omitted.
                let data = constant.value.data();
                let mut runs = Vec::new();
                let mut start = 0;
                while start < data.len() {
                    let mut end = start;
                    while end + 1 < data.len() && data[end + 1] == data[start] {
                        end += 1;
                    }
                    if data[start] != 0 {
                        let range = if end > start { format!("{start}..{end}") } else { start.to_string() };
                        runs.push(format!("{range}: {:#018x}", data[start]));
                    }
                    start = end + 1;
                }
                let words = self.tensor(vec![runs], false);
                format!("const {words} of {} words", data.len())
            }
            TensorExpr::ScalarConstant(constant) => format!("scalar {:#018x}", constant.value),
            TensorExpr::BVLogical(logical) => {
                let operands: Vec<String> = logical
                    .operands
                    .iter()
                    .map(|operand| {
                        let id = self.visit(&operand.expr);
                        if operand.reduce_dims.is_empty() {
                            format!("%{id}")
                        } else {
                            format!("reduce(%{id}, dims={})", list(operand.reduce_dims.iter()))
                        }
                    })
                    .collect();
                format!("{} {}", op_name(logical.op), operands.join(", "))
            }
            TensorExpr::BVGather(gather) => {
                let id = self.visit(&gather.expr);
                let out = self.indices(&gather.indices_out);
                let input = self.indices(&gather.indices_in);
                format!("gather.{} %{id} out={out} in={input}", op_name(gather.op))
            }
            TensorExpr::BVGather2(gather) => {
                let a = self.visit(&gather.a);
                let b = self.visit(&gather.b);
                let out = self.indices(&gather.indices_out);
                let in1 = self.indices(&gather.indices_1);
                let in2 = self.indices(&gather.indices_2);
                format!("gather2.{}.{} %{a}, %{b} out={out} in1={in1} in2={in2}", op_name(gather.op), op_name(gather.inner))
            }
            TensorExpr::BVJaggedPermute(permute) => {
                let id = self.visit(&permute.expr);
                let indices = self.indices(&permute.indices);
                format!("permute %{id} {indices}")
            }
            TensorExpr::BTExpandReduce(change) => {
                let id = self.visit(&change.expr);
                let dims = change.bt_dims.iter().map(|dimension| match dimension {
                    ExpandReduceDimType::Batch(extent) => extent.to_string(),
                    ExpandReduceDimType::Expand(extent) => format!("expand({extent})"),
                    ExpandReduceDimType::Reduce(extent) => format!("reduce({extent})"),
                });
                let op = if change.reduces() { format!(".{}", op_name(change.op)) } else { String::new() };
                format!("expand_reduce{op} %{id} {}", list(dims))
            }
            TensorExpr::BTShift(shift) => {
                let id = self.visit(&shift.expr);
                let kind = match shift.shift_op {
                    ShiftOp::Logical => "lshr",
                    ShiftOp::Arithmetic => "ashr",
                    ShiftOp::Rotate => "rotr",
                };
                format!("{kind} %{id}, {}", shift.shift_amounts)
            }
        };
        let tensor_type = match self.types.infer(expression) {
            Ok(tensor_type) => tensor_type.to_string(),
            Err(error) => format!("<invalid: {error}>"),
        };
        let id = self.ids.len();
        self.lines.push(format!("  %{id} = {text} : {tensor_type}"));
        self.lines.append(&mut self.pending);
        self.ids.insert(key, id);
        id
    }
}

impl fmt::Display for TensorFn {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut printer = Printer {
            lines: Vec::new(),
            ids: HashMap::new(),
            types: TypeCache::default(),
            pending: Vec::new(),
            tensors: 0,
        };
        let root = printer.visit(&self.expression);
        let width = self.expression.infer_type().map(|t| t.width()).unwrap_or(0);
        let mut arguments = String::new();
        for (index, (name, tensor_type)) in self.inputs.iter().enumerate() {
            if index > 0 {
                arguments.push_str(", ");
            }
            write!(arguments, "{name}: bits[{}]", tensor_type.bit_width().unwrap_or(0))?;
        }
        writeln!(out, "fn {}({arguments}) -> bits[{width}] {{", self.name)?;
        for line in &printer.lines {
            writeln!(out, "{line}")?;
        }
        writeln!(out, "  return %{root}\n}}")
    }
}
