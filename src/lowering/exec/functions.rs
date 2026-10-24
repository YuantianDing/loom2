//! Emit the packed `define_fun_*` and constraint functions from loop programs.
//!
//! ```text
//! memref.global "private" constant @<fn>_t<k> : memref<Nxi<W>> = dense<...>
//!
//! func.func private @define_fun_<TensorFn.name>(
//!     %arg0: vector<<ceil(arg0_bits / 64)>xi64>, ...
//! ) -> vector<<ceil(result_bits / 64)>xi64>
//!     attributes {loom.constructor = "<TensorFn.name>"} {
//!   <32-bit programs: vector.bitcast to vector<...xi32>>
//!   <Broadcast / Constant>    arith.constant dense<...> | memref.get_global
//!   <Load / Store>            vector.load (tables) | vector.extract / vector.insert
//!   <And / Or / Xor / Rotl>   arith on whole vectors
//!   <ForLoop>                 scf.for with loop-carried vectors
//!   return %packed_result
//! }
//! ```
//!
//! Step tables that are indexed by a loop live in read-only globals; every
//! other value is an SSA vector. The functions are pure, so production filters
//! may call them concurrently.

use std::collections::{HashMap, HashSet};

use melior::ir::block::BlockLike;
use melior::ir::{Block, Operation, Region, RegionLike, Type, Value, ValueLike};
use mlir_sys::MlirValue;

use super::{arguments, invalid, region, symbol_name, Emitter, Result};
use crate::lowering::loops::{word_count, LoopProgram, Operation as Op, VarId};

const DYNAMIC: i64 = i64::MIN;

/// Where a variable lives during emission.
#[derive(Clone, Copy)]
enum Slot {
    Vector(MlirValue),
    /// A loop induction variable (`index`).
    Index(MlirValue),
    /// Words `base..` of a global table (`base` is an `index`).
    Table(MlirValue, MlirValue),
}

fn value<'c, 'a>(raw: MlirValue) -> Value<'c, 'a> {
    // SAFETY: every raw value stored in a `Slot` comes from an operation or
    // block argument that is alive for the whole function being emitted.
    unsafe { Value::from_raw(raw) }
}

struct FunctionEmitter<'e, 'c> {
    emitter: &'e Emitter<'c>,
    program: &'e LoopProgram,
    element: Type<'c>,
    globals: Vec<Operation<'c>>,
    global_names: HashMap<VarId, String>,
    /// Variables that stay in table memory.
    tables: HashSet<VarId>,
    /// Constant scalars (`Broadcast(var, value, 1)`), for loop bounds.
    scalars: HashMap<VarId, u64>,
}

impl<'c> FunctionEmitter<'_, 'c> {
    fn vector_type(&self, words: usize) -> Type<'c> {
        self.emitter.ty(&format!("vector<{words}xi{}>", self.program.word_width))
    }

    fn signed(&self, word: u64) -> i64 {
        if self.program.word_width == 32 {
            word as u32 as i32 as i64
        } else {
            word as i64
        }
    }

    fn dense(&self, words: &[u64]) -> String {
        let items: Vec<String> = words.iter().map(|&word| self.signed(word).to_string()).collect();
        format!("dense<[{}]>", items.join(", "))
    }

    fn vector_constant(&self, block: &Block<'c>, words: &[u64]) -> MlirValue {
        let ty = self.vector_type(words.len());
        let attribute = self.emitter.attr(&format!("{} : {ty}", self.dense(words)));
        self.emitter.value(block, "arith.constant", &[], ty, &[("value", attribute)]).to_raw()
    }

    fn index(&self, block: &Block<'c>, slot: Slot) -> MlirValue {
        match slot {
            Slot::Index(raw) => raw,
            Slot::Vector(raw) => {
                let element = self.extract(block, raw, None, 0);
                self.emitter.index_cast(block, value(element), "index").to_raw()
            }
            Slot::Table(..) => panic!("a table cannot be used as an index"),
        }
    }

    /// `vector.extract %source[position]` (static, or dynamic `index`).
    fn extract(&self, block: &Block<'c>, source: MlirValue, dynamic: Option<MlirValue>, position: i64) -> MlirValue {
        let mut operands = vec![value(source)];
        let static_position = match dynamic {
            Some(index) => {
                operands.push(value(index));
                DYNAMIC
            }
            None => position,
        };
        let attribute = self.emitter.attr(&format!("array<i64: {static_position}>"));
        self.emitter
            .value(block, "vector.extract", &operands, self.element, &[("static_position", attribute)])
            .to_raw()
    }

    fn insert(&self, block: &Block<'c>, scalar: MlirValue, destination: MlirValue, index: MlirValue) -> MlirValue {
        let ty = value(destination).r#type();
        let attribute = self.emitter.attr(&format!("array<i64: {DYNAMIC}>"));
        self.emitter
            .value(
                block,
                "vector.insert",
                &[value(scalar), value(destination), value(index)],
                ty,
                &[("static_position", attribute)],
            )
            .to_raw()
    }

    fn row_offset(&self, block: &Block<'c>, row: MlirValue, words: usize, word: usize) -> MlirValue {
        let width = self.emitter.index(block, words);
        let start = self.emitter.binary(block, "arith.muli", value(row), width);
        if word == 0 {
            return start.to_raw();
        }
        let offset = self.emitter.index(block, word);
        self.emitter.binary(block, "arith.addi", start, offset).to_raw()
    }

    fn global(&mut self, var: VarId, data: &[u64]) -> String {
        let name = format!(
            "{}_t{}",
            symbol_name(&format!("table_{}", self.program.name)),
            self.global_names.len()
        );
        let ty = format!("memref<{}xi{}>", data.len(), self.program.word_width);
        let tensor = format!("tensor<{}xi{}>", data.len(), self.program.word_width);
        let attributes = [
            ("sym_name", self.emitter.attr(&format!("\"{name}\""))),
            ("sym_visibility", self.emitter.attr("\"private\"")),
            ("type", self.emitter.attr(&ty)),
            ("initial_value", self.emitter.attr(&format!("{} : {tensor}", self.dense(data)))),
            ("constant", self.emitter.attr("unit")),
        ];
        self.globals.push(self.emitter.build("memref.global", &[], &[], &attributes, Vec::new()));
        self.global_names.insert(var, name.clone());
        name
    }

    /// Emit `ops[range]` into `block`, updating `env`.
    fn emit_range(&mut self, block: &Block<'c>, range: std::ops::Range<usize>, env: &mut HashMap<VarId, Slot>) {
        let program = self.program;
        let mut position = range.start;
        while position < range.end {
            let op = &program.ops[position];
            match op {
                Op::Broadcast(dst, word, count) => {
                    if *count == 1 {
                        self.scalars.insert(*dst, *word);
                    }
                    let raw = self.vector_constant(block, &vec![*word; *count]);
                    env.insert(*dst, Slot::Vector(raw));
                }
                Op::Constant(dst, data) => {
                    if self.tables.contains(dst) {
                        let name = self.global(*dst, data);
                        let ty = self.emitter.ty(&format!("memref<{}xi{}>", data.len(), program.word_width));
                        let attribute = self.emitter.attr(&format!("@{name}"));
                        let memory = self.emitter.value(block, "memref.get_global", &[], ty, &[("name", attribute)]);
                        let zero = self.emitter.index(block, 0);
                        env.insert(*dst, Slot::Table(memory.to_raw(), zero.to_raw()));
                    } else {
                        env.insert(*dst, Slot::Vector(self.vector_constant(block, data)));
                    }
                }
                Op::Load(dst, src, index) => {
                    let words = program.words(*dst);
                    let row = self.index(block, env[index]);
                    let slot = match env[src] {
                        Slot::Table(memory, base) => {
                            let offset = self.row_offset(block, row, words, 0);
                            let start = self.emitter.binary(block, "arith.addi", value(base), value(offset)).to_raw();
                            if self.tables.contains(dst) {
                                Slot::Table(memory, start)
                            } else {
                                let loaded = self.emitter.value(
                                    block,
                                    "vector.load",
                                    &[value(memory), value(start)],
                                    self.vector_type(words),
                                    &[],
                                );
                                Slot::Vector(loaded.to_raw())
                            }
                        }
                        Slot::Vector(source) => {
                            let elements: Vec<MlirValue> = (0..words)
                                .map(|word| {
                                    let at = self.row_offset(block, row, words, word);
                                    self.extract(block, source, Some(at), 0)
                                })
                                .collect();
                            let elements: Vec<Value> = elements.into_iter().map(value).collect();
                            let built = self.emitter.value(
                                block,
                                "vector.from_elements",
                                &elements,
                                self.vector_type(words),
                                &[],
                            );
                            Slot::Vector(built.to_raw())
                        }
                        Slot::Index(_) => panic!("cannot load from an index"),
                    };
                    env.insert(*dst, slot);
                }
                Op::Store(dst, index, src) => {
                    let words = program.words(*src);
                    let row = self.index(block, env[index]);
                    let (Slot::Vector(mut target), Slot::Vector(source)) = (env[dst], env[src]) else {
                        panic!("stores operate on vectors");
                    };
                    for word in 0..words {
                        let element = self.extract(block, source, None, word as i64);
                        let at = self.row_offset(block, row, words, word);
                        target = self.insert(block, element, target, at);
                    }
                    env.insert(*dst, Slot::Vector(target));
                }
                Op::Splat(dst, src) => {
                    let Slot::Vector(source) = env[src] else { panic!("splat of a non-vector") };
                    let element = self.extract(block, source, None, 0);
                    let ty = self.vector_type(program.words(*dst));
                    let raw = self.emitter.value(block, "vector.broadcast", &[value(element)], ty, &[]).to_raw();
                    env.insert(*dst, Slot::Vector(raw));
                }
                Op::And(dst, a, b) | Op::Or(dst, a, b) | Op::Xor(dst, a, b) => {
                    let name = match op {
                        Op::And(..) => "arith.andi",
                        Op::Or(..) => "arith.ori",
                        _ => "arith.xori",
                    };
                    let (Slot::Vector(x), Slot::Vector(y)) = (env[a], env[b]) else { panic!("logic on non-vectors") };
                    let raw = self.emitter.binary(block, name, value(x), value(y)).to_raw();
                    env.insert(*dst, Slot::Vector(raw));
                }
                Op::Rotl(dst, a, b) => {
                    let (Slot::Vector(x), Slot::Vector(r)) = (env[a], env[b]) else { panic!("rotl on non-vectors") };
                    let words = program.words(*dst);
                    let width = u64::from(program.word_width);
                    let full = self.vector_constant(block, &vec![width; words]);
                    let low = self.vector_constant(block, &vec![width - 1; words]);
                    let high = self.emitter.binary(block, "arith.shli", value(x), value(r));
                    let back = self.emitter.binary(block, "arith.subi", value(full), value(r));
                    let back = self.emitter.binary(block, "arith.andi", back, value(low));
                    let wrapped = self.emitter.binary(block, "arith.shrui", value(x), back);
                    let raw = self.emitter.binary(block, "arith.ori", high, wrapped).to_raw();
                    env.insert(*dst, Slot::Vector(raw));
                }
                Op::Copy(dst, src) => {
                    let slot = env[src];
                    env.insert(*dst, slot);
                }
                Op::ForLoop(iv, trip) => {
                    let end = program.live(*iv).end;
                    self.emit_loop(block, *iv, self.scalars[trip] as usize, position, end, env);
                    position = end;
                    continue;
                }
            }
            position += 1;
        }
    }

    fn emit_loop(
        &mut self,
        block: &Block<'c>,
        iv: VarId,
        trips: usize,
        position: usize,
        end: usize,
        env: &mut HashMap<VarId, Slot>,
    ) {
        let program = self.program;
        // Loop-carried: written in the body, defined before the loop.
        let mut carried: Vec<VarId> = program.ops[position + 1..end]
            .iter()
            .filter_map(Operation_writes)
            .filter(|var| *var != iv && program.live(*var).start < position && env.contains_key(var))
            .collect();
        carried.sort();
        carried.dedup();
        let initial: Vec<MlirValue> = carried
            .iter()
            .map(|var| match env[var] {
                Slot::Vector(raw) => raw,
                _ => panic!("only vectors can be loop-carried"),
            })
            .collect();
        let types: Vec<Type<'c>> = initial.iter().map(|&raw| value(raw).r#type()).collect();
        let index = self.emitter.ty("index");
        let mut block_arguments = vec![(index, self.emitter.location)];
        block_arguments.extend(types.iter().map(|&ty| (ty, self.emitter.location)));
        let body = Block::new(&block_arguments);
        let mut inner = env.clone();
        inner.insert(iv, Slot::Index(body.argument(0).expect("iv").to_raw()));
        for (offset, var) in carried.iter().enumerate() {
            inner.insert(*var, Slot::Vector(body.argument(offset + 1).expect("carried").to_raw()));
        }
        self.emit_range(&body, position + 1..end, &mut inner);
        let yielded: Vec<Value> = carried
            .iter()
            .map(|var| match inner[var] {
                Slot::Vector(raw) => value(raw),
                _ => panic!("only vectors can be loop-carried"),
            })
            .collect();
        self.emitter.op(&body, "scf.yield", &yielded, &[], &[], Vec::new());
        let lower = self.emitter.index(block, 0);
        let upper = self.emitter.index(block, trips);
        let step = self.emitter.index(block, 1);
        let mut operands = vec![lower, upper, step];
        operands.extend(initial.iter().map(|&raw| value(raw)));
        let looped = self.emitter.op(block, "scf.for", &operands, &types, &[], vec![region(body)]);
        for (offset, var) in carried.iter().enumerate() {
            env.insert(*var, Slot::Vector(looped.result(offset).expect("result").to_raw()));
        }
    }
}

#[allow(non_snake_case)]
fn Operation_writes(op: &Op) -> Option<VarId> {
    op.writes()
}

/// Variables that stay in table memory: constants and table rows that are
/// only ever used as the source of a `Load`.
fn table_variables(program: &LoopProgram) -> HashSet<VarId> {
    let mut load_only: HashMap<VarId, bool> = HashMap::new();
    for op in &program.ops {
        for var in op.reads() {
            let as_source = matches!(op, Op::Load(_, src, _) if *src == var);
            let entry = load_only.entry(var).or_insert(true);
            *entry &= as_source;
        }
    }
    let mut tables = HashSet::new();
    for op in &program.ops {
        match op {
            Op::Constant(dst, _) if load_only.get(dst) == Some(&true) => {
                tables.insert(*dst);
            }
            Op::Load(dst, src, _) if tables.contains(src) && load_only.get(dst) == Some(&true) => {
                tables.insert(*dst);
            }
            _ => {}
        }
    }
    tables
}

/// Name of the packed constructor function for `constructor`.
pub fn define_fun_symbol(constructor: &str) -> String {
    format!("define_fun_{}", symbol_name(constructor))
}

/// The `vector<Nxi64>` type of a value of `bits` bits.
pub fn packed_type<'c>(emitter: &Emitter<'c>, bits: usize) -> Type<'c> {
    emitter.ty(&format!("vector<{}xi64>", word_count(bits)))
}

/// Emit `program` as a function whose `finish` turns the output slot into the
/// returned values. Returns the function and the globals it references.
fn emit_function<'c>(
    emitter: &Emitter<'c>,
    program: &LoopProgram,
    symbol: &str,
    results: &[Type<'c>],
    attributes: &[(&str, melior::ir::attribute::Attribute<'c>)],
    finish: &dyn Fn(&Block<'c>, MlirValue) -> Vec<MlirValue>,
) -> (Operation<'c>, Vec<Operation<'c>>) {
    let mut state = FunctionEmitter {
        emitter,
        program,
        element: emitter.ty(&format!("i{}", program.word_width)),
        globals: Vec::new(),
        global_names: HashMap::new(),
        tables: table_variables(program),
        scalars: HashMap::new(),
    };
    let arguments_types: Vec<Type> = program.input_bits.iter().map(|&bits| packed_type(emitter, bits)).collect();
    let signature = format!(
        "({}) -> ({})",
        arguments_types.iter().map(ToString::to_string).collect::<Vec<_>>().join(", "),
        results.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ")
    );
    let entry = Block::new(&arguments_types.iter().map(|&ty| (ty, emitter.location)).collect::<Vec<_>>());
    let mut env = HashMap::new();
    for (index, argument) in arguments(&entry).into_iter().enumerate() {
        let raw = if program.word_width == 32 {
            let ty = state.vector_type(program.inputs[index].0);
            emitter.value(&entry, "vector.bitcast", &[argument], ty, &[]).to_raw()
        } else {
            argument.to_raw()
        };
        env.insert(VarId::Input(index), Slot::Vector(raw));
    }
    state.emit_range(&entry, 0..program.ops.len(), &mut env);
    let Slot::Vector(output) = env[&VarId::Output(0)] else { panic!("the output must be a vector") };
    let returned = finish(&entry, output);
    let returned: Vec<Value> = returned.into_iter().map(value).collect();
    emitter.op(&entry, "func.return", &returned, &[], &[], Vec::new());
    let region = Region::new();
    region.append_block(entry);
    let mut all_attributes = vec![
        ("sym_name", emitter.attr(&format!("\"{symbol}\""))),
        ("function_type", emitter.attr(&signature)),
        ("sym_visibility", emitter.attr("\"private\"")),
    ];
    all_attributes.extend_from_slice(attributes);
    let function = emitter.build("func.func", &[], &[], &all_attributes, vec![region]);
    (function, state.globals)
}

/// Emit one packed constructor, returning it with its table globals.
pub fn lower_tensor_fn<'c>(emitter: &Emitter<'c>, program: &LoopProgram) -> (Operation<'c>, Vec<Operation<'c>>) {
    let result = packed_type(emitter, program.result_bits);
    let constructor = emitter.attr(&format!("{:?}", program.name));
    emit_function(
        emitter,
        program,
        &define_fun_symbol(&program.name),
        &[result],
        &[("loom.constructor", constructor)],
        &|block, output| {
            if program.word_width == 32 {
                vec![emitter.value(block, "vector.bitcast", &[value(output)], result, &[]).to_raw()]
            } else {
                vec![output]
            }
        },
    )
}

/// Emit `@<symbol>(candidate) -> i1` from a one-bit constraint program.
pub fn lower_constraint<'c>(
    emitter: &Emitter<'c>,
    symbol: &str,
    program: &LoopProgram,
) -> Result<(Operation<'c>, Vec<Operation<'c>>)> {
    if program.result_bits != 1 || program.input_bits.len() != 1 {
        return Err(invalid("constraint lowering", "a constraint must map one candidate to one bit"));
    }
    let element = emitter.ty(&format!("i{}", program.word_width));
    Ok(emit_function(emitter, program, symbol, &[emitter.ty("i1")], &[], &|block, output| {
        let attribute = emitter.attr("array<i64: 0>");
        let bit = emitter.value(block, "vector.extract", &[value(output)], element, &[("static_position", attribute)]);
        let zero = emitter.attr(&format!("0 : i{}", program.word_width));
        let zero = emitter.value(block, "arith.constant", &[], element, &[("value", zero)]);
        vec![emitter.cmpi(block, "ne", bit, zero).to_raw()]
    }))
}
