use crate::ir::value::Value;
use std::rc::Rc;

pub enum Operator {
    Add,
    Sub,
    Not,
    And,
    Or,
    Xor,
    Slt,
}

pub struct BinaryOperation {
    operator: Operator,
    lhs: Rc<Value>,
    rhs: Rc<Value>,
}
