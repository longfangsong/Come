use crate::ir::value::{Value, IsValue};
use std::rc::Rc;
use crate::ir::data_type::DataType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Operator {
    Add,
    Sub,
    Not,
    And,
    Or,
    Xor,
    Slt,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BinaryOperation {
    operator: Operator,
    lhs: Rc<Value>,
    rhs: Rc<Value>,
}

impl IsValue for BinaryOperation {
    fn data_type(&self) -> DataType {
        self.lhs.data_type()
    }
}
