use crate::ir::value::{Value, IsValue};
use std::rc::Rc;
use crate::ir::data_type::DataType;
use crate::ir::basic_block::BasicBlock;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Phi {
    operand1: Rc<Value>,
    basic_block1: Rc<BasicBlock>,
    operand2: Rc<Value>,
    basic_block2: Rc<BasicBlock>,
}

impl IsValue for Phi {
    fn data_type(&self) -> DataType {
        self.operand1.data_type()
    }
}
