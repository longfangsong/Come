use crate::ir::{basic_block::BasicBlock, value::Value};
use std::rc::Rc;

pub enum BranchType {
    EQ,
    NE,
    LT,
    GE,
}

pub struct Branch {
    pub branch_type: BranchType,
    pub operand1: Rc<Value>,
    pub operand2: Rc<Value>,
    pub success: Rc<BasicBlock>,
    pub failure: Rc<BasicBlock>,
}
