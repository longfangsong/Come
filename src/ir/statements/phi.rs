use crate::ir::value::Value;
use std::rc::Rc;

pub struct Phi {
    operand1: Rc<Value>,
}
