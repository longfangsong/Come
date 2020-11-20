use crate::ir::value::Value;
use std::rc::Rc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Store {
    from: Rc<Value>,
    to: Rc<Value>,
}
