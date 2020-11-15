use crate::ir::value::Value;
use std::rc::Rc;

pub struct Store {
    from: Rc<Value>,
    to: Rc<Value>,
}
