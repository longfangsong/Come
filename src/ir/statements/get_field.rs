use crate::ir::{data_type::DataType, value::Value};
use std::rc::Rc;

pub struct GetField {
    from: Rc<Value>,
    index: usize,
    get_type: DataType,
}
