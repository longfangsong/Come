use crate::ir::{data_type::DataType, value::Value};
use std::rc::Rc;

pub struct Load {
    from: Rc<Value>,
    type_name: DataType,
}
