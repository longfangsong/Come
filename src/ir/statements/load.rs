use crate::ir::{data_type::DataType, value::Value};
use std::rc::Rc;
use crate::ir::value::IsValue;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Load {
    from: Rc<Value>,
    data_type: DataType,
}

impl IsValue for Load {
    fn data_type(&self) -> DataType {
        self.data_type.clone()
    }
}
