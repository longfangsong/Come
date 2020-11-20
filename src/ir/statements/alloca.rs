use crate::ir::data_type::DataType;
use crate::ir::value::IsValue;
use crate::ir::data_type::space::Space;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Alloca {
    data_type: DataType,
}

impl IsValue for Alloca {
    fn data_type(&self) -> DataType {
        Space::new(32).into()
    }
}
