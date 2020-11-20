use crate::ir::data_type::DataType;
use crate::ir::value::IsValue;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalDefinition {
    name: String,
    data_type: DataType,
}

impl IsValue for GlobalDefinition {
    fn data_type(&self) -> DataType {
        self.data_type.clone()
    }
}
