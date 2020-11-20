use crate::ir::statements::{BinaryOperation, GetField, GetIndex, Load, Phi};
use crate::ir::global_definition::GlobalDefinition;
use crate::ir::data_type::DataType;
use enum_dispatch::enum_dispatch;
use crate::ir::data_type::integer::Integer;

#[enum_dispatch]
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Value {
    Constant(i64),
    Global(GlobalDefinition),
    Phi(Phi),
    BinaryOperation(BinaryOperation),
    Load(Load),
    GetField(GetField),
    GetIndex(GetIndex),
}

#[enum_dispatch(Value)]
pub trait IsValue {
    fn data_type(&self) -> DataType;
}

impl IsValue for i64 {
    fn data_type(&self) -> DataType {
        Integer::new(true, 32).into()
    }
}