use crate::ir::statements::{BinaryOperation, GetField, GetIndex, Load, Phi};

pub enum Value {
    Constant,
    Global,
    Phi(Phi),
    BinaryOperation(BinaryOperation),
    Load(Load),
    GetField(GetField),
    GetIndex(GetIndex),
}
