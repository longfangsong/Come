mod alloca;
mod binary_operation;
mod branch;
mod get_field;
mod get_index;
mod jump;
mod load;
mod phi;
mod store;

pub use alloca::Alloca;
pub use binary_operation::BinaryOperation;
pub use branch::Branch;
pub use get_field::GetField;
pub use get_index::GetIndex;
pub use jump::Jump;
pub use load::Load;
pub use phi::Phi;
pub use store::Store;

pub enum Statement {
    Alloca,
    Load,
    Store,
    BinaryOperation,
    GetField,
    GetIndex,
}

pub enum Terminator {
    Jump,
    Branch,
}
