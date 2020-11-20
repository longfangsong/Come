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
use crate::ir::basic_block::{BasicBlockBuilder, BasicBlock};
use std::rc::Rc;
use std::fmt::Debug;
use derive_more::From;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Statement {
    Alloca(Alloca),
    Load(Load),
    Store(Store),
    BinaryOperation(BinaryOperation),
    GetField(GetField),
    GetIndex(GetIndex),
}

#[derive(Clone, Debug, Eq, PartialEq, From)]
pub enum Terminator {
    Jump(Jump),
    Branch(Branch),
}

pub trait TerminatorBuilder: Debug {
    fn on_other_basic_block_done(&mut self, builder: Rc<BasicBlock>) -> Option<Terminator>;
}
