use crate::ir::statements::{Phi, Statement, Terminator};

pub struct BasicBlock {
    name: Option<String>,
    phis: Vec<Phi>,
    statements: Vec<Statement>,
    terminator: Terminator,
}
