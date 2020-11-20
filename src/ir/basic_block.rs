use crate::ir::statements::{Phi, Statement, Terminator};
use std::rc::Rc;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BasicBlock {
    pub(crate) name: Option<String>,
    phis: Vec<Rc<Phi>>,
    statements: Vec<Rc<Statement>>,
    terminator: Option<Terminator>,
}

#[derive(Default, Clone, Debug, Eq, PartialEq)]
pub struct BasicBlockBuilder {
    pub name: Option<String>,
    pub phis: Vec<Rc<Phi>>,
    pub statements: Vec<Rc<Statement>>,
    pub terminator: Option<Rc<Terminator>>,
}

impl BasicBlockBuilder {
    fn new() -> Self {
        Default::default()
    }

    pub fn set_name(&mut self, name: String) {
        self.name = Some(name);
    }

    pub fn add_phi(&mut self, phi: Rc<Phi>) {
        self.phis.push(phi);
    }

    pub fn add_statement(&mut self, statement: Rc<Statement>) {
        self.statements.push(statement);
    }

    pub fn set_terminator(&mut self, terminator: Rc<Terminator>) {
        self.terminator.replace(terminator);
    }
}