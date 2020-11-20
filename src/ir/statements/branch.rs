use crate::ir::{basic_block::BasicBlock, value::Value};
use std::rc::{Rc, Weak};
use crate::ir::statements::{TerminatorBuilder, Terminator};
use crate::ir::basic_block::BasicBlockBuilder;
use std::mem;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BranchType {
    EQ,
    NE,
    LT,
    GE,
}

#[derive(Clone, Debug)]
pub struct Branch {
    pub branch_type: BranchType,
    pub operand1: Rc<Value>,
    pub operand2: Rc<Value>,
    pub success: Weak<BasicBlock>,
    pub failure: Weak<BasicBlock>,
}

impl PartialEq for Branch {
    fn eq(&self, other: &Self) -> bool {
        false
    }
}

impl Eq for Branch {}

#[derive(Clone, Debug)]
enum BranchParent {
    Built(Rc<BasicBlock>),
    Builder(Rc<BasicBlockBuilder>),
}

#[derive(Clone, Debug)]
pub struct BranchBuilder {
    branch_type: BranchType,
    operand1: Rc<Value>,
    operand2: Rc<Value>,
    success_name: String,
    success: Weak<BasicBlock>,
    failure_name: String,
    failure: Weak<BasicBlock>,
}

impl BranchBuilder {
    fn new(branch_type: BranchType, operand1: Rc<Value>, operand2: Rc<Value>, success_name: String, failure_name: String) -> Self {
        Self {
            branch_type,
            operand1,
            operand2,
            success_name,
            success: Weak::new(),
            failure_name,
            failure: Weak::new(),
        }
    }
}

impl TerminatorBuilder for BranchBuilder {
    fn on_other_basic_block_done(&mut self, builder: Rc<BasicBlock>) -> Option<Terminator> {
        if let Some(name) = &builder.name {
            if &self.success_name == name {
                self.success = Rc::downgrade(&builder);
            }
            if &self.failure_name == name {
                self.failure = Rc::downgrade(&builder);
            }
        }
        if self.failure.upgrade().is_some() && self.success.upgrade().is_some() {
            Some(Branch {
                branch_type: self.branch_type,
                operand1: self.operand1.clone(),
                operand2: self.operand2.clone(),
                success: mem::take(&mut self.success),
                failure: mem::take(&mut self.failure),
            }.into())
        } else {
            None
        }
    }
}
