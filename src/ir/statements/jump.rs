use crate::ir::basic_block::BasicBlock;
use std::rc::{Rc, Weak};
use crate::ir::statements::{TerminatorBuilder, Terminator};

#[derive(Clone, Debug)]
pub struct Jump {
    to: Weak<BasicBlock>,
}

impl PartialEq for Jump {
    fn eq(&self, other: &Self) -> bool {
        false
    }
}

impl Eq for Jump {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JumpBuilder {
    waiting_for_name: String,
}

impl TerminatorBuilder for JumpBuilder {
    fn on_other_basic_block_done(&mut self, builder: Rc<BasicBlock>) -> Option<Terminator> {
        if let Some(name) = &builder.name {
            if name == &self.waiting_for_name {
                Some(Jump { to: Rc::downgrade(&builder) }.into())
            } else {
                None
            }
        } else {
            None
        }
    }
}
