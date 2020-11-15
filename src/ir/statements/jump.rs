use crate::ir::basic_block::BasicBlock;
use std::rc::Rc;

pub struct Jump {
    to: Rc<BasicBlock>,
}
