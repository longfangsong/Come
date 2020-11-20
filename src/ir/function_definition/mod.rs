use crate::ir::data_type::DataType;
use crate::ir::basic_block::BasicBlock;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct FunctionDefinition {
    name: String,
    params: Vec<DataType>,
    basic_blocks: Vec<Rc<BasicBlock>>,
}


