use crate::ir::data_type::DataTypeTable;
use crate::ir::parsing::Error;
use nom::IResult;
use crate::ir::function_definition::FunctionDefinition;
use std::rc::Rc;
use crate::ir::basic_block::{BasicBlock, BasicBlockBuilder};
use crate::ir::statements::TerminatorBuilder;

#[derive(Clone, Debug)]
pub struct UncompletedBasicBlock {
    block_builder: BasicBlockBuilder,
    terminator_builder: Rc<dyn TerminatorBuilder>,
}

#[derive(Clone, Debug)]
pub struct ParsingContext<'a> {
    pub code: &'a str,
    pub data_type_table: DataTypeTable,
    pub parsed_functions: Vec<Rc<FunctionDefinition>>,
    pub completed_basic_blocks: Vec<Rc<BasicBlock>>,
    pub uncompleted_basic_blocks: Vec<UncompletedBasicBlock>,
}

impl<'a> PartialEq for ParsingContext<'a> {
    fn eq(&self, other: &Self) -> bool {
        false
    }
}

impl<'a> ParsingContext<'a> {
    pub fn new(code: &'a str,
               data_type_table: DataTypeTable,
               parsed_functions: Vec<Rc<FunctionDefinition>>) -> Self {
        Self {
            code,
            data_type_table,
            parsed_functions,
            completed_basic_blocks: vec![],
            uncompleted_basic_blocks: vec![],
        }
    }

    pub fn from_str(s: &'a str) -> Self {
        Self::new(s, DataTypeTable::new(), vec![])
    }
}

pub fn lift<'a, T, F>(
    mut parser: F,
) -> impl FnMut(ParsingContext<'a>) -> IResult<ParsingContext<'a>, T, Error>
    where
        F: FnMut(&'a str) -> IResult<&'a str, T>,
{
    move |context: ParsingContext<'a>| -> IResult<ParsingContext<'a>, T, Error> {
        let (code, data_type_table, parsed_functions, completed_basic_blocks, uncompleted_basic_blocks) =
            (context.code, context.data_type_table, context.parsed_functions, context.completed_basic_blocks, context.uncompleted_basic_blocks);
        parser(code)
            .map(move |(rest_code, result)| {
                (
                    ParsingContext {
                        code: rest_code,
                        data_type_table,
                        parsed_functions,
                        completed_basic_blocks,
                        uncompleted_basic_blocks,
                    },
                    result,
                )
            })
            .map_err(|it| Error::from(it).into())
    }
}
