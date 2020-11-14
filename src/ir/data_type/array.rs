//! An array of other type
use crate::{
    ir::data_type::{address, integer, DataType, DataTypeTable},
    util::{parsing, parsing::integer},
};

use std::{fmt, rc::Rc};
use crate::ir::parsing::{ParsingContext, Error, lift};
use nom::IResult;
use nom::branch::alt;
use nom::combinator::map;
use crate::ir::data_type::{space, struct_type};
use nom::sequence::{pair, delimited};
use nom::bytes::complete::tag;

/// An array of other type
#[derive(Clone, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub struct Array {
    /// The content type
    pub children_type: Box<DataType>,
    /// The length of the array
    pub length: usize,
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.children_type, self.length)
    }
}

fn higher_than_array(context: ParsingContext) -> IResult<ParsingContext, DataType, Error> {
    alt((
        map(lift(integer::parse), DataType::Integer),
        map(lift(space::parse), DataType::Space),
        map(lift(address::parse), DataType::Address),
        map(struct_type::parse, DataType::Struct),
    ))(context)
}

pub fn parse(context: ParsingContext) -> IResult<ParsingContext, Array, Error> {
    map(
        pair(higher_than_array, lift(delimited(tag("["), integer, tag("]")))),
        |(children_type, length)| Array {
            children_type: Box::new(children_type),
            length: length as usize,
        },
    )(context)
}
