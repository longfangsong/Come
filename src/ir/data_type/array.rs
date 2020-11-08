//! An array of other type
use crate::{
    ir::data_type::{address, integer, DataTypeExt, DataTypeRef, DataTypeTable},
    util::{parsing, parsing::integer},
};
use nom::{
    branch::alt,
    bytes::complete::tag,
    combinator::map,
    sequence::{delimited, pair},
    IResult,
};
use std::fmt;

/// An array of other type
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Array {
    /// The content type
    pub children_type: Box<DataTypeRef>,
    /// The length of the array
    pub length: usize,
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.children_type, self.length)
    }
}

impl DataTypeExt for Array {
    fn size(&self, data_type_table: &DataTypeTable) -> usize {
        self.length
            * data_type_table
                .get_type(&self.children_type)
                .unwrap()
                .size(data_type_table)
    }
}

pub fn higher_than_array(code: &str) -> IResult<&str, DataTypeRef> {
    alt((
        map(address::parse, DataTypeRef::Address),
        map(integer::parse, DataTypeRef::Integer),
        map(parsing::ident, DataTypeRef::Struct),
    ))(code)
}

pub fn parse(code: &str) -> IResult<&str, Array> {
    map(
        pair(higher_than_array, delimited(tag("["), integer, tag("]"))),
        |(children_type, length)| Array {
            children_type: Box::new(children_type),
            length: length as usize,
        },
    )(code)
}
