//! An array of other type
use crate::ir::data_type::{address, integer, DataType};

use crate::{
    ir::{
        data_type::{space, struct_type},
        parsing::{lift, Error, ParsingContext},
    },
    util::parsing,
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
        pair(
            higher_than_array,
            lift(delimited(tag("["), parsing::integer, tag("]"))),
        ),
        |(children_type, length)| Array {
            children_type: Box::new(children_type),
            length: length as usize,
        },
    )(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        data_type::{integer::Integer, struct_type::Struct, DataTypeTable},
        parsing::ParsingContext,
    };

    #[test]
    fn test_parse() {
        let i32_type = Integer {
            signed: true,
            bit_width: 32,
        };
        let code = "i32[32]";
        let mut context = ParsingContext {
            code,
            data_type_table: DataTypeTable::new(),
        };
        let (_rest_context, result) = parse(context.clone()).unwrap();
        assert_eq!(
            result,
            Array {
                children_type: Box::new(DataType::Integer(i32_type.clone())),
                length: 32,
            }
        );

        let struct_type = Struct {
            name: "S".to_string(),
            children_type: vec![i32_type.clone().into(), i32_type.clone().into()],
        };
        context.code = "S[32]";
        context
            .data_type_table
            .add_type(DataType::Struct(struct_type.clone()));
        let (_rest_context, result) = parse(context).unwrap();
        assert_eq!(
            result,
            Array {
                children_type: Box::new(DataType::Struct(struct_type)),
                length: 32,
            }
        );
    }
}
