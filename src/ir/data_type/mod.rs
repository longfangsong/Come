//! This module describe the data types exists in Come IR
mod address;
mod array;
mod integer;
mod space;
mod struct_type;

use crate::ir::data_type::space::Space;
use address::Address;
use array::Array;
use derive_more::{From, TryInto};
use integer::Integer;

use crate::ir::parsing::{lift, Error, ParsingContext};
use nom::{branch::alt, combinator::map, IResult};
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};
pub use struct_type::Struct;

/// A type of data
#[derive(Clone, From, TryInto, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub enum DataType {
    Address(Address),
    Integer(Integer),
    Array(Array),
    Struct(Struct),
    Space(Space),
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Integer(x) => x.fmt(f),
            DataType::Array(x) => x.fmt(f),
            DataType::Address(x) => x.fmt(f),
            DataType::Struct(x) => x.fmt(f),
            DataType::Space(x) => x.fmt(f),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataTypeTable {
    structs: HashMap<String, DataType>,
}

impl DataTypeTable {
    pub fn new() -> Self {
        DataTypeTable {
            structs: HashMap::new(),
        }
    }
}

impl DataTypeTable {
    fn add_type(&mut self, data_type: DataType) {
        if let DataType::Struct(x) = data_type {
            let name = x.name.clone();
            let result = DataType::Struct(x);
            self.structs.insert(name, result);
        }
    }

    fn get_type(&self, name: &str) -> Option<DataType> {
        self.structs.get(name).cloned()
    }
}

pub fn parse(context: ParsingContext) -> IResult<ParsingContext, DataType, Error> {
    alt((
        map(lift(integer::parse), DataType::Integer),
        map(lift(space::parse), DataType::Space),
        map(lift(address::parse), DataType::Address),
        map(array::parse, DataType::Array),
        map(struct_type::parse, DataType::Struct),
    ))(context)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::parsing::ParsingContext;

    #[test]
    fn test_parse() {
        let code = "i32";
        let context = ParsingContext::from_str(code);
        let (rest_context, result) = parse(context).unwrap();
        assert_eq!(
            result,
            DataType::Integer(Integer {
                signed: true,
                bit_width: 32,
            })
        );

        let i32_type = result;
        let code = "%S = type { i32, i32 }";
        let context = ParsingContext::new(code, rest_context.data_type_table);
        let (rest_context, result) = parse(context).unwrap();
        assert_eq!(
            result,
            DataType::Struct(Struct {
                name: "S".to_string(),
                children_type: vec![i32_type.clone(), i32_type],
            })
        );

        let old_result = result;
        let code = "S";
        let context = ParsingContext::new(code, rest_context.data_type_table);
        let (rest_context, result) = parse(context).unwrap();
        assert_eq!(result, old_result);

        let code = "S[32]";
        let context = ParsingContext::new(code, rest_context.data_type_table);
        let (_rest_context, result) = parse(context).unwrap();
        assert_eq!(
            result,
            DataType::Array(Array {
                children_type: Box::new(old_result),
                length: 32,
            })
        );
    }
}
