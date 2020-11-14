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
    collections::{HashMap, HashSet},
    fmt::{self, Display, Formatter},
};
use struct_type::Struct;

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
    fn new() -> Self {
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
            self.structs.insert(name, result.clone());
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
        map(struct_type::parse, DataType::Struct),
        map(array::parse, DataType::Array),
    ))(context)
}

// #[cfg(test)]
// mod tests {
//     #[test]
//     fn test_parse() {
//         let code = "i32";
//         let context =
//     }
// }
