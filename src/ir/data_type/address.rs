//! Definition and parser for [`Address`](crate::ir::data_type::address::Address) type.
use crate::ir::data_type::{DataTypeExt, DataTypeTable};
use nom::{bytes::complete::tag_no_case, combinator::map, IResult};
use std::{fmt, fmt::Display};

/// An address in memory
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Address;

impl Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address")
    }
}

impl DataTypeExt for Address {
    fn size(&self, _data_type_table: &DataTypeTable) -> usize {
        32
    }
}

pub fn parse(code: &str) -> IResult<&str, Address> {
    map(tag_no_case("address"), |_| Address)(code)
}
