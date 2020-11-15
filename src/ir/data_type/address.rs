//! Definition and parser for [`Address`](crate::ir::data_type::address::Address) type.

use nom::{bytes::complete::tag_no_case, combinator::map, IResult};
use std::{fmt, fmt::Display};

// todo: in the future we may want to support 64-bit systems
// then `Address` type might be a certain size of <SystemWordLength>
/// An address in memory
#[derive(Clone, Copy, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub struct Address;

impl Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address")
    }
}

pub fn parse(code: &str) -> IResult<&str, Address> {
    map(tag_no_case("address"), |_| Address)(code)
}
