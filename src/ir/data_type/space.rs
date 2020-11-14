use crate::util::parsing;
use nom::{
    bytes::complete::{tag, tag_no_case},
    combinator::map,
    sequence::{delimited, pair},
    IResult,
};
use std::{fmt, fmt::Display};

#[derive(Clone, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub struct Space {
    bit_width: usize,
}

impl Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Space({})", self.bit_width)
    }
}

pub fn parse(code: &str) -> IResult<&str, Space> {
    map(
        pair(
            tag_no_case("space"),
            delimited(tag("("), parsing::integer, tag(")")),
        ),
        |(_, bit_width)| Space {
            bit_width: bit_width as usize,
        },
    )(code)
}
