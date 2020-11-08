//! An arbitrary bit width Integer
use crate::ir::data_type::{DataTypeExt, DataTypeTable};
use nom::{
    branch::alt, bytes::complete::tag, character::complete::digit1, combinator::map,
    sequence::pair, IResult,
};
use std::{fmt, str::FromStr};

/// An arbitrary bit width Integer
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Integer {
    /// whether the integer is signed
    pub signed: bool,
    /// bit width of the integer
    pub bit_width: usize,
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            if self.signed { "i" } else { "u" },
            self.bit_width
        )
    }
}

impl DataTypeExt for Integer {
    fn size(&self, _data_type_table: &DataTypeTable) -> usize {
        self.bit_width
    }
}

pub fn parse(code: &str) -> IResult<&str, Integer> {
    alt((
        map(pair(tag("i"), digit1), |(_, width_str)| Integer {
            signed: true,
            bit_width: usize::from_str(width_str).unwrap(),
        }),
        map(pair(tag("u"), digit1), |(_, width_str)| Integer {
            signed: false,
            bit_width: usize::from_str(width_str).unwrap(),
        }),
    ))(code)
}
