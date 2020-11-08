//! Parsing tools
use crate::util::parsing::ident;
use nom::{bytes::complete::tag, combinator::map, sequence::pair, IResult};

pub fn local(code: &str) -> IResult<&str, String> {
    map(pair(tag("%"), ident), |(_, name)| name)(code)
}
