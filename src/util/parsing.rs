//! Parsing tools
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, digit1, hex_digit1, multispace0},
    combinator::{map, recognize},
    multi::many0,
    sequence::{pair, tuple},
    IResult, Parser,
};

pub fn ident(code: &str) -> IResult<&str, String> {
    map(
        recognize(pair(
            alt((alpha1, tag("_"))),
            many0(alt((alphanumeric1, tag("_")))),
        )),
        |s: &str| s.to_string(),
    )(code)
}

pub fn integer(code: &str) -> IResult<&str, i64> {
    alt((
        map(pair(tag("0x"), hex_digit1), |(_, digits)| {
            i64::from_str_radix(digits, 16).unwrap()
        }),
        map(digit1, |digits| i64::from_str_radix(digits, 10).unwrap()),
    ))(code)
}

pub fn in_multispace<F, I, O, E>(f: F) -> impl FnMut(I) -> IResult<I, O, E>
where
    I: nom::InputTakeAtPosition + Clone,
    <I as nom::InputTakeAtPosition>::Item: nom::AsChar + Clone,
    E: nom::error::ParseError<I>,
    F: Parser<I, O, E>,
{
    map(tuple((multispace0, f, multispace0)), |(_, x, _)| x)
}
