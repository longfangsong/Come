use crate::ir::parsing::ParsingContext;
use nom::{error::ErrorKind, IResult};

#[derive(Debug)]
pub enum Error<'a> {
    Unknown(()),
    ParseStrError(&'a str),
    ParseContextError(ParsingContext<'a>),
}

impl<'a> nom::error::ParseError<ParsingContext<'a>> for Error<'a> {
    fn from_error_kind(input: ParsingContext<'a>, _kind: ErrorKind) -> Self {
        Error::ParseContextError(input)
    }

    fn append(_input: ParsingContext<'_>, _kind: ErrorKind, other: Self) -> Self {
        other
    }
}

impl<'a> From<nom::Err<nom::error::Error<&'a str>>> for Error<'a> {
    fn from(e: nom::Err<nom::error::Error<&'a str>>) -> Self {
        if let nom::Err::Error(e) = e {
            Error::ParseStrError(e.input)
        } else {
            unimplemented!()
        }
    }
}

impl<'a> Into<nom::Err<Error<'a>>> for Error<'a> {
    fn into(self) -> nom::Err<Error<'a>> {
        nom::Err::Error(self)
    }
}
