mod error;

use crate::ir::data_type::DataTypeTable;
pub use error::Error;
use nom::IResult;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParsingContext<'a> {
    pub code: &'a str,
    pub data_type_table: DataTypeTable,
}

impl<'a> ParsingContext<'a> {
    pub fn new(code: &'a str, data_type_table: DataTypeTable) -> Self {
        Self {
            code,
            data_type_table,
        }
    }

    pub fn from_str(s: &'a str) -> Self {
        Self::new(s, DataTypeTable::new())
    }
}

pub fn lift<'a, T, F>(
    mut parser: F,
) -> impl FnMut(ParsingContext<'a>) -> IResult<ParsingContext<'a>, T, Error>
where
    F: FnMut(&'a str) -> IResult<&'a str, T>,
{
    move |context: ParsingContext<'a>| -> IResult<ParsingContext<'a>, T, Error> {
        let (code, data_type_table) = (context.code, context.data_type_table);
        parser(code)
            .map(move |(rest_code, result)| {
                (
                    ParsingContext {
                        code: rest_code,
                        data_type_table,
                    },
                    result,
                )
            })
            .map_err(|it| Error::from(it).into())
    }
}
