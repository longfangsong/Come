//! A user defined struct
use crate::{
    ir::{
        data_type::{DataType, DataTypeTable},
        util::parsing::local,
    },
    util::parsing::in_multispace,
};

use crate::{
    ir::{
        data_type,
        parsing::{lift, Error, ParsingContext},
    },
    util::parsing,
};
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::streaming::space0,
    combinator::map,
    multi::many1,
    sequence::{delimited, pair, tuple},
    IResult,
};
use std::{convert::TryInto, fmt, rc::Rc};
use nom::character::complete::multispace0;

/// A user defined struct
#[derive(Clone, Debug, Hash, Ord, PartialOrd, PartialEq, Eq)]
pub struct Struct {
    /// name of the struct
    pub name: String,
    /// children types
    pub children_type: Vec<DataType>,
}

impl fmt::Display for Struct {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fields: String = self
            .children_type
            .iter()
            .map(|it| format!("  {}", it))
            .collect::<Vec<String>>()
            .join(",\n");
        write!(f, "{} {{\n{}}}", self.name, fields)
    }
}

fn parse_definition(context: ParsingContext) -> IResult<ParsingContext, Struct, Error> {
    map(
        pair(
            lift(tuple((
                local,
                space0,
                tag("="),
                space0,
                tag("type"),
                space0,
            ))),
            delimited(
                lift(in_multispace(tag("{"))),
                many1(map(pair(data_type::parse, lift(pair(tag(","), multispace0))), |x| x.0)),
                lift(in_multispace(tag("}"))),
            ),
        ),
        |((name, _, _, _, _, _), children_type)| Struct {
            name,
            children_type,
        },
    )(context)
}

pub fn parse(context: ParsingContext) -> IResult<ParsingContext, Struct, Error> {
    if let Ok((mut rest_context, result)) = parse_definition(context.clone()) {
        rest_context.data_type_table.add_type(result.clone().into());
        Ok((rest_context, result))
    } else {
        let (code, table) = (context.code, context.data_type_table);
        let (rest_code, name) = parsing::ident(code).map_err(|it| Error::from(it).into())?;
        let result = table.get_type(&name).ok_or_else(|| {
            Error::ParseContextError(ParsingContext {
                code: rest_code,
                data_type_table: table.clone(),
            })
                .into()
        })?;
        Ok((
            ParsingContext {
                code: rest_code,
                data_type_table: table,
            },
            result.try_into().unwrap(),
        ))
    }
}
