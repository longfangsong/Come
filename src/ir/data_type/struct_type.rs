//! A user defined struct
use crate::{
    ir::{
        data_type::{parse_ref, DataTypeExt, DataTypeRef, DataTypeTable},
        util::parsing::local,
    },
    util::parsing::in_multispace,
};
use nom::{
    bytes::complete::tag,
    character::complete::space0,
    combinator::map,
    multi::many1,
    sequence::{delimited, pair, tuple},
    IResult,
};
use std::fmt;

/// A user defined struct
#[derive(Clone)]
pub struct Struct {
    /// name of the struct
    pub name: String,
    /// children types
    pub children_type: Vec<DataTypeRef>,
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

impl DataTypeExt for Struct {
    fn size(&self, data_type_table: &DataTypeTable) -> usize {
        self.children_type
            .iter()
            .map(|it| data_type_table.get_type(it).unwrap())
            .map(|it| it.size(data_type_table))
            .sum()
    }
}

pub fn parse(code: &str) -> IResult<&str, Struct> {
    map(
        tuple((
            local,
            space0,
            tag("="),
            space0,
            tag("type"),
            space0,
            delimited(
                in_multispace(tag("{")),
                many1(in_multispace(map(pair(parse_ref, tag(",")), |x| x.0))),
                in_multispace(tag("}")),
            ),
        )),
        |(name, _, _, _, _, _, children_type)| Struct {
            name,
            children_type,
        },
    )(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let result = parse(
            "%S = type {\
        u64,\
        u32,\
        Address,\
        S0,\
        }",
        )
            .unwrap()
            .1;
        assert_eq!(result.children_type.len(), 4);
    }
}
