use crate::ir::data_type::{DataTypeExt, DataTypeRef, DataTypeTable};
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
