use crate::ir::data_type::{DataTypeExt, DataTypeRef, DataTypeTable};
use std::fmt;

/// An array of other type
#[derive(Clone)]
pub struct Array {
    /// The content type
    pub children_type: Box<DataTypeRef>,
    /// The length of the array
    pub length: usize,
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.children_type, self.length)
    }
}

impl DataTypeExt for Array {
    fn size(&self, data_type_table: &DataTypeTable) -> usize {
        self.length
            * data_type_table
            .get_type(&self.children_type)
            .unwrap()
            .size(data_type_table)
    }
}
