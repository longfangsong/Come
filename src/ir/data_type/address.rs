use crate::ir::data_type::{DataTypeExt, DataTypeTable};
use std::{fmt, fmt::Display};

/// An address in memory
#[derive(Clone, Copy)]
pub struct Address;

impl Display for Address {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Address")
    }
}

impl DataTypeExt for Address {
    fn size(&self, _data_type_table: &DataTypeTable) -> usize {
        32
    }
}
