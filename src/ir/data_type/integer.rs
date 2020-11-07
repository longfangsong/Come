use crate::ir::data_type::{DataTypeExt, DataTypeTable};
use std::fmt;

/// An arbitrary bit width Integer
#[derive(Clone)]
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
