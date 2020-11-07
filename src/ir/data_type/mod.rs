//! This module describe the data types exists in Come IR
mod address;
mod array;
mod integer;
mod struct_type;

use address::Address;
use array::Array;
use derive_more::From;
use enum_dispatch::enum_dispatch;
use integer::Integer;
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
};
use struct_type::Struct;

/// A type of data
#[enum_dispatch]
#[derive(Clone)]
pub enum DataType {
    Address(Address),
    Integer(Integer),
    Array(Array),
    Struct(Struct),
}

// todo: in the future we may want to support 64-bit systems
// then `Address` type might be a certain size of <SystemWordLength>
#[enum_dispatch(DataType)]
pub trait DataTypeExt {
    fn size(&self, data_type_table: &DataTypeTable) -> usize;
}

/// Reference to a `DataType`
#[derive(Clone, From)]
pub enum DataTypeRef {
    Address(Address),
    Integer(Integer),
    Array(Array),
    Struct(String),
}

impl Display for DataType {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DataType::Integer(x) => x.fmt(f),
            DataType::Array(x) => x.fmt(f),
            DataType::Address(x) => x.fmt(f),
            DataType::Struct(x) => x.fmt(f),
        }
    }
}

impl Display for DataTypeRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DataTypeRef::Integer(x) => x.fmt(f),
            DataTypeRef::Array(x) => x.fmt(f),
            DataTypeRef::Address(x) => x.fmt(f),
            DataTypeRef::Struct(x) => x.fmt(f),
        }
    }
}

impl Into<DataTypeRef> for DataType {
    fn into(self) -> DataTypeRef {
        match self {
            DataType::Address(x) => DataTypeRef::Address(x),
            DataType::Integer(x) => DataTypeRef::Integer(x),
            DataType::Array(x) => DataTypeRef::Array(x),
            DataType::Struct(x) => DataTypeRef::Struct(x.name),
        }
    }
}

pub struct DataTypeTable {
    structs: HashMap<String, Struct>,
}

impl DataTypeTable {
    fn new() -> Self {
        DataTypeTable {
            structs: HashMap::new(),
        }
    }
}

impl DataTypeTable {
    fn add_type(&mut self, data_type: DataType) {
        if let DataType::Struct(x) = data_type {
            self.structs.insert(x.name.clone(), x);
        }
    }

    fn get_type(&self, data_type_ref: &DataTypeRef) -> Option<DataType> {
        match data_type_ref {
            DataTypeRef::Address(x) => Some(DataType::Address(*x)),
            DataTypeRef::Integer(x) => Some(DataType::Integer(x.clone())),
            DataTypeRef::Array(x) => Some(DataType::Array(x.clone())),
            DataTypeRef::Struct(x) => self.structs.get(x).map(|it| DataType::Struct(it.clone())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        let i32_type = Integer {
            signed: true,
            bit_width: 32,
        };
        let u64_type = Integer {
            signed: false,
            bit_width: 64,
        };
        let address_type = Address;
        let array_type = Array {
            children_type: Box::new(i32_type.clone().into()),
            length: 5,
        };
        let struct_type = Struct {
            name: "S".to_string(),
            children_type: vec![
                i32_type.clone().into(),
                u64_type.clone().into(),
                address_type.clone().into(),
                array_type.clone().into(),
            ],
        };

        let mut table = DataTypeTable::new();
        table.add_type(struct_type.clone().into());
        let size = struct_type.size(&table);
        assert_eq!(size, 32 + 64 + 32 + 5 * 32)
    }
}
