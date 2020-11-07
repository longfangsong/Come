//! This module describe the data types exists in Come IR
use std::fmt::{self, Formatter, Display};
use std::collections::HashMap;
use enum_dispatch::enum_dispatch;

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
#[derive(Clone)]
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

/// An address in memory
#[derive(Clone, Copy)]
pub struct Address;

impl Display for Address {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Address")
    }
}

impl DataTypeExt for Address {
    fn size(&self, _data_type_table: &DataTypeTable) -> usize {
        32
    }
}

/// An arbitrary bit width Integer
#[derive(Clone)]
pub struct Integer {
    /// whether the integer is signed
    pub signed: bool,
    /// bit width of the integer
    pub bit_width: usize,
}

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", if self.signed { "i" } else { "u" }, self.bit_width)
    }
}

impl DataTypeExt for Integer {
    fn size(&self, _data_type_table: &DataTypeTable) -> usize {
        self.bit_width
    }
}

/// An array of other type
#[derive(Clone)]
pub struct Array {
    /// The content type
    pub children_type: Box<DataType>,
    /// The length of the array
    pub length: usize,
}

impl fmt::Display for Array {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}[{}]", self.children_type, self.length)
    }
}

impl DataTypeExt for Array {
    fn size(&self, data_type_table: &DataTypeTable) -> usize {
        self.children_type.size(data_type_table) * self.length
    }
}

/// A user defined struct
#[derive(Clone)]
pub struct Struct {
    /// name of the struct
    pub name: String,
    /// children types
    pub children_type: Vec<DataTypeRef>,
}

impl fmt::Display for Struct {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let fields: String = self.children_type.iter()
            .map(|it| format!("  {}", it))
            .collect::<Vec<String>>()
            .join(",\n");
        write!(f, "{} {{\n{}}}", self.name, fields)
    }
}

impl DataTypeExt for Struct {
    fn size(&self, data_type_table: &DataTypeTable) -> usize {
        self.children_type.iter()
            .map(|it| data_type_table.get_type(it).unwrap())
            .map(|it| it.size(data_type_table))
            .sum()
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
    structs: HashMap<String, Struct>
}

impl DataTypeTable {
    fn add_type(&mut self, data_type: &DataType) {
        if let DataType::Struct(x) = data_type {
            self.structs.insert(x.name.clone(), x.clone());
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
