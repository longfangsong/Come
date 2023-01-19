use crate::{backend::riscv::instruction::ParsedParam, utility::parsing};
use bitvec::prelude::*;
use nom::{bytes::complete::tag, combinator::map, sequence::delimited, IResult};

use super::IsParamTransformer;

#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub struct BitAt(u8);

impl BitAt {
    pub const fn new(index: u8) -> Self {
        Self(index)
    }

    pub fn parse(code: &str) -> IResult<&str, Self> {
        map(
            delimited(tag("bit_at("), parsing::integer, tag(")")),
            Self::new,
        )(code)
    }

    pub const fn bit_count(&self) -> usize {
        1
    }
}

impl IsParamTransformer for BitAt {
    fn param_to_instruction_part(&self, _address: u64, param: &ParsedParam) -> BitVec<u32> {
        // // it is ok to use `as u32` here, see
        // // https://doc.rust-lang.org/reference/expressions/operator-expr.html#type-cast-expressions
        let param = param.unwrap_immediate() as u32;
        let param_bits = &param.view_bits::<Lsb0>();
        let mut instruction_part = BitVec::new();
        instruction_part.push(param_bits[self.0 as usize]);
        instruction_part
    }

    fn update_param(&self, instruction_part: &BitSlice<u32>, param: &mut ParsedParam) {
        if let ParsedParam::Immediate(param_value) = param {
            let mut param_bits_store = *param_value as u32;
            let param_bits = param_bits_store.view_bits_mut::<Lsb0>();
            param_bits.set(self.0 as usize, instruction_part[0]);
            *param_value = param_bits_store as i32;
        }
    }

    fn default_param(&self) -> ParsedParam {
        ParsedParam::Immediate(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse() {
        let result = BitAt::parse("bit_at(0)").unwrap().1;
        assert_eq!(result, BitAt(0));
        let result = BitAt::parse("bit_at(1)").unwrap().1;
        assert_eq!(result, BitAt(1));
        assert!(BitAt::parse("bits_at(0, 7)").is_err());
    }

    #[test]
    fn param_to_instruction_part() {
        let transformer = BitAt(0);
        let param = ParsedParam::Immediate(0b1010);
        assert_eq!(
            transformer.param_to_instruction_part(0, &param),
            vec![false]
        );
        let transformer = BitAt(1);
        assert_eq!(transformer.param_to_instruction_part(0, &param), vec![true]);
        let transformer = BitAt(7);
        assert_eq!(
            transformer.param_to_instruction_part(0, &param),
            vec![false]
        );
    }

    #[test]
    fn update_param() {
        let transformer = BitAt(0);
        let mut param = ParsedParam::Immediate(0);
        transformer.update_param(&[true], &mut param);
        assert_eq!(param, ParsedParam::Immediate(1));

        let transformer = BitAt(1);
        let mut param = ParsedParam::Immediate(0);
        transformer.update_param(&[false], &mut param);
        assert_eq!(param, ParsedParam::Immediate(0));

        let transformer = BitAt(30);
        let mut param = ParsedParam::Immediate(0);
        transformer.update_param(&[true], &mut param);
        assert_eq!(param, ParsedParam::Immediate(0x40000000));

        let transformer = BitAt(31);
        let mut param = ParsedParam::Immediate(0);
        transformer.update_param(&[true], &mut param);
        assert_eq!(param, ParsedParam::Immediate(-0x8000_0000));
    }
}