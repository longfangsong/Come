use crate::{
    backend::riscv::{function::FunctionCompileContext, register_assign::RegisterAssign},
    ir::{quantity::Quantity, statement::Call},
};

pub fn emit_code(call: &Call, ctx: &mut FunctionCompileContext) -> String {
    let Call {
        to,
        name,
        data_type,
        params,
    } = call;
    // handle builtin functions
    if name == "load_u32" {
        load_u32(to, &params[0], ctx)
    } else if name == "store_u32" {
        store_u32(&params[0], &params[1], ctx)
    } else {
        todo!()
    }
}

fn store_u32(to_address: &Quantity, value: &Quantity, ctx: &mut FunctionCompileContext) -> String {
    let mut result = String::new();
    match value {
        Quantity::RegisterName(logical_register) => {
            let assign = ctx.local_assign.get(logical_register).unwrap();
            match assign {
                RegisterAssign::Register(physical_register) => {
                    result.push_str(&format!("    mv a1, {}\n", physical_register));
                }
                RegisterAssign::StackRef(offset) => {
                    result.push_str(&format!("    lw a1, {}(sp)\n", offset));
                }
                RegisterAssign::StackValue(offset) => {
                    result.push_str(&format!("    lw a1, {}(sp)\n", offset));
                }
                RegisterAssign::MultipleRegisters(_) => todo!(),
            }
        }
        Quantity::GlobalVariableName(_) => todo!(),
        Quantity::NumberLiteral(constant) => result.push_str(&format!("    li a1, {}\n", constant)),
    }
    match to_address {
        Quantity::RegisterName(to_address_register) => {
            let assign = ctx.local_assign.get(to_address_register).unwrap();
            match assign {
                RegisterAssign::Register(physical_register) => {
                    result.push_str(&format!("    mv a0, {}\n", physical_register));
                }
                RegisterAssign::StackRef(offset) => {
                    result.push_str(&format!("    lw a0, {}(sp)\n", offset));
                }
                RegisterAssign::StackValue(offset) => {
                    result.push_str(&format!("    lw a0, {}(sp)\n", offset));
                }
                RegisterAssign::MultipleRegisters(_) => todo!(),
            }
        }
        Quantity::GlobalVariableName(_) => todo!(),
        Quantity::NumberLiteral(constant) => result.push_str(&format!("    li a0, {}\n", constant)),
    }
    result.push_str("    sw a1, 0(a0)\n");
    result
}

fn load_u32(
    to: &Option<crate::ir::RegisterName>,
    from_address: &Quantity,
    ctx: &mut FunctionCompileContext,
) -> String {
    let mut result = String::new();
    match from_address {
        Quantity::RegisterName(register) => {
            let register_assign = ctx.local_assign.get(register).unwrap();
            let load_addr = match register_assign {
                RegisterAssign::Register(register) => format!("    mv a0, {}\n", register),
                RegisterAssign::StackRef(offset) => format!("    lw a0, {}(sp)\n", offset),
                RegisterAssign::StackValue(offset) => format!("    lw a0, {}(sp)\n", offset),
                RegisterAssign::MultipleRegisters(_) => todo!(),
            };
            result.push_str(&load_addr);
        }
        Quantity::NumberLiteral(constant) => {
            result.push_str(&format!("    li a0, {}\n", constant));
        }
        Quantity::GlobalVariableName(_) => todo!(),
    }
    result.push_str(&format!("    lw a0, 0(a0)\n"));
    if let Some(to_register) = to {
        let register_assign = ctx.local_assign.get(to_register).unwrap();
        match register_assign {
            RegisterAssign::Register(register) => {
                result.push_str(&format!("    mv {}, a0\n", register))
            }
            RegisterAssign::StackRef(offset) => {
                result.push_str(&format!("    sw a0, {}(sp)\n", offset));
            }
            RegisterAssign::StackValue(offset) => {
                result.push_str(&format!("    sw a0, {}(sp)\n", offset));
            }
            RegisterAssign::MultipleRegisters(_) => todo!(),
        };
    }
    result
}
