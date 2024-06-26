use std::collections::HashMap;

use itertools::Itertools;
use wasm_encoder::{BlockType, Function, Instruction, MemArg, ValType};

use crate::{
    ir::{
        analyzer::BindedControlFlowGraph,
        function::basic_block::BasicBlock,
        quantity::Quantity,
        statement::{
            branch::BranchType,
            calculate::{binary, unary},
            Alloca, Branch, IRStatement,
        },
        FunctionDefinition, FunctionHeader, RegisterName,
    },
    utility::data_type::{Integer, Type},
};

use super::control_flow::{CFSelector, CFSelectorSegment, ControlFlowElement};

fn lower_type(t: &Type) -> ValType {
    match t {
        crate::utility::data_type::Type::Integer(Integer { signed, width }) => {
            match (signed, width) {
                (true, 32) => ValType::I32,
                (true, 64) => ValType::I64,
                _ => unimplemented!(),
            }
        }
        _ => unimplemented!(),
    }
}

pub fn lower_function_type(header: &FunctionHeader) -> (Vec<ValType>, Vec<ValType>) {
    let parameter_types = header
        .parameters
        .iter()
        .map(|p| &p.data_type)
        .map(lower_type)
        .collect();
    let return_type = if header.return_type == Type::None {
        vec![]
    } else {
        vec![lower_type(&header.return_type)]
    };
    (parameter_types, return_type)
}

fn put_value_onto_stack(
    value: &Quantity,
    register_name_id_map: &HashMap<RegisterName, u32>,
    result: &mut Function,
    data_type: ValType,
) {
    match value {
        crate::ir::quantity::Quantity::RegisterName(register) => {
            let register_id = register_name_id_map[register];
            result.instruction(&Instruction::LocalGet(register_id));
        }
        crate::ir::quantity::Quantity::NumberLiteral(n) => {
            match data_type {
                ValType::I32 => result.instruction(&Instruction::I32Const(*n as i32)),
                ValType::I64 => result.instruction(&Instruction::I64Const(*n)),
                _ => unimplemented!(),
            };
        }
        crate::ir::quantity::Quantity::GlobalVariableName(_) => unimplemented!(),
    }
}

fn lower_unary_calculate(
    result: &mut Function,
    register_name_id_map: &HashMap<RegisterName, u32>,
    unary_calculate: &unary::UnaryCalculate,
) {
    let data_type = lower_type(&unary_calculate.data_type);
    match data_type {
        ValType::I32 => result.instruction(&Instruction::I32Const(0)),
        ValType::I64 => result.instruction(&Instruction::I64Const(0)),
        _ => unimplemented!(),
    };
    put_value_onto_stack(
        &unary_calculate.operand,
        register_name_id_map,
        result,
        data_type,
    );
    match &unary_calculate.operation {
        unary::UnaryOperation::Neg => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Sub),
            ValType::I64 => result.instruction(&Instruction::I64Sub),
            _ => unimplemented!(),
        },
        unary::UnaryOperation::Not => unimplemented!(),
    };
    let result_register_id = register_name_id_map[&unary_calculate.to];
    result.instruction(&Instruction::LocalSet(result_register_id));
}

fn lower_binary_calculate(
    result: &mut Function,
    register_name_id_map: &HashMap<RegisterName, u32>,
    binary_calculate: &binary::BinaryCalculate,
) {
    let data_type = lower_type(&binary_calculate.data_type);
    put_value_onto_stack(
        &binary_calculate.operand1,
        register_name_id_map,
        result,
        data_type,
    );
    put_value_onto_stack(
        &binary_calculate.operand2,
        register_name_id_map,
        result,
        data_type,
    );
    match &binary_calculate.operation {
        binary::BinaryOperation::Add => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Add),
            ValType::I64 => result.instruction(&Instruction::I64Add),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::LessThan => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32LtS),
            ValType::I64 => result.instruction(&Instruction::I64LtS),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::LessOrEqualThan => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32LeS),
            ValType::I64 => result.instruction(&Instruction::I64LeS),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::GreaterThan => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32GtS),
            ValType::I64 => result.instruction(&Instruction::I64GtS),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::GreaterOrEqualThan => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32GeS),
            ValType::I64 => result.instruction(&Instruction::I64GeS),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::Equal => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Eq),
            ValType::I64 => result.instruction(&Instruction::I64Eq),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::NotEqual => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Ne),
            ValType::I64 => result.instruction(&Instruction::I64Ne),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::Sub => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Sub),
            ValType::I64 => result.instruction(&Instruction::I64Sub),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::Or => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Or),
            ValType::I64 => result.instruction(&Instruction::I64Or),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::Xor => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Xor),
            ValType::I64 => result.instruction(&Instruction::I64Xor),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::And => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32And),
            ValType::I64 => result.instruction(&Instruction::I64And),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::LogicalShiftLeft => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Shl),
            ValType::I64 => result.instruction(&Instruction::I64Shl),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::LogicalShiftRight => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32ShrU),
            ValType::I64 => result.instruction(&Instruction::I64ShrU),
            _ => unimplemented!(),
        },
        binary::BinaryOperation::AthematicShiftRight => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32ShrS),
            ValType::I64 => result.instruction(&Instruction::I64ShrS),
            _ => unimplemented!(),
        },
    };
    let result_register_id = register_name_id_map[&binary_calculate.to];
    result.instruction(&Instruction::LocalSet(result_register_id));
}

fn lower_statement(
    result: &mut Function,
    register_name_id_map: &HashMap<RegisterName, u32>,
    offset_table: &HashMap<RegisterName, i32>,
    statement: &IRStatement,
    bb_id: usize,
    cfe_root: &ControlFlowElement,
    register_type: &HashMap<RegisterName, Type>,
    cfg: &BindedControlFlowGraph,
    move_stack_pointer: i32,
) {
    let current = cfe_root.find_node(bb_id).unwrap();
    match statement {
        IRStatement::UnaryCalculate(unary_calculate) => {
            lower_unary_calculate(result, register_name_id_map, unary_calculate)
        }
        IRStatement::BinaryCalculate(binary_calculate) => {
            lower_binary_calculate(result, register_name_id_map, binary_calculate)
        }
        IRStatement::Branch(branch_statement) => {
            // currently there exists 3 kinds of branch target:
            // 1. if-else: this kind is already folded into `if-else`
            // 2. loop, branch to loop header
            // 3. loop, branch out of loop=
            if current.is_if_condition() {
                // in such case, the branch target is already folded into `if-else`
                // thus, both branch targets are nested into if-else, and since we generate if-else
                // block directly after this statement, we just need to generate the condition and put
                // it onto the stack
                generate_if_condition(
                    branch_statement,
                    register_type,
                    register_name_id_map,
                    result,
                );
                result.instruction(&Instruction::If(BlockType::Empty));
            } else {
                let success_target = cfg.basic_block_index_by_name(&branch_statement.success_label);
                let success_target_selector = cfe_root.find_node(success_target).unwrap();
                let failure_target = cfg.basic_block_index_by_name(&branch_statement.failure_label);
                let failure_target_selector = cfe_root.find_node(failure_target).unwrap();
                if let Some(levels) = success_target_selector.levels_before(&current) {
                    // branch back on success
                    result.instruction(&Instruction::BrIf(levels as u32));
                    if let Some(levels) = failure_target_selector.levels_before(&current) {
                        // branch back, maybe into a different loop
                        result.instruction(&Instruction::BrIf(levels as u32));
                    }
                    // or just fallthrough
                } else {
                    // fallthrough on success
                    // if failure is branch back, swap success and failure
                    if let Some(levels) = failure_target_selector.levels_before(&current) {
                        result.instruction(&Instruction::I32Eqz);
                        result.instruction(&Instruction::BrIf(levels as u32));
                    }
                    // or both are fallthrough
                }
            }
        }
        IRStatement::Jump(jump_statement) => {
            let jump_target = cfg.basic_block_index_by_name(&jump_statement.label);
            let jump_target_selector = cfe_root.find_node(jump_target).unwrap();
            println!("{:?}", cfe_root);
            println!("{jump_target_selector}, {current}");
            if let Some(levels) = jump_target_selector.levels_before(&current) {
                println!("{levels}");
                result.instruction(&Instruction::Br(levels as u32));
            }
        }
        IRStatement::Ret(ret) => {
            if move_stack_pointer != 0 {
                result.instruction(&Instruction::GlobalGet(0));
                result.instruction(&Instruction::I32Const(move_stack_pointer));
                result.instruction(&Instruction::I32Sub);
                result.instruction(&Instruction::GlobalSet(0));
            }
            if let Some(return_value) = &ret.value {
                match return_value {
                    Quantity::RegisterName(register) => {
                        let register_id = register_name_id_map[register];
                        result.instruction(&Instruction::LocalGet(register_id));
                    }
                    Quantity::GlobalVariableName(_) => unimplemented!(),
                    Quantity::NumberLiteral(n) => {
                        // fixme: `n`'s data type
                        result.instruction(&Instruction::I32Const(*n as _));
                    }
                }
            }
            result.instruction(&Instruction::Return);
        }
        IRStatement::Load(load) => {
            result.instruction(&Instruction::GlobalGet(0));
            let offset = match &load.from {
                Quantity::RegisterName(register) => offset_table.get(register).unwrap(),
                Quantity::GlobalVariableName(_) => unimplemented!(),
                Quantity::NumberLiteral(_) => unimplemented!(),
            };
            result.instruction(&Instruction::I32Const(*offset));
            result.instruction(&Instruction::I32Sub);
            result.instruction(&Instruction::I32Load(MemArg {
                offset: 0,
                align: 0,
                memory_index: 0,
            }));
            let load_target = register_name_id_map.get(&load.to).unwrap();
            result.instruction(&Instruction::LocalSet(*load_target));
        }
        IRStatement::Store(store) => {
            // todo: handle data types
            result.instruction(&Instruction::GlobalGet(0));
            let offset = match &store.target {
                Quantity::RegisterName(register) => offset_table.get(register).unwrap(),
                Quantity::GlobalVariableName(_) => unimplemented!(),
                Quantity::NumberLiteral(_) => unimplemented!(),
            };
            result.instruction(&Instruction::I32Const(*offset));
            result.instruction(&Instruction::I32Sub);
            match &store.source {
                Quantity::RegisterName(register) => {
                    // fixme: register maybe another memory Address
                    let register_id = register_name_id_map.get(register).unwrap();
                    result.instruction(&Instruction::LocalGet(*register_id));
                }
                Quantity::NumberLiteral(n) => {
                    result.instruction(&Instruction::I32Const(*n as _));
                }
                Quantity::GlobalVariableName(_) => unimplemented!(),
            }
            result.instruction(&Instruction::I32Store(MemArg {
                offset: 0,
                align: 0,
                memory_index: 0,
            }));
        }
        IRStatement::Alloca(_) => (/* already handled in alloca_stack */),

        IRStatement::Phi(_) => unimplemented!(),
        IRStatement::Call(_) => unimplemented!(),
        IRStatement::LoadField(_) => unimplemented!(),
        IRStatement::SetField(_) => unimplemented!(),
    }
}

fn generate_if_condition(
    branch_statement: &Branch,
    register_type: &HashMap<RegisterName, Type>,
    register_name_id_map: &HashMap<RegisterName, u32>,
    result: &mut Function,
) {
    let data_type = decide_branch_operation_type(branch_statement, register_type);
    put_value_onto_stack(
        &branch_statement.operand1,
        register_name_id_map,
        result,
        data_type,
    );
    put_value_onto_stack(
        &branch_statement.operand2,
        register_name_id_map,
        result,
        data_type,
    );
    match branch_statement.branch_type {
        BranchType::EQ => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Eq),
            ValType::I64 => result.instruction(&Instruction::I64Eq),
            _ => unimplemented!(),
        },
        BranchType::NE => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32Ne),
            ValType::I64 => result.instruction(&Instruction::I64Ne),
            _ => unimplemented!(),
        },
        BranchType::LT => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32LtS),
            ValType::I64 => result.instruction(&Instruction::I64LtS),
            _ => unimplemented!(),
        },
        BranchType::GE => match data_type {
            ValType::I32 => result.instruction(&Instruction::I32GeS),
            ValType::I64 => result.instruction(&Instruction::I64GeS),
            _ => unimplemented!(),
        },
    };
}

fn decide_branch_operation_type(
    branch_statement: &Branch,
    register_type: &HashMap<RegisterName, Type>,
) -> ValType {
    let operand1 = &branch_statement.operand1;
    let operand2 = &branch_statement.operand2;
    let data_type = match (operand1, operand2) {
        (Quantity::RegisterName(register), _) | (_, Quantity::RegisterName(register)) => {
            &register_type[register]
        }
        (Quantity::NumberLiteral(_), Quantity::NumberLiteral(_)) => &Type::Integer(Integer {
            signed: true,
            width: 64,
        }),
        _ => unimplemented!(),
    };
    lower_type(data_type)
}

fn lower_basic_block(
    result: &mut Function,
    bb_id: usize,
    block: &BasicBlock,
    binded_cfg: &BindedControlFlowGraph,
    register_name_id_map: &HashMap<RegisterName, u32>,
    offset_table: &HashMap<RegisterName, i32>,
    cfe_root: &ControlFlowElement,
    register_type: &HashMap<RegisterName, Type>,
    move_stack_pointer: i32,
) {
    for statement in &block.content {
        lower_statement(
            result,
            register_name_id_map,
            offset_table,
            statement,
            bb_id,
            cfe_root,
            register_type,
            binded_cfg,
            move_stack_pointer,
        )
    }
}

fn lower_control_flow_element(
    result: &mut Function,
    body: &[BasicBlock],
    element: &ControlFlowElement,
    current: CFSelector,
    binded_cfg: &BindedControlFlowGraph,
    register_name_id_map: &HashMap<RegisterName, u32>,
    offset_table: &HashMap<RegisterName, i32>,
    cfe_root: &ControlFlowElement,
    register_type: &HashMap<RegisterName, Type>,
    move_stack_pointer: i32,
) {
    match element {
        ControlFlowElement::Block { content } => {
            result.instruction(&Instruction::Block(BlockType::Empty));
            for (i, block) in content.iter().enumerate() {
                let mut new_selector = current.clone();
                new_selector.push_back(CFSelectorSegment::ContentAtIndex(i));
                lower_control_flow_element(
                    result,
                    body,
                    block,
                    new_selector,
                    binded_cfg,
                    register_name_id_map,
                    offset_table,
                    cfe_root,
                    register_type,
                    move_stack_pointer,
                );
            }
            result.instruction(&Instruction::End);
        }
        ControlFlowElement::If {
            condition,
            on_success,
            on_failure,
        } => {
            let mut new_selector = current.clone();
            new_selector.push_back(CFSelectorSegment::IfCondition);
            lower_control_flow_element(
                result,
                body,
                condition,
                new_selector,
                binded_cfg,
                register_name_id_map,
                offset_table,
                cfe_root,
                register_type,
                move_stack_pointer,
            );
            // result.instruction(&Instruction::If(BlockType::Empty));
            for (i, success_block) in on_success.iter().enumerate() {
                let mut new_selector = current.clone();
                new_selector.push_back(CFSelectorSegment::IndexInSuccess(i));
                lower_control_flow_element(
                    result,
                    body,
                    success_block,
                    new_selector,
                    binded_cfg,
                    register_name_id_map,
                    offset_table,
                    cfe_root,
                    register_type,
                    move_stack_pointer,
                );
            }
            if !on_failure.is_empty() {
                result.instruction(&Instruction::Else);
                for (i, failure_block) in on_failure.iter().enumerate() {
                    let mut new_selector = current.clone();
                    new_selector.push_back(CFSelectorSegment::IndexInFailure(i));
                    lower_control_flow_element(
                        result,
                        body,
                        failure_block,
                        new_selector,
                        binded_cfg,
                        register_name_id_map,
                        offset_table,
                        cfe_root,
                        register_type,
                        move_stack_pointer,
                    );
                }
            }
            result.instruction(&Instruction::End);
        }
        ControlFlowElement::Loop { content } => {
            result.instruction(&Instruction::Loop(BlockType::Empty));
            for (i, block) in content.iter().enumerate() {
                let mut new_selector = current.clone();
                new_selector.push_back(CFSelectorSegment::ContentAtIndex(i));
                lower_control_flow_element(
                    result,
                    body,
                    block,
                    new_selector,
                    binded_cfg,
                    register_name_id_map,
                    offset_table,
                    cfe_root,
                    register_type,
                    move_stack_pointer,
                );
            }
            result.instruction(&Instruction::End);
        }
        ControlFlowElement::BasicBlock { id } => {
            lower_basic_block(
                result,
                *id,
                &body[*id],
                binded_cfg,
                register_name_id_map,
                offset_table,
                cfe_root,
                register_type,
                move_stack_pointer,
            );
        }
    }
}

pub fn alloca_stack(function: &FunctionDefinition) -> (HashMap<RegisterName, i32>, i32) {
    let alloca_statements = function
        .content
        .iter()
        .flat_map(|it| it.content.iter())
        .flat_map(IRStatement::try_as_alloca);
    let mut offset_table = HashMap::new();
    let mut current_offset = 0;
    for Alloca { to, alloc_type } in alloca_statements {
        let bytes = match alloc_type {
            Type::Address => 4i32,
            Type::Integer(Integer { width, .. }) => *width as i32 / 8,
            Type::StructRef(_) => unimplemented!(),
            Type::None => unreachable!(),
        };
        offset_table.insert(to.clone(), current_offset + bytes);
        current_offset += bytes;
    }
    (offset_table, current_offset)
}

pub fn lower_function_body(
    function: &FunctionDefinition,
    control_flow_root: &ControlFlowElement,
    binded_cfg: &BindedControlFlowGraph,
) -> Function {
    let parameters = function
        .header
        .parameters
        .iter()
        .map(|parameter| (parameter.name.clone(), parameter.data_type.clone()))
        .collect_vec();
    let (offset_table, move_stack_pointer) = alloca_stack(function);
    let locals = parameters
        .into_iter()
        .chain(
            function
                .content
                .iter()
                .flat_map(|block| block.created_registers())
                .filter(|(register, _)| !offset_table.contains_key(register)),
        )
        .collect_vec();
    let body = &function.content;
    let locals_name_type_map: HashMap<RegisterName, Type> = locals.iter().cloned().collect();
    let register_name_id_map: HashMap<RegisterName, u32> = locals
        .iter()
        .enumerate()
        .map(|(a, (b, _))| (b.clone(), a as u32))
        .collect();
    let mut result = Function::new_with_locals_types(locals.iter().map(|(_, t)| lower_type(t)));
    if move_stack_pointer != 0 {
        result.instruction(&Instruction::GlobalGet(0));
        result.instruction(&Instruction::I32Const(move_stack_pointer));
        result.instruction(&Instruction::I32Add);
        result.instruction(&Instruction::GlobalSet(0));
    }
    for (i, control_flow_element) in control_flow_root
        .block_content()
        .unwrap()
        .iter()
        .enumerate()
    {
        lower_control_flow_element(
            &mut result,
            body,
            control_flow_element,
            CFSelector::from_segment(CFSelectorSegment::ContentAtIndex(i)),
            binded_cfg,
            &register_name_id_map,
            &offset_table,
            control_flow_root,
            &locals_name_type_map,
            move_stack_pointer,
        );
    }
    if move_stack_pointer != 0 {
        result.instruction(&Instruction::GlobalGet(0));
        result.instruction(&Instruction::I32Const(move_stack_pointer));
        result.instruction(&Instruction::I32Sub);
        result.instruction(&Instruction::GlobalSet(0));
    }
    match function.header.return_type {
        Type::Integer(Integer {
            signed: true,
            width: 32,
        }) => {
            result.instruction(&Instruction::I32Const(0));
            result.instruction(&Instruction::Return);
        }
        Type::Integer(Integer { .. }) => unimplemented!(),
        Type::StructRef(_) => unimplemented!(),
        Type::Address => unimplemented!(),
        Type::None => (/* pass */),
    }
    result.instruction(&Instruction::End);
    result
}
