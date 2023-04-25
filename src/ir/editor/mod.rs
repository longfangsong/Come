use self::{
    action::{InsertBasicBlock, InsertStatement, IsAction, RemoveStatement, RenameLocal},
    analyzer::{BindedAnalyzer, IsAnalyzer},
};

use super::{
    function::{formalize, FunctionDefinitionIndex},
    quantity::Quantity,
    statement::IRStatement,
    RegisterName,
};

mod action;
/// Analyzers for providing information of an ir function.
pub mod analyzer;
pub use analyzer::Analyzer;
pub struct Editor {
    // todo: remove this pub
    pub content: super::FunctionDefinition,
    pub analyzer: analyzer::Analyzer,
}

impl Editor {
    pub fn new(content: super::FunctionDefinition) -> Self {
        Self {
            content: formalize(content),
            analyzer: analyzer::Analyzer::new(),
        }
    }

    pub fn insert_statement(
        &mut self,
        index: impl Into<FunctionDefinitionIndex>,
        statement: impl Into<IRStatement>,
    ) {
        self.perform_action(InsertStatement::at_index(index, statement));
    }

    pub fn push_front_statement(&mut self, index: usize, statement: impl Into<IRStatement>) {
        self.perform_action(InsertStatement::front_of(index, statement));
    }

    pub fn push_back_statement(&mut self, index: usize, statement: impl Into<IRStatement>) {
        self.perform_action(InsertStatement::back_of(index, statement));
    }

    pub fn remove_statement(&mut self, index: impl Into<FunctionDefinitionIndex>) {
        self.perform_action(RemoveStatement::new(index));
    }

    pub fn remove_statements<T: Into<FunctionDefinitionIndex> + Ord>(
        &mut self,
        indexes: impl IntoIterator<Item = T>,
    ) {
        let mut indexes = indexes.into_iter().collect::<Vec<_>>();
        indexes.sort();
        while !indexes.is_empty() {
            let index = indexes.pop().unwrap();
            self.remove_statement(index);
        }
    }

    pub fn rename_local(&mut self, from: RegisterName, to: impl Into<Quantity>) {
        self.perform_action(RenameLocal::new(from, to));
    }

    fn perform_action(&mut self, action: impl Into<action::Action>) {
        let action = action.into();
        self.analyzer.on_action(&action);
        action.perform_on_function(&mut self.content);
    }

    pub fn binded_analyzer(&self) -> BindedAnalyzer {
        self.analyzer.bind(&self.content)
    }

    pub fn insert_basic_block(&mut self, name: String, index: impl Into<usize>) {
        self.perform_action(InsertBasicBlock::at_index(index, name));
    }

    pub fn create_basic_block(&mut self, name: String) -> usize {
        self.perform_action(InsertBasicBlock::back_of(name));
        self.content.content.len() - 1
    }
}