mod control_flow_loop;
use super::IsAnalyzer;
use crate::{
    ir::{self, editor::action::Action, statement::IRStatement, FunctionDefinition},
    utility::{self},
};
use bimap::BiMap;
pub use control_flow_loop::{Loop, LoopContent};
use itertools::Itertools;
use petgraph::{
    algo::{
        self,
        dominators::{simple_fast, Dominators},
    },
    prelude::*,
};
use std::{
    cell::{OnceCell, Ref, RefCell},
    collections::HashMap,
};

/// A node in the control flow graph.
/// For preventing using wrong index to access the graph,
/// we introduced a new type here instead of using naive usize.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Node(usize);

impl Node {
    pub fn to_block_index(self) -> usize {
        self.0
    }
}

impl From<NodeIndex<usize>> for Node {
    fn from(node_index: NodeIndex<usize>) -> Self {
        Self(node_index.index())
    }
}

impl From<usize> for Node {
    fn from(node_index: usize) -> Self {
        Self(node_index)
    }
}

#[derive(Debug)]
struct ControlFlowGraphContent {
    graph: DiGraph<(), (), usize>,
    frontiers: HashMap<Node, Vec<Node>>,
    bb_name_node_map: BiMap<Node, String>,
    dominators: Dominators<NodeIndex<usize>>,
    from_to_may_pass_blocks: RefCell<HashMap<(Node, Node), Vec<Node>>>,
}

impl ControlFlowGraphContent {
    fn new(function_definition: &FunctionDefinition) -> Self {
        let mut graph = DiGraph::<(), (), usize>::default();
        let bb_name_index_map: BiMap<_, _> = function_definition
            .content
            .iter()
            .enumerate()
            .map(|(index, bb)| (Node(index), bb.name.as_ref().unwrap().clone()))
            .collect();
        for (node, bb) in function_definition.content.iter().enumerate() {
            let last_statement = bb.content.last().unwrap();
            match last_statement {
                IRStatement::Branch(branch) => {
                    let success_node = *bb_name_index_map
                        .get_by_right(&branch.success_label)
                        .unwrap();
                    let failure_node = *bb_name_index_map
                        .get_by_right(&branch.failure_label)
                        .unwrap();
                    graph.extend_with_edges([(node, success_node.0), (node, failure_node.0)]);
                }
                IRStatement::Jump(jump) => {
                    let to_node = *bb_name_index_map.get_by_right(&jump.label).unwrap();
                    graph.extend_with_edges([(node, to_node.0)]);
                }
                IRStatement::Ret(_) => {
                    graph.extend_with_edges([(node, function_definition.content.len())]);
                }
                _ => unreachable!(),
            }
        }
        let dorminators = simple_fast(&graph, 0.into());
        let graph = remove_unreachable_nodes(graph);
        let frontiers = utility::graph::dominance_frontiers(&dorminators, &graph)
            .into_iter()
            .map(|(k, v)| {
                (
                    Node(k.index()),
                    v.into_iter().map(NodeIndex::index).map(Node).collect(),
                )
            })
            .collect();
        Self {
            graph,
            frontiers,
            dominators: dorminators,
            bb_name_node_map: bb_name_index_map,
            from_to_may_pass_blocks: RefCell::new(HashMap::new()),
        }
    }

    fn dominance_frontier(&self, node: Node) -> &[Node] {
        self.frontiers.get(&node).unwrap()
    }

    fn dominates_calculate(&self, visiting: Node, visited: &mut Vec<Node>) {
        if visited.contains(&visiting) {
            return;
        }
        visited.push(visiting);
        let mut imm_dominates = self.immediately_dominates(visiting);
        imm_dominates.retain(|it| !visited.contains(it));
        for it in imm_dominates {
            self.dominates_calculate(it, visited);
        }
    }

    fn immediately_dominates(&self, node: Node) -> Vec<Node> {
        self.dominators
            .immediately_dominated_by(node.0.into())
            .map(|it| Node(it.index()))
            .collect()
    }

    fn dominates(&self, node: Node) -> Vec<Node> {
        let mut visited = Vec::new();
        self.dominates_calculate(node, &mut visited);
        visited
    }

    fn node_by_basic_block_name(&self, name: &str) -> Node {
        *self.bb_name_node_map.get_by_right(name).unwrap()
    }

    fn basic_block_name_by_node(&self, index: Node) -> &str {
        self.bb_name_node_map.get_by_left(&index).unwrap()
    }

    fn may_pass_blocks(&self, from: Node, to: Node) -> Ref<Vec<Node>> {
        let mut from_to_passed_blocks = self.from_to_may_pass_blocks.borrow_mut();
        from_to_passed_blocks.entry((from, to)).or_insert_with(|| {
            let mut passed_nodes = algo::all_simple_paths::<Vec<_>, _>(
                &self.graph,
                from.0.into(),
                to.0.into(),
                0,
                None,
            )
            .flatten()
            .map(|it| Node(it.index()))
            .collect_vec();
            passed_nodes.sort();
            passed_nodes.dedup();
            passed_nodes
        });
        drop(from_to_passed_blocks);
        Ref::map(self.from_to_may_pass_blocks.borrow(), |it| {
            it.get(&(from, to)).unwrap()
        })
    }
}

/// [`ControlFlowGraph`] is for analyzing the control flow graph and related information of a function.
#[derive(Default, Debug)]
pub struct ControlFlowGraph {
    content: OnceCell<ControlFlowGraphContent>,
    loop_item: OnceCell<Loop>,
}

impl ControlFlowGraph {
    /// Creates a new [`ControlFlowGraph`].
    pub fn new() -> Self {
        Self {
            content: OnceCell::new(),
            loop_item: OnceCell::new(),
        }
    }
    fn content(&self, content: &FunctionDefinition) -> &ControlFlowGraphContent {
        self.content
            .get_or_init(|| ControlFlowGraphContent::new(content))
    }
    fn dominance_frontier(&self, content: &ir::FunctionDefinition, node: Node) -> &[Node] {
        self.content(content).dominance_frontier(node)
    }
    fn basic_block_index_by_name(&self, content: &ir::FunctionDefinition, name: &str) -> Node {
        self.content(content).node_by_basic_block_name(name)
    }
    fn basic_block_name_by_index(&self, content: &ir::FunctionDefinition, index: Node) -> &str {
        self.content(content).basic_block_name_by_node(index)
    }
    fn may_pass_blocks(
        &self,
        content: &ir::FunctionDefinition,
        from: Node,
        to: Node,
    ) -> Ref<Vec<Node>> {
        self.content(content).may_pass_blocks(from, to)
    }
    fn dominate(&self, content: &ir::FunctionDefinition, node: Node) -> Vec<Node> {
        self.content(content).dominates(node)
    }
    fn loops(&self, content: &FunctionDefinition) -> &Loop {
        self.loop_item.get_or_init(|| {
            let graph = &self.content(content).graph;
            let nodes: Vec<_> = graph.node_indices().collect();
            Loop::new(graph, &nodes, &[])
        })
    }
}

pub struct BindedControlFlowGraph<'item, 'bind: 'item> {
    pub bind_on: &'bind FunctionDefinition,
    item: &'item ControlFlowGraph,
}

impl<'item, 'bind: 'item> BindedControlFlowGraph<'item, 'bind> {
    /// [Dorminance Frontier](https://en.wikipedia.org/wiki/Dominator_(graph_theory)) of basic block indexed by `node`.
    pub fn dominance_frontier(&self, node: Node) -> &[Node] {
        self.item.dominance_frontier(self.bind_on, node)
    }

    /// Returns the [`Node`] of basic block named `name`.
    pub fn node_by_basic_block_name(&self, name: &str) -> Node {
        self.item.basic_block_index_by_name(self.bind_on, name)
    }

    /// Returns the name of basic block indexed by `node`.
    pub fn basic_block_name_by_node(&self, node: Node) -> &str {
        self.item.basic_block_name_by_index(self.bind_on, node)
    }

    // Returns reference to the basic block indexed by `node`
    pub fn basic_block_by_node(&self, node: Node) -> &ir::BasicBlock {
        &self.bind_on.content[node.0]
    }

    /// Returns the basic blocks that may be passed from `from` to `to`.
    pub fn may_pass_blocks(&self, from: Node, to: Node) -> Ref<Vec<Node>> {
        self.item.may_pass_blocks(self.bind_on, from, to)
    }

    /// Returns the [`Loop`] of the function.
    pub fn loops(&self) -> &Loop {
        self.item.loops(self.bind_on)
    }

    /// Returns the control flow graph.
    fn graph(&self) -> &DiGraph<(), (), usize> {
        &self.item.content(self.bind_on).graph
    }

    /// Returns the [`Node`]s dominated by `node`.
    pub fn dominates(&self, node: Node) -> Vec<Node> {
        self.item.dominate(self.bind_on, node)
    }

    /// Returns the `node`'s predecessor, ie. the nodes which have edges into `node`.
    pub fn predecessors(&self, node: Node) -> impl Iterator<Item = Node> + '_ {
        self.graph()
            .neighbors_directed(node.0.into(), Direction::Incoming)
            .map(|it| Node(it.index()))
    }

    /// Returns the `node`'s successor, ie. the nodes which have edges from `node`.
    pub fn successors(&self, node: Node) -> impl Iterator<Item = Node> + '_ {
        self.graph()
            .neighbors_directed(node.0.into(), Direction::Outgoing)
            .map(|it| Node(it.index()))
    }

    /// Returns the `node`'s predecessor, but not dominated by `node`.
    pub fn not_dominate_successors(&self, node: Node) -> Vec<Node> {
        let successors = self
            .graph()
            .neighbors_directed(node.0.into(), Direction::Incoming)
            .map(|it| Node(it.index()));
        let nodes_dominated = self.dominates(node);
        successors
            .filter(|it| !nodes_dominated.contains(it))
            .collect()
    }
}

impl<'item, 'bind: 'item> IsAnalyzer<'item, 'bind> for ControlFlowGraph {
    type Binded = BindedControlFlowGraph<'item, 'bind>;

    fn on_action(&mut self, _action: &Action) {
        self.content.take();
        self.loop_item.take();
    }

    fn bind(&'item self, content: &'bind ir::FunctionDefinition) -> Self::Binded {
        BindedControlFlowGraph {
            bind_on: content,
            item: self,
        }
    }
}

/// Remove unreachable nodes from a graph.
fn remove_unreachable_nodes(mut graph: DiGraph<(), (), usize>) -> DiGraph<(), (), usize> {
    let mut reachable_nodes = vec![];
    // We start from the node indexed by 0, which represents the entry node for functions.
    let mut dfs = Dfs::new(&graph, 0.into());
    while let Some(node) = dfs.next(&graph) {
        reachable_nodes.push(node);
    }
    graph.retain_nodes(|_, it| reachable_nodes.contains(&it));
    graph
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        ir::{
            function::{basic_block::BasicBlock, test_util::*},
            statement::Ret,
        },
        utility::data_type,
    };

    #[test]
    fn test_loop() {
        let control_flow_graph = ControlFlowGraph::new();
        let function_definition = FunctionDefinition {
            header: ir::FunctionHeader {
                name: "f".to_string(),
                parameters: Vec::new(),
                return_type: data_type::Type::None,
            },
            content: vec![
                BasicBlock {
                    name: Some("bb0".to_string()),
                    content: vec![branch("bb1", "bb2")],
                },
                BasicBlock {
                    name: Some("bb1".to_string()),
                    content: vec![jump("bb3")],
                },
                BasicBlock {
                    name: Some("bb2".to_string()),
                    content: vec![jump("bb6")],
                },
                BasicBlock {
                    name: Some("bb3".to_string()),
                    content: vec![jump("bb4")],
                },
                BasicBlock {
                    name: Some("bb4".to_string()),
                    content: vec![branch("bb5", "bb9")],
                },
                BasicBlock {
                    name: Some("bb5".to_string()),
                    content: vec![branch("bb1", "bb3")],
                },
                BasicBlock {
                    name: Some("bb6".to_string()),
                    content: vec![branch("bb7", "bb8")],
                },
                BasicBlock {
                    name: Some("bb7".to_string()),
                    content: vec![jump("bb2")],
                },
                BasicBlock {
                    name: Some("bb8".to_string()),
                    content: vec![branch("bb7", "bb9")],
                },
                BasicBlock {
                    name: Some("bb9".to_string()),
                    content: vec![Ret { value: None }.into()],
                },
            ],
        };
        let bind = control_flow_graph.bind(&function_definition);
        let loops = bind.loops();
        assert!(loops.content.contains(&LoopContent::Node(0)));
        assert!(loops.content.contains(&LoopContent::Node(9)));
        assert!(loops
            .content
            .iter()
            .any(|it| if let LoopContent::SubLoop(subloop) = it {
                subloop.entries.contains(&1)
            } else {
                false
            }));
        assert!(loops
            .content
            .iter()
            .any(|it| if let LoopContent::SubLoop(subloop) = it {
                subloop.entries.contains(&2)
            } else {
                false
            }));
    }
}
