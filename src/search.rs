use crate::nn;
use crate::nonnan::NonNan;
use crate::search_state::State;
use crate::sim::BasicInstr;
use eyre::{OptionExt, Result};
use rand::prelude::*;
use std::borrow::BorrowMut;
use std::collections::{hash_map, HashMap};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ChildId(u32);

struct RealNode {
    children: BTreeMap<ChildId, NodeId>,

    /// raw NN utility (for non-leaf nodes), or final evaluation (for leaf
    /// nodes)
    /// = U-value
    utility: f32,

    /// = P-values
    policy: [f32; BasicInstr::N_TYPES],

    /// = N
    visits: u32,

    /// = Q-value * N
    value_sum: f32,
}

impl RealNode {
    fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            utility: 0.,
            policy: [0.; BasicInstr::N_TYPES],
            visits: 0,
            value_sum: 0.,
        }
    }

    fn value(&self) -> f32 {
        if self.visits == 0 {
            self.utility
        } else {
            self.value_sum / (self.visits as f32)
        }
    }
}

struct TerminalNode {
    value: f32,
    visits: u32,
}

impl TerminalNode {
    fn new(value: f32) -> Self {
        Self { value, visits: 0 }
    }
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct NodeId(u32);

enum Node {
    Unexpanded,
    Real(RealNode),
    Terminal(TerminalNode),
}

impl Node {
    fn visits(&self) -> u32 {
        match self {
            Node::Unexpanded => 0,
            Node::Real(real_node) => real_node.visits,
            Node::Terminal(terminal_node) => terminal_node.visits,
        }
    }

    fn incr_visits(&mut self) {
        match self {
            Node::Unexpanded => panic!("can't incr unexpanded node"),
            Node::Real(real_node) => real_node.visits += 1,
            Node::Terminal(terminal_node) => terminal_node.visits += 1,
        }
    }

    fn value(&self) -> Option<f32> {
        match self {
            Node::Real(real_node) => Some(real_node.value()),
            Node::Terminal(terminal_node) => Some(terminal_node.value),
            Node::Unexpanded => None,
        }
    }
}

pub struct TreeSearch {
    root: State,
    nodes: Vec<Node>,
    sum_depth: usize,
    max_depth: u32,
    dirichlet_distr: rand_distr::Dirichlet<f32>,
}

impl TreeSearch {
    pub fn new(root: State) -> Self {
        Self {
            root,
            nodes: vec![Node::Unexpanded],
            sum_depth: 0,
            max_depth: 0,
            dirichlet_distr: rand_distr::Dirichlet::new(&[1.0; BasicInstr::N_TYPES]).unwrap(),
        }
    }

    fn node(&self, node_idx: NodeId) -> &Node {
        &self.nodes[node_idx.0 as usize]
    }

    fn node_mut(&mut self, node_idx: NodeId) -> &mut Node {
        &mut self.nodes[node_idx.0 as usize]
    }

    fn get_child(&self, real_node: &RealNode, child_id: ChildId) -> &Node {
        real_node
            .children
            .get(&child_id)
            .map_or(&Node::Unexpanded, |&node_id| self.node(node_id))
    }

    fn get_or_add_child(&mut self, node_idx: NodeId, child_id: ChildId) -> NodeId {
        let nodes_len = self.nodes.len();
        match &mut self.nodes[node_idx.0 as usize] {
            Node::Real(real_node) => {
                let mut needs_pushing = false;
                let child_node_id = real_node
                    .children
                    .entry(child_id)
                    .or_insert_with(|| {
                        let node_id = NodeId(nodes_len.try_into().unwrap());
                        needs_pushing = true;
                        node_id
                    })
                    .to_owned();
                if needs_pushing {
                    self.nodes.push(Node::Unexpanded);
                }
                child_node_id
            }
            _ => panic!("get_or_add_child called for a non-real node"),
        }
    }

    /// Returns None if the child is unexpanded; in this case the heuristic for
    /// unexpanded nodes should be used.
    fn puct(
        &self,
        parent: &RealNode,
        child_id: ChildId,
        default_value_for_unexpanded_child: f32,
    ) -> NonNan {
        let child = self.get_child(parent, child_id);
        let policy = parent.policy[child_id.0 as usize];
        let child_value = child.value().unwrap_or(default_value_for_unexpanded_child);
        let prior = policy * (parent.visits as f32).sqrt() / (1. + child.visits() as f32);
        NonNan::new(child_value + 1.4 * prior).unwrap()
    }

    pub fn search_once<RngT: Rng>(&mut self, rng: &mut RngT, model: &nn::Model) -> Result<()> {
        let mut state = self.root.clone();
        let mut node_idx = NodeId(0);
        let mut path = Vec::new();

        let leaf_utility = loop {
            path.push(node_idx);
            if path.len() >= 500 {
                // assume deadlocked
                break 0.;
            }

            let is_root = node_idx == NodeId(0);

            match self.node(node_idx) {
                Node::Terminal(terminal_node) => {
                    break terminal_node.value;
                }
                Node::Real(real_node) => {
                    let next_updates = state.next_updates().ok().unwrap();

                    let default_value_for_unexpanded_child = {
                        let first_play_urgency_reduction = if is_root {
                            assert!(path.len() == 1);
                            // root has no first play urgency reduction if
                            // Dirichlet noise is disabled
                            // TODO: dirichlet noise
                            0.
                        } else {
                            0.2 * real_node
                                .policy
                                .iter()
                                .enumerate()
                                .filter(|(i, p)| {
                                    self.get_child(real_node, ChildId(*i as u32)).visits() > 0
                                })
                                .map(|(_, p)| p)
                                .sum::<f32>()
                                .sqrt()
                        };
                        real_node.value() - first_play_urgency_reduction
                    };

                    // pick a child based on PUCT
                    let child_id = (0..next_updates.len().try_into().unwrap())
                        .map(ChildId)
                        .max_by_key(|&child_id| {
                            self.puct(real_node, child_id, default_value_for_unexpanded_child)
                        })
                        .unwrap();

                    // move to child, update the state
                    node_idx = self.get_or_add_child(node_idx, child_id);
                    let update = next_updates.get(child_id.0 as usize).unwrap();
                    state.update(*update);
                }
                Node::Unexpanded => {
                    // expand this node
                    if let Some(win) = state.evaluate_final_state() {
                        *self.node_mut(node_idx) = Node::Terminal(TerminalNode::new(win));
                        break win;
                    } else {
                        let next_arm_index = state.instr_buffer.len();
                        let next_arm_pos = state.world.arms[next_arm_index].pos;
                        let (x, y) = nn::features::normalize_position(next_arm_pos)
                            .ok_or_eyre("arm out of nn bounds")?;

                        let eval = model.forward(&*state.nn_features, x, y, is_root)?;

                        let mut real_node = RealNode::new();
                        real_node.policy = eval.policy;
                        if is_root {
                            // add Dirichlet noise
                            // TODO: for this problem, we should stretch out the
                            // Dirichlet noise to more than just the root.
                            let noise = self.dirichlet_distr.sample(rng);
                            for (policy, noise) in real_node.policy.iter_mut().zip(noise) {
                                const EPS: f32 = 0.25;
                                *policy = (1. - EPS) * *policy + EPS * noise;
                            }
                        }
                        real_node.utility = eval.win;

                        *self.node_mut(node_idx) = Node::Real(real_node);
                        break eval.win;
                    }
                }
            }
        };

        // backpropagate
        let mut this_depth = 0u32;
        for &node_idx in path.iter().rev() {
            self.node_mut(node_idx).incr_visits();
            if let Node::Real(real_node) = self.node_mut(node_idx) {
                real_node.value_sum += leaf_utility;
            }

            this_depth += 1;
        }
        self.sum_depth += this_depth as usize;
        self.max_depth = self.max_depth.max(this_depth);

        Ok(())
    }

    pub fn sum_visits(&self) -> u32 {
        self.node(NodeId(0)).visits()
    }

    pub fn avg_depth(&self) -> f32 {
        (self.sum_depth as f32) / (self.sum_visits() as f32)
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    pub fn win(&self) -> f32 {
        match self.node(NodeId(0)) {
            Node::Real(real_node) => real_node.value(),
            _ => panic!("Need tree to be expanded at least once to get win rate"),
        }
    }
}

#[derive(Debug)]
pub struct UpdateWithStats {
    pub instr: BasicInstr,
    pub is_terminal: bool,
    pub value: NonNan,
    pub visits: u32,
}

#[derive(Debug)]
pub struct NextUpdatesWithStats {
    pub root_value: f32,
    pub updates_with_stats: Vec<UpdateWithStats>,
    pub avg_depth: f32,
    pub max_depth: u32,
}

impl NextUpdatesWithStats {
    pub fn best_update(&self) -> BasicInstr {
        self.updates_with_stats
            .iter()
            .max_by_key(|s| (s.visits, s.value))
            .map(|s| s.instr)
            .unwrap()
    }
}

impl TreeSearch {
    pub fn next_updates_with_stats(&self) -> NextUpdatesWithStats {
        match self.node(NodeId(0)) {
            Node::Real(root_real_node) => {
                let updates_with_stats = self
                    .root
                    .next_updates()
                    .ok()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(child_id, &instr)| {
                        let child_id = ChildId(child_id.try_into().unwrap());
                        let child = self.get_child(root_real_node, child_id);
                        UpdateWithStats {
                            instr,
                            is_terminal: matches!(child, Node::Terminal(_)),
                            value: NonNan::new(child.value().unwrap_or(root_real_node.value()))
                                .unwrap(),
                            visits: child.visits(),
                        }
                    })
                    .collect();
                NextUpdatesWithStats {
                    root_value: root_real_node.value(),
                    updates_with_stats,
                    avg_depth: self.avg_depth(),
                    max_depth: self.max_depth(),
                }
            }
            _ => panic!("Need tree to be expanded at least once to get updates"),
        }
    }
}
