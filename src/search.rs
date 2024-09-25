use crate::nn;
use crate::nonnan::NonNan;
use crate::search_state::State;
use crate::sim::BasicInstr;
use eyre::{OptionExt, Result};
use rand::prelude::*;
use std::borrow::BorrowMut;
use std::collections::{hash_map, HashMap};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
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

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
struct NodeId(u32);

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
struct RealNodeId(u32);

enum Node {
    Unexpanded,
    Real(RealNodeId),
    Terminal(f32),
}

pub struct TreeSearch {
    root: State,
    nodes: Vec<Node>,
    real_nodes: Vec<RealNode>,
    sum_depth: usize,
    max_depth: u32,
    model: nn::Model,
    dirichlet_distr: rand_distr::Dirichlet<f32>,
}

impl TreeSearch {
    pub fn new(root: State, model: nn::Model) -> Self {
        Self {
            root,
            nodes: vec![Node::Unexpanded],
            real_nodes: vec![],
            sum_depth: 0,
            max_depth: 0,
            model,
            dirichlet_distr: rand_distr::Dirichlet::new(&[1.0; BasicInstr::N_TYPES]).unwrap(),
        }
    }

    fn node(&self, node_idx: NodeId) -> &Node {
        &self.nodes[node_idx.0 as usize]
    }

    fn node_mut(&mut self, node_idx: NodeId) -> &mut Node {
        &mut self.nodes[node_idx.0 as usize]
    }

    fn real_node(&self, real_node_idx: RealNodeId) -> &RealNode {
        &self.real_nodes[real_node_idx.0 as usize]
    }

    fn real_node_mut(&mut self, real_node_idx: RealNodeId) -> &mut RealNode {
        &mut self.real_nodes[real_node_idx.0 as usize]
    }

    fn visits_f32(&self, node: &Node) -> f32 {
        match node {
            Node::Unexpanded => 0.,
            Node::Real(real_node_idx) => self.real_node(*real_node_idx).visits as f32,
            Node::Terminal(_) => f32::INFINITY,
        }
    }

    fn value(&self, node: &Node) -> Option<f32> {
        match node {
            Node::Real(real_node_idx) => Some(self.real_node(*real_node_idx).value()),
            Node::Terminal(win) => Some(*win),
            Node::Unexpanded => None,
        }
    }

    fn get_child(&self, real_node: &RealNode, child_id: ChildId) -> &Node {
        real_node
            .children
            .get(&child_id)
            .map_or(&Node::Unexpanded, |&node_id| self.node(node_id))
    }

    fn get_or_add_child(&mut self, real_node_idx: RealNodeId, child_id: ChildId) -> NodeId {
        self.real_nodes[real_node_idx.0 as usize]
            .children
            .entry(child_id)
            .or_insert_with(|| {
                let node_id = NodeId(self.nodes.len().try_into().unwrap());
                self.nodes.push(Node::Unexpanded);
                node_id
            })
            .to_owned()
    }

    /// Returns None if the child is unexpanded; in this case the heuristic for
    /// unexpanded nodes should be used.
    fn puct(&self, parent_id: RealNodeId, child_id: ChildId) -> Option<NonNan> {
        let parent = self.real_node(parent_id);
        let child = self.get_child(parent, child_id);
        let policy = parent.policy[child_id.0 as usize];
        let child_value = self.value(child)?;
        let prior = policy * (parent.visits as f32).sqrt() / (1. + self.visits_f32(child));
        Some(NonNan::new(child_value + 1.4 * prior).unwrap())
    }

    pub fn search_once<RngT: Rng>(&mut self, rng: &mut RngT) -> Result<()> {
        self.search_once_with_cb(rng, |_| ())
    }

    pub fn search_once_with_cb<RngT: Rng, F: FnMut(&BasicInstr)>(
        &mut self,
        rng: &mut RngT,
        mut update_cb: F,
    ) -> Result<()> {
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
                Node::Terminal(win) => {
                    break *win;
                }
                &Node::Real(real_node_idx) => {
                    let next_updates = state.next_updates().ok().unwrap();

                    let default_puct_for_unexpanded_child = {
                        let real_node = self.real_node(real_node_idx);
                        let first_play_urgency_reduction = if is_root {
                            assert!(path.len() == 1);
                            // root has no first play urgency reduction
                            // TODO: dirichlet noise
                            0.
                        } else {
                            0.2 * real_node
                                .policy
                                .iter()
                                .enumerate()
                                .filter(|(i, p)| {
                                    self.visits_f32(self.get_child(real_node, ChildId(*i as u32)))
                                        > 0.
                                })
                                .map(|(_, p)| p)
                                .sum::<f32>()
                        };
                        real_node.value() - first_play_urgency_reduction
                    };

                    // pick a child based on PUCT
                    let child_id = (0..next_updates.len().try_into().unwrap())
                        .map(ChildId)
                        .max_by_key(|&child_id| {
                            self.puct(real_node_idx, child_id)
                                .unwrap_or(NonNan::new(default_puct_for_unexpanded_child).unwrap())
                        })
                        .unwrap();

                    // move to child, update the state
                    node_idx = self.get_or_add_child(real_node_idx, child_id);
                    let update = next_updates.get(child_id.0 as usize).unwrap();
                    update_cb(update);
                    state.update(*update);
                }
                Node::Unexpanded => {
                    // expand this node
                    if let Some(win) = state.evaluate_final_state() {
                        *self.node_mut(node_idx) = Node::Terminal(win);
                        break win;
                    } else {
                        let real_node_id = RealNodeId(self.real_nodes.len().try_into().unwrap());
                        self.real_nodes.push(RealNode::new());

                        *self.node_mut(node_idx) = Node::Real(real_node_id);

                        let next_arm_index = state.instr_buffer.len();
                        let next_arm_pos = state.world.arms[next_arm_index].pos;
                        let (x, y) = nn::features::normalize_position(next_arm_pos)
                            .ok_or_eyre("arm out of nn bounds")?;

                        let eval = self.model.forward(&*state.nn_features, x, y, is_root)?;

                        let real_node = self.real_nodes.last_mut().unwrap();
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

                        break eval.win;
                    }
                }
            }
        };

        // backpropagate
        let mut this_depth = 0u32;
        for &node_idx in path.iter().rev() {
            if let Node::Real(real_node_idx) = self.nodes[node_idx.0 as usize] {
                let real_node = self.real_node_mut(real_node_idx);
                real_node.visits += 1;
                real_node.value_sum += leaf_utility;
            }

            this_depth += 1;
        }
        self.sum_depth += this_depth as usize;
        self.max_depth = self.max_depth.max(this_depth);

        Ok(())
    }

    pub fn sum_visits(&self) -> u32 {
        match self.node(NodeId(0)) {
            Node::Real(real_node_idx) => self.real_node(*real_node_idx).visits,
            Node::Unexpanded | Node::Terminal(_) => 0,
        }
    }

    pub fn avg_depth(&self) -> f32 {
        (self.sum_depth as f32) / (self.sum_visits() as f32)
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    pub fn win(&self) -> f32 {
        match self.node(NodeId(0)) {
            Node::Real(real_node_idx) => {
                let node = self.real_node(*real_node_idx);
                node.value()
            }
            _ => panic!("Need tree to be expanded at least once to get win rate"),
        }
    }
}

impl TreeSearch {
    pub fn next_updates_with_stats(&self) -> Vec<(BasicInstr, NonNan, u32)> {
        match self.node(NodeId(0)) {
            &Node::Real(real_node_idx) => self
                .root
                .next_updates()
                .ok()
                .unwrap()
                .iter()
                .enumerate()
                .map(|(child_id, &update)| {
                    let child_id = ChildId(child_id.try_into().unwrap());
                    match self.get_child(self.real_node(real_node_idx), child_id) {
                        Node::Real(real_child_id) => {
                            let child = self.real_node(*real_child_id);
                            (update, NonNan::new(child.value()).unwrap(), child.visits)
                        }
                        Node::Terminal(win) => (update, NonNan::new(*win).unwrap(), u32::MAX),
                        Node::Unexpanded => (update, NonNan::new(0.).unwrap(), 0),
                    }
                })
                .collect(),
            _ => panic!("Need tree to be expanded at least once to get updates"),
        }
    }
}
