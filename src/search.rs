use crate::nonnan::NonNan;
use crate::search_state::State;
use crate::sim::BasicInstr;
use crate::Result;
use rand::prelude::*;
use std::borrow::BorrowMut;
use std::collections::{hash_map, HashMap};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd)]
struct ChildId(u32);

struct RealNode {
    children: BTreeMap<ChildId, NodeId>,
    visits: u32,
    wins: f32,
}

impl RealNode {
    fn new() -> Self {
        Self {
            children: BTreeMap::new(),
            visits: 0,
            wins: 0.,
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
}

pub trait ResultPredictor<RngT: Rng>: Send + Sync {
    fn predict<'a>(
        &'a self,
        state: State,
        rng: &mut RngT,
        update_cb: Box<dyn FnMut(&BasicInstr) + 'a>,
    ) -> f32;
}

pub struct PlayoutResultPredictor;

impl<RngT: Rng> ResultPredictor<RngT> for PlayoutResultPredictor {
    fn predict<'a>(
        &'a self,
        mut state: State,
        rng: &mut RngT,
        mut update_cb: Box<dyn FnMut(&BasicInstr) + 'a>,
    ) -> f32 {
        loop {
            match state.next_updates() {
                Err(final_eval) => {
                    return final_eval;
                }
                Ok(next_updates) => {
                    let next_update_idx = rng.gen_range(0..next_updates.len());
                    let next_update = next_updates.get(next_update_idx).unwrap();
                    update_cb(next_update);
                    state.update(*next_update);
                }
            }
        }
    }
}

impl<RngT: Rng> ResultPredictor<RngT> for Box<dyn ResultPredictor<RngT>> {
    fn predict<'a>(
        &'a self,
        state: State,
        rng: &mut RngT,
        update_cb: Box<dyn FnMut(&BasicInstr) + 'a>,
    ) -> f32 {
        (**self).predict(state, rng, update_cb)
    }
}

impl TreeSearch {
    pub fn new(root: State) -> Self {
        Self {
            root,
            nodes: vec![Node::Unexpanded],
            real_nodes: vec![],
            sum_depth: 0,
            max_depth: 0,
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

    fn get_child(&self, real_node_idx: RealNodeId, child_id: ChildId) -> &Node {
        self.real_node(real_node_idx)
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

    fn uct(&self, parent: &RealNode, child: &Node) -> NonNan {
        match child {
            Node::Real(real_child_idx) => {
                let child = self.real_node(*real_child_idx);
                if child.visits == 0 {
                    NonNan::new(f32::INFINITY).unwrap()
                } else {
                    let exploit = child.wins / (child.visits as f32);
                    let explore = ((parent.visits as f32).ln() / (child.visits as f32)).sqrt();
                    NonNan::new(exploit + 2.4 * explore).unwrap()
                }
            }
            Node::Terminal(win) => NonNan::new(*win).unwrap(),
            Node::Unexpanded => NonNan::new(f32::INFINITY).unwrap(),
        }
    }

    pub fn search_once<RngT: Rng, ResultPredictorT: ResultPredictor<RngT>>(
        &mut self,
        rng: &mut RngT,
        result_predictor: &ResultPredictorT,
    ) -> Result<()> {
        self.search_once_with_cb(rng, |_| (), result_predictor)
    }

    pub fn search_once_with_cb<
        RngT: Rng,
        F: FnMut(&BasicInstr),
        ResultPredictorT: ResultPredictor<RngT>,
    >(
        &mut self,
        rng: &mut RngT,
        mut update_cb: F,
        result_predictor: &ResultPredictorT,
    ) -> Result<()> {
        let mut state = self.root.clone();
        let mut node_idx = NodeId(0);
        let mut path = BTreeSet::new();
        let mut path_count = 0;

        let win = loop {
            path.insert(node_idx);
            path_count += 1;
            if path_count >= 500 {
                // assume deadlocked
                break 0.;
            }

            match self.node(node_idx) {
                Node::Terminal(win) => {
                    break *win;
                }
                &Node::Real(real_node_idx) => {
                    let next_updates = state.next_updates().ok().unwrap();

                    // pick a child based on UCT
                    let child_id = (0..next_updates.len().try_into().unwrap())
                        .map(ChildId)
                        .max_by_key(|&child_id| {
                            let child = self.get_child(real_node_idx, child_id);
                            self.uct(self.real_node(real_node_idx), child)
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

                        break result_predictor.predict(state, rng, Box::new(update_cb));
                    }
                }
            }
        };

        // backpropagate
        let mut this_depth = 0u32;
        for &node_idx in path.iter() {
            if let Node::Real(real_node_idx) = self.nodes[node_idx.0 as usize] {
                let real_node = self.real_node_mut(real_node_idx);
                real_node.visits += 1;
                real_node.wins += win;
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
                node.wins / node.visits as f32
            }
            _ => panic!("Need tree to be expanded at least once to get win rate"),
        }
    }

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
                    match self.get_child(real_node_idx, child_id) {
                        Node::Real(real_child_id) => {
                            let child = self.real_node(*real_child_id);
                            (
                                update,
                                NonNan::new(child.wins / child.visits as f32).unwrap(),
                                child.visits,
                            )
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
