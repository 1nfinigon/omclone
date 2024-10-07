use crate::nn;
use crate::nonnan::NonNan;
use crate::search_state::State;
use crate::sim::BasicInstr;
use atomic_float::AtomicF32;
use eyre::{OptionExt, Result};
use once_cell::sync::OnceCell;
use rand::prelude::*;
use std::sync::atomic::{self, AtomicU32};

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ChildId(u32);

/// Represents a node in one of its three states:
/// - unexpanded (nothing is known about this node yet): OnceCell is not set,
///   and not currently being initialized
/// - expanding (a thread is fleshing out the node and evaluating the nn):
///   OnceCell is being initialized (locked)
/// - expanded (terminal OR non-terminal): OnceCell is set
///
/// This type is intended to be "small" -- approx 2 words.
struct Node(OnceCell<Box<NodeData>>);

impl Node {
    const fn new_unexpanded() -> Self {
        Self(OnceCell::new())
    }

    /// Returns `true` if this thread succeeded in the expansion of this node;
    /// false if another thread did the initialization (or is still doing
    /// initialization, in which case this function will block). If `false`,
    /// then it is guaranteed that `f` was not called. It is also guaranteed
    /// that after this call, `data` will return `Some`.
    fn data_expanding_if_still_unexpanded<F: FnOnce() -> Result<NodeData>>(
        &self,
        f: F,
    ) -> Result<(&NodeData, bool)> {
        let mut this_thread_expanded = false;
        let data = self.0.get_or_try_init(|| -> Result<_> {
            this_thread_expanded = true;
            Ok(Box::new(f()?))
        })?;
        Ok((data, this_thread_expanded))
    }

    /// Returns `None` if unexpanded.
    fn data(&self) -> Option<&NodeData> {
        self.0.get().map(|b| b.as_ref())
    }

    fn visits(&self) -> u32 {
        self.data().map_or(0, |data| data.visits())
    }
}

/// Contains data about an expanded terminal OR non-terminal node.
/// This type is intended to be mutable with only a shared reference.
/// All relevant fields are inherently atomic/sync. Compound operations might be
/// racy but aren't intended to affect search too much.
///
/// This type is not intended to be "small" memory-wise.
struct NodeData {
    // TODO: remove this and use utility/value_sum being NaN instead.
    terminal: bool,

    children: [Node; BasicInstr::N_TYPES],

    /// raw NN utility (for non-leaf nodes), or final evaluation (for leaf
    /// nodes)
    /// = U-value
    utility: f32,

    /// = P-values
    policy: [f32; BasicInstr::N_TYPES],

    /// = N
    visits: AtomicU32,

    /// = Q-value * N
    value_sum: AtomicF32,
}

impl NodeData {
    fn new_terminal(value: f32) -> Self {
        Self {
            terminal: true,
            children: [const { Node::new_unexpanded() }; BasicInstr::N_TYPES],
            utility: value,
            policy: [0.; BasicInstr::N_TYPES],
            visits: 0.into(),
            value_sum: 0.0.into(),
        }
    }

    fn new_nonterminal(nn_utility: f32, nn_policy: [f32; BasicInstr::N_TYPES]) -> Self {
        Self {
            terminal: false,
            children: [const { Node::new_unexpanded() }; BasicInstr::N_TYPES],
            utility: nn_utility,
            policy: nn_policy,
            visits: 0.into(),
            value_sum: 0.0.into(),
        }
    }

    fn is_terminal(&self) -> bool {
        self.terminal
    }

    /// NN utility, if non-terminal. Terminal valuation otherwise
    fn raw_utility(&self) -> f32 {
        self.utility
    }

    fn value(&self) -> f32 {
        if self.terminal {
            self.utility
        } else {
            let visits = self.visits();
            if visits == 0 {
                self.utility
            } else {
                // TODO: fix the race with visits by changing value_sum to just value
                self.value_sum.load(atomic::Ordering::Relaxed) / (visits as f32)
            }
        }
    }

    fn visits(&self) -> u32 {
        self.visits.load(atomic::Ordering::Relaxed)
    }

    fn get_child(&self, child_id: ChildId) -> &Node {
        &self.children[child_id.0 as usize]
    }

    fn puct(&self, child_id: ChildId, default_value_for_unexpanded_child: f32) -> NonNan {
        let child = self.get_child(child_id);
        let policy = self.policy[child_id.0 as usize];
        let child_value = child
            .data()
            .map_or(default_value_for_unexpanded_child, |child| child.value());
        let visits = self.visits.load(atomic::Ordering::Relaxed);
        let prior = policy * (visits as f32).sqrt() / (1. + child.visits() as f32);
        NonNan::new(child_value + 1.4 * prior).unwrap()
    }

    fn incr_visits_and_utility(&self, utility: f32) {
        self.visits.fetch_add(1, atomic::Ordering::Relaxed);
        if !self.is_terminal() {
            self.value_sum.fetch_add(utility, atomic::Ordering::Relaxed);
        }
    }
}

pub struct TreeSearch {
    root: State,
    root_node: Node,
    sum_depth: usize,
    max_depth: u32,
    dirichlet_distr: rand_distr::Dirichlet<f32>,
    tracy_client: tracy_client::Client,
}

impl TreeSearch {
    pub fn new(root: State, tracy_client: tracy_client::Client) -> Self {
        Self {
            root,
            root_node: Node::new_unexpanded(),
            sum_depth: 0,
            max_depth: 0,
            dirichlet_distr: rand_distr::Dirichlet::new(&[1.0; BasicInstr::N_TYPES]).unwrap(),
            tracy_client,
        }
    }

    pub fn search_once<RngT: Rng>(&mut self, rng: &mut RngT, model: &nn::Model) -> Result<()> {
        let span = self
            .tracy_client
            .clone()
            .span(tracy_client::span_location!("search once"), 0);

        let mut state = self.root.clone();
        let mut node = &self.root_node;

        // Contains all nodes from the root, including the newly expanded node
        // that we stopped descending at.
        let mut path: Vec<&NodeData> = Vec::new();

        let leaf_utility = loop {
            let is_root = path.len() == 0;

            let (node_data, this_thread_expanded) =
                node.data_expanding_if_still_unexpanded(|| {
                    // if this callback was called, then we are the lucky ones who
                    // get to expand this previously unexpanded node

                    if let Some(win) = state.evaluate_final_state() {
                        span.emit_text("leaf: terminal node");

                        Ok(NodeData::new_terminal(win))
                    } else {
                        let _span = self.tracy_client.clone().span(
                            tracy_client::span_location!("leaf: unexpanded nonterminal node"),
                            0,
                        );

                        let next_arm_index = state.instr_buffer.len();
                        let next_arm_pos = state.world.arms[next_arm_index].pos;
                        let (x, y) = nn::features::normalize_position(next_arm_pos)
                            .ok_or_eyre("arm out of nn bounds")?;

                        let mut eval = model.forward(&state.nn_features, x, y, is_root)?;

                        if is_root {
                            // add Dirichlet noise
                            // TODO: for this problem, we should stretch out the
                            // Dirichlet noise to more than just the root.
                            // TODO: filter to only valid moves
                            let noise = self.dirichlet_distr.sample(rng);
                            for (policy, noise) in eval.policy.iter_mut().zip(noise) {
                                const EPS: f32 = 0.25;
                                *policy = (1. - EPS) * *policy + EPS * noise;
                            }
                        }

                        Ok(NodeData::new_nonterminal(eval.win, eval.policy))
                    }
                })?;

            path.push(node_data);

            if this_thread_expanded {
                // We were the ones to expand, so stop descending here.
                break node_data.value();
            }

            // Otherwise, the node was already expanded.

            if node_data.is_terminal() {
                span.emit_text("leaf: terminal node");

                break node_data.value();
            }

            // The node is not a leaf. We need to keep descending.

            let _span = self
                .tracy_client
                .clone()
                .span(tracy_client::span_location!("real node"), 0);

            let next_updates = state.next_updates().ok().unwrap();

            let default_value_for_unexpanded_child = {
                let first_play_urgency_reduction = if is_root {
                    assert!(path.len() == 1);
                    // root has no first play urgency reduction if
                    // Dirichlet noise is disabled
                    // TODO: dirichlet noise
                    0.
                } else {
                    0.2 * node_data
                        .policy
                        .iter()
                        .enumerate()
                        .filter_map(|(i, p)| {
                            if node_data.get_child(ChildId(i as u32)).visits() > 0 {
                                Some(p)
                            } else {
                                None
                            }
                        })
                        .sum::<f32>()
                        .sqrt()
                };
                node_data.value() - first_play_urgency_reduction
            };

            // pick a child based on PUCT
            let child_id = (0..next_updates.len().try_into().unwrap())
                .map(ChildId)
                .max_by_key(|&child_id| {
                    node_data.puct(child_id, default_value_for_unexpanded_child)
                })
                .unwrap();

            // move to child, update the state
            node = node_data.get_child(child_id);
            let update = next_updates.get(child_id.0 as usize).unwrap();
            state.update(*update);
        };

        span.emit_value(path.len() as u64);

        {
            let _span = self
                .tracy_client
                .clone()
                .span(tracy_client::span_location!("backprop"), 0);

            // backpropagate
            for node_data in path.iter().rev() {
                node_data.incr_visits_and_utility(leaf_utility);
            }
            let this_depth = path.len().saturating_sub(1);
            self.sum_depth += this_depth;
            self.max_depth = self.max_depth.max(this_depth as u32);
        }

        Ok(())
    }

    pub fn sum_visits(&self) -> u32 {
        self.root_node.visits()
    }

    pub fn avg_depth(&self) -> f32 {
        (self.sum_depth as f32) / (self.sum_visits() as f32)
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth
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
    pub root_raw_utility: f32,
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
        let root_node_data = self
            .root_node
            .data()
            .expect("Need tree to be expanded at least once to get updates");
        let updates_with_stats = self
            .root
            .next_updates()
            .ok()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(child_id, &instr)| {
                let child_id = ChildId(child_id.try_into().unwrap());
                let child = root_node_data.get_child(child_id);
                UpdateWithStats {
                    instr,
                    is_terminal: child.data().is_some_and(|child| child.is_terminal()),
                    value: NonNan::new(
                        child
                            .data()
                            .map_or(root_node_data.value(), |child| child.value()),
                    )
                    .unwrap(),
                    visits: child.visits(),
                }
            })
            .collect();
        NextUpdatesWithStats {
            root_value: root_node_data.value(),
            root_raw_utility: root_node_data.raw_utility(),
            updates_with_stats,
            avg_depth: self.avg_depth(),
            max_depth: self.max_depth(),
        }
    }
}
