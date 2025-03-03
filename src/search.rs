//! Parallel MCTS search implementation

use crate::eval::AsyncEvaluator;
use crate::search_state::State;
use crate::sim::BasicInstr;
use atomic_float::AtomicF32;
use eyre::Result;
use once_cell::sync::OnceCell;
use ordered_float::NotNan;
use std::sync::atomic::{self, AtomicU32, AtomicUsize};

/// How many virtual visits should an in-progress thread count for (in an
/// attempt to avoid multiple threads descending exactly the same line)?
const VIRTUAL_LOSS_MULTIPLIER: u32 = 1;

trait VisitCount {
    fn visits(node_data: &NodeData) -> u32;
}
struct WithVirtualLoss;
impl VisitCount for WithVirtualLoss {
    fn visits(node_data: &NodeData) -> u32 {
        node_data.visits.load(atomic::Ordering::Relaxed)
            + node_data.virtual_loss_count.load(atomic::Ordering::Relaxed) * VIRTUAL_LOSS_MULTIPLIER
    }
}
struct AssertZeroVirtualLoss;
impl VisitCount for AssertZeroVirtualLoss {
    fn visits(node_data: &NodeData) -> u32 {
        assert_eq!(
            node_data.virtual_loss_count.load(atomic::Ordering::Relaxed),
            0
        );
        node_data.visits.load(atomic::Ordering::Relaxed)
    }
}

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
    fn data_expanding_if_still_unexpanded<V, F: FnOnce() -> Result<(NodeData, V)>>(
        &self,
        f: F,
    ) -> Result<(&NodeData, Option<V>)> {
        let mut this_thread_expanded = None;
        let node_data = self.0.get_or_try_init(|| -> Result<_> {
            let (node_data, value) = f()?;
            this_thread_expanded = Some(value);
            Ok(Box::new(node_data))
        })?;
        Ok((node_data, this_thread_expanded))
    }

    /// Returns `None` if unexpanded.
    fn data(&self) -> Option<&NodeData> {
        self.0.get().map(|b| b.as_ref())
    }

    fn visits<V: VisitCount>(&self) -> u32 {
        self.data().map_or(0, |data| data.visits::<V>())
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

    virtual_loss_count: AtomicU32,

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
            virtual_loss_count: 0.into(),
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
            virtual_loss_count: 0.into(),
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

    fn value<V: VisitCount>(&self) -> f32 {
        if self.terminal {
            self.utility
        } else {
            let visits = self.visits::<V>();
            if visits == 0 {
                self.utility
            } else {
                // TODO: fix the race with visits by changing value_sum to just value
                self.value_sum.load(atomic::Ordering::Relaxed) / (visits as f32)
            }
        }
    }

    /// Asserts that virtual loss is zero.
    fn visits<V: VisitCount>(&self) -> u32 {
        V::visits(self)
    }

    fn incr_virtual_loss_count(&self) {
        self.virtual_loss_count
            .fetch_add(1, atomic::Ordering::Relaxed);
    }

    fn decr_virtual_loss_count(&self) {
        let old_value = self
            .virtual_loss_count
            .fetch_sub(1, atomic::Ordering::Relaxed);
        assert!(old_value != 0, "BUG: virtual loss count underflow");
    }

    fn get_child(&self, child_id: ChildId) -> &Node {
        &self.children[child_id.0 as usize]
    }

    fn puct<V: VisitCount>(
        &self,
        child_id: ChildId,
        default_value_for_unexpanded_child: f32,
    ) -> NotNan<f32> {
        let child = self.get_child(child_id);
        let policy = self.policy[child_id.0 as usize];
        let child_value = child
            .data()
            .map_or(default_value_for_unexpanded_child, |child| {
                child.value::<V>()
            });
        let visits = self.visits::<V>();
        let prior = policy * (visits as f32).sqrt() / (1. + child.visits::<V>() as f32);
        NotNan::new(child_value + 1.4 * prior).unwrap()
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
    sum_depth: AtomicUsize,
    max_depth: AtomicU32,
    tracy_client: tracy_client::Client,
}

impl TreeSearch {
    pub fn new(root: State, tracy_client: tracy_client::Client) -> Self {
        Self {
            root,
            root_node: Node::new_unexpanded(),
            sum_depth: 0.into(),
            max_depth: 0.into(),
            tracy_client,
        }
    }

    pub fn clear(&mut self, root: State) {
        self.root = root;
        self.root_node = Node::new_unexpanded();
        self.sum_depth = 0.into();
        self.max_depth = 0.into();
    }
}

enum TreeSearchWorkFibreState {
    Descending,
    //WaitingForExpansion,
    ReachedExpandedLeaf { leaf_utility: f32 },
    Finished,
}

struct TreeSearchWorkFibre<'a> {
    fibre_state: TreeSearchWorkFibreState,
    /// We need to allow `data_expanding_if_still_unexpanded` to take
    /// ownership of `State` if the callback is called, but we need
    /// to retain ownership otherwise. It's hard to convince Rust at
    /// compile-time, so instead do this ownership check at runtime.
    state: Option<Box<State>>,
    /// Contains all nodes that we added virtual loss to.
    virtual_loss_incurred: Vec<&'a NodeData>,
    node: &'a Node,
    tracy_client: tracy_client::Client,
    tree_search: &'a TreeSearch,
}

impl<'a> TreeSearchWorkFibre<'a> {
    fn new(tree_search: &'a TreeSearch) -> Self {
        Self {
            fibre_state: TreeSearchWorkFibreState::Descending,
            state: Some(Box::new(tree_search.root.clone())),
            node: &tree_search.root_node,
            virtual_loss_incurred: Vec::new(),
            tracy_client: tree_search.tracy_client.clone(),
            tree_search,
        }
    }

    fn do_one_descent(&mut self, evaluator: &dyn AsyncEvaluator) -> Result<bool> {
        let span = self
            .tracy_client
            .clone()
            .span(tracy_client::span_location!("do_one_descent"), 0);
        let is_root = self.virtual_loss_incurred.is_empty();

        let (node_data, this_thread_expanded) = {
            let span_need_to_expand = self
                .tracy_client
                .clone()
                .span(tracy_client::span_location!("need to expand?"), 0);

            let (node_data, this_thread_expanded) =
                self.node.data_expanding_if_still_unexpanded(|| {
                    // if this callback was called, then we are the lucky ones who
                    // get to expand this previously unexpanded node

                    let state = self.state.take().unwrap();

                    if let Some(win) = state.evaluate_final_state() {
                        span.emit_text("leaf: terminal node");

                        Ok((NodeData::new_terminal(win), win))
                    } else {
                        let _span = self
                            .tracy_client
                            .clone()
                            .span(tracy_client::span_location!("leaf: expanding node"), 0);

                        let eval_result = evaluator.eval_blocking(*state, is_root)?;

                        Ok((
                            NodeData::new_nonterminal(eval_result.utility, eval_result.policy),
                            eval_result.utility,
                        ))
                    }
                })?;

            if this_thread_expanded.is_some() {
                span_need_to_expand.emit_text("yes");
                span_need_to_expand.emit_color(0x00C000);
            } else {
                span_need_to_expand.emit_text("no (or blocked)");
                span_need_to_expand.emit_color(0xC0C000);
            }

            (node_data, this_thread_expanded)
        };

        if let Some(value) = this_thread_expanded {
            // We were the ones to expand, so stop descending here.
            node_data.incr_visits_and_utility(value);
            self.fibre_state = TreeSearchWorkFibreState::ReachedExpandedLeaf {
                leaf_utility: value,
            };
            return Ok(true);
        }

        // Otherwise, the node was already expanded.

        if node_data.is_terminal() {
            span.emit_text("leaf: terminal node");

            let value = node_data.value::<AssertZeroVirtualLoss>();
            node_data.incr_visits_and_utility(value);
            self.fibre_state = TreeSearchWorkFibreState::ReachedExpandedLeaf {
                leaf_utility: value,
            };
            return Ok(true);
        }

        // The node is not a leaf. We need to keep descending.

        let _span = self
            .tracy_client
            .clone()
            .span(tracy_client::span_location!("real node"), 0);

        let next_updates = self.state.as_ref().unwrap().next_updates().ok().unwrap();

        let default_value_for_unexpanded_child = {
            let first_play_urgency_reduction = if is_root {
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
                        if node_data
                            .get_child(ChildId(i as u32))
                            .visits::<WithVirtualLoss>()
                            > 0
                        {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .sum::<f32>()
                    .sqrt()
            };
            node_data.value::<WithVirtualLoss>() - first_play_urgency_reduction
        };

        // pick a child based on PUCT
        let child_id = (0..next_updates.len().try_into().unwrap())
            .map(ChildId)
            .max_by_key(|&child_id| {
                node_data.puct::<WithVirtualLoss>(child_id, default_value_for_unexpanded_child)
            })
            .unwrap();

        // We wait until now to add virtual loss, to avoid messing up PUCT
        // computations.
        self.virtual_loss_incurred.push(node_data);
        node_data.incr_virtual_loss_count();

        // move to child, update the state
        self.node = node_data.get_child(child_id);
        let update = next_updates.get(child_id.0 as usize).unwrap();
        self.state.as_mut().unwrap().update(*update);

        Ok(true)
    }

    fn do_backprop(&mut self, leaf_utility: f32) {
        let _span = self
            .tracy_client
            .clone()
            .span(tracy_client::span_location!("backprop"), 0);

        // backpropagate
        for node_data in self.virtual_loss_incurred.iter().rev() {
            node_data.incr_visits_and_utility(leaf_utility);
            node_data.decr_virtual_loss_count();
        }
        let this_depth = self.virtual_loss_incurred.len();
        self.tree_search
            .sum_depth
            .fetch_add(this_depth, atomic::Ordering::Relaxed);
        // `fetch_update` returns `Err` on closure returning `None`, but we
        // don't care
        let (Ok(_) | Err(_)) = self.tree_search.max_depth.fetch_update(
            atomic::Ordering::Relaxed,
            atomic::Ordering::Relaxed,
            |max_depth| {
                if max_depth >= this_depth as u32 {
                    None
                } else {
                    Some(this_depth as u32)
                }
            },
        );

        self.fibre_state = TreeSearchWorkFibreState::Finished;
    }

    /// Returns Ok(true) if finished, Ok(false) if blocked.
    fn do_some_work(&mut self, evaluator: &dyn AsyncEvaluator) -> Result<bool> {
        loop {
            match self.fibre_state {
                TreeSearchWorkFibreState::Descending => {
                    let unblocked = self.do_one_descent(evaluator)?;
                    if !unblocked {
                        return Ok(false);
                    }
                }
                TreeSearchWorkFibreState::ReachedExpandedLeaf { leaf_utility } => {
                    self.do_backprop(leaf_utility);
                    assert!(matches!(
                        self.fibre_state,
                        TreeSearchWorkFibreState::Finished
                    ));
                }
                TreeSearchWorkFibreState::Finished => {
                    return Ok(true);
                }
            }
        }
    }
}

impl TreeSearch {
    pub fn search_once(&self, evaluator: &dyn AsyncEvaluator) -> Result<()> {
        let mut fibre = TreeSearchWorkFibre::new(self);
        let finished = fibre.do_some_work(evaluator)?;
        assert!(finished, "async resuming of work not yet implemented");
        Ok(())
    }

    pub fn sum_visits(&self) -> u32 {
        self.root_node.visits::<AssertZeroVirtualLoss>()
    }

    pub fn avg_depth(&self) -> f32 {
        (self.sum_depth.load(atomic::Ordering::Relaxed) as f32) / (self.sum_visits() as f32)
    }

    pub fn max_depth(&self) -> u32 {
        self.max_depth.load(atomic::Ordering::Relaxed)
    }
}

#[derive(Debug)]
pub struct UpdateWithStats {
    pub instr: BasicInstr,
    pub is_terminal: bool,
    pub value: NotNan<f32>,
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
                    value: NotNan::new(
                        child
                            .data()
                            .map_or(root_node_data.value::<AssertZeroVirtualLoss>(), |child| {
                                child.value::<AssertZeroVirtualLoss>()
                            }),
                    )
                    .unwrap(),
                    visits: child.visits::<AssertZeroVirtualLoss>(),
                }
            })
            .collect();
        NextUpdatesWithStats {
            root_value: root_node_data.value::<AssertZeroVirtualLoss>(),
            root_raw_utility: root_node_data.raw_utility(),
            updates_with_stats,
            avg_depth: self.avg_depth(),
            max_depth: self.max_depth(),
        }
    }
}
