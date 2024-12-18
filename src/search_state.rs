//! [sim::WorldWithTapes](crate::sim::WorldWithTapes) but augmented to be
//! suitable for MCTS search

use crate::nn;
use crate::sim::*;

/// The current state of the world at a given search ply. Can represent
/// intermediate states where half of the arms in the current cycle have been
/// assigned an instruction, but not all. The cycle is committed (the `world` is
/// updated and collision detection performed) automatically when all arms have
/// been assigned an instruction.
///
/// Also caches and updates `nn_features` incrementally to stay in sync with
/// `world`.
#[derive(Clone)]
pub struct State {
    /// `true` if a previous update caused a crash -- in this case none of the
    /// other fields apart from this one will have been updated. Hence, always
    /// check this field first!!
    pub errored: bool,

    pub cycle_limit: u64,
    pub world: Box<World>,
    pub instr_buffer: Vec<BasicInstr>,
    pub nn_features: Box<nn::Features>,
}

impl State {
    pub fn new(world: World, cycle_limit: u64) -> Self {
        let n_arms = world.arms.len();
        let mut nn_features = Box::new(nn::Features::new());
        nn_features.set_nontemporal(&world);
        nn_features.init_all_temporal(&world, cycle_limit.saturating_sub(world.cycle));
        Self {
            errored: false,
            cycle_limit,
            world: Box::new(world),
            instr_buffer: Vec::with_capacity(n_arms),
            nn_features,
        }
    }

    pub fn next_arm_index(&self) -> usize {
        self.instr_buffer.len()
    }

    pub fn update(&mut self, update: BasicInstr) {
        assert!(!self.errored);
        let n_arms = self.world.arms.len();
        assert!(self.instr_buffer.len() < n_arms, "instr_buffer should have been committed to the world already, if all arms have instructions");
        self.nn_features
            .set_temporal_instr(0, &self.world, self.instr_buffer.len(), update);
        self.instr_buffer.push(update);
        if self.instr_buffer.len() == n_arms {
            // commit buffer
            let instr_buffer =
                std::mem::replace(&mut self.instr_buffer, Vec::with_capacity(n_arms));
            let mut new_world = (*self.world).clone();
            let mut motions = WorldStepInfo::new();
            let mut float_world = FloatWorld::new();
            let result = new_world.run_step(true, &mut motions, &mut float_world, &instr_buffer);
            match result {
                Ok(()) => {
                    self.nn_features.shift_temporal();
                    self.nn_features.set_temporal_except_instr(
                        0,
                        &new_world,
                        self.cycle_limit.saturating_sub(self.world.cycle),
                    );
                    self.world = Box::new(new_world);
                }
                Err(_) => {
                    self.errored = true;
                }
            }
        }
    }

    pub fn evaluate_final_state(&self) -> Option<f32> {
        if self.errored {
            Some(0.)
        } else if self.instr_buffer.is_empty() {
            if self.world.is_complete() {
                // TODO: take into account score (cost, cycles, area).
                Some(1.)
            } else if self.world.cycle >= self.cycle_limit {
                Some(0.)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn next_updates(&self) -> std::result::Result<&'static [BasicInstr], f32> {
        match self.evaluate_final_state() {
            Some(final_eval) => Err(final_eval),
            None => Ok(&[
                BasicInstr::Empty,
                BasicInstr::RotateClockwise,
                BasicInstr::RotateCounterClockwise,
                BasicInstr::Extend,
                BasicInstr::Retract,
                BasicInstr::Grab,
                BasicInstr::Drop,
                BasicInstr::PivotClockwise,
                BasicInstr::PivotCounterClockwise,
                BasicInstr::Forward,
                BasicInstr::Back,
            ]),
        }
    }
}
