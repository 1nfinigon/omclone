use crate::nn;
use crate::sim::*;
use std::sync::Arc;

#[derive(Clone)]
pub struct State {
    pub errored: bool,
    pub world: Arc<World>,
    pub instr_buffer: Vec<BasicInstr>,
    pub nn_features: Box<nn::Features>,
}

impl State {
    pub fn new(world: World) -> Self {
        let n_arms = world.arms.len();
        let mut nn_features = Box::new(nn::Features::new());
        nn_features.set_nontemporal(&world);
        nn_features.init_all_temporal(&world);
        Self {
            errored: false,
            world: Arc::new(world),
            instr_buffer: Vec::with_capacity(n_arms),
            nn_features,
        }
    }

    pub fn update(&mut self, update: BasicInstr) {
        assert!(!self.errored);
        let n_arms = self.world.arms.len();
        assert!(self.instr_buffer.len() < n_arms, "instr_buffer should have been committed to the world already, if all arms have instructions");
        self.nn_features
            .set_temporal_instr(0, &*self.world, self.instr_buffer.len(), update);
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
                    self.nn_features.set_temporal_except_instr(0, &new_world);
                    self.world = Arc::new(new_world);
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
            if self.world.timestep > 1000 {
                // TODO: don't hardcode this.
                Some(0.)
            } else if self.world.is_complete() {
                println!("DONE");
                // TODO: take into account score (cost, cycles, area).
                Some(1.)
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
