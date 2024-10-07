use crate::search_state::State;
use crate::sim::BasicInstr;
use eyre::Result;

pub struct EvalResult {
    pub utility: f32,
    pub policy: [f32; BasicInstr::N_TYPES],
}

pub trait Evaluator: Send + Sync {
    fn model_name(&self) -> &str;
    fn eval_count(&self) -> usize;
    fn clear(&mut self);
    fn eval_blocking(&self, state: State, is_root: bool) -> Result<EvalResult>;
}
