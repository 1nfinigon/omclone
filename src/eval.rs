//! Non-NN-specific code for evaluating value/policy for leaves in parallel

use crate::search_state::State;
use crate::sim::BasicInstr;
use eyre::{Context, OptionExt, Result};
use rand::{Rng, RngCore};
use std::sync::{
    atomic::{self, AtomicUsize},
    mpsc, Arc, Mutex,
};
use std::thread;

pub struct EvalResult {
    pub utility: f32,
    pub policy: [f32; BasicInstr::N_TYPES],
}

pub trait BatchEvaluator: Send + Sync {
    fn model_name(&self) -> &str;

    /// `states` contains a bool `is_root`
    /// Must return a Vec of the same length
    fn batch_eval_blocking(&self, states: Vec<(State, bool)>) -> Result<Vec<EvalResult>>;
}

pub trait AsyncEvaluator: Send + Sync {
    fn model_name(&self) -> &str;

    // TODO: make this async
    fn eval_blocking(&self, state: State, is_root: bool) -> Result<EvalResult>;

    // TODO: move these?
    fn eval_count(&self) -> usize;
    fn clear(&mut self);
}

struct EvalThreadRequest {
    state: State,
    is_root: bool,
    tx: mpsc::SyncSender<EvalResult>,
}

/// A threaded adapter that takes a blocking BatchEvaluator, and exposes an
/// AsyncEvaluator
pub struct EvalThread<BE> {
    model: Arc<BE>,
    eval_thread_tx: mpsc::SyncSender<EvalThreadRequest>,
    eval_count: AtomicUsize,
}

impl<BE: BatchEvaluator> EvalThread<BE> {
    const MAX_EVAL_BATCH_SIZE: usize = 128;

    fn thread_main(
        model: Arc<BE>,
        rx: mpsc::Receiver<EvalThreadRequest>,
        tracy_client: tracy_client::Client,
    ) {
        let _span = tracy_client
            .clone()
            .span(tracy_client::span_location!("eval thread"), 0);

        let tracy_eval_batch_size_plot = tracy_client::plot_name!("eval batch size");

        loop {
            tracy_client.plot(tracy_eval_batch_size_plot, 0.);

            // These two should always be the same length.
            let mut batch_to_evaluate = Vec::new();
            let mut result_txs = Vec::new();

            // First, let's accumulate as many eval requests as possible
            loop {
                if batch_to_evaluate.len() >= Self::MAX_EVAL_BATCH_SIZE {
                    break;
                }

                let EvalThreadRequest { state, is_root, tx } = {
                    if batch_to_evaluate.is_empty() {
                        match rx.recv() {
                            Ok(request) => request,
                            Err(mpsc::RecvError) => {
                                // no more data will ever come; nothing we can do but
                                // terminate
                                return;
                            }
                        }
                    } else {
                        match rx.try_recv() {
                            Ok(request) => request,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                // no more data will ever come; nothing we can do but
                                // terminate
                                return;
                            }
                            Err(mpsc::TryRecvError::Empty) => {
                                // end of this batch
                                break;
                            }
                        }
                    }
                };

                batch_to_evaluate.push((state, is_root));
                result_txs.push(tx);
            }

            assert!(!batch_to_evaluate.is_empty());
            assert_eq!(batch_to_evaluate.len(), result_txs.len());

            // workaround tracy_client not allowing stairstep plot config
            tracy_client.plot(tracy_eval_batch_size_plot, 0.);

            tracy_client.plot(tracy_eval_batch_size_plot, batch_to_evaluate.len() as f64);

            // Now, let's process our batch.
            let results = {
                let span = tracy_client
                    .clone()
                    .span(tracy_client::span_location!("batch eval"), 0);
                span.emit_value(result_txs.len() as u64);

                // We can't do much with errors (it would be noisy to pass the
                // same error back to every search thread), so just panic
                model
                    .batch_eval_blocking(batch_to_evaluate)
                    .expect("batch evaluation failed")
            };
            assert_eq!(result_txs.len(), results.len());
            let len = results.len();

            for (result_tx, result) in result_txs.into_iter().zip(results.into_iter()) {
                let (Ok(()) | Err(_)) = result_tx.send(result);
            }

            // workaround tracy_client not allowing stairstep plot config
            tracy_client.plot(tracy_eval_batch_size_plot, len as f64);
        }
    }
}

impl<BE: BatchEvaluator + 'static> EvalThread<BE> {
    pub fn new(model: BE, tracy_client: tracy_client::Client) -> Self {
        let model = Arc::new(model);
        let (eval_thread_tx, eval_thread_rx) = mpsc::sync_channel(Self::MAX_EVAL_BATCH_SIZE);
        thread::spawn({
            let model = model.clone();
            let tracy_client = tracy_client.clone();
            move || Self::thread_main(model, eval_thread_rx, tracy_client)
        });
        Self {
            model,
            eval_count: 0.into(),
            eval_thread_tx,
        }
    }
}

impl<BE: BatchEvaluator> AsyncEvaluator for EvalThread<BE> {
    fn model_name(&self) -> &str {
        self.model.model_name()
    }

    fn eval_count(&self) -> usize {
        self.eval_count.load(atomic::Ordering::Relaxed)
    }

    fn clear(&mut self) {
        self.eval_count = 0.into();
    }

    /// Queues an evaluation on the eval thread, and waits for the result
    fn eval_blocking(&self, state: State, is_root: bool) -> Result<EvalResult> {
        let (tx, rx) = mpsc::sync_channel(1);
        let () = self
            .eval_thread_tx
            .send(EvalThreadRequest { state, is_root, tx })
            .wrap_err("eval thread died, not accepting requests")?;

        self.eval_count.fetch_add(1, atomic::Ordering::Relaxed);

        let result = rx
            .recv()
            .wrap_err("eval thread died, never sent response")?;
        Ok(result)
    }
}

pub struct DummyEvaluator {
    rng: Mutex<Box<dyn RngCore + Send>>,
}

impl DummyEvaluator {
    pub fn new(rng: Box<dyn RngCore + Send>) -> Self {
        Self {
            rng: Mutex::new(rng),
        }
    }
}

impl BatchEvaluator for DummyEvaluator {
    fn model_name(&self) -> &str {
        "dummy-evaluator"
    }

    fn batch_eval_blocking(&self, states: Vec<(State, bool)>) -> Result<Vec<EvalResult>> {
        let mut rng = self.rng.lock().ok().ok_or_eyre("can't lock rng")?;
        Ok(states
            .into_iter()
            .map(|_| {
                let mut policy = [0f32; BasicInstr::N_TYPES];
                policy.iter_mut().for_each(|elem| {
                    *elem = rng.sample(rand::distributions::Open01);
                });
                let policy_denom: f32 = policy.iter().sum();
                policy.iter_mut().for_each(|elem| {
                    *elem /= policy_denom;
                });
                EvalResult {
                    utility: rng.sample(rand::distributions::Open01),
                    policy,
                }
            })
            .collect())
    }
}
