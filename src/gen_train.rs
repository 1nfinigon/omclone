use crate::nn;
use crate::parser;
use crate::search_history;
use crate::search_state;
use crate::sim;
use crate::utils;

use eyre::{eyre, Result};
use num_traits::ToPrimitive;
use rand::prelude::*;
use rayon::prelude::*;
use sim::BasicInstr;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

struct ParWriterState {
    next_n: Mutex<usize>,
}

impl ParWriterState {
    const BASEDIR: &str = "test/training_data";
    fn new() -> Result<Self> {
        if let Some((latest_dir_n, latest_dir)) = utils::max_child(Self::BASEDIR) {
            let next_n = utils::max_child(latest_dir).map_or(latest_dir_n, |(n, _)| n + 1);

            println!("found existing training data; starting from {}", next_n);
            Ok(Self {
                next_n: Mutex::new(next_n),
            })
        } else {
            println!("no existing training data; starting from 0");
            Ok(Self {
                next_n: Mutex::new(0),
            })
        }
    }

    fn get(&self) -> usize {
        *self.next_n.lock().unwrap()
    }

    fn next_npz_file(&self) -> PathBuf {
        let mut next_n = self.next_n.lock().unwrap();
        let dir_n = *next_n / 1000 * 1000;
        let dir_path = PathBuf::from(format!("{}/{}", Self::BASEDIR, dir_n));
        if !fs::metadata(&dir_path).is_ok_and(|m| m.is_dir()) {
            fs::create_dir(&dir_path).unwrap();
        }
        let file_path = dir_path.join(format!("{}.npz", *next_n));
        *next_n += 1;
        file_path
    }
}

fn write_npz_files(
    par_writer_state: &ParWriterState,
    features: Box<nn::Features>,
    value: f32,
    visits: [f32; BasicInstr::N_TYPES],
    (x, y): (usize, usize),
    loss_weights: [f32; 3],
) -> Result<()> {
    let spatial_input = features.spatial.to_tensors_for_serializing()?;
    let spatiotemporal_input = features.spatiotemporal.to_tensors_for_serializing()?;
    let temporal_input = features.temporal.to_tensors_for_serializing()?;
    let value_output = tch::Tensor::f_from_slice(&[value, 1. - value])?;
    let policy_output = tch::Tensor::f_from_slice(&visits[..])?.f_softmax(0, None)?;
    let pos = tch::Tensor::f_from_slice(&[x as i64, y as i64])?;
    let loss_weights = tch::Tensor::f_from_slice(&loss_weights)?;

    tch::Tensor::write_npz(
        &[
            ("spatial_input_indices", &spatial_input.indices),
            ("spatial_input_values", &spatial_input.values),
            ("spatial_input_size", &spatial_input.size),
            (
                "spatiotemporal_input_indices",
                &spatiotemporal_input.indices,
            ),
            ("spatiotemporal_input_values", &spatiotemporal_input.values),
            ("spatiotemporal_input_size", &spatiotemporal_input.size),
            ("temporal_input_indices", &temporal_input.indices),
            ("temporal_input_values", &temporal_input.values),
            ("temporal_input_size", &temporal_input.size),
            ("value_output", &value_output),
            ("policy_output", &policy_output),
            ("pos", &pos),
            ("loss_weights", &loss_weights),
        ],
        par_writer_state.next_npz_file(),
    )?;

    Ok(())
}

fn process_one_solution(
    par_writer_state: &ParWriterState,
    fpath: impl AsRef<Path>,
    rng: &mut impl Rng,
    puzzle_map: &utils::PuzzleMap,
) -> Result<f32> {
    let solution = parser::parse_solution(&mut BufReader::new(File::open(fpath.as_ref())?))?;

    let (_, puzzle) = puzzle_map.get(&solution.puzzle_name).unwrap();
    let history_fpath = fpath.as_ref().with_extension("history");
    let history_file =
        search_history::HistoryFile::read(&mut BufReader::new(File::open(&history_fpath)?))?;

    let init = parser::puzzle_prep(puzzle, &solution)?;
    let world = sim::WorldWithTapes::setup_sim(&init)?;

    let mut search_state =
        search_state::State::new(world.world, history_file.timestep_limit as u64);

    let mut tensors = Vec::new();

    for (move_idx, history_item) in history_file.history.0.iter().enumerate() {
        let arm_index = search_state.next_arm_index();
        let instr =
            world.tapes[arm_index].get(search_state.world.timestep as usize, world.repeat_length);

        let next_arm_pos = search_state.world.arms[arm_index].pos;
        let (x, y) = nn::features::normalize_position(next_arm_pos).expect("arm out of nn bounds");

        match history_item.kind {
            search_history::Kind::FromOptimalSolution => {
                // Basically don't use these to train the value head because
                // the current state has no correlation with the final value
                // from MCTS
                let loss_weights = [0.00001, 1.0, 1.0];

                if rng.gen_bool(if instr == BasicInstr::Empty {
                    0.01
                } else {
                    0.1
                }) {
                    // mock up a policy of visiting the right answer 75% of the time
                    // TODO: dunno if this is right/good,
                    // it certainly gets us more data quicker though.
                    let mut visits = [1f32; BasicInstr::N_TYPES];
                    visits[instr.to_usize().unwrap()] = 4.;
                    tensors.push((
                        search_state.nn_features.clone(),
                        visits,
                        (x, y),
                        loss_weights,
                    ));
                }
            }
            search_history::Kind::Mcts => {
                // For now, basically don't use these to train the policy head
                // because they're bad quality. (?)
                let loss_weights = if history_file.final_outcome > 0.5 {
                    [10.0, 5.0, 1.0]
                } else {
                    [0.1, 0.05, 1.0]
                };

                // include 50% of these < 2000 playouts, and 100% of these
                // >= 200 playouts
                let n_playouts: u32 = history_item.playouts.iter().sum();
                if rng.gen_bool(if n_playouts < 2000 { 0.5 } else { 1.0 }) {
                    let visits: Vec<_> = history_item
                        .playouts
                        .iter()
                        .copied()
                        .map(|p| p as f32)
                        .collect();
                    tensors.push((
                        search_state.nn_features.clone(),
                        visits.try_into().unwrap(),
                        (x, y),
                        loss_weights,
                    ));
                }
            }
        }

        if search_state.errored {
            return Err(eyre!("at move {}, search state had errored", move_idx));
        }
        search_state.update(instr);
    }

    let final_result = if let Some(final_result) = search_state.evaluate_final_state() {
        final_result
    } else {
        return Err(eyre!(
            "after playing out the solution, did not reach a final state. Ignoring"
        ));
    };

    assert!((final_result - history_file.final_outcome).abs() < 1e-6);

    for (features, visits, pos, loss_weights) in tensors.into_iter() {
        write_npz_files(
            par_writer_state,
            features,
            final_result,
            visits,
            pos,
            loss_weights,
        )?;
    }

    Ok(final_result)
}

fn gen_for_solution_dir(
    par_writer_state: &ParWriterState,
    puzzle_map: &utils::PuzzleMap,
    solution_dir: impl AsRef<Path>,
) -> Result<()> {
    println!("loading solutions for path {:?}", solution_dir.as_ref());

    let mut solution_paths = Vec::new();
    let mut cb = |fpath: PathBuf| {
        if !fs::exists(fpath.with_extension("sampled")).unwrap() {
            solution_paths.push(fpath);
        }
    };
    utils::read_file_suffix_recurse(&mut cb, ".solution", solution_dir);

    let i = std::sync::atomic::AtomicUsize::new(0);
    let total_final_outcome = std::sync::Mutex::new(0f32);
    solution_paths.par_iter().for_each(|fpath| {
        //let mut rng = rand_pcg::Pcg64::seed_from_u64(123);
        let mut rng = rand::thread_rng();
        let rng = &mut rng;

        println!(
            "{}/{} (current sample id: {})",
            i.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            solution_paths.len(),
            par_writer_state.get(),
        );
        let result = process_one_solution(par_writer_state, fpath, rng, puzzle_map);
        match result {
            Ok(final_outcome) => {
                *total_final_outcome.lock().unwrap() += final_outcome;
                File::create_new(fpath.with_extension("sampled")).unwrap();
            }
            Err(e) => {
                // TODO: figure out where these errors are coming from???????
                println!("ignoring error for {:?}: {}", fpath, e)
            }
        }
    });

    println!(
        "avg final outcome: {:.3}",
        total_final_outcome.into_inner().unwrap() / (i.into_inner() as f32)
    );

    Ok(())
}

pub fn main() -> Result<()> {
    let par_writer_state = ParWriterState::new()?;

    println!("loading seed puzzles");
    let mut puzzle_map = utils::PuzzleMap::new();
    utils::read_puzzle_recurse(&mut puzzle_map, "test/puzzle");

    let (max_game_n, _) = utils::max_child("test/games").expect("no games");
    for game_n in max_game_n.saturating_sub(1000000)..=max_game_n {
        let path = PathBuf::from(format!("test/games/{}", game_n));
        if fs::exists(&path)? {
            gen_for_solution_dir(&par_writer_state, &puzzle_map, path)?;
        }
    }

    Ok(())
}
