mod nn;
mod nonnan;
mod parser;
mod search;
mod search_history;
mod search_state;
mod sim;
mod utils;

use num_traits::ToPrimitive;
use rand::prelude::*;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tch;
use uuid;

#[cfg(feature = "color_eyre")]
use color_eyre::{eyre::Result, install};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{eyre::Result, install};

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "full");
    install()?;

    let model = nn::Model::load()?;
    //let mut rng = rand_pcg::Pcg64::seed_from_u64(123);
    let mut rng = rand::thread_rng();
    let rng = &mut rng;

    println!("loading seed puzzles");
    let mut puzzle_map = utils::PuzzleMap::new();
    utils::read_puzzle_recurse(&mut puzzle_map, "test/puzzle");

    println!("loading current-epoch solutions");
    let mut all_tensors = Vec::new();
    let mut solution_paths = Vec::new();
    let mut cb = |fpath: PathBuf| {
        solution_paths.push(fpath);
    };
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/current-epoch");

    for fpath in solution_paths {
        let solution = parser::parse_solution(&mut BufReader::new(File::open(&fpath)?))?;

        println!("{}", solution.puzzle_name);
        let (_, puzzle) = puzzle_map.get(&solution.puzzle_name).unwrap();
        let history_fpath = format!(
            "{}.history",
            fpath.to_str().unwrap().strip_suffix(".solution").unwrap()
        );
        let history_file =
            search_history::HistoryFile::read(&mut BufReader::new(File::open(&history_fpath)?))?;

        let init = parser::puzzle_prep(puzzle, &solution)?;
        let world = sim::WorldWithTapes::setup_sim(&init)?;

        let mut search_state =
            search_state::State::new(world.world, history_file.timestep_limit as u64);

        let mut tensors = Vec::new();

        for history_item in history_file.history.0.iter() {
            let arm_index = search_state.next_arm_index();
            let instr = world.tapes[arm_index]
                .get(search_state.world.timestep as usize, world.repeat_length);

            match history_item.kind {
                search_history::Kind::FromOptimalSolution => {
                    // include 10% of these
                    if rng.gen_bool(0.1) {
                        // mock up a policy of solely picking the right answer, 90% of the time
                        // TODO: dunno if this is right/good,
                        // it certainly gets us more data quicker though.
                        let mut policy = vec![1f32; sim::BasicInstr::N_TYPES];
                        policy[instr.to_usize().unwrap()] = 9.;
                        tensors.push((search_state.nn_features.clone(), policy));
                    }
                }
                search_history::Kind::MCTS => {
                    // include 100% of these < 200 playouts, and 100% of these
                    // >= 200 playouts
                    let n_playouts: u32 = history_item.playouts.iter().sum();
                    let _ = n_playouts;
                    tensors.push((
                        search_state.nn_features.clone(),
                        history_item
                            .playouts
                            .iter()
                            .copied()
                            .map(|p| p as f32)
                            .collect(),
                    ));
                }
            }

            search_state.update(instr);
        }

        let final_result = if let Some(final_result) = search_state.evaluate_final_state() {
            final_result
        } else {
            println!(
                "WARNING: after playing out the solution, did not reach a final state. Ignoring"
            );
            continue;
        };

        for (input, policy) in tensors {
            all_tensors.push((input, final_result, policy));
        }
    }

    println!("Writing out");
    for (idx, (features, value, policy)) in all_tensors.iter().enumerate() {
        if idx % 1000 == 0 {
            println!("Writing out: {}/{}", idx, all_tensors.len());
        }

        let (spatial_input, spatiotemporal_input, temporal_input) =
            features.to_tensor(tch::Device::Cpu)?;
        let value_output = tch::Tensor::f_from_slice(&[*value])?;
        let policy_output = tch::Tensor::f_from_slice(&policy[..])?;

        tch::Tensor::write_npz(
            &[
                ("spatial_input", &spatial_input),
                ("spatiotemporal_input", &spatiotemporal_input),
                ("temporal_input", &temporal_input),
                ("value_output", &value_output),
                ("policy_output", &policy_output),
            ],
            format!("test/next-training/{}.npz", idx),
        )?;
    }

    Ok(())
}
