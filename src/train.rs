mod nn;
mod nonnan;
mod parser;
mod search;
mod search_history;
mod search_state;
mod sim;
mod utils;

use eyre::{eyre, Result};
use num_traits::ToPrimitive;
use rand::prelude::*;
use sim::BasicInstr;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::{Path, PathBuf};
use tch;
use uuid;

#[cfg(feature = "color_eyre")]
use color_eyre::install;
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::install;

fn write_npz_files(
    index: &mut usize,
    features: Box<nn::Features>,
    value: f32,
    visits: [f32; BasicInstr::N_TYPES],
    (x, y): (usize, usize),
) -> Result<()> {
    let (spatial_input, spatiotemporal_input, temporal_input) =
        features.to_tensor(tch::Device::Cpu)?;
    let value_output = tch::Tensor::f_from_slice(&[value, 1. - value])?;
    let policy_output = tch::Tensor::f_from_slice(&visits[..])?.f_softmax(0, None)?;
    let pos = tch::Tensor::f_from_slice(&[x as i64, y as i64])?;

    tch::Tensor::write_npz(
        &[
            ("spatial_input", &spatial_input),
            ("spatiotemporal_input", &spatiotemporal_input),
            ("temporal_input", &temporal_input),
            ("value_output", &value_output),
            ("policy_output", &policy_output),
            ("pos", &pos),
        ],
        format!("test/next-training/{}.npz", *index),
    )?;

    *index += 1;
    Ok(())
}

fn process_one_solution(
    fpath: impl AsRef<Path>,
    rng: &mut impl Rng,
    puzzle_map: &utils::PuzzleMap,
    file_index: &mut usize,
) -> Result<()> {
    let solution = parser::parse_solution(&mut BufReader::new(File::open(fpath.as_ref())?))?;

    let (_, puzzle) = puzzle_map.get(&solution.puzzle_name).unwrap();
    let history_fpath = format!(
        "{}.history",
        fpath
            .as_ref()
            .to_str()
            .unwrap()
            .strip_suffix(".solution")
            .unwrap()
    );
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
                // include 10% of these
                if rng.gen_bool(if instr == BasicInstr::Empty { 0.01 } else { 0.1 }) {
                    // mock up a policy of visiting the right answer 75% of the time
                    // TODO: dunno if this is right/good,
                    // it certainly gets us more data quicker though.
                    let mut visits = [1f32; BasicInstr::N_TYPES];
                    visits[instr.to_usize().unwrap()] = 4.;
                    tensors.push((search_state.nn_features.clone(), visits, (x, y)));
                }
            }
            search_history::Kind::MCTS => {
                // include 90% of these < 200 playouts, and 100% of these
                // >= 200 playouts
                let n_playouts: u32 = history_item.playouts.iter().sum();
                if rng.gen_bool(if n_playouts < 200 { 0.9 } else { 1.0 }) {
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

    for (features, visits, pos) in tensors {
        write_npz_files(file_index, features, final_result, visits, pos)?;
    }

    Ok(())
}

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
    let mut solution_paths = Vec::new();
    let mut cb = |fpath: PathBuf| {
        solution_paths.push(fpath);
    };
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/current-epoch");

    let mut file_index = 0;
    for (i, fpath) in solution_paths.iter().enumerate() {
        println!(
            "{}/{} ({} files written)",
            i,
            solution_paths.len(),
            file_index
        );
        let result = process_one_solution(fpath, rng, &puzzle_map, &mut file_index);
        match result {
            Ok(()) => (),
            Err(e) => {
                // TODO: figure out where these errors are coming from???????
                println!("ignoring error for {:?}: {}", fpath, e)
            }
        }
    }

    Ok(())
}
