use crate::nn;
use crate::parser;
use crate::search_history;
use crate::search_state;
use crate::sim;
use crate::utils;

use eyre::OptionExt;
use eyre::{eyre, Result};
use num_traits::ToPrimitive;
use rand::prelude::*;
use rayon::prelude::*;
use sim::BasicInstr;
use std::fs::File;
use std::io::BufWriter;
use std::io::{BufReader, Seek, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use zip::{read::ZipArchive, write::ZipWriter};

fn write_npz_files<W: Write + Seek>(
    npz_writer: &Mutex<ZipWriter<W>>,
    // needs to be a path that's guaranteed not to collide with any other
    // threads; it is used temporarily, and cleaned up in this function
    tmp_path: impl AsRef<Path>,
    // parent path inside the .npz archive for these tensors
    sample_name: &str,
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

    let mut npz_entries = Vec::new();
    let mut add_entry =
        |name, tensor| npz_entries.push((format!("{}/{}", sample_name, name), tensor));
    for (sparse_name, sparse_tensors) in [
        ("spatial_input", &spatial_input),
        ("spatiotemporal_input", &spatiotemporal_input),
        ("temporal_input", &temporal_input),
    ] {
        add_entry(format!("{}/indices", sparse_name), &sparse_tensors.indices);
        add_entry(format!("{}/values", sparse_name), &sparse_tensors.values);
        add_entry(format!("{}/size", sparse_name), &sparse_tensors.size);
    }
    add_entry("value_output".to_string(), &value_output);
    add_entry("policy_output".to_string(), &policy_output);
    add_entry("pos".to_string(), &pos);
    add_entry("loss_weights".to_string(), &loss_weights);
    tch::Tensor::write_npz(&npz_entries, tmp_path.as_ref())?;

    {
        let tmp_zip_archive = ZipArchive::new(BufReader::new(File::open(tmp_path.as_ref())?))?;
        let mut npz_writer = npz_writer
            .lock()
            .ok()
            .ok_or_eyre("can't lock npz writer mutex")?;
        npz_writer.merge_archive(tmp_zip_archive)?;
    }
    std::fs::remove_file(tmp_path)?;

    Ok(())
}

fn process_one_solution<W: Write + Seek>(
    npz_writer: &Mutex<ZipWriter<W>>,
    fpath: impl AsRef<Path>,
    rng: &mut impl Rng,
    puzzle_map: &utils::PuzzleMap,
) -> Result<usize> {
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
                // Basically don't use these to train the value head because
                // the current state has no correlation with the final value
                // from MCTS
                let loss_weights = [0.00001, 0.8, 1.0];

                if rng.gen_bool(if instr == BasicInstr::Empty {
                    0.05
                } else {
                    0.5
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
                let loss_weights = [1.0, 0.2, 1.0];

                // include 60% of these < 200 playouts, and 100% of these
                // >= 200 playouts
                let n_playouts: u32 = history_item.playouts.iter().sum();
                if rng.gen_bool(if n_playouts < 200 { 0.6 } else { 1.0 }) {
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

    let n_tensors = tensors.len();
    for (i, (features, visits, pos, loss_weights)) in tensors.into_iter().enumerate() {
        let sample_name = format!("{}_{}", solution.solution_name, i);
        let tmp_path = std::env::temp_dir().join(format!(
            "ae98c2fd-4926-4c37-a654-daacf787e462_{}.zip",
            sample_name
        ));
        write_npz_files(
            npz_writer,
            &tmp_path,
            &sample_name,
            features,
            final_result,
            visits,
            pos,
            loss_weights,
        )?;
    }

    Ok(n_tensors)
}

pub fn main() -> Result<()> {
    println!("loading seed puzzles");
    let mut puzzle_map = utils::PuzzleMap::new();
    utils::read_puzzle_recurse(&mut puzzle_map, "test/puzzle");

    println!("loading current-epoch solutions");
    let mut solution_paths = Vec::new();
    let mut cb = |fpath: PathBuf| {
        solution_paths.push(fpath);
    };
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/current-epoch");

    let i = std::sync::atomic::AtomicUsize::new(0);
    let n_samples_written = std::sync::atomic::AtomicUsize::new(0);
    let npz_writer = Arc::new(Mutex::new(ZipWriter::new(BufWriter::new(
        File::create_new("test/next-training.npz")?,
    ))));
    solution_paths.par_iter().for_each(|fpath| {
        //let mut rng = rand_pcg::Pcg64::seed_from_u64(123);
        let mut rng = rand::thread_rng();
        let rng = &mut rng;

        let npz_writer = Arc::clone(&npz_writer);

        println!(
            "{}/{} ({} samples written so far)",
            i.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            solution_paths.len(),
            n_samples_written.load(std::sync::atomic::Ordering::SeqCst),
        );
        let result = process_one_solution(&npz_writer, fpath, rng, &puzzle_map);
        match result {
            Ok(n) => {
                n_samples_written.fetch_add(n, std::sync::atomic::Ordering::SeqCst);
            }
            Err(e) => {
                // TODO: figure out where these errors are coming from???????
                println!("ignoring error for {:?}: {}", fpath, e)
            }
        }
    });

    println!("{} total samples", n_samples_written.into_inner());

    Ok(())
}
