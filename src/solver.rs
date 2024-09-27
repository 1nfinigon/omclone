use crate::nn;
use crate::parser;
use crate::search;
use crate::search_history;
use crate::search_state;
use crate::sim;
use crate::test;
use crate::utils;

use rand::prelude::*;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};
use uuid;

use eyre::Result;

fn solve_one_puzzle_seeded(
    puzzle_fpath: impl AsRef<Path>,
    seed_puzzle: &parser::FullPuzzle,
    solution_fpath: impl AsRef<Path>,
    seed_solution: &parser::FullSolution,
    model: &nn::Model,
    rng: &mut impl Rng,
) -> Result<()> {
    let mut seed_init = parser::puzzle_prep(&seed_puzzle, &seed_solution)?;

    match test::check_solution(&seed_solution, seed_puzzle, true) {
        test::CheckResult::Ok => (),
        _ => {
            return Ok(());
        }
    }

    println!(
        "====== starting {:?}, seeding with {:?}",
        puzzle_fpath.as_ref(),
        solution_fpath.as_ref()
    );

    // Recentre the solution so that the bounding box is centred around (w/2, h/2)
    if let Some((min, max)) = seed_init.bounding_box() {
        let nn_wh = sim::Pos::new(
            nn::constants::N_WIDTH as i32,
            nn::constants::N_HEIGHT as i32,
        );
        let seed_wh = max - min;
        if seed_wh.x >= nn_wh.x || seed_wh.y >= nn_wh.y {
            println!(
                "skipping because solution footprint is too big: {} vs max {}",
                seed_wh, nn_wh
            );
            return Ok(());
        }
        let delta = (nn_wh - (max + min)) / 2;
        seed_init.move_by(delta);
    }

    let seed_world = sim::WorldWithTapes::setup_sim(&seed_init)?;

    let first_timestep = seed_world.world.timestep;
    let n_arms = seed_world.world.arms.len() as u64;
    let n_moves = seed_solution.stats.as_ref().unwrap().cycles as u64 * n_arms;
    let n_moves_to_search = rng.gen_range(1..=15); // how many moves to leave behind for MCTS to find

    let mut search_state = search_state::State::new(
        seed_world.world.clone(),
        first_timestep + (n_moves + rng.gen_range(1..=30) + n_arms - 1) / n_arms,
    );
    let mut search_history = search_history::History::new();
    let mut tapes: Vec<sim::Tape<sim::BasicInstr>> = Vec::new();
    for _ in 0..seed_world.tapes.len() {
        tapes.push(sim::Tape {
            first: first_timestep as usize,
            instructions: Vec::new(),
        });
    }

    // make some pre-moves from the seed solution

    let n_premoves = n_moves.saturating_sub(n_moves_to_search);
    println!(
        "making {} premoves ({} cycles + {}; {} short of seed solution)",
        n_premoves,
        n_premoves / n_arms,
        n_premoves % n_arms,
        n_moves_to_search
    );

    for _ in 0..n_premoves {
        let arm_index = search_state.next_arm_index();
        let instr = seed_world.tapes[arm_index].get(
            search_state.world.timestep as usize,
            seed_world.repeat_length,
        );
        tapes[arm_index].instructions.push(instr);
        search_state.update(instr);
        search_history.append_from_optimal_solution(instr);
    }

    // search for a solution

    let result_is_success = loop {
        if let Some(result) = search_state.evaluate_final_state() {
            println!("done; result = {}", result);
            break result > 0.;
        }

        let mut tree_search = search::TreeSearch::new(search_state.clone());

        let playouts = if rng.gen_bool(0.75) { 100 } else { 600 };

        for _ in 0..playouts {
            tree_search.search_once(rng, &model)?;
        }

        let stats = tree_search.next_updates_with_stats();

        //println!("{:?}", stats);

        let instr = stats.best_update();
        println!(
            "{}{:<23} #={} (v={:.3} (raw {:.3}) d={:>5.2}/{:>2}) {}",
            if search_state.next_arm_index() == 0 {
                "*"
            } else {
                " "
            },
            format!("{:?}", instr),
            playouts,
            stats.root_value,
            stats.root_raw_utility,
            stats.avg_depth,
            stats.max_depth,
            {
                let v: Vec<_> = stats
                    .updates_with_stats
                    .iter()
                    .map(|u| format!("[{}{:>3}]", u.instr.to_char(), u.visits * 100 / playouts))
                    .collect();
                v.join(" ")
            }
        );

        tapes[search_state.next_arm_index()]
            .instructions
            .push(instr);
        search_state.update(instr);
        search_history.append_mcts(&stats);
    };

    // finalize world for saving

    let solution_name = {
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes[..]);
        uuid::Builder::from_random_bytes(bytes)
            .into_uuid()
            .to_string()
    };
    let solution_stats = if result_is_success {
        Some(sim::SolutionStats {
            cycles: (search_state.world.timestep - first_timestep)
                .try_into()
                .unwrap(),
            cost: search_state.world.cost,
            area: search_state.world.area_touched.len().try_into().unwrap(),
            instructions: sim::compute_tape_instruction_count(&tapes)
                .try_into()
                .unwrap(),
        })
    } else {
        None
    };
    let out_world = {
        let repeat_length = sim::compute_tape_repeat_length(&tapes);
        sim::WorldWithTapes {
            world: seed_world.world.clone(),
            tapes,
            repeat_length,
        }
    };
    let out_solution = parser::create_solution(
        &out_world,
        seed_solution.puzzle_name.clone(),
        solution_name.clone(),
        solution_stats,
    );
    let out_history = search_history::HistoryFile {
        solution_name: solution_name.clone(),
        history: search_history,
        timestep_limit: search_state.timestep_limit.try_into().unwrap(),
    };

    // save solution and search history

    let out_solution_filename =
        PathBuf::from(format!("test/current-epoch/{}.solution", solution_name));
    println!("saving solution to {:?}", out_solution_filename);
    let mut f_out_solution = BufWriter::new(File::create(&out_solution_filename)?);
    parser::write_solution(&mut f_out_solution, &out_solution)?;
    std::mem::drop(f_out_solution);

    let out_history_filename =
        PathBuf::from(format!("test/current-epoch/{}.history", solution_name));
    println!("saving history to {:?}", out_history_filename);
    let mut f_out_history = BufWriter::new(File::create(&out_history_filename)?);
    out_history.write(&mut f_out_history)?;
    std::mem::drop(f_out_history);

    Ok(())
}

pub fn main() -> Result<()> {
    println!(
        "{} spatial features\n{} spatiotemporal features\n{} temporal features",
        nn::feature_offsets::Spatial::SIZE,
        nn::feature_offsets::Spatiotemporal::SIZE,
        nn::feature_offsets::Temporal::SIZE,
    );
    println!(
        "{:?} input tensor size",
        std::mem::size_of::<nn::Features>()
    );

    let model = nn::Model::load()?;
    //let mut rng = rand_pcg::Pcg64::seed_from_u64(123);
    let mut rng = rand::thread_rng();
    let rng = &mut rng;

    //let (seed_puzzle, seed_solution) = utils::get_default_puzzle_solution()?;
    //solve_one_puzzle_seeded(&"", &seed_puzzle, &"", &seed_solution, &model, rng)?;

    println!("loading seed puzzles");
    let mut puzzle_map = utils::PuzzleMap::new();
    utils::read_puzzle_recurse(&mut puzzle_map, "test/puzzle");
    println!("loading seed solutions");
    let mut seed_solution_paths = Vec::new();
    let mut cb = |fpath: PathBuf| {
        seed_solution_paths.push(fpath);
    };
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/solution");
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/om-leaderboard-master");
    println!("shuffling seed solutions");
    seed_solution_paths.shuffle(rng);

    for solution_fpath in seed_solution_paths.iter() {
        if let Some(seed_solution) = utils::verify_solution(solution_fpath, &puzzle_map) {
            let (puzzle_fpath, seed_puzzle) = puzzle_map.get(&seed_solution.puzzle_name).unwrap();

            solve_one_puzzle_seeded(
                puzzle_fpath,
                seed_puzzle,
                solution_fpath,
                &seed_solution,
                &model,
                rng,
            )?;
        }
    }

    Ok(())
}
