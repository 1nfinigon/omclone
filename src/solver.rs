mod nn;
mod nonnan;
mod parser;
mod search;
mod search_history;
mod search_state;
mod sim;
mod utils;

use rand::{Rng, SeedableRng};
use std::path::PathBuf;
use std::{fs::File, io::BufWriter};
use uuid;

#[cfg(feature = "color_eyre")]
use color_eyre::{eyre::Result, install};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{eyre::Result, install};

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "full");
    install()?;

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

    let (seed_puzzle, seed_solution) = utils::get_default_puzzle_solution()?;
    let mut seed_init = parser::puzzle_prep(&seed_puzzle, &seed_solution)?;
    seed_init.centre();
    seed_init.move_by(sim::Pos::new(
        nn::constants::N_WIDTH as i32 / 2,
        nn::constants::N_HEIGHT as i32 / 2,
    ));
    let seed_world = sim::WorldWithTapes::setup_sim(&seed_init)?;

    let mut search_state = search_state::State::new(seed_world.world.clone());
    let mut search_history = search_history::History::new();
    let mut tapes: Vec<sim::Tape<sim::BasicInstr>> = Vec::new();
    for _ in 0..seed_world.tapes.len() {
        tapes.push(sim::Tape {
            first: 0,
            instructions: Vec::new(),
        });
    }

    // make some pre-moves from the seed solution

    let n_premoves = seed_solution.stats.unwrap().cycles - 3;
    println!("making {} premoves", n_premoves);

    for _ in 0..n_premoves {
        assert!(search_state.instr_buffer.is_empty());
        let instructions = seed_world.get_instructions_at(search_state.world.timestep);
        for (i, &instr) in instructions.iter().enumerate() {
            assert_eq!(search_state.next_arm_index(), i);
            tapes[i].instructions.push(instr);
            search_state.update(instr);
            search_history.append_from_optimal_solution(instr);
        }
    }

    // search for a solution

    let mut rng = rand_pcg::Pcg64::seed_from_u64(123);

    let result_is_success = loop {
        if let Some(result) = search_state.evaluate_final_state() {
            println!("done; result = {}", result);
            break result > 0.;
        }

        let mut tree_search = search::TreeSearch::new(search_state.clone());

        for _ in 0..100 {
            tree_search.search_once(&mut rng, &model)?;
        }

        let stats = tree_search.next_updates_with_stats();

        println!("{:?}", stats);

        let instr = stats.best_update();
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
        seed_puzzle.puzzle_name.clone(),
        solution_name.clone(),
        if result_is_success {
            Some(out_world.get_stats())
        } else {
            None
        },
    );
    let out_history = search_history::HistoryFile {
        solution_name: solution_name.clone(),
        history: search_history,
    };

    // save solution and search history

    let out_solution_filename =
        PathBuf::from(format!("test/current-epoch/{}.solution", solution_name));
    println!("saving solution to {}", out_solution_filename.display());
    let mut f_out_solution = BufWriter::new(File::create_new(&out_solution_filename)?);
    parser::write_solution(&mut f_out_solution, &out_solution)?;
    std::mem::drop(f_out_solution);

    let out_history_filename =
        PathBuf::from(format!("test/current-epoch/{}.history", solution_name));
    println!("saving history to {}", out_history_filename.display());
    let mut f_out_history = BufWriter::new(File::create_new(&out_history_filename)?);
    out_history.write(&mut f_out_history)?;
    std::mem::drop(f_out_history);

    Ok(())
}
