mod nn;
mod nonnan;
mod parser;
mod search;
mod search_history;
mod search_state;
mod sim;
mod utils;

use std::fs;

#[cfg(feature = "color_eyre")]
use color_eyre::{eyre::Result, install};
use rand::SeedableRng;
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

    let (puzzle, solution) = utils::get_default_puzzle_solution()?;
    let mut init = parser::puzzle_prep(&puzzle, &solution)?;
    init.centre();
    init.move_by(sim::Pos::new(
        nn::constants::N_WIDTH as i32 / 2,
        nn::constants::N_HEIGHT as i32 / 2,
    ));
    let world = sim::WorldWithTapes::setup_sim(&init)?;

    let mut search_state = search_state::State::new(world.world.clone());
    let mut search_history = search_history::History::new();

    let n_premoves = solution.stats.unwrap().cycles - 3;
    println!("making {} premoves", n_premoves);

    for _ in 0..n_premoves {
        assert!(search_state.instr_buffer.is_empty());
        let instructions = world.get_instructions_at(search_state.world.timestep);
        for (i, &instr) in instructions.iter().enumerate() {
            search_state.update(instr);
            search_history.append_from_optimal_solution(instr);
        }
    }

    //
    let mut rng = rand_pcg::Pcg64::seed_from_u64(123);

    loop {
        if let Some(result) = search_state.evaluate_final_state() {
            println!("done; result = {}", result);
            break;
        }

        let mut tree_search = search::TreeSearch::new(search_state.clone());

        for _ in 0..100 {
            tree_search.search_once(&mut rng, &model)?;
        }

        let stats = tree_search.next_updates_with_stats();

        println!("{:?}", stats);

        let instr = stats.best_update();
        search_state.update(instr);
        search_history.append_mcts(&stats);
    }

    Ok(())
}
