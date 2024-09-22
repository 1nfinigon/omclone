mod nn;
mod nonnan;
mod parser;
mod search;
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
        "{} spatial features, {} spatiotemporal features",
        nn::feature_offsets::Spatial::SIZE,
        nn::feature_offsets::Spatiotemporal::SIZE
    );
    println!("{:?} total input size", std::mem::size_of::<nn::Features>());

    let (puzzle, solution) = utils::get_default_puzzle_solution()?;
    let mut init = parser::puzzle_prep(&puzzle, &solution)?;
    init.centre();
    init.move_by(sim::Pos::new(
        nn::constants::N_WIDTH as i32 / 2,
        nn::constants::N_HEIGHT as i32 / 2,
    ));
    let mut world = sim::WorldWithTapes::setup_sim(&init)?;

    let mut motion = sim::WorldStepInfo::new();
    let mut float_world = sim::FloatWorld::new();
    for _ in 0..120 {
        world.run_step(true, &mut motion, &mut float_world)?;
    }

    let mut tree_search = search::TreeSearch::new(search_state::State::new(world.world));
    let mut rng = rand_pcg::Pcg64::seed_from_u64(123);

    for i in 0..10000 {
        tree_search.search_once(&mut rng, &search::PlayoutResultPredictor)?;
    }

    Ok(())
}
