//! Simulator benchmark

use crate::parser;
use crate::sim;
use crate::utils;

use eyre::Result;

pub fn main() -> Result<()> {
    let (puzzle, sol) = utils::get_default_puzzle_solution()?;
    //println!("Check: {:?}", sol.stats);
    let init = parser::puzzle_prep(&puzzle, &sol)?;

    let mut world = sim::WorldWithTapes::setup_sim(&init)?;
    let mut float_world = sim::FloatWorld::new();
    let mut motions = sim::WorldStepInfo::new();
    /*
    while !world.is_complete() {
        world.run_step(false, &mut motions, &mut float_world)?;
        let stats = world.get_stats();
        println!("Step {:03}", stats.cycles);
    }
    let stats = world.get_stats();
    println!("Complete! {:?}", stats);
    Ok(())
    */

    let start_time = std::time::Instant::now();
    for iteration in 1.. {
        world.run_step(true, &mut motions, &mut float_world)?;
        if iteration % 10000 == 0 {
            let curr_time = std::time::Instant::now();
            let duration = curr_time - start_time;
            let duration_per_step = duration.div_f64(iteration as f64);
            println!(
                "Iterations: {:<9} Duration per step: {:.2}us",
                iteration,
                duration_per_step.as_secs_f32() / 1e-6
            );
        }
    }
    Ok(())
}
