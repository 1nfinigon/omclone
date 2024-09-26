use crate::parser::*;
use crate::sim::*;
use crate::utils::*;
use std::{collections::HashMap, fs, fs::File, io::BufReader, path::{Path, PathBuf}};
const CHECK_AREA: bool = false;

fn check_solution(stats: &mut (usize, usize), fpath: &Path, sol: &FullSolution, puzzle_map: &PuzzleMap) {
    let puzzle = puzzle_map.get(&sol.puzzle_name).expect("at this point solution should have been paired up with a puzzle in puzzle_map already");
    let init = match puzzle_prep(&puzzle, &sol) {
        Err(e) => {
            println!("Failed to prepare {:?}: {}", fpath, e);
            return;
        }
        Ok(s) => s,
    };
    let mut world = match WorldWithTapes::setup_sim(&init) {
        Err(e) => {
            println!("Failed to setup {:?}: {}", fpath, e);
            return;
        }
        Ok(s) => s,
    };
    stats.1 += 1;
    let mut float_world = FloatWorld::new();
    let mut motions = WorldStepInfo::new();
    while !world.world.is_complete() && world.world.timestep < 500_000 {
        let step = world.run_step(CHECK_AREA, &mut motions, &mut float_world);
        if let Err(e) = step {
            println!(
                "Simulation error on step {} puzzle {}: {:?}: {}",
                world.world.timestep, sol.puzzle_name, fpath, e
            );
            return;
        }
    }
    let mut newstats = world.get_stats();
    let oldstats = sol.stats.as_ref().unwrap();
    if puzzle.production || !CHECK_AREA {
        newstats.area = oldstats.area
    };
    if &newstats == oldstats {
        stats.0 += 1;
    } else {
        println!(
            "Stats don't match! {:?} and puzzle {}\n{:?} vs true {:?}",
            fpath, sol.puzzle_name, newstats, oldstats
        );
    }
    if stats.1 % 100 == 0 {
        println!("Current progress: {}/{}", stats.0, stats.1);
    }
}

#[test]
fn check_all() {
    std::env::set_var("RUST_BACKTRACE", "full");
    let mut puzzle_map = PuzzleMap::new();
    read_puzzle_recurse(&mut puzzle_map, PUZZLE_DIR);
    let mut stats = (0, 0);
    let mut cb = |fpath: PathBuf, solution| {
        check_solution(
            &mut stats,
            &fpath,
            &solution,
            &puzzle_map,
        );
    };
    read_solution_recurse(
        &mut cb,
        &puzzle_map,
        "test/solution",
    );
    read_solution_recurse(
        &mut cb,
        &puzzle_map,
        "test/om-leaderboard-master",
    );
    println!("final score = {}/{}", stats.0, stats.1);
}
