use crate::parser::*;
use crate::sim::*;
use crate::utils::*;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
const CHECK_AREA: bool = false;

#[derive(Debug)]
pub enum CheckResult {
    Skipped,
    Ok,
    FailedSetup,
    FailedSim,
    FailedStatMismatch,
}

pub fn check_solution(fpath: &Path, sol: &FullSolution, puzzle_map: &PuzzleMap) -> CheckResult {
    let (_puzzle_fpath, puzzle) = puzzle_map.get(&sol.puzzle_name).expect(
        "at this point solution should have been paired up with a puzzle in puzzle_map already",
    );
    let init = match puzzle_prep(&puzzle, &sol) {
        Err(e) => {
            println!("Failed to prepare {:?}: {}", fpath, e);
            return CheckResult::FailedSetup;
        }
        Ok(s) => s,
    };
    /*
    if init.has_overlap() {
        //println!("skipping due to overlap");
        return CheckResult::Skipped;
    }
    */
    let mut world = match WorldWithTapes::setup_sim(&init) {
        Err(e) => {
            println!("Failed to setup {:?}: {}", fpath, e);
            return CheckResult::FailedSetup;
        }
        Ok(s) => s,
    };
    let mut float_world = FloatWorld::new();
    let mut motions = WorldStepInfo::new();
    while !world.world.is_complete() && world.world.timestep < 500_000 {
        let step = world.run_step(CHECK_AREA, &mut motions, &mut float_world);
        if let Err(e) = step {
            println!(
                "Simulation error on step {} puzzle {}: {:?}: {}",
                world.world.timestep, sol.puzzle_name, fpath, e
            );
            return CheckResult::FailedSim;
        }
    }
    let mut newstats = world.get_stats();
    let oldstats = sol.stats.as_ref().unwrap();
    if puzzle.production || !CHECK_AREA {
        newstats.area = oldstats.area
    };
    if &newstats == oldstats {
        return CheckResult::Ok;
    } else {
        println!(
            "Stats don't match! {:?} and puzzle {}\n{:?} vs true {:?}",
            fpath, sol.puzzle_name, newstats, oldstats
        );
        return CheckResult::FailedStatMismatch;
    }
}

#[test]
fn check_all() {
    std::env::set_var("RUST_BACKTRACE", "full");
    let mut puzzle_map = PuzzleMap::new();
    read_puzzle_recurse(&mut puzzle_map, PUZZLE_DIR);
    let mut stats = (0, 0);
    let mut cb = |fpath: PathBuf, solution| {
        match check_solution(&fpath, &solution, &puzzle_map) {
            CheckResult::Skipped => (),
            CheckResult::FailedSetup => (),
            CheckResult::FailedSim => {
                stats.1 += 1;
            }
            CheckResult::FailedStatMismatch => {
                stats.1 += 1;
            }
            CheckResult::Ok => {
                stats.0 += 1;
                stats.1 += 1;
            }
        }
        if stats.1 % 100 == 0 {
            println!("Current progress: {}/{}", stats.0, stats.1);
        }
    };
    read_solution_recurse(&mut cb, &puzzle_map, "test/solution");
    read_solution_recurse(&mut cb, &puzzle_map, "test/om-leaderboard-master");
    println!("final score = {}/{}", stats.0, stats.1);
}
