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
    Skipped(String),
    Ok,
    FailedPrep(String),
    FailedSetup(String),
    FailedSim(String),
    FailedStatMismatch(String),
}

pub fn check_solution(sol: &FullSolution, puzzle: &FullPuzzle, skip_overlap: bool) -> CheckResult {
    let init = match puzzle_prep(&puzzle, &sol) {
        Err(e) => {
            return CheckResult::FailedPrep(e.to_string());
        }
        Ok(s) => s,
    };
    if skip_overlap && init.has_overlap() {
        return CheckResult::Skipped(format!("has overlap"));
    }
    let mut world = match WorldWithTapes::setup_sim(&init) {
        Err(e) => {
            return CheckResult::FailedSetup(e.to_string());
        }
        Ok(s) => s,
    };
    let mut float_world = FloatWorld::new();
    let mut motions = WorldStepInfo::new();
    while !world.world.is_complete() && world.world.timestep < 500_000 {
        let step = world.run_step(CHECK_AREA, &mut motions, &mut float_world);
        if let Err(e) = step {
            return CheckResult::FailedSim(format!(
                "Simulation error on step {}: {}",
                world.world.timestep, e
            ));
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
        return CheckResult::FailedStatMismatch(format!(
            "Stats don't match! {:?} vs true {:?}",
            newstats, oldstats
        ));
    }
}

#[test]
fn check_all() {
    std::env::set_var("RUST_BACKTRACE", "full");
    let mut puzzle_map = PuzzleMap::new();
    read_puzzle_recurse(&mut puzzle_map, PUZZLE_DIR);
    let mut stats = (0, 0);
    let mut cb = |fpath: PathBuf, solution: FullSolution| {
        let print_err = |kind: &str, details: &str| {
            println!("{}: {:?}: {}", kind, fpath, details);
        };
        let (_puzzle_fpath, puzzle) = puzzle_map.get(&solution.puzzle_name).expect(
            "at this point solution should have been paired up with a puzzle in puzzle_map already",
        );
        match check_solution(&solution, &puzzle, false) {
            CheckResult::Skipped(s) => (),
            CheckResult::FailedPrep(e) => {
                print_err("Failed during prep", &e);
            }
            CheckResult::FailedSetup(e) => {
                print_err("Failed during setup", &e);
            }
            CheckResult::FailedSim(e) => {
                print_err("Simulation error", &e);
                stats.1 += 1;
            }
            CheckResult::FailedStatMismatch(e) => {
                print_err("Stats do not match", &e);
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
