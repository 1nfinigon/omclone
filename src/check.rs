//! Utility functions for checking that a solution is valid.
//!
//! Also a binary for running all community solutions from external repos as a
//! test suite, as setup by `setup.sh`.

use crate::parser::*;
use crate::sim::*;
use crate::utils::*;
use eyre::Result;
use std::path::PathBuf;
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
    let init = match puzzle_prep(puzzle, sol) {
        Err(e) => {
            return CheckResult::FailedPrep(e.to_string());
        }
        Ok(s) => s,
    };
    if skip_overlap && init.has_overlap() {
        return CheckResult::Skipped("has overlap".to_string());
    }
    let mut world = match WorldWithTapes::setup_sim(&init) {
        Err(e) => {
            return CheckResult::FailedSetup(e.to_string());
        }
        Ok(s) => s,
    };
    let mut float_world = FloatWorld::new();
    let mut motions = WorldStepInfo::new();
    while !world.world.is_complete() && world.world.cycle < 500_000 {
        let step = world.run_step(CHECK_AREA, &mut motions, &mut float_world);
        if let Err(e) = step {
            return CheckResult::FailedSim(format!(
                "Simulation error on step {}: {}",
                world.world.cycle, e
            ));
        }
    }
    let mut newstats = world.get_stats();
    let oldstats = sol.stats.as_ref().unwrap();
    if puzzle.production || !CHECK_AREA {
        newstats.area = oldstats.area
    };
    if &newstats == oldstats {
        CheckResult::Ok
    } else {
        CheckResult::FailedStatMismatch(format!(
            "Stats don't match! {:?} vs true {:?}",
            newstats, oldstats
        ))
    }
}

pub fn main() -> Result<()> {
    let mut puzzle_map = PuzzleMap::new();
    read_puzzle_recurse(&mut puzzle_map, PUZZLE_DIR);
    let mut stats = (0, 0);
    let mut cb = |fpath: PathBuf, solution: FullSolution| {
        let (puzzle_fpath, puzzle) = puzzle_map.get(&solution.puzzle_name).expect(
            "at this point solution should have been paired up with a puzzle in puzzle_map already",
        );
        let print_err = |kind: &str, details: &str| {
            println!("{}: {:?} / {:?}: {}", kind, puzzle_fpath, fpath, details);
        };
        match check_solution(&solution, puzzle, false) {
            CheckResult::Skipped(_s) => {
                return;
            }
            CheckResult::FailedPrep(e) => {
                print_err("Failed during prep", &e);
                return;
            }
            CheckResult::FailedSetup(e) => {
                print_err("Failed during setup", &e);
                return;
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
    Ok(())
}
