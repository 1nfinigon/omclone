use crate::parser::*;
use crate::sim::*;
use crate::utils::*;
use std::{collections::HashMap, fs, fs::File, io::BufReader, path::Path};
const CHECK_AREA: bool = false;

fn read_solution_recurse(
    stats: &mut (usize, usize),
    puzzle_map: &PuzzleMap,
    directory: fs::ReadDir,
) {
    for f in directory {
        if let Ok(f) = f {
            let ftype = f.file_type().unwrap();
            if ftype.is_dir() {
                read_solution_recurse(stats, puzzle_map, fs::read_dir(f.path()).unwrap());
            } else if ftype.is_file() {
                let fpath = f.path();
                let lname = fpath.to_str().unwrap().to_ascii_lowercase();
                /*if lname.contains("overlap") || lname.contains("tourney-2019\\week6") || lname.contains("tourney-2021\\week 6"){
                    //println!("----{:?}",fpath);
                    continue;
                }*/
                if lname.contains(".solution") {
                    check_solution(stats, &fpath, puzzle_map);
                }
            }
        }
    }
}

fn check_solution(stats: &mut (usize, usize), fpath: &Path, puzzle_map: &PuzzleMap) {
    let f_sol = File::open(fpath).unwrap();
    let sol_maybe = parse_solution(&mut BufReader::new(f_sol));
    let sol = match sol_maybe {
        Err(e) => {
            println!("Failed to parse solution {:?}: {}", fpath, e);
            return;
        }
        Ok(s) => s,
    };
    let puzzle = puzzle_map.get(&sol.puzzle_name);
    if puzzle.is_none() {
        /*println!("Can't find puzzle {}",sol.puzzle_name);*/
        return;
    }
    if let Some(puzzle) = puzzle {
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
        let oldstats = sol.stats.unwrap();
        if puzzle.production || !CHECK_AREA {
            newstats.area = oldstats.area
        };
        if newstats == oldstats {
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
}

#[test]
fn check_all() {
    std::env::set_var("RUST_BACKTRACE", "full");
    let mut puzzle_map = PuzzleMap::new();
    read_puzzle_recurse(&mut puzzle_map, fs::read_dir(PUZZLE_DIR).unwrap());
    let mut stats = (0, 0);
    read_solution_recurse(
        &mut stats,
        &puzzle_map,
        fs::read_dir("test/solution").unwrap(),
    );
    read_solution_recurse(
        &mut stats,
        &puzzle_map,
        fs::read_dir("test/om-leaderboard-master").unwrap(),
    );
    println!("final score = {}/{}", stats.0, stats.1);
}
