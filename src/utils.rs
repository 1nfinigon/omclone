use crate::parser;
use crate::sim::*;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use eyre::Result;

pub const PUZZLE_DIR: &str = "test/puzzle";

/// A puzzle has two names: the name that is saved in the puzzle file itself, and the
/// "fname" (filename stem). Interestingly, the string encoded in the solution
/// file is the fname, not the puzzle name. So, we key the PuzzleMap by fname.
pub type PuzzleMap = HashMap<String, (PathBuf, parser::FullPuzzle)>;

pub fn read_puzzle_recurse(puzzle_map: &mut PuzzleMap, directory: impl AsRef<Path>) {
    for f in fs::read_dir(directory).unwrap() {
        if let Ok(f) = f {
            let ftype = f.file_type().unwrap();
            if ftype.is_dir() {
                read_puzzle_recurse(puzzle_map, &f.path());
            } else if ftype.is_file() {
                let f_puzzle = File::open(f.path()).unwrap();
                let fname = f.file_name().into_string().unwrap();
                let fname = fname.strip_suffix(".puzzle").unwrap_or(&fname);
                if let Ok(puzzle) = parser::parse_puzzle(&mut BufReader::new(f_puzzle)) {
                    if puzzle.outputs.iter().any(|atoms| {
                        atoms
                            .iter()
                            .any(|atom| atom.atom_type == AtomType::RepeatingOutputMarker)
                    }) {
                        println!("Skipping infinite: {} | {}", fname, puzzle.puzzle_name);
                    } else {
                        let existing_puzzle =
                            puzzle_map.insert(fname.to_string(), (f.path(), puzzle));
                        assert!(existing_puzzle.is_none());
                    }
                } else {
                    println!("Puzzle failed to load: {:?}", f.path());
                }
            }
        }
    }
}

pub fn read_file_suffix_recurse<F: FnMut(PathBuf)>(
    cb: &mut F,
    suffix: &str,
    directory: impl AsRef<Path>,
) {
    for f in fs::read_dir(directory).unwrap() {
        if let Ok(f) = f {
            let ftype = f.file_type().unwrap();
            if ftype.is_dir() {
                read_file_suffix_recurse(cb, suffix, &f.path());
            } else if ftype.is_file() {
                let fname = f.file_name().into_string().unwrap();
                if fname.ends_with(suffix) {
                    cb(f.path());
                }
            }
        }
    }
}

pub fn verify_solution(
    fpath: impl AsRef<Path>,
    puzzle_map: &PuzzleMap,
) -> Option<parser::FullSolution> {
    let f_sol = File::open(fpath.as_ref()).unwrap();
    let sol_maybe = parser::parse_solution(&mut BufReader::new(f_sol));
    let sol = match sol_maybe {
        Err(e) => {
            println!("Failed to parse solution {:?}: {}", fpath.as_ref(), e);
            return None;
        }
        Ok(s) => s,
    };
    if !puzzle_map.contains_key(&sol.puzzle_name) {
        /*println!("Can't find puzzle {}",sol.puzzle_name);*/
        return None;
    }
    Some(sol)
}

pub fn read_solution_recurse<F: FnMut(PathBuf, parser::FullSolution)>(
    cb: &mut F,
    puzzle_map: &PuzzleMap,
    directory: impl AsRef<Path>,
) {
    read_file_suffix_recurse(
        &mut |fpath: PathBuf| {
            if let Some(sol) = verify_solution(&fpath, puzzle_map) {
                cb(fpath, sol)
            }
        },
        ".solution",
        directory,
    )
}

pub fn get_default_path_strs() -> (&'static str, &'static str, &'static str) {
    const DEFAULT_PATHS: &str = include_str!("default_paths.txt");
    let mut path_data = DEFAULT_PATHS.lines();
    let base = path_data.next().unwrap();
    let puzzle = path_data.next().unwrap();
    let solution = path_data.next().unwrap();
    (base, puzzle, solution)
}

pub fn get_default_puzzle_solution() -> Result<(parser::FullPuzzle, parser::FullSolution)> {
    let (base_str, puzzle_str, solution_str) = get_default_path_strs();
    let base_path = Path::new(base_str);
    let f_puzzle = File::open(base_path.join(puzzle_str))?;
    let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puzzle))?;
    let f_sol = File::open(base_path.join(solution_str))?;
    let sol = parser::parse_solution(&mut BufReader::new(f_sol))?;
    Ok((puzzle, sol))
}
