use crate::parser::*;
use crate::sim::*;
use std::{collections::HashMap, fs, fs::File, io::BufReader, path::Path};

pub const PUZZLE_DIR: &str = "test/puzzle";
pub type PuzzleMap = HashMap<String, FullPuzzle>;
pub fn read_puzzle_recurse(puzzle_map: &mut PuzzleMap, directory: fs::ReadDir) {
    for f in directory {
        if let Ok(f) = f {
            let ftype = f.file_type().unwrap();
            if ftype.is_dir() {
                read_puzzle_recurse(puzzle_map, fs::read_dir(f.path()).unwrap());
            } else if ftype.is_file() {
                let f_puzzle = File::open(f.path()).unwrap();
                let fname = f.file_name().into_string().unwrap().replace(".puzzle", "");
                if let Ok(puzzle) = parse_puzzle(&mut BufReader::new(f_puzzle)) {
                    if puzzle.outputs.iter().any(|atoms| {
                        atoms
                            .iter()
                            .any(|atom| atom.atom_type == AtomType::RepeatingOutputMarker)
                    }) {
                        println!("Skipping infinite: {} | {}", fname, puzzle.puzzle_name);
                    } else {
                        puzzle_map.insert(fname, puzzle);
                    }
                } else {
                    println!("Puzzle failed to load: {:?}", f.path());
                }
            }
        }
    }
}

pub fn get_default_path_strs() -> (&'static str, &'static str, &'static str) {
    const DEFAULT_PATHS: &str = include_str!("default_paths.txt");
    let mut path_data = DEFAULT_PATHS.lines();
    let base = path_data.next().unwrap();
    let puzzle = path_data.next().unwrap();
    let solution = path_data.next().unwrap();
    (base, puzzle, solution)
}
