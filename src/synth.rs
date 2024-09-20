mod parser;
mod sim;

#[cfg(feature = "color_eyre")]
use color_eyre::{eyre::Result, install};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{eyre::Result, install};

use std::{fs::File, path::PathBuf};
use std::{io::prelude::*, io::BufReader, io::BufWriter};

fn main() -> Result<()> {
    std::env::set_var("RUST_BACKTRACE", "full");
    install()?;

    let (base_str, puzzle_str, _solution_str) = sim::get_default_path_strs();
    let base = PathBuf::from(String::from(base_str));
    let puzzle = String::from(puzzle_str);
    //let solution = String::from(solution_str);
    let puzzle = File::open(base.join(puzzle))?;
    let mut puzzle = parser::parse_puzzle(&mut BufReader::new(puzzle))?;
    //let solution = parser::parse_solution(&mut BufReader::new(f_sol))?;
    puzzle.puzzle_name.push_str("(SYNTH)");
    //
    let output = File::create("/tmp/out.puzzle")?;
    let mut output = BufWriter::new(output);
    parser::write_puzzle(&mut output, &puzzle)?;

    println!("done");
    Ok(())
}
