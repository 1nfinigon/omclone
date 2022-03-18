mod parser;
mod sim;
mod render_sim;
mod ui;
#[cfg(feature = "color_eyre")]
use color_eyre::{install, eyre::Result};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{install, eyre::Result};


#[cfg(feature = "benchmark")]
fn main() -> Result<()> {
    use std::{fs::File, io::BufReader, path::Path};
    std::env::set_var("RUST_BACKTRACE", "full");
    #[cfg(color_eyre)]
    color_eyre::install()?;
    #[cfg(not(color_eyre))]
    simple_eyre::install()?;

    let (base_str, puzzle_str, solution_str) = ui::get_default_path_strs();
    let base_path = Path::new(base_str);
    let f_puzzle = File::open(base_path.join(puzzle_str))?;
    let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puzzle))?;
    let f_sol = File::open(base_path.join(solution_str))?;
    let sol = parser::parse_solution(&mut BufReader::new(f_sol))?;
    //println!("Check: {:?}", sol.stats);
    let init = parser::puzzle_prep(puzzle, sol)?;

    let mut world = sim::World::setup_sim(&init)?;
    /*while !world.is_complete() {
        world.run_step()?
        let stats = world.get_stats();
        println!("Step {:03}", stats.cycles);
    }
    let stats = world.get_stats();
    println!("Complete! {:?}", stats);*/
	loop {
		world.run_step()?;
    }
    Ok(())
}

#[cfg(not(feature = "benchmark"))]
fn main() -> Result< () >{
    std::env::set_var("RUST_BACKTRACE", "full");
    install()?;

    use miniquad::*;
    let conf = conf::Conf{
        fullscreen: false,
        .. Default::default()
    };
    miniquad::start(conf, |mut ctx| {
        UserData::owning(ui::MyMiniquadApp::new(&mut ctx), ctx)
    });
    Ok(())
}