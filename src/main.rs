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
    let init = parser::puzzle_prep(&puzzle, &sol)?;

    let mut world = sim::World::setup_sim(&init)?;
    /*while !world.is_complete() {
        world.run_step()?
        let stats = world.get_stats();
        println!("Step {:03}", stats.cycles);
    }
    let stats = world.get_stats();
    println!("Complete! {:?}", stats);*/
    let mut float_world = sim::FloatWorld::new();
    let mut motions = sim::WorldStepInfo::new();
	loop {
        world.run_step(true, &mut motions, &mut float_world)?;
    }
}

#[cfg(feature = "main_ui")]
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

#[cfg(feature = "cx_checker")]
fn main() -> Result<()> {
    use std::{fs::File, io::BufReader};
    #[cfg(color_eyre)]
    color_eyre::install()?;
    #[cfg(not(color_eyre))]
    simple_eyre::install()?;

    let solution_str = "transmutation-cx-OM2022_TransmutationCX-1.solution";

    let puzzle_baseline = include_bytes!("../OM2022_TransmutationCX.puzzle");
    let puzzle_read = &mut puzzle_baseline.as_slice();
    let mut puzzle = parser::parse_puzzle(puzzle_read)?;
    //First input is data, atoms after 0 (1-6) are input
    //Only/first output is data, atoms after 0 (1-6) are output
    //But are differently ordered, so do it via checking the positions?

    let f_sol = File::open(solution_str)?;
    let sol = parser::parse_solution(&mut BufReader::new(f_sol))?;
    println!("solution: {}",sol.solution_name);

    let mut any_failures = false;
    let mut slowest = 0;
    let mut float_world = sim::FloatWorld::new();
    let mut motions = sim::WorldStepInfo::new();
    'variants: for variant in 0..64{
        let mut input_arr = [false;6];
        for idx in 0..6{
            input_arr[idx] = (variant>>idx) & 0x1 == 1;
        }
        /*println!("");
        println!("Begin!");
        println!("{:?}",input_arr);*/

        fn rule_cx(a: bool, b: bool, c:bool) -> bool{
            let mut tmp = 0;
            if a {tmp += 4};
            if b {tmp += 2};
            if c {tmp += 1};
            const CX_ARR: [bool;8] = [false, true, true, true, false, true, true, false];
            CX_ARR[tmp]
        }
        fn to_type(a: bool) -> sim::AtomType{
            if a {sim::AtomType::Fire} else {sim::AtomType::Salt}
        }
        for idx in 0..6{
            let output_bool = rule_cx(input_arr[(idx+5)%6],
                                      input_arr[idx],
                                      input_arr[(idx+1)%6]);
            let pos = sim::rot_to_pos(idx as i32);
            //println!("at {},{} : {},{},{} -> {}", pos.x, pos.y ,input_arr[(idx+5)%6],input_arr[(idx+0)%6],input_arr[(idx+1)%6],output_bool);
            for atom in &mut puzzle.inputs[0]{
                if atom.pos == pos{
                    atom.atom_type = to_type(input_arr[idx]);
                }
            }
            for atom in &mut puzzle.outputs[0]{
                if atom.pos == pos{
                    atom.atom_type = to_type(output_bool);
                }
            }
        }
        /*println!("input: ");
        for atom in &puzzle.inputs[0]{
            println!("atom at {},{} is {:?}", atom.pos.x, atom.pos.y ,atom.atom_type);
        }
        println!("output: ");
        for atom in &puzzle.outputs[0]{
            println!("atom at {},{} is {:?}", atom.pos.x, atom.pos.y ,atom.atom_type);
        }*/
        
        println!("simming {}",variant);
        let init = parser::puzzle_prep(&puzzle, &sol)?;
        let mut world = sim::World::setup_sim(&init)?;

        while !world.is_complete() {
            if let Err(error_out) = world.run_step(true, &mut motions, &mut float_world){
                any_failures = true;
                println!("Step {} error: {}", world.timestep, error_out);
                break 'variants;
            }
            if world.timestep > 1_000_000{
                any_failures = true;
                println!("Over 10 million steps simmed, assuming failure");
                break 'variants;
            }
        }
        if world.timestep > slowest{
            slowest = world.timestep;
        }
    }
    if !any_failures{
        println!("All good! Slowest solve: {}",slowest)
    }
    Ok(())
}
