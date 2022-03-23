#[cfg(feature = "cx_checker")]
use wasm_bindgen::prelude::*;
mod parser;
mod sim;

#[cfg(feature = "cx_checker")]
#[wasm_bindgen]
extern "C" {
    fn add_text(s: &str);
    fn add_square(white: bool);
}

#[cfg(feature = "cx_checker")]
#[wasm_bindgen]
pub fn evaluate_solution(solution: &[u8]){
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let puzzle_baseline = include_bytes!("../OM2022_TransmutationCX.puzzle");
    let puzzle_read = &mut puzzle_baseline.as_slice();
    let mut puzzle = parser::parse_puzzle(puzzle_read).unwrap();
    //First input is data, atoms after 0 (1-6) are input
    //Only/first output is data, atoms after 0 (1-6) are output
    //But are differently ordered, so do it via checking the positions?
    let mut reader = solution;
    let sol = parser::parse_solution(&mut reader).unwrap();
    add_text(&format!("solution: {}",sol.solution_name));

    let mut any_failures = false;
    let mut slowest = 0;
    let mut float_world = sim::FloatWorld::new();
    let mut motions = sim::WorldStepInfo::new();
    'variants: for variant in 0..64{
        let mut input_arr = [false;6];
        for idx in 0..6{
            input_arr[idx] = (variant>>idx) & 0x1 == 1;
        }

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
        let mut output_string = "".to_string();
        for idx in 0..6{
            let output_bool = rule_cx(input_arr[(idx+5)%6],
                                      input_arr[idx],
                                      input_arr[(idx+1)%6]);
            let pos = sim::rot_to_pos(idx as i32);
            output_string += if output_bool {"F"} else {"S"};
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
        
        let init = parser::puzzle_prep(&puzzle, &sol).unwrap();
        let mut world = sim::World::setup_sim(&init).unwrap();
        let input_string:String = input_arr.map(|b|if b {'F'} else {'S'}).iter().collect();
        let fail_str = format!("ID {}, input {}, output {}",variant,input_string,output_string);

        while !world.is_complete() {
            if let Err(error_out) = world.run_step(true, &mut motions, &mut float_world){
                any_failures = true;
                add_square(false);
                add_text(&format!("{}: Step {} error: {}", fail_str, world.timestep, error_out));
                continue 'variants;
            }
            if world.timestep > 10_000_000{
                any_failures = true;
                add_square(false);
                add_text(&format!("{}: Over 10 million steps simmed, assuming failure", fail_str));
                continue 'variants;
            }
        }
        add_square(true);
        if world.timestep > slowest{
            slowest = world.timestep;
        }
    }
    if !any_failures{
        add_text(&format!("All good! Slowest solve: {}",slowest));
    } else {
        add_text("Fire/Salt patterns are in counter-clockwise order, starting from the right");
    }
}
