
use crate::sim::*;
use crate::parser::*;
const CHECK_AREA:bool = true;
#[test]
fn check_all(){
    use std::{fs,fs::File, io::BufReader, collections::HashMap};
    std::env::set_var("RUST_BACKTRACE", "full");
    let mut puzzle_map:HashMap<String, FullPuzzle> = HashMap::new();
    fn read_puzzle_recurse(puzzle_map:&mut HashMap<String, FullPuzzle>, directory: fs::ReadDir){
        for f in directory{
            if let Ok(f) = f{
                let ftype = f.file_type().unwrap();
                if ftype.is_dir(){
                    read_puzzle_recurse(puzzle_map,fs::read_dir(f.path()).unwrap());
                } else if ftype.is_file(){
                    let f_puzzle = File::open(f.path()).unwrap();
                    if let Ok(puzzle) = parse_puzzle(&mut BufReader::new(f_puzzle)){
                        if puzzle.outputs.iter().any(|atoms_meta| atoms_meta[0].iter().any(
                                        |atom|atom.atom_type == AtomType::RepeatingOutputMarker)){
                            println!("Skipping infinite: {}",puzzle.puzzle_name);
                        } else {
                            let fname = f.file_name().into_string().unwrap();
                            puzzle_map.insert(fname.replace(".puzzle",""), puzzle);
                        }
                    } else {
                        println!("Puzzle failed to load: {:?}",f.path());
                    }
                }
            }
        }
    }
    read_puzzle_recurse(&mut puzzle_map, fs::read_dir("test/puzzle").unwrap());
    fn read_solution_recurse(stats: &mut(usize, usize), puzzle_map: &HashMap<String, FullPuzzle>, directory: fs::ReadDir){
        let mut float_world = FloatWorld::new();
        let mut motions = WorldStepInfo::new();
        'file_loop: for f in directory{
            if let Ok(f) = f{
                let ftype = f.file_type().unwrap();
                if ftype.is_dir(){
                    read_solution_recurse(stats, puzzle_map,fs::read_dir(f.path()).unwrap());
                } else if ftype.is_file(){
                    let fpath = f.path();
                    /*let lname = fpath.to_str().unwrap().to_ascii_lowercase();
                    if lname.contains("overlap") || lname.contains("tourney-2019\\week6") || lname.contains("tourney-2021\\Week 6"){
                        continue;
                    }*/
                    //println!("----{:?}",fpath);
                    let f_sol = File::open(f.path()).unwrap();
                    let sol_maybe = parse_solution(&mut BufReader::new(f_sol));
                    let sol = match sol_maybe{
                        Err(e) => {/*println!("Failed to parse solution {:?}: {}",fpath, e);*/ continue}
                        Ok(s) => s
                    };
                    let puzzle = puzzle_map.get(&sol.puzzle_name);
                    if puzzle.is_none(){
                        /*println!("Can't find puzzle {}",sol.puzzle_name);*/
                        continue;
                    }
                    if let Some(puzzle) = puzzle{
                        let init = match puzzle_prep(&puzzle, &sol){
                            Err(e) => {/*println!("Failed to prepare {:?}: {}",fpath, e);*/ continue}
                            Ok(s) => s
                        };
                        let mut world = match World::setup_sim(&init){
                            Err(e) => {/*println!("Failed to setup {:?}: {}",fpath, e);*/ continue}
                            Ok(s) => s
                        };
                        stats.1 += 1;
                        while !world.is_complete() && world.timestep < 500_000 {
                            let step = world.run_step(CHECK_AREA, &mut motions, &mut float_world);
                            if let Err(e) = step{
                                println!("Simulation error on step {} puzzle {}: {:?}: {}",world.timestep,sol.puzzle_name,fpath, e);
                                continue 'file_loop;
                            }
                        }
                        let mut newstats = world.get_stats();
                        let oldstats = sol.stats.unwrap();
                        if puzzle.production || !CHECK_AREA {newstats.area = oldstats.area};
                        if newstats == oldstats{
                            stats.0 += 1;
                        } else {
                            println!("Stats don't match! {:?} and puzzle {}\n{:?} vs true {:?}",fpath,sol.puzzle_name,newstats, oldstats);
                        }
                        if stats.1 % 100 == 0{
                            println!("Current progress: {}/{}",stats.0,stats.1);
                        }
                    }
                }
            }
        }
    }
    let mut stats = (0,0);
    read_solution_recurse(&mut stats, &puzzle_map, fs::read_dir("test/solution").unwrap());
    println!("final score = {}/{}",stats.0,stats.1);
}