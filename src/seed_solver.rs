use crate::check;
use crate::nn;
use crate::parser;
use crate::search;
use crate::search_history;
use crate::search_state;
use crate::sim;
use crate::utils;

use rand::prelude::*;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use eyre::Result;

fn solve_one_puzzle_seeded(
    args: &Args,
    tracy_client: tracy_client::Client,
    puzzle_fpath: impl AsRef<Path>,
    seed_puzzle: &parser::FullPuzzle,
    solution_fpath: impl AsRef<Path>,
    seed_solution: &parser::FullSolution,
    model: &nn::Model,
    rng: &mut impl Rng,
) -> Result<()> {
    let mut seed_init = parser::puzzle_prep(seed_puzzle, seed_solution)?;

    match check::check_solution(seed_solution, seed_puzzle, true) {
        check::CheckResult::Ok => (),
        _ => {
            return Ok(());
        }
    }

    println!(
        "====== starting {:?}, seeding with {:?}, model {}",
        puzzle_fpath.as_ref(),
        solution_fpath.as_ref(),
        model.name
    );

    // Recentre the solution so that the bounding box is centred around (w/2, h/2)
    if let Some((min, max)) = seed_init.bounding_box() {
        let nn_wh = sim::Pos::new(
            nn::constants::N_WIDTH as i32,
            nn::constants::N_HEIGHT as i32,
        );
        let seed_wh = max - min;
        if seed_wh.x >= nn_wh.x || seed_wh.y >= nn_wh.y {
            println!(
                "skipping because solution footprint is too big: {} vs max {}",
                seed_wh, nn_wh
            );
            return Ok(());
        }
        let delta = (nn_wh - (max + min)) / 2;
        seed_init.move_by(delta);
    }

    // TODO: randomly rotate the seed world, then check that the post-rotate
    // world still passes checks
    let _ = sim::InitialWorld::rot_by;

    let seed_world = sim::WorldWithTapes::setup_sim(&seed_init)?;

    let first_timestep = seed_world.world.timestep;
    let n_arms = seed_world.world.arms.len() as u64;
    let n_moves = seed_solution.stats.as_ref().unwrap().cycles as u64 * n_arms;
    let n_moves_to_search = rng.gen_range(1..=args.max_cycles_from_optimal.unwrap_or(15)); // how many moves to leave behind for MCTS to find

    let mut search_state = search_state::State::new(
        seed_world.world.clone(),
        first_timestep
            + (n_moves + rng.gen_range(1..=args.max_cycles.unwrap_or(30)) + n_arms - 1) / n_arms,
    );
    let mut search_history = search_history::History::new();
    let mut tapes: Vec<sim::Tape<sim::BasicInstr>> = Vec::new();
    for _ in 0..seed_world.tapes.len() {
        tapes.push(sim::Tape {
            first: first_timestep as usize,
            instructions: Vec::new(),
        });
    }

    // make some pre-moves from the seed solution

    let n_premoves = n_moves.saturating_sub(n_moves_to_search);
    println!(
        "making {} premoves ({} cycles + {}; {} short of seed solution)",
        n_premoves,
        n_premoves / n_arms,
        n_premoves % n_arms,
        n_moves_to_search
    );

    for _ in 0..n_premoves {
        let arm_index = search_state.next_arm_index();
        let instr = seed_world.tapes[arm_index].get(
            search_state.world.timestep as usize,
            seed_world.repeat_length,
        );
        tapes[arm_index].instructions.push(instr);
        search_state.update(instr);
        search_history.append_from_optimal_solution(instr);
    }

    // search for a solution

    let mut still_following_premoves = true;
    let result_is_success = loop {
        if let Some(result) = search_state.evaluate_final_state() {
            println!("done; result = {}", result);
            break result > 0.;
        }

        let tree_search = search::TreeSearch::new(search_state.clone(), tracy_client.clone());

        let playouts = if rng.gen_bool(0.75) { 100 } else { 600 };

        (0..playouts)
            .into_par_iter()
            .map_init(
                || rand::thread_rng(),
                |rng, _| -> Result<()> { Ok(tree_search.search_once(rng, model)?) },
            )
            .collect::<Result<()>>()?;

        let stats = tree_search.next_updates_with_stats();

        //println!("{:?}", stats);

        let instr = stats.best_update();

        println!(
            "{}{:<23} #={} (v={:.3} (raw {:.3}) d={:>5.2}/{:>2}) {}",
            if search_state.next_arm_index() == 0 {
                "*"
            } else {
                " "
            },
            format!("{:?}", instr),
            playouts,
            stats.root_value,
            stats.root_raw_utility,
            stats.avg_depth,
            stats.max_depth,
            {
                let v: Vec<_> = stats
                    .updates_with_stats
                    .iter()
                    .map(|u| {
                        let brackets = if u.is_terminal {
                            ["#", "#"]
                        } else {
                            ["[", "]"]
                        };
                        format!(
                            "{}{}{:>3}{}",
                            brackets[0],
                            u.instr.to_char(),
                            u.visits * 100 / playouts,
                            brackets[1]
                        )
                    })
                    .collect();
                v.join(" ")
            }
        );

        // elbow: at what x value does the unsmoothed curve hit zero?
        // sharpness: how sharp the curve should be
        // intercept: what should the value at 0 be
        let lowconf = |x: f64, elbow: f64, sharpness: f64, intercept: f64| {
            // log(1+exp(6*(1-(x/0.5))))/6
            ((1. + (sharpness * (1. - (x / elbow))).exp()).ln() * intercept / sharpness)
                .clamp(0., 1.)
        };

        if still_following_premoves && rng.gen_bool(lowconf(stats.root_value.into(), 0.5, 6., 0.9))
        {
            let arm_index = search_state.next_arm_index();
            let instr = seed_world.tapes[arm_index].get(
                search_state.world.timestep as usize,
                seed_world.repeat_length,
            );
            println!("Applying true update due to low confidence: {:?}", instr);
            tapes[arm_index].instructions.push(instr);
            search_state.update(instr);
            search_history.append_from_optimal_solution(instr);
        } else {
            still_following_premoves = false;
            tapes[search_state.next_arm_index()]
                .instructions
                .push(instr);
            search_state.update(instr);
            search_history.append_mcts(&stats);
        }
    };

    // finalize world for saving

    let solution_name = {
        let mut bytes = [0u8; 16];
        rng.fill(&mut bytes[..]);
        uuid::Builder::from_random_bytes(bytes)
            .into_uuid()
            .to_string()
    };
    let solution_stats = if result_is_success {
        Some(sim::SolutionStats {
            cycles: (search_state.world.timestep - first_timestep)
                .try_into()
                .unwrap(),
            cost: search_state.world.cost,
            area: search_state.world.area_touched.len().try_into().unwrap(),
            instructions: sim::compute_tape_instruction_count(&tapes)
                .try_into()
                .unwrap(),
        })
    } else {
        None
    };
    let out_world = {
        let repeat_length = sim::compute_tape_repeat_length(&tapes);
        sim::WorldWithTapes {
            world: seed_world.world.clone(),
            tapes,
            repeat_length,
        }
    };
    let out_solution = parser::create_solution(
        &out_world,
        seed_solution.puzzle_name.clone(),
        solution_name.clone(),
        solution_stats,
    );
    let out_history = search_history::HistoryFile {
        solution_name: solution_name.clone(),
        history: search_history,
        timestep_limit: search_state.timestep_limit.try_into().unwrap(),
        final_outcome: if result_is_success { 1.0 } else { 0.0 },
    };

    // save solution and search history

    let out_basedir = PathBuf::from(format!("test/games/{}", model.name));
    if !std::fs::exists(&out_basedir)? {
        std::fs::create_dir(&out_basedir)?;
    }

    let out_solution_filename = out_basedir.join(format!("{}.solution", solution_name));
    println!("saving solution to {:?}", out_solution_filename);
    let mut f_out_solution = BufWriter::new(File::create(&out_solution_filename)?);
    parser::write_solution(&mut f_out_solution, &out_solution)?;
    std::mem::drop(f_out_solution);

    let out_history_filename = out_basedir.join(format!("{}.history", solution_name));
    println!("saving history to {:?}", out_history_filename);
    let mut f_out_history = BufWriter::new(File::create(&out_history_filename)?);
    out_history.write(&mut f_out_history)?;
    std::mem::drop(f_out_history);

    Ok(())
}

#[derive(Default)]
pub struct Args {
    threads: Option<usize>,
    forever: Option<()>,
    seed: Option<u64>,
    reload_model_every: Option<usize>,
    max_cycles: Option<u64>,
    max_cycles_from_optimal: Option<u64>,
}

impl Args {
    pub fn new(mut args: std::env::Args) -> Self {
        let mut self_ = Self::default();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--threads" => {
                    self_.threads = Some(
                        args.next()
                            .and_then(|s| s.parse::<usize>().ok())
                            .expect("--threads should be followed by a count"),
                    );
                }
                "--forever" => {
                    self_.forever = Some(());
                }
                "--seed" => {
                    self_.seed = Some(
                        args.next()
                            .and_then(|s| s.parse::<u64>().ok())
                            .expect("--seed should be followed by an integer seed"),
                    );
                }
                "--reload-model-every" => {
                    self_.reload_model_every = Some(
                        args.next()
                            .and_then(|s| s.parse::<usize>().ok())
                            .expect("--reload-model-every should be followed by a count"),
                    );
                }
                "--max-cycles" => {
                    self_.max_cycles = Some(
                        args.next()
                            .and_then(|s| s.parse::<u64>().ok())
                            .expect("--max-cycles should be followed by a count"),
                    );
                }
                "--max-cycles-from-optimal" => {
                    self_.max_cycles_from_optimal = Some(
                        args.next()
                            .and_then(|s| s.parse::<u64>().ok())
                            .expect("--max-cycles-from-optimal should be followed by a count"),
                    );
                }
                _ => panic!("Unknown arg {}", arg),
            }
        }
        self_
    }
}

fn run_one_epoch(
    args: &Args,
    tracy_client: tracy_client::Client,
    rng: &mut impl Rng,
    device: tch::Device,
    puzzle_map: &utils::PuzzleMap,
    seed_solution_paths: &mut Vec<PathBuf>,
) -> Result<()> {
    println!("shuffling seed solutions");
    seed_solution_paths.shuffle(rng);

    let mut model = nn::Model::load_latest(device, tracy_client.clone())?;
    let mut solves_since_model_reload = 0;
    for solution_fpath in seed_solution_paths.iter() {
        if let Some(seed_solution) = utils::verify_solution(solution_fpath, &puzzle_map) {
            let (puzzle_fpath, seed_puzzle) = puzzle_map.get(&seed_solution.puzzle_name).unwrap();

            solve_one_puzzle_seeded(
                args,
                tracy_client.clone(),
                puzzle_fpath,
                seed_puzzle,
                solution_fpath,
                &seed_solution,
                &model,
                rng,
            )?;

            solves_since_model_reload += 1;
            if solves_since_model_reload > args.reload_model_every.unwrap_or(usize::MAX) {
                solves_since_model_reload = 0;
                model = nn::Model::load_latest(device, tracy_client.clone())?;
            }
        }
    }

    Ok(())
}

pub fn main(args: std::env::Args, tracy_client: tracy_client::Client) -> Result<()> {
    let args = Args::new(args);

    println!(
        "{} spatial features\n{} spatiotemporal features\n{} temporal features",
        nn::feature_offsets::Spatial::SIZE,
        nn::feature_offsets::Spatiotemporal::SIZE,
        nn::feature_offsets::Temporal::SIZE,
    );
    println!(
        "{:?} input tensor size",
        std::mem::size_of::<nn::Features>()
    );

    //let (seed_puzzle, seed_solution) = utils::get_default_puzzle_solution()?;
    //solve_one_puzzle_seeded(&"", &seed_puzzle, &"", &seed_solution, &model, rng)?;

    println!("loading seed puzzles");
    let mut puzzle_map = utils::PuzzleMap::new();
    utils::read_puzzle_recurse(&mut puzzle_map, "test/puzzle");
    println!("loading seed solutions");
    let mut seed_solution_paths = Vec::new();
    let mut cb = |fpath: PathBuf| {
        seed_solution_paths.push(fpath);
    };
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/solution");
    utils::read_file_suffix_recurse(&mut cb, ".solution", "test/om-leaderboard-master");

    let device = nn::get_best_device()?;
    println!("Using device {:?}", device);

    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads.unwrap_or(4))
        .thread_name(|i| format!("search thread {}", i))
        .build()?
        .install(|| -> Result<()> {
            let mut rng: Box<dyn RngCore> = if let Some(seed) = args.seed {
                Box::new(rand_pcg::Pcg64::seed_from_u64(seed))
            } else {
                Box::new(rand::thread_rng())
            };
            let rng = &mut rng;

            match args.forever {
                Some(()) => loop {
                    run_one_epoch(
                        &args,
                        tracy_client.clone(),
                        rng,
                        device,
                        &puzzle_map,
                        &mut seed_solution_paths,
                    )?;
                },
                None => {
                    run_one_epoch(
                        &args,
                        tracy_client.clone(),
                        rng,
                        device,
                        &puzzle_map,
                        &mut seed_solution_paths,
                    )?;
                }
            }
            Ok(())
        })
}
