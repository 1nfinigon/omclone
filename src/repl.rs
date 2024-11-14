//! REPL for scripting the sim. Primarily useful for exporting to PNG

use std::fs::File;
use std::io::{BufReader, IsTerminal, Write};
use std::sync::{mpsc, Arc};
use std::thread;

use crate::parser;
use crate::render_sim;
#[cfg(feature = "search")]
use crate::search_state;
use crate::sim;
use crate::utils;

use eyre::{ensure, eyre, OptionExt, Result, WrapErr};
use miniquad::*;
use once_cell::sync::OnceCell;

struct Handler {
    mq_ctx: Box<dyn RenderingBackend>,
    puzzle_map: Option<utils::PuzzleMap>,
    /// What params to use when (re)loading
    load_params: Option<LoadParams>,

    world: Option<sim::WorldWithTapes>,
    #[cfg(feature = "search")]
    search_state: Option<search_state::State>,

    stdin_thread: Option<thread::JoinHandle<Result<()>>>,
    stdin_rx: mpsc::Receiver<(String, Arc<OnceCell<()>>)>,
}

struct LoadParams {
    puzzle: parser::FullPuzzle,
    solution: parser::FullSolution,
    rotate: Option<sim::RawRot>,
    #[cfg(feature = "torch")]
    recentre_to_fit_within_nn_bounds: bool,
}

impl Handler {
    fn new(mq_ctx: Box<dyn RenderingBackend>) -> Self {
        let (stdin_tx, stdin_rx) = mpsc::sync_channel(0);
        let stdin_thread = thread::Builder::new()
            .name("stdin handler".to_string())
            .spawn(move || {
                let stdin = std::io::stdin();
                let mut stdout = std::io::stdout();
                loop {
                    let mut buf = String::new();
                    if stdin.is_terminal() {
                        write!(stdout, "omclone> ")?;
                        stdout.flush()?;
                    }
                    if stdin.read_line(&mut buf)? == 0 {
                        break Ok(());
                    }

                    let complete = Arc::new(OnceCell::new());
                    stdin_tx.send((buf, complete.clone()))?;
                    if stdin.is_terminal() {
                        complete.wait();
                    }
                }
            })
            .expect("couldn't spawn stdin thread");

        Self {
            mq_ctx,
            puzzle_map: None,
            load_params: None,
            world: None,
            #[cfg(feature = "search")]
            search_state: None,
            stdin_thread: Some(stdin_thread),
            stdin_rx,
        }
    }

    fn get_or_load_puzzle_map(&mut self) -> &utils::PuzzleMap {
        self.puzzle_map.get_or_insert_with(|| {
            let mut puzzle_map = Default::default();
            utils::read_puzzle_recurse(&mut puzzle_map, utils::PUZZLE_DIR);
            puzzle_map
        })
    }

    fn reload(&mut self) -> Result<()> {
        let LoadParams {
            puzzle,
            solution,
            rotate,
            #[cfg(feature = "torch")]
            recentre_to_fit_within_nn_bounds,
        } = self
            .load_params
            .as_ref()
            .ok_or_eyre("load parameters not set")?;
        let mut init = parser::puzzle_prep(puzzle, solution)?;

        if let Some(rot) = rotate {
            init.rot_by(*rot);
        }

        #[cfg(feature = "torch")]
        if *recentre_to_fit_within_nn_bounds {
            use crate::nn;
            println!("Recentre: old bounding box {:?}", init.bounding_box());
            init.recentre_to_fit_within(sim::Pos::new(
                nn::constants::N_WIDTH as i32,
                nn::constants::N_HEIGHT as i32,
            ))?;
            println!("Recentre: new bounding box {:?}", init.bounding_box());
        }

        // suppress `mut` warning if compiling without feature "torch"
        let _ = &mut init;

        let world = sim::WorldWithTapes::setup_sim(&init)?;
        self.world = Some(world.clone());

        #[cfg(feature = "search")]
        {
            self.search_state = Some(search_state::State::new(
                world.world,
                solution
                    .stats
                    .as_ref()
                    .and_then(|stats| stats.cycles.try_into().ok())
                    .unwrap_or(u64::MAX),
            ));
        }
        Ok(())
    }

    fn load(&mut self, load_params: LoadParams) -> Result<()> {
        self.load_params = Some(load_params);
        self.reload()?;
        Ok(())
    }

    fn cmd_load<S: AsRef<str>>(&mut self, args: &[S]) -> Result<()> {
        let mut puzzle_path = None;
        let mut solution_path = None;
        let mut rotate = None;
        #[cfg(feature = "torch")]
        let mut recentre_to_fit_within_nn_bounds = false;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_ref() {
                "--puzzle" => {
                    puzzle_path = Some(
                        args.next()
                            .ok_or_eyre("--puzzle missing filename")?
                            .as_ref()
                            .to_owned(),
                    );
                }
                "--solution" => {
                    solution_path = Some(
                        args.next()
                            .ok_or_eyre("--solution missing filename")?
                            .as_ref()
                            .to_owned(),
                    );
                }
                "--rotate" => {
                    rotate = Some(sim::RawRot(
                        args.next()
                            .ok_or_eyre("--rotate missing int")?
                            .as_ref()
                            .parse::<i32>()?,
                    ));
                }
                #[cfg(feature = "torch")]
                "--recentre-to-fit-within-nn-bounds" => {
                    recentre_to_fit_within_nn_bounds = true;
                }
                arg => {
                    return Err(eyre!("unknown arg {}", arg));
                }
            }
        }

        let (puzzle, solution) = match (puzzle_path, solution_path) {
            (None, None) => utils::get_default_puzzle_solution()?,
            (None, Some(solution_path)) => {
                let (puzzle, solution) =
                    utils::get_solution(&solution_path, self.get_or_load_puzzle_map())?;
                let puzzle = puzzle
                    .ok_or_eyre(
                        "couldn't find puzzle for the given solution, please specify explicitly",
                    )?
                    .clone();
                (puzzle, solution)
            }
            (Some(puzzle_path), Some(solution_path)) => {
                let f_puz = File::open(&puzzle_path)?;
                let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puz))
                    .wrap_err(eyre!("Failed to parse puzzle {:?}", &puzzle_path))?;
                let f_sol = File::open(&solution_path)?;
                let solution = parser::parse_solution(&mut BufReader::new(f_sol))
                    .wrap_err(eyre!("Failed to parse solution {:?}", &solution_path))?;
                (puzzle, solution)
            }
            (Some(_), None) => Err(eyre!("when passing --puzzle, need --solution too"))?,
        };

        self.load(LoadParams {
            puzzle,
            solution,
            rotate,
            #[cfg(feature = "torch")]
            recentre_to_fit_within_nn_bounds,
        })
    }

    fn cmd_cycle<S: AsRef<str>>(&mut self, args: &[S]) -> Result<()> {
        match args {
            [cycle_string] => {
                let world = self.world.as_ref().ok_or_eyre("world not loaded")?;
                let cycle = cycle_string.as_ref().parse::<u64>()?;
                if world.world.cycle > cycle {
                    self.reload()?;
                }

                let world = self.world.as_mut().ok_or_eyre("world not loaded")?;
                #[cfg(feature = "search")]
                let search_state = self.search_state.as_mut().ok_or_eyre("world not loaded")?;

                ensure!(world.world.cycle <= cycle);
                let mut float_world = sim::FloatWorld::new();
                let mut motion = sim::WorldStepInfo::new();
                while world.world.cycle < cycle {
                    #[cfg(feature = "search")]
                    for instr in world.get_instructions() {
                        search_state.update(instr);
                    }

                    world.run_step(false, &mut motion, &mut float_world)?;
                }

                ensure!(world.world.cycle == cycle);
                #[cfg(feature = "search")]
                {
                    ensure!(search_state.world.cycle == cycle);
                    ensure!(!search_state.errored);
                }

                Ok(())
            }
            _ => Err(eyre!("bad args to cycle")),
        }
    }

    fn cmd_render<S: AsRef<str>>(&mut self, args: &[S]) -> Result<()> {
        let world = &self.world.as_ref().ok_or_eyre("no world loaded")?.world;

        let mut width = 256u32;
        let mut height = 256u32;
        let mut output_filename = "output.png".to_string();
        let mut offset = None;
        let mut scale = None;

        let mut args = args.into_iter();
        while let Some(arg) = args.next() {
            match arg.as_ref() {
                "--filename" => {
                    output_filename = args
                        .next()
                        .ok_or_eyre("--output missing filename")?
                        .as_ref()
                        .to_owned();
                }
                "--offset" => {
                    let offset_str: Vec<_> = args
                        .next()
                        .ok_or_eyre("--offset missing arg")?
                        .as_ref()
                        .split(',')
                        .collect();
                    let (x, y) = if let [x_str, y_str] = &offset_str[..] {
                        Ok((x_str.parse::<f32>()?, y_str.parse::<f32>()?))
                    } else {
                        Err(eyre!("bad offset string format, expected x,y"))
                    }?;
                    offset = Some((x, y));
                }
                "--scale" => {
                    scale = Some(
                        args.next()
                            .ok_or_eyre("--scale missing arg")?
                            .as_ref()
                            .parse::<f32>()?,
                    );
                }
                "--size" => {
                    let size_str: Vec<_> = args
                        .next()
                        .ok_or_eyre("--size missing arg")?
                        .as_ref()
                        .split('x')
                        .collect();
                    let (w, h) = if let [w_str, h_str] = &size_str[..] {
                        Ok((w_str.parse::<u32>()?, h_str.parse::<u32>()?))
                    } else {
                        Err(eyre!("bad size string format, expected wxh"))
                    }?;
                    width = w;
                    height = h;
                }
                arg => {
                    return Err(eyre!("unknown arg {}", arg));
                }
            }
        }

        let screen_size = (width as f32, height as f32);

        let color_img = self.mq_ctx.new_render_texture(TextureParams {
            width,
            height,
            format: TextureFormat::RGBA8,
            ..Default::default()
        });

        let offscreen_pass = self.mq_ctx.new_render_pass(color_img, None);

        let render_data = render_sim::RenderDataBase::new(self.mq_ctx.as_mut());
        self.mq_ctx.clear(Some((1., 1., 1., 1.)), None, None);
        self.mq_ctx.begin_pass(
            Some(offscreen_pass),
            PassAction::clear_color(0.5, 0.5, 0.5, 1.0),
        );

        let mut camera = render_sim::CameraSetup::frame_center(world, screen_size);
        println!("Default camera settings: {:?}", camera);
        if let Some((x, y)) = offset {
            camera.offset = [x, y];
        }
        if let Some(scale) = scale {
            camera.scale_base = scale;
        }

        let tracks = render_sim::setup_tracks(self.mq_ctx.as_mut(), &world.track_maps);
        let mut float_world = sim::FloatWorld::new();
        float_world.generate_static(world);

        render_data.draw(
            self.mq_ctx.as_mut(),
            screen_size,
            &camera,
            &tracks,
            false,
            world,
            &float_world,
        );

        self.mq_ctx.end_render_pass();

        let mut color_img_bytes = vec![0u8; (width * height * 4).try_into()?];
        self.mq_ctx
            .texture_read_pixels(color_img, &mut color_img_bytes);

        let color_img = image::RgbaImage::from_raw(width, height, color_img_bytes)
            .ok_or_eyre("failed creating image from raw data")?;
        let mut color_img = image::DynamicImage::ImageRgba8(color_img);
        color_img.apply_orientation(image::metadata::Orientation::FlipVertical);
        color_img.save(output_filename)?;

        Ok(())
    }

    #[cfg(feature = "torch")]
    fn cmd_nn_save_input<S: AsRef<str>>(&mut self, args: &[S]) -> Result<()> {
        use crate::{eval, nn};

        let tracy_client = tracy_client::Client::start();

        let search_state = self.search_state.as_ref().ok_or_eyre("world not loaded")?;

        match args {
            [output_filename] => {
                let model_input_tensors = nn::ModelInputTensors::from_eval_input_batched(
                    &[eval::EvalInput {
                        state: search_state.clone(),
                        is_root: true,
                    }],
                    (tch::Kind::Float, tch::Device::Cpu),
                    tracy_client,
                )?;
                model_input_tensors.save(output_filename.as_ref())?;
                Ok(())
            }
            _ => Err(eyre!("expected filename")),
        }
    }

    fn cmd(&mut self, input: &str) -> Result<()> {
        let mut shlex = shlex::Shlex::new(input);
        if let Some(cmd) = shlex.next() {
            let args: Vec<_> = shlex.by_ref().collect();
            let result = if shlex.had_error {
                Err(eyre!("bad syntax (unclosed quote?)"))
            } else {
                match cmd.as_ref() {
                    "load" => self.cmd_load(&args),
                    "render" => self.cmd_render(&args),
                    "cycle" => self.cmd_cycle(&args),
                    #[cfg(feature = "torch")]
                    "nn.save_input" => self.cmd_nn_save_input(&args),
                    _ => Err(eyre!("unknown command {}", cmd)),
                }
            };
            match result {
                Ok(()) => (),
                Err(e) => {
                    println!("{:#}", e);
                }
            }
        }
        Ok(())
    }
}

impl EventHandler for Handler {
    fn update(&mut self) {
        loop {
            match self.stdin_rx.recv() {
                Ok((buf, complete)) => {
                    self.cmd(&buf).unwrap();
                    complete.set(()).unwrap();
                }
                Err(mpsc::RecvError) => {
                    self.stdin_thread.take().map(|thread| thread.join());
                    window::order_quit();
                    return;
                }
            }
        }
    }

    fn draw(&mut self) {}
}

pub fn main() -> Result<()> {
    miniquad::start(
        miniquad::conf::Conf {
            window_title: "Offscreen renderer".to_owned(),
            window_width: 0,
            window_height: 0,
            fullscreen: false,
            ..Default::default()
        },
        move || {
            #[cfg(target_os = "macos")]
            {
                use objc::{runtime::*, *};
                unsafe {
                    let ns_app: *mut Object = msg_send![class!(NSApplication), sharedApplication];
                    let ns_application_activation_policy_prohibited = 2;
                    let () = msg_send![ns_app, setActivationPolicy: ns_application_activation_policy_prohibited];
                }
            }

            Box::new(Handler::new(window::new_rendering_backend()))
        },
    );

    Ok(())
}
