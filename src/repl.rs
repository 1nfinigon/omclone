use std::fs::File;
use std::io::BufReader;
use std::io::Write;

use crate::parser;
use crate::render_sim;
use crate::sim;
use crate::utils;
use eyre::{eyre, OptionExt, Result, WrapErr};
use miniquad::*;

struct ReplState {
    mq_ctx: Box<dyn RenderingBackend>,
    puzzle_map: Option<utils::PuzzleMap>,
    puzzle: Option<parser::FullPuzzle>,
    solution: Option<parser::FullSolution>,
    world: Option<sim::WorldWithTapes>,
}

impl ReplState {
    fn new(mq_ctx: Box<dyn RenderingBackend>) -> Self {
        Self {
            mq_ctx,
            puzzle_map: None,
            puzzle: None,
            solution: None,
            world: None,
        }
    }

    fn get_or_load_puzzle_map(&mut self) -> &utils::PuzzleMap {
        self.puzzle_map.get_or_insert_with(|| {
            let mut puzzle_map = Default::default();
            utils::read_puzzle_recurse(&mut puzzle_map, utils::PUZZLE_DIR);
            puzzle_map
        })
    }

    fn load(&mut self, puzzle: parser::FullPuzzle, solution: parser::FullSolution) -> Result<()> {
        let init = parser::puzzle_prep(&puzzle, &solution)?;
        self.puzzle = Some(puzzle);
        self.solution = Some(solution);
        self.world = Some(sim::WorldWithTapes::setup_sim(&init)?);
        Ok(())
    }

    fn cmd_load<S: AsRef<str>>(&mut self, args: &[S]) -> Result<()> {
        match args {
            [] => {
                let (puzzle, solution) = utils::get_default_puzzle_solution()?;
                self.load(puzzle, solution)
            }
            [sol_path] => {
                let (puzzle, solution) =
                    utils::get_solution(sol_path.as_ref(), self.get_or_load_puzzle_map())?;
                let puzzle = puzzle
                    .ok_or_eyre(
                        "couldn't find puzzle for the given solution, please specify explicitly",
                    )?
                    .clone();
                self.load(puzzle.clone(), solution)
            }
            [puz_path, sol_path] => {
                let f_puz = File::open(puz_path.as_ref())?;
                let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puz))
                    .wrap_err(eyre!("Failed to parse puzzle {:?}", puz_path.as_ref()))?;
                let f_sol = File::open(sol_path.as_ref())?;
                let solution = parser::parse_solution(&mut BufReader::new(f_sol))
                    .wrap_err(eyre!("Failed to parse solution {:?}", sol_path.as_ref()))?;
                self.load(puzzle, solution)
            }
            _ => Err(eyre!("bad args to load")),
        }
    }

    fn cmd_render<S: AsRef<str>>(&mut self, args: &[S]) -> Result<()> {
        let world = &self.world.as_ref().ok_or_eyre("no world loaded")?.world;

        let mut width = 256u32;
        let mut height = 256u32;
        let mut output_filename = "output.png".to_string();

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
                "--size" => {
                    let size_str: Vec<_> = args
                        .next()
                        .ok_or_eyre("--output missing filename")?
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

        let camera = render_sim::CameraSetup::frame_center(world, screen_size);
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

    fn run(&mut self) -> Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::stdout();
        loop {
            let mut buf = String::new();
            write!(stdout, "> ")?;
            stdout.flush()?;
            if stdin.read_line(&mut buf)? == 0 {
                break Ok(());
            }

            let mut shlex = shlex::Shlex::new(&buf);
            if let Some(cmd) = shlex.next() {
                let args: Vec<_> = shlex.collect();
                let result = match cmd.as_ref() {
                    "load" => self.cmd_load(&args),
                    "render" => self.cmd_render(&args),
                    _ => Err(eyre!("unknown command {}", cmd)),
                };
                match result {
                    Ok(()) => (),
                    Err(e) => {
                        eprintln!("{:#}", e);
                    }
                }
            }
        }
    }
}

struct DummyHandler;

impl EventHandler for DummyHandler {
    fn update(&mut self) {}
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
            let mq_ctx = window::new_rendering_backend();
            let mut repl_state = ReplState::new(mq_ctx);

            repl_state.run().unwrap();

            window::order_quit();

            Box::new(DummyHandler)
        },
    );

    Ok(())
}
