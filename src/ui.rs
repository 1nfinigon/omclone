use miniquad::*;
use crate::{sim::*, parser, render_sim::*};
use std::{fs::File, io::BufReader, io::BufWriter, io::prelude::*, path::Path};
/*#[cfg(feature = "color_eyre")]
use color_eyre::{
    eyre::{bail, ensure, eyre},
    Result,
};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{
    eyre::{bail, ensure, eyre},
    Result,
};*/

use core::ops::Range;
struct TapeBuffer<'a>{
    tape_ref: &'a mut Tape,
    force_reload: &'a mut bool,
    tape_mode: bool,
    held_str: String
}
impl AsRef<str> for TapeBuffer<'_>{
    fn as_ref(&self) -> &str {
        self.held_str.as_str()
    }
}
//Note: this implementation assumes all characters in the string are ASCII/1 byte long
//It should be fine with non-ASCII attempted inserts, which will be removed as invalid instructions
impl egui::widgets::text_edit::TextBuffer for TapeBuffer<'_>{
    fn is_mutable(&self) -> bool {
        true
    }
    fn insert_text(&mut self, text: &str, char_index: usize) -> usize {
        let tape = &mut self.tape_ref;
        let instr_mapped:Vec<Instr> = text.chars().filter_map(|x| Instr::from_char(x)).collect();
        let inserted = instr_mapped.len();

        let empty_extension = if char_index < tape.first{
            let tmp = tape.first-char_index;
            tape.first = char_index;
            tmp
        } else {
            0
        };
        let instr_chain = instr_mapped.into_iter().chain(std::iter::repeat(Instr::Empty).take(empty_extension));
        let splice_index = char_index-tape.first;
        let splice_range = if self.tape_mode {
            splice_index..usize::min(splice_index+text.len(),tape.instructions.len())
        } else {
            splice_index..splice_index
        };
        tape.instructions.splice(splice_range,instr_chain);

        self.held_str = tape.modify_and_string();
        *self.force_reload = true;
        inserted
    }
    fn delete_char_range(&mut self, char_range: Range<usize>) {
        let tape = &mut self.tape_ref;
        if self.tape_mode{
            for x in char_range{
                if x >= tape.first{
                    tape.instructions[x-tape.first] = Instr::Empty;
                }
            }
        } else {
            if char_range.end <= tape.first{
                tape.first -= char_range.end.saturating_sub(char_range.start);
            } else {
                let start = if char_range.start < tape.first{
                    tape.first = char_range.start;
                    0
                } else {
                    char_range.start-tape.first
                };
                let end = (char_range.end-tape.first).min(tape.instructions.len());
                tape.instructions.drain(start..end);
            }
        }
        self.held_str = tape.modify_and_string();
        *self.force_reload = true;
    }

    fn clear(&mut self) {
        self.tape_ref.instructions.clear();
        self.tape_ref.first = 0;
        self.held_str = self.tape_ref.modify_and_string();
        *self.force_reload = true;
    }
    fn replace(&mut self, text: &str) {
        self.tape_ref.instructions.clear();
        self.tape_ref.first = 0;
        self.insert_text(text, 0);
    }
    fn take(&mut self) -> String {
        let s = self.as_ref().to_owned();
        self.clear();
        s
    }
}

struct PathInfo{
    base: String,
    puzzle: String,
    solution: String,
}
struct Loaded{
    base_world: World,
    curr_timestep: usize,
    max_timestep: usize,
    last_world: World,
    last_timestep: usize,
    camera: CameraSetup,
    tape_mode: bool,
    tracks: TrackBindings,
    solution: parser::FullSolution,
    message: Option<String>,
    error_loc: Option<XYPos>,
    partial_timestep: f32,
    running_free: bool,
    show_area: bool,
    run_speed: f32,
    popup_reload: bool,
}
enum AppState{
    NotLoaded,Loaded(Loaded) 
}
enum AppStateUpdate{
    NoChange, LoadFromData, ResetHome
}
use AppState::*;

pub struct MyMiniquadApp {
    egui_mq: egui_miniquad::EguiMq,
    render_data: RenderDataBase,
    app_state: AppState,
    unloaded_info: PathInfo
}

pub fn get_default_path_strs() -> (&'static str, &'static str, &'static str){
    const DEFAULT_PATHS: &str = include_str!("default_paths.txt");
    let mut path_data = DEFAULT_PATHS.lines();
    let base = path_data.next().unwrap();
    let puzzle = path_data.next().unwrap();
    let solution = path_data.next().unwrap();
    (base, puzzle, solution)
}
impl MyMiniquadApp {
    pub fn new(ctx: &mut Context) -> Self {
        
        let render_data = RenderDataBase::new(ctx);
        let (base_str, puzzle_str, solution_str) = get_default_path_strs();
        let base = String::from(base_str);
        let puzzle = String::from(puzzle_str);
        let solution = String::from(solution_str);
        let app_state = AppState::NotLoaded;
        let unloaded_info = PathInfo{base, puzzle, solution};
        Self {
            egui_mq: egui_miniquad::EguiMq::new(ctx),
            render_data,app_state,unloaded_info
        }
    }
}

impl EventHandler for MyMiniquadApp {
    fn update(&mut self, _: &mut Context) {
        if let Loaded(loaded) = &mut self.app_state{
            if loaded.running_free && loaded.last_timestep == loaded.curr_timestep{
                loaded.curr_timestep += 1;
                if loaded.max_timestep < loaded.curr_timestep{
                    loaded.max_timestep = loaded.curr_timestep;
                }
            }
            if loaded.last_timestep > loaded.curr_timestep{
                loaded.last_timestep = 0;
                loaded.last_world = loaded.base_world.clone();
                loaded.partial_timestep = 0.;
                loaded.running_free = false;
            }
            let mut target_step = loaded.curr_timestep;
            if loaded.last_timestep+1 == loaded.curr_timestep{
                loaded.partial_timestep += loaded.run_speed;
                target_step -= 1;
            } else {
                loaded.partial_timestep = 0.;
                loaded.running_free = false;
            }
            if loaded.partial_timestep >= 1.{
                target_step += 1;
                loaded.partial_timestep = 0.;
            }
            if loaded.last_timestep < target_step{
                loaded.error_loc = None;
                for time in loaded.last_timestep..target_step{
                    let output = loaded.last_world.run_step(loaded.show_area);
                    if let Err(output) = output{
                        loaded.message = Some(output.to_string());
                        loaded.error_loc = Some(output.location);
                        loaded.last_world = loaded.base_world.clone();
                        for _ in 0..time{
                            loaded.last_world.run_step(loaded.show_area).unwrap();
                        }
                        loaded.curr_timestep = time;
                        //loaded.max_timestep = time;
                        loaded.running_free = false;
                        break;
                    }
                }
                loaded.last_timestep = loaded.curr_timestep;
            }
        }
    }

    fn draw(&mut self, ctx: &mut Context) {
        ctx.clear(Some((1., 1., 1., 1.)), None, None);
        ctx.begin_default_pass(PassAction::clear_color(0.5, 0.5, 0.5, 1.0));
        if let Loaded(loaded) = &mut self.app_state{
            let world = &loaded.last_world;
            let float_world_try = world.partial_step(loaded.partial_timestep);
            let float_world = if float_world_try.is_ok(){
                float_world_try.unwrap()
            } else {
                world.partial_step(0.).unwrap()
            };
            self.render_data.draw(ctx, &loaded.camera, &loaded.tracks, loaded.show_area, &world, &float_world)
        }
        ctx.end_render_pass();
        
        let mut do_loading = AppStateUpdate::NoChange;
        self.egui_mq.run(ctx, |egui_ctx|{
            match &mut self.app_state{
                Loaded(loaded) => {
                    if let Some(msg) = &loaded.message{
                        let mut opened = true;
                        egui::Window::new("Message").open(&mut opened).show(egui_ctx, |ui| {
                            ui.label(msg);
                        });
                        if !opened {loaded.message = None;}
                    }
                    egui::Window::new("World loaded").show(egui_ctx, |ui| {
                        ui.style_mut().spacing.slider_width = 500.;
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut loaded.curr_timestep, 0..=loaded.max_timestep)
                                .show_value(false));
                            ui.add(egui::DragValue::new(&mut loaded.curr_timestep)
                                .clamp_range(0..=loaded.max_timestep)
                                .speed(0.1));
                            ui.label("Timestep");
                        });
                        ui.horizontal(|ui| {
                            if ui.button("-1").clicked() {
                                if loaded.curr_timestep > 0 {
                                    loaded.curr_timestep -= 1;
                                    loaded.running_free = false;
                                    loaded.partial_timestep = 0.;
                                }
                            }
                            if ui.button("+1").clicked() {
                                if loaded.curr_timestep < loaded.max_timestep {
                                    loaded.curr_timestep += 1;
                                    loaded.running_free = false;
                                    loaded.partial_timestep = 0.;
                                }
                            }
                            ui.label("Max Timestep:");
                            ui.add(egui::DragValue::new(&mut loaded.max_timestep)
                                .speed(0.1));
                            ui.separator();
                            if loaded.max_timestep < loaded.curr_timestep{
                                loaded.max_timestep = loaded.curr_timestep;
                            }

                            let min_size = loaded.base_world.arms.iter()
                                .fold(0, |val, arm| usize::max(val, arm.instruction_tape.instructions.len()));
                            ui.label("Loop length:");
                            ui.add(egui::DragValue::new(&mut loaded.base_world.repeat_length)
                                .clamp_range(min_size..=usize::MAX));
                            ui.separator();
                            ui.checkbox(&mut loaded.show_area, "Show Area");
                            ui.separator();
                            ui.label("Speed:");
                            ui.add(egui::DragValue::new(&mut loaded.run_speed)
                                .clamp_range(0.01..=1.0).fixed_decimals(2).speed(0.002));
                                ui.checkbox(&mut loaded.running_free, "Run");
                        });
                        ui.horizontal(|ui| {
                            if ui.button("Save").clicked() {
                                let dat = &self.unloaded_info;
                                let base_path = Path::new(&dat.base);
                                let solution_path = base_path.join(&dat.solution);
                                let f = File::create(solution_path).unwrap();
                                let mut writer = BufWriter::new(f);
                                let base = &loaded.base_world;
                                let tape_list = base.arms.iter().map(|a| &a.instruction_tape);
                                parser::replace_tapes(&mut loaded.solution, tape_list, base.repeat_length).unwrap();
                                if !loaded.solution.solution_name.contains("omclone"){
                                    loaded.solution.solution_name += "(omclone)";
                                }
                                parser::write_solution(&mut writer, &loaded.solution).unwrap();
                                writer.flush().unwrap();
                                loaded.message = Some(String::from("Saved!"));
                            }
                            ui.separator();
                            if ui.button("Load").clicked() {
                                loaded.popup_reload = true;
                            }
                        });
                    });
                    egui::Window::new("Arms").hscroll(true).show(egui_ctx, |ui| {
                        ui.checkbox(&mut loaded.tape_mode, "Tape/overwrite mode");
                        let end_spacing = loaded.max_timestep.saturating_sub(loaded.curr_timestep);
                        let marker = " ".repeat(loaded.curr_timestep+3)+"V"+&" ".repeat(end_spacing);
                        //+&(" ".repeat(loaded.max_timestep-loaded.curr_timestep));
                        ui.add(egui::Label::new(egui::RichText::new(marker).monospace()).wrap(false));

                        let mut force_reload = false;
                        for (a_num, a) in loaded.base_world.arms.iter_mut().enumerate(){
                            let text = a.instruction_tape.to_string();
                            let mut text_buf = TapeBuffer{
                                tape_ref: &mut a.instruction_tape,
                                force_reload: &mut force_reload,
                                tape_mode: loaded.tape_mode,
                                held_str: text
                            };
                            ui.horizontal(|ui| {
                                ui.label(format!("{:02}",a_num));
                                let text_output = egui::TextEdit::singleline(&mut text_buf)
                                    .code_editor().desired_width(f32::INFINITY)
                                    .show(ui);
                                if let Some(cursor) = text_output.cursor_range{
                                    loaded.curr_timestep = cursor.primary.ccursor.index;
                                }
                            });
                        }
                        if loaded.max_timestep < loaded.curr_timestep{
                            loaded.max_timestep = loaded.curr_timestep;
                        }
                        if force_reload {
                            loaded.last_timestep = usize::MAX;
                            loaded.partial_timestep = 0.;
                            loaded.running_free = false;
                            let min_size = loaded.base_world.arms.iter()
                                .fold(0, |val, arm| usize::max(val, arm.instruction_tape.instructions.len()));
                            if loaded.base_world.repeat_length < min_size{
                                loaded.base_world.repeat_length = min_size;
                            }
                            loaded.message = None;
                        };
                    });
                    egui::Window::new("Reload?").open(&mut loaded.popup_reload).show(egui_ctx, |ui| {
                        if ui.button("Reload current").clicked() {
                            do_loading = AppStateUpdate::LoadFromData;
                        }
                        if ui.button("Load New").clicked() {
                            do_loading = AppStateUpdate::ResetHome;
                        }
                    });
                }
                NotLoaded => {
                    let dat = &mut self.unloaded_info;
                    egui::Window::new("World Not Loaded").show(egui_ctx, |ui| {
                        ui.horizontal(|ui| {
                            ui.label("Base path: ");
                            ui.text_edit_singleline(&mut dat.base);
                        });
                        ui.horizontal(|ui| {
                            ui.label("Puzzle: ");
                            ui.text_edit_singleline(&mut dat.puzzle);
                        });
                        ui.horizontal(|ui| {
                            ui.label("Solution: ");
                            ui.text_edit_singleline(&mut dat.solution);
                        });
                        if ui.button("Load").clicked() {
                            do_loading = AppStateUpdate::LoadFromData;
                        }
                    });
                }
            };
        });
        self.egui_mq.draw(ctx);
        match do_loading{
            AppStateUpdate::NoChange => {},
            AppStateUpdate::ResetHome => {
                self.app_state = NotLoaded;
            }
            AppStateUpdate::LoadFromData => {
                let dat = &self.unloaded_info;
                let base_path = Path::new(&dat.base);
                let f_puzzle = File::open(base_path.join(&dat.puzzle)).unwrap();
                let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puzzle)).unwrap();
                let solution_path = base_path.join(&dat.solution);
                let f_sol = File::open(&solution_path).unwrap();
                let solution = parser::parse_solution(&mut BufReader::new(f_sol)).unwrap();
                println!("Check: {:?}", solution.stats);
                let init = parser::puzzle_prep(&puzzle, &solution).unwrap();
                for (a_num, a) in init.arms.iter().enumerate(){
                    println!("Arms {:02}: {:?}", a_num, a.instruction_tape.to_string());
                }
                let world = World::setup_sim(&init).unwrap();
                let mut test_world = world.clone();

                while !test_world.is_complete() && test_world.timestep < 10000{
                    let res = test_world.run_step(false);
                    if let Err(e) = res{
                        println!("test world error: {:?}", e);
                        break;
                    }
                    if test_world.timestep % 100 == 0{
                        println!("Initial sim step {:03}", test_world.timestep);
                    }
                }
                let camera = CameraSetup::frame_center(&test_world);
                let tracks = setup_tracks(ctx, &test_world.track_map);
                let new_loaded = Loaded{
                    base_world: world.clone(),
                    curr_timestep: 0,
                    max_timestep: test_world.timestep as usize,
                    last_world: world,
                    last_timestep: 0,
                    tape_mode: false,
                    camera, tracks, solution,
                    message: None,
                    error_loc: None,
                    partial_timestep: 0.,
                    running_free: false,
                    show_area: true,
                    run_speed: 0.05,
                    popup_reload: false,
                };
                self.app_state = Loaded(new_loaded);
            }
        }

        // Draw things in front of egui here

        ctx.commit_frame();
    }

    fn mouse_motion_event(&mut self, ctx: &mut Context, x: f32, y: f32) {
        self.egui_mq.mouse_motion_event(ctx, x, y);
    }

    fn mouse_wheel_event(&mut self, ctx: &mut Context, dx: f32, dy: f32) {
        self.egui_mq.mouse_wheel_event(ctx, dx, dy);
    }

    fn mouse_button_down_event(
        &mut self,
        ctx: &mut Context,
        mb: MouseButton,
        x: f32,
        y: f32,
    ) {
        self.egui_mq.mouse_button_down_event(ctx, mb, x, y);
    }

    fn mouse_button_up_event(
        &mut self,
        ctx: &mut Context,
        mb: MouseButton,
        x: f32,
        y: f32,
    ) {
        self.egui_mq.mouse_button_up_event(ctx, mb, x, y);
    }

    fn char_event(
        &mut self,
        _ctx: &mut Context,
        character: char,
        _keymods: KeyMods,
        _repeat: bool,
    ) {
        self.egui_mq.char_event(character);
    }

    fn key_down_event(
        &mut self,
        ctx: &mut Context,
        keycode: KeyCode,
        keymods: KeyMods,
        _repeat: bool,
    ) {
        self.egui_mq.key_down_event(ctx, keycode, keymods);
    }

    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, keymods: KeyMods) {
        self.egui_mq.key_up_event(keycode, keymods);
    }
}