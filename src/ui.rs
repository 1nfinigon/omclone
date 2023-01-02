use miniquad::*;
use crate::{sim::*, parser, render_sim::*};
use std::{fs::File, io::BufReader, io::BufWriter, io::prelude::*, path::Path};

use core::ops::Range;
struct TapeBuffer<'a>{
    tape_ref: &'a mut Tape,
    earliest_edit: &'a mut Option<usize>,
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
        *self.earliest_edit = Some(char_index);
        let tape = &mut self.tape_ref;
        let instr_mapped:Vec<Instr> = text.chars().filter_map(Instr::from_char).collect();
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

        self.held_str = tape.noop_clear_and_string();
        inserted
    }
    fn delete_char_range(&mut self, char_range: Range<usize>) {
        *self.earliest_edit = Some(char_range.start);
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
        self.held_str = tape.noop_clear_and_string();
    }

    fn clear(&mut self) {
        *self.earliest_edit = Some(0);
        self.tape_ref.instructions.clear();
        self.tape_ref.first = 0;
        self.held_str = self.tape_ref.noop_clear_and_string();
    }
    fn replace(&mut self, text: &str) {
        *self.earliest_edit = Some(0);
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
enum RunState{
    Manual(usize), FreeRun, Crashed(XYPos)
}
struct Loaded{
    base_world: World,
    curr_world: World,
    backup_world: World,
    curr_time: f64,
    curr_substep: usize,
    backup_step: u64,
    substep_count: usize,
    float_world: FloatWorld,
    saved_motions: WorldStepInfo,

    max_timestep: usize,
    camera: CameraSetup,
    tracks: TrackBindings,
    solution: parser::FullSolution,
    message: Option<String>,

    tape_mode: bool,
    run_state: RunState,
    show_area: bool,
    run_speed: f64,
    popup_reload: bool,
}
impl Loaded{
    fn reset_to(&mut self, reset_to:u64){
        if self.backup_step < reset_to{
            self.reset_to_backup();
        } else {
            self.reset_world();
        }
    }
    fn reset_world(&mut self) {
        self.saved_motions.clear();
        self.curr_world = self.base_world.clone();
        self.curr_time = 0.;
        self.substep_count = 8;
        self.curr_substep = 0;
        self.message = None;

        self.backup_world = self.base_world.clone();
        self.backup_step = 0;
    }
    fn reset_to_backup(&mut self){
        self.saved_motions.clear();
        self.curr_world = self.backup_world.clone();
        self.curr_time = self.backup_step as f64;
        self.substep_count = 8;
        self.curr_substep = 0;
        self.message = None;
    }
    fn try_set_target_time(&mut self, target: usize){
        if let RunState::Crashed(_) = self.run_state{
            if self.curr_world.timestep as usize > target{
                self.run_state = RunState::Manual(target);
            }
        } else {
            self.run_state = RunState::Manual(target);
        }
    }
}
enum AppState{
    NotLoaded,Loaded(Loaded) 
}
enum AppStateUpdate{
    NoChange, Load, ResetHome
}
use AppState::*;

pub struct MyMiniquadApp {
    egui_mq: egui_miniquad::EguiMq,
    render_data: RenderDataBase,
    app_state: AppState,
    unloaded_info: PathInfo
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
    pub fn load(&mut self, f_puzzle: impl Read, f_sol: impl Read, ctx: &mut Context){
        let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puzzle)).unwrap();
        let solution = parser::parse_solution(&mut BufReader::new(f_sol)).unwrap();
        println!("Check: {:?}", solution.stats);
        let init = parser::puzzle_prep(&puzzle, &solution).unwrap();
        for (a_num, a) in init.arms.iter().enumerate(){
            println!("Arms {:02}: {:?}", a_num, a.instruction_tape.to_string());
        }
        let world = World::setup_sim(&init).unwrap();
        let test_world = world.clone();

        let float_world = FloatWorld::new();
        let mut saved_motions = WorldStepInfo::new();
        saved_motions.clear();
        let camera = CameraSetup::frame_center(&test_world, ctx.screen_size());
        let tracks = setup_tracks(ctx, &test_world.track_map);
        let new_loaded = Loaded{
            base_world: world.clone(),
            backup_world: world.clone(),
            curr_world: world,
            curr_time: 0.,
            curr_substep: 0,
            backup_step: 0,
            substep_count: 8,
            float_world,saved_motions,
        
            max_timestep: 100,//test_world.timestep as usize,
            camera,tracks,solution,
            message: None,
        
            tape_mode: false,
            run_state: RunState::Manual(0),
            show_area: false,
            run_speed: 0.05,
            popup_reload: false,
        };
        self.app_state = Loaded(new_loaded);
    }
}


#[cfg(target_arch = "wasm32")]
static JS_PUZZLE_INPUT: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);
#[cfg(target_arch = "wasm32")]
static JS_SOLUTION_INPUT: std::sync::Mutex<Option<Vec<u8>>> = std::sync::Mutex::new(None);
#[cfg(target_arch = "wasm32")]
static JS_SOLUTION_NAME: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn js_load_puzzle(puzzle: sapp_jsutils::JsObject){
    let mut puzzle_vec = Vec::new();
    puzzle.to_byte_buffer(&mut puzzle_vec);
    *JS_PUZZLE_INPUT.lock().unwrap() = Some(puzzle_vec);
}
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn js_load_solution(solution: sapp_jsutils::JsObject){
    let mut solution_vec = Vec::new();
    solution.to_byte_buffer(&mut solution_vec);
    *JS_SOLUTION_INPUT.lock().unwrap() = Some(solution_vec);
}
#[cfg(target_arch = "wasm32")]
#[no_mangle]
pub extern "C" fn js_load_solution_name(name: sapp_jsutils::JsObject){
    let mut sol_name = String::new();
    name.to_string(&mut sol_name);
    *JS_SOLUTION_NAME.lock().unwrap() = Some(sol_name);
}

#[cfg(all(target_arch = "wasm32", feature="editor_ui"))]
extern "C" {
    fn save_file(filedata: sapp_jsutils::JsObject, filename: sapp_jsutils::JsObject);
}

const EDITOR_ENABLED: bool = cfg!(feature = "editor_ui");

impl EventHandler for MyMiniquadApp {
    fn update(&mut self, _: &mut Context) {
        if let Loaded(loaded) = &mut self.app_state{
            let target_time = match loaded.run_state{
                RunState::Manual(target_timestep) => {
                    let target_timestep = target_timestep as f64;
                    if target_timestep > loaded.curr_time+1.001{
                        target_timestep
                    } else {
                        f64::min(loaded.curr_time+loaded.run_speed,target_timestep)
                    }
                },
                RunState::FreeRun => loaded.curr_time+loaded.run_speed,
                RunState::Crashed(_) => return,
            };
            if target_time < loaded.curr_time{
                loaded.reset_to(target_time as u64);
            } else { //If backup is too out-of-date, catch up
                if loaded.curr_time as u64 >= loaded.backup_step+200 {
                    loaded.reset_to_backup()
                }
            }
            let mut now_time = (loaded.curr_world.timestep as f64)+(loaded.curr_substep as f64 / loaded.substep_count as f64);

            let advance = |now_time: &mut f64, substep_time: &mut f64, loaded: &mut Loaded| -> SimResult<()>{
                if loaded.curr_substep == 0{
                    let failcheck = loaded.curr_world.prepare_step(&mut loaded.saved_motions);
                    if failcheck.is_err(){
                        loaded.saved_motions.clear();
                        return failcheck;
                    }
                    loaded.substep_count = loaded.curr_world.substep_count(&loaded.saved_motions);
                    *substep_time = 1.0/(loaded.substep_count as f64);
                }
                loaded.curr_substep += 1;
                let portion = loaded.curr_substep as f32 / loaded.substep_count as f32;
                *now_time = (loaded.curr_world.timestep as f64)+(portion as f64);
                if loaded.show_area{
                    loaded.float_world.regenerate(&loaded.curr_world, &loaded.saved_motions, portion);
                    loaded.curr_world.mark_area_and_collide(&loaded.float_world, &loaded.saved_motions.spawning_atoms)?;
                }
                if loaded.curr_substep == loaded.substep_count{
                    loaded.curr_substep = 0;
                    loaded.curr_world.finalize_step(&mut loaded.saved_motions)?;
                }
                Ok(())
            };

            let mut substep_time = 1.0/(loaded.substep_count as f64);
            let backup_target_time = target_time.floor() - 100.;
            if now_time <= backup_target_time{
                while now_time <= backup_target_time{
                    let output = advance(&mut now_time, &mut substep_time, loaded);
                    if let Err(output) = output{
                        loaded.run_state = RunState::Crashed(output.location);
                        loaded.message = Some(output.to_string());
                        loaded.curr_time = now_time;
                        return;
                    }
                }
                loaded.backup_step = backup_target_time as u64;
                loaded.backup_world = loaded.curr_world.clone();
                assert_eq!(backup_target_time as u64, loaded.curr_world.timestep);
            }

            while now_time <= target_time-substep_time{
                let output = advance(&mut now_time, &mut substep_time, loaded);
                if let Err(output) = output{
                    loaded.run_state = RunState::Crashed(output.location);
                    loaded.message = Some(output.to_string());
                    loaded.curr_time = now_time;
                    return;
                }
            }
            loaded.curr_time = target_time;
            let display_portion = target_time.fract() as f32;
            if loaded.saved_motions.arms.is_empty(){
                loaded.float_world.generate_static(&loaded.curr_world);
            } else {
                loaded.float_world.regenerate(&loaded.curr_world, &loaded.saved_motions, display_portion);
            }
        }
    }

    fn draw(&mut self, ctx: &mut Context) {
        ctx.clear(Some((1., 1., 1., 1.)), None, None);
        ctx.begin_default_pass(PassAction::clear_color(0.5, 0.5, 0.5, 1.0));
        if let Loaded(loaded) = &mut self.app_state{
            let world = &loaded.curr_world;
            let float_world = &loaded.float_world;
            self.render_data.draw(ctx, &loaded.camera, &loaded.tracks, loaded.show_area, world, float_world)
        }
        ctx.end_render_pass();
        
        let mut do_loading = AppStateUpdate::NoChange;
        let screen_size = ctx.screen_size();
        self.egui_mq.run(ctx, |egui_ctx|{
            match &mut self.app_state{
                Loaded(loaded) => {
                    egui::Window::new("World loaded").default_pos((0.,0.)).show(egui_ctx, |ui| {
                        ui.style_mut().spacing.slider_width = 500.;
                        let mut target_time = loaded.curr_time.floor() as usize;
                        if loaded.max_timestep < target_time{
                            loaded.max_timestep = target_time;
                        }
                        ui.horizontal(|ui| {
                            ui.add(egui::Slider::new(&mut target_time, 0..=loaded.max_timestep)
                                .show_value(false));
                            ui.add(egui::DragValue::new(&mut target_time)
                                .clamp_range(0..=loaded.max_timestep)
                                .speed(0.1));
                            ui.label("Timestep");
                        });
                        ui.horizontal(|ui| {
                            if ui.button("-1").clicked() {
                                if target_time > 0 {
                                    target_time -= 1;
                                }
                            }
                            if ui.button("+1").clicked() {
                                target_time += 1;
                            }
                            ui.label("Max Timestep:");
                            ui.add(egui::DragValue::new(&mut loaded.max_timestep)
                                .speed(0.1));
                            ui.separator();
                            if loaded.max_timestep < target_time{
                                loaded.max_timestep = target_time;
                            }
                            if target_time != loaded.curr_time.floor() as usize{
                                loaded.try_set_target_time(target_time);
                            }

                            let min_size = loaded.base_world.arms.iter()
                                .fold(0, |val, arm| usize::max(val, arm.instruction_tape.instructions.len()));
                            ui.label("Loop length:");

                            let repeat_length_response = ui.add_enabled(EDITOR_ENABLED, egui::DragValue::new(&mut loaded.base_world.repeat_length)
                                .clamp_range(min_size..=usize::MAX));
                            if repeat_length_response.changed(){
                                let length = loaded.base_world.repeat_length;
                                loaded.backup_world.repeat_length = length;
                                loaded.curr_world.repeat_length = length;
                                if loaded.max_timestep < length{
                                    loaded.max_timestep = length;
                                }
                                if length <= loaded.curr_time as usize{
                                    loaded.reset_to(length as u64);
                                }
                            }
                            ui.separator();
                            ui.checkbox(&mut loaded.show_area, "Show Area");
                            ui.separator();
                            ui.label("Speed:");
                            ui.add(egui::DragValue::new(&mut loaded.run_speed)
                                .clamp_range(0.01..=10.0f32).fixed_decimals(2).speed(0.002));
                            let (mut running_free, enabled) = match loaded.run_state{
                                RunState::FreeRun => (true, true),
                                RunState::Manual(_) => (false, true),
                                RunState::Crashed(_) => (false, false),
                            };
                            let was_freerun = running_free;
                            ui.add_enabled_ui(enabled, |ui| ui.checkbox(&mut running_free, "Run"));
                            if running_free{
                                loaded.run_state = RunState::FreeRun;
                            } else if was_freerun{
                                loaded.try_set_target_time(loaded.curr_time.ceil() as usize);
                            }
                        });
                    });
                    if let Some(msg) = &loaded.message{
                        let mut opened = true;
                        egui::Window::new("Message").open(&mut opened).show(egui_ctx, |ui| {
                            ui.label(msg);
                        });
                        if !opened {loaded.message = None;}
                    }
                    egui::Window::new("Camera controls").show(egui_ctx, |ui| {
						ui.horizontal(|ui|{
							ui.label("x:");
							ui.add(egui::DragValue::new(&mut loaded.camera.offset[0])
								.speed(0.1));
							ui.label("y:");
							ui.add(egui::DragValue::new(&mut loaded.camera.offset[1])
								.speed(0.1));
                            if ui.button("Center").clicked(){
                                loaded.camera = CameraSetup::frame_center(&loaded.base_world, screen_size);
                            }
						});
						ui.horizontal(|ui|{
                            ui.label("zoom:");
                            ui.add(egui::Slider::new(&mut loaded.camera.scale_base,0.002 ..= 0.2)
                                .logarithmic(true));
                        });
                    });
					
                    #[cfg(feature = "editor_ui")]{
                        egui::Window::new("File info").show(egui_ctx, |ui| {
                            ui.label("solution:");
                            ui.text_edit_singleline(&mut loaded.solution.solution_name);
                            ui.label("filename:");
                            ui.text_edit_singleline(&mut self.unloaded_info.solution);
                            ui.horizontal(|ui| {
                                if ui.button("Save").clicked() {
                                    let dat = &self.unloaded_info;
                                    
                                    #[cfg(not(target_arch = "wasm32"))]
                                    let mut f = {
                                        let base_path = Path::new(&dat.base);
                                        let solution_path = base_path.join(&dat.solution);
                                        File::create(solution_path).unwrap()
                                    };
                                    #[cfg(target_arch = "wasm32")]
                                    let mut f = {
                                        Vec::<u8>::new()
                                    };

                                    {//scope the writer so it's dropped before trying to save in js
                                        if !loaded.solution.solution_name.contains("omclone"){
                                            loaded.solution.solution_name += "(omclone)";
                                        }
                                        let mut writer = BufWriter::new(&mut f);
                                        let base = &loaded.base_world;
                                        let new_solution = parser::create_solution(base, 
                                            loaded.solution.puzzle_name.clone(),
                                            loaded.solution.solution_name.clone());
                                        parser::write_solution(&mut writer, &new_solution).unwrap();
                                        writer.flush().unwrap();
                                    }

                                    #[cfg(target_arch = "wasm32")]{
                                        let data = f;
                                        let filedata = sapp_jsutils::JsObject::buffer(&data);
                                        let filename = sapp_jsutils::JsObject::string(&dat.solution);
										unsafe {save_file(filedata,filename);}
                                    }

                                    loaded.message = Some(String::from("Saved!"));
                                }
                                ui.separator();
                                if ui.button("Load").clicked() {
                                    loaded.popup_reload = true;
                                }
                            });
                        });
                    }
                    #[cfg(feature = "editor_ui")]{
                        egui::Window::new("Arms").hscroll(true).show(egui_ctx, |ui| {
                            ui.checkbox(&mut loaded.tape_mode, "Tape/overwrite mode");
                            let curr_timestep = loaded.curr_time.floor() as usize;
                            let end_spacing = loaded.max_timestep.saturating_sub(curr_timestep);
                            let marker = " ".repeat(curr_timestep+3)+"V"+&" ".repeat(end_spacing);
                            //+&(" ".repeat(loaded.max_timestep-loaded.curr_timestep));
                            ui.add(egui::Label::new(egui::RichText::new(marker).monospace()).wrap(false));
                            let mut earliest_edit = None;
                            let mut target_time = None;
                            let mut change_arm_id = None;
                            egui::ScrollArea::vertical().show(ui, |ui| {
                                for (a_num, a) in loaded.base_world.arms.iter_mut().enumerate(){
                                    let text = a.instruction_tape.to_string();
                                    let mut text_buf = TapeBuffer{
                                        tape_ref: &mut a.instruction_tape,
                                        earliest_edit: &mut earliest_edit,
                                        tape_mode: loaded.tape_mode,
                                        held_str: text
                                    };
                                    ui.horizontal(|ui| {
                                        ui.label(format!("{:02}",a_num+1));
                                        let text_output = egui::TextEdit::singleline(&mut text_buf)
                                            .code_editor().desired_width(f32::INFINITY)
                                            .show(ui);
                                        if let Some(cursor) = text_output.cursor_range{
                                            target_time = Some(cursor.primary.ccursor.index);
                                        }
                                        if text_output.response.changed(){
                                            change_arm_id = Some(a_num);
                                        }
                                    });
                                }
                            });
                            if let Some(target) = target_time {
                                loaded.try_set_target_time(target);
                            }
                            if let Some(edit_step) = earliest_edit{
                                let arm_id = change_arm_id.unwrap();//If edit is performed, the change must be recorded
                                let min_size = loaded.base_world.arms.iter()
                                    .fold(0, |val, arm| usize::max(val, arm.instruction_tape.instructions.len()));
                                if loaded.base_world.repeat_length < min_size{
                                    loaded.base_world.repeat_length = min_size;
                                    loaded.backup_world.repeat_length = min_size;
                                    loaded.curr_world.repeat_length = min_size;
                                }
                                loaded.backup_world.arms[arm_id].instruction_tape = loaded.base_world.arms[arm_id].instruction_tape.clone();
                                if edit_step <= loaded.curr_time as usize {
                                    if let RunState::Crashed(_) = loaded.run_state{
                                        loaded.run_state = RunState::Manual(loaded.curr_world.timestep as usize);
                                    }
                                    loaded.reset_to(edit_step as u64);
                                } else {
                                    loaded.curr_world.arms[arm_id].instruction_tape = loaded.base_world.arms[arm_id].instruction_tape.clone();
                                }
                            }
                        });
                    }
                    /*egui::Window::new("debug").show(egui_ctx, |ui|{
                        ui.label(format!("curr time: {}",loaded.curr_time as u64));
                        ui.label(format!("backup time: {}",loaded.backup_step));
                    });*/
                    egui::Window::new("Reload?").open(&mut loaded.popup_reload).show(egui_ctx, |ui| {
                        #[cfg(not(target_arch = "wasm32"))]{
                                if ui.button("Reload current").clicked() {
                                    do_loading = AppStateUpdate::Load;
                                }
                                if ui.button("Load New").clicked() {
                                    do_loading = AppStateUpdate::ResetHome;
                                }
                        }
                        #[cfg(target_arch = "wasm32")]{
                            ui.label("Choose files first, then press:");
                            if ui.button("Reset").clicked() {
                                do_loading = AppStateUpdate::ResetHome;
                            }

                        }
                    });
                }
                NotLoaded => {
                    egui::Window::new("World Not Loaded").show(egui_ctx, |ui| {
                        #[cfg(not(target_arch = "wasm32"))]{
                            let dat = &mut self.unloaded_info;
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
                                do_loading = AppStateUpdate::Load;
                            }
                        }
                        #[cfg(target_arch = "wasm32")]
                        {
                            ui.label("Upload the puzzle and the solution");
                            if JS_PUZZLE_INPUT.lock().unwrap().is_some() && JS_SOLUTION_INPUT.lock().unwrap().is_some(){
                                do_loading = AppStateUpdate::Load;
                            }
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
            AppStateUpdate::Load => {
                #[cfg(not(target_arch = "wasm32"))]{
                    let dat = &self.unloaded_info;
                    let base_path = Path::new(&dat.base);
                    let f_puzzle = File::open(base_path.join(&dat.puzzle)).unwrap();
                    let solution_path = base_path.join(&dat.solution);
                    let f_sol = File::open(&solution_path).unwrap();
                    self.load(f_puzzle, f_sol, ctx);
                }
                #[cfg(target_arch = "wasm32")]{
                    let puzzle = JS_PUZZLE_INPUT.lock().unwrap();
                    let f_puzzle = puzzle.as_ref().unwrap();
                    let solution = JS_SOLUTION_INPUT.lock().unwrap();
                    let f_sol = solution.as_ref().unwrap();
                    self.load(f_puzzle.as_slice(), f_sol.as_slice(), ctx);
					let name = JS_SOLUTION_NAME.lock().unwrap().take();
					self.unloaded_info.solution = name.unwrap_or(String::from("omclone.solution"));
                }
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