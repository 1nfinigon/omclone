use miniquad::*;
use crate::{sim::*, parser};
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
type GFXPos = [f32;2];
use std::f32::consts::PI;
fn pos_to_xy(input: &Pos) -> GFXPos{
    let a = input.x as f32;
    let b = input.y as f32;
    [a*2.+b,b*f32::sqrt(3.)]
}
fn rot_to_angle(r: Rot) -> f32{
    (-r as f32)*PI/3.
}

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
                tape.first -= char_range.end - char_range.start;
            } else {
                let start = if char_range.start < tape.first{
                    tape.first = char_range.start;
                    0
                } else {
                    char_range.start-tape.first
                };
                let end = char_range.end-tape.first;
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

type Vert = [f32;2];
//Vertex format: (x, y)
//note: 1 hex has inner radius of 1 (width of 2).
fn setup_bonds(ctx: &mut Context) -> Bindings{
    const BOND_VERT_BUF: [Vert;4] = [
        [ 0.,-0.1,],
        [ 0., 0.1,],
        [ 2., -0.1,],
        [ 2., 0.1,]];
    const BOND_INDEX_BUF: [u16;6] = [
        0, 1, 2,
        0, 2, 3,];
    let vb = Buffer::immutable(ctx, BufferType::VertexBuffer, &BOND_VERT_BUF);
    let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &BOND_INDEX_BUF);
    Bindings {
        vertex_buffers: vec![vb],
        index_buffer: index_buffer,
        images: vec![],
    }
}

fn setup_arms(ctx: &mut Context) -> Bindings{
    const ARM_VERT_BUF: [Vert;14] = [
        //Arm Base
        [ 0.,-0.4,],
        [ 0., 0.4,],
        //Arm edge
        [ 2., 0.,],
        [ 4., 0.,],
        [ 6., 0.,],
        //Grab markers
        [ 1.8, 0.,],
        [ 2.2, -0.4,],
        [ 2.2, 0.4,],
        [ 3.8, 0.,],
        [ 4.2, -0.4,],
        [ 4.2, 0.4,],
        [ 5.8, 0.,],
        [ 6.2, -0.4,],
        [ 6.2, 0.4,],
        ];
    //First triangle is arm, 2nd triangle is optional grab marker
    const ARM_INDEX_BUF: [u16;18] = [
        0, 1, 2,    5, 6, 7,
        0, 1, 3,    8, 9, 10,
        0, 1, 4,    11, 12, 13,];
    let vb = Buffer::immutable(ctx, BufferType::VertexBuffer, &ARM_VERT_BUF);
    let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &ARM_INDEX_BUF);
    Bindings {
        vertex_buffers: vec![vb],
        index_buffer,
        images: vec![],
    }
}

fn setup_tracks(ctx: &mut Context, track: &TrackMap) -> (Bindings, usize){
    let mut verts_vec:Vec<GFXPos> = Vec::new();
    let mut index_vec:Vec<u16> = Vec::new();
    let mut curr_index = 0;
    for (center_pos, track_data) in track{
        //Every minus is matched by a positive so only need one
        if let Some(plus) = &track_data.plus{
            verts_vec.push(pos_to_xy(center_pos));
            index_vec.push(curr_index);
            curr_index += 1;
            verts_vec.push(pos_to_xy(&(center_pos+plus)));
            index_vec.push(curr_index);
            curr_index += 1;
        }
    }

    let vb = Buffer::immutable(ctx, BufferType::VertexBuffer, &verts_vec);
    let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &index_vec);
    (Bindings {
        vertex_buffers: vec![vb],
        index_buffer,
        images: vec![],
    }, index_vec.len())
}

type UvVert = [f32;4];
//(x, y), (u, v)
const GLYPH_COUNT:usize = 15;
fn setup_glyphs(ctx: &mut Context) -> [Bindings;GLYPH_COUNT]{
    const GLYPH_VERT_BUF: [UvVert;4] = [
        [-3.,-3.,    0., 1.],
        [-3., 3.,    0., 0.],
        [ 3.,-3.,    1., 1.],
        [ 3., 3.,    1., 0.]];
    const GLYPH_INDEX_BUF: [u16;6] = [
        0, 1, 2,
        1, 2, 3];
    let vb = Buffer::immutable(ctx, BufferType::VertexBuffer, &GLYPH_VERT_BUF);
    let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &GLYPH_INDEX_BUF);
    let glyph_list:[&[u8];GLYPH_COUNT] = [
        include_bytes!("../images/Ani.png"),
        include_bytes!("../images/Bonder.png"),
        include_bytes!("../images/Calcification.png"),
        include_bytes!("../images/Dispersion.png"),
        include_bytes!("../images/Disposal.png"),
        include_bytes!("../images/Duplication.png"),
        include_bytes!("../images/Equilibrium.png"),
        include_bytes!("../images/Multibond.png"),
        include_bytes!("../images/Projection.png"),
        include_bytes!("../images/Purification.png"),
        include_bytes!("../images/Triplex.png"),
        include_bytes!("../images/Unbonder.png"),
        include_bytes!("../images/Unification.png"),
        include_bytes!("../images/HexGrid.png"),
        include_bytes!("../images/ShadeAtomInOut.png"),
    ];
    glyph_list.map(|byte_data| -> Bindings{
        use image::io::Reader as ImageReader;
        use image::ImageFormat::Png;
        use std::io::Cursor;
        let img = ImageReader::with_format(Cursor::new(byte_data),Png).decode().unwrap().into_rgba8();
        let texture = Texture::from_rgba8(ctx, 256, 256, &img);
        Bindings {
            vertex_buffers: vec![vb],
            index_buffer,
            images:vec![texture],
        }
    })
}
const CIRCLE_VERT_COUNT:usize = 20;
fn setup_circle(ctx: &mut Context) -> Bindings{
    let mut verts = [[0.;2];CIRCLE_VERT_COUNT+1];
    let angle_per = PI*2./(CIRCLE_VERT_COUNT as f32);
    //verts[CIRCLE_VERT_COUNT] = [0.,0.];
    let mut indices: [u16; CIRCLE_VERT_COUNT*3] = [CIRCLE_VERT_COUNT as u16;CIRCLE_VERT_COUNT*3];
    for i in 0..CIRCLE_VERT_COUNT{
        let angle = (i as f32)*angle_per;
        verts[i] = [angle.cos()*0.8,angle.sin()*0.8];
        indices[i*3+1] = (i) as u16;
        indices[i*3+2] = ((i+1)%CIRCLE_VERT_COUNT) as u16;
    }
    let vertex_buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &verts);
    let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &indices);
    Bindings {
        vertex_buffers: vec![vertex_buffer],
        index_buffer,
        images: vec![],
    }
}
struct ShapeStore{
    arm_bindings: Bindings,
    circle_bindings: Bindings,
    bond_bindings: Bindings,
    glyph_bindings: [Bindings;GLYPH_COUNT],
}
struct NotLoaded{
    base: String,
    puzzle: String,
    solution: String,
}
struct CameraSetup{
    scale: f32,
    offset: GFXPos,
}
impl CameraSetup{
    fn process(world: &World) -> Self{
        let mut pos_list = world.glyphs.iter().map(|x| pos_to_xy(&x.pos));
        let (mut lowx, mut lowy, mut highx, mut highy) = pos_list.try_fold(
            (f32::INFINITY,f32::INFINITY,f32::NEG_INFINITY,f32::NEG_INFINITY),
            |(lowx, lowy, highx, highy), [thisx, thisy]| {
                let new_lowx = if thisx < lowx {thisx} else {lowx};
                let new_highx = if thisx > highx {thisx} else {highx};
                let new_lowy = if thisy < lowy {thisy} else {lowy};
                let new_highy = if thisy > highy {thisy} else {highy};
                Some((new_lowx, new_lowy, new_highx, new_highy))
            }
        ).unwrap();
        const BORDER: f32 = 5.;
        lowx -= BORDER;
        lowy -= BORDER;
        highx+= BORDER;
        highy+= BORDER;
        let offset = [-(lowx+highx)/2., -(lowy+highy)/2.];
        let scale_x = 1./(highx-lowx);
        let scale_y = 1./(highy-lowy);
        let scale = if scale_x > scale_y {scale_x} else {scale_y};
        println!("camera: x{}, +{:?}",scale,offset);
        CameraSetup {scale, offset}
    }
}
struct Loaded{
    base_world: World,
    curr_timestep: usize,
    max_timestep: usize,
    last_world: World,
    last_timestep: usize,
    camera: CameraSetup,
    tape_mode: bool,
    track_binds: (Bindings, usize),
    solution: parser::FullSolution,
    message: Option<String>,
}
enum AppState{
    NotLoaded(NotLoaded),Loaded(Loaded) 
}
use AppState::*;

pub struct MyMiniquadApp {
    egui_mq: egui_miniquad::EguiMq,
    pipeline: Pipeline,
    pipeline_glyphs: Pipeline,
    pipeline_tracks: Pipeline,
    shapes: ShapeStore,
    app_state: AppState,
}

#[repr(C)]
struct BasicUniforms{
    color: [f32;3],
    offset: GFXPos,
    world_offset: GFXPos,
    angle: f32,
    scale: f32,
}
#[repr(C)]
struct UvUniforms{
    offset: GFXPos,
    world_offset: GFXPos,
    angle: f32,
    scale: f32,
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
        let shader_meta = ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("color", UniformType::Float3),
                    UniformDesc::new("offset", UniformType::Float2),
                    UniformDesc::new("world_offset", UniformType::Float2),
                    UniformDesc::new("angle", UniformType::Float1),
                    UniformDesc::new("scale", UniformType::Float1),
                    ],
            },
        };
        const V_SHADE: &str = include_str!("basic_vert.vs");
        const F_SHADE: &str = include_str!("basic_frag.fs");
        let shader = Shader::new(ctx, V_SHADE, F_SHADE, shader_meta).unwrap();

        let pipeline = Pipeline::new(
            ctx,
            &[BufferLayout::default()],
            &[VertexAttribute::new("local_pos", VertexFormat::Float2)],
            shader,
        );
        
        let shader_meta_uv = ShaderMeta {
            images: vec!["tex".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("offset", UniformType::Float2),
                    UniformDesc::new("world_offset", UniformType::Float2),
                    UniformDesc::new("angle", UniformType::Float1),
                    UniformDesc::new("scale", UniformType::Float1),
                    ],
            },
        };
        const V_UV_SHADE: &str = include_str!("uv_vert.vs");
        const F_UV_SHADE: &str = include_str!("uv_frag.fs");
        let shader_uv = Shader::new(ctx, V_UV_SHADE, F_UV_SHADE, shader_meta_uv).unwrap();
        use miniquad::graphics::*;
        let pipeline_glyphs = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("local_pos", VertexFormat::Float2),
                VertexAttribute::new("uv", VertexFormat::Float2),
            ],
            shader_uv,
            PipelineParams{
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha))
                ),
                ..Default::default()
            }
        );
        let pipeline_tracks = Pipeline::with_params(
            ctx,
            &[BufferLayout::default()],
            &[VertexAttribute::new("local_pos", VertexFormat::Float2)],
            shader,
            PipelineParams{
                primitive_type:PrimitiveType::Lines,
                ..Default::default()
            }
        );
        let shapes = ShapeStore{
            arm_bindings: setup_arms(ctx),
            circle_bindings: setup_circle(ctx),
            bond_bindings: setup_bonds(ctx),
            glyph_bindings: setup_glyphs(ctx),
        };
        
        let (base_str, puzzle_str, solution_str) = get_default_path_strs();
        let base = String::from(base_str);
        let puzzle = String::from(puzzle_str);
        let solution = String::from(solution_str);
        let app_state = AppState::NotLoaded(NotLoaded{base, puzzle, solution});
        Self {
            egui_mq: egui_miniquad::EguiMq::new(ctx),
            pipeline,pipeline_glyphs,pipeline_tracks,shapes,app_state
        }
    }
}

fn atom_color(t: AtomType) -> [f32;3]{
    use AtomType::*;
    match t{
        Salt  => [0.8, 0.8, 0.8],
        Air   => [0., 1., 1.],
        Earth => [0., 1., 0.],
        Fire  => [1., 0., 0.],
        Water => [0., 0., 1.],
        Vitae => [1., 0.6, 0.6],
        Mors  => [0.4, 0., 0.],
        Quicksilver => [1.,1.,1.],
        Gold => [1., 1., 0.2],
        Silver => [0.3, 0.3, 0.3],
        Copper => [0.8, 0.4, 0.1],
        Iron => [0.2, 0.2, 0.2],
        Tin => [0.4, 0.4, 0.2],
        Lead => [0.3, 0.3, 0.3],
        Quintessence => {
            let t = ((miniquad::date::now()/2.).fract() as f32)*PI*2.;
            let colorize = |o:f32|->f32 {
                (t+o).sin().max(0.)
            };
            [colorize(0.),colorize(PI*2./3.),colorize(-PI*2./3.)]
        },
        RepeatingOutputMarker | ConduitSpace => [0., 0., 0.],
    }
}

impl EventHandler for MyMiniquadApp {
    fn update(&mut self, _: &mut Context) {}

    fn draw(&mut self, ctx: &mut Context) {
        ctx.clear(Some((1., 1., 1., 1.)), None, None);
        ctx.begin_default_pass(PassAction::clear_color(0.5, 0.5, 0.5, 1.0));

        if let Loaded(loaded) = &mut self.app_state{
            if loaded.last_timestep > loaded.curr_timestep{
                loaded.last_timestep = 0;
                loaded.last_world = loaded.base_world.clone();
            }
            if loaded.last_timestep < loaded.curr_timestep{
                for time in loaded.last_timestep..loaded.curr_timestep{
                    let output = loaded.last_world.run_step();
                    if let Err(output) = output{
                        loaded.message = Some(output.to_string());
                        loaded.last_world = loaded.base_world.clone();
                        for _ in 0..time{
                            loaded.last_world.run_step().unwrap();
                        }
                        loaded.curr_timestep = time;
                        loaded.max_timestep = time;
                        break;
                    }
                }
                loaded.last_timestep = loaded.curr_timestep;
            }
            let world = &loaded.last_world;
            let scale = loaded.camera.scale;
            let world_offset = loaded.camera.offset;
            let y_factor = f32::sqrt(3.)*2.0;
            let inv_scale= 1./scale;
            let base_x = ((-inv_scale-world_offset[0])/2.0).ceil()*2.0;
            let base_y = ((-inv_scale-world_offset[1])/y_factor).ceil()*y_factor;

            ctx.apply_pipeline(&self.pipeline_tracks);
            let (track_binds, track_index_count) = &loaded.track_binds;
            ctx.apply_bindings(track_binds);//Hex grid
            ctx.apply_uniforms(&BasicUniforms {
                color:[1., 1., 1.], offset:[0.,0.], world_offset, angle:0., scale
            });
            ctx.draw(0, *track_index_count as i32, 1);

            //Draw input/output atoms
            for glyph in world.glyphs.iter(){
                use GlyphType::*;
                match &glyph.glyph_type{
                    Input(atoms) | Output(atoms,_) => {
                        //Draw in/out atom bonds
                        ctx.apply_pipeline(&self.pipeline);
                        ctx.apply_bindings(&self.shapes.bond_bindings);
                        for atom in atoms {
                            let color = [1., 1., 1.];
                            let offset = pos_to_xy(&atom.pos);
                            for r in 0..6 {
                                let bond = atom.connections[r];
                                if bond == Bonds::NORMAL{
                                    let angle = rot_to_angle(r as Rot);
                                    ctx.apply_uniforms(&BasicUniforms {
                                        color, offset, world_offset, angle, scale
                                    });
                                    ctx.draw(0, 4, 1);
                                }
                            }
                        }
                        //Draw in/out atom circles
                        ctx.apply_bindings(&self.shapes.circle_bindings);
                        for atom in atoms {
                            let color = atom_color(atom.atom_type);
                            let offset = pos_to_xy(&atom.pos);
                            let angle = 0.;
                            ctx.apply_uniforms(&BasicUniforms {
                                color, offset, world_offset, angle, scale
                            });
                            ctx.draw(0, (CIRCLE_VERT_COUNT*3) as i32, 1);
                        }
                    },
                    _ => continue,
                };
            }
            //Draw the Hex grid
            ctx.apply_pipeline(&self.pipeline_glyphs);
            ctx.apply_bindings(&self.shapes.glyph_bindings[13]);
            for x in 0..(inv_scale/3.0).ceil() as i32 +1{
                for y in 0..(inv_scale/y_factor).ceil() as i32 *2+1{
                    let offset = [base_x+(x as f32*6.0),
                                  base_y+(y as f32*y_factor)];
                    ctx.apply_uniforms(&UvUniforms {
                        offset, world_offset, angle:0., scale
                    });
                    ctx.draw(0, 6, 1);
                }
            }
            //Draw glyphs (including half-transparent cover for input/outputs)
            for glyph in world.glyphs.iter(){
                let offset = pos_to_xy(&glyph.pos);
                let angle = rot_to_angle(glyph.rot);
                use GlyphType::*;
                let i = match &glyph.glyph_type{
                    Animismus       => 0,
                    Bonding         => 1,
                    Calcification   => 2,
                    Dispersion      => 3,
                    Disposal        => 4,
                    Duplication     => 5,
                    Equilibrium     => 6,
                    MultiBond       => 7,
                    Projection      => 8,
                    Purification    => 9,
                    TriplexBond     => 10,
                    Unbonding       => 11,
                    Unification     => 12,
                    Input(atoms) | Output(atoms,_) => {
                        ctx.apply_bindings(&self.shapes.glyph_bindings[14]);
                        for atom in atoms{
                            let offset = pos_to_xy(&atom.pos);
                            ctx.apply_uniforms(&UvUniforms {
                                offset, world_offset, angle, scale
                            });
                            ctx.draw(0, 6, 1);
                        }
                        continue
                    },
                    Track(_) | Conduit(_) => continue,
                };
                ctx.apply_bindings(&self.shapes.glyph_bindings[i]);
                ctx.apply_uniforms(&UvUniforms {
                    offset, world_offset, angle, scale
                });
                ctx.draw(0, 6, 1);
            }

            //Draw atom bonds
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&self.shapes.bond_bindings);
            for (_,atom) in world.atoms.atom_map.iter() {
                let color = [1., 1., 1.];
                let offset = pos_to_xy(&atom.pos);
                for r in 0..6 {
                    let bond = atom.connections[r];
                    if bond == Bonds::NORMAL{
                        let angle = rot_to_angle(r as Rot);
                        ctx.apply_uniforms(&BasicUniforms {
                            color, offset, world_offset, angle, scale
                        });
                        ctx.draw(0, 4, 1);
                    }
                }
            }
            //Draw atom circles
            ctx.apply_bindings(&self.shapes.circle_bindings);
            for (_,atom) in world.atoms.atom_map.iter() {
                let color = atom_color(atom.atom_type);
                let offset = pos_to_xy(&atom.pos);
                let angle = 0.;
                ctx.apply_uniforms(&BasicUniforms {
                    color, offset, world_offset, angle, scale
                });
                ctx.draw(0, (CIRCLE_VERT_COUNT*3) as i32, 1);
            }
            //Draw arms
            ctx.apply_bindings(&self.shapes.arm_bindings);
            for arm in world.arms.iter() {
                let color = [0., 0., 0.];
                let offset = pos_to_xy(&arm.pos);
                let triangles_drawn = if arm.grabbing {6} else {3};
                for r in (0..6).step_by(arm.angles_between_arm() as usize) {
                    let angle = rot_to_angle(arm.rot+r);
                    ctx.apply_uniforms(&BasicUniforms {
                        color, offset, world_offset, angle, scale
                    });
                    ctx.draw((arm.len-1)*6, triangles_drawn, 1);
                }
            }
        }
        ctx.end_render_pass();
        let mut do_loading = false;
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
                                }
                            }
                            if ui.button("+1").clicked() {
                                if loaded.curr_timestep < loaded.max_timestep {
                                    loaded.curr_timestep += 1;
                                }
                            }
                            ui.label("Max Timestep: ");
                            ui.add(egui::DragValue::new(&mut loaded.max_timestep)
                                .speed(0.1));
                            ui.separator();
                            if loaded.max_timestep < loaded.curr_timestep{
                                loaded.max_timestep = loaded.curr_timestep;
                            }

                            let min_size = loaded.base_world.arms.iter()
                                .fold(0, |val, arm| usize::max(val, arm.instruction_tape.instructions.len()));
                            ui.label("Loop length: ");
                            ui.add(egui::DragValue::new(&mut loaded.base_world.repeat_length)
                                .clamp_range(min_size..=usize::MAX));
                            ui.separator();

                            if ui.button("Save").clicked() {
                                use std::{fs::File, io::BufWriter, io::prelude::*};
                                let f = File::create("output.solution").unwrap();
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
                        });
                    });
                    egui::Window::new("Arms").hscroll(true).show(egui_ctx, |ui| {
                        ui.checkbox(&mut loaded.tape_mode, "Tape/overwrite mode");
                        let marker = " ".repeat(loaded.curr_timestep+3)+"V"+
                        &(" ".repeat(loaded.max_timestep-loaded.curr_timestep));
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
                            let min_size = loaded.base_world.arms.iter()
                                .fold(0, |val, arm| usize::max(val, arm.instruction_tape.instructions.len()));
                            if loaded.base_world.repeat_length < min_size{
                                loaded.base_world.repeat_length = min_size;
                            }
                            loaded.message = None;
                        };
                    });
                }
                NotLoaded(dat) => {
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
                            do_loading = true;
                        }
                    });
                }
            };
        });
        self.egui_mq.draw(ctx);
        if do_loading{
            if let NotLoaded(dat) = &mut self.app_state{
                use std::{fs::File, io::BufReader, path::Path};
                let base_path = Path::new(&dat.base);
                let f_puzzle = File::open(base_path.join(&dat.puzzle)).unwrap();
                let puzzle = parser::parse_puzzle(&mut BufReader::new(f_puzzle)).unwrap();
                let f_sol = File::open(base_path.join(&dat.solution)).unwrap();
                let solution = parser::parse_solution(&mut BufReader::new(f_sol)).unwrap();
                println!("Check: {:?}", solution.stats);
                let init = parser::puzzle_prep(&puzzle, &solution).unwrap();
                for (a_num, a) in init.arms.iter().enumerate(){
                    println!("Arms {:02}: {:?}", a_num, a.instruction_tape.to_string());
                }
                let world = World::setup_sim(&init).unwrap();
                let mut test_world = world.clone();

                while !test_world.is_complete() && test_world.timestep < 10000{
                    let res = test_world.run_step();
                    if let Err(e) = res{
                        println!("test world error: {:?}", e);
                        break;
                    }
                    if test_world.timestep % 100 == 0{
                        println!("Initial sim step {:03}", test_world.timestep);
                    }
                }
                let camera = CameraSetup::process(&test_world);
                let track_binds = setup_tracks(ctx, &test_world.track_map);
                let new_loaded = Loaded{
                    base_world: world.clone(),
                    curr_timestep: 0,
                    max_timestep: test_world.timestep as usize,
                    last_world: world,
                    last_timestep: 0,
                    tape_mode: false,
                    camera, track_binds, solution,
                    message: None,
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