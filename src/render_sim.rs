//! Render [sim::World](crate::sim::World) into a [miniquad::Context]

use crate::sim::*;
use miniquad::*;

use crate::render_library::*;

use std::{collections::BTreeMap, f32::consts::PI};
pub fn pos_to_xy(input: &Pos) -> GFXPos {
    let a = input.x as f32;
    let b = input.y as f32;
    [a * 2. + b, b * f32::sqrt(3.)]
}
pub fn rot_to_angle(r: RawRot) -> f32 {
    (r.0 as f32) * PI / 3.
}

//note: 1 hex has inner radius of 1 (width of 2).
//hex height, 3 tiles tip-to-tip=sqrt(3)*5/6
const Y_FACTOR: f32 = 1.7320508;
fn setup_arms(ctx: &mut dyn RenderingBackend) -> Bindings {
    const ARM_VERT_BUF: [Vert; 14] = [
        //Arm Base
        [0., -0.4],
        [0., 0.4],
        //Arm edge
        [2., 0.],
        [4., 0.],
        [6., 0.],
        //Grab markers
        [1.8, 0.],
        [2.2, -0.4],
        [2.2, 0.4],
        [3.8, 0.],
        [4.2, -0.4],
        [4.2, 0.4],
        [5.8, 0.],
        [6.2, -0.4],
        [6.2, 0.4],
    ];
    //First triangle is arm, 2nd triangle is optional grab marker
    const ARM_INDEX_BUF: [u16; 18] = [0, 1, 2, 5, 6, 7, 0, 1, 3, 8, 9, 10, 0, 1, 4, 11, 12, 13];
    let vb = ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&ARM_VERT_BUF),
    );
    let index_buffer = ctx.new_buffer(
        BufferType::IndexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&ARM_INDEX_BUF),
    );
    Bindings {
        vertex_buffers: vec![vb],
        index_buffer,
        images: vec![],
    }
}

pub struct TrackBindings {
    bindings: Bindings,
    vert_count: usize,
}
pub fn setup_tracks(ctx: &mut dyn RenderingBackend, tracks: &TrackMaps) -> TrackBindings {
    let mut verts_vec: Vec<GFXPos> = Vec::new();
    let mut index_vec: Vec<u16> = Vec::new();
    let mut curr_index = 0;
    let mut add_from = |track: &TrackMap| {
        for (from, offset) in track {
            let gfxpos = pos_to_xy(from);
            verts_vec.push(gfxpos);
            index_vec.push(curr_index);
            curr_index += 1;
            let gfxoffset = pos_to_xy(offset);
            let outer_pos = [
                gfxpos[0] + gfxoffset[0] * 0.5,
                gfxpos[1] + gfxoffset[1] * 0.5,
            ];
            verts_vec.push(outer_pos);
            index_vec.push(curr_index);
            curr_index += 1;
        }
    };
    add_from(&tracks.plus);
    add_from(&tracks.minus);

    let vb = ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&verts_vec),
    );
    let index_buffer = ctx.new_buffer(
        BufferType::IndexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&index_vec),
    );
    TrackBindings {
        bindings: Bindings {
            vertex_buffers: vec![vb],
            index_buffer,
            images: vec![],
        },
        vert_count: index_vec.len(),
    }
}

#[derive(Copy, Clone, Eq, Ord, PartialEq, PartialOrd, enum_iterator::Sequence)]
enum TextureId {
    Ani,
    Bonder,
    Calcification,
    Dispersion,
    Disposal,
    Duplication,
    Equilibrium,
    Multibond,
    Projection,
    Purification,
    Triplex,
    Unbonder,
    Unification,
    HexGridTiling,
    ShadeAtomInOut,
    ShadeAreaFill,
    BondNormal,
    BondRed,
    BondWhite,
    BondYellow,
}

impl TextureId {
    fn byte_data(self) -> &'static [u8] {
        match self {
            Self::Ani => include_bytes!("../images/Ani.png"),
            Self::Bonder => include_bytes!("../images/Bonder.png"),
            Self::Calcification => include_bytes!("../images/Calcification.png"),
            Self::Dispersion => include_bytes!("../images/Dispersion.png"),
            Self::Disposal => include_bytes!("../images/Disposal.png"),
            Self::Duplication => include_bytes!("../images/Duplication.png"),
            Self::Equilibrium => include_bytes!("../images/Equilibrium.png"),
            Self::Multibond => include_bytes!("../images/Multibond.png"),
            Self::Projection => include_bytes!("../images/Projection.png"),
            Self::Purification => include_bytes!("../images/Purification.png"),
            Self::Triplex => include_bytes!("../images/Triplex.png"),
            Self::Unbonder => include_bytes!("../images/Unbonder.png"),
            Self::Unification => include_bytes!("../images/Unification.png"),
            Self::HexGridTiling => include_bytes!("../images/HexGridTiling.png"),
            Self::ShadeAtomInOut => include_bytes!("../images/ShadeAtomInOut.png"),
            Self::ShadeAreaFill => include_bytes!("../images/ShadeAreaFill.png"),
            Self::BondNormal => include_bytes!("../images/BondNormal.png"),
            Self::BondRed => include_bytes!("../images/BondRed.png"),
            Self::BondWhite => include_bytes!("../images/BondWhite.png"),
            Self::BondYellow => include_bytes!("../images/BondYellow.png"),
        }
    }
}

fn setup_textures(ctx: &mut dyn RenderingBackend) -> BTreeMap<TextureId, Bindings> {
    const HL: f32 = Y_FACTOR * 5. / 3.; //height of 1.5 hex (from tip to tip)
    const TEXTURE_VERT_BUF: [UvVert; 4] = [
        [-3., HL - 6., 0., 1.],
        [-3., HL, 0., 0.],
        [3., HL - 6., 1., 1.],
        [3., HL, 1., 0.],
    ];
    const TEXTURE_INDEX_BUF: [u16; 6] = [0, 1, 2, 1, 2, 3];
    let vb = ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&TEXTURE_VERT_BUF),
    );
    let index_buffer = ctx.new_buffer(
        BufferType::IndexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&TEXTURE_INDEX_BUF),
    );
    enum_iterator::all::<TextureId>()
        .map(|texture_id| {
            use image::ImageFormat::Png;
            use image::ImageReader;
            use std::io::Cursor;
            let img = ImageReader::with_format(Cursor::new(texture_id.byte_data()), Png)
                .decode()
                .unwrap()
                .into_rgba8();
            let params = TextureParams {
                format: TextureFormat::RGBA8,
                wrap: TextureWrap::Repeat,
                width: img.width(),
                height: img.height(),
                ..Default::default()
            };
            let texture = ctx.new_texture_from_data_and_format(&img, params);
            //let texture = Texture::from_rgba8(ctx, width, height, &img);
            let bindings = if texture_id == TextureId::HexGridTiling {
                let tmp_vb = ctx.new_buffer(
                    BufferType::VertexBuffer,
                    BufferUsage::Stream,
                    BufferSource::empty::<UvVert>(4),
                );
                Bindings {
                    vertex_buffers: vec![tmp_vb],
                    index_buffer,
                    images: vec![texture],
                }
            } else {
                Bindings {
                    vertex_buffers: vec![vb],
                    index_buffer,
                    images: vec![texture],
                }
            };
            (texture_id, bindings)
        })
        .collect()
}
const CIRCLE_VERT_COUNT: usize = 20;
fn setup_circle(ctx: &mut dyn RenderingBackend) -> Bindings {
    let mut verts = [[0.; 2]; CIRCLE_VERT_COUNT + 1];
    let angle_per = PI * 2. / (CIRCLE_VERT_COUNT as f32);
    //verts[CIRCLE_VERT_COUNT] = [0.,0.];
    let mut indices: [u16; CIRCLE_VERT_COUNT * 3] =
        [CIRCLE_VERT_COUNT as u16; CIRCLE_VERT_COUNT * 3];
    for i in 0..CIRCLE_VERT_COUNT {
        let angle = (i as f32) * angle_per;
        verts[i] = [angle.cos() * 0.8, angle.sin() * 0.8];
        indices[i * 3 + 1] = (i) as u16;
        indices[i * 3 + 2] = ((i + 1) % CIRCLE_VERT_COUNT) as u16;
    }
    let vertex_buffer = ctx.new_buffer(
        BufferType::VertexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&verts),
    );
    let index_buffer = ctx.new_buffer(
        BufferType::IndexBuffer,
        BufferUsage::Immutable,
        BufferSource::slice(&indices),
    );
    Bindings {
        vertex_buffers: vec![vertex_buffer],
        index_buffer,
        images: vec![],
    }
}
struct ShapeStore {
    arm_bindings: Bindings,
    circle_bindings: Bindings,
    texture_bindings: BTreeMap<TextureId, Bindings>,
}
pub struct CameraSetup {
    pub scale_base: f32,
    pub offset: GFXPos,
}
impl CameraSetup {
    pub fn scale(&self, screen_size: (f32, f32)) -> (f32, f32) {
        (
            1000. / screen_size.0 * self.scale_base,
            1000. / screen_size.1 * self.scale_base,
        )
    }
    pub fn frame_center(world: &World, screen_size: (f32, f32)) -> Self {
        let pos_list = world.glyphs.iter().map(|x| pos_to_xy(&x.pos));
        let (mut lowx, mut lowy, mut highx, mut highy) = pos_list.fold(
            (
                f32::INFINITY,
                f32::INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
            ),
            |(lowx, lowy, highx, highy), [thisx, thisy]| {
                let new_lowx = if thisx < lowx { thisx } else { lowx };
                let new_highx = if thisx > highx { thisx } else { highx };
                let new_lowy = if thisy < lowy { thisy } else { lowy };
                let new_highy = if thisy > highy { thisy } else { highy };
                (new_lowx, new_lowy, new_highx, new_highy)
            },
        );
        const BORDER: f32 = 6.;
        lowx -= BORDER;
        lowy -= BORDER;
        highx += BORDER;
        highy += BORDER;
        let offset = [-(lowx + highx) / 2., -(lowy + highy) / 2.];
        let world_width_scale = screen_size.0 / 1000. * 2. / (highx - lowx);
        let world_height_scale = screen_size.1 / 1000. * 2. / (highy - lowy);
        let scale_base = world_width_scale.min(world_height_scale);
        println!(
            "camera debug: screen {:?}, scales {:?},{:?} wh {:?},{:?}",
            screen_size,
            world_width_scale,
            world_height_scale,
            highx - lowx,
            highy - lowy
        );
        println!("camera: x{}, +{:?}", scale_base, offset);
        CameraSetup { scale_base, offset }
    }
}
pub struct RenderDataBase {
    pipeline: Pipeline,
    pipeline_textured: Pipeline,
    pipeline_tracks: Pipeline,
    shapes: ShapeStore,
    font: FontStorage,
}

impl RenderDataBase {
    pub fn new(ctx: &mut dyn RenderingBackend) -> Self {
        let shader_meta = ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("color", UniformType::Float3),
                    UniformDesc::new("offset", UniformType::Float2),
                    UniformDesc::new("world_offset", UniformType::Float2),
                    UniformDesc::new("angle", UniformType::Float1),
                    UniformDesc::new("scale", UniformType::Float2),
                ],
            },
        };
        const V_SHADE: &str = include_str!("basic_vert.vs");
        const F_SHADE: &str = include_str!("basic_frag.fs");

        let shader = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: V_SHADE,
                    fragment: F_SHADE,
                },
                shader_meta,
            )
            .unwrap();

        let pipeline = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[VertexAttribute::new("local_pos", VertexFormat::Float2)],
            shader,
            Default::default(),
        );

        let shader_meta_uv = ShaderMeta {
            images: vec!["tex".to_string()],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("offset", UniformType::Float2),
                    UniformDesc::new("world_offset", UniformType::Float2),
                    UniformDesc::new("angle", UniformType::Float1),
                    UniformDesc::new("scale", UniformType::Float2),
                ],
            },
        };
        const V_UV_SHADE: &str = include_str!("uv_vert.vs");
        const F_UV_SHADE: &str = include_str!("uv_frag.fs");
        let shader_uv = ctx
            .new_shader(
                ShaderSource::Glsl {
                    vertex: V_UV_SHADE,
                    fragment: F_UV_SHADE,
                },
                shader_meta_uv,
            )
            .unwrap();
        use miniquad::graphics::*;
        let pipeline_textured = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[
                VertexAttribute::new("local_pos", VertexFormat::Float2),
                VertexAttribute::new("uv", VertexFormat::Float2),
            ],
            shader_uv,
            PipelineParams {
                color_blend: Some(BlendState::new(
                    Equation::Add,
                    BlendFactor::Value(BlendValue::SourceAlpha),
                    BlendFactor::OneMinusValue(BlendValue::SourceAlpha),
                )),
                ..Default::default()
            },
        );
        let pipeline_tracks = ctx.new_pipeline(
            &[BufferLayout::default()],
            &[VertexAttribute::new("local_pos", VertexFormat::Float2)],
            shader,
            PipelineParams {
                primitive_type: PrimitiveType::Lines,
                ..Default::default()
            },
        );
        let shapes = ShapeStore {
            arm_bindings: setup_arms(ctx),
            circle_bindings: setup_circle(ctx),
            texture_bindings: setup_textures(ctx),
        };

        let font = FontStorage::new(ctx);

        Self {
            pipeline,
            pipeline_textured,
            pipeline_tracks,
            shapes,
            font,
        }
    }
}

fn atom_color(t: AtomType) -> [f32; 3] {
    use AtomType::*;
    match t {
        Salt => [0.8, 0.8, 0.8],
        Air => [0., 1., 1.],
        Earth => [0., 1., 0.],
        Fire => [1., 0., 0.],
        Water => [0., 0., 1.],
        Vitae => [1., 0.6, 0.6],
        Mors => [0.4, 0., 0.],
        Quicksilver => [1., 1., 1.],
        Gold => [1., 0.85, 0.2],
        Silver => [0.35, 0.3, 0.4],
        Copper => [0.8, 0.4, 0.1],
        Iron => [0.35, 0.25, 0.20],
        Tin => [0.4, 0.4, 0.2],
        Lead => [0.3, 0.3, 0.3],
        Quintessence => {
            let t = ((miniquad::date::now() / 2.).fract() as f32) * PI * 2.;
            let colorize = |o: f32| -> f32 { (t + o).sin().max(0.) };
            [
                colorize(0.),
                colorize(PI * 2. / 3.),
                colorize(-PI * 2. / 3.),
            ]
        }
        RepeatingOutputMarker | ConduitSpace => [0., 0., 0.],
    }
}

impl RenderDataBase {
    fn draw_atoms(
        &self,
        ctx: &mut Context,
        screen_size: (f32, f32),
        atoms: &[FloatAtom],
        camera: &CameraSetup,
    ) {
        let scale = camera.scale(screen_size);
        let world_offset = camera.offset;
        ctx.apply_pipeline(&self.pipeline_textured);

        //Draw atom bonds
        let atoms_copy = atoms;
        for atom in atoms_copy {
            let offset = [atom.pos.x, atom.pos.y];
            for r in 0..3 {
                //4-6 are done by the other atom
                let matches = [
                    (
                        Bonds::NORMAL,
                        &self.shapes.texture_bindings[&TextureId::BondNormal],
                    ),
                    (
                        Bonds::TRIPLEX_R,
                        &self.shapes.texture_bindings[&TextureId::BondRed],
                    ),
                    (
                        Bonds::TRIPLEX_K,
                        &self.shapes.texture_bindings[&TextureId::BondWhite],
                    ),
                    (
                        Bonds::TRIPLEX_Y,
                        &self.shapes.texture_bindings[&TextureId::BondYellow],
                    ),
                ];
                let bond = atom.connections[r];
                for (bondtype, bindtype) in matches {
                    if bond.intersects(bondtype) {
                        let angle = rot_to_angle(RawRot(r as i32)) + atom.rot;
                        ctx.apply_bindings(bindtype);
                        ctx.apply_uniforms(UniformsSource::table(&UvUniforms {
                            offset,
                            world_offset,
                            angle,
                            scale,
                        }));
                        ctx.draw(0, 6, 1);
                    }
                }
            }
        }
        //Draw atom circles
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.shapes.circle_bindings);
        for atom in atoms {
            let color = atom_color(atom.atom_type);
            let offset = [atom.pos.x, atom.pos.y];
            let angle = 0.;
            ctx.apply_uniforms(UniformsSource::table(&BasicUniforms {
                color,
                offset,
                world_offset,
                angle,
                scale,
            }));
            ctx.draw(0, (CIRCLE_VERT_COUNT * 3) as i32, 1);
        }
    }
    //note: assumes ctx is in the middle of a render pass
    pub fn draw(
        &self,
        ctx: &mut Context,
        screen_size: (f32, f32),
        camera: &CameraSetup,
        tracks: &TrackBindings,
        show_area: bool,
        world: &World,
        float_world: &FloatWorld,
    ) {
        let scale = camera.scale(screen_size);
        let world_offset = camera.offset;

        //Draw input/output atoms
        let mut temp_atoms_vec: Vec<FloatAtom> = Vec::new();
        for glyph in world.glyphs.iter() {
            use GlyphType::*;
            temp_atoms_vec.clear();
            match &glyph.glyph_type {
                Input { pattern, .. }
                | Output { pattern, .. }
                | OutputRepeating { pattern, .. } => {
                    for atom in pattern {
                        temp_atoms_vec.push(atom.into());
                    }
                    self.draw_atoms(ctx, screen_size, &temp_atoms_vec, camera);
                }
                _ => continue,
            };
        }
        //Draw the Hex grid
        if camera.scale_base > 0.01 {
            let (inv_scale_x, inv_scale_y) = (1. / scale.0, 1. / scale.1);
            let xc = (-world_offset[0] / 2. - 0.25).fract();
            let yc = ((-world_offset[1] / 2.) / Y_FACTOR + 0.10).fract();
            let xdf = inv_scale_x / 2.;
            let ydf = inv_scale_y / (2. * Y_FACTOR);

            let hex_grid_data: [UvVert; 4] = [
                [-1., -1., xc - xdf, yc - ydf],
                [-1., 1., xc - xdf, yc + ydf],
                [1., -1., xc + xdf, yc - ydf],
                [1., 1., xc + xdf, yc + ydf],
            ];
            ctx.buffer_update(
                self.shapes.texture_bindings[&TextureId::HexGridTiling].vertex_buffers[0],
                BufferSource::slice(&hex_grid_data),
            );
            ctx.apply_pipeline(&self.pipeline_textured);
            ctx.apply_bindings(&self.shapes.texture_bindings[&TextureId::HexGridTiling]);
            ctx.apply_uniforms(UniformsSource::table(&UvUniforms {
                offset: [0., 0.],
                world_offset: [0., 0.],
                angle: 0.,
                scale: (1., 1.),
            }));
            ctx.draw(0, 6, 1);
        }
        //Draw glyphs (including half-transparent cover for input/outputs)
        ctx.apply_pipeline(&self.pipeline_textured);
        for glyph in world.glyphs.iter() {
            let offset = pos_to_xy(&glyph.pos);
            let angle = rot_to_angle(glyph.rot);
            use GlyphType::*;
            let texture_id = match &glyph.glyph_type {
                Animismus => TextureId::Ani,
                Bonding => TextureId::Bonder,
                Calcification => TextureId::Calcification,
                Dispersion => TextureId::Dispersion,
                Disposal => TextureId::Disposal,
                Duplication => TextureId::Duplication,
                Equilibrium => TextureId::Equilibrium,
                MultiBond => TextureId::Multibond,
                Projection => TextureId::Projection,
                Purification => TextureId::Purification,
                TriplexBond => TextureId::Triplex,
                Unbonding => TextureId::Unbonder,
                Unification => TextureId::Unification,
                Input { pattern, .. }
                | Output { pattern, .. }
                | OutputRepeating { pattern, .. } => {
                    ctx.apply_bindings(&self.shapes.texture_bindings[&TextureId::ShadeAtomInOut]);
                    for atom in pattern {
                        //transparent cover
                        let offset = pos_to_xy(&atom.pos);
                        ctx.apply_uniforms(UniformsSource::table(&UvUniforms {
                            offset,
                            world_offset,
                            angle,
                            scale,
                        }));
                        ctx.draw(0, 6, 1);
                    }
                    continue;
                }
                Conduit { locs: pos_vec, .. } => {
                    ctx.apply_bindings(&self.shapes.texture_bindings[&TextureId::ShadeAtomInOut]);
                    for p in pos_vec {
                        let offset = pos_to_xy(p);
                        ctx.apply_uniforms(UniformsSource::table(&UvUniforms {
                            offset,
                            world_offset,
                            angle,
                            scale,
                        }));
                        ctx.draw(0, 6, 1);
                    }
                    continue;
                }
                Track { .. } => continue,
            };
            ctx.apply_bindings(&self.shapes.texture_bindings[&texture_id]);
            ctx.apply_uniforms(UniformsSource::table(&UvUniforms {
                offset,
                world_offset,
                angle,
                scale,
            }));
            ctx.draw(0, 6, 1);
        }
        //Draw conduit numbers
        self.font.set_pipeline(ctx);
        for glyph in world.glyphs.iter() {
            if let GlyphType::Conduit { id, .. } = glyph.glyph_type {
                let offset = pos_to_xy(&glyph.pos);
                let string = id.to_string();
                self.font
                    .render_text_centered(ctx, &string, offset, world_offset, scale);
            }
        }
        //draw area cover
        if show_area {
            ctx.apply_pipeline(&self.pipeline_textured);
            ctx.apply_bindings(&self.shapes.texture_bindings[&TextureId::ShadeAreaFill]);
            for p in &world.area_touched {
                let offset = pos_to_xy(p);
                ctx.apply_uniforms(UniformsSource::table(&UvUniforms {
                    offset,
                    world_offset,
                    angle: 0.,
                    scale,
                }));
                ctx.draw(0, 6, 1);
            }
        }

        //Draw tracks
        ctx.apply_pipeline(&self.pipeline_tracks);
        ctx.apply_bindings(&tracks.bindings);
        ctx.apply_uniforms(UniformsSource::table(&BasicUniforms {
            color: [1., 1., 1.],
            offset: [0., 0.],
            world_offset,
            angle: 0.,
            scale,
        }));
        ctx.draw(0, tracks.vert_count as i32, 1);

        //Draw atoms
        let atoms_slice = &float_world.atoms_xy[..];
        self.draw_atoms(ctx, screen_size, atoms_slice, camera);

        //Draw arms
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.shapes.arm_bindings);
        for f_arm in float_world.arms_xy.iter() {
            let color = [0., 0., 0.];
            let offset = [f_arm.pos.x, f_arm.pos.y];
            let triangles_drawn = if f_arm.grabbing { 6 } else { 3 };
            for r in Rot::step_by(f_arm.arm_type.angles_between_arm()) {
                let angle = f_arm.rot + rot_to_angle(r.raw_nonneg());
                ctx.apply_uniforms(UniformsSource::table(&BasicUniforms {
                    color,
                    offset,
                    world_offset,
                    angle,
                    scale,
                }));
                let rounded_len = (f_arm.len.round() as i32) / 2;
                ctx.draw((rounded_len - 1) * 6, triangles_drawn, 1);
            }
        }
        self.font.set_pipeline(ctx);
        for (num, f_arm) in float_world.arms_xy.iter().enumerate() {
            let offset = [f_arm.pos.x, f_arm.pos.y];
            let string = (num + 1).to_string();
            self.font
                .render_text_centered(ctx, &string, offset, world_offset, scale);
        }
    }
}
