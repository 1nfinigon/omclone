use miniquad::*;
use crate::sim::*;

pub type GFXPos = [f32;2];
use std::f32::consts::PI;
pub fn pos_to_xy(input: &Pos) -> GFXPos{
    let a = input.x as f32;
    let b = input.y as f32;
    [a*2.+b,b*f32::sqrt(3.)]
}
pub fn rot_to_angle(r: Rot) -> f32{
    (-r as f32)*PI/3.
}

type Vert = [f32;2];
//Vertex format: (x, y)
//note: 1 hex has inner radius of 1 (width of 2).
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

pub struct TrackBindings{
    bindings: Bindings,
    vert_count: usize,
}
pub fn setup_tracks(ctx: &mut Context, track: &TrackMap) -> TrackBindings{
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
    TrackBindings{
        bindings: Bindings {
            vertex_buffers: vec![vb],
            index_buffer,
            images: vec![],
        },
        vert_count: index_vec.len()
    }
}

type UvVert = [f32;4];
//(x, y), (u, v)
const TEXTURE_COUNT:usize = 20;
fn setup_textures(ctx: &mut Context) -> [Bindings;TEXTURE_COUNT]{
    const TEXTURE_VERT_BUF: [UvVert;4] = [
        [-3.,-3.,    0., 1.],
        [-3., 3.,    0., 0.],
        [ 3.,-3.,    1., 1.],
        [ 3., 3.,    1., 0.]];
    const TEXTURE_INDEX_BUF: [u16;6] = [
        0, 1, 2,
        1, 2, 3];
    let vb = Buffer::immutable(ctx, BufferType::VertexBuffer, &TEXTURE_VERT_BUF);
    let index_buffer = Buffer::immutable(ctx, BufferType::IndexBuffer, &TEXTURE_INDEX_BUF);
    let texture_list:[&[u8];TEXTURE_COUNT] = [
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
        include_bytes!("../images/HexGrid.png"),//13
        include_bytes!("../images/ShadeAtomInOut.png"),
        include_bytes!("../images/ShadeAreaFill.png"),
        include_bytes!("../images/BondNormal.png"),//16
		include_bytes!("../images/BondRed.png"),
		include_bytes!("../images/BondWhite.png"),
		include_bytes!("../images/BondYellow.png"),
    ];
    texture_list.map(|byte_data| -> Bindings{
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
    texture_bindings: [Bindings;TEXTURE_COUNT],
}
pub struct CameraSetup{
    pub scale: f32,
    pub offset: GFXPos,
}
impl CameraSetup{
    pub fn frame_center(world: &World) -> Self{
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
pub struct RenderDataBase {
    pipeline: Pipeline,
    pipeline_textured: Pipeline,
    pipeline_tracks: Pipeline,
    shapes: ShapeStore,
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
impl RenderDataBase {
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
        let pipeline_textured = Pipeline::with_params(
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
            texture_bindings: setup_textures(ctx),
        };
        
        Self {
            pipeline,pipeline_textured,pipeline_tracks,shapes
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

impl RenderDataBase {
	fn draw_atoms(&self, ctx: &mut Context, atoms:&[FloatAtom], camera: &CameraSetup){
        let scale = camera.scale;
        let world_offset = camera.offset;
        ctx.apply_pipeline(&self.pipeline_textured);

        //Draw atom bonds
        let atoms_copy = atoms;
		for atom in atoms_copy {
			let offset = [atom.pos.x, atom.pos.y];
			for r in 0..6 {
				let matches = [
					(Bonds::NORMAL, &self.shapes.texture_bindings[16]),
					(Bonds::TRIPLEX_R, &self.shapes.texture_bindings[17]),
					(Bonds::TRIPLEX_K, &self.shapes.texture_bindings[18]),
					(Bonds::TRIPLEX_Y, &self.shapes.texture_bindings[19])];
				let bond = atom.connections[r];
				for (bondtype, bindtype) in matches{
					if bond.intersects(bondtype){
						let angle = rot_to_angle(r as Rot)+atom.rot;
						ctx.apply_bindings(bindtype);
						ctx.apply_uniforms(&UvUniforms {
							offset, world_offset, angle, scale
						});
						ctx.draw(0, 4, 1);
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
			ctx.apply_uniforms(&BasicUniforms {
				color, offset, world_offset, angle, scale
			});
			ctx.draw(0, (CIRCLE_VERT_COUNT*3) as i32, 1);
		}
	}
    //note: assumes ctx is in the middle of a render pass
    pub fn draw(&self, ctx: &mut Context, camera: &CameraSetup, tracks: &TrackBindings,
         show_area: bool, world: &World, float_world: &FloatWorld)
    {
        let scale = camera.scale;
        let world_offset = camera.offset;
        let y_factor = f32::sqrt(3.)*2.0;
        let inv_scale= 1./scale;
        let base_x = ((-inv_scale-world_offset[0])/2.0).ceil()*2.0;
        let base_y = ((-inv_scale-world_offset[1])/y_factor).ceil()*y_factor;

        ctx.apply_pipeline(&self.pipeline_tracks);
        ctx.apply_bindings(&tracks.bindings);//Hex grid
        ctx.apply_uniforms(&BasicUniforms {
            color:[1., 1., 1.], offset:[0.,0.], world_offset, angle:0., scale
        });
        ctx.draw(0, tracks.vert_count as i32, 1);

        //Draw input/output atoms
        for glyph in world.glyphs.iter(){
            use GlyphType::*;
            match &glyph.glyph_type{
                Input(atoms_meta) | Output(atoms_meta,_) => {
                    let atoms = atoms_meta[0].iter().map(|x| x.into());
                    let atoms_vec:Vec<FloatAtom> = atoms.collect();
					self.draw_atoms(ctx, &atoms_vec, camera);
                },
                _ => continue,
            };
        }
        //Draw the Hex grid
        ctx.apply_pipeline(&self.pipeline_textured);
        ctx.apply_bindings(&self.shapes.texture_bindings[13]);
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
                Input(atoms_meta) | Output(atoms_meta,_) => {
                    let atoms = &atoms_meta[0];
                    ctx.apply_bindings(&self.shapes.texture_bindings[14]);
                    for atom in atoms{ //transparent cover
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
            ctx.apply_bindings(&self.shapes.texture_bindings[i]);
            ctx.apply_uniforms(&UvUniforms {
                offset, world_offset, angle, scale
            });
            ctx.draw(0, 6, 1);
        }
        //draw area cover
        if show_area{
            ctx.apply_pipeline(&self.pipeline_textured);
            ctx.apply_bindings(&self.shapes.texture_bindings[15]);                
            for p in &world.area_touched{
                let offset = pos_to_xy(p);
                ctx.apply_uniforms(&UvUniforms {
                    offset, world_offset, angle:0., scale
                });
                ctx.draw(0, 6, 1);
            }
        }
		
		//Draw atoms
        let atoms_slice = &float_world.atoms_xy[..];
		self.draw_atoms(ctx, atoms_slice, camera);
		
        //Draw arms
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.shapes.arm_bindings);
        for f_arm in float_world.arms_xy.iter() {
            let color = [0., 0., 0.];
            let offset = [f_arm.pos.x, f_arm.pos.y];
            let triangles_drawn = if f_arm.grabbing {6} else {3};
            for r in (0..6).step_by(Arm::angles_between_arm(f_arm.arm_type) as usize) {
                let angle = f_arm.rot+rot_to_angle(r);
                ctx.apply_uniforms(&BasicUniforms {
                    color, offset, world_offset, angle, scale
                });
                let rounded_len = (f_arm.len.round() as i32)/2;
                ctx.draw((rounded_len-1)*6, triangles_drawn, 1);
            }
        }
    }
}