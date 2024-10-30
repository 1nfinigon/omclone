use crate::parser;
use crate::render_sim;
use crate::sim;
use crate::utils;
use eyre::Result;
use miniquad::*;

struct DummyHandler;

impl EventHandler for DummyHandler {
    fn update(&mut self) {}
    fn draw(&mut self) {}
}

pub fn main() -> Result<()> {
    let (puzzle, sol) = utils::get_default_puzzle_solution()?;
    let init = parser::puzzle_prep(&puzzle, &sol)?;
    let world = sim::WorldWithTapes::setup_sim(&init)?;

    miniquad::start(
        miniquad::conf::Conf {
            window_width: 0,
            window_height: 0,
            fullscreen: false,
            ..Default::default()
        },
        move || {
            let mut mq_ctx = window::new_rendering_backend();

            let width = 256u32;
            let height = 256u32;
            let screen_size = (width as f32, height as f32);

            let color_img = mq_ctx.new_render_texture(TextureParams {
                width,
                height,
                format: TextureFormat::RGBA8,
                ..Default::default()
            });

            let offscreen_pass = mq_ctx.new_render_pass(color_img, None);

            let render_data = render_sim::RenderDataBase::new(mq_ctx.as_mut());
            mq_ctx.clear(Some((1., 1., 1., 1.)), None, None);
            mq_ctx.begin_pass(
                Some(offscreen_pass),
                PassAction::clear_color(0.5, 0.5, 0.5, 1.0),
            );

            let camera = render_sim::CameraSetup::frame_center(&world.world, screen_size);
            let tracks = render_sim::setup_tracks(mq_ctx.as_mut(), &world.world.track_maps);
            let mut float_world = sim::FloatWorld::new();
            float_world.generate_static(&world.world);

            render_data.draw(
                mq_ctx.as_mut(),
                screen_size,
                &camera,
                &tracks,
                false,
                &world.world,
                &float_world,
            );

            mq_ctx.end_render_pass();

            let mut color_img_bytes = vec![0u8; (width * height * 4).try_into().unwrap()];
            mq_ctx.texture_read_pixels(color_img, &mut color_img_bytes);

            let color_img = image::RgbaImage::from_raw(width, height, color_img_bytes).unwrap();
            let mut color_img = image::DynamicImage::ImageRgba8(color_img);
            color_img.apply_orientation(image::metadata::Orientation::FlipVertical);
            color_img.save("output.png").unwrap();

            window::order_quit();

            Box::new(DummyHandler)
        },
    );

    Ok(())
}
