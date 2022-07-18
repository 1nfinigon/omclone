mod parser;
mod sim;
#[cfg(test)]
mod test;

#[cfg(any(feature = "editor_ui",feature = "display_ui",))]
mod render_library;
#[cfg(any(feature = "editor_ui",feature = "display_ui",))]
mod render_sim;
#[cfg(any(feature = "editor_ui",feature = "display_ui",))]
mod ui;

#[cfg(feature = "color_eyre")]
use color_eyre::{install, eyre::Result};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{install, eyre::Result};

#[cfg(any(feature = "editor_ui",feature = "display_ui",))]
fn main() -> Result< () >{
    #[cfg(not(target_arch = "wasm32"))]
    std::env::set_var("RUST_BACKTRACE", "full");
    install()?;

    use miniquad::*;
    let conf = conf::Conf{
        fullscreen: false,
        .. Default::default()
    };
    miniquad::start(conf, |mut ctx| {
        Box::new(ui::MyMiniquadApp::new(&mut ctx))
    });
    Ok(())
}
