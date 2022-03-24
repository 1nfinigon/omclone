mod parser;
mod sim;
#[cfg(test)]
mod test;

#[cfg(feature = "main_ui")]
mod render_sim;
#[cfg(feature = "main_ui")]
mod ui;

#[cfg(feature = "color_eyre")]
use color_eyre::{install, eyre::Result};
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::{install, eyre::Result};

#[cfg(feature = "main_ui")]
fn main() -> Result< () >{
    std::env::set_var("RUST_BACKTRACE", "full");
    install()?;

    use miniquad::*;
    let conf = conf::Conf{
        fullscreen: false,
        .. Default::default()
    };
    miniquad::start(conf, |mut ctx| {
        UserData::owning(ui::MyMiniquadApp::new(&mut ctx), ctx)
    });
    Ok(())
}
