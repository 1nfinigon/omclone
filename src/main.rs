mod check;
mod nonnan;
mod parser;
mod sim;
mod utils;

#[cfg(feature = "benchmark")]
mod benchmark;

#[cfg(feature = "nn")]
mod nn;
#[cfg(feature = "nn")]
mod solver;
#[cfg(feature = "nn")]
mod search;
#[cfg(feature = "nn")]
mod search_history;
#[cfg(feature = "nn")]
mod search_state;
#[cfg(feature = "nn")]
mod train;

#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod render_library;
#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod render_sim;
#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod ui;

use eyre::Result;

#[cfg(feature = "color_eyre")]
use color_eyre::install;
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::install;

fn main() -> Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "full");
    }
    install()?;

    let args: Vec<_> = std::env::args().collect();

    #[cfg(feature = "benchmark")]
    if args.get(1).map(|x| x.as_str()) == Some("benchmark") {
        return benchmark::main();
    }

    #[cfg(feature = "nn")]
    if args.get(1).map(|x| x.as_str()) == Some("solver") {
        return solver::main();
    }

    #[cfg(feature = "nn")]
    if args.get(1).map(|x| x.as_str()) == Some("train") {
        return train::main();
    }

    #[cfg(any(feature = "editor_ui", feature = "display_ui",))]
    {
        use miniquad::*;
        let conf = conf::Conf {
            fullscreen: false,
            ..Default::default()
        };
        miniquad::start(conf, || Box::new(ui::MyMiniquadApp::new()));

        return Ok(());
    }

    #[cfg(not(any(feature = "editor_ui", feature = "display_ui",)))]
    {
        panic!("no command, and not compiled with support for the UI");
    }
}
