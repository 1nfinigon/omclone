mod check;
mod nonnan;
mod parser;
mod revgen;
mod sim;
mod utils;

#[cfg(feature = "benchmark")]
mod benchmark;

#[cfg(feature = "nn")]
mod gen_train;
#[cfg(feature = "nn")]
mod nn;
#[cfg(feature = "nn")]
mod search;
#[cfg(feature = "nn")]
mod search_history;
#[cfg(feature = "nn")]
mod search_state;
#[cfg(feature = "nn")]
mod seed_solver;
#[cfg(feature = "nn")]
mod tb_trim;

#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod render_library;
#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod render_sim;
#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod ui;

use eyre::{eyre, Result};

#[cfg(feature = "color_eyre")]
use color_eyre::install;
#[cfg(not(feature = "color_eyre"))]
use simple_eyre::install;

tracy_client::register_demangler!();

fn main() -> Result<()> {
    /*
    #[cfg(not(target_arch = "wasm32"))]
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "full");
    }
    */

    #[cfg(feature = "nn")]
    unsafe {
        std::env::set_var("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
    }

    install()?;

    let tracy_client = tracy_client::Client::start();

    let mut args = std::env::args();

    let _prog_name = args.next().unwrap();
    let subcommand = args.next();

    match subcommand.as_deref() {
        Some("check") => check::main(),

        #[cfg(feature = "benchmark")]
        Some("benchmark") => benchmark::main(),

        #[cfg(feature = "nn")]
        Some("seed-solver") => seed_solver::main(args, tracy_client),

        #[cfg(feature = "nn")]
        Some("gen-train") => gen_train::main(),

        #[cfg(feature = "nn")]
        Some("tb-trim") => tb_trim::main(),

        Some("revgen") => revgen::main(),

        Some(subcommand) => Err(eyre!("unsupported subcommand {}", subcommand)),

        #[cfg(any(feature = "editor_ui", feature = "display_ui",))]
        None => ui::main(),

        #[cfg(not(any(feature = "editor_ui", feature = "display_ui",)))]
        None => {
            panic!("no command, and not compiled with support for the UI");
        }
    }
}
