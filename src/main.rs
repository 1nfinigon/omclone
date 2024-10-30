//! Opus Magnum simulator, tree-search, NN training utils.
//!
//! Binaries:
//! -   [ui] (`cargo run`)
//! -   [seed_solver] (`cargo run seed-solver`)
//! -   [gen_train] (`cargo run gen-train`)
//! -   [check] (`cargo run check`)
//! -   [benchmark] (`cargo run benchmark`)
//!
//! All binaries are run as subcommands, e.g. `cargo run seed-solver`
//!
//! Important modules:
//!
//! -   Simulator backend
//!     -   [sim]
//!     -   [parser]
//! -   Tree-search
//!     -   [search]
//!     -   [eval]
//! -   NN and training
//!     -   [nn]
//!     -   [gen_train]
//!
//! <details><summary>README.md</summary>
//!
#![doc = include_str!("../README.md")]
//!
//! </details>

mod check;
mod parser;
mod revgen;
mod sim;
mod utils;

#[cfg(feature = "benchmark")]
mod benchmark;

#[cfg(feature = "search")]
mod eval;
#[cfg(all(feature = "search", feature = "torch"))]
mod gen_train;
#[cfg(feature = "search")]
mod nn;
#[cfg(feature = "search")]
mod search;
#[cfg(feature = "search")]
mod search_history;
#[cfg(feature = "search")]
mod search_state;
#[cfg(feature = "search")]
mod seed_solver;
#[cfg(feature = "search")]
mod tb_trim;

#[cfg(any(feature = "editor_ui", feature = "display_ui",))]
mod export_png;
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

#[cfg(feature = "tracy")]
tracy_client::register_demangler!();

fn main() -> Result<()> {
    /*
    #[cfg(not(target_arch = "wasm32"))]
    unsafe {
        std::env::set_var("RUST_BACKTRACE", "full");
    }
    */

    #[cfg(feature = "search")]
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

        #[cfg(feature = "search")]
        Some("seed-solver") => seed_solver::main(args, tracy_client),

        #[cfg(all(feature = "search", feature = "torch"))]
        Some("gen-train") => gen_train::main(),

        #[cfg(feature = "search")]
        Some("tb-trim") => tb_trim::main(),

        Some("revgen") => revgen::main(),

        #[cfg(any(feature = "editor_ui", feature = "display_ui",))]
        Some("export-png") => export_png::main(),

        Some(subcommand) => Err(eyre!("unsupported subcommand {}", subcommand)),

        #[cfg(any(feature = "editor_ui", feature = "display_ui",))]
        None => ui::main(),

        #[cfg(not(any(feature = "editor_ui", feature = "display_ui",)))]
        None => {
            panic!("no command, and not compiled with support for the UI");
        }
    }
}
