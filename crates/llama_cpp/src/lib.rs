//! High-level bindings to [llama.cpp][llama.cpp]'s C API, providing a predictable, safe, and
//! high-performance medium for interacting with Large Language Models (LLMs) on consumer-grade
//! hardware.
//!
//! **Along with llama.cpp, this crate is still in an early state, and breaking changes may occur
//! between versions.** The high-level API, however, is fairly settled on.
//!
//! To get started, create a [`LlamaModel`] and a [`LlamaSession`]:
//!
//! ```no_run
//! use llama_cpp::{LlamaModel, LlamaParams, SessionParams};
//! use llama_cpp::standard_sampler::StandardSampler;
//!
//! // Create a model from anything that implements `AsRef<Path>`:
//! let model = LlamaModel::load_from_file("path_to_model.gguf", LlamaParams::default()).expect("Could not load model");
//!
//! // A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
//! // several gigabytes large, a session is typically a few dozen to a hundred megabytes!
//! let mut ctx = model.create_session(SessionParams::default()).expect("Failed to create session");
//!
//! // You can feed anything that implements `AsRef<[u8]>` into the model's context.
//! ctx.advance_context("This is the story of a man named Stanley.").unwrap();
//!
//! // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
//! let max_tokens = 1024;
//! let mut decoded_tokens = 0;
//!
//! // `ctx.start_completing_with` creates a worker thread that generates tokens. When the completion
//! // handle is dropped, tokens stop generating!
//!
//! let mut completions = ctx.start_completing_with(StandardSampler::default(), 1024).into_strings();
//!
//! for completion in completions {
//!     print!("{completion}");
//!     let _ = io::stdout().flush();
//!
//!     decoded_tokens += 1;
//!
//!     if decoded_tokens > max_tokens {
//!         break;
//!     }
//! }
//! ```
//!
//! ## Dependencies
//!
//! This crate depends on (and builds atop) [`llama_cpp_sys`], and builds llama.cpp from source.
//! You'll need at least `libclang` and a C/C++ toolchain (`clang` is preferred).
//! See [`llama_cpp_sys`] for more details.
//!
//! The bundled GGML and llama.cpp binaries are statically linked by default, and their logs
//! are re-routed through [`tracing`][tracing] instead of `stderr`.
//! If you're getting stuck, setting up [`tracing`][tracing] for more debug information should
//! be at the top of your troubleshooting list!
//!
//! ## Undefined Behavior / Panic Safety
//!
//! It should be **impossible** to trigger [undefined behavior][ub] from this crate, and any
//! UB is considered a critical bug. UB triggered downstream in [llama.cpp][llama.cpp] or
//! [`ggml`][ggml] should have issues filed and mirrored in `llama_cpp-rs`'s issue tracker.
//!
//! While panics are considered less critical, **this crate should never panic**, and any
//! panic should be considered a bug. We don't want your control flow!
//!
//! ## Building
//!
//! Keep in mind that [llama.cpp][llama.cpp] is very computationally heavy, meaning standard
//! debug builds (running just `cargo build`/`cargo run`) will suffer greatly from the lack of optimisations. Therefore, unless
//! debugging is really necessary, it is highly recommended to build and run using Cargo's `--release` flag.
//!
//! ## Minimum Stable Rust Version (MSRV) Policy
//!
//! This crates supports Rust 1.73.0 and above.
//!
//! ## License
//!
//! MIT or Apache 2.0 (the "Rust" license), at your option.
//!
//! [ub]: https://doc.rust-lang.org/reference/behavior-considered-undefined.html
//! [tracing]: https://docs.rs/tracing/latest/tracing/
//! [ggml]: https://github.com/ggerganov/ggml/
//! [llama.cpp]: https://github.com/ggerganov/llama.cpp/

#![warn(missing_docs)]

use llama_cpp_sys::{llama_context, llama_token_data_array};
use thiserror::Error;

mod batch;
mod detail;
mod model;
mod session;

pub use model::*;
pub use session::*;

pub mod grammar;
mod multimodal;
/// The standard sampler implementation.
pub mod standard_sampler;

/// A single token produced or consumed by a [`LlamaModel`], without its associated context.
///
/// Due to the layout of llama.cpp, these can be _created_ from a [`LlamaModel`], but require a
/// [`LlamaSession`] to decode.
///
/// On its own, this isn't useful for anything other than being fed into
/// [`LlamaSession::advance_context_with_tokens`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct Token(pub i32);

/// An error that occurred on the other side of the C FFI boundary.
///
/// GGML and llama.cpp typically log useful information before failing, which is forwarded to this
/// crate's [`tracing`] handler.
///
/// [tracing]: https://docs.rs/tracing/latest/tracing/
#[derive(Error, Debug)]
#[error("an internal assertion failed in llama.cpp; check `tracing` output.")]
pub struct LlamaInternalError;

/// Something which selects a [`Token`] from the distribution output by a
/// [`LlamaModel`].
pub trait Sampler {
    /// Given a [`llama_context`], the tokens in context (to allow for
    /// repetition penalities), and a [`llama_token_data_array`] (which contains
    /// the distribution over the next token as output by the model), selects a
    /// token.
    fn sample(
        &mut self,
        context: *mut llama_context,
        tokens: &[Token],
        candidates_p: llama_token_data_array,
    ) -> Token;
}

/// Memory requirements for something.
///
/// This is typically returned by [`LlamaModel::estimate_session_size`] and
/// [`LlamaModel::estimate_embeddings_session_size`] as an estimation of memory usage.
#[derive(Debug)]
pub struct ResourceUsage {
    /// The host memory required, in bytes.
    pub host_memory: usize,

    /// The device memory required, in bytes.
    ///
    /// The device depends on features used to build this crate, as well as the main gpu selected during model creation.
    ///
    /// If the device is the CPU, this is additional host memory required.
    pub device_memory: usize,
}
