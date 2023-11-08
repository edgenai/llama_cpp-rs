//! High-level bindings to [llama.cpp][llama.cpp]'s C API, providing a predictable, safe, and
//! high-performance medium for interacting with Large Language Models (LLMs) on consumer-grade
//! hardware.
//!
//! **Along with llama.cpp, his crate is still in an early state, and breaking changes may occur
//! between versions.** The high-level API, however, is fairly settled on.
//!
//! To get started, create a [`LlamaModel`] and a [`LlamaSession`]:
//!
//! ```no_run
//! use llama_cpp::LlamaModel;
//!
//! // Create a model from anything that implements `AsRef<Path>`:
//! let model = LlamaModel::load_from_file("path_to_model.gguf").expect("Could not load model");
//!
//! // A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
//! // several gigabytes large, a session is typically a few dozen to a hundred megabytes!
//! let mut ctx = model.create_session();
//!
//! // You can feed anything that implements `AsRef<[u8]>` into the model's context.
//! ctx.advance_context("This is the story of a man named Stanley.").unwrap();
//!
//! // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
//! let max_tokens = 1024;
//! let mut decoded_tokens = 0;
//!
//! // `ctx.get_completions` creates a worker thread that generates tokens. When the completion
//! // handle is dropped, tokens stop generating!
//!
//! let mut completions = ctx.start_completing();
//!
//! while let Some(next_token) = completions.next_token() {
//!     println!("{}", String::from_utf8_lossy(next_token.as_bytes()));
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
//! You'll need `libclang`, `cmake`, and a C/C++ toolchain (`clang` is preferred) at the minimum.
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
use std::ffi::{c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::{ptr, thread};
use tokio::sync::{Mutex, RwLock};

use ctor::{ctor, dtor};
use derive_more::{Deref, DerefMut};
use thiserror::Error;
use tracing::{error, info, trace};

use llama_cpp_sys::{
    llama_backend_free, llama_backend_init, llama_batch_free, llama_batch_init, llama_beam_search,
    llama_context, llama_context_default_params, llama_decode, llama_free, llama_free_model,
    llama_load_model_from_file, llama_log_set, llama_model, llama_model_default_params,
    llama_n_vocab, llama_new_context_with_model, llama_set_n_threads, llama_token_bos,
    llama_token_eos, llama_token_eot, llama_token_get_text, llama_token_middle, llama_token_nl,
    llama_token_prefix, llama_token_suffix, llama_tokenize,
};

/// [ctor](https://docs.rs/ctor/latest/ctor/) wind-up binding to invoke llama.cpp's
/// `llama_backend_init`, which is required before using the library.
///
/// This executes automatically before `main` via some linker shenanigans.
#[ctor]
fn llama_cpp_up() {
    unsafe {
        // SAFETY: This is the only time that `llama_backend_init` is called. We always assume that
        // NUMA is available.
        llama_backend_init(true);
    }

    unsafe {
        // SAFETY: `llama_backend_init` has already been called, so no logging will occur until
        // after `main` has entered.
        llama_log_set(Some(detail::llama_log_callback), ptr::null_mut());
    }
}

/// [ctor](https://docs.rs/ctor/latest/ctor/) teardown binding to invoke llama.cpp's
/// `llama_backend_free`, which frees any memory claimed by [`llama_cpp_up`].
///
/// This executes automatically following `main` via some linker shenanigans.
#[dtor]
fn llama_cpp_down() {
    unsafe {
        // SAFETY: This is the only time that `llama_backend_free` is called.
        llama_backend_free();
    }
}

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

/// An error raised while loading a llama.cpp model.
#[derive(Error, Debug)]
pub enum LlamaLoadError {
    /// The given path couldn't be loaded because it doesn't exist on the filesystem.
    #[error("Path does not exist: {0}")]
    DoesNotExist(PathBuf),

    /// Something went wrong on the other side of the C FFI boundary.
    #[error("Llama.cpp couldn't load the provided model: {0}")]
    LlamaError(#[from] LlamaInternalError),
}

/// An error raised while tokenizing some input for a model.
#[derive(Error, Debug)]
pub enum LlamaTokenizationError {
    /// llama.cpp only supports vectors of length up to `i32::MAX`.
    #[error("Input was too large: {n_bytes} were provided, but llama.cpp only supports up to {max_bytes}")]
    InputTooLarge {
        /// The number of bytes that were being tokenized.
        n_bytes: usize,

        /// The maximum number of bytes that _can_ be tokenized.
        max_bytes: usize,
    },

    /// Something went wrong on the other side of the C FFI boundary.
    #[error("Tokenization failed: {0}")]
    LlamaError(#[from] LlamaInternalError),
}

/// An error that occurred on the other side of the C FFI boundary.
///
/// GGML and llama.cpp typically log useful information before failing, which is forwarded to this
/// crate's [`tracing`] handler.
///
/// [tracing]: https://docs.rs/tracing/latest/tracing/
#[derive(Error, Debug)]
#[error("an internal assertion failed in llama.cpp; check `tracing` output.")]
pub struct LlamaInternalError;

/// The inner part of a [`LlamaModel`].
///
/// This is a thin wrapper over an `Arc<RwLock<*mut llama_model>>`, which is used to share the
/// model across threads.
#[derive(Clone, Deref, DerefMut)]
struct LlamaModelInner(*mut llama_model);

unsafe impl Send for LlamaModelInner {}
unsafe impl Sync for LlamaModelInner {}

impl Drop for LlamaModelInner {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: `drop`ping more than once is unsound [1], so `self.model` cannot have been
            // `free`d yet.
            //
            // [1]: See https://github.com/rust-lang/rust/issues/60977
            llama_free_model(self.0);
        }
    }
}

/// A [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master) model.
///
/// At present, these can only be loaded from GGML's model file format, [GGUF][gguf], via
/// [`LlamaModel::load_from_file`].
///
/// [gguf]: https://github.com/ggerganov/ggml/pull/302
#[derive(Clone)]
pub struct LlamaModel {
    /// A handle to the inner model on the other side of the C FFI boundary.
    model: Arc<RwLock<LlamaModelInner>>,

    /// The size of this model's vocabulary, in tokens.
    vocabulary_size: usize,

    /// The beginning of sentence (BOS) token for this model.
    bos_token: Token,

    /// The end of sentence (EOS) token for this model.
    eos_token: Token,

    /// The newline (NL) token for this model.
    nl_token: Token,

    /// For infilling, the prefix token for this model.
    infill_prefix_token: Token,

    /// For infilling, the middle token for this model.
    infill_middle_token: Token,

    /// For infilling, the suffix token for this model.
    infill_suffix_token: Token,

    /// For infilling, the token for the end of the infill.
    eot_token: Token,
}

unsafe impl Send for LlamaModel {}

impl LlamaModel {
    /// Loads a LLaMA model from a compatible GGUF (`.gguf`) file.
    ///
    /// If the model fails to load on the other side of the C FFI boundary, llama.cpp will log an
    /// error to this crate's `tracing` handler.
    /// If you're stuck here, consider setting up [`tracing`][tracing] to get the whole story.
    ///
    /// [tracing]: https://docs.rs/tracing/latest/tracing/
    pub fn load_from_file(file_path: impl AsRef<Path>) -> Result<Self, LlamaLoadError> {
        let file_path = file_path.as_ref();

        if !file_path.exists() {
            return Err(LlamaLoadError::DoesNotExist(file_path.into()));
        }

        let model = unsafe {
            // SAFETY: Assume that llama.cpp will gracefully fail and return `nullptr` if
            // `llama_load_model_from_file` fails.
            //
            // This is, unfortunately, the best we can do here.
            llama_load_model_from_file(
                CString::new(file_path.to_string_lossy().into_owned().into_bytes())
                    .unwrap_or_else(|_| {
                        unreachable!(
                            "Path {:#?} contained NUL bytes; this should never happen",
                            file_path
                        )
                    })
                    .as_ptr(),
                llama_model_default_params(),
            )
        };

        if model.is_null() {
            Err(LlamaInternalError.into())
        } else {
            let vocabulary_size = unsafe {
                // SAFETY: `model` is not null.
                llama_n_vocab(model)
            };

            Ok(Self {
                model: Arc::new(RwLock::new(LlamaModelInner(model))),
                vocabulary_size: vocabulary_size as usize,
                bos_token: Token(unsafe { llama_token_bos(model) }),
                eos_token: Token(unsafe { llama_token_eos(model) }),
                nl_token: Token(unsafe { llama_token_nl(model) }),
                infill_prefix_token: Token(unsafe { llama_token_prefix(model) }),
                infill_middle_token: Token(unsafe { llama_token_middle(model) }),
                infill_suffix_token: Token(unsafe { llama_token_suffix(model) }),
                eot_token: Token(unsafe { llama_token_eot(model) }),
            })
        }
    }

    /// Converts `content` into a vector of tokens that are valid input for this model.
    ///
    /// This temporarily allocates at the amount of memory consumed by `content`, but shrinks that
    /// allocation shortly after.
    pub fn tokenize_bytes(
        &self,
        content: impl AsRef<[u8]>,
    ) -> Result<Vec<Token>, LlamaTokenizationError> {
        let content = content.as_ref();

        if content.len() > i32::MAX as usize {
            return Err(LlamaTokenizationError::InputTooLarge {
                n_bytes: content.len(),
                max_bytes: i32::MAX as usize,
            });
        }

        let mut out_buf = Vec::with_capacity(content.len());

        let n_written_tokens = unsafe {
            // SAFETY: The pointer ranges specified here are always valid, and `n_written_tokens`
            // is always less than `content.len()`.
            //
            // `content.len()` always fits within an `i32`.
            //
            // `out_buf` is a `Vec<Token>`, and `Token` is `#[repr(transparent)]` over an `i32`.
            llama_tokenize(
                **self.model.try_read().unwrap(),
                content.as_ptr() as *const i8,
                content.len() as i32,
                out_buf.as_mut_ptr() as *mut i32,
                out_buf.capacity() as i32,
                false,
                false,
            )
        };

        if n_written_tokens >= 0 {
            unsafe {
                // SAFETY: if `n_written_tokens` is non-negative, tokenization succeeded, and
                // the value is the number of tokens present in `out_buf`.
                out_buf.set_len(n_written_tokens as usize);
            }

            out_buf.shrink_to_fit();

            Ok(out_buf)
        } else {
            Err(LlamaInternalError.into())
        }
    }

    /// Gets the byte string representation of `token` in this model's vocabulary.
    ///
    /// The returned slice is valid for the lifetime of this session, and typically encodes
    /// a UTF-8 string; consider using [`String::from_utf8_lossy`] if you need to display the
    /// contents.
    fn detokenize(&self, token: Token) -> &[u8] {
        assert!(
            (token.0 as usize) < self.vocabulary_size,
            "{} is out of range for this model's vocabulary range",
            token.0
        );

        unsafe { CStr::from_ptr(llama_token_get_text(**self.model.try_read().unwrap(), token.0)) }.to_bytes()
    }

    /// Creates a new evaluation context for this model.
    ///
    /// The model must live for at least as long as the context, but many contexts can be created
    /// from the same model.
    ///
    /// The vast majority of loaded data (weights) are immutably stored in the model, with a much
    /// smaller state belonging to each context. For Zephyr 7B, this works out to about 4GiB for
    /// the model weights and 100MiB for each session.
    pub fn create_session(&self) -> LlamaSession {
        let params = unsafe {
            // SAFETY: Stack constructor; always safe.
            llama_context_default_params()
        };

        let ctx = unsafe {
            // SAFETY: due to `_model` being declared in the `LlamaContext`, `self` must live
            // for at least the lifetime of `LlamaContext`.
            llama_new_context_with_model(**self.model.blocking_read(), params)
        };

        let cpus = num_cpus::get() as u32;

        unsafe {
            // SAFETY: The presence of `u32::MAX` CPU cores would create a black hole, so you
            // should consider closing your laptop and running.
            llama_set_n_threads(ctx, cpus, cpus);
        }

        LlamaSession {
            model: self.clone(),
            inner: Arc::new(Mutex::new(LlamaContextInner { ptr: ctx }) ),
            history_size: 0,
        }
    }


    /// Returns the beginning of sentence (BOS) token for this context.
    pub fn bos(&self) -> Token {
        self.bos_token
    }

    /// Returns the end of sentence (EOS) token for this context.
    pub fn eos(&self) -> Token {
        self.eos_token
    }

    /// Returns the newline (NL) token for this context.
    pub fn nl(&self) -> Token {
        self.nl_token
    }

    /// Returns the infill prefix token for this context.
    pub fn infill_prefix(&self) -> Token {
        self.infill_prefix_token
    }

    /// Returns the infill middle token for this context.
    pub fn infill_middle(&self) -> Token {
        self.infill_middle_token
    }

    /// Returns the infill suffix token for this context.
    pub fn infill_suffix(&self) -> Token {
        self.infill_suffix_token
    }

    /// Returns the infill end of middle token for this context.
    pub fn eot(&self) -> Token {
        self.eot_token
    }
}

/// The inner part of a [`LlamaSession`].
///
/// This is wrapped in an `Arc` for sharing across thread boundaries.
struct LlamaContextInner {
    /// A pointer to the inner context.
    ptr: *mut llama_context,
}

unsafe impl Send for LlamaContextInner {}
unsafe impl Sync for LlamaContextInner {}

impl Drop for LlamaContextInner {
    fn drop(&mut self) {
        // SAFETY: `drop`ping more than once is unsound [1], so `self.model` cannot have been
        // `free`d yet.
        //
        // [1]: See https://github.com/rust-lang/rust/issues/60977
        unsafe { llama_free(self.ptr) }
    }
}

/// An evaluation session for a llama.cpp model.
///
/// This stores a small amount of state, which is destroyed when the session is dropped.
/// You can create an arbitrary number of sessions for a model using [`LlamaModel::create_session`].
pub struct LlamaSession {
    /// The model this session was created from.
    model: LlamaModel,

    /// A pointer to the llama.cpp side of the model context.
    inner: Arc<Mutex<LlamaContextInner>>,

    /// The number of tokens present in this model's context.
    history_size: usize,
}

/// An error raised while advancing the context in a [`LlamaSession`].
#[derive(Error, Debug)]
pub enum LlamaContextError {
    /// If non-tokens were provided, tokenizing the input failed.
    #[error("tokenization failed: {0}")]
    TokenizationFailed(#[from] LlamaTokenizationError),

    /// Too many tokens were provided.
    ///
    /// llama.cpp only supports vectors of length up to `i32::MAX`.
    #[error("{provided_tokens} were provided, but llama.cpp can only handle {max_tokens}")]
    MaxTokensExceeded {
        /// The number of provided tokens.
        provided_tokens: usize,

        /// The maximum number of tokens.
        max_tokens: usize,
    },

    /// No tokens were provided at all.
    #[error("no tokens were provided")]
    NoTokensProvided,

    /// An error occurred on the other side of the FFI boundary; check your logs.
    #[error("advancing context failed: {0}")]
    LlamaError(#[from] LlamaInternalError),
}

impl LlamaSession {
    /// Advances the inner context of this model with `tokens`.
    ///
    /// The model will generate new tokens from the end of the context.
    pub fn advance_context_with_tokens(
        &mut self,
        tokens: impl AsRef<[Token]>,
    ) -> Result<(), LlamaContextError> {
        let tokens = tokens.as_ref();
        let n_tokens = tokens.len();

        if n_tokens == 0 {
            return Err(LlamaContextError::NoTokensProvided);
        }

        if n_tokens > i32::MAX as usize {
            return Err(LlamaContextError::MaxTokensExceeded {
                provided_tokens: n_tokens,
                max_tokens: i32::MAX as usize,
            });
        }

        let mut batch = unsafe {
            // SAFETY: `llama_batch_init` is a stack constructor, and should never fail.
            llama_batch_init(n_tokens.max(4) as i32, 0, 1)
        };

        batch.n_tokens = n_tokens as i32;

        for (i, token) in tokens.iter().enumerate() {
            trace!("Writing token {i} of {n_tokens} ({token:?})");

            unsafe {
                // SAFETY: For all 0 < i < n_tokens, `llama_batch_init` created each of these
                // offsets; although each offset may be currently uninitialized.
                batch.token.add(i).write(token.0);
                batch.pos.add(i).write(i as i32);
                batch.logits.add(i).write(1);
                batch.n_seq_id.add(i).write(1);

                let seq_ptr = batch.seq_id.add(i);

                if !seq_ptr.is_null() {
                    **seq_ptr = 0;
                }
            }
        }

        unsafe {
            // SAFETY: `batch.logits[n_tokens - 1]` exists, for the same reasons outlined above.
            batch.logits.add(n_tokens - 1).write(1);
        }

        trace!("Wrote {n_tokens} tokens to the token buffer");
        trace!("Starting LLaMA decode for batch");

        if unsafe {
            // SAFETY: `llama_decode` will not fail for a valid `batch`, which we correctly
            // initialized above.
            llama_decode(self.inner.blocking_lock().ptr, batch)
        } != 0
        {
            return Err(LlamaInternalError.into());
        } else {
            trace!("Batch decode completed successfully");
        }

        trace!("Freeing batch");

        unsafe {
            // SAFETY: This is the last time we use `batch`, which is still currently initialized.
            llama_batch_free(batch)
        };

        self.history_size += tokens.len();

        Ok(())
    }

    /// Tokenizes and feeds an arbitrary byte buffer `ctx` into this model.
    ///
    /// `ctx` is typically a UTF-8 string, but anything that can be downcast to bytes is accepted.
    pub fn advance_context(&mut self, ctx: impl AsRef<[u8]>) -> Result<(), LlamaContextError> {
        let tokens = self.model.tokenize_bytes(ctx.as_ref())?.into_boxed_slice();

        self.advance_context_with_tokens(tokens)
    }

    /// Starts generating tokens at the end of the context using llama.cpp's built-in Beam search.
    /// This is where you want to be if you just want some completions.
    pub fn start_completing(&mut self) -> CompletionHandle {
        let (tx, rx) = flume::unbounded();

        info!(
            "Generating completions with {} tokens of history",
            self.history_size,
        );

        let past_tokens = self.history_size;
        let mutex = self.inner.clone();

        thread::spawn(move || unsafe {
            llama_beam_search(
                mutex.blocking_lock().ptr,
                Some(detail::llama_beam_search_callback),
                Box::leak(Box::new(detail::BeamSearchState { tx })) as *mut _ as *mut c_void,
                1,
                past_tokens as i32,
                32_768,
            );
        });

        CompletionHandle { ctx: self, rx }
    }
}

/// An intermediate token generated during an LLM completion.
pub struct CompletionToken<'a> {
    /// The session that generated this token.
    ctx: &'a LlamaSession,

    /// The generated token.
    token: Token,
}

impl<'a> CompletionToken<'a> {
    /// Decodes this token, returning the bytes composing it.
    pub fn as_bytes(&self) -> &[u8] {
        self.ctx.model.detokenize(self.token)
    }

    /// Returns this token as an `i32`.
    pub fn id(&self) -> i32 {
        self.token.0
    }
}

impl<'a> AsRef<Token> for CompletionToken<'a> {
    fn as_ref(&self) -> &Token {
        &self.token
    }
}

impl<'a, T: AsRef<Token>> PartialEq<T> for CompletionToken<'a> {
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.token == *other.as_ref()
    }
}

impl Eq for CompletionToken<'_> {}

/// A handle (and channel) to an ongoing completion job on an off thread.
///
/// If this structure is dropped, the off thread is stopped.
pub struct CompletionHandle<'a> {
    /// The session in charge of this completion.
    ctx: &'a mut LlamaSession,

    /// The token receiver bound to the off thread.
    rx: flume::Receiver<Token>,
}

impl<'a> CompletionHandle<'a> {
    /// Blocks the current thread, resolving to the next completed token, or `None` if EOS is
    /// reached.
    pub fn next_token(&mut self) -> Option<CompletionToken<'_>> {
        self.rx.recv().ok().map(|token| CompletionToken {
            ctx: self.ctx,
            token,
        })
    }

    /// Asynchronously yields the current thread, resolving to the next completed token, or `None`
    /// if EOS is reached.
    pub async fn next_token_async(&mut self) -> Option<CompletionToken<'_>> {
        self.rx
            .recv_async()
            .await
            .ok()
            .map(|token| CompletionToken {
                ctx: self.ctx,
                token,
            })
    }
}

mod detail {
    //! FFI implementation details.

    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::slice_from_raw_parts;
    use tokio::sync::OwnedSemaphorePermit;

    use tracing::{error, info, trace, warn};

    use llama_cpp_sys::{
        ggml_log_level, ggml_log_level_GGML_LOG_LEVEL_ERROR, ggml_log_level_GGML_LOG_LEVEL_INFO,
        ggml_log_level_GGML_LOG_LEVEL_WARN, llama_beams_state,
    };

    use crate::Token;

    pub(crate) struct BeamSearchState {
        pub(crate) tx: flume::Sender<Token>,
    }

    #[no_mangle]
    pub(crate) unsafe extern "C" fn llama_beam_search_callback(
        shared_state_ptr: *mut c_void,
        beam_state: llama_beams_state,
    ) {
        let shared_state = unsafe {
            // SAFETY: `channel` has this type and hasn't been de-allocated.
            &mut *(shared_state_ptr as *mut BeamSearchState)
        };

        if shared_state.tx.is_disconnected() {
            // Close all beams to terminate the search.
            for i in 0..beam_state.n_beams {
                unsafe {
                    // SAFETY: beam_views[i] exists where 0 <= i <= n_beams.
                    *beam_state.beam_views.add(i)
                }
                    .eob = true;
            }
        }

        // Llama.cpp trims the common prefix after every invocation; the presence of
        // `common_prefix_length > 0` means the first `common_prefix_length` tokens have been
        // settled upon.
        if beam_state.common_prefix_length > 0 {
            let first_beam = unsafe {
                // SAFETY: At least one beam always exists.
                &*(beam_state.beam_views)
            };

            let beam_tokens = unsafe {
                // SAFETY: If all beams share a common prefix, at least that many tokens exist in
                // every beam.
                &*slice_from_raw_parts(first_beam.tokens, beam_state.common_prefix_length)
            };

            for unshared_token in beam_tokens {
                let _ = shared_state.tx.send(Token(*unshared_token));
            }
        }

        if beam_state.last_call {
            unsafe {
                // SAFETY: `channel` is heap-allocated, and this is the only time we'll construct
                // a `Box` back over it; this is the last time this function will be called, and
                // the last time this pointer will be seen.
                let _ = Box::from_raw(shared_state);
            }
        }
    }

    #[no_mangle]
    pub(crate) unsafe extern "C" fn llama_log_callback(
        level: ggml_log_level,
        text: *const c_char,
        _user_data: *mut c_void,
    ) {
        let text = unsafe {
            // SAFETY: `text` is a NUL-terminated C String.
            CStr::from_ptr(text)
        };
        let text = String::from_utf8_lossy(text.to_bytes());

        match level {
            ggml_log_level_GGML_LOG_LEVEL_ERROR => error!("ggml: {text}"),
            ggml_log_level_GGML_LOG_LEVEL_INFO => info!("ggml: {text}"),
            ggml_log_level_GGML_LOG_LEVEL_WARN => warn!("ggml: {text}"),
            _ => trace!("ggml: {text}"),
        }
    }
}
