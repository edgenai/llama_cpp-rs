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
//! let mut ctx = model.create_session(SessionParams::default()).unwrap();
//!
//! // You can feed anything that implements `AsRef<[u8]>` into the model's context.
//! ctx.advance_context("This is the story of a man named Stanley.").unwrap();
//!
//! // LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
//! let max_tokens = 1024;
//! let mut decoded_tokens = 0;
//!
//! // `ctx.get_completions_with` creates a worker thread that generates tokens. When the completion
//! // handle is dropped, tokens stop generating!
//!
//! let mut completions = ctx.start_completing_with(StandardSampler::default(), 1024);
//!
//! while let Some(next_token) = completions.next_token() {
//!     println!("{}", String::from_utf8_lossy(&*next_token.detokenize()));
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

use std::cmp::min;
use std::ffi::{c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::{ptr, thread};

use derive_more::{Deref, DerefMut};
use futures::executor::block_on;
use thiserror::Error;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver};
use tokio::sync::{Mutex, RwLock};
use tracing::{error, info, trace, warn};

use llama_cpp_sys::{
    ggml_type, llama_backend_free, llama_backend_init, llama_batch, llama_batch_free,
    llama_batch_init, llama_beam_search, llama_context, llama_context_default_params,
    llama_context_params, llama_decode, llama_free, llama_free_model, llama_get_logits_ith,
    llama_load_model_from_file, llama_log_set, llama_model, llama_model_default_params,
    llama_model_params, llama_n_vocab, llama_new_context_with_model, llama_split_mode,
    llama_split_mode_LLAMA_SPLIT_LAYER, llama_split_mode_LLAMA_SPLIT_NONE,
    llama_split_mode_LLAMA_SPLIT_ROW, llama_token_bos, llama_token_data, llama_token_data_array,
    llama_token_eos, llama_token_eot, llama_token_get_text, llama_token_middle, llama_token_nl,
    llama_token_prefix, llama_token_suffix, llama_token_to_piece, llama_tokenize,
};

/// The standard sampler implementation.
pub mod standard_sampler;

/// The current instance of [`Backend`], if it exists. Also stored is a reference count used for
/// initialisation and freeing.
static BACKEND: Mutex<Option<(Backend, usize)>> = Mutex::const_new(None);

/// Empty struct used to initialise and free the [llama.cpp][llama.cpp] backend when it is created
/// dropped respectively.
///
/// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
struct Backend {}

impl Backend {
    /// Initialises the [llama.cpp][llama.cpp] backend and sets its logger.
    ///
    /// There should only ever be one instance of this struct at any given time.
    ///
    /// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
    fn init() -> Self {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_init(true);

            // SAFETY: performs a simple assignment to static variables. Should only execute once
            // before any logs are made.
            llama_log_set(Some(detail::llama_log_callback), ptr::null_mut());
        }

        Self {}
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_free();
        }
    }
}

/// A "reference" to [`BACKEND`].
///
/// Initialises [`BACKEND`] if there is no [`Backend`] inside. If there are no other references,
/// this drops [`Backend`] upon getting itself dropped.
struct BackendRef {}

impl BackendRef {
    /// Creates a new reference, initialising [`BACKEND`] if necessary.
    async fn new() -> Self {
        let mut lock = BACKEND.lock().await;
        if let Some((_, count)) = lock.as_mut() {
            *count += 1;
        } else {
            let _ = lock.insert((Backend::init(), 1));
        }

        Self {}
    }
}

impl Drop for BackendRef {
    fn drop(&mut self) {
        block_on(async move {
            let mut lock = BACKEND.lock().await;
            if let Some((_, count)) = lock.as_mut() {
                *count -= 1;

                if *count == 0 {
                    lock.take();
                }
            } else {
                error!("Backend as already been freed, this should never happen")
            }
        });
    }
}

impl Clone for BackendRef {
    fn clone(&self) -> Self {
        block_on(Self::new())
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
struct LlamaModelInner {
    #[deref]
    #[deref_mut]
    model: *mut llama_model,
    _backend_ref: BackendRef,
}

unsafe impl Send for LlamaModelInner {}

unsafe impl Sync for LlamaModelInner {}

impl Drop for LlamaModelInner {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: `drop`ping more than once is unsound [1], so `self.model` cannot have been
            // `free`d yet.
            //
            // [1]: See https://github.com/rust-lang/rust/issues/60977
            llama_free_model(self.model);
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
    pub fn load_from_file(
        file_path: impl AsRef<Path>,
        model_params: LlamaParams,
    ) -> Result<Self, LlamaLoadError> {
        let backend_ref = block_on(BackendRef::new());
        info!("Loading model \"{}\"", file_path.as_ref().to_string_lossy());

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
                model_params.into(),
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
                model: Arc::new(RwLock::new(LlamaModelInner {
                    model,
                    _backend_ref: backend_ref,
                })),
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

    /// Loads a LLaMA model from a compatible GGUF (`.gguf`) file asyncronously.
    ///
    /// This is a thin `tokio` wrapper over [`LlamaModel::load_from_file`].
    pub async fn load_from_file_async(
        file_path: impl AsRef<Path>,
        params: LlamaParams,
    ) -> Result<Self, LlamaLoadError> {
        let path = file_path.as_ref().to_owned();

        tokio::task::spawn_blocking(move || Self::load_from_file(path, params))
            .await
            .unwrap()
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
    pub fn detokenize(&self, token: Token) -> &[u8] {
        assert!(
            (token.0 as usize) < self.vocabulary_size,
            "{} is out of range for this model's vocabulary range",
            token.0
        );

        unsafe {
            CStr::from_ptr(llama_token_get_text(
                **self.model.try_read().unwrap(),
                token.0,
            ))
        }
        .to_bytes()
    }

    /// Converts the provided token into a [`String`] piece, using the model's vocabulary.
    ///
    /// Panics if the model is invalid.
    pub fn token_to_piece(&self, token: Token) -> String {
        let initial_size = 8u16;
        let mut buffer = vec![std::os::raw::c_char::from(0); usize::from(initial_size)];
        let size = unsafe {
            llama_token_to_piece(
                **self.model.try_read().unwrap(),
                token.0,
                buffer.as_mut_ptr(),
                std::os::raw::c_int::from(initial_size),
            )
        };

        buffer.resize(size.unsigned_abs() as usize + 1, 0);
        if size < 0 {
            let size = unsafe {
                llama_token_to_piece(
                    **self.model.try_read().unwrap(),
                    token.0,
                    buffer.as_mut_ptr(),
                    std::os::raw::c_int::from(buffer.len() as i32 - 1),
                )
            };
            assert_eq!(
                size as usize + 1,
                buffer.len(),
                "Buffer length doesn't match"
            );
        }

        let c_string = unsafe {
            // SAFETY: llama_token_to_piece should always return a null terminated buffer
            CString::from_vec_with_nul_unchecked(buffer.iter().map(move |x| *x as u8).collect())
        };
        c_string.to_string_lossy().to_string()
    }

    /// Creates a new evaluation context for this model.
    ///
    /// The model must live for at least as long as the context, but many contexts can be created
    /// from the same model.
    ///
    /// The vast majority of loaded data (weights) are immutably stored in the model, with a much
    /// smaller state belonging to each context. For Zephyr 7B, this works out to about 4GiB for
    /// the model weights and 100MiB for each session.
    pub fn create_session(
        &self,
        session_params: SessionParams,
    ) -> Result<LlamaSession, LlamaContextError> {
        let params = llama_context_params::from(session_params);
        let max_batch = params.n_batch;

        let ctx = unsafe {
            // SAFETY: due to `_model` being declared in the `LlamaContext`, `self` must live
            // for at least the lifetime of `LlamaContext`.
            llama_new_context_with_model(**self.model.try_read().unwrap(), params)
        };
        if ctx.is_null() {
            return Err(LlamaContextError::SessionFailed);
        }

        Ok(LlamaSession {
            inner: Arc::new(LlamaSessionInner {
                model: self.clone(),
                ctx: Mutex::new(LlamaContextInner { ptr: ctx }),
                history_size: AtomicUsize::new(0),
                last_batch_size: AtomicUsize::new(0),
                max_batch,
            }),
        })
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
#[derive(Clone)]
pub struct LlamaSession {
    inner: Arc<LlamaSessionInner>,
}

/// The cloned part of a [`LlamaSession`].
struct LlamaSessionInner {
    /// The model this session was created from.
    model: LlamaModel,

    /// A pointer to the llama.cpp side of the model context.
    ctx: Mutex<LlamaContextInner>,

    /// The number of tokens present in this model's context.
    history_size: AtomicUsize,

    /// The number of tokens present in this model's context.
    last_batch_size: AtomicUsize,

    /// Max batch size.
    max_batch: u32,
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
    #[error("failed to create llama context")]
    SessionFailed,

    /// An error occurred on the other side of the FFI boundary; check your logs.
    #[error("advancing context failed (error code {0})")]
    DecodeFailed(i32),
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

        info!("Advancing context with {n_tokens} tokens");

        let batch_size = min(n_tokens, self.inner.max_batch as usize);
        let sequences = tokens.chunks(batch_size);

        if n_tokens > batch_size {
            info!("Number of tokens exceeds the maximum batch size ({}) for this session, splitting the input", self.inner.max_batch);
        }

        let mut batch = Batch::new(batch_size, 0, 1);
        let history_size = self.inner.history_size.load(Ordering::SeqCst);
        let mut local_history = 0;
        let mut last_batch_size = self.inner.last_batch_size.load(Ordering::SeqCst);

        for sequence in sequences {
            batch.clear();

            for token in sequence {
                batch.add(*token, history_size + local_history, &[0], false);
                local_history += 1;
            }

            // Set the logits of the very last token
            if local_history == n_tokens {
                batch.set_logits(sequence.len() - 1, true);
            }

            trace!("Wrote {n_tokens} tokens to the token buffer");
            trace!("Starting LLaMA decode for batch");

            let err = unsafe {
                // SAFETY: `llama_decode` will not fail for a valid `batch`, which we correctly
                // initialized above.
                llama_decode(self.inner.ctx.blocking_lock().ptr, batch.handle())
            };
            if err != 0 {
                return Err(LlamaContextError::DecodeFailed(err));
            }
            trace!("Batch decode completed successfully");

            last_batch_size = sequence.len();
        }

        self.inner
            .history_size
            .fetch_add(local_history, Ordering::SeqCst);

        self.inner
            .last_batch_size
            .store(last_batch_size, Ordering::SeqCst);

        Ok(())
    }

    /// Advances the inner context of this model with `tokens`.
    ///
    /// This is a thin `tokio::spawn_blocking` wrapper around
    /// [`LlamaSession::advance_context_with_tokens`].
    pub async fn advance_context_with_tokens_async(
        &mut self,
        tokens: impl AsRef<[Token]>,
    ) -> Result<(), LlamaContextError> {
        let tokens = tokens.as_ref().to_owned();
        let mut session = self.clone();

        tokio::task::spawn_blocking(move || session.advance_context_with_tokens(tokens))
            .await
            .unwrap()
    }

    /// Tokenizes and feeds an arbitrary byte buffer `ctx` into this model.
    ///
    /// `ctx` is typically a UTF-8 string, but anything that can be downcast to bytes is accepted.
    pub fn advance_context(&mut self, ctx: impl AsRef<[u8]>) -> Result<(), LlamaContextError> {
        let tokens = self
            .inner
            .model
            .tokenize_bytes(ctx.as_ref())?
            .into_boxed_slice();

        self.advance_context_with_tokens(tokens)
    }

    /// Tokenizes and feeds an arbitrary byte buffer `ctx` into this model.
    ///
    /// This is a thin `tokio::spawn_blocking` wrapper around
    /// [`LlamaSession::advance_context`].
    pub async fn advance_context_async(
        &mut self,
        ctx: impl AsRef<[u8]>,
    ) -> Result<(), LlamaContextError> {
        let ctx = ctx.as_ref().to_owned();
        let mut session = self.clone();

        tokio::task::spawn_blocking(move || {
            let tokens = session.inner.model.tokenize_bytes(ctx)?.into_boxed_slice();

            session.advance_context_with_tokens(tokens)
        })
        .await
        .unwrap()
    }

    /// Starts generating tokens at the end of the context using llama.cpp's built-in Beam search.
    /// TODO fix: beam search keeps going even after it should have ended
    pub fn start_completing(&mut self) -> CompletionHandle {
        let (tx, rx) = unbounded_channel();
        let history_size = self.inner.history_size.load(Ordering::SeqCst);
        let session = self.clone();

        info!("Generating completions with {history_size} tokens of history");

        thread::spawn(move || unsafe {
            let state = Box::new(detail::BeamSearchState { tx });
            // SAFETY: `state_ptr` is converted back to a [`Box`] and freed in [`detail::llama_beam_search_callback`]
            let state_ptr = Box::into_raw(state);

            llama_beam_search(
                session.inner.ctx.blocking_lock().ptr,
                Some(detail::llama_beam_search_callback),
                state_ptr as *mut _ as *mut c_void,
                1,
                history_size as i32,
                32_768,
            );
        });

        CompletionHandle { rx }
    }

    /// Start completion.
    pub fn start_completing_with<S>(
        &mut self,
        sampler: S,
        max_predictions: usize,
    ) -> CompletionHandle
    where
        S: Sampler + Send + Sync + 'static,
    {
        let (tx, rx) = unbounded_channel();
        let history_size = self.inner.history_size.load(Ordering::SeqCst);
        let session = self.clone();
        // TODO deal with 0 history size
        info!("Generating completions with {history_size} tokens of history");

        thread::spawn(move || {
            let context = session.inner.ctx.blocking_lock();
            let vocab = session.model().vocabulary_size;
            let end_of_stream = session.model().eos_token;
            let mut count = 0;
            let mut batch = Batch::new(1, 0, 1);
            let mut i = session.inner.last_batch_size.load(Ordering::SeqCst);
            let mut current_pos = history_size;

            loop {
                let mut candidates = unsafe {
                    let logits = llama_get_logits_ith(context.ptr, (i - 1) as i32);

                    let mut candidates = vec![];
                    for id in 0..vocab {
                        candidates.push(llama_token_data {
                            id: id as i32,
                            logit: *logits.add(id),
                            p: 0.0,
                        })
                    }

                    candidates
                };

                let candidates_p = llama_token_data_array {
                    data: candidates.as_mut_ptr(),
                    size: vocab,
                    sorted: false,
                };

                let token = sampler.sample(context.ptr, candidates_p);

                match tx.send(token) {
                    Ok(_) => (),
                    Err(e) => {
                        let token_str =
                            String::from_utf8_lossy(session.inner.model.detokenize(e.0));
                        warn!("Cannot send token ({}): {}", token_str, e);
                        break;
                    }
                };

                if token == end_of_stream || max_predictions <= count {
                    break;
                }

                batch.clear();
                batch.add(token, current_pos, &[0], true);

                let res = unsafe { llama_decode(context.ptr, batch.handle()) };

                if res != 0 {
                    error!("Failed to decode context ({res})");
                    break;
                }

                count += 1;
                i = batch.tokens();

                session.inner.last_batch_size.store(i, Ordering::SeqCst);
                current_pos = session.inner.history_size.fetch_add(i, Ordering::SeqCst);
            }
        });

        CompletionHandle { rx }
    }

    /// Returns the model this session was created from.
    pub fn model(&self) -> LlamaModel {
        self.inner.model.clone()
    }
}

/// A handle (and channel) to an ongoing completion job on an off thread.
///
/// If this structure is dropped, the off thread is stopped.
pub struct CompletionHandle {
    /// The token receiver bound to the off thread.
    rx: UnboundedReceiver<Token>,
}

impl CompletionHandle {
    /// Blocks the current thread, resolving to the next completed token, or `None` if EOS is
    /// reached.
    pub fn next_token(&mut self) -> Option<Token> {
        block_on(self.rx.recv())
    }

    /// Asynchronously yields the current thread, resolving to the next completed token, or `None`
    /// if EOS is reached.
    pub async fn next_token_async(&mut self) -> Option<Token> {
        self.rx.recv().await
    }
}

/// Parameters for llama.
pub struct LlamaParams {
    /// Number of layers to store in VRAM.
    ///
    /// If this number is bigger than the amount of model layers, all layers are loaded to VRAM.
    pub n_gpu_layers: u32,

    /// How to split the model across multiple GPUs
    pub split_mode: SplitMode,

    /// The GPU that is used for scratch and small tensors
    pub main_gpu: u32,

    /// How to split layers across multiple GPUs (size: LLAMA_MAX_DEVICES)
    //const float * tensor_split, TODO

    /// Called with a progress value between 0 and 1, pass NULL to disable
    //llama_progress_callback progress_callback, TODO

    /// Context pointer passed to the progress callback
    //void * progress_callback_user_data, TODO

    /// Override key-value pairs of the model meta data
    //const struct llama_model_kv_override * kv_overrides, TODO

    /// Only load the vocabulary, no weights
    pub vocab_only: bool,

    /// Use mmap if possible
    pub use_mmap: bool,

    /// Force system to keep model in RAM
    pub use_mlock: bool,
}

/// A policy to split the model across multiple GPUs
#[non_exhaustive]
pub enum SplitMode {
    /// Single GPU.
    ///
    /// Equivalent to [`llama_split_mode_LLAMA_SPLIT_NONE`]
    None,

    /// Split layers and KV across GPUs
    ///
    /// Equivalent to [`llama_split_mode_LLAMA_SPLIT_LAYER`]
    Layer,

    /// Split rows across GPUs
    ///
    /// Equivalent to [`llama_split_mode_LLAMA_SPLIT_ROW`]
    Row,
}

impl From<SplitMode> for llama_split_mode {
    fn from(value: SplitMode) -> Self {
        match value {
            SplitMode::None => llama_split_mode_LLAMA_SPLIT_NONE,
            SplitMode::Layer => llama_split_mode_LLAMA_SPLIT_LAYER,
            SplitMode::Row => llama_split_mode_LLAMA_SPLIT_ROW,
        }
    }
}

impl From<llama_split_mode> for SplitMode {
    fn from(value: llama_split_mode) -> Self {
        #![allow(non_upper_case_globals)]
        match value {
            llama_split_mode_LLAMA_SPLIT_NONE => SplitMode::None,
            llama_split_mode_LLAMA_SPLIT_LAYER => SplitMode::Layer,
            llama_split_mode_LLAMA_SPLIT_ROW => SplitMode::Row,
            _ => unimplemented!(),
        }
    }
}

impl Default for LlamaParams {
    fn default() -> Self {
        // SAFETY: Stack constructor, always safe
        let c_params = unsafe { llama_model_default_params() };

        Self {
            n_gpu_layers: c_params.n_gpu_layers as u32,
            split_mode: c_params.split_mode.into(),
            main_gpu: c_params.main_gpu as u32,
            vocab_only: c_params.vocab_only,
            use_mmap: c_params.use_mmap,
            use_mlock: c_params.use_mlock,
        }
    }
}

impl From<LlamaParams> for llama_model_params {
    fn from(value: LlamaParams) -> Self {
        llama_model_params {
            n_gpu_layers: value.n_gpu_layers as i32,
            split_mode: value.split_mode.into(),
            main_gpu: value.main_gpu as i32,
            tensor_split: ptr::null_mut(),
            progress_callback: None,
            progress_callback_user_data: ptr::null_mut(),
            kv_overrides: ptr::null_mut(),
            vocab_only: value.vocab_only,
            use_mmap: value.use_mmap,
            use_mlock: value.use_mlock,
        }
    }
}

/// Session-specific parameters.
pub struct SessionParams {
    /// RNG seed, [`u32::MAX`] for random (default)
    pub seed: u32,

    /// text context, 0 = from model
    pub n_ctx: u32,

    /// prompt processing maximum batch size
    pub n_batch: u32,

    /// number of threads to use for generation
    pub n_threads: u32,

    /// number of threads to use for batch processing
    pub n_threads_batch: u32,

    /// RoPE scaling type, from [`llama_rope_scaling_type`]
    pub rope_scaling_type: i32,

    /// ref: https://github.com/ggerganov/llama.cpp/pull/2054

    /// RoPE base frequency, 0 = from model
    pub rope_freq_base: f32,

    /// RoPE frequency scaling factor, 0 = from model
    pub rope_freq_scale: f32,

    /// YaRN extrapolation mix factor, negative = from model
    pub yarn_ext_factor: f32,

    /// YaRN magnitude scaling factor
    pub yarn_attn_factor: f32,

    /// YaRN low correction dim
    pub yarn_beta_fast: f32,

    /// YaRN high correction dim
    pub yarn_beta_slow: f32,

    /// YaRN original context size
    pub yarn_orig_ctx: u32,

    /// data type for K cache
    pub type_k: u32,

    /// data type for V cache
    pub type_v: u32,

    /// embedding mode only
    pub embedding: bool,

    /// whether to offload the KQV ops (including the KV cache) to GPU
    pub offload_kqv: bool,
}

impl Default for SessionParams {
    fn default() -> Self {
        let c_defaults = unsafe {
            // SAFETY: Stack constructor, always safe.
            llama_context_default_params()
        };

        let threads = num_cpus::get_physical() as u32 - 1;

        Self {
            seed: c_defaults.seed,
            n_ctx: c_defaults.n_ctx,
            n_batch: c_defaults.n_batch,
            n_threads: threads,
            n_threads_batch: threads,
            rope_scaling_type: c_defaults.rope_scaling_type,
            rope_freq_base: c_defaults.rope_freq_base,
            rope_freq_scale: c_defaults.rope_freq_scale,
            yarn_ext_factor: c_defaults.yarn_ext_factor,
            yarn_attn_factor: c_defaults.yarn_attn_factor,
            yarn_beta_fast: c_defaults.yarn_beta_fast,
            yarn_beta_slow: c_defaults.yarn_beta_slow,
            yarn_orig_ctx: c_defaults.yarn_orig_ctx,
            type_k: c_defaults.type_k as u32,
            type_v: c_defaults.type_v as u32,
            embedding: c_defaults.embedding,
            offload_kqv: c_defaults.offload_kqv,
        }
    }
}

impl From<SessionParams> for llama_context_params {
    fn from(value: SessionParams) -> Self {
        Self {
            seed: value.seed,
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            rope_scaling_type: value.rope_scaling_type,
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            yarn_orig_ctx: value.yarn_orig_ctx,
            cb_eval: None,
            cb_eval_user_data: ptr::null_mut(),
            type_k: value.type_k as ggml_type,
            type_v: value.type_v as ggml_type,
            mul_mat_q: true,   // Deprecated
            logits_all: false, // Deprecated
            embedding: value.embedding,
            offload_kqv: value.offload_kqv,
        }
    }
}

/// A safe wrapper around a [`llama_batch`].
struct Batch {
    // TODO
    /// ## Members
    /// * `n_tokens`: [`i32`] - The number of tokens
    /// * `tokens`: `*mut` [`llama_token`][llama_token] - The number of tokens
    /// * `embd`: `*mut` [`f32`] - The number of tokens
    /// * `pos`: `*mut` [`llama_pos`][llama_pos] - The number of tokens
    /// * `n_seq_id`: `*mut` [`i32`] - The number of tokens
    /// * `seq_id`: `*mut *mut` [`llama_seq_id`][llama_seq_id] - The number of tokens
    /// * `logits`: `*mut` [`i8`] - The number of tokens
    /// * `all_pos_0`: [`llama_pos`][llama_pos] - The number of tokens
    /// * `all_pos_1`: [`llama_pos`][llama_pos] - The number of tokens
    /// * `all_seq_id`: [`llama_seq_id`][llama_seq_id] - The number of tokens
    ///
    /// [llama_token]: llama_cpp_sys::llama_token
    /// [llama_seq_id]: llama_cpp_sys::llama_seq_id
    /// [llama_pos]: llama_cpp_sys::llama_pos
    inner: llama_batch,

    /// The maximum number of tokens this batch can have.
    capacity: usize,

    /// The maximum number of sequences that can be generated for this batch.
    max_sequences: usize,
}

impl Batch {
    fn new(capacity: usize, embed: usize, max_sequences: usize) -> Self {
        // Ideally panic shouldn't be used, but this struct is only used inside this crate, so it
        // should be fine.

        if capacity == 0 {
            panic!("Cannot create a batch with no capacity");
        }
        if max_sequences == 0 {
            panic!("At least one sequence must be generated");
        }

        Self {
            inner: unsafe { llama_batch_init(capacity as i32, embed as i32, max_sequences as i32) },
            capacity,
            max_sequences,
        }
    }

    fn clear(&mut self) {
        self.inner.n_tokens = 0;
    }

    fn add(&mut self, token: Token, position: usize, sequence_ids: &[i32], logits: bool) -> usize {
        trace!(
            "Writing token {} of {} ({token:?})",
            self.inner.n_tokens,
            self.capacity
        );

        let i = self.inner.n_tokens as usize;

        if i == self.capacity || self.max_sequences < sequence_ids.len() {
            return usize::MAX;
        }

        unsafe {
            // SAFETY: For all 0 < i < n_tokens, `llama_batch_init` created each of these
            // offsets; although each offset may be currently uninitialized.
            self.inner.token.add(i).write(token.0);
            self.inner.pos.add(i).write(position as i32);
            if logits {
                self.inner.logits.add(i).write(1);
            } else {
                self.inner.logits.add(i).write(0);
            }
            self.inner.n_seq_id.add(i).write(sequence_ids.len() as i32);

            let seq_ptr = *self.inner.seq_id.add(i);

            if !seq_ptr.is_null() {
                for id in 0..sequence_ids.len() {
                    seq_ptr.add(id).write(id as i32);
                }
            }
        }

        self.inner.n_tokens += 1;
        self.inner.n_tokens as usize - 1
    }

    fn set_logits(&self, idx: usize, value: bool) {
        assert!(idx < self.inner.n_tokens as usize, "Index out of bounds");

        unsafe {
            if value {
                self.inner.logits.add(idx).write(1);
            } else {
                self.inner.logits.add(idx).write(0);
            }
        }
    }

    fn tokens(&self) -> usize {
        self.inner.n_tokens as usize
    }

    fn handle(&self) -> llama_batch {
        self.inner
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        trace!("Freeing batch");

        unsafe { llama_batch_free(self.inner) }
    }
}

/// This needs to be documented!
pub trait Sampler {
    /// This needs to be documented!
    fn sample(&self, context: *mut llama_context, candidates_p: llama_token_data_array) -> Token;
}

mod detail {
    //! FFI implementation details.

    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]

    use std::ffi::{c_char, c_void, CStr};
    use std::ptr::slice_from_raw_parts;

    use tokio::sync::mpsc::UnboundedSender;
    use tracing::{error, info, trace, warn};

    use llama_cpp_sys::{
        ggml_log_level, ggml_log_level_GGML_LOG_LEVEL_ERROR, ggml_log_level_GGML_LOG_LEVEL_INFO,
        ggml_log_level_GGML_LOG_LEVEL_WARN, llama_beams_state,
    };

    use crate::Token;

    pub(crate) struct BeamSearchState {
        pub(crate) tx: UnboundedSender<Token>,
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

        if shared_state.tx.is_closed() {
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

        // TODO check if this happens due to some bug
        if text.len() < 2 {
            return;
        }

        let text = if let Some(stripped) = text.strip_suffix('\n') {
            stripped
        } else {
            text.as_ref()
        };

        match level {
            ggml_log_level_GGML_LOG_LEVEL_ERROR => error!("ggml: {text}"),
            ggml_log_level_GGML_LOG_LEVEL_INFO => info!("ggml: {text}"),
            ggml_log_level_GGML_LOG_LEVEL_WARN => warn!("ggml: {text}"),
            _ => trace!("ggml: {text}"),
        }
    }
}
