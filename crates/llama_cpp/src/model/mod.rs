//! Implements the [`LlamaModel`] struct

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{atomic::AtomicUsize, Arc};

use derive_more::{Deref, DerefMut};
use futures::executor::block_on;
use thiserror::Error;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tracing::info;

use llama_cpp_sys::{
    llama_context_params, llama_free_model, llama_load_model_from_file, llama_model, llama_n_vocab,
    llama_new_context_with_model, llama_token_bos, llama_token_eos, llama_token_eot,
    llama_token_get_text, llama_token_middle, llama_token_nl, llama_token_prefix,
    llama_token_suffix, llama_token_to_piece, llama_tokenize,
};

use crate::{
    LlamaContextError, LlamaContextInner, LlamaInternalError, LlamaSession, LlamaSessionInner,
    SessionParams, Token,
};

mod backend;
mod params;

use backend::BackendRef;
pub use params::*;

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
                tokens: RwLock::new(Vec::new()),
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

    /// Returns the number of possible values a [`Token`] can have for this model.
    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }
}
