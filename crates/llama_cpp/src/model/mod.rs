//! Implements the [`LlamaModel`] struct

use std::cmp::min;
use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};
use std::ptr::slice_from_raw_parts;
use std::sync::{atomic::AtomicUsize, Arc};

use derive_more::{Deref, DerefMut};
use futures::executor::block_on;
use thiserror::Error;
use tokio::sync::Mutex;
use tokio::sync::RwLock;
use tracing::info;

use backend::BackendRef;
use llama_cpp_sys::{
    llama_context, llama_context_default_params, llama_context_params, llama_decode,
    llama_free_model, llama_get_embeddings_ith, llama_kv_cache_clear, llama_load_model_from_file,
    llama_model, llama_n_ctx_train, llama_n_embd, llama_n_vocab, llama_new_context_with_model,
    llama_token_bos, llama_token_eos, llama_token_eot, llama_token_get_text, llama_token_middle,
    llama_token_nl, llama_token_prefix, llama_token_suffix, llama_token_to_piece, llama_tokenize,
};
pub use params::*;

use crate::batch::Batch;
use crate::{
    LlamaContextError, LlamaContextInner, LlamaInternalError, LlamaSession, LlamaSessionInner,
    SessionParams, Token,
};

mod backend;
mod params;

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

    /// For embeddings, the length of a single embeddings vector.
    embedding_length: usize,

    /// The number of tokens in the context the model was trained with.
    training_size: usize,
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
                embedding_length: unsafe { llama_n_embd(model) } as usize,
                training_size: unsafe { llama_n_ctx_train(model) } as usize,
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
        add_bos: bool,
        special: bool,
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
                add_bos,
                special,
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

    /// Calls [`LlamaModel::tokenize_bytes`] for each element of the provided slice and returns the resulting vector.
    pub fn tokenize_slice(
        &self,
        slice: &[impl AsRef<[u8]>],
        add_bos: bool,
        special: bool,
    ) -> Result<Vec<Vec<Token>>, LlamaTokenizationError> {
        let mut out = Vec::with_capacity(slice.len());
        let iter = slice
            .iter()
            .map(move |prompt| self.tokenize_bytes(prompt, add_bos, special));

        for item in iter {
            out.push(item?)
        }

        Ok(out)
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
        let params = llama_context_params::from(session_params.clone());
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
                params: session_params,
            }),
        })
    }

    /// Performs embeddings decoding on the given batch and returns the result.
    fn embeddings_decode(
        &self,
        context: *mut llama_context,
        batch: &Batch,
        input_count: usize,
    ) -> Result<Vec<Vec<f32>>, LlamaContextError> {
        let res = unsafe {
            // clear previous kv_cache values (irrelevant for embeddings)
            llama_kv_cache_clear(context);
            llama_decode(context, batch.handle())
        };

        if res < 0 {
            return Err(LlamaContextError::DecodeFailed(res));
        }

        let mut out = Vec::with_capacity(input_count);

        for i in 0..input_count {
            let embedding = unsafe {
                let ptr = llama_get_embeddings_ith(context, i as i32);
                slice_from_raw_parts(ptr, self.embedding_length)
                    .as_ref()
                    .ok_or(LlamaContextError::DecodeFailed(1))?
            };

            // normalize the embedding
            let mut embed_vec = vec![0f32; self.embedding_length];
            let sum = embedding
                .iter()
                .map(move |x| x * x)
                .reduce(move |a, b| a + b)
                .ok_or(LlamaContextError::DecodeFailed(2))?;

            let norm = sum.sqrt();
            for (i, value) in embedding.iter().enumerate() {
                embed_vec[i] = value / norm;
            }

            out.push(embed_vec)
        }

        Ok(out)
    }

    /// Runs embeddings inference for the given inputs vector, returning the result.
    fn embeddings_process(
        &self,
        inputs: Vec<Vec<Token>>,
        params: EmbeddingsParams,
    ) -> Result<Vec<Vec<f32>>, LlamaContextError> {
        let mut total_tokens = 0;
        let mut max_tokens = 0;
        for tokens in &inputs {
            total_tokens += tokens.len();
            if max_tokens < tokens.len() {
                max_tokens = tokens.len();
            }
        }

        let batch_capacity = min(self.training_size, total_tokens);
        let mut batch = Batch::new(batch_capacity, 0, inputs.len());
        let mut out = Vec::with_capacity(inputs.len());

        let context = unsafe {
            // SAFETY: Stack constructor, always safe.
            let mut ctx_params = llama_context_default_params();
            ctx_params.embedding = true;
            ctx_params.n_threads = params.n_threads;
            ctx_params.n_threads_batch = params.n_threads_batch;
            // SAFETY: due to `_model` being declared in the `LlamaContext`, `self` must live
            // for at least the lifetime of `LlamaContext`.
            llama_new_context_with_model(**self.model.try_read().unwrap(), ctx_params)
        };

        if context.is_null() {
            return Err(LlamaContextError::SessionFailed);
        }

        let mut batch_input_count = 0;
        for input in inputs {
            if batch.tokens() + input.len() > batch_capacity {
                out.append(&mut self.embeddings_decode(context, &batch, batch_input_count)?);
                batch.clear();
                batch_input_count = 0;
            }

            for (i, token) in input.iter().enumerate() {
                batch.add(*token, i, &[batch_input_count as i32], false);
            }
            batch_input_count += 1;
        }

        if 0 < batch_input_count {
            out.append(&mut self.embeddings_decode(context, &batch, batch_input_count)?);
        }

        Ok(out)
    }

    /// Runs embeddings inference for the given inputs, returning the result.
    pub fn embeddings(
        &self,
        inputs: &[impl AsRef<[u8]>],
        params: EmbeddingsParams,
    ) -> Result<Vec<Vec<f32>>, LlamaContextError> {
        let inputs = self.tokenize_slice(inputs, true, false)?;
        self.embeddings_process(inputs, params)
    }

    /// Runs embeddings inference for the given inputs, returning the result.
    ///
    /// This is a thin `tokio::spawn_blocking` wrapper around
    /// [`LlamaModel::embeddings`].
    pub async fn embeddings_async(
        &self,
        inputs: &[impl AsRef<[u8]>],
        params: EmbeddingsParams,
    ) -> Result<Vec<Vec<f32>>, LlamaContextError> {
        let inputs = self.tokenize_slice(inputs, true, false)?;
        let model = self.clone();

        tokio::task::spawn_blocking(move || model.embeddings_process(inputs, params))
            .await
            .unwrap()
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

    /// Returns the length of a single embedding vector for this model.
    pub fn embed_len(&self) -> usize {
        self.embedding_length
    }

    /// Returns the number of tokens in the context the model was trained with.
    pub fn train_len(&self) -> usize {
        self.training_size
    }
}

/// Embeddings inference specific parameters.
pub struct EmbeddingsParams {
    /// number of threads to use for generation
    pub n_threads: u32,

    /// number of threads to use for batch processing
    pub n_threads_batch: u32,
}
