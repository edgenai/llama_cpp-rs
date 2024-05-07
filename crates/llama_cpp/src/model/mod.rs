//! Implements the [`LlamaModel`] struct

use std::borrow::Borrow;
use std::cmp::min;
use std::ffi::{c_char, CStr, CString};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::ptr::slice_from_raw_parts;
use std::sync::{atomic::AtomicUsize, Arc, Mutex, RwLock};
use std::usize;

use derive_more::{Deref, DerefMut};
use thiserror::Error;
use tracing::{error, info, trace, warn};

use backend::BackendRef;
use llama_cpp_sys::{
    ggml_row_size, llama_context, llama_context_params, llama_decode, llama_free_model,
    llama_get_embeddings_ith, llama_get_embeddings_seq, llama_kv_cache_clear,
    llama_load_model_from_file, llama_model, llama_model_meta_val_str, llama_n_ctx_train,
    llama_n_embd, llama_n_vocab, llama_new_context_with_model, llama_token, llama_token_bos,
    llama_token_eos, llama_token_eot, llama_token_get_text, llama_token_middle, llama_token_nl,
    llama_token_prefix, llama_token_suffix, llama_token_to_piece, llama_tokenize,
};
pub use params::*;

use crate::batch::Batch;
use crate::{
    LlamaContextError, LlamaContextInner, LlamaInternalError, LlamaSession, LlamaSessionInner,
    ResourceUsage, SessionParams, Token,
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
/// This is a thin wrapper over an `Arc<*mut llama_model>`, which is used to share the
/// model across threads.
#[derive(Deref, DerefMut)]
struct LlamaModelInner {
    #[deref]
    #[deref_mut]
    model: *mut llama_model,
    _backend_ref: BackendRef,
}

unsafe impl Send for LlamaModelInner {}

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
    model: Arc<Mutex<LlamaModelInner>>,

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

    /// The number of layers in the model's network.
    layers: usize,

    /// ???
    kv_heads: usize,
    /// Dimension of keys (d_k). d_q is assumed to be the same, but there are n_head q heads, and only n_head_kv k-v heads
    k_attention: usize,
    /// Dimension of values (d_v) aka n_embd_head
    v_attention: usize,

    /// State Space Models conv kernel
    ssm_d_conv: usize,
    /// State Space Models inner size
    ssm_d_inner: usize,
    /// State Space Models state size
    ssm_d_state: usize,
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
        let backend_ref = BackendRef::new();
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

            let n_embd = unsafe { llama_n_embd(model) } as usize;

            // Lots of redundant fetches here because llama.cpp doesn't expose any of this directly

            let heads = get_metadata(model, "%s.attention.head_count")
                .parse::<usize>()
                .unwrap_or(0);

            let layers = get_metadata(model, "%s.block_count")
                .parse::<usize>()
                .unwrap_or(0);
            let kv_heads = get_metadata(model, "%s.attention.head_count_kv")
                .parse::<usize>()
                .unwrap_or(heads);
            let k_attention = get_metadata(model, "%s.attention.key_length")
                .parse::<usize>()
                .unwrap_or(n_embd / heads);
            let v_attention = get_metadata(model, "%s.attention.value_length")
                .parse::<usize>()
                .unwrap_or(n_embd / heads);
            let ssm_d_conv = get_metadata(model, "%s.ssm.conv_kernel")
                .parse::<usize>()
                .unwrap_or(0);
            let ssm_d_inner = get_metadata(model, "%s.ssm.inner_size")
                .parse::<usize>()
                .unwrap_or(0);
            let ssm_d_state = get_metadata(model, "%s.ssm.state_size")
                .parse::<usize>()
                .unwrap_or(0);

            Ok(Self {
                model: Arc::new(Mutex::new(LlamaModelInner {
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
                embedding_length: n_embd,
                training_size: unsafe { llama_n_ctx_train(model) } as usize,
                layers,
                kv_heads,
                k_attention,
                v_attention,
                ssm_d_conv,
                ssm_d_inner,
                ssm_d_state,
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
    ///
    /// # Parameters
    ///
    /// * `content` - The data slice to be tokenized.
    /// * `add_bos` - Add the beginning of sentence token to the end of `content`.
    /// * `special` - Parse special tokens. If false, special tokens are parsed as if they were plain text.
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

        // With add_bos=true and the string "🦙", having less than `length + 2`
        // slots for tokens will incorrectly return a `LlamaInternalError`.
        let mut out_buf = Vec::with_capacity(content.len() + 2);

        let n_written_tokens = unsafe {
            let model_lock = self.model.lock().unwrap();

            // SAFETY: The pointer ranges specified here are always valid, and `n_written_tokens`
            // is always less than `content.len()`.
            //
            // `content.len()` always fits within an `i32`.
            //
            // `out_buf` is a `Vec<Token>`, and `Token` is `#[repr(transparent)]` over an `i32`.
            llama_tokenize(
                **model_lock,
                content.as_ptr() as *const c_char,
                content.len() as i32,
                out_buf.as_mut_ptr() as *mut llama_token,
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
            let model_lock = self.model.lock().unwrap();
            CStr::from_ptr(llama_token_get_text(**model_lock, token.0))
        }
        .to_bytes()
    }

    /// Converts the provided token into a `Vec<u8>` piece, using the model's vocabulary.
    ///
    /// Panics if the model is invalid.
    pub fn token_to_byte_piece(&self, token: Token) -> Vec<u8> {
        let initial_size = 8u16;
        let mut buffer = vec![0u8; usize::from(initial_size)];
        let model_lock = self.model.lock().unwrap();
        let size = unsafe {
            // SAFETY: Casting `*mut u8` to `*mut i8` is safe because `u8` and
            // `i8` have the same size and alignment.
            llama_token_to_piece(
                **model_lock,
                token.0,
                buffer.as_mut_ptr() as *mut c_char,
                std::os::raw::c_int::from(initial_size),
                false,
            )
        };

        buffer.resize(size.unsigned_abs() as usize, 0);
        if size < 0 {
            let size = unsafe {
                // SAFETY: Casting `*mut u8` to `*mut i8` is safe because `u8`
                // and `i8` have the same size and alignment. The length of
                // buffer is accurate for this reason.
                llama_token_to_piece(
                    **model_lock,
                    token.0,
                    buffer.as_mut_ptr() as *mut c_char,
                    std::os::raw::c_int::from(buffer.len() as i32),
                    false,
                )
            };
            assert_eq!(size as usize, buffer.len(), "Buffer length doesn't match");
        }

        buffer
    }

    /// Converts the provided token into a [`String`] piece, using the model's vocabulary.
    ///
    /// Note that this method cannot handle UTF-8 codepoints that are split into
    /// multiple tokens correctly. Therefore, this method should be avoided for
    /// decoding a sequence of tokens. Instead, use
    /// [`LlamaModel::decode_tokens`] or [`crate::TokensToStrings`].
    ///
    /// Panics if the model is invalid.
    pub fn token_to_piece(&self, token: Token) -> String {
        String::from_utf8_lossy(&self.token_to_byte_piece(token)).to_string()
    }

    /// Decodes a sequence of tokens into a [`String`].
    ///
    /// Panics if the model is invalid.
    pub fn decode_tokens(&self, tokens: impl IntoIterator<Item = impl Borrow<Token>>) -> String {
        let mut buf: Vec<u8> = vec![0; 1024];
        let mut i = 0;

        let mut tokens = tokens.into_iter();
        let mut token = tokens.next();

        while let Some(t) = token.as_ref().map(Borrow::borrow) {
            let token_buf = &mut buf[i..];

            let size = unsafe {
                let model_lock = self.model.lock().unwrap();

                // SAFETY: Casting `*mut u8` to `*mut i8` is safe because `u8` and
                // `i8` have the same size and alignment. The length of token_buf is
                // accurate for this reason.
                llama_token_to_piece(
                    **model_lock,
                    t.0,
                    token_buf.as_mut_ptr() as *mut c_char,
                    token_buf.len() as i32,
                    false,
                )
            };

            if size >= 0 {
                // There was enough space; continue to the next token.
                i += size as usize;
                token = tokens.next();
            } else {
                // There was not enough space; grow the buffer and try again.
                buf.resize(buf.len() + (-size) as usize + 1, 0);
                buf.resize(buf.capacity(), 0);
            }
        }

        buf.truncate(i);
        String::from_utf8_lossy(&buf).to_string()
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
            let model_lock = self.model.lock().unwrap();

            // SAFETY: due to `_model` being declared in the `LlamaContext`, `self` must live
            // for at least the lifetime of `LlamaContext`.
            llama_new_context_with_model(**model_lock, params)
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

    /// Calculates and returns an estimate of how much local memory a [`LlamaSession`] will take.
    ///
    /// At the moment, the value returned should always be more than the real value, possibly double.
    ///
    /// # Parameters
    ///
    /// * `session_params` - the parameters of the session to be created.
    pub fn estimate_session_size(&self, session_params: &SessionParams) -> ResourceUsage {
        let kv_size = session_params.n_ctx as i64; // TODO exception for mamba arch

        // dimension of key embeddings across all k-v heads
        let n_embd_k_gqa = self.k_attention * self.kv_heads;
        // dimension of value embeddings across all k-v heads
        let n_embd_v_gqa = self.v_attention * self.kv_heads;

        // dimension of the rolling state embeddings
        let n_embd_k_s = if self.ssm_d_conv > 0 {
            (self.ssm_d_conv - 1) * self.ssm_d_inner
        } else {
            0
        };
        // dimension of the recurrent state embeddings
        let n_embd_v_s = self.ssm_d_state * self.ssm_d_inner;

        let k_row_size = unsafe {
            ggml_row_size(
                session_params.type_k.into(),
                (n_embd_k_gqa + n_embd_k_s) as i64 * kv_size,
            )
        };
        let v_row_size = unsafe {
            ggml_row_size(
                session_params.type_v.into(),
                (n_embd_v_gqa + n_embd_v_s) as i64 * kv_size,
            )
        };

        let cache_size = self.layers * (k_row_size + v_row_size);
        trace!("KV cache size: {}MB", cache_size / 1024 / 1024);

        let batch = min(session_params.n_ctx, session_params.n_batch) as usize;
        let logits_size = self.vocabulary_size * batch;
        let embed_size = if session_params.embedding {
            self.embedding_length * batch
        } else {
            0
        };
        let output_size = (logits_size + embed_size) * size_of::<f32>();
        trace!("Output buffer size: {}MB", output_size / 1024 / 1024);

        // const LLAMA_MAX_NODES: usize = 8192;
        //
        // let compute_size = unsafe {
        //     ggml_tensor_overhead() * LLAMA_MAX_NODES
        //         + ggml_graph_overhead_custom(LLAMA_MAX_NODES, false)
        // };

        ResourceUsage {
            host_memory: output_size,
            // TODO while llama doesn't offer memory estimation utilities, this is the best that can be done realistically
            // https://github.com/ggerganov/llama.cpp/issues/4315
            device_memory: cache_size + output_size,
        }
    }

    /// Performs embeddings decoding on the given batch and returns the result.
    fn embeddings_decode(
        &self,
        context: *mut llama_context,
        batch: &Batch,
        token_counts: &[usize],
    ) -> Result<Vec<Vec<f32>>, LlamaContextError> {
        let res = unsafe {
            // clear previous kv_cache values (irrelevant for embeddings)
            llama_kv_cache_clear(context);
            llama_decode(context, batch.handle())
        };

        if res < 0 {
            return Err(LlamaContextError::DecodeFailed(res));
        }

        let mut out = Vec::with_capacity(token_counts.len());

        for (i, count) in token_counts.iter().enumerate() {
            let embedding = unsafe {
                let mut ptr = llama_get_embeddings_seq(context, i as i32);

                if ptr.is_null() {
                    ptr = llama_get_embeddings_ith(context, (count - 1) as i32);
                }

                if ptr.is_null() {
                    return Err(LlamaContextError::EmbeddingsFailed(
                        "Could not retrieve embeddings".to_string(),
                    ));
                }

                slice_from_raw_parts(ptr, self.embedding_length)
                    .as_ref()
                    .ok_or(LlamaContextError::EmbeddingsFailed(
                        "Could not parse embeddings".to_string(),
                    ))?
            };

            out.push(self.normalise_embedding(embedding)?)
        }

        Ok(out)
    }

    /// Normalise an embeddings vector.
    fn normalise_embedding(&self, embedding: &[f32]) -> Result<Vec<f32>, LlamaContextError> {
        let mut embed_vec = vec![0f32; self.embedding_length];
        let sum = embedding
            .iter()
            .map(move |x| x * x)
            .reduce(move |a, b| a + b)
            .ok_or(LlamaContextError::EmbeddingsFailed(
                "Could not normalise vector".to_string(),
            ))?;

        let norm = sum.sqrt();
        for (i, value) in embedding.iter().enumerate() {
            embed_vec[i] = value / norm;
        }

        Ok(embed_vec)
    }

    /// Runs embeddings inference for the given inputs vector, returning the result.
    fn embeddings_process(
        &self,
        inputs: Vec<Vec<Token>>,
        params: EmbeddingsParams,
    ) -> Result<Vec<Vec<f32>>, LlamaContextError> {
        let mut total_tokens = 0;
        let mut max_tokens = 0;
        let token_counts: Vec<usize> = inputs.iter().map(|v| v.len()).collect();
        for count in &token_counts {
            total_tokens += count;
            if max_tokens < *count {
                max_tokens = *count;
            }
        }

        let batch_capacity = if max_tokens > self.training_size {
            warn!("Large embedding input requires a context larger than the model's training context.");
            max_tokens
        } else {
            min(self.training_size, total_tokens)
        };
        let mut batch = Batch::new(batch_capacity, 0, 1);
        let mut out = Vec::with_capacity(inputs.len());

        let context_params = params.as_context_params(batch_capacity);
        let context = unsafe {
            let model_lock = self.model.lock().unwrap();

            // SAFETY: due to `_model` being declared in the `LlamaContext`, `self` must live
            // for at least the lifetime of `LlamaContext`.
            llama_new_context_with_model(**model_lock, context_params)
        };

        if context.is_null() {
            return Err(LlamaContextError::SessionFailed);
        }

        let mut batch_input_count = 0;
        let mut submitted = 0;
        for input in inputs {
            if batch.tokens() + input.len() > batch_capacity {
                trace!("Decoding {} embedding tokens", batch.tokens());
                out.append(&mut self.embeddings_decode(
                    context,
                    &batch,
                    &token_counts[submitted..batch_input_count],
                )?);
                batch.clear();
                submitted = batch_input_count;
                batch_input_count = 0;
            }

            trace!("Adding {} tokens to batch", input.len());
            for (i, token) in input.iter().enumerate() {
                batch.add(*token, i, &[batch_input_count as i32], false);
            }
            batch.set_logits(batch.tokens() - 1, true);
            batch_input_count += 1;
        }

        if 0 < batch_input_count {
            trace!("Decoding remaining {} embedding tokens", batch.tokens());
            out.append(&mut self.embeddings_decode(
                context,
                &batch,
                &token_counts[submitted..batch_input_count],
            )?);
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

    /// Return an estimation of how much memory embeddings generation is gonna require for the provided parameters and
    /// input tokens.
    pub fn estimate_embeddings_session_size(
        &self,
        inputs: &[Vec<Token>],
        params: &EmbeddingsParams,
    ) -> ResourceUsage {
        let mut total_tokens = 0;
        let mut max_tokens = 0;
        for tokens in inputs {
            total_tokens += tokens.len();
            if max_tokens < tokens.len() {
                max_tokens = tokens.len();
            }
        }

        let batch_capacity = if max_tokens > self.training_size {
            warn!("Large embedding input requires a context larger than the model's training context.");
            max_tokens
        } else {
            min(self.training_size, total_tokens)
        };

        let context_params = params.as_context_params(batch_capacity);

        let mut ret = self.estimate_session_size(&context_params.into());
        ret.device_memory += ret.device_memory / 4; // bad workaround for device memory, see estimate_session_size
        ret
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

    /// Return the number of layers of the model.
    pub fn layers(&self) -> usize {
        self.layers
    }
}

/// Retrieves a value in string form from a model's metadata.
///
/// # Parameters
///
/// * `model` - a pointer to the model to retrieve values from.
/// * `key` - the key of the metadata value.
///
/// #  Limitations
///
/// At the moment, the implementation will retrieves values of limited length, so this shouldn't be used to retrieve
/// something like the model's grammar.
fn get_metadata(model: *mut llama_model, key: &str) -> String {
    let c_key = if let Some(stripped) = key.strip_prefix("%s") {
        let arch_key = CStr::from_bytes_with_nul(b"general.architecture\0").unwrap(); // Should never fail
        let mut arch_val = vec![0u8; 128];

        let res = unsafe {
            llama_model_meta_val_str(
                model,
                arch_key.as_ptr(),
                arch_val.as_mut_ptr() as *mut c_char,
                arch_val.len(),
            )
        };

        if let Ok(len) = usize::try_from(res) {
            if let Ok(c_str) = CStr::from_bytes_with_nul(&arch_val[..=len]) {
                let formatted = format!("{}{stripped}", c_str.to_string_lossy());
                CString::new(formatted.as_bytes()).unwrap()
            } else {
                // This should be unreachable
                error!("Could not parse architecture metadata");
                return String::new();
            }
        } else {
            // This should be unreachable
            error!("Could not find architecture metadata");
            return String::new();
        }
    } else {
        CString::new(key).unwrap()
    };

    // This implementation assumes large values such as the model's vocabulary will never be queried
    let mut val = vec![0u8; 128];
    let res = unsafe {
        llama_model_meta_val_str(
            model,
            c_key.as_ptr(),
            val.as_mut_ptr() as *mut c_char,
            val.len(),
        )
    };

    if let Ok(len) = usize::try_from(res) {
        if let Ok(val_str) = CStr::from_bytes_with_nul(&val[..=len])
            .map(move |val| val.to_string_lossy().to_string())
        {
            val_str
        } else {
            error!("Failed to parse retrieved metadata");
            String::new()
        }
    } else {
        warn!(key, "Could not find metadata");
        String::new()
    }
}
