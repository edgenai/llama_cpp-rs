//! Functionality for the [`LlamaSession`] struct

use derive_more::{Deref, DerefMut};
use std::cmp::min;
use std::ops::{Bound, RangeBounds};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;

use thiserror::Error;
use tokio::sync::mpsc::unbounded_channel;
use tracing::{error, info, trace, warn};

use llama_cpp_sys::{
    llama_context, llama_copy_state_data, llama_decode, llama_free, llama_get_logits_ith,
    llama_get_state_size, llama_kv_cache_seq_rm, llama_set_state_data, llama_token_data,
    llama_token_data_array,
};

use crate::standard_sampler::StandardSampler;
use crate::{LlamaModel, LlamaTokenizationError, Sampler, Token};

mod completion;
mod params;

use crate::batch::Batch;
pub use completion::CompletionHandle;
pub use completion::*;
pub use params::*;

/// The inner part of a [`LlamaSession`].
///
/// This is wrapped in an `Arc` for sharing across thread boundaries.
#[derive(Deref, DerefMut)]
pub(crate) struct LlamaContextInner {
    /// A pointer to the inner context.
    pub(crate) ptr: *mut llama_context,
}

unsafe impl Send for LlamaContextInner {}

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
    pub(crate) inner: Arc<LlamaSessionInner>,
}

/// The cloned part of a [`LlamaSession`].
// NOTE: Changes made here may need to be reflected in LlamaSession::deep_copy
pub(crate) struct LlamaSessionInner {
    /// The model this session was created from.
    pub(crate) model: LlamaModel,

    /// A pointer to the llama.cpp side of the model context.
    pub(crate) ctx: Mutex<LlamaContextInner>,

    /// The list of tokens within the current context
    pub(crate) tokens: RwLock<Vec<Token>>,

    /// The number of tokens present in this model's context.
    pub(crate) last_batch_size: AtomicUsize,

    /// Max batch size.
    pub(crate) max_batch: u32,

    /// The parameters this session was created with
    pub(crate) params: SessionParams,
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

    /// An error occurred on the other side of the FFI boundary; check your logs.
    #[error("failed to create llama context")]
    SessionFailed,

    /// An error occurred on the other side of the FFI boundary; check your logs.
    #[error("advancing context failed (error code {0})")]
    DecodeFailed(i32),

    /// An error occurred on the other side of the FFI boundary; check your logs.
    #[error("failed to process embeddings (reason: {0})")]
    EmbeddingsFailed(String),

    /// An error occurred operating over kv cache due to invalid range.
    #[error("failed to operate over kv cache due to invalid range")]
    InvalidRange,

    /// Tried to start completing before advancing the context.
    #[error("cannot start completing without any history")]
    NoContext,
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
            return Ok(());
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
        let history_size = self.context_size();
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
                let session_guard = self.inner.ctx.lock().unwrap();

                // SAFETY: `llama_decode` will not fail for a valid `batch`, which we correctly
                // initialized above.
                llama_decode(**session_guard, batch.handle())
            };
            if err != 0 {
                return Err(LlamaContextError::DecodeFailed(err));
            }
            trace!("Batch decode completed successfully");

            last_batch_size = sequence.len();
        }

        self.inner.tokens.write().unwrap().extend_from_slice(tokens);

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
            .tokenize_bytes(ctx.as_ref(), false, true)?
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

        tokio::task::spawn_blocking(move || session.advance_context(ctx))
            .await
            .unwrap()
    }

    /// Starts generating tokens at the end of the context using a greedy
    /// sampler
    pub fn start_completing(&mut self) -> Result<CompletionHandle, LlamaContextError> {
        self.start_completing_with(
            StandardSampler::new_greedy(),
            self.params().n_ctx as usize - self.context_size(),
        )
    }

    /// Start completion.
    pub fn start_completing_with<S>(
        &mut self,
        mut sampler: S,
        max_predictions: usize,
    ) -> Result<CompletionHandle, LlamaContextError>
    where
        S: Sampler + Send + Sync + 'static,
    {
        let history_size = self.context_size();

        if history_size == 0 {
            return Err(LlamaContextError::NoContext);
        }

        let (tx, rx) = unbounded_channel();
        let session = self.clone();

        info!("Generating completions with {history_size} tokens of history");

        thread::spawn(move || {
            let context = session.inner.ctx.lock().unwrap();
            let vocab = session.model().vocabulary_size();
            let end_of_stream = session.model().eos();
            let mut token_buf = session.inner.tokens.write().unwrap();
            let mut batch = Batch::new(1, 0, 1);
            let mut current_pos = history_size;

            // There are no logits; we need to send the last token back through
            // the model
            if session.inner.last_batch_size.load(Ordering::SeqCst) == 0 {
                // Remove last token
                unsafe {
                    llama_kv_cache_seq_rm(**context, -1, token_buf.len() as i32 - 1, -1);
                }

                // Decode last token
                batch.add(*token_buf.last().unwrap(), current_pos, &[0], true);
                let res = unsafe { llama_decode(**context, batch.handle()) };

                if res != 0 {
                    error!("Failed to decode context ({res})");
                    return;
                }

                // Update state with new batch
                session
                    .inner
                    .last_batch_size
                    .store(batch.tokens(), Ordering::SeqCst);
                batch.clear();
            }

            loop {
                // Get logit values from the model and store them in a `llama_token_data_array`
                let mut candidates: Vec<llama_token_data> = {
                    let i = session.inner.last_batch_size.load(Ordering::SeqCst);
                    let logits = unsafe { llama_get_logits_ith(**context, (i - 1) as i32) };
                    let logits = unsafe { std::slice::from_raw_parts(logits, vocab) };

                    logits
                        .iter()
                        .enumerate()
                        .map(|(id, &logit)| llama_token_data {
                            id: id as i32,
                            logit,
                            p: 0.0,
                        })
                        .collect()
                };

                let candidates_p = llama_token_data_array {
                    data: candidates.as_mut_ptr(),
                    size: vocab,
                    sorted: false,
                };

                // Select the next token
                let token = sampler.sample(**context, &token_buf, candidates_p);

                // Send the token to the `CompletionHandle`, exiting on failure
                if let Err(e) = tx.send(token) {
                    let token_str = String::from_utf8_lossy(session.inner.model.detokenize(e.0));
                    warn!("Cannot send token ({}): {}", token_str, e);
                    return;
                }

                // Exit if eos is generated or maximum number of predictions is reached
                if token == end_of_stream || token_buf.len() - history_size >= max_predictions {
                    return;
                }

                // Create a batch with the generated token and decode it
                batch.add(token, current_pos, &[0], true);
                let res = unsafe { llama_decode(**context, batch.handle()) };

                if res != 0 {
                    error!("Failed to decode context ({res})");
                    return;
                }

                // Update state with new token/batch
                session
                    .inner
                    .last_batch_size
                    .store(batch.tokens(), Ordering::SeqCst);
                current_pos = token_buf.len();
                token_buf.push(token);
                batch.clear();
            }
        });

        Ok(CompletionHandle {
            rx,
            model: self.model(),
        })
    }

    /// Returns the model this session was created from.
    pub fn model(&self) -> LlamaModel {
        self.inner.model.clone()
    }

    /// Returns the parameters this session was created with.
    pub fn params(&self) -> &SessionParams {
        &self.inner.params
    }

    /// Returns the number of tokens currently in this session's context
    pub fn context_size(&self) -> usize {
        self.inner.tokens.read().unwrap().len()
    }

    /// Returns the list of tokens in the current context
    pub fn context(&self) -> Vec<Token> {
        self.inner.tokens.read().unwrap().clone()
    }

    /// Removes all tokens within the given range without performing any prompt
    /// processing. If you remove tokens in the middle of context, it is recommended that you keep
    /// the first ~4 tokens of context, per <https://arxiv.org/abs/2309.17453>.
    ///
    /// Note that calling this is not equivalent to calling [`LlamaSession::set_context`] with the
    /// same list of tokens that this method produces.
    pub fn remove_tokens_in_range(
        &mut self,
        range: impl RangeBounds<usize>,
    ) -> Result<(), LlamaContextError> {
        let start_bound = match range.start_bound() {
            Bound::Included(i) => *i as i32,
            Bound::Excluded(i) => *i as i32 + 1,
            Bound::Unbounded => -1,
        };

        let end_bound = match range.end_bound() {
            Bound::Included(i) => *i as i32 + 1,
            Bound::Excluded(i) => *i as i32,
            Bound::Unbounded => -1,
        };

        // -1 here to match all sequences
        let success = unsafe {
            let context = self.inner.ctx.lock().unwrap();

            llama_kv_cache_seq_rm(**context, -1, start_bound, end_bound)
        };

        if !success {
            return Err(LlamaContextError::InvalidRange);
        }

        // If we delete to the end, store 0 to indicate that there are no logits
        if end_bound == -1 || end_bound as usize >= self.context_size() {
            self.inner.last_batch_size.store(0, Ordering::SeqCst);
        }

        self.inner.tokens.write().unwrap().drain(range);

        Ok(())
    }

    /// Removes all but the first `n_tokens` tokens from the context.
    pub fn truncate_context(&mut self, n_tokens: usize) -> Result<(), LlamaContextError> {
        self.remove_tokens_in_range(n_tokens..)
    }

    /// Sets this session's context to the tokens provided.
    ///
    /// This method is more efficient than creating a new session and advancing it, because it only
    /// has to decode the tokens not already in the prefix of the previous context.
    pub fn set_context_to_tokens(
        &mut self,
        new_tokens: impl AsRef<[Token]>,
    ) -> Result<(), LlamaContextError> {
        let new_tokens = new_tokens.as_ref();
        let old_tokens = self.inner.tokens.read().unwrap();

        let shared_prefix = old_tokens
            .iter()
            .zip(new_tokens)
            .position(|(t1, t2)| t1 != t2)
            .unwrap_or(new_tokens.len().min(old_tokens.len()));

        std::mem::drop(old_tokens);

        self.truncate_context(shared_prefix)?;
        self.advance_context_with_tokens(&new_tokens[shared_prefix..])
    }

    /// Sets this session's context to the tokenized version of the provided bytes. See
    /// [`LlamaSession::set_context_to_tokens`] for more information.
    pub fn set_context(&mut self, ctx: impl AsRef<[u8]>) -> Result<(), LlamaContextError> {
        let tokens = self
            .inner
            .model
            .tokenize_bytes(ctx.as_ref(), false, false)?
            .into_boxed_slice();

        self.set_context_to_tokens(tokens)
    }

    /// Sets this session's context to the tokens provided.
    ///
    /// This is a thin `tokio::spawn_blocking` wrapper around
    /// [`LlamaSession::set_context_to_tokens`].
    pub async fn set_context_to_tokens_async(
        &mut self,
        tokens: impl AsRef<[Token]>,
    ) -> Result<(), LlamaContextError> {
        let tokens = tokens.as_ref().to_owned();
        let mut session = self.clone();

        tokio::task::spawn_blocking(move || session.set_context_to_tokens(tokens))
            .await
            .unwrap()
    }

    /// Sets this session's context to the tokenized version of the provided bytes. See
    /// [`LlamaSession::set_context_to_tokens`] for more information.
    ///
    /// This is a thin `tokio::spawn_blocking` wrapper around
    /// [`LlamaSession::set_context_to_tokens`].
    pub async fn set_context_async(
        &mut self,
        ctx: impl AsRef<[u8]>,
    ) -> Result<(), LlamaContextError> {
        let ctx = ctx.as_ref().to_owned();
        let mut session = self.clone();

        tokio::task::spawn_blocking(move || session.set_context(ctx))
            .await
            .unwrap()
    }

    /// Creates a new [`LlamaSession`] with the same contents as `self`. The returned
    /// [`LlamaSession`] can be used and modified independently from `self`.
    ///
    /// This differs from [`LlamaSession::clone`] in that [`LlamaSession::clone`] creates a new
    /// reference to the same underlying [`LlamaSession`].
    pub fn deep_copy(&self) -> Result<LlamaSession, LlamaContextError> {
        let ctx = self.inner.ctx.lock().unwrap();

        #[allow(unused_mut)]
        let mut copy = self.model().create_session(self.inner.params.clone())?;

        let size = unsafe { llama_get_state_size(**ctx) };
        let mut buf = vec![0; size];

        // SAFETY: `llama_copy_state_data` and `llama_set_state_data` should never write/read more than
        // `llama_get_state_size` bytes, so `buf` should be big enough.
        //
        // `copy` was created from the same model as `self` and with the same parameters.
        unsafe {
            let copy_size = llama_copy_state_data(**ctx, buf.as_mut_ptr());
            assert!(copy_size <= size);
            let copy_guard = copy.inner.ctx.lock().unwrap();
            let set_size = llama_set_state_data(**copy_guard, buf.as_mut_ptr());
            assert_eq!(copy_size, set_size);
        }

        // NOTE: Any changes to the fields of a LlamaSession may require that
        // those changes are mirrored here
        *copy.inner.tokens.write().unwrap() = self.inner.tokens.read().unwrap().clone();
        copy.inner.last_batch_size.store(
            self.inner.last_batch_size.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );

        Ok(copy)
    }

    /// Returns the maximum size in bytes this session is occupying in host memory.
    ///
    /// Currently there is no way to check the amount of memory occupied in devices.
    pub fn memory_size(&self) -> usize {
        let ctx = self.inner.ctx.lock().unwrap();
        unsafe { llama_get_state_size(**ctx) }
    }
}
