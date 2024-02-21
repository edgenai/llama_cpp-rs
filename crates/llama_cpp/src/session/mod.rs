//! Functionality for the [`LlamaSession`] struct

use std::cmp::min;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

use futures::executor::block_on;
use thiserror::Error;
use tokio::sync::{
    mpsc::{unbounded_channel, UnboundedReceiver},
    Mutex, RwLock
};
use tracing::{error, info, trace, warn};

use llama_cpp_sys::{
    llama_beam_search, llama_context, llama_copy_state_data, llama_decode, llama_free,
    llama_get_logits_ith, llama_get_state_size, llama_kv_cache_seq_rm, llama_set_state_data,
    llama_token_data, llama_token_data_array,
};

use crate::{detail, LlamaModel, LlamaTokenizationError, Sampler, Token};

mod batch;
mod params;

use batch::Batch;
pub use params::*;

/// The inner part of a [`LlamaSession`].
///
/// This is wrapped in an `Arc` for sharing across thread boundaries.
pub(crate) struct LlamaContextInner {
    /// A pointer to the inner context.
    pub(crate) ptr: *mut llama_context,
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
                // SAFETY: `llama_decode` will not fail for a valid `batch`, which we correctly
                // initialized above.
                llama_decode(block_on(self.inner.ctx.lock()).ptr, batch.handle())
            };
            if err != 0 {
                return Err(LlamaContextError::DecodeFailed(err));
            }
            trace!("Batch decode completed successfully");

            last_batch_size = sequence.len();
        }

        block_on(self.inner.tokens.write()).extend_from_slice(tokens);

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

        tokio::task::spawn_blocking(move || session.advance_context(ctx))
            .await
            .unwrap()
    }

    /// Starts generating tokens at the end of the context using llama.cpp's built-in Beam search.
    /// TODO fix: beam search keeps going even after it should have ended
    pub fn start_completing(&mut self) -> CompletionHandle {
        let (tx, rx) = unbounded_channel();
        let history_size = self.context_size();
        let session = self.clone();

        info!("Generating completions with {history_size} tokens of history");

        thread::spawn(move || unsafe {
            let state = Box::new(detail::BeamSearchState { tx });
            // SAFETY: `state_ptr` is converted back to a [`Box`] and freed in [`detail::llama_beam_search_callback`]
            let state_ptr = Box::into_raw(state);

            llama_beam_search(
                block_on(session.inner.ctx.lock()).ptr,
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
        let history_size = self.context_size();
        let session = self.clone();
        // TODO deal with 0 history size
        info!("Generating completions with {history_size} tokens of history");

        thread::spawn(move || {
            let context = block_on(session.inner.ctx.lock());
            let vocab = session.model().vocabulary_size();
            let end_of_stream = session.model().eos();
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
                let mut token_buf = block_on(session.inner.tokens.write());
                current_pos = token_buf.len();
                token_buf.push(token);
            }
        });

        CompletionHandle { rx }
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
        block_on(self.inner.tokens.read()).len()
    }

    /// Returns the list of tokens in the current context
    pub fn context(&self) -> Vec<Token> {
        block_on(self.inner.tokens.read()).clone()
    }

    /// Removes all but the first `n_tokens` tokens from the context
    pub fn truncate_context(&self, n_tokens: usize) {
        if n_tokens > self.context_size() {
            return;
        }

        let context = block_on(self.inner.ctx.lock());

        unsafe {
            llama_kv_cache_seq_rm(
                context.ptr,
                -1,              // Match all sequences
                n_tokens as i32, // Delete starting at n_tokens
                -1,              // Delete ending at end of context
            )
        }

        block_on(self.inner.tokens.write()).truncate(n_tokens)
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
        let old_tokens = block_on(self.inner.tokens.read());

        let shared_prefix = old_tokens
            .iter()
            .zip(new_tokens)
            .position(|(t1, t2)| t1 != t2)
            .unwrap_or(new_tokens.len().min(old_tokens.len()));

        std::mem::drop(old_tokens);

        self.truncate_context(shared_prefix);
        self.advance_context_with_tokens(&new_tokens[shared_prefix..])
    }

    /// Sets this session's context to the tokenized version of the provided bytes. See
    /// [`LlamaSession::set_context_to_tokens`] for more information.
    pub fn set_context(&mut self, ctx: impl AsRef<[u8]>) -> Result<(), LlamaContextError> {
        let tokens = self
            .inner
            .model
            .tokenize_bytes(ctx.as_ref())?
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
        let ctx = self.inner.ctx.blocking_lock();

        #[allow(unused_mut)]
        let mut copy = self.model().create_session(self.inner.params.clone())?;

        let size = unsafe { llama_get_state_size(ctx.ptr) };
        let mut buf = vec![0; size];

        // SAFETY: `llama_copy_state_data` and `llama_set_state_data` should never write/read more than
        // `llama_get_state_size` bytes, so `buf` should be big enough.
        //
        // `copy` was created from the same model as `self` and with the same parameters.
        unsafe {
            let copy_size = llama_copy_state_data(ctx.ptr, buf.as_mut_ptr());
            assert!(copy_size <= size);
            let set_size = llama_set_state_data(copy.inner.ctx.blocking_lock().ptr, buf.as_mut_ptr());
            assert_eq!(copy_size, set_size);
        }

        // NOTE: Any changes to the fields of a LlamaSession may require that
        // those changes are mirrored here
        *block_on(copy.inner.tokens.write()) = block_on(self.inner.tokens.read()).clone();
        copy.inner.last_batch_size.store(
            self.inner.last_batch_size.load(Ordering::SeqCst),
            Ordering::SeqCst,
        );

        Ok(copy)
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
