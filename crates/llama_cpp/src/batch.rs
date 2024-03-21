//! Implements the [`Batch`] struct

use llama_cpp_sys::{llama_batch, llama_batch_free, llama_batch_init};
use tracing::trace;

use crate::Token;

/// A safe wrapper around a [`llama_batch`].
pub struct Batch {
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
    pub fn new(capacity: usize, embed: usize, max_sequences: usize) -> Self {
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

    pub fn clear(&mut self) {
        self.inner.n_tokens = 0;
    }

    pub fn add(
        &mut self,
        token: Token,
        position: usize,
        sequence_ids: &[i32],
        logits: bool,
    ) -> usize {
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
                for (i, id) in sequence_ids.iter().enumerate() {
                    seq_ptr.add(i).write(*id);
                }
            }
        }

        self.inner.n_tokens += 1;
        self.inner.n_tokens as usize - 1
    }

    pub fn set_logits(&self, idx: usize, value: bool) {
        assert!(idx < self.inner.n_tokens as usize, "Index out of bounds");

        unsafe {
            if value {
                self.inner.logits.add(idx).write(1);
            } else {
                self.inner.logits.add(idx).write(0);
            }
        }
    }

    pub fn tokens(&self) -> usize {
        self.inner.n_tokens as usize
    }

    pub fn handle(&self) -> llama_batch {
        self.inner
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        trace!("Freeing batch");

        unsafe { llama_batch_free(self.inner) }
    }
}
