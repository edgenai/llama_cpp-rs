use std::borrow::Borrow;
use std::ffi::{c_int, CString};
use std::path::{Path, PathBuf};
use std::ptr::addr_of_mut;

use derive_more::{Deref, DerefMut};
use thiserror::Error;
use tracing::{error, info};

use llama_cpp_sys::{
    clip_ctx, clip_free, clip_model_load, llama_n_batch, llava_eval_image_embed,
    llava_image_embed_free, llava_image_embed_make_with_bytes,
};

use crate::{
    LlamaContextError, LlamaLoadError, LlamaModel, LlamaParams, LlamaSession,
    LlamaTokenizationError, SessionParams, Token,
};

/// An error raised while advancing the context in a [`LlavaSession`].
#[derive(Error, Debug)]
pub enum LlavaContextError {
    /// Something went wrong in a call to a LlamaContextError method.
    #[error(transparent)]
    Llama(#[from] LlamaContextError),

    /// Failed to create image embedding.
    #[error("Failed to create image embedding")]
    Embed,

    /// Failed to decode an image.
    #[error("Image decode failed")]
    Decode,
}

/// The inner Clip model, used in multimodal models.
#[derive(Deref, DerefMut)]
struct ClipInner {
    clip: *mut clip_ctx,
}

unsafe impl Send for ClipInner {}

impl Drop for ClipInner {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: `drop`ping more than once is unsound [1], so `self.clip` cannot have been
            // `free`d yet.
            //
            // [1]: See https://github.com/rust-lang/rust/issues/60977
            clip_free(self.clip);
        }
    }
}

pub struct LlavaModel {
    model: LlamaModel,
    clip_path: PathBuf,
}

impl LlavaModel {
    pub fn load_from_file(
        file_path: impl AsRef<Path>,
        model_params: LlamaParams,
        mmproj_path: impl AsRef<Path>,
    ) -> Result<Self, LlamaLoadError> {
        Ok(Self {
            model: LlamaModel::load_from_file(file_path, model_params)?,
            clip_path: mmproj_path.as_ref().to_path_buf(),
        })
    }

    pub async fn load_from_file_async(
        file_path: impl AsRef<Path>,
        params: LlamaParams,
        mmproj_path: impl AsRef<Path>,
    ) -> Result<Self, LlamaLoadError> {
        Ok(Self {
            model: LlamaModel::load_from_file_async(file_path, params).await?,
            clip_path: mmproj_path.as_ref().to_path_buf(),
        })
    }

    pub fn tokenize_bytes(
        &self,
        content: impl AsRef<[u8]>,
        add_bos: bool,
        special: bool,
    ) -> Result<Vec<Token>, LlamaTokenizationError> {
        self.model.tokenize_bytes(content, add_bos, special)
    }

    pub fn tokenize_slice(
        &self,
        slice: &[impl AsRef<[u8]>],
        add_bos: bool,
        special: bool,
    ) -> Result<Vec<Vec<Token>>, LlamaTokenizationError> {
        self.model.tokenize_slice(slice, add_bos, special)
    }

    pub fn detokenize(&self, token: Token) -> &[u8] {
        self.model.detokenize(token)
    }

    pub fn token_to_byte_piece(&self, token: Token) -> Vec<u8> {
        self.model.token_to_byte_piece(token)
    }

    pub fn token_to_piece(&self, token: Token) -> String {
        self.model.token_to_piece(token)
    }

    pub fn decode_tokens(&self, tokens: impl IntoIterator<Item = impl Borrow<Token>>) -> String {
        self.model.decode_tokens(tokens)
    }

    pub fn create_session(
        &self,
        session_params: SessionParams,
    ) -> Result<LlavaSession, LlamaContextError> {
        let path_str = self.clip_path.to_string_lossy();
        let path_str = CString::new(path_str.as_bytes())?;
        let clip = unsafe { clip_model_load(path_str.as_ptr(), 1) };

        Ok(LlavaSession {
            session: self.model.create_session(session_params)?,
            clip: ClipInner { clip },
        })
    }

    pub fn bos(&self) -> Token {
        self.model.bos()
    }
    pub fn eos(&self) -> Token {
        self.model.eos()
    }
    pub fn nl(&self) -> Token {
        self.model.nl()
    }
    pub fn infill_prefix(&self) -> Token {
        self.model.infill_prefix()
    }
    pub fn infill_middle(&self) -> Token {
        self.model.infill_middle()
    }
    pub fn infill_suffix(&self) -> Token {
        self.model.infill_suffix()
    }
    pub fn eot(&self) -> Token {
        self.model.eot()
    }
    pub fn vocabulary_size(&self) -> usize {
        self.model.vocabulary_size()
    }
    pub fn embed_len(&self) -> usize {
        self.model.embed_len()
    }
    pub fn train_len(&self) -> usize {
        self.model.train_len()
    }
    pub fn layers(&self) -> usize {
        self.model.layers()
    }
}

#[derive(Deref, DerefMut)]
pub struct LlavaSession {
    #[deref]
    #[deref_mut]
    session: LlamaSession,
    clip: ClipInner,
}

impl LlavaSession {
    pub fn advance_with_image(&self, image: impl AsRef<[u8]>) -> Result<(), LlavaContextError> {
        let embed = unsafe {
            llava_image_embed_make_with_bytes(
                *self.clip,
                self.session.inner.params.n_threads as c_int,
                image.as_ref().as_ptr(),
                image.as_ref().len() as c_int,
            )
        };

        if embed.is_null() {
            error!("Failed to create image embeddings");
            return Err(LlavaContextError::Embed);
        }

        let token_count = unsafe { (*embed).n_image_pos as usize };
        info!("Advancing context with {token_count} tokens");

        let res = unsafe {
            let ctx_guard = self.session.inner.ctx.lock().unwrap();
            let batch_size = llama_n_batch(**ctx_guard);
            let mut past = self.session.context_size() as c_int;
            let res =
                llava_eval_image_embed(**ctx_guard, embed, batch_size as c_int, addr_of_mut!(past));
            llava_image_embed_free(embed);
            res
        };

        if res {
            let mut guard = self.session.inner.tokens.write().unwrap();
            let image_tokens = vec![Token(0); token_count];
            guard.extend(&image_tokens);
            Ok(())
        } else {
            Err(LlavaContextError::Decode)
        }
    }
}
