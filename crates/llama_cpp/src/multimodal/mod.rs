use crate::{
    LlamaContextError, LlamaLoadError, LlamaModel, LlamaParams, LlamaSession,
    LlamaTokenizationError, SessionParams, Token,
};
use derive_more::{Deref, DerefMut};
use llama_cpp_sys::{clip_ctx, clip_free};
use std::borrow::Borrow;
use std::path::{Path, PathBuf};

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
    ) -> Result<LlamaSession, LlamaContextError> {
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

impl LlavaSession {}
