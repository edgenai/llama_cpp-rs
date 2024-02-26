//! Module containing [`CompletionHandle`] and associated types.

use std::{
    borrow::Borrow,
    pin::Pin,
    task::{Context, Poll},
};

use futures::{executor::block_on, Stream};
use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver};

use crate::{LlamaModel, Token};

/// A handle (and channel) to an ongoing completion job on an off thread.
///
/// If this structure is dropped, the off thread is stopped.
pub struct CompletionHandle {
    /// The token receiver bound to the off thread.
    pub(super) rx: UnboundedReceiver<Token>,

    /// The model that this handle's LlamaSession is associated with.
    pub(super) model: LlamaModel,
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

    /// Iterate or stream over the generated tokens represented as byte pieces.
    pub fn into_bytes(self) -> TokensToBytes<CompletionHandle> {
        let model = self.model.clone();
        TokensToBytes::new(self, model)
    }

    /// Iterate or stream over the generated tokens represented as [`String`] pieces.
    ///
    /// This iterator will defer outputting a UTF-8 codepoint until it recieves
    /// the entire codepoint. This ensures that Emojis and other codepoints that
    /// are split across multiple tokens are decoded correctly.
    ///
    /// Joining the returned strings will yield the same output as
    /// [`LlamaModel::decode_tokens`], with invalid UTF-8 replaced with the
    /// unicode replacement character: "�".
    pub fn into_strings(self) -> TokensToStrings<CompletionHandle> {
        let model = self.model.clone();
        TokensToStrings::new(self, model)
    }

    /// Converts a `CompletionHandle` into a `String` containing the full model
    /// output.
    pub fn into_string(self) -> String {
        self.model.clone().decode_tokens(self)
    }

    /// Converts a `CompletionHandle` into a `String` containing the full model
    /// output, asynchronously.
    pub async fn into_string_async(mut self) -> String {
        let mut tokens = Vec::new();

        while let Some(token) = self.next_token_async().await {
            tokens.push(token);
        }

        self.model.decode_tokens(tokens)
    }
}

impl Iterator for CompletionHandle {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.rx.recv())
    }
}

impl Stream for CompletionHandle {
    type Item = Token;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// A wrapper struct around an iterator or stream of tokens, yielding `Vec<u8>`
/// byte pieces for each token.
pub struct TokensToBytes<I> {
    inner: I,
    model: LlamaModel,
}

impl<I> TokensToBytes<I> {
    /// Creates a new [`TokensToBytes`] iterator/stream from an inner
    /// iterator/stream and a [`LlamaModel`] used to convert each token into its
    /// byte sequence.
    pub fn new(inner: I, model: LlamaModel) -> TokensToBytes<I> {
        Self { inner, model }
    }
}

impl<I: Iterator> Iterator for TokensToBytes<I>
where
    I::Item: Borrow<Token>,
{
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner
            .next()
            .map(|token| self.model.token_to_byte_piece(*token.borrow()))
    }
}

impl<I: Stream + Unpin> Stream for TokensToBytes<I>
where
    I::Item: Borrow<Token>,
{
    type Item = Vec<u8>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let res = std::pin::pin!(&mut self.inner).poll_next(cx);
        res.map(|optional_token| {
            optional_token.map(|token| self.model.token_to_byte_piece(*token.borrow()))
        })
    }
}

/// A struct used to decode `Vec<u8>` tokens from a [`CompletionHandle`] into [`String`] tokens.
///
/// This struct can handle merging split UTF-8 codepoints.
struct TokenDecoder {
    /// A buffer used to store incomplete codepoints between calls to
    /// [`TokenDecoder::add_token`]
    buf: Vec<u8>,
}

impl TokenDecoder {
    /// Creates a new [`TokenDecoder`].
    fn new() -> TokenDecoder {
        TokenDecoder { buf: Vec::new() }
    }

    /// Adds a token to the decoder and returns a full [`String`] representation
    /// of it if possible.
    ///
    /// If the token has a trailing incomplete UTF-8 sequence, this method will
    /// not include it in the output string. Instead, the incomplete sequence
    /// will be stored in the decoder's buffer for the next call to this method.
    fn add_token(&mut self, token: &[u8]) -> String {
        let mut token = token;
        let mut out = String::new();

        if !self.buf.is_empty() {
            self.buf.extend_from_slice(token);
            token = self.buf.as_slice();
        }

        loop {
            match std::str::from_utf8(token) {
                Ok(s) => {
                    out.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(err) => {
                    let valid_len = err.valid_up_to();
                    out.push_str(unsafe { std::str::from_utf8_unchecked(&token[..valid_len]) });

                    if let Some(len) = err.error_len() {
                        out.push(char::REPLACEMENT_CHARACTER);
                        token = &token[valid_len + len..];
                    } else {
                        let mut last_bytes = [0; 4];
                        let last_part_len = token.len() - valid_len;
                        last_bytes[..last_part_len].clone_from_slice(&token[valid_len..]);

                        self.buf.clear();
                        self.buf.extend_from_slice(&last_bytes[..last_part_len]);

                        break;
                    }
                }
            }
        }

        out
    }

    /// Returns the last partial UTF-8 sequence stored in the decoder.
    ///
    /// If there is no partial UTF-8 sequence stored, this method will return `None`.
    fn last_part(&mut self) -> Option<String> {
        (!self.buf.is_empty()).then(|| {
            let out = String::from_utf8_lossy(&self.buf).to_string();
            self.buf.clear();
            out
        })
    }
}

/// A wrapper struct around a `CompletionHandle`, yielding `String` tokens for
/// each byte piece of the model's output.
///
/// See [`CompletionHandle::into_string_completion`] for more information.
pub struct TokensToStrings<I> {
    completion: TokensToBytes<I>,
    decoder: TokenDecoder,
}

impl<I> TokensToStrings<I> {
    /// Creates a new [`TokensToStrings`] iterator/stream from an inner
    /// iterator/stream and a [`LlamaModel`] used to convert each token into its
    /// byte sequence.
    pub fn new(inner: I, model: LlamaModel) -> Self {
        Self {
            completion: TokensToBytes::new(inner, model),
            decoder: TokenDecoder::new(),
        }
    }
}

impl<I: Iterator> Iterator for TokensToStrings<I>
where
    I::Item: Borrow<Token>,
{
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bytes) = self.completion.next() {
            Some(self.decoder.add_token(&bytes))
        } else {
            self.decoder.last_part()
        }
    }
}

impl<I: Stream + Unpin> Stream for TokensToStrings<I>
where
    I::Item: Borrow<Token>,
{
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match std::pin::pin!(&mut self.completion).poll_next(cx) {
            Poll::Ready(Some(bytes)) => Poll::Ready(Some(self.decoder.add_token(&bytes))),
            Poll::Ready(None) => Poll::Ready(self.decoder.last_part()),
            Poll::Pending => Poll::Pending,
        }
    }
}
