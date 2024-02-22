use std::{pin::Pin, task::{Context, Poll}};

use futures::{executor::block_on, Stream};
use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver};

use crate::{LlamaModel, Token};


/// A handle (and channel) to an ongoing completion job on an off thread.
///
/// If this structure is dropped, the off thread is stopped.
pub struct CompletionHandle {
    /// The token receiver bound to the off thread.
    rx: UnboundedReceiver<Token>,

    /// The model that this handle's LlamaSession is associated with.
    model: LlamaModel,
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

    pub fn into_byte_iter(self) -> ByteCompletion {
        ByteCompletion(self)
    }

    pub fn into_string_iter(self) -> StringCompletion {
        StringCompletion {
            completion: ByteCompletion(self),
            decoder: TokenDecoder::new(),
        }
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

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.rx.try_recv() {
            Ok(token) => Poll::Ready(Some(token)),
            Err(TryRecvError::Disconnected) => Poll::Ready(None),
            Err(TryRecvError::Empty) => Poll::Pending,
        }
    }
}

pub struct ByteCompletion(CompletionHandle);

impl Iterator for ByteCompletion {
    type Item = Vec<u8>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|token| self.0.model.token_to_byte_piece(token))
    }
}

impl Stream for ByteCompletion {
    type Item = Vec<u8>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let res = std::pin::pin!(&mut self.0).poll_next(cx);
        res.map(|optional_token| optional_token.map(|token| self.0.model.token_to_byte_piece(token)))
    }
}

struct TokenDecoder {
    buf: Vec<u8>,
}

impl TokenDecoder {
    fn new() -> TokenDecoder {
        TokenDecoder {
            buf: Vec::new(),
        }
    }

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
                    out
                        .push_str(unsafe { std::str::from_utf8_unchecked(&token[..valid_len]) });

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

    fn last_part(&mut self) -> Option<String> {
        (!self.buf.is_empty()).then(|| {
            let out = String::from_utf8_lossy(&self.buf).to_string();
            self.buf.clear();
            out
        })
    }
}

pub struct StringCompletion {
    completion: ByteCompletion,
    decoder: TokenDecoder,
}

impl Iterator for StringCompletion {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(bytes) = self.completion.next() {
            Some(self.decoder.add_token(&bytes))
        } else {
            self.decoder.last_part()
        }
    }
}

impl Stream for StringCompletion {
    type Item = String;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match std::pin::pin!(&mut self.completion).poll_next(cx) {
            Poll::Ready(Some(bytes)) => Poll::Ready(Some(self.decoder.add_token(&bytes))),
            Poll::Ready(None) => Poll::Ready(self.decoder.last_part()),
            Poll::Pending => Poll::Pending,
        }
    }
}
