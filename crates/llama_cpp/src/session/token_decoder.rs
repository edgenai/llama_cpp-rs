use crate::{LlamaModel, Token};

pub struct TokenDecoder {
    inner: String,
    buf: Vec<u8>,
    prev_len: usize,
}

impl TokenDecoder {
    pub fn new() -> TokenDecoder {
        TokenDecoder {
            inner: String::new(),
            buf: Vec::new(),
            prev_len: 0,
        }
    }

    pub fn add_token(&mut self, token: &[u8]) {
        let mut token = token;
        self.prev_len = self.inner.len();

        if !self.buf.is_empty() {
            self.buf.extend_from_slice(token);
            token = self.buf.as_slice();
        }

        loop {
            match std::str::from_utf8(token) {
                Ok(s) => {
                    self.inner.push_str(s);
                    self.buf.clear();
                    break;
                }
                Err(err) => {
                    let valid_len = err.valid_up_to();
                    self.inner
                        .push_str(unsafe { std::str::from_utf8_unchecked(&token[..valid_len]) });

                    if let Some(len) = err.error_len() {
                        self.inner.push(char::REPLACEMENT_CHARACTER);
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
    }

    pub fn current_string(&self) -> &str {
        &self.inner
    }

    pub fn new_string_part(&self) -> &str {
        &self.inner[self.prev_len..]
    }

    pub fn into_string(mut self) -> String {
        self.inner.push_str(&String::from_utf8_lossy(&self.buf));
        self.inner
    }
}

impl LlamaModel {}
