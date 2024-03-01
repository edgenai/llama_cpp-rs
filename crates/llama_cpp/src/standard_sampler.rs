use std::ptr::addr_of_mut;

use llama_cpp_sys::{
    llama_context, llama_sample_min_p, llama_sample_repetition_penalties, llama_sample_tail_free,
    llama_sample_temp, llama_sample_token, llama_sample_token_greedy, llama_sample_token_mirostat,
    llama_sample_token_mirostat_v2, llama_sample_top_k, llama_sample_top_p, llama_sample_typical,
    llama_token, llama_token_data_array,
};

use crate::{Sampler, Token};

/// Functions which change how a [`SoftmaxSampler`] selects its next token.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SamplerStage {
    /// Divide the logits by this value. Ranges from 0 to 2. Lower values yield a more
    /// deterministic output, and higher values yield a more random/creative output.
    Temperature(f32),

    /// Penalizes generating a tokens that is within the `last_n` tokens in various ways.
    RepetitionPenalty {
        /// Divide the token's logit by this value if they appear one or more time in the `last_n`
        /// tokens. 1.0 disables this, and values from 1.0-1.2 work well.
        ///
        /// See page 5 of https://arxiv.org/pdf/1909.05858.pdf
        repetition_penalty: f32,

        /// Subtract this value from the token's logit for each time the token appears in the
        /// `last_n` tokens. 0.0 disables this, and 0.0-1.0 are reasonable values.
        ///
        /// See: https://platform.openai.com/docs/guides/text-generation/parameter-details
        frequency_penalty: f32,

        /// Subtract this value from the token's logit if the token appears in the `last_n` tokens.
        /// 0.0 disables this, and 0.0-1.0 are reasonable values.
        ///
        /// See: https://platform.openai.com/docs/guides/text-generation/parameter-details
        presence_penalty: f32,

        /// How many tokens back to look when determining penalties. -1 means context size, and 0
        /// disables this stage.
        last_n: i32,
    },

    /// Keep the most likely tokens until their total probability exceeds `p`.
    ///
    /// See: https://arxiv.org/abs/1904.09751
    TopP(f32),

    /// Remove tokens with probability less than `p` times the probability of the most likely
    /// token.
    ///
    /// See: https://github.com/ggerganov/llama.cpp/pull/3841
    MinP(f32),

    /// Keep the `k` tokens with the highest probability.
    ///
    /// See: https://arxiv.org/abs/1904.09751
    TopK(i32),

    /// Typical Sampling
    ///
    /// See: https://arxiv.org/abs/2202.00666
    Typical(f32),

    /// Tail Free Sampling
    ///
    /// See: https://www.trentonbricken.com/Tail-Free-Sampling/
    TailFree(f32),
}

/// Determines how the next token is selected from the distribution produced by
/// the model and the [`SamplerStage`]'s.
#[derive(Clone, Debug)]
#[non_exhaustive]
enum TokenSelector {
    /// Selects a token at random, weighted by the distribution
    Softmax,

    /// Always selects the most likely token.
    Greedy,

    /// Selects a token using [Mirostat](https://arxiv.org/pdf/2007.14966.pdf)
    Mirostat { tau: f32, eta: f32, m: i32, mu: f32 },

    /// Selects a token using [Mirostat V2](https://arxiv.org/abs/2007.14966.pdf)
    MirostatV2 { tau: f32, eta: f32, mu: f32 },
}

/// Selects a token with a [`TokenSelector`] after applying multiple [`SamplerStage`]'s to it.
#[derive(Clone, Debug)]
pub struct StandardSampler {
    stages: Vec<SamplerStage>,
    min_keep: usize,
    token_selector: TokenSelector,
}

impl StandardSampler {
    /// Creates a new [`StandardSampler`] that selects a token at random based
    /// on the distribution from the model after the [`SamplerStage`]'s are
    /// applied.
    pub fn new_softmax(stages: Vec<SamplerStage>, min_keep: usize) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::Softmax,
        }
    }

    /// Creates a new [`StandardSampler`] that always selects the next most
    /// token produced by the model.
    pub fn new_greedy() -> StandardSampler {
        StandardSampler {
            stages: Vec::new(),
            min_keep: 0,
            token_selector: TokenSelector::Greedy,
        }
    }

    /// Creates a new [`StandardSampler`] that selects a token using
    /// [Mirostat](https://arxiv.org/pdf/2007.14966.pdf).
    pub fn new_mirostat(
        stages: Vec<SamplerStage>,
        min_keep: usize,
        tau: f32,
        eta: f32,
        m: i32,
    ) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::Mirostat {
                tau,
                eta,
                m,
                mu: 2.0 * tau,
            },
        }
    }

    /// Creates a new [`StandardSampler`] that selects a token using
    /// [Mirostat V2](https://arxiv.org/pdf/2007.14966.pdf).
    pub fn new_mirostat_v2(
        stages: Vec<SamplerStage>,
        min_keep: usize,
        tau: f32,
        eta: f32,
    ) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::MirostatV2 {
                tau,
                eta,
                mu: 2.0 * tau,
            },
        }
    }
}

impl Default for StandardSampler {
    fn default() -> Self {
        Self {
            stages: vec![
                SamplerStage::RepetitionPenalty {
                    repetition_penalty: 1.1,
                    frequency_penalty: 0.0,
                    presence_penalty: 0.0,
                    last_n: 64,
                },
                SamplerStage::TopK(40),
                SamplerStage::TopP(0.95),
                SamplerStage::MinP(0.05),
                SamplerStage::Temperature(0.8),
            ],
            min_keep: 1,
            token_selector: TokenSelector::Softmax,
        }
    }
}

impl Sampler for StandardSampler {
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    fn sample(
        &mut self,
        context: *mut llama_context,
        tokens: &[Token],
        mut candidates_p: llama_token_data_array,
    ) -> Token {
        let p_ptr = addr_of_mut!(candidates_p);
        let min_keep = self.min_keep.max(1);

        unsafe {
            for stage in &self.stages {
                match stage {
                    SamplerStage::RepetitionPenalty {
                        repetition_penalty,
                        frequency_penalty,
                        presence_penalty,
                        last_n,
                    } => {
                        let last_n = if *last_n < 0 {
                            tokens.len()
                        } else {
                            tokens.len().min(*last_n as usize)
                        };

                        llama_sample_repetition_penalties(
                            context,
                            p_ptr,
                            tokens[tokens.len() - last_n..].as_ptr() as *const llama_token,
                            last_n,
                            *repetition_penalty,
                            *frequency_penalty,
                            *presence_penalty,
                        );
                    }
                    SamplerStage::Temperature(temp) => {
                        if *temp == 0.0 {
                            llama_sample_top_k(context, p_ptr, 1, 1);
                        } else {
                            llama_sample_temp(context, p_ptr, *temp);
                        }
                    }
                    SamplerStage::TopP(top_p) => {
                        llama_sample_top_p(context, p_ptr, *top_p, min_keep);
                    }
                    SamplerStage::MinP(min_p) => {
                        llama_sample_min_p(context, p_ptr, *min_p, min_keep);
                    }
                    SamplerStage::TopK(top_k) => {
                        llama_sample_top_k(context, p_ptr, *top_k, min_keep);
                    }
                    SamplerStage::Typical(p) => {
                        llama_sample_typical(context, p_ptr, *p, min_keep);
                    }
                    SamplerStage::TailFree(z) => {
                        llama_sample_tail_free(context, p_ptr, *z, min_keep);
                    }
                }
            }

            let id = match &mut self.token_selector {
                TokenSelector::Softmax => llama_sample_token(context, p_ptr),
                TokenSelector::Greedy => llama_sample_token_greedy(context, p_ptr),
                TokenSelector::Mirostat { tau, eta, m, mu } => {
                    llama_sample_token_mirostat(context, p_ptr, *tau, *eta, *m, addr_of_mut!(*mu))
                }
                TokenSelector::MirostatV2 { tau, eta, mu } => {
                    llama_sample_token_mirostat_v2(context, p_ptr, *tau, *eta, addr_of_mut!(*mu))
                }
            };

            Token(id)
        }
    }
}
