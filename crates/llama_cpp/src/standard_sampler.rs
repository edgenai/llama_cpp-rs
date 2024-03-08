use std::ptr::addr_of_mut;

use llama_cpp_sys::{
    llama_context, llama_sample_entropy, llama_sample_grammar, llama_grammar_accept_token, llama_sample_min_p, llama_sample_repetition_penalties, llama_sample_tail_free, llama_sample_temp, llama_sample_token, llama_sample_token_greedy, llama_sample_token_mirostat, llama_sample_token_mirostat_v2, llama_sample_top_k, llama_sample_top_p, llama_sample_typical, llama_token, llama_token_data_array
};

use crate::{grammar::LlamaGrammar, Sampler, Token};

/// Functions which modify the probability distribution output by the model.
///
/// Standard ordering for samplers (taken from [kobold.cpp](https://github.com/LostRuins/koboldcpp)):
///
/// 1. [`SamplerStage::RepetitionPenalty`]
/// 2. [`SamplerStage::Temperature`], [SamplerStage::DynamicTemperature]
/// 3. [`SamplerStage::TopK`]
/// 4. [`SamplerStage::TailFree`]
/// 5. [`SamplerStage::Typical`]
/// 6. [`SamplerStage::TopP`], [`SamplerStage::MinP`]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum SamplerStage {
    /// Divide the logits by this value. Ranges from 0 to 2. Lower values yield a more
    /// deterministic output, and higher values yield a more random/creative output. This should
    /// not be used with [`SamplerStage::DynamicTemperature`].
    Temperature(f32),

    /// Divide the logits by a dynamically determined value between `min_temp`
    /// and `max_temp`. This should not be used with [`SamplerStage::Temperature`].
    ///
    /// This determines the temperature using the equation:
    ///
    /// ```
    /// (current_entropy / maximum_entropy) ^ exponent_val
    /// ```
    ///
    /// where `current_entropy` is the entropy of the current
    /// distribution over tokens, `maximum_entropy` is the maximum possible
    /// entropy over that distribution, and `exponent_val` is the parameter below.
    ///
    /// See: <https://arxiv.org/pdf/2309.02772.pdf>
    DynamicTemperature {
        /// Determines the minimum possible temperature for this stage. Should be between 0 and 2.
        min_temp: f32,

        /// Determines the maximum possible temperature for this stage. Should be between 0 and 2.
        max_temp: f32,

        /// The `exponent_val` parameter. 1 is a good starting point. Values less than 1 cause the
        /// temperature to approach `max_temp` more quickly at small entropies.
        exponent_val: f32,
    },
    /// Penalizes generating a token that is within the `last_n` tokens of context in various ways.
    RepetitionPenalty {
        /// Divide the token's logit by this value if they appear one or more time in the `last_n`
        /// tokens. 1.0 disables this, and values from 1.0-1.2 work well.
        ///
        /// See page 5 of <https://arxiv.org/pdf/1909.05858.pdf>
        repetition_penalty: f32,

        /// Subtract this value from the token's logit for each time the token appears in the
        /// `last_n` tokens. 0.0 disables this, and 0.0-1.0 are reasonable values.
        ///
        /// See: <https://platform.openai.com/docs/guides/text-generation/parameter-details>
        frequency_penalty: f32,

        /// Subtract this value from the token's logit if the token appears in the `last_n` tokens.
        /// 0.0 disables this, and 0.0-1.0 are reasonable values.
        ///
        /// See: <https://platform.openai.com/docs/guides/text-generation/parameter-details>
        presence_penalty: f32,

        /// How many tokens back to look when determining penalties. -1 means context size, and 0
        /// disables this stage.
        last_n: i32,
    },

    /// Keep the most likely tokens until their total probability exceeds `p`.
    ///
    /// See: <https://arxiv.org/abs/1904.09751>
    TopP(f32),

    /// Remove tokens with probability less than `p` times the probability of the most likely
    /// token.
    ///
    /// See: <https://github.com/ggerganov/llama.cpp/pull/3841>
    MinP(f32),

    /// Keep the `k` tokens with the highest probability.
    ///
    /// See: <https://arxiv.org/abs/1904.09751>
    TopK(i32),

    /// Typical Sampling
    ///
    /// See: <https://arxiv.org/abs/2202.00666>
    Typical(f32),

    /// Tail Free Sampling
    ///
    /// See: <https://www.trentonbricken.com/Tail-Free-Sampling/>
    TailFree(f32),
}

impl SamplerStage {
    /// Applies this [`SamplerStage`] to the provided token data array.
    ///
    /// Ensures that at least `min_keep` tokens remain after the
    /// [`SamplerStage`]'s are applied.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn apply(
        &self,
        context: *mut llama_context,
        tokens: &[Token],
        mut candidates_p: llama_token_data_array,
        min_keep: usize,
    ) -> llama_token_data_array {
        let p_ptr = addr_of_mut!(candidates_p);

        unsafe {
            match self {
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
                SamplerStage::DynamicTemperature {
                    min_temp,
                    max_temp,
                    exponent_val,
                } => {
                    llama_sample_entropy(context, p_ptr, *min_temp, *max_temp, *exponent_val);
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

        candidates_p
    }
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

    /// Selects a token using [Mirostat V2](https://arxiv.org/pdf/2007.14966.pdf)
    MirostatV2 { tau: f32, eta: f32, mu: f32 },
}

impl TokenSelector {
    /// Select and and return a token from a given distribution.
    ///
    /// Note: while this function may take a mutable reference to `self`, the internal state *shouldn't* be altered.
    #[allow(clippy::not_unsafe_ptr_arg_deref)]
    pub fn select(
        &mut self,
        context: *mut llama_context,
        mut candidates_p: llama_token_data_array,
    ) -> Token {
        unsafe {
            let p_ptr = addr_of_mut!(candidates_p);
            let id = match self {
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

/// Selects a token after applying multiple [`SamplerStage`]'s to the
/// probability distribution output by the model.
#[derive(Clone, Debug)]
pub struct StandardSampler {
    stages: Vec<SamplerStage>,
    min_keep: usize,
    grammar: Option<LlamaGrammar>,
    token_selector: TokenSelector,
}

impl StandardSampler {
    /// Creates a new [`StandardSampler`] that modifies the model's raw
    /// distribution using multiple [`SamplerStage`]'s, then selects a random
    /// token from that distrubution.
    ///
    /// Ensures that at least `min_keep` tokens remain after the
    /// [`SamplerStage`]'s are applied.
    pub fn new_softmax(
        stages: Vec<SamplerStage>,
        min_keep: usize,
        grammar: Option<LlamaGrammar>,
    ) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            grammar: grammar,
            token_selector: TokenSelector::Softmax,
        }
    }

    /// Creates a new [`StandardSampler`] that always selects the next most
    /// token produced by the model.
    pub fn new_greedy() -> StandardSampler {
        StandardSampler {
            stages: Vec::new(),
            min_keep: 0,
            grammar: None,
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
            grammar: None,
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
            grammar: None,
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
            grammar: None,
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

        // Note: We should sample grammar before applying other sampling stages.
        if let Some(grammar) = self.grammar.as_mut() {
            unsafe { llama_sample_grammar(context, p_ptr, grammar.grammar.as_ptr()) };
        }

        for stage in &self.stages {
            candidates_p = stage.apply(context, tokens, candidates_p, min_keep);
        }

        let token = self.token_selector.select(context, candidates_p);

        // Note: We must accept the token into the grammar after sampling if a grammar is provided.
        if let Some(grammar) = self.grammar.as_mut() {
            unsafe { llama_grammar_accept_token(context, grammar.grammar.as_ptr(), token.0)}
        }

        token
    }
}
