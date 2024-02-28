use std::ptr::addr_of_mut;

use llama_cpp_sys::{
    llama_context, llama_sample_min_p, llama_sample_repetition_penalties, llama_sample_softmax,
    llama_sample_tail_free, llama_sample_temp, llama_sample_token, llama_sample_token_greedy,
    llama_sample_token_mirostat, llama_sample_token_mirostat_v2, llama_sample_top_k,
    llama_sample_top_p, llama_sample_typical, llama_token, llama_token_data_array,
};

use crate::{Sampler, Token};

// /// The standard sampler.
// pub struct StandardSampler {
// /// number of previous tokens to remember
// pub n_prev: i32,

// /// if greater than 0, output the probabilities of top n_probs tokens.
// pub n_probs: i32,

// /// <= 0 to use vocab size
// pub top_k: i32,

// /// 1.0 = disabled
// pub top_p: f32,

// /// 0.0 = disabled
// pub min_p: f32,

// /// 1.0 = disabled
// pub tfs_z: f32,

// /// 1.0 = disabled
// pub typical_p: f32,

// /// 1.0 = disabled
// pub temp: f32,

// /// last n tokens to penalize (0 = disable penalty, -1 = context size)
// pub penalty_last_n: i32,

// /// 1.0 = disabled
// pub penalty_repeat: f32,

// /// 0.0 = disabled
// pub penalty_freq: f32,

// /// 0.0 = disabled
// pub penalty_present: f32,

// /// 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
// pub mirostat: i32,

// /// target entropy
// pub mirostat_tau: f32,

// /// learning rate
// pub mirostat_eta: f32,

// /// consider newlines as a repeatable token
// pub penalize_nl: bool,

// /// // optional BNF-like grammar to constrain sampling
// pub grammar: String,

// // Classifier-Free Guidance
// // https://arxiv.org/abs/2306.17806
// /// string to help guidance
// pub cfg_negative_prompt: String,

// /// how strong is guidance
// pub cfg_scale: f32,
// //std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
// }

// impl StandardSampler {}

// impl Sampler for StandardSampler {
// #[allow(clippy::not_unsafe_ptr_arg_deref)]
// fn sample(
// &self,
// context: *mut llama_context,
// mut candidates_p: llama_token_data_array,
// ) -> Token {
// let p_ptr = addr_of_mut!(candidates_p);
// let id = unsafe {
// if self.temp < 0.0 {
// llama_sample_softmax(context, p_ptr);
// candidates_p.data.add(0).read().id
// } else if self.temp == 0.0 {
// llama_sample_token_greedy(context, p_ptr)
// } else {
// match self.mirostat {
// 1 => {
// let mirostat_m = 100;
// llama_sample_temp(context, p_ptr, self.temp);
// // TODO confirm that this is correct
// let mut mu = self.mirostat_tau * 2.0;
// let mu_ptr = addr_of_mut!(mu);
// llama_sample_token_mirostat(
// context,
// p_ptr,
// self.mirostat_tau,
// self.mirostat_eta,
// mirostat_m,
// mu_ptr,
// )
// }
// 2 => {
// llama_sample_temp(context, p_ptr, self.temp);
// // TODO confirm that this is correct
// let mut mu = self.mirostat_tau * 2.0;
// let mu_ptr = addr_of_mut!(mu);
// llama_sample_token_mirostat_v2(
// context,
// p_ptr,
// self.mirostat_tau,
// self.mirostat_eta,
// mu_ptr,
// )
// }
// _ => {
// let min_keep = self.n_probs.max(1) as usize;

// llama_sample_top_k(context, p_ptr, self.top_k, min_keep);
// llama_sample_tail_free(context, p_ptr, self.tfs_z, min_keep);
// llama_sample_typical(context, p_ptr, self.typical_p, min_keep);
// llama_sample_top_p(context, p_ptr, self.top_p, min_keep);
// llama_sample_min_p(context, p_ptr, self.min_p, min_keep);
// llama_sample_temp(context, p_ptr, self.temp);

// llama_sample_token(context, p_ptr)
// }
// }
// }
// };

// Token(id)
// }
// }

// impl Default for StandardSampler {
// fn default() -> Self {
// Self {
// n_prev: 64,
// n_probs: 0,
// top_k: 40,
// top_p: 0.95,
// min_p: 0.05,
// tfs_z: 1.0,
// typical_p: 1.0,
// temp: 0.8,
// penalty_last_n: 64,
// penalty_repeat: 1.1,
// penalty_freq: 0.0,
// penalty_present: 0.0,
// mirostat: 0,
// mirostat_tau: 5.0,
// mirostat_eta: 0.1,
// penalize_nl: true,
// grammar: "".to_string(),
// cfg_negative_prompt: "".to_string(),
// cfg_scale: 1.0,
// }
// }
// }

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

        /// How many tokens back to look when determining penalties.
        last_n: usize,
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

#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum TokenSelector {
    Softmax,
    Greedy,
    MirostatV1 { tau: f32, eta: f32, m: i32, mu: f32 },
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
    pub fn new(stages: Vec<SamplerStage>, min_keep: usize) -> StandardSampler {
        StandardSampler {
            stages,
            min_keep,
            token_selector: TokenSelector::Softmax,
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

        unsafe {
            for stage in &self.stages {
                match stage {
                    SamplerStage::RepetitionPenalty {
                        repetition_penalty,
                        frequency_penalty,
                        presence_penalty,
                        last_n,
                    } => {
                        let last_n = tokens.len().min(*last_n);

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
                        llama_sample_top_p(context, p_ptr, *top_p, self.min_keep);
                    }
                    SamplerStage::MinP(min_p) => {
                        llama_sample_min_p(context, p_ptr, *min_p, self.min_keep);
                    }
                    SamplerStage::TopK(top_k) => {
                        llama_sample_top_k(context, p_ptr, *top_k, self.min_keep);
                    }
                    SamplerStage::Typical(p) => {
                        llama_sample_typical(context, p_ptr, *p, self.min_keep);
                    }
                    SamplerStage::TailFree(z) => {
                        llama_sample_tail_free(context, p_ptr, *z, self.min_keep);
                    }
                }
            }

            let id = match &mut self.token_selector {
                TokenSelector::Softmax => llama_sample_token(context, p_ptr),
                TokenSelector::Greedy => llama_sample_token_greedy(context, p_ptr),
                TokenSelector::MirostatV1 { tau, eta, m, mu } => {
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
