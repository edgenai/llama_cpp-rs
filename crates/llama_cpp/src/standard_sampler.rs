use std::ptr::addr_of_mut;

use llama_cpp_sys::{
    llama_context, llama_sample_min_p, llama_sample_softmax, llama_sample_tail_free,
    llama_sample_temp, llama_sample_token, llama_sample_token_greedy, llama_sample_token_mirostat,
    llama_sample_token_mirostat_v2, llama_sample_top_k, llama_sample_top_p, llama_sample_typical,
    llama_token_data_array,
};

use crate::{Sampler, Token};

pub struct StandardSampler {
    /// number of previous tokens to remember
    pub n_prev: i32,

    /// if greater than 0, output the probabilities of top n_probs tokens.
    pub n_probs: i32,

    /// <= 0 to use vocab size
    pub top_k: i32,

    /// 1.0 = disabled
    pub top_p: f32,

    /// 0.0 = disabled
    pub min_p: f32,

    /// 1.0 = disabled
    pub tfs_z: f32,

    /// 1.0 = disabled
    pub typical_p: f32,

    /// 1.0 = disabled
    pub temp: f32,

    /// last n tokens to penalize (0 = disable penalty, -1 = context size)
    pub penalty_last_n: i32,

    /// 1.0 = disabled
    pub penalty_repeat: f32,

    /// 0.0 = disabled
    pub penalty_freq: f32,

    /// 0.0 = disabled
    pub penalty_present: f32,

    /// 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    pub mirostat: i32,

    /// target entropy
    pub mirostat_tau: f32,

    /// learning rate
    pub mirostat_eta: f32,

    /// consider newlines as a repeatable token
    pub penalize_nl: bool,

    /// // optional BNF-like grammar to constrain sampling
    pub grammar: String,

    // Classifier-Free Guidance
    // https://arxiv.org/abs/2306.17806
    /// string to help guidance
    pub cfg_negative_prompt: String,

    /// how strong is guidance
    pub cfg_scale: f32,
    //std::unordered_map<llama_token, float> logit_bias; // logit bias for specific tokens
}

impl StandardSampler {}

impl Sampler for StandardSampler {
    fn sample(
        &self,
        context: *mut llama_context,
        mut candidates_p: llama_token_data_array,
    ) -> Token {
        let p_ptr = addr_of_mut!(candidates_p);
        let id = unsafe {
            if self.temp < 0.0 {
                llama_sample_softmax(context, p_ptr);
                candidates_p.data.add(0).read().id
            } else if self.temp == 0.0 {
                llama_sample_token_greedy(context, p_ptr)
            } else {
                match self.mirostat {
                    1 => {
                        let mirostat_m = 100;
                        llama_sample_temp(context, p_ptr, self.temp);
                        // TODO confirm that this is correct
                        let mut mu = self.mirostat_tau * 2.0;
                        let mu_ptr = addr_of_mut!(mu);
                        llama_sample_token_mirostat(
                            context,
                            p_ptr,
                            self.mirostat_tau,
                            self.mirostat_eta,
                            mirostat_m,
                            mu_ptr,
                        )
                    }
                    2 => {
                        llama_sample_temp(context, p_ptr, self.temp);
                        // TODO confirm that this is correct
                        let mut mu = self.mirostat_tau * 2.0;
                        let mu_ptr = addr_of_mut!(mu);
                        llama_sample_token_mirostat_v2(
                            context,
                            p_ptr,
                            self.mirostat_tau,
                            self.mirostat_eta,
                            mu_ptr,
                        )
                    }
                    _ => {
                        let min_keep = self.n_probs.max(1) as usize;

                        llama_sample_top_k(context, p_ptr, self.top_k, min_keep);
                        llama_sample_tail_free(context, p_ptr, self.tfs_z, min_keep);
                        llama_sample_typical(context, p_ptr, self.typical_p, min_keep);
                        llama_sample_top_p(context, p_ptr, self.top_p, min_keep);
                        llama_sample_min_p(context, p_ptr, self.min_p, min_keep);
                        llama_sample_temp(context, p_ptr, self.temp);

                        llama_sample_token(context, p_ptr)
                    }
                }
            }
        };

        Token(id)
    }
}

impl Default for StandardSampler {
    fn default() -> Self {
        Self {
            n_prev: 64,
            n_probs: 0,
            top_k: 40,
            top_p: 0.95,
            min_p: 0.05,
            tfs_z: 1.0,
            typical_p: 1.0,
            temp: 0.8,
            penalty_last_n: 64,
            penalty_repeat: 1.1,
            penalty_freq: 0.0,
            penalty_present: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            penalize_nl: true,
            grammar: "".to_string(),
            cfg_negative_prompt: "".to_string(),
            cfg_scale: 1.0,
        }
    }
}
