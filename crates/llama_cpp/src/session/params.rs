//! Implements [`SessionParams`], which configures a [`crate::LlamaSession`]

use std::ptr::null_mut;

use llama_cpp_sys::{
    ggml_type, llama_context_default_params, llama_context_params, llama_pooling_type,
    llama_rope_scaling_type,
};

/// whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
#[derive(Clone, Copy, Debug)]
pub enum PoolingType {
    /// Unspecified.
    Unspecified,
    /// Don't pool.
    None,
    /// TODO lookup what this does
    Mean,
    /// TODO lookup what this does
    Cls,
}

impl From<PoolingType> for llama_pooling_type {
    fn from(value: PoolingType) -> Self {
        match value {
            PoolingType::Unspecified => llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED,
            PoolingType::None => llama_pooling_type::LLAMA_POOLING_TYPE_NONE,
            PoolingType::Mean => llama_pooling_type::LLAMA_POOLING_TYPE_MEAN,
            PoolingType::Cls => llama_pooling_type::LLAMA_POOLING_TYPE_CLS,
        }
    }
}

impl From<llama_pooling_type> for PoolingType {
    fn from(value: llama_pooling_type) -> Self {
        #![allow(non_upper_case_globals)]
        match value {
            llama_pooling_type::LLAMA_POOLING_TYPE_UNSPECIFIED => PoolingType::Unspecified,
            llama_pooling_type::LLAMA_POOLING_TYPE_NONE => PoolingType::None,
            llama_pooling_type::LLAMA_POOLING_TYPE_MEAN => PoolingType::Mean,
            llama_pooling_type::LLAMA_POOLING_TYPE_CLS => PoolingType::Cls,
            _ => unimplemented!(),
        }
    }
}

/// A rope scaling type.
#[derive(Clone, Copy)]
pub enum RopeScaling {
    /// Unspecified.
    Unspecified,
    /// None.
    None,
    /// Linear.
    Linear,
    /// Yarn.
    Yarn,
}

impl From<RopeScaling> for llama_rope_scaling_type {
    fn from(value: RopeScaling) -> Self {
        match value {
            RopeScaling::Unspecified => {
                llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED
            }
            RopeScaling::None => llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_NONE,
            RopeScaling::Linear => llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LINEAR,
            RopeScaling::Yarn => llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_YARN,
        }
    }
}

impl From<llama_rope_scaling_type> for RopeScaling {
    fn from(value: llama_rope_scaling_type) -> Self {
        match value {
            llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED => {
                RopeScaling::Unspecified
            }
            llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_NONE => RopeScaling::None,
            llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_LINEAR => RopeScaling::Linear,
            llama_rope_scaling_type::LLAMA_ROPE_SCALING_TYPE_YARN => RopeScaling::Yarn,
            _ => unimplemented!(),
        }
    }
}

/// The type of key or value in the cache.
#[derive(Clone, Copy)]
pub enum CacheType {
    /// 32 bit float.
    F32,
    /// 16 bit float.
    F16,
    /// TODO ???
    Q4_0,
    /// TODO ???
    Q4_1,
    /// TODO ???
    Q5_0,
    /// TODO ???
    Q5_1,
    /// TODO ???
    Q8_0,
    /// TODO ???
    Q8_1,
    /// TODO ???
    Q2K,
    /// TODO ???
    Q3K,
    /// TODO ???
    Q4K,
    /// TODO ???
    Q5K,
    /// TODO ???
    Q6K,
    /// TODO ???
    Q8K,
    /// TODO ???
    IQ2XXS,
    /// TODO ???
    IQ2XS,
    /// TODO ???
    IQ3XXS,
    /// TODO ???
    IQ1S,
    /// TODO ???
    IQ4NL,
    /// TODO ???
    IQ3S,
    /// TODO ???
    IQ2S,
    /// TODO ???
    IQ4XS,
    /// 8 bit integer.
    I8,
    /// 16 bit integer.
    I16,
    /// 32 bit integer.
    I32,
    /// 64 bit integer.
    I64,
    /// 64 bit float.
    F64,
    /// Number of values in this enum. Not applicable to rust.
    Count,
}

impl From<CacheType> for ggml_type {
    fn from(value: CacheType) -> Self {
        match value {
            CacheType::F32 => ggml_type::GGML_TYPE_F32,
            CacheType::F16 => ggml_type::GGML_TYPE_F16,
            CacheType::Q4_0 => ggml_type::GGML_TYPE_Q4_0,
            CacheType::Q4_1 => ggml_type::GGML_TYPE_Q4_1,
            CacheType::Q5_0 => ggml_type::GGML_TYPE_Q5_0,
            CacheType::Q5_1 => ggml_type::GGML_TYPE_Q5_1,
            CacheType::Q8_0 => ggml_type::GGML_TYPE_Q8_0,
            CacheType::Q8_1 => ggml_type::GGML_TYPE_Q8_1,
            CacheType::Q2K => ggml_type::GGML_TYPE_Q2_K,
            CacheType::Q3K => ggml_type::GGML_TYPE_Q3_K,
            CacheType::Q4K => ggml_type::GGML_TYPE_Q4_K,
            CacheType::Q5K => ggml_type::GGML_TYPE_Q5_K,
            CacheType::Q6K => ggml_type::GGML_TYPE_Q6_K,
            CacheType::Q8K => ggml_type::GGML_TYPE_Q8_K,
            CacheType::IQ2XXS => ggml_type::GGML_TYPE_IQ2_XXS,
            CacheType::IQ2XS => ggml_type::GGML_TYPE_IQ2_XS,
            CacheType::IQ3XXS => ggml_type::GGML_TYPE_IQ3_XXS,
            CacheType::IQ1S => ggml_type::GGML_TYPE_IQ1_S,
            CacheType::IQ4NL => ggml_type::GGML_TYPE_IQ4_NL,
            CacheType::IQ3S => ggml_type::GGML_TYPE_IQ3_S,
            CacheType::IQ2S => ggml_type::GGML_TYPE_IQ2_S,
            CacheType::IQ4XS => ggml_type::GGML_TYPE_IQ4_XS,
            CacheType::I8 => ggml_type::GGML_TYPE_I8,
            CacheType::I16 => ggml_type::GGML_TYPE_I16,
            CacheType::I32 => ggml_type::GGML_TYPE_I32,
            CacheType::I64 => ggml_type::GGML_TYPE_I64,
            CacheType::F64 => ggml_type::GGML_TYPE_F64,
            CacheType::Count => ggml_type::GGML_TYPE_COUNT,
        }
    }
}

impl From<ggml_type> for CacheType {
    fn from(value: ggml_type) -> Self {
        match value {
            ggml_type::GGML_TYPE_F32 => CacheType::F32,
            ggml_type::GGML_TYPE_F16 => CacheType::F16,
            ggml_type::GGML_TYPE_Q4_0 => CacheType::Q4_0,
            ggml_type::GGML_TYPE_Q4_1 => CacheType::Q4_1,
            ggml_type::GGML_TYPE_Q5_0 => CacheType::Q5_0,
            ggml_type::GGML_TYPE_Q5_1 => CacheType::Q5_1,
            ggml_type::GGML_TYPE_Q8_0 => CacheType::Q8_0,
            ggml_type::GGML_TYPE_Q8_1 => CacheType::Q8_1,
            ggml_type::GGML_TYPE_Q2_K => CacheType::Q2K,
            ggml_type::GGML_TYPE_Q3_K => CacheType::Q3K,
            ggml_type::GGML_TYPE_Q4_K => CacheType::Q4K,
            ggml_type::GGML_TYPE_Q5_K => CacheType::Q5K,
            ggml_type::GGML_TYPE_Q6_K => CacheType::Q6K,
            ggml_type::GGML_TYPE_Q8_K => CacheType::Q8K,
            ggml_type::GGML_TYPE_IQ2_XXS => CacheType::IQ2XXS,
            ggml_type::GGML_TYPE_IQ2_XS => CacheType::IQ2XS,
            ggml_type::GGML_TYPE_IQ3_XXS => CacheType::IQ3XXS,
            ggml_type::GGML_TYPE_IQ1_S => CacheType::IQ1S,
            ggml_type::GGML_TYPE_IQ4_NL => CacheType::IQ4NL,
            ggml_type::GGML_TYPE_IQ3_S => CacheType::IQ3S,
            ggml_type::GGML_TYPE_IQ2_S => CacheType::IQ2S,
            ggml_type::GGML_TYPE_IQ4_XS => CacheType::IQ4XS,
            ggml_type::GGML_TYPE_I8 => CacheType::I8,
            ggml_type::GGML_TYPE_I16 => CacheType::I16,
            ggml_type::GGML_TYPE_I32 => CacheType::I32,
            ggml_type::GGML_TYPE_I64 => CacheType::I64,
            ggml_type::GGML_TYPE_F64 => CacheType::F64,
            ggml_type::GGML_TYPE_COUNT => CacheType::Count,
            _ => unimplemented!(),
        }
    }
}

/// Session-specific parameters.
#[derive(Clone)]
pub struct SessionParams {
    /// RNG seed, [`u32::MAX`] for random (default)
    pub seed: u32,

    /// text context, 0 = from model
    pub n_ctx: u32,

    /// prompt processing maximum batch size
    pub n_batch: u32,

    /// physical maximum batch size used for computations
    pub n_ubatch: u32,

    /// max number of sequences (i.e. distinct states for recurrent models)
    pub n_seq_max: u32,

    /// number of threads to use for generation
    pub n_threads: u32,

    /// number of threads to use for batch processing
    pub n_threads_batch: u32,

    /// RoPE scaling type, from [`llama_rope_scaling_type`]
    pub rope_scaling_type: RopeScaling,

    /// ref: https://github.com/ggerganov/llama.cpp/pull/2054

    /// RoPE base frequency, 0 = from model
    pub rope_freq_base: f32,

    /// RoPE frequency scaling factor, 0 = from model
    pub rope_freq_scale: f32,

    /// YaRN extrapolation mix factor, negative = from model
    pub yarn_ext_factor: f32,

    /// YaRN magnitude scaling factor
    pub yarn_attn_factor: f32,

    /// YaRN low correction dim
    pub yarn_beta_fast: f32,

    /// YaRN high correction dim
    pub yarn_beta_slow: f32,

    /// YaRN original context size
    pub yarn_orig_ctx: u32,

    /// data type for K cache
    pub type_k: CacheType,

    /// data type for V cache
    pub type_v: CacheType,

    /// embedding mode only
    pub embedding: bool,

    /// whether to offload the KQV ops (including the KV cache) to GPU
    pub offload_kqv: bool,

    /// whether to pool (sum) embedding results by sequence id (ignored if no pooling layer)
    pub pooling: PoolingType,

    /// defragment the KV cache if holes/size > thold, < 0 disabled (default)
    pub defrag_threshold: f32,
}

impl Default for SessionParams {
    fn default() -> Self {
        let c_defaults = unsafe {
            // SAFETY: Stack constructor, always safe.
            llama_context_default_params()
        };

        let threads = num_cpus::get_physical() as u32 - 1;

        Self {
            seed: c_defaults.seed,
            n_ctx: c_defaults.n_ctx,
            n_batch: c_defaults.n_batch,
            n_ubatch: c_defaults.n_ubatch,
            n_seq_max: c_defaults.n_seq_max,
            n_threads: threads,
            n_threads_batch: threads,
            rope_scaling_type: c_defaults.rope_scaling_type.into(),
            rope_freq_base: c_defaults.rope_freq_base,
            rope_freq_scale: c_defaults.rope_freq_scale,
            yarn_ext_factor: c_defaults.yarn_ext_factor,
            yarn_attn_factor: c_defaults.yarn_attn_factor,
            yarn_beta_fast: c_defaults.yarn_beta_fast,
            yarn_beta_slow: c_defaults.yarn_beta_slow,
            yarn_orig_ctx: c_defaults.yarn_orig_ctx,
            type_k: c_defaults.type_k.into(),
            type_v: c_defaults.type_v.into(),
            embedding: c_defaults.embeddings,
            offload_kqv: c_defaults.offload_kqv,
            pooling: c_defaults.pooling_type.into(),
            defrag_threshold: c_defaults.defrag_thold,
        }
    }
}

impl From<SessionParams> for llama_context_params {
    fn from(value: SessionParams) -> Self {
        Self {
            seed: value.seed,
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_ubatch: value.n_ubatch,
            n_seq_max: value.n_seq_max,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            rope_scaling_type: value.rope_scaling_type.into(),
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            yarn_orig_ctx: value.yarn_orig_ctx,
            defrag_thold: value.defrag_threshold,
            cb_eval: None,
            cb_eval_user_data: null_mut(),
            type_k: value.type_k.into(),
            type_v: value.type_v.into(),
            logits_all: false, // Deprecated
            embeddings: value.embedding,
            offload_kqv: value.offload_kqv,
            pooling_type: value.pooling.into(),
            abort_callback: None,
            abort_callback_data: null_mut(),
        }
    }
}

impl From<llama_context_params> for SessionParams {
    fn from(value: llama_context_params) -> Self {
        Self {
            seed: value.seed,
            n_ctx: value.n_ctx,
            n_batch: value.n_batch,
            n_ubatch: value.n_ubatch,
            n_seq_max: value.n_seq_max,
            n_threads: value.n_threads,
            n_threads_batch: value.n_threads_batch,
            rope_scaling_type: value.rope_scaling_type.into(),
            rope_freq_base: value.rope_freq_base,
            rope_freq_scale: value.rope_freq_scale,
            yarn_ext_factor: value.yarn_ext_factor,
            yarn_attn_factor: value.yarn_attn_factor,
            yarn_beta_fast: value.yarn_beta_fast,
            yarn_beta_slow: value.yarn_beta_slow,
            yarn_orig_ctx: value.yarn_orig_ctx,
            type_k: value.type_k.into(),
            type_v: value.type_v.into(),
            embedding: value.embeddings,
            offload_kqv: value.offload_kqv,
            pooling: value.pooling_type.into(),
            defrag_threshold: value.defrag_thold,
        }
    }
}
