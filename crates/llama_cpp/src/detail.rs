//! FFI implementation details.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr::slice_from_raw_parts;

use tokio::sync::mpsc::UnboundedSender;
use tracing::{error, info, trace, warn};

use llama_cpp_sys::{
    ggml_log_level, ggml_log_level_GGML_LOG_LEVEL_ERROR, ggml_log_level_GGML_LOG_LEVEL_INFO,
    ggml_log_level_GGML_LOG_LEVEL_WARN, llama_beams_state,
};

use crate::Token;

pub(crate) struct BeamSearchState {
    pub(crate) tx: UnboundedSender<Token>,
}

#[no_mangle]
pub(crate) unsafe extern "C" fn llama_beam_search_callback(
    shared_state_ptr: *mut c_void,
    beam_state: llama_beams_state,
) {
    let shared_state = unsafe {
        // SAFETY: `channel` has this type and hasn't been de-allocated.
        &mut *(shared_state_ptr as *mut BeamSearchState)
    };

    if shared_state.tx.is_closed() {
        // Close all beams to terminate the search.
        for i in 0..beam_state.n_beams {
            unsafe {
                // SAFETY: beam_views[i] exists where 0 <= i <= n_beams.
                *beam_state.beam_views.add(i)
            }
            .eob = true;
        }
    }

    // Llama.cpp trims the common prefix after every invocation; the presence of
    // `common_prefix_length > 0` means the first `common_prefix_length` tokens have been
    // settled upon.
    if beam_state.common_prefix_length > 0 {
        let first_beam = unsafe {
            // SAFETY: At least one beam always exists.
            &*(beam_state.beam_views)
        };

        let beam_tokens = unsafe {
            // SAFETY: If all beams share a common prefix, at least that many tokens exist in
            // every beam.
            &*slice_from_raw_parts(first_beam.tokens, beam_state.common_prefix_length)
        };

        for unshared_token in beam_tokens {
            let _ = shared_state.tx.send(Token(*unshared_token));
        }
    }

    if beam_state.last_call {
        unsafe {
            // SAFETY: `channel` is heap-allocated, and this is the only time we'll construct
            // a `Box` back over it; this is the last time this function will be called, and
            // the last time this pointer will be seen.
            let _ = Box::from_raw(shared_state);
        }
    }
}

#[no_mangle]
pub(crate) unsafe extern "C" fn llama_log_callback(
    level: ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    let text = unsafe {
        // SAFETY: `text` is a NUL-terminated C String.
        CStr::from_ptr(text)
    };
    let text = String::from_utf8_lossy(text.to_bytes());

    // TODO check if this happens due to some bug
    if text.len() < 2 {
        return;
    }

    let text = if let Some(stripped) = text.strip_suffix('\n') {
        stripped
    } else {
        text.as_ref()
    };

    match level {
        ggml_log_level_GGML_LOG_LEVEL_ERROR => error!(target: "llama.cpp", "{text}"),
        ggml_log_level_GGML_LOG_LEVEL_INFO => info!(target: "llama.cpp", "{text}"),
        ggml_log_level_GGML_LOG_LEVEL_WARN => warn!(target: "llama.cpp", "{text}"),
        _ => trace!("ggml: {text}"),
    }
}
