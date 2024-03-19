//! FFI implementation details.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CStr};

use tracing::{error, info, trace, warn};

use llama_cpp_sys::{
    ggml_log_level, ggml_log_level_GGML_LOG_LEVEL_ERROR, ggml_log_level_GGML_LOG_LEVEL_INFO,
    ggml_log_level_GGML_LOG_LEVEL_WARN,
};

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
