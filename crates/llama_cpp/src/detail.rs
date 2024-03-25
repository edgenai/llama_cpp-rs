//! FFI implementation details.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

use std::ffi::{c_char, c_void, CStr};

#[cfg(not(feature = "sys_verbosity"))]
use tracing::info;
#[cfg(feature = "sys_verbosity")]
use tracing::trace;
use tracing::{debug, error, warn};

use llama_cpp_sys::ggml_log_level;

#[allow(improper_ctypes_definitions)]
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
        #[cfg(feature = "sys_verbosity")]
        ggml_log_level::GGML_LOG_LEVEL_DEBUG => trace!(target: "llama.cpp", "{text}"),
        #[cfg(feature = "sys_verbosity")]
        ggml_log_level::GGML_LOG_LEVEL_INFO => debug!(target: "llama.cpp", "{text}"),
        #[cfg(not(feature = "sys_verbosity"))]
        ggml_log_level::GGML_LOG_LEVEL_DEBUG => debug!(target: "llama.cpp", "{text}"),
        #[cfg(not(feature = "sys_verbosity"))]
        ggml_log_level::GGML_LOG_LEVEL_INFO => info!(target: "llama.cpp", "{text}"),
        ggml_log_level::GGML_LOG_LEVEL_WARN => warn!(target: "llama.cpp", "{text}"),
        ggml_log_level::GGML_LOG_LEVEL_ERROR => error!(target: "llama.cpp", "{text}"),
        _ => unimplemented!(),
    }
}
