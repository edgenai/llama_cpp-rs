//! Implements the [`Backend`] and [`BackendRef`] structs for managing llama.cpp
//! backends

use std::ptr;

use std::sync::Mutex;
use tracing::error;

use llama_cpp_sys::{
    ggml_numa_strategy_GGML_NUMA_STRATEGY_DISTRIBUTE, llama_backend_free, llama_backend_init,
    llama_log_set, llama_numa_init,
};

use crate::detail;

/// The current instance of [`Backend`], if it exists. Also stored is a reference count used for
/// initialisation and freeing.
static BACKEND: Mutex<Option<(Backend, usize)>> = Mutex::new(None);

/// Empty struct used to initialise and free the [llama.cpp][llama.cpp] backend when it is created
/// dropped respectively.
///
/// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
struct Backend {}

impl Backend {
    /// Initialises the [llama.cpp][llama.cpp] backend and sets its logger.
    ///
    /// There should only ever be one instance of this struct at any given time.
    ///
    /// [llama.cpp]: https://github.com/ggerganov/llama.cpp/
    fn init() -> Self {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_init();

            // TODO look into numa strategies, this should probably be part of the API
            llama_numa_init(ggml_numa_strategy_GGML_NUMA_STRATEGY_DISTRIBUTE);

            // SAFETY: performs a simple assignment to static variables. Should only execute once
            // before any logs are made.
            llama_log_set(Some(detail::llama_log_callback), ptr::null_mut());
        }

        Self {}
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: This is only called when no models or sessions exist.
            llama_backend_free();
        }
    }
}

/// A "reference" to [`BACKEND`].
///
/// Initialises [`BACKEND`] if there is no [`Backend`] inside. If there are no other references,
/// this drops [`Backend`] upon getting itself dropped.
pub(crate) struct BackendRef {}

impl BackendRef {
    /// Creates a new reference, initialising [`BACKEND`] if necessary.
    pub(crate) fn new() -> Self {
        let mut lock = BACKEND.lock().unwrap();
        if let Some((_, count)) = lock.as_mut() {
            *count += 1;
        } else {
            let _ = lock.insert((Backend::init(), 1));
        }

        Self {}
    }
}

impl Drop for BackendRef {
    fn drop(&mut self) {
        let mut lock = BACKEND.lock().unwrap();
        if let Some((_, count)) = lock.as_mut() {
            *count -= 1;

            if *count == 0 {
                lock.take();
            }
        } else {
            error!("Backend as already been freed, this should never happen")
        }
    }
}

impl Clone for BackendRef {
    fn clone(&self) -> Self {
        Self::new()
    }
}
