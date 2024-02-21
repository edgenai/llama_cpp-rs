//! Implements the [`Backend`] and [`BackendRef`] structs for managing llama.cpp
//! backends

use std::ptr;

use futures::executor::block_on;
use tokio::sync::Mutex;
use tracing::error;

use llama_cpp_sys::{llama_backend_free, llama_backend_init, llama_log_set};

use crate::detail;

/// The current instance of [`Backend`], if it exists. Also stored is a reference count used for
/// initialisation and freeing.
static BACKEND: Mutex<Option<(Backend, usize)>> = Mutex::const_new(None);

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
            llama_backend_init(true);

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
    pub(crate) async fn new() -> Self {
        let mut lock = BACKEND.lock().await;
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
        block_on(async move {
            let mut lock = BACKEND.lock().await;
            if let Some((_, count)) = lock.as_mut() {
                *count -= 1;

                if *count == 0 {
                    lock.take();
                }
            } else {
                error!("Backend as already been freed, this should never happen")
            }
        });
    }
}

impl Clone for BackendRef {
    fn clone(&self) -> Self {
        block_on(Self::new())
    }
}
