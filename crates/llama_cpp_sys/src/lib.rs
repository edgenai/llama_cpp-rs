//! System-level, **highly** `unsafe` bindings to
//! [llama.cpp](https://github.com/ggerganov/llama.cpp/tree/master).
//!
//! There's a **lot** of nuance here; for a safe alternative, see `llama_cpp`.
//!
//! You need `cmake`, a compatible `libc`, `libcxx`, `libcxxabi`, and `libclang` to build this
//! project, along with a C/C++ compiler toolchain.
//!
//! The code is automatically built for static _and_ dynamic linking using
//! [the `cmake` crate](https://docs.rs/cmake/), with C FFI bindings being generated with
//! [`bindgen`](https://docs.rs/bindgen/latest/bindgen/).

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// [`ash`] is only included to link to the Vulkan SDK.
#[allow(unused)]
#[cfg(feature = "vulkan")]
use ash;

// [`cudarc`] is only included to link to CUDA.
#[allow(unused)]
#[cfg(feature = "cuda")]
use cudarc;

extern crate link_cplusplus;

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
