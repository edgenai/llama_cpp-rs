# llama_cpp-rs

[![Documentation](https://docs.rs/llama_cpp/badge.svg)](https://docs.rs/llama_cpp/)
[![Crate](https://img.shields.io/crates/v/llama_cpp.svg)](https://crates.io/crates/llama_cpp)

Safe, high-level Rust bindings to the C++ project [of the same name](https://github.com/ggerganov/llama.cpp), meant to
be as user-friendly as possible. Run GGUF-based large language models directly on your CPU in fifteen lines of code, no
ML experience required!

```rust
// Create a model from anything that implements `AsRef<Path>`:
let model = LlamaModel::load_from_file("path_to_model.gguf", LlamaParams::default ()).expect("Could not load model");

// A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
// several gigabytes large, a session is typically a few dozen to a hundred megabytes!
let mut ctx = model.create_session(SessionParams::default ()).expect("Failed to create session");

// You can feed anything that implements `AsRef<[u8]>` into the model's context.
ctx.advance_context("This is the story of a man named Stanley.").unwrap();

// LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
let max_tokens = 1024;
let mut decoded_tokens = 0;

// `ctx.start_completing_with` creates a worker thread that generates tokens. When the completion
// handle is dropped, tokens stop generating!
let mut completions = ctx.start_completing_with(StandardSampler::default (), 1024).into_strings();

for completion in completions {
print!("{completion}");
let _ = io::stdout().flush();

decoded_tokens += 1;

if decoded_tokens > max_tokens {
break;
}
}
```

This repository hosts the high-level bindings (`crates/llama_cpp`) as well as automatically generated bindings to
llama.cpp's low-level C API (`crates/llama_cpp_sys`). Contributions are welcome--just keep the UX clean!

## Building

Keep in mind that [llama.cpp](https://github.com/ggerganov/llama.cpp) is very computationally heavy, meaning standard
debug builds (running just `cargo build`/`cargo run`) will suffer greatly from the lack of optimisations. Therefore,
unless
debugging is really necessary, it is highly recommended to build and run using Cargo's `--release` flag.

### Cargo Features

Several of [llama.cpp](https://github.com/ggerganov/llama.cpp)'s backends are supported through features:

- `cuda` - Enables the CUDA backend, the CUDA Toolkit is required for compilation if this feature is enabled.
- `vulkan` - Enables the Vulkan backend, the Vulkan SDK is required for compilation if this feature is enabled.
- `metal` - Enables the Metal backend, macOS only.
- `hipblas` - Enables the hipBLAS/ROCm backend, ROCm is required for compilation if this feature is enabled.

## Experimental

Something that's provided by these bindings is the ability to predict context size in memory, however it should be
noted that this is a highly experimental feature as this isn't something
that [llama.cpp](https://github.com/ggerganov/llama.cpp) itself provides.
The returned values may be highly inaccurate, however an attempt is made to never return values lower than the real
size.

## License

MIT or Apache-2.0, at your option (the "Rust" license). See `LICENSE-MIT` and `LICENSE-APACHE`.