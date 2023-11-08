# llama_cpp-rs

[![Documentation](https://docs.rs/llama_cpp/badge.svg)](https://docs.rs/llama_cpp/)
[![Crate](https://img.shields.io/crates/v/llama_cpp.svg)](https://crates.io/crates/llama_cpp)

Safe, high-level Rust bindings to the C++ project [of the same name](https://github.com/ggerganov/llama.cpp), meant to
be as user-friendly as possible. Run GGUF-based large language models directly on your CPU in fifteen lines of code, no
ML experience required!

```rust
// Create a model from anything that implements `AsRef<Path>`:
let model = LlamaModel::load_from_file("path_to_model.gguf").expect("Could not load model");

// A `LlamaModel` holds the weights shared across many _sessions_; while your model may be
// several gigabytes large, a session is typically a few dozen to a hundred megabytes!
let mut ctx = model.create_session();

// You can feed anything that implements `AsRef<[u8]>` into the model's context.
ctx.advance_context("This is the story of a man named Stanley.").unwrap();

// LLMs are typically used to predict the next word in a sequence. Let's generate some tokens!
let max_tokens = 1024;
let mut decoded_tokens = 0;

// `ctx.get_completions` creates a worker thread that generates tokens. When the completion
// handle is dropped, tokens stop generating!
let mut completions = ctx.get_completions();

while let Some(next_token) = completions.detokenize() {
    println!("{}", String::from_utf8_lossy(&*next_token.as_bytes()));
    decoded_tokens += 1;
    if decoded_tokens > max_tokens {
        break;
    }
}
```

This repository hosts the high-level bindings (`crates/llama_cpp`) as well as automatically generated bindings to
llama.cpp's low-level C API (`crates/llama_cpp_sys`). Contributions are welcome--just keep the UX clean!

## License

MIT or Apache-2.0, at your option (the "Rust" license). See `LICENSE-MIT` and `LICENSE-APACHE`.