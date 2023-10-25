use std::path::PathBuf;
use std::process::exit;
use std::{env, fs};

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/llama.cpp");
const HEADER_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/llama.cpp/llama.h");

fn main() {
    if fs::read_dir(SUBMODULE_DIR).is_err() {
        eprintln!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");

        exit(1);
    }

    let dst = cmake::Config::new(SUBMODULE_DIR)
        .configure_arg("DLLAMA_STATIC=On")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=llama");

    let bindings = bindgen::Builder::default()
        .header(HEADER_DIR)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate_comments(false)
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .allowlist_function("ggml_.*")
        .allowlist_type("ggml_.*")
        .clang_arg("-xc++")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
