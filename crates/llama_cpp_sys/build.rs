use std::path::{Path, PathBuf};
use std::process::exit;
use std::{env, fs, io};

const SUBMODULE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/thirdparty/llama.cpp");

fn copy_recursively(src: &Path, dst: &Path) -> io::Result<()> {
    if !dst.exists() {
        fs::create_dir_all(dst)?;
    }

    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;

        if file_type.is_dir() {
            copy_recursively(&entry.path(), &dst.join(entry.file_name()))?;
        } else {
            fs::copy(entry.path(), dst.join(entry.file_name()))?;
        }
    }

    Ok(())
}

fn main() {
    let submodule_dir = &PathBuf::from(SUBMODULE_DIR);

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());

    let build_dir = out_dir.join("build");
    let header_path = out_dir.join("build/llama.h");
    let git_path = build_dir.join(".git");
    let git_ignored_path = build_dir.join(".git-ignored");
    let build_info_path = build_dir.join("build-info.h");

    if fs::read_dir(submodule_dir).is_err() {
        eprintln!("Could not find {SUBMODULE_DIR}. Did you forget to initialize submodules?");

        exit(1);
    }

    if let Err(err) = fs::create_dir_all(&build_dir) {
        eprintln!("Could not create {build_dir:#?}: {err}");

        exit(1);
    }

    if let Err(err) = copy_recursively(submodule_dir, &build_dir) {
        eprintln!("Could not copy {submodule_dir:#?} into {build_dir:#?}: {err}");

        exit(1);
    }

    // TODO(scriptis): This is a gross hack. `llama.cpp` tries to read `.git` and create/update
    // `build-info.h` in the root of itself, which breaks Cargo's "don't modify the source tree
    // in `build.rs` rule. This is a workaround for that: get rid of `.git` and manually create
    // `build-info.h`.
    if git_path.exists() {
        fs::rename(&git_path, &git_ignored_path).unwrap();
    }

    if !build_info_path.exists() {
        fs::write(build_info_path, "\
            #ifndef BUILD_INFO_H
            #define BUILD_INFO_H

            #define BUILD_NUMBER 1
            #define BUILD_COMMIT \"ffffffff\"
            #define BUILD_COMPILER \"rustc\"
            #define BUILD_TARGET \"rustc\"

            #endif // BUILD_INFO_H

        ").unwrap();
    }
    let dst = cmake::Config::new(&build_dir)
        .configure_arg("-DLLAMA_STATIC=On")
        .configure_arg("-DLLAMA_BUILD_EXAMPLES=Off")
        .configure_arg("-DLLAMA_BUILD_SERVER=Off")
        .configure_arg("-DLLAMA_BUILD_TESTS=Off")
        .build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-search=native={}/lib64", dst.display());
    println!("cargo:rustc-link-lib=static=llama");

    let bindings = bindgen::Builder::default()
        .header(header_path.to_string_lossy())
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
