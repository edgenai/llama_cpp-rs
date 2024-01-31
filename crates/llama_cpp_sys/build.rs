use std::env;
use std::path::{Path, PathBuf};

use cc::Build;
use once_cell::sync::Lazy;

#[cfg(all(feature = "metal", feature = "cuda"))]
compile_error!("feature \"metal\" and feature \"cuda\" cannot be enabled at the same time");
#[cfg(all(feature = "metal", feature = "blas"))]
compile_error!("feature \"metal\" and feature \"hipblas\" cannot be enabled at the same time");
#[cfg(all(feature = "metal", feature = "blas"))]
compile_error!("feature \"metal\" and feature \"hipblas\" cannot be enabled at the same time");
#[cfg(all(feature = "metal", feature = "clblast"))]
compile_error!("feature \"metal\" and feature \"clblast\" cannot be enabled at the same time");

#[cfg(all(feature = "cuda", feature = "blas"))]
compile_error!("feature \"metal\" and feature \"blas\" cannot be enabled at the same time");
#[cfg(all(feature = "cuda", feature = "hipblas"))]
compile_error!("feature \"cuda\" and feature \"hipblas\" cannot be enabled at the same time");
#[cfg(all(feature = "cuda", feature = "clblast"))]
compile_error!("feature \"cuda\" and feature \"clblast\" cannot be enabled at the same time");

#[cfg(all(feature = "hipblas", feature = "blas"))]
compile_error!("feature \"hipblas\" and feature \"blas\" cannot be enabled at the same time");
#[cfg(all(feature = "hipblas", feature = "clblast"))]
compile_error!("feature \"hipblas\" and feature \"clblast\" cannot be enabled at the same time");

#[cfg(all(feature = "blas", feature = "clblast"))]
compile_error!("feature \"blas\" and feature \"clblast\" cannot be enabled at the same time");

static LLAMA_PATH: Lazy<PathBuf> = Lazy::new(|| PathBuf::from("./thirdparty/llama.cpp"));

fn compile_bindings(out_path: &Path) {
    let bindings = bindgen::Builder::default()
        .header(LLAMA_PATH.join("ggml.h").to_string_lossy())
        .header(LLAMA_PATH.join("llama.h").to_string_lossy())
        .allowlist_type("ggml_.*")
        .allowlist_function("llama_.*")
        .allowlist_type("llama_.*")
        .parse_callbacks(Box::new(
            bindgen::CargoCallbacks::new().rerun_on_header_files(false),
        ))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn compile_opencl(cx: &mut Build, cxx: &mut Build) {
    cx.flag("-DGGML_USE_CLBLAST");
    cxx.flag("-DGGML_USE_CLBLAST");

    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
        println!("cargo:rustc-link-lib=clblast");
    }

    cxx.file("./llama.cpp/ggml-opencl.cpp");
}

fn compile_openblas(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/openblas")
        .include("/usr/local/include/openblas");
    println!("cargo:rustc-link-lib=openblas");
}

fn compile_blis(cx: &mut Build) {
    cx.flag("-DGGML_USE_OPENBLAS")
        .include("/usr/local/include/blis")
        .include("/usr/local/include/blis");
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=blis");
}

fn compile_cuda(cxx_flags: &str) {
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-search=native=/opt/cuda/lib64");

    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        println!(
            "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
            cuda_path
        );
    }

    let libs = "cublas culibos cudart cublasLt pthread dl rt";

    for lib in libs.split_whitespace() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    let mut nvcc = cc::Build::new();

    let env_flags = vec![
        ("LLAMA_CUDA_DMMV_X=32", "-DGGML_CUDA_DMMV_X"),
        ("LLAMA_CUDA_DMMV_Y=1", "-DGGML_CUDA_DMMV_Y"),
        ("LLAMA_CUDA_KQUANTS_ITER=2", "-DK_QUANTS_PER_ITERATION"),
    ];

    let nvcc_flags = "--forward-unknown-to-host-compiler -arch=native ";

    for nvcc_flag in nvcc_flags.split_whitespace() {
        nvcc.flag(nvcc_flag);
    }

    for cxx_flag in cxx_flags.split_whitespace() {
        nvcc.flag(cxx_flag);
    }

    for env_flag in env_flags {
        let mut flag_split = env_flag.0.split("=");
        if let Ok(val) = std::env::var(flag_split.next().unwrap()) {
            nvcc.flag(&format!("{}={}", env_flag.1, val));
        } else {
            nvcc.flag(&format!("{}={}", env_flag.1, flag_split.next().unwrap()));
        }
    }

    nvcc.compiler("nvcc")
        .file(LLAMA_PATH.join("ggml-cuda.cu"))
        .flag("-Wno-pedantic")
        .include(LLAMA_PATH.join("ggml-cuda.h"))
        .compile("ggml-cuda");
}

fn compile_ggml(cx: &mut Build, cx_flags: &str) {
    for cx_flag in cx_flags.split_whitespace() {
        cx.flag(cx_flag);
    }

    cx.include(LLAMA_PATH.as_path())
        .file(LLAMA_PATH.join("ggml.c"))
        .file(LLAMA_PATH.join("ggml-alloc.c"))
        .file(LLAMA_PATH.join("ggml-backend.c"))
        .file(LLAMA_PATH.join("ggml-quants.c"))
        .cpp(false)
        .static_flag(true)
        .define("_GNU_SOURCE", None)
        .define("_XOPEN_SOURCE", "600")
        .define("GGML_USE_K_QUANTS", None)
        .compile("ggml");
}

fn compile_metal(cx: &mut Build, cxx: &mut Build) {
    cx.flag("-DGGML_USE_METAL").flag("-DGGML_METAL_NDEBUG");
    cxx.flag("-DGGML_USE_METAL");

    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    println!("cargo:rustc-link-lib=framework=MetalKit");

    cx.include(LLAMA_PATH.join("ggml-metal.h"))
        .file(LLAMA_PATH.join("ggml-metal.m"));
}

fn compile_llama(cxx: &mut Build, cxx_flags: &str, _out_path: impl AsRef<Path>, _ggml_type: &str) {
    for cxx_flag in cxx_flags.split_whitespace() {
        cxx.flag(cxx_flag);
    }

    //let ggml_obj = out_path.as_ref().join("thirdparty/llama.cpp/ggml.o");

    //cxx.object(ggml_obj);

    /*if !ggml_type.is_empty() {
        let ggml_feature_obj = out_path
            .as_ref()
            .join(format!("thirdparty/llama.cpp/ggml-{}.o", ggml_type));
        cxx.object(ggml_feature_obj);
    }*/

    #[cfg(any(target_os = "macos", target_os = "ios", target_os = "dragonfly"))]
    {
        cxx.define("_DARWIN_C_SOURCE", None);
    }

    #[cfg(target_os = "linux")]
    {
        cxx.define("_GNU_SOURCE", None);
    }

    cxx.static_flag(true)
        .file(LLAMA_PATH.join("llama.cpp"))
        .cpp(true)
        .define("_XOPEN_SOURCE", "600")
        .compile("llama");
}

fn main() {
    if std::fs::read_dir(LLAMA_PATH.as_path()).is_err() {
        panic!(
            "Could not find {}. Did you forget to initialize submodules?",
            LLAMA_PATH.to_string_lossy()
        );
    }

    let out_path = PathBuf::from(env::var("OUT_DIR").expect("No out dir found"));

    compile_bindings(&out_path);

    let mut cx_flags = String::from("");
    let mut cxx_flags = String::from("");

    // check if os is linux
    // if so, add -fPIC to cxx_flags
    if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
        cx_flags.push_str(" -std=c11 -Wall -Wextra -Wpedantic -Wcast-qual -Wdouble-promotion -Wshadow -Wstrict-prototypes -Wpointer-arith -pthread");
        cxx_flags.push_str(" -std=c++11 -Wall -Wdeprecated-declarations -Wunused-but-set-variable -Wextra -Wpedantic -Wcast-qual -Wno-unused-function -Wno-multichar -fPIC -pthread");
    } else if cfg!(target_os = "windows") {
        cx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
        cxx_flags.push_str(" /W4 /Wall /wd4820 /wd4710 /wd4711 /wd4820 /wd4514");
    }

    // TODO in llama.cpp's cmake (https://github.com/ggerganov/llama.cpp/blob/9ecdd12e95aee20d6dfaf5f5a0f0ce5ac1fb2747/CMakeLists.txt#L659), they include SIMD instructions manually, however it doesn't seem to be necessary for VS2022's MSVC, check when it is needed
    // TODO add missing windows feature support

    #[cfg(feature = "native")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -march=native -mtune=native");
            cxx_flags.push_str(" -march=native -mtune=native");
        }
    }

    #[cfg(feature = "fma")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mfma");
            cxx_flags.push_str(" -mfma");
        }
    }

    #[cfg(feature = "f16c")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mf16c");
            cxx_flags.push_str(" -mf16c");
        }
    }

    #[cfg(feature = "avx")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mavx");
            cxx_flags.push_str(" -mavx");
        } else if !(cfg!(feature = "avx2") || cfg!(feature = "avx512")) {
            cx_flags.push_str(" /arch:AVX");
            cxx_flags.push_str(" /arch:AVX");
        }
    }

    #[cfg(feature = "avx2")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mavx2");
            cxx_flags.push_str(" -mavx2");
        } else if !cfg!(feature = "avx512") {
            cx_flags.push_str(" /arch:AVX2");
            cxx_flags.push_str(" /arch:AVX2");
        }
    }

    #[cfg(feature = "avx512")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mavx512f -mavx512bw");
            cxx_flags.push_str(" -mavx512f -mavx512bw");
        }
    }

    #[cfg(feature = "avx512_vmbi")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mavx512vbmi");
            cxx_flags.push_str(" -mavx512vbmi");
        }
    }

    #[cfg(feature = "avx512_vnni")]
    {
        if cfg!(target_os = "linux") || cfg!(target_os = "macos") {
            cx_flags.push_str(" -mavx512vnni");
            cxx_flags.push_str(" -mavx512vnni");
        }
    }

    let mut cx = cc::Build::new();

    let mut cxx = cc::Build::new();

    let mut ggml_type = String::new();

    cxx.include(LLAMA_PATH.join("common"))
        .include(LLAMA_PATH.as_path())
        .include("./include");

    if cfg!(feature = "opencl") {
        compile_opencl(&mut cx, &mut cxx);
        ggml_type = "opencl".to_string();
    } else if cfg!(feature = "openblas") {
        compile_openblas(&mut cx);
    } else if cfg!(feature = "blis") {
        compile_blis(&mut cx);
    } else if cfg!(feature = "metal") && cfg!(target_os = "macos") {
        compile_metal(&mut cx, &mut cxx);
        ggml_type = "metal".to_string();
    }

    if cfg!(feature = "cuda") {
        cx_flags.push_str(" -DGGML_USE_CUBLAS");
        cxx_flags.push_str(" -DGGML_USE_CUBLAS");

        cx.include("/usr/local/cuda/include")
            .include("/opt/cuda/include");
        cxx.include("/usr/local/cuda/include")
            .include("/opt/cuda/include");

        if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
            cx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
            cxx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
        }

        compile_ggml(&mut cx, &cx_flags);

        compile_cuda(&cxx_flags);

        compile_llama(&mut cxx, &cxx_flags, &out_path, "cuda");
    } else {
        compile_ggml(&mut cx, &cx_flags);

        compile_llama(&mut cxx, &cxx_flags, &out_path, &ggml_type);
    }

    #[cfg(feature = "compat")]
    {
        compat::redefine_symbols(out_path);
    }
}

#[cfg(feature = "compat")]
mod compat {
    use crate::*;
    use std::process::Command;

    pub fn redefine_symbols(out_path: impl AsRef<Path>) {
        // TODO this whole section is a bit hacky, could probably clean it up a bit, particularly the retrieval of symbols from the library files
        // TODO do this for cuda if necessary

        let (ggml_lib_name, llama_lib_name) = lib_names();
        let (nm_name, objcopy_name) = tool_names();
        println!("Modifying {ggml_lib_name} and {llama_lib_name}, symbols acquired via \"{nm_name}\" and modified via \"{objcopy_name}\"");

        // Modifying symbols exposed by the ggml library

        let output = Command::new(nm_name)
            .current_dir(&out_path)
            .arg(ggml_lib_name)
            .args(["-p", "-P"])
            .output()
            .expect("Failed to acquire symbols from the compiled library.");
        if !output.status.success() {
            panic!(
                "An error has occurred while acquiring symbols from the compiled library ({})",
                output.status
            );
        }
        let out_str = String::from_utf8_lossy(&output.stdout);
        let symbols = get_symbols(
            &out_str,
            [
                Filter {
                    prefix: "ggml",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "ggml",
                    sym_type: 'B',
                },
                Filter {
                    prefix: "gguf",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "quantize",
                    sym_type: 'T',
                },
                Filter {
                    prefix: "dequantize",
                    sym_type: 'T',
                },
            ],
        );

        let mut cmd = Command::new(objcopy_name);
        cmd.current_dir(&out_path);
        for symbol in symbols {
            cmd.arg(format!("--redefine-sym={symbol}=llama_{symbol}"));
        }
        let status = cmd
            .arg(ggml_lib_name)
            .status()
            .expect("Failed to modify global symbols from the ggml library.");
        if !status.success() {
            panic!(
                "An error as occurred while modifying global symbols from library file ({})",
                status
            );
        }

        // Modifying the symbols llama depends on from ggml

        let output = Command::new(nm_name)
            .current_dir(&out_path)
            .arg(llama_lib_name)
            .args(["-p", "-P"])
            .output()
            .expect("Failed to acquire symbols from the compiled library.");
        if !output.status.success() {
            panic!(
                "An error has occurred while acquiring symbols from the compiled library ({})",
                output.status
            );
        }
        let out_str = String::from_utf8_lossy(&output.stdout);
        let symbols = get_symbols(
            &out_str,
            [
                Filter {
                    prefix: "ggml",
                    sym_type: 'U',
                },
                Filter {
                    prefix: "gguf",
                    sym_type: 'U',
                },
            ],
        );

        let mut cmd = Command::new(objcopy_name);
        cmd.current_dir(&out_path);
        for symbol in symbols {
            cmd.arg(format!("--redefine-sym={symbol}=llama_{symbol}"));
        }
        let status = cmd
            .arg(llama_lib_name)
            .status()
            .expect("Failed to modify ggml symbols from library file.");
        if !status.success() {
            panic!(
                "An error has occurred while modifying ggml symbols from library file ({})",
                status
            );
        }
    }

    /// Returns *GGML*'s and *Llama.cpp*'s compiled library names, based on the operating system.
    fn lib_names() -> (&'static str, &'static str) {
        let ggml_lib_name;
        let llama_lib_name;
        if cfg!(target_family = "windows") {
            ggml_lib_name = "ggml.lib";
            llama_lib_name = "llama.lib";
        } else if cfg!(target_family = "unix") {
            ggml_lib_name = "libggml.a";
            llama_lib_name = "libllama.a";
        } else {
            println!("cargo:warning=Unknown target family, defaulting to Unix lib names");
            ggml_lib_name = "libggml.a";
            llama_lib_name = "libllama.a";
        };

        (ggml_lib_name, llama_lib_name)
    }

    /// Returns the names of tools equivalent to [nm][nm] and [objcopy][objcopy].
    ///
    /// [nm]: https://www.man7.org/linux/man-pages/man1/nm.1.html
    /// [objcopy]: https://www.man7.org/linux/man-pages/man1/objcopy.1.html
    fn tool_names() -> (&'static str, &'static str) {
        let nm_names;
        let objcopy_names;
        if cfg!(target_family = "unix") {
            nm_names = vec!["nm", "llvm-nm"];
            objcopy_names = vec!["objcopy", "llvm-objcopy"];
        } else {
            nm_names = vec!["llvm-nm"];
            objcopy_names = vec!["llvm-objcopy"];
        }

        println!("Looking for \"nm\" or an equivalent tool");
        let nm_name = find_tool(&nm_names).expect(
            "No suitable tool equivalent to \"nm\" has been found in \
        PATH, if one is already installed, either add it to PATH or set NM_PATH to its full path",
        );

        println!("Looking for \"objcopy\" or an equivalent tool");
        let objcopy_name = find_tool(&objcopy_names).expect("No suitable tool equivalent to \"objcopy\" has \
        been found in PATH, if one is already installed, either add it to PATH or set OBJCOPY_PATH to its full path");

        (nm_name, objcopy_name)
    }

    /// Returns the first tool found in the system, given a list of tool names, returning the first one found and
    /// printing its version.
    ///
    /// Returns [`Option::None`] if no tool is found.
    fn find_tool<'a>(names: &[&'a str]) -> Option<&'a str> {
        for name in names {
            if let Ok(output) = Command::new(name).arg("--version").output() {
                if output.status.success() {
                    let out_str = String::from_utf8_lossy(&output.stdout);
                    println!("Valid \"tool\" found:\n{out_str}");
                    return Some(name);
                }
            }
        }

        None
    }

    /// A filter for a symbol in a library.
    struct Filter<'a> {
        prefix: &'a str,
        sym_type: char,
    }

    /// Turns **`nm`**'s output into an iterator of [`str`] symbols.
    ///
    /// This function expects **`nm`** to be called using the **`-p`** and **`-P`** flags.
    fn get_symbols<'a, const N: usize>(
        nm_output: &'a str,
        filters: [Filter<'a>; N],
    ) -> impl Iterator<Item = &'a str> + 'a {
        nm_output
            .lines()
            .map(|symbol| {
                // Strip irrelevant information

                let mut stripped = symbol;
                while stripped.split(' ').count() > 2 {
                    let idx = unsafe { stripped.rfind(' ').unwrap_unchecked() };
                    stripped = &stripped[..idx]
                }
                stripped
            })
            .filter(move |symbol| {
                // Filter matching symbols

                if symbol.split(' ').count() == 2 {
                    for filter in &filters {
                        if symbol.ends_with(filter.sym_type) && symbol.starts_with(filter.prefix) {
                            return true;
                        }
                    }
                }
                false
            })
            .map(|symbol| &symbol[..symbol.len() - 2]) // Strip the type, so only the symbol remains
    }
}
