use std::env;
use std::path::{Path, PathBuf};

use cc::Build;
use once_cell::sync::Lazy;

#[cfg(all(
    feature = "metal",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "hipblas",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"metal\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "cuda",
    any(
        feature = "metal",
        feature = "blas",
        feature = "hipblas",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"cuda\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "blas",
    any(
        feature = "cuda",
        feature = "metal",
        feature = "hipblas",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"blas\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "hipblas",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "metal",
        feature = "clblast",
        feature = "vulkan"
    )
))]
compile_error!("feature \"hipblas\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "clblast",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "hipblas",
        feature = "metal",
        feature = "vulkan"
    )
))]
compile_error!("feature \"clblas\" cannot be enabled alongside other GPU based features");

#[cfg(all(
    feature = "vulkan",
    any(
        feature = "cuda",
        feature = "blas",
        feature = "hipblas",
        feature = "clblast",
        feature = "metal"
    )
))]
compile_error!("feature \"vulkan\" cannot be enabled alongside other GPU based features");

static LLAMA_PATH: Lazy<PathBuf> = Lazy::new(|| PathBuf::from("./thirdparty/llama.cpp"));

fn compile_bindings(out_path: &Path) {
    println!("Generating bindings..");
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

/// Add platform appropriate flags and definitions present in all compilation configurations.
fn push_common_flags(cx: &mut Build, cxx: &mut Build) {
    cx.static_flag(true).cpp(false).std("c11");
    cxx.static_flag(true).cpp(true).std("c++14"); // MSVC does not support C++11

    if cfg!(target_family = "unix") {
        cx.flag("-pthread")
            .flag("-Wall")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .flag("-Wcast-qual")
            .flag("-Wdouble-promotion")
            .flag("-Wshadow")
            .flag("-Wstrict-prototypes")
            .flag("-Wpointer-arith");
        cxx.flag("-fPIC")
            .flag("-pthread")
            .flag("-Wall")
            .flag("-Wdeprecated-declarations")
            .flag("-Wextra")
            .flag("-Wpedantic")
            .flag("-Wcast-qual")
            .flag("-Wno-unused-function")
            .flag("-Wno-multichar");
    } else if cfg!(target_family = "windows") {
        cx.flag("/W4")
            .flag("/Wall")
            .flag("/wd4820")
            .flag("/wd4710")
            .flag("/wd4711")
            .flag("/wd4820")
            .flag("/wd4514");
        cxx.flag("/W4")
            .flag("/Wall")
            .flag("/wd4820")
            .flag("/wd4710")
            .flag("/wd4711")
            .flag("/wd4820")
            .flag("/wd4514");
    }

    if cfg!(target_os = "openbsd") {
        cx.define("_XOPEN_SOURCE", "700");
        cxx.define("_XOPEN_SOURCE", "700");
    } else {
        cx.define("_XOPEN_SOURCE", "600");
        cxx.define("_XOPEN_SOURCE", "600");
    }

    if cfg!(target_os = "linux") {
        cx.define("_GNU_SOURCE", None);
        cxx.define("_GNU_SOURCE", None);
    } else if cfg!(any(
        target_os = "macos",
        target_os = "ios",
        target_os = "dragonfly"
    )) {
        cx.define("_DARWIN_C_SOURCE", None);
        cxx.define("_DARWIN_C_SOURCE", None);
    } else if cfg!(target_os = "openbsd") {
        cx.define("_BSD_SOURCE", None);
        cxx.define("_BSD_SOURCE", None);
    } else if cfg!(target_os = "freebsd") {
        cx.define("__BSD_VISIBLE", None);
        cxx.define("__BSD_VISIBLE", None);
    } else if cfg!(target_os = "netbsd") {
        cx.define("_NETBSD_SOURCE", None);
        cxx.define("_NETBSD_SOURCE", None);
    }

    if cfg!(any(target_arch = "arm", target_arch = "aarch64")) {
        if cfg!(target_family = "unix") {
            cx.flag("-mavx512vnni").flag("-mfp16-format=ieee");
            cxx.flag("-mavx512vnni").flag("-mfp16-format=ieee");
        } else if cfg!(target_family = "windows") {
            cx.define("__ARM_NEON", None)
                .define("__ARM_FEATURE_FMA", None)
                .define("__ARM_FEATURE_DOTPROD", None)
                .define("__aarch64__", None);
            cxx.define("__ARM_NEON", None)
                .define("__ARM_FEATURE_FMA", None)
                .define("__ARM_FEATURE_DOTPROD", None)
                .define("__aarch64__", None);
        }
    }
}

/// Add platform appropriate flags and definitions based on enabled features.
fn push_feature_flags(cx: &mut Build, cxx: &mut Build) {
    // TODO in llama.cpp's cmake (https://github.com/ggerganov/llama.cpp/blob/9ecdd12e95aee20d6dfaf5f5a0f0ce5ac1fb2747/CMakeLists.txt#L659), they include SIMD instructions manually, however it doesn't seem to be necessary for VS2022's MSVC, check when it is needed

    if cfg!(any(target_arch = "x86", target_arch = "x86_64")) {
        if cfg!(feature = "native") && cfg!(target_family = "unix") {
            cx.flag("-march=native");
            cxx.flag("-march=native");
        }

        if cfg!(feature = "fma") && cfg!(target_family = "unix") {
            cx.flag("-mfma");
            cxx.flag("-mfma");
        }

        if cfg!(feature = "f16c") && cfg!(target_family = "unix") {
            cx.flag("-mf16c");
            cxx.flag("-mf16c");
        }

        if cfg!(target_family = "unix") {
            if cfg!(feature = "avx512") {
                cx.flag("-mavx512f").flag("-mavx512bw");
                cxx.flag("-mavx512f").flag("-mavx512bw");

                if cfg!(feature = "avx512_vmbi") {
                    cx.flag("-mavx512vbmi");
                    cxx.flag("-mavx512vbmi");
                }

                if cfg!(feature = "avx512_vnni") {
                    cx.flag("-mavx512vnni");
                    cxx.flag("-mavx512vnni");
                }
            }

            if cfg!(feature = "avx2") {
                cx.flag("-mavx2");
                cxx.flag("-mavx2");
            }

            if cfg!(feature = "avx") {
                cx.flag("-mavx");
                cxx.flag("-mavx");
            }
        } else if cfg!(target_family = "windows") {
            if cfg!(feature = "avx512") {
                cx.flag("/arch:AVX512");
                cxx.flag("/arch:AVX512");

                if cfg!(feature = "avx512_vmbi") {
                    cx.define("__AVX512VBMI__", None);
                    cxx.define("__AVX512VBMI__", None);
                }

                if cfg!(feature = "avx512_vnni") {
                    cx.define("__AVX512VNNI__", None);
                    cxx.define("__AVX512VNNI__", None);
                }
            } else if cfg!(feature = "avx2") {
                cx.flag("/arch:AVX2");
                cxx.flag("/arch:AVX2");
            } else if cfg!(feature = "avx") {
                cx.flag("/arch:AVX");
                cxx.flag("/arch:AVX");
            }
        }
    }
}

fn compile_opencl(_cx: &mut Build, _cxx: &mut Build) {
    println!("Compiling OpenCL GGML..");

    todo!();

    // cx.flag("-DGGML_USE_CLBLAST");
    // cxx.flag("-DGGML_USE_CLBLAST");
    //
    // if cfg!(target_os = "linux") {
    //     println!("cargo:rustc-link-lib=OpenCL");
    //     println!("cargo:rustc-link-lib=clblast");
    // } else if cfg!(target_os = "macos") {
    //     println!("cargo:rustc-link-lib=framework=OpenCL");
    //     println!("cargo:rustc-link-lib=clblast");
    // }
    //
    // cxx.file("./llama.cpp/ggml-opencl.cpp");
}

fn compile_openblas(_cx: &mut Build) {
    println!("Compiling OpenBLAS GGML..");

    todo!();

    // cx.flag("-DGGML_USE_OPENBLAS")
    //     .include("/usr/local/include/openblas")
    //     .include("/usr/local/include/openblas");
    // println!("cargo:rustc-link-lib=openblas");
}

fn compile_blis(_cx: &mut Build) {
    println!("Compiling BLIS GGML..");

    todo!();

    // cx.flag("-DGGML_USE_OPENBLAS")
    //     .include("/usr/local/include/blis")
    //     .include("/usr/local/include/blis");
    // println!("cargo:rustc-link-search=native=/usr/local/lib");
    // println!("cargo:rustc-link-lib=blis");
}

fn _compile_cuda(_cxx_flags: &str) {
    println!("Compiling CUDA GGML..");

    todo!();

    // println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    // println!("cargo:rustc-link-search=native=/opt/cuda/lib64");
    //
    // if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
    //     println!(
    //         "cargo:rustc-link-search=native={}/targets/x86_64-linux/lib",
    //         cuda_path
    //     );
    // }
    //
    // let libs = "cublas culibos cudart cublasLt pthread dl rt";
    //
    // for lib in libs.split_whitespace() {
    //     println!("cargo:rustc-link-lib={}", lib);
    // }
    //
    // let mut nvcc = cc::Build::new();
    //
    // let env_flags = vec![
    //     ("LLAMA_CUDA_DMMV_X=32", "-DGGML_CUDA_DMMV_X"),
    //     ("LLAMA_CUDA_DMMV_Y=1", "-DGGML_CUDA_DMMV_Y"),
    //     ("LLAMA_CUDA_KQUANTS_ITER=2", "-DK_QUANTS_PER_ITERATION"),
    // ];
    //
    // let nvcc_flags = "--forward-unknown-to-host-compiler -arch=native ";
    //
    // for nvcc_flag in nvcc_flags.split_whitespace() {
    //     nvcc.flag(nvcc_flag);
    // }
    //
    // for cxx_flag in cxx_flags.split_whitespace() {
    //     nvcc.flag(cxx_flag);
    // }
    //
    // for env_flag in env_flags {
    //     let mut flag_split = env_flag.0.split("=");
    //     if let Ok(val) = std::env::var(flag_split.next().unwrap()) {
    //         nvcc.flag(&format!("{}={}", env_flag.1, val));
    //     } else {
    //         nvcc.flag(&format!("{}={}", env_flag.1, flag_split.next().unwrap()));
    //     }
    // }
    //
    // nvcc.compiler("nvcc")
    //     .file(LLAMA_PATH.join("ggml-cuda.cu"))
    //     .flag("-Wno-pedantic")
    //     .include(LLAMA_PATH.join("ggml-cuda.h"))
    //     .compile("ggml-cuda");
}

fn compile_metal(_cx: &mut Build, _cxx: &mut Build) {
    println!("Compiling Metal GGML..");

    todo!();

    // cx.flag("-DGGML_USE_METAL").flag("-DGGML_METAL_NDEBUG");
    // cxx.flag("-DGGML_USE_METAL");
    //
    // println!("cargo:rustc-link-lib=framework=Metal");
    // println!("cargo:rustc-link-lib=framework=Foundation");
    // println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");
    // println!("cargo:rustc-link-lib=framework=MetalKit");
    //
    // cx.include(LLAMA_PATH.join("ggml-metal.h"))
    //     .file(LLAMA_PATH.join("ggml-metal.m"));
}

fn compile_vulkan(cxx: &mut Build) {
    println!("Compiling Vulkan GGML..");
    println!("cargo:rerun-if-env-changed=VULKAN_SDK");

    if let Some(sdk_path) = option_env!("VULKAN_SDK") {
        println!("Found Vulkan SDK path in system: \"{sdk_path}\"");
        let (lib_folder, lib_name) = if cfg!(target_os = "windows") {
            let folder = if cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64") {
                "Lib"
            } else {
                "Lib32"
            };
            (folder, "vulkan-1")
        } else {
            ("lib", "vulkan")
        };

        println!("cargo:rustc-link-search={sdk_path}/{lib_folder}");
        println!("cargo:rustc-link-lib={lib_name}");
    } else {
        todo!()
    };

    cxx.include("./thirdparty/Vulkan-Headers/include/")
        .file(LLAMA_PATH.join("ggml-vulkan.cpp"))
        .define("GGML_USE_VULKAN", None);
}

fn compile_ggml(mut cx: Build) {
    println!("Compiling GGML..");
    cx.include(LLAMA_PATH.as_path())
        .file(LLAMA_PATH.join("ggml.c"))
        .file(LLAMA_PATH.join("ggml-alloc.c"))
        .file(LLAMA_PATH.join("ggml-backend.c"))
        .file(LLAMA_PATH.join("ggml-quants.c"))
        .compile("ggml");
}

fn compile_llama(mut cxx: Build, _out_path: impl AsRef<Path>, _ggml_type: &str) {
    println!("Compiling Llama.cpp..");
    cxx.include(LLAMA_PATH.as_path())
        .file(LLAMA_PATH.join("llama.cpp"))
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

    let mut cx = Build::new();
    let mut cxx = Build::new();

    push_common_flags(&mut cx, &mut cxx);
    push_feature_flags(&mut cx, &mut cxx);

    let mut ggml_type = String::new();

    if cfg!(feature = "vulkan") {
        compile_vulkan(&mut cxx);
    } else if cfg!(feature = "opencl") {
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
        todo!()

        // cx_flags.push("-DGGML_USE_CUBLAS");
        // cxx_flags.push("-DGGML_USE_CUBLAS");
        //
        // cx.include("/usr/local/cuda/include")
        //     .include("/opt/cuda/include");
        // cxx.include("/usr/local/cuda/include")
        //     .include("/opt/cuda/include");
        //
        // if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        //     cx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
        //     cxx.include(format!("{}/targets/x86_64-linux/include", cuda_path));
        // }
        //
        // compile_ggml(&mut cx, &cx_flags);
        //
        // compile_cuda(&cxx_flags);
        //
        // compile_llama(&mut cxx, &cxx_flags, &out_path, "cuda");
    } else {
        compile_ggml(cx);
        compile_llama(cxx, &out_path, &ggml_type);
    }

    #[cfg(feature = "compat")]
    {
        compat::redefine_symbols(out_path);
    }
}

#[cfg(feature = "compat")]
mod compat {
    use std::collections::HashSet;
    use std::process::Command;

    use crate::*;

    pub fn redefine_symbols(out_path: impl AsRef<Path>) {
        let (ggml_lib_name, llama_lib_name) = lib_names();
        let (nm_name, objcopy_name) = tool_names();
        println!(
            "Modifying {ggml_lib_name} and {llama_lib_name}, symbols acquired via \
        \"{nm_name}\" and modified via \"{objcopy_name}\""
        );

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

        let nm_name;
        println!("cargo:rerun-if-env-changed=NM_PATH");
        if let Some(path) = option_env!("NM_PATH") {
            nm_name = path;
        } else {
            println!("Looking for \"nm\" or an equivalent tool");
            nm_name = find_tool(&nm_names).expect(
                "No suitable tool equivalent to \"nm\" has been found in \
            PATH, if one is already installed, either add it to PATH or set NM_PATH to its full path",
            );
        }

        let objcopy_name;
        println!("cargo:rerun-if-env-changed=OBJCOPY_PATH");
        if let Some(path) = option_env!("OBJCOPY_PATH") {
            objcopy_name = path;
        } else {
            println!("Looking for \"objcopy\" or an equivalent tool..");
            objcopy_name = find_tool(&objcopy_names).expect("No suitable tool equivalent to \"objcopy\" has \
            been found in PATH, if one is already installed, either add it to PATH or set OBJCOPY_PATH to its full path");
        }

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
                    println!("Valid tool found:\n{out_str}");
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
    ) -> HashSet<&'a str> {
        let iter = nm_output
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
            .map(|symbol| &symbol[..symbol.len() - 2]); // Strip the type, so only the symbol remains

        HashSet::from_iter(iter)
    }
}
