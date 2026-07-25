use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=cuda/common.cuh");
    println!("cargo:rerun-if-changed=cuda/micro_block_fused.cu");
    println!("cargo:rerun-if-changed=cuda/particle_router.cu");
    println!("cargo:rerun-if-changed=cuda/particle_aggregator.cu");

    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok();

    if cuda_enabled {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

        let cu_files = vec![
            "cuda/micro_block_fused.cu",
            "cuda/particle_router.cu",
            "cuda/particle_aggregator.cu",
        ];

        // Step 1: Check if NVCC is installed
        let nvcc_check = Command::new("nvcc").arg("--version").output();

        if let Ok(output) = nvcc_check {
            if output.status.success() {
                let obj_ext = if target_os == "windows" { "obj" } else { "o" };
                let mut obj_files = Vec::new();

                // Step 2: Compile each .cu file into an object file using nvcc -c
                for cu_file in &cu_files {
                    let stem = Path::new(cu_file).file_stem().unwrap().to_str().unwrap();
                    let obj_file = out_dir.join(format!("{}.{}", stem, obj_ext));

                    let mut compile_cmd = Command::new("nvcc");
                    compile_cmd
                        .arg("-c")
                        .arg("-O3")
                        .arg("--use_fast_math")
                        .arg("-std=c++17")
                        .arg("-arch=sm_80");

                    if target_env == "msvc" {
                        // Match Rust MSVC dynamic C runtime (/MD) to prevent LNK4098 LIBCMT warning
                        compile_cmd.arg("-Xcompiler").arg("/MD");
                    } else if target_os != "windows" {
                        compile_cmd.arg("-Xcompiler").arg("-fPIC");
                    }

                    compile_cmd.arg("-o").arg(&obj_file);
                    compile_cmd.arg(cu_file);

                    println!("Cargo NVCC Compile step: {:?}", compile_cmd);
                    let status = compile_cmd.status();
                    match status {
                        Ok(s) if s.success() => {
                            obj_files.push(obj_file);
                        }
                        _ => {
                            println!("cargo:warning=Failed to compile CUDA file {}", cu_file);
                            return;
                        }
                    }
                }

                // Step 3: Package object files into a static library via nvcc -lib
                let lib_name = if target_os == "windows" {
                    "annp_cuda.lib"
                } else {
                    "libannp_cuda.a"
                };
                let out_lib_path = out_dir.join(lib_name);

                let mut lib_cmd = Command::new("nvcc");
                lib_cmd.arg("-lib").arg("-o").arg(&out_lib_path);
                for obj in &obj_files {
                    lib_cmd.arg(obj);
                }

                println!("Cargo NVCC Lib step: {:?}", lib_cmd);
                if let Ok(lib_status) = lib_cmd.status() {
                    if lib_status.success() {
                        println!("cargo:rustc-link-search=native={}", out_dir.display());
                        println!("cargo:rustc-link-lib=static=annp_cuda");
                        println!("cargo:rustc-link-lib=cudart");
                        return;
                    }
                }
            }
        }
        println!("cargo:warning=CUDA feature requested but nvcc compilation failed or nvcc was not found. Falling back to Rust CPU runtime.");
    }
}
