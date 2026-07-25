use std::env;
use std::path::PathBuf;
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

        let nvcc_check = Command::new("nvcc").arg("--version").status();

        if let Ok(status) = nvcc_check {
            if status.success() {
                let lib_name = if target_os == "windows" {
                    "annp_cuda.lib"
                } else {
                    "libannp_cuda.a"
                };

                let out_lib_path = out_dir.join(lib_name);

                let mut nvcc_cmd = Command::new("nvcc");
                nvcc_cmd
                    .arg("-lib")
                    .arg("-O3")
                    .arg("--use_fast_math")
                    .arg("-std=c++17")
                    .arg("-arch=sm_80");

                if target_env == "msvc" {
                    // Match Rust MSVC runtime (/MD) to eliminate LIBCMT LNK4098 linker warnings
                    nvcc_cmd.arg("-Xcompiler").arg("/MD");
                }

                nvcc_cmd.arg("-o").arg(&out_lib_path);
                for cu_file in &cu_files {
                    nvcc_cmd.arg(cu_file);
                }

                println!("Compiling CUDA kernels via NVCC: {:?}", nvcc_cmd);
                let compile_status = nvcc_cmd.status();

                if let Ok(c_status) = compile_status {
                    if c_status.success() {
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
