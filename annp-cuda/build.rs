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

        let cu_files = vec![
            "cuda/micro_block_fused.cu",
            "cuda/particle_router.cu",
            "cuda/particle_aggregator.cu",
        ];

        let nvcc_status = Command::new("nvcc").arg("--version").status();

        if let Ok(status) = nvcc_status {
            if status.success() {
                let mut builder = cc::Build::new();
                #[cfg(unix)]
                builder
                    .cuda(true)
                    .flag("-O3")
                    .flag("-std=c++17")
                    .flag("--use_fast_math")
                    .flag("-arch=sm_80") // Default targeting Ampere / Hopper architecture
                    .files(cu_files);

                #[cfg(windows)]
                builder
                    .cuda(true)
                    .flag("-O3")
                    .flag("-allow-unsupported-compiler")
                    .flag("-std=c++17")
                    .flag("--use_fast_math")
                    .flag("-arch=sm_80") // Default targeting Ampere / Hopper architecture
                    .files(cu_files);

                builder.compile("libannp_cuda.a");

                println!("cargo:rustc-link-search=native={}", out_dir.display());
                println!("cargo:rustc-link-lib=static=annp_cuda");
                println!("cargo:rustc-link-lib=cudart");
                return;
            }
        }
        println!("cargo:warning=CUDA feature requested but nvcc compiler was not found or failed. Falling back to Rust CPU runtime.");
    }
}
