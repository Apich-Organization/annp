use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_cuda_lib_dir(target_os: &str) -> Option<PathBuf> {
    // 1. Check CUDA_PATH environment variable
    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let path = PathBuf::from(&cuda_path);
        let lib_dir = if target_os == "windows" {
            path.join("lib").join("x64")
        } else {
            path.join("lib64")
        };
        if lib_dir.exists() {
            return Some(lib_dir);
        }
    }

    // 2. Check CUDA_PATH_V* env vars (Windows versioned paths)
    for (key, val) in env::vars() {
        if key.starts_with("CUDA_PATH_V") {
            let path = PathBuf::from(&val);
            let lib_dir = if target_os == "windows" {
                path.join("lib").join("x64")
            } else {
                path.join("lib64")
            };
            if lib_dir.exists() {
                return Some(lib_dir);
            }
        }
    }

    // 3. Deduce from nvcc binary location via system path lookup
    let where_cmd = if target_os == "windows" {
        "where"
    } else {
        "which"
    };
    if let Ok(where_out) = Command::new(where_cmd).arg("nvcc").output() {
        if where_out.status.success() {
            let path_str = String::from_utf8_lossy(&where_out.stdout);
            if let Some(first_line) = path_str.lines().next() {
                let nvcc_bin = PathBuf::from(first_line.trim());
                if let Some(bin_dir) = nvcc_bin.parent() {
                    if let Some(cuda_root) = bin_dir.parent() {
                        let candidate = if target_os == "windows" {
                            cuda_root.join("lib").join("x64")
                        } else {
                            cuda_root.join("lib64")
                        };
                        if candidate.exists() {
                            return Some(candidate);
                        }
                    }
                }
            }
        }
    }

    // 4. Check standard default installation directories
    if target_os == "windows" {
        let default_base = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
        if default_base.exists() {
            if let Ok(entries) = std::fs::read_dir(default_base) {
                for entry in entries.flatten() {
                    let cand = entry.path().join("lib").join("x64");
                    if cand.exists() {
                        return Some(cand);
                    }
                }
            }
        }
    } else {
        for candidate in &[
            "/usr/local/cuda/lib64",
            "/usr/local/cuda/lib",
            "/usr/lib/x86_64-linux-gnu",
        ] {
            let p = PathBuf::from(candidate);
            if p.exists() {
                return Some(p);
            }
        }
    }

    None
}

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
                    let compile_output = compile_cmd.output();
                    match compile_output {
                        Ok(out) if out.status.success() => {
                            obj_files.push(obj_file);
                        }
                        Ok(out) => {
                            println!(
                                "cargo:warning=NVCC compile error for {}: {}",
                                cu_file,
                                String::from_utf8_lossy(&out.stderr)
                            );
                            return;
                        }
                        Err(e) => {
                            println!("cargo:warning=Failed to execute NVCC: {}", e);
                            return;
                        }
                    }
                }

                // Step 3: Package object files into static library
                let lib_name = if target_os == "windows" {
                    "annp_cuda.lib"
                } else {
                    "libannp_cuda.a"
                };
                let out_lib_path = out_dir.join(lib_name);

                // Try archiving via nvcc -lib
                let mut lib_cmd = Command::new("nvcc");
                lib_cmd.arg("-lib").arg("-o").arg(&out_lib_path);
                for obj in &obj_files {
                    lib_cmd.arg(obj);
                }

                let mut lib_success = false;
                if let Ok(lib_out) = lib_cmd.output() {
                    if lib_out.status.success() {
                        lib_success = true;
                    } else {
                        println!(
                            "cargo:warning=nvcc -lib warning/error: {}",
                            String::from_utf8_lossy(&lib_out.stderr)
                        );
                    }
                }

                // Fallback to MSVC lib.exe on Windows if nvcc -lib failed
                if !lib_success && target_os == "windows" {
                    let mut msvc_lib_cmd = Command::new("lib");
                    msvc_lib_cmd.arg(format!("/OUT:{}", out_lib_path.display()));
                    for obj in &obj_files {
                        msvc_lib_cmd.arg(obj);
                    }
                    if let Ok(msvc_out) = msvc_lib_cmd.output() {
                        if msvc_out.status.success() {
                            lib_success = true;
                        }
                    }
                }

                // Fallback to ar on Linux/Unix if nvcc -lib failed
                if !lib_success && target_os != "windows" {
                    let mut ar_cmd = Command::new("ar");
                    ar_cmd.arg("rcs").arg(&out_lib_path);
                    for obj in &obj_files {
                        ar_cmd.arg(obj);
                    }
                    if let Ok(ar_out) = ar_cmd.output() {
                        if ar_out.status.success() {
                            lib_success = true;
                        }
                    }
                }

                if lib_success {
                    if let Some(cuda_lib_dir) = find_cuda_lib_dir(&target_os) {
                        println!("cargo:rustc-link-search=native={}", cuda_lib_dir.display());
                    }

                    println!("cargo:rustc-link-search=native={}", out_dir.display());
                    println!("cargo:rustc-link-lib=static=annp_cuda");
                    println!("cargo:rustc-link-lib=cudart");
                    return;
                }
            }
        }
        println!("cargo:warning=CUDA feature requested but nvcc compilation failed or nvcc was not found. Falling back to Rust CPU runtime.");
    }
}
