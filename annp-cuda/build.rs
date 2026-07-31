use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn find_nvcc() -> Option<PathBuf> {
    if Command::new("nvcc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        return Some(PathBuf::from("nvcc"));
    }

    if let Ok(cuda_path) = env::var("CUDA_PATH") {
        let nvcc = PathBuf::from(cuda_path).join("bin").join(if cfg!(windows) {
            "nvcc.exe"
        } else {
            "nvcc"
        });
        if nvcc.exists() {
            return Some(nvcc);
        }
    }

    let default_linux = PathBuf::from("/usr/local/cuda/bin/nvcc");
    if default_linux.exists() {
        return Some(default_linux);
    }

    if cfg!(windows) {
        let default_base = Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA");
        if default_base.exists() {
            if let Ok(entries) = std::fs::read_dir(default_base) {
                for entry in entries.flatten() {
                    let cand = entry.path().join("bin").join("nvcc.exe");
                    if cand.exists() {
                        return Some(cand);
                    }
                }
            }
        }
    }
    None
}

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

    // 3b. Try our find_nvcc to deduce lib path
    if let Some(nvcc_path) = find_nvcc() {
        if let Some(bin_dir) = nvcc_path.parent() {
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
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
    println!("cargo:rerun-if-changed=cuda/common.cuh");
    println!("cargo:rerun-if-changed=cuda/micro_block_fused.cu");
    println!("cargo:rerun-if-changed=cuda/particle_router.cu");

    let nvcc_path_opt = find_nvcc();
    let cuda_enabled = env::var("CARGO_FEATURE_CUDA").is_ok() || nvcc_path_opt.is_some();

    if cuda_enabled {
        let nvcc_path = nvcc_path_opt.unwrap_or_else(|| PathBuf::from("nvcc"));
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

        let cu_files = vec!["cuda/micro_block_fused.cu", "cuda/particle_router.cu"];

        let msvc_tool = if target_env == "msvc" {
            let target =
                env::var("TARGET").unwrap_or_else(|_| "x86_64-pc-windows-msvc".to_string());
            Some(cc::Build::new().target(&target).get_compiler())
        } else {
            None
        };

        let mut nvcc_check = Command::new(&nvcc_path);
        nvcc_check.arg("--version");
        if let Some(tool) = &msvc_tool {
            for (k, v) in tool.env() {
                nvcc_check.env(k, v);
            }
            if let Some(parent) = tool.path().parent() {
                let current_path = env::var_os("PATH").unwrap_or_default();
                let mut new_path = parent.to_path_buf().into_os_string();
                new_path.push(";");
                new_path.push(&current_path);
                nvcc_check.env("PATH", new_path);
            }
        }
        let nvcc_check_res = nvcc_check.output();

        if let Ok(output) = nvcc_check_res {
            if output.status.success() {
                let obj_ext = if target_os == "windows" { "obj" } else { "o" };
                let mut obj_files = Vec::new();

                for cu_file in &cu_files {
                    let stem = Path::new(cu_file).file_stem().unwrap().to_str().unwrap();
                    let obj_file = out_dir.join(format!("{}.{}", stem, obj_ext));

                    let mut compile_cmd = Command::new(&nvcc_path);
                    if let Some(tool) = &msvc_tool {
                        for (k, v) in tool.env() {
                            compile_cmd.env(k, v);
                        }
                        if let Some(parent) = tool.path().parent() {
                            let current_path = env::var_os("PATH").unwrap_or_default();
                            let mut new_path = parent.to_path_buf().into_os_string();
                            new_path.push(";");
                            new_path.push(&current_path);
                            compile_cmd.env("PATH", new_path);
                        }
                    }
                    let cuda_arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());
                    compile_cmd
                        .arg("-c")
                        .arg("-O3")
                        .arg("--use_fast_math")
                        .arg("-std=c++17")
                        .arg(format!("-arch={}", cuda_arch));

                    if target_env == "msvc" {
                        compile_cmd.arg("-Xcompiler").arg("/MT");
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

                let lib_name = if target_os == "windows" {
                    "annp_cuda.lib"
                } else {
                    "libannp_cuda.a"
                };
                let out_lib_path = out_dir.join(lib_name);

                let mut lib_cmd = Command::new(&nvcc_path);
                if let Some(tool) = &msvc_tool {
                    for (k, v) in tool.env() {
                        lib_cmd.env(k, v);
                    }
                    if let Some(parent) = tool.path().parent() {
                        let current_path = env::var_os("PATH").unwrap_or_default();
                        let mut new_path = parent.to_path_buf().into_os_string();
                        new_path.push(";");
                        new_path.push(&current_path);
                        lib_cmd.env("PATH", new_path);
                    }
                }
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

                if !lib_success && target_os == "windows" {
                    let mut msvc_lib_cmd = Command::new("lib");
                    if let Some(tool) = &msvc_tool {
                        for (k, v) in tool.env() {
                            msvc_lib_cmd.env(k, v);
                        }
                        if let Some(parent) = tool.path().parent() {
                            let current_path = env::var_os("PATH").unwrap_or_default();
                            let mut new_path = parent.to_path_buf().into_os_string();
                            new_path.push(";");
                            new_path.push(&current_path);
                            msvc_lib_cmd.env("PATH", new_path);
                        }
                    }
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
                    println!("cargo:rustc-cfg=cuda_available");
                    return;
                }
            }
        }
        println!(
            "cargo:warning=CUDA feature requested but nvcc compilation failed or nvcc was not found. Falling back to Rust CPU runtime."
        );
    }
}
