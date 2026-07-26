use std::ffi::c_void;

#[cfg(cuda_available)]
extern "C" {
    fn cudaStreamCreate(stream: *mut *mut c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut c_void) -> i32;
}

/// CUDA Stream and Prefetch Ring Buffer Abstraction.
pub struct CudaStreamManager {
    stream_ptr: *mut c_void,
    #[allow(dead_code)]
    owned: bool,
}

unsafe impl Send for CudaStreamManager {}

impl CudaStreamManager {
    pub fn new() -> Self {
        Self {
            stream_ptr: std::ptr::null_mut(),
            owned: false,
        }
    }

    pub fn from_raw(stream_ptr: *mut c_void, owned: bool) -> Self {
        Self { stream_ptr, owned }
    }

    pub fn create() -> Self {
        #[cfg(cuda_available)]
        {
            let mut ptr = std::ptr::null_mut();
            unsafe {
                if cudaStreamCreate(&mut ptr) == 0 {
                    return Self {
                        stream_ptr: ptr,
                        owned: true,
                    };
                }
            }
        }
        Self::new()
    }

    pub fn stream_ptr(&self) -> *mut c_void {
        self.stream_ptr
    }

    pub fn synchronize(&self) {
        #[cfg(cuda_available)]
        {
            if !self.stream_ptr.is_null() {
                unsafe {
                    cudaStreamSynchronize(self.stream_ptr);
                }
            }
        }
    }
}

impl Drop for CudaStreamManager {
    fn drop(&mut self) {
        #[cfg(cuda_available)]
        {
            if self.owned && !self.stream_ptr.is_null() {
                self.synchronize();
                unsafe {
                    cudaStreamDestroy(self.stream_ptr);
                }
            }
        }
    }
}

impl Default for CudaStreamManager {
    fn default() -> Self {
        Self::new()
    }
}
