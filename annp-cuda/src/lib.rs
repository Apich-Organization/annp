pub mod ffi;
pub mod stream;

pub use ffi::{
    CudaMicroBlockRunner, CudaParticleAggregator, CudaParticleRouter, ParticleCudaHeader,
};
pub use stream::CudaStreamManager;

pub fn is_cuda_available() -> bool {
    #[cfg(cuda_available)]
    {
        true
    }
    #[cfg(not(cuda_available))]
    {
        false
    }
}
