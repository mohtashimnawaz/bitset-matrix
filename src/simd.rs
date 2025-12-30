//! SIMD helpers.
//!
//! Historically we tried using `packed_simd_2` which fails on some stable
//! toolchains/targets; for portability the `simd` feature currently uses a
//! scalar fallback. In future we can wire `std::simd` or a packed backend
//! behind this feature when we detect compiler support.

mod imp {
    /// Scalar fallback for block-wise AND.
    pub fn block_and(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        for i in 0..n { dst[i] &= src[i]; }
    }

    /// Scalar fallback for block-wise OR.
    pub fn block_or(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        for i in 0..n { dst[i] |= src[i]; }
    }

    /// Scalar fallback for block-wise XOR.
    pub fn block_xor(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        for i in 0..n { dst[i] ^= src[i]; }
    }
}

pub use imp::{block_and, block_or, block_xor};
