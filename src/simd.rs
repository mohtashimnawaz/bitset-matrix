//! Feature-gated SIMD helpers (fall back to scalar loops when `simd` feature is disabled).

#[cfg(feature = "simd")]
mod imp {
    use packed_simd_2::u64x4;

    pub fn block_and(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        let mut i = 0usize;
        while i + 4 <= n {
            let a = u64x4::from_slice_unaligned(&dst[i..]);
            let b = u64x4::from_slice_unaligned(&src[i..]);
            let r = a & b;
            r.write_to_slice_unaligned(&mut dst[i..]);
            i += 4;
        }
        while i < n { dst[i] &= src[i]; i += 1; }
    }

    pub fn block_or(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        let mut i = 0usize;
        while i + 4 <= n {
            let a = u64x4::from_slice_unaligned(&dst[i..]);
            let b = u64x4::from_slice_unaligned(&src[i..]);
            let r = a | b;
            r.write_to_slice_unaligned(&mut dst[i..]);
            i += 4;
        }
        while i < n { dst[i] |= src[i]; i += 1; }
    }

    pub fn block_xor(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        let mut i = 0usize;
        while i + 4 <= n {
            let a = u64x4::from_slice_unaligned(&dst[i..]);
            let b = u64x4::from_slice_unaligned(&src[i..]);
            let r = a ^ b;
            r.write_to_slice_unaligned(&mut dst[i..]);
            i += 4;
        }
        while i < n { dst[i] ^= src[i]; i += 1; }
    }
}

#[cfg(not(feature = "simd"))]
mod imp {
    pub fn block_and(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        for i in 0..n { dst[i] &= src[i]; }
    }
    pub fn block_or(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        for i in 0..n { dst[i] |= src[i]; }
    }
    pub fn block_xor(dst: &mut [u64], src: &[u64]) {
        let n = dst.len().min(src.len());
        for i in 0..n { dst[i] ^= src[i]; }
    }
}

pub use imp::{block_and, block_or, block_xor};
