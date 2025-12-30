#![cfg(feature = "simd")]

use bitset_matrix::BitMatrix;

#[test]
fn simd_and_matches_scalar() {
    let mut a = BitMatrix::new(4, 130);
    let mut b = BitMatrix::new(4, 130);
    // set a few deterministic bits
    for r in 0..4 { for c in 0..130 { if (r * c) % 5 == 0 { a.set(r, c, true); } if (r + c) % 7 == 0 { b.set(r, c, true); } }}
    let mut c = a.clone();
    c.bitand_assign(&b);

    // Elementwise check
    for r in 0..4 { for col in 0..130 { let expected = a.get(r, col) & b.get(r, col); assert_eq!(c.get(r, col), expected); }}
}
