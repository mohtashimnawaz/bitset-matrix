use quickcheck::quickcheck;
use bitset_matrix::BitMatrix;

fn eq_elementwise_and(a: BitMatrix, b: BitMatrix) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() { return true; }
    let mut c = a.clone();
    c.bitand_assign(&b);
    let va = a.to_vec();
    let vb = b.to_vec();
    let mut expected = va.clone();
    for r in 0..va.len() {
        for cidx in 0..va[0].len() {
            expected[r][cidx] = va[r][cidx] & vb[r][cidx];
        }
    }
    c.to_vec() == expected
}

quickcheck! {
    fn prop_and(rows: usize, cols: usize, _seed: u64) -> bool {
        let rows = (rows % 8) + 1; // keep sizes small for quick tests
        let cols = (cols % 128) + 1;
        // random-ish generation from seed is deterministic here; build small matrices
        let mut a = BitMatrix::new(rows, cols);
        let mut b = BitMatrix::new(rows, cols);
        for r in 0..rows { for c in 0..cols { let v = (r*c) % 3 == 0; a.set(r,c,v); let v2 = (r+c) % 2 == 0; b.set(r,c,v2); }}
        eq_elementwise_and(a,b)
    }
}
