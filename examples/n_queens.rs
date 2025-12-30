use bitset_matrix::BitMatrix;

// Simple demonstration: represent columns available per row as the row bits.
fn solve_n_queens(n: usize) -> usize {
    fn rec(row: usize, n: usize, cols: &BitMatrix, diag1: &BitMatrix, diag2: &BitMatrix) -> usize {
        if row == n { return 1; }
        let mut count = 0usize;
        let mut avail = Vec::new();
        for c in 0..n { if cols.get(0, c) && diag1.get(0, c) && diag2.get(0, c) { avail.push(c); } }
        for &c in &avail {
            // place at c: clear column and diagonals for next row (naive example)
            let mut cols2 = cols.clone(); cols2.set(0, c, false);
            let mut d1 = diag1.clone(); d1.set(0, c, false);
            let mut d2 = diag2.clone(); d2.set(0, c, false);
            count += rec(row+1, n, &cols2, &d1, &d2);
        }
        count
    }

    let mut cols = BitMatrix::new(1, n);
    let mut d1 = BitMatrix::new(1, n);
    let mut d2 = BitMatrix::new(1, n);
    for c in 0..n { cols.set(0, c, true); d1.set(0, c, true); d2.set(0, c, true); }
    rec(0, n, &cols, &d1, &d2)
}

fn main() {
    println!("Solutions for 8-queens: {}", solve_n_queens(8));
}
