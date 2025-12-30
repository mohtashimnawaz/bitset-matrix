use bitset_matrix::BitMatrix;

fn main() {
    // naive demonstration of candidate sets per cell using a 9x9 matrix of candidates
    let mut cand = BitMatrix::new(9, 9);
    // set candidate for cell (0,0) = 1 as true
    cand.set(0, 0, true);
    println!("candidate (0,0)=1: {}", cand.get(0,0));
}
