use criterion::{criterion_group, criterion_main, Criterion};
use bitset_matrix::BitMatrix;
use rand::Rng;

fn rand_matrix(rows: usize, cols: usize) -> BitMatrix {
    let mut m = BitMatrix::new(rows, cols);
    let mut rng = rand::thread_rng();
    for r in 0..rows {
        for c in 0..cols {
            if rng.gen_bool(0.5) { m.set(r, c, true); }
        }
    }
    m
}

fn bench_bitwise(c: &mut Criterion) {
    let a = rand_matrix(256, 256);
    let b = rand_matrix(256, 256);

    c.bench_function("matrix_and", |ben| {
        let mut a = a.clone();
        ben.iter(|| { a.bitand_assign(&b); });
    });

    c.bench_function("matrix_or", |ben| {
        let mut a = a.clone();
        ben.iter(|| { a.bitor_assign(&b); });
    });

    c.bench_function("matrix_xor", |ben| {
        let mut a = a.clone();
        ben.iter(|| { a.bitxor_assign(&b); });
    });
}

criterion_group!(benches, bench_bitwise);
criterion_main!(benches);
