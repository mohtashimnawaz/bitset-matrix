# bitset-matrix

A compact, row-major 2D bitset matrix with fast bitwise operations across rows and columns.

Features
- Dense, row-major `u64`-backed storage
- Fast row-wise block operations (SIMD feature available)
- Column ops and iterators for convenience
- Small, dependency-free core; optional `simd` feature for acceleration

Quick example: N-Queens helper

```rust
use bitset_matrix::BitMatrix;

// Example: for backtracking you'd store available columns as a row of bits
let mut m = BitMatrix::new(1, 8);
for c in 0..8 { m.set(0, c, true); }
// pick col 3
m.set(0, 3, false);
```

Quick example: Sudoku pencil marks

```rust
use bitset_matrix::BitMatrix;
// 9x9 board, each cell can have 1..9 candidates encoded per row per submatrix as needed
let mut board = BitMatrix::new(9, 9);
board.set(0, 0, true); // candidate marker
```

See `examples/` for a runnable N-Queens example.
