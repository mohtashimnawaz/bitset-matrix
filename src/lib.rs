//! A compact, row-major 2D bitset matrix with fast row-wise bitwise operations.
//!
//! The matrix stores bits in contiguous `u64` words per row. Row-wise operations (AND/OR/XOR)
//! are implemented as word-wise loops for speed. Column-wise operations are supported but
//! are naturally slower because bits are packed across words.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitMatrix {
    rows: usize,
    cols: usize,
    words_per_row: usize,
    data: Vec<u64>,
}

// SIMD helper module (feature-gated)
mod simd;

impl BitMatrix {
    /// Create a new `rows x cols` zeroed bit matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        let words_per_row = (cols + 63) / 64;
        let data = vec![0u64; rows * words_per_row];
        let mut m = Self { rows, cols, words_per_row, data };
        m.clear_unused_bits();
        m
    }

    /// Number of rows.
    pub fn rows(&self) -> usize { self.rows }

    /// Number of columns.
    pub fn cols(&self) -> usize { self.cols }

    fn index(&self, row: usize, col: usize) -> (usize, u64) {
        assert!(row < self.rows, "row out of bounds");
        assert!(col < self.cols, "col out of bounds");
        let word = col / 64;
        let bit = (col % 64) as u64;
        (row * self.words_per_row + word, 1u64 << bit)
    }

    /// Set a bit at (row, col).
    pub fn set(&mut self, row: usize, col: usize, val: bool) {
        let (idx, mask) = self.index(row, col);
        if val { self.data[idx] |= mask; } else { self.data[idx] &= !mask; }
    }

    /// Get the bit at (row, col).
    pub fn get(&self, row: usize, col: usize) -> bool {
        let (idx, mask) = self.index(row, col);
        (self.data[idx] & mask) != 0
    }

    /// Returns a slice of the words for `row`.
    pub fn row_words(&self, row: usize) -> &[u64] {
        assert!(row < self.rows, "row out of bounds");
        let start = row * self.words_per_row;
        &self.data[start..start + self.words_per_row]
    }

    fn row_words_mut(&mut self, row: usize) -> &mut [u64] {
        assert!(row < self.rows, "row out of bounds");
        let start = row * self.words_per_row;
        &mut self.data[start..start + self.words_per_row]
    }

    fn last_word_mask(&self) -> u64 {
        let rem = self.cols % 64;
        if rem == 0 { !0u64 } else { (1u64 << rem) - 1 }
    }

    fn clear_unused_bits(&mut self) {
        if self.cols % 64 == 0 { return; }
        let mask = self.last_word_mask();
        for r in 0..self.rows {
            let idx = r * self.words_per_row + (self.words_per_row - 1);
            self.data[idx] &= mask;
        }
    }

    /// Count number of set bits in the matrix.
    pub fn count_ones(&self) -> usize {
        let mut sum = 0usize;
        let mask = self.last_word_mask();
        for r in 0..self.rows {
            let start = r * self.words_per_row;
            for w in 0..self.words_per_row {
                let mut v = self.data[start + w];
                // mask last word in row
                if w + 1 == self.words_per_row { v &= mask; }
                sum += v.count_ones() as usize;
            }
        }
        sum
    }

    /// Bitwise AND producing a new matrix. Requires same shape.
    pub fn bitand(&self, other: &Self) -> Self {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        let mut out = self.clone();
        out.bitand_assign(other);
        out
    }

    /// In-place AND with `other`.
    pub fn bitand_assign(&mut self, other: &Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for r in 0..self.rows {
            let start = r * self.words_per_row;
            let end = start + self.words_per_row;
            simd::block_and(&mut self.data[start..end], &other.data[start..end]);
        }
        self.clear_unused_bits();
    }

    /// In-place OR with `other`.
    pub fn bitor_assign(&mut self, other: &Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for r in 0..self.rows {
            let start = r * self.words_per_row;
            let end = start + self.words_per_row;
            simd::block_or(&mut self.data[start..end], &other.data[start..end]);
        }
        self.clear_unused_bits();
    }

    /// In-place XOR with `other`.
    pub fn bitxor_assign(&mut self, other: &Self) {
        assert_eq!(self.rows, other.rows);
        assert_eq!(self.cols, other.cols);
        for r in 0..self.rows {
            let start = r * self.words_per_row;
            let end = start + self.words_per_row;
            simd::block_xor(&mut self.data[start..end], &other.data[start..end]);
        }
        self.clear_unused_bits();
    }

    /// Fast in-place row-wise AND: `dst_row` &= `src_row`.
    pub fn row_and_assign(&mut self, dst_row: usize, src_row: usize) {
        let w = self.words_per_row;
        let mask = self.last_word_mask();
        let dst_start = dst_row * w;
        let src_start = src_row * w;
        for i in 0..w {
            let dst_i = dst_start + i;
            let src_i = src_start + i;
            self.data[dst_i] &= self.data[src_i];
        }
        self.data[dst_start + w - 1] &= mask;
    }

    /// Fast in-place row-wise OR: `dst_row` |= `src_row`.
    pub fn row_or_assign(&mut self, dst_row: usize, src_row: usize) {
        let w = self.words_per_row;
        let mask = self.last_word_mask();
        let dst_start = dst_row * w;
        let src_start = src_row * w;
        for i in 0..w {
            let dst_i = dst_start + i;
            let src_i = src_start + i;
            self.data[dst_i] |= self.data[src_i];
        }
        self.data[dst_start + w - 1] &= mask;
    }

    /// Fast in-place row-wise XOR: `dst_row` ^= `src_row`.
    pub fn row_xor_assign(&mut self, dst_row: usize, src_row: usize) {
        let w = self.words_per_row;
        let mask = self.last_word_mask();
        let dst_start = dst_row * w;
        let src_start = src_row * w;
        for i in 0..w {
            let dst_i = dst_start + i;
            let src_i = src_start + i;
            self.data[dst_i] ^= self.data[src_i];
        }
        self.data[dst_start + w - 1] &= mask;
    }

    /// In-place column-wise AND: for each row r, `dst_col[r] &= src_col[r]`.
    pub fn col_and_assign(&mut self, dst_col: usize, src_col: usize) {
        assert!(dst_col < self.cols && src_col < self.cols);
        let dst_word = dst_col / 64;
        let dst_mask = 1u64 << (dst_col % 64);
        let src_word = src_col / 64;
        let src_mask = 1u64 << (src_col % 64);
        for r in 0..self.rows {
            let base = r * self.words_per_row;
            let src_idx = base + src_word;
            if (self.data[src_idx] & src_mask) == 0 {
                let dst_idx = base + dst_word;
                self.data[dst_idx] &= !dst_mask;
            }
        }
    }

    /// In-place column-wise OR: for each row r, `dst_col[r] |= src_col[r]`.
    pub fn col_or_assign(&mut self, dst_col: usize, src_col: usize) {
        assert!(dst_col < self.cols && src_col < self.cols);
        let dst_word = dst_col / 64;
        let dst_mask = 1u64 << (dst_col % 64);
        let src_word = src_col / 64;
        let src_mask = 1u64 << (src_col % 64);
        for r in 0..self.rows {
            let base = r * self.words_per_row;
            let src_idx = base + src_word;
            if (self.data[src_idx] & src_mask) != 0 {
                let dst_idx = base + dst_word;
                self.data[dst_idx] |= dst_mask;
            }
        }
    }

    /// In-place column-wise XOR: for each row r, `dst_col[r] ^= src_col[r]`.
    pub fn col_xor_assign(&mut self, dst_col: usize, src_col: usize) {
        assert!(dst_col < self.cols && src_col < self.cols);
        let dst_word = dst_col / 64;
        let dst_mask = 1u64 << (dst_col % 64);
        let src_word = src_col / 64;
        let src_mask = 1u64 << (src_col % 64);
        for r in 0..self.rows {
            let base = r * self.words_per_row;
            let src_idx = base + src_word;
            if (self.data[src_idx] & src_mask) != 0 {
                let dst_idx = base + dst_word;
                self.data[dst_idx] ^= dst_mask;
            }
        }
    }

    /// Get a column as a Vec<bool> (col-wise access is slower).
    pub fn column(&self, col: usize) -> Vec<bool> {
        assert!(col < self.cols);
        let mut v = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            v.push(self.get(r, col));
        }
        v
    }

    /// Set a column from a slice of bools.
    pub fn set_column(&mut self, col: usize, src: &[bool]) {
        assert!(col < self.cols);
        assert!(src.len() == self.rows);
        for r in 0..self.rows { self.set(r, col, src[r]); }
    }

    /// Row iterator (yields booleans across columns for a row).
    pub fn iter_row(&self, row: usize) -> RowIter<'_> {
        assert!(row < self.rows);
        RowIter { m: self, row, col: 0 }
    }

    /// Column iterator (yields booleans across rows for a column).
    pub fn iter_col(&self, col: usize) -> ColIter<'_> {
        assert!(col < self.cols);
        ColIter { m: self, col, row: 0 }
    }

    /// Convert to Vec<Vec<bool>>.
    pub fn to_vec(&self) -> Vec<Vec<bool>> {
        let mut out = Vec::with_capacity(self.rows);
        for r in 0..self.rows {
            let mut row = Vec::with_capacity(self.cols);
            for c in 0..self.cols { row.push(self.get(r, c)); }
            out.push(row);
        }
        out
    }
}

/// Row iterator type
pub struct RowIter<'a> { m: &'a BitMatrix, row: usize, col: usize }

impl<'a> Iterator for RowIter<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.col >= self.m.cols { return None; }
        let v = self.m.get(self.row, self.col);
        self.col += 1;
        Some(v)
    }
}

/// Column iterator type
pub struct ColIter<'a> { m: &'a BitMatrix, col: usize, row: usize }

impl<'a> Iterator for ColIter<'a> {
    type Item = bool;
    fn next(&mut self) -> Option<bool> {
        if self.row >= self.m.rows { return None; }
        let v = self.m.get(self.row, self.col);
        self.row += 1;
        Some(v)
    }
}

impl From<Vec<Vec<bool>>> for BitMatrix {
    fn from(v: Vec<Vec<bool>>) -> Self {
        let rows = v.len();
        let cols = if rows == 0 { 0 } else { v[0].len() };
        let mut m = BitMatrix::new(rows, cols);
        for (r, rowv) in v.into_iter().enumerate() {
            assert_eq!(rowv.len(), cols);
            for (c, b) in rowv.into_iter().enumerate() {
                if b { m.set(r, c, true); }
            }
        }
        m
    }
}

impl BitMatrix {
    /// Create from a Vec<Vec<bool>> by value.
    pub fn from_vec(v: Vec<Vec<bool>>) -> Self { BitMatrix::from(v) }

    /// Convert into Vec<Vec<bool>> consuming self.
    pub fn into_vec(self) -> Vec<Vec<bool>> { self.to_vec() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_set_get() {
        let mut m = BitMatrix::new(3, 130); // more than 2 words per row
        assert_eq!(m.rows(), 3);
        assert_eq!(m.cols(), 130);
        assert!(!m.get(1, 1));
        m.set(1, 1, true);
        assert!(m.get(1, 1));
        m.set(1, 129, true);
        assert!(m.get(1, 129));
        assert_eq!(m.count_ones(), 2);
    }

    #[test]
    fn row_ops_and_matrix_ops() {
        let mut a = BitMatrix::new(2, 70);
        let mut b = BitMatrix::new(2, 70);
        a.set(0, 1, true);
        a.set(0, 69, true);
        b.set(0, 1, true);
        b.set(0, 2, true);

        a.row_and_assign(0, 0); // no-op
        assert!(a.get(0, 1));
        a.row_and_assign(0, 0); // still ok

        // test matrix and/or/xor
        let c = a.bitand(&b);
        assert!(c.get(0, 1));
        assert!(!c.get(0, 2));

        a.bitor_assign(&b);
        assert!(a.get(0, 2));

        a.bitxor_assign(&b);
        // XORing twice with b reverts bits that were only in b
        assert!(!a.get(0, 2));
    }

    #[test]
    fn column_get_set() {
        let mut m = BitMatrix::new(4, 10);
        m.set_column(3, &[true, false, true, false]);
        let col = m.column(3);
        assert_eq!(col, vec![true, false, true, false]);
    }

    #[test]
    fn column_ops_and_iterators() {
        let mut m = BitMatrix::new(4, 10);
        // Set a few bits in cols 1 and 2
        m.set(0, 1, true);
        m.set(1, 1, true);
        m.set(2, 2, true);
        m.set(3, 2, true);

        // OR column 3 with column 1
        m.col_or_assign(3, 1);
        assert!(m.get(0, 3));
        assert!(m.get(1, 3));
        assert!(!m.get(2, 3));

        // XOR column 3 with column 2
        m.col_xor_assign(3, 2);
        // row 2 and 3 had col2 set
        assert!(m.get(2, 3));
        assert!(m.get(3, 3));

        // AND column 3 with column 2 (clear bits where col2==0)
        m.col_and_assign(3, 2);
        assert!(!m.get(0, 3));
        assert!(!m.get(1, 3));

        // iterators
        let row0: Vec<bool> = m.iter_row(0).collect();
        assert_eq!(row0.len(), 10);
        let col2: Vec<bool> = m.iter_col(2).collect();
        assert_eq!(col2.len(), 4);

        // to/from vec conversions
        let v = m.to_vec();
        let m2 = BitMatrix::from(v.clone());
        assert_eq!(m2.to_vec(), v);
    }

    #[test]
    fn masks_keep_bounds() {
        let mut m = BitMatrix::new(1, 70); // 70 -> 2 words, last word only 6 valid bits
        m.set(0, 69, true);
        assert!(m.get(0, 69));
        // outside of bounds should panic when directly accessed
    }
}
