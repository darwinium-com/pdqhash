extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro::TokenTree;

// This is a proc macro to generate the discrete cosine transform matrix at compile time.
// it only exists because Rust doesn't want to support floats in const functions


// ----------------------------------------------------------------
// See comments on dct64To16. Input is (0..63)x(0..63); output is
// (1..16)x(1..16) with the latter indexed as (0..15)x(0..15).
//
// * numRows is 16.
// * numCols is 64.
// * Storage is row-major
// * Element i,j at row i column j is at offset i*16+j.
fn fill_dct_matrix_64_cached<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>() -> [[f32; OUT_NUM_COLS]; OUT_NUM_ROWS] {
    let mut buffer = [[0.0; OUT_NUM_COLS]; OUT_NUM_ROWS];
    let matrix_scale_factor = f32::sqrt(2.0 / 64.0);
    for i in 0..OUT_NUM_ROWS {
        for j in 0..OUT_NUM_COLS {
            buffer[i][j] = matrix_scale_factor *
                f32::cos((std::f32::consts::PI / 2.0 / 64.0) * (i + 1) as f32 * (2 * j + 1) as f32);
        }
    }
    buffer
}

struct CommaPutter<I: Iterator<Item=TokenTree>> {
    source: I,
    wants_comma: bool,
}

impl<I: Iterator<Item=TokenTree>> CommaPutter<I> {
    fn new(source: I) -> Self {
        Self{
            source,
            wants_comma: false,
        }
    }
}

impl<I: Iterator<Item=TokenTree>> Iterator for CommaPutter<I> {
    type Item = TokenTree;

    fn next(&mut self) -> Option<Self::Item> {
        if self.wants_comma {
            self.wants_comma = false;
            Some(TokenTree::Punct(proc_macro::Punct::new(',', proc_macro::Spacing::Alone)))
        } else {
            let out = self.source.next();
            if out.is_some() {
                self.wants_comma = true;
            }
            out
        }
    }
}

#[proc_macro]
pub fn make_matrix(_item: TokenStream) -> TokenStream {
    let matrix = fill_dct_matrix_64_cached::<16, 64>();
    let outer_array: TokenStream = CommaPutter::new(matrix.iter().map(|row|{
        let inner_array: TokenStream = CommaPutter::new(row.iter().map(|val|{
            TokenTree::Literal(proc_macro::Literal::f32_unsuffixed(*val))
        })).collect();
        TokenTree::Group(proc_macro::Group::new(proc_macro::Delimiter::Bracket, inner_array))
    })).collect();
    TokenTree::Group(proc_macro::Group::new(proc_macro::Delimiter::Bracket, outer_array)).into()
}
