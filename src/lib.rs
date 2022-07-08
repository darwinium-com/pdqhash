///! Compute PDQ hash of an image.
///! The PDQ algorithm was developed and open-sourced by Facebook (now Meta) in 2019.
///! It specifies a transformation which converts images into a binary format ('PDQ Hash') whereby 'perceptually similar’ images produce similar outputs.
///! It was designed to offer an industry standard for representing images to collaborate on threat mitigation.
use std::ops::Deref;

pub use image;
use image::GenericImageView;

const LUMA_FROM_R_COEFF: f32 = 0.299;
const LUMA_FROM_G_COEFF: f32 = 0.587;
const LUMA_FROM_B_COEFF: f32 = 0.114;

mod dct;

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Minimum size tested.
const MIN_HASHABLE_DIM: u32 = 5;

//  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// Tent filter.
const PDQ_NUM_JAROSZ_XY_PASSES: usize = 2;

const DOWNSAMPLE_DIMS: u32 = 512;

trait ToLuma: image::Pixel {
    fn to_luma(&self) -> f32;
}

impl ToLuma for image::Rgb<u8> {
    fn to_luma(&self) -> f32 {
        (self.0[0] as f32) * LUMA_FROM_R_COEFF
            + (self.0[1] as f32) * LUMA_FROM_G_COEFF
            + (self.0[2] as f32) * LUMA_FROM_B_COEFF
    }
}

impl ToLuma for image::Rgb<u16> {
    fn to_luma(&self) -> f32 {
        (self.0[0] as f32) / 256.0 * LUMA_FROM_R_COEFF
            + (self.0[1] as f32) / 256.0 * LUMA_FROM_G_COEFF
            + (self.0[2] as f32) / 256.0 * LUMA_FROM_B_COEFF
    }
}

impl ToLuma for image::Rgba<u8> {
    fn to_luma(&self) -> f32 {
        (self.0[0] as f32) * LUMA_FROM_R_COEFF
            + (self.0[1] as f32) * LUMA_FROM_G_COEFF
            + (self.0[2] as f32) * LUMA_FROM_B_COEFF
    }
}

impl ToLuma for image::Rgba<u16> {
    fn to_luma(&self) -> f32 {
        (self.0[0] as f32) / 256.0 * LUMA_FROM_R_COEFF
            + (self.0[1] as f32) / 256.0 * LUMA_FROM_G_COEFF
            + (self.0[2] as f32) / 256.0 * LUMA_FROM_B_COEFF
    }
}

impl ToLuma for image::Bgr<u8> {
    fn to_luma(&self) -> f32 {
        (self.0[0] as f32) * LUMA_FROM_B_COEFF
            + (self.0[1] as f32) * LUMA_FROM_G_COEFF
            + (self.0[2] as f32) * LUMA_FROM_R_COEFF
    }
}

impl ToLuma for image::Bgra<u8> {
    fn to_luma(&self) -> f32 {
        (self.0[0] as f32) * LUMA_FROM_B_COEFF
            + (self.0[1] as f32) * LUMA_FROM_G_COEFF
            + (self.0[2] as f32) * LUMA_FROM_R_COEFF
    }
}

impl ToLuma for image::Luma<u8> {
    fn to_luma(&self) -> f32 {
        self.0[0] as f32
    }
}

impl ToLuma for image::Luma<u16> {
    fn to_luma(&self) -> f32 {
        self.0[0] as f32 / 256.0
    }
}

impl ToLuma for image::LumaA<u8> {
    fn to_luma(&self) -> f32 {
        self.0[0] as f32
    }
}

impl ToLuma for image::LumaA<u16> {
    fn to_luma(&self) -> f32 {
        self.0[0] as f32 / 256.0
    }
}

trait ToLumaImage {
    fn to_luma_image(&self) -> (usize, usize, Vec<f32>);
}

impl<P, Container> ToLumaImage for image::ImageBuffer<P, Container>
where
    P: ToLuma + 'static,
    P::Subpixel: 'static,
    Container: Deref<Target = [P::Subpixel]>,
{
    fn to_luma_image(&self) -> (usize, usize, Vec<f32>) {
        let width = self.width();
        let height = self.height();
        let out = self.pixels().map(<P as ToLuma>::to_luma).collect();
        (width as usize, height as usize, out)
    }
}

fn to_luma_image(image: &image::DynamicImage) -> (usize, usize, Vec<f32>) {
    match image {
        image::DynamicImage::ImageLuma8(image) => image.to_luma_image(),
        image::DynamicImage::ImageLumaA8(image) => image.to_luma_image(),
        image::DynamicImage::ImageRgb8(image) => image.to_luma_image(),
        image::DynamicImage::ImageRgba8(image) => image.to_luma_image(),
        image::DynamicImage::ImageBgr8(image) => image.to_luma_image(),
        image::DynamicImage::ImageBgra8(image) => image.to_luma_image(),
        image::DynamicImage::ImageLuma16(image) => image.to_luma_image(),
        image::DynamicImage::ImageLumaA16(image) => image.to_luma_image(),
        image::DynamicImage::ImageRgb16(image) => image.to_luma_image(),
        image::DynamicImage::ImageRgba16(image) => image.to_luma_image(),
    }
}

fn compute_jarosz_filter_window_size(old_dimension: usize, new_dimension: usize) -> usize {
    (old_dimension + 2 * new_dimension - 1) / (2 * new_dimension)
}

fn jarosz_filter_float(
    buffer1: &mut [f32], // matrix as num_rows x num_cols in row-major order
    num_rows: usize,
    num_cols: usize,
    window_size_along_rows: usize,
    window_size_along_cols: usize,
    nreps: usize,
) {
    let mut temp_buf = Vec::new();
    temp_buf.resize(buffer1.len(), 0.0);
    for _ in 0..nreps {
        box_along_rows_float(
            buffer1,
            temp_buf.as_mut_slice(),
            num_rows,
            num_cols,
            window_size_along_rows,
        );
        box_along_cols_float(
            temp_buf.as_slice(),
            buffer1,
            num_rows,
            num_cols,
            window_size_along_cols,
        );
    }
}

// This is called from two places, one has a constant stride, the other a variable stride
// It should compile a version for each.
#[inline(always)]
fn box_one_d_float(
    invec: &[f32],
    in_start_offset: usize,
    outvec: &mut [f32],
    vector_length: usize,
    stride: usize,
    full_window_size: usize,
) {
    let half_window_size = (full_window_size + 2) / 2; // 7->4, 8->5

    let phase_1_nreps = half_window_size - 1;
    let phase_2_nreps = full_window_size - half_window_size + 1;

    let oi_off = phase_1_nreps * stride;
    let li_off = phase_2_nreps * stride;

    let mut sum = 0.0;
    let mut current_window_size = 0.0;

    let phase_1_end = oi_off + in_start_offset;

    // PHASE 1: ACCUMULATE FIRST SUM NO WRITES
    for ri in (in_start_offset..phase_1_end).step_by(stride) {
        let value = invec[ri];
        sum += value;
        current_window_size += 1.0;
    }

    let phase_2_end = full_window_size * stride + in_start_offset;
    // PHASE 2: INITIAL WRITES WITH SMALL WINDOW
    for ri in (phase_1_end..phase_2_end).step_by(stride) {
        let oi = ri - oi_off;
        sum += invec[ri];
        current_window_size += 1.0;
        outvec[oi] = sum / current_window_size;
    }

    let phase_3_end = vector_length * stride + in_start_offset;
    // PHASE 3: WRITES WITH FULL WINDOW
    for ri in (phase_2_end..phase_3_end).step_by(stride) {
        let oi = ri - oi_off;
        let li = oi - li_off;
        sum += invec[ri];
        sum -= invec[li];
        outvec[oi] = sum / (current_window_size);
    }

    let phase_4_start = (vector_length - half_window_size + 1) * stride + in_start_offset;
    // PHASE 4: FINAL WRITES WITH SMALL WINDOW
    for oi in (phase_4_start..phase_3_end).step_by(stride) {
        let li = oi - li_off;
        sum -= invec[li];
        current_window_size -= 1.0;
        outvec[oi] = sum / current_window_size;
    }
}

// ----------------------------------------------------------------
fn box_along_rows_float(
    input: &[f32],      // matrix as num_rows x num_cols in row-major order
    output: &mut [f32], // matrix as num_rows x num_cols in row-major order
    n_rows: usize,
    n_cols: usize,
    window_size: usize,
) {
    for i in 0..n_rows {
        box_one_d_float(input, i * n_cols, output, n_cols, 1, window_size);
    }
}

// ----------------------------------------------------------------
fn box_along_cols_float(
    input: &[f32],      // matrix as num_rows x num_cols in row-major order
    output: &mut [f32], // matrix as num_rows x num_cols in row-major order
    n_rows: usize,
    n_cols: usize,
    window_size: usize,
) {
    for j in 0..n_cols {
        box_one_d_float(input, j, output, n_rows, n_cols, window_size);
    }
}

// ----------------------------------------------------------------
fn decimate_float<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    input: &[f32], // matrix as in_num_rows x in_num_cols in row-major order
    in_num_rows: usize,
    in_num_cols: usize,
) -> [[f32; OUT_NUM_COLS]; OUT_NUM_ROWS] {
    let mut output = [[0.0; OUT_NUM_COLS]; OUT_NUM_ROWS];
    // target centers not corners:
    for outi in 0..OUT_NUM_ROWS {
        let ini = ((outi * 2 + 1) * in_num_rows) / (OUT_NUM_ROWS * 2);
        for outj in 0..OUT_NUM_COLS {
            let inj = ((outj * 2 + 1) * in_num_cols) / (OUT_NUM_COLS * 2);
            output[outi][outj] = input[ini * in_num_cols + inj];
        }
    }
    output
}

// ----------------------------------------------------------------
// This is all heuristic (see the PDQ hashing doc). Quantization matters since
// we want to count *significant* gradients, not just the some of many small
// ones. The constants are all manually selected, and tuned as described in the
// document.
fn pdq_image_domain_quality_metric<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    buffer64x64: &[[f32; OUT_NUM_COLS]; OUT_NUM_ROWS],
) -> f32 {
    let mut gradient_sum = 0.0;

    for i in 0..(OUT_NUM_ROWS - 1) {
        for j in 0..OUT_NUM_COLS {
            let u = buffer64x64[i][j];
            let v = buffer64x64[i + 1][j];
            let d = (u - v) / 255.;
            gradient_sum += d.abs();
        }
    }
    for i in 0..OUT_NUM_ROWS {
        for j in 0..(OUT_NUM_COLS - 1) {
            let u = buffer64x64[i][j];
            let v = buffer64x64[i][j + 1];
            let d = (u - v) / 255.;
            gradient_sum += d.abs();
        }
    }

    // Heuristic scaling factor.
    let quality = gradient_sum / 90.;
    if quality > 1.0 {
        1.0
    } else {
        quality
    }
}

const BUFFER_W_H: usize = 64;

const DCT_OUTPUT_W_H: usize = 16;
const DCT_OUTPUT_MATRIX_SIZE: usize = DCT_OUTPUT_W_H * DCT_OUTPUT_W_H;

const HASH_LENGTH: usize = DCT_OUTPUT_MATRIX_SIZE / 8;

/// Perform a discrete cosine transform from a 64x64 matrix and compute only a 16x16 corner of it. Quicker than computing the whole thing.
fn dct64_to_16<const OUT_NUM_ROWS: usize, const OUT_NUM_COLS: usize>(
    input: &[[f32; OUT_NUM_COLS]; OUT_NUM_ROWS],
) -> [f32; DCT_OUTPUT_MATRIX_SIZE] {
    let mut intermediate_matrix = [[0.0; OUT_NUM_COLS]; DCT_OUTPUT_W_H];
    for i in 0..DCT_OUTPUT_W_H {
        for j in 0..OUT_NUM_COLS {
            let mut sumk = 0.0;
            for k in 0..BUFFER_W_H {
                sumk += f32::from_bits(dct::DCT_MATRIX[i][k]) * input[k][j];
            }

            intermediate_matrix[i][j] = sumk;
        }
    }

    let mut output = [0.0; DCT_OUTPUT_MATRIX_SIZE];
    for i in 0..DCT_OUTPUT_W_H {
        for j in 0..DCT_OUTPUT_W_H {
            let mut sumk = 0.0;
            for k in 0..BUFFER_W_H {
                sumk += intermediate_matrix[i][k] * f32::from_bits(dct::DCT_MATRIX[j][k]);
            }
            output[i * DCT_OUTPUT_W_H + j] = sumk;
        }
    }
    output
}

// Quickly find the median
fn torben_median(m: &[f32]) -> Option<f32> {
    let mut min = m.iter().cloned().reduce(f32::min)?;
    let mut max = m.iter().cloned().reduce(f32::max)?;

    let half = (m.len() + 1) / 2;
    loop {
        let guess = (min + max) / 2.0;
        let mut less = 0;
        let mut greater = 0;
        let mut equal = 0;
        let mut maxltguess = min;
        let mut mingtguess = max;
        for val in m {
            if *val < guess {
                less += 1;
                if *val > maxltguess {
                    maxltguess = *val;
                }
            } else if *val > guess {
                greater += 1;
                if *val < mingtguess {
                    mingtguess = *val;
                }
            } else {
                equal += 1;
            }
        }
        if less <= half && greater <= half {
            return Some(if less >= half {
                maxltguess
            } else if less + equal >= half {
                guess
            } else {
                mingtguess
            });
        } else if less > greater {
            max = maxltguess;
        } else {
            min = mingtguess;
        }
    }
}

fn pdq_buffer16x16_to_bits(input: &[f32; DCT_OUTPUT_MATRIX_SIZE]) -> [u8; HASH_LENGTH] {
    let dct_median = torben_median(input).unwrap();
    let mut hash = [0; HASH_LENGTH];

    for i in 0..HASH_LENGTH {
        let mut byte = 0;
        for j in 0..8 {
            let val = input[i * 8 + j];
            if val > dct_median {
                byte |= 1 << j;
            }
        }
        hash[HASH_LENGTH - i - 1] = byte;
    }
    hash
}

/// Returns PDQ hash and quality of an image without first downscaling.
///
/// It is bit-for-bit compatible with the expected output from the Java version provided by facebook.
pub fn generate_pdq_full_size(image: &image::DynamicImage) -> ([u8; HASH_LENGTH], f32) {
    let (num_cols, num_rows, mut image) = to_luma_image(image);
    let window_size_along_rows = compute_jarosz_filter_window_size(num_cols, BUFFER_W_H);
    let window_size_along_cols = compute_jarosz_filter_window_size(num_rows, BUFFER_W_H);

    jarosz_filter_float(
        image.as_mut_slice(),
        num_rows,
        num_cols,
        window_size_along_rows,
        window_size_along_cols,
        PDQ_NUM_JAROSZ_XY_PASSES,
    );

    let buffer64x64 =
        decimate_float::<BUFFER_W_H, BUFFER_W_H>(image.as_slice(), num_rows, num_cols);

    let buffer16x16 = dct64_to_16(&buffer64x64);
    (
        pdq_buffer16x16_to_bits(&buffer16x16),
        pdq_image_domain_quality_metric(&buffer64x64),
    )
}

/// Returns PDQ hash and quality of an image.
///
/// Returns None if image is too small to generate a useful hash.
/// This will first downsize the image in RGB space using image crate, which is more efficient than computing PDQ on the full size image. Some divergence from reference implementation is expected.
pub fn generate_pdq(image: &image::DynamicImage) -> Option<([u8; HASH_LENGTH], f32)> {
    if image.width() < MIN_HASHABLE_DIM || image.height() < MIN_HASHABLE_DIM {
        return None;
    }

    let out = if image.width() > DOWNSAMPLE_DIMS || image.height() > DOWNSAMPLE_DIMS {
        generate_pdq_full_size(&image.thumbnail_exact(
            DOWNSAMPLE_DIMS.min(image.width()),
            DOWNSAMPLE_DIMS.min(image.height()),
        ))
    } else {
        generate_pdq_full_size(&image)
    };
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load() {
        fn load(data: &[u8]) -> String {
            let hash = generate_pdq_full_size(&image::load_from_memory(data).unwrap()).0;
            hex::encode(hash)
        }

        assert_eq!(
            "f8f8f0cee0f4a84f06370a22038f63f0b36e2ed596621e1d33e6b39c4e9c9b22",
            load(include_bytes!("test_data/bridge-1-original.jpg"))
        );
        assert_eq!(
            "30a10efd71cc3d429013d48d0ffffc52e34e0e17ada952a9d29685211ea9e5af",
            load(include_bytes!("test_data/bridge-2-rotate-90.jpg"))
        );
        assert_eq!(
            "adad5a64b5a142e75b62a09857da895ae63b847fc23794b766b319361bc93188",
            load(include_bytes!("test_data/bridge-3-rotate-180.jpg"))
        );
        assert_eq!(
            "a5f0a457a48995e8c9065c275aaa5498b61ba4bdf8fcf80387c32f8b1bfc4f05",
            load(include_bytes!("test_data/bridge-4-rotate-270.jpg"))
        );
        assert_eq!(
            "f8f80f31e0f417b20e37f5cd028f980fb36ed02a9662c1e233e64c634e9c64dd",
            load(include_bytes!("test_data/bridge-5-flipx.jpg"))
        );
        assert_eq!(
            "0dad2599b1a1bd1a5362576742da32a5e63b7380c2374b4866b366c91bc9ce77",
            load(include_bytes!("test_data/bridge-6-flipy.jpg"))
        );
        assert_eq!(
            "f0a5e102f1ccc0bd945308720fff038de34ef1e8ada9a956d2967ade5ea91a50",
            load(include_bytes!("test_data/bridge-7-flip-plus-1.jpg"))
        );
        assert_eq!(
            "a5f05aa8a4896a17c906a2d85aaaab07b61b5b42f8fc07fc87c3d0741bfcb0fa",
            load(include_bytes!("test_data/bridge-8-flip-minus-1.jpg"))
        );
    }
}
