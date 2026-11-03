//! Batch tensor construction and recognition execution helpers (#9).

use ndarray::Array4;
use ort::session::Session;

use crate::rec::{resize_norm_into, CtcDecoder};
use crate::{REC_HEIGHT, REC_WIDTH};

/// Run recognition on a single batch and populate the destination results array directly.
///
/// # Arguments
/// * `session` - Mutable reference to the assigned ONNX recognition session.
/// * `decoder` - CTC greedy decoder populated with the model vocabulary.
/// * `crops` - Slice of original crops `(original_index, image)`.
/// * `batch_indices` - Indices within `crops` belonging to this execution batch.
/// * `results` - Destination slice to populate with recognized text and confidence.
pub fn run_rec_batch(
    session: &mut Session,
    decoder: &CtcDecoder,
    crops: &[(usize, image::RgbImage)],
    batch_indices: &[usize],
    results: &mut [(String, f32)],
) -> Result<(), String> {
    let local_res = run_rec_batch_local(session, decoder, crops, batch_indices)?;
    for (orig_idx, text, conf) in local_res {
        results[orig_idx] = (text, conf);
    }
    Ok(())
}

/// Run recognition on a batch and return `(original_index, text, confidence)` tuples.
///
/// Prepares a uniform canvas sized to the maximum aspect ratio within the batch,
/// evaluates the ONNX forward pass, and performs CTC greedy argmax decoding.
pub fn run_rec_batch_local(
    session: &mut Session,
    decoder: &CtcDecoder,
    crops: &[(usize, image::RgbImage)],
    batch_indices: &[usize],
) -> Result<Vec<(usize, String, f32)>, String> {
    let max_ratio = batch_indices
        .iter()
        .map(|&ci| crops[ci].1.width() as f32 / crops[ci].1.height().max(1) as f32)
        .fold(0.0f32, f32::max)
        .max(REC_WIDTH as f32 / REC_HEIGHT as f32);
    let batch_w = ((REC_HEIGHT as f32 * max_ratio).ceil() as usize).max(1);
    let batch_size = batch_indices.len();
    let mut rec_input = Array4::<f32>::zeros((batch_size, 3, REC_HEIGHT as usize, batch_w));
    for (bi, &ci) in batch_indices.iter().enumerate() {
        let crop = &crops[ci].1;
        let ratio = crop.width() as f32 / crop.height().max(1) as f32;
        let target_w = ((REC_HEIGHT as f32 * ratio).ceil() as usize).min(batch_w);
        let view = rec_input.slice_mut(ndarray::s![bi, .., .., 0..target_w]);
        resize_norm_into(crop, target_w, REC_HEIGHT as usize, view);
    }
    let rec_tensor = ort::value::Tensor::from_array(rec_input).expect("rec tensor");
    let out = session
        .run(ort::inputs![rec_tensor])
        .map_err(|e| format!("rec run: {e}"))?;
    let (shape, data) = out[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("rec extract: {e}"))?;
    let num_classes = shape[2] as usize;
    let num_timesteps = shape[1] as usize;
    let mut local_res = Vec::with_capacity(batch_size);
    for (bi, &ci) in batch_indices.iter().enumerate() {
        let orig_idx = crops[ci].0;
        let offset = bi * num_timesteps * num_classes;
        let (text, conf) = decoder.decode(&data[offset..], num_classes, num_timesteps);
        local_res.push((orig_idx, text, conf));
    }
    Ok(local_res)
}
