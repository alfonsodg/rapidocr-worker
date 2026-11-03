use std::time::Instant;

use ort::session::Session;

use crate::batch_runner::{run_rec_batch, run_rec_batch_local};
use crate::cls::{can_skip_remaining, evaluate_sample_orientation, run_cls_chunk, CLS_BATCH};
use crate::config::WorkerConfig;
use crate::det::{cv_resize_bilinear, det_resize_target, normalize_chw, DbPostProcess};
use crate::order::sort_boxes;
use crate::perspective::perspective_crop;
use crate::rec::{plan_rec_batches, CtcDecoder};
use crate::session::{build_session, trim_allocator_heap, StageTimings};

/// Orchestrates the multi-stage OCR pipeline (detection, direction classification, recognition).
pub struct OcrEngine {
    det: Session,
    cls: Option<Session>,
    rec_sessions: Vec<Session>,
    decoder: CtcDecoder,
    post: DbPostProcess,
    det_normalization: ([f32; 3], [f32; 3]),
    limit_side_len: u32,
    rec_batch: usize,
    cls_sampled: bool,
}

impl OcrEngine {
    /// Initialize the OCR engine from the parsed worker configuration (#9).
    ///
    /// # Arguments
    /// * `config` - Validated command-line configuration specifying model paths and thread budgets.
    pub fn new(config: &WorkerConfig) -> Self {
        let arena_enabled = !config.no_arena;
        let det = build_session(&config.det_path, config.threads, arena_enabled);
        if config.cls_disabled_without_flag() {
            eprintln!("warning: --cls-model not provided; direction classification disabled");
        }
        let cls = if config.no_cls || config.cls_path.is_empty() {
            None
        } else {
            Some(build_session(
                &config.cls_path,
                config.threads,
                arena_enabled,
            ))
        };

        let num_workers = config.rec_workers.max(1);
        let rec_threads = (config.threads / num_workers).max(1);
        let mut rec_sessions = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            rec_sessions.push(build_session(&config.rec_path, rec_threads, arena_enabled));
        }

        let meta_val: String = rec_sessions[0]
            .metadata()
            .ok()
            .and_then(|m| m.custom("character"))
            .unwrap_or_default();
        // PP-OCRv6 ships no `character` metadata, so an external keys file is the
        // only source for its 18,708-symbol dictionary. Without either
        // source, failing here is correct: starting with a wrong dictionary would
        // decode every page into garbage.
        let decoder = if !meta_val.trim().is_empty() {
            CtcDecoder::from_metadata(&meta_val).expect("decoder")
        } else if let Some(ref keys_path) = config.keys_path {
            CtcDecoder::from_keys_file(std::path::Path::new(keys_path))
                .unwrap_or_else(|error| panic!("{error}"))
        } else {
            panic!(
                "rec model has no embedded 'character' metadata and no --keys file was given; \
                 pass --keys <path> for models that store the dictionary externally"
            );
        };

        Self {
            det,
            cls,
            rec_sessions,
            decoder,
            post: match config.det_profile {
                crate::config::DetProfile::Ppocrv4 => DbPostProcess::default(),
                crate::config::DetProfile::Ppocrv6 => DbPostProcess::ppocrv6(),
            },
            det_normalization: config.det_normalization.mean_std(),
            limit_side_len: config.limit_side_len,
            rec_batch: config.rec_batch,
            cls_sampled: config.cls_sampled,
        }
    }

    /// Process a single image through the complete OCR pipeline.
    ///
    /// # Arguments
    /// * `image_bytes` - Raw encoded image payload (JPEG, PNG, etc.).
    ///
    /// # Returns
    /// * `Ok((joined_text, mean_confidence, timings))` on success.
    /// * `Err(message)` if decoding or inference fails.
    pub fn process(&mut self, image_bytes: &[u8]) -> Result<(String, f32, StageTimings), String> {
        let img = image::load_from_memory(image_bytes)
            .map_err(|e| format!("decode: {e}"))?
            .to_rgb8();
        let (ow, oh) = (img.width(), img.height());
        if ow == 0 || oh == 0 {
            return Err("empty image".into());
        }

        // --- Detection Stage ---
        let mut timings = StageTimings::default();
        let stage_clock = Instant::now();
        let (nw, nh) = match det_resize_target(ow, oh, self.limit_side_len) {
            Some((nw, nh)) => (nw, nh),
            None => return Err("det resize target is empty".into()),
        };
        let (det_mean, det_std) = self.det_normalization;
        // Drop `resized` before allocating `det_tensor`, and drop `det_tensor` + `det_out`
        // before `boxes_from_bitmap` allocates `bx` so downstream vectors never land above
        // the ~200 MB detection execution-frame peak in glibc's sbrk [heap].
        let det_input = {
            let resized = cv_resize_bilinear(&img, nw, nh);
            normalize_chw(&resized, det_mean, det_std)
        };
        let boxes = {
            let (det_w, det_h, bitmap) = {
                let det_tensor = ort::value::Tensor::from_array(det_input).expect("det tensor");
                let det_out = self
                    .det
                    .run(ort::inputs![det_tensor])
                    .map_err(|e| format!("det run: {e}"))?;
                let (det_shape, det_data) = det_out[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("det extract: {e}"))?;
                (
                    det_shape[3] as usize,
                    det_shape[2] as usize,
                    det_data.to_vec(),
                )
            };
            let (mut bx, _sc) =
                self.post
                    .boxes_from_bitmap(&bitmap, det_w, det_h, ow as usize, oh as usize);
            drop(bitmap);
            sort_boxes(&mut bx);
            timings.det_ms = stage_clock.elapsed().as_secs_f32() * 1000.0;
            bx
        };
        if boxes.is_empty() {
            trim_allocator_heap();
            return Ok((String::new(), 0.0, timings));
        }

        // --- Crop Extraction ---
        let num_boxes = boxes.len();
        let mut crops: Vec<(usize, image::RgbImage)> = boxes
            .iter()
            .enumerate()
            .map(|(i, box_pts)| (i, perspective_crop(&img, box_pts)))
            .collect();
        drop(boxes);
        drop(img);

        // Rotate vertical crops to horizontal before orientation classification
        for (_, crop) in crops.iter_mut() {
            if crop.height() as f32 / crop.width().max(1) as f32 >= 1.5 {
                *crop = image::imageops::rotate270(crop);
            }
        }

        // --- Direction Classification Stage ---
        let cls_clock = Instant::now();
        if let Some(ref mut cls_session) = self.cls {
            let mut order: Vec<usize> = (0..crops.len()).collect();
            order.sort_by(|&a, &b| {
                let ra = crops[a].1.width() as f32 / crops[a].1.height().max(1) as f32;
                let rb = crops[b].1.width() as f32 / crops[b].1.height().max(1) as f32;
                ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut chunks_iter = order.chunks(CLS_BATCH);
            if self.cls_sampled && crops.len() > CLS_BATCH {
                // Fast-path: sample the first batch to check if the document is uniformly upright
                if let Some(sample_chunk) = chunks_iter.next() {
                    let mut sample_decisions = Vec::with_capacity(sample_chunk.len());
                    run_cls_chunk(
                        cls_session,
                        sample_chunk,
                        &mut crops,
                        Some(&mut sample_decisions),
                    )?;
                    let sample_pairs: Vec<(&str, f32)> = sample_decisions
                        .iter()
                        .map(|(l, s)| (l.as_str(), *s))
                        .collect();
                    // Only a confident upright sample set allows skipping the
                    // remaining crops; any other evidence falls back to
                    // per-box classification (Python parity).
                    let orientation = evaluate_sample_orientation(&sample_pairs);
                    if !can_skip_remaining(orientation) {
                        for chunk in chunks_iter {
                            run_cls_chunk(cls_session, chunk, &mut crops, None)?;
                        }
                    }
                }
            } else {
                for chunk in chunks_iter {
                    run_cls_chunk(cls_session, chunk, &mut crops, None)?;
                }
            }
            timings.cls_ms = cls_clock.elapsed().as_secs_f32() * 1000.0;
        }

        // --- Recognition Stage ---
        let rec_clock = Instant::now();
        // Compute aspect ratios and sort crops for optimal bucketing
        crops.sort_by(|a, b| {
            let ra = a.1.width() as f32 / a.1.height().max(1) as f32;
            let rb = b.1.width() as f32 / b.1.height().max(1) as f32;
            ra.partial_cmp(&rb).unwrap_or(std::cmp::Ordering::Equal)
        });

        let crop_ratios: Vec<(usize, f32)> = crops
            .iter()
            .enumerate()
            .map(|(i, (_, c))| (i, c.width() as f32 / c.height().max(1) as f32))
            .collect();
        let batches = plan_rec_batches(&crop_ratios, self.rec_batch);

        let mut results: Vec<(String, f32)> = vec![(String::new(), 0.0); num_boxes];

        if self.rec_sessions.len() <= 1 {
            // Single session execution
            let session = &mut self.rec_sessions[0];
            for batch_indices in batches {
                run_rec_batch(session, &self.decoder, &crops, &batch_indices, &mut results)?;
            }
        } else {
            // Concurrent execution across session workers using scoped threads
            let num_workers = self.rec_sessions.len();
            let mut worker_batches: Vec<Vec<Vec<usize>>> =
                (0..num_workers).map(|_| Vec::new()).collect();
            for (bi, batch) in batches.into_iter().enumerate() {
                worker_batches[bi % num_workers].push(batch);
            }

            let crops_ref = &crops;
            let decoder_ref = &self.decoder;

            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(num_workers);
                for (session, assigned_batches) in self.rec_sessions.iter_mut().zip(worker_batches)
                {
                    handles.push(s.spawn(move || {
                        let mut local_results: Vec<(usize, String, f32)> = Vec::new();
                        for batch_indices in &assigned_batches {
                            let batch_res = run_rec_batch_local(
                                session,
                                decoder_ref,
                                crops_ref,
                                batch_indices,
                            )?;
                            local_results.extend(batch_res);
                        }
                        Ok::<Vec<(usize, String, f32)>, String>(local_results)
                    }));
                }

                for handle in handles {
                    let local_results = handle
                        .join()
                        .map_err(|_| "worker thread panicked".to_string())??;
                    for (orig_idx, text, conf) in local_results {
                        results[orig_idx] = (text, conf);
                    }
                }
                Ok::<(), String>(())
            })?;
        }

        drop(crops);
        trim_allocator_heap();
        timings.rec_ms = rec_clock.elapsed().as_secs_f32() * 1000.0;

        let valid_results: Vec<&(String, f32)> = results
            .iter()
            .filter(|(t, c)| *c >= 0.5 && !t.trim().is_empty())
            .collect();
        let texts: Vec<&str> = valid_results.iter().map(|(t, _)| t.as_str()).collect();
        let confidences: Vec<f32> = valid_results.iter().map(|(_, c)| *c).collect();
        let joined = texts.join("\n");
        let mean_conf = if confidences.is_empty() {
            0.0
        } else {
            confidences.iter().sum::<f32>() / confidences.len() as f32
        };
        Ok((joined, mean_conf, timings))
    }
}
