// 源分离推理：直接移植 mb-roformer-mlx 的 examples/infer.rs（full）
// 与 examples/infer_wsa.rs（windowed sink）的真实管线。

use anyhow::Result;
use babycat::{
    constants::{DECODING_BACKEND_SYMPHONIA, RESAMPLE_MODE_LIBSAMPLERATE},
    Signal, Waveform, WaveformArgs,
};
use mb_roformer_mlx::{MelBandRoformer, WsaMbRoformer};
use mlx_rs::{
    ops::{self, indexing},
    Array,
};
use std::path::Path;

use crate::db::OutputFile;

const AUDIO_CHANNELS: i32 = 2;
const MODEL_SAMPLE_RATE: u32 = 44_100;

fn load_audio(input_path: &Path) -> Result<Array> {
    let path = input_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("input path is not valid UTF-8"))?;
    let waveform_args = WaveformArgs {
        frame_rate_hz: MODEL_SAMPLE_RATE,
        resample_mode: RESAMPLE_MODE_LIBSAMPLERATE,
        decoding_backend: DECODING_BACKEND_SYMPHONIA,
        ..Default::default()
    };
    let waveform = Waveform::from_file(path, waveform_args)?;
    anyhow::ensure!(waveform.num_frames() > 0, "input audio contains no decoded frames");
    anyhow::ensure!(waveform.num_channels() > 0, "input audio contains no decoded channels");

    let audio = Array::from_slice(
        waveform.to_interleaved_samples(),
        &[waveform.num_frames() as i32, waveform.num_channels() as i32],
    );
    let audio = ops::transpose_axes(&audio, &[1, 0])?;
    if waveform.num_channels() == 1 {
        Ok(ops::repeat_axis::<f32>(audio, AUDIO_CHANNELS, 0)?)
    } else if waveform.num_channels() == 2 {
        Ok(audio)
    } else {
        let indices = Array::from_slice(&[0i64, 1], &[2]);
        Ok(indexing::take_axis(&audio, &indices, 0)?)
    }
}

fn save_audio(audio: &Array, output_path: &Path) -> Result<()> {
    let path = output_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("output path is not valid UTF-8"))?;
    let interleaved = ops::transpose_axes(audio, &[1, 0])?;
    let interleaved = ops::flatten(&interleaved, None, None)?;
    let waveform = Waveform::new(
        MODEL_SAMPLE_RATE,
        AUDIO_CHANNELS as u16,
        interleaved.as_slice::<f32>().to_vec(),
    );
    waveform.to_wav_file(path)?;
    Ok(())
}

// ---------- full variant (infer.rs) ----------

fn reflect_pad_for_overlap(audio: &Array, border: i32) -> Result<Array> {
    let shape = audio.shape();
    let audio_len = shape[1];
    let padded_len = audio_len + border * 2;
    let pos = ops::arange::<_, i32>(0, padded_len, None)?;
    let left_source = ops::subtract(&Array::from_slice(&[border], &[]), &pos)?;
    let middle_source = ops::subtract(&pos, &Array::from_slice(&[border], &[]))?;
    let right_source = ops::subtract(&Array::from_slice(&[2 * audio_len + border - 2], &[]), &pos)?;
    let left_mask = pos.lt(&Array::from_slice(&[border], &[]))?;
    let right_mask = pos.ge(&Array::from_slice(&[audio_len + border], &[]))?;
    let indices = ops::which(&left_mask, &left_source, &middle_source)?;
    let indices = ops::which(&right_mask, &right_source, &indices)?;
    Ok(indexing::take_axis(audio, &indices, 1)?)
}

fn windowing_array(chunk_size_samples: i32) -> Result<Array> {
    let fade_size = chunk_size_samples / 10;
    let pos = ops::arange::<_, f32>(0.0, chunk_size_samples as f32, None)?;
    let ones = ops::ones::<f32>(&[chunk_size_samples])?;
    let denom = Array::from_slice(&[(fade_size - 1) as f32], &[]);
    let fade_in = ops::divide(&pos, &denom)?;
    let fade_out = ops::divide(
        &ops::subtract(&Array::from_slice(&[(chunk_size_samples - 1) as f32], &[]), &pos)?,
        &denom,
    )?;
    let left_mask = pos.lt(&Array::from_slice(&[fade_size as f32], &[]))?;
    let right_mask = pos.ge(&Array::from_slice(&[(chunk_size_samples - fade_size) as f32], &[]))?;
    let window = ops::which(&left_mask, &fade_in, &ones)?;
    Ok(ops::which(&right_mask, &fade_out, &window)?)
}

fn pad_chunk_to_model_size(part: &Array, length: i32, chunk_size_samples: i32) -> Result<Array> {
    if length == chunk_size_samples {
        return Ok(part.clone());
    }
    if length > chunk_size_samples / 2 + 1 {
        let pos = ops::arange::<_, i32>(0, chunk_size_samples, None)?;
        let right_source = ops::subtract(&Array::from_slice(&[2 * length - 2], &[]), &pos)?;
        let in_audio = pos.lt(&Array::from_slice(&[length], &[]))?;
        let indices = ops::which(&in_audio, &pos, &right_source)?;
        Ok(indexing::take_axis(part, &indices, 1)?)
    } else {
        let padding = ops::zeros::<f32>(&[AUDIO_CHANNELS, chunk_size_samples - length])?;
        Ok(ops::concatenate_axis(&[part, &padding], 1)?)
    }
}

fn scatter_add_overlap(
    destination: &Array,
    updates: &Array,
    start: i32,
    length: i32,
    total_length: i32,
) -> Result<Array> {
    let channel_offsets = ops::multiply(
        &ops::arange::<_, i64>(0i64, AUDIO_CHANNELS as i64, None)?.reshape(&[1, AUDIO_CHANNELS, 1])?,
        &Array::from_slice(&[total_length as i64], &[]),
    )?;
    let time_offsets = ops::arange::<_, i64>(start as i64, (start + length) as i64, None)?
        .reshape(&[1, 1, length])?;
    let indices = ops::add(&channel_offsets, &time_offsets)?;
    let indices = ops::reshape(&indices, &[AUDIO_CHANNELS * length])?;
    let flat_destination = ops::reshape(destination, &[AUDIO_CHANNELS * total_length])?;
    let flat_updates = ops::reshape(updates, &[AUDIO_CHANNELS * length, 1])?;
    let flat_output = indexing::scatter_add_single(flat_destination, &indices, &flat_updates, 0)?;
    Ok(ops::reshape(&flat_output, &[1, AUDIO_CHANNELS, total_length])?)
}

fn demix_track(model: &mut MelBandRoformer, audio: &Array, num_overlap: i32, chunk_size_samples: i32) -> Result<Array> {
    let step = chunk_size_samples / num_overlap;
    let fade_size = chunk_size_samples / 10;
    let border = chunk_size_samples - step;
    let original_len = audio.shape()[1];
    let padded = original_len > 2 * border && border > 0;
    let mix = if padded {
        reflect_pad_for_overlap(audio, border)?
    } else {
        audio.clone()
    };
    let total_length = mix.shape()[1];
    let base_window = windowing_array(chunk_size_samples)?;
    let window_pos = ops::arange::<_, i32>(0, chunk_size_samples, None)?;
    let window_ones = ops::ones::<f32>(&[chunk_size_samples])?;
    let mut result = ops::zeros::<f32>(&[1, AUDIO_CHANNELS, total_length])?;
    let mut counter = ops::zeros::<f32>(&[1, AUDIO_CHANNELS, total_length])?;
    let mix_strides = mix.strides();
    let mut offset = 0;

    while offset < total_length {
        let length = (total_length - offset).min(chunk_size_samples);
        let part = ops::as_strided(
            &mix,
            &[AUDIO_CHANNELS, length],
            &[mix_strides[0] as i64, mix_strides[1] as i64],
            offset as usize * mix_strides[1],
        )?;
        let part = pad_chunk_to_model_size(&part, length, chunk_size_samples)?;
        let model_input = part.expand_dims(0)?;
        let model_output = model.infer(&model_input)?;
        let output_strides = model_output.strides();
        let model_output = ops::as_strided(
            &model_output,
            &[1, AUDIO_CHANNELS, length],
            &[
                output_strides[0] as i64,
                output_strides[1] as i64,
                output_strides[2] as i64,
            ],
            0,
        )?;

        let window = if offset == 0 {
            let mask = window_pos.lt(&Array::from_slice(&[fade_size], &[]))?;
            ops::which(&mask, &window_ones, &base_window)?
        } else if offset + chunk_size_samples >= total_length {
            let mask = window_pos.ge(&Array::from_slice(&[chunk_size_samples - fade_size], &[]))?;
            ops::which(&mask, &window_ones, &base_window)?
        } else {
            base_window.clone()
        };
        let window_strides = window.strides();
        let window = ops::as_strided(&window, &[length], &[window_strides[0] as i64], 0)?;
        let window = window.reshape(&[1, 1, length])?;
        let weighted = ops::multiply(&model_output, &window)?;
        let counter_update = ops::broadcast_to(&window, &[1, AUDIO_CHANNELS, length])?;
        result = scatter_add_overlap(&result, &weighted, offset, length, total_length)?;
        counter = scatter_add_overlap(&counter, &counter_update, offset, length, total_length)?;
        offset += step;
    }

    let estimated = ops::divide(&result, &counter)?;
    let estimated = estimated.squeeze_axes(&[0])?;
    if padded {
        let strides = estimated.strides();
        Ok(ops::as_strided(
            &estimated,
            &[AUDIO_CHANNELS, original_len],
            &[strides[0] as i64, strides[1] as i64],
            border as usize * strides[1],
        )?)
    } else {
        Ok(estimated)
    }
}

// ---------- wsa variant (infer_wsa.rs) ----------

fn chunk_audio(audio: &Array, chunk_size_samples: i32) -> Result<(Array, i32)> {
    let audio_samples = audio.shape()[1];
    let full_samples =
        ((audio_samples + chunk_size_samples - 1) / chunk_size_samples) * chunk_size_samples;
    let padded = if full_samples > audio_samples {
        let padding = ops::zeros::<f32>(&[AUDIO_CHANNELS, full_samples - audio_samples])?;
        ops::concatenate_axis(&[audio, &padding], 1)?
    } else {
        audio.clone()
    };
    let clips = full_samples / chunk_size_samples;
    let strides = padded.strides();
    let chunks = ops::as_strided(
        &padded,
        &[AUDIO_CHANNELS, clips, chunk_size_samples],
        &[
            strides[0] as i64,
            (chunk_size_samples as usize * strides[1]) as i64,
            strides[1] as i64,
        ],
        0,
    )?;
    Ok((ops::transpose_axes(&chunks, &[1, 0, 2])?, audio_samples))
}

fn infer_chunks(model: &mut WsaMbRoformer, chunks: &Array, batch_size: i32) -> Result<Array> {
    let clips = chunks.shape()[0];
    let channels = chunks.shape()[1];
    let chunk_size_samples = chunks.shape()[2];
    let strides = chunks.strides();
    let mut pointer = 0;
    let mut outputs = Vec::new();
    while pointer < clips {
        let current_batch = (clips - pointer).min(batch_size);
        let batch_chunks = ops::as_strided(
            chunks,
            &[current_batch, channels, chunk_size_samples],
            &[strides[0] as i64, strides[1] as i64, strides[2] as i64],
            pointer as usize * strides[0],
        )?;
        outputs.push(model.infer(&batch_chunks)?);
        pointer += current_batch;
    }
    Ok(ops::concatenate_axis(&outputs, 0)?)
}

fn restore_audio(outputs: &Array, audio_samples: i32) -> Result<Array> {
    let clips = outputs.shape()[0];
    let channels = outputs.shape()[1];
    let chunk_size_samples = outputs.shape()[2];
    let output = ops::transpose_axes(outputs, &[1, 0, 2])?;
    let output = ops::reshape(&output, &[channels, clips * chunk_size_samples])?;
    let strides = output.strides();
    Ok(ops::as_strided(
        &output,
        &[channels, audio_samples],
        &[strides[0] as i64, strides[1] as i64],
        0,
    )?)
}

fn write_stems(audio: &Array, vocal: &Array, work_dir: &Path, stem: &str) -> Result<Vec<OutputFile>> {
    let instrumental = ops::subtract(audio, vocal)?;
    let vname = format!("{stem}_vocal.wav");
    let iname = format!("{stem}_instrumental.wav");
    save_audio(vocal, &work_dir.join(&vname))?;
    save_audio(&instrumental, &work_dir.join(&iname))?;
    Ok(vec![
        OutputFile { name: vname, kind: "audio".into(), label: "Vocal".into() },
        OutputFile { name: iname, kind: "audio".into(), label: "Instrumental".into() },
    ])
}

pub fn run_full(
    model: &mut MelBandRoformer,
    input_path: &Path,
    work_dir: &Path,
    chunk_size_sec: i32,
    overlap: i32,
    stem: &str,
) -> Result<Vec<OutputFile>> {
    anyhow::ensure!(chunk_size_sec > 0, "chunk-size must be positive");
    anyhow::ensure!(overlap > 0, "overlap must be positive");
    let chunk_size_samples = chunk_size_sec * MODEL_SAMPLE_RATE as i32;
    let audio = load_audio(input_path)?;
    let vocal = demix_track(model, &audio, overlap, chunk_size_samples)?;
    write_stems(&audio, &vocal, work_dir, stem)
}

pub fn run_wsa(
    model: &mut WsaMbRoformer,
    input_path: &Path,
    work_dir: &Path,
    chunk_size_sec: i32,
    batch_size: i32,
    stem: &str,
) -> Result<Vec<OutputFile>> {
    anyhow::ensure!(chunk_size_sec > 0, "chunk-size must be positive");
    anyhow::ensure!(batch_size > 0, "batch-size must be positive");
    let chunk_size_samples = chunk_size_sec * MODEL_SAMPLE_RATE as i32;
    let audio = load_audio(input_path)?;
    let (chunks, audio_samples) = chunk_audio(&audio, chunk_size_samples)?;
    let outputs = infer_chunks(model, &chunks, batch_size)?;
    let vocal = restore_audio(&outputs, audio_samples)?;
    write_stems(&audio, &vocal, work_dir, stem)
}
