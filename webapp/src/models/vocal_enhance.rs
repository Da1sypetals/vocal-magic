// 人声增强：移植 srs-inference 的批处理推理管线（Renaissance）。

use anyhow::Result;
use mlx_rs::Array;
use srs_inference::audio::{AudioBuffer, SAMPLE_RATE};
use srs_inference::spectral::SpectralTransform;
use srs_inference::Renaissance;
use std::path::Path;

use crate::db::OutputFile;

pub fn run(
    model: &mut Renaissance,
    input_path: &Path,
    work_dir: &Path,
    stem: &str,
) -> Result<Vec<OutputFile>> {
    let spectral = SpectralTransform::new();
    let mut buf = AudioBuffer::load(input_path)?.preprocess()?;
    let peak = buf.normalize();

    let input = Array::from_slice(&buf.samples, &[1, buf.samples.len() as i32]);
    let spec = spectral.stft(&input)?;
    let out_spec = model.forward(&spec)?;
    let wav = spectral.istft(&out_spec)?;

    let samples: Vec<f32> = wav.as_slice::<f32>().iter().map(|s| s * peak).collect();
    let out = AudioBuffer {
        samples,
        sample_rate: SAMPLE_RATE,
    };
    let fname = format!("{stem}_enhanced.wav");
    let out_path = work_dir.join(&fname);
    out.save_f32_wav(&out_path)?;

    Ok(vec![OutputFile {
        name: fname,
        kind: "audio".into(),
        label: "Enhanced".into(),
    }])
}
