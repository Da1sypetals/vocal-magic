//! In-memory job store + a single model-resident worker thread.
//!
//! The HTTP layer submits jobs (writing the upload to disk and queuing an id);
//! a dedicated thread loads each MLX model once (lazily) and drains the queue,
//! updating job status as it progresses. A single worker keeps Metal command
//! submission serial, which the GAME model requires.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use babycat::{
    Signal, Waveform, WaveformArgs,
    constants::{DECODING_BACKEND_SYMPHONIA, RESAMPLE_MODE_LIBSAMPLERATE},
};
use game_mlxrs::{GameVocalTranscriber, midi, score_json};
use mb_roformer_mlx::{MelBandRoformer, WsaMbRoformer};
use mlx_rs::{
    Array, Device,
    ops::{self, indexing},
};
use srs_inference::Renaissance;
use srs_inference::audio::{AudioBuffer, SAMPLE_RATE};
use srs_inference::spectral::SpectralTransform;

const APP_SUPER_RESOLUTION: &str = "super-resolution";
const APP_TRANSCRIBE_MIDI: &str = "transcribe-midi";
const APP_SEPARATION: &str = "separation";
const MODEL_MELBAND_ROFORMER: &str = "melband-roformer";
const MODEL_WSA_MB_ROFORMER: &str = "wsa-mb-roformer-win10-sink8";
const SEPARATION_AUDIO_CHANNELS: i32 = 2;
const SEPARATION_SAMPLE_RATE: u32 = 44_100;
const MELBAND_CHUNK_SIZE_SAMPLES: i32 = 352_800;
const MELBAND_NUM_OVERLAP: i32 = 2;
const WSA_CHUNK_SECONDS: u32 = 8;
const WSA_BATCH_SIZE: i32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobStatus {
    Queued,
    Processing,
    Done,
    Failed,
}

/// Tunable parameters for the transcription (audio → MIDI) pipeline.
#[derive(Debug, Clone)]
pub struct MidiParams {
    pub t0: f32,
    pub nsteps: i32,
    pub seg_threshold: f32,
    pub seg_radius: f32,
    pub est_threshold: f32,
    pub tempo: f32,
    pub language: Option<String>,
    pub quantize: bool,
    pub quantize_equal_weight: bool,
}

impl Default for MidiParams {
    fn default() -> Self {
        Self {
            t0: 0.0,
            nsteps: 8,
            seg_threshold: 0.2,
            seg_radius: 0.02,
            est_threshold: 0.2,
            tempo: 120.0,
            language: None,
            quantize: false,
            quantize_equal_weight: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SubmitReq {
    pub name: String,
    pub app: String,
    pub model: String,
    pub model_name: String,
    pub midi: Option<MidiParams>,
}

#[derive(Debug, Clone)]
pub struct JobArtifact {
    pub key: String,
    pub label: String,
    pub kind: &'static str,
    pub path: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JobKind {
    SuperResolution,
    TranscribeMidi,
    Separate(SeparationModel),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeparationModel {
    MelBand,
    Wsa,
}

impl SeparationModel {
    fn from_id(model: &str) -> Result<Self, String> {
        match model {
            MODEL_MELBAND_ROFORMER => Ok(Self::MelBand),
            MODEL_WSA_MB_ROFORMER => Ok(Self::Wsa),
            _ => Err(format!("unsupported separation model: {model}")),
        }
    }
}

impl JobKind {
    fn from_request(app: &str, model: &str) -> Result<Self, String> {
        match app {
            APP_SUPER_RESOLUTION => Ok(Self::SuperResolution),
            APP_TRANSCRIBE_MIDI => Ok(Self::TranscribeMidi),
            APP_SEPARATION => Ok(Self::Separate(SeparationModel::from_id(model)?)),
            _ => Err(format!("unsupported app: {app}")),
        }
    }

    fn product(self) -> &'static str {
        match self {
            Self::TranscribeMidi => "midi",
            Self::SuperResolution | Self::Separate(_) => "audio",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Job {
    pub id: String,
    pub name: String,
    pub app: String,
    pub model: String,
    pub model_name: String,
    pub product: &'static str, // "audio" | "midi"
    pub status: JobStatus,
    pub created_at: u64,
    pub input_path: PathBuf,
    pub output_path: Option<PathBuf>, // wav (audio) or .mid (midi)
    pub score_path: Option<PathBuf>,  // score json (midi only)
    pub notes_path: Option<PathBuf>,  // piano-roll notes json (midi only)
    pub artifacts: Vec<JobArtifact>,
    pub note_count: Option<usize>,
    pub error: Option<String>,
    pub duration: Option<f64>,
    pub rtf: Option<f64>,
    kind: JobKind,
    midi: Option<MidiParams>,
}

#[derive(Default)]
struct Store {
    jobs: HashMap<String, Job>,
    /// Job ids in newest-first order.
    order: Vec<String>,
}

pub struct Engine {
    store: Arc<Mutex<Store>>,
    work_dir: PathBuf,
    worker_tx: Sender<String>,
    counter: AtomicU64,
}

/// Paths to the model assets the worker may need.
pub struct ModelPaths {
    pub sr_checkpoint: PathBuf,
    pub game_weights: PathBuf,
    pub game_config: PathBuf,
    pub separation_melband_checkpoint: PathBuf,
    pub separation_wsa_checkpoint: PathBuf,
}

#[derive(Default)]
struct SeparationModels {
    melband: Option<MelBandRoformer>,
    wsa: Option<WsaMbRoformer>,
}

impl Engine {
    pub fn new(paths: ModelPaths, work_dir: PathBuf) -> Self {
        let store = Arc::new(Mutex::new(Store::default()));
        let worker_tx = start_worker(paths, store.clone());
        Self {
            store,
            work_dir,
            worker_tx,
            counter: AtomicU64::new(0),
        }
    }

    fn next_id(&self) -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let n = self.counter.fetch_add(1, Ordering::Relaxed);
        format!("job-{nanos}-{n}")
    }

    pub async fn submit(&self, req: SubmitReq, data: &[u8]) -> std::io::Result<Job> {
        let id = self.next_id();
        let kind = JobKind::from_request(&req.app, &req.model)
            .map_err(|message| std::io::Error::new(std::io::ErrorKind::InvalidInput, message))?;
        let ext = Path::new(&req.name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("bin")
            .to_lowercase();
        let input_path = self.work_dir.join(format!("{id}.in.{ext}"));
        tokio::fs::write(&input_path, data).await?;

        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let job = Job {
            id: id.clone(),
            name: req.name,
            app: req.app,
            model: req.model,
            model_name: req.model_name,
            product: kind.product(),
            status: JobStatus::Queued,
            created_at,
            input_path,
            output_path: None,
            score_path: None,
            notes_path: None,
            artifacts: Vec::new(),
            note_count: None,
            error: None,
            duration: None,
            rtf: None,
            kind,
            midi: req.midi,
        };

        {
            let mut store = self.store.lock().unwrap();
            store.jobs.insert(id.clone(), job.clone());
            store.order.insert(0, id.clone());
        }

        if self.worker_tx.send(id.clone()).is_err() {
            self.mark_failed(&id, "推理服务不可用（worker 已退出）".into());
        }
        Ok(self.get(&id).unwrap_or(job))
    }

    pub fn get(&self, id: &str) -> Option<Job> {
        self.store.lock().unwrap().jobs.get(id).cloned()
    }

    pub fn list(&self, page: usize, per_page: usize) -> (Vec<Job>, usize) {
        let store = self.store.lock().unwrap();
        let total = store.order.len();
        let start = (page - 1) * per_page;
        let items = store
            .order
            .iter()
            .skip(start)
            .take(per_page)
            .filter_map(|id| store.jobs.get(id).cloned())
            .collect();
        (items, total)
    }

    fn mark_failed(&self, id: &str, msg: String) {
        set_status(&self.store, id, |j| {
            j.status = JobStatus::Failed;
            j.error = Some(msg);
        });
    }
}

fn set_status(store: &Arc<Mutex<Store>>, id: &str, f: impl FnOnce(&mut Job)) {
    let mut s = store.lock().unwrap();
    if let Some(j) = s.jobs.get_mut(id) {
        f(j);
    }
}

fn start_worker(paths: ModelPaths, store: Arc<Mutex<Store>>) -> Sender<String> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        Device::set_default(&Device::gpu());

        let spectral = SpectralTransform::new();
        let mut renaissance: Option<Renaissance> = None;
        let mut transcriber: Option<GameVocalTranscriber> = None;
        let mut separators = SeparationModels::default();

        while let Ok(id) = rx.recv() {
            set_status(&store, &id, |j| j.status = JobStatus::Processing);

            let job = {
                let s = store.lock().unwrap();
                s.jobs.get(&id).cloned()
            };
            let Some(job) = job else { continue };

            let result = match job.kind {
                JobKind::SuperResolution => {
                    run_super_resolution(&mut renaissance, &spectral, &paths, &job, &store, &id)
                }
                JobKind::TranscribeMidi => run_midi(&mut transcriber, &paths, &job, &store, &id),
                JobKind::Separate(separation_model) => {
                    run_separation(&mut separators, separation_model, &paths, &job, &store, &id)
                }
            };

            if let Err(e) = result {
                set_status(&store, &id, |j| {
                    j.status = JobStatus::Failed;
                    j.error = Some(e);
                });
            }
        }
    });
    tx
}

fn run_super_resolution(
    model: &mut Option<Renaissance>,
    spectral: &SpectralTransform,
    paths: &ModelPaths,
    job: &Job,
    store: &Arc<Mutex<Store>>,
    id: &str,
) -> Result<(), String> {
    if model.is_none() {
        let m =
            Renaissance::load(&paths.sr_checkpoint).map_err(|e| format!("模型加载失败: {e}"))?;
        *model = Some(m);
    }
    let model = model.as_mut().unwrap();

    let output_path = job.input_path.with_extension("out.wav");
    let started = Instant::now();
    let audio_secs = enhance(model, spectral, &job.input_path, &output_path)?;
    let elapsed = started.elapsed().as_secs_f64();
    let rtf = if audio_secs > 0.0 {
        elapsed / audio_secs
    } else {
        0.0
    };

    set_status(store, id, |j| {
        j.status = JobStatus::Done;
        j.output_path = Some(output_path.clone());
        j.artifacts = vec![JobArtifact {
            key: "output".to_owned(),
            label: "已修复".to_owned(),
            kind: "audio",
            path: output_path.clone(),
        }];
        j.duration = Some(elapsed);
        j.rtf = Some(rtf);
    });
    Ok(())
}

/// Renaissance enhancement; returns the input audio duration in seconds.
fn enhance(
    model: &mut Renaissance,
    spectral: &SpectralTransform,
    input_path: &Path,
    output_path: &Path,
) -> Result<f64, String> {
    let decoded = AudioBuffer::load(input_path).map_err(|e| e.to_string())?;
    let mut waveform = decoded.preprocess().map_err(|e| e.to_string())?;
    let normalization = waveform.normalize();
    let audio_secs = waveform.samples.len() as f64 / f64::from(SAMPLE_RATE);

    let input = Array::from_slice(&waveform.samples, &[1, waveform.samples.len() as i32]);
    let spectrum = spectral.stft(&input).map_err(|e| e.to_string())?;
    let enhanced = model.forward(&spectrum).map_err(|e| e.to_string())?;
    enhanced.eval().map_err(|e| e.to_string())?;

    let wave = spectral.istft(&enhanced).map_err(|e| e.to_string())?;
    let samples: Vec<f32> = wave
        .as_slice::<f32>()
        .iter()
        .map(|s| s * normalization)
        .collect();

    AudioBuffer {
        samples,
        sample_rate: SAMPLE_RATE,
    }
    .save_f32_wav(output_path)
    .map_err(|e| e.to_string())?;
    Ok(audio_secs)
}

fn run_midi(
    transcriber: &mut Option<GameVocalTranscriber>,
    paths: &ModelPaths,
    job: &Job,
    store: &Arc<Mutex<Store>>,
    id: &str,
) -> Result<(), String> {
    let params = job.midi.clone().unwrap_or_default();

    if transcriber.is_none() {
        if !paths.game_weights.exists() {
            return Err(format!(
                "GAME 权重缺失：{}（请提供 .safetensors 与 config.yaml 后重试）",
                paths.game_weights.display()
            ));
        }
        if !paths.game_config.exists() {
            return Err(format!("GAME 配置缺失：{}", paths.game_config.display()));
        }
        let mut t = GameVocalTranscriber::new(&paths.game_weights, &paths.game_config);
        t.load().map_err(|e| format!("GAME 模型加载失败: {e}"))?;
        *transcriber = Some(t);
    }
    let t = transcriber.as_mut().unwrap();
    t.t0 = params.t0;
    t.nsteps = params.nsteps;
    t.seg_threshold = params.seg_threshold;
    t.seg_radius = params.seg_radius;
    t.est_threshold = params.est_threshold;

    let language = params.language.as_deref();
    let started = Instant::now();
    let notes = if params.quantize {
        let weighted = !params.quantize_equal_weight;
        t.transcribe_quantized(&job.input_path, language, weighted)
            .map_err(|e| e.to_string())?
    } else {
        t.transcribe_with_language(&job.input_path, language)
            .map_err(|e| e.to_string())?
    };
    let elapsed = started.elapsed().as_secs_f64();

    let (durations, presence, scores) = game_mlxrs::notes_to_raw(&notes);
    let midi_path = job.input_path.with_extension("out.mid");
    let score_path = job.input_path.with_extension("out.json");
    let notes_path = job.input_path.with_extension("notes.json");

    midi::save_midi(&midi_path, &durations, &presence, &scores, params.tempo)
        .map_err(|e| format!("MIDI 写入失败: {e}"))?;
    score_json::save_json(&score_path, &durations, &presence, &scores)
        .map_err(|e| format!("JSON 写入失败: {e}"))?;

    // Compact notes JSON for the piano-roll preview.
    let notes_json: Vec<_> = notes
        .iter()
        .map(|n| serde_json::json!({ "onset": n.onset, "offset": n.offset, "pitch": n.pitch }))
        .collect();
    let span = notes.iter().map(|n| n.offset).fold(0.0f32, f32::max);
    let payload = serde_json::json!({ "notes": notes_json, "count": notes.len(), "span": span });
    std::fs::write(&notes_path, payload.to_string()).map_err(|e| e.to_string())?;

    let count = notes.len();
    set_status(store, id, |j| {
        j.status = JobStatus::Done;
        j.output_path = Some(midi_path.clone());
        j.score_path = Some(score_path.clone());
        j.notes_path = Some(notes_path.clone());
        j.artifacts = vec![
            JobArtifact {
                key: "midi".to_owned(),
                label: "MIDI".to_owned(),
                kind: "file",
                path: midi_path.clone(),
            },
            JobArtifact {
                key: "json".to_owned(),
                label: "JSON".to_owned(),
                kind: "file",
                path: score_path.clone(),
            },
            JobArtifact {
                key: "notes".to_owned(),
                label: "Notes".to_owned(),
                kind: "file",
                path: notes_path.clone(),
            },
        ];
        j.note_count = Some(count);
        j.duration = Some(elapsed);
    });
    Ok(())
}

fn run_separation(
    models: &mut SeparationModels,
    separation_model: SeparationModel,
    paths: &ModelPaths,
    job: &Job,
    store: &Arc<Mutex<Store>>,
    id: &str,
) -> Result<(), String> {
    let vocal_path = job.input_path.with_extension("vocal.wav");
    let instrumental_path = job.input_path.with_extension("instrumental.wav");
    let started = Instant::now();
    let audio_secs = match separation_model {
        SeparationModel::MelBand => {
            if models.melband.is_none() {
                if !paths.separation_melband_checkpoint.exists() {
                    return Err(format!(
                        "MelBand checkpoint missing: {}",
                        paths.separation_melband_checkpoint.display()
                    ));
                }
                let mut model = MelBandRoformer::new();
                model
                    .load(&paths.separation_melband_checkpoint)
                    .map_err(|e| format!("MelBand model load failed: {e}"))?;
                models.melband = Some(model);
            }
            separate_melband(
                models.melband.as_mut().unwrap(),
                &job.input_path,
                &vocal_path,
                &instrumental_path,
            )?
        }
        SeparationModel::Wsa => {
            if models.wsa.is_none() {
                if !paths.separation_wsa_checkpoint.exists() {
                    return Err(format!(
                        "WSA checkpoint missing: {}",
                        paths.separation_wsa_checkpoint.display()
                    ));
                }
                let mut model = WsaMbRoformer::new();
                model
                    .load(&paths.separation_wsa_checkpoint)
                    .map_err(|e| format!("WSA model load failed: {e}"))?;
                models.wsa = Some(model);
            }
            separate_wsa(
                models.wsa.as_mut().unwrap(),
                &job.input_path,
                &vocal_path,
                &instrumental_path,
            )?
        }
    };
    let elapsed = started.elapsed().as_secs_f64();
    let rtf = if audio_secs > 0.0 {
        elapsed / audio_secs
    } else {
        0.0
    };

    set_status(store, id, |j| {
        j.status = JobStatus::Done;
        j.output_path = Some(vocal_path.clone());
        j.artifacts = vec![
            JobArtifact {
                key: "vocal".to_owned(),
                label: "Vocal".to_owned(),
                kind: "audio",
                path: vocal_path.clone(),
            },
            JobArtifact {
                key: "instrumental".to_owned(),
                label: "Instrumental".to_owned(),
                kind: "audio",
                path: instrumental_path.clone(),
            },
        ];
        j.duration = Some(elapsed);
        j.rtf = Some(rtf);
    });
    Ok(())
}

fn separate_melband(
    model: &mut MelBandRoformer,
    input_path: &Path,
    vocal_path: &Path,
    instrumental_path: &Path,
) -> Result<f64, String> {
    let audio = load_separation_audio(input_path)?;
    let audio_samples = audio.shape()[1];
    let audio_secs = audio_samples as f64 / f64::from(SEPARATION_SAMPLE_RATE);
    let vocal = demix_melband_track(model, &audio)?;
    let instrumental = ops::subtract(&audio, &vocal).map_err(|e| e.to_string())?;
    save_separation_audio(&vocal, vocal_path)?;
    save_separation_audio(&instrumental, instrumental_path)?;
    Ok(audio_secs)
}

fn separate_wsa(
    model: &mut WsaMbRoformer,
    input_path: &Path,
    vocal_path: &Path,
    instrumental_path: &Path,
) -> Result<f64, String> {
    let audio = load_separation_audio(input_path)?;
    let audio_samples = audio.shape()[1];
    let audio_secs = audio_samples as f64 / f64::from(SEPARATION_SAMPLE_RATE);
    let chunk_size_samples = WSA_CHUNK_SECONDS
        .checked_mul(SEPARATION_SAMPLE_RATE)
        .ok_or_else(|| "WSA chunk-size * sample-rate overflowed".to_owned())?
        as i32;
    let (chunks, audio_samples) = chunk_wsa_audio(&audio, chunk_size_samples)?;
    let outputs = infer_wsa_chunks(model, &chunks, WSA_BATCH_SIZE)?;
    let vocal = restore_wsa_audio(&outputs, audio_samples)?;
    let instrumental = ops::subtract(&audio, &vocal).map_err(|e| e.to_string())?;
    save_separation_audio(&vocal, vocal_path)?;
    save_separation_audio(&instrumental, instrumental_path)?;
    Ok(audio_secs)
}

fn path_to_str(path: &Path) -> Result<&str, String> {
    path.to_str()
        .ok_or_else(|| format!("path is not valid UTF-8: {path:?}"))
}

fn load_separation_audio(path: &Path) -> Result<Array, String> {
    let input_metadata = Waveform::from_file(
        path_to_str(path)?,
        WaveformArgs {
            decoding_backend: DECODING_BACKEND_SYMPHONIA,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    if input_metadata.frame_rate_hz() != SEPARATION_SAMPLE_RATE {
        eprintln!(
            "input sample rate is {} Hz; resampling to {} Hz for separation",
            input_metadata.frame_rate_hz(),
            SEPARATION_SAMPLE_RATE
        );
    }

    let waveform = Waveform::from_file(
        path_to_str(path)?,
        WaveformArgs {
            frame_rate_hz: SEPARATION_SAMPLE_RATE,
            resample_mode: RESAMPLE_MODE_LIBSAMPLERATE,
            decoding_backend: DECODING_BACKEND_SYMPHONIA,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    if waveform.num_frames() == 0 {
        return Err("input audio contains no decoded frames".to_owned());
    }
    if waveform.num_channels() == 0 {
        return Err("input audio contains no decoded channels".to_owned());
    }

    let audio = Array::from_slice(
        waveform.to_interleaved_samples(),
        &[waveform.num_frames() as i32, waveform.num_channels() as i32],
    );
    let audio = ops::transpose_axes(&audio, &[1, 0]).map_err(|e| e.to_string())?;
    if waveform.num_channels() == 1 {
        ops::repeat_axis::<f32>(audio, SEPARATION_AUDIO_CHANNELS, 0).map_err(|e| e.to_string())
    } else if waveform.num_channels() == 2 {
        Ok(audio)
    } else {
        let indices = Array::from_slice(&[0i64, 1], &[2]);
        indexing::take_axis(&audio, &indices, 0).map_err(|e| e.to_string())
    }
}

fn chunk_wsa_audio(audio: &Array, chunk_size_samples: i32) -> Result<(Array, i32), String> {
    let audio_samples = audio.shape()[1];
    let full_samples =
        ((audio_samples + chunk_size_samples - 1) / chunk_size_samples) * chunk_size_samples;
    let padded = if full_samples > audio_samples {
        let padding = ops::zeros::<f32>(&[SEPARATION_AUDIO_CHANNELS, full_samples - audio_samples])
            .map_err(|e| e.to_string())?;
        ops::concatenate_axis(&[audio, &padding], 1).map_err(|e| e.to_string())?
    } else {
        audio.clone()
    };
    let clips = full_samples / chunk_size_samples;
    let strides = padded.strides();
    let chunks = ops::as_strided(
        &padded,
        &[SEPARATION_AUDIO_CHANNELS, clips, chunk_size_samples],
        &[
            strides[0] as i64,
            (chunk_size_samples as usize * strides[1]) as i64,
            strides[1] as i64,
        ],
        0,
    )
    .map_err(|e| e.to_string())?;

    Ok((
        ops::transpose_axes(&chunks, &[1, 0, 2]).map_err(|e| e.to_string())?,
        audio_samples,
    ))
}

fn infer_wsa_chunks(
    model: &mut WsaMbRoformer,
    chunks: &Array,
    batch_size: i32,
) -> Result<Array, String> {
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
        )
        .map_err(|e| e.to_string())?;
        outputs.push(model.infer(&batch_chunks).map_err(|e| e.to_string())?);
        pointer += current_batch;
    }

    ops::concatenate_axis(&outputs, 0).map_err(|e| e.to_string())
}

fn restore_wsa_audio(outputs: &Array, audio_samples: i32) -> Result<Array, String> {
    let clips = outputs.shape()[0];
    let channels = outputs.shape()[1];
    let chunk_size_samples = outputs.shape()[2];
    let output = ops::transpose_axes(outputs, &[1, 0, 2]).map_err(|e| e.to_string())?;
    let output = ops::reshape(&output, &[channels, clips * chunk_size_samples])
        .map_err(|e| e.to_string())?;
    let strides = output.strides();
    ops::as_strided(
        &output,
        &[channels, audio_samples],
        &[strides[0] as i64, strides[1] as i64],
        0,
    )
    .map_err(|e| e.to_string())
}

fn reflect_pad_for_overlap(audio: &Array, border: i32) -> Result<Array, String> {
    let audio_len = audio.shape()[1];
    let padded_len = audio_len + border * 2;
    let pos = ops::arange::<_, i32>(0, padded_len, None).map_err(|e| e.to_string())?;
    let left_source =
        ops::subtract(&Array::from_slice(&[border], &[]), &pos).map_err(|e| e.to_string())?;
    let middle_source =
        ops::subtract(&pos, &Array::from_slice(&[border], &[])).map_err(|e| e.to_string())?;
    let right_source = ops::subtract(&Array::from_slice(&[2 * audio_len + border - 2], &[]), &pos)
        .map_err(|e| e.to_string())?;
    let left_mask = pos
        .lt(&Array::from_slice(&[border], &[]))
        .map_err(|e| e.to_string())?;
    let right_mask = pos
        .ge(&Array::from_slice(&[audio_len + border], &[]))
        .map_err(|e| e.to_string())?;
    let indices =
        ops::which(&left_mask, &left_source, &middle_source).map_err(|e| e.to_string())?;
    let indices = ops::which(&right_mask, &right_source, &indices).map_err(|e| e.to_string())?;
    indexing::take_axis(audio, &indices, 1).map_err(|e| e.to_string())
}

fn melband_windowing_array() -> Result<Array, String> {
    let fade_size = MELBAND_CHUNK_SIZE_SAMPLES / 10;
    let pos = ops::arange::<_, f32>(0.0, MELBAND_CHUNK_SIZE_SAMPLES as f32, None)
        .map_err(|e| e.to_string())?;
    let ones = ops::ones::<f32>(&[MELBAND_CHUNK_SIZE_SAMPLES]).map_err(|e| e.to_string())?;
    let denom = Array::from_slice(&[(fade_size - 1) as f32], &[]);
    let fade_in = ops::divide(&pos, &denom).map_err(|e| e.to_string())?;
    let fade_out = ops::divide(
        &ops::subtract(
            &Array::from_slice(&[(MELBAND_CHUNK_SIZE_SAMPLES - 1) as f32], &[]),
            &pos,
        )
        .map_err(|e| e.to_string())?,
        &denom,
    )
    .map_err(|e| e.to_string())?;
    let left_mask = pos
        .lt(&Array::from_slice(&[fade_size as f32], &[]))
        .map_err(|e| e.to_string())?;
    let right_mask = pos
        .ge(&Array::from_slice(
            &[(MELBAND_CHUNK_SIZE_SAMPLES - fade_size) as f32],
            &[],
        ))
        .map_err(|e| e.to_string())?;
    let window = ops::which(&left_mask, &fade_in, &ones).map_err(|e| e.to_string())?;
    ops::which(&right_mask, &fade_out, &window).map_err(|e| e.to_string())
}

fn pad_melband_chunk_to_model_size(part: &Array, length: i32) -> Result<Array, String> {
    if length == MELBAND_CHUNK_SIZE_SAMPLES {
        return Ok(part.clone());
    }

    if length > MELBAND_CHUNK_SIZE_SAMPLES / 2 + 1 {
        let pos = ops::arange::<_, i32>(0, MELBAND_CHUNK_SIZE_SAMPLES, None)
            .map_err(|e| e.to_string())?;
        let right_source = ops::subtract(&Array::from_slice(&[2 * length - 2], &[]), &pos)
            .map_err(|e| e.to_string())?;
        let in_audio = pos
            .lt(&Array::from_slice(&[length], &[]))
            .map_err(|e| e.to_string())?;
        let indices = ops::which(&in_audio, &pos, &right_source).map_err(|e| e.to_string())?;
        indexing::take_axis(part, &indices, 1).map_err(|e| e.to_string())
    } else {
        let padding = ops::zeros::<f32>(&[
            SEPARATION_AUDIO_CHANNELS,
            MELBAND_CHUNK_SIZE_SAMPLES - length,
        ])
        .map_err(|e| e.to_string())?;
        ops::concatenate_axis(&[part, &padding], 1).map_err(|e| e.to_string())
    }
}

fn scatter_add_overlap(
    destination: &Array,
    updates: &Array,
    start: i32,
    length: i32,
    total_length: i32,
) -> Result<Array, String> {
    let channel_offsets = ops::multiply(
        &ops::arange::<_, i64>(0i64, SEPARATION_AUDIO_CHANNELS as i64, None)
            .map_err(|e| e.to_string())?
            .reshape(&[1, SEPARATION_AUDIO_CHANNELS, 1])
            .map_err(|e| e.to_string())?,
        &Array::from_slice(&[total_length as i64], &[]),
    )
    .map_err(|e| e.to_string())?;
    let time_offsets = ops::arange::<_, i64>(start as i64, (start + length) as i64, None)
        .map_err(|e| e.to_string())?
        .reshape(&[1, 1, length])
        .map_err(|e| e.to_string())?;
    let indices = ops::add(&channel_offsets, &time_offsets).map_err(|e| e.to_string())?;
    let indices =
        ops::reshape(&indices, &[SEPARATION_AUDIO_CHANNELS * length]).map_err(|e| e.to_string())?;
    let flat_destination = ops::reshape(destination, &[SEPARATION_AUDIO_CHANNELS * total_length])
        .map_err(|e| e.to_string())?;
    let flat_updates = ops::reshape(updates, &[SEPARATION_AUDIO_CHANNELS * length, 1])
        .map_err(|e| e.to_string())?;
    let flat_output = indexing::scatter_add_single(flat_destination, &indices, &flat_updates, 0)
        .map_err(|e| e.to_string())?;
    ops::reshape(&flat_output, &[1, SEPARATION_AUDIO_CHANNELS, total_length])
        .map_err(|e| e.to_string())
}

fn demix_melband_track(model: &mut MelBandRoformer, audio: &Array) -> Result<Array, String> {
    let step = MELBAND_CHUNK_SIZE_SAMPLES / MELBAND_NUM_OVERLAP;
    let fade_size = MELBAND_CHUNK_SIZE_SAMPLES / 10;
    let border = MELBAND_CHUNK_SIZE_SAMPLES - step;
    let original_len = audio.shape()[1];
    let padded = original_len > 2 * border && border > 0;
    let mix = if padded {
        reflect_pad_for_overlap(audio, border)?
    } else {
        audio.clone()
    };
    let total_length = mix.shape()[1];
    let base_window = melband_windowing_array()?;
    let window_pos =
        ops::arange::<_, i32>(0, MELBAND_CHUNK_SIZE_SAMPLES, None).map_err(|e| e.to_string())?;
    let window_ones = ops::ones::<f32>(&[MELBAND_CHUNK_SIZE_SAMPLES]).map_err(|e| e.to_string())?;
    let mut result = ops::zeros::<f32>(&[1, SEPARATION_AUDIO_CHANNELS, total_length])
        .map_err(|e| e.to_string())?;
    let mut counter = ops::zeros::<f32>(&[1, SEPARATION_AUDIO_CHANNELS, total_length])
        .map_err(|e| e.to_string())?;
    let mix_strides = mix.strides();
    let mut offset = 0;

    while offset < total_length {
        let length = (total_length - offset).min(MELBAND_CHUNK_SIZE_SAMPLES);
        let part = ops::as_strided(
            &mix,
            &[SEPARATION_AUDIO_CHANNELS, length],
            &[mix_strides[0] as i64, mix_strides[1] as i64],
            offset as usize * mix_strides[1],
        )
        .map_err(|e| e.to_string())?;
        let part = pad_melband_chunk_to_model_size(&part, length)?;
        let model_input = part.expand_dims(0).map_err(|e| e.to_string())?;
        let model_output = model.infer(&model_input).map_err(|e| e.to_string())?;
        let output_strides = model_output.strides();
        let model_output = ops::as_strided(
            &model_output,
            &[1, SEPARATION_AUDIO_CHANNELS, length],
            &[
                output_strides[0] as i64,
                output_strides[1] as i64,
                output_strides[2] as i64,
            ],
            0,
        )
        .map_err(|e| e.to_string())?;

        let window = if offset == 0 {
            let mask = window_pos
                .lt(&Array::from_slice(&[fade_size], &[]))
                .map_err(|e| e.to_string())?;
            ops::which(&mask, &window_ones, &base_window).map_err(|e| e.to_string())?
        } else if offset + MELBAND_CHUNK_SIZE_SAMPLES >= total_length {
            let mask = window_pos
                .ge(&Array::from_slice(
                    &[MELBAND_CHUNK_SIZE_SAMPLES - fade_size],
                    &[],
                ))
                .map_err(|e| e.to_string())?;
            ops::which(&mask, &window_ones, &base_window).map_err(|e| e.to_string())?
        } else {
            base_window.clone()
        };
        let window_strides = window.strides();
        let window = ops::as_strided(&window, &[length], &[window_strides[0] as i64], 0)
            .map_err(|e| e.to_string())?;
        let window = window.reshape(&[1, 1, length]).map_err(|e| e.to_string())?;
        let weighted = ops::multiply(&model_output, &window).map_err(|e| e.to_string())?;
        let counter_update = ops::broadcast_to(&window, &[1, SEPARATION_AUDIO_CHANNELS, length])
            .map_err(|e| e.to_string())?;
        result = scatter_add_overlap(&result, &weighted, offset, length, total_length)?;
        counter = scatter_add_overlap(&counter, &counter_update, offset, length, total_length)?;
        offset += step;
    }

    let estimated = ops::divide(&result, &counter).map_err(|e| e.to_string())?;
    let estimated = estimated.squeeze_axes(&[0]).map_err(|e| e.to_string())?;
    if padded {
        let strides = estimated.strides();
        ops::as_strided(
            &estimated,
            &[SEPARATION_AUDIO_CHANNELS, original_len],
            &[strides[0] as i64, strides[1] as i64],
            border as usize * strides[1],
        )
        .map_err(|e| e.to_string())
    } else {
        Ok(estimated)
    }
}

fn save_separation_audio(audio: &Array, output_path: &Path) -> Result<(), String> {
    let interleaved = ops::transpose_axes(audio, &[1, 0]).map_err(|e| e.to_string())?;
    let interleaved = ops::flatten(&interleaved, None, None).map_err(|e| e.to_string())?;
    let waveform = Waveform::new(
        SEPARATION_SAMPLE_RATE,
        SEPARATION_AUDIO_CHANNELS as u16,
        interleaved.as_slice::<f32>().to_vec(),
    );
    waveform
        .to_wav_file(path_to_str(output_path)?)
        .map_err(|e| e.to_string())
}
