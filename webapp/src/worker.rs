use anyhow::{bail, Context, Result};
use mlx_rs::Device;
use rusqlite::Connection;
use serde_json::Value;
use std::path::Path;
use std::sync::mpsc::Receiver;

use crate::apps;
use crate::db::{self, OutputFile};
use crate::events::{EventTx, JobEvent};
use crate::models::vocal2midi::Vocal2MidiParams;
use crate::models::{source_separation, vocal2midi, vocal_enhance, Loaded};
use crate::storage::Paths;
use game_mlxrs::GameVocalTranscriber;
use mb_roformer_mlx::{MelBandRoformer, WsaMbRoformer};
use srs_inference::Renaissance;

pub struct JobRequest {
    pub job_id: String,
    pub done: tokio::sync::oneshot::Sender<Result<db::Job, String>>,
}

pub struct Worker {
    paths: Paths,
    conn: Connection,
    rx: Receiver<JobRequest>,
    event_tx: EventTx,
    cache: Option<(String, Loaded)>,
}

impl Worker {
    pub fn new(paths: Paths, conn: Connection, rx: Receiver<JobRequest>, event_tx: EventTx) -> Self {
        Worker {
            paths,
            conn,
            rx,
            event_tx,
            cache: None,
        }
    }

    pub fn run_forever(mut self) {
        Device::set_default(&Device::gpu());
        while let Ok(req) = self.rx.recv() {
            let result = self.handle(&req.job_id);
            let _ = req.done.send(result);
        }
    }

    fn handle(&mut self, job_id: &str) -> Result<db::Job, String> {
        if let Err(e) = db::set_status(&self.conn, job_id, "running") {
            return Err(format!("{e:#}"));
        }
        // 广播 running 状态
        if let Ok(Some(job)) = db::get_job(&self.conn, job_id) {
            let _ = self.event_tx.send(JobEvent::Updated { job });
        }

        match self.process(job_id) {
            Ok(outputs) => {
                if let Err(e) = db::set_done(&self.conn, job_id, &outputs) {
                    return Err(format!("{e:#}"));
                }
                match db::get_job(&self.conn, job_id) {
                    Ok(Some(job)) => {
                        let _ = self.event_tx.send(JobEvent::Updated { job: job.clone() });
                        Ok(job)
                    }
                    _ => Err("job not found after completion".into()),
                }
            }
            Err(e) => {
                let msg = format!("{e:#}");
                tracing::error!("job {job_id} failed: {msg}");
                let _ = db::set_error(&self.conn, job_id, &msg);
                if let Ok(Some(job)) = db::get_job(&self.conn, job_id) {
                    let _ = self.event_tx.send(JobEvent::Updated { job });
                }
                Err(msg)
            }
        }
    }

    fn process(&mut self, job_id: &str) -> Result<Vec<OutputFile>> {
        let job = db::get_job(&self.conn, job_id)?
            .ok_or_else(|| anyhow::anyhow!("job {job_id} not found"))?;

        let app = apps::find_app(&job.app)
            .ok_or_else(|| anyhow::anyhow!("unknown app: {}", job.app))?;
        let model = app
            .find_model(&job.model)
            .ok_or_else(|| anyhow::anyhow!("unknown model: {}", job.model))?;

        let model_dir = self.paths.model_dir(&model.dir);
        let ckpt = model_dir.join(&model.checkpoint);
        let input_path = Path::new(&job.input_path);
        let work_dir = Path::new(&job.work_dir);
        let params = &job.params;
        let stem = Path::new(&job.input_filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output")
            .to_string();

        Device::set_default(&Device::gpu());

        self.ensure_model(&model_dir, &ckpt, &model.variant, &app.id)?;
        let loaded = &mut self.cache.as_mut().unwrap().1;

        match (app.id.as_str(), model.variant.as_str()) {
            ("source-separation", "full") => {
                let chunk = p_i64(params, "chunk_size", 8) as i32;
                let overlap = p_i64(params, "overlap", 2) as i32;
                let Loaded::MbrFull(m) = loaded else {
                    bail!("model cache type mismatch for source-separation full");
                };
                source_separation::run_full(m, input_path, work_dir, chunk, overlap, &stem)
            }
            ("source-separation", "wsa") => {
                let chunk = p_i64(params, "chunk_size", 8) as i32;
                let batch = p_i64(params, "batch_size", 4) as i32;
                let Loaded::MbrWsa(m) = loaded else {
                    bail!("model cache type mismatch for source-separation wsa");
                };
                source_separation::run_wsa(m, input_path, work_dir, chunk, batch, &stem)
            }
            ("vocal-enhance", _) => {
                let device = p_str(params, "device", "gpu");
                if device == "cpu" {
                    Device::set_default(&Device::cpu());
                } else {
                    Device::set_default(&Device::gpu());
                }
                let Loaded::Srs(m) = loaded else {
                    bail!("model cache type mismatch for vocal-enhance");
                };
                vocal_enhance::run(m, input_path, work_dir, &stem)
            }
            ("vocal2midi", _) => {
                let lang = p_str(params, "language", "none");
                let vp = Vocal2MidiParams {
                    t0: p_f64(params, "t0", 0.0) as f32,
                    nsteps: p_i64(params, "nsteps", 8) as i32,
                    seg_threshold: p_f64(params, "seg_threshold", 0.2) as f32,
                    seg_radius: p_f64(params, "seg_radius", 0.02) as f32,
                    est_threshold: p_f64(params, "est_threshold", 0.2) as f32,
                    tempo: p_f64(params, "tempo", 120.0) as f32,
                    language: if lang == "none" { None } else { Some(lang) },
                };
                let Loaded::Game(t) = loaded else {
                    bail!("model cache type mismatch for vocal2midi");
                };
                vocal2midi::run(t, input_path, work_dir, &vp, &stem)
            }
            other => bail!("unsupported app/variant: {other:?}"),
        }
    }

    fn ensure_model(
        &mut self,
        model_dir: &Path,
        ckpt: &Path,
        variant: &str,
        app_id: &str,
    ) -> Result<()> {
        let key = ckpt.to_string_lossy().to_string();
        if self.cache.as_ref().map(|(k, _)| *k == key).unwrap_or(false) {
            return Ok(());
        }
        anyhow::ensure!(ckpt.exists(), "checkpoint not found: {}", ckpt.display());
        self.cache = None;

        let loaded = match (app_id, variant) {
            ("source-separation", "full") => {
                let mut m = MelBandRoformer::new();
                m.load(ckpt).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Loaded::MbrFull(m)
            }
            ("source-separation", "wsa") => {
                let mut m = WsaMbRoformer::new();
                m.load(ckpt).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Loaded::MbrWsa(m)
            }
            ("vocal-enhance", _) => {
                let m = Renaissance::load(ckpt).map_err(|e| anyhow::anyhow!("{e:?}"))?;
                Loaded::Srs(m)
            }
            ("vocal2midi", _) => {
                let cfg = model_dir.join("config.yaml");
                anyhow::ensure!(cfg.exists(), "GAME config not found: {}", cfg.display());
                let mut t = GameVocalTranscriber::new(ckpt, &cfg);
                t.load().context("load GAME model")?;
                Loaded::Game(t)
            }
            other => bail!("unsupported app/variant for load: {other:?}"),
        };
        self.cache = Some((key, loaded));
        Ok(())
    }
}

fn p_i64(v: &Value, key: &str, default: i64) -> i64 {
    v.get(key).and_then(|x| x.as_i64()).unwrap_or(default)
}
fn p_f64(v: &Value, key: &str, default: f64) -> f64 {
    v.get(key).and_then(|x| x.as_f64()).unwrap_or(default)
}
fn p_str(v: &Value, key: &str, default: &str) -> String {
    v.get(key)
        .and_then(|x| x.as_str())
        .unwrap_or(default)
        .to_string()
}
