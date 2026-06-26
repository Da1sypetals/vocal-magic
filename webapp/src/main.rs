//! A local audio workbench: a multi-app web frontend for on-device audio models.
//!
//! Layout mirrors a creative-tools console: the left rail lists apps, the middle
//! column takes the input audio + per-app parameters, and the right column shows
//! produced assets ("产物") with live status. Apps:
//!   - "音频超分修复"   — restores degraded vocals (srs-inference / MLX)
//!   - "歌声转 MIDI"    — transcribes singing to MIDI notes (game-mlxrs / MLX)
//!   - "人声分离"       — separates a mix into vocal/instrumental stems (MB-RoFormer / MLX)

mod jobs;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::Router;
use axum::extract::{DefaultBodyLimit, Multipart, Path as AxumPath, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use clap::Parser;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tower_http::cors::CorsLayer;

use jobs::{Engine, JobStatus, MidiParams, ModelPaths, SubmitReq};

#[derive(Debug, Parser)]
#[command(about = "Local audio workbench — on-device audio model web app")]
struct Args {
    /// Super-resolution checkpoint (.safetensors).
    #[arg(long, default_value = "../smule-renaissance-small.safetensors")]
    sr_checkpoint: PathBuf,

    /// GAME transcription weights (.safetensors).
    #[arg(long, default_value = "../models/game/GAME-1.0-large.safetensors")]
    game_weights: PathBuf,

    /// GAME transcription config (config.yaml); lang_map.json must sit beside it.
    #[arg(long, default_value = "../models/game/config.yaml")]
    game_config: PathBuf,

    /// MelBand RoFormer separation checkpoint (.safetensors).
    #[arg(
        long,
        default_value = "../checkpoints/melband-roformer-tensors/melband-roformer.mlx.safetensors"
    )]
    separation_melband_checkpoint: PathBuf,

    /// Windowed-sink MB-RoFormer separation checkpoint (.safetensors).
    #[arg(long, default_value = "../checkpoints/mbr-win10-sink8.mlx.safetensors")]
    separation_wsa_checkpoint: PathBuf,

    /// Working directory for uploaded inputs and produced outputs.
    #[arg(long, default_value = "./.studio")]
    work_dir: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8080)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    engine: Arc<Engine>,
}

struct ApiError(StatusCode, String);
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.0, axum::Json(json!({ "error": self.1 }))).into_response()
    }
}
impl<E: std::fmt::Display> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
    }
}

async fn index() -> Html<&'static str> {
    Html(include_str!("web/index.html"))
}

async fn health() -> &'static str {
    "ok"
}

/// Static catalog describing the apps (left rail), their models, and every
/// exposed parameter. The frontend renders parameter controls from this schema.
async fn apps() -> impl IntoResponse {
    axum::Json(json!({
        "apps": [
            {
                "id": "super-resolution",
                "name": "音频超分修复",
                "short": "人声还原 · 高频补全",
                "desc": "上传录音，还原受损人声、补全高频细节，输出 48kHz WAV。",
                "icon": "sparkle",
                "product": "audio",
                "enabled": true,
                "models": [ { "id": "vocal-restore-v1", "name": "smule-renaissance-small.safetensors" } ],
                "params": [
                    { "key": "format", "label": "输出格式", "type": "select", "default": "wav",
                      "options": [ {"v":"wav","t":"WAV · 48kHz"} ] }
                ]
            },
            {
                "id": "transcribe-midi",
                "name": "歌声转 MIDI",
                "short": "人声 → 音符 / MIDI",
                "desc": "将清唱人声转写为音符序列，导出 MIDI 与 JSON 乐谱。",
                "icon": "notes",
                "product": "midi",
                "enabled": true,
                "models": [ { "id": "game-1.0-large", "name": "GAME-1.0-large.safetensors" } ],
                "params": [
                    { "key": "quantize", "label": "量化到半音", "type": "toggle", "default": false },
                    { "key": "quantize_equal_weight", "label": "量化等权重（默认时长加权）",
                      "type": "toggle", "default": false },
                    { "key": "language", "label": "语言代码", "type": "text", "default": "",
                      "placeholder": "如 zh（留空自动）", "advanced": true },
                    { "key": "t0", "label": "D3PM 起始 T", "type": "number", "default": 0.0,
                      "min": 0.0, "max": 0.95, "step": 0.05, "advanced": true },
                    { "key": "nsteps", "label": "D3PM 步数", "type": "number", "default": 8,
                      "min": 1, "max": 64, "step": 1, "advanced": true },
                    { "key": "seg_threshold", "label": "分段阈值", "type": "number", "default": 0.2,
                      "min": 0.0, "max": 1.0, "step": 0.01, "advanced": true },
                    { "key": "seg_radius", "label": "分段半径 (秒)", "type": "number", "default": 0.02,
                      "min": 0.0, "max": 0.5, "step": 0.005, "advanced": true },
                    { "key": "est_threshold", "label": "音符估计阈值", "type": "number", "default": 0.2,
                      "min": 0.0, "max": 1.0, "step": 0.01, "advanced": true },
                    { "key": "tempo", "label": "MIDI 速度 (BPM)", "type": "number", "default": 120,
                      "min": 20, "max": 300, "step": 1, "advanced": true }
                ]
            },
            {
                "id": "separation", "name": "人声分离", "short": "人声 / 伴奏拆分",
                "desc": "将混音分离为人声与伴奏轨道。", "icon": "split",
                "product": "audio", "enabled": true,
                "models": [
                    { "id": "melband-roformer", "name": "melband-roformer.mlx.safetensors" },
                    { "id": "wsa-mb-roformer-win10-sink8", "name": "mbr-win10-sink8.mlx.safetensors" }
                ],
                "params": []
            }
        ]
    }))
}

#[derive(Serialize)]
struct JobView {
    id: String,
    name: String,
    app: String,
    model: String,
    model_name: String,
    product: String,
    status: String,
    created_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rtf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    artifacts: Vec<ArtifactView>,
}

#[derive(Serialize)]
struct ArtifactView {
    key: String,
    label: String,
    kind: String,
}

fn status_str(s: &JobStatus) -> &'static str {
    match s {
        JobStatus::Queued => "queued",
        JobStatus::Processing => "processing",
        JobStatus::Done => "done",
        JobStatus::Failed => "failed",
    }
}

fn view(j: &jobs::Job) -> JobView {
    JobView {
        id: j.id.clone(),
        name: j.name.clone(),
        app: j.app.clone(),
        model: j.model.clone(),
        model_name: j.model_name.clone(),
        product: j.product.to_string(),
        status: status_str(&j.status).to_string(),
        created_at: j.created_at,
        duration: j.duration,
        rtf: j.rtf,
        note_count: j.note_count,
        error: j.error.clone(),
        artifacts: j
            .artifacts
            .iter()
            .map(|artifact| ArtifactView {
                key: artifact.key.clone(),
                label: artifact.label.clone(),
                kind: artifact.kind.to_owned(),
            })
            .collect(),
    }
}

fn f32_field(p: &Value, key: &str, default: f32) -> f32 {
    p.get(key)
        .and_then(Value::as_f64)
        .map(|v| v as f32)
        .unwrap_or(default)
}

async fn create_job(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let mut data: Option<axum::body::Bytes> = None;
    let mut name = "audio.wav".to_string();
    let mut model = String::new();
    let mut model_name = String::new();
    let mut app = "super-resolution".to_string();
    let mut params: Value = Value::Null;

    while let Some(field) = multipart.next_field().await? {
        match field.name().unwrap_or_default() {
            "file" => {
                if let Some(fname) = field.file_name() {
                    name = fname.to_string();
                }
                data = Some(field.bytes().await?);
            }
            "app" => app = field.text().await?,
            "model" => model = field.text().await?,
            "model_name" => model_name = field.text().await?,
            "params" => {
                let txt = field.text().await?;
                params = serde_json::from_str(&txt).unwrap_or(Value::Null);
            }
            _ => {
                let _ = field.bytes().await;
            }
        }
    }

    let data = data.ok_or_else(|| ApiError(StatusCode::BAD_REQUEST, "未收到音频文件".into()))?;

    let midi = if app == "transcribe-midi" {
        let lang = params
            .get("language")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        Some(MidiParams {
            t0: f32_field(&params, "t0", 0.0),
            nsteps: params
                .get("nsteps")
                .and_then(Value::as_f64)
                .map(|v| v as i32)
                .unwrap_or(8),
            seg_threshold: f32_field(&params, "seg_threshold", 0.2),
            seg_radius: f32_field(&params, "seg_radius", 0.02),
            est_threshold: f32_field(&params, "est_threshold", 0.2),
            tempo: f32_field(&params, "tempo", 120.0),
            language: lang,
            quantize: params
                .get("quantize")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            quantize_equal_weight: params
                .get("quantize_equal_weight")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        })
    } else {
        None
    };

    let req = SubmitReq {
        name,
        app,
        model,
        model_name,
        midi,
    };
    let job = state.engine.submit(req, &data).await?;
    Ok(axum::Json(view(&job)).into_response())
}

#[derive(serde::Deserialize)]
struct ListQuery {
    page: Option<usize>,
    per_page: Option<usize>,
}

async fn list_jobs(State(state): State<AppState>, Query(q): Query<ListQuery>) -> impl IntoResponse {
    let page = q.page.unwrap_or(1).max(1);
    let per_page = q.per_page.unwrap_or(7).clamp(1, 100);
    let (items, total) = state.engine.list(page, per_page);
    let views: Vec<JobView> = items.iter().map(view).collect();
    axum::Json(json!({ "jobs": views, "total": total, "page": page, "per_page": per_page }))
}

async fn get_job(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Response, ApiError> {
    let j = state
        .engine
        .get(&id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "任务不存在".into()))?;
    Ok(axum::Json(view(&j)).into_response())
}

fn mime_for(path: &std::path::Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
        .as_str()
    {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        "m4a" | "aac" => "audio/mp4",
        "flac" => "audio/flac",
        "ogg" | "opus" => "audio/ogg",
        "mid" | "midi" => "audio/midi",
        "json" => "application/json",
        _ => "application/octet-stream",
    }
}

#[derive(serde::Deserialize)]
struct AudioQuery {
    which: Option<String>,
    download: Option<u8>,
}

async fn job_audio(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<AudioQuery>,
) -> Result<Response, ApiError> {
    let job = state
        .engine
        .get(&id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "任务不存在".into()))?;
    let which = q.which.as_deref().unwrap_or("output");
    let path = match which {
        "input" => Some(job.input_path.clone()),
        key => job
            .artifacts
            .iter()
            .find(|artifact| artifact.kind == "audio" && artifact.key == key)
            .map(|artifact| artifact.path.clone())
            .or_else(|| {
                if key == "output" {
                    job.output_path.clone()
                } else {
                    None
                }
            }),
    };
    let path = path.ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "音频尚未就绪".into()))?;
    serve_file(
        &path,
        q.download == Some(1),
        &download_base_name(&job.name, which),
    )
    .await
}

#[derive(serde::Deserialize)]
struct FileQuery {
    kind: Option<String>,
}

async fn job_file(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
    Query(q): Query<FileQuery>,
) -> Result<Response, ApiError> {
    let job = state
        .engine
        .get(&id)
        .ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "任务不存在".into()))?;
    let kind = q.kind.as_deref().unwrap_or("midi");
    let path = job
        .artifacts
        .iter()
        .find(|artifact| artifact.kind == "file" && artifact.key == kind)
        .map(|artifact| artifact.path.clone())
        .or_else(|| match kind {
            "json" => job.score_path.clone(),
            "notes" => job.notes_path.clone(),
            _ => job.output_path.clone(),
        });
    let path = path.ok_or_else(|| ApiError(StatusCode::NOT_FOUND, "产物尚未就绪".into()))?;
    serve_file(&path, true, &download_base_name(&job.name, kind)).await
}

fn download_base_name(base_name: &str, key: &str) -> String {
    let stem = std::path::Path::new(base_name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    match key {
        "input" | "output" | "midi" | "json" => stem.to_owned(),
        suffix => format!("{stem}_{suffix}"),
    }
}

async fn serve_file(
    path: &std::path::Path,
    download: bool,
    download_stem: &str,
) -> Result<Response, ApiError> {
    let bytes = tokio::fs::read(path).await?;
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, mime_for(path).parse().unwrap());
    if download {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("bin");
        headers.insert(
            header::CONTENT_DISPOSITION,
            format!("attachment; filename=\"{download_stem}.{ext}\"")
                .parse()
                .unwrap(),
        );
    }
    Ok((headers, bytes).into_response())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    tokio::fs::create_dir_all(&args.work_dir).await?;

    println!("Local Audio Workbench");
    println!("  sr checkpoint : {}", args.sr_checkpoint.display());
    println!("  game weights  : {}", args.game_weights.display());
    println!("  game config   : {}", args.game_config.display());
    println!(
        "  melband ckpt: {}",
        args.separation_melband_checkpoint.display()
    );
    println!(
        "  wsa ckpt     : {}",
        args.separation_wsa_checkpoint.display()
    );
    println!("  work dir      : {}", args.work_dir.display());
    println!("  listening     : http://{}:{}", args.host, args.port);

    let paths = ModelPaths {
        sr_checkpoint: args.sr_checkpoint,
        game_weights: args.game_weights,
        game_config: args.game_config,
        separation_melband_checkpoint: args.separation_melband_checkpoint,
        separation_wsa_checkpoint: args.separation_wsa_checkpoint,
    };
    let engine = Arc::new(Engine::new(paths, args.work_dir));
    let state = AppState { engine };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/health", get(health))
        .route("/api/apps", get(apps))
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/{id}", get(get_job))
        .route("/api/jobs/{id}/audio", get(job_audio))
        .route("/api/jobs/{id}/file", get(job_file))
        .layer(DefaultBodyLimit::max(1024 * 1024 * 1024))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let listener = TcpListener::bind((args.host.as_str(), args.port)).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
