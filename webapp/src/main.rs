mod apps;
mod db;
mod events;
mod metallib;
mod models;
mod storage;
mod worker;

use anyhow::Result;
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Path as AxumPath, State},
    http::{header, StatusCode},
    response::{Html, IntoResponse, Response},
    routing::get,
    Json, Router,
};
use clap::Parser;
use futures::stream::Stream;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::broadcast;

use crate::db::Job;
use crate::events::{EventTx, JobEvent};
use crate::storage::Paths;
use crate::worker::JobRequest;

#[derive(Parser, Debug)]
#[command(
    name = "vocal-magic-webapp",
    about = "Voice Magic on-device audio backend"
)]
struct Cli {
    /// 产物根目录
    #[arg(long)]
    output_dir: Option<PathBuf>,

    /// checkpoint 根目录（必填）
    #[arg(long)]
    checkpoints_dir: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    host: String,

    #[arg(long, default_value_t = 8736)]
    port: u16,
}

#[derive(Clone)]
struct AppState {
    conn: Arc<Mutex<rusqlite::Connection>>,
    worker_tx: Arc<Mutex<std::sync::mpsc::Sender<JobRequest>>>,
    event_tx: EventTx,
    paths: Paths,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    let cli = Cli::parse();
    let output_dir = cli.output_dir.unwrap_or_else(|| {
        dirs::home_dir()
            .expect("cannot resolve home directory")
            .join(".vocal-magic")
    });

    let paths = Paths::new(output_dir, cli.checkpoints_dir);
    paths.init()?;
    tracing::info!("output dir: {}", paths.output_dir.display());
    tracing::info!("checkpoints: {}", paths.checkpoints_dir.display());

    metallib::ensure_mlx_metallib_colocated()?;

    let api_conn = db::open(&paths.db_path)?;
    db::fail_orphans(&api_conn)?;

    let (event_tx, _) = events::channel();

    let worker_conn = db::open(&paths.db_path)?;
    let (wtx, wrx) = std::sync::mpsc::channel::<JobRequest>();
    {
        let worker_paths = paths.clone();
        let worker_event_tx = event_tx.clone();
        std::thread::Builder::new()
            .name("inference".into())
            .spawn(move || {
                worker::Worker::new(worker_paths, worker_conn, wrx, worker_event_tx).run_forever()
            })?;
    }

    let state = AppState {
        conn: Arc::new(Mutex::new(api_conn)),
        worker_tx: Arc::new(Mutex::new(wtx)),
        event_tx,
        paths: paths.clone(),
    };

    let app = Router::new()
        .route("/api/apps", get(get_apps))
        .route("/api/jobs", get(list_jobs).post(create_job))
        .route("/api/jobs/:id", get(get_job).post(run_job).delete(delete_job_handler))
        .route("/api/jobs/:id/files/:name", get(get_file))
        .route("/api/events", get(sse_events))
        .route("/logo.png", get(logo_png))
        .route("/favicon.ico", get(favicon))
        .fallback(get(index_html))
        .layer(DefaultBodyLimit::disable())
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let addr = format!("{}:{}", cli.host, cli.port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;
    tracing::info!("listening on http://{addr}");
    axum::serve(listener, app).await?;
    Ok(())
}

const INDEX_HTML: &str = include_str!("../static/index.html");
const LOGO_PNG: &[u8] = include_bytes!("../static/logo.png");

async fn index_html() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn logo_png() -> Response {
    ([(header::CONTENT_TYPE, "image/png")], Body::from(LOGO_PNG)).into_response()
}

async fn favicon() -> Response {
    ([(header::CONTENT_TYPE, "image/png")], Body::from(LOGO_PNG)).into_response()
}

struct ApiError(anyhow::Error);
impl<E: Into<anyhow::Error>> From<E> for ApiError {
    fn from(e: E) -> Self {
        ApiError(e.into())
    }
}
impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let msg = format!("{:#}", self.0);
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": msg })),
        )
            .into_response()
    }
}

async fn get_apps(State(st): State<AppState>) -> Json<Value> {
    Json(apps::apps_json(&st.paths.checkpoints_dir))
}

async fn list_jobs(State(st): State<AppState>) -> Result<Json<Vec<Job>>, ApiError> {
    let conn = st.conn.lock().unwrap();
    let jobs = db::list_jobs(&conn)?;
    Ok(Json(jobs))
}

async fn get_job(
    State(st): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Job>, ApiError> {
    let conn = st.conn.lock().unwrap();
    let job = db::get_job(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("job not found"))?;
    Ok(Json(job))
}

// POST /api/jobs: 同步阻塞直到推理完成，返回最终 job 状态
async fn create_job(State(st): State<AppState>, mut mp: Multipart) -> Result<Json<Job>, ApiError> {
    let mut app_id = String::new();
    let mut model_id = String::new();
    let mut params: Value = json!({});
    let mut filename = String::from("input");
    let mut bytes: Vec<u8> = Vec::new();

    while let Some(field) = mp.next_field().await? {
        match field.name().unwrap_or("") {
            "app" => app_id = field.text().await?,
            "model" => model_id = field.text().await?,
            "params" => {
                let txt = field.text().await?;
                params = serde_json::from_str(&txt).unwrap_or(json!({}));
            }
            "file" => {
                if let Some(fname) = field.file_name() {
                    filename = fname.to_string();
                }
                bytes = field.bytes().await?.to_vec();
            }
            _ => {}
        }
    }

    let app = apps::find_app(&app_id).ok_or_else(|| anyhow::anyhow!("unknown app: {app_id}"))?;
    let model = app
        .find_model(&model_id)
        .ok_or_else(|| anyhow::anyhow!("unknown model: {model_id}"))?;
    if bytes.is_empty() {
        return Err(anyhow::anyhow!("no input file uploaded").into());
    }

    let id = uuid::Uuid::new_v4().to_string();
    let work_dir = st.paths.work_dir(&app.id, &id);
    let input_path = storage::create_work_dir_with_input(&work_dir, &filename, &bytes)?;
    let input_path = storage::maybe_decode_ncm(&input_path)?;

    let job = Job {
        id: id.clone(),
        app: app.id.clone(),
        model: model.id.clone(),
        status: "queued".into(),
        created_at: db::now_ms(),
        updated_at: db::now_ms(),
        params,
        input_filename: filename,
        input_path: input_path.to_string_lossy().to_string(),
        work_dir: work_dir.to_string_lossy().to_string(),
        outputs: vec![],
        error: None,
    };

    {
        let conn = st.conn.lock().unwrap();
        db::insert_job(&conn, &job)?;
    }

    // 广播 created 事件
    let _ = st.event_tx.send(JobEvent::Created { job: job.clone() });

    // 发送到 worker 并同步等待完成
    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    st.worker_tx
        .lock()
        .unwrap()
        .send(JobRequest {
            job_id: id.clone(),
            done: done_tx,
        })
        .map_err(|_| anyhow::anyhow!("worker channel closed"))?;

    // 阻塞等待推理完成
    let result = done_rx
        .await
        .map_err(|_| anyhow::anyhow!("worker dropped"))?;
    match result {
        Ok(final_job) => Ok(Json(final_job)),
        Err(msg) => {
            // 即使出错也返回完整 job（含 error 字段），前端可以渲染
            let conn = st.conn.lock().unwrap();
            let job = db::get_job(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("job lost"))?;
            Ok(Json(job))
        }
    }
}

// POST /api/jobs/:id — 跑这个 job（可选 body 里带 params 覆盖）
async fn run_job(
    State(st): State<AppState>,
    AxumPath(id): AxumPath<String>,
    body: axum::body::Bytes,
) -> Result<Json<Job>, ApiError> {
    {
        let conn = st.conn.lock().unwrap();
        let mut job = db::get_job(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("job not found"))?;
        // 如果 body 里有 params，更新 job 的参数
        if !body.is_empty() {
            if let Ok(v) = serde_json::from_slice::<Value>(&body) {
                if let Some(p) = v.get("params") {
                    job.params = p.clone();
                    db::update_params(&conn, &id, &job.params)?;
                }
            }
        }
        let work_dir = std::path::Path::new(&job.work_dir);
        if work_dir.exists() {
            for entry in std::fs::read_dir(work_dir)? {
                let path = entry?.path();
                let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                if !name.starts_with("input_") {
                    let _ = std::fs::remove_file(&path);
                }
            }
        }
        db::set_status(&conn, &id, "queued")?;
    }
    {
        let conn = st.conn.lock().unwrap();
        if let Ok(Some(job)) = db::get_job(&conn, &id) {
            let _ = st.event_tx.send(JobEvent::Updated { job });
        }
    }

    let (done_tx, done_rx) = tokio::sync::oneshot::channel();
    st.worker_tx
        .lock()
        .unwrap()
        .send(JobRequest { job_id: id.clone(), done: done_tx })
        .map_err(|_| anyhow::anyhow!("worker channel closed"))?;

    let result = done_rx.await.map_err(|_| anyhow::anyhow!("worker dropped"))?;
    match result {
        Ok(job) => Ok(Json(job)),
        Err(_) => {
            let conn = st.conn.lock().unwrap();
            let job = db::get_job(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("job lost"))?;
            Ok(Json(job))
        }
    }
}

async fn delete_job_handler(
    State(st): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<Value>, ApiError> {
    let work_dir = {
        let conn = st.conn.lock().unwrap();
        db::delete_job(&conn, &id)?
    };
    if let Some(wd) = work_dir {
        let _ = std::fs::remove_dir_all(&wd);
    }
    let _ = st.event_tx.send(JobEvent::Deleted { id: id.clone() });
    Ok(Json(json!({ "deleted": id })))
}

async fn get_file(
    State(st): State<AppState>,
    AxumPath((id, name)): AxumPath<(String, String)>,
) -> Result<Response, ApiError> {
    if name.contains('/') || name.contains('\\') || name == ".." {
        return Err(anyhow::anyhow!("invalid file name").into());
    }
    let job = {
        let conn = st.conn.lock().unwrap();
        db::get_job(&conn, &id)?.ok_or_else(|| anyhow::anyhow!("job not found"))?
    };
    let path = std::path::Path::new(&job.work_dir).join(&name);
    let bytes = std::fs::read(&path)?;
    let ct = match path.extension().and_then(|e| e.to_str()) {
        Some("wav") => "audio/wav",
        Some("mid") => "audio/midi",
        Some("json") => "application/json",
        _ => "application/octet-stream",
    };
    Ok(([(header::CONTENT_TYPE, ct)], Body::from(bytes)).into_response())
}

// GET /api/events: SSE 端点，广播所有 job 生命周期事件
async fn sse_events(State(st): State<AppState>) -> Response {
    let rx = st.event_tx.subscribe();
    let stream = make_event_stream(rx);
    let body = Body::from_stream(stream);
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("X-Accel-Buffering", "no")
        .body(body)
        .unwrap()
}

fn make_event_stream(
    mut rx: broadcast::Receiver<JobEvent>,
) -> impl Stream<Item = Result<String, Infallible>> {
    async_stream::stream! {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    let data = serde_json::to_string(&event).unwrap();
                    yield Ok(format!("data: {data}\n\n"));
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!("SSE client lagged by {n} events");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => break,
            }
        }
    }
}
