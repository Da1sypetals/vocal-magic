use anyhow::{Context, Result};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Serialize, Deserialize, Clone)]
pub struct OutputFile {
    pub name: String,
    pub kind: String,
    pub label: String,
}

#[derive(Serialize, Clone)]
pub struct Job {
    pub id: String,
    pub app: String,
    pub model: String,
    pub status: String, // queued | running | done | error
    pub created_at: i64,
    pub updated_at: i64,
    pub params: Value,
    pub input_filename: String,
    pub input_path: String,
    pub work_dir: String,
    pub outputs: Vec<OutputFile>,
    pub error: Option<String>,
}

pub fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64
}

pub fn open(path: &Path) -> Result<Connection> {
    let conn = Connection::open(path).with_context(|| format!("open sqlite {}", path.display()))?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            app TEXT NOT NULL,
            model TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            params TEXT NOT NULL,
            input_filename TEXT NOT NULL,
            input_path TEXT NOT NULL,
            work_dir TEXT NOT NULL,
            outputs TEXT NOT NULL,
            error TEXT
        )",
        [],
    )?;
    Ok(conn)
}

pub fn insert_job(conn: &Connection, job: &Job) -> Result<()> {
    conn.execute(
        "INSERT INTO jobs (id, app, model, status, created_at, updated_at, params, input_filename, input_path, work_dir, outputs, error)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        rusqlite::params![
            job.id,
            job.app,
            job.model,
            job.status,
            job.created_at,
            job.updated_at,
            serde_json::to_string(&job.params)?,
            job.input_filename,
            job.input_path,
            job.work_dir,
            serde_json::to_string(&job.outputs)?,
            job.error,
        ],
    )?;
    Ok(())
}

pub fn set_status(conn: &Connection, id: &str, status: &str) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET status = ?2, updated_at = ?3 WHERE id = ?1",
        rusqlite::params![id, status, now_ms()],
    )?;
    Ok(())
}

pub fn set_done(conn: &Connection, id: &str, outputs: &[OutputFile]) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET status = 'done', outputs = ?2, updated_at = ?3 WHERE id = ?1",
        rusqlite::params![id, serde_json::to_string(outputs)?, now_ms()],
    )?;
    Ok(())
}

pub fn set_error(conn: &Connection, id: &str, error: &str) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET status = 'error', error = ?2, updated_at = ?3 WHERE id = ?1",
        rusqlite::params![id, error, now_ms()],
    )?;
    Ok(())
}

fn row_to_job(row: &rusqlite::Row) -> rusqlite::Result<Job> {
    let params_str: String = row.get("params")?;
    let outputs_str: String = row.get("outputs")?;
    Ok(Job {
        id: row.get("id")?,
        app: row.get("app")?,
        model: row.get("model")?,
        status: row.get("status")?,
        created_at: row.get("created_at")?,
        updated_at: row.get("updated_at")?,
        params: serde_json::from_str(&params_str).unwrap_or(Value::Null),
        input_filename: row.get("input_filename")?,
        input_path: row.get("input_path")?,
        work_dir: row.get("work_dir")?,
        outputs: serde_json::from_str(&outputs_str).unwrap_or_default(),
        error: row.get("error")?,
    })
}

pub fn get_job(conn: &Connection, id: &str) -> Result<Option<Job>> {
    let mut stmt = conn.prepare("SELECT * FROM jobs WHERE id = ?1")?;
    let mut rows = stmt.query_map([id], row_to_job)?;
    match rows.next() {
        Some(r) => Ok(Some(r?)),
        None => Ok(None),
    }
}

pub fn list_jobs(conn: &Connection) -> Result<Vec<Job>> {
    let mut stmt = conn.prepare("SELECT * FROM jobs ORDER BY created_at DESC")?;
    let rows = stmt.query_map([], row_to_job)?;
    let mut jobs = Vec::new();
    for r in rows {
        jobs.push(r?);
    }
    Ok(jobs)
}

pub fn delete_job(conn: &Connection, id: &str) -> Result<Option<String>> {
    let work_dir: Option<String> = conn
        .query_row("SELECT work_dir FROM jobs WHERE id = ?1", [id], |row| {
            row.get(0)
        })
        .ok();
    conn.execute("DELETE FROM jobs WHERE id = ?1", [id])?;
    Ok(work_dir)
}

// 启动时把残留的 running/queued job 标记为 error（进程重启，内存任务已丢失）
pub fn fail_orphans(conn: &Connection) -> Result<()> {
    conn.execute(
        "UPDATE jobs SET status = 'error', error = 'interrupted by server restart', updated_at = ?1
         WHERE status IN ('queued', 'running')",
        rusqlite::params![now_ms()],
    )?;
    Ok(())
}
