use serde::Serialize;
use serde_json::{json, Value};
use std::path::Path;

// 单个可调参数的描述，前端据此渲染控件
#[derive(Clone, Serialize)]
pub struct ParamDef {
    pub key: String,
    pub label: String,
    pub kind: String, // "int" | "float" | "select"
    pub default: Value,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
    pub unit: Option<String>,
    pub options: Option<Vec<String>>,
}

impl ParamDef {
    fn int(key: &str, label: &str, default: i64, min: f64, max: f64, unit: &str) -> Self {
        ParamDef {
            key: key.into(),
            label: label.into(),
            kind: "int".into(),
            default: json!(default),
            min: Some(min),
            max: Some(max),
            step: Some(1.0),
            unit: Some(unit.into()),
            options: None,
        }
    }
    fn float(key: &str, label: &str, default: f64, min: f64, max: f64, step: f64) -> Self {
        ParamDef {
            key: key.into(),
            label: label.into(),
            kind: "float".into(),
            default: json!(default),
            min: Some(min),
            max: Some(max),
            step: Some(step),
            unit: None,
            options: None,
        }
    }
    fn select(key: &str, label: &str, default: &str, options: &[&str]) -> Self {
        ParamDef {
            key: key.into(),
            label: label.into(),
            kind: "select".into(),
            default: json!(default),
            min: None,
            max: None,
            step: None,
            unit: None,
            options: Some(options.iter().map(|s| s.to_string()).collect()),
        }
    }
}

// 一个算法下的具体模型（模型 - checkpoint），带自己的参数列表
// 每个模型的资源单独放在 <checkpoints-root>/<dir>/ 下：权重文件 + （GAME 额外的 config.yaml/lang_map.json）
#[derive(Clone, Serialize)]
pub struct ModelDef {
    pub id: String,
    pub label: String,
    pub dir: String,        // checkpoints 根目录下的子目录
    pub checkpoint: String, // 子目录内的权重文件名
    pub variant: String,
    pub params: Vec<ParamDef>,
}

// 一个算法（左栏的一项）
#[derive(Clone, Serialize)]
pub struct AppDef {
    pub id: String,
    pub label: String,
    pub icon: String,
    pub description: String,
    pub models: Vec<ModelDef>,
}

pub fn all_apps() -> Vec<AppDef> {
    vec![
        AppDef {
            id: "vocal-enhance".into(),
            label: "Vocal enhance".into(),
            icon: "sparkle".into(),
            description: "Restore and enhance a vocal recording with the Smule Renaissance model. 48 kHz mono output.".into(),
            models: vec![ModelDef {
                id: "renaissance".into(),
                label: "renaissance".into(),
                dir: "renaissance".into(),
                checkpoint: "smule-renaissance-small.safetensors".into(),
                variant: "default".into(),
                params: vec![
                    ParamDef::select("device", "Device", "gpu", &["gpu", "cpu"]),
                    ParamDef::select("mode", "Mode", "batch", &["batch", "streaming"]),
                    ParamDef::int("chunk_frames", "Chunk frames", 8, 1.0, 64.0, "frames"),
                    ParamDef::int("left_context", "Left context", 129, 1.0, 256.0, "frames"),
                    ParamDef::int("right_context", "Right context", 129, 0.0, 256.0, "frames"),
                ],
            }],
        },
        AppDef {
            id: "vocal2midi".into(),
            label: "Vocal to MIDI".into(),
            icon: "piano-keys".into(),
            description: "Transcribe singing voice into continuous-pitch note events with GAME. Exports MIDI and JSON.".into(),
            models: vec![ModelDef {
                id: "GAME-1.0-large".into(),
                label: "GAME-1.0-large".into(),
                dir: "GAME-1.0-large".into(),
                checkpoint: "GAME-1.0-large.safetensors".into(),
                variant: "default".into(),
                params: vec![
                    ParamDef::float("t0", "D3PM start (t0)", 0.0, 0.0, 1.0, 0.05),
                    ParamDef::int("nsteps", "Denoising steps", 8, 1.0, 64.0, "steps"),
                    ParamDef::float("seg_threshold", "Boundary threshold", 0.2, 0.0, 1.0, 0.05),
                    ParamDef::float("seg_radius", "Boundary radius", 0.02, 0.0, 0.5, 0.01),
                    ParamDef::float("est_threshold", "Presence threshold", 0.2, 0.0, 1.0, 0.05),
                    ParamDef::float("tempo", "MIDI tempo", 120.0, 20.0, 300.0, 1.0),
                    ParamDef::select("language", "Language", "none", &["none", "en", "ja", "yue", "zh"]),
                ],
            }],
        },
        AppDef {
            id: "source-separation".into(),
            label: "Source separation".into(),
            icon: "arrows-split".into(),
            description: "Split a stereo mix into vocal and instrumental stems with Mel-Band Roformer. 44.1 kHz stereo WAV.".into(),
            models: vec![
                ModelDef {
                    id: "melband-roformer".into(),
                    label: "melband-roformer".into(),
                    dir: "melband-roformer".into(),
                    checkpoint: "melband-roformer.mlx.safetensors".into(),
                    variant: "full".into(),
                    params: vec![
                        ParamDef::int("chunk_size", "Chunk size", 8, 1.0, 30.0, "sec"),
                        ParamDef::int("overlap", "Overlap", 2, 1.0, 8.0, "windows"),
                    ],
                },
                ModelDef {
                    id: "mbr-win10-sink8".into(),
                    label: "mbr-win10-sink8".into(),
                    dir: "mbr-win10-sink8".into(),
                    checkpoint: "mbr-win10-sink8.mlx.safetensors".into(),
                    variant: "wsa".into(),
                    params: vec![
                        ParamDef::int("chunk_size", "Chunk size", 8, 1.0, 30.0, "sec"),
                        ParamDef::int("batch_size", "Batch size", 4, 1.0, 16.0, "chunks"),
                    ],
                },
            ],
        },
    ]
}

pub fn find_app(id: &str) -> Option<AppDef> {
    all_apps().into_iter().find(|a| a.id == id)
}

impl AppDef {
    pub fn find_model(&self, model_id: &str) -> Option<ModelDef> {
        self.models.iter().find(|m| m.id == model_id).cloned()
    }
}

// 序列化为前端用的 JSON，并注入每个模型的 available（资源是否就绪）
pub fn apps_json(checkpoints_root: &Path) -> Value {
    let apps: Vec<Value> = all_apps()
        .into_iter()
        .map(|app| {
            let models: Vec<Value> = app
                .models
                .iter()
                .map(|m| {
                    let mdir = checkpoints_root.join(&m.dir);
                    let ckpt_ok = mdir.join(&m.checkpoint).exists();
                    let config_ok =
                        app.id != "vocal2midi" || mdir.join("config.yaml").exists();
                    let available = ckpt_ok && config_ok;
                    let mut reason = String::new();
                    if !ckpt_ok {
                        reason = format!("missing checkpoint {}/{}", m.dir, m.checkpoint);
                    } else if !config_ok {
                        reason = format!("missing {}/config.yaml", m.dir);
                    }
                    let mut v = serde_json::to_value(m).unwrap();
                    v["available"] = json!(available);
                    v["unavailable_reason"] = json!(reason);
                    v
                })
                .collect();
            json!({
                "id": app.id,
                "label": app.label,
                "icon": app.icon,
                "description": app.description,
                "models": models,
            })
        })
        .collect();
    json!({ "apps": apps })
}
