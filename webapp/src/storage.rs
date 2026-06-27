use anyhow::{Context, Result};
use ncmdump::Ncmdump;
use std::io::Cursor;
use std::path::{Path, PathBuf};

// 产物目录布局版本。改动不兼容的目录/数据库结构时递增此值，
// 旧目录会在启动时被直接清空（不背历史包袱）。
pub const LAYOUT_VERSION: &str = "1";

#[derive(Clone)]
pub struct Paths {
    pub output_dir: PathBuf,
    pub apps_dir: PathBuf,
    pub db_path: PathBuf,
    pub checkpoints_dir: PathBuf,
}

impl Paths {
    pub fn new(output_dir: PathBuf, checkpoints_dir: PathBuf) -> Self {
        let apps_dir = output_dir.join("apps");
        let db_path = apps_dir.join("jobs.sqlite");
        Paths {
            output_dir,
            apps_dir,
            db_path,
            checkpoints_dir,
        }
    }

    fn version_file(&self) -> PathBuf {
        self.apps_dir.join(".layout-version")
    }

    // 初始化产物目录：检测版本，不兼容则清空旧 apps 目录后重建
    pub fn init(&self) -> Result<()> {
        std::fs::create_dir_all(&self.output_dir)
            .with_context(|| format!("create output dir {}", self.output_dir.display()))?;

        if self.apps_dir.exists() {
            let stored = std::fs::read_to_string(self.version_file()).ok();
            let compatible = stored.as_deref().map(|s| s.trim()) == Some(LAYOUT_VERSION);
            if !compatible {
                tracing::warn!(
                    "incompatible layout version (stored={:?}, current={}); wiping {}",
                    stored,
                    LAYOUT_VERSION,
                    self.apps_dir.display()
                );
                std::fs::remove_dir_all(&self.apps_dir)
                    .with_context(|| format!("wipe apps dir {}", self.apps_dir.display()))?;
            }
        }

        std::fs::create_dir_all(&self.apps_dir)
            .with_context(|| format!("create apps dir {}", self.apps_dir.display()))?;
        std::fs::write(self.version_file(), LAYOUT_VERSION)
            .with_context(|| "write layout version file")?;
        Ok(())
    }

    pub fn work_dir(&self, app_id: &str, job_id: &str) -> PathBuf {
        self.apps_dir.join(app_id).join(job_id)
    }

    // 某个模型的资源目录：<checkpoints-root>/<dir>/
    pub fn model_dir(&self, dir: &str) -> PathBuf {
        self.checkpoints_dir.join(dir)
    }
}

// 把上传的输入复制到 job 工作目录，返回工作目录里的输入文件路径
pub fn create_work_dir_with_input(
    work_dir: &Path,
    input_filename: &str,
    bytes: &[u8],
) -> Result<PathBuf> {
    std::fs::create_dir_all(work_dir)
        .with_context(|| format!("create work dir {}", work_dir.display()))?;
    let safe = sanitize_filename(input_filename);
    let input_path = work_dir.join(format!("input_{safe}"));
    std::fs::write(&input_path, bytes)
        .with_context(|| format!("write input to {}", input_path.display()))?;
    Ok(input_path)
}

// 若输入是网易云 .ncm 加密格式，就地解密成普通音频（mp3/flac）并返回解密后文件路径；
// 否则原样返回。原始 .ncm 仍保留在工作目录里。
pub fn maybe_decode_ncm(input_path: &Path) -> Result<PathBuf> {
    let is_ncm = input_path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("ncm"))
        .unwrap_or(false);
    if !is_ncm {
        return Ok(input_path.to_path_buf());
    }
    let bytes = std::fs::read(input_path)
        .with_context(|| format!("read ncm {}", input_path.display()))?;
    let mut ncm =
        Ncmdump::from_reader(Cursor::new(bytes)).map_err(|e| anyhow::anyhow!("ncm open: {e:?}"))?;
    // 必须在 get_info 之前调 get_data，否则 get_info 会移动内部游标导致解密数据损坏
    let data = ncm.get_data().map_err(|e| anyhow::anyhow!("ncm decrypt: {e:?}"))?;
    let ext = detect_audio_ext(&data);
    let out = input_path.with_extension(format!("decoded.{ext}"));
    std::fs::write(&out, &data).with_context(|| format!("write decoded {}", out.display()))?;
    Ok(out)
}

fn detect_audio_ext(data: &[u8]) -> &'static str {
    if data.starts_with(b"fLaC") {
        "flac"
    } else if data.starts_with(b"ID3") || data.get(..2) == Some(&[0xFF, 0xFB]) {
        "mp3"
    } else if data.starts_with(b"OggS") {
        "ogg"
    } else {
        "mp3"
    }
}

fn sanitize_filename(name: &str) -> String {
    let base = Path::new(name)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("input");
    base.chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_') {
            c
        } else {
            '_'
        })
        .collect()
}
