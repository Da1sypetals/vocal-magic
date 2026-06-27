use anyhow::{anyhow, Result};
use std::fs;
use std::path::{Path, PathBuf};

// 把构建产物里的 mlx.metallib 复制到可执行文件同目录，
// MLX Metal 后端运行时需要它（与 mb-roformer 示例的逻辑一致）。
pub fn ensure_mlx_metallib_colocated() -> Result<()> {
    let exe_path = std::env::current_exe()?;
    let exe_dir = exe_path
        .parent()
        .ok_or_else(|| anyhow!("current executable path has no parent: {exe_path:?}"))?;
    let output_path = exe_dir.join("mlx.metallib");
    if output_path.exists() {
        return Ok(());
    }
    let source_path = find_built_mlx_metallib()?;
    fs::copy(&source_path, &output_path)?;
    Ok(())
}

fn find_built_mlx_metallib() -> Result<PathBuf> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let profile = if cfg!(debug_assertions) { "debug" } else { "release" };
    let build_dir = manifest_dir.join("target").join(profile).join("build");
    let mut candidates = Vec::new();
    collect_mlx_metallibs(&build_dir, &mut candidates)?;
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("could not find mlx.metallib under {}", build_dir.display()))
}

fn collect_mlx_metallibs(dir: &Path, candidates: &mut Vec<PathBuf>) -> Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if path.is_dir() {
            collect_mlx_metallibs(&path, candidates)?;
        } else if path.file_name().and_then(|n| n.to_str()) == Some("mlx.metallib") {
            candidates.push(path);
        }
    }
    Ok(())
}
