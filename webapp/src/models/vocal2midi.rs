// 人声转 MIDI：使用 game-mlxrs 的高层 GameVocalTranscriber。

use anyhow::Result;
use game_mlxrs::{midi, notes_to_raw, score_json, GameVocalTranscriber};
use std::path::Path;

use crate::db::OutputFile;

pub struct Vocal2MidiParams {
    pub t0: f32,
    pub nsteps: i32,
    pub seg_threshold: f32,
    pub seg_radius: f32,
    pub est_threshold: f32,
    pub tempo: f32,
    pub language: Option<String>,
}

pub fn run(
    transcriber: &mut GameVocalTranscriber,
    input_path: &Path,
    work_dir: &Path,
    params: &Vocal2MidiParams,
    stem: &str,
) -> Result<Vec<OutputFile>> {
    transcriber.t0 = params.t0;
    transcriber.nsteps = params.nsteps;
    transcriber.seg_threshold = params.seg_threshold;
    transcriber.seg_radius = params.seg_radius;
    transcriber.est_threshold = params.est_threshold;

    let (samples, sample_rate) = game_mlxrs::mel::load_audio(input_path)?;
    let lang = params.language.as_deref();
    let notes = transcriber.transcribe_pcm(&samples, sample_rate, lang)?;

    let (durations, presence, scores) = notes_to_raw(&notes);
    let mname = format!("{stem}.mid");
    let jname = format!("{stem}.json");
    midi::save_midi(&work_dir.join(&mname), &durations, &presence, &scores, params.tempo)?;
    score_json::save_json(&work_dir.join(&jname), &durations, &presence, &scores)?;

    Ok(vec![
        OutputFile { name: mname, kind: "midi".into(), label: "MIDI".into() },
        OutputFile { name: jname, kind: "json".into(), label: "Notes (JSON)".into() },
    ])
}
