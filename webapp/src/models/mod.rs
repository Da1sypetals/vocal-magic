pub mod source_separation;
pub mod vocal2midi;
pub mod vocal_enhance;

use game_mlxrs::GameVocalTranscriber;
use mb_roformer_mlx::{MelBandRoformer, WsaMbRoformer};
use srs_inference::Renaissance;

// 已加载到内存的模型（每个进程同一时刻缓存一个，切换时丢弃旧的）
pub enum Loaded {
    MbrFull(MelBandRoformer),
    MbrWsa(WsaMbRoformer),
    Srs(Renaissance),
    Game(GameVocalTranscriber),
}
