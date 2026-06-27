use serde::Serialize;
use tokio::sync::broadcast;

use crate::db::Job;

#[derive(Clone, Serialize)]
#[serde(tag = "type")]
pub enum JobEvent {
    #[serde(rename = "created")]
    Created { job: Job },
    #[serde(rename = "updated")]
    Updated { job: Job },
    #[serde(rename = "deleted")]
    Deleted { id: String },
}

pub type EventTx = broadcast::Sender<JobEvent>;

pub fn channel() -> (EventTx, broadcast::Receiver<JobEvent>) {
    broadcast::channel(256)
}
