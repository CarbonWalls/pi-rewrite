//! Append-only JSONL session persistence (pi-compatible line-per-entry shape).

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// One durable session entry. The message shape matches what the agent loop feeds
/// the LLM; `id`/`parentId` give a stable cursor and branch linkage like pi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_id: Option<String>,
    pub timestamp: String,
    #[serde(flatten)]
    pub message: serde_json::Value,
}

pub struct Session {
    pub file: PathBuf,
    pub id: String,
    leaf_id: Option<String>,
}

impl Session {
    /// Open (append to) or create a session file under `sessions_dir`.
    pub fn open(sessions_dir: &Path, cwd: &Path) -> std::io::Result<Self> {
        std::fs::create_dir_all(sessions_dir)?;
        let id = Uuid::new_v4().to_string();
        let host = sanitize(cwd);
        let ts = chrono::Utc::now().format("%Y-%m-%dT%H-%M-%S%.3fZ").to_string();
        let file = sessions_dir.join(host).join(format!("{ts}_{id}.jsonl"));
        Ok(Self { file, id, leaf_id: None })
    }

    pub fn append(&mut self, entry: &SessionEntry) -> std::io::Result<()> {
        let mut raw = serde_json::to_string(entry).unwrap_or_else(|_| "{}".into());
        raw.push('\n');
        if let Some(dir) = self.file.parent() {
            std::fs::create_dir_all(dir)?;
        }
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.file)?;
        f.write_all(raw.as_bytes())?;
        self.leaf_id = Some(entry.id.clone());
        Ok(())
    }

    pub fn leaf_id(&self) -> Option<&str> {
        self.leaf_id.as_deref()
    }

    pub fn entry_id() -> String {
        Uuid::new_v4().to_string()
    }
}

fn sanitize(p: &Path) -> String {
    let s = p.to_string_lossy().to_string();
    let s = s.trim_start_matches('/').to_string();
    if s.is_empty() {
        "--root--".to_string()
    } else {
        s.replace('/', "--")
    }
}
