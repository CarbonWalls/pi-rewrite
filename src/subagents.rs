//! Native peer subagent control: every model is an agent. There is no
//! orchestrator — agents discover each other through a shared registry and ping
//! each other through per-agent inboxes on disk (compatible with the
//! `pi-collaborating-agents` layout under <config>/collaborating-agents/).
//!
//!   registry/<Name>.json          -> running agent record
//!   inbox/<Name>/<msgid>.json      -> one message per file

use crate::multikey::now_ms;
use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRecord {
    pub name: String,
    #[serde(default)]
    pub pid: Option<u32>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub session_file: Option<String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub started_at: Option<String>,
    #[serde(default)]
    pub last_seen_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncomingMessage {
    pub id: String,
    pub from: String,
    pub to: String,
    pub message: String,
    pub ts: String,
}

pub struct AgentRegistry {
    dir: PathBuf,
}

impl AgentRegistry {
    pub fn new(config_dir: &Path) -> Self {
        Self { dir: config_dir.join("collaborating-agents") }
    }

    fn registry_dir(&self) -> PathBuf {
        self.dir.join("registry")
    }
    fn inbox_dir(&self) -> PathBuf {
        self.dir.join("inbox")
    }

    fn ensure_dirs(&self) -> Result<()> {
        std::fs::create_dir_all(self.registry_dir())?;
        std::fs::create_dir_all(self.inbox_dir())?;
        Ok(())
    }

    /// Register (or heartbeat) the current agent.
    pub fn register(&self, name: &str, pid: u32, session_file: Option<&Path>, cwd: &Path, model: &str) -> Result<()> {
        self.ensure_dirs()?;
        let now = Utc::now().to_rfc3339();
        let rec = AgentRecord {
            name: name.to_string(),
            pid: Some(pid),
            session_id: None,
            session_file: session_file.map(|p| p.to_string_lossy().to_string()),
            cwd: Some(cwd.to_string_lossy().to_string()),
            model: Some(model.to_string()),
            started_at: Some(now.clone()),
            last_seen_at: Some(now),
        };
        let path = self.registry_dir().join(format!("{name}.json"));
        std::fs::write(&path, serde_json::to_string_pretty(&rec)? + "\n")?;
        Ok(())
    }

    #[allow(dead_code)]
    pub fn unregister(&self, name: &str) -> Result<()> {
        let path = self.registry_dir().join(format!("{name}.json"));
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        Ok(())
    }

    /// List peers as (name, model).
    pub fn list(&self) -> Result<Vec<(String, String)>> {
        let reg = self.registry_dir();
        if !reg.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&reg)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let raw = std::fs::read_to_string(entry.path())?;
            if let Ok(rec) = serde_json::from_str::<AgentRecord>(&raw) {
                let model = rec.model.unwrap_or_default();
                out.push((rec.name, model));
            }
        }
        out.sort();
        Ok(out)
    }

    /// Deliver a message into a peer's inbox.
    pub fn deliver(&self, from: &str, to: &str, message: &str) -> Result<()> {
        self.ensure_dirs()?;
        let inbox = self.inbox_dir().join(to);
        std::fs::create_dir_all(&inbox)?;
        let id = uuid::Uuid::new_v4().to_string();
        let msg = IncomingMessage {
            id,
            from: from.to_string(),
            to: to.to_string(),
            message: message.to_string(),
            ts: Utc::now().to_rfc3339(),
        };
        let path = inbox.join(format!("{}.json", msg.id));
        std::fs::write(&path, serde_json::to_string(&msg)?)?;
        Ok(())
    }

    /// Deliver a message to every other registered agent.
    pub fn broadcast(&self, from: &str, message: &str) -> Result<usize> {
        let peers = self.list()?;
        let mut n = 0;
        for (name, _) in peers {
            if name == from {
                continue;
            }
            if self.deliver(from, &name, message).is_ok() {
                n += 1;
            }
        }
        Ok(n)
    }

    /// Read and delete all messages in this agent's inbox.
    pub fn drain_inbox(&self, name: &str) -> Result<Vec<IncomingMessage>> {
        let inbox = self.inbox_dir().join(name);
        if !inbox.is_dir() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        for entry in std::fs::read_dir(&inbox)? {
            let entry = entry?;
            if entry.path().extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            match std::fs::read_to_string(entry.path())
                .ok()
                .and_then(|raw| serde_json::from_str::<IncomingMessage>(&raw).ok())
            {
                Some(msg) => out.push(msg),
                None => continue,
            }
            let _ = std::fs::remove_file(entry.path());
        }
        out.sort_by(|a, b| a.ts.cmp(&b.ts));
        Ok(out)
    }

    #[allow(dead_code)]
    pub fn config_dir(&self) -> Result<&Path> {
        self.dir
            .parent()
            .context("collaborating-agents has no parent config dir")
    }

    #[allow(dead_code)]
    pub fn now_banner(&self) -> u64 {
        now_ms()
    }
}
