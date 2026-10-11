//! The agent loop: prompt -> stream -> handle tool calls -> repeat.
//! Native system prompt support and peer-subagent inbox draining are folded in.

use crate::config::{MultikeyConfig, PoolConfig, PoolModel, Settings};
use crate::llm::{ChatMessage, LlmClient, ToolCall, ToolDef, Usage};
use crate::multikey::KeyPool;
use crate::session::Session;
use crate::subagents::AgentRegistry;
use crate::tools::{ToolRegistry, ToolResult};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const DEFAULT_SYSTEM_PROMPT: &str = "You are pi-rewrite, a fast native coding agent. You work in the user's project directory.\nUse tools to inspect and modify the repository: read before you edit, run bash to build and test, write/edit to change files.\nBe direct. Prefer concrete behavior over summaries. When a task is done, say so briefly.\nYou can collaborate with other agents: list peers with agents_list, message one with agents_ping, or send to all with agents_broadcast.";

#[derive(Debug, Clone, Default)]
pub struct AgentOptions {
    /// Replace the default system prompt (text or path to a file).
    pub system_prompt: Option<String>,
    /// Appended after the base system prompt (text or path to a file).
    pub append_system_prompt: Option<String>,
    /// Overrides for provider/model selection.
    pub provider: Option<String>,
    pub model: Option<String>,
    /// Agent identity for peer collaboration.
    pub agent_name: Option<String>,
    pub max_turns: usize,
    pub quiet: bool,
}

/// Resolve a CLI/system-prompt argument that may be either literal text or a path.
fn resolve_prompt_arg(value: &str) -> String {
    let p = Path::new(value);
    if p.is_file() {
        std::fs::read_to_string(p).unwrap_or_else(|_| value.to_string())
    } else {
        value.to_string()
    }
}

pub fn build_system_prompt(settings: &Settings, opts: &AgentOptions) -> String {
    let mut parts: Vec<String> = Vec::new();

    if let Some(base) = &opts.system_prompt {
        parts.push(resolve_prompt_arg(base));
    } else if let Some(base) = settings.system_prompt.as_deref() {
        parts.push(resolve_prompt_arg(base));
    } else {
        parts.push(DEFAULT_SYSTEM_PROMPT.to_string());
    }

    if let Some(extra) = &opts.append_system_prompt {
        parts.push(resolve_prompt_arg(extra));
    } else if let Some(extra) = settings.append_system_prompt.as_deref() {
        parts.push(resolve_prompt_arg(extra));
    }

    parts.join("\n\n")
}

/// Resolve the (pool, model) pair from settings, CLI overrides, or the first pool.
pub fn resolve_provider<'a>(
    multikey: &'a MultikeyConfig,
    settings: &Settings,
    opts: &AgentOptions,
) -> Result<(&'a PoolConfig, PoolModel)> {
    // CLI overrides win.
    if let (Some(p), Some(m)) = (opts.provider.as_deref(), opts.model.as_deref()) {
        if let Some(pair) = multikey.resolve_model(p, m) {
            return Ok(pair);
        }
    }
    // settings.json defaults (defaultProvider = pool id, defaultModel = model id)
    if let (Some(p), Some(m)) = (settings.default_provider.as_deref(), settings.default_model.as_deref()) {
        if let Some(pair) = multikey.resolve_model(p, m) {
            return Ok(pair);
        }
    }
    // A single "pool/model" string in --model.
    if let Some(m) = opts.model.as_deref() {
        if m.contains('/') {
            if let Some((pool, mid)) = m.split_once('/') {
                // pool may itself be "pool/model"
                if let Some(pair) = multikey.resolve_model(pool, mid) {
                    return Ok(pair);
                }
                // try whole string as pool id with empty model
                if let Some(pair) = multikey.resolve_model(m, "") {
                    return Ok(pair);
                }
            }
        }
    }
    // First pool with any model.
    let pool = multikey
        .pools
        .first()
        .context("no multikey pools configured; add one in ~/.pi/agent/multikey.json")?;
    let model = pool
        .models
        .first()
        .context(format!("pool \"{}\" has no models", pool.id))?
        .clone();
    Ok((pool, model))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurnEvent {
    pub kind: String,
    pub text: Option<String>,
    pub tool_name: Option<String>,
    pub tool_args: Option<String>,
    pub tool_result: Option<String>,
}

#[derive(Clone)]
pub struct Agent {
    pub client: LlmClient,
    pub pool: Arc<KeyPool>,
    pub model: PoolModel,
    pub tools: ToolRegistry,
    pub registry: Option<Arc<AgentRegistry>>,
    pub opts: AgentOptions,
}

impl Agent {
    pub fn new(
        pool: PoolConfig,
        model: PoolModel,
        tools: ToolRegistry,
        registry: Option<Arc<AgentRegistry>>,
        opts: AgentOptions,
    ) -> Self {
        Self {
            client: LlmClient::new(),
            pool: Arc::new(KeyPool::new(pool)),
            model,
            tools,
            registry,
            opts,
        }
    }

    /// Run the agent loop over one user prompt. Returns the final assistant text.
    pub async fn run(&self, prompt: &str, system_prompt: &str, session: &mut Session) -> Result<String> {
        let max_turns = if self.opts.max_turns == 0 { 40 } else { self.opts.max_turns };

        // Drain this agent's peer inbox into the opening context.
        let mut messages: Vec<ChatMessage> = vec![ChatMessage::system(system_prompt)];

        if let Some(name) = &self.opts.agent_name {
            if let Some(reg) = &self.registry {
                let inbox = reg.drain_inbox(name).unwrap_or_default();
                for msg in inbox {
                    messages.push(ChatMessage::user(format!(
                        "[ping from {}] {}",
                        msg.from, msg.message
                    )));
                    let entry = crate::session::SessionEntry {
                        id: crate::session::Session::entry_id(),
                        parent_id: session.leaf_id().map(|s| s.to_string()),
                        timestamp: chrono::Utc::now().to_rfc3339(),
                        message: serde_json::json!({ "role": "user", "ping_from": msg.from, "content": msg.message }),
                    };
                    let _ = session.append(&entry);
                }
            }
        }

        messages.push(ChatMessage::user(prompt));
        let entry = crate::session::SessionEntry {
            id: crate::session::Session::entry_id(),
            parent_id: session.leaf_id().map(|s| s.to_string()),
            timestamp: chrono::Utc::now().to_rfc3339(),
            message: serde_json::json!({ "role": "user", "content": prompt }),
        };
        let _ = session.append(&entry);

        let defs: Vec<ToolDef> = self.tools.defs();

        let mut last_text = String::new();
        for turn in 0..max_turns {
            let output = self
                .client
                .stream_chat(self.pool.clone(), &self.model, &messages, &defs)
                .await?;

            // Record the assistant message.
            let assistant = ChatMessage::assistant(output.content.clone(), output.tool_calls.clone());
            messages.push(assistant);
            let entry = crate::session::SessionEntry {
                id: crate::session::Session::entry_id(),
                parent_id: session.leaf_id().map(|s| s.to_string()),
                timestamp: chrono::Utc::now().to_rfc3339(),
                message: serde_json::json!({
                    "role": "assistant",
                    "content": output.content,
                    "tool_calls": output.tool_calls.iter().map(|t| serde_json::json!({"id": t.id, "name": t.name, "arguments": t.arguments})).collect::<Vec<_>>(),
                    "stop_reason": output.stop_reason,
                }),
            };
            let _ = session.append(&entry);

            if let Some(text) = &output.content {
                last_text = text.clone();
                if !self.opts.quiet {
                    print!("{text}");
                    use std::io::Write;
                    let _ = std::io::stdout().flush();
                }
            }

            if output.tool_calls.is_empty() {
                if !self.opts.quiet {
                    println!();
                }
                return Ok(last_text);
            }

            // Execute each tool call and feed results back.
            for call in &output.tool_calls {
                let result = self.execute_tool_call(call).await;
                if !self.opts.quiet {
                    eprintln!("\n[{}] {}", call.name, summarize_result(&result));
                }
                let entry = crate::session::SessionEntry {
                    id: crate::session::Session::entry_id(),
                    parent_id: session.leaf_id().map(|s| s.to_string()),
                    timestamp: chrono::Utc::now().to_rfc3339(),
                    message: serde_json::json!({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": result.content,
                        "is_error": result.is_error,
                    }),
                };
                let _ = session.append(&entry);
                messages.push(ChatMessage::tool_result(&call.id, result.content));
            }
            let _ = turn;
        }

        if !self.opts.quiet {
            eprintln!("\n[reached max turns ({max_turns}), stopping]");
        }
        Ok(last_text)
    }

    async fn execute_tool_call(&self, call: &ToolCall) -> ToolResult {
        match self.tools.execute(&call.name, &call.arguments).await {
            r @ ToolResult { is_error: false, .. } => r,
            r => r,
        }
    }
}

fn summarize_result(r: &ToolResult) -> String {
    let single = r.content.replace('\n', " ");
    let single: String = single.chars().take(160).collect();
    if r.is_error {
        format!("ERROR: {single}")
    } else {
        single
    }
}

#[allow(dead_code)]
pub fn unused(_p: &PathBuf, _u: Usage) {}
