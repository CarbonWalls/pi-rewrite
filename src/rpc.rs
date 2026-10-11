//! JSONL RPC over stdio, speaking the upstream pi `--mode rpc` protocol.
//! Any pi client (or pi itself, via its RpcClient conventions) can drive this
//! binary as a fast backend. Records: commands on stdin, responses + session
//! events on stdout, diagnostics on stderr.

use crate::agent::{Agent, AgentOptions};
use crate::config::{MultikeyConfig, PoolConfig, PoolModel, Settings};
use crate::llm::ChatMessage;
use crate::session::Session;
use crate::subagents::AgentRegistry;
use crate::tools::ToolRegistry;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, BufReader};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum RpcCommand {
    #[serde(rename = "prompt")]
    Prompt { id: Option<String>, message: String },
    #[serde(rename = "abort")]
    Abort { id: Option<String> },
    #[serde(rename = "get_state")]
    GetState { id: Option<String> },
    #[serde(rename = "get_messages")]
    GetMessages { id: Option<String> },
    #[serde(rename = "get_last_assistant_text")]
    GetLastAssistantText { id: Option<String> },
    #[serde(rename = "set_model")]
    SetModel { id: Option<String>, provider: String, model_id: String },
    #[serde(rename = "get_available_models")]
    GetAvailableModels { id: Option<String> },
    #[serde(rename = "bash")]
    Bash { id: Option<String>, command: String },
    #[serde(rename = "new_session")]
    NewSession { id: Option<String> },
    #[serde(rename = "get_session_stats")]
    GetSessionStats { id: Option<String> },
}

fn out(record: &impl Serialize) {
    let mut line = serde_json::to_string(record).unwrap_or_else(|_| "{}".into());
    line.push('\n');
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let _ = lock.write_all(line.as_bytes());
    let _ = lock.flush();
}

fn respond(id: Option<&str>, command: &str, success: bool, error: Option<&str>, data: serde_json::Value) {
    out(&serde_json::json!({
        "id": id,
        "type": "response",
        "command": command,
        "success": success,
        "error": error,
        "data": data,
    }));
}

fn event(record: serde_json::Value) {
    out(&record);
}

/// State carried across RPC commands.
#[allow(dead_code)]
pub struct RpcState {
    pub agent: Agent,
    pub settings: Settings,
    pub multikey: MultikeyConfig,
    pub pool: PoolConfig,
    pub model: PoolModel,
    pub registry: Option<Arc<AgentRegistry>>,
    pub config_dir: PathBuf,
    pub session: Session,
    pub messages: Vec<ChatMessage>,
    pub system_prompt: String,
    pub aborted: bool,
}

impl RpcState {
    fn emit_agent_start(&self) {
        event(serde_json::json!({"type": "agent_start"}));
        event(serde_json::json!({"type": "turn_start"}));
    }
    fn emit_agent_end(&self) {
        event(serde_json::json!({"type": "agent_end", "messages": [], "willRetry": false}));
        event(serde_json::json!({"type": "agent_settled", "aborted": self.aborted}));
    }
}

/// Run the RPC loop until stdin closes.
pub async fn run_rpc(
    mut state: RpcState,
    _agent_opts: AgentOptions,
) -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // stdin closed: orderly shutdown
        }
        let trimmed = line.trim_end_matches('\n').trim_end_matches('\r');
        if trimmed.is_empty() {
            continue;
        }

        let cmd: RpcCommand = match serde_json::from_str(trimmed) {
            Ok(c) => c,
            Err(e) => {
                respond(
                    None,
                    "parse",
                    false,
                    Some(&format!("Failed to parse command: {e}")),
                    serde_json::Value::Null,
                );
                continue;
            }
        };

        match cmd {
            RpcCommand::Prompt { id, message } => {
                state.aborted = false;
                respond(
                    id.as_deref(),
                    "prompt",
                    true,
                    None,
                    serde_json::json!({"disposition": "started"}),
                );
                state.emit_agent_start();

                event(serde_json::json!({
                    "type": "message_start",
                    "message": {"role": "user", "content": message}
                }));
                event(serde_json::json!({
                    "type": "message_end",
                    "message": {"role": "user", "content": message}
                }));

                let result = state
                    .agent
                    .run(&message, &state.system_prompt.clone(), &mut state.session)
                    .await;

                match result {
                    Ok(text) => {
                        event(serde_json::json!({
                            "type": "message_start",
                            "message": {"role": "assistant", "content": []}
                        }));
                        event(serde_json::json!({
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": text}
                        }));
                        event(serde_json::json!({
                            "type": "message_end",
                            "message": {"role": "assistant", "content": text}
                        }));
                        state.messages.push(ChatMessage::assistant(Some(text), vec![]));
                    }
                    Err(e) => {
                        event(serde_json::json!({
                            "type": "message_update",
                            "assistantMessageEvent": {"type": "error", "reason": "error", "error": e.to_string()}
                        }));
                    }
                }
                state.emit_agent_end();
            }
            RpcCommand::Abort { id } => {
                state.aborted = true;
                respond(id.as_deref(), "abort", true, None, serde_json::Value::Null);
            }
            RpcCommand::GetState { id } => {
                let model = serde_json::json!({
                    "id": state.model.id,
                    "name": state.model.name.clone().unwrap_or_default(),
                    "api": state.pool.api.clone().unwrap_or_else(|| "openai-completions".into()),
                    "provider": state.pool.id,
                    "baseUrl": state.pool.base_url,
                    "reasoning": state.model.reasoning.unwrap_or(true),
                    "contextWindow": state.model.context_window.unwrap_or(128_000),
                    "maxTokens": state.model.max_tokens.unwrap_or(16_384),
                });
                respond(
                    id.as_deref(),
                    "get_state",
                    true,
                    None,
                    serde_json::json!({
                        "model": model,
                        "isStreaming": false,
                        "isCompacting": false,
                        "sessionFile": state.session.file.to_string_lossy(),
                        "sessionId": state.session.id,
                        "messageCount": state.messages.len(),
                        "pendingMessageCount": 0,
                    }),
                );
            }
            RpcCommand::GetMessages { id } => {
                respond(
                    id.as_deref(),
                    "get_messages",
                    true,
                    None,
                    serde_json::json!({"messages": state.messages.iter().map(|m| serde_json::to_value(m).unwrap_or(serde_json::Value::Null)).collect::<Vec<_>>()}),
                );
            }
            RpcCommand::GetLastAssistantText { id } => {
                let text = state
                    .messages
                    .iter()
                    .rev()
                    .find(|m| m.role == "assistant")
                    .and_then(|m| m.content.clone());
                respond(
                    id.as_deref(),
                    "get_last_assistant_text",
                    true,
                    None,
                    serde_json::json!({"text": text}),
                );
            }
            RpcCommand::SetModel { id, provider, model_id } => {
                match state.multikey.resolve_model(&provider, &model_id) {
                    Some((pool, model)) => {
                        state.pool = pool.clone();
                        state.model = model.clone();
                        state.agent.pool = Arc::new(crate::multikey::KeyPool::new(pool.clone()));
                        state.agent.model = model.clone();
                        respond(
                            id.as_deref(),
                            "set_model",
                            true,
                            None,
                            serde_json::json!({
                                "id": model.id,
                                "name": model.name.clone().unwrap_or_default(),
                                "api": pool.api.clone().unwrap_or_else(|| "openai-completions".into()),
                                "provider": pool.id,
                                "baseUrl": pool.base_url,
                            }),
                        );
                    }
                    None => {
                        respond(
                            id.as_deref(),
                            "set_model",
                            false,
                            Some(&format!("Model not found: {provider}/{model_id}")),
                            serde_json::Value::Null,
                        );
                    }
                }
            }
            RpcCommand::GetAvailableModels { id } => {
                let models: Vec<serde_json::Value> = state
                    .multikey
                    .pools
                    .iter()
                    .flat_map(|p| {
                        p.models.iter().map(move |m| {
                            serde_json::json!({
                                "id": m.id,
                                "name": m.name.clone().unwrap_or_default(),
                                "api": p.api.clone().unwrap_or_else(|| "openai-completions".into()),
                                "provider": p.id,
                                "baseUrl": p.base_url,
                                "reasoning": m.reasoning.unwrap_or(true),
                                "contextWindow": m.context_window.unwrap_or(128_000),
                                "maxTokens": m.max_tokens.unwrap_or(16_384),
                            })
                        })
                    })
                    .collect();
                respond(
                    id.as_deref(),
                    "get_available_models",
                    true,
                    None,
                    serde_json::json!({"models": models}),
                );
            }
            RpcCommand::Bash { id, command } => {
                let reg = ToolRegistry::new(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
                let res = reg.execute("bash", &format!("{{\"command\": {}}}", serde_json::to_string(&command).unwrap_or_else(|_| "\"\"".into()))).await;
                respond(
                    id.as_deref(),
                    "bash",
                    true,
                    None,
                    serde_json::json!({
                        "output": res.content,
                        "exitCode": if res.is_error { 1 } else { 0 },
                        "cancelled": false,
                        "truncated": false,
                    }),
                );
            }
            RpcCommand::NewSession { id } => {
                state.session = Session::open(
                    &state.config_dir.join("sessions"),
                    &std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
                )?;
                state.messages.clear();
                respond(id.as_deref(), "new_session", true, None, serde_json::json!({"cancelled": false}));
            }
            RpcCommand::GetSessionStats { id } => {
                respond(
                    id.as_deref(),
                    "get_session_stats",
                    true,
                    None,
                    serde_json::json!({
                        "sessionFile": state.session.file.to_string_lossy(),
                        "sessionId": state.session.id,
                        "userMessages": state.messages.iter().filter(|m| m.role == "user").count(),
                        "assistantMessages": state.messages.iter().filter(|m| m.role == "assistant").count(),
                        "totalMessages": state.messages.len(),
                    }),
                );
            }
        }
    }

    Ok(())
}
