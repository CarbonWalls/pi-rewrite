//! OpenAI-compatible streaming LLM client with native multikey rotation.
//!
//! Ports the request/rotation behavior of `pi-multikey/src/stream.ts`:
//! acquire a key lease, send the streaming chat request with that key, and on
//! 429/401/403 (or a matching error body) mark the key and rotate to the next.
//! A full attempt is only committed once it succeeds, so a rotated attempt never
//! produces partial output.

use crate::config::PoolModel;
use crate::multikey::{KeyOutcome, KeyPool, Lease};
use anyhow::Result;
use bytes::BytesMut;
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Message types (OpenAI wire format; all pools are OpenAI-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_calls: Option<Vec<ToolCallWire>>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: Some(content.into()), tool_calls: None, tool_call_id: None }
    }
    pub fn assistant(content: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        let calls = if tool_calls.is_empty() {
            None
        } else {
            Some(
                tool_calls
                    .into_iter()
                    .map(|t| ToolCallWire {
                        id: t.id,
                        kind: "function".into(),
                        function: FunctionCall { name: t.name, arguments: t.arguments },
                    })
                    .collect(),
            )
        };
        Self { role: "assistant".into(), content, tool_calls: calls, tool_call_id: None }
    }
    pub fn tool_result(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallWire {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Default)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
#[allow(dead_code)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    #[serde(rename = "total_tokens", default)]
    pub total_tokens: u64,
}

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct AssistantOutput {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub stop_reason: Option<String>,
    pub usage: Usage,
}

// ---------------------------------------------------------------------------
// Rate-limit / invalid-key classification
// ---------------------------------------------------------------------------

fn classify_message(msg: &str) -> KeyOutcome {
    let lower = msg.to_lowercase();
    if lower.contains("free limit reached on model") {
        return KeyOutcome::QuotaExhausted;
    }
    if lower.contains("rate limit")
        || lower.contains("too many requests")
        || lower.contains("quota exceeded")
        || lower.contains("quota limit")
        || lower.contains("requests per minute")
        || lower.contains("429")
    {
        return KeyOutcome::RateLimited;
    }
    if lower.contains("unauthorized")
        || lower.contains("forbidden")
        || lower.contains("invalid api key")
        || lower.contains("incorrect api key")
        || lower.contains("authentication")
        || lower.contains("401")
        || lower.contains("403")
    {
        return KeyOutcome::Invalid;
    }
    KeyOutcome::Error
}

/// Parse retry-after header (seconds) into milliseconds.
fn retry_after_ms(value: Option<&str>) -> Option<u64> {
    let v = value?;
    let seconds: f64 = v.trim().parse().ok()?;
    (seconds > 0.0).then(|| (seconds * 1000.0) as u64)
}

// ---------------------------------------------------------------------------
// Streaming request with rotation
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct LlmClient {
    http: reqwest::Client,
}

impl LlmClient {
    pub fn new() -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60 * 60)) // providers can be slow
            .build()
            .expect("build reqwest client");
        Self { http }
    }

    pub async fn stream_chat(
        &self,
        pool: Arc<KeyPool>,
        model: &PoolModel,
        messages: &[ChatMessage],
        tools: &[ToolDef],
    ) -> Result<AssistantOutput> {
        let max_attempts = std::cmp::max(1, pool.size());
        let mut last_problem = String::from("no attempts made");

        for _attempt in 0..max_attempts {
            let lease = pool.acquire().await;
            let result = self
                .try_once(&pool, &lease, model, messages, tools)
                .await;
            let outcome = match &result {
                Ok(_) => KeyOutcome::Ok,
                Err(e) => classify_message(&e.to_string()),
            };

            match result {
                Ok(out) => {
                    pool.report(&lease, KeyOutcome::Ok, None).await;
                    pool.release(&lease).await;
                    return Ok(out);
                }
                Err(e) => {
                    let msg = e.to_string();
                    tracing::debug!("multikey[{}]: {} on {} -> rotating", pool.config.id, msg, lease.label);
                    let cooldown_ms = if outcome == KeyOutcome::RateLimited {
                        // The error carries no retry-after; the streaming path has it.
                        None
                    } else {
                        None
                    };
                    pool.report(&lease, outcome, cooldown_ms).await;
                    pool.release(&lease).await;
                    last_problem = msg;
                }
            }
        }

        anyhow::bail!("all {max_attempts} keys exhausted (last: {last_problem})")
    }

    /// One attempt on a leased key. Returns the assembled assistant message, or an
    /// error whose message is classifiable for rotation.
    async fn try_once(
        &self,
        pool: &KeyPool,
        lease: &Lease,
        model: &PoolModel,
        messages: &[ChatMessage],
        tools: &[ToolDef],
    ) -> Result<AssistantOutput> {
        let cfg = &pool.config;
        let url = cfg.endpoint(&cfg.base_url);

        let mut body = serde_json::json!({
            "model": model.id,
            "messages": messages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        if !tools.is_empty() {
            let tool_objs: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    serde_json::json!({
                        "type": "function",
                        "function": {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,
                        }
                    })
                })
                .collect();
            body["tools"] = serde_json::Value::Array(tool_objs);
            body["tool_choice"] = serde_json::json!("auto");
        }

        let mut req = self.http.post(&url).json(&body);
        if cfg.uses_api_key_header() {
            req = req.header("x-api-key", &lease.key);
        } else {
            req = req.bearer_auth(&lease.key);
        }

        let resp = req.send().await.map_err(|e| anyhow::anyhow!("request failed: {e}"))?;
        let status = resp.status();

        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            let ra = resp
                .headers()
                .get(reqwest::header::RETRY_AFTER)
                .and_then(|v| v.to_str().ok());
            let ms = retry_after_ms(ra);
            let _ = ms;
            // The caller classifies the message; embed retry context via a marker.
            let note = ms.map(|m| format!(" (retry-after {m}ms)")).unwrap_or_default();
            anyhow::bail!("429 rate limited{note}");
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            anyhow::bail!("HTTP {} unauthorized/forbidden", status.as_u16());
        }
        if !status.is_success() {
            anyhow::bail!("HTTP {}", status.as_u16());
        }

        // 2xx: read the SSE stream.
        parse_sse(resp.bytes_stream()).await
    }
}

// ---------------------------------------------------------------------------
// SSE parsing
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

fn process_chunk(
    chunk: &serde_json::Value,
    tool_accums: &mut Vec<ToolCallAccum>,
    stop_reason: &mut Option<String>,
    usage: &mut Usage,
) {
    let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else { return };
    if let Some(u) = chunk.get("usage") {
        merge_usage(usage, u);
    }
    for choice in choices {
        if let Some(sr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
            if !sr.is_empty() {
                *stop_reason = Some(sr.to_string());
            }
        }
        let Some(delta) = choice.get("delta") else { continue };

        if let Some(tcs) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for tc in tcs {
                let index = tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                while tool_accums.len() <= index {
                    tool_accums.push(ToolCallAccum::default());
                }
                let a = &mut tool_accums[index];
                if let Some(id) = tc.get("id").and_then(|v| v.as_str()) {
                    a.id = id.to_string();
                }
                if let Some(f) = tc.get("function") {
                    if let Some(n) = f.get("name").and_then(|v| v.as_str()) {
                        a.name = n.to_string();
                    }
                    if let Some(arg) = f.get("arguments").and_then(|v| v.as_str()) {
                        a.arguments.push_str(arg);
                    }
                }
            }
        }
    }
}

fn merge_usage(usage: &mut Usage, u: &serde_json::Value) {
    if let Some(v) = u.get("input_tokens").and_then(|v| v.as_u64()).or_else(|| u.get("prompt_tokens").and_then(|v| v.as_u64())) {
        usage.input = v;
    }
    if let Some(v) = u.get("output_tokens").and_then(|v| v.as_u64()).or_else(|| u.get("completion_tokens").and_then(|v| v.as_u64())) {
        usage.output = v;
    }
    if let Some(v) = u.get("total_tokens").and_then(|v| v.as_u64()) {
        usage.total_tokens = v;
    }
    if let Some(v) = u.get("cache_read_input_tokens").and_then(|v| v.as_u64()).or_else(|| u.get("prompt_cache_hit_tokens").and_then(|v| v.as_u64())) {
        usage.cache_read = v;
    }
    if let Some(v) = u.get("cache_creation_input_tokens").and_then(|v| v.as_u64()).or_else(|| u.get("prompt_cache_miss_tokens").and_then(|v| v.as_u64())) {
        usage.cache_write = v;
    }
}

async fn parse_sse(
    stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>>,
) -> Result<AssistantOutput> {
    let mut buf = BytesMut::new();
    let mut stream = Box::pin(stream);

    let mut content = String::new();
    let mut tool_accums: Vec<ToolCallAccum> = Vec::new();
    let mut stop_reason: Option<String> = None;
    let mut usage = Usage::default();

    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| anyhow::anyhow!("stream error: {e}"))?;
        buf.extend_from_slice(&chunk);

        loop {
            let Some(pos) = buf.iter().position(|&b| b == b'\n') else { break };
            let line = buf.split_to(pos + 1);
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches('\n').trim_end_matches('\r');

            // A SSE comment or keep-alive line (": ..." or empty).
            let data = line.strip_prefix("data:").map(|s| s.trim());
            let Some(data) = data else { continue };

            if data == "[DONE]" {
                let out = assemble_output(content, tool_accums, stop_reason, usage);
                return Ok(out);
            }
            if data.is_empty() {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(data)
                .map_err(|e| anyhow::anyhow!("bad SSE JSON: {e}"))?;
            if let Some(delta_content) = v
                .get("choices")
                .and_then(|c| c.get(0))
                .and_then(|c| c.get("delta"))
                .and_then(|d| d.get("content"))
            {
                if let Some(s) = delta_content.as_str() {
                    content.push_str(s);
                }
            }
            process_chunk(&v, &mut tool_accums, &mut stop_reason, &mut usage);
        }
    }

    // Stream ended without [DONE].
    Ok(assemble_output(content, tool_accums, stop_reason, usage))
}

fn assemble_output(
    content: String,
    tool_accums: Vec<ToolCallAccum>,
    stop_reason: Option<String>,
    usage: Usage,
) -> AssistantOutput {
    let tool_calls = tool_accums
        .into_iter()
        .filter(|t| !t.name.is_empty())
        .map(|t| ToolCall { id: t.id, name: t.name, arguments: t.arguments })
        .collect();
    let content = if content.is_empty() { None } else { Some(content) };
    AssistantOutput { content, tool_calls, stop_reason, usage }
}
