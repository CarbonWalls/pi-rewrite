//! Built-in agent tools: read, write, edit, bash, glob, grep, plus the native
//! peer-subagent tools that make every model an agent that can ping the others.

use crate::llm::ToolDef;
use crate::subagents::AgentRegistry;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Command;

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub content: String,
    pub is_error: bool,
}

impl ToolResult {
    pub fn ok(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: false }
    }
    pub fn err(content: impl Into<String>) -> Self {
        Self { content: content.into(), is_error: true }
    }
}

#[derive(Clone)]
pub struct ToolRegistry {
    pub cwd: PathBuf,
    pub agents: Option<Arc<AgentRegistry>>,
    pub self_agent: Option<String>,
    pub max_output: usize,
}

fn obj(schema: serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "object",
        "properties": schema,
        "required": []
    })
}

fn prop(ty: &str, desc: &str) -> serde_json::Value {
    serde_json::json!({ "type": ty, "description": desc })
}

impl ToolRegistry {
    pub fn new(cwd: PathBuf) -> Self {
        Self { cwd, agents: None, self_agent: None, max_output: 200_000 }
    }

    pub fn defs(&self) -> Vec<ToolDef> {
        let mut defs = vec![
            ToolDef {
                name: "read".into(),
                description: "Read a UTF-8 text file, returning line-numbered content. Use to inspect files before editing.".into(),
                parameters: obj(serde_json::json!({
                    "path": prop("string", "Path to the file to read."),
                    "offset": {"type":"integer","description":"Optional 1-based first line to return (default 1)."},
                    "limit": {"type":"integer","description":"Optional maximum number of lines to return."},
                })),
            },
            ToolDef {
                name: "write".into(),
                description: "Create or fully replace a UTF-8 text file. Use for new files or full rewrites.".into(),
                parameters: obj(serde_json::json!({
                    "path": prop("string", "Path to write, resolved against the working directory."),
                    "content": prop("string", "Full UTF-8 text content to write."),
                })),
            },
            ToolDef {
                name: "edit".into(),
                description: "Edit an existing UTF-8 text file by replacing a literal string. Use for targeted changes.".into(),
                parameters: obj(serde_json::json!({
                    "path": prop("string", "Path to the file to edit."),
                    "old_string": prop("string", "Exact literal text to replace (must appear in the file)."),
                    "new_string": prop("string", "Replacement text. Use an empty string to delete the match."),
                    "replace_all": {"type":"boolean","description":"Optional: replace all occurrences. Default false."},
                })),
            },
            ToolDef {
                name: "bash".into(),
                description: "Execute a shell command and return stdout/stderr and the exit code. Use for building, testing, git, and file operations.".into(),
                parameters: obj(serde_json::json!({
                    "command": prop("string", "The shell command to run (bash -c)."),
                    "workdir": prop("string", "Optional working directory. Defaults to the project root."),
                    "timeout_ms": {"type":"integer","description":"Optional timeout in milliseconds."},
                })),
            },
            ToolDef {
                name: "glob".into(),
                description: "Find files whose paths match a glob pattern (e.g. '**/*.rs', 'src/**').".into(),
                parameters: obj(serde_json::json!({
                    "pattern": prop("string", "Glob pattern to match file paths against."),
                    "path": prop("string", "Optional directory to search in. Defaults to the project root."),
                })),
            },
            ToolDef {
                name: "grep".into(),
                description: "Search file contents with a regular expression and return matching lines with line numbers.".into(),
                parameters: obj(serde_json::json!({
                    "pattern": prop("string", "Regular expression to search for."),
                    "path": prop("string", "Optional file or directory to search. Defaults to the project root."),
                    "include": prop("string", "Optional glob filter for which files to search (e.g. '*.ts')."),
                })),
            },
        ];

        if self.agents.is_some() {
            defs.push(ToolDef {
                name: "agents_list".into(),
                description: "List every peer agent in the collaboration mesh (name and model).".into(),
                parameters: obj(serde_json::json!({})),
            });
            defs.push(ToolDef {
                name: "agents_ping".into(),
                description: "Send a message to a specific peer agent. The agent receives it in its inbox and answers at its next turn. This is peer-to-peer: any agent can ping any other, no orchestrator.".into(),
                parameters: obj(serde_json::json!({
                    "agent": prop("string", "Name of the target agent."),
                    "message": prop("string", "Message text to deliver."),
                })),
            });
            defs.push(ToolDef {
                name: "agents_broadcast".into(),
                description: "Send a message to every other agent in the mesh at once.".into(),
                parameters: obj(serde_json::json!({
                    "message": prop("string", "Message text to deliver."),
                })),
            });
        }

        defs
    }

    pub async fn execute(&self, name: &str, arguments: &str) -> ToolResult {
        let args: serde_json::Value = match serde_json::from_str::<serde_json::Value>(arguments) {
            Ok(v) if v.is_object() => v,
            Ok(v) => return ToolResult::err(format!("arguments must be a JSON object, got {}", v)),
            Err(e) => return ToolResult::err(format!("invalid tool arguments JSON: {e}")),
        };
        let s = |k: &str| args.get(k).and_then(|v| v.as_str()).map(|s| s.to_string());
        let i = |k: &str| args.get(k).and_then(|v| v.as_u64());

        let res = match name {
            "read" => Self::read(s("path").as_deref(), i("offset"), i("limit"), self).await,
            "write" => Self::write(s("path").as_deref(), s("content").as_deref(), self).await,
            "edit" => Self::edit(s("path").as_deref(), s("old_string").as_deref(), s("new_string").as_deref(), &args, self).await,
            "bash" => Self::bash(s("command").as_deref(), s("workdir").as_deref(), i("timeout_ms"), self).await,
            "glob" => Self::glob_tool(s("pattern").as_deref(), s("path").as_deref(), self).await,
            "grep" => Self::grep_tool(s("pattern").as_deref(), s("path").as_deref(), s("include").as_deref(), self).await,
            "agents_list" => self.agents_list().await,
            "agents_ping" => self.agents_ping(s("agent").as_deref(), s("message").as_deref()).await,
            "agents_broadcast" => self.agents_broadcast(s("message").as_deref()).await,
            other => ToolResult::err(format!("unknown tool: {other}")),
        };
        res
    }

    fn resolve(&self, path: Option<&str>) -> PathBuf {
        match path {
            None => self.cwd.clone(),
            Some(p) => {
                let p = PathBuf::from(p);
                if p.is_absolute() {
                    p
                } else {
                    self.cwd.join(p)
                }
            }
        }
    }

    fn truncate(&self, mut s: String) -> String {
        if s.len() > self.max_output {
            s.truncate(self.max_output);
            s.push_str("\n… (output truncated)");
        }
        s
    }

    async fn read(path: Option<&str>, offset: Option<u64>, limit: Option<u64>, reg: &ToolRegistry) -> ToolResult {
        let Some(p) = path else { return ToolResult::err("read: missing 'path'") };
        let path = reg.resolve(Some(p));
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("read: {e}")),
        };
        let lines: Vec<&str> = raw.lines().collect();
        let start = offset.unwrap_or(1).saturating_sub(1) as usize;
        let start = start.min(lines.len());
        let count = limit.unwrap_or(lines.len() as u64).max(1) as usize;
        let end = (start + count).min(lines.len());
        let mut out = String::new();
        for (idx, line) in lines[start..end].iter().enumerate() {
            out.push_str(&format!("{:>6}\t{}\n", start + idx + 1, line));
        }
        if end < lines.len() {
            out.push_str(&format!("… ({} more lines)\n", lines.len() - end));
        }
        ToolResult::ok(reg.truncate(out))
    }

    async fn write(path: Option<&str>, content: Option<&str>, reg: &ToolRegistry) -> ToolResult {
        let (Some(p), Some(c)) = (path, content) else {
            return ToolResult::err("write: missing 'path' or 'content'");
        };
        let path = reg.resolve(Some(p));
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                let _ = std::fs::create_dir_all(parent);
            }
        }
        match std::fs::write(&path, c) {
            Ok(_) => ToolResult::ok(format!("wrote {} bytes to {}", c.len(), path.display())),
            Err(e) => ToolResult::err(format!("write: {e}")),
        }
    }

    async fn edit(
        path: Option<&str>,
        old: Option<&str>,
        new: Option<&str>,
        args: &serde_json::Value,
        reg: &ToolRegistry,
    ) -> ToolResult {
        let (Some(p), Some(old)) = (path, old) else {
            return ToolResult::err("edit: missing 'path' or 'old_string'");
        };
        let new = new.unwrap_or("");
        let path = reg.resolve(Some(p));
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("edit: {e}")),
        };
        if !raw.contains(old) {
            return ToolResult::err("edit: old_string not found in file (read the file first to get exact text)");
        }
        let replace_all = args.get("replace_all").and_then(|v| v.as_bool()).unwrap_or(false);
        let count = if replace_all { usize::MAX } else { 1 };
        let updated: String = if replace_all {
            raw.replace(old, new)
        } else {
            raw.replacen(old, new, count)
        };
        match std::fs::write(&path, &updated) {
            Ok(_) => ToolResult::ok(format!("edited {}", path.display())),
            Err(e) => ToolResult::err(format!("edit: {e}")),
        }
    }

    async fn bash(command: Option<&str>, workdir: Option<&str>, timeout_ms: Option<u64>, reg: &ToolRegistry) -> ToolResult {
        let Some(cmd) = command else { return ToolResult::err("bash: missing 'command'") };
        let wd = reg.resolve(workdir);
        let default_timeout = std::time::Duration::from_secs(10 * 60);
        let timeout = std::time::Duration::from_millis(timeout_ms.unwrap_or(default_timeout.as_millis() as u64));

        let output = tokio::time::timeout(timeout, async {
            Command::new("bash")
                .arg("-c")
                .arg(cmd)
                .current_dir(&wd)
                .output()
                .await
        })
        .await;

        match output {
            Err(_) => ToolResult::err(format!("bash: timed out after {}ms\ncommand: {cmd}", timeout.as_millis())),
            Ok(Err(e)) => ToolResult::err(format!("bash: failed to spawn: {e}")),
            Ok(Ok(out)) => {
                let mut body = String::new();
                let stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);
                if !stdout.is_empty() {
                    body.push_str(&stdout);
                }
                if !stderr.is_empty() {
                    if !body.is_empty() {
                        body.push('\n');
                    }
                    body.push_str("[stderr]\n");
                    body.push_str(&stderr);
                }
                body.push_str(&format!("\n[exit code: {}]", out.status.code().unwrap_or(-1)));
                let res = reg.truncate(body);
                if out.status.success() {
                    ToolResult::ok(res)
                } else {
                    ToolResult { content: res, is_error: true }
                }
            }
        }
    }

    async fn glob_tool(pattern: Option<&str>, path: Option<&str>, reg: &ToolRegistry) -> ToolResult {
        let Some(pat) = pattern else { return ToolResult::err("glob: missing 'pattern'") };
        let base = reg.resolve(path);
        let out = run_glob(&base, pat, 100);
        ToolResult::ok(reg.truncate(out.join("\n")))
    }

    async fn grep_tool(pattern: Option<&str>, path: Option<&str>, include: Option<&str>, reg: &ToolRegistry) -> ToolResult {
        let Some(pat) = pattern else { return ToolResult::err("grep: missing 'pattern'") };
        let re = match Regex::new(pat) {
            Ok(r) => r,
            Err(e) => return ToolResult::err(format!("grep: invalid regex: {e}")),
        };
        let base = reg.resolve(path);
        let out = run_grep(&base, &re, include, 250);
        ToolResult::ok(reg.truncate(out))
    }

    // ---- subagent tools ----
    async fn agents_list(&self) -> ToolResult {
        let Some(agents) = &self.agents else {
            return ToolResult::err("agents tool unavailable");
        };
        let rows = agents.list().unwrap_or_default();
        let text = rows
            .into_iter()
            .map(|(name, model)| format!("{name}\t{model}"))
            .collect::<Vec<_>>()
            .join("\n");
        ToolResult::ok(if text.is_empty() { "(no agents registered)".to_string() } else { text })
    }

    async fn agents_ping(&self, target: Option<&str>, message: Option<&str>) -> ToolResult {
        let (Some(target), Some(message)) = (target, message) else {
            return ToolResult::err("agents_ping: missing 'agent' or 'message'");
        };
        let Some(agents) = &self.agents else { return ToolResult::err("agents tool unavailable") };
        let from = self.self_agent.as_deref().unwrap_or("anonymous");
        match agents.deliver(from, target, message) {
            Ok(()) => ToolResult::ok(format!("pinged {target}")),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }

    async fn agents_broadcast(&self, message: Option<&str>) -> ToolResult {
        let Some(message) = message else { return ToolResult::err("agents_broadcast: missing 'message'") };
        let Some(agents) = &self.agents else { return ToolResult::err("agents tool unavailable") };
        let from = self.self_agent.as_deref().unwrap_or("anonymous");
        match agents.broadcast(from, message) {
            Ok(n) => ToolResult::ok(format!("broadcast to {n} agent(s)")),
            Err(e) => ToolResult::err(e.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// Glob
// ---------------------------------------------------------------------------

fn glob_to_regex(pattern: &str) -> Regex {
    let mut rx = String::from("^");
    for c in pattern.chars() {
        match c {
            '*' => rx.push_str(".*"),
            '?' => rx.push('.'),
            '.' | '+' | '(' | ')' | '|' | '^' | '$' | '@' | '%' | '{' | '}' | '[' | ']' | '\\' => {
                rx.push('\\');
                rx.push(c);
            }
            other => rx.push(other),
        }
    }
    rx.push('$');
    Regex::new(&rx).expect("glob regex")
}

fn run_glob(base: &Path, pattern: &str, limit: usize) -> Vec<String> {
    let re = glob_to_regex(pattern);
    let mut out = Vec::new();
    if base.is_file() {
        let rel = rel_path(base);
        if re.is_match(&rel) || re.is_match(&base.to_string_lossy()) {
            out.push(rel);
        }
        return out;
    }
    let anchor = pattern.contains('/');
    for entry in walkdir::WalkDir::new(base)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        if !entry.file_type().is_file() {
            continue;
        }
        let p = entry.path();
        let rel = p.strip_prefix(base).unwrap_or(p);
        let rel_s = rel.to_string_lossy().to_string();
        let candidate = if anchor { rel_s } else {
            let basename = p.file_name().map(|b| b.to_string_lossy().to_string()).unwrap_or_default();
            basename.clone()
        };
        if re.is_match(&candidate) {
            out.push(rel_path(p));
            if out.len() >= limit {
                break;
            }
        }
    }
    out
}

fn rel_path(p: &Path) -> String {
    if p.is_absolute() {
        p.to_string_lossy().to_string()
    } else {
        p.to_string_lossy().to_string()
    }
}

// ---------------------------------------------------------------------------
// Grep
// ---------------------------------------------------------------------------

fn run_grep(base: &Path, re: &Regex, include: Option<&str>, limit: usize) -> String {
    let include_re = include.map(glob_to_regex);
    let mut out = String::new();
    let mut count = 0;

    let files: Vec<PathBuf> = if base.is_file() {
        vec![base.to_path_buf()]
    } else {
        walkdir::WalkDir::new(base)
            .follow_links(false)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file())
            .map(|e| e.path().to_path_buf())
            .collect()
    };

    for path in files {
        if let Some(re) = &include_re {
            let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if !re.is_match(&name) {
                continue;
            }
        }
        let Ok(raw) = std::fs::read_to_string(&path) else { continue };
        for (i, line) in raw.lines().enumerate() {
            if re.is_match(line) {
                out.push_str(&format!("{}:{}\t{}\n", rel_path(&path), i + 1, line));
                count += 1;
                if count >= limit {
                    return out;
                }
            }
        }
    }
    if out.is_empty() {
        "(no matches)".to_string()
    } else {
        out
    }
}
