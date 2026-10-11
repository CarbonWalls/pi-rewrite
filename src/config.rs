//! Config resolution: find the active pi config directory (native ppi support)
//! and load settings.json / auth.json / multikey.json / models-store.json.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
pub const APP_NAME: &str = "pi-rewrite";

/// Resolve the active config directory, in priority order:
///   1. `--profile <name>` CLI flag (resolved against `~/.pi/profiles/<name>`)
///   2. `PI_CODING_AGENT_DIR` environment variable (set by ppi's `use`)
///   3. `~/.pi/profiles/default.json` -> active profile dir
///   4. `~/.pi/agent` (stock pi config)
pub fn resolve_config_dir(profile: Option<&str>) -> Result<PathBuf> {
    let pi_root = env_pi_root()?;

    if let Some(name) = profile {
        let dir = pi_root.join("profiles").join(name);
        if !dir.is_dir() {
            anyhow::bail!("profile \"{name}\" does not exist at {}", dir.display());
        }
        return Ok(dir);
    }

    if let Ok(dir) = std::env::var("PI_CODING_AGENT_DIR") {
        let p = PathBuf::from(dir);
        if p.is_dir() {
            return Ok(p);
        }
    }

    let default_json = pi_root.join("profiles").join("default.json");
    if default_json.exists() {
        if let Ok(raw) = std::fs::read_to_string(&default_json) {
            if let Ok(obj) = serde_json::from_str::<ProfileDefault>(&raw) {
                let dir = pi_root.join("profiles").join(&obj.default);
                if dir.is_dir() {
                    return Ok(dir);
                }
            }
        }
    }

    Ok(pi_root.join("agent"))
}

/// `~/.pi` root, honoring `PI_ROOT` if set (used by ppi tests).
pub fn env_pi_root() -> Result<PathBuf> {
    if let Ok(p) = std::env::var("PI_ROOT") {
        return Ok(PathBuf::from(p));
    }
    let home = dirs::home_dir().context("could not determine home directory")?;
    Ok(home.join(".pi"))
}

fn profiles_dir() -> Result<PathBuf> {
    Ok(env_pi_root()?.join("profiles"))
}

#[derive(Debug, Deserialize, Serialize, Default)]
struct ProfileDefault {
    #[serde(default)]
    default: String,
}

// ---------------------------------------------------------------------------
// settings.json
// ---------------------------------------------------------------------------

/// Tolerant mirror of pi's settings.json. Unknown fields are ignored so different
/// pi versions keep working.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct Settings {
    pub last_changelog_version: Option<String>,
    #[serde(rename = "packages")]
    pub packages: Vec<String>,
    pub tui_mode: Option<String>,
    pub system_prompt: Option<String>,
    pub append_system_prompt: Option<String>,
    pub default_provider: Option<String>,
    pub default_model: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Settings {
    pub fn load_from(config_dir: &Path) -> Result<Self> {
        let path = config_dir.join("settings.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// multikey.json
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct MultikeyConfig {
    pub pools: Vec<PoolConfig>,
}

impl Default for MultikeyConfig {
    fn default() -> Self {
        Self { pools: Vec::new() }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PoolConfig {
    pub id: String,
    pub name: Option<String>,
    pub base_url: String,
    /// "openai-completions" is the common value; we also accept "openai-responses".
    pub api: Option<String>,
    /// "bearer" (default) or "api-key" (x-api-key header).
    pub auth: Option<String>,
    pub cooldown_ms: Option<u64>,
    pub invalid_key_cooldown_ms: Option<u64>,
    pub keys: Vec<PoolKey>,
    pub models: Vec<PoolModel>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: None,
            base_url: String::new(),
            api: None,
            auth: None,
            cooldown_ms: None,
            invalid_key_cooldown_ms: None,
            keys: Vec::new(),
            models: Vec::new(),
            extra: BTreeMap::new(),
        }
    }
}

impl PoolConfig {
    pub fn cooldown(&self) -> u64 {
        self.cooldown_ms.unwrap_or(20_000)
    }
    pub fn invalid_key_cooldown(&self) -> u64 {
        self.invalid_key_cooldown_ms.unwrap_or(600_000)
    }
    pub fn uses_api_key_header(&self) -> bool {
        self.auth.as_deref() == Some("api-key")
    }
    pub fn endpoint(&self, base_url: &str) -> String {
        let base = base_url.trim_end_matches('/');
        let api = self.api.as_deref().unwrap_or("openai-completions");
        match api {
            "openai-responses" => format!("{base}/responses"),
            _ => format!("{base}/chat/completions"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PoolKey {
    pub key: String,
    pub label: Option<String>,
    pub enabled: Option<bool>,
}

impl Default for PoolKey {
    fn default() -> Self {
        Self { key: String::new(), label: None, enabled: None }
    }
}

impl PoolKey {
    pub fn is_enabled(&self) -> bool {
        self.enabled.unwrap_or(true)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PoolModel {
    pub id: String,
    pub name: Option<String>,
    pub reasoning: Option<bool>,
    pub context_window: Option<u64>,
    pub max_tokens: Option<u64>,
    pub input: Option<Vec<String>>,
    pub api: Option<String>,
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

impl Default for PoolModel {
    fn default() -> Self {
        Self {
            id: String::new(),
            name: None,
            reasoning: None,
            context_window: None,
            max_tokens: None,
            input: None,
            api: None,
            extra: BTreeMap::new(),
        }
    }
}

impl MultikeyConfig {
    pub fn load_from(config_dir: &Path) -> Result<Self> {
        let path = config_dir.join("multikey.json");
        if !path.exists() {
            return Ok(Self::default());
        }
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("reading {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("parsing {}", path.display()))
    }

    pub fn find_pool(&self, id: &str) -> Option<&PoolConfig> {
        self.pools.iter().find(|p| p.id == id)
    }

    /// Resolve `<poolId>/<modelId>` or `<poolId>` + separate model id.
    pub fn resolve_model(&self, provider: &str, model: &str) -> Option<(&PoolConfig, PoolModel)> {
        // provider may be "pool/model" (from --model pool/model w/ no provider)
        if let Some((pool_id, model_id)) = provider.split_once('/') {
            let pool = self.find_pool(pool_id)?;
            let m = pool.models.iter().find(|m| m.id == model_id)?;
            return Some((pool, m.clone()));
        }
        let pool = self.find_pool(provider)?;
        let m = if model.is_empty() {
            pool.models.first()?.clone()
        } else {
            // model may itself be "pool/model"
            if let Some((_, mid)) = model.split_once('/') {
                pool.models.iter().find(|m| m.id == mid)?.clone()
            } else {
                pool.models.iter().find(|m| m.id == model)?.clone()
            }
        };
        Some((pool, m))
    }
}

// ---------------------------------------------------------------------------
// auth.json (stock pi auth: subscriptions / api keys)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct AuthConfig {
    #[serde(flatten)]
    pub extra: BTreeMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Profile (ppi) management — built-in, supersedes pi-profiles/ppi-tui
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ProfileInfo {
    pub name: String,
    pub path: PathBuf,
    pub is_default: bool,
}

pub fn list_profiles() -> Result<Vec<ProfileInfo>> {
    let dir = profiles_dir()?;
    let default = default_profile_name()?;
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        if entry.path().is_dir() {
            let name = entry.file_name().to_string_lossy().to_string();
            out.push(ProfileInfo {
                name: name.clone(),
                path: entry.path(),
                is_default: default.as_deref() == Some(&name),
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

pub fn default_profile_name() -> Result<Option<String>> {
    let p = profiles_dir()?.join("default.json");
    if !p.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&p)?;
    let obj = serde_json::from_str::<ProfileDefault>(&raw)?;
    Ok(if obj.default.is_empty() { None } else { Some(obj.default) })
}

pub fn set_default_profile(name: &str) -> Result<()> {
    let dir = profiles_dir()?;
    let profile_dir = dir.join(name);
    if !profile_dir.is_dir() {
        anyhow::bail!("profile \"{name}\" does not exist at {}", profile_dir.display());
    }
    std::fs::create_dir_all(&dir)?;
    std::fs::write(
        dir.join("default.json"),
        serde_json::to_string_pretty(&ProfileDefault { default: name.to_string() })? + "\n",
    )?;
    Ok(())
}

#[allow(dead_code)]
pub fn profile_exists(name: &str) -> Result<bool> {
    Ok(profiles_dir()?.join(name).is_dir())
}

const NAME_REGEX: &str = r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$";

pub fn validate_profile_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 64
        || !regex::Regex::new(NAME_REGEX).unwrap().is_match(name)
    {
        anyhow::bail!(
            "invalid profile name \"{name}\": must match {} and be at most 64 characters",
            NAME_REGEX
        );
    }
    Ok(())
}

/// Create a profile. `from_base` copies the stock `~/.pi/agent` dir (preserving
/// symlinks) excluding sessions; `from` copies from an existing profile.
pub fn create_profile(name: &str, from_base: bool, from: Option<&str>) -> Result<PathBuf> {
    validate_profile_name(name)?;
    let dir = profiles_dir()?;
    let dest = dir.join(name);
    if dest.exists() {
        anyhow::bail!("profile \"{name}\" already exists at {}", dest.display());
    }
    let pi_root = env_pi_root()?;
    let agent = pi_root.join("agent");

    if from_base || from.is_some() {
        let src = match from {
            Some(p) => {
                validate_profile_name(p)?;
                dir.join(p)
            }
            None => agent.clone(),
        };
        copy_dir_preserving_symlinks(&src, &dest, "sessions")?;
        std::fs::create_dir_all(dest.join("sessions"))?;
    } else {
        std::fs::create_dir_all(&dest)?;
        for sub in ["extensions", "skills", "tools", "prompts", "sessions"] {
            std::fs::create_dir_all(dest.join(sub))?;
        }
        std::fs::write(dest.join("settings.json"), "{}\n")?;
    }
    Ok(dest)
}

pub fn delete_profile(name: &str) -> Result<()> {
    validate_profile_name(name)?;
    let dir = profiles_dir()?;
    let path = dir.join(name);
    if !path.is_dir() {
        anyhow::bail!("profile \"{name}\" does not exist");
    }
    std::fs::remove_dir_all(&path)?;
    if default_profile_name()?.as_deref() == Some(name) {
        let dj = dir.join("default.json");
        if dj.exists() {
            std::fs::remove_file(dj)?;
        }
    }
    Ok(())
}

/// Recursive copy preserving symlinks; skips contents of `exclude_dir` but creates it.
pub fn copy_dir_preserving_symlinks(src: &Path, dest: &Path, exclude_dir: &str) -> Result<()> {
    std::fs::create_dir_all(dest)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dest_path = dest.join(entry.file_name());
        let meta = entry.metadata()?;
        if meta.file_type().is_symlink() {
            let target = std::fs::read_link(&src_path)?;
            symlink(&target, &dest_path)?;
        } else if meta.is_dir() {
            if entry.file_name() == exclude_dir {
                std::fs::create_dir_all(&dest_path)?;
            } else {
                copy_dir_preserving_symlinks(&src_path, &dest_path, exclude_dir)?;
            }
        } else {
            std::fs::copy(&src_path, &dest_path)?;
        }
    }
    Ok(())
}
