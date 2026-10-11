//! Native multikey: one pool, many API keys, least-loaded + least-recently-used
//! selection, cooldown tracking, and 429/401/403-aware rotation.
//!
//! Faithful port of the `pi-multikey` plugin's `pool.ts` semantics:
//!   * acquire()  -> pick enabled key not in cooldown with fewest in-flight
//!                   requests, tie-broken by least-recently-used.
//!   * report()   -> record outcomes and drive cooldowns.

use crate::config::PoolConfig;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use tokio::time::{sleep, Duration};

pub fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyOutcome {
    Ok,
    RateLimited,
    QuotaExhausted,
    Invalid,
    Error,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Lease {
    pub key: String,
    pub label: String,
    pub acquired_at: u64,
}

#[derive(Debug, Clone, Default)]
struct KeyStat {
    ok: u64,
    rate_limited: u64,
    quota_limited: u64,
    invalid: u64,
    errors: u64,
    inflight: u32,
    cooldown_until: u64,
    cooldown_reason: Option<String>,
    last_used: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyStatus {
    pub label: &'static str,
}

pub struct KeyPool {
    pub config: Arc<PoolConfig>,
    stats: Mutex<HashMap<String, KeyStat>>,
}

impl KeyPool {
    pub fn new(config: PoolConfig) -> Self {
        Self { config: Arc::new(config), stats: Mutex::new(HashMap::new()) }
    }

    pub fn size(&self) -> usize {
        self.config.keys.iter().filter(|k| k.is_enabled()).count()
    }

    fn mask(key: &str) -> String {
        if key.len() <= 8 {
            "***".to_string()
        } else {
            format!("{}…{}", &key[..4], &key[key.len() - 4..])
        }
    }

    fn enabled_keys(&self) -> Vec<(String, String)> {
        self.config
            .keys
            .iter()
            .filter(|k| k.is_enabled())
            .map(|k| (k.key.clone(), k.label.clone().unwrap_or_else(|| Self::mask(&k.key))))
            .collect()
    }

    /// Acquire a key lease. If every key is in cooldown, wait until the earliest
    /// recovery (bounded to 60s, floor 250ms).
    pub async fn acquire(&self) -> Lease {
        loop {
            let now = now_ms();
            let mut candidates: Vec<(String, String, KeyStat)> = {
                let stats = self.stats.lock().await;
                self.enabled_keys()
                    .into_iter()
                    .filter_map(|(key, label)| {
                        let s = stats.get(&key).cloned().unwrap_or_default();
                        (s.cooldown_until <= now).then(|| (key, label, s))
                    })
                    .collect()
            };

            if !candidates.is_empty() {
                candidates.sort_by(|a, b| {
                    a.2.inflight.cmp(&b.2.inflight).then(a.2.last_used.cmp(&b.2.last_used))
                });
                let (key, label, mut s) = candidates.remove(0);
                s.inflight += 1;
                s.last_used = now;
                self.stats.lock().await.insert(key.clone(), s);
                return Lease { key, label, acquired_at: now };
            }

            // All keys cooling: wait for the earliest recovery.
            let stats = self.stats.lock().await;
            let earliest = self
                .enabled_keys()
                .into_iter()
                .map(|(key, _)| stats.get(&key).map(|s| s.cooldown_until).unwrap_or(0))
                .min()
                .unwrap_or(now);
            drop(stats);
            let wait = earliest.saturating_sub(now).clamp(250, 60_000);
            sleep(Duration::from_millis(wait)).await;
        }
    }

    pub async fn release(&self, lease: &Lease) {
        let mut stats = self.stats.lock().await;
        if let Some(s) = stats.get_mut(&lease.key) {
            s.inflight = s.inflight.saturating_sub(1);
        }
    }

    pub async fn report(&self, lease: &Lease, outcome: KeyOutcome, cooldown_ms: Option<u64>) {
        let (cooldown, invalid_cooldown) = (self.config.cooldown(), self.config.invalid_key_cooldown());
        let now = now_ms();
        let mut stats = self.stats.lock().await;
        let s = stats.entry(lease.key.clone()).or_default();
        match outcome {
            KeyOutcome::Ok => {
                s.ok += 1;
                s.cooldown_until = 0;
                s.cooldown_reason = None;
            }
            KeyOutcome::RateLimited => {
                s.rate_limited += 1;
                s.cooldown_until = s.cooldown_until.max(now + cooldown_ms.unwrap_or(cooldown));
                s.cooldown_reason = Some("429".to_string());
            }
            KeyOutcome::QuotaExhausted => {
                s.quota_limited += 1;
                s.cooldown_until =
                    s.cooldown_until.max(now + cooldown_ms.unwrap_or(30 * 60_000));
                s.cooldown_reason = Some("daily limit".to_string());
            }
            KeyOutcome::Invalid => {
                s.invalid += 1;
                s.cooldown_until = s.cooldown_until.max(now + invalid_cooldown);
                s.cooldown_reason = Some("invalid".to_string());
            }
            KeyOutcome::Error => {
                s.errors += 1;
            }
        }
    }

    /// Human-readable status snapshot for `pi multikey status`.
    pub async fn status_rows(&self) -> Vec<String> {
        let now = now_ms();
        let stats = self.stats.lock().await;
        self.config
            .keys
            .iter()
            .map(|k| {
                let s = stats.get(&k.key).cloned().unwrap_or_default();
                let label = k.label.clone().unwrap_or_else(|| Self::mask(&k.key));
                let state = if !k.is_enabled() {
                    "disabled".to_string()
                } else if s.cooldown_until > now {
                    format!(
                        "cooldown {}s ({})",
                        (s.cooldown_until - now + 999) / 1000,
                        s.cooldown_reason.as_deref().unwrap_or("?")
                    )
                } else if s.inflight > 0 {
                    format!("active x{}", s.inflight)
                } else {
                    "idle".to_string()
                };
                format!(
                    "  {label:<12} {:18} {state:<30} ok:{} 429:{} quota:{} bad:{} err:{}",
                    Self::mask(&k.key),
                    s.ok,
                    s.rate_limited,
                    s.quota_limited,
                    s.invalid,
                    s.errors
                )
            })
            .collect()
    }
}
