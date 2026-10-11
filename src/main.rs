//! pi-rewrite: fast native reimplementation of pi with native multikey,
//! system prompt, peer subagent control, and ppi (profiles).

mod agent;
mod config;
mod llm;
mod multikey;
mod rpc;
mod session;
mod subagents;
mod tools;

use agent::{Agent, AgentOptions};
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::{
    create_profile, default_profile_name, delete_profile, list_profiles, resolve_config_dir,
    set_default_profile, MultikeyConfig, Settings,
};
use rpc::{run_rpc, RpcState};
use session::Session;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use subagents::AgentRegistry;
use tools::ToolRegistry;

#[derive(Parser, Debug)]
#[command(name = "pi-rewrite", version, about = "Fast native pi with multikey, system prompt, subagents, ppi")]
struct Cli {
    /// Profile to use (native ppi): resolves ~/.pi/profiles/<name> as the config dir
    #[arg(long, global = true)]
    profile: Option<String>,

    /// Provider (pool id) override
    #[arg(long, global = true)]
    provider: Option<String>,

    /// Model id override (may be <pool>/<model>)
    #[arg(long, global = true)]
    model: Option<String>,

    /// Replace the system prompt (literal text or a path to a file)
    #[arg(long, global = true)]
    system_prompt: Option<String>,

    /// Append to the system prompt (literal text or a path to a file)
    #[arg(long, global = true)]
    append_system_prompt: Option<String>,

    /// Agent name for peer collaboration (defaults to the profile name or "main")
    #[arg(long, global = true)]
    agent: Option<String>,

    /// Maximum agent turns before stopping
    #[arg(long, global = true, default_value_t = 40)]
    max_turns: usize,

    /// Print machine-readable events instead of human output
    #[arg(long, global = true)]
    quiet: bool,

    /// Run mode: one-shot prompt (default), json, or rpc (JSONL over stdio)
    #[arg(long, global = true, default_value = "prompt")]
    mode: String,

    #[command(subcommand)]
    command: Option<Commands>,

    /// Prompt text for one-shot mode
    prompt: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Native ppi: manage configuration profiles
    Profile {
        #[command(subcommand)]
        action: ProfileAction,
    },
    /// Native multikey: pool and key status
    Multikey {
        #[command(subcommand)]
        action: MultikeyAction,
    },
    /// Native peer subagent control
    Agent {
        #[command(subcommand)]
        action: AgentAction,
    },
}

#[derive(Subcommand, Debug, Clone)]
enum ProfileAction {
    /// List all profiles
    List,
    /// Show the active profile
    Show,
    /// Set the default profile
    Use { name: String },
    /// Set the default profile explicitly
    SetDefault { name: String },
    /// Create a new profile
    Create {
        name: String,
        /// Copy from the stock ~/.pi/agent config
        #[arg(long)]
        from_base: bool,
        /// Copy from an existing profile
        #[arg(long)]
        from: Option<String>,
    },
    /// Delete a profile
    Delete { name: String },
}

#[derive(Subcommand, Debug, Clone)]
enum MultikeyAction {
    /// Live per-key state: in-flight, cooldowns, 429 counts
    Status,
    /// List pools, keys, and models
    List,
}

#[derive(Subcommand, Debug, Clone)]
enum AgentAction {
    /// List registered peer agents
    List,
    /// Send a message to one peer agent
    Ping { agent: String, message: String },
    /// Send a message to every other peer agent
    Broadcast { message: String },
}

fn print_profiles() -> Result<()> {
    let profiles = list_profiles()?;
    if profiles.is_empty() {
        println!("(no profiles; create one with: pi-rewrite profile create <name> --from-base)");
        return Ok(());
    }
    for p in &profiles {
        let marker = if p.is_default { " *" } else { "  " };
        println!("{marker}{}    {}", p.name, p.path.display());
    }
    Ok(())
}

fn handle_profile(action: ProfileAction) -> Result<()> {
    match action {
        ProfileAction::List => print_profiles(),
        ProfileAction::Show => {
            let name = default_profile_name()?;
            match name {
                Some(n) => {
                    let profiles = list_profiles()?;
                    if let Some(p) = profiles.into_iter().find(|p| p.name == n) {
                        println!("{}  {}", p.name, p.path.display());
                    } else {
                        println!("{n}  (profile dir missing)");
                    }
                }
                None => println!("(no default profile set)"),
            }
            Ok(())
        }
        ProfileAction::Use { name } | ProfileAction::SetDefault { name } => {
            set_default_profile(&name)?;
            println!("default profile: {name}");
            Ok(())
        }
        ProfileAction::Create { name, from_base, from } => {
            let path = create_profile(&name, from_base, from.as_deref())?;
            println!("created profile {name} at {}", path.display());
            Ok(())
        }
        ProfileAction::Delete { name } => {
            delete_profile(&name)?;
            println!("deleted profile {name}");
            Ok(())
        }
    }
}

fn handle_multikey(action: MultikeyAction, config_dir: &PathBuf) -> Result<()> {
    let multikey = MultikeyConfig::load_from(config_dir)?;
    match action {
        MultikeyAction::Status => {
            for pool in &multikey.pools {
                println!("▶ {} — {}  ({})", pool.id, pool.name.clone().unwrap_or_default(), pool.base_url);
                let kp = multikey::KeyPool::new(pool.clone());
                let rows = futures::executor::block_on(kp.status_rows());
                for row in rows {
                    println!("{row}");
                }
                println!();
            }
            Ok(())
        }
        MultikeyAction::List => {
            for pool in &multikey.pools {
                println!("{}  {}  ({} keys, {} models)", pool.id, pool.base_url, pool.keys.len(), pool.models.len());
                for m in &pool.models {
                    println!("    {}  ctx={}  max={}", m.id, m.context_window.unwrap_or(0), m.max_tokens.unwrap_or(0));
                }
            }
            Ok(())
        }
    }
}

fn handle_agent(action: AgentAction, config_dir: &PathBuf) -> Result<()> {
    let reg = AgentRegistry::new(config_dir);
    match action {
        AgentAction::List => {
            let agents = reg.list()?;
            if agents.is_empty() {
                println!("(no agents registered)");
            } else {
                for (name, model) in agents {
                    println!("{name}\t{model}");
                }
            }
            Ok(())
        }
        AgentAction::Ping { agent, message } => {
            reg.deliver("cli", &agent, &message)?;
            println!("pinged {agent}");
            Ok(())
        }
        AgentAction::Broadcast { message } => {
            let n = reg.broadcast("cli", &message)?;
            println!("broadcast to {n} agent(s)");
            Ok(())
        }
    }
}

/// Build the runtime state shared by one-shot, json, and rpc modes.
fn build_state(cli: &Cli) -> Result<(RpcState, AgentOptions)> {
    let config_dir = resolve_config_dir(cli.profile.as_deref())?;
    let settings = Settings::load_from(&config_dir)?;
    let multikey = MultikeyConfig::load_from(&config_dir)?;

    let opts = AgentOptions {
        system_prompt: cli.system_prompt.clone(),
        append_system_prompt: cli.append_system_prompt.clone(),
        provider: cli.provider.clone(),
        model: cli.model.clone(),
        agent_name: cli.agent.clone(),
        max_turns: cli.max_turns,
        quiet: cli.quiet,
    };

    let (pool, model) = agent::resolve_provider(&multikey, &settings, &opts)?;
    let pool = pool.clone();
    let system_prompt = agent::build_system_prompt(&settings, &opts);

    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    let mut tools = ToolRegistry::new(cwd.clone());

    let registry = Arc::new(AgentRegistry::new(&config_dir));
    let agent_name = cli
        .agent
        .clone()
        .or_else(|| cli.profile.clone())
        .unwrap_or_else(|| "main".to_string());
    tools.agents = Some(registry.clone());
    tools.self_agent = Some(agent_name.clone());

    let _ = registry.register(
        &agent_name,
        std::process::id(),
        None,
        &cwd,
        &format!("{}/{}", pool.id, model.id),
    );

    let agent = Agent::new(pool.clone(), model.clone(), tools, Some(registry.clone()), opts.clone());

    let session = Session::open(&config_dir.join("sessions"), &cwd)?;

    let state = RpcState {
        agent,
        settings,
        multikey,
        pool: pool.clone(),
        model: model.clone(),
        registry: Some(registry),
        config_dir,
        session,
        messages: Vec::new(),
        system_prompt,
        aborted: false,
    };

    Ok((state, opts))
}

#[tokio::main]
async fn main() -> Result<ExitCode> {
    let cli = Cli::parse();

    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "warn");
    }
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    // Subcommands that do not need an agent runtime.
    if let Some(command) = &cli.command {
        match command {
            Commands::Profile { action } => {
                handle_profile(action.clone())?;
                return Ok(ExitCode::SUCCESS);
            }
            Commands::Multikey { action } => {
                let config_dir = resolve_config_dir(cli.profile.as_deref())?;
                handle_multikey(action.clone(), &config_dir)?;
                return Ok(ExitCode::SUCCESS);
            }
            Commands::Agent { action } => {
                let config_dir = resolve_config_dir(cli.profile.as_deref())?;
                handle_agent(action.clone(), &config_dir)?;
                return Ok(ExitCode::SUCCESS);
            }
        }
    }

    let (state, opts) = build_state(&cli)?;

    match cli.mode.as_str() {
        "rpc" => {
            run_rpc(state, opts).await?;
            Ok(ExitCode::SUCCESS)
        }
        "json" => {
            let prompt = cli.prompt.clone().context("json mode needs a prompt argument")?;
            one_shot(state, prompt, true).await
        }
        _ => {
            let prompt = cli.prompt.clone().context("prompt mode needs a prompt argument")?;
            one_shot(state, prompt, false).await
        }
    }
}

async fn one_shot(mut state: RpcState, prompt: String, json: bool) -> Result<ExitCode> {
    state.aborted = false;
    let agent = state.agent.clone();
    let system_prompt = state.system_prompt.clone();

    if json {
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "type": "session",
                "version": 3,
                "id": state.session.id,
                "cwd": std::env::current_dir().ok().map(|p| p.to_string_lossy().to_string())
            }))?
        );
    }

    let result = agent.run(&prompt, &system_prompt, &mut state.session).await;

    match result {
        Ok(_) => Ok(ExitCode::SUCCESS),
        Err(e) => {
            eprintln!("[pi-rewrite] error: {e}");
            Ok(ExitCode::from(1))
        }
    }
}
