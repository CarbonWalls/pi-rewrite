# pi-rewrite

A fast, native rewrite of [pi](https://github.com/earendil-works/pi) (the `@earendil-works/pi-coding-agent` CLI). Same config, same provider model, same RPC protocol — but with the features the user actually stacks as plugins folded in as first-class native code, and no Node.js runtime to boot.

## What it is

`pi-rewrite` is a single Rust binary that reads your existing `~/.pi` configuration and talks to your existing multikey pools. It is a drop-in speedup layer, not a fork of pi's source: the config formats, the profile layout, the collaborating-agents registry, and the stdio RPC protocol are all upstream-compatible, so it works with whatever pi version you have installed.

Measured on the reference machine (aarch64, release build), against the installed upstream pi 1.1.0:

| path | pi-rewrite | upstream | speedup |
|---|---|---|---|
| process startup (`--help` / `--version`) | ~21 ms | ~900 ms | ~40x |
| profile + multikey config load (`profile list`) | ~26 ms | ~1.7 s (`ppi list`) | ~65x |
| one-shot prompt, same pool and model | ~6.7 s | ~52 s | ~8x |

Startup and config-load wins come from being a native binary with no plugin loader, no provider re-registration, and no telemetry handshake. The one-shot win comes from skipping plugin/extension resolution and doing a single streaming HTTP call with no retry scaffolding in the hot path.

## Native features (no plugins needed)

- **multikey** — every pool in `~/.pi/agent/multikey.json` is a provider. Requests lease the least-loaded, least-recently-used key; a 429 cools that key down (`cooldownMs`, or `retry-after`) and rotates to the next key instantly; a 401/403 marks the key invalid for `invalidKeyCooldownMs`. Concurrent subagents each hold their own lease, so parallel work spreads across keys automatically. `pi-rewrite multikey status` shows live per-key state.
- **system prompt** — `--system-prompt` replaces the base prompt, `--append-system-prompt <file>` appends to it (the file is read when given a path, used as literal text otherwise). `settings.json` `systemPrompt` / `appendSystemPrompt` are honored per profile.
- **peer subagents** — every model is an agent, and no agent is an orchestrator. Agents self-register in `<config>/collaborating-agents/registry/` (upstream layout) and ping each other through `<config>/collaborating-agents/inbox/<name>/`. Any agent can message any other at any time. The model gets `agents_list`, `agents_ping`, and `agents_broadcast` tools; the CLI mirrors them as `pi-rewrite agent list|ping|broadcast`.
- **ppi (profiles)** — `pi-rewrite profile list|show|use|set-default|create|delete` manages `~/.pi/profiles`, writes the same `profiles/default.json` pointer, and resolves the active profile through the same precedence as `ppi`: `--profile` flag, then `PI_CODING_AGENT_DIR`, then `default.json`, then `~/.pi/agent`.

## Requirements

- Rust 1.85+ (edition 2021) and cargo. On Debian/Ubuntu: `apt install rustc cargo`.

## Build

```sh
cargo build --release
# binary: target/release/pi-rewrite
```

## Install

```sh
cargo install --path . --root ~/.local
# or copy the binary somewhere on PATH and, if you want the name, symlink it as `pi`
ln -s target/release/pi-rewrite ~/.local/bin/pi
```

`pi-rewrite` never modifies your upstream pi install. Run it alongside `pi`; both read the same config.

## Usage

One-shot prompt (uses `settings.json` `defaultProvider` / `defaultModel`, or overrides):

```sh
pi-rewrite --provider 7 --model glm-5.3 "Summarize this repo"
```

Override the system prompt (file or literal):

```sh
pi-rewrite --append-system-prompt ~/.pi/wye.txt "Fix the failing tests"
```

JSON event stream (same record shapes as pi's `--mode json`):

```sh
pi-rewrite --mode json "List files" | jq -c 'select(.type == "message_end")'
```

RPC backend (same JSONL framing as pi's `--mode rpc`; drives it from any language):

```sh
pi-rewrite --mode rpc --provider 7 --model glm-5.3 <<'EOF'
{"id":"r1","type":"get_state"}
{"id":"r2","type":"prompt","message":"Reply with exactly: RPC_OK"}
{"id":"r3","type":"get_available_models"}
EOF
```

Profiles:

```sh
pi-rewrite profile list
pi-rewrite profile use general
pi-rewrite --profile roblox "Check the place file"
pi-rewrite profile create experiment --from-base
```

Multikey status:

```sh
pi-rewrite multikey status
pi-rewrite multikey list
```

Peer agents:

```sh
pi-rewrite agent list
pi-rewrite agent ping RapidNova "can you look at src/main.rs?"
pi-rewrite agent broadcast "build is broken on main"
```

Inside a session the model reaches the same mesh through tools: `agents_list`, `agents_ping`, `agents_broadcast`.

## Config resolution

The active config directory is chosen in this order:

1. `--profile <name>` — `~/.pi/profiles/<name>`
2. `PI_CODING_AGENT_DIR` — set by `ppi use`
3. `~/.pi/profiles/default.json` — `{"default": "<name>"}`
4. `~/.pi/agent` — stock pi config

Model selection, in order: `--provider` + `--model`, then `settings.json` `defaultProvider` + `defaultModel`, then the first pool's first model. `--model` accepts `<pool>/<model>`.

## Tools

`read`, `write`, `edit`, `bash`, `glob`, `grep`, plus the three `agents_*` peer tools. `bash` runs through `bash -c` in the working directory with a 10-minute default timeout.

## Sessions

Append-only JSONL under `<config>/sessions/<cwd>/`, one entry per message and tool result. Each entry carries a UUID `id` and its parent's id.

## What is intentionally not reimplemented

The TUI, MCP server support, codemode, and the extension/skill/package loader. `pi-rewrite` targets the headless paths — one-shot, JSON, RPC — where the speed matters, and leaves interactive pi sessions to upstream pi. The RPC protocol it speaks is the same one pi's own clients use, so a TUI built against pi's RPC works against this binary unchanged.

## License

MIT.
