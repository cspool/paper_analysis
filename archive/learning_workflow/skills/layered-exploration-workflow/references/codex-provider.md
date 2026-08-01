# Codex CLI Provider

## Position in the architecture

`codex-cli` is an optional sibling adapter:

```text
layered_exploration_orchestrator.py
├── deepseek-cli  → claude CLI → DeepSeek endpoint
└── codex-cli     → codex exec → local Codex authentication
```

Do not ask a Claude/DeepSeek Agent to shell out to Codex. The deterministic outer orchestrator chooses the provider for each role.

## Local capabilities

Required Codex CLI features:

- `codex exec` for non-interactive turns;
- `--json` for JSONL events;
- `--output-schema <file>` for final structured output;
- `codex exec resume <session-id>` for loop continuity;
- `--sandbox read-only`;
- `--ask-for-approval never`;
- `--cd <isolated-dir>`;
- existing user execution policy remains in force.

Authentication can be an existing ChatGPT login, API key login, or access-token login. `doctor` reports the login mode without exposing credentials.

## First turn

Write the action schema to a run-local file, then invoke:

```bash
codex \
  --ask-for-approval "never" \
  --sandbox "read-only" \
  --cd "<work-dir>/provider_sandbox/<conversation-key>" \
  exec \
  --json \
  --output-schema "<schema-file>" \
  --skip-git-repo-check \
  --model "<configured-model>" \
  -
```

Pass the prompt on stdin. Parse the session/thread ID from JSONL and save it in workflow state.

## Later loop turns

This resume form is used only for roles configured to retain provider history.
Judge/Evidence review turns default to stateless reconstruction from local
Bundle + Q&A state.

```bash
codex \
  --ask-for-approval "never" \
  --sandbox "read-only" \
  --cd "<work-dir>/provider_sandbox/<conversation-key>" \
  exec resume "<session-id>" \
  --json \
  --output-schema "<schema-file>" \
  --skip-git-repo-check \
  --model "<configured-model>" \
  -
```

Include a compact canonical checkpoint in every prompt. Codex session history is a convenience; it is not the source of truth.

## Tool-isolation limitation

Codex CLI does not expose a direct equivalent of Claude CLI `--tools ""`.

For this workflow:

1. create a dedicated empty provider directory;
2. use a read-only sandbox and approval policy `never`;
3. do not pass project paths except evidence text already selected by the orchestrator;
4. instruct the role not to call tools;
5. inspect JSONL events;
6. reject the turn if a tool, shell, file, MCP, web, or mutation event occurs.

This is strict result admission, not a proof that the model was technically unable to attempt a read. Use `deepseek-cli` for roles requiring a hard no-tools switch unless a future Codex CLI version exposes one.

## Output parsing

Retain raw JSONL. Extract:

- thread/session ID;
- final assistant message;
- usage and terminal status;
- any tool-call event;
- error events.

The final message must parse under the supplied output schema and then pass the same domain validators as every other provider.

## Resume and provider mixing

- Never resume a DeepSeek session with Codex or vice versa.
- Store `{provider, model, session_id}` together.
- Provider selection can be global or role-specific.
- Changing provider mid-role starts a new session and requires the full local checkpoint/history.
- Results from different providers are not automatically consensus. If both review the same object, preserve two reviews or run an explicit adjudication step.

## Suggested use

- DeepSeek default: high-volume discovery and the validated expert-review behavior.
- Codex optional: independent second-pass critic, schema-sensitive curation, or comparative evaluation.
- Do not assume the more expensive provider is more accurate; compare against source validation and gold review examples.

Select Codex for every role:

```bash
python3 scripts/layered_exploration_orchestrator.py init \
  --topic "<topic>" \
  --work-dir "<run-dir>" \
  --provider codex-cli
```

Or retain DeepSeek discovery/curation and use Codex only for Direction review:

```bash
python3 scripts/layered_exploration_orchestrator.py init \
  --topic "<topic>" \
  --work-dir "<run-dir>" \
  --provider deepseek-cli \
  --review-provider codex-cli
```
