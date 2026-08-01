# DeepSeek Provider Through Claude CLI

## Runtime contract

Use the existing validated invocation path, not a new SDK:

```text
claude CLI 2.x
--model deepseek-v4-flash[1m]
Anthropic-compatible endpoint from ANTHROPIC_BASE_URL
credential from ANTHROPIC_AUTH_TOKEN
```

The provider is locally configured when:

- `claude` resolves on `PATH`;
- its version command succeeds;
- `ANTHROPIC_BASE_URL` and `ANTHROPIC_AUTH_TOKEN` exist;
- a model name is configured.

Never print environment values. `doctor` reports local binary/environment
presence only and does not spend tokens on a live endpoint probe. Endpoint/model
reachability is confirmed by the first real call and its raw log.

## One-shot discovery call

Use a fresh UUID and no tools:

```bash
claude -p "<prompt>" \
  --model "deepseek-v4-flash[1m]" \
  --session-id "<uuid>" \
  --output-format "stream-json" \
  --verbose \
  --json-schema '<schema>' \
  --tools "" \
  --permission-mode "bypassPermissions" \
  --add-dir "/data3/paper_analysis" \
  --max-budget-usd "<cap>"
```

The outer script supplies bounded evidence snippets in the prompt. The model does not search files and does not write artifacts.

## Loop call

Keep a separate resumable session ID for:

- each bounded Anchor-curation claim batch;
- each Direction construction loop.

Direction Judge/Evidence calls default to stateless turns reconstructed from
the complete canonical Bundle and local Q&A ledger. This prevents repeated
Bundle copies from accumulating in provider history. The optional
`--review-session-mode resume` keeps separate Judge and Evidence sessions when
explicitly selected.

First turn uses `--session-id`; later turns use:

```bash
claude -p "<next payload>" \
  --resume "<session-id>" \
  --model "deepseek-v4-flash[1m]" \
  --output-format "stream-json" \
  --verbose \
  --json-schema '<schema>' \
  --tools "" \
  --permission-mode "bypassPermissions" \
  --add-dir "/data3/paper_analysis" \
  --max-budget-usd "<cap>"
```

Persist the session ID and last accepted action in `state.json`. A resumed CLI session is context continuity, not canonical state; the script still includes a compact checkpoint anchor in every payload.

## Output parsing

For each JSON line:

- retain raw lines in the model-call log;
- collect visible assistant text;
- retain the terminal `result` object and telemetry;
- accept a structured object from known CLI fields or parse the terminal result text;
- validate it locally against the action schema and domain invariants.

Ignore partial text deltas when a complete terminal result is available. Never accept a `result` with `is_error=true`.

## Structure repair

The CLI supports `--json-schema`, but model/provider output still receives local validation.

If the terminal result is not a valid object:

1. save the invalid raw output;
2. resume the same session once;
3. send a repair prompt containing the validation errors and schema name;
4. forbid retrieval, new reasoning, or state reset;
5. accept only a complete corrected object.

A second failure becomes `failed_retriable` with checkpoint retained.

## Timeout and process cleanup

- Start each call in its own process group.
- On timeout, terminate the group, then kill if it does not exit.
- Track active processes and clean them on SIGINT/SIGTERM.
- Do not mark the task done merely because the process exited with code 0.

## Concurrency

Parallelize independent discovery calls with a bounded worker pool. Do not run two turns concurrently on the same session ID. Curator and each review conversation are sequential.

## Logs and budgets

Record:

- role, task/direction, attempt, session ID;
- model name;
- start/end time and exit status;
- terminal telemetry, token usage, and cost if present;
- prompt hash and schema hash;
- raw log and parsed output path.

Enforce a per-call timeout and `--max-budget-usd`. The configured cap is permission to stop spending, not a guarantee that every compatible proxy reports identical billing semantics.
