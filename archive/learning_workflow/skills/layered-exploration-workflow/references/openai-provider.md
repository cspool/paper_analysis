# OpenAI Feasibility Note (Not Used by This Workflow)

The default implementation uses [deepseek-provider.md](deepseek-provider.md),
matching the validated legacy invocation path; an optional authenticated CLI
path is documented in [codex-provider.md](codex-provider.md). This file only
records whether the current environment could support a separate direct OpenAI
API adapter.

## Readiness

The current Python environment can support an OpenAI Python SDK/Responses API adapter. Runtime readiness would require:

- importable `openai` Python package;
- `OPENAI_API_KEY` in the process environment;
- network access to the OpenAI API;
- a model available to the caller's project.

The current environment has the Python SDK but no `OPENAI_API_KEY`; therefore it is not currently authenticated. The new orchestrator does not implement or depend on this adapter.

## API shape

Use:

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model=model,
    instructions=role_instructions,
    input=runtime_payload,
    text={
        "format": {
            "type": "json_schema",
            "name": schema_name,
            "strict": True,
            "schema": schema,
        }
    },
    store=False,
)
```

Read structured JSON from `response.output_text` and validate it again locally. Structured Outputs reduce syntax drift; they do not validate source truth or graph semantics.

## Conversation state

Default:

- `store=false`;
- save turns locally in run artifacts;
- send a bounded, compact history with each loop turn.

This keeps the run resumable without relying on a remote response retention window and is appropriate for local knowledge-base content.

Optional remote chaining:

- `store=true`;
- pass `previous_response_id`;
- save the response ID in state;
- understand that prior input tokens remain billable and response objects are normally retained remotely for a limited period.

Remote chaining must be explicit, not the default.

## Retries

Retry only transient transport/rate/server failures with bounded exponential backoff. Do not retry:

- missing API key;
- invalid schema;
- source validation failure;
- deterministic graph invariant failure;
- model refusal without first recording it.

A separate one-time structure repair may be used when a provider returns parseable text that violates the expected response contract.

## Privacy and logs

- Do not log environment secrets or authorization headers.
- Log provider, model, response ID, usage metadata, schema name, input hash, and output artifact path.
- Raw prompts may contain local evidence. Keep them inside the user-selected run directory.
- Allow a future redaction policy, but do not silently remove evidence required for reproducibility.

## Official references

- Quickstart and API key environment behavior: <https://developers.openai.com/api/docs/quickstart>
- Responses API model guidance: <https://developers.openai.com/api/docs/guides/latest-model>
- Structured Outputs: <https://developers.openai.com/api/docs/guides/structured-outputs>
- Conversation state and `previous_response_id`: <https://developers.openai.com/api/docs/guides/conversation-state>
