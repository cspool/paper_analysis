# `decision-line-protocol-v1`

Return one decision line and at most one single-line guidance line:

```text
decision = RUN_WORKER
guidance = 关注当前结论尚未覆盖的性能机制，不改变 Script 绑定。
```

Role rules:

- `decision` is exactly one literal injected under `[ALLOWED_DECISIONS]`.
- Decision should provide `guidance` for retry branches so the retried Agent
  receives the semantic error and bounded correction. The Script treats
  guidance as optional opaque text and does not inspect its meaning.
- `guidance` is non-authoritative and cannot select the Task action, object
  kind, target, schema, or state transition.
- Do not return JSON, Markdown fences, prose outside these fields, multiple
  decisions, or control syntax inside guidance.
