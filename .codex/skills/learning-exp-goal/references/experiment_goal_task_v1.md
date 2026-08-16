# EXP Goal Task Contract

`experiment_goal_task.json` is a Script-owned immutable input. Its fields are:

| Field | Meaning |
|---|---|
| `experimentId` | Stable identifier for this experimental Goal. |
| `goalRef` | Parent Learning Workflow Goal. |
| `sourceDecisionTurnRef` | Audit reference to the Decision that selected the branch; do not read it as an instruction. |
| `sourceDecisionContextRef` | Frozen parent context audit reference; use named task inputs instead of mining Controller state. |
| `anchorWork` | Required immutable Anchor Work Result reference. |
| `directionWork` | Optional immutable Direction Work Result reference. It is null for Anchor baseline/headroom diagnosis. |
| `experimentObjective` | Required bounded empirical question copied from Decision guidance. |
| `workspaceRef` | Durable directory for code, environments, logs, raw data, analysis, and attempt history. |

The task deliberately does not prescribe stages, a simulator, repository,
hardware, or fixed implementation plan. Choose and revise those within the
Goal according to actual evidence. It also does not ask for an Agent-authored
JSON result; the Controller stores the final natural-language conclusion and
provider Goal status in its own record.
