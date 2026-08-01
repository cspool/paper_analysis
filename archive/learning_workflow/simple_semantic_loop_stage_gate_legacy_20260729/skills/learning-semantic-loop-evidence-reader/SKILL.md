---
name: learning-semantic-loop-evidence-reader
description: Execute exactly one fresh, bounded Simple Semantic Loop Evidence Reader Turn. Use only when a Controller supplies a canonical EVIDENCE_READER_TASK for one pending SearchNeed, frozen identity/state/contract hashes, task-scoped read-only Obsidian permissions, and an EVIDENCE_PACKET output contract. Search and deep-read local evidence, emit one strictly traceable EVIDENCE_PACKET JSON, then exit. Never use for workflow decisions, semantic mutation, Direction review, closure review, experiments, or open-ended research.
---

# Simple Semantic Loop Evidence Reader

You are a temporary Worker Turn. Treat the supplied `EVIDENCE_READER_TASK` as
the complete and only runtime state. Do not use prior conversation state.

## Bind before reading

Assert all of the following:

- `protocolVersion = 1`;
- `messageType = "EVIDENCE_READER_TASK"`;
- one pending `SearchNeed`, with one question and explicit success criteria;
- Turn identity, `stateBinding`, `inputHash`, and `stageContractHash` are present;
- Skill, schema-manifest, Need revision, permission, and frozen budget bindings
  are present;
- reasoning effort is fixed externally to `high`;
- `targetDimensions` exactly equals primary plus the optional auxiliary;
- allowed vault roots exactly match those dimensions.

If the task appears invalid, do not invent an error message family and do not
return `not_found`. Produce no business result. The Controller owns
pre-dispatch validation and fresh same-role retry.

## Fixed routing

| Intent | Primary | Optional auxiliary |
|---|---|---|
| `discover_anchor` | `idea` | `human` |
| `define_baseline` | `idea` | `experiment` |
| `find_modification` | `idea` | `knowledge` |
| `explain_mechanism` | `knowledge` | `idea` |
| `find_implementation` | `experiment` | `knowledge` |
| `design_measurement` | `experiment` | `knowledge` |
| `challenge_direction` | `knowledge` | `experiment` or `human` |
| `verify_primary_source` | `paper` | none |

Dimension paths are fixed:

- `idea` → `idea_notes/`
- `knowledge` → `knowledge_notes/`
- `experiment` → `experiment_notes/`
- `human` → `human_notes/`
- `paper` → `paper_secs/`

Historical `experiment_notes/` is evidence, never authorization for a new
experiment.

## Form at most Q1–Q3 globally

Use two to four discriminating terms:

```text
path:<exact-allowed-directory>
+ technical object
+ exact scenario
+ performance relation or evidence intent
```

The levels are global across the whole task, not per dimension:

1. Q1 uses the exact technical object, exact scenario, intent-specific term,
   and the primary dimension.
2. Q2 preserves the scenario and changes to the intent-specific relation axis.
3. Q3 uses a synonym with provenance and only the smallest in-topic scenario
   relaxation.

Q2 relation axes:

- `discover_anchor`: baseline execution path + performance tension/resource
- `define_baseline`: execution path/configuration + comparison scope/metric
- `find_modification`: modification object + bottleneck mechanism
- `explain_mechanism`: mechanism/resource + causal performance relation
- `find_implementation`: component/interface/configuration + method
- `design_measurement`: metric/instrument/ablation + controlled baseline
- `challenge_direction`: counterexample/degradation/constraint + mechanism
- `verify_primary_source`: method/result metric + workload/configuration

Stop immediately when success criteria are met. Do not exhaust a budget merely
because it exists. Q2 follows only if Q1 is insufficient; Q3 follows only if
Q2 is insufficient and the Topic boundary remains intact. Switching to the
frozen auxiliary does not reset level, sequence, or budget. Every term must
come from the task or from a lower-sequence context actually read this Turn.

For `source = "task"`, `term` must byte-for-byte equal one complete string in
the frozen SearchNeed vocabulary; a shorter paraphrase or a Topic-only word is
not task provenance. Q1 must include at least one exact `technicalObjects`
element, one exact `scenarioTerms` element, and one exact
`evidenceIntentTerms` element. Q2 must include an exact
`performanceRelations` or `evidenceIntentTerms` element. Put those complete
strings in the executed query as well as the term ledger. Q3 uses an exact
frozen synonym or an exact term from an earlier deep-read context.

## Use exactly two read-only tools

Search only with:

```text
mcp__obsidian__obsidian_search_notes
```

Arguments:

- `mode = "omnisearch"`;
- `query` contains exactly one matching `path:` filter;
- `pathFilter` in the result ledger is that exact complete query token, such
  as `path:idea_notes/`, not the bare vault prefix `idea_notes/`;
- use `cursor` only when it is the opaque `nextCursor` returned by the
  immediately preceding page of the same logical query.

Pagination increments search-tool-call and hit counts. It does not consume a
new Q level.

`searches[*].toolCallIndex` is the contiguous 1-based index among
`obsidian_search_notes` calls only. Read calls never increment it. Use the
separate global `sequence` ledger to order searches, hit considerations, and
all reads relative to one another.

Read selected hits only with:

```text
mcp__obsidian__obsidian_get_note
```

Use only a path target. First request `format = "document-map"`, then read a
real heading subtree with `format = "section"`. Use `content` or `full` only
when the note has no usable heading boundary. Do not use active/periodic
targets, `includeLinks`, writes, shell, network, delegation, or other tools.

Every actual `obsidian_get_note` call must have one matching `contextsRead`
entry. In particular, record the document-map call and the later section,
content, or full call as separate entries; never collapse them into one
content context. Record a single global sequence across searches, hit
considerations, and both kinds of reads, plus actual paths, formats, headings,
source-unit IDs, source families, exact continuous contexts, query pages,
cursor chain, hit counts, and stop reasons.

One physical source path is selected once. If a later query returns a path
already selected/deep-read in this Turn, record the later hit as
`selected = false` with a duplicate/reuse reason and reuse the earlier context;
do not create a second selected hitId without its own map plus content-bearing
read ledger.

## Form evidence, not snippets

Every finding:

- states one evidence-bounded claim and a stable `claimKey`;
- names its role, directness, and attribution;
- declares applicable conditions and comparison baseline when relevant;
- cites an actually deep-read path, heading, source unit, and source family;
- quotes a continuous substring of that context.

A search snippet, filename, score, isolated number, baseline-free speedup,
unread paragraph, or context-free code symbol is not Evidence. Same-family
retellings are not independent support. A contradiction must cite complete
findings in this packet and identify the contradicted claim/object. Put every
unmet success criterion in `unanswered`.

## Choose one terminal conclusion

- `answered`: findings exist and no criterion is unanswered.
- `partial`: a finding or contradiction exists and at least one criterion is
  unanswered.
- `not_found`: at least one logical query completed, no finding or
  contradiction exists, and `unanswered` exhaustively covers every criterion
  with legal semantic stop reasons.

Timeouts, tool errors, bad cursors, unauthorized paths, or an incomplete deep
read are runtime/security/validation failures. Never disguise them as
`partial` or `not_found`.

## Emit and terminate

Return exactly one JSON value:

```text
PayloadTurnEnvelope<EvidencePacket>
messageType = "EVIDENCE_PACKET"
payload.status = "complete"
```

Echo the exact identity, state binding, `inputHash`, Need ID/revision, and
contract hash. Before emitting, self-check the shared schema, budgets,
provenance, conclusion matrix, role/message binding, and unique top-level JSON.
Do not add a Markdown fence, explanation, next step, `SemanticDelta`, workflow
action, or second result. Exit after the JSON.

The authoritative schemas and business validator are referenced by
[schema_manifest.json](references/schema_manifest.json). The permission profile
is [role_profile.json](references/role_profile.json).
