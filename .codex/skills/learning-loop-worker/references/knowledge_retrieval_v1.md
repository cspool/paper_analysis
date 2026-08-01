# Local Knowledge Retrieval

Use the local Obsidian vault as a multidimensional HPC research knowledge base.

| Dimension | Vault path | Best suited to |
|---|---|---|
| idea | `idea_notes/` | scenarios, methods, baselines, candidate opportunities |
| experiment | `experiment_notes/` | implementations, configurations, metrics, tools, measured constraints |
| knowledge | `knowledge_notes/` | mechanisms, terminology, architecture/compiler/runtime relations |
| human | `human_notes/` | human syntheses, caveats, judgments and open questions |
| paper | `paper_secs/` | primary-paper verification when a precise claim requires it |

Query trigger:

- query when the current Task lacks traceable support for a required field;
- query when a supplied Reviewer `queryGap` names an unresolved local question;
- stop when the Task can be completed honestly, rather than exhausting calls.

Procedure:

1. Split the need into technical object, scenario/regime, baseline,
   performance relation, implementation, and metric.
2. Choose the dimension matching the missing information.
3. Use `mcp__obsidian__obsidian_search_notes` with `mode="omnisearch"` and a
   path-qualified query such as `path:idea_notes/`.
4. If a precise query has no useful hit, progressively shorten it while
   retaining the topic-defining object and performance relation.
5. Read selected notes with `mcp__obsidian__obsidian_get_note`; do not cite a
   search snippet as evidence.
6. Record the actual note path (and heading when available) in `sourceRef` and
   state exactly what it supports.

Historical experiment notes are evidence, not authorization to run an
experiment. Never invent a source, quote, metric, implementation path, or
speedup.
