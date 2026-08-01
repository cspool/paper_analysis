# Closure finding registry

The Controller validator, not free-form interpretation, owns this exact
mapping.

| Check | Codes | Type | Recovery classification |
|---|---|---|---|
| `stopProofRevisionCurrent` | `stale_stop_proof_revision` | `state_inconsistency` | `REPAIR_STATE` |
| `stopProofMatchesCanonical` | `stop_proof_canonical_mismatch` | `state_inconsistency` | `REPAIR_STATE` |
| `mechanicalPreflightPassed` | `mechanical_preflight_failed` | `state_inconsistency` | `REPAIR_STATE` |
| `topicScopePreserved` | `topic_scope_silently_narrowed` | `knowledge_gap` | `REOPEN_FRONTIER` |
| `noKnowledgeAnswerableCriticalNeed` | `knowledge_answerable_open_need` | `knowledge_gap` | `REOPEN_FRONTIER` |
| `allAnchorsClosed` | `anchor_not_closed`, `anchor_missing_saturation_reason`, `anchor_missing_status_reason` | `knowledge_gap` | `REOPEN_FRONTIER` |
| `allDirectionsTerminal` | `direction_nonterminal`, `direction_missing_terminal_reason` | `knowledge_gap` | `REOPEN_FRONTIER` |
| `lastTopicExpansionNoDelta` | `last_topic_expansion_missing`, `last_topic_expansion_not_quiet` | `knowledge_gap` | `REOPEN_FRONTIER` |
| `noUnconsumedOrUncommittedWork` | `pending_task`, `in_flight_task`, `pending_output_retry`, `unconsumed_result`, `uncommitted_delta`, `unresolved_validation_failure`, `failed_task` | `state_inconsistency` | `REPAIR_STATE` |
| `criticalContradictionsReviewed` | `unreviewed_critical_contradiction` | `knowledge_gap` | `REOPEN_FRONTIER` |
| `experimentHandoffsComplete` | `experiment_handoff_missing`, `experiment_handoff_invalid` | `incomplete_handoff` | `COMPLETE_HANDOFF` |
| `runtimeEligibleForCompletion` | `runtime_budget_exhausted`, `runtime_failed_or_paused` | `runtime_pause` | `RESUME_RUNTIME` |
| `finalOutputTraceable` | `final_output_missing_field`, `final_output_untraceable` | `state_inconsistency` | `REPAIR_STATE` |

These recovery values classify the finding. They are not Agent roles, Stage
types, or permission to mutate state.

