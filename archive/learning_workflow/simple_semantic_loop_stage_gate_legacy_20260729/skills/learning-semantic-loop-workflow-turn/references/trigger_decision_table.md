# Registered trigger decision table

The task permission envelope may narrow these sets; it may never broaden them.

| Trigger | Question | Registered actions |
|---|---|---|
| `INITIALIZE_TOPIC` | How can the user topic become a non-narrowed TopicFrame? | `RUN_STAGE` |
| `COMMITTED_RESULT_REQUIRES_INTEGRATION` | Does one committed Evidence/Review result imply one legal semantic mutation or independent review? | `RUN_STAGE`, `REQUEST_EVALUATION` |
| `FRONTIER_SELECTION_REQUIRED` | Which focus has one bounded executable gap? | `RUN_STAGE`, `REQUEST_EVALUATION`, `PROPOSE_COMPLETE` |
| `MULTIPLE_NON_EQUIVALENT_STAGES_RUNNABLE` | Which non-equivalent branch should run first? | `RUN_STAGE`, `REPLAN` |
| `GATE_FAILED_WITHOUT_RECOVERY_RULE` | Retry the frozen contract, change route, evaluate, ask, or block? | `RETRY_STAGE`, `REPLAN`, `REQUEST_EVALUATION`, `ASK_USER`, `REPORT_BLOCKED` |
| `PLAN_EXHAUSTED_OBJECTIVE_OPEN` | What Stage is missing, or do current facts support a candidate? | `RUN_STAGE`, `REQUEST_EVALUATION`, `REPLAN`, `PROPOSE_COMPLETE`, `ASK_USER`, `REPORT_BLOCKED` |
| `EVIDENCE_CONTRADICTION` | Which object is affected and should it be challenged, revised, or reviewed? | `RUN_STAGE`, `REQUEST_EVALUATION`, `REPLAN` |
| `NO_PROGRESS_THRESHOLD_REACHED` | Is another legal route available? | `REPLAN`, `ASK_USER`, `REPORT_BLOCKED`, `PROPOSE_PAUSE` |
| `CLOSURE_REJECTED` | Which exact rejected scope must reopen? | `RUN_STAGE`, `REPLAN`, `ASK_USER` |
| `NO_RUNNABLE_STAGE` | Is plan topology missing, external/user input needed, the run blocked/paused, or a closure candidate supported? | `RUN_STAGE`, `REQUEST_EVALUATION`, `REPLAN`, `ASK_USER`, `REPORT_BLOCKED`, `PROPOSE_PAUSE`, `PROPOSE_COMPLETE` |
| `USER_DECISION_REQUIRED` | What is the smallest missing user choice or authorization? | `ASK_USER` |

Output structure, binding, and pre-Gate semantic/authority errors are not
semantic triggers. The Controller may create at most one fresh same-role
replacement Turn for the immutable logical task and inject a bounded,
hash-bound validation-error packet. A valid result that fails the already
frozen effective Gate is instead a workflow event and may reach
`GATE_FAILED_WITHOUT_RECOVERY_RULE`; it is never repaired by rewriting the
same output.
