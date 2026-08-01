# Source Provenance

This skill is a new implementation. Its source material remains read-only.

Human-readable extraction:

- `draft/learning_workflow_reusable_knowledge_extraction.md`

Rule-by-rule source file and line index:

- `draft/learning_workflow_source_provenance.md`

Current workflow design:

- `draft/learning_workflow_optimization_discussion.md`

Primary legacy sources:

- `scripts/learning_scheduler.ts`
- `scripts/idea_review_orchestrator.ts`
- `.claude/skills/learning-experiment-from-notes-{question,answer,horizon,vertical}/SKILL.md`
- `.claude/skills/idea_question/SKILL.md`
- `.claude/skills/idea_answer/SKILL.md`
- `.claude/skills/idea_question/references/*.md`
- `draft/review_draft.md`
- `review_notes/*_review.md`

Explicitly excluded:

- `.claude/skills/idea_brainstorm/`
- any ideastorm skill

When a reference rule and the current design conflict, apply this precedence:

```text
explicit user requirements
> draft/learning_workflow_optimization_discussion.md
> extracted reusable principles
> legacy implementation behavior
```

