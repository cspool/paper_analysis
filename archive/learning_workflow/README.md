# Learning Workflow 历史实现与正式运行快照归档

> 初始归档日期：2026-07-29  
> 状态：历史实现只供追溯；`official_runs/` 保存明确标记生命周期的正式审计快照

本目录集中保存 Simple Semantic Loop 重构前形成的 Learning Skill 草案、
工作流设计、实现计划、能力评估、standalone Layered Exploration Python
workflow 和已退役 Codex 运行时。正文的业务设计和旧运行时代码保持为历史
快照，仅修正归档后失效的索引路径。

当前实现以以下文档为准：

- [01 Agent 类型与调度职责](../../draft/learning_workflow_scheduler_agent_refactor_plans/01_agent_types_and_scheduling_responsibilities.md)
- [02 Loop 需求闭合](../../draft/learning_workflow_scheduler_agent_refactor_plans/02_loop_requirement_closure_design.md)
- [03 消息与持久化契约](../../draft/learning_workflow_scheduler_agent_refactor_plans/03_script_agent_message_and_storage_contract_design.md)
- [04 JSON 通信契约](../../draft/learning_workflow_scheduler_agent_refactor_plans/04_script_agent_json_communication_contract_inventory.md)
- [05 最小 Controller 校验](../../draft/learning_workflow_scheduler_agent_refactor_plans/05_minimal_controller_validation_and_semantic_handoff_design.md)
- [06 外循环记忆与原子 Direction](../../draft/learning_workflow_scheduler_agent_refactor_plans/06_outer_loop_memory_trajectory_and_atomic_direction_design.md)
- [07 冻结快照与语义收敛](../../draft/learning_workflow_scheduler_agent_refactor_plans/07_frozen_decision_snapshots_semantic_convergence_and_runtime_delta_dedup_design.md)
- [08 正式快照、轮次授权与增量合并](../../draft/learning_workflow_scheduler_agent_refactor_plans/08_official_snapshot_round_lease_convergence_probe_and_delta_batching_design.md)

## 正式运行快照

- [多模态推理延迟优先 v6：12 轮正式审计快照](official_runs/multimodal_inference_latency_first_v6_20260801_round12/OFFICIAL_SNAPSHOT.md)

该快照同时冻结运行目录和当时实际使用的实现。它保持源运行真实的 `PAUSED`
生命周期，不把“正式保存”误写成工作流已经 `FINISHED`。

### 2026-07-29 四角色 Simple Semantic Loop 归档

- [SQLite/Stage/Gate 运行时、测试、Schema 与四个 Skill](simple_semantic_loop_stage_gate_legacy_20260729/)

该快照是本次三角色固定 Loop 重构的直接参考来源。它已从活动
`scripts/simple_semantic_loop/` 和 `.codex/skills/` 移出。

## 归档内容

### Skill 与协作模式草案

- [三类 Agent 的编写与协作模式](learning_idea_skill_draft.md)

### 旧版工作流设计链

- [Learning Workflow 优化讨论](learning_workflow_optimization_discussion.md)
- [Agent 编排与脚本协作设计约束](learning_workflow_agent_orchestration_design.md)
- [可复用知识提炼](learning_workflow_reusable_knowledge_extraction.md)
- [来源与引用索引](learning_workflow_source_provenance.md)
- [Codex CLI 新实现计划](learning_workflow_codex_implementation_plan.md)
- [Review-driven Backfill Loop 优化讨论稿](learning_workflow_review_driven_backfill_optimization.md)

### 已退役 Skill 实现

以下 Skill 已从 `.codex/skills/` 迁出，因此不会再被 Codex 当作当前可用 Skill 发现：

- [Learning Anchor Stage Controller](skills/learning-anchor-stage-controller/SKILL.md)
- [Learning Anchor Evidence Worker](skills/learning-anchor-evidence-worker/SKILL.md)
- [Learning Anchor Curator Worker](skills/learning-anchor-curator-worker/SKILL.md)
- [Learning Direction Planner](skills/learning-direction-planner/SKILL.md)
- [Learning Direction Reviewer](skills/learning-direction-reviewer/SKILL.md)
- [Learning Review Evidence Worker](skills/learning-review-evidence-worker/SKILL.md)

每个目录保留原 `SKILL.md`、Agent profile 及已有 reference 文件。它们不在
`.codex/skills/` 下，因此不会被当作当前可用 Skill 发现。

### 已退役 Codex 运行时

- [旧 CLI 入口](scripts/codex_learning_workflow.ts)
- [旧运行时模块与测试](scripts/codex_learning_workflow/)

该运行时仍记录上述六个旧 Skill 名称和 persistent Agent 设计，只能用于历史
审计，不能作为现行工作流启动。代码中的旧命令、原始路径和 Skill 查找逻辑按
历史快照保留，不表示这些位置仍是有效入口。

### 已退役 Standalone Layered Exploration

- [Python orchestrator](scripts/layered_exploration_orchestrator.py)
- [Python core runtime](scripts/layered_exploration_core.py)
- [离线测试](scripts/test_layered_exploration_workflow.py)
- [配套单体 Skill](skills/layered-exploration-workflow/SKILL.md)

这套实现是后续 Codex 工作流之前的单体角色、JSON action、独立 task graph
方案。后续设计明确不 import 该运行时，并将单一 Skill 的 Discovery、Curator、
Direction、Judge、Evidence 职责拆开；现行 Simple Semantic Loop 又进一步
替换为确定性 Controller 和 Decision、Worker、Reviewer 三类 fresh Turn。因此整套实现只保留作来源追溯，
不再位于 `scripts/` 或 `.claude/skills/` 的活动发现路径。

### 支撑性能力评估

- [Codex CLI 持久 Agent 与脚本编排能力结论](codex_cli_orchestration_capability_assessment.md)

这些材料记录了从“持久调度 Agent”方案演进到“确定性 Controller + 按需单 Turn Agent”架构之前的设计过程。若归档内容与当前实现计划冲突，应以当前实现计划为准。
