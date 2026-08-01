# 新版 Learning Workflow：来源与引用索引

> 归档状态：旧版设计的来源索引，仅供设计追溯。

## 0. 引用约定

- 本索引引用的是旧实现中的“规则来源”，不表示新版运行时依赖旧文件。
- 行号基于 2026-07-27 工作区中的只读源文件。
- `采用` 表示保留原则；`改写` 表示保留意图但改变数据结构；`不采用` 表示明确避免。

## 1. 新版设计的直接来源

| 新版规则 | 来源 | 处理 |
|---|---|---|
| Topic 下有多个 Anchor，每个 Anchor 每层可有多个 entry，Direction 只选其中兼容子集 | `archive/learning_workflow/learning_workflow_optimization_discussion.md:1-24` | 采用 |
| Global Catalog、Anchor Layer Map、Direction Bundle 是三个不同视图 | `archive/learning_workflow/learning_workflow_optimization_discussion.md:26-70` | 采用 |
| Anchor 由 phase、regime、backend、bottleneck、primary baseline、metric 确定 | `archive/learning_workflow/learning_workflow_optimization_discussion.md:72-111` | 采用 |
| LayerEntry 原子化、entity 全局复用、所有事实回指证据 | `archive/learning_workflow/learning_workflow_optimization_discussion.md:113-145` | 采用 |
| 跨层边连接 Entry ID，而不是 Layer 名 | `archive/learning_workflow/learning_workflow_optimization_discussion.md:147-183` | 采用 |
| Anchor 级 BaselineSet 与 entry 级 baseline 关系同时存在 | `archive/learning_workflow/learning_workflow_optimization_discussion.md:185-204` | 采用 |
| 专家评阅单位是 Direction + BaselineSet，而不是整个 Anchor Map | `archive/learning_workflow/learning_workflow_optimization_discussion.md:206-231` | 采用 |
| 最终事实对象为 Entity、Claim、Anchor、Entry、Edge、Direction、Bundle | `archive/learning_workflow/learning_workflow_optimization_discussion.md:233-260` | 采用 |

## 2. 从 learning scheduler 提炼的编排来源

| 可复用规则 | 来源 | 处理 |
|---|---|---|
| phase 化流水线、独立 agent 任务和 worker pool | `scripts/learning_scheduler.ts:3-10` | 改写为 discovery、curation、direction、review、render |
| L1–L6 的基本层定义 | `scripts/learning_scheduler.ts:26-88` | 采用层坐标，不采用固定问题模板 |
| `pending/running/done` 与 worker/时间戳 | `scripts/learning_scheduler.ts:110-130` | 采用并增加 failed/retriable |
| skill body 与运行参数分离 | `scripts/learning_scheduler.ts:221-244` | 采用为新 skill reference + runtime payload |
| 中断时 running 回滚，产物信号与 checkpoint 对账 | `scripts/learning_scheduler.ts:246-311` | 采用，完成条件改为 schema/hash 校验 |
| 子进程集中跟踪、信号清理、超时终止 | `scripts/learning_scheduler.ts:403-463` | 采用 provider 级 timeout/cleanup 原则 |
| 独立层任务并发启动和完成检查 | `scripts/learning_scheduler.ts:465-550` | 采用为 L1–L6 × value axis task graph |
| worker 循环从 pending 池取任务并逐次 checkpoint | `scripts/learning_scheduler.ts:579-638` | 采用 |
| Horizon 把 answer 汇成单层 summary | `scripts/learning_scheduler.ts:640-723` | 不采用；改为 lossless entry catalog |
| Vertical 只读取 horizon summary | `scripts/learning_scheduler.ts:725-782` | 不采用；改为从 claim/entry/edge 原图构造 Direction |

## 3. 从 learning skills 提炼的知识来源

| 可复用规则 | 来源 | 处理 |
|---|---|---|
| 每层围绕方法、实现、实验环境展开 | `.claude/skills/learning-experiment-from-notes-question/SKILL.md:28-60` | 改写为 exploration、implementation reuse、method 三价值轴，baseline 横贯三轴 |
| L1–L6 的覆盖范围 | `.claude/skills/learning-experiment-from-notes-question/SKILL.md:39-48` | 采用 |
| 问题先拆逻辑链节点，每个节点长 query→短 query | `.claude/skills/learning-experiment-from-notes-answer/SKILL.md:38-67` | 采用 |
| 搜索保留节点映射，低命中逐级降级 | `.claude/skills/learning-experiment-from-notes-answer/SKILL.md:69-153` | 采用，检索 backend 改为可插拔 |
| 去重后保留 path→node/query 反向索引 | `.claude/skills/learning-experiment-from-notes-answer/SKILL.md:155-161` | 采用 |
| 证据不足显式写 uncertainty，区分直接证据、推断和 Web | `.claude/skills/learning-experiment-from-notes-answer/SKILL.md:237-240,280-286` | 采用并固化为 claim 字段 |
| 各层需要不同技术粒度 | `.claude/skills/learning-experiment-from-notes-answer/SKILL.md:198-236` | 采用为 entry 完整性检查 |
| 同义方法去重但不丢实现、实验和来源 | `.claude/skills/learning-experiment-from-notes-horizon/SKILL.md:27-64` | 采用为 GlobalEntity + Claim ledger |
| 关系类型包括替代、互补、依赖 | `.claude/skills/learning-experiment-from-notes-horizon/SKILL.md:88-94` | 采用并扩展 conflict/controls/enables |
| 跨层组合优先兼容性明确，缺证据要标注 | `.claude/skills/learning-experiment-from-notes-vertical/SKILL.md:33-50` | 采用为 Direction 约束 |
| 强制 L1→L6 完整组合 | `.claude/skills/learning-experiment-from-notes-vertical/SKILL.md:53-128,192-202` | 不采用 |

## 4. 从 idea review 编排提炼的来源

| 可复用规则 | 来源 | 处理 |
|---|---|---|
| 双持久角色、脚本只做代理、校验、路由、记录 | `scripts/idea_review_orchestrator.ts:3-18` | 采用概念；新脚本不调用旧 broker |
| Judge 无检索工具，Evidence Agent 只读 | `scripts/idea_review_orchestrator.ts:31-77` | 采用角色隔离 |
| 显式 session/round/history/reference/next-entry 状态 | `scripts/idea_review_orchestrator.ts:120-162` | 采用为本地 workflow state |
| stream 输入输出、预算、resume 和工具权限由脚本指定 | `scripts/idea_review_orchestrator.ts:217-254` | 改写为 provider abstraction |
| 每轮超时；协议不完整时只做一次原 session 修复 | `scripts/idea_review_orchestrator.ts:450-573` | 采用 |
| marker payload 解析后再转发，禁止叙述性输出混入协议 | `scripts/idea_review_orchestrator.ts:607-815` | 用 JSON Schema Structured Outputs 替代 |
| reference 请求由编排器白名单校验并且每类只注入一次 | `scripts/idea_review_orchestrator.ts:817-850` | 采用 |
| LOOP 被翻译成明确的下一任务、期待信号和保存状态 | `scripts/idea_review_orchestrator.ts:908-1099` | 采用为显式 action state machine |
| checkpoint 和 raw log 可恢复终止判断或中断问题 | `scripts/idea_review_orchestrator.ts:1101-1192` | 采用为 state + append-only event/model-call log |
| 最终产物同时保留 judgment 与逐轮问答 | `scripts/idea_review_orchestrator.ts:1194-1221` | 采用；另保留 bundle 与 claim 引用 |
| 主循环严格 QA→AA→QA，逐轮校验 round 和 marker | `scripts/idea_review_orchestrator.ts:1570-1663` | 采用为 Direction review loop |

## 5. 从 idea review skill 提炼的专家来源

| 可复用规则 | 来源 | 处理 |
|---|---|---|
| 首轮广覆盖、之后 candidate/uncertain/low 入队逐维度处理 | `.claude/skills/idea_question/SKILL.md:87-155` | 采用为 coarse screen + review queue |
| reference 按需加载，每轮只聚焦一个维度 | `.claude/skills/idea_question/SKILL.md:159-222` | 采用 |
| 回答充分、无价值或缺证据决定 ready/low/follow-up | `.claude/skills/idea_question/SKILL.md:226-245` | 采用，状态名改为 accepted/baseline_reference/needs_evidence/rejected |
| 最终评判前覆盖所有候选维度 | `.claude/skills/idea_question/SKILL.md:248-285,304-328` | 采用 |
| 五层完整性模板关注负载、编译、调度、Kernel、硬件因果链 | `.claude/skills/idea_question/SKILL.md:330-340` | 采用，但不要求每个 Direction 覆盖全部层 |
| 旧最终价值为 relevance/reference/depth | `.claude/skills/idea_question/SKILL.md:342-359` | 不采用；由 exploration > implementation reuse > method 替代 |
| Evidence Agent 维护 loaded_paths/evidence_summary、按需补查、缺口不编造 | `.claude/skills/idea_answer/SKILL.md:8-19,143-217,227-274` | 采用为 EvidenceClaim ledger + information gaps |
| 回答保留负载→编译→调度→Kernel→硬件的因果关系 | `.claude/skills/idea_answer/SKILL.md:276-288` | 采用为 edge 评审模板 |

## 6. 从专家 reference 提炼的评判来源

| 专家问题 | 来源 |
|---|---|
| 子计算独立性、运行时动态参数、瓶颈资源正交性、额外开销能否被隐藏 | `.claude/skills/idea_question/references/01-background-and-demand.md:3-27` |
| 并发原语选择、粒度与同步、编译/运行时分工、资源竞争、实现集成 | `.claude/skills/idea_question/references/02-concurrency-implementation.md:3-27` |
| 硬件原语能力边界、多原语竞争、空间/时间并行、可编程粒度与迁移 | `.claude/skills/idea_question/references/03-hardware-mechanisms.md:3-23` |
| memory hierarchy、访问模式、NUMA/NoC、硬限制与软限制 | `.claude/skills/idea_question/references/04-architecture-limits.md:3-23` |
| 工具粒度、并发因素覆盖、模拟误差、未来架构可扩展性 | `.claude/skills/idea_question/references/05-experiment-tools.md:3-22` |

这些 reference 的问题结构被采用，但不原样复制旧的高/中/低总评分。新版用它们验证 Direction 中的 scenario、entry、edge、baseline 和 experiment plan。

## 7. 人工与已验证产物的校准来源

| 校准用途 | 来源 | 提炼结果 |
|---|---|---|
| 人工会保留“潜在机会 + 怀疑点 + 实现线索”，而非只保留论文结论 | `draft/review_draft.md` 全文，尤其 `review` 列 | entry 允许 opportunity、constraint、implementation、evaluation 等不同角色 |
| 高质量 review 同时保留量化 claim、推断边界、缺失证据和复现步骤 | `review_notes/*_review.md` | ExpertReview 必须保存 evidence refs、gaps、baseline、ablation 与 reproduction plan |
| 深挖问题围绕适用条件、退化、资源竞争、实现障碍和测量边界迭代 | `review_notes/Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs_review.md` | Direction review 采用反例驱动的多轮 loop |
| characterization/tool 类论文即使不是新加速方法，也可成为强 baseline 或实验资产 | `review_notes/Demystifying the Placement Policies of the NVIDIA GPU Thread Block Scheduler for Concurrent Kernels_review.md` | baseline/tool 轨道独立保留，不因 exploration 低而删除 |

## 8. Provider 设计引用

新版实际运行采用原脚本相同的 DeepSeek/Claude CLI 路径：

| 接入决策 | 来源 |
|---|---|
| `claude` CLI + `deepseek-v4-flash[1m]` | `scripts/idea_review_orchestrator.ts:217-239` |
| 独立任务使用 `claude -p` 与 `stream-json` | `scripts/learning_scheduler.ts:429-450` |
| loop 会话使用 session ID、resume、stream-json | `scripts/idea_review_orchestrator.ts:217-239,336-470` |
| 当前凭证来自 Anthropic-compatible 环境变量 | 本机环境存在 `ANTHROPIC_BASE_URL`、`ANTHROPIC_AUTH_TOKEN`；索引只记录变量名，不记录值 |

OpenAI 只保留为环境可行性说明，不是新版运行 provider：

| 接入决策 | 官方依据 |
|---|---|
| Python SDK 已安装，但当前无 `OPENAI_API_KEY`，因此不能直接鉴权调用 | OpenAI Developer Quickstart：<https://developers.openai.com/api/docs/quickstart> |
| 若未来增加可选 adapter，可使用 Responses API 与 Structured Outputs | OpenAI Model Guidance：<https://developers.openai.com/api/docs/guides/latest-model>；OpenAI Structured Outputs：<https://developers.openai.com/api/docs/guides/structured-outputs> |

Codex CLI 可作为不依赖 `OPENAI_API_KEY` 的可选第二 provider：

| 接入决策 | 本机依据 |
|---|---|
| 非交互调用与 JSONL 事件 | `codex exec --json`，Codex CLI 0.144.1 本机 help |
| 最终输出 schema | `codex exec --output-schema <FILE>`，本机 help |
| loop session | `codex exec resume <SESSION_ID>`，本机 help |
| 当前认证 | `codex login status` 返回 `Logged in using ChatGPT` |
| 隔离边界 | 支持 read-only sandbox 和 approval=never，但没有 Claude CLI `--tools ""` 的等价硬开关；新版必须检测并拒收 tool-call turn |
