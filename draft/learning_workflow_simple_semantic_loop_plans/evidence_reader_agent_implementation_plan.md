# Evidence Reader Turn Agent 实现计划

## 1. 计划定位

本文实现一个只处理单一 SearchNeed 的短生命周期知识检索 Worker。

- 角色：`evidence_reader`
- Skill：`learning-semantic-loop-evidence-reader`
- 生命周期：一个 `EVIDENCE_READER_TASK` 对应一个 fresh Turn
- Reasoning effort：固定 `high`
- 允许工具：Obsidian Omnisearch 与 note read
- 输出：一个 `EVIDENCE_PACKET`
- 状态写入、Agent 调度和实验：禁止

依赖：

- [共享契约规范](shared_contracts.md)
- [Workflow Turn Agent 实现计划](workflow_turn_agent_implementation_plan.md)
- [Scheduler Script 实现计划](scheduler_script_implementation_plan.md)

现有 `.codex/skills/learning-semantic-loop-evidence-reader/` 只作为查询与 provenance 规则的迁移输入，不能直接注册到新 Controller：它仍使用旧 `EVIDENCE_TASK`、Skill-local schema 和旧 Envelope。本计划负责完成单点协议迁移。

## 2. 在动态 Workflow 中的位置

```text
Workflow Turn 提出 SearchNeed + EVIDENCE_READ Stage/Gate 草案
→ Controller 校验并冻结合同
→ Controller 启动 fresh Evidence Reader Turn
→ Reader 使用 Omnisearch 与 note read
→ Reader 输出 EVIDENCE_PACKET 并退出
→ Controller 运行 Evidence Gate
→ Controller 提交 validated result
→ Controller 触发新的 COMMITTED_RESULT_REQUIRES_INTEGRATION Workflow Turn
```

Reader 不直接接收 Workflow Agent 的调用，也不把结果发送给另一个 Agent。所有输入、结果和后续 trigger 都由 Controller 管理。

## 3. 对需求的责任

| 需求 | 本 Turn 的责任 |
|---|---|
| `REQ-19` | 保持 idea/baseline、knowledge、experiment、human、paper 各维度的表达差异 |
| `REQ-20` | 根据已冻结 SearchNeed 使用 Obsidian Omnisearch 查询指定维度 |
| `REQ-21` | 将给定问题转换为 Q1–Q3 有界查询，并记录关键词来源 |
| `REQ-23` | 提供可被 Workflow Turn 集成为性能优化潜力的证据 |
| `REQ-25` | 接受格式化 task，输出严格结构化 EvidencePacket |

本 Turn 不负责最终人类输出。

## 4. 输入契约

输入为：

```ts
PayloadTurnEnvelope<EvidenceReaderTask>
```

Envelope 必须使用 `messageType = "EVIDENCE_READER_TASK"`，并包含当前 TurnIdentity、StateBinding、`inputHash` 和 `stageContractHash`。Controller 构造的 dispatch packet 还必须固定 Skill name/version/hash、schema manifest hash、frozen StageContract/GateDefinition、task-scoped permission envelope、expected output schema 和本 Turn 终止条件；Reader 不从 Skill 或历史消息补这些运行事实。

`EvidenceReaderTask` 至少包含：

- 一个完整、pending、已校验的 SearchNeed；
- TopicFrame 相关范围；
- 当前 Anchor/Direction 的最小 focus projection；
- primary dimension 和至多一个 auxiliary dimension；
- technical objects、scenario terms、performance relations、known terms、synonyms；
- previous queries、reads、consumed/excluded source units；
- success criteria；
- 与 frozen StageContract canonical-equal 的 `TurnBudget`；其中 `evidenceRead` 固定 query、search-call、hit、selected-source 和 context 上限；
- allowed vault roots；
- schema manifest hash。

以下是 pre-dispatch task validator 的 fail-closed 条件，Controller 不得启动 Reader：

- 缺少一个明确问题；
- primary/auxiliary dimension 与 SearchIntent 路由不符；
- `targetDimensions` 不是“一个 primary 加零或一个 auxiliary”，或请求三个及以上知识维度；
- path 超出 allowlist；
- 要求“继续研究”“搜索更多资料”等开放目标；
- StageContract hash、Need revision 或 state binding 不一致。

Reader 在 Turn 开始时再次断言这些条件。若认为输入契约错误，不得把它伪装成 `not_found`、生成 Evidence，或自行写 attempt 状态；它应拒绝产生业务结果。Controller 重新运行权威 task validator：只有 validator 也失败时才记录 `input_contract_invalid`，重建合法 task 后再决定是否 dispatch；同一无效 task 不做格式重试。若 task 仍有效，则 Reader 的拒绝只是无效产出，进入普通同角色 fresh retry。协议不为此增加错误消息族。

Budget 必须满足共享契约的 Evidence role invariant：`maxLogicalQueries` 为 `1..3`，`maxSearchToolCalls` 计入 pagination 且不超过总 `maxToolCalls`，`maxHitsConsidered` 不超过 50，selected-source/context 上限关系合法。这些值是最大值而非完成目标；达到 success criteria 必须提前停止。

## 5. 知识维度与路由

| Dimension | 路径 | 主要回答 |
|---|---|---|
| `idea` | `idea_notes/` | 场景、baseline、论文方法、报告收益、候选机会 |
| `knowledge` | `knowledge_notes/` | 机制、接口、约束、适用边界 |
| `experiment` | `experiment_notes/` | 已有实现、配置、测量、失败和历史实验 |
| `human` | `human_notes/` | 本地经验、会议判断、环境限制和踩坑 |
| `paper` | `paper_secs/` | 原文定量值、实验条件和精确实现声明 |

固定 SearchIntent 路由：

| SearchIntent | Primary | Optional auxiliary |
|---|---|---|
| `discover_anchor` | idea | human |
| `define_baseline` | idea | experiment |
| `find_modification` | idea | knowledge |
| `explain_mechanism` | knowledge | idea |
| `find_implementation` | experiment | knowledge |
| `design_measurement` | experiment | knowledge |
| `challenge_direction` | knowledge | experiment 或 human |
| `verify_primary_source` | paper | 无 |

不得因为目录存在就自动查询。`experiment_notes/` 是历史证据，不能解释为授权新实验。

## 6. 查询形成算法

查询结构：

```text
path:<allowed-directory>
+ technical object
+ exact scenario
+ performance relation or evidence intent
```

每次使用 2–4 个区分度高的 term：

```text
Q1：精确 technical object + 精确执行场景 + intent-specific 精确词
Q2：保留执行场景，改用该 SearchIntent 的关系轴
Q3：使用有来源的同义词，并只做仍在 Topic 内的最小场景放宽
```

Q2 的关系轴按 intent 固定：

| SearchIntent | Q2 关系轴 |
|---|---|
| `discover_anchor` | baseline execution path + performance tension/resource |
| `define_baseline` | execution path/configuration + comparison scope/metric |
| `find_modification` | modification object + bottleneck mechanism |
| `explain_mechanism` | mechanism/resource + causal performance relation |
| `find_implementation` | target component/interface/configuration + method |
| `design_measurement` | metric/instrument/ablation + controlled baseline |
| `challenge_direction` | counterexample/degradation/constraint + mechanism |
| `verify_primary_source` | method/result metric + workload/configuration |

Q1–Q3 是整个 task 共用的全局降级级别，不是每个维度各自拥有三次查询。执行规则：

1. 先发 Q1。
2. Q1 无命中或上下文不足时才发 Q2。
3. Q2 仍不足时才发 Q3。
4. Q3 会失去 Topic 场景边界时不执行。
5. 某级已满足 success criteria 后停止。
6. 不超过 task budget。
7. 重复 query 只有在成功条件、场景、实体、同义词、排除来源或证据意图实质变化时允许。
8. Q1 必须查询 primary dimension；Q2/Q3 只有在当前未满足 criterion 明确属于已冻结 auxiliary dimension 时，才可选择 auxiliary。
9. 切换维度不重置 query level、sequence 或 budget；整个 task 最多执行 Q1、Q2、Q3 各一次。

关键词只能来自 task 或本 Turn 已深读的上下文。每个 term 记录来源和使用 sequence。
其中 `source = "task"` 的 `term` 必须逐字等于 SearchNeed 冻结词汇数组中的
一个完整元素，不能把 Topic-only 词或短语片段标成 task provenance；Q1 的
technical object、scenario、evidence intent 三轴以及 Q2 的 relation/intent
轴都按完整字符串校验，执行 query 也必须包含这些完整字符串。

## 7. Omnisearch 与深读

对每个 query：

1. 只调用 `mcp__obsidian__obsidian_search_notes`，固定 `mode = "omnisearch"`，并在 query 中放入与该 dimension 完全匹配的单一 `path:` filter；
2. 将 Q-level 视为一个 logical query；只有返回 `nextCursor` 且 hit budget 未耗尽时才可用 opaque cursor 取下一页，cursor 不得推导或改写；
3. pagination 不重置 Q-level，也不增加 semantic query 配额；每个 page tool event 和累计 hit count 都进入审计，累计结果仍受 Omnisearch 上游 50-hit cap 与 task budget 约束；
4. 记录 query、dimension、level、terms、page count、hit count、outcome；
   `EvidenceSearch.pathFilter` 保存 query 中实际执行的完整 token（例如
   `path:idea_notes/`），不能只保存裸前缀 `idea_notes/`；
   `toolCallIndex` 只对 `obsidian_search_notes` 调用做从 1 开始的连续编号，
   get-note 读取不增加它；跨 search/hit/read 的真实顺序由 `sequence` 表达；
5. 选择少量真正可能回答 success criteria 的 hit；
6. 只对选中 hit 的实际 vault-relative path 调用 `mcp__obsidian__obsidian_get_note`，先使用 `format = "document-map"`；
7. 再使用 `format = "section"` 和 document map 中存在的 heading target 读取完整相关 subtree；
8. 必要时读取 setup、baseline、result、limitation 相邻段；
9. 仅当 note 没有可用 heading boundary 时，才允许 `format = "content"` 或 `"full"`；
10. 记录实际 path、heading、source unit 和 exact context。

每次实际 `obsidian_get_note` 调用都必须有且仅有一个对应的
`contextsRead` 项；document-map 与后续 section/content/full 读取分别记账，
不得把两次调用折叠成一个内容上下文。

同一物理 path 在一个 Turn 内只 selected 一次。后续 query 再命中已
selected/deep-read 的 path 时，后续 hit 必须 `selected = false` 并说明
duplicate/reuse；不能创建一个没有自己 map + content read ledger 的第二个
selected hitId。

`obsidian_get_note` 只允许 `target = {"type":"path","path":...}`；`section` 只在 `format = "section"` 时出现，且第一版仅允许 document map 中实际存在的 `type = "heading"` target。`includeLinks`、active/periodic target 以及其他参数均不开放。

不能单独成为 Evidence：

- 搜索 snippet；
- 文件名或标题；
- BM25 score；
- 没有实验设置的数字；
- 没有 baseline 的 speedup；
- 无上下文代码符号；
- 未实际读取的段落。

## 8. Evidence 形成

每个 finding 必须：

- 只表达一个 evidence-bounded claim；
- 使用稳定 claimKey；
- 标注 evidenceRole；
- 标注 direct/inferred；
- 区分 source_report/workflow_inference；
- 声明 applicable conditions；
- 报告收益时注明 comparison baseline；
- 引用 path、heading、sourceUnitId、sourceFamily；
- 保存一个连续的 exact context；
- 使用当前 task 内实际读取的 context。

同一 source family 的多份转述不算独立支持。矛盾必须引用完整 finding。每个未满足 success criterion 都进入 `unanswered`。

## 9. 输出和 Gate

只输出：

```text
PayloadTurnEnvelope<EvidencePacket>
```

`EvidencePacket.status` 固定为 `complete`；它只表示本 Turn 已终止，不表示 SearchNeed 已产生语义增量或 workflow 已完成。

Conclusion：

- `answered`：全部 success criteria 有可追溯 finding；
- `partial`：至少一个 finding，但仍有 unanswered criterion；
- `not_found`：没有 finding 或 contradiction，至少成功完成一个 logical query，且 `unanswered` 覆盖全部未满足 criterion；每个未执行的后续 level 都有 `topic_boundary` 或 frozen budget 等合法 stop reason。

Provider timeout、tool error、非法/丢失 cursor、未授权路径或未完成必需 deep read 不是 `partial`/`not_found` 证据，不能提交 EvidencePacket；Controller 按 runtime、security 或 validation failure 处理。正常耗尽 frozen semantic budget 可以产生 `partial`/`not_found`，但必须记录实际计数和 stop reason。

发出响应前，Reader 必须对 expected schema、TurnIdentity、StateBinding、`inputHash`、Need revision、role/message、结论一致性和唯一顶层 JSON 做一次内部自检。

输出后立即终止，不输出下一步、不创建 SemanticDelta、不请求其他 Agent。

Controller 的 Evidence Gate 至少检查：

- Envelope/contract/state binding；
- Need ID/revision；
- role/message type；
- query path 和 tool events；
- SearchIntent/dimension 路由；
- budget；
- task budget 与 frozen StageContract canonical-equal，logical-query、search-call、read/tool event 分别计数；
- 每个 tool call 的 tool name、mode/format、target path 和单一 `path:` filter；
- Q1–Q3 的全局单调顺序、跨维度不重置和 searches/contextsRead 共享的唯一 sequence；
- pagination cursor 来自上一页、page/hit budget 合法，且不被计成新的 Q-level；
- selected hit 与 contextsRead 对应；
- finding 引用真实 context；
- 每个 contradiction 引用 packet 内完整 finding 和被反驳 claim/object，不能脱离 finding 单独存在；
- exact context 连续性；
- term provenance 只引用 task 或更低 sequence 的已读 context；
- source/query dedup；
- conclusion 与 findings/unanswered 一致；
- terminal conclusion 的 stop reasons 完整，且无 provider/tool/security failure 被伪装成知识结论；
- 无写入、执行或实验事件。

Gate 通过只提交 EvidencePacket；是否产生语义变化由下一次 Workflow Turn 判断。

## 10. 权限配置

```json
{
  "role": "evidence_reader",
  "lifecycle": "fresh_turn",
  "reasoningEffort": "high",
  "tools": [
    {
      "name": "mcp__obsidian__obsidian_search_notes",
      "allowedOperations": [
        {"mode": "omnisearch"}
      ],
      "allowedArguments": [
        "mode",
        "query",
        "cursor"
      ]
    },
    {
      "name": "mcp__obsidian__obsidian_get_note",
      "allowedFormats": [
        "document-map",
        "section",
        "content",
        "full"
      ],
      "allowedArguments": [
        "format",
        "target",
        "section"
      ],
      "allowedTargetTypes": [
        "path"
      ],
      "allowedSectionTypes": [
        "heading"
      ]
    }
  ],
  "allowedVaultPaths": [
    "idea_notes/",
    "knowledge_notes/",
    "experiment_notes/",
    "human_notes/",
    "paper_secs/"
  ],
  "vaultWrite": false,
  "filesystem": "none",
  "shell": false,
  "network": false,
  "delegation": false,
  "goals": false,
  "experimentExecution": false,
  "allowedInputMessageTypes": [
    "EVIDENCE_READER_TASK"
  ],
  "allowedOutputMessageTypes": [
    "EVIDENCE_PACKET"
  ]
}
```

实际 task 只开放其 primary/auxiliary 目录子集。Runtime admission 必须逐个检查 tool event；仅在最终 JSON 中声称使用了合法路径不够。

## 11. 实现文件

```text
.codex/skills/learning-semantic-loop-evidence-reader/
├── SKILL.md
├── references/
│   ├── role_profile.json
│   └── schema_manifest.json
└── scripts/
    └── validate_evidence_packet.py

scripts/simple_semantic_loop/
├── prompt_templates/evidence_reader.ts
├── role_profiles/evidence_reader.json
├── validators/evidence_packet_validator.ts
└── tests/fixtures/evidence_reader/
```

Skill 内旧 schema 要么删除并引用 shared manifest，要么由 shared schema 自动生成并通过 hash 测试；不能手工维护分叉版本。Python helper 与 TypeScript runtime validator 也必须共享同一组生成类型/测试向量，不能各自维护不同的业务规则。

## 12. 实现工作包

### ER-1：契约迁移

- 接入 TurnIdentity、StateBinding 和 StageContract hash；
- 对齐 shared EvidenceTask/EvidencePacket schema；
- 将旧 `EVIDENCE_TASK` 单点迁移为 canonical `EVIDENCE_READER_TASK`，同步 Skill、role profile、schema、fixture 和 validator，迁移后不双重接受；
- 增加 manifest hash 校验。

验收：旧格式输出不能绕过新 Envelope validator。

### ER-2：Role profile

- 只开放两个精确 MCP tool name，并限制 search mode 与 note-read format；
- 为 task 收窄 path；
- 禁用写入、shell、network、Goal 和 delegation。

验收：越权 tool event 使 attempt `security_invalid`。

### ER-3：Query planner

- 实现 intent routing；
- 实现 Q1–Q3；
- 用 intent-specific Q2 模板替换旧的“所有 intent 都必须含 bottleneck term”规则；
- 实现全局 query-level、auxiliary 切换规则；
- 实现 logical-query pagination provenance（page count、opaque cursor chain、累计 hits）；
- 实现 term provenance、共享 sequence 和 dedup。

验收：每条查询能追溯到 task 或更早的已读 context。

### ER-4：Deep read 与 Evidence

- document map/section read；
- source family/unit；
- exact context；
- direct/inferred 和 baseline。

验收：snippet、孤立数字和未读引用不能通过。

### ER-5：Terminal result

- 实现 answered/partial/not_found；
- 实现 terminal output 自检清单；
- 输出一个 Envelope 后结束；
- Controller Gate 和 result commit 集成测试。

验收：Reader 不能创建 Delta 或触发下一 Turn；`input_contract_invalid` 不得
伪装成 `not_found`；structure、binding 或 pre-Gate semantic output failure
只允许 Controller 在固定总预算内以同一 Evidence Reader logical task 启动
最多两次带 `correctionFeedback` 的 fresh replacement attempt。

### ER-6：角色级测试

至少覆盖：

- 全部 SearchIntent 和合法 primary/auxiliary 组合；
- 每个 SearchIntent 的 Q2 关系轴；
- primary+auxiliary 合法、两个 auxiliary/三个维度非法；
- Q1/Q2/Q3 全局顺序、切换维度不重置、sufficient 后停止；
- opaque cursor pagination、50-hit cap 和 page/hit budget；
- 非法或不可行 TurnBudget、task 放大 frozen budget、logical query 与 pagination tool call 混计；
- query term origin、shared sequence、重复 query 与 consumed source；
- selected hit → document map → section/full fallback → finding provenance；
- exact context、同源重复、contradiction、answered/partial/not_found；
- `not_found` 隐藏 finding 或 contradiction；
- timeout/tool error/坏 cursor/未完成 deep read 伪装成 `partial` 或 `not_found`；
- paper verification、multiple path filters、path escape、非法 mode/format/tool；
- stale binding、旧 message type、same-role output retry 和 No Experiment。

## 13. 完成标准

1. 每个 task 只处理一个 SearchNeed。
2. 不同知识维度按 SearchIntent 明确路由。
3. 查询关键词和触发降级都有可审计来源。
4. 所有 finding 都来自本 Turn 实际深读上下文。
5. `not_found` 是合法、可复现的 terminal result。
6. Agent 不创建语义对象、不判断终态、不调度其他 Agent。
7. Agent 无写入、shell、网络、实验和 Goal 权限。
8. 输出通过 Controller Gate 后可触发新的 Workflow integration Turn。
9. `REQ-19`、`REQ-20`、`REQ-21` 的角色级验收 fixture 全部通过。
10. 每次 attempt 使用固定 `high` effort，不能被 task 或 run config 覆盖。
11. Agent 在输出前完成契约自检，且不调用或依赖格式修复辅助 Agent。
12. 输入契约错误与知识检索 `not_found` 严格分离，前者不能提交 EvidencePacket。
13. runtime tool-event admission 能证明所有搜索和读取都在 task-scoped allowlist 内。
