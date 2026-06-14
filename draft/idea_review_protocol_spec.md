# Idea Review — 协议与逻辑规格 (v4)

从 `idea_review_orchestrator.ts` + `idea_question/SKILL.md` + `idea_answer/SKILL.md` 提取，
为 skill 重写提供唯一事实来源。

---

## 核心模型

```
QA (盲评提问方)                    AA (独占 idea note 的回答方)
  │                                    │
  ├─ 首轮：固定五大类别总览问题 ──────►│
  │◄───── 声明对象 + 逐类回答 ─────────┤
  │                                    │
  ├─ 初筛候选维度                       │
  │                                    │
  ├─ 逐维度追问 ◄══════════════════►│  (多轮)
  │   (加载 reference → 生成问题 → 收回答 → 评估)   │
  │                                    │
  ├─ 所有维度处理完毕 → 最终评判         │
  │                                    │
  ▼                                    ▼
  ___JUDGMENT_COMPLETE___             (被关闭)
```

- **QA 主动**：为生成 review 进行多轮提问，自己决定何时结束
- **AA 被动**：收到问题 → 获取上下文 → 回答，不主动提问
- **单向前进**：两个 agent 都不回退，不重试已完成的步骤
- **编排器透明**：正常路径只解析 marker、转发 payload；仅在 CLI/适配器异常终止时执行会话级恢复

### Marker 与 LOOP 的分工

系统中两套标记，职责完全不同：

| | Marker（协议标记） | LOOP 标记 |
|------|---------|-------|
| 作用 | **消息类型标识**——编排器据此判断转发方向 | **Agent 进度 checkpoint**——编排器保存并翻译为下次执行语义 |
| 示例 | `___QA_QUESTION___`, `___AA_OUTPUT_START___`, `___JUDGMENT_COMPLETE___` | `[LOOP: §DIM_EVAL \| await=AA_OUTPUT \| dimension=硬件机制]` |
| 谁写入 | Agent（在协议块内/外） | Agent（在协议块外） |
| 谁读取 | 编排器 parse → 判断类型 → 转发 | 编排器提取字段 → 保存 → 与下次 marker 联合翻译 |
| 编排器是否理解语义 | 是——知道 question→转发给AA, judgment→结束 | 是——将入口、等待信号和状态字段翻译为具体执行语义 |
| Agent 是否依赖它 | 是——按 skill 规定的格式输出 | Agent 负责输出；下次接收的是翻译后的自然语言语义 |

编排器**不根据 marker 推断 agent 状态**。它只做：收到 question → 转发给 AA；收到 judgment → 结束。Agent 自己管理自己的 loop 进度，LOOP 标记是它的"书签"。

---

## 持久 Session 的核心问题

两个 agent 都是持久多轮 session。第一条消息（skill 全文）只在初始化时发送一次。
后续消息只是协议载荷。模型在多轮后会遗忘：

| Agent | 容易遗忘什么 |
|-------|------------|
| QA | 当前在处理哪个候选维度、哪些已 review_ready、距离完成 review 还有多远 |
| AA | 上下文账本（loaded_paths 去重集合）、三级获取预算是否已用尽 |

**解决方案**：
1. **Skill 用带 `GOTO/YIELD/TERMINATE` 的伪代码和命名 § 定义工作流**
2. **Agent 每次 YIELD 时附带下一次输入到达后的执行起点**
3. **编排器下次输入时将该 checkpoint 与实际 marker 联合翻译为具体执行语义**，强迫 agent 从指定 § 恢复

### 执行与停止语义

- `§` 是可跳转的程序标签，不天然构成一次 Agent 响应的边界
- Agent 收到一条输入后，从翻译出的任务块入口开始，按线性步骤执行，并根据最后一个控制流步骤连续执行其他 § 或暂停
- 内部 § 不输出协议块，也不得停止；只有 `YIELD` 或 `TERMINATE` 才结束本次响应
- `YIELD` 输出恰好一个协议块和一个 LOOP；LOOP 中的 § 表示**下一次执行起点**
- `TERMINATE` 输出最终 judgment 后结束工作流
- 协议块与 LOOP 必须位于可见最终输出，不得只停留在 thinking/内部推理
- 仅输出 LOOP、自然语言过程说明或内部 § 结果，均不构成合法响应
- 为抵抗长上下文中的规则遗忘，每个 `GOTO`、`[LOOP: ...]`、`YIELD`、`TERMINATE` 动作都必须在动作出现处就地写清语义；不能只依赖工作流开头的全局定义

### 动作就地语义规则

- **`GOTO §X`**：紧邻动作说明“仅改变当前响应内执行位置、不输出、不停止，并继续执行目标 §”
- **`[LOOP: §X | await=Y | ...]`**：紧邻标记说明“写入可见输出，记录下一次输入 Y 到达后的恢复点 X；LOOP 本身不跳转、不停止”
- **`YIELD <协议块>`**：紧邻动作说明“当前协议块与 LOOP 已输出，立即停止当前响应并等待指定输入；本次响应不得提前执行 LOOP 指向的 §”
- **`TERMINATE <协议块>`**：紧邻动作说明“输出最终协议块与终止 LOOP 后结束整个工作流，不再恢复执行”
- **`[PROTOCOL_REPAIR]`**：这是输入恢复指令，不是输出或响应边界；保持现有状态，从未完成的控制流动作继续。repair 阶段禁止调用工具或重复检索，必须实际输出完整 marker + LOOP，不能只声称“已完成/YIELD”
- 不在每个 § 开头重复一段全局停止语义；局部说明应贴着真正影响控制流的动作，避免动作与含义在长上下文中分离

### AA 空终止恢复

- 若 AA 返回空可见响应，且 CLI `result.stop_reason` 为 `tool_use` 或 `null`，视为 CLI/模型适配器已异常终止当前 turn，而不是仍有工具调用在运行
- 这类状态不向原 AA 会话发送 `[PROTOCOL_REPAIR]`；实践中原会话通常会再次返回 `0 input_tokens / 0 output_tokens`
- 编排器关闭原 AA，会用新 session 接管当前轮次，并注入 canonical 论文信息、最近问答历史、当前问题和字面输出骨架
- 接管 session 通过 CLI `--tools ""` 禁用全部内建工具，并保留文件/MCP/网络 deny-list；证据不足只能写入 `information_gaps`；其 session ID 与 `aa_no_tools=true` 写入 checkpoint
- 后续轮次及 `--resume` 继续使用该无工具 AA，不反复创建替代 session
- 若无工具 AA 仍发生同类空终止，保留 checkpoint 并报错，不继续自动替换

---

## 1. QA 工作流（loop 定义）

QA 不维护外部状态机。Skill 伪代码定义内部 `GOTO`；只有到达 `YIELD/TERMINATE` 的 § 才输出 marker。
Marker 是 QA 告诉编排器“本次执行结果”的信号，LOOP 是下一次执行起点。

```
§INIT
  输入: skill 初始化消息
  初始化 DIM_QUEUE / review_material / loaded references
  输出 ready
  [LOOP: §R1 | await=START_REVIEW]
    语义: 写入下一次 START_REVIEW 到达后的恢复点 §R1；LOOP 本身不跳转、不停止
  YIELD ready → STOP
    语义: ready 与 LOOP 输出后停止当前响应；本次响应不执行 §R1

§R1 — 首轮盲问
  输入: START_REVIEW
  输出:
    ___QA_QUESTION___
    {
      "round": 1,
      "question_level": 1,
      "question_category": "五大类别总览"
    }
    ___QA_QUESTION_TEXT___
    <固定五大类别总览问题正文>
    ___QA_QUESTION_TEXT_END___
    ___QA_QUESTION_END___

    [LOOP: §SCREEN | await=AA_OUTPUT | round=1]
      语义: 写入下一次 round 1 AA_OUTPUT 到达后的恢复点 §SCREEN；本次响应不执行 §SCREEN
  YIELD ___QA_QUESTION___ → STOP
    语义: 只输出固定 question 与 LOOP，然后停止当前响应

§SCREEN — 初筛维度
  收到 AA 首轮回答后：
  - 识别研究对象及核心机制
  - 五大类别各标记: candidate / uncertain / low
  - 全部 candidate + uncertain 加入 DIM_QUEUE
  - 设置 next_question_round=2
  - 若队列为空 → GOTO §JUDGE (relevance=低); 语义: 不输出、不停止，立即继续 §JUDGE
  - 否则 → GOTO §DIM_NEXT; 语义: 不输出、不停止，立即继续 §DIM_NEXT

§DIM_NEXT — 选取下一维度
  从 DIM_QUEUE 取出第一个 status=pending 的维度，设为当前维度
  若队列全部 review_ready 或 low → GOTO §JUDGE; 语义: 不输出、不停止，立即继续 §JUDGE
  否则 → GOTO §DIM_REF; 语义: 不输出、不停止，立即继续 §DIM_REF

§DIM_REF — 加载专家知识
  若当前维度的 reference 未加载:
    输出:
      ___QA_REFERENCE_REQUEST___
      { "round": <next_question_round>, "question_category": "<当前维度>" }
      ___QA_REFERENCE_REQUEST_END___

      [LOOP: §DIM_ASK | await=QA_REFERENCE | dimension=<维度名> | round=<next_question_round>]
        语义: 写入下一次 QA_REFERENCE 到达后的恢复点 §DIM_ASK；本次响应不执行 §DIM_ASK
    YIELD ___QA_REFERENCE_REQUEST___ → STOP
      语义: 只输出 request 与 LOOP，然后停止当前响应

  若已加载 → GOTO §DIM_ASK; 语义: 不输出、不停止，立即继续 §DIM_ASK

§DIM_ASK — 生成追问
  输入: reference 注入，或从 §DIM_REF / §DIM_EVAL 内部跳转
  结合 已有回答 + reference 中 2-3 个引导要点 + 适用五层模板 → 生成具体问题
  输出:
    ___QA_QUESTION___
    {
      "round": N,
      "question_level": 2,
      "question_category": "<当前维度>",
      "question_subcategory": "<可选>"
    }
    ___QA_QUESTION_TEXT___
    <具体追问正文>
    ___QA_QUESTION_TEXT_END___
    ___QA_QUESTION_END___

    [LOOP: §DIM_EVAL | await=AA_OUTPUT | dimension=<维度名> | round=<N>]
      语义: 写入下一次该 round AA_OUTPUT 到达后的恢复点 §DIM_EVAL；本次响应不执行 §DIM_EVAL
  YIELD ___QA_QUESTION___ → STOP
    语义: 只输出 question 与 LOOP，然后停止当前响应

§DIM_EVAL — 评估回答
  收到 AA 回答后，对照五层模板 + 该类别评估标准:
  - 设置 next_question_round=当前 AA_OUTPUT.round+1
  - 信息充分 → 标记 review_ready → GOTO §DIM_NEXT; 语义: 不输出、不停止，立即继续 §DIM_NEXT
  - 缺关键证据 → GOTO §DIM_ASK; 语义: 不输出、不停止，立即继续 §DIM_ASK 聚焦追问
  - 确认无价值 → 标记 low → GOTO §DIM_NEXT; 语义: 不输出、不停止，立即继续 §DIM_NEXT

§JUDGE — 最终评判
  汇总所有 review_ready 维度的 review_material
  输出:
    ___JUDGMENT_COMPLETE___
    {
      "relevance": "高"|"中"|"低",
      "relevance_reason": "...",
      "reference_value": "高"|"中"|"低",
      "reference_reason": "...",
      "depth_value": "高"|"中"|"低",
      "depth_reason": "...",
      "summary": "...(含高价值方法摘要条目)..."
    }
    ___JUDGMENT_COMPLETE_END___

    [LOOP: §TERMINATED | done]
      语义: 写入终止 checkpoint；不表示未来还会恢复执行
  TERMINATE ___JUDGMENT_COMPLETE___
    语义: 只输出 judgment 与终止 LOOP，然后结束整个工作流
```

---

## 2. AA 工作流（loop 定义）

AA 只有两个 § 步骤。上下文管理规则和输出格式都嵌入在步骤内部。

```
§INIT
  1. 收到 skill + idea note
  2. 从 idea note 提取线索标题，在 paper_secs 中定位论文，并从论文主文件 H1 确认保留原始标点的 canonical paper_title
  3. 执行一级上下文获取:
     obsidian_search_notes(query="<标题>", pathPrefix="experiment_notes/")
     obsidian_search_notes(query="<标题>", pathPrefix="knowledge_notes/")
     obsidian_search_notes(query="<标题>", pathPrefix="review_notes/")
     → 差集 → 按相关性最多读 5 个唯一新路径
     → 加入 loaded_paths，记录 evidence_summary
  4. 输出初始化完成协议:
     ___AA_INIT_COMPLETE___
     {"paper_title":"<canonical title>","paper_subdir":"<vault-relative path>"}
     ___AA_INIT_COMPLETE_END___

     [LOOP: §ANSWER | await=QA_QUESTION | L1_done=true | loaded_paths=<N>]
       语义: 写入下一次 QA_QUESTION 到达后的恢复点 §ANSWER；本次响应不执行 §ANSWER
  YIELD ___AA_INIT_COMPLETE___ → STOP
    语义: 只输出 init 协议与 LOOP，然后停止当前响应

§ANSWER — 接收问题并回答
  1. 收到 ___QA_QUESTION___，提取 question / question_level / question_category
  2. 判断首轮: question_category="五大类别总览" → 按首轮规则组织回答
  3. 上下文检查:
     a. 先从 session 记忆中的 evidence_summary 提取（已在之前轮次读取并摘要）
     b. 若记忆模糊 → 可重读 loaded_paths 中已有路径（不消耗新获取预算）
     c. 仍不足 ∧ L2 未用? → 二级获取 (新路径):
        obsidian_search_notes(query="<标题>", pathPrefix="paper_secs/")
        → 与 loaded_paths 差集 → ≤5 唯一新路径 → 加入 loaded_paths
     d. 仍不足 ∧ 本轮 L3 未用? → 三级获取 (新路径):
        从缺口选 1 个关键词 → obsidian_search_notes(mode="omnisearch")
        → 与 loaded_paths 差集 → ≤1 唯一新路径 → 加入 loaded_paths
     e. 仍不足 → 写入 information_gaps，不编造
  4. 按五层执行链组织回答 (参见 §6 模板)
  5. 输出:
     ___AA_OUTPUT_START___
     {
       "round": N
     }
     ___AA_SOURCES_START___
     - <引用的 vault-relative 路径，每行一个；没有则留空>
     ___AA_SOURCES_END___
     ___AA_GAPS_START___
     - <信息缺口，每行一个；没有则留空>
     ___AA_GAPS_END___
     ___AA_ANSWER_START___
     <原始 Markdown 回答——无需转义换行或双引号>
     ___AA_ANSWER_END___
     ___AA_OUTPUT_END___

     [LOOP: §ANSWER | await=QA_QUESTION | completed_round=<N> | L1=done | L2=<done|pending> | L3_rounds=<已用轮次> | loaded_paths=<N>]
       语义: 写入下一次 QA_QUESTION 到达后的恢复点 §ANSWER；本次响应不继续生成下一轮
  6. YIELD ___AA_OUTPUT___ → STOP
     语义: 只输出 AA_OUTPUT 与 LOOP，然后停止当前响应并等待下一问题

路径去重 (贯穿 §INIT 步骤 3 和 §ANSWER 步骤 3):
  - 所有路径规范化为 vault-relative path
  - loaded_paths 记录已访问过的路径（写入 LOOP 标记，编排器每轮保存并翻译状态，永不丢失）
  - 优先从 session 记忆中的 evidence_summary 提取信息（快，不消耗预算）
  - 若记忆模糊或证据不足 → 可以重读已加载路径（不消耗新获取预算）
  - 搜索结果与 loaded_paths 差集后才计入各级预算——已加载路径不算"新获取"
  - 各级路径上限针对"新获取"，不要求凑满
  - 预算耗尽后仍不足 → 写入 information_gaps

首轮回答特殊规则 (question_category="五大类别总览"):
  - 先声明研究对象身份（标题、方法名、核心问题、主要 claim）
  - 逐类回应五大价值维度，优先给定量数字、来源路径
  - 不相关类别明确写"不相关" + 理由
  - 回答必须自包含——QA 看不到 idea note
```

---

## 3. 编排器调度逻辑

### 3.1 启动

```
1. spawn QA session (--disallowedTools=ALL, 19个工具全部阻断)
2. spawn AA session (--disallowedTools=仅阻断写入, 保留读/搜)
3. send QA: QA_SKILL 全文
4. send AA: AA_SKILL 全文 + idea_note (前5000字符)
5. 等待双方就绪信号
```

### 3.2 每轮转发循环

```
loop:
  parse agent 输出的 LOOP 标记 → 保存为 agent_next_entry
  
  若 parseQAMarker → judgment:
    writeReview → close both → exit
  
  若 parseQAMarker → question:
    anchored_input = 翻译执行语义(AA, qaOutput中的LOOP, QA_QUESTION) + formatMarkerPayload(question)
    send AA → parseAAMarker → anchored_input = 翻译执行语义(QA, aaOutput中的LOOP, AA_OUTPUT) + formatAAOutput(answer)
    send QA → resolveQAReferenceRequests → 回到 loop
  
  若 parseQAMarker → reference_request:
    validate → 读文件 → send QA reference → parse QA 新输出 → 回到 loop
```

### 3.3 下一执行语义翻译格式

编排器向 agent 发送下一条输入时，在协议载荷前拼接：

```
[第 N 轮 — <Agent>]
本次执行语义：
从 `<数字入口 / 名称别名 — 任务块名称>` 开始；已经收到 `<本次 marker 信号>`。
<将 LOOP 状态字段翻译后的具体状态，例如“当前维度为架构影响，当前轮次为 5”>。
<该任务块针对此入口的具体动作，例如“评估本轮回答，更新当前维度状态，并按结果继续追问或处理下一维度”>。

── 协议载荷 ──
<marker + payload>
```

`agent_next_entry` 直接取自 agent 上一次输出的 `[LOOP: ...]` 行。原始 LOOP 只作为编排器内部 checkpoint，不机械回注给 Agent。编排器根据 Agent 角色、数字入口/名称别名、实际收到的 marker 和状态字段，将其翻译为上述具体自然语言执行语义。初始化输入和隔离 AA 恢复输入使用同一翻译机制。

### 3.4 QA reference 注入守卫

- `question_category` 必须在白名单内（5 个精确类别名）
- 同一类别不重复注入（由 `qa_loaded_references` 跟踪）
- 注入后 QA 必须产出 question 或 judgment，不得连续请求 reference

---

## 4. 协议格式速查

详细格式已在 QA loop (§1) 和 AA loop (§2) 各步骤中给出。下表仅做速查。

| 方向 | Marker 对 | Body | 对应 loop 步骤 |
|------|----------|------|---------------|
| AA→编排器 | `___AA_INIT_COMPLETE___` / `___AA_INIT_COMPLETE_END___` | JSON `{paper_title, paper_subdir}`；标题来自论文主文件 H1，保留原始标点 | §INIT |
| QA→编排器 | `___QA_QUESTION___` / `___QA_QUESTION_END___` | JSON `{round, question_level, question_category, question_subcategory?}` + `___QA_QUESTION_TEXT___`/`___QA_QUESTION_TEXT_END___` 裸写问题正文 | §R1, §DIM_ASK |
| QA→编排器 | `___QA_REFERENCE_REQUEST___` / `___QA_REFERENCE_REQUEST_END___` | JSON `{round, question_category}` | §DIM_REF |
| QA→编排器 | `___JUDGMENT_COMPLETE___` / `___JUDGMENT_COMPLETE_END___` | JSON `{relevance, relevance_reason, reference_value, reference_reason, depth_value, depth_reason, summary}` | §JUDGE |
| AA→编排器 | `___AA_OUTPUT_START___` / `___AA_OUTPUT_END___` | JSON `{round}` + sources/gaps/answer 三个裸文本 marker 区段 | §ANSWER |
| 编排器→QA | `___AA_OUTPUT_START___` / `___AA_OUTPUT_END___` | 同 AA 输出（编排器规范化后转发） | — |
| 编排器→QA | `___QA_REFERENCE_START___` / `___QA_REFERENCE_END___` | `question_category: <类别>` + 正文 | — |
| 编排器→AA | `___QA_QUESTION___` / `___QA_QUESTION_END___` | 同 QA 输出 | — |

统一约束:
- marker 独占一行；JSON 只承载固定结构字段，双引号按 JSON 规则转义
- sources/gaps/answer 裸写段无需转义；sources/gaps 每行一个条目
- 协议块外零文本（LOOP 标记除外——编排器单独解析）
- `___JUDGMENT_COMPLETE___` 优先级最高——出现即结束

---

## 5. Q 的提问逻辑细节

### 5.1 首轮固定问题

round=1, question_level=1, question_category="五大类别总览"。硬编码文本，直接输出，不做搜索。

问题覆盖五个价值维度：背景与需求 / 方法与实现 / 硬件机制 / 架构影响 / 实验证据。
要求 AA 先声明研究对象 → 逐类回应 → 标注不相关/证据不足的类别。

### 5.2 维度初筛

首轮回答后，五类别分别标记 candidate / uncertain / low：
- candidate：已显示价值信号
- uncertain：可能有价值但证据不足
- low：AA 明确说明不相关且理由可信

全部 candidate+uncertain 加入队列。首轮不做最终价值判定。

### 5.3 追问生成（4 要素）

1. 已有回答：承接已给出的具体机制，避免重复
2. 专家引导：从 reference 选 2-3 个最能填补缺口的要点
3. 五层模板：检查适用层次，缺失但适用的纳入问题
4. Review 证据：追问机制、设计选择、定量结果、基线、开销、边界、退化场景、来源

### 5.4 五层覆盖模板

| 层次 | 追问切入点 |
|------|-----------|
| 负载层 | 该机制对应哪些独立子计算或瓶颈？在什么负载条件下出现？ |
| 编译层 | 编译流程在哪个阶段发现并发机会？IR/codegen 如何表达？ |
| 调度层 | 运行时如何作出调度决策并映射资源？决策开销和退化条件？ |
| Kernel层 | Kernel 内部如何切分/同步/流水执行？关键资源约束？ |
| 硬件层 | 依赖哪些硬件原语？资源竞争、吞吐上限和跨平台边界？ |

不是机械层数门槛，不适用层要求 AA 说明原因。

### 5.5 评判标准

| 维度 | 高 | 中 | 低 |
|------|-----|-----|-----|
| relevance | ≥1 维度满足高价值标准，直接涉及并发 | 间接涉及，需特定视角关联 | 所有大类无高价值信号 |
| reference_value | 框架已集成、有定量证据、可迁移设计模式 | 有定性描述但缺定量证据 | 信息碎片化、概念性描述 |
| depth_value | 非平凡机制/设计/trade-off，五层链清晰 | 已知模式变体，深入收益有限 | 常规工程细节 |

高价值 = (relevance=高 ∧ reference_value=高) ∨ (depth_value=高)

---

## 6. 五层执行链（AA §ANSWER 步骤 4 的参考模板）

AA 的每个回答按此链组织——从负载定义向下追踪到硬件执行。
尽可能覆盖五层（即使问题只显式问一层），某层不可得 → 写 gaps，不编造。

| 层次 | 覆盖内容 | 关键问题 |
|------|---------|---------|
| 负载层 | 并发潜力在哪、独立子计算、运行时动态参数、瓶颈环节 | 哪些子计算可并行？粒度？ |
| 编译层 | IR 并发依赖表达、哪个阶段发现并发、AoT vs JIT | 编译框架如何表达和发现并发？ |
| 调度层 | 调度策略、静态 vs 动态、调度粒度 | 运行时如何映射并发到硬件？ |
| Kernel层 | Tile 约束、Warp/Thread 分工、persistent vs 静态、同步开销 | Kernel 内部如何执行并发？ |
| 硬件层 | 硬件并发原语、能力边界和退化、跨平台替代 | 什么硬件使并发成为可能？ |

五层间的**因果关系**（上层设计选择如何影响下层实现）是回答的核心价值。

---

## 7. Q 的专家知识注入

### 7.1 白名单

| question_category | reference 文件 |
|-------------------|---------------|
| 动态(调度/并发)的背景/需求 | references/01-background-and-demand.md |
| 并发方法的应用和实现 | references/02-concurrency-implementation.md |
| 提供并发机制的硬件模块/架构 | references/03-hardware-mechanisms.md |
| 影响并发的架构/机制 | references/04-architecture-limits.md |
| 架构性能和开销的实验工具 | references/05-experiment-tools.md |

### 7.2 约束

- QA 无文件访问工具——只能通过 `___QA_REFERENCE_REQUEST___` 获取
- 每个类别全 session 最多注入一次
- 请求不占用问答轮次，但注入后必须产出 question 或 judgment
- 不得连续请求多个 reference

---

## 8. 编排器实现细节

### 8.1 Session 参数

```
--model deepseek-v4-flash[1m]
--input-format stream-json --output-format stream-json
--permission-mode bypassPermissions
--session-id <uuid>
--add-dir /data3/paper_analysis
```

### 8.2 工具限制

| | QA (盲评) | AA (需读文件) |
|------|---------|------------|
| Obsidian MCP | 全部 12 个禁用 | 仅 write/append/patch/replace/delete_note 禁用 |
| 文件系统 | Bash, Read, Write, Edit, Glob | Write, Edit, NotebookEdit |
| Web | WebSearch, WebFetch | 无 |

### 8.3 解析函数

| 输入 | 函数 | 提取方式 |
|------|------|---------|
| AA 初始化输出 | `parseAAInitMarker` | 解析 canonical `paper_title` 与 `paper_subdir`，写入 checkpoint 并决定最终 review 文件名 |
| QA 输出 | `parseQAMarker` | judgment > reference_request > question 优先级 |
| AA 输出 | `parseAAMarker` | JSON round (`___AA_OUTPUT_START___`..`___AA_SOURCES_START___`) + sources/gaps/answer 三个 marker 区段；兼容读取合法 v6 JSON payload |
| LOOP 标记 | `parseLoopMarker` (新增) | 匹配 `[LOOP: ...]` 行，字符串透传 |

入站容错边界：
- QA question 已有合法 JSON，且 `___QA_QUESTION_TEXT___`/`___QA_QUESTION_TEXT_END___` 正文边界完整时，可恢复缺失的冗余 `___QA_QUESTION_END___`
- AA output 已有合法 round JSON，且 sources/gaps/answer 三个区段边界完整时，可恢复缺失的冗余 `___AA_OUTPUT_END___`
- Marker JSON 只解析正文中的第一个完整、括号平衡的 JSON 对象；其后的 Markdown 可安全包含 `batch={...}`、代码示例或其他花括号
- 若模型适配层错误地将完整协议块 + LOOP 仅放入 stream-json thinking block，且可见 text/result 不含完整协议响应，编排器可只提取并规范化该协议块与 LOOP；不得转发 thinking 中的分析正文
- 若 Agent 只输出内部步骤说明并停在 `GOTO`、未产生协议块与 LOOP，编排器在同一 session 内发送一次 `[PROTOCOL_REPAIR]`；Agent 必须保持内部状态并继续到下一个 YIELD/TERMINATE。修复后仍无完整协议才终止运行
- 编排器解析后使用 formatter 规范化转发，补齐外层 end marker；缺失正文 end marker 的真正截断输出仍视为协议错误

### 8.4 ConversationState

```typescript
{
  protocol_version: 8,
  idea_note_path, idea_note_title,
  paper_title, paper_subdir, // AA 初始化确认的 canonical 信息
  qa_session_id, aa_session_id,
  round: number,
  qa_history: [{round, question, answer}],
  qa_loaded_references: string[],
  qa_next_entry: string,   // QA 下一次输入到达后的执行起点
  aa_next_entry: string,   // AA 下一次输入到达后的执行起点
  final_judgment: any | null,
  started_at, updated_at
}
```

### 8.5 LOOP 标记规范

Agent 在协议块外输出（编排器单独解析，不参与 marker 匹配）：
```
[LOOP: §NEXT_SECTION | await=<下一输入类型> | key=value]
```

编排器提取后保存到 state，下次转发时与实际 marker 联合翻译为具体执行语义。每次 Agent 响应（包括终止 judgment）缺失 LOOP 都是协议错误，编排器不得静默沿用旧起点。

若 checkpoint 已保存 `[LOOP: §TERMINATED | done]`，但因旧解析错误未保存 `final_judgment`，`--resume` 只能从 `QA_raw.jsonl` 恢复最终 judgment、写入 review 并退出；不得重新启动已终止的 agent session。

未终止 checkpoint 支持两种自动恢复：
- `qa_history.length === round - 1` 且 QA checkpoint 为 `await=AA_OUTPUT`：AA 在当前 round 回答中断。编排器从 `QA_raw.jsonl` 恢复该 round 的最后一个完整问题，用 Claude CLI `--resume <session-id>` 恢复 AA，并发送禁止工具调用的协议 repair。
- `qa_history.length === round` 且 QA checkpoint 为 `await=AA_OUTPUT`：AA 回答已保存但 QA 后续处理中断，恢复原 QA session 并继续后续控制流。

repair 后必须再次验证完整协议块与 LOOP；再次失败时错误需区分空响应、纯叙述响应、缺 marker、缺 LOOP，并附带可用的 token/cost telemetry。不得再把所有 repair 失败统一报告为 `missing LOOP`。

AA 回答阶段的稳定性约束：
- 每个 `§ANSWER` round 最多 2 次工具调用；完成第 2 次调用后禁止再计划或发起检索，立即以 information gaps 收束。
- 编排器在每轮 AA anchor 中重复注入该 2 次上限；若 `aa_no_tools=true`，则改为注入“禁止工具、直接回答”的恢复约束。
- 回答正文以 30 行为精炼目标而非硬门槛；禁止为行数反复计数/重写，完整输出 `AA_OUTPUT + LOOP` 的优先级最高。

---

## 9. Skill 文档结构约定

两个 skill 统一按以下顺序：

```
1. 你的角色（身份 + 核心约束，3-5 句）
2. 工作 loop（用 § 标签命名的显式流程，引用末尾约束）
3. 协议格式（你接收什么、输出什么）
4. LOOP 标记格式（每次输出必须附带）
5. 约束速查表（完整规则罗列，模型需要时查阅）
```

不描述编排器行为，只描述自己的行为。
