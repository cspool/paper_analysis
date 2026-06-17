---
name: learning-experiment-from-notes-answer
description: Answer Agent —— 接收单个问题，通过问题的逻辑拆解 + obsidian api 搜索 + 读取笔记加入上下文，按该层粒度要求具体回答，写入<qid>_<lid>_answer.md 后结束。由 scheduler.ts Phase 2 worker 线程派发。
---

# Learning Experiment from Notes — Answer Agent

只读。用中文回答。不要人为缩短。每个结构化示例（公式/伪代码/流程图/时序图）后必须跟随「注解」节。

## 本地搜索后端硬限制

- **所有 vault 笔记检索只能通过 Obsidian API 完成**：搜索使用 `obsidian_search_notes`，读取使用 `obsidian_get_note`。
- 禁止使用文件系统搜索或目录遍历作为证据检索手段，包括但不限于 `rg`、`grep`、`find`、`ls`、Python 脚本扫描、shell 通配符扫描。
- 允许使用 Web/联网搜索作为**外部补充证据**，但不能替代本地 Obsidian API 搜索；若 Obsidian API 未命中，必须先记录「该链条节点/关键词无 note evidence」，再把 Web 结果单独标注为「Web evidence」。
- Obsidian API 搜索范围视为仅覆盖 vault 中的 Markdown 笔记；omnisearch query 必须同时使用 `path:<绝对目录>` 限定搜索范围。
- 文件系统只允许用于读取调度器传入的问题空间文件、写入答案文件，以及必要的非证据性状态检查；不能用来替代 Obsidian API 搜索笔记内容。

本 skill 是 **Answer Agent**，负责读取 vault 笔记并具体回答单个问题。**输出到文件后结束。** 答案文件必须 ≥100 行。

## 输入

scheduler.ts 将参数直接嵌入此段（`fillSkillInput`），替换本文档中本段占位内容：

- 问题 ID: <qid>
- 层: <lid> <层名称>
- 问题空间文件: <work-dir>/<lid>_问题空间.md
- 输出文件: <work-dir>/<qid>_<lid>_answer.md
- 模型负载: <用户指定>
- 后端平台: <用户指定>
- 请求模式: <用户指定>
- 计算场景: <用户指定>
- 侧重: <侧重标签>
- 侧重配置: <JSON, 含 label/primary/secondary>
- 完成后在输出文件末尾写入 [ANSWER_AGENT_DONE] <qid>

## Workflow

### Step 1: 语义分割 + vault 搜索

#### 1.1 读取问题文本和预关键词

从问题空间文件中提取该问题的预关键词。

#### 1.2 语义分割

将问题先拆成**逻辑链条**，再对链条中的每个部分做语义分割和检索，以便获取更准确的上下文知识。每个链条节点构造一个**长到短的搜索阶梯**：先用能表达该节点完整语义的长匹配 query 做高精度召回，再逐步降级到短语和单关键词召回。

逻辑链条拆分要求：
- 先识别问题的逻辑角色，例如：`对象/模型负载`、`后端/硬件平台`、`请求模式/计算场景`、`核心机制`、`约束/指标`、`需要回答的实现或实验问题`。
- 按因果、条件、对比、递进、并列、目标-手段等关系拆成 `S1/S2/S3...` 链条节点。
- 每个节点必须记录：`节点ID`、`原文片段`、`逻辑角色`、`核心概念`、`长匹配 query`、`降级 query 列表`。
- 后续搜索和上下文读取必须以节点为单位进行，不能只对整道问题做一次统一搜索。
- 如果某个节点没有笔记证据，也要保留该节点，并在答案中说明该节点证据不足。

示例：

| 节点 | 原文片段 | 逻辑角色 | 搜索重点 |
|------|----------|----------|----------|
| S1 | GPU/NPU 上的并发 kernel | 后端/计算场景 | `"concurrent kernel"`、`GPU occupancy`、`NPU` |
| S2 | compute density 和 tile selection | 约束/指标 | `"compute density"`、`"tile selection"`、`tiling` |
| S3 | 如何实现和实验评估 | 实现/实验问题 | `implementation`、`benchmark`、`profiling` |

关键词提取规则：
- 长匹配 query 只用于同一个逻辑链条节点内，最多包含 2-4 个核心概念；不要把整个问题或跨节点关键词全部拼在一起。
- 多词固定短语用英文双引号包起来，例如 `"GPU occupancy"`、`"concurrent kernel"`、`"tile selection"`。
- 对同一概念生成中英文/缩写/连字符变体，并分别搜索，例如：`occupancy`、`GPU occupancy`、`"concurrent kernel"`、`并发 kernel`、`tile selection`、`tiling`。
- 如果长匹配零命中或命中太少，必须拆短降级；不要因为长匹配失败就判定没有笔记证据。

#### 1.3 六目录 Obsidian API 并行搜索

对每个逻辑链条节点，在且仅在六个绝对目录分别执行**长匹配优先、逐级降级**搜索。所有搜索必须调用 `obsidian_search_notes`；不得用文件系统搜索替代。**omnisearch 模式通过 `path:` 内嵌查询字符串限定绝对目录**（该模式没有独立的 `pathPrefix` 参数，`pathPrefix` 仅 `text` 模式可用）。`path:` 放在 query 开头，便于 Omnisearch 先应用目录和 Markdown 文件过滤。

搜索必须保留节点映射：

| 节点 | Query Level | Dir | Query | Path | Score |
|------|-------------|-----|-------|------|-------|
| S1 | A | /data3/paper_analysis/paper_secs | `path:/data3/paper_analysis/paper_secs  "concurrent kernel" "GPU occupancy"` | ... | ... |
| S2 | C | /data3/paper_analysis/knowledge_notes | `path:/data3/paper_analysis/knowledge_notes  tiling` | ... | ... |

每个目录都按以下顺序搜索：

**Level A：长匹配 / 组合语义优先**

先搜索同一逻辑链条节点内的组合 query，保留最多 5 条高分结果：

```
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  \"GPU occupancy\" \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/knowledge_notes  \"GPU occupancy\" \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/experiment_notes  \"GPU occupancy\" \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/idea_notes  \"GPU occupancy\" \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  \"GPU occupancy\" \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  \"GPU occupancy\" \"concurrent kernel\" tiling")
```

长匹配 query 的原则：
- 只组合语义强相关的概念，例如 `occupancy + concurrent kernel + tiling`。
- 不混入另一层/另一问题的概念，例如不要把 `GPU occupancy`、`NPU compute density`、`tile selection`、`serving scheduler` 全部拼成一个 query。
- 长匹配有结果时优先保留；但如果单目录结果少于 3 条，或全局去重后少于 15 条，继续执行后续降级搜索。

**Level B：中等短语匹配**

将长 query 拆成 2 个概念左右的短语组合：

```
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  \"GPU occupancy\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/knowledge_notes  \"GPU occupancy\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/knowledge_notes  \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  \"GPU occupancy\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  \"concurrent kernel\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  \"GPU occupancy\" tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  \"concurrent kernel\" tiling")
```

**Level C：单概念 / 变体匹配**

再逐个关键词或短语搜索：

```
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  \"GPU occupancy\"")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  occupancy")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  \"concurrent kernel\"")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/paper_secs  tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  \"GPU occupancy\"")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  occupancy")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  \"concurrent kernel\"")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/human_notes  tiling")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  \"GPU occupancy\"")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  occupancy")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  \"concurrent kernel\"")
obsidian_search_notes(mode="omnisearch", query="path:/data3/paper_analysis/learning_outputs  tiling")
```

搜索目录：/data3/paper_analysis/paper_secs/、/data3/paper_analysis/knowledge_notes/、/data3/paper_analysis/experiment_notes/、/data3/paper_analysis/idea_notes/、/data3/paper_analysis/human_notes/、/data3/paper_analysis/learning_outputs/

每目录每 query 至多 5 条结果。omnisearch 上游硬上限 50 条，用 `-exclusion` 和 `path:`/`ext:` 过滤缩小范围。

#### 1.3.1 零命中/低命中降级策略

如果某个逻辑链条节点在六个目录的 Level A/Level B 结果不足，按下面顺序降级：

1. 搜同义/缩写/中英文变体，例如 `"concurrent kernel"` → `kernel concurrency` → `并发 kernel`。
2. 拆成更短的单概念 query，例如 `"GPU occupancy"` → `occupancy`，`"tile selection"` → `tiling`。
3. 如果 omnisearch 仍为零，仍只能调用 Obsidian API；切换到 `text` 模式，并用 `pathPrefix` 限定目录：
   ```
   obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="/data3/paper_analysis/paper_secs")
   obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="/data3/paper_analysis/knowledge_notes")
   obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="/data3/paper_analysis/experiment_notes")
   obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="/data3/paper_analysis/idea_notes")
   obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="/data3/paper_analysis/human_notes")
   obsidian_search_notes(mode="text", query="<keyword>", pathPrefix="/data3/paper_analysis/learning_outputs")
   ```
4. 仍无本地结果时，记录「该链条节点/关键词无 note evidence」。不要使用文件系统搜索或无关本地搜索结果补证；如需补充，用 Web 搜索并在答案中单独标注为「Web evidence」。

#### 1.4 去重合并

合并所有搜索结果，按 omnisearch 分数降序去重。保留分数最高的 15-20 条。

去重时必须保留反向索引：
- 一个笔记路径可以对应多个链条节点和多个 query。
- 记录每个路径命中了哪些节点、哪些 query、最高分数和命中目录。
- 后续回答时，每个逻辑链条节点至少尝试引用 1-3 条相关笔记；如果没有相关笔记，明确写「该链条节点无 note evidence」。

### Step 2: 读取上下文笔记

使用 `obsidian_get_note` 逐条读取去重后的笔记路径。**`target` 参数为 discriminated union，必须通过 `{type: "path", path: "..."}` 指定绝对路径**：

- 默认读取全文：
  ```
  obsidian_get_note(format="content", target={type: "path", path: "<absolute-note-path>"})
  ```
- 笔记较长且内容集中在特定标题下时，用 section 精准读取（嵌套标题用 `::` 分隔）：
  ```
  obsidian_get_note(format="section", target={type: "path", path: "<absolute-note-path>"}, section={type: "heading", target: "<标题名>"})
  ```
- 需要 frontmatter 元数据时：
  ```
  obsidian_get_note(format="full", target={type: "path", path: "<absolute-note-path>"})
  ```
- 需要发现可用标题/block/frontmatter 目标时，先用 document-map：
  ```
  obsidian_get_note(format="document-map", target={type: "path", path: "<absolute-note-path>"})
  ```

读取后按逻辑链条节点组织上下文：

```markdown
## 检索上下文映射
### S1: <逻辑角色/原文片段>
- `<absolute-path>` (score: X.X, query: `...`): <与 S1 相关的证据>

### S2: <逻辑角色/原文片段>
- `<absolute-path>` (score: X.X, query: `...`): <与 S2 相关的证据>
```

答案生成时必须先覆盖各链条节点，再做综合回答，避免只根据某一类搜索结果回答整题。

### Step 3: 按层次粒度要求具体回答

**这是最重要的步骤。不同层次必须有对应粒度的具体内容：**

#### L1 算法 Pipeline 粒度

**方法必须具体到伪代码或计算过程**：
- 完整的伪代码（输入张量形状、输出、循环结构）
- 每个算子标注维度变化和并发机会
- 每个代码块后跟随「注解」节（变量含义、复杂度分析、数据依赖、并发可行性）

#### L2 Serving 调度 / L3 编译框架 粒度

**必须具体到框架运行模拟的例子**：
- L2：调度框架执行模拟（步骤 1..N，含伪时间线、SM 分区、Dispatcher 逻辑）
- L3：编译流程模拟（IR 转换链 → pass pipeline → codegen 输出）
- 每个模拟示例后跟随「注解」节（数据结构、并发粒度、编译 vs 运行时权衡）

#### L4 Kernel 调度 粒度

**必须具体到伪代码实现 + 指令和 pipeline 编排**：
- Kernel 伪代码（grid/block 配置、shared memory tile、register 分配）
- 指令 Pipeline 时序图（T0/T1/T2 的时间重叠展示）
- 每个示例后跟随「注解」节（tile 约束、CTA 分配、occupancy、bank conflict 避免）

#### L5 硬件架构 粒度

**必须具体到数据流设计和计算/控制模块设计和功能说明**：
- Mermaid flowchart 数据流路径（HBM→L2→L1→RF→Tensor Core 完整路线）
- 控制模块详解表（Warp Scheduler、Scoreboard、TMA、MIG 等，每行含功能描述和并发支持说明）
- 每个图/表后跟随「注解」节（MAC 数量、带宽数值、功耗、流水线级数）

#### L6 芯片设计 粒度

**必须具体到设计和评估**：
- Mermaid flowchart 芯片拓扑（Chiplet 互联、Die-to-Die 接口、PIM 集成位置）
- 评估表（工艺节点、die 面积、功耗、互联协议、带宽、TOPS/W）
- 每个图/表后跟随「注解」节（工艺约束、热约束、良率影响）

### Step 4: 工具和 notes 引用来确保证据

- 每个方法必须标注 vault 笔记路径和 omnisearch 分数
- 区分「笔记显示」「可推断」「Web 显示」；Web 只能作为外部补充，不能伪装成本地笔记证据
- 笔记未说明则写「笔记未明确说明」，不编造

### Step 5: 按侧重组织答案

| 侧重标签 | 输出节顺序 |
|----------|-----------|
| 硬件架构和运行时 | 方法细节（数据流/架构/运行时调度）→ 实验环境 → 实现(aux) → 是什么？(aux) → 方法对比表 |
| 编译框架 | 方法细节（编译流程）→ 实现 → 实验环境(aux) → 是什么？(aux) → 方法对比表 |
| Kernel 调度 | 方法细节（kernel/pipeline）→ 实现 → 实验环境(aux) → 是什么？(aux) → 方法对比表 |
| 实验和实现 | 实验环境 → 实现 → 方法细节(aux) → 是什么？(aux) → 方法对比表 |
| 方法和创新 | 方法细节 → 是什么？ → 实现(aux) → 实验环境(aux) → 方法对比表 |
| 全栈均衡 | 方法细节 → 是什么？ → 实现 → 实验环境 → 方法对比表 |

- 主内容完整展开（≥3 条笔记证据 + 达到该层粒度要求）
- 辅内容简要概述（标注 `(辅助说明)`，≥1 条笔记证据）

### Step 6: 写入答案文件

```markdown
# <QID> — <问题摘要>

## 笔记证据概览
| 路径 | 分数 | 关键信息 |
|------|------|----------|
| /data3/paper_analysis/paper_secs/... | 15.2 | ... |

---

## <按侧重顺序的第一节>

### 方法1: <方法名>
**笔记证据**: `<absolute-path>` (score: X.X)
**方法细节**（达到该层粒度要求的伪代码/模拟/指令pipeline/数据流）:
...
**注解**: ...
**实验环境**: ...

---

## 方法对比表
| 方法 | 核心机制 | 实现框架 | 硬件平台 | 关键指标 | 笔记证据 |
|------|----------|----------|----------|----------|----------|
| ... | ... | ... | ... | ... | ... |

## 不确定性
<证据缺失、推断不确定之处>

[ANSWER_AGENT_DONE] <qid>
```

## Mermaid 语法安全规则

1. 始终双引号节点文本和边标签
2. 禁止字符：`^`→`#Hat;`，`×`→`x`，`&`→`&amp;`，`<`/`>`→`&lt;`/`&gt;`
3. 节点 ID 仅字母数字，多行用 `<br/>`

## 公式指南

- 块级 `$$...$$` 单独行，行内 `$...$`
- ASCII 变量名，`\mathrm{Label}` 或 `\operatorname{name}`

## 质量自检

- [ ] 问题文本 + 预关键词完整读取
- [ ] 已按逻辑链条拆成 S1/S2/S3...，每个节点记录原文片段、逻辑角色、核心概念和 query 阶梯
- [ ] 每个逻辑链条节点已按六目录执行长匹配优先搜索
- [ ] 本地搜索和笔记读取仅使用 Obsidian API；未使用 `rg`/`grep`/`find`/文件系统扫描进行本地证据检索
- [ ] 长匹配低命中时已降级到中等短语、单概念/变体、必要时 text 模式
- [ ] 上下文按链条节点组织，保留节点→query→path→score 映射
- [ ] 去重后读取分数最高的 15-20 条笔记
- [ ] 答案文件 ≥100 行
- [ ] **方法细节达到该层粒度要求**（L1 伪代码/L2-L3 框架模拟/L4 指令pipeline/L5 数据流+控制模块/L6 芯片拓扑+评估）
- [ ] 每个结构化示例后跟「注解」节
- [ ] 主内容 ≥3 条笔记证据，辅内容 ≥1 条
- [ ] 笔记证据标注绝对路径和 omnisearch 分数
- [ ] Web 补充证据如有使用，已单独标注链接，且未替代本地 note evidence
- [ ] 不编造 — 笔记未说明则写「笔记未明确说明」
- [ ] `[ANSWER_AGENT_DONE]` 在文件末尾
- [ ] 中文回答
