# AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding

- 导出时间：2026-05-21 10:56 CST
- 来源：当前 Codex 可见对话上下文
- 保存路径：conv_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md
- 说明：本记录覆盖当前可见对话上下文；若更早内容未出现在当前上下文中，则未被记录。

## 对话记录

### User
- 使用 `$obsidian-keyword-explainer`，围绕当前笔记 `idea_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md`，要求解释 AdaServe 将一次 decoding iteration 从“每个请求只生成 1 个 token”改为：
  - Speculation：小 draft model 对每个请求做 d 步 beam search，beam width 为 w，生成 candidate token tree。
  - SLO-customized selection：根据每个请求的 TPOT SLO 缺口，优先给更紧急请求分配高 path probability 节点。
  - Throughput-optimized selection：SLO 基本满足后，剩余 token budget 分给全局 path probability 最高的节点。
  - Tree-based verification：target LLM 一次并行验证所有 selected trees。
- 后续继续追问：`Speculation：draft model 做 d 步 beam search 生成 candidate token tree。Tree-based verification：target LLM 一次并行验证所有 selected trees。`
- 再次要求：`更具体`，希望进一步细化 speculation 和 tree-based verification 的执行细节。
- 最后使用 `$export-conversation-notes` 要求保存本轮可见对话。

### Codex
- 使用 `obsidian-keyword-explainer` 技能，并通过 Obsidian MCP/本地工具读取和检索 AdaServe 相关笔记。
- 读取或引用的主要笔记包括：
  - `idea_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md`
  - `experiment_notes/算法实验笔记/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md`
  - `experiment_notes/系统实验笔记/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md`
  - `knowledge_notes/算法知识笔记/Beam Search Speculation（束搜索推测）.md`
  - `knowledge_notes/算法知识笔记/SLO-Customized Token Tree Construction（SLO定制化Token树构造）.md`
  - `knowledge_notes/算法知识笔记/Speculative Decoding（投机解码）.md`
  - `knowledge_notes/系统知识笔记/Multi-SLO LLM Serving（多SLO大模型推理服务）.md`
  - `knowledge_notes/系统知识笔记/Token Budget in Speculative Decoding Serving（投机解码推理中的Token预算）.md`
  - `knowledge_notes/系统知识笔记/SLO Attainment（SLO达成率）.md`
  - `knowledge_notes/系统知识笔记/Goodput（有效吞吐量）.md`
- 第一轮详细解释中，将用户段落切分为 5 个语义单元：
  1. decoding iteration 从每请求 1 token 变为可变多 token 推进；
  2. draft model 做 d 步 beam search 生成 candidate token tree；
  3. SLO-customized selection 根据 TPOT SLO gap/slack 给紧急请求分配高概率节点；
  4. Throughput-optimized selection 将剩余 token budget 分配给全局最高 path probability 节点；
  5. Tree-based verification 由 target LLM 并行验证 selected trees。
- 解释了核心公式和直觉：
  - path probability：`p_path(v_l)=prod_j q(t_j | x,t_<j)`；
  - SLO 推进目标：`m_i=max(0, latency_elapsed_i / TPOT_SLO_i - tokens_generated_i)`；
  - slack：`slack_i = TPOT_SLO_i * tokens_generated_i - latency_elapsed_i`；
  - throughput-optimized objective：在 token budget `B` 下最大化 selected nodes 的 path probability 总和。
- 给出多个结构化例子：
  - Mermaid flowchart 展示 traditional continuous batching 与 AdaServe iteration 的区别；
  - d=3、w=2 的 beam-search candidate tree toy example；
  - SLO gap 表格示例，展示 coding/chatbot/summarization 的紧急程度；
  - Phase 1 / Phase 2 selection 的伪代码；
  - target LLM tree-based verification 的 sequence diagram 和伪代码。
- 对用户追问“更具体”时，进一步细化为：
  - candidate token tree 中每个节点可包含 `request_id`、`node_id`、`parent_id`、`token`、`depth`、`draft_prob`、`path_prob`；
  - candidate tree 不等于 selected tree，selected tree 是在 token budget 和 SLO/throughput 目标下被选中的子树；
  - 若选择深层节点，通常需要保留祖先路径，形成 ancestor-closed selected tree；
  - target LLM 的 tree-based verification 需要 tree-aware attention mask，使节点只能看见原始 prefix、自己路径上的 ancestors 和合法历史节点，不能看见 sibling 分支或其他请求节点；
  - 用两个请求 R1/R2 的完整 toy example 说明：R1 strict SLO 可在一次 verification 中接受 `arr` 和 `)` 两个 token，R2 relaxed SLO 的 draft token `the` 被拒绝并返回 correction token `a`。
- 明确区分：
  - draft model 负责“猜”；
  - scheduler 负责“选”；
  - target LLM 负责“验”；
  - 最终输出仍以 target LLM 验证结果为准。
- 使用 `$export-conversation-notes` 技能时，识别当前论文/笔记标题为 `AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding`，准备保存到 `conv_notes/`。

## 已产生的文件或修改

- `conv_notes/AdaServe_ Accelerating Multi-SLO LLM Serving with SLO-Customized Speculative Decoding.md`：保存当前可见对话记录。

## 后续待办

- 若需要进一步深化，可继续补充 AdaServe tree-based verification 的实际 attention mask / packing / FlashInfer kernel 改造细节；当前对话中已说明概念与 toy-level 执行流程，但未展开 AdaServe 具体 kernel layout。

