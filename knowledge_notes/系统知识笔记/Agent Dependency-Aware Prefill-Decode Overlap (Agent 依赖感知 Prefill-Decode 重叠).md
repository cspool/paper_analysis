## Agent Dependency-Aware Prefill-Decode Overlap (Agent 依赖感知 Prefill-Decode 重叠)

术语是什么？
一种针对 MoA serving 的系统优化技术，利用 agent 间的数据依赖关系将前驱 agent 的解码阶段与后继 agent 的 prefilling 阶段重叠执行，隐藏后继 agent 的预填延迟。核心思路：将依赖 agent 的输入 prompt 按前驱 agent 的输出槽分割为独立前缀段 + 依赖段。前缀段无数据依赖，可立即 prefill 并缓存 KV blocks；依赖段随着前驱 agent 解码出的 token chunk 流式到来，利用已驻留 HBM 的 prefix KV 做增量 prefilling（仅计算新增 token 的 KV），实现 decode-prefill 时间重叠。

从系统架构角度拆解：
以 agent 3 依赖 agent 1、agent 2 的输出为例，全流程：

```
Step 1: 依赖识别
  Shell Router 解析 agent 3 的 prompt
  → 发现两个前驱输出槽: slot_1 (agent 1), slot_2 (agent 2)
  → Prompt 分割: [prefix][slot_1][slot_2]

Step 2: 前缀立即 Prefill (无需等待)
  /prefill_only(prefix) → PE 计算 prefix KV
  → KV blocks 驻留 HBM (不传输到 DE，因为尚未解码)
  → 此阶段与 agent 1, agent 2 的解码并行

Step 3: 增量 Prefill Loop (与解码重叠)
  while slot_1 未完成:
    chunk = APC.fetch(agent_1)       // 轮询 Agent Prompt Cache
    if chunk:
      append(chunk, after prefix)     // 追加到已 prefilled 段之后
      /prefill_only(new_tokens_only)  // 仅计算新 token KV
      // prefix KV 从 HBM 复用 → ~100% KV cache hit rate
  // agent 1 解码完成 = slot_1 prefill 完成 (重叠)

Step 4: 重复增量 Prefill for slot_2
  while slot_2 未完成:
    chunk = APC.fetch(agent_2)
    if chunk:
      append(chunk, after slot_1_kv)
      /prefill_only(new_tokens_only)

Step 5: 所有 slot 完成
  agent 3 的完整输入已在重叠中被 prefilled
  → 转发 /generate 到 native PD router → 标准自回归解码
```

关键优势：在 vanilla MoA 中，agent 3 必须等 agent 1 和 agent 2 全部解码完成后才开始 prefill（串行瓶颈）；在 Faster-MoA 中，agent 3 的 prefill 被 agent 1/2 的 decode 时间大部分隐藏。

术语一般如何实现？如何使用？
- 依赖两个 API entrypoint：标准 /generate 和轻量 /prefill_only（仅计算并缓存 KV，不触发 PE→DE 传输）
- 需要 Agent Prompt Cache (APC) 存储部分解码文本/token，支持轮询式增量 prefilling
- chunk size 需权衡：小 chunk → 更多增量 prefill 请求（PE 初始化开销）；大 chunk → 更多暴露延迟
- 使用同模型家族（共享 tokenizer）时 APC 存 tokens 而非 text，避免 tokenize-detokenize 开销
- 基于 SGLang v0.5.3 实现，通过 Shell Router 编排

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap
