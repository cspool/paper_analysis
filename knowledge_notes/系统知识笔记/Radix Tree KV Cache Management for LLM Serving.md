## Radix Tree KV Cache Management for LLM Serving

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Radix Tree KV Cache Management（基数树 KV 缓存管理）是 LLM serving 系统中的一种内存管理技术，将系统内所有请求的 KV cache 组织为一棵 radix tree（基数树/前缀树），实现 multi-level prefix sharing（多级前缀共享）。在 radix tree 中，根节点存储共享的系统 prompt（如 "You are a helpful assistant"），子节点存储不同的 few-shot examples 或用户 prompt，叶节点对应各独立请求的 unique suffix。不同请求共享相同前缀时，它们在 radix tree 中共享该前缀对应的 KV cache 节点，从而避免为每个请求重复存储相同的 KV 数据。Radix tree 是一种压缩前缀树（压缩 Trie），每个节点可存储任意长度的 token 序列（而非单个字符/词），因此适合 LLM 的 token 级前缀共享。该技术在 SGLang（Zheng et al., 2023a）中首次以系统级形式提出，后续被多个系统采用（如 FastTree、DeFT、Cascade Attention）。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

Radix tree KV cache 在 SGLang + FastTree 系统中的完整运作流程：

```
1. 请求到达：
   客户端发送多个请求，共享 different-level 前缀
   例: System Prompt (A1: "You are a helpful assistant", 3193 tokens)
       → Few-shot Example Group 1 (B1: 20-shot examples)
         → Question 1.1, 1.2, ..., 1.16
       → Few-shot Example Group 2 (B2: 20-shot examples)
         → Question 2.1, 2.2, ..., 2.16

2. Radix Tree 构建（SGLang）：
   全局 KV cache 以非连续 paged blocks 存储，按 radix tree 组织：
   
        [A1: system prompt KV]        ← Root, 3193 tokens
        /                    \
   [B1: example 1 KV]   [B2: example 2 KV]  ← Level 1, 459 tokens each
      /  \                   /  \
   [Q1] [Q2] ...        [Q17] [Q18] ...       ← Level 2, unique questions

3. KV Cache 分配与复用：
   - 每个 tree node 对应一组 paged KV blocks (非连续内存)
   - 同一 prefix 的多个请求复用相同 KV blocks（引用计数管理）
   - 新请求到达：沿 radix tree 查找最长匹配前缀 → 复用已有 KV blocks
     → 仅为 unique suffix 分配新 KV blocks
   - 请求完成：引用计数减 1 → 计数为 0 时释放 KV blocks

4. FastTree 接入点：
   FastTree 读取该 radix tree 结构 → greedy heuristic 生成
   context-queries grouping plan → attention kernel 按 group 聚合计算
```

Radix tree 的关键系统优势：(1) 消除冗余 KV cache 存储——从 O(N × prefix_len) 降至 O(prefix_len + N × suffix_len)；(2) 提升内存效率 → 可服务更多并发请求 → 提升吞吐；(3) 支持动态 tree 结构——请求加入/退出时增量更新引用计数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SGLang 中的 radix tree 实现基于 paged KV cache（借鉴 vLLM 的分页机制）。Radix tree node 存储：token sequence（用于匹配）、KV block indices（指向实际 GPU 内存中的 paged blocks）、child pointers、reference count。匹配算法：新请求的 token sequence 从 root 沿 tree 遍历，匹配最长公共前缀。KV cache 管理：当 node 的 ref count 从 0 变为 1 时分配 blocks；当 node 的 ref count 降为 0 时释放 blocks。与 paged KV cache 的结合：tree node 对应的 KV blocks 可以在 HBM 中非连续（通过 block table 索引），但 tree structure 提供逻辑上的层次化组织。FastTree 开源在 https://github.com/PanZaifeng/FastTree-Artifact，其 attention kernel 利用 SGLang 维护的 radix tree 结构做 grouping 优化。

涉及论文标题：
- FastTree Optimizing Attention Kernel and Runtime for Tree-Structured LLM Inference
