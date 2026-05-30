## Iteration-Level Scheduling / Selective Batching（迭代级调度）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Iteration-Level Scheduling 是 LLM serving 系统中的细粒度请求调度策略，由 Orca 系统首次提出。传统 serving 的 request-level batching 将不同请求 padding 到相同长度后批处理——padding token 浪费计算、已完成生成但未结束的请求需等待同批次慢请求（head-of-line blocking）。Iteration-level scheduling 在每次 decoder iteration 粒度上动态决定参与批处理的请求集合：新请求可立即加入、已完成请求立即退出（返回给用户），无需等待同批次其他请求。配合 Selective Batching 技术，允许不同请求在同一 batch 内处理不同数量的 token。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。

```
Iteration-Level Scheduling (Orca):

请求队列:
  R1: [Prompt 100 tok | output 必须 ≥ 50 tok]
  R2: [Prompt 30 tok  | output 必须 ≥ 20 tok]
  R3: [Prompt 200 tok | 刚到达，待处理]

时间线 (每步 = 1 iteration ≈ 1 token generation):

iter_0 (prefill):
  Batch: [R1_prefill(100 tok)]
  R1: prefill完成, KV cache建立
  
iter_1:
  Batch: [R1_decode, R2_prefill(30 tok)]
  R1: 生成 token_1
  R2: prefill完成
  
iter_2:
  Batch: [R1_decode, R2_decode, R3_prefill(200 tok)]
  R1: token_2; R2: token_1; R3: prefill完成

iter_3..20:
  R2 在 iter_20 生成 EOS → 立即返回 → 从batch中移除
  R3 在 iter_3 即时加入batch（无需等待R2完成）

iter_4..50:
  Batch: [R1_decode, R3_decode]  (R2已退出)

iter_50: R1生成EOS → 返回 → Batch: [R3_decode] 继续

关键：每iteration重新选择batch成员，消除padding和等待。
```

进化形式：(i) SARATHI/DeepSpeed-FastGen 的 chunked-prefill——将长 prefill 拆分为多个 chunk，与 decode 交替调度，减少 prefill 对 decode 延迟的干扰；(ii) Splitwise 的 prefill-decode disaggregation——将 prefill（compute-heavy）和 decode（memory-heavy）分配到不同机器；(iii) FastServe 的 iteration-level preemptive scheduling + proactive KV cache swapping。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现于 vLLM（iteration-level scheduler + block manager）、Orca（首个提出 selective batching）、SARATHI（chunked-prefill schedule）、DeepSpeed-MII（DeepSpeed-FastGen dynamic split-fuse）。核心要求：(i) 调度器可在每 iteration 动态调整 batch 成员；(ii) KV cache 管理需支持随机位置的新请求插入（PagedAttention 天然支持）；(iii) attention kernel 需支持变长序列（无需 padding）。现代 LLM serving 框架普遍采用 iteration-level scheduling 作为基础调度原语。

涉及论文标题：
- A Survey of Resource-efficient LLM and Multimodal Foundation Models
