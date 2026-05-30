## Memory-Conscious Draft Model Execution（内存感知的 Draft 模型执行）

术语解释
Memory-Conscious Draft Model Execution 是 SpecMoEOff 针对 speculative decoding 中 draft model KV cache 也需要 offloading 的问题提出的解决方案。在 throughput-oriented MoE offloading 场景下（large batch + long sequence），draft model 虽然参数量小（<2GB），但其 KV cache 随 batch size 和 sequence length 线性增长，可能超过 GPU HBM 容量（如 142GB target KV cache + 17.75GB draft KV cache > 24GB GPU）。

术语是什么？
核心策略：按 batch 维度将 draft model 执行分离为两个并行部分：
- **GPU Part**: KV cache 完全在 GPU HBM → attention + FFN 均在 GPU 执行
- **CPU Part**: KV cache 在 CPU DRAM → attention 在 CPU 执行（类似 target model）→ hidden states 传回 GPU 做 FFN

优先级原则：draft model 的数据应优先使用 GPU HBM（因为 draft model 在每 iteration 中被调用 k 次，而 target model 仅调用 1 次——相同数据放在 GPU 对 draft model 有 k 倍的 arithmetic intensity 提升）。

从系统架构角度拆解术语：
```
# Dynamic Separate Ratio Mechanism
初始阶段（generation 开始）:
  GPU Part: more requests（seqlen 短, KV cache 小, GPU 内存充足）
  CPU Part: fewer requests

随着 generation 进行:
  seqlen 增长 → GPU memory 不足以容纳全部 draft KV cache
  → 部分 requests 的 draft KV cache 从 GPU 迁移到 CPU
  → CPU Part 增大, GPU Part 减小

请求完成时:
  GPU memory 释放 → 将 CPU Part 中 requests 的 draft KV cache 
  → 动态加载回 GPU → 提升后续 iteration 性能
```

术语一般如何实现？如何使用？
- Partition 维度选择：batch 维度（vs head/sequence）——避免 head 维度的 synchronization 开销和 sequence 维度的 partial score 组合开销
- 两部分并行执行：GPU Part 和 CPU Part 的 attention 同时执行，FFN 统一在 GPU（利用 GPU 闲置计算资源）
- 动态调整：与 Hyperparameter Optimizer 协作，per-iteration 调整 GPU/CPU 分离比例

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding
