## Decoding-Time KV Cache Compression (解码阶段KV Cache压缩)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Decoding-Time KV Cache Compression 是 R-KV 提出的压缩时机范式：与现有方法（SnapKV, PyramidKV 等）在 prefill 阶段一次性 select 并固定 KV cache 不同，R-KV 在 autoregressive decoding 过程中周期性（每 B_buffer tokens）触发压缩。这一设计源于推理模型的特性——generation output 远长于 input prompt（R1-Llama-8B 的 AIME24 平均生成 ~15.5K tokens vs prompt 仅 ~300 tokens），因此主要的 KV cache 增长和冗余发生在 decoding 阶段而非 prefill 阶段。

机制：(1) 分配固定大小的 cache budget B_budget 和 buffer B_buffer（用于存储新生成 tokens）；(2) 每生成 B_buffer 个 token 后触发压缩，将现有 cache（B_budget tokens）+ buffer（前 B_buffer−α tokens）合并为 n = B_budget + B_buffer − α 个候选 KV tokens；(3) 通过 joint selection score（Z = λ·I − (1−λ)·R）选出 top B_budget tokens 保留；(4) 始终保留最后 α 个 observation tokens。

从算法pipeline角度拆解：

```
# Decoding-Time Compression 时序
# 超参: B_budget=1536, B_buffer=128, alpha=8, lambda=0.1

Timeline:
t=0:    生成 [prompt 处理] → KV_cache ← prefill阶段的完整KV
t=128:  触发压缩#1: cache=1536 + buffer=128 → 选top 1536 + α=8 = 1544
t=256:  触发压缩#2: 再次压缩(1544 → 1536 + α)
t=384:  触发压缩#3: ...
...
压缩周期: 每B_buffer=128 tokens一次
压缩操作: n=B_budget+B_buffer-α=1656候选 → B_budget=1536保留 + α=8 obs
```

与 prefill-time compression 的对比：
- Prefill-time (SnapKV): prefill 阶段一次性 select → decode 阶段复用固定 KV，无运行时压缩 → 缺点：prefill 时 unknown 后续 generation 的注意力分布，可能错误淘汰将在 long CoT 中关键的 token
- Decoding-time (R-KV): decode 过程中周期性 re-evaluate → 每次压缩基于最新 observation tokens 的 attention → 动态适应生成过程中的注意力变化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Decoding-time compression 的实现核心是内存管理：需预先分配 B_budget + B_buffer 的 KV cache 内存（而非 FullKV 的动态增长），避免频繁的 GPU memory allocation。在 PyTorch/HuggingFace 中通过 pre-allocate fixed-size tensors 实现。当前局限性：R-KV 的 decoding-time compression 与 paged attention（vLLM 的 core KV cache 管理机制）不兼容，因为 paged attention 的物理 page 分配是动态的且无 compression 专用接口（Appendix D）。需 serving framework 提供 dedicated KV compression API 支持 efficient memory reallocation。与 training-time compression（如 LoRA-based KV reduction、RL 训练产出更短 CoT）正交，可叠加使用（R-KV 作为 inference-time 加速 + training 产出更少冗余）。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

---
