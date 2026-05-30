## Multi-Token vs Single-Token Inference (for BLR Models)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-Token Inference（n>1, prefill阶段）和Single-Token Inference（n=1, decode阶段）是LLM推理的两种模式。在BLR压缩模型上下文中，两种模式的瓶颈性质根本不同：单token推理的memory流量由权重读取主导（o·i），激活(n·i)可忽略，压缩权重直接加速；多token推理的激活流量随n线性增长，BLR的block结构产生dense不存在的b×n×r中间张量（Monarch: 4bnr bytes, BLAST: 8bnr bytes），将compute-bound线性层推入memory-bound区域。

从算法pipeline角度拆解术语：
A40 BF16 Q/K/V/Oproj layer (i=o=4096, r=1024, b=16) 的roofline分析：

| 场景 | 方法 | FLOP | Memory Traffic | α=FLOP/Bytes | 瓶颈 |
|------|------|------|----------------|-------------|------|
| n=1 | Dense | 34M | 32KB | ~1074 | Compute |
| n=1 | BLAST | 17M | 40KB | ~425 | Compute |
| n=1024 | Dense | 34G | 34MB | ~994 | Compute |
| n=1024 | Low-Rank | 17G | 17MB | ~978 | Compute |
| n=1024 | Monarch | 17G | 138MB | ~123 | **Memory!** |
| n=1024 | BLAST | 17G | 266MB | ~64 | **Memory!** |

α阈值 ≈ 155。α_Monarch=123 < 155 → memory-bound（比dense慢1.14-1.68×）。α_BLAST=64 << 155 → strongly memory-bound（比dense慢2.63-4.31×）。尽管FLOP减半，额外数据移动反而使推理变慢。

术语一般如何实现？如何使用？
实际LLM服务中prefill和decode交替发生：prefill（n=prompt_len, 多token）→ decode（n=1, 逐token生成）。BLR模型在decode阶段天然受益于权重压缩（memory-bound→压缩直接加速），在prefill阶段需kernel级优化（如论文Triton kernel的partial fusion、memory layout optimization）将FLOP减少转化为实际加速。论文最终实现BLAST ⑤在end-to-end推理中1.13-1.48× over dense，证明多token瓶颈可以被kernel优化克服。

涉及论文标题：
- Memory-Efficient Acceleration of Block Low-Rank Foundation Models on Resource Constrained GPUs

---
