## Memory Wall (存储墙 / Memory Bandwidth Bottleneck in GPU Computing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Memory Wall 是描述 GPU 计算中计算吞吐（compute throughput）增速远超内存带宽（memory bandwidth）增速的现象和由此产生的性能瓶颈。在 FlashFuser 的 quantitative evidence 中：NVIDIA H100 的 FP16 peak compute 从前代 A100 的 300 TFLOPS 增长到约 1000 TFLOPS（3.3×），而 HBM bandwidth 仅从 2 TB/s 增长到 3 TB/s（1.5×），增速差约 2.2×。这使得许多 deep learning workloads（特别是 GEMM-dominated 的 FFN 层和卷积块）从 compute-bound 变为 memory-bound——算子执行受制于 HBM bandwidth 而非 compute capability。Table I 显示在典型推理配置下（seq_len=512），FFN 层在 GPT-6.7B 占 61.28%、LLaMA-1B 占 57.44% 的总执行时间，均呈现 memory-bound 特征。

从系统架构角度拆解术语：
Memory Wall 在 FlashFuser 中的影响和缓解路径：
```
H100 memory hierarchy (近核→远核, bandwidth):
  Reg: ~256KB/SM, ~20TB/s (highest)
  SMEM: ~228KB/SM, ~19TB/s
  DSM: ~0.2-3.6MB [cluster size 1-16], ~4-8TB/s ← FlashFuser 利用此层
  L2: ~50MB, ~12TB/s
  HBM: 80GB, 3.35TB/s ← 传统方法必经此层 (bottleneck)
  
FlashFuser缓解: 将中间tensor的data path从 "SMEM→HBM→HBM→SMEM" 
改为 "SMEM→DSM→SMEM", 减少58% global memory access
```

Memory wall 的具体影响：(1) GEMM chain fusion 受 SMEM 容量限制——当中间 tensor > 227KB/SM，fusion 必须失败或回退到 HBM round-trip；(2) 仅增加 compute 无法解决——即使 H100 有 1000 TFLOPS，如果 HBM bandwidth 无法匹配 data movement demand，实际 throughput 由 bandwidth 主导；(3) DSM 作为 L1.5 cache 是硬件层面的 partial solution，但其 bandwidth 仍低于 SMEM 且随 cluster size 变化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Memory Wall 是硬件层面的物理约束，解决方案跨多个层次：(1) 硬件层——增加 HBM bandwidth (HBM3→HBM3e)、增大 on-chip memory (L2 cache 增大)、引入 inter-core connection (DSM)；(2) 编译框架/kernel层——kernel fusion 减少 HBM round-trip (FlashFuser)、tiling 使 working set fit in on-chip memory；(3) 算法层——quantization (FP8/INT4) 减少 data volume、sparse attention 跳过近零计算。FlashFuser 的方法属于编译框架层的 optimization——通过利用 DSM 扩展 on-chip memory 来缓解 memory wall。

涉及论文标题：
- FlashFuser: Expanding the Scale of Kernel Fusion for Compute-Intensive Operators via Inter-Core Connection
