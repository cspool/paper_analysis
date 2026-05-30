## Fine-grained Expert Prefetching via Pre-gating（基于预门控的细粒度专家预取）

术语是什么？
Fine-grained Expert Prefetching 是 Read-ME 利用 pre-gating 先验信息实现的层粒度 expert 预取。传统 MoE 推理中 expert 加载无法与计算重叠（不知道下一层需要哪些 expert）。Read-ME 在计算 Layer i FFN 时异步启动 Layer i+1 所需 expert 从 host memory → GPU 的 PCIe 传输（cudaMemcpyAsync），compute stream 与 loading stream 流水线重叠，使仅有首层加载延迟进入关键路径。Prefetching 模式在受限 cache 下比 On-demand Loading 延迟低最多 30%。

从系统架构角度拆解术语：
流水线时序：

```
Timeline →
Compute:  |--L1_Attn--|--L1_Expert--|--L2_Attn--|--L2_Expert--|...
Loading:  |--Load L1--|              |--Load L2--|              |...
          ^                          ^
          仅L1在关键路径              L2加载被L1计算隐藏

vs On-demand Loading:
Compute:  |--L1..--|..wait..|--L1_Expert--|..wait..|--L2_Expert--|
Loading:  |--Load L1--|      |--Load L2--|
          每次加载均在关键路径
```

术语一般如何实现？如何使用？
- 基于 DeepSpeed 修改：MoE forward 循环中对 layer i 发起计算的同时创建独立 CUDA stream 执行 cudaMemcpyAsync(layer i+1 experts, CPU→GPU)。
- 前提：pre-gating 已确定所有层 expert 需求 + expert weights 在 host memory（CPU RAM）。
- 与 Belady Caching 结合：cache hit 的层无需 prefetch（expert 已在 GPU），仅 cache miss 触发 prefetch。
- 局限：首层加载延迟无法隐藏；若 PCIe 带宽不足，计算可能等待传输。

涉及论文标题：
- Read-ME: Refactorizing LLMs as Router-Decoupled Mixture of Experts with System Co-Design
