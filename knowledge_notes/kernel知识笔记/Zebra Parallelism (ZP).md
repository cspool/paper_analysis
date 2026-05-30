## Zebra Parallelism (ZP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Zebra Parallelism (ZP) 是 HeterMoE 提出的面向异构 GPU 集群的 MoE 训练并行策略，替代传统 Expert Parallelism (EP)。在 ZP group 中，expert 模块分布在 N 个 expert GPU（older generation），attention blocks 和其余模块（gate、embedding）复制在 M 个 attention GPU（newer generation）。ZP 将 input batch 分为 R 个 microbatch，attention GPU 和 expert GPU 同时处理不同 microbatch，形成 "zigzag" 式的跨 GPU 流水线。与 Pipeline Parallelism 在 layer 级别切分不同，ZP 在单个 transformer layer 内部切分 attention 和 expert 模块到不同 GPU。

ZP 的关键特征：(1) 不引入额外通信——EP 本就通过 all-to-all 交换 token，ZP 仅将 exchange 从 "attention GPU↔attention GPU" 变为 "attention GPU↔expert GPU"，数据总量不变；(2) 每 GPU 内 3 个 CUDA stream（2 通信 + 1 计算）并行执行 dispatch、combine 和 compute，dispatch 和 combine 方向相反在独立 stream 上不发生带宽竞争；(3) Theorem 1 证明了最优 task ordering——bipartite 通信（M 个 attention GPU 与 N 个 expert GPU 之间）的 ZP schedule 为最小化总迭代时间的最优调度。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
ZP group: M attention GPUs + N expert GPUs, R microbatches

Optimal Forward Schedule (Theorem 1):
  Attention GPU compute: (A_{1,1}^F...A_{1,R}^F)...(A_{L,1}^F...A_{L,R}^F)
  Expert GPU compute:    (E_{1,1}^F...E_{1,R}^F)...(E_{L-1,1}^F...E_{L-1,R}^F)

  A_{i,j}^F: layer i, microbatch j, attention forward
  E_{i,j}^F: layer i, microbatch j, expert forward

Stream Architecture (per GPU):
  Stream 0 (compute): attention/expert 计算
  Stream 1 (comm D):  dispatch all-to-all (Attn→Exp 方向)
  Stream 2 (comm C):  combine all-to-all  (Exp→Attn 方向)
  Sync via CUDA events between streams

依赖约束:
  t(A_{i,j}^F) ≥ t(C_{i-1,j}^F) + T_C            (数据依赖)
  |t(A_{i,j}^F) - t(A_{i',j'}^F)| ≥ T_A   (stream 顺序执行)

Overlap 示例 (R=3, forward):
  Attn GPU: [Disp0][A_{1,0}^F][Comb0][Disp1][A_{1,1}^F]...
  Exp GPU:  [==== E_{1,0}^F ====][Disp1][E_{1,1}^F]...
  // Dispatch 和 Combine 方向相反，在独立 stream 上无带宽竞争
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 基于 PyTorch v2.2 + DeepSpeed v0.14（3K 行 Python）
- ZP engine 初始化时创建 3 CUDA stream，为每个 microbatch 分配 receive buffer
- 创建分离的 NCCL dispatch/combine all-to-all group
- 通过 PyTorch NCCL all-to-all wrapper 传入不等 split size（因 Asym-EA 可能导致不同 GPU 处理不同数量 tokens）
- Gate backward 特殊处理：gate 的 top-k confidence scores 形式 "residual" 连接，backward 分两路传播——一路经 confidence scores 到 gate weights，另一路经 expert outputs。HeterMoE 在 attention outputs 处停止第二分支的 backward，等 expert GPU 梯度后 accumulated
- ZP 可与 data parallelism 组合（多 ZP group 间做 DP）
- 论文声明将开源（截至分析时未找到公开代码仓库）

涉及论文标题：
- HeterMoE: Efficient Training of Mixture-of-Experts Models on Heterogeneous GPUs
