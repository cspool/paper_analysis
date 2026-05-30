## Tensor-Vector Parallelism Scheduling (ILP/TLP for Softmax)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Tensor-Vector Parallelism是FlashAttention-T提出的调度范式：在fused attention kernel中，将softmax计算拆分为tensorized部分（executed on Tensor Core via repurposed MMA）和vectorized部分（executed on CUDA Core），并通过架构特定的调度技术使两部分并行执行，充分利用GPU异构计算资源。与FlashAttention-3的GEMM-Softmax pipeline（仅overlap GEMM和softmax，softmax全在vector unit）和Pingpong scheduling（两个warpgroup的GEMM和全vector softmax交替）不同，FlashAttention-T的tensor-vector parallelism在同一softmax计算内部实现了tensor unit和vector unit的并行。

两种架构特定实现：

**Ampere ILP (Instruction-Level Parallelism) Scheduling**：
- Split策略：Horizontal split（同一tile内按行分割tensorized/vectorized部分，ratio≈1:1）或Vertical split（跨tile分割，ratio固定1:1，因warp register capacity限2 tiles）
- ILP interleaving：在warp内均匀交叉repurposed tensor MMA指令和vector指令，使vector指令在tensor MMA的issue bubble中执行
- 效果：t'_softmax < t_vec（baseline全vector softmax时间），t'_vec = t'_softmax - (t_vec - t'_softmax)

**Hopper TLP (Thread-Level Parallelism) Scheduling**：
- Split策略：仅tensorize P̃ row-summation（leaf-stage nature最小化跨stage register dependency，避免WGMMA serialization）
- TLP overlap：将repurposed WGMMA row-sum加入下一iteration的QK^T+PV WGMMA batch，与另一warpgroup的vector S/O rescaling并行
- 效果：vector interval ratio降至2.7%，远优于ILP（因WGMMA允许更灵活的dynamic tensor-vector overlap，不受static instruction ordering约束）

从kernel调度角度拆解术语：

Ampere ILP scheduling的timeline（图7a, per warp）：
```
// 原FlashAttention-2 iteration:
| QK^T MMA |---- vector softmax (t_vec) ----| PV MMA |
           | max | exp | mul | add | rowsum |         ← 全部vector unit

// FlashAttention-T ILP iteration:
| QK^T MMA |--- t'_softmax (tensor+vector interleaved) ---| PV MMA |
           | tensorized: scaling, FMA, rowsum (repurposed MMA) |
           | vectorized: max (REDUX), exp (MUFU.EX2)           |
           | ← ILP overlap → |
Vector interval: t'_vec = t'_softmax - (t_vec - t'_softmax)
```

Hopper TLP scheduling的timeline（图7b, 2 warpgroups, per iteration）:
```
WG1 Iter(i):   | QK^T WGMMA | PV WGMMA + rowsum WGMMA | signal WG2 → Vec(S/O rescale) |
WG2 Iter(i-1): | Vec(S/O rescale) | ← signal | QK^T WGMMA | PV WGMMA + rowsum WGMMA | ...
Tensor Unit:   |████ WG1 ████|████ WG2 ████|████ WG1 ████|
Vector Unit:   |████ WG2 ████|████ WG1 ████|████ WG2 ████|
Vector interval: t'_vec ≈ 2.7% of t'_iter
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Tensor-vector parallelism的实现需要：(1) 分析目标架构的tensor unit和vector unit的指令吞吐和issue constraints；(2) 确定split策略——搜索最优的tensorized/vectorized比例（Hopper上受nvcc compiler约束，仅tensorize leaf stages）；(3) 对于Ampere ILP：手写CUDA PTX inline assembly实现exact instruction interleaving pattern，预生成通用B fragments复用；(4) 对于Hopper TLP：使用wgmma.commit_group/wgmma.wait_group管理异步WGMMA的commit和sync，利用bar.sync协调warpgroup间的pipeline order。此概念可推广到其他具有异构计算单元的加速器：任何存在高吞吐专用单元（如tensor engine、matrix engine）和通用单元（vector ALU）且workload可被拆分的场景。

涉及论文标题：
- FlashAttention-T: Towards Fully Tensorized Attention by Exploiting Tensor-Vector Parallelism
