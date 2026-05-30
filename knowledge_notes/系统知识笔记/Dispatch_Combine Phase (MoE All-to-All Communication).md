## Dispatch/Combine Phase (MoE All-to-All Communication)

术语解释
Dispatch Phase 和 Combine Phase 是 Expert Parallelism 中两次 all-to-all 通信的阶段名称：Dispatch 阶段将 token 从 gate 所在 GPU 发送到 expert 所在 GPU；Combine 阶段将 expert 输出从 expert 所在 GPU 拉回原 GPU 重构序列。

术语是什么？
在 MoE 的 Expert Parallelism 中，每个 MoE layer 需要两次 all-to-all 通信：
- **Dispatch Phase**: gate network 输出每个 token 的目标 expert 后，token 通过 all-to-all scatter 发送到持有对应 expert 的 GPU。通信模式是全对全的——每个 GPU 可能向每个其他 GPU 发送 token。
- **Combine Phase**: expert FFN 计算完成后，每个 GPU 上的 token 输出需要通过 all-to-all gather 拉回其原始 GPU，以便重构序列并继续执行下一 block 的 attention 层。

从系统架构角度拆解术语：
一次 MoE Layer 中 Dispatch 和 Combine 的完整时序：

```
Timeline (4 GPU, 8 sequences):

GPU 0: |-- Attention --|-- Gate --|====== Dispatch All-to-All ======|
GPU 1: |-- Attention --|-- Gate --|====== Dispatch All-to-All ======|
GPU 2: |-- Attention --|-- Gate --|====== Dispatch All-to-All ======|
GPU 3: |-- Attention --|-- Gate --|====== Dispatch All-to-All ======|

GPU 0: |== Expert 0 FFN ==|====== Combine All-to-All ======|
GPU 1: |== Expert 1 FFN ==|====== Combine All-to-All ======|
GPU 2: |== Expert 2 FFN ==|====== Combine All-to-All ======|
GPU 3: |== Expert 3 FFN ==|====== Combine All-to-All ======|

GPU 0: |-- Next Attention (序列已重构) --|
GPU 1: |-- Next Attention (序列已重构) --|
...

通信瓶颈:
  Dispatch + Combine 占 training iteration time 的 18.1%-47.5%
  (取决于 model, batch size, expert 数量)
```

术语一般如何实现？如何使用？
- 底层通信原语：NCCL all-to-all（GPU）、PyTorch distributed
- 优化方向：
  - LUFFY Sequence Migration: 改变 Combine 的目标 GPU，减少跨 GPU 拉取
  - LUFFY Token Condensation: 减少 Dispatch 的 token 数量
  - Lina: all-to-all 与 allreduce 的优先级调度 + micro-op pipelining
  - DeepSpeed-MoE: Hierarchical All-to-All（节点内+节点间分层）
  - Tutel: 自适应并行度 + all-to-all overlap
  - ExFlow: 通过共享 context 将两次 all-to-all 减少为一次

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---
