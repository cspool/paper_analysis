## Expert Transfer (MoE Training Paradigm) (专家迁移)

术语解释
Expert Transfer 是分布式 MoE 训练中与 Expert Parallelism 竞争的一种通信优化范式：不通过 all-to-all 传输 token，而是将需要的 expert 参数复制到本地 GPU。代表性工作包括 Janus (SIGCOMM 2023) 和 FasterMoE (PPoPP 2022)。

术语是什么？
在标准 Expert Parallelism 中，token 通过 all-to-all 在 GPU 间移动。Expert Transfer 反其道而行：当某个 GPU 上的 token 需要访问远程 expert 时，将远程 expert 的权重参数复制到本地 GPU，在本地执行 FFN 计算。这种方法的动机是：在某些场景下 expert 参数大小可能小于需要传输的 token 总大小。

从系统架构角度拆解术语：
Expert Transfer 的执行流程：

```
=== 3 GPU, 3 Experts, Top-2 Gating ===

策略: 当 GPU 0 上的 token 需要 Expert 1,2 时:

Option A: Expert Parallelism (传 token)
  GPU 0 → GPU 1: send tokens → GPU 1 runs Expert 1 → send back results
  GPU 0 → GPU 2: send tokens → GPU 2 runs Expert 2 → send back results
  通信: all-to-all (2 × all-to-all per MoE layer)

Option B: Expert Transfer (传 expert 权重)
  GPU 0: 从 GPU 1 拉取 Expert 1 权重 → 本地执行
  GPU 0: 从 GPU 2 拉取 Expert 2 权重 → 本地执行
  通信: P2P expert weight transfer (仅传输 expert 参数)

Expert Transfer 导致的问题:
  GPU 0 现在有 3 个 experts (本地 Expert 0 + 远程 Expert 1,2)
  → Expert computation time 增长 (MoE-BERT-Large: 3 experts → 1.88×)
  → GPU 显存竞争
  → Expert parallelism 度降低
```

术语一般如何实现？如何使用？
- Janus (SIGCOMM 2023): 设计算法决定何时以何种方式获取远程 expert，采用 data-centric 范式
- FasterMoE (PPoPP 2022): 动态 shadowing 策略，仅传输 popular expert
- 适用条件：expert 参数大小 < 需要传输的 token 总大小时有效
- 局限：随 expert 增大（如 Mixtral 8×7B 每个 expert ~300MB），expert 传输成本增加；多 expert 共享 GPU 导致资源竞争和并行度下降
- LUFFY 的立场：完全禁止 expert 移动，通过 Sequence Migration + Token Condensation 替代

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---
