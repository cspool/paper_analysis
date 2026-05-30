## SSMB (Sequence-Sharded MoE Blocks / 序列分片MoE块)

术语是什么？

SSMB 是 X-MoE 提出的混合并行策略，在 TP（Tensor Parallelism）+ EP（Expert Parallelism）的并行方案中，进入 MoE block 时将输入序列切分到各 EP rank，使每个 rank 仅处理序列的一段（1/G，G=TP group size），从而将 Adispatch 和 Acombine 的激活内存减少 G 倍。MoE block 结束后通过 all-gather 恢复完整序列以兼容下游 TP block。

从系统架构角度拆解：

SSMB 执行流程（以 TP=2, EP=4 为例）：

```
# === 进入 MoE Block 前（TP Phase） ===
# Device 0,1: 各有完整序列 A0 的拷贝
# Device 2,3: 各有完整序列 A1 的拷贝

# === SSMB 进入 MoE Block ===
# (1) Drop partial tokens: 切分序列
Device 0: 保留 A0[0:S/2]  # 前半段
Device 1: 保留 A0[S/2:S]  # 后半段
Device 2: 保留 A1[0:S/2]
Device 3: 保留 A1[S/2:S]

# (2) EP MoE: gating → PFT dispatch → alltoall → sequential GeMM → combine
# 每个device仅处理其保留的序列片段
# Adispatch, Acombine 尺寸均缩小为 1/G

# (3) All-gather: 恢复完整序列
A0_full = all_gather([Device0_output, Device1_output])
A1_full = all_gather([Device2_output, Device3_output])

# === 回到 TP Phase ===
# 下游 dense block 继续使用完整序列
```

SSMB vs TED（Tensor-Expert-Data parallelism）的内存收益比：
$$r = \frac{k}{H_{FFN}}$$
当 $r > \frac{2}{c \cdot S}$ 时 SSMB 更优（c=capacity factor, S=sequence length）。对 expert-specialized MoE，k ∝ m, HFFN ∝ 1/m，因此 m 越大 SSMB 越有利。

SSMB vs Activation Checkpointing：
- SSMB：激活内存减少 G×，无额外通信（仍然是 4 alltoall/layer），无重计算
- Checkpointing：需额外 2 alltoall（共 6 alltoall/layer）+ 重计算开销

术语一般如何实现？

SSMB 要求 MoE block 内所有操作是 token-wise（无跨 token 依赖），这天然成立（gating、dispatch、expert FFN、combine 均独立 per-token）。Backward pass 中 SSMB 先 drop 对应 partial sequence 的梯度，执行 expert backward + alltoall，最后 all-gather 恢复完整梯度。

涉及论文标题：
- X-MoE: Enabling Scalable Training for Emerging Mixture-of-Experts Architectures on HPC Platforms
