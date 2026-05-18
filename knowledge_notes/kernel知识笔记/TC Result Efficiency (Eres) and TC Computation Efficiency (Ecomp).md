## TC Result Efficiency (Eres) and TC Computation Efficiency (Ecomp)

术语是什么？

Eres（TC Result Efficiency）和 Ecomp（TC Computation Efficiency）是 Drawloom 定义的两个衡量 Tensor Core 在 SpMV 计算中利用效率的指标。Eres = (# of valid output elements) / (mma_m × mma_n)，衡量 TC 指令产生的输出中有多少实际贡献给最终向量 Y。Ecomp = (# of non-zero elements in A) / (mma_m × mma_k)，衡量 TC block 内实际参与计算的非零元素比例。两者分别从输出有效性（Eres）和计算有效性（Ecomp）两个维度刻画 TC 在稀疏计算中的利用效率。高 Eres 低 Ecomp 意味着大量零值参与计算但输出位置合理；低 Eres 高 Ecomp 意味着计算高效但输出位置错配。两者需同时优化。

从kernel调度角度拆解术语：

Eres 和 Ecomp 的计算例子（以 m4n2k4 TC，V=2 为例）：

```
// 场景：row strip (2行) 含 5 个非零元
// TC shape: mma_m=4, mma_n=2, mma_k=4
// V = mma_m/mma_n = 2

// Ecomp计算：
// 输入：5 nonzeros → 填充到 4×4 TC A矩阵
// nnz_in_block = 5
// Ecomp = 5 / (4 × 4) = 5/16 = 31.25%

// Eres计算：
// 输出：TC输出 4×2=8 个结果位置
// 其中4个位置对应实际Y元素（valid outputs沿对角线）
// valid = 4
// Eres = 4 / (4 × 2) = 4/8 = 50%

// DASP m8n8k4: Ecomp≈60%, Eres 较高
// DASP naive m4n2k4(v=2): Ecomp 增加但 Eres 降至 25%
// Drawloom ArbitWeave: Ecomp≈60% 保持，Eres 显著提升
```

Drawloom 的 ArbitWeave 通过 column compression 保持高 Ecomp 同时利用大 TC shape 改善 Eres。SpMM naive approach SpMV 的 Ecomp 仅 11.78%（因稀疏导致大量零值填充到 TC block）——说明 SpMM 优化不能直接用于 SpMV。

术语一般如何实现？如何使用？

Eres 和 Ecomp 是分析性指标，用于指导 TC shape 选择和 mapping 策略优化。在实际实现中，preprocessing 阶段可基于这两个指标预估每种 TC shape 的理论效率，选择最优的 TC shape。Drawloom 在 FP16 下选择 m16n8k16（Ecomp 平均 60.15% vs DASP 61.20%，Eres 显著提升），FP64 下选择 H100 的 m16n8k16 FP64 TC。

涉及论文标题：
- Exploiting Efficient Mapping and Pipelined Execution for Accelerating SpMV on Tensor Cores

