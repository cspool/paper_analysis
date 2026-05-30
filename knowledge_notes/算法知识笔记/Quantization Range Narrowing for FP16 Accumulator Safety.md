## Quantization Range Narrowing for FP16 Accumulator Safety

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Quantization Range Narrowing for FP16 Accumulator Safety（量化范围收缩以保证 FP16 累加器安全）是 SageAttention2++ 提出的一种量化策略。与传统的"最大范围量化"（如 FP8 E4M3 格式将值量化到其完整表示范围 [-448, 448]）不同，该策略有意将量化 scale factor 调大（即缩小量化后的数值范围），使量化后张量的值域远小于 FP8 格式的上界。目的是确保在 mma.m16n8k32 指令的 FP16 累加器中累积 32 个乘积项后，累加结果不会超出 FP16 的最大可表示值 65504。数学约束为：

$$P_r \times V_r \leq 2047 \quad \text{（其中 } P_r = \max(|\tilde{P}|)/\delta_P,\; V_r = \max(|V|)/\delta_V \text{）}$$

即量化后 P 和 V 的元素上界乘积不超过 2047（65504/32）。若启用 Delayed FP32 Buffering（两次 MMA 结果在 FP16 中累加），约束变为 $P_r \times V_r \leq 1023.5$。

从算法pipeline角度拆解术语：

该策略位于 attention 量化 pipeline 的 P×V 阶段。标准 SageAttention2 pipeline 中，P 和 V 分别量化到 FP8 E4M3 完整范围（$P_r=448, V_r=448$）。SageAttention2++ 修改 scale factor 计算：

原 SageAttention2：
```
δ_P = max(|P̃|) / 448       # E4M3 完整范围
δ_V = colmax(|V|) / 448     # E4M3 完整范围
P̂ = round(P̃ / δ_P)          # P̂ ∈ [-448, 448]
V̂ = round(V / δ_V)          # V̂ ∈ [-448, 448]
# 使用 mma.f32.f8.f8.f32 (FP32 acc, 不怕溢出)
O = P̂V̂ * δ_P * δ_V
```

SageAttention2++ narrowing：
```
δ_P = max(|P̃|) / 224        # 缩小范围，P̂ ∈ [-224, 224]
δ_V = colmax(|V|) / 4.5     # 缩小范围，V̂ ∈ [-4.5, 4.5]
P̂ = round(P̃ / δ_P)          # 每个元素 |P̂| ≤ 224
V̂ = round(V / δ_V)          # 每个元素 |V̂| ≤ 4.5
# 使用 mma.f16.f8.f8.f16 (FP16 acc, 需要范围安全)
# 验证: |32 × P̂ × V̂| ≤ 32 × 224 × 4.5 = 32256 ≤ 65504 ✓
O = P̂V̂ * δ_P * δ_V
```

关键设计权衡：缩小 V 的量化范围（V_r=4.5 << 448）会导致 V 的量化精度下降，但由于 P 的范围相应扩大（P_r=224，仍小于 448），两者的乘积 $P_r \times V_r$ 保持不变（精度等价）。实验（Table 2）表明 (P_r=224, V_r=4.5) 与 (P_r=448, V_r=448) 的 CosSim 均为 99.97%、L1 误差一致，证明该"置换"几乎无损。

术语一般如何实现？如何使用？

该策略的实现方式是修改量化 kernel 中 scale factor 的计算逻辑，将除数从 FP8 格式最大值（448 for E4M3）改为自定义的 $P_r$ 和 $V_r$。$P_r$ 和 $V_r$ 作为超参数由实验确定，选择满足精度约束和累加器安全约束的最优对。SageAttention2++ 通过 Table 2 的网格搜索确定了 (224, 4.5) 为最优参数：在 CosSim=99.97%、L1=0.01862 的条件下最大化性能。

该策略适用于任何使用低精度累加器（如 FP16, BF16）进行高精度 Matmul（如 FP8, INT8 输入）的场景——只要累加器的表示范围小于操作数乘积的最大可能值，就需要缩小操作数的量化范围。典型场景包括 FP8/INT8 MMA with FP16/BF16 accumulator 的 GEMM kernel、attention kernel、FFN kernel 等。

涉及论文标题：
- SageAttention2++: A More Efficient Implementation of SageAttention2
