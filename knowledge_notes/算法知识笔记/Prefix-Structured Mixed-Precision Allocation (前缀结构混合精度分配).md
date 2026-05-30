## Prefix-Structured Mixed-Precision Allocation (前缀结构混合精度分配)

术语解释
Prefix-Structured Mixed-Precision Allocation 是一种基于 expert 重要性排名的逐 expert bitwidth 分配策略，在固定每层平均 bitwidth budget 下，按重要性降序（前缀结构）将更高精度分配给更重要 experts，以最大化量化增益（减少 MSE vs FP16 reference）。

术语是什么？
在 GPU-NDP MoE 推理中，NDP 设备的计算吞吐有限，FP16 expert 执行会成为瓶颈。Prefix-Structured Allocation 的核心假设：更重要的 expert（激活频率高、路由评分高）需要更高的量化精度，次要 expert 可承受更粗量化。通过枚举所有满足 budget 的前缀结构 bitwidth 分配 (n4, n3, n2, n1)（即让最重要的 n4 个 expert 用 4-bit、其次 n3 个用 3-bit、n2 个用 2-bit、其余用 1-bit），选择累积损失降低最大的分配方案。

从算法pipeline角度拆解术语：

```
=== 符号说明 ===
E_ndp: NDP-resident experts per layer
b_bar: target average bitwidth (e.g., 2, 3)
R = E_ndp * (b_bar - 1): bitwidth increment budget
L_i(b): pre-measured MSE loss of expert i at bitwidth b

=== Step 1: 按重要性降序排列 ===
importance_scores = {S_{l,e} = α·P̃_{l,e} + (1-α)·W̃_{l,e}}
idx = argsort(importance_scores, descending=True)  # 最重要的在前

=== Step 2: 预计算前缀累积增益 ===
# 从 1-bit baseline 升级到更高精度的 loss 降低
Δ_i(2) = L_i(1) - L_i(2)
Δ_i(3) = L_i(1) - L_i(3)
Δ_i(4) = L_i(1) - L_i(4)

# 前缀累积: C_b(k) = sum_{i=1..k} Δ_i(b)
C_2 = prefix_sum(Δ(2))  # 前 k 个 upgrades to 2-bit 的增益
C_3 = prefix_sum(Δ(3))  # 前 k 个 upgrades to 3-bit 的增益
C_4 = prefix_sum(Δ(4))  # 前 k 个 upgrades to 4-bit 的增益

=== Step 3: 枚举最优分配 ===
best_gain = -∞
for n4 in 0..E_ndp:
    if 3*n4 > R: break
    for n3 in 0..(E_ndp - n4):
        if 3*n4 + 2*n3 > R: break
        n2 = R - 3*n4 - 2*n3
        if n2 < 0 or n4 + n3 + n2 > E_ndp: continue
        n1 = E_ndp - n4 - n3 - n2
        
        # 前缀结构增益
        gain = C_4(n4)                          # n4 个 most important → 4-bit
             + (C_3(n4+n3) - C_3(n4))           # 其次 n3 个 → 3-bit
             + (C_2(n4+n3+n2) - C_2(n4+n3))     # 再次 n2 个 → 2-bit
             + 0                                 # 其余 n1 个 → 1-bit (baseline)
        
        if gain > best_gain:
            best_gain = gain
            best = (n4, n3, n2, n1)
    end
end

=== Step 4: 分配 bitwidth ===
# 前缀结构: 最重要的 → 高 bitwidth
b[idx[0:n4]] = 4
b[idx[n4:n4+n3]] = 3
b[idx[n4+n3:n4+n3+n2]] = 2
b[idx[n4+n3+n2:]] = 1
```

复杂度：每层 O(E_ndp²)，总共 O(L·E_ndp²)，远小于推理成本。

术语一般如何实现？如何使用？
- 离线预计算 per-expert per-bitwidth loss table（使用 calibration data 如 C4）
- 运行时 prefix search 枚举所有可行分配，使用 prefix sums 在 O(1) 评估每个配置
- 约束：固定每层平均 bitwidth（如 b_bar=2 或 3），控制 NDP 总计算量
- 关键参数：重要性 mixing coefficient α（0.5 平衡 activation frequency 和 routing score）
- 适用场景：NDP/边缘设备的混合精度 expert 量化，多精度硬件（FP16+INT4+INT2+INT1）
- 实测效果：Ours-2bit with prefix selector vs without: +3.2% avg accuracy on 8 benchmarks

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
