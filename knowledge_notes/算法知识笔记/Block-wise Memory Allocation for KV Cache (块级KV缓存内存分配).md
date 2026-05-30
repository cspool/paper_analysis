## Block-wise Memory Allocation for KV Cache (块级KV缓存内存分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Block-wise Memory Allocation 是 PM-KVQ 提出的将 KV Cache 量化位宽按 transformer block 粒度非均匀分配的技术。核心观察：不同深度的 transformer block 对 KV Cache 量化误差的敏感度差异显著——深层 block 以及 Qwen 架构的首个 block 比浅层 block 敏感数倍。Uniform bit-width 无法利用这一差异，在显存充足但不足以全局升档时造成浪费。

PM-KVQ 通过 Integer Programming 形式化块级分配问题：`min Σ_i Σ_b x_{i,b}·s_{i,b}` s.t. `Σ_i Σ_b x_{i,b}·Mem(Q_b(K_i)+Q_b(V_i)) ≤ M`，其中敏感度 `s_{i,b} = ||G_{K_i} ⊙ (K_i - Q_b(K_i))||_1 + ||G_{V_i} ⊙ (V_i - Q_b(V_i))||_1` 使用一阶 Taylor 近似估计。CVXPY 在数秒内求解，为敏感 block 分配更高 Fbit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === 离线阶段 ===
# 输入: N 个 transformer blocks, 校准数据, 可选位宽集合 B, 显存预算 M

for each block i = 1 to N:
    前向传播得 K_i, V_i
    反向传播得梯度 G_K_i, G_V_i
    for each bit-width b in B:
        Q_b_K = fake_quant(K_i, bit=b, group=128)
        Q_b_V = fake_quant(V_i, bit=b, group=128)
        s_{i,b} = ||G_K_i ⊙ (K_i - Q_b_K)||_1
                + ||G_V_i ⊙ (V_i - Q_b_V)||_1

# IP 求解
CVXPY.solve(
    minimize Σ_i Σ_b x_{i,b} * s_{i,b}
    s.t. Σ_b x_{i,b} = 1,  x_{i,b} ∈ {0,1}
         Σ_i Σ_b x_{i,b} * Mem(b) ≤ M
)
# 输出: per-block Fbit (如 block 1→4bit, block 28→4bit, 其余→2bit)
```

**Annotations**: 敏感度使用 L1 范数聚合 per-element 梯度×量化误差。IP 求解耗时 < 5s per model。一阶 Taylor 假设 FP16 附近局部线性，在 2-4 bit 量化下基本成立。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现分三步：(1) 校准阶段逐 block 计算敏感度矩阵；(2) 调用 CVXPY 求解 IP；(3) 推理时每个 block 独立执行渐进量化至分配的 Fbit。该方法的一个局限是敏感度基于单一校准集估计，对分布偏移敏感。当 batch size 减少（单样本显存增加）时，块级分配可将多余显存定向分配给敏感 block，提升 0.84% pass@1。代码开源：https://github.com/thu-nics/PM-KVQ。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
