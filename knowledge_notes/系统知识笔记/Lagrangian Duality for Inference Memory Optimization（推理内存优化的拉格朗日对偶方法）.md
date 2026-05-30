## Lagrangian Duality for Inference Memory Optimization（推理内存优化的拉格朗日对偶方法）

术语是什么？
Lagrangian Duality 是 Remoe 用于求解 remote experts 内存规格分配问题 P₂ 的优化方法。原始问题为非线性整数规划（NP-hard），Remoe 通过三个步骤转化为可解形式：
1. **连续松弛**：将离散内存规格 y_{l,v} ∈ {0,1} 连续化为 ỹ_l ∈ [m₁, m_{V^e}]
2. **函数拟合**：构造指数衰减函数 T̃_l^rem = θ₁exp(-θ₂ỹ_l) + θ₃ 描述 CPU 推理时间与内存的单调递减收敛关系
3. **对偶求解**：证明目标函数的凸性（Theorem 2），利用 Slater 条件（Lemma 1）保证强对偶性，通过 KKT 条件（Theorem 3）找到全局最优解

从系统架构角度拆解术语：
优化问题 P₂ 的形式：
```
min_{ỹ}  P₂ = (1+η) Σ_l s̃_l (T̃_l^rem + t_l^rem/s̃_l) (H_w + c_c·ỹ_l)

其中 T̃_l^rem = θ₁exp(-θ₂ỹ_l) + θ₃  // remote expert 推理时间
     H_w = c_g·M_g + c_c·Σ w_v'·m_v'  // 主模型单位时间开销
     s̃_l = Σ_k x_{l,k}·s̃_{l,k}  // 该层 remote expert 总激活概率
     η ≈ 0.1  // prefilling 相对 decoding 的时间比例上界
```

凸性条件（Theorem 2）：g(ỹ_l) = (T̃_l^rem + t_l^rem/s̃_l)(H_w + c_c·ỹ_l) 在 ỹ_l ≥ 2/θ₂ - H_w/c_c 上严格凸。对于大多数 MoE 模型，θ₂ >> 2c_c/H_w，因此 g(ỹ_l) 在 (0,∞) 上全局凸。

对偶问题 P₂^D：
```
max_λ  P₂^D = (1+η) Σ_l s̃_l·g(ỹ_l) + Σ_j Σ_l λ_{l,j}·q_{l,j}^c(ỹ_l)
s.t.    λ_{l,1..4} ≥ 0

其中 q_{l,j}^c 为 TPOT 约束 + ỹ_l 范围约束，λ_{l,j} 为对偶变量
```
通过 KKT 条件求解，ỹ* 同时为 primal 和 dual 的最优解。

术语一般如何实现？如何使用？
- 使用前提：(1) 已有 MoE 模型在不同 vCPU 内存规格下的 expert 推理时间 profiling 数据（用于拟合 θ₁, θ₂, θ₃）；(2) GPU/CPU 价格比 c_g/c_c ≥ 3（商业平台通常满足）；(3) remote expert 调用开销 t_l^rem 可通过 warm-start 降低
- 求解效率：对偶问题维度 = L×4（L 层 × 4 约束），远小于原始整数规划的 V^e × L 维度
- 可推广性：该方法可用于任何资源-性能单调递减且收敛的 serverless 资源分配场景

涉及论文标题：
- Remoe: Towards Efficient and Low-Cost MoE Inference in Serverless Computing
