## Integer Programming for Mixed-Precision Bit-width Allocation（混合精度位宽分配的整数规划）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Integer Programming for Mixed-Precision Bit-width Allocation 是 PM-KVQ 用来为不同 Transformer block 的 KV Cache 分配最优位宽的数学优化方法。输入是每个 block 在各候选位宽下的量化敏感度 s_{i,b} 和内存预算 M，输出是每个 block 的最优位宽选择 x_{i,b}（one-hot 编码，每个 block 恰好选择一个位宽）。该问题被建模为带约束的 0-1 整数规划，使用开源求解器 CVXPY（Diamond & Boyd, 2016, JMLR）在几秒内完成求解。

从编译框架角度拆解术语：

**问题建模**（PM-KVQ Section 3.2, Equations 6-8）：

```
目标函数（最小化全局量化敏感度）：
min Σ_{i=1}^{N} Σ_{b∈B} x_{i,b} · s_{i,b}

约束条件：
1. 每个 block 恰好选择一个位宽：  Σ_{b∈B} x_{i,b} = 1,    ∀i ∈ [1,N]
2. 总内存不超过预算：
   Σ_{i=1}^{N} Σ_{b∈B} x_{i,b} · (Mem(Q_b(K_i)) + Mem(Q_b(V_i))) ≤ M
3. 0-1 变量约束： x_{i,b} ∈ {0,1},    b ∈ B
```

**变量与参数说明**：

| 符号 | 含义 | 示例 (DeepSeek-Qwen-7B) |
|------|------|--------------------------|
| N | Transformer block 数量 | 28 |
| B | 候选位宽集合 | {2, 4} |
| s_{i,b} | block i 在 b-bit 下的敏感度 | Eq.5: ∥G_Ki⊙(K_i-Q_b(K_i))∥₁ + ∥G_Vi⊙(V_i-Q_b(V_i))∥₁ |
| Mem(Q_b(K_i)) | block i 的 Key Cache 在 b-bit 下的内存占用 | 序列长度 × num_kv_heads × head_dim × b/8 bytes |
| M | KV Cache 总内存预算 | 由 target GPU memory 和 batch size 决定 |
| x_{i,b} | 0-1 选择变量 | x_{3,4}=1 表示 block 3 使用 4-bit |

**求解流程**（离线预处理阶段）：

```
// Step 1: Profiling——校准数据上一次前向传播
for each transformer_block i in 1..N:
    for each candidate_bitwidth b in B:
        K_i, V_i = forward_and_get_kv(block_i)
        K_i_q = asymmetric_group_quantize(K_i, bits=b, group_size=128)
        V_i_q = asymmetric_group_quantize(V_i, bits=b, group_size=128)
        s_{i,b} = taylor_sensitivity(K_i, K_i_q, V_i, V_i_q)

// Step 2: 求解
problem = cvxpy.Problem(
    objective = cvxpy.Minimize(sum(x[i,b] * s[i,b] for all i,b)),
    constraints = [
        sum(x[i,b] for b in B) == 1 for each i,       // 每个 block 一个位宽
        sum(x[i,b] * mem[i,b] for all i,b) <= M,       // 总内存约束
        x[i,b] in {0,1} for all i,b                    // 0-1 约束
    ]
)
problem.solve()  // CVXPY 调用底层 MILP 求解器（如 ECOS_BB, GLPK_MI）
result = {i: b for (i,b) where x[i,b] == 1}  // 每个block的Fbit
```

**求解器选择**：CVXPY 作为高层 Python 接口，底层可调用多种 MILP 求解器。由于该问题规模小（N×|B| 个变量，N+|B| 个线性约束；如 28×2=56 变量），CVXPY 的默认求解器即可在几秒内完成。PM-KVQ 论文在 8×A100-80G 服务器上完成 profiling + 求解全过程。

术语一般如何实现？如何使用？

实现关键：(1) CVXPY 作为 Python 嵌入式优化建模语言，使整数规划定义简洁（与论文公式一一对应）；(2) 一阶泰勒近似敏感度无需反向传播到模型参数，仅需 K/V 张量梯度——可在 PyTorch hook 中高效捕获；(3) 由于 profiling 在 FP16 下进行，敏感度值不受量化先验影响——该方法是通用的，可迁移到不同模型/量化方案。

适用场景：(1) 混合精度量化方案中的位宽分配（不仅限 KV Cache，也可用于 weight/activation 混合精度）；(2) 当存在多级可选精度配置和硬性内存约束时；(3) 该方法的替代方案包括贪心分配、启发式搜索和敏感度排序——整数规划在此问题中是最优解，且求解开销可忽略。

局限性：(1) 依赖校准数据代表性——校准数据与实际推理数据分布偏差可能导致次优分配；(2) B 集合大小被限制为 2 个候选值以保证搜索效率和避免过拟合；(3) 未考虑跨 block 的误差传播效应（即浅层 block 量化误差如何通过残差连接放大到深层）。

涉及论文标题：
- PM-KVQ: Progressive Mixed-precision KV Cache Quantization for Long-CoT LLMs
