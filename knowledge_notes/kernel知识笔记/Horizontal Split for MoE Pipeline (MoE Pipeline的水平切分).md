## Horizontal Split for MoE Pipeline (MoE Pipeline的水平切分)

术语解释
在 MoE Expert Pipeline 调度中，对输入张量按行（token 维度）切分，同时将 MoE 权重按 expert 切分。相对于传统 TP 的垂直切分（按列切分输入+按列切分权重），水平切分避免了参数矩阵的重复内存 I/O，并支持将 GroupGemm 退化为 DenseGemm 以获得更高计算效率。

术语是什么？
MoE 模型的 Expert Pipeline 调度需要将输入数据划分为多个 pipeline stage。两种切分方式：
- **垂直切分（Vertical Split）**：按列切分输入张量，权重也按列切分。每个 stage 需要加载完整权重矩阵，导致重复 I/O。
- **水平切分（Horizontal Split）**：按行切分输入张量（按 token 分组），权重按 expert 切分。每个 stage 只加载对应专家组的权重，利用 MoE 稀疏激活避免重复 I/O。

从kernel调度角度拆解术语：

```
# === 垂直切分 (类似 TP) ===
# 输入 [m, K]，权重 [E*K, N] 完整加载
for pipe_i in range(N):
    x_slice = input[:, pipe_i*K//N : (pipe_i+1)*K//N]  # [m, K/N]
    w_subset = weights[pipe_i*K//N : (pipe_i+1)*K//N, :]
    for expert_j in range(E):  # 仍需 per-expert I/O
        y_j += x_j @ w_subset_j  # GroupGemm, group=E
# I/O: V_vertical = m*P0 + E*W + N*m*P1

# === 水平切分 (EPS-MoE) ===
for pipe_i in range(N):
    tokens_i = tokens_for_experts[pipe_i*E//N : (pipe_i+1)*E//N]  # [m_i, K]
    W_i = expert_weights[pipe_i*E//N : (pipe_i+1)*E//N]  # [(E/N)*K, N]
    # Load-aware GEMM choice
    if m_i >= 4096:
        y_i = DenseGemm(tokens_i, W_i)  # cublas
    else:
        y_i = GroupGemm(tokens_i, W_i, groups=E/N)
# I/O: V_horizontal = m*P0 + E*W + m*P1  (无重复参数I/O)

# 当 N=E 时: GroupGemm 退化为 DenseGemm
# 计算时间比较:
# T_vertical = E*C / R(FLOPS | group=E)
# T_horizontal = E*C / R(FLOPS | group=E/N)
```

术语一般如何实现？如何使用？
- 基于 vLLM 的 MoE FFN 前向传播路径修改输入切分逻辑
- token routing 后按 expert 分组 token，再按 pipeline 数细分为子组
- 权重在初始化时即按 expert 维度切分存储，避免运行时切分开销
- 与 all-to-all 通信流水线配合：第 i 组 all2all 通信与第 i-1 组 GEMM 计算重叠
- 适用于 top-k 较大或专家数较多的 MoE 模型

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs
