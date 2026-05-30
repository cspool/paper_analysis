## HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy

- 属于算法pipeline的实现是什么？实验比较什么？
  - HEXA-MoE 提出三个算法层面的 MoE 训练优化：(1) **Expert-Specific Operators**：用三个专用算子——ESMM (Expert-Specific Matrix Multiplication)、ESS (Expert-Specific Summation)、ESTMM (Expert-Specific Transposed Matrix Multiplication)——替代传统 GeMM 或 grouped GeMM 接口，实现 in-place 计算，消除 token padding/discarding 带来的冗余 FLOPs；(2) **Data-Centric 与 Model-Centric 双模式并行**：对大规模 workload 使用 data-centric 配置（tensor parallelism 切分 FFN intermediate size，各设备 all gather 完整 MoE 参数后本地计算），对小规模 workload 使用 model-centric 配置（all gather 同步本地数据批次，各设备计算本地参数 chunk）。引入 pipeline-shared cache 解决 data-centric 模式下 backward pass 需保存全部 MoE 参数导致内存膨胀的问题；(3) **Heterogeneous-Aware Expert Allocation**：基于各设备计算能力（通过 benchmark 测量平均延迟）按反比分配 workload——data-centric 下调整各设备 local batch size，model-centric 下调整各设备 FFN intermediate sub-dimension。
  - 实验比较：(1) Memory Analysis: HEXA-MoE vs Tutel vs MegaBlocks 的 GPU 内存占用，Swin-MoE Small/Base, 8 global experts, top-1~top-8 routing；(2) Latency Analysis: 平均每步训练延迟对比，4 homogeneous GPUs, 4 experts, 不同 batch size；(3) Data-Centric vs Model-Centric: 不同 batch size 下的延迟对比；(4) Heterogeneous Experiments: 异构设备（TITAN RTX + RTX 2080 Ti）上 data-centric 和 model-centric 配置下不同 workload 分配比例的延迟对比；(5) Ablation: expert-specific operators、pipeline-shared cache、fused kernel、data-/model-centric、memory optimization 各组件的 memory footprint 和 latency 贡献分解。

- 硬件平台是什么，配置是什么。
  - 同构机器 M_homo：CPU 2× Intel Xeon Platinum 8352V 2.10GHz, 1008 GB RAM；GPU 4× NVIDIA GeForce RTX 4090 (24 GB)。同构实验均在 M_homo 上进行。
  - 异构机器 M_hete：CPU 2× Intel Xeon Gold 6130 2.10GHz, 62.5 GB RAM；GPU D0: 1× NVIDIA TITAN RTX (24 GB), D1: 1× NVIDIA GeForce RTX 2080 Ti (11 GB)。
  - 软件栈：PyTorch + NCCL 通信后端，automatic mixed precision 训练。

- 模型是什么。数据集和bench分别是什么。
  - 模型：Swin-Transformer-MoE（Small 和 Base 两种规模），遵循 Tutel (Hwang et al., 2023) 的配置。全局 experts 数=4 或 8，routing 从 top-1 到 top-8。默认使用 top-k routing + atomicAdd 聚合各 expert 输出。
  - Benchmark：Swin-MoE 训练过程作为 benchmark，评估指标为平均 GPU 内存占用 (GB) 和每训练步平均延迟 (s)。Latency 实验记录 2k steps 的平均值，Memory 实验记录各设备平均 GPU 内存占用。
  - 计算能力 benchmark（异构实验）：Algorithm 5 中的 proxy task——1024 次循环大矩阵乘法（size=2048），测量完成时间作为计算能力指标。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源链接：https://github.com/UNITES-Lab/HEXA-MoE（另有作者个人仓库 https://github.com/luoshuqing2001/hexa_moe）
  - 算法 pipeline 伪代码（以 top-1 routing 为例）：

```python
# === HEXA-MoE Forward & Backward with Expert-Specific Operators ===
# 输入: x [N, D_i], 路由选择 R(x) [N], 权重 W1 [E, D_i, D_mid], W2 [E, D_mid, D_o]
# Forward:
y1 = ESMM(x, W1, b1, R(x))       # [N, D_mid], 每个 token 用其路由 expert 的 W1 计算
y2 = F(y1)                         # 激活函数 (如 GELU)
y  = ESMM(y2, W2, b2, R(x))       # [N, D_o]

# Backward (auto-diff 提供 ∂ℓ/∂y):
∂ℓ/∂b2 = ESS(∂ℓ/∂y, R(x))         # [E, D_o], 按 expert 累加
∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x))   # [E, D_mid, D_o]
∂ℓ/∂y2 = ESMM(∂ℓ/∂y, W2^T, null, R(x))  # [N, D_mid]
∂ℓ/∂y1 = ∂ℓ/∂y2 ⊙ F'(y1)          # element-wise
∂ℓ/∂b1 = ESS(∂ℓ/∂y1, R(x))        # [E, D_mid]
∂ℓ/∂W1 = ESTMM(x, ∂ℓ/∂y1, R(x))   # [E, D_i, D_mid]
∂ℓ/∂x  = ESMM(∂ℓ/∂y1, W1^T, null, R(x))  # [N, D_i]
```

  - Expert-Specific Operators 定义：
    - ESMM(x, W, b, R): y_i = x_i @ W_{R(x_i)} + b_{R(x_i)}，每个 token 仅与其路由 expert 的权重做矩阵乘法
    - ESS(x, R): y[e] = Σ_{i: R(x_i)=e} x_i，按 expert 分组累加
    - ESTMM(x1, x2, R): y[e, i, j] = Σ_{m: R(x_m)=e} x1[m,i] · x2[m,j]，expert-wise 外积累加

  - Top-k routing 扩展（k>1）：对 k 个路由选择分别执行 ESMM，最终输出为 k 个 ESMM 结果的累加（使用 atomicAdd）。中间结果 tensor 的内存分配扩展为 k 倍。

  - Data-Centric 配置（大规模 workload）：
    ```
    # 各设备沿 FFN intermediate size 切分 expert 权重
    # 每个 MoE layer: all gather 完整参数 → ESMM 本地计算 → 下一层
    # pipeline-shared cache: 每设备分配额外 HBM 区域动态缓存 gathered shards
    # all gather 与 attention/router 计算 overlap
    ```

  - Model-Centric 配置（小规模 workload）：
    ```
    # 各设备沿 FFN intermediate size 切分 expert 权重
    # 每个 MoE layer: all gather 数据批次 → ESMM 用本地参数 chunk 计算 → all reduce sum 聚合
    ```

  - Heterogeneous Allocation:
    ```
    # 先测量各设备计算能力 t_i (proxy task 延迟)
    # Data-centric: B_i = (1/t_i) / Σ(1/t_j) · B_global
    # Model-centric: h_i = (1/t_i) / Σ(1/t_j) · H
    ```

  - 对比 baseline (Tutel) 的 pipeline：Tutel 使用 GeMM + token padding/discarding → dispatch/combine + all-to-all 通信；HEXA-MoE 使用 ESMM in-place 计算 + tensor parallelism 替代 expert parallelism → 无 token padding、无冗余 FLOPs、无 all-to-all 通信。
