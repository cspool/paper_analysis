## MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出 MPMoE，一个基于 PyTorch 的高性能 MoE 训练库，核心实现包括三个组件：
    - **Micro-Batch Pipeline Parallelism**：将 mini-batch 按 batch size 维度切分为多个 micro-batch，在 MoE 层的三个阶段（S: 第一个 All-to-All dispatch → C: Expert 计算 → R: 第二个 All-to-All collect）之间实现 pipeline 并行，使通信和计算重叠执行。与 FasterMoE 按 node 维度切分不同，MPMoE 按 batch 维度切分，保留了 NCCL All-to-All 的优化能力，且 pipeline granularity n 可灵活调整。
    - **Memory Reuse Strategies**：针对 MoE 训练中 activation tensors 和 temporary buffers 占主要内存的观察，提出 4 种内存复用策略（S1-S4），通过不同方式恢复前向中被覆盖的 tensors（T_DI 和 T_M）：S1（T_DI/T_M 均 offload 到 CPU）、S2（T_DI 通信恢复 + T_M offload）、S3（T_DI offload + T_M recompute）、S4（T_DI 通信恢复 + T_M recompute）。将所需 activation buffer 从 n 份压缩为 1 份。
    - **Joint Optimization**：配置 (n, S) 的联合优化——n 为 pipeline granularity，S 为内存复用策略。提供两种方法：(a) MPMoE-pb：profile-based 搜索算法（Algorithm 1），利用单调性和抛物线假设减少搜索空间；(b) MPMoE-pm：基于 3 种 pipeline paradigm（范式1/2/3，如图 8）和 piecewise 性能模型（如图 9），在运行时估算不同配置的执行时间。
  - 实验比较：
    - 端到端训练速度：MPMoE-pb vs MPMoE-pm vs FasterMoE vs FastMoE，在 Adira（64 A100）和 Valor（16 V100）两个集群上。
    - 内存占用：MPMoE vs FastMoE vs FasterMoE vs PMoE（无内存复用的 MPMoE 变体），在不同 pipeline stage 数 n=2/4/8 下。
    - 理论内存节省上限 vs 实际内存节省（Equation 6 验证）。
    - 消融实验：(a) 通信效率 micro-benchmark（FasterMoE vs MPMoE 不同 n 下的 All-to-All dispatch/recovery 时间）；(b) Pipeline granularity 敏感度分析（不同 B 和 n 的性能变化）；(c) 内存复用策略开销分析（S1-S4 在不同 N 和 B 下的表现）。
    - 性能分解与开销分析（TensorCore 加速率、data partition 开销、profiling 开销）。
    - 多节点可扩展性（1/2/4/8 nodes on Adira，throughput 对比）。

- 硬件平台是什么，配置是什么。
  - **Adira 集群**：8 台 NVIDIA DGX A100 服务器，每节点 8×A100 40GB GPU（共 64 GPU），200 Gbps HDR InfiniBand 互联，节点内第 3 代 NVLink。
  - **Valor 集群**：4 节点，16× NVIDIA Tesla V100 16GB HBM GPU，每节点 4 GPU，56 Gbps HDR InfiniBand 互联，节点内第 2 代 NVLink。
  - 软件栈：PyTorch 1.9、CUDA Toolkit 11.1、NCCL 2.7、Ubuntu 18.04。

- 模型是什么。数据集和bench分别是什么。
  - 模型：3 种 MoE 配置（见表 3）：
    - MoE-GPT-S: d_model=768, d_hidden=3072, #experts=64 或 16
    - MoE-GPT-XL: d_model=2048, d_hidden=8192, #experts=64 或 16
    - MoE-BERT-L: d_model=1024, d_hidden=4096, #experts=64 或 16
  - 数据集：Dummy dataset（随机生成的 tokens），因为评估目标是训练系统的 throughput 和 memory footprint，而非模型精度。
  - 优化器：Adam。
  - 评估指标：平均训练时间（用于 speedup 计算）、峰值内存占用（peak memory footprint）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：论文未提供开源链接，经搜索（论文页面、武大 ICS 实验室页面）未发现公开代码仓库。论文的 conference 版本 MPipeMoE 发表于 IPDPS 2023。
  - 算法 pipeline 伪代码：

```
输入: 当前 MoE layer 的 input tensor T_I, shape (N, B, M)
      N = device 数, B = batch size, M = model dimension
      配置: pipeline granularity n, memory reuse strategy S ∈ {S1, S2, S3, S4}
输出: MoE layer 的 output tensor T_O, shape (N, B, M)

// ===== 0. 确定最优配置 (Section 4) =====
// MPMoE-pb: profile-based (Algorithm 1)
if MPMoE-pb:
    if B not in cache C:
        在 G 中找最近区间 (n_l, n_h)
        n_best = searchBestGran(B, (n_l, n_h))  // 抛物线终止搜索
        更新 G 和 C
    n = C[B]
    for each S in {S1, S2, S3, S4}:
        用 n 执行 profiling，取执行时间最短的 S
// MPMoE-pm: performance model
else:  // MPMoE-pm
    对每种配置 (n, S):
        根据 paradigm (1/2/3) 和 piecewise 速度模型估算 T(n,S)
        // Piecewise 速度: W_comp(B/n), W_comm(B/n), W_mem(B/n)
        // α 干扰因子: α(comm,comp), α(comp,comm), α(comm,mem)
    取 T 最小的 (n, S)

// ===== 1. Micro-Batch Pipeline Parallelism (Section 3.2) =====
将 T_I 沿 batch 维度切分为 n 个 micro-batch:
    T_I[0], T_I[1], ..., T_I[n-1], 每个 shape (N, B/n, M)

定义 pipeline stages:
    S(i): 第 i 个 micro-batch 的 All-to-All dispatch (T_I[i] → T_DI[i])
    C(i): 第 i 个 micro-batch 的 Expert FFN 计算 (T_DI[i] → T_M[i] → T_DO[i])
    R(i): 第 i 个 micro-batch 的 All-to-All collect (T_DO[i] → T_O[i])

Pipeline 调度 (如图 7 timeline，交替执行 S 和 R 以增强 memory locality):
    时间 t0: S(0) 启动
    时间 t1: S(0) 完成 → 同时启动 C(0) 和 S(1)
    时间 t2: R(0) 在 C(0) 完成后启动, S(2) 启动
    ...
    // Tensor Shape Flow (以 expert FFN 为例):
    // T_DI[i]: (B/n, M) → Linear1: W1(M, H) → T_M[i]: (B/n, H)
    // T_M[i] → GeLU(in-place) → Linear2: W2(H, M) → T_DO[i]: (B/n, M)

// ===== 2. Memory Reuse (Section 3.3) =====
// 原本每个 partition 独立分配 buffer，现改为共享:
Buffer_DI = alloc(B/n * M * sizeof(fp16))   // n partitions 共享
Buffer_M  = alloc(B/n * H * sizeof(fp16))   // n partitions 共享
Buffer_DO = alloc(B/n * M * sizeof(fp16))   // n partitions 共享

// 前向: 各 micro-batch 的 tensors 依次复用同一 buffer
// 后向: 需恢复被覆写的 T_DI, T_M，根据策略 S:
switch S:
    case S1:  // offload T_DI, offload T_M
        forward:  // Paradigm 2
            C(i) 完成后: D2H_copy(T_DI[i])   // 异步拷贝到 CPU
            S(i) 完成后: D2H_copy(T_M[i])   // 异步拷贝到 CPU
        backward:  // Paradigm 3
            H2D_copy(T_M[i])                // 先从 CPU 取回
            H2D_copy(T_DI[i])
            计算梯度
    case S2:  // comm restore T_DI, offload T_M
        forward:  // Paradigm 2
            仅 offload T_M[i]
        backward:  // Paradigm 3
            H2D_copy(T_M[i])
            T_DI[i] = All-to-All_replay(T_I[i])  // 重新通信
            计算梯度
    case S3:  // offload T_DI, recompute T_M
        forward:  // Paradigm 2
            仅 offload T_DI[i]
        backward:
            H2D_copy(T_DI[i])
            T_M[i] = FFN_forward(T_DI[i])  // 重新计算
            计算梯度
    case S4:  // comm restore T_DI, recompute T_M
        forward:  // Paradigm 1 (无 memory copy)
            不 offload 任何 tensor
        backward:
            T_DI[i] = All-to-All_replay(T_I[i])
            T_M[i] = FFN_forward(T_DI[i])
            计算梯度

// ===== 3. Memory Footprint Calculation (Section 2.2) =====
原始 M_act = 4*B*M + B*H          // Equation 2
原始 M_buf = B*M + B*H            // Equation 3
Pipeline 后的 M_act^pipe = M_buf^pipe = 4*B*M + B*H  // Equation 4
Memory reuse 后节省:
    ΔM_act = ΔM_buf = B * (2M*(n-2)/n + H*(n-1)/n)  // Equation 5
Memory 节省率:
    φ = (ΔM_act + ΔM_buf) / (M_ms + M_act^pipe + M_buf^pipe)  // Equation 6
// 其中 M_ms = 4 * (E*M + 2*H*M)  // 包含 params, grads, momentum, variance

// ===== 4. Performance Model (Section 4.2) =====
// Piecewise 速度函数 (Figure 9):
W_comp(volume) = { k1_comp * volume,  if volume < V_threshold_comp
                 { k2_comp * volume,  otherwise
W_comm(volume)  = { k1_comm * volume,  if volume < V_threshold_comm
                 { k2_comm * volume,  otherwise
// 带 α 干扰因子的实际执行时间:
// 以 Paradigm 1 的 P2 阶段为例:
T_P2 = max( (t_S + t_R) / α(comm,comp), t_C / α(comp,comm) )
```

- 关键配置与结果：
  - Pipeline granularity n: 2/4/8（B < 8k 时 n=2 最优，8k-22k 时 n=4，>22k 时 n=8，Figure 14 验证了单调性假设）。
  - 内存复用策略选择：N 小（如 8 GPU）时 S1/S2 更优，N 大（如 64 GPU）时 S4 更优（Figure 15）。
  - MPMoE-pb 平均 1.66× speedup vs FasterMoE，MPMoE-pm 平均 1.55×；vs FastMoE 分别 2.34× 和 2.20×（Figure 10）。
  - 内存节省：n=2/4/8 时分别平均节省 23%/34%/38%，最高比 FastMoE/FasterMoE 节省 53%（Figure 11）。
  - 实际内存节省达到理论上限的约 95%（Figure 12）。
  - 8 节点扩展比：MPMoE 5.76×（72% ideal），FasterMoE 5.4×（Figure 17）。
  - Profiling overhead: MPMoE-pb <3%，MPMoE-pm <1%（Figure 16）。
