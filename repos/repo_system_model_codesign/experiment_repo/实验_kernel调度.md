# 实验_kernel调度

## SageBwd

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是基于 OpenAI Triton 的自定义 INT8 注意力 kernel（前向+反向），在 FlashAttention 风格的 tiled 计算框架中嵌入 per-block INT8 量化和 K-smoothing。核心 kernel 优化包括：(1) 前向 pass——Q 分块后先做 K-smoothing（K = K - mean_row(K)），然后 per-block INT8 量化 Q,K,V，在 tiled online softmax 循环中使用 INT8 Tensor Core 执行 Q̂K̂ᵀ 和 P̂V̂；(2) 反向 pass——在 FlashAttention 风格的 tiled 反向循环中，P 和 dO 做 per-block INT8 量化用于 dV = PᵀdO，保持 dP = dOVᵀ 为 FP16，dS 做 per-block INT8 量化后用于 dQ = d̂SK 和 dK = d̂SᵀQ；(3) 所有量化使用均匀 INT8（scale = max(|X|)/127），per-block 粒度（block 为 FlashAttention tile）。
  实验比较：SageBwd kernel vs FlashAttention2（PyTorch SDPA/Triton/xFormers 实现），在 RTX 4090 上以 TOPS/s 为指标，head dim D=64 和 D=128 下前向+反向端到端吞吐。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 4090，单 GPU。BF16/FP16 精度。使用 INT8 Tensor Core 加速量化后的矩阵乘法。Block 参数：B_q, B_kv（FlashAttention tile size，论文未明确列出具体数值，参照 FlashAttention 典型设置为 B_q=128, B_kv=64）。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：FlashAttention2（Triton 实现和 xFormers 实现）作为 baseline。SageBwd 使用 OpenAI Triton 从头实现自定义注意力 kernel。修改内容：(1) 在 FlashAttention tiled 循环中嵌入 per-block INT8 量化/反量化操作（quant→INT8 MatMul→dequant）；(2) 添加 K-smoothing 预处理（K = K - mean_row(K)）；(3) 反向 pass 选择性量化——dP 保持 FP16，其余 MatMul 使用 INT8；(4) 所有量化使用对称均匀 INT8（scale = max(|X|)/127），无 calibration data。
  
  前向 kernel 核心流程（基于 FlashAttention online softmax tiling）：
  ```
  1. 分块：Q→{Q_i}, K→{K_j}, V→{V_j}
  2. K_sm_j = K_j - mean_row(K_j)           // K-smoothing per block
  3. s_Q, Q̂_i = INT8_quant(Q_i)             // per-block, scale=max(|Q_i|)/127
  4. s_K, K̂_j = INT8_quant(K_sm_jᵀ)        // per-block, transposed
  5. s_V, V̂_j = INT8_quant(V_j)             // per-block
  6. for each Q_i block:
       for each K_j/V_j block:
         Ŝ_ij = Q̂_i × K̂_j                   // INT8 Tensor Core MatMul
         S_ij = Ŝ_ij × s_Q × s_K            // dequant to FP16/FP32
         // ... online softmax ...
         P̂_ij = P̃_ij / s_P                  // per-token INT8 quant
         O_ij += P̂_ij × V̂_j × s_P × s_V     // INT8 Tensor Core MatMul, dequant accumulate
       O_i = O_i / l_i                       // final rescale
  ```
  
  反向 kernel 核心流程：
  ```
  1. D = rowsum(dO ∘ O)
  2. for each K_j/V_j block:
       for each Q_i block:
         P_ij = exp(recompute_S_ij - L_i)    // recompute P from forward quantized Q,K
         s_P, P̂_ij = INT8_quant(P_ij)        // per-block
         s_dO, dÔ_i = INT8_quant(dO_i)      // per-block
         dV_j += P̂_ijᵀ × dÔ_i × s_P × s_dO  // INT8 MatMul: dV
         dP_ij = dO_i × V_jᵀ                 // FP16 MatMul: dP (NOT quantized!)
         dS_ij = P_ij ∘ (dP_ij - D_i)       // softmax gradient
         s_dS, dŜ_ij = INT8_quant(dS_ij)    // per-block
         dQ_i += dŜ_ij × K̂_j × s_dS × s_K   // INT8 MatMul: dQ
         dK_i += dŜ_ijᵀ × Q̂_i × s_dS × s_Q  // INT8 MatMul: dK
  ```

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源仓库：https://github.com/thu-ml/SageAttention（预计 2025 年 7 月 15 日开源 SageAttention3/SageBwd）。实现基于 OpenAI Triton（https://github.com/triton-lang/triton）。
  
  **评估原理**：测量 attention kernel 在 RTX 4090 上的 wall-clock 时间 t，计算吞吐量 TOPS/s = 理论 FLOPs / t。理论 FLOPs 包含 attention 前向+反向的所有 MatMul 运算量（不考虑量化节省的 bit-level 计算）。与 FlashAttention2（PyTorch SDPA、Triton FA2、xFormers FA2）对比，计算加速比 = FA2_time / SageBwd_time。实验测得最高 1.67× 加速比（head dim=128）。
  
  **Kernel 输入到性能输出全过程**（RTX 4090 上 head dim=128 为例）：
  1. **输入**：Q,K,V ∈ R^{N×128}（FP16/BF16），dO（反向时），block size B_q, B_kv
  2. **K-smoothing**：K 减去列均值（per-block），消除通道异常值，降低有效量化步长
  3. **Per-block INT8 量化**：对每个 FlashAttention tile 计算 scale = max(|X|)/127，round(X/δ_X)，得到 INT8 表示
  4. **INT8 Tensor Core MatMul**：Q̂_i × K̂_j 和 P̂_ij × V̂_j（前向），P̂_ijᵀ × dÔ_i、dŜ_ij × K̂_j、dŜ_ijᵀ × Q̂_i（反向）全部使用 GPU INT8 Tensor Core，输出为 INT32
  5. **反量化累加**：INT32 结果 × s_A × s_B 恢复为 FP32，累加到 FP16 输出缓冲区
  6. **墙钟计时**：使用 CUDA event 测量 kernel launch 到完成的端到端时间，排除 Python 开销
  7. **输出**：O（前向），dQ,dK,dV（反向），TOPS/s = FLOPs / t_mean（多次 warmup + repeat 取均值）

## SLA2

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现在 FlashAttention 基础之上构建 SLA2 的自定义 CUDA kernel（前向+反向）。核心优化：(1) block-wise 计算——将 Q 分为 T_m = N/b_q 块、K/V 分为 T_n = N/b_k 块，仅对 M_c[i,j]=1 的 block pair 执行 sparse attention matmul QKᵀ 和 PV，跳过 M_c[i,j]=0 的 block；(2) 线性 attention 分支使用 Q(KᵀV) 重排——对 M_c[i,j]=0 的 block 预计算 h_j = (K_j^φ)ᵀV_j（局部 KᵀV）和 z_j（归一化因子），用增量累加避免显式 (QKᵀ)V 的 O(N²) 计算；(3) 低比特量化 kernel——前向对 Q、K、P、V 执行 INT8/FP8 量化+反量化，遵循 SageAttention2++ 量化方案，在 FlashAttention block-wise 流水线中嵌入 quant/dequant，反向保持 FP16 精度；(4) 对 K 做列均值减法平滑（K = K - colmean(K)），继承 SageAttention 的稳定性技术。
  实验比较：SLA2 kernel vs FlashAttn2（dense baseline）、VMoBA kernel、VSA kernel，在 RTX 5090 上不同 sparsity（90%/95%/97%）下的 TOPS（Tera Operations Per Second）和加速比。同时评测 QAT（含量化）vs 无 QAT（仅稀疏）的 kernel 加速比（量化带来约 1.3× 额外加速）。

- 后端平台是什么，配置是什么。
  NVIDIA RTX 5090，FP16 精度。单 GPU 评测。Kernel 性能使用 C/t 指标，其中 C = 4N²d 为理论计算量，t 为执行延迟。Wan2.1-14B-720P 模型超出单卡显存时启用 sequential CPU offloading（延迟已排除 offload 开销）。Block 参数：b_q=128, b_kv=64。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：FlashAttn2（https://github.com/Dao-AILab/flash-attention）作为 dense baseline kernel。修改内容：(1) 在 FlashAttention block-wise 框架上新增 M_c 掩码路由逻辑——遍历 block 对时根据 M_c[i,j] 分支到稀疏路径（quant→matmul→dequant→softmax→quant×quant→dequant）或线性路径（累加局部 KᵀV 和归一化因子）；(2) 嵌入低比特量化/反量化操作（SageAttention2++ 方案）；(3) 反向 kernel 手动推导 Q,K,V,Q^φ,K^φ 梯度，预计算 dH_i 和 dZ_i 使主循环仅涉及单次矩阵加法。
  
  前向 kernel（Algorithm 2 核心逻辑）：
  1. 预处理：K = K - colmean(K)，计算 Q^φ,K^φ = φ(Q),φ(K)，pool 压缩得 Q̄,K̄
  2. 路由：P_c = softmax(proj_q(Q̄)proj_k(K̄)ᵀ/√d)，M_c = Top-k(P_c, k%)
  3. Block 遍历（i=1..T_m, j=1..T_n）：
     - M_c[i,j]=1（稀疏路径）：quant(Q_i)→quant(K_j)→dequant(matmul)→softmax→quant(P)→quant(V_j)→dequant(matmul)→O_s 累加（含 FlashAttention 的 online softmax rescaling）
     - M_c[i,j]=0（线性路径）：H_i += (K_j^φ)ᵀV_j; Z_i += rowsum((K_j^φ)ᵀ)
  4. 组合：O_i^l = Q_i^φ H_i / (Q_i^φ Z_i)，O = α⊙O^s + (1-α)⊙O^l

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  SLA2 kernel 代码未开源。Baseline kernel（FlashAttn2、SLA、VSA、VMoBA）均使用官方开源实现。
  
  **评估原理**：测量 SLA2 attention kernel 在 RTX 5090 上的 wall-clock 执行时间 t，计算 TOPS = C/t，其中 C=4N²d。与 FlashAttn2 dense kernel、VMoBA、VSA kernel 对比，在不同 sparsity 下计算加速比 = FlashAttn2_time / SLA2_time。同时对比 QAT（量化稀疏）vs 无 QAT（纯稀疏）评估低比特量化的额外加速效果。
  
  **Kernel 输入到性能输出全过程**（以 Wan2.1-T2V-1.3B 单层 attention 为例）：
  1. **输入**：Q,K,V ∈ R^{N×d}（FP16），α ∈ R^{N/b_q×1}，proj_q,proj_k ∈ R^{d×d}，M_c ∈ {0,1}^{N/b_q×N/b_k}（预计算），b_q=128, b_kv=64
  2. **分块**：Q 分为 T_m 块（每块 b_q 行），K/V 分为 T_n 块（每块 b_k 行）
  3. **稀疏路径**（M_c[i,j]=1）：加载 Q_i block → 加载 K_j block → quant(Q_i) 到 INT8/FP8 → quant(K_j) 到 INT8/FP8 → Tensor Core INT8/FP8 matmul(Q_i,K_jᵀ) → dequant 回 FP16 → /√d → 与 online softmax 状态 m_ij, l_ij 合并 → quant(P_ij) → quant(V_j) → Tensor Core matmul(P_ij,V_j) → dequant → 累加 O_s
  4. **线性路径**（M_c[i,j]=0）：加载 K_j^φ, V_j → 累加 H_i += (K_j^φ)ᵀV_j → 累加 Z_i
  5. **组合**：O_i = α_i ⊙ O_i^s + (1-α_i) ⊙ (Q_i^φ H_i / (Q_i^φ Z_i))
  6. **输出**：O ∈ R^{N×d}（FP16）
  7. **性能测量**：计时从 Q,K,V 加载到 O 写回 HBM 的完整 kernel 执行时间，计算 TOPS 和加速比

## Inference Time Context Sparsity

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现在 FlashInfer 之上构建稀疏 decode kernel，支持 per-token、per-query、per-head 的细粒度不规则稀疏模式。给定稀疏索引和对应权重后，kernel 仅对选中的 KV cache 行计算加权 attention。在 GQA（Hq=32, Hkv=8, D=128, page size 16, NHD layout, 128K context）下与 FlashInfer dense decode baseline 比较延迟和加速比。同时评估含索引器开销（Double Sparsity 8 通道 16-bit）的端到端性能。

- 后端平台是什么，配置是什么。
  NVIDIA H100 80GB HBM3 GPU，FP16 精度。单 GPU 评测。GQA 配置 Hq=32, Hkv=8, D=128, page size 16, NHD layout, 128K context。batch size B ∈ {1, 4, 8, 16}。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：FlashInfer（flashinfer.ai）作为 baseline dense decode kernel。修改内容：在 FlashInfer 的 paged KV-cache 后端之上实现稀疏 decode kernel——不再读取完整 KV cache 页，而是根据稀疏索引 gather 选中的 token 行，然后执行分块 attention 计算。利用 KV cache 向量维度 d=128 提供的连续内存访问来使不规则稀疏模式在 GPU 上有效。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/skylight-org/sparse-attention-hub（Apache 2.0），kernel 实现论文未明确说明是否在该仓库中还是独立仓库。评估原理与全流程：

  **评估原理**：对比 sparse decode kernel 与 FlashInfer dense decode kernel 在相同硬件、相同 KV cache 配置下的 wall-clock 延迟（ms），计算加速比 = FlashInfer_time / sparse_time。稀疏度 S× 表示每个 query-head 仅关注 1/S 比例的 token。

  **Kernel 输入到输出全过程**：
  1. **输入**：Q ∈ R^{B×Hq×d}（query）、稀疏索引 I ∈ R^{B×Hq×k}（每个 query-head 选中的 k 个 token 位置）、稀疏权重 W ∈ R^{B×Hq×k}（对应 attention weight）、KV cache 页表。
  2. **Gather 阶段**：根据 I 从 paged KV cache 中 gather K_sparse、V_sparse。由于 token 级别稀疏无块结构约束，gather 操作利用 d=128 维度的连续读取来摊销随机访问开销。
  3. **Attention 计算**：对每个 batch 元素和每个 head 执行 `softmax(Q_h @ K_sparse_h^T / sqrt(d)) @ V_sparse_h`。使用 FlashInfer 风格的分块/tile 策略。
  4. **输出**：O ∈ R^{B×Hq×d}（attention output）。

  **关键性能结果**（Table 1，不含索引器开销的 kernel-only 加速比 vs FlashInfer）：

  | B | 2× | 4× | 10× | 20× | 50× | 100× | 200× | 500× |
  |---|-----|-----|-------|-------|-------|--------|--------|--------|
  | 1 | 0.32× | 0.63× | 1.45× | 2.58× | 5.57× | 10.25× | 11.05× | 11.14× |
  | 4 | 0.33× | 0.66× | 1.64× | 3.18× | 7.45× | 13.36× | 24.25× | 42.04× |
  | 8 | 0.38× | 0.77× | 1.90× | 3.75× | 8.88× | 16.82× | 29.64× | 76.14× |
  | 16| 0.45× | 0.89× | 2.21× | 4.35× | 10.54×| 20.09× | 37.32× | 76.77× |

  GQA 下：10× 稀疏度附近 break-even；50–100× 稀疏度下 5.5–20× 加速；500× 极端稀疏下大 batch 达 76× 加速。

  **含索引器开销**（Table 2，Double Sparsity 8 通道 16-bit 模拟索引器）：MHA 下 2× 稀疏度即 break-even，100× 稀疏度达 4.17× 加速；GQA 下 10–20× 稀疏度 break-even，100× 达 2.81×。更轻量索引器（HashAttention、PQCache、低精度 Double Sparsity）预期进一步扩大加速比。

## FuseFlow

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：FuseFlow 中的数据流排序（Dataflow Ordering）和并行化（Parallelization）属于 kernel 级调度优化，具体包括：
    1. **Dataflow Ordering 选择**：对每个稀疏 tensor algebra 表达式（如 SpMM、嵌套 matmul），FuseFlow 枚举不破坏融合约束的有效 dataflow order。不同 order 对应不同的稀疏迭代策略（如 Gustavson's algorithm 的 `j → i` 顺序 vs inner product 顺序），直接影响：
       - **坐标处理开销**：不同 order 下 intersect/union 操作的坐标流量级不同
       - **计算数据局部性**：value stream 的复用模式不同
       - **渐进复杂度**：对于特定稀疏格式，某些 order 可能是 discordant（与存储格式不一致），导致渐进复杂度退化
    2. **Partial Order Graph (POG) 约束下的合法调度枚举**：POG 编码了（i）用户局部 dataflow order、（ii）各张量的存储 mode order 约束。FuseFlow 对 POG 进行 topological sort，输出所有合法的全局 dataflow order。当施加局部最优 order 约束后，搜索空间可缩减 68.5%-99.9%。
    3. **Parallelization 调度**：通过 stream parallelizer/serializer 原语，对 SAMML 图中的指定索引变量进行并行化。支持 nested parallelism（两个不同的索引变量同时并行化）。编译器自动 partition 坐标空间、复制计算子图、合并并行结果流。
    4. **Factored Iteration 的运行时效果**：与 global iteration 将所有坐标处理集中到一个大的 input iteration subgraph 不同，factored iteration 将坐标处理分解为多个较小的子图，分别与计算阶段交错。这本质上是一种 **pipeline 级调度策略**，将坐标处理开销分散到多个较小的 pipeline stage 中。
  - **实验比较**：
    - **Dataflow Order sweep**（图 18）：对 nested matmul 枚举不同 dataflow order，最差 order 比最优慢 ~29×。每个 kernel 选择最优 dataflow order 可获得 ~29× 端到端 speedup。
    - **Parallelization factor sweep**（图 16a）：对 BigBird attention 的单个索引变量，parallelization factor 从 1 到 64，性能随并行度线性增长。
    - **Nested Parallelization**（图 16b）：同时并行化两个索引变量（各 factor=4）可获得 ~15.9× speedup。

- 后端平台是什么，配置是什么。
  - **后端平台**：可重构数据流架构（RDA）/ Coarse-Grained Reconfigurable Array (CGRA)。
  - **模拟器**：Comal cycle-accurate dataflow simulator，基于 DAM simulation framework [81]（Rust 1.87.0），集成 HBM2 内存模型（Ramulator 2.0 [48]）。
  - **FPGA**：Xilinx VU9P (AWS F1)，通过 Vitis HLS 综合。
  - **模拟器模拟内容**：对 SAMML 图中每个 dataflow 原语（Level Scanner, Intersect/Union, Repeater, ALU, Reducer, Level Writer, Coordinate Dropper, Stream Parallelizer, Stream Serializer）进行 cycle-accurate 的 fully pipelined 行为建模。模拟器追踪每条 stream 的 token 流动、pipeline stall、内存访问延迟。

- 评估性能的软件/脚本是什么。修改了什么。
  - **评估软件**：FuseFlow 自带的 benchmark 脚本（Docker 容器内运行），通过 Comal 模拟器在 SAMML 图上执行 cycle-accurate 仿真。对每种配置（fusion granularity、dataflow order、parallelization factor）生成独立的 SAMML 图并仿真。
  - **修改内容**：论文从零构建了 FuseFlow 编译器 + Comal 模拟器，非修改已有评估软件。关键组件：
    - **Comal 模拟器**：用 Rust 实现，基于 DAM framework [81]。接收 SAMML JSON/graph 描述作为输入，实例化每个 IR node 的模拟模型，按 dataflow 语义执行 token-passing 仿真。
    - **Fusion Heuristic**：独立的快速分析工具，不依赖仿真。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：GitHub fuseflow-artifact 仓库 + Figshare DOI（reserved），Docker 部署，MIT License。
  - **评估脚本使用示例**（来自论文 artifact appendix）：
    ```bash
    # 在 Docker 容器内生成 Figure 12（fusion 粒度对比）
    python3 scripts/figure_12.py --workers=2
    
    # 生成 Figure 16（parallelization sweep）
    python3 scripts/figure_16.py --workers=2
    ```
  - **Comal 模拟器评估原理**：Comal 模拟器基于 **token-passing dataflow 模型**：
    1. **输入**：SAMML dataflow graph（JSON 表示），包含 nodes（LS, Intersect, Rep, ALU, Red, LW, CD 等）和 edges（streams of crd/ref/val tokens）。附带 tensor metadata（维度、稀疏格式、稀疏度）。
    2. **Token 生成**：模拟器从输入张量的 Level Scanner nodes 开始，按 fibertree 格式生成坐标 tokens（crd）和值 tokens（val）。例如 CSR 矩阵 B 产生 `(i_crd → i_ref → j_crd → val)` 的 token 序列。
    3. **Token 流动与处理**：Tokens 沿 dataflow edges 流向下游 nodes：
       - **Intersect/Union nodes**：比较多个输入坐标流的当前 token，输出匹配（intersect）或全部（union）坐标。这是稀疏计算的核心——通过坐标对齐跳过零值。
       - **Rep nodes**：广播 value stream，按 repeat signal 复制值。
       - **ALU/Red nodes**：执行元素级计算和归约。Red node 累积 partial sums 直到接收归约结束信号。
       - **LW/CD nodes**：将结果坐标和值写回内存，丢弃零坐标。
    4. **Cycle 建模**：每个 node 的处理延迟（cycles）基于 pipeline depth 建模。Stream edges 在 token 队列满时产生 back-pressure stall。内存访问（HBM2）的延迟和带宽通过 Ramulator 2.0 模拟。
    5. **并行化建模**：Stream Parallelizer node 将坐标空间 partition 为多个并行子流，复制下游计算子图。Stream Serializer node 等待所有并行子流完成，合并结果。
    6. **输出**：仿真完成后的 cycle count（latency）、FLOPs count（每个 ALU/Red node 的操作数累加）、bytes transferred（每条 stream edge 的 token count × token width）。
    7. **性能指标计算**：
       - Speedup = baseline_cycles / fused_cycles
       - Operational Intensity = total_flops / total_bytes
       - 所有仿真结果通过 dense PyTorch 实现进行功能正确性验证。

## FlashAttention-4

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：FlashAttention-4 是一套针对 NVIDIA Blackwell (B200/GB200) GPU 架构的 attention kernel 实现，包含以下 kernel 级创新：
    1. **前向 warp-specialized 流水线**：采用 ping-pong 调度（类似 FA-3），两个 warpgroup 各处理一个 Q tile 的 softmax，另一个 warpgroup 驱动 tensor core MMA 和 TMA 访存，通过 tensor memory (TMEM) 解耦 P 的传输，将 output rescaling 分离到独立的 "correction" warpgroup 使其脱离关键路径。
    2. **后向 5-MMA 流水线重排**：利用 TMEM 允许更大的调度自由度，将前一迭代的 dQ/dK MMA 与当前迭代的 softmax 计算重叠。通过让 S 和 P 共享 TMEM 块，dP/dS/dQ 共享另一块，在 4 个 accumulator tile 的 TMEM 容量内实现深度流水。
    3. **2-CTA MMA 模式**：利用 Blackwell 的 CTA pair 协同执行单次 MMA 的能力（M=256, N=K=128），每个 CTA 只 stage 一半的 B 操作数，将后向 shared memory 流量从 3328 cycles 降至 2688 cycles（M=N=d=128 时）。dQ 步骤通过 DSMEM (distributed shared memory) 在 CTA pair 间交换半个 dS tile，将归约轴重新分区使得每个 CTA 做 `(M/2, 2N) × (2N, d)` MMA，dQ 的全局 atomic add 次数减半。
    4. **确定性后向 pass**：通过 semaphore lock 序列化全局归约，结合 "shortest-processing-time-first" (SPT) CTA 调度（对 causal masking：KV blocks 降序、query blocks 从对角线升序、dQ reduction 按降序 query block index），使确定性模式达到非确定性 1-CTA 后向的 75% 性能。
    5. **LPT (longest-processing-time-first) 调度**：对 causal masking 按 reverse mblock 顺序遍历（最长 tile 先处理）；对 variable sequence length (varlen) 通过预处理 kernel 按 per-worktile 执行时间排序 batches，生成 virtual→actual batch index mapping。
    6. **软件指数函数模拟**：用 FMA 单元上的多项式近似（degree-3，Sollya 优化系数，Horner 方法求值）+ IEEE 754 位操作实现 $2^x$，部分替代 MUFU.EX2（10-25% 条目用模拟），将指数吞吐量瓶颈分散到 FMA 单元。
    7. **条件 softmax rescaling**：仅当 $m_j - m_{j-1} > \tau$（$\tau = \log_2(256) = 8.0$，对应 rescaling factor 256.0）时才执行 rescaling，跳过不必要的重缩放操作以减少非 MMA 计算。
  - **实验比较**：FlashAttention-4 vs cuDNN 9.13、Triton 3.6、FlashAttention-2、Gluon、PyTorch 在 B200 GPU 上的 forward/backward pass TFLOPS。消融实验包括 deterministic vs non-deterministic backward、不同 CTA 调度策略（SPT、LPT with reverse mblock、LPT、naive）。

- 后端平台是什么，配置是什么。
  - **GPU**：NVIDIA B200 180GB SXM6 (1000W)，Blackwell 架构。
  - **关键硬件参数**：Tensor core BF16 吞吐 8192 ops/clock/SM（Hopper 的 2×），MUFU 指数单元 16 ops/clock/SM（与 Hopper 相同），SMEM 读带宽 128 bytes/clock/SM（与 Hopper 相同），TMEM 256KB/SM。
  - **软件栈**：CUDA 13.1, FlashAttention 2.8.3, Triton 3.6, PyTorch 2.10.0, CuTe-DSL 4.4.1。
  - **Benchmark 设置**：warmup 5 次，重复 10 次取平均。序列长度 1k-32k，batch size 使总 token 数为 32k。hidden dimension 2048，head dimension 64/128/(192,128)。FP16/BF16 输入。

- 评估性能的软件/脚本是什么。修改了什么。
  - **评估软件**：基于 FlashAttention 库的 benchmark 脚本（`flash_attn/cute` 目录下），使用 CuTe-DSL 编写全部 kernel，无 CUDA C++ 组件。
  - **修改内容**：从 FlashAttention-3 的 Hopper 专用实现完全重写为 Blackwell 适配的 CuTe-DSL 实现。核心修改包括：
    - Hopper MMA (64×128 accumulator in registers) → Blackwell MMA (128×128 accumulator in TMEM, fully asynchronous)
    - 寄存器 accumulator → TMEM accumulator（解耦 MMA 和 softmax 的寄存器争用）
    - warp 分配从 4 threads/row（interleaved）→ 128 threads/warpgroup（每线程一整行），消除 inter-warp shuffle
    - 引入 2-CTA MMA mode（Hopper 不支持）
    - 引入 DSMEM 跨 CTA 数据交换（Hopper 不支持）

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute（permissive license）
  - **评估原理**：测量 attention forward/backward 的 wall-clock 时间，转换为 TFLOPS/s。Forward FLOPs = `4 × seqlen² × head_dim × num_heads`（causal 时除以 2），Backward FLOPs = Forward FLOPs × 2.5（前向 2 次 matmul，后向 5 次 matmul + recomputation）。
  - **使用示例**：
    ```python
    from flash_attn_cute import flash_attn_func

    # Forward pass
    o = flash_attn_func(q, k, v, causal=True)

    # Backward pass (autograd)
    o.sum().backward()
    ```
  - **Kernel 输入到性能输出的全过程**：
    1. **输入**：Q, K, V ∈ R^{N×d}（BF16/FP16），可选 causal mask、varlen metadata。
    2. **Grid 启动**：CTA grid 维度为 (mblocks, heads, batches)。LPT 调度器按 batches（最外层）→ sections of heads（L2 cache 容量内）→ reverse mblocks 顺序遍历。对 varlen，预处理 kernel 按执行时间排序 batches，生成 virtual→actual batch index mapping。
    3. **前向 kernel 主循环**：每个 CTA 固定一个 Q tile（M=128 或 256），沿 KV 序列长度维度迭代。ping-pong 两个 Q tile：一个 tile 的 softmax（两个 warpgroup 交替执行 max reduction → exp → row sum）与另一个 tile 的 MMA（QK^T 和 PV）重叠。指数计算中 10-25% 条目走软件模拟路径（FMA polynomial），其余走 MUFU.EX2。条件 rescaling：仅当 row max 增量超过 τ=8.0 时才执行 $O = e^{m_{old}-m_{new}} O + e^{S-m_{new}} V$，否则跳过 rescaling。
    4. **后向 kernel 主循环**：5 次 MMA（S=QK^T, dP, dV=P^T dO, dS, dK=dS^T Q, dQ=dS K）。2-CTA 模式下：S, dP, dV, dK 的 MMA tile 为 M=256；dQ 使用 M=128 但双倍归约 N=256（通过 DSMEM 交换半个 dS tile）。dQ 的全局 atomic add 减半（每个 CTA 只写一半 tile）。确定性模式通过 semaphore lock 序列化 global reduction。
    5. **输出**：O ∈ R^{N×d}（forward），dQ, dK, dV ∈ R^{N×d}（backward）。性能结果以 TFLOPS/s 报告，peak 达到 1613 TFLOPS/s（71% 理论峰值利用率）。

## MAC-Attention: Match-Amend-Complete Attention for Efficient Long-Context Inference

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 MAC-Attention 的 fused persistent BF16 CUDA decode kernel（`mac_persistent_decode_bf16`），在单次 kernel launch 中完成所有操作：(1) in-kernel query-cache 匹配（L2 最近邻搜索）；(2) per-head hit/miss 分类；(3) load scheduling（根据匹配位置决定每个 head 的 KV 访问范围）；(4) 部分 attention 计算（full-KV for misses、rectification band + tail for hits）；(5) 稳定的 log-sum-exp merge（将复用 attention 与新鲜 attention 融合）；(6) cache writeback。
  实验比较：(1) kernel 级延迟对比：MAC-Attention kernel vs FlashInfer decode kernel 在 64K–256K context 下的 token-level attention 延迟；(2) hit curve benchmark：在不同命中率（0.0–1.0）和 context 长度（64K–127K）下的延迟 breakdown（Match、Compute、Merge 各阶段）。

- 后端平台是什么，配置是什么。
  NVIDIA Hopper GPU（H100 级别），BF16 精度，CUDA 13.0 环境。单 GPU 评测。batch size=1（decode benchmark）。主要配置参数：MAC_THRESHOLD=0.45（匹配阈值）、MAC_LOOKBACK_TOKENS_LEFT=512（ring cache 搜索窗口）、r=256（rectification band 宽度）。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：FlashInfer（`flashinfer-python`）作为 baseline decode kernel。评估脚本位于 `portable_plugin_repro/run_standalone_full_curve.sh`（hit curve benchmark）和 `portable_plugin_repro/run_correctness.sh`（正确性验证）。
  修改内容：不使用 FlashInfer 的 dense decode kernel，而是实现自有的 fused persistent kernel，包含以下 CUDA 源文件（JIT 编译 via `torch.utils.cpp_extension`）：
  - `mac_decode_persistent.cu`：主 fused MAC decode kernel
  - `mac_decode_rope_preserve.cu`：fused RoPE/query-preservation helper
  - `mac_merge_downdate_cache.cu`：prefill cache merge/update-downdate
  - `mac_prefill_update_cache.cu`：prefill cache update helper

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/YJHMITWEB/MAC-Attention.git（MLSys 2026，ACM AE Badge）。

  **评估原理**：对比 MAC-Attention fused decode kernel 与 FlashInfer dense decode kernel 在相同硬件、相同 KV cache 配置下的 per-token attention wall-clock 延迟（μs）。hit curve benchmark 通过控制匹配命中率（0.0–1.0）来解耦 Match/Compute/Merge 各阶段开销；不同 context 长度（64K–127K）展示 constant-complexity 特性。

  **Kernel 输入到性能输出全过程**：
  1. **输入**：
     - Q_n ∈ R^{Hq×d}（当前 token 的 query，pre-RoPE 和 post-RoPE 两份）
     - K_cache, V_cache：paged KV cache（FlashInfer page table 格式）
     - ring buffer：Q_ring ∈ R^{κ×Hq×d}, A_ring ∈ R^{κ×Hq×d}（滑动窗口缓存的 query 和 attention output）
     - 参数：threshold τ=0.45, window κ=512, rectification band r=256, current position n
  2. **Match 阶段（in-kernel）**：对每个 query head，在 Q_ring 中计算 pre-RoPE L2 距离 `||Q_n_pre_rope - Q_ring[i]||_2`。使用 reduction 找到最小距离对应的 index。若 min_dist > τ，标记为 miss。
  3. **Load Scheduling**：对每个 head 根据匹配结果确定 KV 访问范围——未命中 head：读取完整 KV_{0~n}；命中 head：仅读取 KV_{m−r~m}（rectification band）和 KV_{m~n}（tail）。
  4. **Attention Computation**：使用 persistent thread block 设计，每个 CTA 处理一个 head。对未命中 head 执行完整 FlashInfer 风格的 tiled attention；对命中 head，并行计算 rectification band attention 和 tail attention，再用 log-sum-exp merge 融合。
  5. **Merge 阶段**：命中 head 的 merge 公式——
     `A_n = logsumexp_merge(A_prefix_amended, A_tail)`
     其中 A_prefix_amended = A_m ⊖ Attn(Q_m, K_{m−r~m}, V_{m−r~m}) ⊕ Attn(Q_n, K_{m−r~n}, V_{m−r~n})。merge 使用在线 softmax 的数值稳定实现（跟踪 max logit 防止 exp 溢出）。
  6. **Cache Writeback**：将 Q_n_pre_rope 和 A_n 写入 ring buffer（FIFO 替换）。
  7. **输出**：A_n ∈ R^{Hq×d}（attention output for current token）。

  **Kernel 设计关键点**：
  - **Persistent kernel**：单次 launch 处理所有 heads 和所有阶段，避免多次 kernel launch 和中间结果的 HBM 往返。
  - **in-kernel matching**：L2 距离搜索在共享内存/L2 cache 中完成，避免把 Q_ring 写回 HBM 再读出。
  - **FP32 partial workspace**：通过 `MAC_PERSISTENT_PARTIAL_FP32=1` 启用 FP32 中间精度，保证数值稳定性。

  **关键性能结果**（120K context，batch=1）：
  | KV% | Full Attn. | Quest | RocketKV | Multipole | MAC-Attn. |
  |------|------------|-------|----------|-----------|-----------|
  | 1%   | 234.2 μs   | 581.2 | 822.8    | 192.4     | 62.9 μs   |
  | 5%   | 234.2 μs   | 594.7 | 844.7    | 210.8     | 64.0 μs   |
  | 10%  | 234.2 μs   | 608.5 | 1042.5   | 265.4     | 78.1 μs   |
  | 20%  | 234.2 μs   | 640.5 | 1855.6   | 324.6     | 103.8 μs  |

  MAC-Attention 在 attention 阶段比 FlashInfer 快 14.3×（256K 达 ~46×），end-to-end speedup 2.6×。

## PuzzleMoE

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 PuzzleMoE 的 **bit-packed encoding + 自定义 CUDA GEMV kernel**，用于高效执行稀疏合并后的 MoE 推理。核心包含两部分：
  1. **Bit-packed Encoding Scheme**：观察到 Bfloat16 权重的指数域在推理时仅占 5-bit 范围（值域 112-128），将指数整体偏移 112 后释放 3 个冗余 bit，用于嵌入：expert_i 的 mask bit（bit[13]）、expert_j 的 mask bit（bit[12]）、expert_i 的 sign bit（bit[15]）、expert_j 的 sign bit（bit[14]）。合并后的 W_{merged} 存储在标准 Bfloat16 格式中，masks 和 signs 零额外存储嵌入其中。指数 shift 操作无 perplexity 损失（Mixtral-8x7B 4.37→4.37, Deepseek-MoE 6.88→6.88），因为 FP16→BF16 转换中指数部分有相同处理。
  2. **Custom CUDA GEMV Kernel with On-the-fly Decoding**：设计专门的 GEMV kernel，在数据加载路径上对每个权重 W[i,j] 动态解码 mask bit 和 sign bit，解码后立即用于乘加计算。解码的伪代码见 Algorithm 1（论文 Section 3.2.2）：先从 packed Bfloat16 中提取 mask bit → mask=0 则输出 0（pruned weight）→ mask=1 则提取 sign bit + 重建指数（加回 112）+ 重建 Bfloat16 值 → 参与 GEMV 计算。解码逻辑是计算量极小的原地操作，搭载在 kernel 已有数据加载路径上（该路径已通过 warp 调度和合并访存优化到高吞吐），消除物化解码矩阵的内存开销和延迟。
  实验比较：(1) 内存使用对比：压缩后 Mixtral-8x7B 从 2×A100-80GB 降至 1×A100-80GB，Qwen3-MoE 从 2×A100-40GB 降至 1×A100-40GB；(2) 推理加速：Mixtral-8x7B 1.28× speedup，Qwen3-MoE 1.19× speedup（prefill=1024, decode=512）；(3) 压缩时间对比：PuzzleMoE 2min vs D2 55min（Mixtral-8x7B），PuzzleMoE 10min vs HC-SMoE 210min+（Deepseek-MoE, 64 experts）。

- 后端平台是什么，配置是什么。
  NVIDIA A100-80GB GPU（Mixtral-8x7B 评测），A100-40GB GPU（Qwen3-MoE 评测）。Bfloat16 精度推理。Custom CUDA kernel 利用 A100 的 CUDA cores 和 warp-level scheduling 优化 GEMV 计算。预填充长度 1024，解码长度 512。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：自研 CUDA GEMV kernel（自定义实现，非修改已有库）。Kernel 从零构建支持 on-the-fly mask/sign 解码的 GEMV 路径，而非基于 cuBLAS 或 CUTLASS 修改。
  修改内容：非修改已有软件。核心创新：
  - **On-the-fly Decoding**：在 GEMV 的 weight loading stage 内联解码，将 `if mask==0 → skip; else {decode_sign, rebuild_exp, multiply}` 融入 register-level 数据流。
  - **Bit-packed Memory Layout**：Weight 不存储 mask/sign 辅助数组，所有信息嵌入 Bfloat16 张量本身的未使用指数位中。
  - **无物化解码矩阵**：解码输出直接经 register → 乘加单元，不写回任何 off-chip 或 on-chip memory。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/Supercomputing-System-AI-Lab/PuzzleMoE
  评估原理与全流程：

  **评估原理**：对比 PuzzleMoE 压缩后模型在 custom CUDA GEMV kernel 上的 wall-clock 推理延迟（ms/token）与原始模型在标准 dense GEMV（PyTorch/cuBLAS）上的延迟。内存使用通过 PyTorch 的 `torch.cuda.memory_allocated()` 测量模型加载后的 GPU memory footprint。

  **Kernel 输入到性能输出全过程**：
  1. **输入**：
     - W_{merged} ∈ R^{d×h}（Bfloat16 packed weights，masks 和 signs 嵌入指数位）
     - X ∈ R^{B×d}（输入激活，Bfloat16）
     - expert_pos ∈ {0,1}（当前激活的 expert 在合并对中的位置）
  2. **Global Memory Load（合并访存）**：通过 warp-level 合并访存（coalesced access）从 HBM 加载 W_{merged} tile 和 X tile 到 registers/shared memory。与标准 GEMV 数据加载路径一致。
  3. **On-the-fly Decode（register-level）**：
     ```
     for each weight w in W_tile:
         mask_bit = (w >> (13 - expert_pos)) & 1      # 提取 mask bit
         if mask_bit == 0:
             w_decoded = 0                             # pruned, 跳过乘法
         else:
             sign_bit = (w >> (15 - expert_pos)) & 1   # 提取 sign bit
             exp = (w & 0x0F80) + (112 << 7)           # 重建指数 (base+112)
             mantissa = w & 0x007F                      # 保留尾数
             w_raw = (sign_bit << 15) | exp | mantissa  # 重建 Bfloat16 位模式
             w_decoded = reinterpret_as_bfloat16(w_raw)
     ```
     解码操作开销极低（~2 bit shifts + ~2 bitwise AND/OR + 1 integer add），在 A100 的 INT32 ALU 上执行，与 FP32 FMA 并行。
  4. **Fused Multiply-Accumulate**：解码后的 w_decoded 直接与对应 x 相乘累加 → `acc += w_decoded * x`。由于解码在 register 中完成，无额外内存往返。
  5. **Warp-level Reduction**：各 warp 内的 partial sums 通过 warp shuffle 归约，最终写入 output。
  6. **输出**：O ∈ R^{B×h}（Bfloat16 output activation）。
  7. **性能指标**：延迟（ms/token）、内存占用（GB）、加速比 vs baseline。

  **关键设计点**：
  - **Decoding 搭载在数据加载路径**：GEMV 的 weight loading 本身受 HBM 带宽约束（memory-bound），在此期间 INT32 ALU 空闲。Decoding 利用这些空闲 ALU cycles，不增加延迟。
  - **无物化开销**：对比 CSR 格式存储 50% 非结构化稀疏时由于索引存储开销导致零内存节省（Lasby et al. 2025），bit-packing 以零额外存储实现 mask 和 sign 编码。
  - **3-bit 嵌入 > 2-bit 冗余**：Pairwise 合并需要 2 mask bits + 2 sign bits = 4 bits，但通过压缩 mask（log₂3≈1.58 bit，利用三种状态而非四种），压缩后仅需 ~3.58 bit，嵌入 3 个冗余 bit 中（结合 per-group scale 进一步分摊到 3.35 bit/weight）。
  - **合并 3 个 expert 不兼容**：3-expert 合并需要 5 bit（3 mask + 2 sign 或等效编码），超出 3 个冗余 bit 容量，且性能下降（Mixtral PPL 4.36→5.22）。

  **关键系统结果**（50% sparsity，vs 全模型 dense GEMV）：
  | 模型 | 全模型 GPU | 压缩后 GPU | 加速比 |
  |------|-----------|-----------|--------|
  | Mixtral-8x7B | 2×A100-80GB | 1×A100-80GB | 1.28× |
  | Qwen3-MoE | 2×A100-40GB | 1×A100-40GB | 1.19× |

## Reducing GPU Memory Fragmentation via Spatio-Temporal Allocation Planning (STAlloc)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 STAlloc——一个面向深度学习框架的 GPU 内存分配器，通过离线规划（offline planning）+ 在线分配（online allocation）的混合范式减少训练时的 GPU 内存碎片。其 runtime 组件包括：(1) **Static Allocator**：按预先生成的 Static Allocation Plan 以 O(1) 查找直接返回预计算地址，消除 PyTorch 的 online best-fit 搜索开销；(2) **Dynamic Allocator**：对 MoE 等动态模型，在 Static Allocation Plan 的空闲区间（Dynamic Reusable Space）中按 best-fit 策略在线分配，无法容纳时 fallback 到 CUDA Caching Allocator；(3) **Allocation Profiler**：monkey-patch 方式轻量插桩，直接调用 cudaMalloc/cudaFree 记录每个 allocation request 的 size、lifespan（t_s, t_e）、computation phase、module name 和动态性标记。Plan Synthesizer 为离线组件，在训练前运行一次，通过 HomoPhase Group/HomoSize Group 的空时解耦规划生成近乎最优的 allocation plan，复杂度 O(N log N)。
  
  实验比较：(1) 内存效率/碎片率对比：STAlloc vs PyTorch Caching Allocator、PyTorch expandable_segments (ES)、GMLake 在多种训练配置（recomputation、Virtual Pipeline、ZeRO、offload 的组合）下的 memory efficiency 和 fragmentation ratio；(2) 训练吞吐量 overhead：各 allocator 的 end-to-end training throughput (FLOPS)，验证 STAlloc 不引入额外开销；(3) 可扩展性：不同模型规模（7B/14B/32B/72B）和集群规模（8-128 GPUs）下的 memory efficiency 和 OOM 行为；(4) Micro-batch size 鲁棒性：batch size 1-64 下各 allocator 的 memory efficiency 变化；(5) 训练框架通用性：Megatron-LM 和 Colossal-AI 两个框架下的效果；(6) 性能 breakdown：Static Allocator vs Dynamic Allocator 各自对碎片减少的贡献；(7) Profiling + Plan Synthesis 时间开销。

- 后端平台是什么，配置是什么。
  **NVIDIA 平台 (主测试)**：1 节点 8× NVIDIA A800-80GB GPU（Intel Xeon Platinum 8358 128-Core CPU），用于多训练配置评估。扩展性评估使用最多 16 节点 × 8× NVIDIA H200-141GB GPU（Intel Xeon Platinum 8558 192-Core CPU）。**AMD 平台**：8 节点 × 8× AMD MI210-64GB GPU（AMD EPYC 7K62 48-Core CPU），验证跨厂商通用性。软件栈：PyTorch 2.0/2.3/2.6、Megatron-LM、Colossal-AI、CUDA。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：在 Megatron-LM 和 Colossal-AI 训练脚本中通过 PyTorch PluggableAllocator 接口加载 STAlloc，替换原有 CUDA Caching Allocator。对比 baseline 包括 PyTorch 原生 Caching Allocator、PyTorch expandable_segments (ES) 和 GMLake（通过官方 Docker 镜像部署）。
  
  修改内容：STAlloc 约 3100 行 Python + 2300 行 C/C++ 全新实现，非修改已有 allocator。核心修改包括：
  - **PyTorch PluggableAllocator 接口实现**：通过 PyTorch 的 PluggableAllocator API 在训练前加载，接管所有 malloc/free 调用。兼容任何支持 PluggableAllocator 的 PyTorch 版本和 GPU 平台。
  - **Monkey-patching 插桩**：对训练框架进行轻量级插桩（不超过 5 行代码），记录 module name 和 computation phase（forward/backward）用于 spatio-temporal regularity 识别。
  - **Runtime Allocator**：训练初始化时通过 cudaMalloc 预分配 contiguous memory block 作为 static memory pool，后续 static allocation 通过地址偏移直接分配（无额外 GPU API 调用），dynamic allocation 通过计算 candidate intervals `A_c = A_a ∩ A_i` 并 best-fit 选择。使用 PyTorch hook API 追踪 model module 执行以确定当前层，路由请求到正确的 allocator。
  - **Plan Synthesizer（离线工具）**：独立运行的规划工具，输入 profiler 记录的 allocation traces M，输出 Static Allocation Plan D_s 和 Dynamic Reusable Space A_i。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/infinigence/STAlloc（GitHub 公开仓库）。Zenodo artifact：https://zenodo.org/records/17173036。

  **评估原理**：对比不同 allocator 在相同训练配置下的 (1) memory efficiency = M_a / M_r（实际分配大小 / 总预留大小）和 fragmentation ratio = 1 - efficiency；(2) end-to-end training throughput in FLOPS，计算公式由训练框架提供（per-iteration FLOPs / iteration_time）。

  **STAlloc 内存分配全流程**（以 Llama2-7B 训练 + recomputation 在 8×A800 上为例）：

  1. **Profiling 阶段**（训练前，运行 3 个 iteration）：
     - Allocation Profiler 通过 monkey-patch 拦截所有 torch 级别 malloc/free 调用。
     - 对每个 allocation request 记录：size s、allocation timestamp t_s、free timestamp t_e、computation phase p_s/p_e（forward/backward）、micro-batch ID、module name。
     - 对 dynamic layers（MoE expert layers）额外记录 originating module name l_s 和 l_e。
     - 输出：Memory request list M = {m := (s, t_s, t_e, p_s, p_e, dyn)}。以 Llama2-7B-R 为例，约 86,721 个 requests/iteration。

  2. **Plan Synthesis 阶段**（训练前，约 136 秒）：
     - **Static Request 分区**：M_s = {m | m.dyn == False}，M_d = {m | m.dyn == True}。
     - **HomoPhase Group 构建**：按 computation phase pair (p_s, p_e) 将 M_s 中的 requests 分组。相邻 groups 尝试 fusion（当 TMP 提升时合并）。每个 group 输出 local plan D_g（relatively addressed）。
     - **HomoSize Group 构建**：将 HomoPhase Group 的输出 plans 按 size 分组。对每个 size S 的 requests，Algorithm 1（Memory-Layer Construction）将 non-overlapping 的 requests 堆叠到同一 memory-layer 上，最小化 memory-layer 数量。
     - **Global Planning**：按 size 降序处理 HomoSize Groups——大 size 优先放置，小 size 尝试填入大 groups 的空闲 intervals。剩余 requests 构建新的 memory-layer。最终每个 request 被赋予具体的 memory address a。
     - **Dynamic Reusable Space 定位**：对 dynamic requests M_d，按 (l_s, l_e) 分组为 HomoLayer Groups。对每组 G(a,b)，计算 Static Allocation Plan 中在时间区间 T(a,b) 内完全空闲的连续地址段 A_i(a,b) = A \ A_o(a,b)，作为该组的 Dynamic Reusable Space。
     - 输出：Static Allocation Plan D_s = {d := (s, t_s, t_e, p_s, p_e, dyn, a)} + Dynamic Reusable Space maps per HomoLayer Group。

  3. **Runtime Allocation 阶段**（训练时）：
     - **初始化**：STAlloc 通过 cudaMalloc 预分配 contiguous memory block（size = D_s 的峰值内存），同时初始化 fallback Caching Allocator。
     - **Static Request 处理**：当 module hook 识别到当前层的 request 为 static（m.dyn == False 且 m ∈ D_s），Static Allocator 直接返回 pre-planned address a。O(1) 查找，无搜索、无 GPU API 调用。
     - **Dynamic Request 处理**：当 request 为 dynamic（m.dyn == True），Dynamic Allocator：
       1. 识别当前 request 所属的 HomoLayer Group G(a,b)。
       2. 获取该组的 Dynamic Reusable Space A_i。
       3. 计算实际可用区间 A_c = A_a ∩ A_i（A_a 为当前 static pool 的空闲区间）。
       4. 在 A_c 中按 best-fit 策略选择区间并分配。
       5. 若无法满足（A_c 中无足够大区间），fallback 到 Caching Allocator。
     - **地址冲突防护**：Static requests 使用 immutable planned addresses；Dynamic requests 仅在 pre-vetted Dynamic Reusable Space 内分配，保证不与 future static requests 冲突。

  4. **性能输出**：
     - Memory efficiency：训练框架通过 PyTorch API 查询 max_allocated_memory / max_reserved_memory → E。
     - Throughput：训练框架输出 per-iteration FLOPs ÷ iteration_time → FLOPS。

  **关键实验结果**：
  - Dense models（GPT-2, Llama2-7B）：>95%（up to 100%）memory efficiency，fragmentation reduction 90.3% vs PyTorch、93.4% vs GMLake、87.8% vs PyTorch ES。
  - MoE model（Qwen1.5-MoE-A2.7B）：93.7%-97.8% memory efficiency，fragmentation reduction 74.9% vs PyTorch、77.2% vs GMLake、34.0% vs PyTorch ES。
  - 扩展性：Qwen2.5 7B-72B on 8-128 H200 GPUs，under recomputation memory efficiency 99.1%（saving up to 56.3 GB）；under VPP 99%+（saving 15.7 GB avg）。
  - Throughput overhead：<0.05% vs PyTorch（statistical noise level）。
  - Enablement：Qwen2.5-14B on 16 GPUs，STAlloc 使 VPP+TP=2 配置免于 OOM，比次优可运行配置 throughput 高 5.4%-32.5%。

## MixLLM: LLM Quantization with Global Mixed-Precision between Output and Embeddings

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是面向混合精度（4-bit/8-bit）weight + 8-bit activation 的定制 CUDA GEMM kernel，核心包含三项优化：
  1. **Two-Step Dequantization**：Step 1 部分反量化 weight `(W_q − z)` 将 uint4 转 int8；Step 2 int8 Tensor Core 执行 MMA `A_q × (W_q − z)`；Step 3 fp16 scale 乘 `s_a × s_w`。两步设计确保利用 int8 Tensor Core 而非 FP16 路径——关键挑战是 per-group zero-point 的 uint4→int8 转换在 Tensor Core 前的正确性保证。
  2. **Fast Int-to-Float (I2F) Conversion**：使用 bias `0x4b400000` 将整数映射到连续的 float 表示区间，将昂贵的 I2F 指令转化为单次 float 减法。Bias 融合进 Tensor Core MMA accumulator 的初始化 `D = A*B + D`，消除显式 I2F 开销。
  3. **Multi-Level Software Pipeline**：引入 quantization group tile（128 元素为一组）作为新的 tile 粒度，与 warp tile 和 block tile 协同。使用两个输出 buffer——per-group accumulation buffer（I2F + scale multiply）和 global accumulation buffer。通过重叠 Global→Shared→Register 内存加载、Tensor Core MMA、SIMT 反量化实现流水线。Weight tensor 离线 prepacked 以优化 global memory 加载模式。使用 `vsub4` intrinsic 做 zero-point 减法。
  实验比较：(1) W4A8、W8A8、W4.8A8 vs float16 baseline 在多种矩阵维度下的 kernel 延迟和吞吐量；(2) vs TRT-LLM W4A16（SOTA weight-only 4-bit kernel）；(3) vs QoQ 在相同 bit-width 下的吞吐量对比。

- 后端平台是什么，配置是什么。
  NVIDIA A100 80GB GPU，CUDA 12.1。CUDA Graph 用于并行调度不同精度分区子问题的 kernel launch。单 A100 评测 ≤8B 模型，4×A100 评测 70B 模型。MatMul kernel 支持 int8 Tensor Core（mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32）。

- 评估性能的软件/脚本是什么。修改了什么。
  评估软件：自研 CUDA kernel（43.7% C++、18.0% CUDA），测试脚本 `mixllm/test/test_kernel.py`（参数 -m/-n/-k 矩阵维度、-r INT8 比例）。集成 vLLM v0.9.0（通过 `vllm_v0.9.0_patch/` 补丁）进行端到端 serving 性能基准测试，benchmark 脚本 `vllm/run_benchmark.sh`。
  修改内容：从零构建了混合精度 CUDA GEMM kernel——非修改已有 kernel。关键设计：
  - **Two-Step Dequant 融合 MMA**：将 uint4→int8 的 zero-point 减法融入 MMA Pipeline
  - **Fast I2F with Bias Fusion**：I2F 转为 float 减法并融合进 accumulator 初始化
  - **Prepacked Weight Layout**：离线将 weight 重排为 kernel 友好的 tile 格式

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  开源代码：https://github.com/microsoft/MixLLM（MIT License）。评估原理与全流程：

  **Kernel 测试使用示例**：
  ```bash
  # 测试混合精度 kernel，矩阵维度 M=4096, N=4096, K=4096, 10% INT8
  python mixllm/test/test_kernel.py -m 4096 -n 4096 -k 4096 -r 0.1
  ```
  参数 `-r` 控制 INT8 比例：r=0 全 INT4、r=1 全 INT8、r=0.1 即 W4.4A8 (10% INT8 + 90% INT4)。

  **Kernel 输入到性能输出全过程**：
  1. **输入**：
     - W_q_int4 ∈ R^{M×K}（uint4 packed weight，离线 prepacked tile 格式）
     - W_q_int8_idx ⊆ [0, M)（标记 INT8 通道索引，~10% 数量）
     - W_scales_int4 ∈ R^{M×K/128}、W_zeros_int4 ∈ R^{M×K/128}（per-group scales/zeros, group=128）
     - W_scales_int8 ∈ R^{M×K/128}（INT8 通道 per-group scales）
     - A_q_int8 ∈ R^{K×N}（INT8 量化的 activation，group=128）
     - A_scales ∈ R^{K/128×N}
  2. **Kernel Launch（CUDA Graph）**：通过 CUDA Graph 将 INT4 分区和 INT8 分区的 kernel 并行 launch，epilogue 分散输出到目标索引位置。
  3. **INT4 Kernel Path（~90% 通道）**：
     (a) **Weight Load**：从 global memory 加载 prepacked uint4 weight tile 到 shared memory
     (b) **Two-Step Dequant**：`W_deq = (W_q_uint4 − z)` → int8（使用 vsub4 intrinsic）
     (c) **Activation Load**：从 global memory 加载 int8 activation tile
     (d) **MMA**：`int8 Tensor Core: D += A_q_int8 × W_deq_int8`（mma.sync m16n8k32）
     (e) **I2F + Scale Multiply**：`out_float = (float(D) − bias_float) × s_a × s_w`（fast I2F 通过单次 float sub 实现）
  4. **INT8 Kernel Path（~10% 通道）**：
     (a) Weight load uint8（symmetric，无 zero-point）
     (b) MMA: int8 Tensor Core（同 INT4 路径的 MMA instr）
     (c) Scale multiply: `out_float = float(D) × s_a × s_w`
  5. **Software Pipeline 重叠**：Global Memory Load（weight tile i+1）、Shared Memory Load（activation tile i）、Tensor Core MMA（tile i）、SIMT Dequant（tile i+1）四级流水线。Quantization group tile（128 元素）作为 pipeline 的最小调度单元。
  6. **Fused Epilogue**：分别从 INT4 和 INT8 路径的输出 buffer scatter 到最终输出矩阵的正确位置（与 INT8 通道索引对应），开销"basically costless"。
  7. **输出**：O_fp16 ∈ R^{M×N}（FP16 output matrix）。

  **关键性能结果**（Figure 5）：
  - vs float16 baseline：W4A8 平均 1.90×、W8A8 平均 2.75×、W4.8A8 平均 1.88× speedup
  - vs TRT-LLM W4A16（SOTA weight-only）：W4A8 平均 1.26×、W8A8 平均 1.78× speedup
  - vs QoQ 同 bit-width：吞吐量 0.99× 但精度更优

## Streaming Tensor Programs (STeP)

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：STeP 在 SDA 上实现了三种此前 SDA 编程抽象无法表达的 kernel/runtime 级别调度优化：
    1. **Dynamic Tiling（动态分块）**：Tile size 在运行时根据实际 workload 确定。对于 MoE 的 batch dimension，tokens 按每个 expert 的 token 数量自适应分组为 tiles（而非 padding 到固定大小）。在 STeP 中通过 Promote 替代 Reshape 实现——使 Accum 可以 accumulate dynamically-shaped tiles。实现原理：去掉 Reshape 的 padding 限制，允许 Accum operator 在遇到 stop token 时动态结束当前 tile 的累积。
    2. **Configuration Time-multiplexing（配置时分复用）**：同一套 compute/memory 配置在不同 expert branch 之间动态时分复用。在 STeP 中通过 EagerMerge + RandomOffChipLoad 实现：EagerMerge 将不同 expert 的输入按到达顺序汇聚，RandomOffChipLoad 动态加载对应 expert 的 weight，而非使用固定地址的 LinearOffChipLoad。避免为每个可能的 branch 分配专用 compute 和 memory 资源。
    3. **Dynamic Parallelization（动态并行化）**：Work 一旦 downstream parallel pipeline 可用即分发，而非固定分配。在 STeP 中通过 Partition + EagerMerge 组合实现：Partition 将 requests 路由到多个 parallel region，selector stream 由两路 merge 形成——一路 FlatMap 做 round-robin 初始分配，一路 EagerMerge 信号 parallel region 可用状态。在 attention decoding 场景下，KV cache 长度不同的 requests 不会因固定分配导致负载不均衡。
  - **实验比较**：
    - **Dynamic Tiling vs Static Tiling**：MoE layer (Qwen3-30B-A3B: 128 experts, top-8 active; Mixtral-8x7B: 8 experts, top-2 active)。Metric: cycle count, on-chip memory。batch=64 时 PID 分别为 Mixtral 1.33× 和 Qwen3 2.11×；batch=1024 时 PID 分别为 1.86× 和 1.87×。Dynamic tiling 通过自适应 tile size 避免了 static tiling 的 trade-off（small tile → 频繁 off-chip reload vs large tile → padding waste）。
    - **Configuration Time-multiplexing vs 每 expert 独立配置**：Qwen3-30B-A3B MoE layer (batch=64)。Sweep 共享配置的 expert 数（1-128）。Static tiling (tile=32): compute utilization 提升 2.64×（<1% overhead）；Dynamic tiling: compute utilization 提升 2.51×（~5% overhead）。释放 62% compute + 46% memory 资源。
    - **Dynamic Parallelization vs Static Parallelization**：Attention layer，parallelize batch dimension by 4。KV cache length 分布来自 AzureLLMInferenceDataset（5000 requests sampled, 3 classes: lowest/highest variability + median SD）。Low variation: 1.14×~1.26× speedup；High variation: 1.47×~1.57× speedup。vs Static Coarse-grained (batch=16): 2.72× speedup (因为 coarse-grained 下多个 parallel region idle)。
    - **End-to-end Model**：Qwen3-30B-A3B, Mixtral-8x7B 完整 Transformer decoder layer fused 为 single STeP graph。Qwen3: 1.15× speedup + 69% less on-chip memory + 54% fewer compute。Mixtral: 1.27× speedup + 20% less on-chip memory。

- 后端平台是什么，配置是什么。
  - **目标后端**：Spatial Dataflow Accelerators (SDAs)——reconfigurable architectures with spatially distributed compute units and memory units, communicating via hardware FIFOs and NoC。
  - **模拟器**：STeP Rust cycle-approximate simulator（based on DAM framework [55]）。核心配置：
    - Compute units: 16×16 BFloat16 tiles, initiation interval = 1
    - On-chip memory bandwidth: 64 bytes/cycle（per unit）
    - Off-chip memory bandwidth: 1024 bytes/cycle（HBM, matching SN40L [33] configuration）
    - HBM node: Ramulator 2.0 cycle-accurate DRAM simulation
    - Inner-product matrix multiplication：partial-input tiles storage (16 × in_tile_col + |weight tile| + |output tile|)
  - **对比验证平台**：Bluespec SystemVerilog HDL model, cycle-accurate BlueSim simulator, HBM2 (8 stacks), on-chip mem BW = 256 bytes/cycle。

- 评估性能的软件/脚本是什么。修改了什么。
  - **评估软件**：STeP symbolic Python frontend + Rust simulator。Baseline 为 STeP 实现的 Revet-equivalent schedule（Revet [38] 作为 baseline 因其在显式 memory hierarchy 的 SDA 抽象中支持动态行为最广泛，但其 dataflow thread model 限制和缺乏动态 tile 支持使其无法表达 proposed optimizations）。
  - **修改/新增内容**：
    - **Dynamic Tiling kernel 实现**：将 static tiling 的 Reshape（固定 chunk_size + padding）替换为 Promote（允许 Accum 累积动态大小的 tile）。核心流程：
      ```
      # Static tiling: Reshape 将 [D_i, 1] 分块为固定的 [ceil(D_i/4), 4]
      # Dynamic tiling: Promote 跳过 padding，Accum 动态累积
      inp → Partition → Promote → Accum(RetileRow) → ...
      # Accum 遇到 S_b stop token 时自动结束当前 tile，开始新 tile
      ```
    - **Configuration Time-multiplexing kernel 实现**：在 MoE graph 的 compute region 前后插入 control-flow operators：
      ```
      EagerMerge(inputs) → RandomOffChipLoad(dynamic_addr) → Map(matmul) → Partition(outputs)
      # EagerMerge: 按到达顺序汇聚各 expert 输入
      # RandomOffChipLoad: 根据 dynamic selector 加载对应 expert weight
      ```
    - **Dynamic Parallelization kernel 实现**：
      ```
      Partition(requests, merged_selector) → [region_0, region_1, ..., region_N] → EagerMerge(outputs)
      # merged_selector = EagerMerge(FlatMap(round_robin), EagerMerge(availability_signals))
      ```
  - **调度评估原理**：
    - STeP simulator 将每个 STeP graph node 映射到 virtual compute/memory unit，追踪 FIFO 状态和 unit 占用。
    - Higher-order operators 通过 Roofline model 计算延迟：max(input_size/on_chip_BW, total_FLOPs/compute_BW, output_size/on_chip_BW)。
    - HBM node 使用 Ramulator 2.0 模拟实际 DDR 时序（而非简单带宽除法）。
    - 所有 scheduling decisions（tile sizes, parallelization factor, time-multiplexing degree）在 STeP graph 中显式编码。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **评估软件使用流程**：
    1. **编写 STeP 程序**：在 symbolic Python frontend 中用 STeP operators 表达 workload kernel graph（如 MoE layer、attention layer），同时指定 tiling、parallelization、memory placement schedule。
    2. **Frontend 符号分析**：Python frontend 自动推导 stream shapes 和符号化性能指标表达式（off-chip traffic, on-chip memory）。可代入具体数值获得 bound 估计。
    3. **Simulator 执行**：Rust simulator 读取 STeP graph，执行 cycle-approximate 模拟。每个 graph node 对应 virtual unit，edges 对应 FIFOs。Higher-order operators 按 Roofline model 计 cycle，HBM node 按 Ramulator 2.0 计 DRAM access cycle。
    4. **结果聚合**：Symbolic frontend 的符号表达式绑定到 simulator 的 runtime 具体值 → 输出 concrete metrics（cycles, traffic, memory, utilization）。
  - **评估原理（以 Dynamic Tiling 为例）**：
    ```
    Static tiling kernel 流程：
    [D_i, 1] → Reshape(chunk=4) → [ceil(D_i/4), 4] → Accum → [ceil(D_i/4), 4, 64] → matmul
    其中 ceil(D_i/4) 引入 padding，waste compute + memory。
    
    Dynamic tiling kernel 流程：
    [D_i, 1] → Promote → [1, D_i, 1] → Accum(dynamic) → [1, D_i, 64] 
    Accum 在 D_i 维度上动态累积，遇 S_1 自动结束 → 无 padding。
    
    性能差异来源：动态 tile 消除 padding FLOPs + 减少 on-chip memory（不需要预分配 ceil 空间）。
    ```
  - **复现流程**：
    ```bash
    # 1. 克隆仓库
    git clone --recursive https://github.com/stanford-ppl/step_artifact.git
    git clone https://github.com/stanford-ppl/step-artifact-hdl.git
    # 2. 构建 Docker 镜像
    docker build -f step_artifact/Dockerfile -t step_artifact .
    # 3. 启动容器
    docker run -dit step_artifact bash
    docker attach <CONTAINER_ID>
    # 4. 设置环境
    cd /root/step_artifact && source setup.sh
    # 5. 运行所有实验（~7 小时 Core，~24.5 小时 Full）
    source ae_cmd.sh  # Figures 9,10,12,13,14,15,21
    # 6. HDL 验证实验
    cp /root/step_artifact/hdl_validation/fig8.csv /root/step-artifact-hdl/step_reference.csv
    cd /root/step-artifact-hdl && ./run_dse_and_figure.sh
    ```

## Tilus

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - **实现**：Tilus 生成面向 NVIDIA GPU 的**低精度矩阵乘法 kernel**，通过代数布局系统和 thread-block-level 指令集实现高效的低精度计算。核心 kernel 优化包括：
    1. **高效权重加载流水线**：对于低精度权重（如 uint4），采用 (1) pipelined 异步 global→shared memory copy（cp.async），(2) shared→register 加载，(3) 无代价的寄存器张量 reinterpretation（利用布局代数，将同一物理 bits 重新解释为不同 dtype 和 layout），(4) 向量化 casting（PRMT/LOP3/bitwise 指令）。相比 Triton（需要 shared memory 做 layout conversion）和 Ladder（缺少 software pipelining），Tilus 的流水线消除了布局转换开销并支持 pipelining。
    2. **Software Pipelining**：支持 global memory load 与 compute 的异步重叠（CopyAsync + CopyAsyncCommitGroup + CopyAsyncWaitGroup）。
    3. **k-dimension 并行化**：支持沿 k 维度的并行分解策略。
    4. **自动向量化**：对 memory load/store 使用 cp.async.v4, lds128, ldg128 等向量化指令。
    5. **指令选择**：当寄存器张量布局与 `spatial(8,4).repeat(1,4)` 兼容时，选择 ldmatrix PTX 指令替代 lds。
    6. **Single Program Template**：一个参数化 program template（仅调 tile sizes）覆盖所有量化数据类型（uint1-uint8, int2-int8, float3-float8）和所有 batch sizes。
  - **实验比较**：
    - Tilus kernel vs cuBLAS FP16 kernel（作为 baseline，speedup=1.0 参考线）
    - Tilus kernel vs Triton 生成的 kernel
    - Tilus kernel vs Ladder 生成的 kernel
    - Tilus kernel vs QuantLLM 手工 kernel
    - Tilus kernel vs Marlin 手工 kernel
    - 数据类型：u8, f6e3m2, i4, u4, u2, u1（图 10）；uint1-uint8, int2-int8, float3-float8 全谱（图 11）
    - 算子维度：BS=1/16, K=8192, N=57344（Llama-3.3-70B 的矩阵乘法）
    - 指标：speedup（相对 cuBLAS FP16），latency（ms，CUDA Events 测量，取 50 次执行的中位数，每次执行前清空 L2 cache）

- 后端平台是什么，配置是什么。
  - **主平台**：NVIDIA L40S GPU（48 GiB, Ada Lovelace 架构），GPU driver 565.57.01，CUDA Toolkit 12.6.3。
  - **扩展平台**：NVIDIA A100（Ampere 架构）和 NVIDIA H100（Hopper 架构），用于跨架构验证。
  - **Tensor Core 利用**：使用 PTX mma 指令（如 mma.m16n8k16）进行 Tensor Core 矩阵乘法累加。激活类型支持 float16, bfloat16, int8。

- 评估性能的软件/脚本是什么。修改了什么。
  - **评估脚本**：artifact 通过 `bash run.sh` 一键执行所有实验（Docker 容器内自动运行）。
  - **Benchmark 方法**：每个 kernel 执行 50 次，使用 CUDA Events 测量延迟，取中位数。每次执行前清空 L2 cache 消除连续运行的 artifact。模型实验执行 10 次取中位数。
  - **集成评估**：Tilus 量化 kernel 集成到 vLLM v0.5.3 中，使用 continuous batching，对比 vLLM FP16 和 Ladder 量化 kernel。
  - **Auto-tuning**：Triton 和 Ladder 启用 auto-tuning；QuantLLM 使用启发式策略选择 kernel hyperparameters。Tilus 每个 operator 约 200 个配置，编译约 1 分钟，从中 auto-tune 选择最优配置。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：https://github.com/NVIDIA/tilus（compiler），https://github.com/yaoyaoding/tilus-artifacts（artifact, Apache 2.0），DOI: 10.5281/zenodo.16756859。Docker 镜像（~21 GiB）含全部依赖。
  - **Kernel 输入到性能输出的全过程**（以 FP16 × INT6 matmul, BS=16, K=8192, N=57344 为例）：
    1. **输入**：A 矩阵（FP16, shape [M,K]=[16,8192]），B 矩阵（INT6, shape [K,N]=[8192,57344]），两者位于 GPU global memory。
    2. **预处理（host 端）**：调用 Tilus transform program 将 B 从 i6[K,N] 变换为 u8[K/BK, N/BN, BK×BN×6/8] 紧凑布局（每 32 线程 24 bits = 3×u8 = 4×i6），确保后续 LoadGlobal 为连续对齐的字节访问（coalesced memory access）。
    3. **Kernel 启动**：Launch grid(M/BM, N/BN) thread blocks，BM=16, BN=8, BK=16。
    4. **主循环（k 维度 reduction）**：每轮迭代：
       a. LoadGlobal: 异步从 global memory 加载 A_tile (f16[16,16]) 和 B_tile (u8[...]) 到寄存器。
       b. View (reinterpret): 将 B 的寄存器张量从 u8[dtype] 无代价 reinterpret 为 i6[dtype]，利用布局代数保证两者 per-thread bit 数一致（24 bits/thread）。
       c. Cast: PRMT/LOP3 向量化指令将 i6→f16。
       d. Dot (mma): PTX mma.m16n8k16 指令执行 f16×f16 矩阵乘法累加到 f32 accumulator。
    5. **输出**：Cast f32→f16，StoreGlobal 写入 C 矩阵。
    6. **性能测量**：CUDA Event record start → kernel launch → CUDA Event record stop → cudaEventSynchronize → cudaEventElapsedTime 计算延迟。50 次 warm-up + 50 次测量，取中位数。speedup = T_cuBLAS_FP16 / T_Tilus。
