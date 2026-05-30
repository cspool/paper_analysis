## mHC Manifold-Constrained Hyper-Connections

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 mHC 设计了一套定制化融合 kernel 系统来消除 HC 引入的显存带宽瓶颈，使 n=4 时额外时间开销仅 6.7%。
    - **Kernel Fusion 策略**：
      1. **RMSNorm 重排序优化**：将 RMSNorm 的除以 norm 操作重排到矩阵乘法之后，减少高维 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$ 上的延迟。
      2. **融合核 1（Eq.14-15）**：将两次对 $\vec{x}_l$ 的扫描（一次用于 RMSNorm r 的计算，一次用于线性投影 $\vec{x}_l \varphi_l$）融合为单一 kernel，利用矩阵乘法单元最大化显存带宽利用率。前向单核完成；反向两个矩阵乘法同样融合为单核，避免重复加载 $\vec{x}_l$。前后向 kernel 均包含精细的流水线（load, cast, compute, store）处理混合精度。
      3. **融合核 2（Eq.16-18）**：将小系数上的轻量操作（RMSNorm 归一化 + gating factor 乘法 + bias 加法 + Sigmoid 激活）机会主义融合为单一 kernel，显著减少 kernel launch 开销。
      4. **融合核 3（Eq.19）**：Sinkhorn-Knopp 迭代（20 次交替行列归一化）实现在单一 kernel 内。反向 pass 实现自定义反向 kernel，在片上重新计算中间结果并遍历整个迭代过程。
      5. **映射应用融合核**：将 $\mathcal{H}_l^{\text{post}}$ 和 $\mathcal{H}_l^{\text{res}}$ 的应用与 residual merging 融合——将读取元素从 $(3n+1)C$ 降至 $(n+1)C$，写入元素从 $3nC$ 降至 $nC$。
    - **Recomputing 策略**：丢弃 mHC 所有中间激活的前向结果，在反向 pass 中重新执行 mHC kernel（不含沉重的 $\mathcal{F}$ 层计算）。对于 $L_r$ 连续层，仅需持久化首层输入 $\mathbf{x}_{l_0}$。最优块大小 $L_r^* \approx \sqrt{nL/(n+2)}$ 与 pipeline stage 中的层数对齐。
    - **DualPipe 通信重叠**：扩展 DualPipe schedule（DeepSeek-V3）来重叠 mHC 在 pipeline stage 边界的通信和计算。MLP 层的 $\mathcal{F}_{post,res}$ kernel 在专用高优先级 compute stream 上执行，避免阻塞通信流；attention 层避免 persistent kernel 防止长时间 stall。
  - 实验比较：
    - 系统开销度量：mHC (n=4) 仅引入 6.7% 的额外训练时间。
    - 对比 Baseline（标准残差连接）和 HC 的 I/O 开销分析（Tab. 2）：HC 将显存访问从 ~3C 增加到约 $(8n+2)C$，mHC 通过融合 kernel 缓解。

- 后端平台是什么，配置是什么。
  - 论文未明确说明 GPU 型号和规模。利用 TileLang 框架实现 kernel，涉及 bfloat16/tfloat32/float32 混合精度计算。
  - 训练通信涉及 NVLink（节点内）+ NIC（节点间），使用 DualPipe pipeline parallelism。

- 评估性能的软件/脚本是什么。修改了什么。
  - **TileLang** (Wang et al., 2025)：用于实现大部分 mHC kernel（除 Eq.14-15 的矩阵乘法融合核），TileLang 简化了复杂计算过程的 kernel 实现，以最小工程代价充分利用显存带宽。
  - **DualPipe** (DeepSeek-V3)：在 DualPipe schedule 基础上扩展，在 pipeline stage 边界增加通信-计算重叠。
  - **Kernel 修改详情**：
    - 新增 3 个专门的 mHC 计算 kernel（计算 $\mathcal{H}_l^{\text{pre}}, \mathcal{H}_l^{\text{post}}, \mathcal{H}_l^{\text{res}}$）
    - 新增 2 个映射应用 kernel（$\mathcal{F}_{pre}$ 和 $\mathcal{F}_{post,res}$）
    - Sinkhorn-Knopp 前向+自定义反向单 kernel 实现

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明 mHC kernel 代码是否开源。TileLang (https://github.com/anthropics/tilelang) 是开源框架。
  - **mHC kernel 的完整执行流程**（以单层前向为例）：
    1. **输入数据**：$\mathbf{x}_l \in \mathbb{R}^{n \times C}$ (bfloat16)，weights $\varphi_l \in \mathbb{R}^{nC \times (n^2+2n)}$ (tfloat32)，bias $\mathbf{b}_l \in \mathbb{R}^{1 \times (n^2+2n)}$ (float32)，scalars $\alpha$ (float32)
    2. **Kernel 1 — 融合线性投影+Norm**：
       - flatten $\mathbf{x}_l \to \vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$
       - 单 kernel 完成：(a) 计算 $r = \|\vec{\mathbf{x}}_l\|_2 / \sqrt{nC}$；(b) 计算 $\vec{\mathbf{x}}_l \varphi_l$ 得到 $[\tilde{H}^{\text{pre}}, \tilde{H}^{\text{post}}, \tilde{H}^{\text{res}}]$
       - 混合精度流水线：load (bfloat16) → cast (float32) → compute (tfloat32 MMA) → store (float32)
       - 评估原理：此 kernel 将两份独立的内存扫描合并为一份，消除重复从 HBM 读取 $\vec{\mathbf{x}}_l$ 的开销
    3. **Kernel 2 — 融合后处理**：
       - 输入：Kernel 1 的 float32 输出 + $\alpha$ scalars + bias
       - 计算：$1/r \cdot [\alpha^{\text{pre}}\tilde{H}^{\text{pre}}, \alpha^{\text{post}}\tilde{H}^{\text{post}}, \alpha^{\text{res}}\tilde{H}^{\text{res}}] + \mathbf{b}_l$
       - 随后：$\sigma(\cdot)$ 和 $2\sigma(\cdot)$ 应用于 pre/post 分支
       - 评估原理：全部小系数上的逐元素操作融合为单 kernel launch，避免数十次微 kernel 的 launch 开销
    4. **Kernel 3 — Sinkhorn-Knopp 迭代**：
       - 输入：$\tilde{H}^{\text{res}}$
       - 计算：$\mathbf{M}^{(0)} = \exp(\tilde{H}^{\text{res}})$，然后 20 次交替行列归一化
       - 评估原理：完整迭代在单 kernel 内执行，避免多次 kernel launch 和中间结果的 HBM 读写
    5. **Kernel 4 — Pre 映射应用**：
       - 计算：$\mathcal{F}_{pre} = H^{\text{pre}} \cdot \mathbf{x}_l \in \mathbb{R}^{1 \times C}$（n-stream → 1-stream 聚合）
    6. **层计算**：标准 attention/FFN $\mathcal{F}(\mathcal{F}_{pre}, W_l) \in \mathbb{R}^{1 \times C}$
    7. **Kernel 5 — Post+Res 融合映射应用**：
       - 计算：$\mathbf{x}_{l+1} = H^{\text{res}} \mathbf{x}_l + H^{\text{post}^\top} \cdot \mathcal{F}(\cdot)$
       - 融合 residual merging：读取量从 $(3n+1)C$ 优化为 $(n+1)C$，写入量从 $3nC$ 优化为 $nC$
    8. **反向 pass**：
       - 重新执行 Kernel 1-5（不含层函数 $\mathcal{F}$）计算梯度所需中间激活
       - Sinkhorn-Knopp 反向为定制 kernel，在片上重计算中间结果
  - **性能对比**：
    - Baseline（标准残差连接）I/O: 读 2C, 写 C
    - HC（无融合 kernel）I/O: 读 $(5n+1)C + n^2 + 2n$, 写 $(3n+1)C + n^2 + 2n$
    - mHC（融合 kernel 后）: 额外训练时间开销仅 6.7%（n=4）

## MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoESys 在 kernel/运行时层面做了三项优化：（1）**Hierarchical AlltoAll Communication**——针对 MoE expert parallelism 中的 AlltoAll 通信，利用网络拓扑层次（intra-node NVSwitch + inter-node NIC/switch），先 intra-node AlltoAll 通过 NVSwitch 收集数据，再按相同 rank 的 GPU 分组做 inter-node AlltoAll，避免跨不同 rank 的 NIC 通道产生交换机路由冲突；（2）**Custom H2D/D2H Kernels**——使用 CUDA Pinned Memory 优化 Host-to-Device/Device-to-Host 数据传输，减少层间传输延迟；（3）**Fused Multi-head Attention Kernel**——来自 NVIDIA BERT MLPerf 1.1 实现的 Fused MHA kernel，减少 kernel launch 次数。
  - 实验比较：（1）Hierarchical AlltoAll vs baseline AlltoAll 的通信时间占比和 end-to-end training time breakdown（FWD/BWD/OPT/Comm），在 2 nodes(16 GPUs) 和 4 nodes(32 GPUs) 下对比不同参数规模的 MoE 模型；（2）整体 training throughput（tokens/s）提升；（3）Cross-wise comparison 中各 kernel/通信优化的 peak memory 和 computation speed 贡献。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A100 80GB，单节点 8 GPU 通过 NVSwitch 互联（NVLink 900 GB/s），节点间通过 NIC（100G Mellanox ConnectX 系列）+ leaf/spine switch 互联。
  - 网络拓扑：m 个 cluster，每个 cluster 含 p 个 GPU node。Leaf switch 按 rank 分组，同一 rank 的 NIC 直连同一 leaf switch，跨 rank 需经过 spine switch（带宽低于 leaf switch）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 框架：PaddlePaddle / PaddleFleetX 分布式训练框架。
  - 修改的通信原语：将标准 AlltoAll 替换为 Hierarchical AlltoAll——在 NCCL 或 PaddlePaddle 通信层实现两阶段 AlltoAll（intra-node via NVSwitch → inter-node via NIC grouped by rank）。
  - CUDA kernel 修改：H2D/D2H 使用 cudaHostAlloc 分配 pinned memory + 异步 cudaMemcpyAsync；Fused MHA 集成自 NVIDIA MLPerf BERT 实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - MoESys 基于开源 PaddlePaddle/PaddleFleetX。论文称代码将发布于 PaddlePaddle GitHub，截至搜索未找到独立 MoESys 仓库。
  - Hierarchical AlltoAll 的执行原理与全流程：
    1. **输入**：MoE Gate 网络已确定每个 token 的 expert 分配，每个 GPU 需要将 tokens 发送到对应 expert 所在的 GPU。
    2. **阶段一——Intra-node AlltoAll via NVSwitch**：
       - 8 GPU per node，每个 GPU 通过 NVSwitch 以 900 GB/s 带宽做全交换。
       - 目标：将跨 rank 的数据先在本节点内通过 NVLink 搬运到对应 rank 的 GPU。
       - 例如 GPU0 (rank 0) → GPU7 (rank 7) 的数据：通过 NVLink 从 GPU0 搬到 GPU7。
    3. **阶段二——Inter-node AlltoAll via NIC (grouped by rank)**：
       - 按 rank 分组——所有 node 的 GPU0 (rank 0) 组成通信组，通过 NIC + leaf switch（不走 spine switch）做 AlltoAll。
       - 所有 node 的 GPU7 (rank 7) 同理。
       - 优势：同 rank 的 NIC 接入同一 leaf switch，不经过 spine switch（低带宽瓶颈），避免跨 rank 通信的 spine switch 路由开销。
    4. **对比 Baseline AlltoAll**（如 GPU0 of Node1 in Cluster A ↔ GPU7 of Node2 in Cluster B）：
       - Baseline 路径：NIC1 → LE1 → SPq → LE1 → NICn（经过 spine switch，高延迟 + 带宽竞争）
       - Hierarchical 路径：GPU0 → NVLink → GPU7（intra-node）→ NICn → NICn（inter-node，同一 leaf switch）
    5. **性能输出**：通信时间占比显著下降。以 80.7B MoE model / 4 nodes 32 GPUs 为例，end-to-end training 性能提升 10.3%，通信阶段 speedup 15.5%。peer-to-peer 通信效率提升 p 倍（p=单节点 GPU 数）。
  - Custom H2D/D2H Kernel 原理：
    1. 使用 `cudaHostAlloc` 分配 pinned (page-locked) host memory，避免默认 pageable memory 的额外 copy。
    2. `cudaMemcpyAsync` 在独立 CUDA stream 上异步执行 H2D/D2H transfer，与 GPU computation kernel 在 default stream 上的执行重叠。
    3. 在 MoE layer 切换时，将下一层 expert 参数从 CPU pinned memory 异步传输到 GPU global memory，同时 GPU 执行当前 layer 的计算。

## MoE-Lens: Towards the Hardware Limit of High-Throughput MoE LLM Serving Under Resource Constraints

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-Lens 实现了 **手工优化的 CPU Decode Attention kernel** 用于 CPU-GPU 混合 MoE 推理系统的 CPU 端 attention 计算：
    1. **AVX512 SIMD intrinsics 实现**：使用 hand-written SIMD intrinsics（manual vectorization）实现 decode attention，支持 BF16 KV cache 数据格式，upconvert 到 FP32 进行 dot product 和 saxpby 计算（Equation 6）。包含 loop unrolling 和 data prefetching 优化。
    2. **Cache-optimized memory access**：针对 CPU 端 KV cache 访问模式优化，通过 prefetch 指令减少 cache miss。
    3. **多线程 scaling**：支持 multi-threaded execution，实测在超过 20 threads 后吞吐增益饱和（memory controller contention）。
  - 实验比较：(1) MoE-Lens intrinsics kernel vs auto-vectorized baseline（均使用 AVX512 ISA）的 KV cache tokens attended per second；(2) 不同线程数（1-28 threads）下两者的吞吐对比和 scaling behavior；(3) 与 system throughput requirement（假设 KV cache size = 2× model size）的对比。

- 后端平台是什么，配置是什么。
  - CPU: Intel Platinum 8380（支持 AVX512），单 socket 使用（numactl 限制）。
  - GPU: NVIDIA A40 48GB（simulated to 16-24GB）。
  - Memory: DDR4-3200，单 socket 8 channels，~150GB/s measured aggregate bandwidth。

- 评估性能的软件/脚本是什么。修改了什么。
  - MoE-Lens 自身实现的 C++ CPU decode attention kernel（PyTorch extension）。论文未开源（arXiv: 2504.09345），无公开代码。
  - 对比 baseline：auto-vectorized 版本（依赖编译器自动向量化，同样使用 AVX512 ISA）。
  - 修改：手工替换编译器自动生成的向量化代码为 hand-tuned SIMD intrinsics 实现。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？
  - **MoE-Lens 未开源**。
  - **评估原理（基于论文描述）**：
    1. **Kernel 输入**：decode 序列的 query vectors（shape: $[n_q, d]$）和 KV cache blocks（存储在 pinned CPU memory，BF16 格式），以及 GQA group size s。
    2. **Kernel 计算**：对每个 query 在 KV cache 中做 attention——vector dot product（query·key）→ softmax → saxpby（attention weights × value）。手动向量化利用 AVX512 512-bit registers 一次处理 16 个 BF16 元素（converted to FP32 = 8 elements per register）。
    3. **优化策略**：loop unrolling 减少分支和循环开销；data prefetching 指令提前将下一轮 KV cache 数据加载到 CPU cache；FP32 累加保证数值精度；BF16→FP32 upconvert 和 FP32→BF16 rounding 每一步显式处理。
    4. **性能输出**：throughput = KV cache tokens attended per second。Intrinsics kernel 单线程 4.7× auto-vectorized、全线程 3.1× auto-vectorized。满足 system target（KV cache = 2× model size 时所需的 attention throughput）。
  - **运行时集成**：CPU attention kernel 在 VSLPipe 的 CPU Task (C) 阶段被调用，与 GPU Task B 的 GEMM 计算并行执行（图 9 pipeline）。CPU attention 结果通过 H2D transfer 回传 GPU 用于后续 O projection 和 MoE layer。

## MoE-Gen: High-Throughput MoE Inference on a Single GPU with Module-Based Batching

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-GEN 实现了 **CPU attention kernel** 用于卸载 self-attention mechanism（$QK^T$）计算到 CPU，核心包含：
    1. 基于 **AVX intrinsics** 的 Grouped Query Attention（GQA）实现，使用 BF16 数据格式。由于原生 BF16 硬件支持仅限较新高端 CPU，在 FP32 中表示 BF16 数据（显式清零低 16 位尾数），所有计算和累加在 FP32 精度，每次点积累加后按 BF16 舍入规则舍入并清零尾数位，保证与 PyTorch GPU attention 数值一致。
    2. CPU kernel 针对 cache 性能优化（类似 FlashAttention 的 CPU 版本思想），使 CPU 处理 GEMV（$QK^T$ 计算是 matrix-vector 乘法）的速率达到与 PCIe4.0 传输 KV-cache + GPU 计算的时间可比。
    3. 运行时调度：在 attention 阶段，按 $\omega$ 比例将 tokens 分配至 CPU 执行 self-attention，GPU 和 CPU 并行执行各自的 attention 计算，结果在 Post-Attention 前 concatenate。CPU kernel 直接访问 host memory 中的 KV-cache，无需 HtoD 拷贝，节省 PCIe 带宽给 expert weight 预取。
  - 实验比较：(1) MoE-GEN(G)（纯 GPU，$\omega=0$）vs MoE-GEN(H)（CPU attention 卸载，$\omega > 0$）的 decoding throughput；(2) 不同 $\omega$ 值（0-100%）对 throughput 的影响曲线；(3) 不同 CPU 计算能力（C1/C2: AMD 7453 28-Core vs C3: AMD 7313P 16-Core）下的最优 $\omega$ 选择；(4) 不同模型下的 CPU:GPU 最优 split ratio（Mixtral-8x7B: 6:4, Mixtral-8x22B: 7:3, DeepSeek-V2: 0:10）。

- 后端平台是什么，配置是什么。
  - GPU: NVIDIA A5000 (24GB, PCIe 4.0), NVIDIA A6000 (48GB, PCIe 4.0)
  - CPU: AMD EPYC 7453 28-Core (C1/C2), AMD EPYC 7313P 16-Core (C3)
  - Host Memory: 256GB (C1), 512GB (C2), 480GB (C3)

- 评估性能的软件/脚本是什么。修改了什么。
  - 自研 MoE-GEN Engine（约 3000 行 C++ + 2000 行 Python）。benchmark 脚本位于开源仓库 https://github.com/EfficientMoE/MoE-Gen。
  - 修改：在 MoE-GEN Engine 的 attention 阶段插入 CPU kernel dispatch 路径，按 $\omega$ 比例将 self-attention 计算分流到 CPU。
  - 对比的 baseline 系统：vLLM、Llama.cpp、DeepSpeed-Inference、FlexGen*、MoE-Lightning* 均在 GPU 上执行所有 attention 计算。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？
  - **评估原理**：通过 CUDA events 插桩每个 module 的 forward pass 测量计算延迟，通过 `torch.memory_stats` API 测量峰值内存（CUDA context + KV-cache + activations），通过 `cudaMemcpy` 计时测量 PCIe 带宽。离线 profiling 各模块在不同 batch size 和 sequence length 下的数据后，由 DAG-based scheduler 选择最优配置（包括 $\omega$）。
  - **从 kernel 输入到性能输出的过程**：
    1. Profiling 阶段：对 attention 模块在 GPU 和 CPU 上分别测量不同 $b_a$ 下的 latency（CUDA events / CPU timer），以及 KV-cache HtoD copy 时间。
    2. Search 阶段：scheduler 枚举 $\omega \in \{0, 0.1, ..., 1.0\}$，对每个配置估算 attention 阶段总时间 = $\max(\omega \cdot T_\text{CPU\_attn}, (1-\omega) \cdot (T_\text{KV\_copy} + T_\text{GPU\_attn}))$ + 其它固定开销。选择使 critical path 最短的 $\omega$。
    3. Runtime 阶段：Engine 在执行 attention 时，将 $\omega \times b_a$ 个 tokens dispatch 到 CPU kernel（使用 AVX BF16-GQA），$(1-\omega) \times b_a$ 个 tokens 走 GPU 路径。结果通过 concatenate 合并后进入 Post-Attention。
    4. 输出：throughput = $B / T_\text{DAG}$（tokens/s）。

## MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：为 MoDES 的 expert skipping 推理编写了高效 **CUDA kernel**，核心包含三部分：
    1. **Router 内核内嵌 DMT 逻辑**：在计算 router logits 和 top-k 后，直接在 router kernel 内部使用 branch-free masked comparison 将 modality-specific threshold 与 importance score 比较，跳过的 expert route 被赋值为哨兵 expert ID（如 M+1）。此操作不引入额外 kernel launch 或单独的决策 pass，仅在 top-k 列表上添加少量 element-wise 操作。
    2. **Dispatch/Gather 时过滤**：在 MoE dispatch 和 gather 阶段，哨兵条目被自动过滤掉，不进行调度和执行，从而同时减少 expert compute 和 expert loading。
    3. **Group GEMM**：使用 Grouped General Matrix Multiplication，将多个保留 expert 的矩阵乘法合并入单个统一 kernel launch 中并发执行。每个 expert 的计算作为 group 内独立子任务。离线对不同的代表性 activation pattern 做 grid search 确定最优 kernel tile size，确保各种动态 workload 下的高计算吞吐。
  - 实验比较：在单张 H200 GPU 上测量 Kimi-VL-A3B-Instruct（83% skip）和 Qwen3-VL-MoE-30B-A3B-Instruct（88% skip）的 prefill 和 decoding 推理速度。MoDES 实现 prefilling 约 2× 加速，decoding 约 1.2× 加速。Baseline（DiEP 等）在相同 skipping ratio 下的 kernel 加速比与 MoDES 类似（差别 <1%），但 MoDES 在 benchmark 精度上远超 baseline。

- 后端平台是什么，配置是什么。
  - 单张 NVIDIA H200 GPU。Prefilling batch size=8，decoding sequence length=1024。校准与搜索阶段使用 8×H200 GPU。

- 评估性能的软件/脚本是什么。修改了什么。
  - 论文作者自行编写 CUDA kernel 用于 MoE layer 的推理加速，并非基于现有评估框架修改。主要修改了 MoE layer 的三个环节：
    1. Router kernel：内嵌 DMT 判断逻辑（branch-free masked comparison + sentinel ID assignment），无额外 kernel launch。
    2. MoE dispatch/gather kernel：增加 sentinel entry 过滤逻辑。
    3. MoE compute kernel：使用 Group GEMM 替代逐 expert 独立 kernel launch，配合离线 profiling 确定最优 tile size。
  - 评估原理：测量给定 input 下从 token 进入 MoE layer 到输出完成的总 wall-clock 时间，对比原始模型（k=6 或 k=8 所有 expert 激活）与 MoDES 跳过部分 expert 后的加速比。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/ModelTC/MoDES
  - Kernel 执行全过程（以单 token 经过第 l 层 MoE FFN 为例）：

```
输入: 单 token hidden state x ∈ R^d

1. [Router Kernel - 标准部分]
   r = W_router @ x                          # routing logits, (M,)
   pi = softmax(r)                           # routing probabilities, (M,)
   topk_vals, topk_indices = topk(pi, k)     # 取 top-k

2. [Router Kernel - DMT 嵌入逻辑，无额外 kernel launch]
   alpha_tilde = precomputed_alpha_tilde[l]   # 标量, 离线获得
   for i in [0..k-1]:
       s_i = alpha_tilde * topk_vals[i]       # Eq.(3) importance score
       threshold = is_text ? tau_t : tau_v    # modality-specific
       # Branch-free mask: s_i < threshold → sentinel
       mask = (s_i >= threshold)
       topk_indices[i] = mask ? topk_indices[i] : M+1  # M+1 = sentinel
       topk_vals[i]    = mask ? topk_vals[i]    : 0

3. [Dispatch/Gather - Sentinel 过滤]
   有效 routes = filter(topk_indices != M+1)  # 仅 dispatch 保留的 expert
   仅保留的 expert 被激活，哨兵 expert 不调度

4. [Group GEMM - 统一 kernel launch]
   # 所有保留 expert 的矩阵乘法在一个 kernel 中并发执行
   # 使用离线 profiling 确定的最优 tile size
   for each valid expert i:
       # 作为 group 内子任务执行
       output_i = x @ W_expert_i
   y = sum(pi_i * output_i for valid i)      # 加权聚合

输出: 更新后的 token hidden state y ∈ R^d
```

关键性能特征：(1) DMT 决策开销极小——仅对 k 个 top-k 条目做 element-wise 操作，warp divergence 可忽略；(2) Group GEMM 避免了逐 expert 独立 kernel launch 的 overhead；(3) Decoding 阶段加速比较低（~1.2× vs ~2×）因 decoding 为 memory-bound 且仅处理 text token（text token 的 expert skipping ratio 低于 vision token）。

## MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现自定义 **collective communication runtime**（约 6K 行 C++），基于 NCCL 扩展，支持 **in-training topology reconfiguration**。核心组件：
    1. **Traffic Monitor**：运行时追踪 EP 的 traffic demands，利用 MoE 训练框架已有的 token dispatch 信息收集机制（如 Megatron-LM token_dispatcher.py）预测后续 all-to-all 通信模式。
    2. **Topology Controller**：去中心化 greedy 算法（Algorithm 1）——根据预测的 traffic demands，迭代识别 bottleneck links（最长完成时间的 GPU 对），优先为这些 pairs 分配直接 OCS 电路，生成 NUMA-optimized NIC 映射。
    3. **Topology-Aware EP Routing**（5 步流程）：(1) 各 GPU 查拓扑确定 intra-server delegation GPU → (2) intra-host gather via NVSwitch（数据汇聚到 delegation GPU）→ (3) inter-host all-to-all via OCS（优先）+ EPS（fallback）→ (4) intra-host all-to-all among local experts via NVSwitch → (5) delegation GPU scatter 数据到最终目标。步骤 (3) 和 (4) 通过 overlap 减少完成时间。
    4. **DP Hierarchical All-Reduce**：intra-host reduction via NVSwitch → inter-host ring all-reduce via EPS → intra-host broadcast via NVSwitch。多 EPS NIC 时使用 multi-ring all-reduce。
    5. **Traffic Demand Prediction (MixNet-Copilot, §B.1)**：使用 Sequential Least Squares Programming (SLSQP via scipy.optimize) 估计 conditional probability matrix P（前一层 expert load → 当前层 expert load），基于最近 k 次迭代的加权平均。利用预测结果提前重配置 OCS 拓扑以处理 FP 第一个 all-to-all。
  - 实验比较：
    - 原型：MixNet custom runtime + OCS reconfiguration vs 4×100G EPS baseline（NCCL all-to-all + all-reduce），训练 3 个 MoE 模型。
    - 仿真：MixNet topology-aware EP routing vs Fat-tree / Rail-optimized / TopoOpt 的 collective communication（NCCL-based）。

- 后端平台是什么，配置是什么。
  - 原型：4 台 server，每台 8×NVIDIA A100 GPU + 4×Mellanox ConnectX-6 100G NICs。3 NIC per server 接 Polatis OCS（RoCEv2），1 NIC 接 SN3700 Ethernet switch。每 server 内部 4×NVLink（相邻 GPU 对互联）。RDMA 通信使用 FuseLink raw ibverbs library。
  - 仿真：每 server 8 GPU（NVSwitch 900 GB/s）+ 8 NIC（带宽 B）。EPS fabric 用 fat-tree 拓扑。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 **NCCL**（NVIDIA Collective Communications Library）+ **Megatron-LM** 构建。修改内容：
    1. 实现自定义 collective communication runtime（C++ ~6K LoC），通过 RDMA raw ibverbs（FuseLink）进行高速数据传输。
    2. 将 MixNet runtime 移植到 Python 以集成 Megatron-LM，实现通信原语：`mixnet.all_to_all` 和 `mixnet.all_reduce`（类似 torch.dist 接口）。
    3. DP 和 PP 通信复用了 NCCL 的高性能 intra-host 和 inter-host all-reduce/point-to-point 通信。
  - 对比 baseline：4×100G EPS 配置（全部 NIC 经 Ethernet switch），使用标准 NCCL。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://mixnet-project.github.io/
  - 通信 runtime 全栈执行流程（以 EP all-to-all，4 servers × 8 GPUs, EP degree=16 为例）：
    ```
    // ===== Step 1: Topology Lookup =====
    // 每个 GPU 查重配置后的 OCS 拓扑，确定 intra-server delegation GPU
    // delegation_gpu[src_server][dst_server] = 与 dst_server 有直接 OCS 链路的本地 GPU
    // 优先选择 OCS 直接连接的 GPU 对，否则 fallback 到 EPS
    
    // ===== Step 2: Intra-host Gather (NVSwitch) =====
    // 每 server 内部：各 GPU 通过 NVSwitch gather 出站数据到 delegation GPU
    // 数据按目标 server 分组，发送到对应的 delegation GPU
    // NUMA balancing: topology controller 确保多 OCS NIC 分布在多个 NUMA node
    
    // ===== Step 3a: Inter-host All-to-All via OCS (RDMA) =====
    // delegation GPU 之间通过 OCS 直连电路执行 RDMA write
    // 使用 FuseLink raw ibverbs: ibv_post_send(qp, wr, &bad_wr)
    // OCS 电路提供专用高带宽路径，无 packet switching overhead
    
    // ===== Step 3b: Inter-host All-to-All via EPS (RDMA fallback) =====
    // 无 OCS 直连的 GPU 对通过 EPS fabric 传输
    // 使用标准 RDMA over Ethernet
    
    // ===== Step 4: Intra-host All-to-All (NVSwitch) =====
    // 每 server 内部：接收到的 remote expert data 通过 NVSwitch
    // all-to-all 分发给本地 expert 对应的 GPU
    // Step 3 和 Step 4 通过 CUDA stream overlap 并行执行
    
    // ===== Step 5: Intra-host Scatter (NVSwitch) =====
    // delegation GPU 将收到的 all-to-all 数据 scatter 到最终目标 GPU
    
    // ===== DP Hierarchical All-Reduce 流程 =====
    // Stage 1 (intra-host, NVSwitch): GPU 内部 reduce → gateway DP GPU
    // Stage 2 (inter-host, EPS): ring all-reduce among gateway GPUs
    //   - 多 EPS NIC 时使用 multi-ring all-reduce 充分用满带宽
    // Stage 3 (intra-host, NVSwitch): gateway GPU broadcast → 所有 GPU
    
    // ===== OCS Topology Reconfiguration Algorithm (Algorithm 1) =====
    // 输入: E (expert all-to-all communication demands), α (optical degree),
    //       N (#servers), V (server node set)
    // 输出: S (NIC-level mapping in OCS)
    //
    // 1. D = CALCULATE_SERVER_DEMAND(E)
    //    // 将 expert-level traffic matrix 映射到 server-level demand
    //    // TX+RX 合并为 upper triangular matrix
    // 2. while True:
    //      (i,j) = FINDBOTTLENECKLINK(T, C, V)
    //      // 找完成时间最长的 server pair（T[i][j] = D[i][j] / C[i][j]）
    //      if avail_ocs[i] > 0 and avail_ocs[j] > 0:
    //        C[i][j]++, C[j][i]++  // 分配 OCS 链路
    //        avail_ocs[i]--, avail_ocs[j]--
    //      else: break  // 所有 NIC 端口已分配完
    //      T[i][j] = D[i][j] / C[i][j]  // 更新完成时间矩阵
    // 3. S = GetNICMapping(C)
    //    // 将链路分配矩阵 C 转化为 TX/RX NIC 配对
    // 4. S = permuteLinks(S)
    //    // NUMA-aware permutation 避免 intra-host congestion
    // 5. RECONFIGUREOCS(S)
    //    // 向 OCS 发送 TL1 commands 执行物理重配置
    ```

  - 关键性能结果（原型）：
    - MixNet（12 optical + 4 electrical ports）达到与 4×100G EPS baseline（16 electrical ports）相当的训练性能，证明用更少的总端口数（低成本）即可匹配全电气方案。
    - Reconfiguration turnaround time: 平均 41-47ms（1-16 pairs），99th percentile <70ms，NIC activation time 平均 ~5.67s（受限于 commodity transceiver 未针对快速重配置优化——论文排除此时间）。

## MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现两大类自定义 CUDA kernel：(1) **高效 scatter/gather 算子**——基于 token routing 结果预计算 input→output 行映射表，直接通过 CUDA 执行数据传输，替代 Megatron-LM 使用的 torch.scatter_add / torch.gather；(2) **Intra-operator 通信-计算融合 kernel**——将通信操作以 tile 粒度嵌入 GEMM/GroupedGEMM 计算 kernel 中，使用 device memory barrier 实现 tile 级同步（无需 host CPU 干预）。
  - 四类 fused kernel：
    a. **A2A+GEMM**（SP attention Output Projection）：all-to-all 通信与 GEMM 同时启动，remote data tile 到达后通过 signal 通知 GEMM kernel 继续计算该 tile。使用 GPU copy engine 处理通信，SM 全部用于计算。
    b. **GEMM+A2A**（SP attention QKV Projection）：all-to-all 嵌入 GEMM kernel，每个 GEMM tile 计算完成后直接发起 remote data transfer 写入目标 rank。
    c. **AG+Scatter+GroupedGEMM**（FFN token dispatch）：对 token 按 routed expert index → source rank index 排序，使每个 computation tile 依赖尽可能少的 source rank。将 local scatter 融合进 kernel（按 index mapping 选择输入行），GroupedGEMM 按 tile 分块执行。
    d. **GroupedGEMM+Gather+RS**（FFN token combine）：类似 (c) 的逆过程。
  - 实验比较：ablation study 中逐步开启 intra-operator overlap（Table 5，+6% 吞吐）。六种模型下 fused vs non-fused 通信+计算总时间对比（Figure 16，1.2-4.7× 减少）。

- 后端平台是什么，配置是什么。
  - NVIDIA H800 SXM GPU（989 TFLOPS compute, 80 GB HBM, 3.4 TB/s memory BW, 400 GB/s NVLink），intra-node 8 GPU via NVLink。
  - 也测试 A100（312 TFLOPS, NVLink 600 GB/s）和 H20（148 TFLOPS, NVLink 900 GB/s）。
  - CUDA 编程模型，使用 device memory barrier + GPU copy engine。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 Megatron-LM（commit f1f03922）构建。MegaScale-MoE 将 MoE layer 的 attention 和 FFN 模块分解为独立的 GPU kernel 算子（而非 Megatron-LM 中依赖 torch.autograd 的 monolithic 执行），实现细粒度调度。
  - 修改包括：(1) 自定义 CUDA scatter/gather 替代 PyTorch 内置算子；(2) 实现四种 fused communication-computation kernel；(3) SM allocation tuning：为 A2A+GEMM 模式中的通信分配少量 SM（tuned to match computation latency via swizzling 重排 tile 顺序避免 NVLink contention）。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未提供开源仓库链接；基于 Megatron-LM 开源（github.com/NVIDIA/Megatron-LM）。
  - Intra-operator overlap kernel 执行原理（以 AG+Scatter+GroupedGEMM 为例）：
    1. 输入：token hidden states [b*s, h] + routing decisions (每个 token 的 expert assignment)
    2. Token 排序：按 routed expert index 排序所有 token（使连续 token 属同一 expert），再按 source rank index 二次排序（使同一 expert 内同源 rank token 连续）
    3. 切分为 computation tiles：排序后的 token 序列按固定 tile size 切片
    4. 并行执行：每个 tile 启动 GroupedGEMM 前检查所需 source rank 的数据是否已到达（device memory barrier polling）。仅依赖本地/已到达 rank 的 tile 可立即开始计算
    5. Scatter 内联：通过预计算的 row index mapping 直接选取输入矩阵对应行，无额外 kernel launch
    6. 输出：Expert FFN 输出按 tile 组装回 token 维度，进入 gather + reduce-scatter 阶段
  - SM allocation 策略：通信分配少量 SM（数量 tuned 使 comm≈comp latency），其余 SM 用于 GEMM。Swizzling 重排 tile 顺序使各 rank 的 remote data 到达节奏与 computation tile 消费节奏对齐，降低 NVLink contention。

## MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  MegaScale-Infer 实现两类 kernel 级优化：
  1. **High-Performance M2N Communication Library**：针对 MoE 推理中 attention 模块到 expert 模块的 M（发送者数量）×N（接收者数量）通信模式，自研 M2N 通信库替代 NCCL 的 peer-to-peer primitives。核心优化包括：
     - **消除 GPU-to-CPU 数据拷贝**：使用 GPUDirect 技术，数据直接从 GPU 内存通过 RDMA 发送，无需经过 CPU proxy。
     - **消除 Group Initialization Overhead**：NCCL 的 group operations 按 batch of 8 处理（随 N 增大性能恶化），M2N 使用独立的 point-to-point RDMA write with immediate，无 group batching 限制。
     - **消除 GPU Synchronization**：M2N Sender 使用 CUDA events 等待 kernel 完成 → cuStreamWaitValue32 阻塞 stream → Core Sender RDMA 传输 → poll completion queue → 通过共享内存 flag 唤醒 stream。避免了 NCCL 中复杂的 GPU 同步操作和 device memory accesses（这些是性能不稳定来源）。
     - **M2N Receiver**：使用 GDRCopy 进行 GPU 内存 flush 操作确保数据一致性，无需 GPU-to-GPU 拷贝。
     - **Traffic-oriented optimizations**：(a) 高优先级 ACK packets：ACK 与数据包隔离到高优先级队列，避免双向通信中的 ACK 排队延迟；(b) 拥塞控制微调：针对不均衡通信场景微调 congestion control 算法，减少 rate-limiting 效应。
     - 与 DeepEP 对比：M2N 使用 CPU 进行 inter-node 通信（单线程 CPU 足以在几百 KB 数据量下饱和带宽），DeepEP 使用 GPU-to-GPU 通信（消耗 GPU SM 资源但并行处理能力更强）。M2N 不需要 PTX 级别的 low-level 优化。
  2. **Fused Kernels**：
     - **TP Communication-Computation Fusion**：使用 Flux 将 tensor parallelism 的 all-gather/reduce-scatter 通信与相邻 GEMM 操作融合为单 kernel，消除 TP 通信开销。
     - **Sequential Memory-Intensive Operator Fusion**：将 gating network（router 计算）、top-k expert 选择、per-expert token count 计算、normalized token weights 计算、token scatter 等多个连续的小型 memory-intensive 操作融合为一个 kernel，减少 kernel launch 和 memory access。
  实验比较了 M2N 通信库 vs NCCL（median/P99 latency, throughput）在不同 data size（1KB–2048KB）和不同 M/N 配置下，以及与 perftest（CPU baseline）的对比。同时通过 ablation study 测量了 M2N 优化带来的端到端 throughput 增益。

- 后端平台是什么，配置是什么。
  同构集群：8 节点，每节点 8×NVIDIA 80GB Ampere GPU，128 CPUs，2 TB host memory，8×200 Gbps InfiniBand NICs（每 GPU 一张 NIC），节点内 400 GB/s NVLink。
  异构集群：NVIDIA H20（96 GB, 4096 GB/s bandwidth, 900 GB/s NVLink, 4×400 Gbps NICs）+ NVIDIA L40S（48 GB, 864 GB/s bandwidth, PCIe intra-node, 2×400 Gbps NICs）。
  bfloat16 所有计算。

- 评估性能的软件/脚本是什么。修改了什么。
  - M2N 通信库：实现为 PyTorch C++/CUDA extension，约 4,900 行 C/C++ + 5,000 行 Python。核心依赖：GPUDirect（GPU 内存直接 RDMA）、GDRCopy（GPU memory flush via CPU）、RDMA write with immediate、CUDA driver API (cuStreamWaitValue32)、CUDA runtime API (cudaEventQuery)、NVIDIA RDMA 完成队列。
  - 对比基线：NCCL peer-to-peer primitives（group operations + send/recv）和 perftest（CPU-side networking microbenchmark，作为延迟下界）。
  - 测试方法：micro-benchmark 测量不同 data size（1KB–2048KB）和不同 M×N 配置（8×8, 16×16, 32×32）下的 latency 分布（median, P99）和 throughput。
  - 与 NCCL 的关键区别：M2N 避免 NCCL 的 GPU-to-CPU 中间拷贝（即使 user buffer registration 也无法完全消除）、batch size=8 的 group operation 限制、general collective operation setup 开销、GPU 同步/memory access 导致的不稳定性。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  论文未声明开源。M2N 通信库是 ByteDance 内部实现的生产级组件。

  **M2N 通信全流程（从 GPU kernel 输出到 Remote GPU 显存可用）**：
  ```
  // ===== M2N Sender 端（Attention GPU → Expert GPU） =====
  // 步骤 1: 等待前序 kernel 完成
  cudaEventRecord(event_start, stream);          // 记录 kernel 完成事件
  cudaEventSynchronize(event_start);             // 确保 tensor 数据就绪
  
  // 步骤 2: 阻塞 CUDA stream（防止后续 kernel 覆盖正在传输的数据）
  cuStreamWaitValue32(stream, &flag,             // 阻塞直到 flag 被远端更新
                      CU_STREAM_WAIT_VALUE_EQ, 0);
  
  // 步骤 3: RDMA 数据传输（CPU Core Sender 执行）
  // 使用 RDMA write with immediate：
  //   - 直接将 GPU 显存中的数据写入远端 Expert GPU 的注册内存
  //   - "immediate" 值携带 metadata（如 token count、expert ID）
  //   - 无需 CPU proxy buffer 拷贝（GPUDirect RDMA）
  ibv_post_send(qp, wr, &bad_wr);                // 提交 RDMA write + immediate
  // 每个 QP 以 doorbell ring 方式并行发送
  // 单线程 CPU 即可饱和带宽（每连接几百 KB 数据量下）
  
  // 步骤 4: 轮询完成队列确认传输
  while (ibv_poll_cq(cq, 1, &wc) == 0) { /* spin */ }
  // 确认 RDMA write 完成（数据已写入远端 GPU 显存）
  
  // 步骤 5: 唤醒 CUDA stream
  flag = 1;                                      // 共享内存 flag 更新
  // cuStreamWaitValue32 检测到 flag 变化 → 唤醒 stream
  // 后续 kernel 可以安全地复用 registered tensor 内存
  
  // ===== M2N Receiver 端（Expert GPU 侧） =====
  // 步骤 1: 等待接收 buffer 可用
  cudaEventRecord(event, stream);
  cudaEventSynchronize(event);
  
  // 步骤 2: 阻塞 CUDA stream
  cuStreamWaitValue32(stream, &recv_flag, CU_STREAM_WAIT_VALUE_EQ, 0);
  
  // 步骤 3: 轮询完成队列，确认数据已到达
  while (ibv_poll_cq(recv_cq, 1, &wc) == 0) { /* spin */ }
  
  // 步骤 4: GPU 缓存一致性 flush（关键！）
  // 因为数据是通过 RDMA 直接写入 GPU 显存的
  // GPU 的 L2 cache 可能持有该内存区域的 stale data
  // 使用 GDRCopy 执行 flush 操作确保后续 kernel 读到最新数据
  gdr_copy_to_mapping(...);                      // GDRCopy flush via CPU BAR mapping
  // 等效于 NCCL 的 GDR flush operation
  // 参考：https://github.com/NVIDIA/nccl/issues/683
  
  // 步骤 5: 唤醒 CUDA stream
  recv_flag = 1;
  // Expert FFN kernel 开始执行，读取接收到的 token embeddings

  // ===== 性能测量原理 =====
  // Latency: 从 Sender 端 event_start 到 Receiver 端 recv_flag 被设置的 wall-clock time
  // Throughput = total_bytes_transferred / T_c（含所有 QP 并行传输）
  // T_c = max(send_time, recv_time) per Equation 6
  // 
  // Key optimization comparison with NCCL:
  // NCCL 额外开销来源：
  //   1. GPU→CPU proxy buffer copy（即使 user buffer registration 也无法完全消除）
  //   2. Group operation 的 batch-of-8 处理
  //   3. Group init/launch/topology verification overhead
  //   4. GPU 同步操作（cudaDeviceSynchronize）引发的不稳定性
  // M2N 消除所有这些开销，仅保留：
  //   1. cudaEventSynchronize（kernel 完成等待）
  //   2. RDMA write + poll CQ
  //   3. GDRCopy flush
  ```

  关键性能结果：
  - 256KB data size（最常见 MoE serving 场景）：68.2% median latency 降低、92.9% P99 latency 降低、4.2× throughput 提升 vs NCCL。
  - P99 latency 稳定性：随 N 增大 NCCL 的 tail latency 显著上升（32 receivers 时 P99 > 1000μs），M2N 保持稳定（<100μs），因消除 GPU synchronization 消除了主要的不稳定性来源。
  - Throughput scalability：跨不同 M/N 配置（8×8 到 32×32）M2N 均保持 3.3×–5.8× throughput 提升。
  - Ablation study：disaggregated architecture alone 提升 4.66× vs colocated baseline；加上 M2N 优化额外提升 1.53×（因通信可被 pipeline 完全覆盖）。

## MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现自定义 **block-sparse GPU kernels**（SDD, DSD, DDS）用于高效执行 MoE 的动态、负载不均衡的 expert 计算。基于 CUTLASS 2.5 扩展，核心 kernel 设计包括：
    - **Hybrid Blocked-CSR-COO 编码（§5.1.3）**：以 BCSR (Blocked Compressed Sparse Row) 为主要稀疏矩阵格式，同时额外物化每个 non-zero block 的行索引，使得 SDD 操作的并行化无需搜索 row offsets（直接 O(1) 查找 block 坐标）。存储开销可忽略（每个 128×128 block = 16384 个非零值仅需 1 个索引）。
    - **Transpose Indices（§5.1.4）**：为高效支持 block-sparse 矩阵的转置访问（向后传播需要），构造转置后的元数据（等效于 BCSC 编码），包含 column offsets 和转置顺序的 non-zero block 偏移索引数组。无需显式转置非零值（避免数据复制），通过一层间接索引实现转置迭代。类似数据库的 secondary index。
    - **Block Size 选择（§5.1.2）**：基于 CUTLASS 的 tile dimension benchmark，选择 128×128 block size，因为它在 A100 上对所有 tile 配置表现最优（图 5），与 cuBLAS 为 dense Transformer 模型选择的配置一致。
    - **Permutation kernel 融合（§5.2）**：将 token padding（每个 expert 的 token 数填充到 128 的倍数）融合进自定义 permutation kernel。同时在前向开始时构造 block-sparse 矩阵元数据和转置元数据，摊销到后续多次矩阵乘法。
  - 实验比较：
    - Block-sparse kernels vs cuBLAS batched GEMM（§6.3, Figure 9）：18 种 problem configuration（MoE-XS/Small/Medium × 6 operations），平均达到 cuBLAS 98.6% 吞吐量（标准差 4%，最大 104%，最小 91%）。
    - Block-sparse kernels vs Triton Blocksparse（Appendix C, Figure 13）：平均 9× 吞吐量优势（含 Triton Blocksparse 的 sparse matrix preprocessing 开销）。
    - End-to-end：MegaBlocks dMoE vs Tutel dMoE（含动态 capacity factor，§6.1）：1.38×–4.35× 训练加速；MegaBlocks dMoE vs Tutel token-dropping MoE（§6.2）：1.18×–1.38× 加速。

- 后端平台是什么，配置是什么。
  - GPU：NVIDIA A100 SXM4 80GB。单卡 micro-benchmark 和 8-GPU 端到端训练实验。
  - CUDA 11.5 + CUTLASS 2.5。A100 Tensor Cores 用于 FP16 + FP32 accumulation。
  - 8-way expert model parallelism（MoE 层）+ data parallelism（其他层），通过 gradient accumulation 实现 batch size 512 sequences。使用最大不 OOM 的 micro_batch_size（见表 3）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 CUTLASS 2.5 扩展实现 block-sparse GEMM kernels (SDD, DSD, DDS)，支持所有 transposed/non-transposed 输入组合。
  - 修改了 Megatron-LM 的 MoE layer：将 batched matrix multiplication 替换为 block-sparse matrix multiplication。
  - Custom CUDA kernel 用于构造 sparse matrix topology (make_topology) 和 permutation (padded_gather/padded_scatter)。
  - Micro-benchmark：对 18 种 problem configuration（3 模型 × 6 operations: SDD, DSD, DDS, SDD^T, DS^T D, DSD^T），每种执行 100 次取平均吞吐量。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/stanford-futuredata/megablocks，Apache-2.0
  - 评估原理（kernel 输入到性能输出全过程）：

```
// ===== SDD Kernel (Sparse = Dense × Dense, Figure 11) =====
// 输入: Matrix a (m × k dense), Matrix b (k × n dense),
//       SparseMatrix c (m × n, block-sparse output)
// 输出: SparseMatrix c（写入每个 non-zero block 的计算结果）

__global__ void sdd(Matrix a, Matrix b, SparseMatrix c) {
    // (1) Load block coordinates from hybrid Blocked-CSR-COO
    //     每个 threadblock 处理一个 non-zero block
    int row    = c.row_idxs[blockIdx.x];    // 来自 BCOO 行索引
    int column = c.column_idxs[blockIdx.x]; // 来自 BCSR 列索引

    // (2) Zero accumulator (128×128 tile)
    Tile<128, 128> tile_c(0);

    // (3) Main loop: iterate over k dimension in 128-step
    for (int i = 0; i < k; i += 128) {
        Tile<128, 128> tile_a = LoadTile(a, row, i);
        Tile<128, 128> tile_b = LoadTile(b, i, column);
        tile_c += tile_a * tile_b;  // Tensor Core MMA
    }

    // (4) Store result to sparse output
    StoreTile(tile_c, c);
}

// ===== DSD Kernel (Dense = Sparse × Dense, Figure 12) =====
// 输入: SparseMatrix a (m × k, BCSR), Matrix b (k × n dense)
// 输出: Matrix c (m × n dense)

__global__ void dsd(SparseMatrix a, Matrix b, Matrix c) {
    // (1) Each threadblock computes one 128×128 tile of dense output
    int row    = blockIdx.x;  // output row tile index
    int column = blockIdx.y;  // output column tile index

    // (2) Load BCSR row offset and compute number of non-zeros in this row
    int offset_a = a.row_offsets[row];
    int nnz      = a.row_offsets[row + 1] - offset_a;

    // (3) Zero accumulator
    Tile<128, 128> tile_c(0);

    // (4) Main loop: iterate over non-zero blocks in row
    for (int i = 0; i < nnz; i++) {
        Tile<128, 128> tile_a = LoadTile(a, offset_a, i);
        // (5) Load column index of this non-zero block from a
        //     to determine which row to load from b
        int row_b = a.column_idxs[offset_a + i];
        Tile<128, 128> tile_b = LoadTile(b, row_b, column);
        tile_c += tile_a * tile_b;
    }

    // (6) Store result
    StoreTile(tile_c, c);
}

// ===== Transpose Indices for DSD^T / DDS^T =====
// 当 sparse operand 需要转置时（如 DS^T D）：
// 使用 transpose indices（图 6 中 Transpose Indices 数组）
// 它存储按 transposed 顺序排列的 non-zero block 偏移
// 在 DSD kernel 的 main loop 中：
//    loaded_offset = transpose_indices[offset_a + i]  // 间接索引
//    Tile<128,128> tile_a = LoadTile(a, loaded_offset, ...);
// 避免显式转置整个 sparse matrix（节省内存和时间）

// ===== 从输入到性能输出的全流程 =====
// 1. 输入: token tensor x (num_tokens × hidden_size, FP16)
// 2. Router: indices, weights = router(x)
//    indices: (num_tokens,), 每个 token 分配的 expert ID
// 3. make_topology(indices) → 构造图 6 的稀疏矩阵元数据:
//    - Blocked-CSR-COO: row_offsets, row_idxs, column_idxs
//    - Transpose indices: 转置访问的间接索引
// 4. padded_gather: 按 expert 分组 tokens + padding to 128 倍数
// 5. sdd(x, w1, topology): 每个 non-zero block 启动 1 个 threadblock
//    - 128×128 tile, FP16 Tensor Core MMA (m=128, n=128, k=128)
//    - 输出: block-sparse intermediate (每 expert batch 的结果)
// 6. dsd(intermediate, w2): 每个 dense output tile 启动 1 个 threadblock
//    - 迭代对应 row 的 non-zero blocks
//    - 输出: dense tensor
// 7. padded_scatter + weight scaling → MoE layer output
// 8. 性能度量: TFLOPs = (total math ops) / (elapsed time)
//    vs cuBLAS 的 relative throughput = TFLOPs_kernel / TFLOPs_cuBLAS
```
  - 关键性能结果：
    - 128×128 block size 在 A100 上实测优于其他配置（图 5），对应 128×128 CUTLASS tile dimensions。
    - Metadata 内存开销 <0.1%（得益于大 block size），使得 hybrid CSR-COO 和 transpose indices 的额外存储可忽略。
    - SDD 操作通过 BCOO 行索引避免 row offset 搜索的开销，对高 expert count（稀疏度 >90%）至关重要。
    - DSD^T 和 DDS^T 中 transpose indices 引入的间接访问降低了空间局部性，导致 <10% 吞吐量损失，但由于这两类操作仅占端到端运行时间很小比例，影响极小。
    - Micro batch size 影响：Tutel 因 padding 导致内存占用大，micro_batch_size 被迫缩小 2×–8×（表 3），降低了 GPU 利用率和硬件效率。

## MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - MPipeMoE 在 MoE 训练中对三种 CUDA stream（computation / communication / memory copy）进行运行时调度，核心调度机制包括：
    - **Micro-batch pipeline 调度**：将 MoE 层的 All-to-All dispatch (S)、Expert FFN (C)、All-to-All collect (R) 三种 kernel 以 pipeline 方式跨 CUDA stream 调度。S 和 R 交替执行以增强 memory access locality，C 独立在 computation stream 上运行。Pipeline granularity n 决定 micro-batch 的大小，影响 kernel launch 频率和 GPU 利用率。
    - **Interference profiling（Figure 3）**：通过 micro-benchmark 量化三种操作并行执行时的相互 slowdown 因子：μ（通信受其他 stream 干扰的 slowdown）、σ（计算的 slowdown，实测 ≈1，即计算几乎不受影响）、η（memory copy 的 slowdown）。实验发现通信与 memory copy 并行时因带宽竞争导致显著 slowdown，而通信与计算重叠可行（μ_comm > 0.5, σ_comp > 0.5）。
    - **Pipeline paradigm 性能建模**：将 pipeline 划分为 5 个阶段（P0-P4），每个阶段由瓶颈 stream 的执行时间决定。执行时间 C = (1/W_comp) * max(q1, q2*α/μ, q3*β/η)，其中 α=W_comp/W_comm, β=W_comp/W_mem，Q=[q1,q2,q3] 为各类型的操作量。
    - **4 种 memory reuse scheduling 策略的 kernel 调度差异（Table II, Figure 7）**：
      - S1: forward 3 条 stream（comp + comm + mem D2H），backward 3 条 stream（comp + comm + mem H2D）
      - S2: forward 3 条 stream，backward 3 条 stream（额外通信恢复 T_DI）
      - S3: forward 3 条 stream，backward 3 条 stream（额外重计算恢复 T_M）
      - S4: forward 2 条 stream（comp + comm，无 memory copy），backward 3 条 stream（额外通信 + 重计算）
    - **Adaptive strategy selection**：基于 Eq 10 性能模型，在运行时根据 N（GPU 数量）和 B（batch size）选择开销最小的 (n, S) 组合。
  - 实验比较：
    - 通信效率 micro-benchmark：对比 FasterMoE（按 node 切分→P2P）vs MPipeMoE（按 batch 切分→保留 All-to-All）在不同 pipeline granularity n 下的 dispatch/recovery 时延。
    - Pipeline granularity 敏感度分析：不同 n（1/2/4/8/16）在不同 B（2k-32k）下的训练时间（Figure 12）。
    - 内存复用策略 S1-S4 在不同 (N, B) 组合下的 overhead 对比（Figure 13）。
    - 性能分解（Figure 11）：在 memory-time 坐标系下，PipeMoE(n=4) / PipeMoE(adaptive n) / MPipeMoE 的 trade-off。

- 后端平台是什么，配置是什么。
  - 8 台 NVIDIA DGX A100 服务器，每节点 8×A100 SXM 40GB GPU（共 64 GPU），200 Gbps HDR InfiniBand，节点内 NVLink 3.0 + NVSwitch。
  - CUDA 11.1、NCCL（All-to-All collective operator）、PyTorch 1.9.0。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 PyTorch 1.9.0 + CUDA 11.1 实现自定义 MoE 训练库。修改了 MoE layer 的 forward/backward，引入多 CUDA stream 管理（computation stream、communication stream、memory copy stream）。
  - 自定义 micro-benchmark 用于测量 W_comp、W_comm、W_mem 的 piecewise 速度（区分小/大 volume 的不同硬件利用率）以及 μ、σ、η 干扰因子（Figure 3, Figure 9）。
  - Gating network 默认 top-1 routing（k=1），使用 NCCL All-to-All collective operator 进行 token dispatching/collecting。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源：https://github.com/whuzhangzheng/MPipeMoE
  - 评估原理（kernel 输入到性能输出全过程）：

```
// ===== Interference Profiling Micro-benchmark (Section II-C) =====
// 目的：测量 W_comp, W_comm, W_mem 及 slowdown 因子 μ, σ, η
// 原理：在 3 条 CUDA stream 上分别运行 GeMM kernel / NCCL All-to-All /
//       cudaMemcpy D2H，测量单独执行和并行执行的 wall-clock time

输入: 模型配置 (M, H, B), GPU 拓扑
输出: W_comp(B), W_comm(B), W_mem(B) piecewise速度,
      μ(干扰源类型), σ(干扰源类型), η(干扰源类型)

stream_comp:  launch GeMM kernel (M×H × B tokens)
stream_comm:  launch NCCL All-to-All (B×M bytes)
stream_mem:   launch cudaMemcpy D2H (B×M bytes)

// 单独执行
W_comp = FLOPs / t_comp_alone
W_comm = Bytes / t_comm_alone
W_mem  = Bytes / t_mem_alone

// 并行执行（测量 slowdown）
// e.g., comp + comm 并行:
t_comp_parallel = 在 comp+comm 并行中测得 comp kernel 实际耗时
t_comm_parallel = 在 comp+comm 并行中测得 comm kernel 实际耗时
μ_comm = W_comm * t_comm_alone / (Bytes / t_comm_parallel)
        // μ_comm < 1 表示通信因并行而减速
σ_comp = 类似定义
η_all  = 三流并行时 memory copy 的 slowdown

// Piecewise 速度函数（Figure 9 profile 结果）：
// 小 volume: GPU 利用率低，throughput 随 volume 线性增长
// 大 volume: GPU 饱和，throughput 稳定在峰值
// V_threshold_comp 为 GeMM 饱和阈值，V_threshold_comm 为 All-to-All 饱和阈值

// ===== Pipeline 执行时间模型 (Eq 10) =====
// 以 strategy S4 为例（Q_fw=[2,2,0], Q_bw=[5,3,0]）
// Forward pass: 2 个 GeMM + 2 个 All-to-All，无 memory copy
// Backward pass: 5 个 GeMM + 3 个 All-to-All，无 memory copy
// μ_all = μ_comp（仅 comp+comm 并行场景，无 mem stream）

b = B/n  // micro-batch size
v0_comp = b * H * M     // Eq 7: GeMM FLOPs per micro-batch
v0_comm = b * M         // Eq 8: All-to-All bytes per micro-batch
v0_mem  = b * M         // Eq 9: D2H/H2D bytes per micro-batch

// Forward 阶段执行时间
T_comp_fw = q1_fw * v0_comp / (σ * W_comp(b))
T_comm_fw = q2_fw * v0_comm / (μ * W_comm(b))
T_mem_fw  = q3_fw * v0_mem  / (η * W_mem(b))
T_fw = max(T_comp_fw, T_comm_fw, T_mem_fw)

// Backward 同理
T_bw = max(T_comp_bw, T_comm_bw, T_mem_bw)

// 端到端时间（n 个 micro-batch pipeline）
T_total ≈ max(T_fw, T_bw) * n  // 瓶颈 stream 决定

// 选择 T_total 最小的 (n, S) 组合作为运行时最优配置

// ===== 实际调度 =====
// 在 Python API 层面通过设置 pipeline=True, memory_reuse=True 启用
import pmoe
moe_layer = pmoe.MoELayer(d_model=1024, pipeline=True, memory_reuse=True)
```

  - 关键 kernel 调度结果：
    - 计算几乎不受其他 stream 影响（σ≈1），可与通信安全重叠。
    - 通信受并行计算影响但 slowdown 可接受（μ_comm > 0.5），重叠可行。
    - 通信与 memory copy 并行时因 PCIe/NVLink 带宽竞争导致显著 slowdown（η_all 较小），S2 在 N 大时性能差即因此（Figure 13）。
    - N 小时（8 GPU）S1/S2 更优（I/O bound 场景可容忍额外 memory copy）；N 大时（64 GPU）S4 更优（避免 memory bandwidth 竞争，重计算开销可被通信瓶颈掩盖）。
    - B 变化对策略选择不敏感（Figure 13），但 n 需要随 B 自适应调整（Figure 12）。

## MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-Lightning 实现了一个基于 Intel MKL 的自定义 **CPU Grouped Query Attention (GQA) kernel** 和 **Paged Weight Transfer** 机制：
    1. **CPU GQA Kernel**：在 decode 阶段，每层 transformer 的 attention softmax + weighted sum 在 CPU 上执行（而非 GPU），避免将 KV cache 从 CPU 传输到 GPU 占用 PCIe 带宽。Kernel 利用 Intel MKL batch GEMM 进行 query·key 和 attention-weights·value 矩阵乘法。GQA 场景下多个 query head 共享 KV head，通过 MKL 批量矩阵运算高效实现。
    2. **Paged Weight Transfer**：每层 MoE FFN expert weights 存入 CPU memory，分 n 页（n = 微批次数）。传输采用两阶段流水线：CPU memory → pinned memory (memcpy) → GPU memory (cudaMemcpyAsync)，两阶段在连续 pages 间重叠。GPU kernel 通过 page table 查找对应 weight pages 的 GPU 地址。
  - 实验比较：(1) 单层 CPU attention kernel latency vs KV cache transfer latency (CPU→GPU) vs MoE FFN GPU kernel latency 在不同微批次大小 μ 和 context length 下的对比 (Fig. 9)；(2) CPU attention kernel 在所有测试配置下比 KV cache transfer 快 3-4×，接近 CPU memory bandwidth 与 PCIe bandwidth 的比值；(3) MoE FFN kernel latency 在不同 μ 下基本不变（memory-bound）。

- 后端平台是什么，配置是什么。
  - CPU: Intel Xeon @ 2.20GHz (S2) / 2.30GHz (S1/S6-S9), 24-core/32-core。CPU attention kernel 利用 Intel MKL (oneAPI Math Kernel Library) 加速矩阵运算。
  - GPU: NVIDIA T4 (16GB HBM, ~65T FLOPS FP16) / L4 (24GB) / Multi-T4，通过 PCIe Gen3/4 连接。
  - Memory hierarchy: GPU HBM (16-64GB) → PCIe (~16-32 GB/s) → CPU pinned memory → CPU DRAM (192-416GB)。
  - Ablation (Fig. 9) 使用 L4 GPU + 24-core Intel Xeon @ 2.20GHz 测量单层 kernel latency。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估基于 MoE-Lightning 自建系统（Python + C++），CPU GQA kernel 和 paged weight transfer 为自研 C++ 模块，编译为 PyTorch extension。
  - 修改/创新：
    - 替代 GPU attention + KV cache transfer 路径：传统方法（如 FlexGen $S_4$）将 KV cache 从 CPU cudaMemcpy 到 GPU 再做 Flash Attention → MoE-Lightning 改为直接在 CPU 上执行 MKL GQA kernel，仅将 hidden states (shape: [μ, d]，远小于 KV cache [μ, s, d]) 通过 H2D 传输回 GPU。
    - Paged weight transfer 替代整层一次性传输：传统方法整层所有 experts weights 一次性 H2D → MoE-Lightning 分 n 页交错传输，避免阻塞后续微批次的 hidden states H2D。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：https://github.com/caoshiyi/artifacts/tree/asplos25（ASPLOS 2025 artifact）
  - **CPU GQA Kernel 评估（Fig. 9 原理）**：
    1. **Kernel 输入**：
       - Query vectors: shape [μ, n_q, d]，μ 为微批次大小，n_q=32 (Mixtral 8x7B)，d=128 (per-head dim)。
       - KV cache blocks: 存储在 CPU pinned memory 中的历史 key/value vectors，shape [s, n_kv, d]，s 为 context length，n_kv=8 (GQA 分组)。
       - GQA group size = n_q / n_kv = 4。
    2. **Kernel 计算流程**：
       - Step 1: 对每个 query head，使用 Intel MKL SGEMM 计算 query·key^T → attention scores [μ, n_q, s]。
       - Step 2: Softmax over sequence dimension（向量化实现）。
       - Step 3: 使用 MKL SGEMM 计算 attention weights × value → attended outputs [μ, n_q, d]。
       - GQA reuse: n_q=32 query heads 共享 n_kv=8 KV heads，实际计算量 = n_kv × attention per KV group。
    3. **Kernel 输出**：attended hidden states [μ, n_q×d] = [μ, 4096]，作为 PostAttn (O projection) 的输入通过 H2D 传输回 GPU。
    4. **性能指标**：kernel latency (ms/层)，在 μ ∈ {32, 64, 128, 256}、context length ∈ {128, 256, 512, 1024, 2048} 下测量。
    5. **关键结论**：CPU attention 比 KV cache H2D 快 3-4×，接近 CPU BW (~200 GB/s) / PCIe BW (~50 GB/s) ≈ 4× 的理论比值。随着 μ 和 context length 增大，CPU attention 可能成为瓶颈。
  - **Paged Weight Transfer 原理**：
    1. **输入**：存储在 CPU DRAM 中的 expert FFN weights [n_e=8, h1×h2×2 = 4096×14336×2]。
    2. **分页**：weights 被分为 n_pages = n_ub (微批次数) 页，每页大小 = total_expert_weights / n_pages。
    3. **两阶段流水线传输**：Page k 在 CPU→pinned (memcpy) 传输时，Page k-1 同时在 Pinned→GPU (cudaMemcpyAsync)。GPU PostAttn kernel 通过 page table 查表访问对应 pages。
    4. **性能输出**：paged 传输消除了整层一次性传输导致的 hidden states H2D 阻塞，使 GPU 利用率更高。

## MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoE-SpeQ 实现了 **fuseMoE CUDA kernel**——一个单次发射的融合 kernel，用于加速量化 MoE draft 阶段的细粒度 expert 计算。核心理念是将多个小 GEMM kernel launch（每个 expert 独立的 W_gate, W_up, W_down）融合为一个 monolithic kernel，降低 kernel launch overhead 并提高 GPU 硬件利用率。
    - 背景动机：在细粒度 MoE 模型（如 Qwen2-MoE，K=1408, N=2048）上，Marlin 量化 GEMM 后端性能甚至低于 PyTorch FP16 实现——因为每个 expert 矩阵太小，single kernel 无法占满 GPU SM，大量时间消耗在 kernel launch overhead 上。
    - 融合策略：将 L 层中所有需要计算的 expert 的 INT4 GEMM 操作合并为单一 kernel 调用，一次性完成 gate projection、up projection、SiLU activation、gate×up 逐元素乘、down projection。减少 kernel launch 次数（从 per-expert per-layer 变为 per-layer），提高 GPU occupancy。
  - 实验比较：（1）消融实验：DeepSeek-V2-Lite 上 Full (13.02 tok/s) vs without async prefetch (12.37 tok/s, 95%) vs without fused kernel (8.88 tok/s, 68.2%) vs both off (8.29 tok/s, 63.7%)；（2）图12 量化感知 fused expert kernel 对 throughput 和 latency 的影响。

- 后端平台是什么，配置是什么。
  - NVIDIA A100-40GB GPU（PCIe 4.0 x16，理论双向 32GB/s 聚合带宽）。
  - CUDA multi-stream（4 条独立 stream）+ CUDA events 管理同步互斥。
  - Intel Xeon Silver 4310 CPU（24-core），256GB RAM。

- 评估性能的软件/脚本是什么。修改了什么。
  - 基于 Hugging Face Transformers 框架，使用 GPTQ 量化库创建 INT4 draft 模型，使用 Marlin 后端做低比特推理。
  - 修改/新增内容：
    (1) **fuseMoE CUDA kernel**：自研的融合 CUDA kernel，将 per-expert 多次 GEMM kernel launch 合并为单次 launch。具体融合：loop over experts in layer → W_gate * h → W_up * h → SiLU(gate_out) ⊙ up_out → W_down * fused_out。所有中间结果保持 on-chip（SM shared memory + register），避免写回 global memory 再读回。
    (2) **Marlin 后端适配**：针对 MoE 细粒度场景（小 K/N 维度）优化 Marlin kernel 调用方式——通过 batch 多个 expert 的 GEMM 为单次更大矩阵运算，使 Marlin 达到接近 4× 加速期望。
    (3) **Computation Reordering**：verify 阶段前分析 ELB，重排 batch tokens 使同 expert 的计算连续执行，最大化 expert weights 的 L1/L2 cache 命中。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 开源情况：论文未提供开源代码仓库链接。
  - fuseMoE Kernel 评估原理与全流程（基于论文 §3.4.3 和 §4.5）：
    1. **Kernel 输入**：
       - Draft token hidden states: shape [k, d_model]（k 为 draft length，d_model=2048/4096）。
       - MoE layer 的 INT4 quantized weights：每组 expert 的 W_gate[inter_dim, d_model], W_up[inter_dim, d_model], W_down[d_model, inter_dim]，量化参数 scales [group_size] 和 zeros [group_size]（group_size=128）。
       - Router 选定 per-token per-layer 的 top-k expert IDs。
    2. **Kernel 计算流程**：
       - Step 1 - Expert Grouping: 遍历 k 个 token × L 层，按 expert ID 去重分组得到 S = union of selected experts across all tokens。
       - Step 2 - Batch GEMM: 将所有分配给同一 expert 的 tokens 的 hidden states 拼接为 mini-batch [n_tokens_for_expert, d_model] → Marlin INT4 GEMM 乘法 W_gate @ h^T → gate_out [n_tokens, inter_dim]。同理做 W_up @ h^T → up_out [n_tokens, inter_dim]。
       - Step 3 - Fused Activation: SiLU(gate_out) ⊙ up_out → fused [n_tokens, inter_dim]。此步与 Step 2 在同一 kernel 内完成，无中间 global memory write。
       - Step 4 - Down Projection: W_down @ fused^T → expert_output [n_tokens, d_model]。
       - Step 5 - Reduction: 对每个 token 的多个 expert 输出按 router softmax 权重加权求和。
    3. **Kernel 输出**：每层 draft hidden states [k, d_model]，传递至下一层或作为 output logits。
    4. **性能指标**：end-to-end tokens/sec (消融对比: 13.02 vs 8.88 vs 8.29 tok/s)。Prefill/decode latency comparison (图 11：Marlin vs other backends)。
    5. **优化原理**：融合前每个 expert 需要 3 次 GEMM kernel launch（W_gate, W_up, W_down），细粒度 MoE 中 inter_dim 仅 1408（Qwen2-MoE）或 6400（Phi-MoE），单个 GEMM 仅占用极少 SM，launch overhead >> compute。融合后所有 expert 的 GEMMs 在单次 kernel launch 中完成，batch size = Σ n_tokens_per_expert，矩阵维度增大 → GPU occupancy + SM utilization 提升 → 6.8% → 36.3% speedup (from ablation: 68.2% vs 100% normalized speed)。

## MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：MoEBlaze 设计并实现了两类自定义 CUDA kernel：
    1. **高效 Dispatch 数据结构构建 Kernel**（Section 4.2）：替代传统基于 multi-pass radix sort 的 token 调度方法，采用 3 步 atomic-free 的 GPU 并行构建流程：
       - **Step 1 - Build Dense Token-Expert Map**：分配 L×E 的 dense map，每个 warp 分配不相交的 token rows（i），将 top-k expert ID 写入 dense_token_map[i, e_{i,k}] = i。每个 (i,e) pair 最多写入一次（expert ID per token 唯一），guaranteed no intra-warp collision。
       - **Step 2 - Compute Expert Lengths**：CTA grid 按列（expert）映射，每个 CTA 专用于一个 expert e_i，通过 warp-level reduction 计数该列非零项，产出 expert_lengths 数组。
       - **Step 3 - Route Indices to Gates**：两阶段构建 location map：(i) tile-level scan——每 CTA 处理 contiguent tokens，shared memory 内做 exclusive scan（prefix sum）；(ii) 局部 scan 结果加 expert 的全局 expert_offsets，得到每个 entry 在 expert_token_indices 中的最终位置。最后通过并行 kernel 将 dense_token_map 中非零项直接写入对应位置，无原子操作。
    2. **Fused SwiGLU MoE Training Kernel**（Section 5）：将 SwiGLU FFN 的两个第一层 projection (W1, W2) 和激活 epilogue 融合为单 kernel：
       - **Forward**：一次性加载 input x，同时流式通过 W1 和 W2 的 GEMM，在 register/shared memory 中计算 SiLU(a) 并与 b 做 element-wise 乘法，仅写最终输出到 global memory，消除 a、b、σ(a)、SiLU(a)、final product 等中间结果的 global memory 写。
       - **Backward**：融合两个分支的 activation derivatives 计算，通过 tiled reduction 做 in-place 梯度聚合，消除临时 global buffers。采用 activation checkpoint 策略——forward 中不保存 SiLU 中间结果，backward 时 recompute SiLU（element-wise 操作，memory bandwidth bound，recompute 开销极低）。

  - 实验比较：(1) 训练速度（1.4×–6.2× speedup vs MegaBlocks，取决于配置和激活函数）；（2）激活内存消耗（最高 4× reduction）；（3）SiLU vs SwiGLU 两种激活函数的性能差异。

- 后端平台是什么，配置是什么。
  - 单张 NVIDIA H100 Tensor Core GPU（80GB HBM）。利用 H100 的硬件加速特性：warp-group matrix multiplication（WGMMA）、Tensor Memory Accelerator（TMA）。

- 评估性能的软件/脚本是什么。修改了什么。
  - 软件栈：PyTorch 2.0.1 + CUDA 12.1。Baseline：MegaBlocks（基于 block-sparse 操作的高性能 MoE 训练系统）。MoEBlaze 以自定义 CUDA kernel 替换了 MegaBlocks 的 token dispatch（sort-based bucketize）和 expert FFN 计算流程。
  - 内存测量：使用 PyTorch 的 saved tensor hooks 追踪训练中分配的所有中间激活张量大小。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - 论文未明确说明开源链接。
  - 评估原理：对比 MoEBlaze 与 MegaBlocks 在单层 MoE 的 forward+backward（Sparse-to-Sparse 阶段）的 wall-clock time 和 peak activation memory。
  - Kernel 输入到性能输出全过程（以 SwiGLU 为例）：
    1. **输入**：token 张量 x ∈ R^{L×d}（L = batch_size × seq_len），gate 权重 W_g，expert 权重 W1_i, W2_i, W3_i（每个 expert），所有数据在 HBM 中。
    2. **Gating**：W_g · x → softmax → TopK，生成 topk_experts（L×K 的 int32 索引）。
    3. **Dispatch 数据结构构建**（3-step kernel）：
       - Launch L×E grid → 填充 dense_token_map → 写入 HBM（L×E 个 int32，稀疏矩阵）
       - Launch E CTAs → warp-level reduction → expert_lengths[E] → expert_offsets[E+1]
       - Launch E CTAs → tile-level scan + global offset add → 并行写入 expert_token_indices[L×K]
       - 同时构建 token_expert_indices[L×K] 和 token_index_map[L×K]
    4. **Fused SwiGLU FFN Forward**（per expert）：
       - HBM → register：加载 x[token_ids]（on-the-fly gather via expert_token_indices）
       - Tensor Core (WGMMA)：x @ W1 → a，x @ W2 → b（两个 GEMM 在一个 kernel 内流式完成）
       - Register/shared memory：compute SiLU(a) = a·σ(a)，y_swi = SiLU(a) ⊙ b
       - Store to HBM：仅保存 a, b, y_swi 用于 backward（SiLU(a) 不保存，backward 时 recompute）
       - HBM → register：y_swi @ W3 → y_out，store to HBM
    5. **Backward**：
       - ∇W3 = y_swi^T · ∇y_out（GEMM）
       - ∇y_swi = ∇y_out · W3^T（GEMM）
       - Recompute SiLU(a) from a（element-wise，memory bandwidth bound）
       - ∇a = ∇y_swi ⊙ b ⊙ ∇SiLU(a)，∇b = ∇y_swi ⊙ SiLU(a)_recomp（fused tiled reduction）
       - FusedBwdW：∇W1 = a^T · ∇a，∇W2 = b^T · ∇b（fused kernel）
       - FusedBwdX：∇x = ∇a · W1^T + ∇b · W2^T（in-place aggregation）
    6. **测量**：PyTorch saved_tensors_hooks 追踪所有通过 ctx.save_for_backward 保存的 tensor → 计算总字节数作为 activation memory。Wall-clock time 由 CUDA events 测量，排除 optimizer 更新时间。
  - 性能要点：相比 MegaBlocks 的 sort-based dispatch（multi-pass radix sort 需要多次 global memory passes + 多次 kernel launch），MoEBlaze 的 3-step 构建仅需单次 kernel launch chain，利用 shared memory 内的 prefix sum 和 warp-level reduction 避免 atomics，大幅减少 global memory traffic 和 kernel launch latency。

## Optimizing Dynamic Neural Networks with Brainstorm

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  实现是 Brainstorm 的高效 GPU kernel 集合（~3,000 LOC C++/CUDA），包含：(1) **Cell 重排列 kernel**——单 GPU 内根据 Router 的路由决策（Routes tensor）将 tensor 中的 Cell 按目标 branch 重新排列，生成每个 branch 对应的连续输出 tensor，借鉴 Tutel 的 MoE token rearrangement 思路但泛化到任意动态网络；(2) **Sparse All-to-All 通信原语**——多 GPU 场景下用点对点通信集合（point-to-point communication）替代 dense all-to-all，仅传输实际需要跨 GPU 的 Cell，避免因 routing 不均衡导致的 padding 冗余通信；(3) **动态水平融合 kernel 调度**——基于 Router profile 中的 Cell 负载分布，为每个 branch 编译多个不同 shape 的 tuned kernel（通过 TVM auto-tuning），运行时根据实际 Cell 数选择 nearest tuned kernel 并 padding 输入，一个 fused kernel 通过一次 GPU launch 并发执行所有激活的 branch。实验通过 micro-benchmark 比较：(a) 稀疏通信 vs PyTorch all-to-all (NCCL) 的延迟，变化 branch 数（2~8）、GPU 数（2~8）、Cell 大小（32~2048 float32）；(b) 动态水平融合 vs 串行执行（PyTorch）vs 仅 vertical fusion 的延迟，变化 branch 数（2~10）；(c) Profile-guided placement 通信延迟对比；(d) Speculative routing 和 preloading 的 hit/miss 延迟对比。

- 后端平台是什么，配置是什么。
  单 GPU：NVIDIA A100 80GB，CUDA 11.3 + cuDNN 8.6。多 GPU：8× NVIDIA V100 32GB（NVLink），CUDA 11.3 + cuDNN 8.2，NCCL 通信后端。CPU：单 GPU 实验用 AMD EPYC 7V13，多 GPU 实验用 Intel Xeon E5-2690 v4。

- 评估性能的软件/脚本是什么。修改了什么。
  Brainstorm 自定义 CUDA kernel 代码。micro-benchmark 和 end-to-end benchmark 脚本在 GitHub artifact（osdi2023ae 分支）中，提供 shell 脚本自动复现 Figures 12-23。修改内容：
  1. Cell 重排列 kernel：在单个 GPU 上接收 (input_tensor, routes_tensor, num_branches) → 扫描 routes 统计每个 branch 的 Cell 数量 → 分配输出 buffer → 根据 routes 将每个 Cell 从源 tensor scatter copy 到对应 branch 的输出 tensor。实现为 CUDA kernel，支持并行处理多个 branch 的 scatter 操作。
  2. 稀疏通信：将多 GPU 的 Cell routing 实现为 N×N 个点对点 send/recv（而不是一次 all-to-all collective），每对 GPU 间只传输实际需要的 Cell 数量（无 padding）。例如 1024 Cells 分发到 4 个 GPU 上 4 个 branch 各一个，PyTorch all-to-all 需将每个 GPU→每个 GPU 通道 padding 为等量（max 量），而稀疏通信仅传输实际数据。
  3. 动态水平融合的 kernel dispatch：fused kernel 编译时包含多个 tuned 子 kernel（如 Conv2D for 4/32/64 cells），运行时通过 Router 得知每个 branch 的实际 Cell 数后，选择 nearest tuned kernel 并 padding 到对应 shape，计算 global input pointer offsets 后单次 GPU launch 执行所有激活 branch。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  全开源：https://github.com/Raphael-Hao/brainstorm，OSDI 2023 artifact。Docker 镜像预配置，artifact 中每个 Figure 有对应的 shell 脚本自动复现。

  **评估原理与 kernel 全流程（以 sparse all-to-all 和动态水平融合为例）**：

  **Sparse All-to-All 通信原理（Fig 12/14）**：
  1. **输入**：GPU-0 上有 1024 个 Cell（每个 512 float32），Router 决策将 Cell 不均匀分发到 4 个 GPU 上的各 4 个 branch（共 16 个目标）。例如 GPU-0 的 Branch-0 收到 200 个 Cell，Branch-1 收到 10 个，Branch-2 收到 50 个，Branch-3 收到 100 个；GPU-1 同理。
  2. **Baseline (PyTorch all-to-all)**：将 1024 Cells reshape 为 (4, 256)（4 GPU 均匀分片），调用 NCCL all-to-all。由于实际分布不均，每个 GPU 接收 400 Cells 的 buffer 中仅部分有效（其余为 padding），且总的传输量为 1024×4=4096 Cells（含 padding）。
  3. **Brainstorm 稀疏通信**：Router 输出 routes，统计每个 (src_gpu, dst_gpu) pair 的实际 Cell 数量。生成 N×N 个 point-to-point ncclSend/ncclRecv，每对仅发送实际需要的 Cell 数。例如 GPU-0→GPU-1 仅需传 60 个 Cell（含 padding 的 256 相比），总传输量 = sum(实际 Cell 数) 而非 N×总 Cell 数。
  4. **性能输出**：1024 Cells × 512 float32，4 branch/GPU，2 GPU 加速 2.13×，8 GPU 加速 2.66×。加速来源为避免 padding 的额外带宽消耗。

  **动态水平融合 kernel 原理（Fig 13）**：
  1. **Profile 阶段**：Router 收集 branch-0/1/2/3 的历史 Cell 负载分布。发现 P50=4 patches, P90=9 patches, P100=27 patches。
  2. **Tuning 阶段**：Brainstorm 对每个 branch 的 Conv2D 算子用 TVM 为 shape (4, C, H, W) 和 (27, C, H, W) 各 auto-tune 一个 kernel。
  3. **Fused kernel 构建**：生成一个 fused kernel，内含两个子 kernel（4-patch kernel 和 27-patch kernel），通过 global input pointer offsets 指向不同 branch 的输入 buffer。
  4. **运行时**：Router 得知当前 branch-0 收到 4 patches → 使用 4-patch kernel（无 padding）。branch-1 收到 8 patches → 选择 27-patch kernel（padding 19）。branch-2 收到 0 patches → 跳过。一次 GPU launch 并发执行 branch-0 和 branch-1，显著提高 CU utilization。
  5. **性能输出**：vs PyTorch 串行执行加速最高 41.8×（其中 kernel tuning 贡献 13.1×，并发执行贡献 3.18×）。Branch 数较少（≤3）时 overhead 略大于收益，BRT+HF 比 BRT+VF 慢仅 12.3μs。

## Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

- 属于kernel调度/运行时计算的实现是什么？实验比较什么？
  - 实现：在 wafer-scale multi-chiplet GPU 的 Global Command Processor 上运行的两个运行时算法：(1) **Task Allocation Algorithm (Alg. 1)**——将 MoE kernel 计算按 expert 拆分为 per-die 子 kernel，基于 expert placement 信息和 cost model（考虑 DRAM access、computation、D2D communication）将子任务分配到各 die；(2) **Data-Driven Predictor**——利用 cross-token heatmap 预测下一 token 的热门 expert，为 hardware-managed HBM 提供 duplication 指导（cp_en bits），在 kernel 执行期间自动将远程热门 expert 缓存到本地 HBM。
  - 实验比较：(1) Allo Only vs Base vs EP 的 throughput 和 hop count（Task Allocation 的独立效果）；(2) Pred Only vs Base vs EP 的 throughput 和 hop count（Predictor 的独立效果）；(3) Allo+Pred 的组合效果；(4) DRAM access breakdown（local reads / remote reads / local writes）展示两种策略如何将远程读转化为本地读；(5) Host CPU-based 实现 vs GPU CP-based 实现的 overhead 对比（Dojo vs Dojo-Enhanced）。

- 后端平台是什么，配置是什么。
  - **模拟平台**：自研 Python 事件驱动 multi-chiplet GPU simulator（开源：https://github.com/zhongkaiyu/waferscale_gpu_moe_sim）。
  - **硬件配置**（两种 chiplet 拓扑）：
    - **Tesla Dojo**：5×5 2D mesh（25 dies），每 die H100-like（1000 TFLOPS FP16, 80GB HBM, 3.35 TB/s HBM BW, 1.7 TB/s D2D BW）。LLC: 64 MB, 100ns hit latency。D2D: 200ns link latency, XY routing。
    - **TSMC SoW**：8×3 2D mesh（24 dies），每 die 相同 H100-like 配置。
    - **Dojo-Enhanced**：5×5 mesh，每 die B300-like（4500 TFLOPS FP16, 180GB HBM, 8 TB/s BW, 2 TB/s D2D BW）。
  - **模型**：DeepSeek V3 (671B, 256 experts), Kimi K2 (1000B, 384 experts), Llama4 Maverick (402B, 128 experts), Qwen3-235B (235B, 128 experts)。
  - **workload**：real traces from MMLU, MMLU Pro (Chinese), ChineseSimpleQA, LiveCodeBench (>24,000 requests per model)。

- 评估性能的软件/脚本是什么。修改了什么。
  - 评估软件：自研 Python multi-chiplet GPU simulator（`main_ae.py` 入口），模拟 wafer-scale GPU 上 MoE decode 阶段的 kernel 执行。
  - 修改/新增的核心算法：
    1. **Task Allocation Algorithm (Alg. 1)**：
       - 输入：`expert_reqs_dict`（每个 expert 的请求数）、`expert_die_map`（Expert Distribution Table 提供的每个 expert 所在 die 信息）
       - 候选机制 (Candidate Mechanism)：对每个 expert，候选 die 列表 = 存有该 expert 的 die + 距离为 1 的邻居 die，按负载排序后限制为 `max_split_num`（由 expert 请求数决定）
       - 块粒度分配 (Block-Granularity Distribution)：以 block size = 50 为单位分配请求，每 block 用 cost model（DRAM access + compute + D2D communication）选择最优 die
       - 合并：将分配到同 die 的任务合并为最终 allocation plan
    2. **Data-Driven Predictor**：
       - 输入：当前 MoE kernel 中各 expert 被选择的情况 + cross-token heatmap（离线 pre-computed）
       - 从 heatmap 中定位当前被选择的 expert 对应行（①），每行取 top-n 下一 token 最可能的 expert（②）→ 预测结果为这些 expert 的并集（③）
       - 输出：每个 die 的 cp_en bits（标记哪些 expert 应被本地缓存），写入 PDU prediction table
    3. **Cost Model**：考虑 DRAM access latency（local 300ns, remote 多跳 D2D + remote DRAM）、computation time（基于 TFLOPS）、D2D communication（bandwidth + hop count contention）
  - 实验 metric：decode 阶段 MoE layer throughput（tokens/s），hop count（所有 cross-die 通信的 Manhattan 距离之和），DRAM access breakdown。

- 开源情况。基于开源文档和论文，使用例子解释评估软件/脚本如何使用？至少具体到评估软件的评估原理和kernel输入到性能输出的全过程。
  - **开源**：完整代码仓库 https://github.com/zhongkaiyu/waferscale_gpu_moe_sim（Apache-2.0），DOI: 10.5281/zenodo.19617713。提供 `main_ae.py` 入口脚本自动下载 traces → 运行实验 → 生成 CSV → 生成 Figure 12。
  - **评估原理与全流程（以 DeepSeek V3 on TSMC SoW 8×3，batch size 16384 为例）**：
    1. **Trace 加载**：从 HuggingFace 下载 DeepSeek V3 的 expert selection traces（JSON 格式，每层每 token 的 top-k expert IDs）。TSMC SoW 8×3 拓扑初始化——24 个 die 对象，每个含 HBM/LLC/compute/D2D link model。Expert 按 EP-like 均匀分配到 24 个 die。
    2. **Batch 构建**：按 MMLU → MMLU-Pro (CH) → ChineseSimpleQA → LiveCodeBench 顺序填充请求直到达到 target batch size 16384。从 traces 中提取对应请求的 expert selection。
    3. **每层 MoE kernel 执行（decode stage）**：
       - **Global CP 阶段**：(a) 统计当前 batch 中每个 expert 的请求数 → expert_reqs_dict；(b) 运行 Task Allocation Algorithm——对每个 expert（按请求数升序），生成候选 die 列表 = 存有该 expert 的 die + 邻居 die → cost model 评估每个候选 → block size 50 贪心分配 → 合并生成分配计划；(c) 运行 Predictor——从 cross-token heatmap 查当前 expert selection 的对应行 → 每行取 top-n → 预测下一 token 热门 expert → 生成 cp_en bits。
       - **执行阶段**：Global CP 通过 D2D 网络发送子 kernel + prediction 信息到各 Local CP。Each Local CP 分配任务到 SM。SM 请求数据时：(i) PDU 检查 is_local → 若已缓存，ATU 翻译地址从本地读取；(ii) 若远程且未缓存 → D2D XY routing 多跳读取 → 返回时 PDU 检查 cp_en → 若需缓存，写入本地 HBM + 更新 ATU。
    4. **性能统计**：(a) 每层执行时间 = max(all die completion time)，汇总所有 MoE 层得到总执行时间 → throughput = batch_tokens / total_time；(b) Hop count = 所有 D2D 请求的 Manhattan 距离之和；(c) DRAM access 分类统计 local read / remote read / local write。
    5. **关键结果**：Allo+Pred 实现 7.5× throughput on TSMC（vs 6.0× on Dojo，因 TSMC 矩形布局 die 间距更大故 baseline 更差）。Hop count 降低 >213×。Allo Only 的 hop 降低 142× 证明 allocation 已将多数请求分配到本地 die；Pred Only 额外将 remote DRAM reads 转换为 local reads。
