## HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

- 属于Serving调度的实现是什么？实验比较什么？
  - HybridEP 修改 Tutel 分布式 MoE 训练框架的 Expert Parallelism (EP) 通信模式，实现跨数据中心场景下的高效 EP 扩展，核心实现包括三部分：
    1. **Stream-Based Modeling（流建模）**：将 MoE 训练过程解耦为计算流（GeMM 建模：Attention + FFN + Expert）和通信流（A2A + AG），通过重叠建模分析两流之间的关系，推导出以最小化训练延迟为目标的优化问题，求解最优的 A2A/AG 混合比例 p。当 p=1 时退化为标准 EP（纯 A2A），当 p=0 时全部使用 AG 通信。模型根据配置自动判断三种 case（Case 1: 混合 A2A+AG，Case 2.1: 混合 A2A+AG，Case 2.2: 仅 AG）。
    2. **Domain-Based Partition（域分区）**：定义 Expert Domain（专家域）为一组仅使用 AG 通信的 DC 集合。遵循"域内用 AG、域间用 A2A"的规则，通过 Multilevel Description（多层级架构抽象，用 scaling factor SF^i 表示层级扩展）→ Location Renumbering（GPU 全局编号按 PyTorch 格式重映射到多层级位置）→ Topology Construction（Algorithm 1: 通信拓扑构建，对每对 GPU 逐层判断 AG/A2A/None）三步将通信模式映射到 GPU 级别的具体通信拓扑，确保与现有层级硬件架构兼容。
    3. **Parameter-Efficient Migration（参数高效迁移）**：通过 SR-Based Expert Compression 和 Asynchronous Communicator 实现轻量级专家迁移。压缩分为 SREncode（共享+残差分离 → Top-k 稀疏压缩）和 SRDecode（恢复+加法 fused）两阶段。异步通信器分 Initialization 阶段（SREncode 与上一 iteration optimizer step 融合）和 Asyn-comm 阶段（AG 通信与 pre-expert computation 重叠）。压缩比最高 50× 不损失精度。
  - 实验比较：
    - Baselines：Tutel、FasterMoE、SmartMoE（均针对 MoE HPC 环境优化）
    - 建模验证：computation/communication latency 估计精度、最优 p 搜索（4 种 case 各 4 个候选 p）
    - 端到端加速比：不同 data traffic (6-192MB) 和不同 expert size (2-32MB) 下的 iteration 时间
    - 消融实验：Domain-Based Partition（baseline）vs +Parameter-Efficient Migration（完整 HybridEP）
    - 迁移分析：SR 压缩 loss 对比、时间分解（SREncode/SRDecode overhead 及 fusion 效果）
    - 特性对比：EP vs HybridEP 的通信流量（input-dependent vs fixed-bound traffic）和频率（A2A→AG 转换）
    - 大规模仿真：SimAI 模拟 1000 DCs 下的加速比（固定 S_ED vs 固定 p）

- 硬件平台是什么，配置是什么。
  - **Cluster-S**（单 DC）：8 × NVIDIA A800 GPU in a single node
  - **Cluster-M**（2 DCs）：16 × NVIDIA A800 GPU on 2 nodes（每个 node 视为一个 DC）
  - **Cluster-L**（4 DCs）：32 × NVIDIA A800 GPU on 4 nodes
  - 节点内互联：PCIe 3.0 x16 (128 Gbps)
  - 跨节点（DC 间）互联：Ethernet (10 Gbps)，模拟跨 DC 低带宽场景
  - 软件环境：Ubuntu 18.04, CUDA 11.3, cuDNN 7.6, NCCL 2.10
  - 大规模仿真平台：SimAI (USENIX NSDI 2025)
  - 优化器：Adam (lr=1e-4)，PyTorch DDP (All-Reduce 同步梯度)

- 开源Serving框架是什么。修改了什么。
  - 框架：**Tutel** (https://github.com/microsoft/tutel)，Microsoft 的自适应 MoE 分布式训练框架，支持多维分层 All-to-All 和动态 expert 分配。
  - HybridEP 代码未公开开源。论文未提供开源链接，在 Tutel + PyTorch v1.12.1 之上以原型系统实现。
  - 核心修改：
    1. **通信模式替换**：将 Tutel 的纯 A2A 通信替换为混合 A2A+AG 通信。在训练前通过 Stream-Based Modeling 根据环境配置（B, G, P_E, D, Lat_comp^PE）计算最优 p 值，决定 A2A/AG 比例。
    2. **Domain-Based Partition 新增**：新增 Expert Domain 管理层（Multilevel Description + Location Renumbering + Topology Construction Algorithm 1），将 model 输出的通信比例映射到 GPU 级通信拓扑。
    3. **Parameter-Efficient Migration 新增**：SR-Based Expert Compression（shared expert + residual Top-k 压缩，value-index 稀疏传输格式）+ Asynchronous Communicator（Send Queue/Recv Queue 管理，与 optimizer step 和 pre-expert computation 重叠执行）。
    4. **模型驱动配置**：训练前根据模型和环境参数自动计算最优 p 值和 expert domain 大小 S_ED^l。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  - HybridEP 代码未公开开源。论文未提供开源链接。
  - HybridEP MoE 训练全流程（以 Cluster-L: 4 DCs × 8 GPUs, EP=32, p=0.5 为例）：

  ① **训练前：环境配置采集与建模求解**
     - 输入：G=32 GPUs, B_intra=128Gbps, B_inter=10Gbps, P_E, D, Lat_comp^PE
     - Stream-Based Modeling 求解：Formula (1)-(12) → 判断 2D - G·P_E 正负 → 确定 case → 输出最优 p
     - 根据 p 计算 S_ED：p = (G - S_ED) / G → S_ED = G(1-p)

  ② **训练前：Domain-Based Partition 构建拓扑**
     - Multilevel Description：4 DCs × 8 GPUs → 2 levels，SF^0=4, SF^1=8
     - 根据 optimal p 设定 S_ED^0, S_ED^1
     - Location Renumbering (Eq.13)：32 个 GPU 全局索引 m → (x_0, x_1)
     - Topology Construction (Algorithm 1)：对每对 GPU (m,n)，逐层判断通信类型——同域内且同 offset→AG；不同域且同 offset→A2A；否则 None

  ③ **训练 iteration 开始（Forward Pass）**
     - Token batch (B tokens) → Embedding → pre-expert 层（Attention + FFN），按 DP 分布在各 GPU
     - **Asyn-comm 阶段**（与 pre-expert computation 重叠）：
       - Send Queue 弹出上一 iteration SREncode 的压缩 expert 残差
       - NCCL All-Gather：域内 GPU 间收集压缩后的 expert 参数
       - Recv Queue 接收 → SRDecode 恢复完整 expert = shared_expert + decompress(residual)

  ④ **Gate Network + Expert Dispatch**
     - Gate 计算 routing weights → Top-K 选择 activated experts（每个 token 选 K 个 expert）
     - **域间 A2A**（仅对 p 比例的数据）：token data 按 routing result 通过 NCCL All-to-All 发送到对应 expert 所在 GPU
     - **域内 AG**（对 1-p 比例的数据）：对应的 token 不再通过 A2A 传输，因为 expert 已通过 AG 收集到本地

  ⑤ **Expert FFN 计算（GPU 本地）**
     - 每 GPU 对收到的 tokens 执行本地 experts 的 FFN 前向
     - W_gate GEMM → SiLU activation → W_up GEMM → element-wise multiply → W_down GEMM
     - cuBLAS GEMM kernel 在 NVIDIA A800 Tensor Cores 上执行

  ⑥ **Expert Combine（A2A 逆向）**
     - 域间 A2A 通信将 expert 输出按原 token 位置合并回各 GPU

  ⑦ **Parameter-Efficient Migration（与 optimizer step 融合）**
     - SREncode：计算 expert 残差 = expert - shared_expert → Top-k 压缩 → 稀疏 value-index 格式 → 存入 Send Queue
     - 与当前 iteration 的 optimizer step（Adam 参数更新）融合执行，减少 ~30% 的编码 overhead

  ⑧ **Backward Pass**
     - Expert FFN backward → A2A combine backward → A2A dispatch backward → Attention backward
     - Shared expert 梯度通过 All-Reduce 在 backward 阶段同步

  ⑨ **输出**：完成一层的 forward+backward，所有 experts 和 shared expert 参数更新完毕，Send Queue 已为下一 iteration 准备压缩后的 expert 残差，进入下一 iteration
