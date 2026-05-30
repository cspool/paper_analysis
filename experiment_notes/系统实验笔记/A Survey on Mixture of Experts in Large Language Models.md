## A Survey on Mixture of Experts in Large Language Models

- 属于Serving调度的实现是什么？实验比较什么？
  本论文为综述，不提供原始实验。它在系统层面（Section 5）对 MoE 系统的 Computation、Communication、Storage 三方面进行了系统性分类：
  - **Expert Parallelism 流程**（Fig 8a）：Gate Routing → Input Encode → All-to-All Dispatch → Expert Computation → All-to-All Combine → Output Decode
  - **混合并行策略**（Fig 8b-d）：Data+Expert+Tensor Parallelism、Data+Expert+Pipeline Parallelism、Expert+Tensor Parallelism
  - **计算优化（Section 5.1）**：动态专家放置（SE-MoE, Tutel, FlexMoE, SmartMoE）、动态影子专家策略（FasterMoE）、定制 GPU kernel（DeepSpeed-MoE, FastMoE, HetuMoE, Tutel）
  - **通信优化（Section 5.2）**：分层 All-to-All（DeepSpeed-MoE, HetuMoE, ScheMoE）、拓扑感知路由（FasterMoE, TA-MoE, SE-MoE）、专家亲和性预分配（ExFlow）、计算-通信流水线重叠（Tutel, FasterMoE, PipeMoE, MPipeMoE, Lancet）、架构解耦打破通信依赖（ScMoE, Arctic Dense-MoE hybrid）
  - **存储优化（Section 5.3）**：层级存储专家 offloading（SE-MoE, Pre-gated MoE, EdgeMoE: GPU HBM → CPU → SSD）、缓存/预取（expert selection forecasting + prefetching）、激活值内存优化（MPipeMoE: buffer sharing + recomputation/CPU offload）

- 硬件平台是什么，配置是什么。
  综述覆盖的硬件平台为多 GPU 分布式系统（NVIDIA GPU），通信通道涵盖 intra-node（PCIe, pre-4th-gen NVLink）和 inter-node（Ethernet, InfiniBand, 4th-gen NVLink）。

- 开源Serving框架是什么。修改了什么。
  **开源框架**（Table 4，截至2024年6月 Star 数）：
  - DeepSpeed-MoE（Microsoft, 33K stars）：在 DeepSpeed 上增加 MoE 支持（expert parallelism + 分层 All-to-All + 定制 GPU kernel）
  - ColossalAI/OpenMoE（38K stars）：MoE 训练与推理支持
  - Tutel（Microsoft, 672 stars）: 自适应并行策略切换 + 通信计算重叠
  - FastMoE（Tsinghua, 1.4K stars）: 定制GPU kernel + 通信优化
  - Fairseq-moe（Meta, 29K stars）：MoE 多语言训练
  - Megablocks（Stanford, 1.1K stars）：块稀疏 GPU kernel
  - ScatterMoE（Mila, 140 stars）：ParallelLinear 散操作
  - SE-MoE（Baidu, 21K stars, 基于 PaddlePaddle）：动态放置 + 通信 + 存储
  - HetuMoE（PKU, 236 stars）：定制 kernel + 分层 All-to-All
  - Mesh-TensorFlow（Google, 1.6K stars）

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文关联资源仓库：https://github.com/withinmiaov/A-Survey-on-Mixture-of-Experts-in-LLMs。
  各框架开源链接见 Table 4。

  **Expert Parallelism 端到端执行过程（以 DeepSpeed-MoE 为例）**：
  1. **输入**：batch tokens 经由数据并行分布在 N 个 GPU 上，每 GPU 持有部分 expert + 全部非 expert 参数（Attention, LayerNorm, Router 等）
  2. **Attention + Router 计算**（本地 GPU 执行，无通信）
  3. **Input Encode**：每个 GPU 将需要发送到同一 expert 的 tokens 聚合为连续内存块
  4. **All-to-All Dispatch**：将编码后的 token 数据发送到持有对应 expert 的 GPU（跨 GPU 通信）
  5. **Expert Computation**：各 GPU 对接收到的 tokens 执行本地 FFN 计算
  6. **All-to-All Combine**：将 expert 输出传回原始 token 所在的 GPU
  7. **Output Decode**：恢复原始 token 排序，加权合并 expert 输出
  8. **继续下一层**：重复 2-7

  **关键优化技术示例**：
  - **分层 All-to-All（DeepSpeed-MoE）**：优先利用高带宽 intra-node 通道（NVLink），减少低带宽 inter-node 数据交换
  - **拓扑感知路由（TA-MoE）**：将 token 优先路由到同节点的 expert，minimize cross-node 通信
  - **Lancet 重叠**：将非 MoE 计算（Attention, LayerNorm）插入 All-to-All 通信缝隙，延长重叠窗口
