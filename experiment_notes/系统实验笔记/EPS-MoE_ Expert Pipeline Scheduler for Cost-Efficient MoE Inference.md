## EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

- 属于Serving调度的实现是什么？实验比较什么？
  EPS-MoE 提出了一种用于 MoE 推理的 Expert Pipeline Scheduler，包含三个核心模块：(1) 并行策略选择——对 Attention 块使用 DP 或 TP、对 MoE 块使用 EP，支持 TP+TP、DP+EP、TP+EP 三种模式的理论分析；(2) Expert Pipeline Scheduler——基于水平切分输入张量和按专家切分权重的方法，实现 load-aware 动态切换 GroupGemm 和 DenseGemm；(3) 计算与通信重叠——在 kernel 级别将 GEMM 计算与 all2all 通信 pipeline 并行，并通过控制 SM 数量优化重叠效率。
  
  实验比较：
  - Baseline: vLLM 的 TP+TP 实现（已 kernel tuning 至最佳性能）
  - EPS-MoE 的 DP+EP（DeepSeekV2）或 TP+EP（Mixtral/DBRX）模式
  - 测试 PN=1（仅 GEMM 切换，无 pipeline）、PN=2/5/8/20（pipeline 调度）
  - 消融实验：GEMM 切换 vs 重叠的贡献分解、FP8 通信影响、SM 数量控制必要性、不同输出维度影响、最优 PN 选择

- 硬件平台是什么，配置是什么。
  - DeepSeekV2: 8xH800-80GB SXM
  - Mixtral 8x7B: 4xH800-80GB SXM  
  - DBRX: 8xH800-80GB SXM
  - Snowflake Arctic: 8xH800-80GB SXM
  - 消融实验: 4xH800-80GB SXM（单层 Gate/Up/Down GEMM 测试）
  - 分析用: NVIDIA A100-80GB SXM（NVLink，用于通信和计算 profiling）
  - 矩阵计算库：cutlass、cublas

- 开源Serving框架是什么。修改了什么。
  基于 vLLM 框架集成 EPS-MoE。具体修改包括：
  1. **并行策略替换**：将 vLLM 原生的 TP+TP 替换为 TP+EP（对 MHA/GQA/MQA Attention）或 DP+EP（对 MLA Attention）。将 ncclAllReduce 分解为 ncclReduceScatter + all2all（dispatch 阶段）和 all2all + ncclAllGather（combine 阶段），减少 MoE 模型通信量。
  2. **Expert Pipeline Scheduler**：实现水平切分（行切分）输入张量、按专家切分 MoE 权重的调度策略，每次只传输当前专家组所需的 token 到对应设备。动态根据负载切换 GroupGemm（小 token 数时更优）和 DenseGemm（大 token 数时，如 m≥4096 时更优）。
  3. **SM 数量控制**：通过限制 GEMM kernel 占用的 SM 数量，为通信 kernel 留出 SM 资源，实现计算与通信的完美重叠。最优配置为 GEMM 116 SM + 通信 16 SM（H800 共 132 SM）。

- 开源情况。基于开源文档和论文，使用例子解释Serving框架如何使用？作用是什么？至少具体到框架输入到硬件执行的全过程。
  论文未明确提供开源代码链接（作者来自美团，arXiv:2410.12247）。论文说明了与 vLLM 的集成方式。
  
  **Serving 框架使用流程（以 Mixtral 8x7B 推理为例）**：
  
  1. **输入阶段**：用户请求 batch 到达，tokenized 为输入序列。batch 中的 token 被分发到各 GPU 设备。
  
  2. **Attention 块（TP 模式）**：Attention 权重按 TP 切分在各设备上。每设备持有 1/D 的 KVCache 和权重。输入 activation 经 TP 通信（ncclAllReduce，在 EPS-MoE 中分解为 ReduceScatter+AllGather+Flux 融合）完成 Attention 计算。
  
  3. **MoE 块（EP 模式 + Expert Pipeline）**：
     - **Router/Gating**：每个 token 经 gating 网络计算 top-k 专家选择。
     - **Dispatch 通信**：通过 all2all 将每个 token 发送到其选中的专家所在设备。EPS-MoE 使用 ReduceScatter+all2all 替代 ncclAllReduce，通信量更小。
     - **Expert Pipeline Scheduler**：
       - 输入张量按行水平切分（split rows），权重按专家切分。
       - 将专家分组为 N 个 pipeline stage，每个 stage 顺序提交一组专家的 GEMM 计算。
       - 若 m ≥ 4096（compute-bound prefill 阶段），切换到 DenseGemm（cublas）；若 m < 2048（memory-bound decode 阶段），使用 GroupGemm（cutlass）。
       - GEMM 计算与下一轮 all2all 通信重叠：通过限制 GEMM 占用 116 SM，通信占用 16 SM，两者在不同 SM 上并行执行。
     - **Combine 通信**：all2all + ncclAllGather 聚合各 token 的专家输出。
  
  4. **输出阶段**：MoE FFN 输出与 Attention 输出经 residual 连接，经 LayerNorm 后进入下一层。最终经 LM Head 输出 token 概率分布。
  
  **全流程数据路径**：
  `Input tokens → Tokenization → [Attention(TP) → Dispatch(all2all) → Expert FFN(GroupGemm/DenseGemm pipeline) → Combine(all2all+AllGather)] × L layers → LM Head → Output tokens`
  
  **关键加速器**：H800 SXM GPU (NVLink 互联)，每 GPU 132 SM，通过 SM 分区实现 GEMM 和通信的并行调度。
