## Llama 3 Meets MoE: Efficient Upcycling

- baseline方法是什么？
  - **Baseline**：Llama 3-8B dense 模型继续进行 Continued Training (CT)，即不进行 upcycling，直接使用 dense 模型在同量数据上继续训练。全栈执行例子：
    - **算法 Pipeline**：标准 Transformer decoder-only，每层包含 Multi-Head Self-Attention + SiLU-gated FFN，8B 参数全部激活，单 token 前向 FLOPs 约 4.7e14
    - **系统框架**：Megatron-Core + NeMo 分布式训练框架，使用标准 TP+PP+DP 并行策略
    - **编译框架**：论文未明确说明
    - **Kernel 调度**：标准 cuBLAS/cuDNN kernel，无 MoE-specific kernel
    - **硬件架构**：512× H100 GPUs，bfloat16 训练
  - Baseline 缺陷：(1) 扩展模型容量需等比增加计算量（参数翻倍 ≈ FLOPs 翻倍）；(2) 已投入的预训练 GPU 小时无法复用，每次扩展需从头训练；(3) Dense 模型在给定 compute budget 下存在性能天花板。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：Sparse Upcycling——从 Llama 3-8B dense checkpoint 出发，将部分 FFN 层复制 N=8 次初始化 expert（总计 34.4B 参数，仅 11.8B 激活），添加随机初始化 router，仅用 100B tokens (<1% 预训练 compute) 完成 MoE 训练。全栈执行例子：
    - **算法 Pipeline**：
      1. Upcycling：复制 FFN 权重 N 次为 experts，随机初始化 router（Mixtral-type: KeepTopK→Softmax，确保初始输出与 dense 一致）
      2. Expert Capacity Factor (CF=4)：控制每个 expert 最大处理 token 数，隐式引入正则化，防止 expert 过拟合
      3. 仅 100B tokens 训练（vs dense pre-training 数 T tokens），11K GPU hours（vs 估计 1.6M GPU hours from scratch）
    - **系统框架**：
      1. Online Upcycling in NeMo：按并行配置分片 dense checkpoint，各设备独立 upcycle，无需跨设备权重复制
      2. MoE Parallel Folding：解耦 Attention (TP×CP×DP×PP) 和 MoE (Expert-TP×EP×Expert-DP×PP) 并行映射，将通信密集操作折叠到 NVLink 高带宽域
      3. 5-D Hybrid Parallelism (TP+EP+PP+CP+DP ZeRO-1)
    - **编译框架**：论文未明确说明
    - **Kernel 调度**：Megatron-Core 提供的 AllToAll-based token dispatcher（TopK=1-4 时优于 AllGather-based）；对早期训练阶段 MoE 层启用 recomputation
    - **硬件架构**：512× H100 GPUs，bfloat16 训练，NVLink intra-node 高带宽通信 + InfiniBand inter-node
  - **解决 Baseline 缺陷的映射**：
    - 缺陷1（容量扩展与计算量等比增加）→ Upcycling 后 34.4B 总参数仅需 1.6× FLOPs（11.8B 激活参数），实现参数-计算解耦
    - 缺陷2（预训练投入不可复用）→ 直接复用 Llama 3-8B 预训练权重，100B tokens (<1% compute) 完成训练，11K vs 1.6M GPU hours
    - 缺陷3（dense 性能天花板）→ E8T2 MoE 在 MMLU 0-shot 提升 2%（65.20→64.00 in 5-shot, 62.10→64.10 in 0-shot），整体平均提升 ~1.2%
  - **关键设计选择**：
    - CF=4 为 accuracy-MFU 最佳平衡点（MMLU 0-shot 64.0 vs CF=1 的 63.7，MFU 39.4% vs 46.8%）
    - Mixtral-type router 比 ST-type 收敛更快（初始 loss 更低，因 upcycling 后初始输出与 dense 一致）
    - MoE Parallel Folding 下 TP1CP2EP8 配置达 46.8% MFU（128 H100）
