## MoEQuant Enhancing Quantization for Mixture-of-Experts Large Language Models

- baseline方法是什么？
  - Baseline 是 **Wanda (2:4 结构化稀疏)** 和 **传统 post-training weight pruning 方法**（如 SparseGPT）。这些方法对 LLM 的线性层权重矩阵做非结构化或半结构化稀疏，虽然能减少总参数数量，但依赖专用硬件（FPGA 等）才能实现高效部署。此外还有 **Random Expert Pruning**（随机丢弃专家）和 **Frequency-based Expert Pruning**（按校准数据上的激活频率丢弃专家）作为 MoE 专家级 baseline。
  - 全栈执行例子（以 Wanda 2:4 baseline 在 Mixtral 8x7B 上一个 token 的推理为例）：
    - **算法层**：Wanda 逐层对每个线性层的权重计算 importance score = |W_{ij}|·‖X_j‖_2，在 2:4 模式下每 4 个连续权重保留 2 个，其余置零 → 得到稀疏权重矩阵。Mixtral 8x7B 上 Wanda 2:4 实现约 50% 参数减少，但推理速度反而低于 dense 模型（0.91-0.92× speedup），因为 2:4 结构化稀疏需要特定硬件加速（NVIDIA Ampere Sparse Tensor Core 或 FPGA），通用 GPU 上无加速优势。此外权重稀疏对 MoE 架构无针对性优化——专家参数占总参数 ~96%，但 Wanda 对所有权重同等对待，不区分专家重要性差异。
    - **系统框架层**：基于 PyTorch semi_structured_sparse（https://pytorch.org/tutorials/prototype/semi_structured_sparse.html）实现。推理时使用 HuggingFace Transformers 标准加载管线，稀疏权重以 dense 格式存储（mask + values），无专门 MoE 优化。原始 Mixtral 8x7B (bf16) 需 2 张 A100-80G GPU 部署，Wanda 2:4 后仍需 2 张 GPU（51GB 内存）。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel，无自定义 sparse kernel。2:4 结构化稀疏在通用 GEMM kernel 上无加速（甚至因 mask 检查额外开销而减速）。
    - **硬件架构层**：NVIDIA A100-80G GPU，未使用专用稀疏硬件加速器。
  - **Baseline 的核心缺陷**：
    1. **权重级稀疏与 MoE 结构不匹配**：Wanda/SparseGPT 对所有 FFN/Attention 权重均匀裁剪，忽略 MoE 架构中专家才是主要参数载体（8 个专家占 Mixtral 8x7B 总参数 96%）这一结构特征。专家作为独立的 FFN 子网络可整体移除而无需改变其余模型结构，这是权重级稀疏无法利用的。
    2. **需要专用硬件才能实现推理加速**：2:4 结构化稀疏的加速依赖 FPGA 或支持 sparse MMA 的 Tensor Core，通用 GPU 上实际减速。部署不具 plug-and-play 特性。
    3. **Random/Frequency baseline 对专家重要性估计不准**：Random 丢弃不考虑专家贡献；Frequency-based 仅按校准数据上的激活频率排序，忽略不同 token 对专家使用的差异——论文实验显示 activation frequency baseline 甚至比 random 更差，因为 MoE 模型可能对特定专家有路由偏好但偏好不完全等于重要性。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法：Expert Pruning + Dynamic Expert Skipping**
  - **(1) Expert Pruning 解决缺陷 1 和 3**：
    - 将稀疏化粒度从"权重元素"提升到"专家"级别，利用 MoE 结构特征：每个 MoE 层包含 n 个独立专家 FFN，专家是天然的可移除单元。通过逐层枚举保留 r 个专家的组合，以最小化 Frobenius 范数量化重构损失 ‖F'(x,C) − F(x)‖_F 为目标，在 token 级别评估专家子集的重要性。
    - 重构损失在 token 输出层面度量（而非权重层面），直接衡量专家组合对模型最终输出的影响，比 activation frequency 更准确地反映专家贡献。在 Mixtral 8x7B 上，论文方法 r=6 时平均性能仅下降 2.9 点（vs Random 4.5 点, Frequency 6.8 点）。
    - 领域特定剪枝：将校准数据集从 C4 切换到 MATH 训练集，使剪枝过程聚焦领域知识保留。在 GSM8K 5-shot 上 C4 剪枝 r=6 仅 41.02 vs MATH 剪枝 51.25——校准数据选择对领域性能至关重要。
  - **(2) Dynamic Expert Skipping 解决加速问题**：
    - 推理时根据路由权重比值 w_{e1}/w_{e0} 与逐层阈值 β（校准集上该比值的中位数）动态决定是否跳过次优专家。不依赖硬件稀疏支持，是纯软件层面的推理加速——通过减少每个 token 实际执行的专家 FFN 数量来减少 FLOPs。
    - 跳过机制基于权重比而非固定阈值，自适应不同 token 的路由分布。β 取中位数使跳过概率约 50%，在精度与加速间取得平衡。
  - **(3) 组合使用实现全局优化**：
    - 剪枝（减少静态参数 → 内存节省）+ 动态跳过（减少运行时计算 → FLOPs 节省）正交互补。r=6 剪枝 + 动态跳过的组合（62.91 avg accuracy）比 r=4 纯剪枝（59.57）精度更高，但推理速度相当（1.23× vs 1.27×）——以更少专家数获得更高精度，证明动态跳过的效率。
  - 论文方法全栈执行例子（以 Mixtral 8x7B 一个 token 推理为例）：
    - **算法层**：
      1. **离线剪枝阶段**：加载 C4/MATH 校准集（128 条 × 2048 tokens）→ 逐层前向传播缓存输入-输出对 → 逐层枚举 expert combinations（C(n,r)），对每个组合 C 计算 F'(x,C) = Σ w̃_{e_j}·E_{e_j}(x)，取 min ‖F'(x,C)−F(x)‖_F 的组合 → 修改模型 config 仅保留 r 个专家 → 保存 pruned checkpoint
      2. **离线 β 校准阶段**：pruned checkpoint 前向校准集 → 逐层收集 w_{e1}/w_{e0} 比值 → β[l] = median(ratios)
      3. **推理阶段**：token x → 每 MoE 层：路由计算 top-2 (e0, e1) → 若 w_{e1} < β[l]·w_{e0} 则 y = E_{e0}(x)，否则 y = w̃_{e0}·E_{e0}(x) + w̃_{e1}·E_{e1}(x)
    - **系统框架层**：HuggingFace Transformers，仅修改模型 config 中的 expert 数量即可加载剪枝模型——不需要修改模型代码或引入新的 layer type。动态跳过的路由逻辑通过自定义 MoE layer forward 实现，核心改动不到 20 行代码。
    - **编译框架层**：论文未明确说明。
    - **kernel调度层**：标准 PyTorch CUDA kernel。减少 GPU 间通信（从 2 张 GPU 降为 1 张 GPU 后无需跨 GPU 通信）是推理加速的主要来源。动态跳过减少每个 token 的 expert FFN 计算量（平均少执行 0.5 个 expert/token）。
    - **硬件架构层**：NVIDIA A100-80G GPU。剪枝后单卡部署消除了跨 GPU NCCL 通信开销。论文无需专用硬件——plug-and-play 部署。
  - 关键设计动机映射：
    - 权重级稀疏需要专用硬件 → 专家级稀疏（整体移除/跳过专家）在标准 GPU 上即插即用
    - Activation frequency 不反映真实重要性 → 基于 token 重构损失的枚举搜索准确评估专家贡献
    - 静态剪枝无法减少 FLOPs → 动态跳过在线减少激活专家数，真正减少计算量
    - 通用校准集不适合领域任务 → 切换校准数据集到领域数据实现 task-specific pruning
    - 剪枝后模型仍有性能下降 → 通过微调（MetaMathQA 900 步）几乎完全恢复性能，r=7 剪枝模型在 GSM8K 上超越 8-expert 原始模型
