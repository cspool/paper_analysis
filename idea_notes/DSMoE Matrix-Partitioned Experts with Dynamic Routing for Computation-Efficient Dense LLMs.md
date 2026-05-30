## DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

- baseline方法是什么？
  **Baseline 包含两类方法**：
  
  1. **剪枝方法**：
     - **LLM-Pruner (channel-wise)**：结构化剪枝，沿 hidden dimension 缩减通道数，将 LLaMA-1B 从 d=2048 剪至 d=1215，激活参数 889M；LLaMA-7B 从 d=4096 剪至 d=2401，激活参数 3.95B
     - **LLM-Pruner (block-wise)**：结构化剪枝，沿 FFN intermediate dimension 缩减，LLaMA-1B 平均 D=3896.4，激活参数 735M；LLaMA-7B 平均 D=6256.5，激活参数 3.94B
     - **SparseGPT**：非结构化剪枝，权重级别稀疏化（50% 稀疏度），LLaMA-1B 激活参数 735M，LLaMA-7B 激活参数 3.93B
  
  2. **LLaMA-MoE**：将 FFN 划分为 8 个 expert（LLaMA-1B: D=1024×8, topK=3；LLaMA-7B: D=1376×8, topK=3），遵循 Switch Transformer 范式以固定 top-k 激活 + 传统 MoE 训练目标（含 load balancing loss），expert 由预训练权重 warm-start 初始化。
  
  **Baseline 全栈执行例子（以 LLaMA-7B SparseGPT 推理一个 token 为例）**：
  
  - **算法层**：输入 token embedding → attention 计算 → FFN 层使用 50% 稀疏的权重矩阵执行 GEMM（非结构化稀疏，需稀疏计算库支持加速）→ 下一层 attention → ... → 最后一层输出 → LM head 预测下一个 token。稀疏模式在剪枝时一次性确定，对所有输入固定不变。
  - **系统框架层**：论文未明确说明（标准 PyTorch/HuggingFace Transformers 推理）
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：非结构化稀疏需要专用稀疏 GEMM kernel（如 cuSPARSE）才能实际加速，否则稀疏权重仍需完整计算（论文仅评估 FLOPs 减少，未实现实际 wall-clock 加速）
  - **硬件架构层**：论文未明确说明 GPU 型号
  
  **LLaMA-MoE 全栈执行例子（以 LLaMA-7B, 8 experts, topK=3, 推理一个 token 为例）**：
  
  - **算法层**：token → attention → Router 计算 softmax(gate) → 选 top-3 expert → 仅 3 个 expert 的 FFN 参与计算 → 加权求和（由 softmax 门控值加权） → 下一层 → 输出。每个 token 固定激活 3 个 expert，与输入复杂度无关。
  - **系统框架层**：论文未明确说明
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：论文未明确说明
  - **硬件架构层**：论文未明确说明

  **Baseline 的核心缺陷**：
  1. **剪枝永久丢弃知识**：LLM-Pruner 和 SparseGPT 通过永久移除参数实现效率，被丢弃的权重中可能包含对特定输入模式有价值的知识，且无法根据输入复杂度动态调整计算量——简单 token 和复杂 token 处理量完全相同
  2. **固定 top-k 激活缺乏灵活性**：LLaMA-MoE 每个 token 固定激活 3 个 expert，无法根据输入实际需要（简单输入可能只需 1-2 个 expert，困难输入可能需要更多）自适应调节
  3. **传统 MoE 训练范式不适合预训练模型转换**：LLaMA-MoE 的 Router 从随机初始化训练，expert 的 warm-start 优势在 Router 未充分训练时被稀释；top-k softmax 路由使未被选中的 expert 难以接收有效梯度
  4. **Load balancing loss 与稀疏化目标冲突**：传统 MoE 的 load balancing 鼓励 expert 均匀负载，而 DSMoE 的目标是学习稀疏激活模式——两者方向相反
  5. **非结构化剪枝的实际加速困难**：SparseGPT 的 50% 非结构化稀疏需要专用硬件/库才能转化为实际 wall-clock 加速，否则 FLOPs 减少不代表推理变快

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：DSMoE = FFN 矩阵分区 + Sigmoid 门控动态路由 + Straight-Through Estimator + 稀疏损失，将预训练 Dense FFN 转换为输入自适应的稀疏 MoE，保留全部预训练知识的同时实现动态计算分配。
  
  **Defect → Design 映射**：

  | Baseline 缺陷 | DSMoE 设计选择 | 解决机制 |
  |---|---|---|
  | 剪枝永久丢弃知识 | FFN Partitioning：将原始 FFN 矩阵沿 intermediate 维切分为 n 个 expert，全部参数保留 | 所有 expert 输出之和在数学上等价于原始 FFN（公式6），知识零损失 |
  | 固定 top-k 缺乏灵活性 | Sigmoid 门控 + 阈值 τ：每个 expert 独立判断是否激活（σ(xY_i) > τ） | 简单 token 自动激活少 expert，复杂 token 激活多 expert，激活数由输入复杂度决定 |
  | Router 梯度阻断导致 "死 expert" | Straight-Through Estimator：S(x) = sg(G(x)) + x - sg(x) | 前向保持硬阈值稀疏，反向门控参数 Y_i 在所有 expert 上均接收梯度（公式16），非激活 expert 也能学习何时该激活 |
  | Dense 模型天然倾向全激活 | Sparse Loss：L1 惩罚 Σ G(σ(ĥY_n)) | 与门控梯度形成对抗，鼓励抑制不重要 expert，学习选择性激活 |
  | Load balancing 与稀疏化目标冲突 | 不引入 load balancing loss | 模型自由学习稀疏激活模式，不受均匀负载约束 |
  | 非结构化稀疏无法实际加速 | 结构化 expert 分区 + 硬阈值门控 | 激活/未激活 expert 边界清晰，可直接跳过未激活 expert 的矩阵乘法，实现实际计算节省 |

  **论文方法全栈执行例子（以 DSMoE LLaMA-7B, 8 experts, τ=0.5, 推理一个 token 为例）**：
  
  - **算法层**：
    1. Token embedding 输入 → Self-Attention → hidden state ĥ: [1, 4096]
    2. Gate 计算：ĥ @ Y [4096×8] → sigmoid → [g₁, ..., g₈]，e.g. [0.72, 0.13, 0.61, 0.08, 0.55, 0.02, 0.91, 0.04]
    3. 硬阈值 (τ=0.5)：激活 expert 1, 3, 5, 7（4/8 个），其余值置零
    4. 激活 expert 并行计算 SwiGLU FFN（每个 D=1376）：
       - Expert 1: silu(ĥ@W₁) ⊙ (ĥ@U₁) @ V₁ → o₁: [1, 4096]
       - Expert 3: ... → o₃
       - Expert 5: ... → o₅
       - Expert 7: ... → o₇
    5. 加权求和：h = o₁·0.72 + o₃·0.61 + o₅·0.55 + o₇·0.91
    6. 归一化：× 8/4 = ×2 → 最终 FFN 输出
    7. 进入下一 Transformer 层，重复 2-6
    8. 不同层、不同 token 激活不同数量 expert（形成 W 形层间激活模式：首尾层高激活、中间层突起、其余层低激活）
  - **系统框架层**：论文未明确说明（标准 PyTorch/HuggingFace Transformers 继续预训练 + 推理）
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：推理时可直接跳过未激活 expert 的矩阵乘法（结构化跳过，无需稀疏计算库），论文未明确说明具体 kernel 实现
  - **硬件架构层**：论文未明确说明

- baseline方法是什么？
  **Baseline 为三种分布式 MoE 训练方案**：
  
  1. **Vanilla (DeepSpeed-MoE Expert Parallelism)**：每 GPU 持有若干 expert，self-attention 层复制。Dispatch phase 通过 all-to-all 将 token 发送到对应 expert 所在 GPU，combine phase 再通过 all-to-all 将处理后的 token 拉回原 GPU 重构序列。通信量随 batch size 和 expert 数线性增长（MoE-BERT-Large 4 experts 时 all-to-all 通信 6.73GB/batch，占总时间 36.6%；8 experts 时占 47.5%）。
  
  2. **EXT (Expert Transfer, Janus)**：不移动 token，而是将远程 expert 复制到需要它的 GPU 上本地执行。减少 all-to-all 通信，但引入 expert 传输开销和 GPU 资源竞争——多个 expert 挤在同一 GPU 导致 expert computation 时间增长（如 MoE-BERT-Large 3 experts/GPU → computation 1.88×）。
  
  3. **HYT (Hybrid Token+Expert Transfer, FasterMoE)**：策略性地将 popular expert 复制到所有 GPU，结合 token 传输和 expert 传输。但仍有 GPU 资源竞争和 expert parallelism 降低的问题。
  
  **Baseline 全栈执行例子（以 Vanilla Expert Parallelism, 4 GPU, MoE-TransformerXL 训练一个 batch 为例）**：
  
  - **算法层**：输入 8 个 sequences → 各 GPU 独立执行 self-attention → Router (top-2 gating) 计算每个 token 的目标 expert → token→expert 映射
  - **系统框架层**：DeepSpeed-MoE expert parallelism → 4 GPU 各持有 1 个 expert + 完整 attention 参数 → Dispatch All-to-All (NCCL) → Expert FFN 计算 → Combine All-to-All (NCCL) → 序列重构 → 下一 block 的 attention
  - **编译框架层**：论文未明确说明（PyTorch eager execution + NCCL 通信原语）
  - **Kernel/运行时调度层**：All-to-All dispatch/combine 以大张量形式一次发射 → 通信期间 GPU SM 大量空闲 → Expert FFN 使用标准 cuBLAS GEMM → 无通信-计算重叠
  - **硬件架构层**：16× V100 GPU (16GB)，PCIe 互联（无 NVLink）→ PCIe 带宽瓶颈放大 all-to-all 通信延迟
  
  **Baseline 的核心缺陷**：
  1. **All-to-All 通信是系统瓶颈**：dispatch 和 combine 两次 all-to-all 导致大量跨 GPU token 传输，通信时间占 batch training time 的 18.1%-47.5%，且随 expert 数增加而恶化
  2. **Expert Transfer 方案牺牲并行度**：移动 expert 替代移动 token 可减少网络流量，但多 expert 共享同一 GPU 导致资源竞争，computation time 增长 1.88×（MoE-BERT-Large 3 experts/GPU），且随 expert 数增加专家传输本身也成为开销
  3. **现有方案忽略 token 冗余**：被路由到同一 expert 的大量 token 高度相似（MoE-TransformerXL 中约 62% 的 token 对相似度 >0.75），但现有系统无条件传输所有 token
  4. **Combine Phase 的通信路径未被优化**：所有 token 必须拉回原 GPU 重构序列，即使某序列的大部分 token 在另一 GPU 被 expert 处理
  5. **Attention 计算效率被忽略**：现有工作过度关注 expert 通信，但 attention 是 MoE 中最 compute-intensive 的组件，序列长度不均导致的 padding zeros 浪费 GPU 计算和内存

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：LUFFY = Sequence Migration（序列迁移）+ Token Condensation（令牌凝聚），两个正交技术分别优化 combine phase 和 dispatch phase 的通信效率，同时保持最大 expert parallelism（不移动 expert 参数）。
  
  **Defect → Design 映射**：
  
  | Baseline 缺陷 | LUFFY 设计选择 | 解决机制 |
  |---|---|---|
  | All-to-All dispatch 通信量大 | Token Condensation: 识别并凝聚相似 token，消除冗余传输 | 约 62% 的相似 token 可被凝聚 → dispatch 通信量大幅减少 |
  | All-to-All combine 通信量大 | Sequence Migration: 将序列迁移到其大部分 token 被处理的 GPU 上重构 | combine 的跨 GPU token 拉取路径被隐藏为 intra-GPU 路径 |
  | Expert Transfer 牺牲并行度 | 禁止 expert 移动，通过 token 级优化减少通信 | Expert parallelism 始终保持最大（每 GPU 固定持有 expert） |
  | 忽略 token 冗余 | Fast Similarity Measurement: 三步法快速识别相似 token（expert activation filter + historical lookup + cosine） | 大部分 token 对通过 O(1) 查找直接判断，仅少量需 real cosine 计算 |
  | Attention 计算低效 | Sequence Migration 同时优化 attention: 将相似长度序列聚集到同一 GPU | 减少 padding zeros → GPU 内存节省 + attention 计算加速 |
  | 固定阈值破坏收敛 | Adaptive Token Condensation: 根据 loss 下降动态调整阈值 h_t | 训练早期保留更多 token（h_t 大），训练后期可凝聚更多（h_t 小） |

  **论文方法全栈执行例子（以 LUFFY, 4 GPU, MoE-TransformerXL 训练一个 batch 为例）**：
  
  - **算法层**：
    1. Attention 计算（各 GPU 本地执行已分配的 sequences）
    2. Token Condensation: attention 输出 → DGL 图构建 → Fast Similarity Measurement（三步法）→ 自适应阈值 h_t 剪枝 → 连通分量凝聚 → 仅 representative tokens 进入 dispatch
    3. Expert Computation: 各 GPU 对收到的 condensed tokens 执行 FFN 计算（token 少 → 计算少）
    4. Sequence Migration: Controller 收集 token_to_gpu 分布 → Algorithm 1 决策每个 sequence 的重构 GPU（最小化 combine 流量 + 优化 attention batch 效率）→ 迁移决策分发
    5. Combine: 根据迁移决策将 token 路由到目标 GPU 重构序列
    6. 下一 Block Attention: 相似长度 sequences 在同一 GPU → padding 最小化
  
  - **系统框架层**：PyTorch + ~4.5K 行自定义代码 → plug-and-play 插件 → Sequence Migration Controller (集中式决策) + Token Condensation Scheduler (每 GPU 独立 CUDA stream) + 三张哈希表 (token_to_sequence, token_to_gpu, sequence_to_gpu) 管理路由状态
  
  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution）
  
  - **Kernel/运行时调度层**：Token Condensation Scheduler 在独立 CUDA stream 上与 expert computation 并行执行 → DGL 图操作 GPU 加速 → `torch.distributed.rpc` 指导 combine phase 的 token 交换路线 → Cost model T_att(B,L) 在线估算 attention 时间（平均误差 ~5%）
  
  - **硬件架构层**：16× V100 GPU (16GB) PCIe 互联 → 通信减少后 PCIe 瓶颈缓解 → 通信时间从 36.6%-47.5% 显著下降 → Computation speedup 1.16×-1.57×（因 expert 计算量减少和 attention batch 优化）
  
  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Vanilla Expert Parallelism):
  Attention → [All-to-All Dispatch 全量 tokens] → Expert FFN 
  → [All-to-All Combine 全量 tokens 回原 GPU] → Next Attention
  通信瓶颈: 18.1%-47.5% iteration time

  LUFFY:
  Attention → [Token Condensation: 凝聚相似 token] 
  → [All-to-All Dispatch 仅 representative tokens]
  → Expert FFN (更少 token → 计算减少)
  → [Sequence Migration: 决策 combine 目标 GPU] 
  → [All-to-All Combine 减少跨 GPU 拉取]
  → Next Attention (相似长度 sequences batch → padding 减少)
  通信加速: 1.76×-3.72×, 计算加速: 1.16×-1.57×
  ```

  **关键设计决策对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | 不移动 expert（保持 expert parallelism） | Expert Transfer 的资源竞争 → computation 增长 1.88× | LUFFY computation 反降为 1.16×-1.57× speedup |
  | Sequence Migration (非 Expert Transfer) | combine 流量 + attention batch 效率 | MoE-GPT2 sequence migration 单独贡献 1.72× speedup |
  | Token Condensation (非单纯调度) | dispatch 冗余传输 | MoE-TransformerXL token condensation 单独贡献 1.74× speedup |
  | Fast Similarity Measurement (三步法) | naive pairwise 计算不可行 | 大部分 token 对通过 O(1) lookup 直接判断 |
  | Adaptive Threshold h_t | 固定阈值破坏收敛 | h=0.3 → F1 从 90.82 降至 85.41; LUFFY adaptive → 89.17 |
  | Cost Model T_att(B,L) | 迁移决策需准确估算 attention 时间 | 平均估计误差 ~5% |
  | 联合优化 Communication + Computation | 现有工作仅关注通信 | LUFFY 2.73× speedup vs Vanilla (16 experts) |
