# idea库

## Continual Pre-training of MoEs How robust is your router

- baseline方法是什么？
  Baseline 包含两层：(1) **密集 Transformer CPT**：使用 replay + LR re-warming + re-decaying 对 dense decoder-only transformer 进行持续预训练（Ibrahim et al., 2024），从衰减 checkpoint 恢复后用 Cosine Annealing 重新 warming + decaying；(2) **Full Re-training**：将 FineWeb 和下游数据（Stack/German）混合后从头完整训练。Baseline 在 MoE 场景下的核心缺陷：没有针对 MoE 路由算法的 CPT 行为分析——不清楚路由算法在分布偏移下是否会加剧遗忘、是否能维持负载均衡、现有的 dense CPT 策略（replay + LR re-warming/decaying）对 MoE 是否同样有效。

  **Baseline 全栈执行例子（以 570M dense transformer + 64×A100, FineWeb→German CPT 为例）**：
  - **算法层**：输入 batch 含 1024 个 sequence（seq_len=2048），通过 24 层 Llama3-style decoder-only transformer，每层经过 Multi-Head Self-Attention (16 heads) + GEGLU FFN（intermediate=2816→output=1024）。CPT 时从衰减 checkpoint (η=3e-5) 开始，用 Cosine Annealing re-warm 到 η_max=3e-4 再 decay 到 η_min=3e-5。40% replay：每 batch 中 410 samples 来自 FineWeb，614 samples 来自 German CC。
  - **系统框架层**：基于 GPT-NeoX 训练框架，64×A100 GPU，data parallel + ZeRO-1。ZeRO-1 将 optimizer states (AdamW m, v) 分片到 64 张 GPU，每张 GPU 持有 1/64 的 optimizer states。前向：每张 GPU 独立计算 full batch slice 的前向 loss。反向：每张 GPU 计算梯度后 AllReduce 聚合，然后各 GPU 用本地 optimizer state 分片更新参数，再 AllGather 参数。
  - **Kernel层**：标准 PyTorch 操作，包括 cuBLAS GEMM (FFN MatMul)、Flash Attention (self-attention)、LayerNorm、GeLU 激活。论文未明确说明 kernel 细节。
  - **硬件层**：64×NVIDIA A100 GPU（80GB 或 40GB SXM），NVLink + NVSwitch 互联，每步耗时约 880ms（dense），MFU 约 111 TFLOPs。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文首次系统研究 MoE CPT 的完整行为，通过三项关键贡献解决 baseline 在 MoE 场景下的未知问题：

  **(1) 构建四种 MoE 架构的 CPT 实验矩阵 → 填补 MoE CPT 知识空白**
  - 系统对比了当前 SOTA 的两种路由算法（PBTk: z-loss + aux loss; SBTk: Sinkhorn-Knopp 迭代近似线性分配问题）和两种架构（Granular: 31 routed + 1 shared, K=3; Switch: 8 routed, K=1, full FFN）
  - 训练规模：570M active / 2B total，各训练 600B tokens（400B FineWeb + 200B Stack/German），严格 overtraining regime (>10× Chinchilla optimal)
  - 所有 MoE 与 FLOP-matched dense baseline 对比，确保公平

  **(2) 证明 MoE 路由算法对分布偏移具有"令人惊讶的鲁棒性" → 消除对 CPT 破坏路由负载均衡的担忧**
  - SBTk MoEs 在分布偏移时 MRI 几乎不变（因为显式的 Sinkhorn balancing 步骤强制均衡）
  - PBTk MoEs 经历短暂的 MRI spike（分布偏移后），但在 500 steps 内恢复至比 SBTk 更低的 MRI 水平
  - 两种路由算法最终的负载均衡均优于或等于 full re-training baseline
  - 提出 Maximum Routing Imbalance (MRI) 指标作为 MoE 推理延迟的代理：MRI(t,j) = max_i [∑_x 𝟙{i∈I_k(x)} / |B|]，独立于具体硬件部署

  **(3) 建立 MoE CPT 最佳实践 → CPT MoE 可匹配 full re-training 性能且大幅降低成本**
  - Infinite LR schedule (CosineInf) + replay 是 MoE CPT 的最佳组合：在 FineWeb→Stack (30% replay) 和 FineWeb→German (40% replay) 两个任务上，CPT MoE 的验证 loss、English/German/Code benchmark 均匹配或超越 full re-training baseline
  - CPT 仅消耗 full re-training 约 1/3 的计算成本
  - MoE 在 CPT 期间保持对 FLOP-matched dense 的 sample efficiency 优势
  - 路由行为分析发现：CPT 期间路由决策变化主要发生在早期层 (layers 0-2) 和后期层 (layers 13-23)；0% replay 的 checkpoint 在早期层变化最大且遗忘最多，说明早期层的剧烈路由变化与遗忘相关

  **论文方法全栈执行例子（以 Granular PBTk MoE, FineWeb→German, 40% replay + CosineInf 为例）**：
  - **算法层**：输入的 1024 个 sequence（2048 tokens/seq = 2,097,152 tokens/batch）进入 24 层 MoE transformer。每层 MoE block：Self-Attention → Router (W_r: 1024→31, linear projection + softmax) → Top-3 expert selection → shared expert (GEGLU, intermediate=704) + 3 selected experts (GEGLU, intermediate=704) 的加权组合。Router 输出通过 z-loss (coeff=0.001) 和 aux loss (coeff=0.01) 惩罚大 logit 和负载不均衡。CPT 从 CosineInf schedule 的 η_const=1.65e-4 平滑过渡继续训练 95,370 steps。Replay: 每 batch 410 FineWeb + 614 German CC tokens。
  - **系统框架层**：GPT-NeoX + Megablocks grouped GEMM kernel。Megablocks 将同一 batch 中不同 token 被路由到不同 expert 的 FFN 计算打包为单次 grouped GEMM：将 batch 中所有 tokens 按 target expert 分组，同一 expert 的 tokens 拼接为连续矩阵块，一次性完成 batched MatMul，避免逐个 expert 的小矩阵乘法开销。
  - **Kernel层**：论文未明确说明 Megablocks 的详细 kernel 参数，但 grouped_gemm 仓库提供 CUDA kernel 实现。Megablocks 的基本原理：将稀疏 MoE 的多个 expert FFN 的 GEMM 操作融合为一次 grouped GEMM，减少 kernel launch overhead 和内存碎片。
  - **硬件层**：64×A100，data parallel + ZeRO-1。Granular MoE 每步约 1680ms（dense 的 ~2×），forward ~485ms，backward ~1091ms，MFU 约 78 TFLOPs。Sinkhorn 版本因迭代求解额外增加约 110ms/步。

## Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

- baseline方法是什么？
  Baseline 是标准 MoE 架构下的 **Top-1 Token-Choice Routing (TCR)**，使用 capacity factor 1.1 约束 expert 容量（防止 token drop），搭配 auxiliary loss 做负载均衡。以 Mixtral 8×7B 为 backbone。Baseline 存在三个缺陷：(1) **路由效率瓶颈**：仅由 token 单向选择 expert（TCR），无法保证 expert 获得最适合处理的 token；训练后期 class-irrelevant token 呈各向同性分布时，TCR 成功率受限于 C/s（C 为容量，s 为 token 数）；(2) **Expert 同质化**：TCR 的 top-k softmax routing 无法主动维持 expert 间的专业化分工，导致多 expert 学到相似表征，冗余计算；(3) **All-to-All 通信气泡**：固定容量策略导致部分 expert 过载而其他空闲，padding 浪费计算和通信资源。

  **Baseline 全栈执行例子（以 Mixtral 8×7B + 32 Ascend NPU 下单个 MoE block 的一次训练前向为例）**：
  - **算法层**：输入 batch 包含 s 个 token（seq_len=32768），传统 MLP Router 计算 gate_logits = Softmax(W_g · x_t + ε)，W_g ∈ R^{d×n}（全参数矩阵，O(d²) 复杂度）。Top-1 选择最高 logit 的 expert 分配 token，capacity_factor=1.1 意味着每个 expert 容量上限 C = 1.1 × s/n。超出容量的 token 被 drop（残差连接直通）。auxiliary loss L_aux 惩罚 expert 使用不均。训练过程经历 Phase 1 (Router training，expert 学习接收对应类别 token) 和 Phase 2 (Expert training，expert 学习解决问题)，但两个阶段用同一 TCR 策略。
  - **系统框架层**：基于华为 MindSpeed-LLM（Megatron-LM 风格）训练框架，在 Ascend NPU 上运行。32N 配置：TP=4（张量切分 attention/FFN 权重）、PP=4（流水线切分 32 层 MoE block）、DP=2（数据并行）、EP=2（expert 并行，每张 NPU 持有 4 个 expert）。每步训练中的 All-to-All 通信完成 token 在 EP 维度上的重新分发和聚合——token dispatch（send）将 token 发送至持有目标 expert 的 NPU，token combine（recv）将 expert 输出聚合回原 NPU。
  - **编译框架/Kernel 层**：论文未详细说明编译器栈。CoC（Communication Over Computation）优化前，矩阵乘法和集合通信操作串行执行——即先完成 FFN 的 MatMul 计算，再发起 All-to-All 通信。Ascend 的 AI CORE 负责矩阵乘法/卷积，AI VECTOR CORE 负责向量并行计算，AI CPU 负责专用指令。论文未明确说明 kernel 编译细节。
  - **硬件层**：Ascend 910B3 NPU，单颗 20 AI Cores @ 1.8GHz（fp16 313T 算力），HBM 64GB @ 1.6GHz（1.6T 带宽）。8 颗 NPU 组成 Atlas 800T A2 服务器全 mesh 互联。服务器间通过 NPU 直连网络互联。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **ETR（Expert-Token Resonance）**，通过三方面设计解决 baseline 缺陷：

  **(1) GrAP 路由层替代 MLP Router → 解决计算复杂度和 expert 同质化**
  - 将传统全连接 Router W_g ∈ R^{d×n} 替换为 Grouped Average Pooling 生成的**对角稀疏亲和力矩阵** W_aff ∈ R^{d×n}（非零元仅在 d/n 分组内）
  - 参数量降至 1/n，计算 O(d²/n) vs 传统 O(d²)
  - 正交性天然确保各 expert 对应的 w_i 向量互相正交，防止多个 expert 学到相似路由模式
  - 使用**余弦相似度**（非 Softmax logit）作为亲和力分数 δ_{t,i} = cos(x_t, w_i)

  **(2) TCR+ECR 双向选择 → 解决路由质量和 expert 专业化**
  - Step 1 (TCR)：每个 token 按 δ 选 top-ℓ experts
  - Step 2 (ECR)：每个 expert 从已分配 token 中按 δ 选 top-C tokens
  - Expert 主动选择最相关 token，"共振效应"加速 expert 专业化
  - 理论证明（Theorem 5）：早期训练 TCR 更优（成功率 Θ(C·∑p_i/s)），后期 ECR 更优（成功率 → 1 当 C ≥ 2C*）；双向动态协调最大化全训练过程的成功率

  **(3) 自适应容量 + Locality Loss → 解决通信气泡和负载不均**
  - 容量 C 动态计算：C_min = (1/n)·exp(d·δ_max²/(2−δ_max²))，随训练进度自适应降低（最大降 40%）
  - Locality Loss L_loc = μ·KL(D_c||D_l)：鼓励 token 路由至本地 NPU 的 expert，减少跨节点 All-to-All 通信

  **ETR 全栈执行例子（以 Mixtral 8×7B + 32 Ascend NPU 下单个 MoE block 的一次训练前向为例）**：
  - **算法层**：输入 batch s 个 token，GrAP 生成 W_aff（对角稀疏，仅 d 个参数项，循环移位分组平均池化）。对每个 token t 和 expert i 计算 cos(x_t, w_i) 得亲和力矩阵 δ ∈ R^{s×n}。TCR 阶段：每个 token 取 top-ℓ experts（论文中 ℓ 由亲和力分数和阈值确定）。ECR 阶段：每个 expert i 从分配给它的 token 中 Bottom-C（保留最高亲和力的 C 个），动态 C = max(C_min, adaptive_by_progress)。仅 C 个最高亲和力 token 进入 expert FFN 计算。总 loss = task_loss + α·L_aux + β·L_loc（L_loc = μ·KL(current_dist || local_dist)，local_dist 偏向同节点 expert）。
  - **系统框架层**：基于 MindSpeed-LLM，增加双向路由模块（token dispatch 后并行执行 expert 侧 token 过滤）。Locality Loss 通过感知 EP 拓扑（哪些 expert 在同节点）计算 KL 散度。CoC 优化将 MatMul 和 All-to-All 通信融合为统一 kernel，通过 Ascend MTE（Memory Transfer Engine）的远程内存访问实现流水线并行——计算当前 batch 时预取下一 batch 的通信数据。Token rearrangement 引入的 TopK/IndexPutV2 操作有少量开销，但 FFN MatMul 实测达 17× 加速（相比 baseline）、2.6× 相比 LocMoE。
  - **编译框架/Kernel 层**：CoC 优化在 Ascend CANN（Compute Architecture for Neural Networks）编译框架层面将 MatMul + All-to-All 融合——论文未详细说明 CANN 版本或融合策略细节。Ascend AI CORE 执行 FFN MatMul（Cube 计算单元），AI VECTOR CORE 执行亲和力分数的 cosine 计算和 TopK/BottomC，AI CPU 执行 token rearrangement 的 IndexPutV2 等控制流操作。论文未明确说明 kernel 调度策略细节。
  - **硬件层**：同 baseline 的 Ascend 910B3 NPU 集群。但 locality loss 减少跨服务器通信量（token 优先本地 expert），通信 idle 时间占比下降（见图 5 的 3D 柱状图对比），显存峰值降低 4.57%-16.27%（因容量自适应减少 padding）。

- baseline方法是什么？
  Baseline 是 MoE-LLaVA [25] 提出的 **MoE-tuning** 三阶段方法：Pretraining（MLP Projector 对齐视觉和语言模态）→ Stage II MoE-tuning（复制 FFN 参数初始化多个 expert + 训练线性 router 做 top-k 选择）→ Stage III Instruction Tuning。Baseline 存在两个核心缺陷：(1) **Expert Uniformity**：通过复制（replication）初始化 MoE expert，导致 expert 趋同，失去 MoE 架构的多样化优势——实验证明 shuffle router 几乎不影响性能，说明 expert 之间没有实质差异；(2) **Router Rigidity**：使用静态线性 router 对所有 token 做统一路由，无法区分视觉 token 和文本 token 的差异（KDE 密度图显示两种模态的 logit 分布高度重叠），导致 router 输出对输入模态不敏感。

  **Baseline 全栈执行例子（以 MoE-LLaVA + Qwen-1.8B 单个 token 的前向推理为例）**：
  - **算法层**：视觉 token 经 CLIP-L 编码后通过 MLP Projector 映射到 LLM hidden space。每个 LLM decoder layer 的 FFN 被替换为 4 个 MoE expert（由原始 FFN 复制初始化），线性 router R 对每个 token 计算 logits = W_r · h，Softmax 后选 top-2 experts 做加权求和输出。训练时加载均衡辅助损失 L_aux 惩罚 expert 间 token 分配不均。
  - **系统框架层**：基于 LLaVA 1.5 代码框架实现（PyTorch + HuggingFace Transformers），使用 DeepSpeed ZeRO-2 做分布式训练。FFN 复制 + MoE 层替换在模型初始化阶段完成，不涉及运行时框架修改。
  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution，无自定义编译器 pass）。
  - **Kernel 调度层**：MoE FFN 的 expert 间计算为标准的串行循环或 batch-gemm，单个 token 的 top-2 expert 结果由加权 sum 合并。无自定义 CUDA kernel。通信层面 expert parallelism 使用标准 all-to-all 或 all-gather（论文未详述）。
  - **硬件架构层**：8x NVIDIA A100-80G GPU，无特定硬件加速。Expert 和 router 均在 GPU 通用计算单元上执行。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  EvoMoE 在 MoE-tuning 的 Stage II/III 分别引入两项算法创新对应解决两个缺陷：

  **(1) Expert Evolution → 解决 Expert Uniformity**

  将"复制初始化"改为"进化初始化"：仅训练一个 FFN（Expert 1），其他 expert 通过 EMA 形式从 Expert 1 的参数和梯度演化而来：
  ```
  θ_n ← β_n · θ_1 + (1 - β_n) · ∇θ_1   (n = 2, 3, 4)
  ```
  β_n 随机采样自不同范围（[0.9,0.99]、[0.8,0.89]、[0.7,0.79]），每个 expert 以不同速率"吸收"梯度信息，从而自然产生功能分化。对比：复制初始化的所有 expert 有相同的起点和几乎相同的梯度轨迹 → 趋同；进化初始化的 expert 因 β 不同而拥有不同的参数方向 → 多样。

  效果验证：独立评估每个演化后的 expert（不用 router）发现 Expert 2/3/4 在多个 benchmark 上一致优于 Expert 1（原 FFN），甚至 β=0.9（仅保留 10% 梯度更新）也能优于 Expert 1，证明演化产生的多样性是有效而非随机的。

  **(2) Dynamic Token-aware Router (DTR) → 解决 Router Rigidity**

  将"静态线性 router"替换为"hypernetwork 驱动的动态 router"：
  ```
  z' = MSA(LN(z_prev)) + z_prev
  Θ_up^τ, Θ_down^τ = H^τ(z')            # 两个 hypernetwork H_V / H_T
  E^τ = Θ_up^τ · SwiGLU(Θ_down^τ · z')   # 动态生成的 up/down 投影
  ρ^τ = φ(E^τ)                            # 最终 linear 层预测 expert 分布
  ```
  关键设计：视觉 token 和文本 token 分别通过不同的 hypernetwork 生成专属投影参数，确保：(a) 模态感知——视觉 token 经 H_V 路由，文本 token 经 H_T 路由；(b) token 级自适应——每个 token 拥有独立的路由计算权重，而非 shared linear layer。可视化表明 DTR 的 expert 分配在不同模态间有明显差异（MoE-tuning baseline 则几乎均匀），实现了"让 visual expert 处理图像、text expert 处理文本"的功能分化。

  **EvoMoE 全栈执行例子（与 baseline 同一 token 的前向推理对比）**：
  - **算法层**：与 baseline 相同流经 CLIP-L + MLP Projector → LLM decoder layers，但在每个 MoE layer：(a) expert 参数由 Expert Evolution 生成（4 个多样化 FFN 而非 4 个近似相同的复制 FFN）；(b) router 由 DTR 替代线性 router：H_V 或 H_T 动态生成投影矩阵 → SwiGLU → φ 输出 expert 概率 → 仅 top-1 expert 被激活（比 baseline 的 top-2 少一半激活参数）。Stage III 仅训练 H_V / H_T / φ（共约额外 34760 参数），experts 冻结。
  - **系统框架层**：同 baseline（LLaVA 1.5 + DeepSpeed ZeRO-2），但在模型定义中将 FFN 层替换为 EvoMoE 层（含 4 个演化 expert + DTR）。Stage II 仅 Expert 1 需要 optimizer state 和梯度，Stage III 仅 DTR 参数需要 optimizer state——训练参数量比 baseline 更少。
  - **编译框架层**：论文未明确说明（与 baseline 相同的 PyTorch eager execution）。
  - **Kernel 调度层**：论文未明确说明（与 baseline 相同的标准 expert computation，无自定义 kernel）。
  - **硬件架构层**：同 baseline（8x A100-80G）。论文未提出硬件层面的修改。

## Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

- baseline方法是什么？
  Baseline 是传统的大规模 MoE 模型训练范式：依赖高端 GPU（H100/H800）集群进行同步分布式训练（All-Reduce），所有计算在同类高端加速器上执行，存储和 I/O 使用传统并行文件系统（如 GPFS）。

  **Baseline 全栈执行例子（以 Ling-Plus 290B MoE 单步训练为例）**：
  - **算法层**：标准 MoE 架构，top-k token-to-expert routing，所有 expert 等容量设计。训练初期 router 随机初始化导致某些 expert 严重过载或闲置——早期 expert 负载崩溃是最常见的训练失败模式。Loss spike 发生时无自动处理机制，可能导致 wide spike 持续多个 step 使 benchmark 降到随机水平。
  - **系统框架层**：Megatron-LM / DeepSpeed 同步 All-Reduce 分布式训练。所有 worker 完成整个 step 后 barrier 同步——慢节点（straggler）瓶颈所有节点。checkpoint 写操作：Megatron 默认所有 DP group 的 rank_0 负责 checkpoint 数据聚合和写入，导致这些 rank_0 集中到少数物理节点，CPU 和网络带宽竞争激烈。
  - **编译框架层**：论文未明确说明（使用标准 PyTorch + CUDA 编译路径，依赖框架搭建）。
  - **Kernel 调度层**：同步 All-Reduce 每 step 做全局通信 barrier，无法重叠通信与计算。算子和通信 kernel（group_gemm、permute/unpermute、all2all、expert parallelism）在不同加速器平台实现不一致。无有效的 kernel 级性能诊断工具——需要全量监控（NVTX profiler）消耗大量内存而难以长期在生产环境使用。
  - **硬件架构层**：高端 H100/H800 GPU，NVLINK/NVSwitch 互联。设备类型单一。训练成本极高：Ling-Plus 在 Device D（989 TFLOPS）上训练 1T tokens 需约 635 万 RMB。

  **Baseline 核心痛点**：
  1. 高端 GPU 供需严重失衡——商业部署高峰期高端 GPU 被抢占用于在线推理，研发团队面临长期短缺
  2. 异构设备间计算/通信算子实现不一致（group_gemm, all2all, permute/unpermute），跨平台迁移导致精度累积偏差
  3. StoCworker 问题导致大规模同步训练效率剧烈下降（1000+ 节点时 baseline 速度降至 5.49e-2 step/s）
  4. MoE 训练早期极其不稳定——expert 负载崩溃、loss spike/divergence
  5. 跨集群存储同步慢（PB 级数据 OSS List 需 >6h）、checkpoint I/O 瓶颈（rank_0 集中写入）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出系统性方案在**低规格异构硬件**上完成 300B MoE 模型训练，核心是 5 个系统层级优化：

  1. **模型架构层面的 MoE 稳定性**：Fine-Grained Experts + Shared Expert 设计提升专家专业化的同时保持通用能力；Stochastic Routing Warmup 在训练早期引入受控随机路由噪声防止 expert 崩溃；NormHead 通过 L2 归一化 LM-Head 权重抑制 loss divergence；Skip Loss Spikes + Sample Retry 自动检测/跳过/重试 spike 步。
  2. **EDiT 异步分布式训练**：layer-wise sync + pseudo gradient penalty + time-based sync 替代同步 All-Reduce，解决 straggler 问题，最大加速 66.1%。
  3. **XPUTimer 轻量诊断**：selective tracing + async CUDA event + 数据压缩，90% 内存节省，O(1) 快速定位。
  4. **PCache + Babel 存储优化**：FUSE+shm 消除用户/内核态切换；AI co-design 分散 DP rank_0 写入（checkpoint 写延迟 -50%，峰值内存 -60%）；并行 metadata prefetch 加速跨集群同步 36×。
  5. **Flood 纯 PP 离线推理**：放弃 TP 避免低互联带宽下的通信开销，Segment Cache 替代 PageAttention。

  **论文方法全栈执行例子（以 Ling-Plus 290B MoE 单步训练为例）**：
  - **算法层**：Fine-grained experts (N 个低维 expert) + Shared expert（所有 token 不经路由全量通过）。Stochastic Routing Warmup：step i 时 ŝ_t = (i/W)·s_t + (1-i/W)·(μ_s+σ_s·ε)，ε~N(0,I)。NormHead：h_o = W_lm_head/||W_lm_head||₂ · h。Load balance loss (α=0.015) + router z-loss (α=1e-4)。训练阶段：初始预训练 9T tokens (4K ctx) → 长上下文 150B tokens (16K ctx, RoPE θ 10K→600K) → 退火（inverse sqrt decay lr 1.2e-4→1.2e-8）。Loss spike 发生 → 跳过当前 update → 保存数据 → 随机重注入后续 batch → 持续 spike 则 lr *= decay_factor。
  - **系统框架层**：DLRover 统一管理 DeepSpeed/Megatron-LM/Megatron vendor version 跨平台部署。EDiT 实现：每 worker 逐层 forward→backward→逐层 sync（非全局 barrier）。Pseudo gradient penalty 三步：(i) EMA 追踪 pseudo_grad 检测异常 worker→排除；(ii) 剩余 worker 按 pseudo_grad norm 加权平均；(iii) 统一梯度裁剪。Time-based sync：到达时间阈值而非固定步数触发同步。checkpoint 写入：PCache 将 DP group 写入任务 round-robin 分配到不同物理节点避免集中竞争。
  - **编译框架层**：论文未明确说明。使用 FlagScale 等开源分布式训练框架，针对不同加速器平台做底层算子一致性验证（matmul、linear、Attention、MLP、Router）。
  - **Kernel 调度层**：XPUTimer 运行时轻量追踪——Python 层通过 TRACED_PYTHON_API 环境变量动态拦截 API；C++/CUDA 层框架无关 kernel 监控（cuBLAS, Flash Attention, NCCL, 自定义算子）；CUDA event pool 复用 + 异步后台线程日志 + 数据压缩（仅记录时间戳和 kernel 输入 layout，~1.5MB/加速器/step）。EDiT 中通信与计算重叠：layer-wise sync 的 prefetch 机制在 backward 时同步下一层权重。PCache 使用 FUSE + shm 实现用户-内核态零拷贝写入 NVMe SSD。
  - **硬件架构层**：五种异构加速器混合训练（Device A 370 TFLOPS 64GB 无 FP8 → Device D 989 TFLOPS 80GB FP8 → Device E 147 TFLOPS 96GB FP8）。通过跨平台操作一致性验证（matmul, linear, attention, MLP, router forward → backward alignment）确保不同硬件上训练精度收敛一致——即便单个操作精度差异微小，累积后也会导致 loss 收敛巨大偏差，因此必须逐 operator 和逐 framework module 进行 forward+backward 完整对齐。节省约 20% 计算成本。

- baseline方法是什么？
  Baseline 是 vLLM 中的 TP+TP（Attention TP + MoE TP）并行策略，使用 cutlass GroupGemm 作为 MoE FFN 的默认 GEMM 实现，ncclAllReduce 作为通信原语。

  **Baseline 全栈执行例子（以 Mixtral 8x7B 单个 token 推理为例）**：
  - **算法层**：MoE gating 选择 top-k experts，FFN 包含 Gate/Up/Down 三个 GEMM
  - **系统框架层**：vLLM TP+TP 模式——Attention 和 MoE 权重均在 D 个设备上 TP 切分，所有 token 在所有设备上存在
  - **编译框架层**：论文未明确说明（使用标准 PyTorch + CUDA 编译路径）
  - **Kernel 调度层**：cutlass GroupGemm 单次 kernel launch 处理所有 expert 的 GEMM（所有 expert 共享一个 kernel grid）。ncclAllReduce 进行 TP 通信，每层 2×(D-1)×P/D 通信量。GEMM 和通信串行执行，无重叠。GEMM kernel 独占所有 132 SM。
  - **硬件架构层**：NVIDIA H800 SXM GPU，NVLink 互联。GEMM 计算在 Tensor Cores，all2all 通信经 NVLink。SM 全部分配给 GroupGemm，通信在 GEMM 完成后才启动，导致 SM 资源在通信期间空闲。

  **Baseline 核心痛点**：
  1. TP+TP 通信量最大（V_{TP+TP} > V_{TP+EP} > V_{DP+EP}），且 ncclAllReduce 不拆解，无法与 GEMM 并行
  2. GroupGemm 在大输入规模（prefill 阶段 m≥4096）效率低于 DenseGemm，但 baseline 固定使用 GroupGemm
  3. GEMM 和 all2all 串行执行，GPU SM 资源在通信阶段闲置，资源利用率低

- 论文方法是什么？如何对应解决Baseline的缺陷？
  EPS-MoE 提出三模块组合方案：
  1. **并行策略重设计**：MoE 块从 TP 切换到 EP，Attention 块保持 TP（MHA/GQA）或 DP（MLA）。将 ncclAllReduce 拆解为 ReduceScatter+all2all（dispatch）和 all2all+AllGather（combine），通信量从 V_{TP+TP} 降至 V_{TP+EP}。
  2. **Expert Pipeline Scheduler**：水平切分输入（按行），权重按专家切分。每次只传输当前专家组所需的 token，将 GroupGemm 的 group 数从 E 降至 E/N。当 N=E 时 GroupGemm 退化为 DenseGemm。根据负载（m 大小）动态选择 GEMM 类型。
  3. **SM 控制的计算-通信重叠**：限制 GEMM 的 SM 数（如 116 SM），留出 SM 给通信 kernel（16 SM），使 GEMM 计算与 all2all 通信在不同 SM 上并行执行。

  **论文方法全栈执行例子（以 Mixtral 8x7B 单个 token 推理为例）**：
  - **算法层**：同 baseline，MoE gating 不变
  - **系统框架层**：vLLM TP+EP 模式——Attention 权重 TP 切分，MoE 专家按 EP 分布。每个 token 经 router 确定 top-k experts 后，只将 token 发送到对应专家所在的设备（all2all），而非广播到所有设备
  - **编译框架层**：论文未明确说明（cutlass/cublas 库调用）
  - **Kernel 调度层**：
    - 输入按行水平切分 N 组（N=PN），权重按专家切分
    - Dispatch 阶段：ncclReduceScatter + all2all，token 只传输到目标专家设备
    - 第 1 组 token all2all 完成后启动第 1 组专家的 GEMM（根据 m/N 大小选 GroupGemm 或 DenseGemm），同时第 2 组 token 开始 all2all
    - GEMM 限制 116 SM，通信占用 16 SM，两者在不同 SM 上并行
    - Combine 阶段：all2all + ncclAllGather 聚合结果
    - ⚡ 关键改进：GEMM 类型自适应切换（小 m → GroupGemm，大 m → DenseGemm），计算与通信在 SM 级并行
  - **硬件架构层**：同 NVIDIA H800 SXM GPU，SM 资源被划分为计算区（116 SM）和通信区（16 SM），两者同时工作，消除通信阶段的 SM 空闲

  **痛点映射**：
  | Baseline 痛点 | EPS-MoE 解决方案 |
  |---|---|
  | TP+TP 通信量最大 | TP+EP / DP+EP 减少通信量，all2all 替代 AllReduce |
  | GroupGemm 在大 load 下效率低 | 水平切分 + 按专家切权重，m 大时自动切换 DenseGemm（cublas） |
  | GEMM 和通信串行，SM 闲置 | SM 控制 + pipeline 重叠，GEMM 和 all2all 在不同 SM 上并行 |
  | 固定调度策略忽略负载特性 | Load-aware 自适应调度，根据 m 动态选择 GEMM 类型和 pipeline 数 PN |

  实验效果：DeepSeekV2 prefill throughput 从 100K 提升至 121.8K tokens/s (+21.8%)；Mixtral 8x7B TTFT 最多降低 24.3%；DBRX TTFT 最多降低 30.5%。

## EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

- baseline方法是什么？
  Baseline 是**直接对 MoE-LLM 使用 dense LLM 的标准量化和剪枝方法**，即：(1) **GPTQ 均匀位宽量化**：对所有 expert 施加相同位宽量化（2/3-bit），不考虑 MoE 路由器的 expert selection 特性，导致量化后路由器选错 expert（expert-shift 问题），模型精度严重退化。(2) **静态混合精度量化（PMQ/BSP）**：基于校准集统计 expert 选择频率分配不同位宽，但忽略了不同任务类别中 expert 重要性截然不同的规律，导致严重的跨任务过拟合。(3) **逐 token 动态剪枝（EES/ODP）**：对每个 token 剪枝贡献度最小的 expert，但仅减少部分 expert 的输入大小，加速效果有限（~5-8%），且未利用序列级 expert 选择频率的稀疏性。

  Baseline 全栈执行例子（以 Mixtral-8x7B 推理为例）：
  - **算法层**：标准 GPTQ 量化：W_fp16 → W_intB（group-wise asymmetric, 128 groupsize），使用 Hessian-based 误差补偿，但 MoE router 保持量化前权重不变。量化后每个 token 通过 router 选择 top-2 expert，router 输出因量化噪声偏离全精度模型，导致选错 expert（expert-shift）。以 3-bit GPTQ 量化为例，量化本身导致 PPL 从 3.84 升至 4.16，expert-shift 进一步恶化至 4.65。
  - **系统框架层**：HuggingFace Transformers + GPTQ 推理。所有 8 个 expert 的全精度权重加载到 GPU memory（~94GB），显存压力大。推理时每 token 计算 router logits（MatMul）→ Softmax → Top-2 选择 → 2 个 expert FFN 前向传播 → 加权求和。无论 expert 是否被频繁选择，所有 expert 权重均需常驻显存。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明（标准 cuBLAS GEMM 执行 MoE expert 计算）。GPTQ 量化后使用 BitBLAS 处理 INT 权重的混合精度 BLAS 操作。
  - **硬件架构层**：NVIDIA A100 40G GPU（量化）/ RTX 3090（部署）。直接 GPTQ 量化的 Mixtral-8x7B 在 3.03-bit 下仍需 ~19GB 显存，精度损失~3.7%。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **EAC-MoE = QESC + PESF**，从 MoE 模型最核心的 expert selection 机制出发，分别从"确保选对 expert"（量化前校准）和"跳过不重要 expert"（推理中剪枝）两个维度解决 Baseline 缺陷：

  **(1) QESC: Quantization with Expert-Selection Calibration（解决 expert-shift 和跨任务过拟合）**
  Baseline 缺陷：(a) GPTQ 量化只最小化重建误差 ||WX - W_qX||_2²，不感知 expert selection 偏差；(b) PMQ/BSP 用静态校准集分配位宽，跨任务泛化差。

  QESC 设计：
  - 逐层校准：量化每层 MHSA 后，用校准数据前向传播获得该层量化输入 x̂_l，然后用 TopK-MSE Loss 校准该层 router 权重，使量化后的 router 输出尽可能匹配全精度 router 在 top-K 上的输出，防止 expert-shift 逐层累积。
  - TopK-MSE Loss：仅对 top-K 最高概率的 expert 计算 MSE Loss（而非所有 N 个 expert）。图 4 证明 95.9% 的 shifted expert 仍在 top-16 概率内（64 expert 中），而对所有 expert 计算 MSE 会被低概率 expert 的噪声主导。TopK-MSE 让优化聚焦于"更可能被选中的 expert"。
  - 效果：在 3.03-bit 下，Mixtral-8x7B 准确率损失 <0.5%，Deepseek-moe-16b-base 准确率损失 <0.2%，远超 GPTQ/BSP/PMQ。

  **(2) PESF: Pruning based on Expert-Selection Frequency（解决加速效果有限）**
  Baseline 缺陷：EES/ODP 逐 token 剪枝低权重 expert，仅减少部分 expert 输入大小（并非完全跳过 expert 计算），加速比仅 1.05-1.08x。

  PESF 设计：
  - 序列级动态剪枝：收集当前序列所有 token 的 expert 选择统计，若某专家被选中次数 c_i < (l*K/N) * α，则直接跳过该 expert 的全部计算（而非仅减少输入）。
  - 基于 Section 3.3 的核心洞察：同一任务类型内 expert 选择频率高度相似（cosine similarity >0.8），跨任务类型显著不同。因此动态统计能准确反映当前任务的 expert 重要性。
  - 保守策略（α=0.3）：准确率几乎无损，加速 1.08-1.14x，显著优于 EES/ODP。
  - 激进策略（α=0.7）：加速 1.30-1.47x，准确率下降约 1.5%。
  - 限制：仅适用于 prefill 阶段（需要 l 个 token 的统计信息），不适用于逐 token 的 generate 阶段。

  论文方法全栈执行例子（EAC-MoE = QESC 3.03-bit + PESF α=0.3，以 Mixtral-8x7B 推理 512 token 序列为例，batch=4）：
  - **算法层**：
    Layer 0: x_0 → 量化 MHSA (4-bit, 量化权重 W_q^{attn}) → x_0' → Router (FP16 权重, 经 QESC 校准) → Top-2 expert 选择 (expert e_a, e_b)
    → 执行 PESF: 统计本层所有 512 个 token 的 expert 选择, c_i 统计 → 若 c_j < (512*2/8)*0.3 = 38.4, 剪枝 expert j
    → 仅对被保留 expert (如 e_a, e_b) 计算 quantized FFN (3-bit 权重) → 加权求和 → x_1
    ...逐层重复...
    Layer 31: 最终输出 token hidden states。
    Router 在每层都经过 QESC 校准维持正确的 expert 选择，PESF 动态跳过约 10-15% 的 expert 计算。
  - **系统框架层**：BitBLAS 加载量化权重并执行混合精度 BLAS。量化后 Mixtral-8x7B 权重从 93.41GB → 18.98GB（4.92x 压缩），可在单张 RTX 3090 (24GB) 上部署。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：BitBLAS 处理 group-wise INT4/INT3 权重的混合精度 GEMM 操作。PESF 剪枝的 expert 对应 kernel launch 被跳过，减少实际计算量。论文未提供自定义 kernel 实现。
  - **硬件架构层**：RTX 3090 24GB GPU。QESC 量化：显存 18.98GB，batch=4 seq=512: total 加速 1.54x。QESC+PESF: 加速 1.68x。准确率：71.68% (vs baseline 72.64%, 损失 <1%)。

- baseline方法是什么？
  Baseline 是 **固定计算预算的 Dense LLM 推理**，即每个 token 在每层都经过相同大小 FFN 处理，无论 token 复杂度如何。具体痛点：(1) **计算浪费**：简单 token（如标点、常见词）与困难 token（如专有名词、代码关键字）消耗相同计算量，导致资源利用率低；(2) **缺乏自适应性**：无法根据输入复杂度动态分配计算，限制了效率-精度 Pareto frontier；(3) **MoE router 训练次优**：传统 MoE router 使用 per-layer load balancing loss，强制每层内均匀分配 token 给各 expert，限制了跨层的灵活计算分配，导致路由模式偏离理论最优。

  Baseline 全栈执行例子（以 Dense 12-layer Llama-style 1.4B 模型推理一个 token 为例）：
  - **算法层**：标准 Dense Transformer，每层 Attention + FFN（inner_dim=10240, SwiGLU），每 token 固定经过 12 层相同规模 FFN。Router 不存在，无任何动态路由。
  - **系统框架层**：标准 PyTorch/HuggingFace Transformers 推理 pipeline。无 adaptive batching 或 dynamic compute allocation 机制。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：标准 cuBLAS GEMM 执行 FFN 矩阵乘法。每层 FFN 对每 token 执行相同的 M×K×N 矩阵乘，无 sparsity 或 conditional execution kernel。每个 token 触发完整的 12 层 FFN forward。
  - **硬件架构层**：论文未明确说明（推断为 NVIDIA GPU，如 A100/H100），标准 GPU 执行 dense matmul。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **Duo-LLM 框架**，通过在每层 FFN 中加入 big + small 两个模块，并研究 oracle 最优路由与 learned router 的差距，系统性地研究自适应计算。三个关键设计对应 Baseline 缺陷：

  **(1) Duo FFN 模块（解决计算浪费）**：每层 FFN 包含 big（inner_dim=10240）和 small（inner_dim=640, 16x smaller）两个模块。简单 token 可被路由到 small 模块以节省计算，困难 token 路由到 big 模块以获得更高精度。训练时以 random routing（p=0.5）确保两个模块可互换。
  
  **(2) Oracle 最优路由（揭示理论上界）**：穷举所有 2^L（或 3^L 含 skip）条路由路径，在给定计算预算下选择最小化 perplexity 的路由。发现核心洞察：
  - 仅激活 1 个 big layer 的 oracle 路由 perplexity **低于**所有 12 层都用 big module；
  - 最优 big layer 数量为 6/12（而非 12/12），因为 12C6 候选路径最多，增大了选到优质路径的概率；
  - 预算有限时（4 big layers），oracle 优先将 big 分配给**后层**；预算充足时（8 big layers），优先分配给**前层**；
  - 后层存在"容量阈值"——满足阈值后才值得给前层增加计算。

  **(3) Budget loss 替代 per-layer load balancing（解决 Router 次优）**：不同于传统 MoE 的 per-layer load balancing loss（强制每层内 expert 使用均匀），Duo-LLM 使用全局 budget loss `L_budget = (mean(P_big) - target_budget)^2`，允许 router 跨层灵活分配计算。暴露了 trained router 与 oracle 的巨大差距——router 的 perplexity 更接近 fixed pattern 而非 oracle，证明了现有 MoE router 训练的次优性。

  论文方法全栈执行例子（以 Duo-LLM 推理一个 token，预算=4 big/12 layers 为例）：
  - **算法层**：Token 输入第 1 层 → shared Attention → Router W_{r,1} 计算 P_big/P_small → 若 P_small > P_big 则走 SmallFFN (inner_dim=640) → 残差连接。逐层推进，直至 12 层中恰好使用了 4 个 big FFN。Oracle 模式下，路由决策由 exhaustive search 预先确定（前层多用 small、后层多用 big）。Learned router 模式下，路由由 softmax(W_r * x) 采样决定。最终 output logits 用于计算 token loss。
  - **系统框架层**：论文未实现端到端 serving 框架。论文提到 Megablocks 的 block-sparse matmul 可在单 GPU 上高效执行 Duo-LLM，但实际 efficient implementation "beyond the scope of this work"。
  - **编译框架层**：论文未明确说明。
  - **Kernel调度层**：论文未明确说明。理论上，每层根据路由决策执行不同尺寸的 GEMM（big: 2560×10240 或 small: 2560×640），可利用条件执行或 block-sparse matmul kernel 减少计算。
  - **硬件架构层**：论文未明确说明。

## Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models

- baseline方法是什么？
  Baseline 是 **micro-batch level LBL**（LBL_micro）。在主流开源 MoE 训练框架（Deepspeed-MoE、Tutel、Megablocks、Megatron-Core）中，LBL 在每个 parallel group（即每个 GPU 的 micro-batch）内独立计算 f_i 和 P_i，然后 all-gather 平均得到 LBL_micro。其问题在于：(1) 大模型训练中 micro-batch 通常仅含极少序列（数千 tokens），LBL 几乎退化到序列级均衡约束；(2) 由于数据多样性控制，一个 micro-batch 通常由同域数据打包而成，但 micro-batch LBL 仍然强制将这些同域 token 均匀分配到所有 expert，抑制了 expert 的 domain specialization。Baseline 全栈执行例子：训练时，每个 GPU 上的 micro-batch 含 ≤4 条序列 → Router 计算出 token-expert 分配后，在 GPU 本地计算 f_i、P_i 并计算 LBL → all-gather 平均各 GPU 的 LBL → 反向传播时，Router 被梯度强制学习在每个 micro-batch 内均匀分配 token → 结果是各 domain token 被几乎无差别分配，expert 没有 domain 级 specialization。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文方法是将 LBL 从 **micro-batch 级别**改为 **global-batch 级别**计算（LBL_global），通过两个机制实现：(1) **跨并行组同步专家选择频率 f_i**：在各 Data Parallel 组之间 all-reduce f_i（仅 N_E 维向量），用全局频率 f̄_i 替换本地 f_i 计算 LBL，从而将均衡约束从"每序列内均匀"放松为"全语料库均匀"；(2) **Buffer 近似机制**：当节点有限、微批总和小于全局批大小时，在 GA 各步缓冲累积同步后的 c_i，逐步逼近 global f̄_i。该方法直接解决了 Baseline 的核心缺陷——micro-batch LBL 将约束定得太紧（序列级），阻止了 router 将特定域 token 集中分配给特定 expert。Global-batch LBL 全栈执行例子：各 GPU 的 micro-batch 完成 Router 前向并获得 c_i → all-reduce 同步 c_i 得到全局计数 → 用全局 f̄_i（若 GA 则用 Buffer 累积近似）计算 LBL → 反向传播时，Router 被梯度鼓励在 global-batch 整体上均衡，但不要求每个 micro-batch 内均衡 → 因此 Router 可以将 SFT-Math 的 token 倾向于某些 expert、SFT-Code 的 token 倾向于另一些 expert → expert domain specialization 自然涌现（如图 multi-domain 选择频率差异达 0.2+）。Shuffle LBL_micro 消融证实：性能提升来自 token 多样性（引入不同域数据），而非 token 数量的方差降低。额外开销：通信 f_i 仅 ~1% latency，局部负载不均可通过加微量 micro-batch LBL（1% weight）恢复速度至 2.6% 以内。

- baseline方法是什么？
  Baseline 为传统 Dense Transformer 模型（GPT-like NLG），以及 PyTorch 分布式推理作为 MoE inference baseline。具体痛点：(1) **Dense 模型训练成本高**：随模型规模增大，训练 FLOPs 线性增长，达到 6.7B/175B/530B 级别需要数千 GPU 数月训练（MT-NLG 530B 需 >2000 A100 GPUs × 3 个月），继续 scale 不可行。(2) **Standard MoE 参数效率低**：现有 MoE（如 Switch Transformer）需要 10x dense 参数量才能达到质量持平，海量参数导致训练内存需求大、推理延迟高（推理为 memory bandwidth bound，参数量即延迟瓶颈）。(3) **Standard MoE 所有层专家数相同**：未利用深层/浅层学习不同表征的特性（CV 中已知深层学习 task-specific 特征，浅层学 general 特征），导致专家结构浪费。(4) **Top-2 gating 通信开销大**：增加 expert capacity 能提精度但 all-to-all 通信量翻倍，训练/推理速度显著下降。(5) **PyTorch 分布式推理低效**：现有的 MoE 分布式推理使用 naive PyTorch (tensor-slicing + expert-parallelism)，all-to-all 通信瓶颈大、kernel 效率低、无法 scale 到多节点。

  **Baseline 全栈执行例子（以 PyTorch MoE inference, 1.3B+MoE-128, 52B params, 128 GPUs, EP=128 推理一个 batch token 为例）**：
  - **算法层**: Standard MoE with 128 experts, top-1 gating, 24 layers (12 MoE layers), GPT-like architecture. 52B total params, 1.3B activated per token.
  - **系统框架层**: PyTorch distributed — 使用 basic expert parallelism (128-way) + tensor-slicing for non-expert params。All-to-all 通信使用默认 NCCL via torch.distributed。无 hierarchical all-to-all，无 parallelism-coordinated optimization。
  - **编译框架层**: 论文未明确说明（PyTorch eager execution）。
  - **Kernel调度层**: Sparse-dense einsum 实现 token routing（gating→one-hot mask→einsum sort→expert FFN→einsum unsort），复杂度 S×E×M×ce（大量与零的无效乘加）。Gating 函数由多个独立 kernel 调用完成（top-k, cumsum, scatter, mask creation）。
  - **硬件架构层**: 8×NVIDIA A100 GPU/节点，NVLink intra-node，Mellanox InfiniBand inter-node。TDP capped by single GPU memory bandwidth for large dense models。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **(1) MoE for Auto-Regressive NLG**：首次系统性将 MoE 应用于 GPT-like 自回归 NLG 模型（对比先前工作仅关注 encoder-decoder）。每两层 dense feedforward 之一替换为 128-expert MoE 层，top-1 gating 使每 token 计算量与 base dense 相同但模型质量远超 dense。350M+MoE-128 质量对标 1.3B dense（4x），1.3B+MoE-128 质量对标 6.7B dense（5x training cost reduction）。Training throughput: 1.3B+MoE-128 = 372 samples/sec vs 6.7B dense = 70 samples/sec on 128 A100 GPUs。

  **(2) PR-MoE (Pyramid-Residual MoE)**：基于两个关键发现 —— Phenomenon-I（深层 MoE 对模型质量贡献远大于浅层，Second-Half-MoE >> First-Half-MoE）和 Phenomenon-II（Residual-MoE: 固定 MLP + 可变 expert 等价于 Top-2 gating 精度但仅需 Top-1 通信量）。Pyramid-MoE 在深层使用更多 experts（如 350M+PR-MoE-32/64，前 10 层 32 experts，后 2 层 64 experts）。Residual-MoE 每 token 同时经固定 dense MLP + 选定 expert 处理（残差相加）。PR-MoE 组合两者：350M+PR-MoE-32/64 (4B) 精度对标 350M+MoE-128 (13B) → 3x 参数减少；1.3B+PR-MoE-64/128 (31B) 精度对标 1.3B+MoE-128 (52B) → 40% 参数减少。训练层面：设计 multi-expert + multi-data parallelism 灵活策略支持不同层不同 expert 数，避免 load imbalance 和 batch size 降低。

  **(3) MoS (Mixture-of-Students, Staged KD)**：MoE-to-MoE 知识蒸馏（非 MoE-to-Dense 蒸馏），学生保留 MoE 架构的稀疏优势。发现全程 KD 损失伤害精度（后期 underfitting），提出 staged KD：前 400K steps 使用 KD loss（L = L_CE + α·L_KD），后期停用 KD 仅优化标准 LM loss。学生层数减少 12.5%（24→21 层），350M+PR-MoE+L21+MoS (3.5B) 保留 99.5% 教师性能，1.3B+PR-MoE+L21+MoS (27B) 保留 99.1%。PR-MoE + MoS 组合减少参数至 3.7x。

  **(4) DeepSpeed-MoE Inference System - Multi-Dimensional Parallelism**：Expert 参数使用 expert parallelism (EP) + expert-slicing（tensor-slicing of experts）；Non-expert 参数使用 tensor-slicing (intra-node) + data parallelism (inter-node)。critical data path 降至每 token = 1.3B（仅 base dense），远小于 6.7B dense counterpart。

  **(5) Hierarchical All-to-All + Parallelism-Coordinated Communication**：Hierarchical all-to-all: 两步 intra-node → inter-node all-to-all（数据布局变换 + P2P），hops O(p) → O(G+p/G)。Parallelism-Coordinated: 当 EP + TP 组合时，利用 TP all-reduce 造成的数据复制，限定 all-to-all 仅在同 TP rank 子集内进行，延迟 O(p) → O(p/L)（L=TP degree）。解决 baseline 中 NCCL all-to-all 随设备数线性增长不 scale 的问题。

  **(6) Optimized MoE Kernels**：Gating fusion (top-k + Blelloch scan cumsum + scatter) → 单 kernel, dense mapping table 替代 sparse mask。Data-layout transformation 替代 sparse einsum: 复杂度 S×E×M×ce → S×M×ce, 消除 (E-1)/E 的零运算，融合 gating probability scaling。实现 6x+ MoE kernel 延迟降低，这是 PyTorch baseline 完全无法做到的。

  **论文方法全栈执行例子（以 DeepSpeed-MoE inference, 1.3B+MoE-128, 52B params, 128 GPUs, EP=128, TP=8 推理一个 batch token 为例）**：
  - **算法层**: PR-MoE (可选) + MoS (可选) architecture。Top-1 sparse gating。每 token critical path = base dense size (1.3B)。
  - **系统框架层**: DeepSpeed-MoE inference。128-way EP + 8-way TP for non-expert (within node) + data-parallel across nodes for non-expert。Multi-dimensional parallelism 协同调度：expert partition decisions based on EP/expert-slicing; non-expert partition via TP/DP。
  - **编译框架层**: 论文未明确说明（DeepSpeed 为 PyTorch-based framework，不使用编译框架）。
  - **Kernel调度层**: Fused gating kernel (Blelloch scan cumsum + dense mapping)。Data-layout transform kernel (sort/unsort by expert without sparse einsum)。Parallelism-Coordinated all-to-all: O(p/L) = O(128/8) = 16-hop all-to-all（vs PyTorch O(128)=128 hops）。Hierarchical all-to-all: intra-node (8 GPUs) + inter-node (16 nodes)。Token re-order with gating probability scaling fused。
  - **硬件架构层**: 128+ A100 GPUs (16+ nodes), NVLink intra-node + InfiniBand inter-node。Microsoft SCCL optimized all-to-all。MoE kernel per GPU 仅处理 1 expert（EP=128）→ 极低 latency。Throughput: super-linear scaling（per GPU throughput 随 GPU 数增加而增加）。Max scale: 2T param model in <25ms latency。

## DeepSeek-V3 Technical Report

- baseline方法是什么？
  Baseline 为 DeepSeek-V2 架构（MLA + DeepSeekMoE + auxiliary-loss-based load balancing + next-token prediction only），以及 BF16 训练范式。具体痛点：(1) **Auxiliary Loss 干扰模型性能**：传统 MoE 使用 auxiliary loss 强制负载均衡，但过大的 auxiliary loss 会损伤模型性能（trade-off between load balance and model quality）；序列级 auxiliary loss 要求每个序列内部负载均衡，限制了专家的领域特化能力。(2) **Next-token prediction 训练信号稀疏**：每个 position 仅预测下一个 token，训练信号密度低，数据效率有限。(3) **BF16 训练通信与内存开销大**：BF16 训练的 activation、通信和 optimizer state 占用大量 GPU 内存和带宽；跨节点 MoE 通信开销与计算量之比约为 1:1，成为训练瓶颈。(4) **Pipeline parallelism bubble 大**：传统 1F1B 和 ZB1P pipeline parallelism 仍存在显著的 bubble 和通信-计算串行问题。(5) **推理部署资源需求大**：MoE 推理需要大量 GPU 才能高效运行，小型团队难以部署。

  **Baseline 全栈执行例子（以 DeepSeek-V2, 236B total/21B activated, 单 token decode 为例）**：
  - **算法层**: MLA (d_c=512) + DeepSeekMoE (2 shared + 160 routed, K_r=6, Sigmoid gating with top-K normalization, auxiliary-loss-based load balancing)。Next-token prediction only。
  - **系统框架层**: HAI-LLM 框架，BF16 训练，HAI-LLM 框架上的标准 1F1B/ZB1P pipeline parallelism + expert parallelism + ZeRO-1 data parallelism。All-to-all 通信使用 NCCL，无 overlap。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 FlashAttention kernel，NCCL all-to-all 通信 kernel。BF16 GEMM on Tensor Cores。
  - **硬件架构层**: 8×H800 GPU/节点 × N nodes。NVLink + NVSwitch 节点内，InfiniBand 跨节点。BF16 参数/activation 存储，FP32 optimizer states。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **(1) Auxiliary-Loss-Free Load Balancing**：引入每个 expert 的 bias 项 b_i，仅在 routing 时加到 affinity score 上（s_{i,t}+b_i 决定 Top-K），每 training step 结束时动态调整：过载专家 b_i -= γ(0.001)，欠载专家 b_i += γ(0.001)。Gating value 仍使用原始 s_{i,t}。仅保留极小的 sequence-wise balance loss (α=0.0001) 防止极端不均衡。消除 auxiliary loss 对性能的负面影响，同时通过 batch-wise balancing 允许专家在不同 domain 上特化（Pile-test 证实 auxiliary-loss-free 模型展现更强的 domain-specific expert specialization patterns）。

  **(2) Multi-Token Prediction (MTP)**：增加 1-depth MTP 模块，每个 position 额外预测下下个 token，保持完整 causal chain。MTP 模块包含 shared embedding layer、shared output head、独立 Transformer block TRM_1 和 projection matrix M_1。训练时 λ=0.3 (first 10T) → 0.1 (last 4.8T)。推理时 MTP 模块可丢弃（正常推理）或用于 speculative decoding（第二 token 接受率 85-90%，1.8x TPS 加速）。稠化训练信号，提升数据效率；MTP 消融实验：small MoE (15.7B) 和 large MoE (228.7B) 上均一致提升 benchmark 性能。

  **(3) FP8 Mixed Precision Training**：fine-grained quantization（activation: 1×128 tile-wise, weight: 128×128 block-wise），E4M3 for all tensors, online quantization (per-tile max)。CUDA Core FP32 promotion：每 N_c=128 个 WGMMA 结果拷贝到 CUDA Core 做完整 FP32 累积+dequantization。BF16 optimizer states（first/second moments 不用 FP32），activation cached in FP8 (E5M6 for attention inputs, FP8 for SwiGLU inputs)。低精度通信：MoE up-projection 前将 activation 量化为 FP8 再 dispatch。BF16 → FP8 训练 loss error <0.25%，训练速度理论加倍，GPU 内存显著减少。

  **(4) DualPipe Pipeline Parallelism**：双向流水线调度，将每个 chunk 拆分为 attention/dispatch/MLP/combine 四组件，后向再拆分 backward for input 和 backward for weights。通过手动调整 SM 比例实现 all-to-all 和 PP 通信与计算的完全重叠。Bubble = (PP-1)/(PP)*(F&B-3W)/(F+B-W)，比 1F1B 和 ZB1P 更小。仅需 pipeline stages 和 micro-batches 可被 2 整除。支持跨节点 fine-grained experts 而通信开销近零。

  **(5) Custom Cross-Node All-to-All Kernels**：warp specialization + 20 SMs/10 channels。IB send → IB-to-NVLink forward → NVLink receive 流水线处理 dispatching；NVLink send → NVLink-to-IB forward+accumulate → IB receive+accumulate 处理 combining。PTX 定制指令 + auto-tuned chunk size 减小 L2 cache 污染。配合 node-limited routing (M=4, avg 3.2 experts/node)，仅 20/132 SMs 即可跑满 IB+NVLink 带宽。

  **(6) Inference Deployment Strategy**：Prefill-Decoding 分离部署（prefill: 4 nodes/32 GPUs, TP4+SP+DP8+EP32；decode: 40 nodes/320 GPUs, TP4+SP+DP80+EP320）。冗余专家部署：每 10 分钟检测高负载 expert 并复制，prefill 阶段 32 冗余 expert，decode 阶段 64 GPU 承载冗余+共享 expert。Micro-batch 双流水线重叠（prefill: 重叠两 micro-batch 的 attention/MoE 和 dispatch/combine；decode: 重叠 attention 和 dispatch+MoE+combine）。正在探索 dynamic redundancy（每 GPU 16 experts，仅激活 9 个，运行时全局最优路由计算）。

  **论文方法全栈执行例子（以 DeepSeek-V3, 671B total/37B activated, 跨节点训练一个 forward-backward chunk pair 为例）**：
  - **算法层**: MLA (d_c=512, d_c'=1536) + DeepSeekMoE (1 shared + 256 routed, K_r=8, M=4 nodes, Sigmoid gating, bias-based aux-loss-free routing, no token dropping) + MTP (D=1, λ=0.3, shared emb/head + independent TRM_1)。FP8 E4M3 fine-grained quantized GEMM。
  - **系统框架层**: HAI-LLM framework。16-way PP + 64-way EP (8 nodes) + ZeRO-1 DP。DualPipe 双向调度：正向 chunk [Attn|Dispatch|MLP|Combine]+PP_Comm 与反向 chunk [Attn_BW_In|Attn_BW_W|Disp_BW|MLP_BW_In|MLP_BW_W|Comb_BW]+PP_Comm 重叠。RMSNorm + MLA up-projection recomputation。EMA 异步在 CPU 更新。Shared Embedding+Output Head 部署在首尾相同 PP rank。
  - **编译框架层**: 论文未明确说明（自研 HAI-LLM 为内部训练框架，非开源编译框架）。
  - **Kernel调度层**: Cross-node all-to-all kernel — warp specialization, 20 SMs/10 channels, IB+NVLink fully overlapped pipeline。FP8 GEMM — WGMMA on Tensor Core, N_c=128 interval CUDA Core FP32 promotion, per-group scaling dequantization fused。PTX instruction + auto-tuned chunk size + L2 cache interference minimization。
  - **硬件架构层**: H800 GPU, 132 SMs, Tensor Core (14-bit accumulation hardware limit), NVLink 160 GB/s + NVSwitch intra-node, IB 50 GB/s inter-node。FP8 activation caching (E5M6 for attn inputs), BF16 optimizer states, FP32 master weights+gradients。Cost: 180K H800 GPU hours/trillion tokens → 14.8T tokens = 2.664M GPU hours ($5.328M) pre-training。

## DeepSeek-V2 A Strong, Economical, and Efficient Mixture-of-Experts Language Model

- baseline方法是什么？
  Baseline 为传统 Dense Transformer（以 DeepSeek 67B 为代表）：使用标准 Multi-Head Attention (MHA) 和 Dense FFN。MHA 每个 token 每层需缓存 2×n_h×d_h×l 个 KV 元素（DeepSeek 67B 约 1.9M elements），严重限制推理时的最大 batch size 和序列长度。Dense FFN 每个 token 激活全部参数（67B），训练 FLOPs 随参数量线性增长，成本高昂。GQA 和 MQA 虽能减少 KV cache 但性能显著弱于 MHA；传统 MoE（如 GShard）的粗粒度专家分割导致专家特化不足和知识冗余。
  
  **Baseline 全栈执行例子（以 DeepSeek 67B, MHA+Dense, 推理一个 decode token 为例）**：
  - **算法层**: MHA — Q=W^Q@h, K=W^K@h, V=W^V@h, O=Softmax(QK^T/√d_h)V, 缓存完整 K,V（~1.9M elements/token/layer）。Dense FFN — h'=h+FFN(h)，每个 token 激活全部 67B 参数。
  - **系统框架层**: HAI-LLM 训练框架，无专家并行（纯 dense），使用 pipeline parallelism + data parallelism。
  - **编译框架层**: 论文未明确说明（使用标准 PyTorch/FlashAttention 等底层库）。
  - **Kernel调度层**: 标准 FlashAttention kernel（无 MLA 吸收优化），标准 GEMM kernel for FFN。
  - **硬件架构层**: 8×H800 GPU/节点，NVLink + NVSwitch 节点内互联，InfiniBand 跨节点。MHA KV cache 随序列长度线性增长，decode 阶段 memory-bound。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **(1) MLA (Multi-head Latent Attention)**：通过低秩 KV 联合压缩 c^{KV}=W^{DKV}@h 将 KV cache 压缩到 d_c 维（512 vs MHA 的 16384），并在推理时将上投影矩阵 W^{UK}/W^{UV} 吸收进 W^{UQ}/W^O，避免显式计算 K/V。解耦 RoPE 通过额外的小维度多头 query+共享 key 承载位置信息，解决 RoPE 与低秩压缩的兼容性问题。KV cache 从 MHA 的 ~1.9M elements 降至 ~34.6K elements（减少 93.3%），且性能优于 MHA。
  
  **(2) DeepSeekMoE**：细粒度专家分割（160 路由专家，每 token 激活 6 个）提升专家特化程度，隔离 2 个共享专家减少路由专家间知识冗余。Device-Limited Routing (M=3) 限制每 token 的目标设备数以控制 all-to-all 通信开销。三层辅助损失 (Expert/Device/Communication Balance) + Token-Dropping 保证分布式训练负载均衡。21B 激活参数即可达到与 67B-72B dense 模型相当的 top-tier 性能。
  
  **论文方法全栈执行例子（以 DeepSeek-V2, 236B total/21B activated, 推理一个 decode token 为例）**：
  - **算法层**: MLA — c^{KV}=W^{DKV}@h (仅 512 维需缓存), q^C=W^{UQ}@c^Q, q^R=RoPE(W^{QR}@c^Q), k^R=RoPE(W^{KR}@h)。吸收优化：W^{UK} 融入 W^{UQ}, W^{UV} 融入 W^O，attention 时 K/V 无需显式重建。DeepSeekMoE — h'=u+ΣFFN_i^{(s)}(u)+Σg_{i,t}·FFN_i^{(r)}(u)，仅 6/160 路由专家激活。
  - **系统框架层**: HAI-LLM + 16-way ZB-Pipeline Parallelism + 8-way Expert Parallelism + ZeRO-1 Data Parallelism。无需 Tensor Parallelism（激活参数少）。共享专家计算与 expert parallel all-to-all 通信重叠。vLLM 作为推理后端。
  - **编译框架层**: 论文未明确说明。使用改进版 FlashAttention-2 优化 MLA。
  - **Kernel调度层**: 自研 CUDA kernels 加速 all-to-all 通信、routing 算法、跨专家 fused linear 计算。MLA 的 W^{UK}/W^{UV} 吸收优化避免 decode 时 K/V 重建计算。
  - **硬件架构层**: 8×H800 GPU/节点。FP8 精度部署 + KV cache 6-bit 量化进一步压缩。MLA 使 KV cache 仅 ~25.9KB/token (FP8+6bit quant)，远小于 MHA，decode 阶段从 memory-bound 变为 compute-bound。单节点生成吞吐 >50K tokens/s（5.76× DeepSeek 67B）。

## DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

- baseline方法是什么？
  **Baseline 是 Vanilla Upcycling**：将预训练 Dense 模型中的原始 FFN 层复制 N 份作为 N 个 MoE 专家的初始权重，再引入一个随机初始化的 Router 进行专家选择。训练后每个专家权重 W_i 独立更新。
  
  **Baseline 的核心缺陷**：引入大量冗余参数。以 MoE-LLaVA-Phi 为例，2.52B 额外参数（占模型的 50%+），这些参数中存在极高的冗余——训练后专家权重与初始 FFN 权重的余弦相似度 > 0.999，专家间余弦相似度也 > 0.999。这意味着 Δ_i = W_i - W_base 是微小且冗余的调整。
  
  **Baseline 全栈执行例子（以 MoE-LLaVA-Phi Vanilla Upcycling 推理一个 visual token 为例）**：
  - **算法层**：图像经过 CLIP-Large 视觉编码器 → visual token embedding → 每间隔一个 Transformer block：attention 计算 → Router 计算 top-2 softmax 路由分数 → 激活 2 个专家 FFN（每个专家是独立复制的 W_i，共 N=4 个独立权重矩阵各占 2560×10240）→ 加权求和输出 → 下一层 → LM head 预测。每个 MoE 层存储 4 个完整 FFN 权重矩阵（每个 ~26M 参数），合计 ~105M/层。
  - **系统框架层**：论文未明确说明（标准 PyTorch/HuggingFace Transformers 推理框架，标准 MoE 前向传播）
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：标准 dense GEMM kernel 执行每个激活专家的 FFN 计算；Router 的 top-k 选择为稀疏激活，但专家权重本身为 dense 矩阵
  - **硬件架构层**：NVIDIA A100 80GB GPU 执行标准 CUDA kernel

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：DeRS (Decompose, Replace, Synthesis) 范式，基于 upcycled MoE 专家共享相同初始权重 W_base 的特性，将 N 个专家重构为一个共享基础权重 + N 个轻量 delta 权重的形式，从而消除参数冗余。
  
  **具体设计如何解决 Baseline 缺陷**：
  
  1. **Decompose（分解）**：观察到 W_i = W_base + Δ_i，且余弦相似度 > 0.999 表明 Δ_i 是微小调整。将专家权重显式分解为共享部分和增量部分，使得冗余的 Δ_i 可以被独立压缩。
  
  2. **Replace（替换）**：用轻量表示替换冗余的 Δ_i：
     - DeRS Compression 中：对已训练的 Δ_i 应用后处理稀疏化（随机 drop + rescale）或量化（降低位宽），MoE 层参数从 4×K 降至 K+4×k
     - DeRS Upcycling 中：从训练开始就用稀疏矩阵（紧凑索引+值向量）或低秩矩阵（A@B）表示增量权重，训练参数从 N·d·d_h 降至 d·d_h+N·r·(d+d_h)，实现高达 2270× 参数减少
  3. **Synthesis（合成）**：推理时按需合成 Ŵ_i = W_base + F(Δ_i)，不增加推理延迟的额外开销。
  
  **关键创新点**：这是首次利用 upcycled MoE 特有的"同源初始化"特性进行专家参数去冗余。Vanilla MoE 从 scratch 训练时各专家随机初始化，无法应用此分解方法。
  
  **论文方法全栈执行例子（以 DeRS-LM Upcycling + MoE-LLaVA-Phi 推理一个 visual token 为例）**：
  - **算法层**：图像经过 CLIP-Large 视觉编码器 → visual token embedding → 每间隔一个 block：attention 计算 → Router 计算 top-2 路由分数 → 对于被选中的 2 个专家：
    1. 合成专家权重：W_i = W_shared + A_i @ B_i（A_i: 2560×1, B_i: 1×10240，低秩分解，共仅 ~12.8K 参数 vs 26M）
    2. 使用合成权重执行 FFN 计算
    → 加权求和输出 → 下一层。每个 MoE 层存储 1 个共享 W_shared (26M) + 4×(A_i+B_i) (~51K×4=0.2M)，合计 ~26.2M/层 vs Vanilla 的 ~105M/层。
  - **系统框架层**：论文未明确说明（标准 PyTorch 训练/推理，使用 torch.scatter 实现稀疏矩阵映射，低秩矩阵使用标准 matmul）
  - **编译框架层**：论文未明确说明
  - **Kernel/运行时调度层**：W_shared + A_i@B_i 的合成操作为轻量级加法/矩阵乘法，可在 GPU 上高效执行；稀疏矩阵版本需要在推理时从紧凑向量重构为 sparse/dense 矩阵（torch.scatter）
  - **硬件架构层**：NVIDIA A100 80GB GPU；推理内存从 Vanilla 的 10.5G 降至 DeRS-LM 的 5.9G (43.8% 减少)

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

## Aria An Open Multimodal Native Mixture-of-Experts Model

- baseline方法是什么？
  Baseline 主要对比两类模型：

  1. **Dense Multimodal Models**：Llama3.2-11B（11B dense，视觉编码器 + 语言解码器的标准 VLM 架构），同规模下全部参数参与每次推理，无法利用稀疏激活降低推理成本。
  
  2. **Modality-Specialized MoE Models**：Pixtral-12B（基于 Mixtral 的 multimodal MoE）、MoE-LLaVA（从 dense 模型 upcycling 的 multimodal MoE）、MoMa（modality-aware expert 架构）。这些 MoE 要么从 dense 模型 "upcycle" 而来（非原生的多模态稀疏训练），要么为不同模态设计专属 expert（增加架构复杂度和模块化成本）。
  
  **Baseline 全栈执行例子（以 Llama3.2-11B 推理一个图文混合请求为例）**：

  - **算法层**：输入 "Describe this image" + image → Visual Encoder (ViT) 将图像编码为 N 个 visual tokens → 与 text tokens 拼接为 [T_vis, T_txt] → 进入 11B dense Transformer decoder → 每层 self-attention (所有 token 相互 attend, O((N+M)²·d)) → 每层 FFN (d_model → d_ff → d_model, 全部 11B 参数参与计算) → auto-regressive 生成每个 output token。dense 模型中每个 token 都激活全部 11B 参数，无 expert 路由。

  - **系统框架层**：HuggingFace Transformers 标准推理。Pixtral-12B 使用 mistral-inference 框架。无特殊的多模态 serving 优化，visual encoder 独立前向计算后送入 LLM decoder。

  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution，使用 Flash-Attention 加速 attention 计算）。

  - **Kernel级**：标准 GEMM kernel 执行 attention projection 和 FFN。Pixtral-12B 的 MoE 层使用 token-choice routing（Top-2 experts），每个 token 的 expert 选择独立，batch 内不同 token 可能激活不同 expert 组合，需要 group_gemm 实现。

  - **硬件架构层**：NVIDIA A100/H100 GPU。dense 11B 模型 bf16 约 22GB 显存（不含 KV cache），对 consumer GPU（<24GB）压力大。

  **Baseline 的核心缺陷**：
  1. **Modality Performance Gap**：现有 open multimodal 模型在跨模态能力上不均衡——Pixtral-12B 在长视频理解（LongVideoBench 47.4 vs ARIA 65.3）严重落后，且缺乏对视觉/语言/代码的统一 high-quality 能力。dense 模型在扩展到 multimodal 后往往损害纯语言能力（knowledge forgetting）。
  2. **非原生多模态 MoE**：先前 multimodal MoE（MoE-LLaVA, MoMa）依赖从 dense checkpoint upcycling 初始化和/或设计 modality-specific expert，限制了 expert specialization 的自然涌现。
  3. **Context Window 受限**：多数 open multimodal 模型的 context window 有限（通常 <8K），无法处理长视频或多页文档等 long-context multimodal 输入。
  4. **训练效率 vs 模型能力的平衡**：dense 模型全部参数参与每次前向，推理 FLOPs 随参数线性增长；而 MoE 可用更少激活参数达到同等能力，但训练 non-trivial 的 load balancing 和 expert specialization。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ARIA 提出 **multimodal native fine-grained MoE** 配合 4-stage 渐进式预训练 pipeline：

  **方法核心**：
  1. **Fine-grained MoE with Modality-Generic Experts**：66 experts/layer (2 shared + 64 routed)，每个 expert FFN hidden dim 仅 1664（远小于 dense 模型的标准 FFN），每 token 激活 2 shared + 6 routed experts（3.5B activated / 24.9B total）。关键点：所有 expert 是 modality-generic 的（不预设某 expert 只处理视觉/文本），expert specialization 在训练中自然涌现（Section 4.2 可视化证明了 visual-specialized experts 的存在）。
  
  2. **Multimodal Native Pre-training from Scratch**：不从 dense checkpoint upcycle，而是从随机初始化开始，language 和 multimodal 数据混合训练。MoE decoder 同时在 text 和 visual tokens 上做 next-token prediction，无需 modality-specific 专家架构。
  
  3. **4-Stage Training Pipeline**：渐进式赋予模型不同能力——Stage 1 (语言基础) → Stage 2 (多模态理解) → Stage 3 (长上下文) → Stage 4 (指令遵循)。每阶段维护前阶段能力的同时增加新能力。
  
  4. **Lightweight Visual Encoder with Cross-Attention Projection**：438M 参数的 ViT + cross-attention projection module，将变长/变分辨率图像压缩为固定数量 visual tokens（128/256），降低 decoder 的计算负担。
  
  5. **Group-Level Load Balancing**：用 8-expert group 级别的 load balancing 替代 per-expert balancing，避免 fine-grained MoE（64 routed experts）场景下过强的 load balancing 约束压制 expert specialization。

  **论文方法全栈执行例子（以 ARIA 处理 "Describe this video" 请求为例）**：

  - **算法层**：输入 64K-length video (N frames) + text prompt → 每帧经 ViT-SO400M 编码为 patches → projection module (cross-attn with 128 learnable queries) 将每帧压缩为 128 visual tokens → 共 M 个 visual tokens 与 text tokens 拼接 → MoE decoder (28 layers, 每层 66 experts):
    Layer l: Self-Attention → RMSNorm → Router(W_router · x) → Top-6 expert selection → 2 shared experts (always active) + 6 routed experts 分别计算 SwiGLU_FFN(x) → 加权求和。训练时用 group-level load balancing loss (8 experts/group)，推理时 expert 激活稀疏化——仅 3.5B/25.3B 参数参与计算，比同能力 dense 模型 (如 InternVL2-40B) 推理效率高得多。

  - **系统框架层**：训练基于修改版 Megatron-LM（expert parallelism + ZeRO-1 data parallelism，无 tensor parallelism 以减少 all-reduce 通信），推理支持 HuggingFace Transformers（AutoModelForCausalLM + AutoProcessor）和 vLLM（RadixAttention prefix caching 加速）。单张 A100 80GB 即可 bf16 推理。

  - **编译框架层**：论文未明确说明。依赖 Flash-Attention 和 grouped_gemm 等性能库加速 MoE expert 调度和 attention 计算。

  - **Kernel级**：expert parallelism 下，66 experts 分布在多个 GPU 上，每个 GPU 持有 expert 子集 + 一份完整的 attention 参数和 shared expert 参数。Router 输出通过 all-to-all 通信将 token 路由到目标 expert 所在 GPU，expert FFN 计算完成后 all-to-all 回传。Group-level load balancing 减少了路由坍缩风险。ZeRO-1 将 optimizer states 分片到各 GPU。

  - **硬件架构层**：多节点 NVIDIA A100/H100 GPU 集群。expert parallelism 将 expert 参数分布在多个 GPU 上，batch 内 token 通过 all-to-all 路由到对应 expert。训练中视觉 encoder 参数复制到所有 GPU（数据并行），MoE decoder 的 attention 参数也使用 ZeRO-1 数据并行（仅分片 optimizer states），expert FFN 参数使用 expert parallelism 分片。

  **对应解决 baseline 缺陷**：
  - 缺陷1（跨模态能力不均衡）→ 4-stage 渐进训练 + 语言/多模态/代码混合数据，确保各能力同步增长
  - 缺陷2（非原生 MoE）→ 从 scratch multimodal native 训练 + modality-generic experts，让 expert specialization 自然涌现
  - 缺陷3（context window 受限）→ Stage 3 专门扩展 context 到 64K + RoPE theta 从 100K 提升到 5M
  - 缺陷4（训练/推理效率）→ fine-grained MoE (3.5B/25.3B) + expert parallelism + ZeRO-1，无需 tensor parallelism

- baseline方法是什么？
  Baseline 为两类：

  1. **Dense 模型**：与 MoE 模型 FLOP-aligned (FA) 或 Parameter-aligned (PA) 的标准 Llama3 架构 dense Transformer。FA Dense 与 MoE 有相同的 active parameters（即相同的每 token 推理 FLOPs），PA Dense 与 MoE 有相同的 total parameters（即相同的内存占用）。

  2. **标准稀疏 MoE 模型**：Llama3 架构 + MoE 层（8 total experts, 2 active per token, Token Choice routing），使用标准 load balancing loss (Fedus et al., 2022)。推理时需要 expert offloading 以在端侧设备上运行——未使用的 experts 从 GPU 卸载到 CPU，每个 token 生成可能触发 expert 切换并引发 offload 操作。

  **Baseline 全栈执行例子（以 Phone-sized MoE, 1.37B active / 3.75B total, 推理一个 token 为例）**：

  - **算法层**：输入 token x_t → Router: softmax(W_router · x_t) → Top-2 选择 expert (如 expert 3, 5) → Expert 3 FFN 和 Expert 5 FFN 分别计算 → 加权求和 → 输出 y_t。每个 token 独立路由，连续 token 间 expert 选择无关联约束。Offloading 推理时：若当前 token 选中的 expert 集合 S_t ≠ S_{t-1}（前一个 token 的选中集合），需将不再需要的 expert 从 GPU 卸载，将新需要的 expert 从 CPU 加载到 GPU。

  - **系统框架层**：HuggingFace Transformers 标准推理（无 Serving 框架修改）。Expert offloading 逻辑：维护 GPU 上的活跃 expert 集合（不超过 2 active experts），每个 token 生成后检查是否需要 offload/load。

  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution）。

  - **Kernel/运行时调度层**：标准 cuBLAS GEMM 执行 expert FFN（SwiGLU: gate_proj → SiLU → up_proj → × → down_proj）。Offload 时 CPU↔GPU 数据传输串行化在 token 生成之间，导致 4-20× 推理延迟增加。

  - **硬件架构层**：CPU + GPU 服务器环境。端侧设备 (Phone) 假设 <6GB RAM。GPU 仅保留 active experts 参数（≈ FA Dense），其余 experts 驻留 CPU memory。

  **Baseline 的核心缺陷**：
  1. **参数量膨胀（Memory）**：MoE 的 total parameters 远超 active parameters（Phone: 3.75B vs 1.37B, 2.7×），超出端侧设备内存限制，必须依赖 expert offloading。
  2. **Offloading 导致的延迟（Latency）**：标准 MoE 在 offloading 场景下 Expert Replacement Ratio 高达 43.82%，意味着几乎每 2-3 个 token 就需要一次 expert 加载/卸载操作。每次 offload 引入明显的 PCIe/内存带宽延迟，导致 4-20× 的推理减速。
  3. **Dense vs MoE 的不公平比较**：以往研究在比较 Dense 和 MoE 时存在混杂因素（不同训练数据、不同训练配方、不同架构），无法明确归因 MoE 组件本身的贡献。
  4. **Expert 参数冗余**：8 个 expert 中每个都有完整的 FFN 权重矩阵（d_model × d_ff），但每个 expert 实际只需"专门化"处理约 1/8 的 token，全秩矩阵可能存在参数低效。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：CoSMoEs 通过两个正交的算法创新解决端侧 MoE 的三个维度挑战（Quality, Memory, Latency）：

  1. **Weight-Decomposed (WD) Experts**：将 expert FFN 的权重矩阵替换为低秩分解（类似 LoRA 但用于预训练），减少总参数量同时通过鼓励"expert specialization"提升模型质量。
  2. **Block-wise Expert Selection (BlES) Loss**：在训练阶段引入 sequence-level 的辅助损失，鼓励连续 token 选择相同的 expert 集合，从而在推理时减少 expert offloading 次数，降低延迟。

  **Defect → Design 映射**：

  | Baseline 缺陷 | CoSMoEs 设计选择 | 解决机制 |
  |---|---|---|
  | 参数量膨胀 → memory 超出端侧限制 | WD Experts: 低秩分解 M → L×R (r=n/2) | 减少每个 expert 的参数量，在参数对齐比较下可堆叠更多层/注意力头（WD MoE: 26L/20H vs MoE: 24L/18H） |
  | Offloading 导致延迟 → 43.82% expert replacement | BlES Loss: H_norm × L_norm 惩罚连续 token 的 expert 切换 | 减少 6× expert replacement (43.82% → 6.55%)，1.54× 生成速度提升 |
  | Dense vs MoE 不公平比较 → 无法归因 MoE 贡献 | 严格控制混杂因素：相同训练数据(FW-edu)、相同训练配方、架构对齐 | MoE 比 FA Dense 平均提升 +2.35%，可明确归因为 MoE 架构贡献 |
  | Expert 参数冗余 → full-rank 矩阵低效 | WD: 利用"特殊化"直觉——每个 expert 只需处理 1/E 的 token | WD MoE 比标准 MoE 额外 +1.1%，且总参数更少 (3.65B vs 3.75B) |
  | Load balancing 跨层可被 exploit | Sequence-level load balancing（每层独立计算） | 防止模型通过跨层"分工"欺骗 BlES loss 和 load balancing loss |

  **CoSMoEs 全栈执行例子（以 Phone-sized WD MoE + BlES, 推理多个 token 为例）**：

  - **算法层**：
    1. Token t=1: x_1 → Router: Top-2 选择 expert 3, 5
    2. WD Expert FFN: expert 3 的 gate_proj 用 L_gate_3 [d_ff, r] × R_gate_3 [r, d_model] 替代直接矩阵乘法 → 同理 up_proj 和 down_proj → SwiGLU 前向
    3. Token t=2: x_2 → Router: Top-2 仍是 expert 3, 5（BlES 训练的效果——连续 token 倾向相同 expert）
    4. 无需 offloading（S_t = S_{t-1}）→ 无 GPU↔CPU 数据传输
    5. Token t=3: 仍需 expert 3, 5 → 继续无 offloading
    6. 平均每 15 tokens 才触发一次 expert 切换（vs baseline 每 2-3 tokens）
    7. Expert replacement ratio: 6.55%（hard selection 统计），生成速度: 23.10 tok/s

  - **系统框架层**：HuggingFace Transformers + gpt-fast 推理。Offloading 逻辑与 baseline 相同，但因 BlES 损失训练后 expert 切换频率大幅降低，offload 操作的实际发生频率降低 6.7×。

  - **编译框架层**：论文未明确说明。

  - **Kernel/运行时调度层**：WD Expert FFN 的计算路径：x → x@R_gate^T → @L_gate^T（两次小矩阵乘法替代一次大矩阵乘法）→ SiLU → 同理 up_proj → × → @R_down^T → @L_down^T。低秩分解增加了矩阵乘法次数（3→6 次），但总 FLOPs 减少（r ≪ n,m）。在 batch=1 的端侧推理场景下，小矩阵乘法更 cache-friendly。

  - **硬件架构层**：GPU 显存仅保留 active experts（2 个 WD experts ≈ FA Dense 大小）。CPU memory 保留全部 8 个 WD experts。由于 WD 减少了每个 expert 的参数量，offloading 时的数据传输量也相应减少。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (标准 MoE + Offloading):
  Pre-train MoE on FW-edu → Router 自由选择 → 推理: 每个 token 
  → Router → 可能切换 expert → CPU↔GPU offload (每 2-3 token 一次)
  → 高延迟 (15.02 tok/s, ExRep=43.82%)
  
  CoSMoEs (WD MoE + BlES + Offloading):
  Pre-train WD MoE on FW-edu → 同时施加 BlES loss (H_norm × L_norm) 
  + Sequence-level load balancing → 推理: 每个 token → Router → 
  倾向保持 expert 选择稳定 → CPU↔GPU offload (每 15 token 一次)
  → 低延迟 (23.10 tok/s, ExRep=6.55%)
  ```

  **关键创新总结**：CoSMoEs 的核心洞察是将端侧部署的三个约束（Quality, Memory, Latency）分别通过算法手段解决——WD 提升 Quality 并减少 Memory，BlES 降低 Offloading Latency。两个创新正交：WD 专注于"每个 expert 内部如何更高效"，BlES 专注于"expert 之间如何协作以降低切换频率"。与 inference-time 优化方法（如 MoE-Infinity, EdgeMoE）相比，CoSMoEs 的训练时优化是正交的——可直接叠加使用。

## Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

- baseline方法是什么？
  **Baseline 为未剪枝的 MoE LLM（DeepSeek-V2-Lite / Qwen1.5-MoE-A2.7B）以及已有的 expert pruning 方法（Random Pruning、Seer Prune、Group&Merge）。**

  **已有剪枝方法的原理**：
  - Random Pruning：随机选择 experts 删除。
  - Seer Prune：基于 gate activation 统计（推理时收集 gate 激活频率），保留最频繁激活的 experts，删除低频 experts。
  - Group&Merge：将专家按某种相似度分组后合并。

  **这些 baseline 方法的共性问题**：
  1. 忽略 **intra-layer expert homogeneity**（层内专家功能冗余）：同一层内多个专家因训练动态发展出功能重叠。
  2. 忽略 **inter-layer similarity patterns**（跨层相似模式）：深层比浅层包含更多同质专家，冗余度随深度递增。
  3. 将专家视为独立单元处理，缺乏跨层全局视角。

  **Baseline 全栈执行例子（以 DeepSeek-V2-Lite, Seer Prune 为例）**：

  - **算法层**: 在 calibration 数据上收集每层每个 expert 的 gate 激活次数 → 按激活频率排序 → 保留 top-k 高频 experts，删除其余 → 路由权重不做调整（直接丢弃被删 expert 的路由条目）。该方法仅依赖 gate 统计信号，不考虑 expert 参数本身的功能相似性。
  - **系统框架层**: 论文未明确说明推理框架。使用 HuggingFace transformers 标准推理流程。Seer Prune 等 baseline 方法在模型层面操作（修改 model.state_dict()），不涉及推理框架修改。
  - **编译框架层**: 论文未明确说明（使用 PyTorch + HuggingFace transformers 标准推理）。
  - **Kernel/运行时调度层**: 论文未明确说明。MoE 层使用标准 sparse MoE kernel（top-k gating + 分组 GEMM）。剪枝后 expert 数量减少，对应 expert FFN 的 GEMM 计算量减少。
  - **硬件架构层**: 32× NVIDIA A100 80GB GPU 集群，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **Gate-only 信号局限**：Seer Prune 仅依赖 gate 激活频率判断专家重要性，忽略专家参数本身的功能相似性。两个功能几乎相同的专家可能都获得较高激活频率，导致冗余未被识别和消除。
  2. **层隔离假设失效**：现有方法逐层独立决策剪枝，忽略了深层专家更同质的趋势。Group&Merge 虽考虑了层内相似性，但未利用跨层同质模式优化全局剪枝策略。
  3. **丢弃式剪枝损失信息**：Random Pruning 和 Seer Prune 直接丢弃被剪专家及其路由权重，导致的功能损失无法恢复。Group&Merge 虽有合并但无权重自适应调节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: C-PRUNE = Layerwise Expert Clustering（层内聚类）+ Global Cluster Pruning（全局聚类剪枝）+ Parameterized Expert Merging（参数化专家合并）。

  **Defect → Design 映射**：

  | Baseline 缺陷 | C-PRUNE 设计选择 | 解决机制 |
  |---|---|---|
  | Gate-only 信号忽略参数相似性 | Expert Embedding: 用 calibration 数据计算每个 expert 的实际输出向量 φ(f_i) 作为功能特征，而非 gate 激活次数 | 通过参数空间分析（cosine similarity）直接衡量专家功能的冗余度，而非间接依赖路由统计。Expert activation-based embedding 捕获了专家的实际计算行为 |
  | 层隔离假设，忽略跨层同质趋势 | Global Cluster Pruning: 建立跨所有层的 unified importance score，惩罚深层 expert（depth penalty），全局统一排序决定剪枝 | 利用"深层专家更同质"的先验，在全局优化中对深层施加更大的剪枝压力。Layerwise ratio 0.2 + Global ratio 0.1 的配置达到最佳 trade-off |
  | 丢弃式剪枝损失信息 | Parameterized Expert Merging: 剪枝 cluster 中所有专家通过 weighted averaging 合并为一个专家，ω_i ∝ exp(γ·A_ik)（相似度越高权重越大）；路由权重更新时加入 exploration noise 维持多样性 | 保留被剪专家的功能信息（加权合并），同时通过温度 γ 控制融合锐度，避免信息丢失 |
  | 无自适应剪枝阈值 | Adaptive Clustering Threshold τ^(l) = mean_deviation + δ·σ^(l)，层越深阈值越大 | 根据每层的实际专家嵌入分布自适应调节聚类半径，深层允许更大的聚类（更大的冗余容忍度） |

  **C-PRUNE 全栈执行例子**：

  - **算法层**: 
    1. Expert Embedding: 在 task-specific calibration 数据上前向传播，每层每个 expert 取 K 个样本输出的均值作为嵌入 φ(f_i) ∈ R^d
    2. 亲和矩阵: A_ij = σ(α·cos(φ(f_i), φ(f_j)))
    3. Hierarchical Agglomerative Clustering: 层内自底向上合并最相似的 expert/cluster
    4. 聚类后合并: θ̂_k = Σ softmax(γ·A_ik)·θ_i（保留信息，而非丢弃）
    5. 全局剪枝: 跨层统一评分，深层专家受 depth penalty 更可能被剪
    6. 路由更新: Ŵ_k = mean(W_i) + ε·N(0,I)
    7. 可选 task-specific fine-tuning: 在目标 domain 数据上微调剪枝后模型
    结果：20% pruning rate 下，DeepSeek 15.7B→13.0B，MMLU 仅降 1.4%（vs Random 降 64%）；Qwen 14.3B→11.8B，保留 88% MMLU。GSM8K 上 C-PRUNE 反超 base model（33.56 vs 32.21）。

  - **系统框架层**: 论文未明确说明推理框架。C-PRUNE 在 HuggingFace transformers 上实现，对 model 的 expert parameters 进行修改（合并/删除）。不涉及 Serving 框架的调度修改。

  - **编译框架层**: 论文未明确说明（使用 PyTorch 标准推理流程）。剪枝后的 MoE 模型可无缝加载到标准推理框架中，因为 expert 合并后模型结构不变（仍是 MoE FFN layers，仅 expert 数量减少）。

  - **Kernel/运行时调度层**: 论文未明确说明具体 kernel 实现。剪枝后 routed experts 减少（64→52），每个 token 的 MoE 计算量减少（top-8 from 64 candidates vs from 52 candidates），但 top-k gating + 分组 GEMM 的 kernel 模式不变。论文提到 1.2× 推理加速（来自 expert 数量减少带来的计算量下降）。

  - **硬件架构层**: 32× NVIDIA A100 80GB GPU，无自定义硬件。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Seer Prune):
  Calib Data → Gate Forward → Act Count → Sort by Freq → Discard Low-Freq Experts → Pruned Model
  
  C-PRUNE (Ours):
  Calib Data → Expert Forward → Expert Embedding φ(f_i) 
  → Cosine Affinity Matrix A → Hierarchical Clustering (per-layer)
  → Weighted Expert Merging (within clusters) → Global Importance Scoring (cross-layer)
  → Global Pruning + Routing Update → Pruned Model
  → (Optional) Task-Specific Fine-tuning
  ```

## C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

- baseline方法是什么？
  **Baseline 为 MoE LLM 的默认 Router（端到端训练的 top-k gating）**：pretraining 阶段与模型参数联合训练的 router/gate 为每个 token 选择 top-k experts，形成固定的 expert pathway。推理时，router 权重冻结，所有样本共享同一套路由逻辑。

  **Baseline 全栈执行例子（以 OLMoE-1B-7B, 16 layers, 64 experts/layer, top-8 激活为例）**：

  - **算法层**: 输入 token x 经过第 l 层 → Gate 计算 routing logits = x · W_gate (shape: [64]) → Softmax → Top-8 选择 → 仅被选中的 8 个 expert FFN 处理 → h_l = x + Σ_{j∈top8} w_j · Expert_j(x)。所有 16 层使用相同的预训练 router，路径固定不可调。
  - **系统框架层**: 论文未明确说明推理框架。C3PO 通过替换 HuggingFace transformers 中的 `olmoe_modeling.py`，在 forward 时注入修改后的 routing weights。无 Serving 框架层面的调度修改。
  - **编译框架层**: 论文未明确说明（使用 PyTorch + HuggingFace transformers 标准推理）。
  - **Kernel/运行时调度层**: 论文未明确说明具体 kernel 实现。MoE 层使用标准 sparse MoE kernel（top-k gating + 分组 GEMM）。C3PO 在算法层面修改 routing weights，不涉及 kernel 修改。
  - **硬件架构层**: 论文未明确说明具体 GPU 硬件。

  **Baseline 的核心缺陷**：
  1. **Router 次优性 (Sub-optimality)**：预训练的 end-to-end router 对困难样本或分布外样本产生次优的 expert pathway，导致显著的 accuracy gap。实验表明 base model 与 Oracle（知道 ground truth 的最优 routing）之间存在 10-20% 的 accuracy gap（Table 1: OLMoE base 69.9% vs Oracle 85.2%，gap=15.3%；DeepSeekMoE base 66.4% vs Oracle 80.8%，gap=14.4%）。
  2. **静态路由缺乏样本级自适应性**：所有测试样本使用同一套预训练 router，无法根据具体样本特征动态调整 expert 选择。
  3. **Expert 利用不充分**：大多数 expert 被欠利用（仅 12-20 个 expert 被频繁激活），路由缺乏 specialization 导致计算资源浪费。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: C3PO = Critical-Layer (关键层选择) + Core-Expert (核心专家选择) + Collaborative Pathway Optimization (协同路径优化)。核心思路是 test-time adaptation——对每个测试样本，利用参考集中相似样本的 "成功 pathway" 来优化当前样本的 routing weights，无需模型参数更新。

  **Defect → Design 映射**：

  | Baseline 缺陷 | C3PO 设计选择 | 解决机制 |
  |---|---|---|
  | Router 次优性导致 10-20% gap | NGD: 用邻居 loss 作 surrogate objective 做梯度下降优化 ω | 将 routing weights 从固定值变为可优化变量，test-time gradient-based search 逼近 Oracle 性能（NGD 可达 Oracle 的 85-95%） |
  | 静态路由缺乏样本适应性 | Collaborative Pathway Optimization: 基于 embedding 相似度检索 k=3 邻居，用邻居的 successful pathway 指导优化 | 每个样本获得定制化的 routing weights，动态适应样本特征 |
  | Expert 利用率低、缺乏 specialization | Core-Expert: 只优化 top-20 experts 的 routing weights（覆盖最终 top-8 的 99.8%）；Critical-Layer: 只优化最后 5 层 | 激活更集中（Figure 7），强化高频 expert 的 specialization，减少冗余 |

  **论文方法全栈执行例子（OLMoE-C3PO, NGD 变体, k=3, Gaussian kernel, 10 steps, last token）**：

  - **算法层**: 
    1. 测试样本 x 用 NV-Embed-V2 获取 embedding E(x)
    2. 在参考集中 kNN 检索 k=3 个相似样本 {(x_i, y_i, ω_i)}
    3. 提取 x 的初始 ω_0（仅最后 5 层, top-20 experts）
    4. NGD 10 步迭代: L = Σ K_gaussian(x_i, x)·ℓ(f(x_i, ω), y_i) / Σ K_gaussian(x_i, x)，cosine LR 1e-2→1e-5，更新 ω
    5. 用优化后的 ω* 推理 f(x, ω*)
  - **系统框架层**: 替换 HuggingFace transformers 的 `olmoe_modeling.py`，注入优化后的 routing weights。通过 `olmoe_optimizer.py` 执行优化流程。未修改 Serving 框架。
  - **编译框架层**: 论文未明确说明。
  - **Kernel/运行时调度层**: 论文未明确说明。路由权重在算法层面被修改，不改变底层 MoE kernel 的执行方式。
  - **硬件架构层**: 论文未明确说明具体 GPU 硬件。

## BrownoutServe SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

- baseline方法是什么？
  **Baseline 为 vLLM**：一个高性能 LLM 推理引擎，支持 ContinuousBatching、FlashAttention、PagedAttention 等优化技术。vLLM 对 MoE 模型采用静态配置，所有 token 的 expert routing 为标准的 top-k gating——每个 token 经过 gate 后由 top-k 个原 experts 处理（zero-brownout），所有 experts 全部参与计算而不做任何降级。

  **Baseline 全栈执行例子（以 Qwen1.5-MoE-A2.7B-Chat, 60 experts/layer, batch_size=64, 4×A100-40GB 为例）**：

  - **算法层**: 输入 token x_t → Gate 计算 s_{i,t} = x_t^T · e_i → Top-2 Softmax routing → 60 个 experts 中被选中的 top-2 expert 处理 → h_t = x_t + Σ FFN_i(x_t)。所有 expert 按实际路由需求参与计算，不做降级。
  - **系统框架层**: vLLM Scheduler (FCFS) → ContinuousBatching 管理 batch → PagedAttention 管理 KV cache → Fused MoE kernel 执行 expert FFN。负载高峰时等待队列积压，请求延迟增大，可能出现 SLO 违规。静态配置下无自适应调节能力。
  - **编译框架层**: 论文未明确说明（vLLM 使用 PyTorch + CUDA/C++ kernel）。
  - **Kernel/运行时调度层**: CPU 端 block table → CPU→GPU 传输 → FlashAttention kernel → Gate kernel → Fused MoE kernel（多 expert FFN 合并为一次 sparse GEMM）。MoE 阶段成为瓶颈——prefill 阶段 MoE 占 transformer layer latency 81.23%，decoding 阶段占 93.89%（Fig. 1）。
  - **硬件架构层**: 4× A100-PCIE-40GB GPU，60 experts 分布在各 GPU 上。Cold experts（少数 token 路由到的 expert）无法充分利用 GPU 并行性——token 分布呈长尾模式，少量 hot expert 处理大量 token，多数 cold expert 利用率低。bursty workload 下 GPU 资源紧张，水平扩展（Kubernetes HPA/新实例）冷启动延迟 30-90s 模型加载 + 30-60s 实例初始化，无法及时响应。

  **Baseline 的核心缺陷**：
  1. **Expert 负载不均衡导致 GPU 利用率低**: 仅少量 hot experts 处理大量 token，cold experts 的 GPU SM 空闲，token 分布的 long-tail pattern 导致资源浪费。
  2. **Bursty workload 下静态配置无法自适应**: 突发流量时无降级机制，请求排队延迟急剧上升，SLO 违规严重。传统水平扩展冷启动延迟 1-2 分钟，不适用于短时突发。
  3. **MoE 模块是推理瓶颈**: prefill 阶段 MoE 占 81.23%，decoding 占 93.89%，但 vLLM 未针对 MoE 做专门的延迟优化。
  4. **全 expert 激活的巨大计算压力**: 大 batch 下 quasi-dense 激活，所有 expert 参与计算，通信带宽和计算资源压力大。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: BrownoutServe = United Expert 知识蒸馏 + Brownout 路由算法 + SALC 动态阈值调节。核心思路是借鉴电力系统的 brownout（降级供电）概念，在推理时用精度换延迟：将冷门 experts 的 token 交由 united experts 处理，减少 expert 访问次数，通过 SALC 算法动态维持延迟在 SLO 范围内。

  **Defect → Design 映射**：

  | Baseline 缺陷 | BrownoutServe 设计选择 | 解决机制 |
  |---|---|---|
  | Expert 负载不均衡 → cold expert GPU 利用率低 | United Experts 合并多 expert 知识，partial-brownout 将 cold expert 的 token 聚合到 united expert | 减少 expert 访问次数（如 8 experts → 5 次访问），增大每个被访问 expert 的 batch size，提升 GPU 并行度 |
  | Bursty workload 下无法自适应 | SALC 算法：P90 latency < warning_line → threshold↑（提升精度）；P90 latency > SLO → threshold↓（降级保延迟） | 闭环反馈控制，threshold 在 [0,1] 动态调整，突发时自动触发 brownout，突发结束后恢复 |
  | MoE 模块是延迟瓶颈 | Brownout 减少 expert 访问次数，Triton 重写 MoE kernel，优化 PagedAttention block table | MoE latency 随 expert 访问次数线性下降；Triton kernel 进一步提升效率 |
  | 全 expert 激活的计算压力 | Brownout threshold 控制参与计算的专家比例 | threshold=0.4 时仅 37.5% experts 直接参与，其余由 united experts 替代 |

  **论文方法全栈执行例子（以 Qwen1.5-MoE-A2.7B-Chat, partial-brownout, way=8, threshold=0.4, bursty workload 为例）**：

  - **算法层**: x_t → Gate 计算 affinity → Top-K routing → 统计各 expert token 数 → 降序排列 → 前 40% token 由原 experts 处理（S1），后 60% token 按 way=8 分组由 united experts 处理（S2）→ h_t = x_t + ΣFFN^{(s)} + Σp_{i,t}·FFN^{(r)} + Σq_{i,t}·FFN^{(u)}。United experts 通过离线 MSE 蒸馏训练：L_MSE^j = (1/k) Σ ||H_u^j - H_o^{j·k+i}||²。
  - **系统框架层**: BrownoutServe Scheduler (FCFS + ContinuousBatching) → SALC 每 iteration 监控 P90 latency → 对比 SLO warning line 和 SLO → 动态更新 threshold。例如：突发前 threshold=0.8，delay<warning_line → threshold 逐渐增至 0.9；t=75s 突发 → latency 超 SLO → threshold × 0.8 降至 0.72 → 更多 token 走 united experts → latency 回落到 warning_line~SLO 之间 → 突发后逐步恢复 threshold。
  - **编译框架层**: 论文未明确说明（PyTorch + Triton kernel 编译为 GPU 代码）。
  - **Kernel/运行时调度层**: GPU 端 block table → FlashAttention → Gate kernel → Brownout 划分 kernel（GPU sort + partition）→ S1 Fused MoE kernel（原 experts）→ S2 Triton United Expert kernel（concat tokens → FFN）→ 输出。PagedAttention block table GPU 化消除 CPU→GPU 传输。
  - **硬件架构层**: 4× A100-PCIE-40GB，united experts 权重常驻 GPU 显存（与原 expert 同参数规模，总 united experts 数远少于 experts），无需 extra GPU 资源。way 切换时需通过 Experts Loader 重新加载不同的 united experts 权重（GPU memory ↔ CPU memory/disk），但 threshold 调整 zero-overhead。


## Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

- baseline方法是什么？
  **Baseline 为 Token-level MoE (Token-MoE)**：在 Multilingual NMT 中，MoE Transformer 的 encoder 和 decoder 都使用 token-level routing——每个输入 token x_s 独立经过 router GATE(x_s) 计算 top-2 experts，不同 token 可能路由到不同的 experts。模型使用 top-2 gating，E 个 experts 的 FFN 替换 Transformer 的 alternate layers。训练时使用 auxiliary load balancing loss 确保 expert 利用率均衡。

  **Baseline 全栈执行例子（以 Token-MoE 32 experts WMT En→Fr 翻译为例）**：
  - **算法层**: 输入 token "Bonjour" → GATE(x_s) = TopK(Softmax(W_g · x_s), k=2) → 不同 token 独立选 expert（"Bon"→expert 3,7; "jour"→expert 12,5; "##s"→expert 1,8）→ y_s = G[3]·FFN_3(x_s) + G[7]·FFN_7(x_s)。Decoder 自回归每步：每个新 token 重新 router → 可能路由到不同的 expert，需要加载全部 32 experts 的动态通信。
  - **系统框架层**: GShard (TensorFlow/Lingvo) → Expert Parallelism: 32 experts 分布在多个 TPU 设备上 → all-to-all dispatch tokens → expert FFN 计算 → all-to-all combine → 自回归解码每步重复 all-to-all → 通信开销占总 step time 26.9%。
  - **编译框架层**: 论文未明确说明（Google 内部 TensorFlow/XLA 编译）。
  - **Kernel/运行时调度层**: 每个 TPU core 加载部分 experts → Router kernel → All-to-all token dispatch kernel → FFN GEMM kernel → All-to-all combine kernel → 每 decoding step 重复。
  - **硬件架构层**: 32 Cloud TPU V3 cores → Decoder 221M params 需跨多 TPU 设备 → dynamic routing 导致跨设备通信 → 解码器每步时间是 encoder 的 200x → peak throughput 1.3×10^5 tokens/s。

  **Baseline 的核心缺陷**：
  1. **Decoder 参数膨胀**: 全部 E experts 需常驻或可访问 → decoder 221M-6.5B params，超出单加速器内存 → 必须模型并行。
  2. **自回归解码通信放大**: 每 decoding step 都需要 all-to-all 通信 → 通信开销 26.9%-36% × N_decoding_steps。
  3. **小 batch 设备利用不足**: 小 batch 下只有少量 expert 被激活 → 大量设备空闲。
  4. **蒸馏损失**: 蒸馏 Token-MoE → Dense 模型仅保留 32% BLEU 增益。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: Task-level MoE (Task-MoE) = 将 MoE 路由从 token-level 改为 task-level，使同一 task（language pair / target language）的所有 token 路由到相同的 experts。配合 hybrid 策略：encoder 用 token-level routing（保持灵活性），decoder 用 task-level routing（dominant inference cost）。

  **Defect→Design 映射**：

  | Baseline 缺陷 | Task-MoE 设计选择 | 解决机制 |
  |---|---|---|
  | Decoder 参数膨胀需模型并行 | Task-level routing: 每 task 仅需 K=2 experts | Decoder 参数从 221M→25M (↓88%), 6.5B→201M (↓97%)，单加速器即可容纳 |
  | 每 decoding step all-to-all 通信 | 所有 token 路由到相同 experts（同设备） | 通信开销从 26.9%→0.0% (WMT), 36%→0.2% (large-scale) |
  | 小 batch 设备利用率低 | 每 task 独立部署 sub-network | 不同 task 可在不同设备上独立并行解码 |
  | 蒸馏仅保留 32% BLEU 增益 | Sub-network extraction 保留 100% 增益 | Task-MoE BLEU 29.0 vs Distillation 26.9 (+2.1) |
  | Token-MoE encoder 灵活性丧失 | Hybrid: Token encoder + Task decoder | Encoder 保持 token-level 灵活性（Xx-En 更好），decoder 获得 task-level 效率（decoder 占 200x 时间） |

  **Task-MoE 全栈执行例子（以 WMT En→Fr 翻译，32 experts，hybrid Token/Target 策略为例）**：

  - **算法层**: 
    Encoder: x_s (source token) → GATE(x_s) = TopK(Softmax(W_g · x_s), k=2) → token-level routing → 不同 source token 可走不同 expert → 输出 source hidden states。Decoder: target language "Fr" → task_emb = Embedding("Fr") → GATE(task_emb) = TopK(Softmax(W_g · task_emb), k=2) → 如 expert 5, 17 → 所有 decoder tokens 走 expert 5 + 17 → y_s = G[5]·FFN_5(x_s) + G[17]·FFN_17(x_s)。Task boundary 由 target language 定义（French→English 和 German→English 同 task "English" 选相同 experts），或由 language pair 定义（各自独立选 experts）。

  - **系统框架层**: Task-specific sub-network 提取 → 每个 task 仅加载 K=2 experts 到单 TPU device → 无 all-to-all → 无跨设备通信 → Decoder 前向: Router (task_emb) 一次计算，所有 decoder step 复用 → Expert FFN 计算（仅 2 experts）→ peak throughput 2.3×10^5 tokens/s。

  - **编译框架层**: 论文未明确说明（Google 内部 TensorFlow/XLA）。

  - **Kernel/运行时调度层**: Router kernel: task_emb lookup → Softmax → TopK（仅需运行一次 per task，非 per token）。FFN kernel: 仅 2 experts GEMM（非全部 32/128）。无 all-to-all dispatch/combine kernel。Decoder 自回归每步 kernel 执行路径缩短 → peak throughput 1.87x-2.6x。

  - **硬件架构层**: 32-128 Cloud TPU V3 cores → 每 task 分配专用 sub-network → 不同 task 可在不同 core groups 独立解码 → communication overhead 0.0%-0.2% → decoder step time 大幅缩短。

  **路由决策分析（Section 5.4）**：对 Token-MoE 的 gating decisions 可视化发现——decoder 中 task-level 决策自然出现（related languages share similar expert distributions，如 Spanish-Catalan, Russian-Ukrainian），encoder 中所有 Xx-En 任务偏好相同的少数 experts。这为 hybrid 策略（Token encoder + Task decoder）提供了实证支持。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Task-level routing on decoder | Decoder 参数从 221M→25M (WMT), 6.5B→201M (200 langs) | Peak throughput 1.87x (WMT), 2.6x (200 langs) |
  | Hybrid Token encoder + Task decoder | Encoder 保持灵活性，decoder 获得效率 | BLEU 23.6 (vs Token/Token 22.6, +1.0) |
  | Target language 作为 task boundary | 最大化 transfer: xx→En 共享同组 experts | Target/Target BLEU 22.9 vs LanguagePair/LanguagePair 21.4 |
  | Sub-network extraction 替代蒸馏 | 100% 保留 MoE BLEU 增益 vs 蒸馏 32% | Task-MoE 29.0 vs Distillation 26.9 BLEU |
  | Per-task expert 不共享 | 不同 task 加载不同 sub-network 独立解码 | No communication overhead (0.0-0.2% vs 26.9-36%) |

  **创新总结**: Task-MoE 的核心洞察是——在 Multi-task learning（MNMT）场景中，task boundary 是已知先验，可以直接利用来替代 token-level routing 的 dynamic selection。通过将路由决策从 "per token" 提升到 "per task"，将不可控的 token 级动态通信转化为可控的 task 级静态部署，从而在不蒸馏、不量化、不剪枝的情况下实现高效推理。其局限在于仅适用于 task boundary 明确的多任务场景（如 multilingual NMT），不适用于通用单任务 LLM。

## Branch-Train-MiX Mixing Expert LLMs into a Mixture-of-Experts LLM

- baseline方法是什么？
  BTX 对比两类 baseline：

  **(1) Branch-Train-Merge (BTM)**：复制 seed 模型为多个 domain expert，各 expert 在领域数据上独立训练。推理时通过 TF-IDF 计算输入与各 expert 训练数据的相似度，选 Top-k expert 模型的输出 logits 做平均 ensemble。执行流程：输入 prompt → TF-IDF 嵌入 → cosine similarity 选 expert → 各选中 expert 独立 forward → 输出 logits 加权平均 → 预测 token。BTM 是特殊的 BTX（100% compute 给 expert training，0% 给 MoE finetune）。

  **(2) Sparse Upcycling**：从 seed dense checkpoint 将每层 FFN 复制为多个 identical expert，随机初始化 router，然后在混合数据上做 MoE 训练。这是 BTX 的另一特殊形式（0% expert training，100% MoE finetune）。执行流程：seed dense FFN → 复制为 4 个 identical expert → + 随机 router → Top-2 MoE training on mixed data → 统一 single model。

  **Baseline 全栈执行例子（以 BTM 处理 math/code 混合输入为例）**：
  - **算法层**：输入 prompt → TF-IDF 嵌入 → cosine sim 选 Top-2 expert（Math + Code）→ Math expert 独立 decoder forward（32层 Llama-2 FFN）→ Code expert 独立 decoder forward（32层）→ 两个 logits 向量直接平均 → argmax→输出 token。各 expert 之间无信息交换，无 token 级路由。
  - **系统框架层**：论文未明确说明（标准 PyTorch forward，无特殊 serving 框架）。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：论文未明确说明。
  - **硬件架构层**：论文未明确说明（Meta FAIR 训练集群，GPU 细节未给出）。

  **Baseline 的核心缺陷**：
  1. **BTM 无统一模型**：expert 独立存在，不能做 SFT/RLHF 等后续微调（对齐 LLM 的关键步骤）
  2. **BTM 路由粗糙**：TF-IDF 整句级路由，无法做 token 级细粒度 routing，不同 token 需要的 expert 组合不同
  3. **BTM 无学习路由**：TF-IDF 相似度是静态的，无法学习最优 token→expert 映射
  4. **Sparse Upcycling 同步训练**：全部 compute 用于 MoE 训练，all-to-all 通信成本随 expert 数增长，训练吞吐低
  5. **Sparse Upcycling 无领域专门化**：expert 从同一 checkpoint 复制，无 domain specialization，性能不均衡

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BTX 将 embarrassingly parallel 的 expert training 与 MoE finetune 结合：先异步独立训练 domain expert（继承 BTM 的训练效率），再将 expert FFN 组合为 MoE 层并 finetune router（解决 BTM 的无统一模型和无学习路由问题），同时 attention 参数平均（假设 attention 层 domain specialization 弱于 FFN）。

  | Baseline 缺陷 | BTX 设计选择 | 解决机制 |
  |---|---|---|
  | BTM 无统一模型，无法 SFT/RLHF | MoE finetune 阶段将所有 expert 合并为统一 MoE LLM | 最终模型是标准的 MoE Transformer，可直接做 SFT/RLHF |
  | BTM TF-IDF 整句路由粗糙 | MoE router 做 token 级 Top-k routing: g(W_l x) | 每个 token 独立选 top-2 expert，同一序列可用所有 domain expert |
  | BTM 路由无学习 | 随机初始化 W_l 然后 MoE finetune 训练 | Router 通过 80B tokens finetune 学习最优 token→expert 映射 |
  | Sparse Upcycling 同步训练通信高 | Expert 训练阶段 embarrassingly parallel | 无 all-to-all 通信直至 MoE finetune 阶段，训练吞吐线性 scaling |
  | Sparse Upcycling 无 domain specialization | 每个 expert 在独立领域数据上单独训练 | Math expert 在 math 任务上从 2.5→18.8 (MATH)，Code expert 从 12.8→31.7 (HumanEval) |
  | Expert 训练导致 catastrophic forgetting | 保留 Seed 模型作为 generalist expert + MoE finetune 混合所有数据 | BTX 在 reasoning 上 63.5 vs seed 63.3（无退化），Knowledge 41.0 vs 37.4（+3.6） |

  **BTX 方法全栈执行例子（以 Llama-2 7B seed + 4 experts + Top-2 routing 推理一个 token 为例）**：

  - **算法层**：
    token x 进入 layer l → Attention: 使用 4 expert 的平均 attention 权重 (W_q, W_k, W_v, W_o 均平均) → Router: logits = x @ W_l [4096, 4] → TopK(logits, k=2) → SoftMax(top2_vals) → 激活 2/4 FFN experts → y = w_1·FFN_math(x) + w_2·FFN_code(x) → 输出。Math 输入时 router 偏好 Math/Code expert，Knowledge 输入时偏好 Wiki/Llama-2 expert，Reasoning 时均衡使用 Math/Llama-2 expert。
  - **系统框架层**：论文未明确说明（标准 PyTorch MoE forward）。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：MoE 层仅计算 top-2 experts 的 FFN GEMM（4 个 expert 中 2 个），激活参数 11.1B，总参数 4×7B=28B。
  - **硬件架构层**：论文未明确说明（Meta FAIR 训练集群，GPU 细节未给出）。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Embarrassingly parallel expert training | Sparse upcycling 同步训练通信开销大 | Training time 7.8 GPU-days, BTX 训练 533B tokens vs sparse upcycling 252B tokens (同 compute) |
  | Expert FFN → MoE layer (仅 FFN 做 expert) | Attention 层 domain specialization 弱 | Freeze experts 后仅训练 router+attention: 性能几乎不变 (34.7 vs 34.7) |
  | Self-attention 权重平均 | 避免 attention 参数膨胀 | 无新增 attention 参数，与 seed model 结构完全兼容 |
  | Top-2 routing with load balancing (α=0.01) | Dead expert 问题（Code expert 无负载均衡时几乎不被激活） | Load balancing 使 Code expert 从 dead→在 math/code domain 主导 |
  | 保留 seed model 为 generalist expert | 防止 catastrophic forgetting，保留原有 general knowledge | Knowledge 41.0 vs seed 37.4 (+3.6), Reasoning 63.5 vs 63.3 (+0.2) |
  | MoE finetune 80B tokens (vs expert training 200B+ tokens each) | 仅用少量 compute 学会 router 和调优平均的 attention | BTX 47.9 vs BTM 43.4 (+4.5 average), BTX 47.9 vs Sparse Upcycling (DM) 46.3 (+1.6) |

  **创新总结**：BTX 的核心洞察是将 LLM 能力提升分解为两个解耦阶段——(1) expert training 阶段通过 embarrassingly parallel 训练最大化 compute 投入产出（线性 scaling），(2) MoE finetune 阶段将分离的 knowledge source 通过可学习的 token-level router 融合为统一模型。这使 BTX 同时获得 BTM 的训练效率和 MoE 的统一模型优势，避免了 BTM 的"无统一模型"和 sparse upcycling 的"同步训练通信高"两个极端。Blending experts 实验（将各 domain FFN 分块再混合）导致的性能大幅下降（Average 34.7→22.2）验证了"保留 domain specialization + 学习 router"而非"强制混合 domain knowledge"的设计决策正确性。

## BTS Harmonizing Specialized Experts into a Generalist LLM

- baseline方法是什么？
  Baseline 方法分为两大类：
  1. **Expert Merging（无学习连接的合并）**：
     - BTM (Branch-Train-Merge)：对 Seed 和各 Expert 的输出 logits 做 Bayes 规则加权 ensemble，不做任何训练。执行流程：输入 → 每个模型独立 forward → 输出 logits → 加权平均 → 输出 token。
     - Model Soup：直接对 Seed 和各 Expert 的权重做均匀平均。执行流程：参数空间线性插值 → 合并后的单模型 forward。
     - Expert Routing：训练一个线性路由器 ∈ R^{dim×n}，基于 prompt 平均 embedding 选择 Seed 或某个 Expert 处理整个序列。执行流程：输入 embedding → 路由器分类 → 选择单模型 → 该模型处理全部 token。
  2. **Expert Upcycling（破坏模块性的 MoE 转换）**：
     - BTX：将 Seed 和 Expert 的 FFN 拷贝为 MoE Expert，训练全部参数。执行流程：输入 → Attention → Router 选 Expert(s) → 加权 FFN 输出。
     - BAM：将 Attention 和 FFN 都改为 MoE/MoA 结构（所有参数参与训练）。

  全栈执行例子（以 BTM 为例）：
  - 算法层：输入 prompt → Seed + Code Expert + Math Expert + Multilingual Expert 各自 forward（每个 20 层 Transformer）→ 各模型分别计算 4 个 logit 向量 → Bayes 加权 ensemble → argmax 输出下一个 token。
  - 系统框架层：论文未明确说明（推理使用标准 PyTorch forward，无特殊 Serving 框架）。
  - 编译/kernel/硬件层：论文未明确说明。

  Baseline **核心缺陷**：BTM/Model Soup 在 Expert 之间**缺乏可学习的中间表示连接**，合并是仅在输出层/参数空间的静态合并，表达能力受限，尤其在跨领域任务（cross-capability）上表现差。BTX/BAM 虽然通过 MoE 训练获得学习连接，但**破坏了模块性**（所有参数参与训练，Expert 不再保持完整独立），无法灵活增删 Expert。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  BTS 方法在保持 Expert 参数完全冻结（模块性）的前提下，通过在 Seed（Hub）和 Expert（Spoke）模型层之间插入并训练轻量 **Stitch Layer**，提供**可学习的中间表示连接**，实现 token 级粒度的 Expert 表示融合。

  **设计对缺陷的映射**：
  | Baseline 缺陷 | BTS 设计选择 |
  |---|---|
  | BTM 无学习连接 → 表达能力弱 | 插入可训练的 stitch 层 w_gate + w_proj（264M 参数） |
  | BTX/BAM 破坏模块性 → Expert 不可增删 | 仅训练 stitch 层，Expert 完全冻结 |
  | Expert Routing 整序列级选单模型 → 不能 context-switch | Stitch 层在每个 token 重新计算 gate → token 级 Expert 动态选择 |
  | 无 Cross-capability（交叉领域） | 交替 Experts-into-Hub / Hub-into-Experts 架构使 Expert 之间双向信息流动 |

  **全栈执行例子（BTS 推理流程）**：
  - 算法层：输入 prompt → Layer 1-4：Seed/Experts 各自 forward → Stitch Layer 1（Hub-into-Experts）：对每个 Expert，sigmoid gate 控制 Hub 信息注入比例 → Layer 5-9 forward → Stitch Layer 2（Experts-into-Hub）：softmax gate 控制各 Expert 投影到 Hub 空间并加权合并 → Layer 10-14 forward → Stitch Layer 3（Hub-into-Experts）→ Layer 15-19 forward → Stitch Layer 4（Experts-into-Hub）→ Hub 最后层输出 → LM head → token 预测。Gate 值可视化验证：Math 任务时 Math Expert gate → 1，Translation 任务时 Multilingual Expert + Seed gate 交替激活，Context-switching 场景下 gate 动态切换。
  - 系统框架层：论文未明确说明（使用标准 PyTorch forward，无需修改 Serving 框架）。
  - 编译/kernel/硬件层：论文未明确说明。

- baseline方法是什么？
  **Baseline**: DeepSpeed-MoE 的 **expert parallelism (EP)** : 每 GPU 持有若干完整 expert（全部 W_i, W_o 矩阵），self-attention 和 router 层复制。Forward pass: (1) 每 GPU 独立执行 router，分配 token→expert; (2) all-to-all scatter 将 token 发送到持有对应 expert 的 GPU; (3) 各 GPU 本地执行 expert FFN (x · W_i · W_o); (4) all-to-all gather 结果回源 GPU。使用 capacity factor (CF) 限制每 expert 最大 token 数，超限 token 被丢弃。

  **Baseline 缺陷**:
  1. **Load Imbalance**: 真实推理中 expert popularity 高度倾斜——以 Switch 128-expert 模型为例，最后一层 14 个 expert 收不到任何 token，最繁忙 expert 收到 3105 tokens。导致部分 GPU 过载、部分 GPU 空闲，端到端延迟由最繁忙 GPU 决定。
  2. **Token Dropping**: 使用 CF 缓解不平衡会丢弃超限 token，直接损害模型精度。
  3. **专家复制方案的开销**: 其他方案（Lazarus, Prophet）通过复制热门 expert 到多 GPU 平衡负载，但需要 profiling、重调度和额外 GPU 内存。
  4. **Batch 越大越严重**: 即使 router skew 参数固定，batch size 增大时 token 分配的绝对差异也增大，imbalance 方案（如 DeepSpeed）中 GPU idle time 绝对值增加。

  **Baseline 全栈执行例子（以 4 GPU, 128 expert Switch-Base encoder 推理一个 MoE block 为例）**:
  - **算法层**: Switch Transformer top-1 gating, 128 FFN experts → token→expert 路由
  - **系统框架层**: DeepSpeed-MoE expert parallelism → 4 GPU 各持有 32 个完整 expert → all-to-all scatter/gather 通信原语 → CF=min(128, 50)=50 限制 token 数
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + NCCL all-to-all）
  - **Kernel/运行时调度层**: 每 GPU 对持有的 32 个 expert 执行完整矩阵乘法 x·W_i·W_o（单个或多个 kernel launch），GPU 间负载不均导致部分 GPU kernel 提前完成等待 all-to-all gather barrier
  - **硬件架构层**: 4× A100 80GB NVLink 互联 → 最忙 GPU 处理最多 token（SM 全占用），最闲 GPU 提前 idle（SM 空闲等待 all-to-all barrier）

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: MoEShard = **Expert Tensor Sharding (TS)** 替代 Expert Parallelism: 将每个 expert 的 W_i 列切分、W_o 行切分到所有 GPU → 每 GPU 持有所有 expert 的 partial shard → 所有 GPU 处理所有 token 的 partial computation → pointwise sum 恢复完整输出。配合两个 kernel 优化: (a) per-expert token concatenation 减少 kernel launch 数; (b) MegaBlocks block-sparse MM 将全部 expert shard 计算融合为单次操作。

  **Defect→Design 映射**:

  | Baseline 缺陷 | MoEShard 设计选择 | 解决机制 |
  |---|---|---|
  | 路由倾斜导致 GPU 负载不均 | W_i 列切分 + W_o 行切分的 expert tensor sharding | 所有 GPU 处理完全相同数量的计算（全部 token × 全部 expert shard），天然 perfect load balancing |
  | CF 丢 token 损害精度 | 所有 token 全程保留 | 每 token 在所有 GPU 上参与 partial computation 并最终求和，零 token dropping |
  | 专家复制需要 profiling 和额外内存 | 无专家复制，无 profiling | 每 GPU 只需每个 expert 的 1/|G| 列/行 shard，总参数量与 EP 相同 |
  | batch 增大加剧 idle time | 计算量与 batch size 线性 scaling | 所有 GPU 的计算量始终相等，无论 batch 多大 |

  **MoEShard 全栈执行例子（以 4 GPU, 128 expert Switch-Base encoder, batch=250, seq=120, h=768 推理一个 MoE block 为例）**:

  - **算法层**: 同 Baseline —— Switch Transformer top-1 gating, 128 FFN experts。区别：sharding 而非 placement 策略改变，路由机制不变。
  - **系统框架层**: MoEShard 自定义 PyTorch forward pass (Algorithm 1) → 每 GPU 复制 router + self-attention → Step 2 metadata exchange (all-to-all broadcast per-expert token counts) → Step 3 token scatter (all GPU send all tokens, NVLink ~0.15ms) → Step 4 sharded expert computation (每 GPU 对每 expert 执行 x · W_i^g · W_o^g) → Step 5 gather + pointwise sum partial outputs → **无 all-to-all scatter/gather 的 barrier 等待**（全部 GPU 计算量相同，同时完成）
  - **编译框架层**: 论文未明确说明（PyTorch eager execution）
  - **Kernel/运行时调度层**: Fusion opt 1: per-expert token concatenation → 128 kernel launches (|E|) vs 512 (|E|×|G|)。Fusion opt 2: MegaBlocks block-sparse MM → 1 kernel launch 处理全部 expert shard。每 kernel 内 SM 计算均匀。
  - **硬件架构层**: 4× A100 80GB NVLink → 4 GPU SM 均计算 x·W_i^g·W_o^g (同数据量，同计算量) → 同时完成 → 直接进入 gather → 无 SM idle。NVLink 仅在 Step 3 scatter 和 Step 5 gather 使用（带宽充足）。

  关键洞察：MoEShard 通过将"按 expert 分配 GPU"改为"按 tensor 维度分配 GPU"，将不可控的路由倾斜问题转化为可控的均匀张量计算问题。代价是 token 全复制（NVLink 吸收）和 partial output 求和（pointwise addition, negligible）。

## Accelerating Distributed MoE Training and Inference with Lina

- baseline方法是什么？
  **Baseline**: DeepSpeed MoE 的混合并行（Data Parallelism + Expert Parallelism），使用独立的 CUDA streams 分别处理 all-to-all（expert-parallel 通信）和 allreduce（data-parallel 通信），不做跨 stream 协调。Inference 使用 uniform expert-device allocation（每 device 1 个 expert）。

  **Baseline 缺陷**:
  1. **Training 缺陷**: backward pass 中 all-to-all (Stream b) 与 allreduce (Stream c) 并发时公平共享 InfiniBand 带宽，all-to-all 是阻塞式操作（无法与计算并行），被延长 median 1.83x（worst 4.14x）。且 PyTorch DDP gradient bucketing 导致 allreduce 实际 bucket size 变化剧烈，无法预估精确 arrival/running time 做静态调度。
  2. **Inference 缺陷**: 真实推理请求下 expert popularity 高度倾斜（最 popular expert 收到 4.02x~5.56x tokens），uniform allocation 导致 popular expert device 过载，unpopular expert device 空闲（最大 idle time 29.4%）；且 all-to-all 的各 link 使用不均衡，带宽未充分利用。

  **Baseline 全栈执行例子（以 16-expert Transformer-XL Training 一个 MoE layer backward pass 为例）**:
  - **算法层**: top-2 gating, 16 FFN experts → Gate 输出 (token→expert) 映射
  - **系统框架层**: DeepSpeed MoE → 16 GPU (1 expert/GPU) → Data Parallelism (gradient allreduce) + Expert Parallelism (token all-to-all dispatch/combine)
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + NCCL 通信原语）
  - **Kernel/运行时调度层**: Stream a (FFN backward kernel) → 完成后梯度分入 Stream b (all-to-all, 完整大 tensor 一次发射) 和 Stream c (PyTorch DDP gradient bucketing → allreduce, 完整大 tensor 一次发射)，两 stream 并发，NCCL 底层 fair-share 带宽
  - **硬件架构层**: A100 SMs 计算 FFN backward → gradient 经 PCIe/NVSwitch 进入 IB HCA → InfiniBand 传输；GPU SM efficiency 在 all-to-all 期间仅 3.7%（大量空闲等待）

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: Lina = **Training 端**: micro-op priority scheduler (tensor partitioning + priority queue + pipelining) + expert packing; **Inference 端**: expert popularity estimation (token-level expert selection pattern profiling) + two-phase dynamic resource scheduling.

  **Defect→Design 映射**:

  | Baseline 缺陷 | Lina 设计选择 | 解决机制 |
  |---|---|---|
  | all-to-all 与 allreduce 无协调争抢带宽 | Tensor Partitioning (30MB micro-ops) + Priority Queue | all-to-all micro-op 始终优先，allreduce micro-op 仅 idle 时发射 |
  | DDP gradient bucketing → allreduce size 不可预测 | 每个 gradient 独立 partition，不跨 gradient 混合 chunk | 所有 micro-op 大小均匀，调度器可精确控制 |
  | 大 tensor 一次发射阻塞时间长 | Micro-op pipelining: all-to-all 也分区 → 每 micro-op 完成后即启动对应 token 的 FFN | 消除 bubble: FFN time 被 all-to-all 覆盖 |
  | FFN micro-op << all-to-all micro-op 导致 pipeline bubble | Expert Packing: 2^n 递增每 device expert 数 → FFN total time 对齐 all-to-all | Pipeline efficiency: 33%→86% (Transformer-XL) |
  | Inference 中 expert popularity 倾斜且无法提前获知 | Token-level expert selection pattern profiling + sample path estimation | 在 gate 执行前估算 expert popularity 做预调度 |
  | 事后调度（gate 后）阻塞过长 | Two-phase 调度: phase 1 (预调度, 与计算重叠) + phase 2 (少量微调, ~23% cases) | 调度 overhead 从每层 blocking 降为大部分被重叠 |
  | Uniform all-to-all 各 link 负载不均 | Unequal split all-to-all (按实际 token 量 split) | 匹配 popular expert link 高带宽需求 |

  **Lina 全栈执行例子（同 16-expert Transformer-XL Training backward pass 对比 baseline）**:
  - **算法层**: 同 baseline (top-2 gating, 不变模型精度)
  - **系统框架层**: DeepSpeed MoE + Lina Communication Scheduler → 修改 PyTorch DDP bucketing → gradient 不 fuse 而是独立 partition 为 30MB micro-ops
  - **编译框架层**: 论文未明确说明
  - **Kernel/运行时调度层**: 
    1. FFN backward 完成后 gradient tensor 入 priority queue
    2. Scheduler: `chunk(grad, 30MB)` → 5 micro-ops
    3. 若队列有 all-to-all micro-op → launch NCCL all-to-all → 等待完成
    4. 若队列无 all-to-all → launch allreduce micro-op（但 combine computation 阶段停止发射）
    5. All-to-all micro-op 1 完成 → 对应 tokens 进入 FFN → 覆盖计算延迟
    6. Expert Packing: 2 experts/device → FFN total time 增长至接近 all-to-all micro-op → pipeline efficiency 86%
  - **硬件架构层**: A100 SMs 在 all-to-all 期间不再全 idle（pipelining 使 FFN 计算重叠）→ GPU utilization +17.6%；all-to-all 获满 100Gbps IB 带宽 → all-to-all time speedup 2.21x

  **Lina Inference 全栈对比**:
  - Baseline: Gate → [Uniform All-to-All Dispatch] → Expert Compute (popular expert 过载) → [All-to-All Combine] → 尾部延迟拉长
  - Lina: Profile path patterns → Phase 1: 估算 popularity (layers 1-3 warm-up) → Phase 1 piggback on all-to-all → Scheduler compute mapping (popular expert → multi-device replica) → Expert swap → [Unequal All-to-All] → Balanced Expert Compute → Phase 2: check accuracy (~23% need re-schedule) → [Unequal All-to-All Combine] → median inference 1.45x faster, tail 95%ile 1.63x faster



## A Survey on Mixture of Experts in Large Language Models

- baseline方法是什么？
  本论文为综述，其隐含 baseline 是：在没有系统性 MoE 知识组织的情况下，研究者分散地探索 MoE 的算法、系统和应用，缺乏统一的分类框架指导设计选择。具体而言：
  **(1) 算法 baseline**：固定 top-k 门控（k=1 或 k=2）、标准 FFN 专家（无 fine-grained segmentation）、无共享专家、无 PEFT-MoE 融合，训练从头开始。
  **(2) 系统 baseline**：基础 expert parallelism（GShard 式 All-to-All dispatch/combine），无计算优化（无定制 GPU kernel）、无通信优化（无分层/拓扑感知）、无存储优化（所有 expert 常驻 GPU）。
  **(3) 应用 baseline**：MoE 仅用于 NLP 领域的大模型预训练，缺乏跨域（CV/Recommendation/Multimodal）的系统性应用指导。

  **Baseline 全栈执行例子（以 Mixtral-8x7B 推理一个 token 为例）**：
  - 算法层：top-2 token-choice gating（Linear-Softmax-TopK）→ 2 out of 8 FFN experts 激活 → 加权求和输出 → 无 shared expert → L_aux = 0.01 负载均衡
  - 系统框架层：Expert parallelism（每个 GPU 持有部分 experts）→ Gate Routing → All-to-All Dispatch → Expert Computation → All-to-All Combine → Output Decode
  - 编译框架层：论文未明确说明（标准 PyTorch 执行，无 MoE 专用编译）
  - Kernel 调度层：标准 cuBLAS GEMM kernel → 无 block-sparse 优化 → token dropping 由 expert capacity 限制
  - 硬件架构层：标准 NVIDIA GPU HBM → 无 offloading → 全部 expert 参数驻留显存

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本论文为**综述**，不提出新方法，而是建立三层分类学（Algorithm- System-Application Taxonomy，Figure 3），系统性地组织和对比现有 MoE 研究，并识别七项关键挑战（Section 7）。

  **综述对 baseline 缺陷的诊断与分类解决路径**：

  | Baseline 缺陷 | 综述识别的方法方向 | 代表性工作 |
  |---|---|---|
  | 固定 top-k 门控导致负载不均和训练不稳定 | 创新门控算法（Expert-Choice, BASE, DSelect-k）、软门控（SMEAR, Lory）、hash/随机门控 | Expert-Choice Gating [92], BASE [72], Lory [39] |
  | FFN 专家粗粒度、知识冗余 | Fine-grained expert segmentation、共享专家、新兴专家架构（MoA, MoH, LoRA experts） | DeepSeekMoE [67], Qwen1.5-MoE [102], MoA [80] |
  | 训练从头开始资源消耗大 | Dense-to-Sparse（Sparse Upcycling）、Sparse-to-Dense（蒸馏/剪枝）、Expert Models Merging（BTX） | Sparse Upcycling [47], BTX [52], DS-MoE [62] |
  | All-to-All 通信成为瓶颈 | 分层通信、拓扑感知路由、计算-通信重叠、架构解耦 | DeepSpeed-MoE [64], Lancet [143], ScMoE [108] |
  | 稀疏运算 GPU 利用率低 | 块稀疏 GEMM kernel（MegaBlocks）、PIT 编译器、ParallelLinear（ScatterMoE） | MegaBlocks [137], PIT [139], ScatterMoE [138] |
  | 专家参数超出单 GPU 显存 | 层级存储 offloading（GPU→CPU→SSD）、预测+预取、低精度加载 | SE-MoE [131], EdgeMoE [148], HOBBIT |
  | 跨域应用缺乏指导 | NLP → CV → Recommender Systems → Multimodal 系统化应用分类 | V-MoE [6], LIMoE [153], MMoE [59] |

  **综述方法论全栈执行例子**：
  本综述的方法论是通过三层分类学自上而下组织知识：
  - **算法层**：Gating Function（Sparse/Dense/Soft）× Expert Network（FFN/Attention/CNN/LoRA）× Training Scheme（Dense-to-Sparse/Sparse-to-Dense/Expert Merging）→ 构成 3×4×3 的设计空间
  - **系统框架层**：Computation（GPU kernel + 负载均衡放置）× Communication（分层 All-to-All + 拓扑感知 + 流水线重叠）× Storage（层级 offloading + 预取）→ 三维度覆盖系统全栈
  - **编译框架层**：论文提及 PIT 编译器（Permutation Invariant Transformation 变换 tile 为 dense 计算）但未深入展开
  - **Kernel 调度层**：Block-sparse kernel（MegaBlocks）、ParallelLinear grouped GEMM（ScatterMoE）、定制 encode/decode kernel（DeepSpeed-MoE/FastMoE/Tutel）
  - **硬件架构层**：层级存储（GPU HBM + CPU Memory + SSD），论文指出稀疏运算在硬件加速器上的非均匀性是关键挑战（Section 7）

## A Survey on Inference Optimization Techniques for Mixture of Experts Models

- baseline方法是什么？
  在MoE推理优化领域，没有单一的baseline。但在综述的分类框架中，隐含的baseline是：
  **(1) 模型级baseline**：标准MoE架构使用固定top-K路由（如top-2 gating），所有expert以FP16全精度存储，无剪枝、无量化、无蒸馏。
  **(2) 系统级baseline**：
  - Expert Parallelism baseline：DeepSpeed-MoE和FasterMoE（标准all-to-all通信，无特殊调度优化）
  - Expert Offloading baseline：Mixtral-Offloading（按层加载expert，简单LRU缓存，无预取）
  **(3) 硬件级baseline**：传统GPU架构针对稠密计算优化，缺乏对MoE稀疏激活和动态expert调度的硬件支持。

  **Baseline全栈执行例子（以Mixtral-8x7B推理一个token为例）**：
  - 算法层：top-2固定gating，2/8 expert激活 → FP16 FFN计算
  - 系统框架层：无offloading时全部expert常驻GPU显存；或简单按层加载expert（如Mixtral-Offloading）→ LRU缓存，无预取
  - 编译框架层：论文未明确说明（使用标准PyTorch/Transformers执行）
  - Kernel调度层：标准cuBLAS GEMM kernel，无专为稀疏expert优化的kernel
  - 硬件架构层：标准NVIDIA GPU SM架构，无NDP/PIM/FPGA加速

- 论文方法是什么？如何对应解决Baseline的缺陷？
  本论文是**综述**，本身不提出新方法，而是建立三层分类框架（模型级-系统级-硬件级），系统性地组织和对比现有方法，并识别关键挑战和未来方向。

  **综述对baseline缺陷的诊断与分类解决路径**：

  | Baseline缺陷 | 综述识别的方法方向 | 代表性工作 |
  |---|---|---|
  | 固定top-K浪费计算 | 动态门控（根据token复杂度自适应） | DynMoE、XMoE、AdapMoE |
  | 全精度expert占用过多显存 | 量化压缩（INT4/INT2/INT1） | QMoE、MC-MoE、MoQE |
  | 冗余expert浪费参数 | 剪枝/合并expert | TSEP、MoE-Pruner、MC-SMoE |
  | All-to-All通信瓶颈 | 分层通信、数据压缩、减少通信次数 | Tutel、ExFlow、Janus |
  | Expert加载延迟（offloading场景） | 预取+智能缓存+低精度加载 | HOBBIT、ProMoE、ExpertFlow |
  | 负载不均衡导致GPU闲置 | 性能建模+greedy搜索expert放置 | Prophet、FlexMoE、Lazarus |
  | GPU硬件对稀疏计算低效 | NDP/PIM/FPGA专用加速 | MoNDE、Duplex、FLAME |

  **综述方法论全栈执行例子（以优化后的MoE推理一个token为例）**：
  - **算法层**：动态门控根据输入复杂度自适应选择expert数量（非固定top-2）→ 量化后的INT4 expert权重进行低精度FFN计算
  - **系统框架层**：Expert cache中保留高频expert（LRU+LFU+LHU混合策略）→ 基于当前gate输出预取下层expert（跨层预测准确率~90%）→ CPU辅助处理低精度cold expert → GPU和通信流水线重叠
  - **编译框架层**：论文未明确说明
  - **Kernel调度层**：专用CUDA kernel处理量化权重的反量化+浮点计算（如MoE-CSP的4-bit kernel）→ FPGA上双缓冲expert权重加载（如FLAME）
  - **硬件架构层**：Hot expert在GPU执行 + Cold expert通过CXL发送到NDP核在LPDDR内执行（如MoNDE的Activation Movement模式）

## APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

- baseline方法是什么？
  **Baseline 为 Mobius（ASPLOS 2023）**，它是带宽受限 GPU 节点上结合流水线并行和 offloading 技术的微调系统。Mobius 将模型分区为多于 GPU 数量的 stage，相邻 stage 映射到不同 GPU，通过跨根复合体映射（cross-mapping）减少带宽竞争。在前向过程中，Mobius 在执行当前 stage 时预取下一 stage 的参数，完成后卸载参数和激活到 host memory；反向过程再从 host memory 预取参数和激活。

  **Baseline（Mobius）全栈执行例子（以 MoE 模型在一个 micro-batch 上的 forward 为例）**：
  - 算法层：标准 MoE top-2 gating（Linear-Softmax-TopK），所有 expert 在 GPU 上计算，无 CPU 参与计算，无 expert 热度利用
  - 系统框架层：流水线并行（GPipe-style stage partition）+ Offloading（host↔GPU）。每个 GPU 持有多个 stage，当前 stage 执行时预取下一 stage 全部参数到 GPU 内存，无选择性加载
  - 编译框架层：论文未明确说明（标准 PyTorch eager 执行）
  - Kernel 调度层：标准 CUDA stream 执行，无专门的数据移动调度优化。所有数据移动按 stage 粒度顺序执行，不同 stage 的数据移动可能互相阻塞
  - 硬件架构层：NVIDIA A800 GPU (40GB) × 4，PCIe Switch 互联，无 NVLink/InfiniBand。Intel Xeon Gold 6348 CPU 仅用于 host memory 存储，不参与计算

  **Baseline 的核心缺陷**：MoE 架构下 data-to-computation ratio 显著增加（expert 数量多而每个 token 仅激活 k 个），导致 Mobius 的 stage 级全量加载方式出现**计算阻塞问题**——数据加载时间超过计算时间，GPU 等待数据而闲置。此外，Mobius 未利用 MoE 的 expert 热度偏斜特性（少数 expert 承担大部分 token），也未利用 CPU 的计算资源。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  APTMoE 提出**亲和感知 offloading（Affinity-Aware Offloading）**，核心思想是基于 expert 热度和计算亲和性将部分低热度 expert 的计算分配到 CPU，减少 GPU 数据搬移量并提升计算效率。具体通过两个策略实现：

  1. **分层加载策略（Hierarchical Loading Strategy）**：将 Mobius 的 stage 级粗粒度加载细分为三层：
     - **Inter-stage loading**：基于历史 expert 热度（少数 expert 跨时间持续高激活），在 stage 切换时预取高热度 expert 到 GPU
     - **Inter-layer loading**：基于预测器（predictor）提前预测目标层 expert 热度，在当前层计算时预加载下一层高需求 expert
     - **Inter-expert loading**：基于实时 gate 输出，在同一层内动态决定哪些 expert 加载到 GPU、哪些留在 CPU 计算
     使用 Equation 1 (R = ΣCPU_time / ΣLoad_time) 作为 GPU/CPU 分配决策阈值，R=1 为停止加载边界。

  2. **需求优先级调度策略（Demand-Priority Scheduling Strategy）**：解决三层加载对同一 PCIe 带宽的竞争，通过 PriorityQueue 按 inter-expert > inter-layer > inter-stage 优先级动态协调加载顺序，采用 CUDA Event 前探机制在 kernel 启动前调度，隐藏 launch overhead。

  **APTMoE 方法全栈执行例子（以 MoE 模型在一个 micro-batch 上的 forward 为例）**：
  - 算法层：Expert 热度预测器（与 gate 同结构，共享权重初始化，提前若干层放置）预测目标层 expert 激活分布 → Equation 1 计算 CPU/GPU 分配阈值 → 高热度 expert 加载到 GPU 用 cuBLAS GEMM 计算 → 低热度 expert 留在 host memory 由 CPU 就地计算（使用 PyTorch CPU tensor），无需加载到 GPU
  - 系统框架层：流水线并行（相邻 stage 映射到不同 GPU）+ 三层加载流水线（inter-stage→inter-layer→inter-expert）→ 每层仅加载部分 expert 参数（而非全量），减少 PCIe 数据搬移量 → 反向过程中所有 expert 热度已知，inter-stage 阶段一次性全局最优分配
  - 编译框架层：论文未明确说明（PyTorch eager 执行，未涉及编译优化）
  - Kernel 调度层：comp_stream 执行计算与 load_stream 执行数据移动并行 → PriorityQueue 动态调度三层加载 → CUDA Event 前探隐藏 kernel launch latency → Inter-stream event 保证 data dependency 正确性
  - 硬件架构层：NVIDIA A800 GPU × 4（PCIe Switch），Intel Xeon Gold 6348 CPU 参与低热度 expert 计算（非仅存储）。三种设备拓扑（C1+G4/G2/G1）验证 CPU 核心数对性能的影响。最大扩展至 16 GPU（3 节点）

  **关键设计对应关系**：
  | Baseline 缺陷 | APTMoE 解决方案 | 具体机制 |
  |---|---|---|
  | MoE 数据量增大导致计算阻塞（data-to-computation ratio 高） | 分层加载策略 | 三层加载（inter-stage/inter-layer/inter-expert）按 expert 热度选择性加载，非全量加载 |
  | Mobius 未利用 expert 热度偏斜 | 亲和感知分配 | Expert 热度预测器 + Equation 1 阈值决策，高热度→GPU，低热度→CPU 就地计算 |
  | CPU 仅作存储，不参与计算 | CPU 参与低热度 expert 计算 | 低 token 数量时 CPU 计算时间与 GPU 可比（受限于 compute-bound→memory-bound 转换），减少数据搬移 |
  | 多加载阶段竞争同一 PCIe 带宽 | 需求优先级调度 | PriorityQueue + CUDA Event 前探，inter-expert > inter-layer > inter-stage 优先级动态协调 |

## AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach

- baseline方法是什么？
  **Baseline 为传统稀疏 MoE 架构（以 Mixtral 8x7B 为代表）**。在传统 MoE 中，专家（FFN）和门控网络通过同一混合数据集联合训练，专家未经过任务级训练，因此不存在与特定任务对应的专家。路由仅使用简单的 top-K 门控（Linear-SoftMax-TopK），无任务层面的分组或层级路由，专家分配在所有层上对不同任务分布相似，缺乏可控性和可解释性。Mixtral 8x7B 论文中观察到"no obvious patterns in the assignment of experts based on the topic"。

  **Baseline 全栈执行例子（以传统 MoE 处理复合医学查询"四肢无力+开中药方"为例）**：
  - 算法层：top-2 token-choice gating → 2 out of N FFN experts 激活 → 专家由混合数据训练，无任务领域区分 → 加权求和输出 → 无 task-aware routing → 对复合意图无法区分诊断和处方的不同权重需求
  - 系统框架层：标准 MoE 推理框架（HuggingFace Transformers / vLLM）→ Router 计算 → TopK → Expert Computation → Combine，所有专家平等对待
  - 编译框架层：论文未明确说明
  - Kernel 调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  **Baseline 的核心缺陷**：
  1. **专家缺乏任务专门化**：传统 MoE 的专家和路由器使用相同混合数据联合训练，没有任务级训练，导致专家间知识混合冗余，未形成明确的任务领域模式
  2. **路由不可控不可解释**：对复合多意图指令，传统 top-K 路由无法区分不同子任务的重要性权重，无法按"功能类别→具体子任务"的层次来分配专家
  3. **所有层路由相同**：传统 MoE 对不同 Transformer 层使用相同的路由机制，未考虑不同层关注不同抽象级别特征（低层偏基础特征，高层偏抽象语义）

- 论文方法是什么？如何对应解决Baseline的缺陷？
  AT-MoE 提出**自适应任务规划混合专家架构**，通过三个关键设计解决 baseline 缺陷：

  1. **任务级 LoRA 专家训练**：不在混合数据上联合训练，而是先用 PEFT（LoRA）在不同任务场景上**分别训练**任务特定专家，使每个专家有明确的任务领域属性（如医学中的功能类专家：诊断/处方/分诊；领域类专家：外科/放射科/病理科；风格类专家：严谨型/建议型）。每个 LoRA 专家 ΔW = BA（低秩分解），在原始 LLM 权重 W_0 基础上独立微调。

  2. **自适应分组路由（Adaptive Grouped Routing）**：对复杂多意图指令使用两层路由：
     - **第一层 - 群组级路由**：W_G (N_dim × N_G) 先将输入嵌入映射为跨组权重 W'_G（温度 SoftMax），在组维度做全局权重分配。例如对"四肢无力+开中药方"查询，功能类组权重较高。
     - **第二层 - 组内路由**：W_D (N_G × N_M) 在组内做局部 SoftMax 归一化，如功能组内"诊断"和"处方"权重高，领域组内"消化内科"和"中医"权重高。
     - 不足 N_M 专家的组用 -inf padding 不参与 SoftMax。

  3. **层级路由矩阵（Layer-wise Routing）**：不同 Transformer 层使用独立的路由矩阵（共 N_T 个），高层关注功能性/风格性特征，低层关注基础领域知识特征，实现了层级的自适应权重分配。

  最终输出：y_i = (λ·F_G(W̄_e) + (1-λ)·W_p)x_i + W_0·x_i，其中 F_G 为分组路由函数，W_p 为在所有任务混合数据上训练的预合并通用 LoRA 专家，λ 为平衡参数。

  **AT-MoE 方法全栈执行例子（以处理复合医学查询"四肢无力+开中药方"为例）**：
  - 算法层：查询嵌入 x → 群组路由 W_G: 功能组权重 0.6，领域组权重 0.3，风格组权重 0.1 → 组内路由 W_D: 功能组内"诊断"0.5+"处方"0.4+"分诊"0.1；领域组内"消化内科"0.5+"中医"0.4+"放射科"0.1；风格组内"严谨型"0.8 → λ=0.7 合并任务专家 + (1-λ)=0.3 合并通用专家 → 输出综合答案。同时，第 L 层（高层）侧重功能和风格专家，第 L-5 层（低层）侧重领域知识专家
  - 系统框架层：论文未明确说明（方法为算法层面设计，未讨论 Serving 框架集成）
  - 编译框架层：论文未明确说明
  - Kernel 调度层：论文未明确说明
  - 硬件架构层：论文未明确说明

  **关键设计对应关系**：
  | Baseline 缺陷 | AT-MoE 解决方案 | 具体机制 |
  |---|---|---|
  | 专家缺乏任务专门化，知识混合冗余 | 任务级 LoRA 专家分别训练 | 先在各任务数据上独立训练 LoRA 模块，冻结后再训练路由，确保每个专家有明确任务领域属性 |
  | 复合意图指令无法区分子任务权重 | 自适应分组路由（Group→Within-group 两层） | 群组级路由做全局组权重分配 + 组内路由做局部专家 SoftMax 归一化，实现"功能类别→具体子任务"的层次化权重 |
  | 所有层路由机制相同，不区分抽象级别 | 层级路由矩阵 | N_T 个独立路由矩阵，不同层关注不同特征（高层偏功能/风格，低层偏领域知识） |
  | 路由不可控不可解释 | 可追溯的权重分配路径 | 群组→组内→最终权重的分配路径可追踪：功能组 0.6 → 诊断 0.5 → 最终诊断 LoRA 权重 0.6×0.5×0.7=0.21 |
  | 单专家模型无法处理复合任务 | 多 LoRA 加权融合 + 通用专家 | 多个任务专家加权融合 + 预合并通用专家 λ 平衡，而非像 CoE 仅选单个专家

## Ada-K Routing Boosting the Efficiency of MoE-based LLMs

- baseline方法是什么？
  **Baseline 为静态 Top-K 路由 MoE**：每个 MoE layer 中，router 计算 Softmax(W · x_i) 得到专家概率分布，固定选择 top-k 个专家激活。所有 token 无论其重要性、语义复杂度或所在任务难度，都激活相同数量的 k 个专家。各模型使用各自默认的 k 值（如 Mixtral-8x22B k=2, DeepSeek-MoE-16B k=6）。

  **Baseline 全栈执行例子（以 Mixtral-8x7B 推理一个 token 为例）**：
  - **算法层**：token embedding x_i → Router: Softmax(W · x_i) → Top-2 路由 → 仅激活 2/8 FFN experts → 加权求和输出。无论 token 是简单介词还是复杂名词，都固定激活 2 个专家。
  - **系统框架层**：HuggingFace Transformers / vLLM → Gate 计算 → TopK 选择 → Expert FFN 计算 → Combine 输出。专家资源分配无自适应机制。
  - **编译框架层**：论文未明确说明（标准 PyTorch eager execution）。
  - **Kernel/运行时调度层**：标准 cuBLAS GEMM kernel 执行 expert FFN 计算。所有 token 的 expert 激活数相同，计算量均匀。
  - **硬件架构层**：NVIDIA A800 80GB GPU → 所有 expert 参数驻留 HBM → token 计算量不因内容而变化。

  **Baseline 的核心缺陷**：
  1. **固定激活数不考虑 token 重要性差异**：简单 token（如标点符号、连词、虚词）和复杂 token（如承载关键语义的名词/动词、需要复杂推理的 token）消耗完全相同的计算资源。简单 token 被过度处理（浪费计算），复杂 token 可能资源不足。
  2. **无法根据任务难度自适应**：简单任务（如常识问答）和困难任务（如多跳推理、数学）使用相同的专家激活策略，无法将更多计算资源集中于困难样本。
  3. **所有层使用相同路由策略**：不考虑浅层（基本特征提取）和中间层（复杂语义整合）对专家资源需求的不同。
  4. **性能-效率 Tradeoff 不可调**：Top-K 路由降低 k 值直接导致显著性能损失（如 Mixtral-8x7B k=1 vs k=2 平均准确率下降 7.68 点），无法灵活平衡性能与效率。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: Ada-K = 在每个 MoE layer 插入轻量级可学习 **allocator**（线性层，≈1M 总参数），对每个 token 动态采样决定激活专家数量 k*，并使用 **PPO 强化学习**端到端训练 allocator（绕过采样不可微分问题），同时加入 **activation regularization** 最小化专家激活数量。Warm-start 阶段使用 Top-P 核采样生成伪标签预训练 allocator。

  **Defect→Design 映射**:

  | Baseline 缺陷 | Ada-K 设计选择 | 解决机制 |
  |---|---|---|
  | 固定 k 不考虑 token 重要性 | Allocator 动态采样 k* ~ Softmax(W_alloc · x_i) | 每个 token 获得定制化专家数量：简单 token 用 1-2 experts，关键 token 用 3-8 experts |
  | 简单/困难任务相同资源分配 | PPO reward = 仅最后一层 log P(token) | 训练目标为优化最终预测质量，agent 自动学会对困难任务分配更多专家（BBH: Act=3.43 vs Collection: Act=2.58） |
  | 所有层相同策略 | 每层独立 allocator，层间独立决策 | 中间层自动分配更多专家（整合复杂特征），浅层和深层自动减少（基础特征提取和输出精炼） |
  | 降低 k 导致性能暴跌 | PPO loss + regularization loss (λ) 联合优化 | 通过 λ 实现灵活 trade-off：在 activation reduction rate 达 44% 前性能始终高于 baseline；Ada-K 在 34.4% reduction 下性能 +0.77 |
  | 采样操作不可微分 | PPO 强化学习 + reinforce with baseline advantage | 无需梯度通过采样操作；以默认 Top-K 路由输出为 baseline 降低方差；仅 2 PPO epochs 即可收敛 |
  | 冷启动采样不稳定 | P-Warm 策略 (Top-P 核采样伪标签) | 选择 p* 使平均专家数接近默认 k，用 n_j(p*) 预训练 allocator，避免随机初始化导致的任意采样 |

  **Ada-K 方法全栈执行例子（以 Mixtral-8x7B 推理一个 token 为例）**：
  - **算法层**：token embedding x_i → Allocator: P_alloc = Softmax(W_alloc · x_i) → 采样 k* ~ P_alloc（如对"the"采样 k*=1，对"photosynthesis"采样 k*=3）→ Router: TopK(P_router, k*) → 激活 k* 个 expert → 加权求和。内容词（名词/动词）平均激活 3.1 experts，虚词（介词/连词）平均 1.8 experts。
  - **系统框架层**：HuggingFace Transformers 推理 → 每层 forward 增加一次 allocator 采样（与 router 同级，计算量可忽略）→ 其余流程不变。Allocator 作为可插拔模块，无需修改 Serving 框架。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：Allocator 采样决定 k* 后，expert FFN 执行 k* 个 expert 的 GEMM（平均 k*=1.40 vs baseline k=2），FLOPs 从 6.56T 降至 4.42T（↓32.6%）。推理加速 1.28×。
  - **硬件架构层**：NVIDIA A800 GPU → k* 减小使 GPU kernel launch 和计算量均减少 → SM 利用率在大量简单 token 上降低（节省能耗），困难 token 上增加（质量提升）。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Per-layer allocator | 不同层对不同 token 有不同专家需求 | 中间层平均 3.2 experts，浅层 2.1，深层 2.3 (Qwen1.5-MoE) |
  | PPO with last-layer-only reward | 端到端优化 token 预测质量而非中间层局部指标 | Advantage 曲线持续上升 (Figure 8)，loss 持续下降 |
  | Activation regularization as loss | 直接可微分地最小化专家数量期望 | "As Loss" 模式准确率 +0.70 vs "As Reward" +0.21 |
  | P-Warm start (Top-P pseudo labels) | 避免采样空间过大导致随机初始化训练不稳定 | P-Warm Acc=55.13 vs Random=54.18 vs K-Warm=54.97 |
  | λ=3e-3 trade-off coefficient | 灵活平衡性能与效率 | 在 reduction rate 达 44% 前性能始终高于 baseline |
  | 数据域不敏感 | 训练数据域不影响效果 | Pretrain data Acc=55.78 vs SFT data Acc=55.13（均高于 baseline 54.43） |
  | Allocator ratio scaling | 每层部署 vs 部分层部署 | 全部层 FLOPs=0.92T vs 12.5%层 FLOPs=1.19T，训练参数仅增长 0.37M→2.95M |
  | 保持负载均衡 | Router 冻结避免破坏现有 expert load balance | 训练前后各 expert 激活概率分布几乎不变 (Figure 6) |

  **创新总结**：Ada-K 首次将 MoE 路由从"固定策略"转变为"学习策略"，通过极低训练成本（<8 GPU-hours, <0.002% 参数）实现了 25%+ FLOPs 节省和 20%+ 推理加速，同时提升性能。其核心洞察是：将非微分的离散路由决策问题通过 PPO 转化为可学习的策略优化问题，使 expert 资源分配从"一刀切"变为"按需分配"。

## Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

- baseline方法是什么？
  **Baseline 1**: DeepSpeed MoE training with static uniform expert replication + ZeRO-1 optimizer offloading。每个 expert class 分配相同数量的 replica (r = sN/E)，expert capacity 固定为 `capacity_factor × tokens_per_batch / E`，超出的 token 直接丢弃。Optimizer state 与 expert instance 绑定（共置在同一 GPU 或 EDP group 内）。
  
  **Baseline 2**: FlexMoE adaptive expert replication，根据 expert popularity 非均匀复制 expert，但 optimizer state 仍与 expert instance 绑定，rebalancing 时需搬运 optimizer state（8× weight size），因此只能粗粒度 rebalance（每 50-100 iterations），且每次 rebalancing iteration 延迟为正常的 2.46×–4.10×。
  
  **Baseline 共同缺陷**:
  1. **Convergence-Latency Tradeoff**: 静态 replication 无法匹配动态变化的 expert popularity（Figure 2: 16× fluctuation in 3 iterations），热门 expert 成为 latency bottleneck，冷门 expert 资源闲置。capacity_factor 调低→latency 改善但 token drop 增加→收敛变慢（Table 1）。
  2. **Optimizer Migration Overhead**: 自适应 replication 方案（FlexMoE）因 optimizer state (16B/param) 与 expert weights (2B/param) 绑定，rebalancing 需搬运两者，严重制约 rebalancing 频率（50-100 iters），无法跟踪 per-iteration popularity 变化。
  3. **Auxiliary Loss Tuning Burden**: 静态系统依赖 auxiliary load-balancing loss 来平衡 expert utilization，但高系数干扰主 loss 收敛（Figure 11），低系数导致高 drop rate。
  
  **Baseline 全栈执行例子（以 DeepSpeed static replication on GPT-Small, 16 GPUs, E=16, s=4 为例）**:
  - **算法层**: Switch Transformer Top-1 gating → router assigns tokens to experts → 固定 capacity_factor=1.0，每个 expert class 固定 capacity → 超容量 token 丢弃
  - **系统框架层**: DeepSpeed MoE → Expert Parallelism (16 GPUs, 4 slots/GPU) + Data Parallelism (EDP group, 4 replicas per expert) → ZeRO-1 optimizer offload (optimizer sharded within EDP group, binding optimizer to expert placement)
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + NCCL collectives）
  - **Kernel/运行时调度层**: Forward: 2× all-to-all (dispatch tokens + combine outputs) → Expert FFN compute → Backward: 2× all-to-all (scatter + gather gradients) → All-reduce within EDP groups for gradient sync → Optimizer step: PCIe transfer gradient → CPU Adam update → PCIe write back weights
  - **硬件架构层**: A100 GPU HBM 存储 expert weights + activations → host CPU DRAM 存储 optimizer state → PCIe 4.0 32GB/s GPU↔CPU → 100Gbps IB GPU↔GPU → 热门 expert device 成为 all-to-all 和 compute 瓶颈 → GPU utilization 不均匀

- 论文方法是什么？如何对应解决Baseline的缺陷？
  
  **论文方法**: SYMI 通过 **Model-Optimizer State Decoupling** 实现 per-iteration no-overhead adaptive expert replication。
  
  核心设计三步：
  1. **Decouple**: 将 optimizer state 从 expert instance 解耦 → optimizer 均匀静态分片到所有 N 个节点的 host memory，永不迁移
  2. **Repurpose**: 利用 optimizer step 中已有的 weight update 通信（Grad Communication → Optimizer Update → Weight Communication），将 updated weights 发送到新 placement 对应的 slot，而不是原 slot — 通信量完全相同（sNW），不引入任何额外数据搬运
  3. **Predict**: Expert Placement Scheduler 以 previous iteration 的 popularity 为 proxy（simple yet effective），proportionally 分配 replica counts（Algorithm 1），per-iteration 更新 placement
  
  **Defect→Design 映射**:

  | Baseline 缺陷 | SYMI 设计选择 | 解决机制 |
  |---|---|---|
  | Static replication → token drops & latency bottleneck | Adaptive expert replication per-iteration based on popularity | r_i ∝ popularity_i，热门 expert 获更多 replica→有效 capacity 自动拓展，冷门 expert 减少 replica→无资源闲置（Figure 9） |
  | Optimizer state binding → rebalancing overhead (2.46×–4.10× latency) | Decouple optimizer state from expert placement | Optimizer static uniform sharding across ALL N nodes；rebalancing 仅需 weight 重定向（same data volume），optimizer 永久不动 |
  | Infrequent rebalancing (50-100 iters) → cannot track rapid popularity shifts (16× in 3 iters) | No-overhead per-iteration rebalancing | Weight Communication Phase 的数据量不因 expert assignment 改变而变化 = sNW；locality shift 仅引入 1.52% 额外通信时间 |
  | Auxiliary loss tuning → convergence vs balance tradeoff | Adaptive replication eliminates need for auxiliary loss as system necessity | SYMI 在任何 auxiliary loss coefficient 下均保持 ∼10% token drops（vs DeepSpeed ∼40%），auxiliary loss 变为 quality knob 而非 system necessity（Figure 11） |
  | NCCL 不支持 intra-rank expert data parallelism → 20% extra token drops | Intra+Inter Rank All-Reduce | 三步梯度同步（intra-rank sum → inter-rank allreduce → intra-rank broadcast），expert 可自由放置于任意 slot |
  | Dynamic NCCL group creation → 1000s+ overhead in large clusters | Pre-register contiguous-rank communication groups at init | 仅需 N(N-1)/2 个 groups（非 2^N），跨 expert 和 layer 复用 |

  **SYMI 全栈执行例子（同 GPT-Small, 16 GPUs, E=16, s=4, per-iteration rebalancing）**:
  - **算法层**: Top-1 gating → router assigns tokens to experts → SYMI 扩展 router 做 global popularity all-reduce（E × 4B 通信，可忽略）→ 无固定 expert capacity，effective capacity = slot_capacity × r_i（r_i 随 iteration 动态变化）
  - **系统框架层**: 基于 DeepSpeed 修改 → Expert Parallelism（16 ranks, 4 slots/rank） + SYMI Optimizer（解耦式 optimizer state 管理）→ Expert Placement Scheduler（per-iteration 计算 placement）→ Layer Metadata Store（缓存 popularity 供 scheduler 读取）
  - **编译框架层**: 论文未明确说明（NCCL + PyTorch distributed batch point-to-point 通信）
  - **Kernel/运行时调度层**: 
    1. Forward: Router → popularity all-reduce → Token dispatch per dynamic placement (all-to-all) → Expert FFN
    2. Backward: Expert FFN backward → Intra-rank gradient sum → Inter-rank all-reduce (representatives only) → Intra-rank broadcast → SYMI Optimizer gradient collection (Algorithm 2, batch P2P, local-prioritized) → PCIe to host
    3. Optimizer Step: CPU Adam update → Expert Placement Scheduler (Algorithm 1, local) → Weight distribution (batch P2P to new placement) → PCIe to GPU
    4. 关键不变性: 每 iteration 传输的总数据量 = sNW (Grad) + sNW (Weight) = 与 DeepSpeed static 完全相同！
  - **硬件架构层**: A100 GPU HBM 存储 expert weights（动态 placement）→ host CPU DRAM 存储 optimizer state（静态 uniform shard）→ PCIe 4.0 32GB/s（optimizer ↔ GPU）→ 100Gbps IB（GPU↔GPU）→ 所有 GPU 负载均衡（adaptive replication）→ 无 hotspot bottleneck → iteration latency 略低于 DeepSpeed（new collectives 更高效）

  **关键创新对比**:
  - vs DeepSpeed: SYMI 增加 per-iteration adaptive replication，不增加 iteration latency（实际上减少 2.8%-9.3% 因更高效的 collectives），减少 69% token drops，time-to-convergence 加速 30.5%
  - vs FlexMoE: SYMI 的 rebalancing 无 optimizer migration overhead → 可 per-iteration rebalance → FlexMoE-10 需 35% 更高平均 iteration latency 才能达到相同收敛速度 → SYMI time-to-convergence 更快 25.9%

## AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

- baseline方法是什么？
  **Baseline 为 vanilla MoE with fixed top-k routing**: 每个 MoE layer 中，router 计算 Softmax(W_g · x) 得到所有 n 个 expert 的概率分布，固定选择 top-k 个 expert 激活。所有 token 无论语义重要性或复杂度，都激活相同数量 k 个 expert。典型例子：Mixtral-8x7B 使用 n=8 FFN experts, k=2 top-2 routing；Mo-LoRA 使用 n=4 LoRA experts, k=1 or k=2 routing。

  **Baseline 全栈执行例子（以 Mixtral-8x7B 推理一个 token x 为例）**：
  - **算法层**: token embedding x → Router: G(x) = Softmax(TopK(x · W_g, k=2)) → 2/8 FFN experts 激活 → y = G(x)_1 · E_1(x) + G(x)_2 · E_2(x) → 所有 token 固定使用 2 experts。无论 "apple" 还是 "the" 都消耗相同 FLOPs。
  - **系统框架层**: HuggingFace Transformers / vLLM → Gate 计算 → TopK (k=2) → Expert FFN 计算（各 expert 独立 GEMM）→ Weighted sum combine。无自适应机制。
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution）。
  - **Kernel/运行时调度层**: 论文未明确说明（标准 cuBLAS GEMM，2 experts per token）。
  - **硬件架构层**: 论文未明确说明（标准 GPU，所有 expert 参数驻留 HBM）。

  **Baseline 的核心缺陷**：
  1. **固定 expert 激活数不考虑 token 差异**: 语义丰富的 token（名词、动词）和功能 token（标点、连词、<EOS>）消耗相同计算量。论文通过 SocialIQA 上 Mixtral-8x7B 的路由分布分析验证：各层 token 路由概率分布的 sharpness 差异巨大——部分 token 极度倾向单一 expert，而另一部分 token 分散到超过 2 个 expert，证明固定 k 对所有 token 并非最优。
  2. **无法按需分配计算资源**: 简单 token 被过度处理（浪费 FLOPs），复杂 token 可能资源不足。无法根据计算预算灵活调整 expert 负载。
  3. **Expert-Choice Routing 的因果不适配**: Expert-choice routing 可实现不等量选择，但依赖 future tokens 做 top-k token selection，不适合 auto-regressive text generation。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: AdaMOE = 在 MoE layer 的 expert set 中引入固定数量 m 的 **null experts**（zero mapping, 零 FLOPs），并将 TopK 的 k 值增大。每个 token 仍做固定 top-k 选择，但当 null experts 被选中时无计算开销，因此实际激活的 true expert 数量随 token 自适应变化。通过修改 load balancing loss（null experts 间不做负载均衡）和 annealing α 训练策略控制平均 expert 负载。

  **Defect→Design 映射**:

  | Baseline 缺陷 | AdaMOE 设计选择 | 解决机制 |
  |---|---|---|
  | 固定 k 使所有 token 消耗相同 FLOPs | 引入 m 个 null experts + 增大 k 值 | Token 在 top-k 中可选 0~k 个 true expert（null expert 不消耗 FLOPs），实现 token-adaptive routing。简单 token 倾向选更多 null expert，复杂 token 倾向选更多 true expert |
  | 无法按计算预算调整 expert 负载 | 通过调整 m（null expert 数量）和 α（load balancing loss 系数）控制平均 true expert 使用率 | 增大 m → 更多 null experts → 降低 true expert 负载；annealing α（先紧后松）实现 performance-efficiency tradeoff。ARC-C 上 FLOPs 减少 14.5% 同时 accuracy 提升 1.69% |
  | Expert-Choice Routing 不适合自回归 | Token-choice routing 天然适配 causal LM | 每个 token 独立选择自己的 top-k experts，不依赖 future tokens，与标准 transformer 推理完全兼容 |
  | 传统 load balancing 对 null experts 施加不必要约束 | 修改 load balancing loss: null experts 之间不做负载均衡 | 将所有 null experts 视为同质，用平均 f_j 替代各自 f_j，避免对 router 施加无意义的约束。实验验证 ℓ_null 比 ℓ_bal 在 RTE/COLA/SQA/OQA 上显著提升 accuracy |
  | Top-k 增大后 normalization 方式选择 | 仅对 top-k 中的 true experts 做 Softmax normalization | 保证加权输出与 vanilla MoE 的数值尺度一致（option 2 在 SIQA 上 accuracy 81.27 vs option 1 80.19） |

  **AdaMOE 方法全栈执行例子（以 Mixtral-8x7B + AdaMOE (m=8, k=3) 推理一个 token x 为例）**：
  - **算法层**: token embedding x → Router: G(x) = Softmax(TopK(x · W_g, k=3)) where W_g ∈ R^{d × (8+8)} → 若选出 {E_2, E_5, null_3} → 仅对 E_2, E_5 做 Softmax (option 2) → y = w_2 · E_2(x) + w_5 · E_5(x) → 实际仅 2 true experts FLOPs。若选出 {null_1, null_3, null_6} → y = 0，token 完全绕过此 MoE layer，类似 MoD 的 "bypass" 行为。平均 Load = 1.66（baseline Load = 2.00）。
  - **系统框架层**: HuggingFace Transformers 推理 → 原始 gate module 扩展 gate2 维度（gate2 output=8 for m=8 null experts）→ Router 拼接 gate+gate2 输出 → TopK(k=3) → 仅执行 true experts 的 FFN 计算 → null experts 不触发任何 GEMM kernel → 减少 FLOPs（↓14.5% on ARC-C）。
  - **编译框架层**: 论文未明确说明。
  - **Kernel/运行时调度层**: 论文未明确说明（实际 expert 激活数减少 → 减少 FFN GEMM kernel launch 次数 → 推理延迟降低）。
  - **硬件架构层**: 论文未明确说明（标准 GPU，减少的 FLOPs 直接转化为能耗和延迟节省）。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Null experts (zero FLOPs) | 实现 token-adaptive routing 而不破坏 top-k 机制 | Load 从 2.00 降至 1.66 (↓17%), FLOPs ↓14.5% on ARC-C |
  | m 和 k 超参数配置 | 按计算预算调整 expert 利用率 | m=32,k=6: Load=1.54, accuracy 仍高于 baseline；m=40,k=8: Load=1.34 |
  | Annealing α (α1=0.02→α2=0.0001) | 先紧后松: epoch 1 建立负载均衡, epoch 2 释放 token 自由度 | WINO accuracy: epoch1=76.24 → epoch2=81.93 (+5.69%) with minimal Load increase |
  | ℓ_null (no balancing among null experts) | 消除对 null experts 间的不必要约束 | ℓ_null vs ℓ_bal: RTE 67.51 vs 56.68, COLA 85.01 vs 83.68 |
  | Plugin-and-play (对 vanilla LLMs 和 MoE-LLMs) | 无缝集成，无需改动模型架构 | Llama2-7B Mo-LoRA: 各配置均超 baseline；Mixtral-8x7B: accuracy +1.69% @ FLOPs-14.5% |
  | 鲁棒性 (不同 epochs/LoRA ranks) | 方法对各种超参数不敏感 | Epoch 1 vs 10: AdaMOE 48.88→88.54 (baseline 45.95→87.19); Rank 8 vs 32: AdaMOE 48.88→49.01 (baseline 45.95→46.72) |

  **创新总结**: AdaMOE 通过在最简单的位置（expert set）插入最简单的操作（null expert = 0 FLOPs），以最小代价实现了 token-adaptive routing。核心洞察：将 "选择多少 expert" 的离散决策问题转化为 "选哪些 expert" 的连续路由问题（增加 m 个 null expert 并增大 k），配合 load balancing loss 自动调整平均 null/true expert 使用率。方法实现简单（仅需扩展 router 输出维度 + 修改 loss），兼容现有 (MoE-)LLM，可直接 fine-tune 使用，无需 pretrain from scratch。

## Adaptive Gating in Mixture-of-Experts based Language Models

- baseline方法是什么？
  **Baseline 为 Top-2 Gating MoE**：在训练阶段，每个 MoE layer 中 router 计算 Softmax(x · W_G) 得到 E 个专家的概率分布，固定选择 top-2 个专家激活并加权求和（y = Σ_{i∈E} R_i · FFN_i(x)）。推理时切换为 top-1 gating 以减少延迟。所有 token 无论语义复杂度如何，训练时都消耗 2 个 expert 的 FLOPs，all-to-all 通信量也固定为 top-1 的 2 倍。

  **Baseline 的核心缺陷**：
  1. **固定计算量浪费资源**：大量 token（≥55%）的概率分布显著偏向 top-1 expert（top-1 与 top-2 概率差异大），这些 token 仅需单 expert 即可有效处理，但 top-2 gating 仍强制为其激活 2 个 expert，造成不必要的计算和通信开销。
  2. **训练效率与模型性能的 trade-off 不明**：top-2 gating 是否真的比 top-1 gating 带来性能提升并足以 justify 额外的计算成本，缺乏系统性分析。实际上 top-1 gating（Switch Transformer）在 4/6 任务上训练收敛更慢，训练时间甚至超过 top-2，说明单纯减少 k 不减训练时间。
  3. **Token 间计算时间不均导致训练瓶颈**：即使部分 token 使用 top-1 节省计算，Attention 层需要完整序列输入，训练 step 时间仍由 batch 中最慢的 top-2 token 决定，计算节省无法完全转化为时间节省（Table 1: FLOPs 节省 40% 但运行时间仅节省 24%）。

  **Baseline 全栈执行例子（以 BERT-Base MoE + top-2 gating 训练一个 MoE layer 为例）**：
  - **算法层**: token embedding x → Gate: R = Softmax(x · W_G) → TopK(R, 2) → 固定激活 2/16 FFN experts → y = R_1·FFN_1(x) + R_2·FFN_2(x)。所有 token 语义简单或复杂都消耗 2× FFN FLOPs，包括 "a", "the", "is" 等虚词。
  - **系统框架层**: HuggingFace Transformers + PyTorch → 8× A100 expert parallelism (每 GPU 2 experts) → all-to-all scatter token → Expert FFN compute → all-to-all gather → attention layer（需要完整序列，等待所有 token 完成）。
  - **编译框架层**: 论文未明确说明（PyTorch eager execution）。
  - **Kernel/运行时调度层**: Attention layer barrier 等待所有 token 完成 MoE 计算 → 即使 80% token 已完成 top-1 FFN，仍需等待剩余 20% token 的 top-2 FFN 完成。
  - **硬件架构层**: 8× A100 40GB NVLink 互联 → all-to-all 每次传输 top-2 所需全部 token（2× top-1 的数据量）。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: Adaptive Gating in MoE = 引入阈值 T 根据 expert 概率分布动态决定每个 token 激活 1 或 2 个 expert，配合 modified load balancing loss（仅约束 top-1 决策）和 curriculum learning（按复杂度重排训练数据）。

  **Defect→Design 映射**:

  | Baseline 缺陷 | Adaptive Gating 设计选择 | 解决机制 |
  |---|---|---|
  | 固定 top-2 对所有 token 浪费计算 | 阈值 T 自适应门控：prob_diff ≤ T → top-2，否则 → top-1 | 概率分布偏斜的 token（≥55%）自动降为 top-1，节省计算 FLOPs。Sentiment analysis 仅 11.3% token 使用 top-2 |
  | top-1 gating 训练收敛慢导致训练时间不降反升 | 保留 top-2 用于困难 token + modified load balancing loss | 困难 token 仍获得双专家处理（保证收敛速度），简单 token 节省计算。自适应方案在 6/6 任务上训练时间 < top-2 且 ≤ dense |
  | Token 计算时间不均→Attention barrier 成为瓶颈 | Curriculum learning: 按复杂度重排训练数据 | 将相似复杂度的样本分组训练，减少同 batch 内 top-2 token 比例方差，缓解"快 token 等待慢 token"问题。平均减少 13.7% 额外训练时间 |
  | 负载均衡对灵活 expert 数不适应 | Modified load balancing: 仅对 top-1 gating 施加软约束 | top-2 决策自由不受负载均衡限制，避免对需要双专家的 token 施加不合理的 expert 分布约束 |

  **论文方法全栈执行例子（以 BERT-Base MoE + Adaptive Gating 训练一个 MoE layer 为例）**：
  - **算法层**: token x → Gate: R = Softmax(x · W_G) → 计算 prob_diff = R_top1 - R_top2 → if diff ≤ T(0.1): route to top-2 expert; else: route to top-1 expert → 输出 y = (仅单/双 expert FFN 加权和)。虚词 "the", "a" 用 1 expert，情感承载词用 2 experts。
  - **系统框架层**: HuggingFace Transformers + PyTorch → 8× A100 expert parallelism（16 experts 均匀分布）→ all-to-all scatter（token 数减少，因多数 token 仅需 top-1 目标 GPU 通信）→ Expert FFN compute（总 FLOPs 减少）→ all-to-all gather → Attention layer（同 batch 内 token 的计算时间差异减小，因 curriculum learning 将相似复杂度样本分组）。
  - **编译框架层**: 论文未明确说明（PyTorch eager execution）。
  - **Kernel/运行时调度层**: Attention barrier 等待时间减少：同 batch 内 top-2 token 比例方差降低（curriculum learning 效果）→ top-1 token 完成后短暂等待即可进入 attention → 端到端 step 时间减少（最多 22.5%）。
  - **硬件架构层**: 8× A100 40GB → all-to-all 通信量因多数 token 仅需发往 1 个 expert GPU 而减少 → MoE layer 运行时间从 1x 降至 0.76x–0.92x（取决于 top-1 比例，Table 1）。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | 阈值 T=0.1 自适应门控 | 概率偏斜 token 节省 1 expert FLOPs | Sentiment: 11.3% top-2, FLOPs 3.28G→2.30G (↓30%), 训练时间 ↓23% |
  | Modified load balancing (仅约束 top-1) | top-2 决策自由度 | 防止 expert 集中，同时不干扰双专家 token 的路由学习 |
  | Curriculum learning (余弦相似度排序) | 缓解同 batch 内 token 计算时间差异 | 去除后训练时间膨胀平均 13.7%，推理性能最大下降 0.21 F1 |
  | 小阈值 0.1-0.2 为最优 | 平衡计算节省与模型精度 | T=0.2 在多数任务上性能等于 top-2 且训练时间更短；T=0.4 不总等于 top-2 性能 |
  | 任务相关的自适应路由分析 | 理解哪些 token 需要双专家及原因 | Sentiment: 中性意见/反讽 token; Translation: 复杂从句; QA: 限定问题范围的关键词; Summarization: 代词/主旨 token |

  **创新总结**: Adaptive Gating 首次在 MoE 训练中将"每个 token 固定 k 个 expert"改为"基于概率分布的灵活 expert 数"，其核心洞察是门控概率分布本身就包含了 token 复杂度的信息——top-1 与 top-2 概率差异大的 token 天然仅需单专家。配合 curriculum learning 解决了灵活 expert 数带来的 batch 内负载不均问题，实现了训练时间降低 22.5% 的同时保持模型质量。

## AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts

- baseline方法是什么？
  **Baseline: MoLE (Mixture of LoRA Experts) with 静态 top-k 门控**。在每层 Transformer 的 self-attention 权重矩阵（Wq, Wk, Wv, Wo）上，用 N 个 LoRA 专家（各 rank=r）替代单个 LoRA（rank=N×r），路由器计算 Softmax(W_g x) 得到 N 个专家权重，通过固定 top-k（k=2 或 k=3）选择最高的 k 个专家，其余权重置零，归一化后加权求和。或者使用固定阈值 τ=1/N 的硬阈值策略。

  **Baseline 缺陷**：
  1. **static top-k 对所有 token 同等对待**：无论是简单还是复杂的 token/任务，始终激活固定数量专家，无法根据输入复杂度灵活调整资源分配。
  2. **固定阈值 τ=1/N 缺乏上下文感知**：阈值无法随输入语义变化，不能区分何时需要更多专家（如复杂推理）或更少专家（如简单语法）。
  3. **资源浪费或欠利用**：简单 token 激活过多专家浪费计算，复杂 token 可能激活不足导致精度损失。

  **Baseline 全栈执行例子（以 Llama-2-7B + MoLE top-2，处理单个 token x 为例）**：
  - **算法层**：输入 x 经 router 计算 8 个专家权重 p = Softmax(W_g x) → TopK(p, k=2) 选出权重最高的 2 个专家 → 输出 h = W_0 x + (p_1 E_1(x) + p_2 E_2(x)) / (p_1 + p_2)，每个 E_i(x) = B_i A_i x ∈ R^d
  - **系统框架层**：HuggingFace Transformers + PEFT → 替换 self-attention 四矩阵的 LoRA adapter 为 8 个 LoRA expert → forward 时 router 和 top-k 在 PyTorch eager 模式执行 → 所有 expert 的 A_i, B_i 矩阵都已加载到 H100 GPU 显存
  - **编译框架层**：论文未明确说明（PyTorch eager execution）
  - **Kernel/运行时调度层**：无论输入简单或复杂，始终启动 2 个 LoRA expert 的 GEMM kernel（每个 expert = r×k · d×r 两次矩阵乘法）→ 简单 token 浪费 1 个 expert 的 kernel 计算 → 复杂 token 可能 2 个 expert 仍不足
  - **硬件架构层**：单张 NVIDIA H100 GPU → top-2 的 2 次 LoRA GEMM 消耗固定 CUDA core 和显存带宽 → 简单 token 下 SM 做无效计算

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: AdaMoLE = LoRA + 自适应 MoE，引入可学习的动态阈值网络替代静态 top-k 选择。关键设计：
  1. **动态阈值网络**：τ = τ_max · σ(W_τ x + b_τ)，τ_max = 1/N，单层线性层 + sigmoid 使 τ 随输入 x 自适应变化
  2. **基于阈值的专家选择**：激活所有 p_i ≥ τ 的专家，而非固定 k 个
  3. **可导门控公式**：用 (p_i - τ) 替代原始 p_i，确保 τ 参与反向传播梯度计算，使阈值网络可学习
  4. **参数等价性**：8 个 rank-4 专家 = 单个 rank-32 LoRA，总参数量相同（除门控和阈值网络额外参数）

  **Defect→Design 映射**：

  | Baseline 缺陷 | AdaMoLE 设计选择 | 解决机制 |
  |---|---|---|
  | static top-k 对所有 token 同等对待 | 动态阈值 τ(x) = τ_max · σ(W_τ x + b_τ) | 每个 token 根据自身特征计算独立阈值，复杂 token 得到较低 τ（激活更多专家），简单 token 得到较高 τ（激活更少专家） |
  | 固定阈值 τ=1/N 缺乏上下文感知 | 阈值网络以输入 x 为条件 | τ 成为 x 的函数，不同语义输入自然产生不同阈值和专家激活数 |
  | 简单 token 浪费计算 | 高 τ 只激活极少数专家 | Table 4 显示 τ∈[0,3/(2N)] 时平均仅激活 1.26 专家（CommonsenseQA），远少于 top-2 的固定 2 个 |
  | 复杂 token 精度不足 | 低 τ 允许更多专家参与 | τ∈[0,1/(2N)] 时平均激活 6.59 专家（CommonsenseQA），远多于 top-2，准确率 78.95% vs top-2 的 77.15% |

  **论文方法全栈执行例子（以 Llama-2-7B + AdaMoLE，阈值范围 [0, 1/N]，处理单个 token x 为例）**：
  - **算法层**：输入 x → router 计算 8 维权重 p = Softmax(W_g x) → 阈值网络计算 τ = (1/N) · σ(W_τ x + b_τ) → 激活集合 S = {i | p_i ≥ τ} → 输出 h = W_0 x + Σ_{i∈S} (p_i - τ) B_i A_i x / Σ_{j∈S} (p_j - τ) → S 的大小随 x 动态变化，平均 3.46（CommonsenseQA）或 4.56（COPA）
  - **系统框架层**：HuggingFace Transformers + PEFT → 额外加入阈值网络（单层 Linear + Sigmoid）→ forward 时 router、阈值计算、条件专家激活均在 PyTorch eager 执行 → 所有 8 个 LoRA expert 矩阵常驻 H100 显存
  - **编译框架层**：论文未明确说明（PyTorch eager execution）
  - **Kernel/运行时调度层**：简单 token（τ 高）→ 仅 1-2 个 expert 的 GEMM kernel 启动 → 复杂 token（τ 低）→ 可能 6-8 个 expert GEMM kernel 启动 → 实际 kernel launch 数动态变化，理论计算量与输入复杂度成正比
  - **硬件架构层**：单张 NVIDIA H100 GPU → 简单 token 少用 CUDA core/显存带宽，复杂 token 多用 → 相比 top-2 固定 2 expert，AdaMoLE 能对简单 token 节省 ~40% 计算（1.26 vs 2 expert），对复杂 token 额外利用更多 expert 提升精度

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | τ = τ_max · σ(W_τ x + b_τ) 动态阈值 | 替代 static top-k 实现上下文自适应专家选择 | CommonsenseQA: 78.71% vs MoLE top-2 77.15% (+1.56%); COPA: 94.00% vs 92.00% (+2.00%) |
  | (p_i - τ) 可导门控公式 | 使阈值网络可端到端训练 | 阈值网络参数可通过交叉熵 loss 反向传播学习 |
  | Threshold sensitivity τ∈[0,1/(2N)] vs [0,1/N] | 调整计算-精度权衡 | τ∈[0,1/(2N)]: CommonsenseQA 78.95%, 平均激活 6.59 专家; τ∈[0,1/N]: 78.71%, 平均激活 3.46 专家 |
  | 较低层更多专家激活 | 利用 LLM 层次特征：低层处理多样基础特征需更多专家 | Figure 2: Layers 1-10 激活更多专家，Layers 25-32 激活较少（CommonsenseQA & COPA） |
  | Hyperparameter robustness (N×r) | 方法对各配置鲁棒 | N=4×r=4: 78.38%; N=8×r=4: 78.71%; N=16×r=4: 78.13%; 均超各自 MoLE baseline |
  | Cross-model generalization | 方法适用于不同基础模型 | Gemma-7B: 81.00% vs LoRA 80.51% (+0.49%); Llama-2-13B: 81.74% vs LoRA 79.77% (+1.97%) |

## AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

- baseline方法是什么？
  **Baseline 为手动设计的 homogeneous MoE（以 Switch Transformer 为代表）**。在传统 MoE 中，专家采用均匀设计：所有层中 expert 数量相同（如每层 4 个或每隔一层 4 个），所有 expert 的 FFN 尺寸相同（如 intermediate size = 2048 或 3072），encoder 和 decoder 的层数也固定（均为 6 层）。专家放置位置采用 ad-hoc 规则：每隔一层（Fedus et al. 2022b; Kim et al. 2021）、每四层（Zoph et al. 2022），或最后几层（Rajbhandari et al. 2022）。

  **Baseline 全栈执行例子（以 6-layer encoder-decoder SwitchTransformer-Big + 手动 homogeneous MoE 推理一个 token 为例）**：
  - **算法层**: token embedding x → Router: Softmax(x · W_g) → top-1 gating → 每层固定激活 1/4 experts（所有 expert FFN 尺寸相同，intermediate=3072）→ 所有 token 走相同大小的专家计算
  - **系统框架层**: fairseq (PyTorch) → encoder forward（6 layers, 4 experts each）→ decoder autoregressive forward（6 layers, 4 experts each, 200× 每步时间 vs encoder at peak throughput）→ 所有 decoder layer 每步激活相同数量 expert
  - **编译框架层**: 论文未明确说明（PyTorch eager execution + fairseq）
  - **Kernel/运行时调度层**: 论文未明确说明（标准 cuBLAS GEMM 执行 expert FFN）
  - **硬件架构层**: Intel Xeon CPU → decoder latency 占总量 90%+ → 6 decoder layers × 4 experts 激活 → total FLOPs 10.6G, latency ~2199ms (CPU)

  **Baseline 的核心缺陷**：
  1. **Homogeneous 设计导致计算浪费**：所有 expert 尺寸相同，但不同 token 需要不同计算量——对简单 token，"大专家"浪费 FLOPs
  2. **无自适应计算（adaptive compute）**：相同数量/大小的 expert 参数应用到每个输入，不支持"不同 token 使用不同计算量"
  3. **手动设计效率低下**：expert 放置（每层/隔层/每四层）是 ad-hoc 选择，未系统性地优化 FLOPs 和 latency
  4. **MoE 设计不考虑硬件约束**：expert 数量/大小的选择与目标部署硬件（CPU latency, memory）脱节，模型可能在 CPU 上 latency 过高

- 论文方法是什么？如何对应解决Baseline的缺陷？
  
  **论文方法**: AutoMoE = 通过 NAS 在异构 MoE 搜索空间中自动搜索最优架构。核心设计三步：
  1. **异构搜索空间**：每层可变 expert 数量（{1,...,M}）+ 每 expert 可变 FFN 尺寸（{1024,2048,3072}）+ 可变 decoder 层数（{1-6}）→ 形成指数级搜索空间
  2. **Supernet 训练 + 演化搜索**：Supernet 通过 weight sharing 联合训练所有子架构，演化算法以 validation loss 为性能信号、以目标设备 latency 为约束，迭代搜索 Pareto 最优
  3. **自适应计算（Adaptive Compute）**：异构设计使不同 token 通过 routing 自然分配到不同大小的 expert，简单 token 走小 expert（节省计算），复杂 token 走大 expert（保持质量）

  **Defect→Design 映射**:

  | Baseline 缺陷 | AutoMoE 设计选择 | 解决机制 |
  |---|---|---|
  | Homogeneous expert size → 所有 token 相同计算量 | 可变 expert FFN 尺寸（per expert） | Token 路由至不同大小的 expert，"简单" token 走小 expert（节省 FLOPs），"困难" token 走大 expert（保证质量），实现 adaptive compute |
  | 手动 ad-hoc expert 放置 | NAS 自动搜索每层 expert 数和放置 | 演化算法发现最优配置：encoder 中间层（3rd, 5th）分配最多 expert，decoder 首层最多 → encoder 承担 71% 专家 |
  | 不考虑硬件约束 | Latency constraint（CPU ≤ 600ms）作为搜索约束 | 演化搜索在 latency 约束内优化 BLEU，产生的架构天然满足部署硬件要求 |
  | 固定 decoder 层数（6 层）→ decoder 延迟主导 | 搜索可变 decoder 层数（1-6） | 减少 decoder 层数（从 6 → 3 或 4），补偿为增加首层 expert 数。decoder latency 降低 30%+ |
  | Expert 选择缺乏系统性优化 | Supernet weight sharing + 演化搜索 | 联合优化 expert 数量、大小、decoder 层数、attention heads、hidden size 等全部 Transformer 超参数 |

  **AutoMoE 方法全栈执行例子（以 WMT'14 En-De AutoMoE 6-expert 搜索到的架构推理一个 token 为例）**：
  - **算法层**: token x → Router → top-1 gating → encoder: 层层 expert 数分别为 [5,1,1,1,2,1]，expert FFN 尺寸各异（1024-3072）；decoder: 4 layers, experts [1,1,1,1], FFN 尺寸全 3072 → 大多数计算集中在 encoder 中间层（容量大），decoder 轻量化（layer 数从 6 → 4, experts 少）
  - **系统框架层**: fairseq (PyTorch) → encoder forward: 中间层激活多 expert + 大 FFN（处理源语言语义信息）→ decoder forward: 4 layers only, 每层 1 expert（轻量级生成）→ 总 latency 504ms (CPU) vs baseline SwitchTransformer 2199ms
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution + fairseq）
  - **Kernel/运行时调度层**: 论文未明确说明。总 FLOPs 从 10.6G → 2.9G (↓3.7×)，expert 激活数大幅减少
  - **硬件架构层**: Intel Xeon CPU → encoder 承担主要计算（中间层多 expert + 大 FFN, latency ~45ms），decoder 极轻（4 layers × 1 expert × 3072 FFN, latency ~459ms）→ 总 latency 504ms = 4.4× speedup

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | 可变 decoder 层数 | decoder 延迟主导（>90%） | FLOPs 随 decoder 层数增加而增加（Fig 3a）；AutoMoE 自动选择 3-4 decoder layers |
  | Encoder 中间层多 expert | Encoder 需要高容量处理语义 | Encoder 3rd/5th layer 分配最多 expert（Fig 3c），encoder 占总 expert 71% |
  | Decoder 首层多 expert | 补偿 decoder 层数减少的容量损失 | Decoder 首层 expert 最多，逐层递减（Fig 3d） |
  | 异质 expert 尺寸（fract-expert） | 实现 adaptive compute | 70% expert layers 有 ≥2 experts，>75% 含可变 expert 尺寸。WMT'14 En-De AutoMoE: BLEU 28.2, FLOPs 2.9G, Latency 504ms |
  | Identity/dummy experts（FFN size=0） | 允许部分 token "跳过" FFN 计算 | BLEU 28.1 (↓0.1), FLOPs 2.7G (↓6.9%) — 质量轻微损失但 FLOPs 显著降低 |
  | Latency constraint（而非仅 FLOPs） | 更严格的硬件控制 | Latency constraint 下模型充分利用 budget 且 FLOPs 更优；FLOPs constraint 下 latency 偏高（Table 6） |

  **创新总结**: AutoMoE 首次将 NAS 引入 MoE 设计，将 MoE 架构从"手动 homogeneous 设计"转变为"自动异构搜索"。其核心洞察是：MoE 架构的各维度（expert 数量、大小、decoder 层数）之间存在复杂的性能-效率 trade-off，通过 Supernet 的 weight sharing 和演化搜索，可以在短时间内（224 GPU-hours vs Evolved Transformer 的 2,192,000 GPU-hours）找到 Pareto 最优的异质配置。异构设计自然实现 adaptive compute——不同 token 路由到不同大小的 expert，无需额外机制即可实现"按需分配计算"。

## AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

- baseline方法是什么？
  **Baseline 1: 从头训练（From Scratch）**。大模型（32B MoE）完全随机初始化，在全部 5345B tokens 上从头预训练。需要最大规模的集群（1024 devices × 240 GFLOPS），训练吞吐仅 25B tokens/day，需 213.8 GPU-days 总训练时间。
  **Baseline 2: bert2BERT 初始化（FPI/Stacking AKI）**。使用 FPI（Function Preserving Initialization）或 stacking-based AKI 扩展小模型权重初始化大模型。FPI 导致对称权重（net2net 固有缺陷），stacking 导致层间输出空间不匹配（last layer output ≠ first layer input）。

  **Baseline 全栈执行例子（以训练 32B MoE 为例）**:
  - **算法层**: 随机初始化所有参数 → MoE router N(0, 0.02) → 从零开始学习所有 token 表征 → 需要全量 5345B tokens → load balancing loss + z-loss 稳定训练
  - **系统框架层**: 分布式训练框架（PyTorch + 自研 AI 框架）→ 1024 GPUs 数据并行 + expert 并行 → all-to-all scatter/gather 通信 → 训练吞吐 25B tokens/day
  - **编译框架层**: 论文未明确说明（BAAI 内部 AI 框架，可能基于 PyTorch + 定制算子）
  - **Kernel/运行时调度层**: 每 GPU 执行持有的 expert 的 FFN GEMM kernel → MoE layer 涉及 all-to-all 通信 token dispatch → router 计算 + top-k 选择 kernel
  - **硬件架构层**: 1024 × Ascend-like 240 GFLOPS accelerators → 25B tokens/day 吞吐 → From Scratch 需 5345B/25=213.8 天等效训练时间

  **Baseline 缺陷**:
  1. **计算和数据浪费**: 从头训练需要 5345B tokens，每个 token 都需要从零学习基础语言知识。
  2. **FPI 权重对称**: 扩展时将权重简单复制/拆分，导致对称权重在训练中梯度相同，有效参数减半。
  3. **Stacking 层间不匹配**: StackBERT 的层堆叠方法使第 L_1-1 层输出空间与第 0 层输入空间不匹配，导致训练初期的 loss spike 和不稳定。
  4. **GQA 不兼容**: 原始 AKI 仅支持 MHA，无法处理 Group Query Attention 模型的 attention head 扩展。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  **论文方法**: EfficientScale = Scale-Up (AKI-Pro) + Scale-Out (Sparse Upcycling)

  1. **AKI-Pro 解决 FPI 对称性**: 使用相邻层权重而非同层复制来扩展宽度（继承 bert2BERT AKI），避免对称初始化，保证有效参数不减少。
  2. **Interpolation 替代 Stacking**: 深度扩展使用 `W'_l = floor(l × L_2 / L_1)` 插值而非直接复制堆叠，保证相邻层输出空间平滑过渡，训练更稳定（验证: FPI-Interpolation loss 3.31 vs FPI-Stacking loss 4.30 at M(32,4096)）。
  3. **GQA 兼容性改造**: 在源和目标模型 group 数一致的前提下，将每个 GQA group 视为独立 MHA block 进行 AKI 扩展，使 AKI 支持 GQA 模型。
  4. **Sparse Upcycling 解决从头训练**: 将 dense 模型 MLP 直接复制为 MoE experts，保留已学知识，仅需 545B tokens 微调（vs 5345B 从头训练）。

  **Defect→Design 映射**:

  | Baseline 缺陷 | EfficientScale 设计选择 | 解决机制 |
  |---|---|---|
  | 从头训练需 5345B tokens | Scale-Up 知识迁移 + Scale-Out 复用 dense 权重 | 仅需 3600(7B) + 1200(16B) + 545(MoE) = 5345B tokens total，但 7B 训练可复用已有小模型，实际额外训练仅 1745B |
  | FPI 对称权重导致有效参数减半 | AKI-Pro 用相邻层权重打破对称 | Validation loss 降低: M(32,4096) AKI-Pro 7.81 vs FPI 4.30 |
  | Stacking 层间不匹配导致训练不稳定 | Interpolation 深度扩展 | FPI-Interpolation loss 3.31 << FPI-Stacking loss 4.30 |
  | 原始 AKI 不支持 GQA | 将每个 group 视为独立 MHA block 扩展 | 支持 GQA 架构（16B 模型 8 KV groups），扩展后训练收敛正常 |
  | 从头训练 32B MoE 需 213.8 天 | Scale-Up + Scale-Out pipeline | 时间节省 4.12×，算力节省 3.35× |

  **EfficientScale 全栈执行例子（以 1.3B → 7B → 16B → 8×16B MoE 为例）**:

  - **算法层**:
    Phase 1: 加载 Aquila2-1.3B M(24,2048) 预训练权重 → Phase 2: AKI-Pro 宽度扩展 (768→4096 hidden dim, 2048→14336 intermediate) + Interpolation 深度扩展 (24→32 layers) + GQA 保持 32 groups → 初始化 M(32,4096) 7B 模型 → 连续预训练 3.6T tokens → 再次 Scale-Up AKI-Pro: M(32,4096)→M(40,5120), 深度 interpolation 32→40, 宽度 4096→5120, GQA 32→8 groups → 连续预训练 1.2T tokens → Phase 3: Sparse Upcycling ×8 experts → router N(0,0.02) 随机初始化 → top-2 routing → 连续预训练 545B tokens
  - **系统框架层**: BAAI 自研 AI 框架 → 1024 GPU data+expert parallelism → all-to-all token dispatch/gather → load balancing loss (λ=0.001) + z-loss (λ=0.01) → full BF16 训练
  - **编译框架层**: 论文未明确说明（BAAI 内部框架，可能基于 PyTorch+XLA 或自研编译器）
  - **Kernel/运行时调度层**: MoE layer forward: router Softmax + top-2 selection → all-to-all scatter tokens to expert-holding GPUs → 每 GPU 执行 2 experts' FFN GEMM (W_gate·x, W_up·x, W_down·h) → all-to-all gather → residual add + LN。Dense layer: 标准 Transformer block GEMM kernels
  - **硬件架构层**: Phase 1: 480 × 989.5 GFLOPS GPU → 279B tokens/day。Phase 2-3: 1024 × 240 GFLOPS accelerators → Scale-Up 70B/day, Scale-Out 25B/day。时间节省 4.12× (213.8→51.84 GPU-days), 算力节省 3.35×

  **关键数值验证**（Table 1, 4, 6）:
  | 指标 | Baseline (From Scratch) | EfficientScale | 提升 |
  |---|---|---|---|
  | M(32,4096) validation loss | 12.22 (random init) | 7.81 (AKI-Pro) | 初始 loss 降低 36% |
  | GSM8K-gen | - | 54.51 (8×16B MoE) vs 7.81 (7B) | MoE 大幅超越 dense |
  | 总训练时间 | 213.8 等效天 | 51.84 天 | **4.12× 时间节省** |
  | 总算力 | 52,592,640 GFLOPS-days | 15,705,343 GFLOPS-days | **3.35× 算力节省** |
  | MMLU-ppl | - | AquilaMoE 61.00 vs AquilaDense-16B 57.11 | +3.89 点 |

## BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

- baseline方法是什么？
  **Baseline 为单一 brain foundation model 预训练于 resting-state fMRI**：现有 brain foundation models（BrainLM, BrainJEPA, BrainMass）将 fMRI 分析建模为自监督预训练+下游微调范式。预训练阶段仅使用 resting-state fMRI 数据（或最多加入一种 tasking state），通过 mask reconstruction（MAE 或 JEPA）学习 BOLD 或 FC 的 latent feature representation。下游微调使用 SVM 或简单 MLP 做分类/回归。

  **Baseline 全栈执行例子（以 BrainMass 处理 ABIDE Autism 分类为例）**：
  - **算法层**: fMRI raw BOLD → AAL atlas 分区 → FC 矩阵 X ∈ R^{116×116} → 随机 mask → Encoder(Bottleneck) → Z ∈ R^{2048} → Decoder → X̂ → L = ||X̂ - X||² → 预训练 29,951 resting-state scans → 下游：提取 Z → SVM/MLP → 2-class Autism 分类（F1=67.81% with MLP+68k）
  - **系统框架层**: PyTorch（推断）→ 单 GPU RTX 6000 Ada → 训练/推理在单卡完成 → 无分布式通信
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution）
  - **Kernel/运行时调度层**: 标准 cuBLAS GEMM 执行 transformer encoder FFN 和 attention 计算
  - **硬件架构层**: 1× NVIDIA RTX 6000 Ada GPU → 所有模型参数常驻显存 → 推理 37.08ms/sample

  **Baseline 的核心缺陷**：
  1. **忽略认知状态异质性**：大规模 fMRI 数据集（UKB、HCP）包含多种认知状态（resting + 11 种 tasking），但现有模型仅使用 resting-state 数据（~30k scans），忽略了 >38k tasking fMRI。直接混合所有认知状态训练单一模型反而因不同认知状态间的异质性导致 suboptimal 特征表示（信息瓶颈理论）。
  2. **数据扩展边际收益递减**：从 30k→68k 预训练数据（加入 11 种 tasking states），BrainMass+MLP 在 ABIDE 上仅 +1.00 F1，在 SZ 上反而 -2.22 F1；BrainJEPA 在 HCPA 上 -6.42 F1。说明简单扩大数据规模不解决多认知状态异质性问题。
  3. **输入类型和预处理管线依赖**：BOLD 模型和 FC 模型对不同下游数据集的性能差异大，模型缺乏对输入类型和预处理管线的鲁棒性。不同预处理 pipeline（如 [33] vs 本文 pipeline）的数据分布差异进一步降低模型泛化。
  4. **Late Fusion MoE 无效**：直接使用 Late Fusion（各 expert 独立预测后加权融合）无法有效利用 expert 多样性——路由器总是倾向于选择数据量最大的单一 expert（如 Rest, n=29,971），无法学习专家间的协同。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  
  **论文方法**: BrainMoE = 将 fMRI 数据按认知状态分层预训练多个 brain expert + Router 做 expert 选择 + Cognition Adapter（Transformer Decoder with cross-attention）混合 cognition embeddings。

  **Defect→Design 映射**:

  | Baseline 缺陷 | BrainMoE 设计选择 | 解决机制 |
  |---|---|---|
  | 忽略认知状态异质性 | 按 12 种认知状态分层预训练 12 个独立 expert | 每个 expert 专门学习一种认知状态下的 brain activity pattern，避免不同状态间相互干扰 |
  | 数据扩展边际收益递减 | Stratified pre-training + MoE fine-tuning | 不是简单混合所有数据训练一个模型，而是让每个 expert 成为特定认知状态的"专家"，再通过 adapter 联合利用。Taowu (n=40) 上 +18.28 F1 over 68k BrainMass |
  | 输入类型/预处理管线依赖 | Cognition Adapter 对 expert 架构和数据格式无要求 | Adapter 通过 cross-attention 将 FC 矩阵信息注入 task embeddings，不依赖 expert 的具体内部实现。支持 BOLD/FC 混合 expert |
  | Late Fusion MoE 无效 | Router + Cognition Adapter（Transformer Decoder）替代 weighted sum | Router 学习多样化的 dual expert 组合（而非单一 expert 主导），Adapter 通过 self-attention 混合 expert 嵌入产生新表示 |

  **BrainMoE 方法全栈执行例子（以 ABIDE Autism 分类为例）**：
  - **算法层**: fMRI → AAL116 FC 矩阵 X ∈ R^{116×116} → 12 个冻结 expert 分别产 Z_rest, Z_emotion, Z_gambling, ..., Z_language ∈ R^{2048} → Router: P = Softmax(W_r · [Z_1,...,Z_12]), Top-k → 选 Rest+Emotion expert → Cognition Adapter: Self-Attention(Z_topk + Q_task) → Cross-Attention(Q=FC, K=Z_bar, V=FC) → FFN → Linear → 2-class Autism 分类（F1=70.26% vs Baseline 67.81%）
  - **系统框架层**: PyTorch → 1× RTX 6000 Ada → 12 experts + adapter 常驻显存 → 推理 157.60ms/sample（vs baseline 37.08ms，4× 时间增加）
  - **编译框架层**: 论文未明确说明（标准 PyTorch eager execution）
  - **Kernel/运行时调度层**: 12 experts 前向并行 → Router Top-k → Adapter Self-Attention + Cross-Attention → 所有计算在单 GPU 上顺序执行
  - **硬件架构层**: 1× NVIDIA RTX 6000 Ada → 709M params 常驻显存 → 4× 推理时间 overhead（trade robustness for latency）

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | 按认知状态分层预训练 12 experts | 解决简单混合多状态数据的 suboptimal 问题 | BrainMoE 在所有 7 个数据集上超越单个 expert（Fig 4），task-specific experts（Language→AD, WM→PD）超越 Rest expert |
  | Cognition Adapter (Transformer Decoder) | 替换 MLP adapter 的可扩展性不足 | 709M params, FC recon. expert 在 phenotypic 分类 4/7 数据集 rank 1st（Table 3） |
  | Cross-attention with FC matrix | 解决输入类型依赖，统一 BOLD/FC 多模态信息 | All-in-one BrainMoE (36 experts) 在 sex 分类 4/7 数据集 rank 1st（Table 4） |
  | Router Top-k with adaptive selection | 解决 Late Fusion 单一 expert 主导 | BrainMoE 学习 diverse dual expert 组合（Fig 5a vs 5b），expert 嵌入相关性 < 0.5（Fig 5c） |
  | 多 cognitive state 数据利用 | 解决 >38k tasking fMRI 被忽略的浪费 | 68,251 scans 全部利用，smallest dataset (Taowu n=40) 上 BrainMoE +43.76 F1 over BrainJEPA |
  | Age regression 泛化 | 验证方法在连续回归任务上的效果 | ABIDE (6-58 yrs): MSE 36.77→4.86（↓87%），HCPYA: 5.46→3.45 |
  | fMRI-EEG 多模态 | 验证跨模态鲁棒性 | CBraMod(EEG)+BrainMoE(fMRI) 在 8-task 分类 68.73% vs CBraMod only 67.66% |

  **创新总结**: BrainMoE 首次将 MoE 框架引入脑 fMRI 基础模型，核心洞察是：大规模脑影像数据中丰富的认知状态信息不应被忽视，也不应被简单混合——而应该通过"分而治之"（stratified pre-training）+ "智能混合"（cross-attention adapter）的方式利用。每个 brain expert 成为特定认知状态的"脑活动专家"，Cognition Adapter 通过 Transformer decoder 的自注意力和交叉注意力机制学习如何为每类下游任务组合这些专家知识。方法对 expert 架构和数据格式无要求，可适配任意 brain foundation model。

## Brainformers Trading Simplicity for Efficiency

- baseline方法是什么？
  **Baseline 为 GLaM（manually crafted sparse Transformer）**：标准稀疏 MoE Transformer，采用 uniform block 设计——每个 Transformer block 固定为 attention + FFN（dense block）或 attention + MoE（sparse block），两者严格交替排列。所有层使用相同的 Top-2 token-based gating、固定的 model dimension 和 expansion ratio（FFN hidden = 4× model dim）。架构由人工设计，无自动搜索优化。

  **Baseline 全栈执行例子（以 GLaM 8B/64E 模型训练一个 token 为例）**：
  - **算法层**：input token → Layer 1: Attention(4096-dim, 32 heads) + Dense FFN(4096→16384→4096) → Layer 2: Attention + MoE(Top-2 routing, 64 experts, each 4096→16384→4096) → Layer 3: Attention + Dense FFN → ... 交替重复 → 每 token 固定激活 2/64 experts → 计算 FLOPs 由固定 expansion ratio 决定
  - **系统框架层**：GLaM 训练框架（Google 内部，推测基于 TensorFlow/XLA + TPU） → Expert Parallelism 分布 64 experts 到多 TPU device → all-to-all token dispatch → TPU 间通信
  - **编译框架层**：论文未明确说明（Google 内部 XLA 编译）
  - **Kernel/运行时调度层**：TPU matrix unit 执行 dense FFN GEMM（4096×16384）和 MoE expert GEMM → Top-2 routing kernel → 每层严格交替导致不同层间计算量不均匀（MoE 层重、FFN 层轻）
  - **硬件架构层**：512 Cloud TPU-V4 chips → 143B total params 分布 → steps/sec = 0.39 → 训练收敛慢

  **Baseline 的核心缺陷**：
  1. **Uniform 架构限制效率**：固定 attention-FFN/MoE 交替导致架构缺乏灵活性，无法根据计算需求调整不同层的宽度和类型。GLaM 的 uniform 设计使得 MoE 层和 dense 层计算量差距大，层间负载不均。
  2. **固定 Top-2 gating 不是最优**：所有 token 固定激活 2 experts，但 Expert Choice routing 可能更优（允许 perfect load balance 且每 token 激活数量可变）。
  3. **固定 expansion ratio 浪费参数**：GLaM 使用固定 4× expansion（4096→16384），但 MoE 的多 expert 已提供宽度，不需要如此大的 expansion。
  4. **手动设计无法系统化优化**：人工调整架构维度（层数、宽度、expert 数）缺乏系统性，难以找到最优配置。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Brainformer = 通过演化搜索（Regularized Evolutionary Search）自动发现非均匀 Transformer block 架构，打破 uniform 交替限制，在固定训练时间预算下联合优化层类型序列、层宽度、gating 机制、routing 策略和激活函数。

  **Defect→Design 映射**：

  | Baseline 缺陷 | Brainformer 设计选择 | 解决机制 |
  |---|---|---|
  | Uniform attention-FFN 交替 → 架构不灵活 | Non-uniform block-wise architecture: 每层独立选择 F_attn/F_moe/F_ffn | 搜索发现最优层序列（如 Brainformer Block 1: 8 sub-layers 含 2 attention + 3 MoE + 3 FFN），减少 attention 频率降低计算 |
  | 固定 Top-2 routing → 负载不均 | 搜索同时优化 gating function（Top-2 vs Expert Choice） | 搜索选择 Expert Choice gating + capacity factor=1，实现 perfect load balance 和极致稀疏 |
  | 固定 4× expansion → 参数浪费 | 搜索可变 model dim + MoE/FFN hidden dim | 搜索选择更大 model dim(1024) + 更小 expansion factor，利用 MoE 多 expert 的宽度替代单层大 expansion |
  | 手动设计 → 非系统化 | Evolutionary search + fixed training time constraint | 在固定 wall clock time 下自动采样、训练、评估、选择最优架构 |
  | 训练预算分配不公 → 稀疏模型吃亏 | Fixed training time search + inference time constraint | 以训练时间和计算成本（而非参数总量）为比较基准，允许模型以更快 step time 换取更多 training steps |
  | Sparse model scaling 效率低 | Block-wise stacking: 搜索到的 block 通过 ScaleModelDim + StackNTimes 扩展到目标规模 | 100M→1B→8B 线性扩展，保持 block 结构不变 |

  **Brainformer 方法全栈执行例子（以 Brainformer-1 8B/64E 训练一个 token 为例）**：

  - **算法层**：input token → Brainformer Block（8 sub-layers）：
    Sub-layer 1: F_attn (model_dim=1024, 20 heads) → Multi-head Self-Attention
    Sub-layer 2: F_moe (model_dim=1024, moe_hidden=2048, ExpertChoice gating, capacity=1) → 每 token 平均路由至 1 expert，64 experts，perfect load balance
    Sub-layer 3: F_ffn (model_dim=1024, ffn_hidden=2048) → Gated GeLU activation
    Sub-layer 4: F_attn → Self-Attention
    Sub-layer 5: F_moe → Expert Choice MoE
    Sub-layer 6: F_ffn → Dense FFN
    Sub-layer 7: F_moe → Expert Choice MoE
    Sub-layer 8: F_ffn → Dense FFN
    → Block 重复 N 次（stacking）→ LM head → token prediction。相比 GLaM：attention 频率降低（2 vs 每层都有），expert 激活数降低（~1 vs 2），model dim 更大（1024 vs 4096 但 expansion 更小），总 activated params 更少（7.4B vs 9.8B）
  
  - **系统框架层**：Google 内部 TPU 训练框架 → Expert Parallelism（64 experts 分布 512 TPU V4）→ Expert Choice routing 天然 load balance → 无 auxiliary loss 即可均衡 → 减少通信等待。Brainformer-1 实现 1.96 steps/sec vs GLaM 0.39（5× faster）
  
  - **编译框架层**：论文未明确说明（Google 内部 XLA 编译，自动融合 TPU 计算图）
  
  - **Kernel/运行时调度层**：TPU V4 matrix unit 执行 MoE expert FFN GEMM（1024×2048, 1 expert/token avg → 远小于 GLaM 的 4096×16384×2）→ Attention kernel（1024-dim, 20 heads → 少于 GLaM 的 4096-dim, 32 heads）→ 总计算量大幅降低 → step time 5× faster
  
  - **硬件架构层**：512 Cloud TPU-V4 chips → 158B total params（高于 GLaM 143B），但仅 7.4B activated（低于 GLaM 9.8B）→ 更少的 per-chip 计算量 + Expert Choice 的 load balance → TPU 利用率更高 → 2× training convergence speedup

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Non-uniform block（8 sub-layers, variable types） | 打破 uniform 交替限制，灵活组合计算 | Brainformer-1 PPLX 1.99 vs GLaM 2.12 at 8B scale |
  | Expert Choice gating (capacity=1) | 替代 Top-2 实现 perfect load balance + 更稀疏 | 每 token avg 1 expert, step time 5× faster |
  | 搜索可变 model dim + hidden dim | 优化 expansion ratio 匹配 MoE 的宽度 | Model dim 1024 + MoE hidden 2048（expansion~2× vs baseline 4×） |
  | Fixed training time search | 公平比较不同架构在相同预算下的质量 | Brainformer 在相同训练时间下达到更低 PPLX |
  | Block-wise stacking (ScaleModelDim + StackNTimes) | 从搜索到的小规模块扩展到生产规模 | 100M block → 1B → 8B 线性扩展 |
  | 减少 attention 频率 | Attention 在长序列上成本高 | Block 仅 2 attention 层（vs baseline 每层 attention） |
  | 2× convergence speedup + 5× step time speedup | 整体训练效率 | 512 TPU V4, same hardware, 更快达到目标 PPLX |

## BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

- baseline方法是什么？
  **Baseline 为 On-Demand Expert Loading with Predictive Prefetching**。在 memory-constrained（GPU 显存不足以容纳全部 MoE expert）的推理场景中，系统将 inactive expert offload 到 CPU memory，仅在需要时通过 PCIe 加载到 GPU。为了隐藏传输延迟，prefetching 系统（如 MoE-Infinity, Pre-gated MoE）基于历史激活频率或 auxiliary gating 预测下一层的 expert 需求并提前加载。

  **Baseline 全栈执行例子（以 DeepSeek-V2-Lite, 64 experts/layer, cache rate=0.75, prefetching 预测一个 token 为例）**：

  - **算法层**: token x → Router: TopK(Softmax(W_g · x), k=6) → 选中 6 个 experts → 检查 GPU residency 状态 M。Prefetching 预测器基于历史激活频率或前一层 attention output 预测下一层需要的 experts → 提前异步加载到 GPU cache。预测成功 → 下一层的 expert 已在 GPU，直接计算。预测失败（cache miss）→ 同步 CPU→GPU 传输（~9-10ms over PCIe），pipeline stall。Drop 策略：跳过该 expert（准确率严重损失）。
  
  - **系统框架层**: llama.cpp / MoE-Infinity → Router → Prefetch predictor → Expert cache manager (GPU cache 保留 c 比例 expert, 基于 LRU/LFU 替换) → CPU offloading engine → 预测命中时传输被计算隐藏（~0ms overhead），预测失败时同步传输 ~9-10ms → 该延迟是推理瓶颈（占 85-94% inference latency on edge devices）。

  - **编译框架层**: 论文未明确说明（llama.cpp 使用 CUDA backend 编译为 GPU kernel）。

  - **Kernel/运行时调度层**: Gate kernel → Prefetch decision kernel → Expert FFN GEMM kernel → PCIe 传输 kernel（cudaMemcpy）。当 prefetch 失败时，GPU kernel 线程等待 PCIe DMA 完成（GPU SM idle），cache hit 的 token 也无法继续（需要完整 batch 完成）。

  - **硬件架构层**: 1× A100 PCIe GPU + Intel Xeon Platinum 8457C CPU。PCIe 4.0 带宽 32 GB/s → 单个 expert 传输 ~10ms（expert 参数 ~300MB+ for DeepSeek-V2-Lite 的每个 expert），而 GPU 计算仅需 ~1-2ms → inference 从 compute-bound 变为 I/O-bound。随着 expert 数量增加（64→256→2048），GPU cache 命中率下降 → prefetch miss 率上升 → 延迟恶化。

  **Baseline 的核心缺陷**：
  1. **Prefetch 预测本质不完美**：MoE 的 expert 路由是输入依赖的（context-dependent），无法完美预测。historical frequency-based heuristics 和 auxiliary gate-based prediction 都因路由的随机性而存在不可避免的 misprediction——预测失败时强制同步加载，延迟 ~10ms，淹没了多次成功 prefetch 带来的收益。
  2. **Cache Miss 的惩罚是二元的**：现有系统的选择要么承受完整的 PCIe 传输延迟（on-demand loading），要么完全跳过 expert 计算（expert dropping），两者都是 unacceptable 的极端——前者的延迟惩罚，后者的精度惩罚。缺乏一个优雅的中间方案。
  3. **Expert 冗余未被利用**：大量 MoE expert 存在功能冗余（functional redundancy）——从 similarity heatmap（Figure 4）和 prior work（MoE 容忍 aggressive pruning down to 4 bits）已验证——但这种冗余仅被当作可剪枝的"浪费"，未被作为系统资源利用。
  4. **Expert 激活分布的偏斜浪费 cache 容量**：少数"popular" expert 占有不成比例的激活（Figure 6），大量 expert 仅被稀疏激活但仍占用 GPU cache 槽位，导致 cache 效率低下。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**: BuddyMoE = 基于 expert 功能冗余的 buddy expert 替代系统。核心思想是将 cache miss 从"catastrophic stall"转化为"low-cost approximation"：当 prefetch 失败时，不等待同步传输，而是用 GPU cache 中功能相似的 buddy expert 即时替代，以极小的精度损失换取显著的延迟降低。

  **Defect→Design 映射**：

  | Baseline 缺陷 | BuddyMoE 设计选择 | 解决机制 |
  |---|---|---|
  | Prefetch 预测不完美 → miss 导致 10ms stall | Buddy replacement: 用 GPU-resident buddy 替代 CPU-resident expert | 查找操作 ~0ms vs 同步传输 ~10ms，pipeline 不 stall |
  | Miss 处理仅有两个极端（full stall 或 drop expert） | 三阶段 safety gate（TAE + Distribution Gate + Buddy Priority Score） | 提供 fine-grained trade-off：conservative settings 最大化 accuracy，aggressive settings 最大化 throughput |
  | Expert 冗余未被利用（仅被视为可剪枝浪费） | Co-activation matrix analysis → buddy list construction via CFT | 将冗余转化为功能性资源：buddy expert 在功能上足够相似，替代后 accuracy degradation 可控（-5.4% acc 换取 +10.3% t/s at c=0.375） |
  | 激活分布偏斜 → cache 效率低 | Co-activation 引导的 buddy 选择使替换倾向高频共激活的 peers | buddy 替代不仅避免了传输，还在有意义的 expert 间做替换（非随机），accuracy 比 random replacement 高 176%-400% |
  | 无 token 级别的敏感度评估 | TAE (Token Activating Entropy) gate: τ 控制替换阈值 | peaked routing token（低 TAE）不替换 → 保护对替换敏感的 token 质量；diffuse routing token（高 TAE）允许替换 → 大量无害替换 |
  | 无 batch 级别的风险控制 | Distribution Gate: δ 控制 batch 级 CPU expert 比例阈值 | 当 batch 中过多 expert 在 CPU 时禁止替换 → 防止级联误差累积 |
  | 无拓扑感知的 buddy 选择 | Ψ score 中的 hop(j) 惩罚项 + sharding awareness | 在分布式设置中优先选择同 GPU/同 partition 的 buddy，减少跨设备通信 |

  **论文方法全栈执行例子（以 DeepSeek-V2-Lite, 64 experts/layer, cache rate=0.75, τ=0.95, \|B\|=16, ρ=3 推理一个 token 为例）**：

  - **算法层**: token x → Router: TopK(Softmax(W_g · x), k=6) → 选出 {e_3, e_7, e_15, e_22, e_41, e_58}。检查 GPU residency: e_3/e_7/e_22 在 GPU，e_15/e_41 被 prefetched 成功已在 GPU，e_58 在 CPU（prefetch miss）。
    → TAE gate: TAE(x) = 0.97 > τ=0.95 → token 对 expert 选择较 diffuse，允许替换。
    → Distribution gate: 本 batch 中仅 2 个 expert 在 CPU，δ < β → 允许替换。
    → Buddy selection: 查 B_ℓ(e_58; 0.95) = [e_22, e_41, e_7, ...] → e_22 已在 S' 中（被映射到相同 buddy 需避免 → penalty 降低优先级），e_41 也在 S' 中 → 选 e_7 作为 buddy。e_7 已在 GPU 且在 token 的 expert set 中（被映射到相同 buddy 意味着减少多样性）。下一个候选 e_23 在 GPU 且不在 U_t → 替换成功: S'[t][5] = e_23。
    → Expert 计算: y = p̃_3·FFN_3(x) + p̃_7·FFN_7(x) + p̃_15·FFN_15(x) + p̃_22·FFN_22(x) + p̃_41·FFN_41(x) + p̃_23·FFN_23(x)（e_58 被 e_23 替代，功能相似但 ~0ms 延迟 vs ~10ms 传输）。

  - **系统框架层**: llama.cpp → Router → [BuddyMoE Runtime 中间层: TAE gate → Distribution gate → Buddy selection CUDA kernel] → Expert FFN → "Buddy profile" lookup table 序列化加载随 model checkpoint，O(16×64) = 1024 entries/layer 存储可忽略。BuddyMoE 不替代 prefetching 而是补充：prefetch 成功时正常工作，prefetch 失败时用 buddy replacement 避免同步 stall。

  - **编译框架层**: 论文未明确说明（llama.cpp CUDA backend，标准 NVCC 编译）。

  - **Kernel/运行时调度层**: CUDA buddy_substitute_kernel: grid(T,1,1) × block(K,1,1)，每个 thread block 处理 1 个 token的 K=6 个 expert 替换。__shared__ bool U_t[64] 维护 token 的已分配 expert set → 每 thread 依次检查自己的 expert → if M[e_id]==false: for r in 1..H: b_id = B[e_id][r]; if M[b_id] && atomicCAS(&U_t[b_id], false, true): S'[t][k] = b_id; break。Atomic CAS 操作是无锁的（lock-free），保证 uniqueness constraint。所有操作在 GPU memory 内完成，无 CPU↔GPU 传输。

  - **硬件架构层**: 1× A100 PCIe GPU + Intel Xeon Platinum 8457C CPU。Buddy replacement 完全在 GPU 内部完成：查找 B lookup table（GPU HBM 常驻）→ 检查 M 布尔 mask（GPU register/shared memory）→ atomic CAS（GPU L2 cache atomic operation）→ 结果写入 S'。不触发任何 PCIe 传输。PCIe read 带宽使用比 baseline 减少约 20%（Figure 8）。Expert 权重的 GPU/CPU residency 管理由 llama.cpp native offloading 机制维护。

  **性能 Trade-off 数值（Table 2-4）**：

  | Cache Rate | 配置 | Acc | t/s | vs Original | vs Random |
  |---|---|---|---|---|---|
  | 0.75 | τ=0.95, \|B\|=16, ρ=3 | 0.695 | 36.75 | +7.4% t/s, -5.4% acc | +26.4% acc |
  | 0.50 | τ=0.95, \|B\|=16, ρ=3 | 0.635 | 30.21 | +5.8% t/s | +176% acc |
  | 0.375 | τ=0.95, \|B\|=16, ρ=3 | 0.645 | 27.33 | +10.3% t/s | +303% acc |

  在极端内存约束（c=0.375）下，BuddyMoE 保持 0.645 accuracy（vs random 0.16），同时比 original baseline 提升 10.3% 吞吐量——证明 buddy replacement 在内存最紧张时效益最大。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Co-activation-based buddy identification (CFT) | 替代 random replacement 的无意义替换 | c=0.375: BuddyMoE Acc=0.645 vs Random=0.16 (4× better) |
  | TAE gate (τ=0.95) | 保护 peaked-routing token 不被替换 | τ=0.75 Acc=0.64 vs τ=0.95 Acc=0.525 (higher τ more accurate) |
  | Distribution gate (δ) | 防止 batch 级级联误差 | 当 δ 高时禁止替换 → 保留 on-demand 作为 safety net |
  | Replacement budget (ρ=3) | 限制每 token 最大替换次数，防止过替换 | ρ=3: Acc=0.695 vs ρ=4: Acc=0.54（更多替换 → 更大精度损失） |
  | Buddy list size (\|B\|=16) | 足够大的候选集提高 GPU 命中概率 | \|B\|=16: 覆盖 α=0.95 co-activation mass → 高 buddy 命中率 |
  | Per-layer calibration (α_ℓ) | 适应不同层间的冗余模式差异 | Early layers: broader redundancy; later layers: tighter expert clusters |
  | CUDA kernel parallelization | 消除 CPU-GPU kernel launch overhead | Atomic CAS + shared memory → ~0ms overhead |

  **创新总结**: BuddyMoE 的核心洞察是将 MoE 模型的 expert 冗余从"可剪枝的浪费"转变为"可调度的系统资源"，实现了 MoE 推理中 cache miss 损失函数的范式转换——从二元的"miss=stall/drop"变为连续的"miss=approximate"。通过 co-activation 模式分析、三阶段 safety gate 和 CUDA 并行化替换，BuddyMoE 在最大化吞吐的同时保持了可控的精度退化，在极端内存约束下的收益最大（c=0.375 时 +10.3% throughput, accuracy 为 random 的 4×）。关键在于 buddy 的选择不是随机的——co-activation 编码了真实的功能相似性，使替换成为"有意义的近似"而非"随机的扰动"。——通过打乱 Transformer 的 uniform block 结构，让演化搜索在更大的架构空间中自动发现高效的层类型组合和维度配置。搜索到的架构倾向于：(1) 减少 attention 层频率（attention 计算昂贵），(2) 使用 Expert Choice routing 替代 Top-2（天然 load balance + 更稀疏），(3) 增大 model dim 同时减小 expansion factor（利用 MoE 多 expert 替代大 FFN）。这种"非均匀+稀疏"的设计使 Brainformer 在更少 activated params 下实现更高质量和 5× step time speedup。

## Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

- baseline方法是什么？
  **Baseline 为 Expert Parallelism 下的 dropless MoE 推理**：在 Megatron-LM 分布式推理框架中，MoE 层使用标准 Token-Choice Top-K gating（每个 token 独立选 top-k 个 expert），不做容量约束，所有选中的 token 都参与对应 expert 的 FFN 计算。训练时虽使用了 auxiliary balance loss（如 Switch-Transformer 的 load balancing loss）鼓励均衡分配，但在推理时 token 分布仍高度不均衡。

  **Baseline 全栈执行例子（以 OLMoE-Instruct, 8×H20 GPU, 8-way EP, batch 8K × seq 512 为例）**：

  - **算法层**：输入 token x [N, d] → Gate: logits = x · W_g [d, 64] → Softmax → TopK(k=8) → 8/64 expert 激活 → y = Σ_{i∈top8} w_i · FFN_i(x)。所有 token 的 top-8 expert 均完整参与计算，不做任何 token 丢弃或重路由。
  - **系统框架层**：Megatron-LM → 8-way Expert Parallelism（每 GPU 8 个 expert）+ 8-way Data Parallelism → Router → All-to-All Dispatch → Expert FFN → All-to-All Combine。高负载 expert 处理大量 token → GPU 间负载不均 → 低负载 GPU 提前完成但等待 All-to-All barrier。
  - **编译框架层**：论文未明确说明（PyTorch eager execution + NCCL collectives）。
  - **Kernel/运行时调度层**：论文未明确说明具体 kernel 实现。MoE FFN 使用标准 grouped GEMM，各 expert 的 token batch 大小取决于路由结果。
  - **硬件架构层**：8× H20 GPU（96GB HBM3），PCIe/NVLink 互联。

  **Baseline 的核心缺陷**：
  1. **Straggler Effect 导致延迟瓶颈**：推理时 token 分布极不均衡——OLMoE 中最高负载 expert 收到超过 7× N̄ 的 token（Figure 1, 2），MoE 层延迟 L ∝ max({N_i})，由最繁忙 expert 决定，低负载 expert 和 GPU 空闲等待。
  2. **训练时 balance loss 在推理时失效**：auxiliary balance loss 仅约束训练过程，无法保证推理时 test data 的 token 分配均衡。从 scratch 训练的 MoE（如 OLMoE）比 upcycling 模型（如 Mixtral）更不均衡。
  3. **现有方案资源开销大**：DeepSeek-V3 通过复制高负载 expert 冗余部署缓解，但增加 GPU 资源消耗；auxiliary loss 调参复杂。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Capacity-Aware Inference = Capacity-Aware Token Drop（对高负载 expert 施加容量约束丢弃超额 token）+ Capacity-Aware Expanded Drop（利用低负载 expert 剩余容量，扩展候选集处理溢出 token）。核心思路是在 All-to-All 通信前对 token-to-expert assignment 施加容量控制和重分配，减少 straggler expert 的负载并提升低负载 expert 的利用率。

  **Defect→Design 映射**：

  | Baseline 缺陷 | Capacity-Aware 设计选择 | 解决机制 |
  |---|---|---|
  | Straggler Effect → 最高负载 expert 决定延迟 | Token Drop: 容量约束 C=γN̄，丢弃超载 expert 的溢出 token | L ∝ max(N_i) ≤ γN̄，γ<1 时延迟上限由容量因子直接控制，如 γ=1.5 时 Mixtral 获 1.85× 加速 |
  | 低负载 expert 空闲等待 | Expanded Drop: top-k+m 扩展候选 + 容量约束内重分配 | 低负载 expert 吸收溢出 token → 提升利用率和模型性能（+0.2% on Mixtral） |
  | 训练 balance loss 在推理时失效 | 推理时 capacity constraint 强制执行负载均衡 | 无需依赖训练时的 balance loss，推理时直接控制每 expert 的最大 token 数 |
  | Expert 复制方案资源开销大 | Token Drop/Expanded Drop 是纯路由逻辑 | 零额外参数、零额外通信（Expanded Drop 仅限本地设备内扩展），计算开销可忽略 |
  | Gating score 分布平坦 → 丢弃候选 expert 质量高 | 使用 Score-based metric 而非 Order/Random | Token Drop 以 gating score 为重要性度量，丢弃低分 token；Expanded Drop 利用 score 分布平坦特性扩展候选 |

  **论文方法全栈执行例子（以 Mixtral-8×7B-Instruct, 8×H20 GPU, 8-way EP, γ=1.5, Expanded Drop 为例）**：

  - **算法层**：
    1. x [N,d] → Gate: logits [N,8] → Softmax → TopK(k=2) → topk_scores, topk_idx
    2. Expanded Drop: 候选集 = topk_idx ∪ local_expert_ids ([N, 2+m], 本GPU 1-2个expert)
    3. 构建 exp_mask [N,8] 标记扩展候选 → masked_scores = scores × exp_mask
    4. cap = int(1.5 × (N×2) / 8) → 逐 expert 取 top-cap token (dim=0)
    5. 超载 expert 的低分 token 被丢弃，未被丢弃的 token 可能路由到 >2 个 expert（w/o max constraint）
    6. 最终 expert 输出加权求和 → 允许每 token 激活超过 k 个 expert（利用低负载 expert 剩余容量）

  - **系统框架层**：Megatron-LM 修改 → 在 Gate 之后、All-to-All Dispatch 之前插入 Expanded Drop → Router → Expanded Drop（本地）→ All-to-All Dispatch（仅传输保留 token）→ Expert FFN → All-to-All Combine → Add Residual。关键：Expanded Drop 额外 expert 仅限本地设备（m = 本地 expert 数），无跨设备通信。Device-Level 变体将约束从 per-expert N_i ≤ γN̄ 放宽为 ΣN_i ≤ n_l·γN̄，进一步减少过度丢弃。

  - **编译框架层**：论文未明确说明（PyTorch eager execution + NCCL）。

  - **Kernel/运行时调度层**：论文未明确说明。Gate kernel 输出 scores 和 indices → Expanded Drop kernel（topk + cat + scatter + capacity topk，O(N·(k+m)) 纯逻辑操作）→ 后续标准 MoE dispatch/compute/combine kernel 不变。

  - **硬件架构层**：8× H20 GPU → 高负载 GPU 的 expert 处理 token 数因 capacity constraint 从峰值 ~7N̄ 降至 γN̄=1.5N̄ → GPU SM 负载更均衡 → All-to-All barrier 等待时间减少 → 1.85× 端到端加速。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Score-based dropping metric | Order/Reverse Order 不稳定（序列shuffle改变丢弃token），Random无区分度 | Score在γ=1.0时Avg 61.1 vs Order 51.8 (Table 1) |
  | 容量因子 γ=1.5 | 在性能和效率间取得平衡 | γ≥1.5性能接近baseline (Figure 12)，γ=1.5 Mixtral获1.85×加速 |
  | Expanded Drop overselection (k+m) | 低负载expert利用率提升 | Expanded Drop Avg 74.5 vs Token Drop 73.8 on Mixtral (Table 2) |
  | 不限制每token最多k个expert (w/o max) | 允许灵活利用低负载expert剩余容量 | w/o max Avg 61.2 vs w/ max 61.0 (Table 11) |
  | Device-Level capacity (ΣN_i ≤ n_l·γN̄) | 减少因单expert超限的过度丢弃 | Device-Level Avg 74.8 vs Expert-Level 73.9, speedup 1.31× vs 1.23× (Table 3) |
  | Image First 多模态丢弃策略 | 图像token冗余度高，优先丢弃对性能影响小 | Image First Percep. 1362.1 vs Uniform 1307.6 (Table 4) |
  | Expanded Drop本地设备限制 | 避免跨设备通信增加 | Figure 6: 全局扩展增加communication和permutation时间 |

  **创新总结**：Capacity-Aware Inference 的核心洞察是将 MoE 推理中的不可控 routing skew 通过容量约束转化为可控的延迟上限，然后用扩展候选集在容量约束内"回收"丢弃 token 的表示能力。其本质是在 pre-communication 阶段做一个轻量的 token-to-expert 重调度——不是改变 dispatch 通信模式，而是改变 dispatch 的输入（哪些 token 去哪些 expert）。这使其能无缝集成到任何 Expert Parallelism 框架中（Megatron-LM、DeepSpeed 等），无需修改底层通信或 kernel。额外开销仅来自 topk/capacity/scatter 等逻辑操作（vs expert FFN 和 All-to-All 通信可忽略），收益来自 straggler expert 负载的降低和 GPU 利用率均衡。

## Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

- baseline方法是什么？
  **Baseline 为传统并行 MoE（Standard MoE）**：每层 Transformer 的 MoE 模块中，所有 expert 独立并行工作——Router 一次性为每个 token 计算 gating scores（$s_i = \text{Softmax}(e_i^\top x)$），从 N 个 expert 中 TopK 选择 K 个，然后并行执行：$y = \sum_{i=1}^{K} g_i \cdot E_i(x)$。Expert 之间无任何交互或信息传递，所有 expert 在单步 forward pass 中独立完成计算。

  **Baseline 全栈执行例子（以 MoE K=8, C=1, 544M 模型推理一个 token 为例）**：

  - **算法层**：token embedding x → Router: $s_i = \text{Softmax}(e_i^\top x)$ → TopK(s, 8) → 并行激活 8/63 routed experts + 1 shared expert → $y = \sum_{i\in\text{top8}} g_i \cdot E_i(x)$。所有 8 个 expert 看到的是同一个原始输入 x，各自独立计算，无顺序依赖，最后简单加权求和。
  - **系统框架层**：PyTorch + veRL FSDP Trainer（https://github.com/volcengine/verl），标准 MoE forward。Expert 计算可并行化——8 个 expert 的 FFN 可同时执行（batch = 8 expert × tokens_per_expert），最大化 GPU 矩阵乘法并行度。
  - **编译框架层**：论文未明确说明（PyTorch eager execution）。
  - **Kernel/运行时调度层**：论文未明确说明。标准 MoE kernel：Gate kernel → TopK selection → grouped GEMM（8 expert FFN 合并计算）→ weighted sum combine。单步完成，无迭代 loop。
  - **硬件架构层**：NVIDIA H100 GPU，单设备。所有 64 experts（63 routed + 1 shared）参数常驻 GPU 显存。

  **Baseline 的核心缺陷**：
  1. **Expert 独立并行无交互**：expert 之间完全独立，无法进行互补推理——每个 expert 只能从原始输入 x 中提取信息，而非基于其他 expert 已精炼的中间表示。限制了 expert 组合的多样性（最多 C(N,2K) 种组合）。
  2. **静态路由不可迭代调整**：每个 token 在单步中被静态分配到固定的 K 个 expert，无法根据中间计算结果重新评估和调整路由决策。
  3. **"深度"仅能通过增加 Transformer 层数实现**：增加表示深度的唯一方式是加层（L↑），导致参数量和内存线性增长。无 within-layer 的深度扩展机制。
  4. **Scaling 效率低**：扩展模型 capacity 只能通过 width scaling（增加 expert 数 N 或每 token 选择数 K）或 depth scaling（增加层数 L），均带来显著的内存和计算开销。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Chain-of-Experts (CoE) = 将传统 MoE 层的单步并行 expert 计算改为 C 步迭代顺序计算，每步使用独立 Router 基于前一步的中间表示重新选择 expert，并加入 inner residual connection 稳定训练。

  **Defect→Design 映射**：

  | Baseline 缺陷 | CoE 设计选择 | 解决机制 |
  |---|---|---|
  | Expert 独立并行无交互 | C 步迭代处理：$x^{(t)} = \sum g_{t,i} \cdot E_i(x^{(t-1)}) + x^{(t-1)}$ | 每步 expert 的输入是前一步所有 expert 处理后 + residual 的中间表示，形成 "relay race" 式的顺序精炼 |
  | 静态路由不可迭代调整 | Iteration-based Independent Routing：每步独立 Router 参数 $e_{t,i}$ | 第 t 步 Router 基于 $x^{(t-1)}$ 动态重新评估，可自适应选择不同的 expert 集合 |
  | 深度扩展需加层导致内存线性增长 | C 步迭代作为新的 scaling axis（depth through iteration） | C=2, L=4 匹配 MoE L=12 性能但减少 42% memory；C 增加不增加参数或层数 |
  | 组合多样性受限（C(N,2K)种） | 两次 TopK 独立选择：C(N,K)² 种组合 | N=64, K=4 → 823× 更多 expert 组合，显著提升 representational capacity |
  | 缺乏 inner residual 导致训练不稳定 | 每步 inner residual：$x^{(t-1)}$ 直接加到 $x^{(t)}$ | 消融实验：inner residual loss=1.12 vs outer=1.21 vs init=1.18，inner residual 显著优于其他设计 |

  **论文方法全栈执行例子（以 CoE K=4, C=2, 544M 模型推理一个 token 为例）**：

  - **算法层**：
    1. 初始化：$x^{(0)} = x$
    2. 第一步（t=1）：$s_{1,i} = \text{Softmax}(e_{1,i}^\top \cdot x^{(0)})$ → TopK(s₁, 4) → 选择 expert 集合 A（如 experts 3, 15, 28, 42）→ $h_1 = \sum_{i\in A} g_{1,i} \cdot E_i(x^{(0)})$ → $x^{(1)} = h_1 + x^{(0)}$（inner residual）
    3. 第二步（t=2）：$s_{2,i} = \text{Softmax}(e_{2,i}^\top \cdot x^{(1)})$ → TopK(s₂, 4) → 基于 $x^{(1)}$ 重新选择 expert 集合 B（如 experts 5, 18, 33, 60，与 A 可能完全不同）→ $h_2 = \sum_{i\in B} g_{2,i} \cdot E_i(x^{(1)})$ → $x^{(2)} = h_2 + x^{(1)}$
    4. 输出：$y = x^{(2)}$
    总 expert 计算量 = 4+4 = 8，与 baseline MoE (K=8) 完全相同，但多了 inner residual 的 element-wise add（可忽略）。

  - **系统框架层**：PyTorch + 修改的 veRL FSDP Trainer → 扩展支持 multi-round expert execution。与 baseline 相比，CoE 的 forward 增加了 C-1 次额外的 Router+TopK（低成本）和 inner residual add（element-wise，忽略不计）。但由于每步只选 K/C=4 个 expert（vs baseline K=8），单步 grouped GEMM 的并行度减半，这是论文提到的 "time overhead" 来源——H100 上大 batch grouped GEMM 对小 expert 数的利用率下降。

  - **编译框架层**：论文未明确说明（PyTorch eager execution）。

  - **Kernel/运行时调度层**：论文未明确说明。CoE forward kernel 执行序列：Gate_1 → TopK_1 → GroupedGEMM_4experts → ResidualAdd_1 → Gate_2 → TopK_2 → GroupedGEMM_4experts → ResidualAdd_2。vs Baseline：Gate → TopK → GroupedGEMM_8experts → Combine。CoE 多了 kernel launch 次数（2× Gate + 2× TopK + 2× FFN vs 1×）但总 GEMM FLOPs 相同。

  - **硬件架构层**：NVIDIA H100 GPU，单设备。CoE 的主要 hardware-level tradeoff：sequential processing 减少了单步并行度（4 experts/step vs 8），但通过 "depth through iteration" 在总计算量不变的情况下提升模型表达力。在 H100 上的 time overhead（Limitations 中提及）来自小 batch grouped GEMM 的 GPU 利用率下降。

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | C>1 的 iteration depth | MoE 单步并行无交互 → 提供新的 scaling axis | C=2, L=4 匹配 L=12 MoE 性能(-42% memory)；C=2, N=48 匹配 N=64 MoE (-17.6% memory) |
  | Iteration-independent Router | 静态路由无法根据中间状态调整 | 消融：共享 router 导致 loss plateau 在 1.5（远差于独立 router 1.12 + MoE baseline 1.20） |
  | Inner residual every iteration | 多步训练不稳定 | Inner 1.12 vs Outer 1.21 vs Init 1.18 |
  | 保持 sparsity（每步 K/C experts） | 总计算量不变但增加 expert 组合多样性 | Validation loss 1.20→1.12（相同 FLOPs）；823× expert 组合数提升 |
  | 理论分析：C(N,k)² > C(N,2k) | Demonstrates why iterative routing increases representational capacity | 组合空间指数级扩展 |

  **创新总结**：CoE 的核心洞察是将 MoE 的 "shallowly parallel expert processing" 重新定义为 "sequential expert reasoning process"。这不需要修改 expert 架构、不需要增加参数或 FLOPs，仅通过改变 "router 何时调用、基于什么中间状态调用" 来解锁 expert 之间的通信能力。其设计本质是一种 within-layer 的 "recurrence"——类似 Universal Transformer 的跨层参数复用，但 CoE 的 expert 复用发生在同一层内，且每步 Router 独立重新决策。这种设计的代价是减少了单步的矩阵乘法并行度（K/C vs K），在论文的小模型规模下 H100 上用 1 GPU hour 可完成验证；局限性在于尚未在大规模模型（>1B）和多节点训练上验证，且 C>2 时观察到 diminishing returns 甚至不稳定。

**创新总结**：Capacity-Aware Inference 的核心洞察是将 MoE 推理中的不可控 routing skew 通过容量约束转化为可控的延迟上限，然后用扩展候选集在容量约束内"回收"丢弃 token 的表示能力。其本质是在 pre-communication 阶段做一个轻量的 token-to-expert 重调度——不是改变 dispatch 通信模式，而是改变 dispatch 的输入（哪些 token 去哪些 expert）。这使其能无缝集成到任何 Expert Parallelism 框架中（Megatron-LM、DeepSpeed 等），无需修改底层通信或 kernel。额外开销仅来自 topk/capacity/scatter 等逻辑操作（vs expert FFN 和 All-to-All 通信可忽略），收益来自 straggler expert 负载的降低和 GPU 利用率均衡。

## CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

- baseline方法是什么？
  **Baseline 为标准的 LoRA-based MoE PEFT 方法（MixLoRA、MOELoRA、MiLoRA、OMoE 等），以及非 MoE 的 PEFT 方法（LoRA、DoRA）。**

  **Baseline MoE-PEFT 的核心机制**：
  - 将 LoRA 的低秩矩阵 A, B 替换为 n 个并行 expert {E_i = B_i A_i}，通过 Router g(x; G) 进行 top-k 稀疏激活
  - Router 计算每个 expert 的 importance score，选择 top-k 激活，其余 expert 输出贡献为 0
  - 输出: y' = W₀·x + Σ_{i∈T} ĝ_i · E_i(x)

  **这些 baseline 方法的共性问题**：
  1. **Expert Knowledge Redundancy（专家知识冗余）**：缺乏足够的专业化约束导致不同 expert 学习到重叠/相似的功能，浪费 MoE 的容量。
  2. **Expert Load Imbalance（专家负载不均）**：训练中仅部分 expert 被频繁激活，其他 expert 利用不足，违背 MoE 设计初衷。
  3. **Capacity Underutilization（容量利用不足）**：简单堆叠更多 expert 不会线性提升性能（Qian et al., 2024），反而遇到性能瓶颈。

  现有方法（如 load balance loss, localized balancing constraint）尝试缓解上述问题但远不足够。

  **Baseline 全栈执行例子（以 MixLoRA on LLaMA-2 7B 为例）**：

  - **算法层**: 输入 token x → Router 计算 n=8 个 expert 的 gating score → top-2 激活 → 仅 top-2 expert 的 LoRA 输出参与残差计算 → 交叉熵损失反向传播。Router 无专业化约束，expert 参数更新仅由下游任务 loss 驱动，导致 expert 功能趋同。训练后约 2.9% 参数可训练。
  - **系统框架层**: 论文未明确说明推理框架。使用 HuggingFace PEFT + transformers 标准训练/推理流程。MoE-LoRA 的 expert 计算本质是独立的低秩矩阵乘法并行执行后加权求和，不涉及推理框架修改。
  - **编译框架层**: 论文未明确说明（使用 PyTorch eager mode + HuggingFace transformers）。
  - **Kernel/运行时调度层**: 论文未明确说明。expert 的 LoRA 前向为 standard GEMM（B·A·x），多个 expert 在 PyTorch 层面并行计算后 weighted sum。top-k routing 通过 argmax + mask 实现。
  - **硬件架构层**: NVIDIA A6000 48GB GPU，无自定义硬件。

  **Baseline 的核心缺陷**：
  1. **无专业化信号**：Router 仅基于 task loss 隐式学习路由，缺乏显式的 "expert 应差异化" 信号。导致 expert 学习到相似的参数分布（OMoE 论文已通过实验证明 vanilla MoE 的 expert 分布坍缩）。
  2. **非激活 expert 被浪费**：训练中 top-k 只激活少数 expert，非激活 expert 的前向输出被丢弃（乘以 0），反向传播中这些 expert 的参数更新梯度仅间接来自未来可能被选中的概率，缺乏直接利用。
  3. **Balance loss 治标不治本**：Load balance loss 强制 expert 负载均匀但不保证 expert 功能差异化——可能导致不同 expert 在相同功能上轮流激活而非真正专业化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **CoMoE (Contrastive Representation for MoE)** 在 MoE-based PEFT 训练中引入基于 InfoNCE 的对比学习辅助目标，从信息论角度促进 expert 专业化。

  **关键创新——将非激活 expert 从 "浪费" 变为 "负样本"**：
  CoMoE 将 top-k routing 下的非激活 expert 重新利用为对比学习的负样本（negative keys），同时将激活 expert 作为正样本（positive keys）。这种设计使得每个 expert 在训练中同时受到两个方向的梯度信号：
  - **正信号**（来自激活 expert 的 L_CE + 正对比样本）：学习匹配当前输入
  - **负信号**（来自非激活 expert 的负对比样本）：被推离当前输入的表示空间

  **对比 Baseline，CoMoE 全栈执行例子（LLaMA-2 7B, n=4, k=2）**：

  - **算法层**: 
    1. 输入 token x → Router 计算 4 个 expert 的 gating score → top-2 激活（如 expert 1 和 3）
    2. 标准前向：y' = W₀·x + ĝ₁·E₁(x) + ĝ₃·E₃(x) → 计算 L_CE
    3. 对比前向：收集所有 4 个 expert 的输出 E₁(x)...E₄(x) 作为表示向量（不经 weighted sum）
    4. 随机选一个激活 expert（如 expert 1）为 anchor q = Normalize(E₁(x))
    5. 正样本 P = {Normalize(E₃(x))}（另一个激活 expert）
    6. 负样本 N = {Normalize(E₂(x)), Normalize(E₄(x))}（非激活 expert）
    7. 计算 cosine similarity: s_pos = q·E₃(x)/τ, s_neg = q·E₂(x)/τ + q·E₄(x)/τ
    8. InfoNCE: L_con = -log( exp(s_pos) / (exp(s_pos) + Σexp(s_neg)) )
    9. 总损失: L_total = L_CE + 0.01·L_con → 反向传播
    10. **效果**：激活 expert 的表示被拉近（专业化协作），非激活 expert 的表示被推远（避免冗余）。多个 task 自然分配给不同 expert 组合（Figure 4 可视化验证）。

  - **系统框架层**: 论文未明确说明推理框架。训练基于 HuggingFace PEFT + transformers。对比损失计算仅依赖 expert 输出表示（前向传播中已计算的中间结果），不增加额外前向开销。推理时无需对比损失——仅标准 top-k routing 前向，因此推理延迟与 MixLoRA 同级甚至更优（3,789ms vs 4,217ms）。
  
  - **编译框架层**: 论文未明确说明（使用 PyTorch eager mode）。

  - **Kernel/运行时调度层**: 论文未明确说明。对比损失在 Python 层面计算（normalize + dot product + softmax + log），张量操作量 O(n·D) vs expert FFN 计算 O(d_model²)，可忽略不计。

  - **硬件架构层**: NVIDIA A6000 48GB，无自定义硬件。单卡可完成全量实验。训练 3.5h（multi-task, n=4）。

  **CoMoE 解决 Baseline 缺陷的映射关系**：

  | Baseline 缺陷 | CoMoE 解决方案 | 实现机制 |
  |--------------|---------------|---------|
  | Expert 功能冗余（无专业化信号） | 对比损失显式惩罚 expert 表示相似性 | 正样本拉近同类 expert，负样本推远异类 expert → 表示空间分散化（Figure 5 可视化验证） |
  | 非激活 expert 信息浪费 | 将非激活 expert 用作负样本，赋予其训练信号 | s_neg 梯度更新非激活 expert 参数，推动其学习不同于当前输入的功能 |
  | Balance loss 治标不治本 | 对比损失自然产生专业化分工，无需显式 balance loss | Figure 4: 加入 contrastive loss 后不同 task 自然分配到不同 expert 组合（如 ARC-c→expert{1,3}, BoolQ→expert{1,4}） |
  | 堆叠 expert 不线性提升性能 | 通过提升每个 expert 的利用率而非数量来提升容量 | n=4 在 multi-task 上 avg +1.3 超过 n=8 的 MixLoRA（参数效率 2×） |
  | 多任务性能退化（ST→MT 下降 7-12%） | 专业化 expert 更好地处理异质数据分布 | ST→MT 下降仅 0.1-1.8%（vs baseline 4.4-8%） |

  **理论保证**：
  Theorem 1 证明 InfoNCE loss 是对 MI Gap ΔI = I(x; M⁺) - I(x; M⁻) 的紧下界：ΔI ≥ log(N) - L_NCE。最大化该下界 = 最大化输入与激活专家的互信息同时最小化输入与非激活专家的互信息，从信息论角度保证 expert 专业化。

  **核心结果**：
  - Multi-task 平均 accuracy +1.3 (LLaMA-2 7B, vs 最强 baseline)
  - 参数效率 2×：1.45% tunable params vs MixLoRA 2.9%，性能更优
  - 推理延迟降低 10% vs MixLoRA (3,789ms vs 4,217ms)
  - GPU 内存节省 465 MiB vs MixLoRA
  - 固定负样本采样策略将训练复杂度从 O(n) 降至 O(1)，性能无损

## CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory

- baseline方法是什么？
  **Baseline 为 Samba-CoE（MICRO 2024）**：目前唯一探索大规模 CoE 模型部署的系统。Samba-CoE 使用 **FCFS（First-Come, First-Served）请求调度**和 **LRU（Least Recently Used）专家淘汰策略**，在 NUMA 设备上将专家 offload 到 CPU memory/SSD，按需加载到 GPU HBM 进行推理。

  **Baseline 全栈执行例子（以电路板缺陷检测 CoE, 300+ 专家, NUMA RTX3080Ti 12GB, 推理 3 个请求 R1→Expert1, R2→Expert2, R3→Expert1 为例）**：

  - **算法层**：输入组件图像 → Router 确定所需 Expert（如 Expert1 for 组件A, Expert2 for 组件B）→ Expert FFN（ResNet101 或 YOLOv5）执行分类/检测推理。路由规则由用户预定义（人工指定各组件对应专家），CoE 模型不涉及 token-level 动态路由（与 MoE 不同）。

  - **系统框架层**：Samba-CoE FCFS 调度 → R1 到达：Expert1 加载到 GPU → 推理完成 → R2 到达：Expert1 不在队列中可被淘汰 → LRU 淘汰 Expert1 加载 Expert2 → 推理完成 → R3 到达：Expert1 需从 CPU/SSD 重新加载 → 触发 expert switching。在 NUMA 设备上，从 SSD 切换专家占推理延迟 90%+；UMA 设备上占 60%+。

  - **编译框架层**：论文未明确说明（PyTorch 标准执行）。

  - **Kernel/运行时调度层**：标准 PyTorch GPU kernel（ResNet101 conv + YOLOv5 detection head）。Expert 从 CPU→GPU 或 SSD→GPU 通过 PCIe/NVMe 传输，传输时间串行化在推理前。无 kernel 级优化。

  - **硬件架构层**：NUMA 设备（RTX3080Ti 12GB GPU + Xeon CPU 16GB + SSD）或 UMA 设备（Apple M2 24GB 统一内存 + SSD）。GPU 显存不足以容纳全部 300+ 专家（>60GB），需 tiered storage（GPU→CPU→SSD）。

  **Baseline 的核心缺陷**：
  1. **FCFS 调度忽视请求间专家依赖**：依赖同一专家的多个请求在队列中可能被不相关的请求分隔，导致专家被不必要地淘汰后重新加载。例如 R1(Expert1)→R2(Expert2)→R3(Expert1)，R2 可能淘汰 Expert1，R3 再重新加载 Expert1 → 产生可避免的 expert switching。
  2. **LRU 淘汰依赖历史统计而非未来使用概率**：LRU 仅基于过去访问时间预测未来使用，在 CoE 场景中不准确。CoE 的路由规则是预先定义的，可以精确计算每个专家的使用概率，LRU 未利用这一信息。
  3. **静态内存分配未平衡专家加载与 batch 推理**：更大的 batch size 降低平均延迟但消耗更多中间结果内存 → 减少可常驻 GPU 的专家数量 → 增加 expert switching 频率。这一 trade-off 因不同处理器（CPU/GPU）和不同设备架构而异，手工调优困难。
  4. **单 executor 或 naive round-robin 多 executor 未优化负载分配**：不同专家的计算量和访问频率不同，静态均分请求导致部分 executor 过载而其他空闲。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：CoServe = Dependency-aware Request Scheduling + Dependency-aware Expert Management + Offline Profiler。核心洞察是 CoE 系统具有**专家依赖（Expert Dependency）**特性——请求间的依赖（多个请求需要同一专家）和专家间的依赖（后续专家依赖前置专家的输出）——可以用于减少不必要的专家切换。

  **Defect → Design 映射**：

  | Baseline 缺陷 | CoServe 设计选择 | 解决机制 |
  |---|---|---|
  | FCFS 导致可避免的 expert switching | Dependency-aware Request Arranging: 将新请求排在队列中同 expert 请求之后 | 同 expert 请求成组处理，一次加载服务所有同组请求 → expert switching 减少 78.5%-93.87% |
  | FCFS + round-robin 负载不均 | Dependency-aware Request Assigning: 预测额外推理延迟，选择使最大队列时间最小化的 executor | 动态平衡各 executor 负载，最小化总任务时间 |
  | LRU 淘汰低效（纯历史统计） | Dependency-aware Expert Management: 两阶段淘汰——先淘汰无前置依赖的后续专家，再按使用概率升序淘汰 | 使用预评估的专家使用概率（来自路由规则）替代历史统计 → 淘汰决策更准确 |
  | 静态内存分配无法适应不同硬件 | Offline Profiler + Sliding Decay Window: 通过 microbenchmarks + CDF 搜索最优专家数量 | 自动适应不同设备（NUMA/UMA），找到专家加载 vs batch 推理的最佳平衡点 |
  | 静态 executor 数量配置 | Offline Profiler 搜索最优 executor 组合（如 G3C1 vs G4C1） | 根据 workload 特征自动选择最优 executor 配置 |

  **CoServe 全栈执行例子（以同一场景 R1→Expert1, R2→Expert2, R3→Expert1, NUMA RTX3080Ti, 3 GPU executors 为例）**：

  - **算法层**：同 Baseline——Router 确定所需 Expert，Expert FFN 执行分类。CoServe 不修改算法层（不改变专家模型本身）。

  - **系统框架层**（关键差异）：
    1. Request Scheduler 预测 R1 在各 executor 队列的额外延迟 → 分配给 executor 1
    2. R2 到达 → Scheduler 预测：R2 需要 Expert2，各队列均无 Expert2 → 选择总时间最小的 executor（如 executor 2）
    3. R3 到达 → Scheduler 预测：R3 需要 Expert1 → executor 1 队列中有 Expert1 的请求 → 切换延迟=0 → 分配给 executor 1
    4. Request Arranging: R3 排在 R1 之后（同 Expert1）
    5. executor 1 处理：R1 batch + R3 batch 一起用 Expert1 推理，Expert1 仅加载一次
    6. 若 Expert1 不在 model pool：Expert Manager Stage 1 淘汰无前置依赖的闲置专家 → Stage 2 按使用概率淘汰 → 加载 Expert1

  - **编译框架层**：论文未明确说明（PyTorch eager execution）。

  - **Kernel/运行时调度层**：GPU executor 执行 expert FFN（ResNet101/YOLOv5）→ CPU executor 并行执行低优先级 batch → Expert loading (SSD→GPU) 与 GPU 推理可部分重叠（通过并行 executor）。Request scheduling 由 CPU 执行，与 GPU 推理并行。

  - **硬件架构层**：同 Baseline（RTX3080Ti 12GB + Xeon + SSD / Apple M2 24GB + SSD）。Offline profiler 自动确定 GPU 加载 35 个专家（Task A）或 34 个专家（Task B），剩余内存用于 batch 推理。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Samba-CoE):
  R1(Expert1) → R2(Expert2) → R3(Expert1)
  FCFS: Expert1 load → infer R1 → LRU evict Expert1 → 
  Expert2 load → infer R2 → Expert2 in pool → 
  Expert1 reload → infer R3
  专家切换: 3 次加载

  CoServe (Dependency-aware):
  R1(Expert1) → R3(Expert1) → R2(Expert2)
  Scheduling: R1→executor1, R3→executor1 (same expert, 排在R1后), R2→executor2
  Expert1 load → infer R1+R3 (batch) → Expert2 load → infer R2
  专家切换: 2 次加载（减少 33%）

  更复杂场景下（300+ 专家，大量请求），效果放大：
  CoServe 减少 expert switching 78.5%-93.87% → 吞吐量提升 4.5×-12×
  ```

  **关键创新总结**：CoServe 的核心洞察是将 CoE 系统与 MoE 系统的**关键区别**（CoE 的路由规则是预定义的、可离线分析的）转化为系统优化的机会：
  1. CoE 的路由规则可以预计算专家使用概率（MoE 无法做到，因为 MoE router 在推理时动态输出）→ 替代 LRU 的历史统计实现更准确的淘汰决策
  2. CoE 的专家依赖关系（后续专家等待前置专家结果）可以用于优先级排序 → 两阶段淘汰策略（优先淘汰"尚未需要的"而非"最近最少用的"）
  3. CoE 的请求-专家映射可以从路由规则提前获知 → 请求调度可以在知道未来需求的情况下做出安排（而非仅依赖 FCFS）

  这些优化都源于 CoE 区别于 MoE 的根本属性：路由的预定义性和可分析性。

## Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

- baseline方法是什么？
  Baseline 为两类 SOTA MoE offloading 系统：

  1. **DeepSpeed-ZeRO-Inference [3]**（batch=1 类代表）：利用 MoE 的稀疏激活特性，每次只加载激活的 expert weights 到 GPU HBM，通过 expert prediction 和 hot expert caching 降低传输开销。适用于小 batch size，在 batch=1 时 GPU 利用率仅 0.76%（Mixtral 8x7B on A30）。

  2. **MoE-Lightning [6]**（throughput-oriented 代表）：通过增大 batch size 对抗 MoE 稀疏性，专家权重加载后 GPU 可并行处理多个 input。KV cache 维护在 CPU DRAM 中，attention 计算在 CPU 执行。但因 CPU DRAM 容量限制，batch size 仍有限，理论 GPU 利用率仅 3.13%。

  **Baseline 全栈执行例子（以 MoE-Lightning on Mixtral-8x7B, A30 GPU 为例）**：

  - **算法层**：输入 tokens → Router 计算 top-2 experts → 加载 expert weights (CPU DRAM→GPU HBM) → Expert FFN (3×矩阵乘法: W_gate, W_up, W_down) → CPU Attention (GEMV 对 KV cache in CPU DRAM) → 逐 token 自回归生成。每次 forward 仅处理 1 个 token per request。无 speculative decoding。

  - **系统框架层**：MoE-Lightning 系统 → batch 拆分为 micro-batches → GPU Other1→CPU Attention→GPU Other2→GPU MoE 流水线 → expert weights 异步预取 → KV cache 在 CPU DRAM, attention 在 CPU 执行。

  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA 执行）。

  - **Kernel/运行时调度层**：CPU Attention kernel (GEMV, 单 token) → GPU MoE kernel (GEMM, expert FFN) → HtoD Transfer (expert weights 加载)。瓶颈分析 (Figure 3 Roofline): MoE layer 主要受 CPU-GPU memory transfer for expert weights 限制（memory-bound），Attention layer 受 CPU memory access for KV cache 限制（memory-bound）。

  - **硬件架构层**：A30 GPU (165 TFLOPS, 933 GB/s) + Intel Xeon Gold 6426Y CPU + 250 GB CPU DRAM。GPU 和 CPU 通过 PCIe 连接 (25 GB/s)。Mixtral-8x7B FP16 占 87GB，CPU DRAM 中约 160GB 留给 KV cache。

  **Baseline 的核心缺陷**：
  1. **GPU 利用率极低**：batch=1 方案 GPU 利用率仅 0.76%，MoE-Lightning 仅 3.13%。根本原因是 expert loading 的 I/O 瓶颈和 MoE 稀疏激活导致每次 forward 处理 token 数太少，GPU compute 能力远未被充分利用。
  2. **CPU-GPU transfer 瓶颈**：MoE layer 的 CPU→GPU expert weight 传输几乎占满 PCIe 带宽，成为 memory-bound bottleneck。
  3. **CPU memory access 瓶颈**：Attention layer 的 CPU KV cache 访问受限于 CPU memory bandwidth，CPU computational power 也未被充分利用。
  4. **固定 hyperparameter 配置**：MoE-Lightning 使用固定 batch size/micro-batch size/缓存策略，无法在不同硬件和 workload 下自适应调节。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：SpecMoEOff = speculative decoding (EAGLE) + CPU chunked attention kernel + memory-conscious draft execution + hyperparameter optimizer。核心洞察：speculative decoding 增大每次 forward 的 workload（从 1 token 变为 k+1 tokens），天然对齐 MoE offloading 场景下 GPU 资源闲置的问题。

  **Defect→Design 映射**：

  | Baseline 缺陷 | SpecMoEOff 设计选择 | 解决机制 |
  |---|---|---|
  | GPU 利用率仅 0.76%-3.13% | Speculative decoding: draft model 生成 k tokens, target model 一次性验证 | 每次 forward 处理 k+1 tokens 而非 1 token → 增大 operational intensity → GPU MoE 层从 memory-bound 转为更接近 compute-bound |
  | CPU-GPU transfer 瓶颈 (memory-bound) | 增大每次 forward 的 batch size (k 倍 token 数) | 相同 transfer 量下计算更多 token → 每 token 的 transfer cost 分摊 |
  | CPU KV cache memory access 瓶颈 | Chunked attention: 合并 k 个 draft tokens 的 KV cache access 为一次读取 | d× 次 KV cache 读取 → 1× 次，充分利用 CPU memory bandwidth |
  | Attention 在 CPU 上低效 (per-token GEMV) | CPU chunked attention kernel (Intel MKL GEMM) | GEMV→GEMM (Q@K^T), 利用 CPU SIMD/MIMD 高效执行 |
  | Draft model KV cache 溢出 GPU HBM | CPU/GPU separation: batch 维度分片, GPU Part (全 GPU 执行) + CPU Part (CPU attention + GPU FFN) | 动态调整分离比例，优先级: draft model > target model (draft 多次调用，target 仅一次) |
  | 固定 hyperparameter 无法自适应 | Hyperparameter optimizer: 凸优化预决定 + profiling estimator DAG 模拟 | 自动搜索最优 k, 适应不同硬件/模型/workload |
  | Naive mask 存储浪费内存 | Mask 压缩: 仅存储 draft-to-draft 子区域 (n×n) | O(n·(l+n)) → O(n²) mask 内存 |

  **SpecMoEOff 全栈执行例子（以 Mixtral-8x7B + EAGLE draft, k=5 draft tokens, 单 iteration 为例）**：

  - **算法层**：
    1. Draft model (EAGLE, <2GB in GPU HBM): 输入当前 hidden state → iterative generate k=5 draft tokens (GPU Part + CPU Part 并行 attention → GPU FFN)
    2. Target model verify: 将 original tokens + 5 draft tokens 拼接 → CPU Chunked Attention (Intel MKL GEMM, Q[5,4096]@K^T[4096,517]→[5,517]) → GPU MoE (expert weights CPU→GPU, FFN x(5+1)tokens) → accepted = a(5) ≈ 3-4 tokens
    3. Next iteration: update KV cache with accepted tokens

  - **系统框架层**：SGLang + SpecMoEOff:
    - Target model pipeline: microbatch 1 的 GPU Other1 → CPU Attention (concurrent with microbatch 2 的 GPU Other1) → microbatch 1 的 GPU Other2 → GPU MoE (concurrent with microbatch 2 的 CPU Attention)
    - Next layer expert weights: async HtoD transfer (separate CUDA Stream, concurrent with current layer compute)
    - Draft model: GPU Part + CPU Part 并行 → 生成 k draft tokens → verification
    - Memory Manager: expert cache (GPU HBM) + KV cache (CPU DRAM target, GPU/CPU split draft)

  - **编译框架层**：论文未明确说明。

  - **Kernel/运行时调度层**：CPU Chunked Attention (Intel MKL GEMM → softmax → MKL GEMM, mask 压缩仅 5×5) → GPU MoE GEMM (expert FFN, 3× matrix per expert, fused MoE implementation) → HtoD Transfer (separate CUDA Stream, CUDA Event synchronization) → GPU Attention (draft model GPU Part, FlashAttention-style)。

  - **硬件架构层**：A30/4090D + Intel Xeon CPU + CPU DRAM → GPU 和 CPU 通过 PCIe (25/23 GB/s) 连接 → 所有 expert weights 均在 CPU DRAM → hot expert cache 在 GPU HBM (如 5.25 GB) → KV cache 在 CPU DRAM (target) + GPU/CPU split (draft) → speculation 减少 per-token PCIe transfer cost。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (MoE-Lightning):
  iteration: [1 token forward] x until generation done
  per iteration: CPU Attention(GEMV,1 token) → CPU→GPU load experts → GPU MoE(1 token)
  GPU utilization: 3.13%, bottleneck: PCIe transfer + CPU mem bandwidth

  SpecMoEOff:
  iteration: [draft k tokens → verify (k+1) tokens] x until generation done
  per iteration: Draft: GPU/CPU parallel attn + GPU FFN (k times, tiny model)
              → Verify: CPU Chunked Attention(GEMM, k+1 tokens) → 
                CPU→GPU load experts → GPU MoE(k+1 tokens)
  GPU utilization: improved (more compute per load), 
  speedup: 2.5× decode throughput over MoE-Lightning
  ```

  **关键设计对应关系**：

  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Speculative decoding (EAGLE) | GPU utilization from 3.13% higher | 1.45×-1.53× speedup over w/o-sd baseline |
  | CPU chunked attention (Intel MKL) | CPU attention 成为 bottleneck (Figure 13) | CPU Attention: 4.29s actual, 10.6% estimation error |
  | CPU/GPU draft separation | Draft KV cache > GPU HBM (A30 24GB) | 动态分离比例, 初始更多GPU, 随seqlen增长迁移CPU |
  | Hyperparameter optimizer | 固定 k 无法最优 (Figure 10 shows non-monotonic) | Dynamic k 比 fixed-k best 额外 +2% throughput |
  | Mask 压缩 | 减少 CPU DRAM 占用 | O(n×(l+n))→O(n²) |

  **Roofline 分析的核心发现**：
  - MoE layer (GPU compute + CPU-GPU transfer): arithmetic intensity 仅 3.13% of GPU peak → speculative decoding 增大 b×k 倍 token 数 → 直接提升 operational intensity
  - Attention layer (CPU memory access): KV cache access 是瓶颈 → chunked attention 将 n 次读取合并为 1 次
  - Speculative decoding 恰好同时缓解两个瓶颈

## BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

- baseline方法是什么？
  Baseline 为 **Fine-Grained MoE（DeepSeekMoE 式）**，基于 CDAC（communicate-descend-ascend-communicate）执行方式：

  在 CDAC 方式下，每个 MoE 层的执行顺序为：
  1. **Communicate**：将 token 通过 All-to-All 发送到对应 expert 所在设备（此时 token 维度为 full hidden dimension h）
  2. **Descend**：各 expert 内部通过 descending projection ($W_{i,\downarrow}$) 将 token 从 h 降维到 FFN intermediate dimension $h_f$
  3. **Ascend**：各 expert 内部通过 ascending projection ($W_{i,\uparrow}$) 将 token 从 $h_f$ 升维回 h
  4. **Communicate**：再次 All-to-All 将各 expert 输出汇集回源设备

  All-to-All 通信量 C = 2 × top_k × (ep-1)/ep × b × s × h，正比于 hidden dimension h。当 top_k 增大时（如从 top-1 到 top-8），通信量线性增长 7.1-7.3 倍，成为训练和推理的主导延迟（占比高达 90%+）。

  **Baseline 全栈执行例子（以 GPT-Fine-Grained, top_k=8, ep=32 推理一个 MoE layer 为例）**：

  - **算法层**：输入 x ∈ R^{batch×seq×2048} → Router: TopK(Softmax(x·W_gate), k=8) → All-to-All dispatch（高维 2048 通信，1,488 GB transfer for ep=32）→ 64 个 expert 各自执行 E_i(x) = σ(x·W_{i,↓})·W_{i,↑}（每个 expert h=2048→h_f→2048）→ All-to-All combine（高维 2048）→ 输出 y。通信占迭代时间 79.2-90.6%（随 top_k 增大而增加）。
  - **系统框架层**：Megatron-LM EP（expert parallelism）=32，跨 32 GPU 分发 64 experts。All-to-All 使用 NCCL 通信原语，token dispatch 和 combine 各一次全交换。Megatron 中 TP-SP 通信还涉及 All-to-All、All-Gather、Reduce-Scatter（在 TP group 内），均在高维度进行。
  - **编译框架层**：论文未明确说明（NCCL + cuBLAS 标准执行）。
  - **Kernel/运行时调度层**：All-to-All kernel（NCCL）→ Expert FFN GEMM kernel（64 experts 分布在 32 GPU 上，每 GPU 2 experts）→ 高维 token 搬运占通信时间主导。
  - **硬件架构层**：32 GPUs（48 GB HBM each, PCIe 4.0 x16），4 节点 × 8 GPU，100 Gbps InfiniBand。All-to-All 通信成为瓶颈，top_k=8 时占总延迟 91.8%（训练）/ 90.6%（推理）。

  **Baseline 的核心缺陷**：
  1. **All-to-All 在高维度进行**：CDAC 模式下，expert 内部降维-升维投影在 All-to-All 之后，导致通信始终在全维度 h 进行，通信量巨大。
  2. **top_k 增大会加剧通信瓶颈**：fine-grained MoE 需要更多 small experts 和更大 top_k 来保证性能，但通信量与 top_k 线性增长，限制模型扩展。
  3. **系统级优化效果有限**：Tutel 的 overlap 和 Lina 的带宽协调等系统优化在 fine-grained MoE 场景下效果有限——计算量被大幅减少后（small experts），overlap 窗口太小，带宽优化空间被压缩。
  4. **压缩方案损害模型质量**：ScheMoE 的 ZFP 压缩等 lossy 压缩方法会降低模型质量，且引入额外压缩/解压计算开销。
  5. **Expert capacity 限制丢 token**：为缓解 imbalanced routing 带来的 straggler 问题，传统 MoE 设置 expert capacity（capacity factor f=1~1.25），超限 token 被丢弃，直接损害模型质量。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **BigMac**，核心创新是将 fine-grained MoE 的 CDAC 执行顺序改为 **DCCA（descend-communicate-communicate-ascend）**，配合重新设计的 expert 结构：

  **DCCA 策略**：
  1. **Descend**：在 All-to-All 通信之前插入 descending projection ($W'_{\downarrow}$)，将 token 从 h 压缩到 r·h（r=0.25，如 2048→512）
  2. **Communicate**：All-to-All dispatch —— 在低维度 r·h 进行
  3. **Communicate**：All-to-All combine —— 继续在低维度 r·h 进行
  4. **Ascend**：ascending projection ($W'_{\uparrow}$) 将 token 从 r·h 恢复到 h

  通信量降至原来的 r 倍（-75%），仅增加 +4.54% FLOPs 和 +1.35% 参数。

  **BigMac Expert 重新设计**：由于 DCCA 已将输入维度降至 r·h，expert 内部改为先升维再降维（$E_i(x) = \sigma(xW_{i,\uparrow})W_{i,\downarrow}$），保证与 fine-grained MoE expert 相同的参数量和计算复杂度。

  **Defect → Design 映射**：

  | Baseline 缺陷 | BigMac 设计选择 | 解决机制 |
  |---|---|---|
  | All-to-All 在高维度进行 | DCCA：先 descend 再 communicate | 通信维度从 h 降至 r·h（-75%), All-to-All 通信量减少至 25% |
  | top_k 增大会加剧通信瓶颈 | 通信量与 top_k 解耦（DCCA 框架下 top_k 的影响被 r 打折扣） | Top8 BigMac 比 Top4 Fine-Grained 快 27.7-55.4% |
  | 系统优化在 fine-grained 场景效力有限 | 算法层直接减少通信量（不依赖系统层 overlap） | Megatron 上 2.45-3.07× training speedup；Tutel 上 1.71-3.09× speedup |
  | 压缩方案损害模型质量 | DCCA 是结构级优化（非 lossy compression），无精度损失 | BigMac 质量与 Fine-Grained 相当或更优（Table 6,7） |
  | expert capacity 丢 token | 通信减少后可移除 capacity 限制实现 dropless routing | 进一步改善模型质量（enables dropless token routing） |

  **BigMac 全栈执行例子（以 GPT-BigMac, top_k=8, r=0.25, ep=32 推理一个 MoE layer 为例）**：

  - **算法层**：输入 x ∈ R^{batch×seq×2048} → Router: TopK(Softmax(x·W_gate), k=8)（在 full dimension 2048 做路由以保证精度）→ Descend: x' = x·W'↓（2048→512, 压缩至 r=0.25）→ All-to-All dispatch（低维 512 通信，仅 372 GB transfer，-75% vs baseline）→ 64 experts 各自执行 E_i(x') = σ(x'·W_{i,↑})·W_{i,↓}（先在 expert 内升维 512→h_f→512，保证 expert 复杂度）→ All-to-All combine（低维 512）→ Ascend: y = y'·W'↑（512→2048, 恢复原始维度）→ 输出。额外优势：可承受更大 top_k（如 Top8）而无通信代价惩罚；可移除 expert capacity 限制实现 dropless routing。

  - **系统框架层**：Megatron-LM / Tutel / DeepSpeed-Inference 无需修改系统逻辑——DCCA 仅改变模型结构（projection 顺序），通信调用方式不变。在 Megatron 中，TP-SP 通信（All-to-All、All-Gather、Reduce-Scatter）也从高维降为低维，减少 1.42-2.34×。

  - **编译框架层**：论文未明确说明。

  - **Kernel/运行时调度层**：All-to-All kernel 搬运的 token 维度从 2048 降至 512（-75% data volume）。Expert FFN 计算量不变（通过先升维保证），但通信 kernel 耗时大幅缩短。Tutel 的 2DH All-to-All + overlap 与 BigMac 正交叠加。

  - **硬件架构层**：32 GPUs × 4 nodes, 100 Gbps InfiniBand。BigMac 将 All-to-All 通信瓶颈从占 91.8% 显著降低，端到端训练延迟加速 1.53-3.09×（跨 Megatron/Tutel），推理吞吐提升 1.62-3.11×（跨 Megatron/Tutel/DeepSpeed-Inference）。

  **关键设计对应关系**：

  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | DCCA（descend before communicate） | 通信在高维度 | All-to-All 从 1,488 GB → 372 GB (-75%) |
  | BigMac Expert（先升后降） | DCCA 降维后 expert 参数减少 | Param: 3.73B → 3.78B (+1.35% only) |
  | r=0.25 downscaling factor | 平衡通信节省与模型质量 | FLOPs +4.54% 换取通信 -75% |
  | Routing at full dimension | 保证门控精度（低维路由会退化） | 选择 x（非 x'）作为 gate input |
  | Dropless routing | 通信减少后移除 capacity 限制 | 进一步提升模型质量（避免丢 token 损失） |

  **创新总结**：BigMac 的核心洞察是——fine-grained MoE 的 CDAC 方式将 All-to-All 放在了最高维度（通信最贵），通过重新排列 projection 和通信的顺序（DCCA），只需增加极少的 FLOPs（+4.54%）和参数（+1.35%），即可将通信量减少 75%。这是一个纯算法/模型结构层面的优化，与系统级优化（Tutel、Lina 等）正交叠加。更重要的是，BigMac 解耦了 top_k 与通信成本的关系，使 fine-grained MoE 可以使用更大的 top_k 以获得更好的模型质量，而无需承受通信代价。


## Comet Fine-grained Computation-communication Overlapping for Mixture-of-Experts

- baseline方法是什么？
  **Baseline 为 Megatron-LM 中的标准 MoE 执行方式**，包括四种变体：

  1. **Megatron-Cutlass**: Megatron-LM 默认实现，使用 CUTLASS GroupGEMM 作为 expert FFN kernel，通信（NCCL all-to-all）和计算（GEMM）顺序执行，**无任何通信-计算重叠**。

  2. **Megatron-TE (Transformer Engine)**: 使用 NVIDIA Transformer Engine 的 GEMM 实现，同样无通信-计算重叠。TP 下因 Transformer Engine API 调用开销，部分配置下性能甚至差于 Megatron-Cutlass。

  3. **FasterMoE**: 通过自定义 Scatter 和 Gather 算子将 expert 计算分为 2 个 chunk 实现 pipeline overlap（pipeline degree=2，即 coarse-grained）。仅 hide 29.2% 通信延迟，同时引入了额外的 local indexing 开销。

  4. **Tutel**: 通过自适应并行策略切换、2D 分层 all-to-all 和启发式搜索实现 partial overlap。hide 68.6% 通信延迟，但 expert 数量大时（如 Qwen2 的 E=64）CPU 端 scheduling overhead 增大导致优势衰减。

  **Baseline 全栈执行例子（以 Mixtral 8x7B, EP=8, TP=1, M=16384, 一个 MoE layer 为例）**：

  - **算法层**: Router 计算 top-2 gating → 全量 all-to-all dispatch tokens → 等待通信完成 → 各 GPU 执行持有的 experts 的 GEMM（layer0 → activation → layer1）→ 全量 all-to-all combine → 等待通信完成 → top-K reduce → 输出。整个 pipeline 完全顺序执行：通信阻塞计算，计算阻塞通信。

  - **系统框架层**: Megatron-LM (Expert Parallelism + Data Parallelism)。MoE layer forward = `token_permutation → alltoall_dispatch (NCCL) → expert_gemm (CUTLASS GEMM, 每 expert 独立 kernel launch) → alltoall_combine (NCCL) → token_unpermutation → reduce`。Host CPU 需要为每步通信和计算分别 launch kernel，kernel launch overhead 在小 M 时占比显著。

  - **编译框架层**: PyTorch eager execution + NCCL + CUTLASS。无通信-计算融合编译优化。

  - **Kernel 调度层**: Expert GEMM 使用 CUTLASS group_gemm。所有 expert 的 GEMM tile 统一调度，无 tile 级重排序。通信使用 NCCL all-to-all，按完整大 tensor 一次传输。通信和计算通过独立的 CUDA streams 发射（如 Tutel/FasterMoE），但 coarse-grained chunk 划分导致重叠效率低——初始和最后的通信阶段无计算可重叠，产生 pipeline bubble。

  - **硬件架构层**: 8× H800 GPU (NVLink)。通信阶段 GPU SM 大量空闲（仅 NCCL kernel 使用少量 SM 做数据搬运），计算阶段通信链路空闲。GPU compute utilization 在通信期间接近 0。

  **Baseline 的核心缺陷**：
  1. **粒度不匹配（Granularity Mismatch）**: MoE 的通信以 token 为单位（单个 token 是最小数据搬运单元），但 GEMM 以 tile（如 128×128）为计算粒度——一个 GEMM tile 需要 128 个 token 的数据，这些 token 可能分布在多个 remote GPU 上。Coarse-grained pipeline（FasterMoE/Tutel）必须等一个 chunk 中所有 token 到齐才能启动计算，导致 tile 粒度以下的等待无法消除。
  2. **数据依赖复杂**: MoE gate 在运行时动态决定 token→expert 映射，每个 GEMM tile 所需的 token 随机分布在多 GPU 上。计算 tile 不能开始直到其依赖的所有 token（local + remote）可用，但在 coarse-grained 通信中远程 token 只能按 chunk 整体到达，无法按 tile 粒度就绪。
  3. **通信和计算负载动态变化**: MoE 的 token 分布不均衡（不同 expert 接收不同数量的 token），通信量和计算量在运行时动态变化。将通信和计算封装在独立 kernel 中（FasterMoE, Tutel 的做法）使得 GPU SM 资源分配在编译时固定，无法根据运行时负载自适应调整，导致重叠中的气泡（bubble）。
  4. **Host 端 scheduling overhead**: 多个独立 kernel launch 之间需要 CPU 端调度（尤其是 Tutel 的 adaptive scheduling 和 FasterMoE 的 multi-expert kernel），在小 M（短序列）时 CPU scheduling 成为 dominant overhead。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Comet** 提出两个核心设计解决 MoE 中通信-计算的 fine-grained overlapping：

  **Defect → Design 映射**：

  | Baseline 缺陷 | Comet 设计选择 | 解决机制 |
  |---|---|---|
  | 粒度不匹配（token 通信 vs tile 计算） | **Shared Tensor Based Dependency Resolving**: 沿 M（layer0）或 N（layer1）维度分解 shared tensor → tile 计算重排序 | 将 token 级通信与 tile 级计算对齐——local token tiles 优先计算，remote token tiles 延后（此时通信已并发完成） |
  | 数据依赖复杂（tile 级等待） | Tile 按数据依赖重排序：layer0 按 source rank 排序优先计算最小依赖 tile，layer1 改为 column-wise 并行 | 每个 tile 能在其依赖的 tokens 就绪后立即开始（而非等待整个 chunk） |
  | 通信/计算负载动态变化 | **Adaptive Workload Assignment**: Thread block specialization + 自适应分配 n^c/n^p 比例 | 通信和计算在同一个 fused kernel 内执行但隔离到不同 thread blocks，运行时根据 M 和 parallelism 选择最优比例 |
  | Host 端 scheduling overhead | 通信和计算融合在单个 GPU kernel 中 | 消除多次 kernel launch 的 CPU↔GPU 往返延迟 |

  **Comet 方法全栈执行例子（同 Mixtral 8x7B, EP=8, TP=1, M=16384, 一个 MoE layer）**：

  - **算法层（不变）**: 同 baseline，Router 计算 top-2 gating → 决定 token→expert 映射。

  - **系统框架层**: Megatron-LM + Comet Python API。MoE layer forward 从多步 kernel launch 变为两个 Comet fused kernel：
    ```
    # Baseline: ~6+ kernel launches
    token_permute → alltoall_dispatch → expert0_gemm → expert1_gemm → ... → alltoall_combine → reduce
    
    # Comet: 2 fused kernel launches
    comet_layer0(shared_tensor, routing_map, expert_weights)  # NVSHMEM recv + GroupGEMM
    comet_layer1(shared_tensor, routing_map, expert_weights)  # column-wise GEMM + reduce + NVSHMEM send
    ```

  - **编译框架层**: 论文未明确说明（CUDA C++ + CUTLASS 模板 + NVSHMEM，无 Triton/TVM 编译层）。

  - **Kernel 调度层（核心创新）**:
    
    **Layer0 (Communication→Computation Pipeline)**:
    1. Shared Tensor 识别: layer0 的 shared tensor = dispatch buffer [M×topk, N]，是通信(producer)的输出和 GEMM(consumer)的输入
    2. 沿 M 维度分解: shared tensor 按行（token 粒度）分解 → 每个 token 独立可作为 GEMM 的输入
    3. Token 重排序: 所有需要参与 GroupGEMM 的 tokens 按 source rank 排序 → local tokens 聚集在前（无需通信，立即可用），remote tokens 聚集在后
    4. GroupGEMM tile 调度: tile 计算顺序重新编排——仅含 local tokens 的 tile 优先计算 → 含部分 remote tokens 的 tile 等 NVSHMEM 完成 → 纯 remote tile 最后。在计算早期 tiles 的同时，NVSHMEM 通信 thread blocks 正在拉取后续 tiles 所需的 remote tokens
    5. Thread block 隔离: 通信 TB 执行 NVSHMEM `get` → shared tensor buffer。计算 TB 执行 CUTLASS GroupGEMM（TMA async copy + tensor core MMA）。两套 TB 由 SM hardware scheduler 并发调度，互不干扰。

    **Layer1 (Computation→Communication Pipeline)**:
    1. Shared Tensor 识别: layer1 的 shared tensor = GEMM 输出 buffer [M×topk, N]，是 GEMM(producer)的输出和 reduce+通信(consumer)的输入
    2. 沿 N 维度分解: shared tensor 按列（hidden dim 粒度）分解为 N/T^N 个列块
    3. Column-wise GEMM: 所有 expert 并行计算第 1 个列块 → T^N 列完成后立即 top-K reduce → NVSHMEM write 回 source rank → 同时继续计算第 2 个列块 → ...
    4. 重叠效果: reduce+通信 与 后续列的 GEMM 计算完全重叠。Baseline 必须等所有 expert 全部列计算完才开始 reduce+通信。

    **Adaptive Assignment**: 预编译内核库含多个 n^c/n^p 比例（n^c=18/26/46...）→ deployment 前 profile → runtime 按 M 和 (EP,TP) 查表选择最优 kernel。

  - **硬件架构层**: 8× H800 GPU (NVLink)。NVSHMEM 分配 buffer = 2×M×N bytes（M=16384 时 128MB），跨所有 MoE layers 全局复用。Comet 对 GEMM thread block 使用标准 CUTLASS Hopper 实现（TMA + MMA），通信 thread block 额外占用 SM 资源但通过隔离避免了干扰 GEMM 的异步计算流水线。

  **对比 Baseline 的核心改进路径**：
  ```
  Baseline (Megatron-Cutlass/FasterMoE/Tutel):
  Tokens → NCCL all-to-all (full tensor, wait complete)
  → Expert GEMM (all tiles, sequential)
  → NCCL all-to-all (full tensor, wait complete)
  → Reduce
  通信占 47% total time, zero 或 partial overlap (29-69%)
  
  Comet:
  Tokens → [Fused Kernel Start]
    ├─ Comm TB: NVSHMEM get token-by-token → shared tensor buffer
    ├─ Compute TB: GroupGEMM tile 0 (local tokens, no wait)
    ├─ Comm TB: more NVSHMEM get (background)
    ├─ Compute TB: GroupGEMM tile 1 (ready tokens)
    ├─ ... (fine-grained interleaving)
    ├─ Compute TB: Column-wise GEMM col 0 → reduce → NVSHMEM send
    ├─ Compute TB: Column-wise GEMM col 1 (while col 0 reducing)
    └─ ...
  → [Fused Kernel End]
  Hide 86.5% communication, 1.96× single-layer speedup, 1.71× end-to-end speedup
  ```

  **关键设计对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Shared tensor 沿 M/N 维分解 | 打破 coarse-grained 数据依赖，token 和 tile 粒度对齐 | 通信隐藏从 29-69% → 86.5% |
  | Tile 按数据依赖重排序（local→remote） | Complex data dependency 导致的 tile 级等待 | Expert compute efficiency 不受 partitioning 影响（t_1+t_2 ≈ t） |
  | Thread block specialization（隔离通信/计算） | Fine-grained I/O 拖慢 GEMM (尤其是 Hopper TMA 流水线) | GEMM TB 使用标准 CUTLASS，零性能退化 |
  | Adaptive n^c/n^p 分配 | 运行时负载动态变化（不同 M, TP, EP 下最优分配不同） | 不同配置自动选择最优 kernel, 无需人工 tuning |
  | NVSHMEM 替代 NCCL | Token 级 fine-grained remote I/O (NCCL 只支持 coarse-grained) | Unified Virtual Address 逐 token 访问 |
  | 单 fused kernel 替代多 kernel launch | Host 端 scheduling overhead (小 M 时 dominant) | 小 M (256-1024) 时 speedup 更高 (2.37×) |

  **创新总结**: Comet 的核心洞察是——MoE 通信和计算之间存在复杂的 token-tile 数据依赖，coarse-grained pipelining（按 chunk 重叠）无法消除这种依赖导致的等待。通过将 shared tensor 沿正确维度分解并重调度 tile 计算顺序，Comet 将粗粒度的 chunk 级重叠升级为 fine-grained 的 tile 级重叠。进一步地，通过 thread block specialization 将通信和计算隔离到同一 kernel 的不同 TB，避免了 fine-grained I/O 干扰 high-performance GEMM（尤其是 Hopper 的 TMA 异步流水线）。这种方法本质上是将 MoE 系统优化从 "kernel 间调度" 下沉到 "kernel 内调度"，消除了 CPU 端 scheduling overhead 并实现了精准的 GPU SM 资源分配。

## Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

- baseline方法是什么？
  **Baseline 为现有的 MoE 推理方案**：GPU 内存中驻留全部 expert 参数，或使用 offloading 策略将非激活 expert 存储在 CPU 主存中，通过 PCIe 总线按需传输到 GPU。当前 offloading 系统面临的关键瓶颈是 PCIe 带宽有限（PCIe 4.0 约 32 GB/s），远低于 GPU 内存带宽（约 300 GB/s），导致专家加载延迟。**现有压缩方案（量化）**虽然能减少传输参数大小，但低比特量化（1-4 bit）会导致不可控、不可预测的误差，严重损害生成性能（如 QMoE 的 20× 压缩伴随 6.7% 精度下降，CMoE 的 150× 压缩伴随 23.81% 精度下降）。此外，之前的 MoE 压缩工作缺乏对不同 expert、不同层在面临压缩误差时敏感性差异的系统性理解，导致无法设计针对性的分层压缩策略（如对误差敏感层分配更精确的压缩，对鲁棒层使用更高压缩比）。

  **Baseline 全栈执行例子（以 Moonlight-16B MoE 模型在 GPU 内存受限的单 GPU 上推理一个 GSM8K 数学题为例）**：

  - **算法层**：输入 token 序列 → Self-Attention（QKV 投影 + Attention 计算 + Output 投影）→ Router（top-6 gating）选择 6 个 expert → 选中的 expert 参数若不在 GPU 内存中则从 CPU 主存传输 → 各 expert FFN 计算（$W_{up}, W_{gate}, W_{down}$）→ 加权求和输出 → 下一层。非激活 expert 占约 70% 参数（~66.6 GB of 94 GB for Mixtral-8x7B）浪费 GPU 内存。
  - **系统框架层**：Offloading 框架（如 MoE-Infinity/Pre-gated MoE）管理 GPU resident store（驻留常访问 expert）+ CPU main memory（存储所有 expert 参数）+ GPU staging buffer（预取动态数据）。当 GPU 访问不在 resident store 的 expert 时，计算 stall 直到 PCIe 传输完成。
  - **编译框架层**：论文未明确说明（基于 PyTorch/HuggingFace Transformers 推理 pipeline）。
  - **Kernel/运行时调度层**：PCIe DMA 传输 expert 权重（FP16）→ GPU kernel 执行 GEMM。传统无压缩时每个 expert 满精度传输。量化方案（如 4-bit）减少传输量但引入不可控量化误差。论文未明确说明具体 kernel 实现。
  - **硬件架构层**：论文未明确说明 GPU 型号和 CPU 配置。分析依赖 PCIe 带宽（~32 GB/s for PCIe 4.0）vs GPU 内存带宽（~300 GB/s）的对比。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出使用 **error-bounded lossy compression（有界误差有损压缩）**，如 SZ3（CPU）和 CuSZp（GPU），替代传统量化方案来压缩 MoE 非激活 expert 参数，并对不同 expert 的压缩误差敏感性进行系统性分析。核心思路：
  
  - **有界误差保证**：与传统量化产生"不可控、不可预测"的误差不同，error-bounded 压缩算法保证每个重建参数与原始参数的绝对误差 ≤ 预设的 error bound ê。这使得压缩对推理精度的影响可控、可预测。
  - **高压缩比且低误差**：SZ3/CuSZp 等预测-量化-编码管线可以在给定 error bound 下实现比简单量化更高的压缩比，因为利用了参数值的空间相关性（如 Lorenzo predictor 利用相邻参数的梯度预测）。
  - **分层误差敏感性分析**：通过全面的 error injection 实验（正态分布 N(0, ê) 误差注入模拟压缩解压后的参数状态），系统性地揭示了三层分层敏感性：
    - **浅层（Shallow layers, L1-L10）**：主要负责 attention 机制和输入 token→向量表示的转换，对压缩误差高度鲁棒，引入误差后推理精度下降最小（如 Layer1 Top-6 experts 注入 80% error: ICA 0.74 vs baseline 0.85）。可以承受更高压缩比。
    - **中层（Middle layers, L13）**：负责核心模型推理/逻辑分析，对压缩误差极为敏感，误差会显著损害推理精度（如 Layer13 全部 64 个 expert 注入 80% error: ICA 暴跌至 0.38 vs baseline 0.86, PIA 降至 0.65）。需要最保守的压缩策略，或使用更小的 error bound。
    - **深层（Deep layers, L20-L26）**：负责指令跟随和输出整合，对误差有"反直觉"的鲁棒性甚至增益——注入 bounded error 有时会提升推理精度（如 Layer26 Top-6 experts 注入 80% error: ICA 升至 0.90 vs baseline 0.85），可能源于深层 expert 的隐性集成效应（implicit ensemble effect），引入噪声使模型生成更多样化的输出整合。

  - **路由机制的适应性保护**：实验发现当高激活频率 expert 参数被扰动时，路由机制会动态调整路由权重，将任务重新分配给其他功能完整的 expert，保护核心推理能力（PIA 保持稳定 ≥ 94%）。
  - **功能解耦发现**：ICA（指令合规精度）和 PIA（纯推理精度）对参数误差的解耦响应说明，semantic generation 和 instruction parsing 在 MoE 架构中是功能分离的，分别由不同层的 expert 负责。

  **论文方法全栈执行例子（以 Moonlight-16B 推理 GSM8K 为例，error-bounded compression + 感知分层的 offloading 流程）**：

  - **算法层**：Input token → Self-Attention → Router 选择 top-6 experts → **分层压缩策略**：浅层 expert 使用较大 error bound（高压缩比），中层 expert 使用极小 error bound（保守压缩），深层 expert 可使用中等 error bound（甚至利用噪声增益）→ SZ3/CuSZp 压缩后 expert 参数通过 PCIe 传输 → GPU 端解压重建（参数含 bounded error）→ Expert FFN → 加权输出。压缩后参数 = 原始参数 + 误差（误差 ∈ [-ê, ê] 由压缩算法保证）。
  - **系统框架层**：在现有 offloading 框架基础上，在 GPU staging buffer 和 CPU main memory 之间插入压缩/解压模块。CPU 端压缩 expert（SZ3）→ 压缩数据通过 PCIe 传输（数据量减少）→ GPU 端解压（CuSZp）→ 解压参数加载到 GPU resident store → Expert FFN 计算。pipeline 可设计为：压缩/解压与传输重叠，进一步减少延迟。
  - **编译框架层**：论文未明确说明。
  - **Kernel/运行时调度层**：CPU 端 SZ3 使用多线程 Lorenzo predictor + Huffman 编码压缩 expert 权重张量；GPU 端 CuSZp 使用 CUDA kernel 并行解压。传输数据量 = 原始 size × (1/压缩比)，压缩比由 error bound ê 决定。论文未明确说明具体 kernel 实现细节。
  - **硬件架构层**：论文未明确说明 GPU/CPU 具体型号。分析框架适用于支持 PCIe 通信的 GPU-CPU 异构系统。

## Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

- baseline方法是什么？
  **Baseline 为两类**：
  
  1. **MoNDE [18]（GPU-NDP, context-agnostic expert placement）**：基于全局历史专家激活频率统计进行静态 hot/cold 分类——hot experts 常驻 GPU HBM（FP16），cold experts 驻留 CXL-NDP 设备执行。当 router 选中的 expert 不在 GPU 时，触发 on-demand 迁移（expert weights 从 NDP → GPU via PCIe）或由 NDP 执行。核心问题：
     - **静态/全局频率忽略 context dependence**：同一 expert 对不同输入序列的 heat 程度不同，全局统计无法捕捉 per-sequence 的激活变化
     - **on-demand expert migration**：decoding 期间频繁触发参数传输（expert weight ~数百 MB/次），PCIe 带宽成为瓶颈，GPU 利用率下降
     - **NDP compute pressure 未解决**：cold experts 以 FP16 在 NDP 上执行，NDP 计算吞吐有限（64×(4×4) systolic arrays），成为系统瓶颈
  
  2. **HOBBIT [31]（GPU-only mixed-precision offloading）**：所有 experts 和 attention 均在 GPU 执行，使用混合精度（FP16 + INT4）从 CPU memory 加载 expert weights。缺陷：无 NDP 近数据执行优势，expert weights 仍需通过 PCIe 传输到 GPU。

  **Baseline 全栈执行例子（以 MoNDE, Mixtral-8×7B, 4 GPU/4 NDP experts/layer 推理一个序列为例）**：
  
  - **算法层**：输入 tokens → Self-Attention → Router (top-2 gating) → 每 token 选择 2 个 experts → 若选中 expert 在 GPU → 本地 FP16 FFN；若选中 expert 在 NDP → 若 on-demand policy → expert weight (FP16, ~每 expert ~170MB for Mixtral-8×7B) 从 NDP DDR 经 PCIe → GPU HBM → GPU FFN 计算，或 activation 经 PCIe → NDP → NDP FP16 计算 → activation 回传 GPU
  - **系统框架层**：MoNDE runtime → GPU memory manager 维护 hot expert cache → cold experts 在 NDP DDR → expert migration triggered per layer per decoding step → 迁移延迟 90%+ of transformer block time
  - **编译框架层**：论文未明确说明（标准 PyTorch + custom runtime）
  - **Kernel调度/运行时计算层**：GPU 端标准 cuBLAS GEMM；NDP 端 systolic array FP16 GEMM → NDP compute units bottleneck（64 arrays × 16 MAC 仅 ~1 TOPS vs H100 989 TFLOP/s）
  - **硬件架构层**：H100 GPU + DDR-based NDP via PCIe Gen4 ×16 → PCIe BW ~32 GB/s vs HBM3 ~3.35 TB/s → 参数传输主导延迟

  **Baseline 的核心缺陷**：
  1. **Context-agnostic placement 忽略 expert activation dynamics**：不同输入序列、不同 decoding step 的 expert 激活模式不同，静态/频率统计无法捕捉
  2. **Parameter Movement 开销巨大**：Expert weight 传输数百 MB vs activation 仅数 KB —— MoNDE 将计算延迟转化为参数传输延迟
  3. **NDP compute pressure**：FP16 在受限 NDP 硬件（64× systolic arrays）上执行成为瓶颈，低量化位宽是释放 NDP 潜力的关键
  4. **频繁 migration 的带宽争用**：on-demand swapping 在 decoding 期间持续触发 GPU↔NDP 传输，挤占 pipeline 效率
  5. **Prefill 信息被浪费**：Prefill 阶段自然产生的 expert 激活统计信息未被利用，decoding 从零开始做决策

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：Context-Aware MoE Inference on CXL-NDP = **Prefill-Guided Expert Placement** + **Context-Aware Mixed-Precision Quantization for NDP**。核心洞察：prefill 阶段的 expert 激活分布与 decoding 阶段高度相似（cosine similarity avg=0.89），因此 prefill 统计可作为 decoding 阶段 placement 和 quantization 的可靠先验。

  **Defect → Design 映射**：

  | Baseline 缺陷 | 论文设计选择 | 解决机制 |
  |---|---|---|
  | Context-agnostic placement 忽略动态性 | Prefill-guided placement: S_{l,e} = αP̃_{l,e} + (1-α)W̃_{l,e} 基于 prefill 统计 | Prefill-decode activation 相似度 0.89 → per-sequence once placement，捕捉 context-dependent hot/cold 模式 |
  | Parameter Movement 开销（~数百 MB/expert） | NDP 近数据执行 + single migration: expert 仅在 prefill 后迁移一次 | 转化为 Activation Movement (~8KB/token) → 数据移动量减少 10^4-10^5× |
  | NDP compute pressure (FP16 瓶颈) | Context-aware mixed precision: 1-4 bit per-expert based on importance | NDP 量化执行 ~5-8× latency reduction，硬件计算压力降低 |
  | 频繁 migration 带宽争用 | Once-per-sequence fixed placement: decoding 期间 zero migration | PCIe 仅用于 activation 传输，无 expert weight 争用 |
  | Prefill 信息被浪费 | Prefill 阶段即收集 (P_{l,e}, W_{l,e})，驱动 placement + bitwidth 决策 | 零额外推理开销（counter update 可忽略），信息复用最大化 |
  | 统一量化忽略 expert heterogeneity | Prefix-structured 分配：重要 expert 高 bitwidth，次要 expert 低 bitwidth | 在固定平均 bitwidth budget 下最大化量化增益，Ours-2bit + selector vs w/o selector: +3.2% avg accuracy |

  **论文方法全栈执行例子（以 Mixtral-8×7B, K=4 GPU/4 NDP experts/layer, Ours-3bit, 一个推理请求为例）**：

  - **算法层**：
    1. Prefill: tokens → Attention → Router → 收集 (P_{l,e}, W_{l,e}) for all 32 layers, 8 experts/layer
    2. Importance: S_{l,e} = 0.5P̃_{l,e} + 0.5W̃_{l,e} → per-layer top-4 → H_l (GPU), bottom-4 → C_l (NDP)
    3. Bitwidth: 4 NDP experts, b_bar=3 → R=8 increments → prefix-structured search → e.g., (n4=2, n3=2): top-2 NDP experts → 4-bit GPTQ, bottom-2 → 2-bit GPTQ
    4. Decoding: per token → Router → top-2 experts → if GPU → H100 FP16 FFN; if NDP → activation sent via PCIe → NDP systolic array (3-bit/2-bit) → result back → weighted sum
    5. Overlap: GPU computing layer l hot experts while NDP computing layer l-1 cold experts → pipeline efficiency

  - **系统框架层**：
    - Prefill 统计收集器：轻量级 per-layer counter array (8 counters × 2 metrics × 32 layers = 512 values)
    - Expert Placement Module：prefill 后执行一次 O(L·E) 排序，单次 expert migration (PCIe weight transfer)
    - Expert Bitwidth Selector：O(L·E_NDP^2) 前缀枚举搜索，per-sequence 执行一次
    - Decoding Runtime：固定 placement，在 GPU 和 NDP 间按 router 结果分派计算

  - **编译框架层**：论文未明确说明。

  - **Kernel调度/运行时计算层**：
    - GPU 端：H100 tensor cores 执行 FP16 GEMM（hot experts），标准 cuBLAS
    - NDP 端：量化 GEMM on 64×(4×4) systolic arrays——不同 bitwidth 的 effective throughput 不同（4-bit ~4× faster than FP16 equivalent, 1-bit uses XNOR+popcount）
    - PCIe activation transfer：per-token 8KB (4096-dim FP16) vs per-expert weight ~170MB — 约 2×10^4× 减少
    - GPU-NDP pipeline overlap：GPU stream 1 (hot FFN) || NDP stream (cold FFN via PCIe)

  - **硬件架构层**：
    - H100 GPU: 80GB HBM3, 132 SMs, 989.4 TFLOP/s → 处理 hot experts + attention + router
    - CXL-NDP: 512GB DDR, 512 GB/s internal BW, 64×(4×4) systolic arrays @ 1 GHz → 处理 cold experts (量化)
    - PCIe Gen4 ×16: ~32 GB/s → activation movement 通道（非 parameter movement 通道）
    - 关键：NDP 512 GB/s 内部带宽 >> PCIe 32 GB/s → 近数据执行利用 NDP 高内部带宽

  **对比 Baseline 的核心改进路径**：
  Baseline (MoNDE, context-agnostic):
  Prefill (no stats) → Decoding: per step per layer: Router → if cold expert needed → [Parameter Movement: ~170MB expert weight NDP→GPU via PCIe] OR [NDP FP16 compute: systolic arrays bottleneck] → GPU wait → FFN → next layer
  瓶颈: PCIe parameter transfer OR NDP compute (FP16)

  Ours (Context-Aware):
  Prefill → [Collect (P,W) stats] → [Importance + Placement + Bitwidth, once] → [Single expert migration, once] → Decoding: per step per layer: Router → if GPU: local FP16 FFN; if NDP: [Activation Movement: ~8KB via PCIe] → [NDP b-bit compute, 5-8× faster] → [Activation back via PCIe]; GPU hot FFN || NDP cold FFN (overlap)
  优势: Activation Movement + NDP low-bit compute + zero decoding migration + overlap

  **关键设计决策对应关系**：
  | 设计选择 | 解决的具体问题 | 数值验证 |
  |---|---|---|
  | Prefill-guided once-per-sequence placement | 消除 decoding 期间 expert migration | 6.6-8.3× speedup vs MoNDE |
  | Context-aware mixed-precision (1-4 bit) | 降低 NDP compute pressure | NDP latency 5× (3-bit) / 8× (2-bit) reduction |
  | Prefix-structured bitwidth allocation | 在固定 avg bitwidth 下最大化精度 | +3.2% accuracy (Ours-2bit w/ selector vs w/o) |
  | Prefill statistics (activation freq + routing score) | 捕捉 context-dependent expert importance | Prefill-decode cosine sim 0.89 |
  | Parameter → Activation Movement | 从 ~170MB→~8KB per expert invocation | PCIe 带宽使用大幅降低 |
  | GPU-NDP pipeline overlap | NDP compute hidden behind GPU compute | 端到端 8.7× decoding throughput vs MoNDE |
  | 保留部分 FP16 experts on GPU (K per layer) | 关键 experts 无损精度 | 仅 0.13% avg accuracy drop (Ours-3bit) |

## Continual Pre-training of MoEs: How robust is your router?

- baseline方法是什么？
  **Baseline 为 Dense Transformer 的持续预训练策略（直接应用于 MoE）**：
  
  从 Ibrahim et al. (2024) 确立的 Dense CPT 策略出发——LR Re-warming + LR Re-decaying + Replay——直接应用于 MoE，不针对 MoE 特有的路由机制做任何修改。具体配置：(1) 从衰减后 checkpoint 开始 CPT，用 cosine decay 重新 warmup 到 $\eta_{max}$ 再 decay；(2) 按固定 replay 比例混合新旧数据。

  **Baseline 全栈执行例子（以 PB Granular MoE, 400B FineWeb → 200B German, decayed checkpoint CPT, 一个训练 step 为例）**：
  
  - **算法层**：输入 1024 sequences × 2048 tokens → 24 层 MoE decoder → 每层：Attention → Router (linear proj [1024, 32] softmax top-3) → 3 细粒度 GEGLU FFN experts + 1 shared expert → weighted sum → Aux Loss + Z-Loss 加入总 loss → next layer
  - **系统框架层**：GPT-NeoX + Megablocks grouped GEMM kernel → 64 A100 GPU Dataparallel + ZeRO-1 → dropless MoE 前向（Megablocks 处理稀疏 expert dispatch）→ AdamW optimizer step
  - **编译框架层**：论文未明确说明（PyTorch eager execution + GPT-NeoX 框架）
  - **Kernel/运行时调度层**：Megablocks grouped GEMM kernel 处理 MoE 稀疏前向（group tokens by expert → batched GEMM）→ NCCL All-Reduce for Data Parallel → ZeRO-1 分散 optimizer states
  - **硬件架构层**：64× A100 GPU，标准数据中心配置，无特殊硬件

  **Baseline 的核心缺陷**：
  1. **路由算法对分布偏移的鲁棒性未知**：MoE 的 PBTk/SBTk 路由是在 IID 预训练数据上设计的，分布偏移可能导致 router 在新旧分布间失衡，加剧遗忘或破坏 expert 负载均衡
  2. **LR Re-warming 对路由的影响未知**：从衰减 checkpoint 大幅 warmup LR 可能导致 router 经历"混沌期"，token 分配剧烈波动
  3. **Replay 对 MoE 路由的影响未知**：replay 旧数据可能干扰 router 对新分布的适应，也可能帮助维持旧分布上的负载均衡
  4. **缺乏 CPT 场景下的路由诊断工具**：没有类似 MRI 的指标来量化分布偏移对 MoE 最坏情况延迟的影响
  5. **MoE CPT 与 Full Re-training 的性能差距未知**：不知道 CPT MoE 能否匹配重训练 MoE 的性能

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文方法**：系统性实证研究，不提出新算法，而是(1) 验证现有 Dense CPT 策略（Infinite LR + Replay）对 MoE 的适用性；(2) 量化两种路由算法和两种 MoE 架构在 CPT 下的鲁棒性；(3) 提出 MRI 指标和三种路由行为分析指标来诊断 CPT 中的路由变化；(4) 将 CPT MoE 与 Full Re-training MoE 直接对比。
  
  **Defect → Design/Discovery 映射**：

  | Baseline 缺陷 | 论文方法/发现 | 解决机制 |
  |---|---|---|
  | 路由对分布偏移鲁棒性未知 | 实验证明 PBTk/SBTk 均对分布偏移**惊人鲁棒**——即使 0% replay，MRI 也在 500 step 内恢复到比 SBTk 更好的水平 | PBTk 的 Aux Loss + Z-Loss 足以在 CPT 中维持负载均衡；SBTk 显式平衡更稳定但最终 MRI 更高 |
  | LR Re-warming 对路由影响未知 | 对比 decayed vs non-decayed checkpoint CPT：Non-decayed (CosineInf) 减少遗忘且不牺牲适应，路由混沌期更短 | CosineInf schedule 在 CPT 阶段不 warmup，直接从 $\eta_{const}$ 继续，避免剧烈 LR 变化对 router 的冲击 |
  | Replay 对 MoE 路由影响未知 | Replay 对 MoE 和 Dense 的效果相似：减少遗忘但稍损害适应。对 PBTk 的 MRI spike 有轻微缓解作用 | Replay 保持 router 对旧分布的记忆 → 减少早期层路由变化 → 减少遗忘 |
  | 缺乏 CPT 路由诊断工具 | 提出 **MRI**（Maximum Routing Imbalance）作为最坏情况延迟代理；扩展三个路由行为指标（Router Saturation / Vocabulary Specialization / Expert Co-activation）到 CPT 场景 | MRI = $\max_i$ ( routed tokens to expert i / total tokens)，直接量化延迟风险 |
  | 不知道 CPT MoE 能否匹配 Full Re-training | CPT PB Granular MoE 在 German 和 Code 任务上匹配甚至超越 Full Re-training MoE 性能（<1% accuracy 差距），同时仅用约 1/3 计算量 | MoE 的更大参数量在 CPT 中起正则化作用 → 比 Dense 忘记更少 |
  | Switch MoE 早期层 MRI 不稳定 | 发现 Granular MoE 架构在早期层 MRI 更稳定 → 推荐 CPT 使用 Granular MoE | 细粒度 experts + shared expert 提供更稳定的路由分布 |

  **论文方法全栈执行例子（以 PB Granular MoE, CosineInf + 40% Replay, 400B FineWeb → 200B German CPT, 一个训练 step 为例）**：
  
  - **算法层**：
    1. Data Sampling: 1024 sequences × 2048 tokens，其中 40% (410 seqs) 从 FineWeb replay，60% (614 seqs) 从 German CC
    2. 24 层 decoder 前向：每层 Attention → Router (W_r [1024, 32] · x → softmax → top-3 experts from 31 routed + 1 shared) → 3 细粒度 GEGLU FFN (intermediate=704) + 1 shared GEGLU FFN → weighted combination → 输出
    3. Loss = CrossEntropy(LM) + 0.01×Aux Loss + 0.001×Z-Loss（Aux Loss 鼓励 31 experts 均匀负载，Z-Loss 惩罚大 router logits）
    4. Backward: AdamW (β1=0.9, β2=0.95, wd=0.1, grad clip=1.0), LR=1.65×10^{-4} (CosineInf constant phase)
    5. MRI 记录：per-layer max(load_i / total_tokens)，monitor 最坏情况延迟风险
  
  - **系统框架层**：
    - GPT-NeoX 框架 + Llama3 tokenizer (128K vocab)
    - Megablocks grouped GEMM kernel：将分配到同一 expert 的 tokens 分组 → batched GEMM → dropless 执行
    - 64 A100 GPU：Data Parallel + ZeRO-1 (optimizer states 分片)

  - **编译框架层**：论文未明确说明（PyTorch eager + Megablocks kernel 编译为 CUDA）
  
  - **Kernel/运行时调度层**：
    - Megablocks grouped GEMM：输入 [S, H] tokens → Router → token-to-expert mapping → group tokens by expert → per-expert GEMM (batched) → scatter output to original token positions
    - ZeRO-1 All-Reduce：gradient synchronization across 64 GPUs → optimizer step per GPU → broadcast updated params
    - 关键：Granular MoE 的 forward pass (485ms) 比 Switch MoE (449ms) 慢约 8%，backward 慢约 13%（更多 experts 的 dispatch overhead）
  
  - **硬件架构层**：
    - 64× A100 GPU，标准配置。论文未明确说明互联方式（NVLink/PCIe）或显存容量
    - MoE step time ~1679ms (PB Granular) vs Dense ~880ms → MoE 约 2× slower per step，但因样本效率优势总体更强

  **关键发现与设计指导**：
  | 发现 | 对 CPT MoE 的指导 |
  |---|---|
  | CosineInf (non-decayed) 优于 Cosine Decay (decayed) CPT | CPT 应在预训练阶段就使用 Infinite LR Schedule |
  | Replay 对 MoE 和 Dense 效果相似 | 按 Dense CPT 经验选择 Replay 比例即可 |
  | PBTk 性能始终优于 SBTk（首次在大规模实验中证实） | CPT 优先使用 PBTk (Aux Loss + Z-Loss) 路由 |
  | Granular MoE 优于 Switch MoE，且早期层 MRI 更稳定 | CPT 优先使用细粒度 expert 架构 |
  | 早期 MoE 层（0-2）路由变化最大 → 与遗忘最相关 | 未来可研究对早期层特殊处理（freeze/lower LR）以减少遗忘 |
  | CPT MoE 匹配 Full Re-training MoE，仅需 ~1/3 计算量 | CPT 是 Full Re-training 的高效替代方案 |
  | MRI 在 PBTk 下分布偏移后短暂飙升但 500 step 内恢复 | CPT 中短期 MRI spike 无需特别干预 |

## DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

- baseline方法是什么？
  Baseline 为 DeepSeek-VL（LLaVA-style 架构，hybrid vision encoder: SigLIP-384 + SAM-B-1024, dense LLM 7B, 固定双分辨率 384×384 和 1024×1024）。具体痛点：(1) **固定分辨率限制**：DeepSeek-VL 的 hybrid vision encoder 仅支持固定 1024×1024 和 384×384 两种分辨率，无法高效处理极端宽高比的高分辨率图像（如 InfographicVQA 中的超长图），导致细节信息丢失。(2) **Dense LLM 参数效率低**：DeepSeek-VL 使用 7B dense LLM，所有参数在每次推理时均激活，计算和显存效率低，模型规模扩展困难。(3) **视觉定位能力缺失**：DeepSeek-VL 不支持视觉定位（visual grounding），无法输出目标物体的 bounding box，限制了在 embodied AI 和 agent 场景中的应用。(4) **训练数据质量不足**：开源图像描述数据集质量参差不齐（短描述、文本不匹配、幻觉），影响模型的多模态理解能力。

  **Baseline 全栈执行例子（以 DeepSeek-VL 7B, 单张 1024×1024 图像 + 文本 query decode 为例）**：
  - **算法层**: Hybrid Encoder: SigLIP-384 (coarse) + SAM-B-1024 (fine) -> Concat -> MLP Projector -> Dense 7B LLM Decoder。固定双分辨率策略，不支持动态 tile。无 visual grounding 能力。Next-token prediction only (visual+text tokens)。
  - **系统框架层**: HAI-LLM 框架，标准数据并行+流水线并行训练。dense LLM 全参数激活，每 token 计算量固定。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 Transformer attention kernel, FFN kernel on A100 Tensor Cores。
  - **硬件架构层**: NVIDIA A100 GPU 集群训练，7B dense 模型可部署在单 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **(1) Dynamic Tiling Strategy**：替代 DeepSeek-VL 的 hybrid fixed-resolution encoder。通过候选分辨率集 C={(m·384, n·384)|1≤m,n≤9} 最小化 padding 面积选择最佳分辨率，将高分辨率图像动态切分为 m×n 个 384×384 local tiles + 1 global thumbnail tile。所有 tile 通过单个 SigLIP-SO400M-384 共享编码，再经 2×2 pixel shuffle 压缩（27×27→14×14, 196 tokens/tile），通过 <tile_newline> 和 <view_separator> special tokens 组织 visual sequence。优势：(a) 支持任意宽高比高分辨率图像，不限固定 1024×1024；(b) tile 数可控（≤81+1），视觉 token 数随分辨率线性增长而非平方增长（local attention 特性）；(c) 统一使用单一 SigLIP 编码器，简化架构。

  **(2) DeepSeekMoE LLM with MLA**：将 dense LLM 替换为 MoE + MLA 架构。MLA 通过低秩压缩 K_t^{C}=W^{UK}·(W^{DKV}·h_t)，V_t^{C}=W^{UV}·(W^{DKV}·h_t) 大幅减少 KV cache（rank=512 vs embedding dim=2048/2560），提高推理吞吐。MoE 使用 2 shared + 64~72 routed experts，每 token 仅激活 Top-6 routed+2 shared=8 experts，3B→0.57B/16B→2.4B/27B→4.1B activated params。优势：(a) 稀疏激活大幅降低每 token 计算量；(b) MLA 压缩 KV cache 使长序列推理更高效；(c) 总参数大但激活参数少，训练推理效率高。

  **(3) 精细化三阶段训练 + 数据质量管控**：Stage 1 VL Alignment: 冻结 LLM 训练 vision encoder+MLP 实现 visual-textual 对齐。Stage 2 VL Pretraining: 全参数训练 ~800B tokens (70% VL + 30% text-only)，通过内部 captioner + DeepSeek Chat 质量评分过滤低质量描述。Stage 3 SFT: ~20B tokens 涵盖 VQA/OCR/文档/图表/数学/定位/grounded conversation 等多任务。数据增强包括：negative samples 防止幻觉定位、中英双语多轮对话消除语言混用、Visual Prompt QA 支持箭头/框/圈/涂鸦理解。

  **(4) Visual Grounding**：引入 <|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|> special tokens 实现：(a) 视觉定位——给定文字描述输出 bounding box；(b) Grounded conversation——在对话回复中引用具体目标位置；(c) In-context visual grounding——跨图像目标的参照理解。

  **论文方法全栈执行例子（以 DeepSeek-VL2, 4.5B activated, 单张高分辨率 3000×1000 图像 + text query decode 为例）**：
  - **算法层**: Dynamic Tiling 选择 m=8,n=2 分辨率 (3072×768, padding=174,336)，切 16 local tiles + 1 global thumbnail。SigLIP-SO400M-384 编码 17 tiles × 196 tokens = 3,332 visual tokens + special tokens → ~3,400 total visual tokens。MLA (rank=512) 压缩 KV: c_KV->k_C,v_C (latent dim 512, up-projected per head), decoupled RoPE (d_h^R=64)。DeepSeekMoE: 2 shared + 72 routed, Sigmoid gating+expert bias correction, Top-6 activation -> 8 experts per token。MLA+MoE: 每 token 仅 4.1B activated / 8 experts active，KV cache 大小为同等 dense MHA 的 512/2560≈20%。
  - **系统框架层**: HAI-LLM 框架，pipeline parallelism（fine-grained vision encoder layer division 防止 pipeline bubble）+ tensor parallelism + expert parallelism。Image tile load balancing across data parallel ranks。双 pipeline strategy 按数据是否为纯文本切换。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 FlashAttention kernel on A100。MLA attention 需自定义 fused kernel（低秩压缩+上投影+RoPE 融合）。MoE all-to-all dispatch/combine 通信。
  - **硬件架构层**: NVIDIA A100 GPU 集群（336 GPUs for DeepSeek-VL2, 7×8 GPU nodes），节点内 NVLink，节点间 InfiniBand。FP32 optimizer states，无 BF16 optimizer（Tiny/Small）或 BF16 optimizer（DeepSeek-VL2）。推理部署在单 GPU (80GB)。

## Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

- baseline方法是什么？
  Baseline 为两种视角：(1) **未压缩 MoE 模型**：Mixtral-8×7B (47B total/13B activated, 87.7GB memory) 和 DeepSeek-MoE-16B (30.8GB memory)，包含全量 experts、全量 layers 和全量 blocks。(2) **现有压缩方法**：Expert Drop（Lu et al. 2024, Muzio et al. 2024）按重要性评分移除不重要 expert，减少参数量和内存但仍保留 MoE 层内的昂贵计算和 expert 间通信开销，speedup 不足 1%；Pruning（Wanda/SparseGPT）和 Quantization（GPTQ/AWQ）作为独立 Expert Slimming 技术存在但未与结构化 Expert Trimming 集成。Baseline 的核心痛点：(a) Expert Drop 移除 expert 后仍保留 MoE layer 内的 costly computation（expert FFN 的前向计算）和 communication overhead（分布式环境下的 All-to-All 通信），导致尽管参数减少但 inference speedup 微乎其微；(b) Expert Drop 破坏路由模式——部分 expert 被移除后，router 对某些输入可能选中"错误"的剩余 expert，导致性能大幅下降（如 MMLU 23% 下降 at 25% experts dropped）；(c) Expert Slimming 技术（pruning/quantization）仅关注单个 expert 内部压缩，未联合解决跨 expert 的结构冗余；(d) dense pruning 方法（Wanda/SparseGPT）在 MoE 上应用时，不考虑 MoE 的 inductive bias（如 shared expert vs routed expert 的不同冗余特性），对 shared expert 误剪枝导致额外性能损失（+3.6% for shared expert exclusion）。

  **Baseline 全栈执行例子（以 Mixtral-8×7B, Expert Drop 12.5%, 128 sequence × 2048 tokens batch 为例）**：
  - **算法层**: 加载 Mixtral-8×7B checkpoint。Expert Drop: 每层按 G(x) 平均路由分选择 7/8 expert 保留（丢弃 1 expert），更新 router weight G ← G_{i∈T'}。保留的 7 experts 仍做 MoE FFN（x → router Top-2 → active expert FFN → weighted sum）。FLOPs 不变（仍激活 2/7 experts，expert FFN 计算不变），内存减少约 1/8 expert 参数量，speedup < 1% 因 expert 内计算和通信未减少。
  - **系统框架层**: HuggingFace Transformers + AWQ/GPTQ 量化框架 + LM Evaluation Harness。未修改 serving 框架。Batch forward pass on input seq_len=2048, batch_size=1~8。论文未明确说明 serving framework。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 PyTorch FP16 matrix multiply kernels on NVIDIA GPUs。量化模型使用 INT4 GEMM kernels (AWQ/GPTQ 量化后)。论文未明确说明具体 kernel 实现。
  - **硬件架构层**: NVIDIA GPU (RTX 3090 作为部署目标提及)。Mixtral-8×7B FP16 需 87.7GB，量化后 24.4GB (AWQ 4-bit)，可在 24GB consumer GPU 部署。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出统一的 MoE 压缩框架，从两个互补视角系统性地解决 Baseline 缺陷：

  **(1) Layer Drop 解决 Expert Drop 的"保留专家内计算+通信"问题**：不是移除单个 expert，而是直接移除整个 MoE 层（含 Norm 模块）。通过 S^{(NM)} = cos_sim(x', x'+MoE(Norm(x'))) 评估每层的输入输出相似度来选择冗余层。移除整个 MoE 层后：(a) 消除了该层内所有 expert FFN 计算（彻底避免 cost computation within experts）；(b) 消除了该层的 All-to-All 通信（彻底避免 complex communication among experts）；(c) 减少参数量和内存（移除整层所有 expert）。实验：Mixtral-8×7B 用 Layer Drop 移除 8/32 MoE 层后 MMLU 仅降 1%，speedup 显著提升 vs Expert Drop 的 <1%。

  **(2) Block Drop 进一步解决 Layer Drop 中 Attention 计算保留问题**：Layer Drop 保留了 computation-costly attention layers。Block Drop 通过评估 block 级 S^{(NM)} = cos_sim(x^l, y^l) 移除整个 Transformer block（Attention + MoE + Norms）。移除 block 后：(a) 减少了 Attention 的 O(S^2·d) 矩阵乘法和 Softmax 计算；(b) 移除了对应层的 KV-Cache（如 batch=128, seq_len=2048 时节约 5GB KV-Cache）；(c) 移除了 FFN 计算。与 Expert Drop 的"精准但效果有限"相比，Layer/Block Drop 是"粗粒度但高效"的互补策略。实验：Mixtral-8×7B 移除 5/32 blocks 仍保持 >90% 性能，speedup 优于同压缩率的 Layer Drop。

  **(3) 集成 Expert Trimming + Expert Slimming 解决分别优化的"孤岛"问题**：将 Expert Slimming（AWQ 4-bit quantization）与 Expert Trimming（Layer/Block Drop）按"S+T"顺序组合——先对所有 expert 量化，再基于量化后模型计算相似度执行 Layer/Block Drop。量化减少每个 expert 的内存（→24.4GB from 87.7GB），Layer/Block Drop 进一步减少 FLOPs 和通信（→42.9T from 54.4T），两者互补达成 6.05× speedup + 77.1% 内存节省（20GB）。量化"保性能"（98%+ of original accuracy），Layer/Block Drop "增效率"（speedup 和 memory reduction）。

  **(4) 发现 MoE Layers 比 Dense 更冗余**：同深度 Mixtral-8×7B (MoE) vs Mistral-7B (Dense)，相同 Layer/Block Drop 下 MoE 模型性能衰减显著更小（Drop 8 layers: MoE -7.0 vs Dense -24.3 on MMLU）。这一发现验证了 MoE 架构中存在更高程度的结构冗余，Layer/Block Drop 特别适合 MoE 压缩。

  **(5) Post-Finetuning 解决压缩后的性能 gap**：在 Alpaca-GPT4 数据集上对压缩模型 full-finetune 3 epochs，性能 gap 从显著缩小（DeepSeek-MoE-16B Block Drop: 从 -5.5% 恢复到 -0.6%）。

  **(6) Expert Slimming 消融：Shared Expert 不可压缩性发现**：DeepSeek-MoE-16B 使用残差 MoE（2 shared + 64 routed），发现 shared expert 比 routed expert 更不可压缩——pruning 不含 shared expert 相比 pruning shared expert 提升平均精度 3.6%（Wanda）到 1.5%（SparseGPT）。

  **论文方法全栈执行例子（以 Mixtral-8×7B, AWQ + Block Drop B5/32, 128 sequence × 2048 tokens batch decode 为例）**：
  - **算法层**: 
    1. Expert Slimming: AWQ 4-bit 量化所有 32 层 × 8 experts 的 FFN 权重。W_i_quant = AWQ(W_i, 4-bit, group_size=128)。量化后模型总内存 24.4GB。
    2. 用 128 个 C4 样本在量化模型上计算每个 block 的 S^{(NM)}_l = mean(cos_sim(x^l, x^l+Block_l(Norm(x^l))))。
    3. 按 S^{(NM)} 降序排序 blocks，移除 Top-5 highest-similarity blocks（深层更冗余→多 drop 深层 blocks）。移除后 FLOPs 从 54.4T 降至 46.0T。
    4. Router: 保留 layers 的 router 不变（移除层无需 routing）。移除 block: block 的 attention + MoE + 2 Norms 全部移除，对应 KV-Cache 移除。
    5. Inference forward: tokens → embedding → 剩余 27 blocks（每 block: Attention + Norm + MoE FFN with 8 quantized experts, Top-2 routing）→ LM Head → next token。Speedup=5.94×, Memory=21.9GB。
  - **系统框架层**: HuggingFace Transformers 加载量化模型（AutoAWQ）+ PyTorch forward pass 测试 speedup/FLOPs（input seq_len=2048）+ EleutherAI LM Evaluation Harness 评估 zero-shot benchmarks。论文未修改 serving framework。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 论文未明确说明。AWQ 量化后使用 INT4 GEMM kernels；标准 PyTorch FP16 attention kernels。
  - **硬件架构层**: NVIDIA GPU（RTX 3090 24GB 为部署目标的提及）。量化后 Mixtral-8×7B 从 87.7GB → 20.0~24.4GB，满足 24GB consumer GPU 部署条件。Speedup 基于 forward pass on seq_len=2048 测量。

## Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

- baseline方法是什么？
  Baseline 是 **Standard TopK MoE routing with sparse backward pass**。在标准 TopK MoE 中，Router 通过线性变换 Wx 产生 logits，经 Softmax 得到 expert weights π，TopK 选择 K 个 expert 处理 token，输出 y = Σ_{i∈A} π_i E_i(x)。在反向传播时，由于 TopK 选择是离散不可微操作，使用 straight-through estimator 绕过：∂y/∂π_i = E_i(x) if i∈TopK else 0。未被选中的 experts (N-K 个) 对 Router 的梯度贡献为 0，Router 无法从这些 experts 获得反馈信号。这导致：(1) Router 学习效率低——只能根据已激活 experts 的输出调整路由策略，无法评估未激活 experts 是否更合适；(2) 训练不稳定——Router 更新不完整，在 load imbalance 时容易产生 loss spike；(3) 最大 stable learning rate 受限——因为只有部分 Router 参数得到更新，较大 LR 会导致少数被更新的行产生过大变化。Baseline 全栈执行例子：训练时每个 token x → Router 计算 π = Softmax(Wx) → TopK 选择 expert {i_1,...,i_K} → 仅 K 个 expert 计算前向 E_i(x) → 前向输出 y = Σ_{i∈A} π_i E_i(x) → 反向传播时 ∂y/∂π_i = E_i(x) for i∈A, 0 for i∉A → dL/dW = (dL/dy) · Σ_{i∈A} E_i(x) · (∂π_i/∂W) → Router 仅根据 K 个已激活 expert 的输出更新，N-K 个未激活 expert 对应的 W 行不参与梯度更新。训练框架使用 gpt-neox + Megablocks + liger kernel (Triton)，dropless MoE，global-batch auxiliary loss (0.01)，AdamW optimizer。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DefaultMoE**：为每个 expert 维护一个 EMA default vector Ê_i = EMA(E_i(x))，在反向传播时将未激活 expert 输出替换为 default vector，使 Router 收到 dense gradient。直接解决 Baseline 的三大缺陷：

  **(1) Router 梯度不完整** → 解决：Default vector 填充缺失梯度。Standard TopK 的 Router 梯度误差为 ε_TopK = (∂L/∂y) Σ_{i∉A} E_i(x) · (∂π_i/∂W)，即丢失了 N-K 个未激活 experts 的梯度项。DefaultMoE 使用 Ê_i ≈ E[E_i(x)] 替代，梯度误差变为 ε_default = (∂L/∂y) Σ_{i∉A} (E_i(x) - E[E_i(x)]) · (∂π_i/∂W)，在期望上为 0（因为 E[E_i(x) - E[E_i(x)]] = 0）。通过实验验证：DefaultMoE 的 Router gradient 在所有 K 值下都与 dense gradient（K=N 全激活）更相似，尤其在 K=1 的前几层（Router entropy 高时）差异最为显著。

  **(2) 训练稳定性差** → 解决：更完整的 Router 更新允许更大的 stable learning rate。Baseline 在 LR=9×10⁻⁴ 时会出现 loss spike（某次迭代 load 极不均衡导致单步大幅更新），DefaultMoE 在相同 LR 下稳定训练，且 DefaultMoE 在所有 LR 下均优于 baseline。这是因为 DefaultMoE 同时更新所有 N 个 expert 对应的 Router 行，避免了个别行更新过大。

  **(3) 收敛速度慢** → 解决：DefaultMoE 在 160B token 训练中比 TopK baseline 减少约 9~15% 的 token 需求（达到相同 target PPL），且最终 PPL 更低。在 12 个下游 benchmark 上，8c1 平均提升 2.1%，8c2 提升 5.0%。该方法对所有 MoE 配置（8c1~32c4）、所有模型规模（557M~7.33B）、所有 sparsity 比例均有效。

  DefaultMoE 全栈执行例子（8c1, 1.96B model, training step）：
  - **算法层**: token x 经 Router 得到 π = Softmax(Wx) → TopK=1 选择 expert 3 → 前向时 E_3(x) 真实计算，E_0,E_1,E_2,E_4,E_5,E_6,E_7 用 EMA buffer Ê_i 替代 → y = Σ π_i [if i=3: E_3(x) else Ê_i] → 反向传 Router: ∂y/∂π_i = E_3(x) for i=3, Ê_i for i≠3 → dL/dW[i,:] += (dL/dy) · Ê_i · x^T for all i ≠ 3 → 所有 8 个 expert 对应的 Router 行均获得梯度更新。EMA 同步更新: Ê_3 = β·Ê_3 + (1-β)·E_3(x)，其他 expert 的 EMA 保持不变。
  - **系统框架层**: gpt-neox + Megablocks 实现 dropless MoE 训练（data-parallel only），64 GPUs AWS，global aux loss 跨节点 reduce。
  - **编译框架层**: 论文未明确说明（gpt-neox eager execution + Triton JIT compiled kernels via liger kernel）。
  - **Kernel调度层**: Megablocks 的 sparse matmul kernel 处理 expert FFN 的批量计算，liger kernel (Triton) 提供优化的 cross-entropy 等 loss kernel。EMA 更新和 default vector 替换均在 PyTorch eager 层面完成，额外开销：O(1) memory per expert × hidden_dim（如 1024 维 × 8 experts × 16 layers ≈ 0.03% 参数增量）。
  - **硬件架构层**: 64 GPUs AWS 集群，无特殊硬件要求。Throughput overhead: 1.96B 模型 1 GPU 上 26,393 (TopK) vs 25,913 (DefaultMoE) tokens/sec = -1.85%；7.33B 模型 per-node 1,393 vs 1,391 tokens/sec = -0.18%（统计噪声级别）。随模型增大，matmul 占比增加，EMA 开销占比趋近于 0。

## Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 是 **Sparse MoE (SMoE) 训练 + Sparse Inference**：传统 MoE 训练中，每个 token 仅激活 top-K 个 expert（通常 K=2），反向传播时梯度仅通过这 K 个激活的 expert 和对应的 Router 分数传递。Baseline 使用 dMoE (MegaBlocks) 实现，Switch Transformer style 训练（Fedus et al. 2022），使用 switch loss 进行负载均衡。这种稀疏训练范式导致 MoE 模型的参数效率远低于 Dense 模型——需要 2-4× 更多参数才能匹配 Dense 模型性能。

  **Baseline 全栈执行例子（SMoE-5B, D_emb=3072, N_ffd=16, top-2 sparse training, 训练 step）**：
  - **算法层**: token X [3072] → Router S = Softmax(h(X)) → TopK=2 选择 expert 3, 7 → 仅 E_3(X), E_7(X) 计算前向 → O = S_3·E_3(X) + S_7·E_7(X) → 反向传播仅更新 Router 对应 expert 3,7 的行和 E_3, E_7 的参数 → expert 0,1,2,4,5,6,8-15 不获得梯度。Switch Loss 作为辅助损失 per layer: L_switch = α · N · Σ_i f_i · P_i，其中 f_i 是分给 expert i 的 token 比例，P_i 是 Router 给 expert i 的平均概率。
  - **系统框架层**: dMoE (Gale et al. 2023, MegaBlocks) 实现 MoE 训练（expert parallelism + data parallelism），FSDP (Zhao et al. 2023; Rajbhandari et al. 2020) 分片优化器状态和参数。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: MegaBlocks 的 block-sparse GEMM kernel 处理 grouped expert FFN 的批量前向/反向计算。稀疏训练中仅有 top-2 expert 被调度计算。
  - **硬件架构层**: NVIDIA H100 80GB × 32。训练 3B/6B-scale 需要 64h/124h。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 DS-MoE（Dense Training, Sparse Inference）：

  **(1) Sparse Training 的参数低效** → 解决：Dense Training（所有 expert 在训练时全部激活）：
  
  Baseline SMoE 的 Router 梯度在反向传播时被 binary mask M ∈ {0,1}^N 截断：∇S = [e_1(X),...,e_N(X)]^T ∇O ⊙ M，仅激活的 top-K expert 对 Router 梯度有贡献。DS-MoE 保留完整梯度：∇S = [e_1(X),...,e_N(X)]^T ∇O，所有 N 个 expert 的梯度均参与 Router 更新。每个 expert e_i 的梯度也变为 ∇e_i(X) = S_i · ∇O，获得按其 Router score S_i 加权的完整梯度。这意味着：(a) Router 能学习所有 expert 的能力分布（类似 Dense model 的 FFN 全参数学习）；(b) 所有 expert 参数均被持续优化（而非只有被激活的）；(c) 参数效率达到 Dense 模型水平。

  **(2) 训练后如何实现 Sparse Inference** → 解决：Mutual Information (MI) Loss：
  
  Dense Training 不能直接退化为 Sparse Inference——如果没有额外的正则化，Router 会均匀分配所有 expert（无 sparsity）。DS-MoE 引入 MI Loss：L_MI = -H(e) + (1/|X|)Σ H(e|X)，其中：
  - 最大化 H(e) = -Σ p(e_i) log p(e_i)：保证 expert 负载均衡（所有 expert 被充分训练）。
  - 最小化 H(e|X) = -Σ p(e|x) log p(e|x)：鼓励 Router 为每个 token 做出确定性选择（sparse concentration）。
  
  两者形成"对抗"平衡：负载均衡 vs 专家集中。训练后，Router 自然产生 sparsity——仅 top-K 或超阈值 expert 被激活（30-40% parameters during inference）。α 参数（MI loss weight）控制 sparsity 程度。

  **(3) Baseline SMoE 的 Attention 层仍是 dense** → 解决：Mixture of Attention Head (MoA)：
  
  多数 SMoE 模型仅在 FFN 层使用 expert（Attention 仍为 dense）。DS-MoE 将 Attention 也改为 MoA (Zhang et al. 2022)：每个 expert 计算 N_head 个 query vectors，共享 KV cache。MoA 在推理时也可 sparse，进一步减少计算。

  **论文方法全栈执行例子（DS-MoE-3B, D_emb=3072, N_ffd=32, D_ffd=384, N_att=8, 训练 step）**：
  - **算法层**: 
    1. Dense Training Forward: token X → Router S = Softmax(h(X)) → 计算所有 32 个 expert 的输出 E_i(X) = GeLU(X@W_up_i + b_up_i)@W_down_i + b_down_i → O = Σ_{i=1..32} S_i · E_i(X)。计算量 ≈ 32 × 2 × 3072 × 384 = 75.5M FLOPs/token/layer（≈ Dense-3B 的相当水平，因 Dense-3B 的 D_ffd=12288 而非 32×384=12288，实际 FLOPs 相同）。
    2. Dense Training Backward: ∇O → ∇S = [E_1(X),...,E_32(X)]^T ∇O（所有 32 个 expert 对 Router 梯度有贡献）→ ∇e_i(X) = S_i · ∇O → Router 和所有 expert 参数同步更新。Router 学习到所有 32 个 expert 的全景分布。
    3. MI Loss per batch: 统计 P(e_i) = mean over batch of S_i（expert 边际分布）→ H(e) = -Σ P(e_i) log P(e_i)；统计 P(e_i|x) = S_i（per-token expert 分布）→ H(e|x) = -Σ S_i log S_i → L_MI = -H(e) + mean(H(e|x))。
    4. Sparse Inference: Router 计算 S → TopK=6 → ParallelLinear dispatch X to selected experts → 仅 6 experts 执行 → O = Σ_{i∈top-6} S_i·E_i(X)。Active parameters: 6/32 = 18.75% of expert params, ~34% of total hidden params (accounting for attention and norms)。
  - **系统框架层**: PyTorch + FSDP (fully sharded data parallelism) + activation checkpointing。使用 Flash Attention (通过 PyTorch SDPA 或手动指定) 优化注意力的 HBM I/O。dMoE (MegaBlocks) 仅用于 SMoE baseline 实现。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: MLP 层使用 SimpleMoE 的 ParallelLinear（原生 PyTorch eager, group-based expert parallelism kernel）；Attention 层使用 torch.nn（dense inference, 因 sparsity >60% 时 sparse overhead 超过 dense）。训练使用 PyTorch 原生 eager execution + FSDP 分片通信。
  - **硬件架构层**: H100 80GB × 8 (1B) / × 32 (3B, 6B)。推理 evaluation on A100-80GB 和 H100-80GB。DS-MoE-3B 训练 64h on 32×H100；DS-MoE-6B 训练 124h on 32×H100。

  **核心设计洞察**：传统观点认为 MoE 的稀疏性来自训练阶段的稀疏激活（sparse gradient）。DS-MoE 发现稀疏性可以作为训练后的"退火"行为自然浮现——通过 dense training 保留参数效率，通过 MI loss 在训练过程中隐式塑造 sparsity pattern，最终在推理时仅激活 top-K expert。这从根本上不同于"稀疏训练后推理"的经典范式。

## DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

- baseline方法是什么？
  现有 MoE 专家剪枝方法分为两类：(1) **Feature Statistics 类**（M-SMoE、Expert Trimming）：统计每个专家的 activation frequency 或 feature similarity，在每层独立地删除频率最低或合并相似的专家。M-SMoE 将低频专家合并到高频专家，但在层内做 activation count normalization 后跨层信息被抹除，隐含假设各层冗余程度相同。(2) **Greedy Search 类**（NAEE、S-SMoE）：NAEE 在每个 MoE 层内枚举所有 k-expert 组合，通过最小化 reconstruction loss 选出最优子集；S-SMoE 基于相似度做 pruning+merging。两类方法的核心缺陷是**所有层使用统一的剪枝比例**，忽略了不同 MoE 层之间专家冗余程度的显著差异（如 CKA 可视化所示，浅层 1-15 的 intra-layer similarity 模式与深层 16-32 明显不同）。对于 64 experts/layer 的模型，仅 12.5% sparsity 就需要评估 C(64,8) ≈ 4×10^8 种组合，使全局 exhaustive search 在计算上不可行。

  **Baseline 全栈执行例子（以 NAEE on Mixtral 8×7B, 50% sparsity 即每层 8→4 experts, 推理一个 token 为例）**：
  - **算法pipeline层**: 在每层 MoE 内枚举 C(8,4)=70 种专家组合，计算每种组合的 reconstruction loss，选出 loss 最小的子集作为保留专家。32 层 × 70 组合 = 2240 次评估。每层独立选择，无跨层信息传递。
  - **系统框架层**: HuggingFace Transformers 加载模型 → 逐层执行 expert combination search → 输出 pruned checkpoint。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 PyTorch FFN forward，无 custom kernel。
  - **硬件架构层**: 4× NVIDIA A800 GPU。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DiEP (Differentiable Expert Pruning)**，将离散专家选择重新表述为连续优化问题。核心设计：(1) **Intra-layer + Inter-layer 双层次重要性学习**：定义 intra-layer 重要性 α_i^(l)（每层内专家相对重要性）和 inter-layer 重要性 β^(l)（层对全局模型的贡献），通过 softmax 归一化实现连续松弛，将全局离散搜索空间（指数级）转换为连续可微空间，直接解决 NAEE 等 exhaustive search 的计算不可行问题。(2) **交替梯度优化**：以 α:β = 3:1 的比例交替更新，解耦两个参数组的梯度路径避免优化冲突（与 DiffPruning 的单调梯度下降形成对比），目标函数组合 CE loss + Reconstruction Regularization (∥F' − F∥_F)，无需 validation set。(3) **全局统一排序剪枝**：s_i^(l) = α_i^(l) · β^(l)，全局排序所有 L×N 个专家后按 ratio r 统一删除 bottom-K。这种 cross-layer global ranking 解决了 baseline 中"每层统一比例"导致的浅层/深层冗余差异被忽视问题——浅层自动保留更多专家（因为 β 和 α 学到的浅层重要性更高，符合 CKA 可视化结果）。(4) **Adaptive Expert Skipping 在线推理加速**：γ = γ1（routing weight ratio 中位数）× γ2（CKA similarity ratio），当 token 的次要专家 routing weight 低于 γ 倍的主要专家时跳过，消除冗余专家计算。额外获得 1.2−1.3× 推理加速。

  **DiEP 方法全栈执行例子（以 Mixtral 8×7B, 50% sparsity, 推理一个 token 为例）**：
  - **算法pipeline层**: α_i^(l) (32×8=256 参数量) + β^(l) (32 参数) → 仅约 0.01% 额外参数。Calibration: 128 C4 samples, 10 epochs, lr=5e-3 cosine schedule, batch=16。前向计算 y'^(l+1) = β^(l) · Σ_i softmax(α_i^(l)) · FFN_i(x) → 交替优化 3:1 更新 α 和 β → 收敛后全局排序 s_i^(l) → 删除 128 个最不重要专家 → 保留约 92% 原模型性能（MMLU avg 57.9 vs full 67.9, 50% sparsity）。
  - **系统推理层**: 加载 pruned checkpoint → 对每个 MoE layer 的 Top-2 激活专家计算 γ 阈值 → 若 w_e1 < γ·w_e0 跳过 e1 的 FFN 计算 → 减少 GPU 计算量和 memory access。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 标准 PyTorch FFN forward，adaptive skipping 通过条件判断跳过冗余专家计算，无 custom kernel。
  - **硬件架构层**: 4× NVIDIA A800 GPU。50% sparsity + skipping 下 GPU memory 降至 0.52× 原模型，token generation speedup 1.28×。

## DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

- baseline方法是什么？
  Baseline 有两个层面：(1) **Standard MoE Inference without Computation Dropping**：预训练 MoE 模型的 naive 推理，每 token 在每层激活 Top-K 个 experts，所有 activated experts 的 FFN 计算全部执行，不做任何 token-expert 计算丢弃。Expert 按照预训练时的粒度（如 Mixtral-8×7B 的 8 experts, Top-2）运行。在分布式场景下使用标准 EP + ETP（Expert-Tensor Parallelism）部署，通信模式为 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll"，存在多轮 kernel launch 和同步开销。(2) **Prior Sparsity-based Acceleration Methods**：EES (Efficient Expert Skipping) 根据第二高 gating score 与第一高 score 的比值动态跳过 expert 计算，但 accuracy degradation 显著（GSM8K -2.4%）；EEP (Efficient Expert Pruning) 静态剪枝不常用 experts 实现模型压缩，但 accuracy loss 大（r=6: -8.0%, r=4: -25.9%）；Wanda 等权重剪枝方法在 2:4 sparsity 下 GSM8K accuracy 下降 50.7%。这些方法的共同缺陷：(a) 以动态 tensor-level sparsity 换静态压缩，破坏 MoE 的动态路由优势；(b) 高 drop/prune rate 下 accuracy 剧烈下降；(c) 细粒度 neuron-level sparsity 难以在现有 GPU hardware/kernel 设计上翻译为实际 speedup；(d) 未利用 neuron-level 激活稀疏性（SwiGLU FFN 中大量 neuron 的 gating score × activation 乘积接近零但不为零）；(e) 未考虑 EP 分布式推理中的 load imbalance 问题。

  **Baseline 全栈执行例子（以 Mixtral-8×7B, 8×H20, TP=8, 推理 batch of tokens 为例）**：
  - **算法层**: Standard MoE with 8 experts, Top-2 gating, SwiGLU FFN experts (d_ffn=14336), 32 decoder layers with MoE layers alternating with attention. 每 token 激活 2/8 experts，所有激活的 expert FFN 计算完整执行。
  - **系统框架层**: SGLang framework，使用 TP=8 做 tensor parallelism for non-expert layers，EP 的通信使用标准 ETP 模式（AlltoAll + AllGather）。无 token-expert computation dropping，无 load-aware thresholding。
  - **编译框架层**: 论文未明确说明（SGLang Python/Triton-based execution）。
  - **Kernel调度层**: 标准 Triton grouped-GEMM kernel for expert computation。所有 activated experts 使用完整权重矩阵 W₁, W₂, W₃ 计算。Gating 函数使用标准 top-k + softmax。
  - **硬件架构层**: 8× NVIDIA H20 GPU，单节点 NVLink/NVSwitch 互联。每个 token 的 MoE 计算量 = 2×(3×d_model×d_ffn) FLOPs = 2×3×4096×14336 FLOPs。每个 EP device 负载不均，总推理时间由最繁忙 device 决定。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **DualSparse-MoE**，核心创新是协调利用 MoE 架构中天然存在的 **双重稀疏性**（tensor-level + neuron-level），并通过 expert partition 在 post-training 阶段增强 tensor-level sparsity 而无需重训练。

  **(1) Expert Partition（解决baseline中专家粒度固定问题）**：Baseline 中 MoE 的 expert granularity 在 pre-training 时确定，部署时无法改变。论文提出两种 post-training expert 划分方法，保持数学一致性：(a) Complete Transformation 将 E experts 划分为 E×P 个 finer-grained experts，提高 fine-tuning accuracy（Mixtral 8→32 experts: fine-tuning loss 降低，downstream accuracy +0.59%）；(b) Partial Transformation 保持 gating network 不变，仅重映射 expert indices，支持 S-ETP 优化。

  **(2) 2T-Drop with Neuron Reconstruction（解决baseline中单一阈值丢弃的精度损失）**：Baseline 的简单 token-expert dropping 面临显著 accuracy degradation。论文的解决方案：(a) Static Neuron Importance Profiling：在 calibration samples 上对每个 expert 内 neuron 做 importance profiling（四种方法：accumulated gate、abs gate、gate-up、abs gate-up），按重要性排序后重构为 major（高重要性）和 minor（低重要性）sub-expert；(b) Dual-Threshold Dropping：对 major sub-expert 使用较低阈值 T²_major（保守保留），对 minor sub-expert 使用较高阈值 T²_minor（激进丢弃），gating score 在 dual-threshold 之间的 experts 仅计算 major half neurons。结果：~25% drop rate 下仅 loss 0.08%-0.28% average accuracy（Mixtral: 71.12→71.04, OLMoE: 65.91→65.63, DeepSeek: 67.83→67.65）。

  **(3) Load-Aware Thresholding（解决baseline中EP负载不均问题）**：Baseline 的 EP 推理中，所有 device 使用相同 drop threshold，但不同 device 负载差异大，均匀丢弃在 overloaded device 上加速不够、underloaded device 上精度损失不必要。论文方案：每个 device 根据 actual_load / ideal_load 比值动态调整 threshold，overloaded device 用高 threshold（激进丢弃）、underloaded device 用低 threshold（保守保留），以最小精度损失实现负载均衡。结果：load-aware 2T-Drop → 1.41× MoE module speedup, 1.13× end-to-end speedup，仅 0.5% average accuracy loss。

  **(4) S-ETP（解决baseline中ETP通信复杂问题）**：Baseline 的 ETP 使用 "AlltoAll+AllGather" 或 "ReduceScatter+AlltoAll" 多轮通信，引入额外 kernel launch 和同步开销。S-ETP 通过 partial transformation 将 TP 职责转移到算法层面，仅需单次 AlltoAll 通信。结果：real H20 带宽提升 3.0%-29.9%，NVL72 模拟提升 10.2%-80.4%。

  **DualSparse-MoE 方法全栈执行例子（以 Mixtral-8×7B, P=4 (32 experts), 8×H20 TP=8, 推理一个 batch of tokens, ~25% drop rate 为例）**：
  - **算法pipeline层**: Preprocessing: (a) partial transformation: 8→32 experts, P=4, Top-2→Top-8；(b) neuron importance profiling on MMLU calibration, 按 accumulated absolute gate value 排序，每个原 expert 重构为 major (top 50%) + minor (bottom 50%) sub-expert。Inference: 每 token 计算 gating scores → normalize → 对每个 activated expert 判断 normalized score 与 dual thresholds (T²_major=0.07, T²_minor=0.09) 的关系 → 决定 skip/major-only/full 计算 → ~24% token-expert pairs dropped。
  - **系统框架层**: SGLang framework + DualSparse-MoE modifications。Preprocessing 完成 expert partition + neuron reconstruction。Inference 时 gating function 融合 dual-threshold decision logic。通信使用标准 AlltoAll（可选择 S-ETP 简化模式）。如启用 load-aware thresholding：gather 各 device 负载 ratio → 动态调整各 device 的 drop threshold。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 优化的 Triton grouped-GEMM kernel 支持变粒度计算模式（skip/major-only/full）。Token-expert dispatch 根据 dual-threshold decision 将 token 分组到不同的 GEMM kernel 调用。2T-Drop 的细粒度计算与 1T-Drop 粗粒度计算实现相近的实际 speedup（因为优化避免了额外 kernel launch）。
  - **硬件架构层**: 8× NVIDIA H20 GPU, NVLink intra-node。~24% drop rate → 1.17-1.23× MoE module speedup, 1.07-1.12× end-to-end speedup。Tensor-level 丢弃粒度天然适配 GPU grouped-GEMM，区别于 neuron-level sparsity 需要专用 hardware/kernel 才能在低 drop rate 下实现 speedup。

## Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

- baseline方法是什么？
  **Vanilla MoE dLLM 并行解码**：在 diffusion LLM 的 block-based parallel decoding 中，每个 token 通过独立的 gating/routing function 选择 Top-K experts（标准 softmax gating + TopK selection）。MoE 层输出为 MoE(x) = Σ_{i∈S} (G(x)_i / Σ_{j∈S} G(x)_j) · E_i(x)，其中 S = TopK(G(x), K)。N 个并行 token 导致 unique expert load |∪_{n=1}^N S_n| 近乎线性增长（"expert explosion"），HBM→SRAM weight fetching cost 主导延迟。

  全栈执行例子（LLaDA2.0-Mini 16B, block size 32）：
  - **算法pipeline层**: 每 token 独立 softmax gating → Top-8 selection → weighted sum of 8 expert FFNs。32 tokens × 8 experts/token → unique activated experts ≈84 per layer。
  - **系统框架层**: dInfer inference framework + Fast-dLLM KV cache（0.9 confidence threshold）。HBM 加载 84 个 expert 权重（~0.98 GB/layer MoE component），memory-bound 运行。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: PyTorch native kernels 执行 gating、softmax、topk、scatter/gather 等碎片化算子链（12+ kernels）。
  - **硬件架构层**: NVIDIA B200 GPU，HBM→SRAM bandwidth 瓶颈。Expert weight fetching cost b >> per-token compute cost a，导致 memory-bound。

  现有 expert skipping baseline（NAEE、MC-MoE）的缺陷：token-centric 优化仅减少 per-token compute（a 项），不减少 unique expert load（b 项）。在 dLLM 场景下 accuracy 严重退化（LLaDA2.0-Mini 上仅保留 ~46% relative accuracy），因为静态阈值无法适应并行 token 间多样的 gating 分布。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Dynamic Expert Sharing（DES）** 将优化从 per-token pruning 转为 sequence-level coreset selection。核心洞察：dLLM 并行解码的 token 共享语义上下文，expert 需求存在显著重叠。通过识别 compact、high-utility expert coreset C，限制所有 token 仅在 C 内路由，最大化 expert 复用。

  全栈执行例子（DES-Vote, β=0.15, LLaDA2.0-Mini 16B, block size 32）：
  - **算法pipeline层**: 
    1. **Saliency-Aware Voting**：mask 每 token 的非 Top-K router scores → 跨序列聚合投票 V_i = Σ_{n=1}^N Masked(I_{n,i}) → Top-M_core experts 组成 coreset C（M_core = β×M = 0.15×M）。
    2. **Constrained Local Routing**：每 token 从 C 中选择 Top-K experts，重新归一化 gate weights。
    3. 结果：unique experts 从 84→38（-55%），accuracy 保留 99.5%。
    对比 baseline：Token-centric expert skipping（NAEE/MC-MoE）仅跳过低分 experts 减少 compute（a 项），但每 token 独立选择意味 |∪S_n| 几乎不变；DES 通过跨 token 共享最大化了 |∪S_n| 的降低，直接减少 weight-fetching cost（b 项）。
  - **系统框架层**: dInfer + Fast-dLLM，DES 在每 MoE 层插入 coreset selection step。Memory footprint 从 0.98 GB/layer 降至 0.45 GB/layer。DES-Vote 的 β 参数（连续值）提供灵活的 budget 控制，可绕过 DES-Seq 每 token 至少 1 expert 的下限。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 自定义 fused CUDA kernel 将 12 个碎片化算子（softmax + topk + masked reduction + topk）融合为 2 个 kernel：Kernel 1 利用 register-level 计算 + atomic instructions 完成 per-token softmax/ TopK/weighted vote accumulation；Kernel 2 执行 threshold-governed final ranking。实现 6× speedup over PyTorch baseline。
  - **硬件架构层**: NVIDIA B200 GPU。MoE layer latency 降低 38.0%（LLaDA2.0-Mini），end-to-end GPU kernel time 降低 8.2-14.3%。DES-Vote 在不同 block sizes（8/16/32/64）下保持恒定低 expert count，彻底解耦了 memory overhead 与并行度的绑定关系。

## Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

- baseline方法是什么？
  Baseline 包含三个层面：(1) **Full SMoE Model (Zero-Shot)**：Mixtral 8×7B-Instruct 使用 8 experts (Top-2)，每 token 激活 2/8 experts 进行推理，47B 总参数、13B 激活参数。Router 网络（single-layer perceptron）在高维 hidden space 中划分 expert 分配决策，但存在 expert activation imbalance 问题。(2) **Frequency-based Expert Pruning [37]**：对训练集统计每个 expert 的激活频次，剪枝掉激活频次最低的 experts。缺陷：不考虑 router weighting 的实际贡献，仅依赖频次统计，在低 expert budget 下容易 collapse。(3) **Soft Activation Pruning [37]**：累积 router weighting（soft activation value），剪枝累积值最低的 experts。缺陷：路由权重在后续层移位后剪枝效果不可靠。(4) **NAEE [34]**：逐层穷举所有剪枝方案，选择与 full model 输出差异最小的方案。缺陷：(a) 穷举计算量随 expert 数组合爆炸（Qwen 的 60+ experts 无法穷举，只能随机采样 5000/2000 种方案，实际性能接近 random）；(b) 仅基于 output discrepancy 做选择，不考虑下游任务实际表现；(c) 剪枝后不做 knowledge recovery，高 sparsity 下性能下降显著。

  **Baseline 全栈执行例子（以 Mixtral 8×7B-Instruct, 2×A100, 单个 token 推理为例）**：
  - **算法层**: SMoE with 8 experts (SwiGLU FFN), Top-2 gating。每个 expert θ_i = {W_{1i}, W_{2i}, W_{3i}}。Router 计算 G = softmax(ZW_G) ∈ R^{n×E}，TopK(G_j) 选择 2 个 expert 激活。H_j = Σ_{i∈TopK} G_{ji} · FFN_i(Z_j)。
  - **系统框架层**: 标准 HuggingFace Transformers + PyTorch。加载 8 experts 全部参数 (45B out of 47B 为 expert 参数)，batch inference 时全部 experts 驻留显存。
  - **编译框架层**: 论文未明确说明（标准 PyTorch + CUDA graph 路径）。
  - **Kernel调度层**: 标准 grouped-GEMM for expert computation。所有 8 个 experts 的 W₁, W₂, W₃ 完整加载并计算，每个 activated expert 的 FFN = SwiGLU(Z_sub, W₁i, W₃i) · W₂i。Router top-k softmax 为标准核函数。
  - **硬件架构层**: 2× NVIDIA A100 GPU。8 experts → 显存占用 ~88.6 GB (FP16 约 94 GB 参数 + activations)。全部 experts 进入 HBM，prefill/decode 阶段完整执行 2 个 expert 的 FFN 计算。

  **Baseline 核心痛点**：
  1. **Expert 冗余性未被充分利用**：单个 expert 即可维持合理的推理性能（仅微小下降），但 8 个 experts 全部保留在显存中，造成 ~72% 参数量能被剪枝而性能不降的浪费。现有的剪枝方法（Frequency/Soft Activation/NAEE）在 expert 粒度选择上精度不足，高 sparsity 下 collapse 严重。
  2. **Router 网络在高维空间分配不精确**：Router 作为 single-layer perceptron 难以精确划分高维 hidden space，导致 expert activation imbalance 和 sub-optimal routing。但剪枝改变了 routing 行为——剩余 experts 的 routing weights re-normalize，提供了 routing 优化的机会。
  3. **梯度式 fine-tuning 资源需求过高**：传统剪枝范式中剪枝后需用 SGD fine-tuning 恢复性能，需要大量 GPU 显存和计算时间。权重复用只能通过 "select subset by importance criteria" 或 "distillation" 两种范式，缺乏高效的 weight merging 范式。
  4. **剪枝后 expert knowledge 丢失**：直接丢弃 pruned experts 导致知识丢失，现有方法只做 selection 不做 recovery。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出 **EEP (Efficient Expert Pruning)**，一种无梯度进化策略，通过设计 Router Mapping (WRM) 和 Expert Merging (WEM) 两个参数空间矩阵，在两阶段进化搜索中完成 expert 剪枝和知识合并。

  **(1) Expert Pruning Phase — 进化搜索发现最优剪枝模式（解决 pain point #1 "粗糙剪枝"）**：
  Baseline 的 Frequency/Soft Activation 使用固定的单一统计量指标选择 expert，无法适应不同下游任务的 expert 贡献分布。EEP 通过进化策略在巨大搜索空间（每层从 C(E, E') 种组合中选择）中搜索最优剪枝模式。WRM 和 WEM 初始化为 one-hot rows，且约束 WRM = WEM，只选择保留 expert 并保持离散性。每次迭代中个体按累积 F 分数排名，Top M_CP 作为 candidate parents，通过 Crossover（沿 expert 维度组合）和 Mutation（随机替换 pruned experts）产生后代。搜索 40 轮找到最优子集。
  结果：EEP (Prune Only) 在 4/8 expert 保留时大幅超越 baselines（Avg. 70.3 vs NAEE 60.5 vs Frequency 45.8），在 2/8 expert 保留时仍维持有效性能（Avg. 59.7）。

  **(2) Expert Merging Phase — 权重合并恢复知识（解决 pain point #3 "缺乏高效 fine-tuning" 和 #4 "知识丢失"）**：
  Baseline 范式中剪枝后 knowledge recovery 要么不需要（selection-only 方法精度低），要么需要梯度-based fine-tuning（资源要求高）。EEP 引入一种第三范式：**Weight Merging**。WRM 和 WEM 解耦后元素从离散 0/1 过渡到连续值，通过 block-wise weighted sum 将 pruned experts 的知识合并到 retained experts 中：
  - θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}
  - 其中 ω_ji 为连续值，可包含负值（负系数说明某些 expert 的知识对下游任务无益）
  - 进化搜索 160 轮完成 continuous optimization
  结果：Merging 后在几乎所有数据集上实现 5%-7% 的额外提升（如 WIC 57.8→65.0, CB 69.6→75.0, SQuAD 75.2→80.6）。且整个过程无梯度计算，可在仅支持推理的设备上运行。

  **(3) Router 重新聚焦（解决 pain point #2 "Router 不精确"）**：
  Baseline 中 Router 需要在 8 个 experts 间分配高维 hidden space，剪枝后仅剩 4 或 2 个 experts，router re-normalization 使决策空间大幅缩小。实验显示剪枝后 expert 的 accumulated activation times、accumulated routing weights 和 activation correlation 发生明显变化，路由更加聚焦，部分数据集上即使不做 parameter update 也能超越 full model（如 SQuAD: 4 experts 下从 53.4%→75.2%）。

  **(4) 双重使用场景（解决 pain point #3 in broader deployment）**：
  - Use case 1 (减少 total experts): 8→4→2，节省 47%-71% GPU 显存
  - Use case 2 (减少 active experts): Top-2→Top-1，实现 prefill 1.63× 加速，decode 1.34× 加速
  - 组合使用: 4 total + 1 active → 47% 显存节省 + 1.41× 推理加速

  **EEP 方法全栈执行例子（以 Mixtral 8×7B, Top-2→Top-2, 8→4 experts per block, 搜索 SQuAD 训练子集为例）**：
  - **算法pipeline层**: Input Z ∈ R^{n×d}。
    阶段 I (Pruning): WRM = WEM ∈ R^{4×8} (one-hot rows)。G' = WRM · softmax(ZW_G)（路由降维）。θ'_j = WEM row_j 选择的原始 expert 权重。进化搜索 40 iterations, population size=|P|, 每天代评估 F(W·Θ)（下游任务 accuracy 作为 fitness）。Crossover 沿 expert dimension 交换；Mutation 随机替换 one-hot 位置。
    阶段 II (Merging): WRM, WEM ∈ R^{4×8} (continuous)。θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}。进化搜索 160 iterations, Mutation=Gaussian noise。Expert weights 按深度分为 4 groups（或 32 groups per dataset），组内共享 merging coefficients 减少优化参数。
    最终 model: 4 experts per MoE block（参数量从 45B→~12.8B experts），权重为 merged form。
  - **系统框架层**: 无特定 serving framework（标准 PyTorch/HuggingFace 推理）。EEP search 过程在 inference-only 环境完成。搜索完成后将 merged model weights 导出为标准 HF format，在目标部署平台使用。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 论文未明确说明 KERNEL 级别实现。标准 grouped-GEMM kernel for expert computation（4 experts vs 8 experts 减少了 GEMM 调用次数和权重加载量）。Active expert 减少（2→1）后 decode 阶段 FP compute 和 HBM→SRAM weight fetching 减半。
  - **硬件架构层**: 2× NVIDIA A100 GPU。显存占用从 88.6GB→46.6GB（4 experts）→25.6GB（2 experts）。Prefill speedup 1.11×(4E)/1.18×(2E)/1.63×(1 active)/1.75×(4E+1 active)，Decode speedup 1.29-1.60×。Merging phase 在精度恢复阶段的贡献与显存/计算节省关系：merging 增加 ~floating-point operations in post-training 但不增加 inference cost。

## Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

- baseline方法是什么？
  Baseline 分为三类：
  (1) **Non-LLM 方法**：针对每个 DP 任务单独训练的专用模型或规则系统，如 Raha(ED)、Ditto/DeepBlocker(EM/Blocking)、Baran(DC)、RECA(CTA)、TURL(RE/EL)、IPM(DI)、MAVE(AVE) 等。这些方法依赖手工特征工程或任务特定架构，无法跨任务泛化。
  (2) **LLM-based 方法**：JellyFish(13B) 和 TableLLaMa(7B) 采用 MTL 范式，用一个大而密的 LLM 同时处理所有 DP 任务。需要在海量 task-specific 语料上预训练（数千 GPU hours），且单一大模型难以同时学好多个离散的 DP 任务子空间。
  (3) **MoE 模型（Mixtral 8×7B）**：内置 MoE layer，router 和 experts 联合训练。但缺乏独立 expert 微调的灵活性，且 load balancing 不均匀，在简单/封闭域任务（如 EM、DC）上性能差。

  **Baseline 全栈执行例子**（以 JellyFish 13B 处理 EM 任务为例）：
  - **算法pipeline层**: JellyFish 使用单一 dense 13B LLM，通过 MTL 在所有 DP 任务上联合训练。给定 query q=(t1,t2)，instruction prompt + few-shot demonstrations 输入 LLM，输出 match/mismatch 判断。由于所有任务共享同一参数空间，EM 任务的学习会受其他任务（如 RE、CTA）的梯度干扰，且 13B 模型需要 tensor parallelism 跨 GPU 部署。
  - **系统框架层**: 训练需数千 GPU hours 预训练（如 TableLLaMa 在百万 Wikipedia webtables 上预训练）。推理时 JellyFish 需 4-bit 量化才能在单 3090 上运行，量化带来性能损失。推理框架为标准 transformers/HuggingFace pipeline。
  - **编译框架层**: 论文未明确说明。
  - **Kernel调度层**: 论文未明确说明。使用标准 PyTorch GEMM kernel 执行 dense LLM forward pass。
  - **硬件架构层**: 训练需多 GPU（A100 级别或以上），推理时 JellyFish 13B 在单 RTX 3090 上需 4-bit 量化（GPTQ），Mixtral 56B 即使 4-bit 量化也无法部署在单 3090 上（OOM）。1×3090 上 JellyFish 4-bit 推理吞吐约 1.3× MELD，但 model process time 慢 10×。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  MELD 通过四个关键设计解决 Baseline 缺陷：

  **(1) 增强型 RAG + 自标注 → 解决 few-shot 数据稀缺（针对 Non-LLM 和 LLM baseline 的过拟合问题）**：
  Baseline 在 few-shot 场景下易过拟合（Theorem 2 证明 single expert error bound 受样本数 N 限制）。MELD 用 fine-tuned sentence-bert 进行跨域相似 entry 检索，用 contrastive learning 训练 RAG 模型后进行自标注（self-annotation），将少量标注数据扩增为大规模自标注训练集。RAG 模型还能通过 query 变换实现跨任务数据增广（如将 EM query 转换为 DI query）。

  **(2) Meta-path 数据增强 → 解决跨域泛化能力不足（针对 Multi-task learning 的任务子空间离散问题）**：
  Baseline MTL 方法难以处理离散且远离的 DP 任务子空间。MELD 用启发式贪心搜索找到每个 task T_i 的 meta-path（如 EM_BLK → DI → AVE → EM），沿 meta-path 调用多个 experts 进行数据增强，为原始 task 补充结构化信息和跨域特征。这对半结构化数据和低质量数据（如 semi-text-watch, amazon-google）尤为重要（ablation 显示 w/o meta-path 性能下降 10-30%）。

  **(3) 信息瓶颈引导的 Expert 精炼 → 解决 "多 expert 训练不收敛/不均衡" 问题**：
  Mixtral 的内置 MoE layer 中 expert 训练不均衡（load imbalance），且不能独立 fine-tune 单个 expert。MELD 采用 Min-Max 优化目标（基于 Information Bottleneck）：min I(X; θ) 确保训练数据多样性以防止过拟合，max I(Y; θ) 确保 expert 与标签的相关性以防止欠拟合。通过迭代控制 RAG 数据增强 + LoRA fine-tune 实现该目标，每个 expert 既能保持对自身 domain 的高性能，又对跨域 query 具有鲁棒性。实验显示 w/o MoE（单 expert per task）在所有数据集上性能下降。

  **(4) 独立 Router Network + 动态 LoRA 切换 → 解决 MoE 推理开销大和部署限制（针对 Mixtral 的 load imbalance 和部署问题）**：
  Mixtral 的 router 与 experts 联合训练，不可独立调整，且 56B total params 无法在单 3090 部署。MELD 的 router 是独立的轻量 transformer（共享 M_RAG 编码层），用对比学习训练，为每个 query 选择 top-k diverse 且 relevant 的 experts。推理时通过 Punica + vLLM 实现 dynamic LoRA switch，单 3090 可同时 serving 200 个 LoRA experts，无需 merge 操作，model process time 比 Mixtral 快 30×。4×3090 吞吐量为 Mixtral 的 5.6×（MELD 用 data parallelism，而 Mixtral 需 tensor parallelism 的跨 GPU 通信）。

  **MELD 方法全栈执行例子**（以 EM 任务、query q = (t1="Apple iPhone 13", t2="iPhone 13 by Apple")、k=3 experts 为例）：
  - **算法pipeline层**: Query q 经 serializer 序列化为 dict{instruction: "Entity Matching", tuples: [{t1}, {t2}], meta: {table: "Products", columns: ["name","brand"]}}。M_RAG 编码 q 为 emb_q，检索跨域相似 entries 作为 demonstrations D_EM（来自 Walmart-Amazon、Ant-Buy 等域）。若 meta-path E_EM = {e_BLK, e_DI, e_EM}，则先由 e_BLK 判断候选对、e_DI 填补缺失属性、e_EM 最终判断。精炼后的 e_EM^{aug} 不仅学会 EM 分类边界，还融合了其他 experts 的知识特征。
  - **系统框架层（Serving调度）**: Router N 计算 top-3 experts: e_EM^{aug}, e_DI^{aug}, e_CTA^{aug}（权重 [0.5, 0.3, 0.2]）。Punica + vLLM 在单 3090 上动态加载对应的 3 个 LoRA adapter 到 base Mistral-7B。每个 expert 独立推理后加权融合输出。vLLM 的 PagedAttention 管理 KV cache，continuous batching 处理并发 queries。Load balancing 机制将同类型 queries 路由到同一 GPU 以提高 cache 命中率。
  - **编译框架层**: 论文未明确说明。LoRA adapter 的前向计算通过标准 PyTorch linear + low-rank matrix multiplication 实现。
  - **Kernel调度层**: 论文未明确说明。LoRA forward: y = W_0·x + (B·A)·x，其中 B∈R^{d×r}, A∈R^{r×d}（r=rank，远小于 d）。Punica 优化了多个 LoRA 的 GEMM 融合，减少 kernel launch overhead。
  - **硬件架构层**: RTX 3090 (24GB VRAM)。Base Mistral-7B 约 14GB (FP16)，每个 LoRA adapter 约 10-50MB（取决于 rank），200 个 LoRA 约 2-10GB 额外显存。3 个 experts 的推理延迟 ≈ 1 个 expert 的延迟 + LoRA switch 开销（Punica 报告接近零开销）。相比 Mixtral 需要 >48GB 显存（无法在单 3090 部署）或 JellyFish 需 4-bit 量化，MELD 在 consumer GPU 上实现 full-precision MoE 推理。

## Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

- baseline方法是什么？
  **All-to-All 全连接 MoA**：现有 MoA 系统采用多层 all-to-all agent 连接拓扑——相邻层间的 agent 全连接，每个 agent 接收上一层所有 agent 的输出，经 aggregator 融合生成最终答案。硬件部署上，MoA serving 缺乏对 agent 间复杂数据依赖和异构延迟的支持：前驱 agent 解码与后继 agent prefilling 被视作严格串行，无 overlap 机会；且每层延迟由最慢 agent 决定（`T_ℓ^{all} = max_i t_{ℓ,i}`），导致 GPU 利用率低（pipeline stall time）。
  
  **全栈执行例子**：
  - **算法pipeline层**: 用户 query → Layer 1 所有 N 个 proposer agents（如 9 个，各用不同 LLM 骨干）并行生成答案 → Layer 1 所有输出拼接为 Layer 2 每个 aggregator agent 的输入（全连接，输入长度 = 原始 prompt + N × 输出长度）→ Layer 2 所有 aggregator 并行生成 → ... → 最终 aggregator 融合所有 Layer L-1 输出为最终答案。全连接拓扑导致：(a) 冗余连接传递无用信息；(b) 大模型（32B）与小模型（4B）同时启动但大模型慢得多，小模型完成后 GPU idle 等待；(c) aggregator 输入 context 极长（9×输出长度）。
  - **系统框架层**: Naive PD disaggregation 下，dependent agent（如 Layer 2 aggregator）必须等待所有前驱解码完成并收集输出后才能开始 prefill → 前驱解码期间后继 GPU 完全空闲。
  - **编译框架层**: 论文未明确说明（基于 SGLang/vLLM 原生 PD pipeline，无 agent-aware 优化）。
  - **Kernel调度层**: 论文未明确说明。
  - **硬件架构层**: All-to-all topology 下最慢 agent（如 32B 模型）决定层延迟 → 该层其他已完成 agent 的 GPU SM 空闲等待 barrier 同步 → 低硬件利用率。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **Faster-MoA** 通过三个协同设计解决上述缺陷：
  
  1. **层次化树状拓扑替换 All-to-All**：以 9-3-1 三层树结构替换全连接。Layer 1 的 9 个 agents 分为 3 个 clusters，Layer 2 的每个 agent 仅连接其对应 cluster（而非全部 9 个），Layer 3 的 root aggregator 连接所有 Layer 2 agents。效果：(a) 减少冗余连接——每个后继仅处理局部 cluster 输出而非全部；(b) 输入 context 长度从 9× 输出降至 3× 输出，prefill 成本线性下降；(c) 子树间可独立并发，不因跨 cluster 慢 agent 而被阻塞（`T_ℓ^{tree} ≈ max_{a_{ℓ,j}} max_{c∈C(a_{ℓ,j})} t_c`）。

  2. **语义引导动态 Early-Exit**：利用 FrobCosSim + 几何平均置信度计算早退概率 Q。当小 agent（4B/8B）输出语义一致且高置信时，以概率 Q 终止未完成的大 agent（32B），跳过其等待时间。这解决了 all-to-all 中"必须等最慢 agent"的问题，且 Q 自适应任务难度——难任务（IFBench）大模型被调用更多，简任务（GSM8K）大模型较少被调用。

  3. **依赖感知增量 Prefilling**：Shell Router 将依赖 agent 的 prompt 按前驱 agent 输出槽分割。前缀段无依赖立即 prefill；前驱解码出的 token chunk 流式 append 到 APC → shell router 周期性 fetch → 增量 /prefill_only update（复用已驻留在 HBM 的 prefix KV，近 100% cache hit）→ prefilling 计算被前驱 decoding 重叠隐藏。解决了 naive PD disaggregation 中"依赖 agent 必须等前驱全部完成后才开始 prefill"的串行瓶颈。

  **全栈执行例子（与 Baseline 对比）**：
  - **算法pipeline层**: 用户 query → Layer 1：9 agents 分为 3 clusters 并行执行，每个 cluster 内含 4B/8B/32B 三模型 → 完成 4B 和 8B 后 → 计算 Q = √(C̄·B)^(1/τ) → 若 Q 足够高，概率性终止 32B（不再等待）→ Layer 2：3 agents 各继承自己的 cluster 输出（仅 3 个，非全部 9 个）→ Layer 3：root aggregator 融合全部 Layer 2 输出为最终答案。
  - **系统框架层**: Shell router 接收到 Layer 2 agent 的请求 → 识别依赖 Layer 1 特定 cluster 输出 → 将 prompt 按前驱输出槽分割 → 前缀段立即发 /prefill_only → 监控 APC 中前驱 agent 的 decode chunk → 增量 /prefill_only（复用 prefix KV）→ decode 完成时 prefilling 已完成（被 decode 时间隐藏）→ 转发 /generate → 前驱解码与后继 prefilling 时间重叠。
  - **编译框架层**: 论文未明确说明。基于 SGLang v0.5.3 原生 xgrammar/torch.compile 机制，增量 prefilling 通过 KV cache reuse 机制实现 token 追加而非完整重计算。
  - **Kernel调度层**: GPU PE 在执行 /prefill_only 时维护 prefix KV blocks 在 HBM，增量 token 通过 FlashAttention 仅计算新 token 的 KV 并追加，prefix 部分直接从 HBM 读取（near 100% cache hit）。
  - **硬件架构层**: 6×H200 GPU，3 组 PE/DE pair。不同 size 模型的 decode 和 prefill 在不同 GPU 上并行，空闲 SM 被增量 prefill 任务利用。最终效果：~90% E2E 延迟减少，准确率 ±1%。

## Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism

- baseline方法是什么？
  Baseline 是 MegaScale-Infer [36] 中的 Ping-Pong Pipeline Parallelism (PPPipe) 算法，该算法是 DEP 架构下最先进的任务调度方法。

  **Baseline 全栈执行例子（以 DeepSeek-V2、ag 个 AG GPU + eg 个 EG GPU 为例）**：
  - **算法层**：MoE 层包含 Multi-Head Latent Attention (MLA) + Shared Expert + Routed Experts。每个 token 经 router 选择 top-k experts。Shared Expert 对所有 token 计算。
  - **系统框架层**：DEP 将 GPUs 分为 Attention Group (AG) 和 Expert Group (EG)。AG 存储所有 Attention 层参数和 Shared Expert 参数（全复制到 ag 个 GPU），EG 存储 E 个 sparse experts（分布在 eg 个 GPU，每个 GPU 持有 E/eg 个专家）。PPPipe 将 mini-batch 切分为 r1 个 micro-batch，使 AG 和 EG 可并行执行。A2E 和 E2A 通信通过 NCCL 实现，无 group 内通信（AG 内参数全复制、EG 内 token 按 expert 路由不跨设备）。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA + NCCL 编译路径）
  - **Kernel 调度层**：PPPipe 以 micro-batch 为粒度进行流水线调度。在 AG 端，Attention + Shared Expert 在每个 micro-batch 内顺序执行，完成后启动 A2E 通信。在 EG 端，每个 expert 对其分配的 token 执行 GEMM（Gate/Up/Down 三层投影），每个 micro-batch 的 expert 计算和 A2E/E2A 通信串行执行（仅 micro-batch 间有重叠）。A2E 与 Shared Expert 不可并行（PPPipe 将 Shared Expert 视为 Attention 的一部分）。
  - **硬件架构层**：四种 GPU 平台（8×A6000 48GB Ampere NVLink、8×A10 24GB Ampere PCIe only、8×H20 96GB Hopper NVLink、32×H20 四节点 NVLink）。计算在 Tensor Cores，通信通过 NVLink/PCIe。

  **Baseline 核心痛点**：
  1. Shared Expert 计算调度不当：PPPipe 假设无 Shared Expert，将 Shared Expert 视为 Attention 的一部分串行执行，但实际 Shared Expert 与 A2E 通信和 routed expert 计算之间无数据依赖，可以并行，造成 GPU 空闲。
  2. Micro-batch 级别流水线不足以完全隐藏通信：PPPipe 仅做 coarse-grained micro-batch 重叠，未能进一步将 A2E/E2A 通信与 expert 计算重叠。粗粒度下，一个 micro-batch 内的通信仍占用较长时间，导致另一端 GPU 等待。
  3. 解空间巨大难以找到最优调度：引入 shared expert 支持和细粒度流水线后，r1/ma/r2/me 的组合搜索空间爆炸，Brute-force 枚举不可行。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  FinDEP 提出三方面创新：

  **1) 细粒度任务切分与流水线（解决痛点 1 和 2）**：
  将 AG 端任务沿 batch 维度切分为 r1 个 pipeline 段（每 GPU 处理 ma 个样本），将 EG 端任务沿 token 维度进一步切分为 r2 个 fine-grained 段（每个 expert 处理 me 个 token）。通过 r1 和 r2 两级流水线实现：AG 内 Attention 与 Shared Expert 交替执行（ASAS 策略），A2E 通信与 Shared Expert 计算并行，A2E/E2A 通信与 EG 内的 expert 计算重叠。

  **2) 形式化优化问题（解决痛点 3）**：
  建立线性性能模型（α-β 模型）预测 GEMM/Attention/A2E/E2A 的执行时间，将 DEP 推理时间形式化为包含 r1、ma、r2、me 和任务顺序的优化问题（目标函数 Eq.13），证明目标函数关于 ma 单调递增（Theorem 1/2）、关于 r1 单调非递减（Theorem 3）、关于 1/r2 是凸函数（Theorem 4）。

  **3) 高效近似最优求解算法（解决痛点 3）**：
  Algorithm 1 基于单调性和凸性约束搜索空间：(a) 利用 ma 和 r1 的单调性，只在 Pareto 前沿上搜索 (ma, r1) 组合；(b) 对每个组合，固定 r1/ma 后对 1/r2 做凸优化快速收敛；(c) 同时评估 ASAS 和 AASS 两种执行顺序。算法复杂度 O(C·√M)，实际求解 < 1s，使在线自适应成为可能。

  **论文方法全栈执行例子（以 DeepSeek-V2 为例，与 Baseline 对比）**：
  - **算法层**：同 baseline，MoE gating/routing 不变
  - **系统框架层**：DEP 架构不变，但在 AG 端增加两种执行顺序选择：(a) AASS（先全部 Attention、再全部 Shared Expert）——使 A2E 最早启动，利于 EG 尽早计算；(b) ASAS（Attention 与 Shared Expert 交替执行）——提高 AG 内 GPU 利用率。通过 Algorithm 1 在线选择最优顺序。A2E/E2A 通信从每个 micro-batch 一次变为每个 fine-grained 段一次（r2 倍频率但每次数据量减少），与 expert 计算重叠。
  - **编译框架层**：论文未明确说明（标准 PyTorch + CUDA 路径）
  - **Kernel 调度层**：
    AG 端：ma 个 sample 的 Attention → ma 个 sample 的 Shared Expert → me 个 token 的 A2E 通信，pipeline 重复 r1 次。ASAS 策略下 Shared Expert 与下一 micro-batch 的 Attention（或无数据依赖的 A2E）交替执行。
    EG 端：r2 段 fine-grained pipeline，每段处理 me 个 token。A2E 收到 me 个 token 后立即启动 expert 的 GEMM 计算，同时下一段 me 个 token 的 A2E 通信可并行进行。关键调度参数：X(ma)=ta+ts, Y(me)=max(te, ta2e), F(ma,me)=max(X, r2·Y) 控制各段之间的时序约束。
  - **硬件架构层**：同 baseline 的 4 种 GPU 平台。效果：在 8×A6000 DeepSeek-V2 S=4096 下，非重叠通信时间从 PPPipe 的 528.94ms 降至 309.81ms（1.7× 减少）。在 32×H20 Qwen3-MoE S=4096 下吞吐提升 1.24×。

## Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

- baseline方法是什么？
  Baseline 是 EPLB (Expert Parallelism Load Balancer) 的 token routing 算法——将每个 expert 的 token 均匀分配到其所有 replicas 上（"token-balancing"）。EPLB 的完整流程包含两步：(1) expert replication：按上一时间窗口各 expert 处理 token 数的比例创建 replicas；(2) expert placement：将 replicas 放置在 GPU 上以平衡各 GPU 期望处理的 token 数；(3) token routing：将每个 expert 的 token 均匀分配到其 replicas 上。Baseline 的核心假设是"GPU runtime 与处理的 token 数成正比"，这在 compute-bound 场景（prefill）下成立，但在 memory-bound 场景（decode）下不成立。

  **Baseline 全栈执行例子（以 Qwen3-30B decode batch=32 tokens/GPU, 8×A100, 1.5× replication 为例）**：
  - **算法层**：MoE router 为每个 token 计算 top-k experts；EPLB token routing 将每个 expert 的 token 均匀分配到所有 replicas——若 expert e 有 3 个 replicas 和 9 个 token，则每个 replica 分到 3 个 token
  - **系统框架层**：vLLM EP——Attention 层 DP，MoE FFN 层 EP 分布在 8 个 GPU 上。EPLB 的 expert placement/replication 根据历史 token 分布周期性更新
  - **编译框架层**：vLLM CUDA Graph compilation——prefill 和 decode 的 MoE 层计算被编译为 CUDA Graphs（论文未修改编译框架）
  - **Kernel 调度层**：decode 阶段每个 MoE layer 执行：local top-k → all-to-all dispatch → FFN (cutlass GroupGemm) → all-to-all combine。由于令牌均匀分布，每个 GPU 上可能激活更多 expert replicas（每个 expert 的 token 分散到多个 replica），导致加载 expert weight 的内存流量增加
  - **硬件架构层**：8×NVIDIA A100 40GB, 600 GB/s NVLink。Memory-bound 的 decode 阶段，FFN runtime 由 HBM → Tensor Core 的 weight 加载带宽决定，而非 Tensor Core 计算。更多 activated experts = 更多 weight 需加载 = 更长的 memory traffic = 更高的 decode latency

  **Baseline 核心痛点**：
  1. EPLB token-balancing 在 memory-bound decode 阶段错误地增加了 activated expert replicas 数量（1.5× replication 下 activated experts 增加 ~30% vs no-replication），因为均匀分配迫使更多 replica 被激活
  2. 增加的 activated experts 导致更多 expert weights 需从 HBM 加载到 Tensor Cores，memory traffic 增加，decode latency 恶化（1.5× replication 下 +14% TPOT）
  3. 虽然 replication 能改善 compute-bound prefill 性能（-17% TTFT），但对 memory-bound decode 的退化反而导致 overall token throughput 下降（-10% at 1.5× replication）
  4. 换言之，EPLB 强制 prefill 和 decode 使用同一套 load-balancing 策略，无法为 memory-bound decode 阶段做针对性优化

- 论文方法是什么？如何对应解决Baseline的缺陷？
  METRO 提出将 token routing 目标从 "balance tokens across GPUs" 改为 "minimize activated experts across GPUs"，因为 memory-bound regime 下 GPU runtime ∝ activated experts 数量而非 token 数量。具体设计包括三个组件：

  1. **MIN-EXP-ROUTING 问题形式化**：将最小化 activated experts 建模为 ILP 问题，通过 Lemma 1 证明只需将每个 expert 的所有 token 路由到单个 replica 即可——这从根本上避免了 token-balancing 导致的 replica inflation
  2. **GPU-native 贪心近似算法**：由于 ILP 最优解计算开销过大（31%-104% FFN time），METRO 使用 O(|A|) 复杂度的 greedy 算法——每个 expert i 并行选择当前 activated experts 最少的 GPU g*，全序加锁避免死锁，运行在单 SM 上仅需 17-26us
  3. **All-gather dispatch 替换 all-to-all**：使每个 GPU 获得全局 top-k 知识 T[1..N]，作为 Algorithm 1 的输入。在 memory-bound 小 batch 下的开销可忽略（~3us bandwidth vs ~100us NCCL launch latency）

  **论文方法全栈执行例子（以同样 Qwen3-30B decode batch=32 tokens/GPU, 8×A100, 1.5× replication 为例）**：
  - **算法层**：MoE router top-k 不变。METRO routing：对于每个有 token 的 expert e，查 placement matrix 获取其候选 GPU 集合，选择当前 activated expert 计数最少的 GPU 进行路由（greedy min-L），该 expert 所有 token 路由到同一 GPU。例如 expert e 有 3 个 replicas (GPU 0, 3, 5)，当前 L=[2,1,1,1,2,0,1,1]，则选 GPU 5 (L[5]=0)，e 的所有 token 全部路由到 GPU 5，仅激活一个 replica
  - **系统框架层**：vLLM EP + METRO——Attention 层 DP 不变。METRO 仅替换 decode phase 的 token routing 逻辑，prefill phase 继续用 EPLB routing。METRO 与 EPLB 共用 expert placement/replication 策略，不干扰 prefill
  - **编译框架层**：vLLM CUDA Graph compilation——METRO routing kernel 被编译进 decode phase CUDA Graphs，power-of-two batch sizes 预编译（论文未修改编译框架本身，仅添加 kernel 到 graph）
  - **Kernel 调度层**：decode 阶段每个 MoE layer 执行：
    1. all-gather tokens（替换 all-to-all dispatch）→ 每个 GPU 获得全局 ~256 tokens
    2. 每个 GPU 在全局 token 集上计算 top-k → 构建 T[1..128]
    3. METRO CUDA kernel（单 SM）greedy 路由 → 每个 expert 匹配到单一 GPU
    4. Expert FFN 仅计算分配给本 GPU 的 activated experts → 仅加载被激活 expert 的 weight
    5. all-to-all combine 返回结果
    ⚡ 关键改进：activated experts 减少 up to 42.3% → memory traffic 减少 → FFN 时间减少 up to 81us/layer → end-to-end decode latency 降低 11%-22%
  - **硬件架构层**：8×NVIDIA A100 40GB, 600 GB/s NVLink。METRO 不改变硬件使用方式，但通过减少 activated experts 数量直接降低了 HBM → Tensor Core 的 weight 加载量。在 memory-bound regime 下，weight 加载是 runtime bottleneck，减少 activated experts = 减少 memory traffic = 降低 latency

  **痛点映射**：
  | Baseline 痛点 | METRO 解决方案 |
  |---|---|
  | Token-balancing inflate activated experts (+30%) | MIN-EXP-ROUTING 最小化 activated experts per GPU (-42.3% vs EPLB) |
  | 更多 activated experts → 更多 weight 内存流量 → decode latency 恶化 | Greedy routing 使每个 expert 的 token 集中到单一 replica，减少 activated experts → 减少 memory traffic → decode latency 降低 11%-22% |
  | EPLB 强制 prefill/decode 用同一 routing 策略 | METRO 仅应用于 memory-bound decode phase，prefill 继续用 EPLB token-balancing |
  | ILP 最优解计算开销过大（31%-104% FFN time） | O(|A|) greedy algorithm on single SM：17-26us，near-optimal (within 10.9% of optimal) |
  | All-to-all dispatch 无法提供全局 top-k 信息 | All-gather dispatch 使各 GPU 获得全局 T[1..N]，overhead 在 memory-bound 小 batch 下可忽略 |

  实验效果：(a) decode latency 降低 11%-22%；(b) total token throughput 提升 3%-21%（co-deployed prefill+decode）；(c) 在固定 SLO 下 decode throughput 达 EPLB 的 1.98x-4.11x；(d) 这些增益在 Qwen3-30B、Qwen3-235B、DeepSeek-V3 等多种模型和 InstructCoder、NuminaMath、Humaneval、GSM8K 等多种 workload 上一致。

## ElasticMoE: An Efficient Auto Scaling Method for Mixture-of-Experts Models

- baseline方法是什么？
  Baseline 方法包括四种自动缩放策略，均基于 vLLM 实现：(1) **Horizontal (Replica)**：启动完整独立推理实例副本（如 DP2-TP2-EP4 的 4 NPU 副本需再加 4 NPU），旧实例继续服务，新实例冷启动。粗粒度（最小增量 = 完整副本规模），高延迟（容器启动+权重加载+通信初始化+KV cache 分配，可达数十秒到分钟级），参数冗余（每个实例独立复制 expert 权重，内存浪费）；(2) **Vertical (Cold Restart)**：停止旧实例，在新 NPU 集合上重启扩展配置（如 4→6 NPU 仅需 6 而非 10），细粒度但引入 downtime（旧实例销毁期间无服务）；(3) **Vertical (Extravagant)**：新实例在独立 NPU 上并行启动，旧实例继续服务（4→6 需 10 NPU 临时总量），无 downtime 但资源翻倍，成本高；(4) **Vertical (Colocated)**：新实例在同组 NPU 上启动，4 颗共享 NPU 需临时持有两份模型权重和 KV cache，为避免 OOM 需提前缩小 KV cache，降低吞吐量。

  **Baseline 全栈执行例子（以 DeepSeek V2 Lite 模型，4 NPU 响应流量突发，Horizontal 缩放为例）**：
  - **算法层**：MoE gating 选择 top-k experts（64 experts，6 activated/token），attention + FFN 标准 Transformer decode
  - **系统框架层**：vLLM 部署在 DP2-TP2-EP4 配置（4 NPU）。当流量突发触发缩放时：(a) 检测到 SLO 持续低于阈值 → 触发 scale-out；(b) Kubernetes/Ray Serve 调度新容器/进程；(c) 新实例从磁盘加载完整模型权重到新 NPU 4-7；(d) 初始化通信组（HCCL init_process_group）；(e) 分配 KV cache 内存；(f) 新实例就绪后 Coordinator 更新路由表分流请求。整个过程需要数十秒到分钟级延迟，期间 4 NPU 的旧实例过载持续违反 SLO
  - **编译框架层**：论文未明确说明（使用标准 PyTorch CANN 编译路径）
  - **Kernel 调度层**：Ascend CANN API 管理 NPU 内存分配（aclrtMalloc），HCCL 管理通信（all-to-all for EP, all-reduce for TP），GEMM 在 NPU 计算单元执行。缩放时每个新 NPU 独立执行磁盘→HBM 加载（最慢链路）
  - **硬件架构层**：Huawei CloudMatrix384，Ascend 910C NPU（64 GB HBM），Unified Bus 互联。旧实例 4 NPU 满负荷运行，新 4 NPU 冷启动时从磁盘串行加载专家权重，Unified Bus 闲置

  **Baseline 核心痛点 (L1-L5)**：
  L1：高缩放延迟——新实例需冷启动（权重加载+通信初始化+KV cache 分配），数十秒到分钟级，无法应对突发短流量
  L2：高 downtime——Vertical Cold Restart 需销毁旧实例再启动新实例，in-flight 请求丢失，新请求排队积压
  L3：粗粒度缩放——Horizontal 必须启动完整副本（DeepSeek V3 最小 32 NPU），微小流量波动也需大量过度分配
  L4：低效 expert 重分配——Horizontal 每个实例独立复制 expert 权重，EP 度局限在实例内，无法跨实例统一 token 路由和负载均衡
  L5：高峰值内存——Vertical Colocated 在共享 NPU 上临时持有两份模型权重和 KV cache，OOM 风险或需缩 KV cache 牺牲吞吐

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ElasticMoE 提出三个核心机制解决上述五个缺陷：

  **(1) HBM 管理与推理执行解耦 (HMM + IMM) → 解决 L1, L5**
  - HMM 以持久守护进程独立管理模型权重和 KV cache，权重仅从磁盘加载一次
  - IMM 推理进程不直接加载权重，而是通过 zero-copy IPC 获取 HMM 的引用句柄
  - 旧实例终止不触发权重卸载，新实例无需磁盘 I/O 即可附加已有权重
  - 消除缩放期间的冗余内存分配峰值

  **(2) In-Place 缩放 + 零拷贝复用 + P2P 传输 → 解决 L2, L3**
  - 固定 TP 度不变，仅调整 DP 和 EP 度：共享 NPU 上 attention 权重和 KV cache 布局不变，可直接 zero-copy 复用
  - 新增 NPU 通过 HCCL P2P 传输获取权重（比磁盘快约一个数量级），绕过 host memory
  - "Scale-while-serve" 模型：旧实例持续服务 in-flight 请求，新实例在后台准备，准备就绪后无缝切换流量（零 downtime）
  - 支持增量为 2 NPU 的细粒度缩放（vs Horizontal 的 32-320 NPU 完整副本）

  **(3) 虚拟 Expert 管理 (vpage-remap) → 解决 L4**
  - 专家权重按非连续物理页存储但通过虚拟地址映射为连续逻辑张量（满足 GEMM kernel 要求）
  - EP 度变化时只需更新虚拟→物理映射而非全量拷贝/重新分配大缓冲区
  - 支持跨 NPU 灵活重新分配 expert 权重，避免 Horizontal 的 expert 复制冗余
  - 降低 peak memory 和使用中的延迟

  **论文方法全栈执行例子（以 DeepSeek V2 Lite，4→6 NPU scale-up 为例）**：
  - **算法层**：同 baseline，MoE gating 和 FFN 计算不变
  - **系统框架层**：
    1. Coordinator 的 SLO-aware Estimator 检测 SLO<90% → 触发 scale-up (DP2→DP3, EP4→EP6, TP2 固定)
    2. HMM 分析当前 vs 目标配置，生成最小代价计划：NPU 0-3 上 attention/KV cache 零拷贝复用，NPU 4-5 通过 HCCL P2P 从 NPU 0-1 接收 attention 权重
    3. Expert 权重全局 remap：p2p-copy 迁移到 NPU 4-5，vpage-remap 更新映射（旧映射保持活跃）
    4. IMM 从 LRU cache 取 pre-initialized 6-NPU 实例 → zero-copy attach HMM 权重和 KV cache → 标记 ready
    5. Coordinator 停止向旧实例路由新请求 → 等待 in-flight 完成 → 旧实例标记 inactive → 流量切到新实例
    全程旧实例持续服务（无 downtime），新实例准备期间与旧实例共享同一份 KV cache
  - **编译框架层**：论文未明确说明（CANN API 管理 NPU 计算图编译，PyBind11 桥接 C++/Python）
  - **Kernel 调度层**：
    - IpcSafeAllocator 拦截 torch.ones/empty/full → CANN IPC-compatible aclrtMalloc
    - p2p-copy: HCCL isend/irecv/broadcast + aclrtMemcpyAsync，经 Unified Bus 直接 NPU-to-NPU
    - zero-copy: rtIpcSetMemoryName → rtSetIpcMemPid → UNIX socket → rtIpcOpenMemory → torch::from_blob
    - vpage-remap: aclrtMallocPhysical (非连续物理页) → aclrtReserveMemAddress (连续虚拟地址) → aclrtMapMem (映射)
  - **硬件架构层**：Ascend 910C NPU × 6，Unified Bus 全互联。P2P 传输经 Unified Bus 而非 PCIe/host memory，延时极低。旧实例 4 NPU 和新实例 6 NPU 共享 NPU 0-3 上的 attention 权重和 KV cache 物理内存（通过 IPC 引用）

  **五个缺陷的对应解决**：
  | Baseline 缺陷 | ElasticMoE 解决方案 |
  |---|---|
  | L1 高缩放延迟 (数十秒到分钟) | HMM 持久权重 + IMM pre-initialized 实例 + P2P 快于磁盘 + zero-copy 消除重复加载 → scale-up 2.43s |
  | L2 高 downtime (Cold Restart) | Scale-while-serve: 旧实例持续服务直到新实例就绪 → 0 downtime |
  | L3 粗粒度 (32-320 NPU 增量) | 仅调 DP+EP，支持 2 NPU 增量细粒度缩放 → 灵活匹配需求 |
  | L4 低效 expert 重分配 | 全局 EP 重配置 + vpage-remap 无拷贝 expert 迁移 → 避免 expert 复制，提升 KV cache 容量 |
  | L5 高峰值内存 (OOM 风险) | 共享 NPU zero-copy 复用（非复制），新 NPU 仅 P2P 传输必要权重 → peak memory 仅比 Cold Restart 高 2-3%，比 Extravagant 低 35-40% |

  实验效果：(a) scale-up latency 为最佳 baseline 的 ≈0.11×（≈9× 改善），scale-down latency 为最佳 baseline 的 <0.15×；(b) 零 downtime；(c) peak memory 接近 Cold Restart 最优值（仅高 2-3%）；(d) 缩放期间 throughput 达 Cold Restart 的 ≈2×；(e) 在递增 RPS 下维持 SLO≥90% 到 ~8.7 RPS，远超 Cold Restart 和 Colocated baselines；(f) scale-down 后 SLO-per-NPU 最高（成本效率最优）。

## Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

- baseline方法是什么？
  Baseline 是传统的 KD (Sanh, 2019) 和 GKD (Agarwal et al., 2024)。KD 使用 forward KL divergence 最小化 teacher 和 student 在 token 级别的分布差异（使用 teacher 的原始 Top-k routing 选择 activated experts）。GKD 则使用 student 生成的 on-policy 数据 + reverse KL divergence，但同样仅依赖 MoE teacher 的 Top-k routing 机制。

  **Baseline 全栈执行例子（以 Llama-MoE-3.5B (4/16) → Sheared-Llama-1.3B 蒸馏为例）**：
  - **算法层**：MoE 教师使用 Noise Top-k Gating 计算 gate logits → softmax → Top-k selection。对于每个输入 token，仅激活 k=4 个 expert，gate probabilities 中 non-activated experts 的总概率 >50%（大部分层的 activated experts gate prob sum <50%），意味着大部分 expert knowledge 未被利用。教师输出 logits 与 student 之间计算 KL divergence 作为蒸馏损失。
  - **系统框架层**：论文未明确说明（标准 PyTorch/HuggingFace Transformers 训练循环，无特定 serving 框架修改）
  - **编译框架层**：论文未明确说明（使用 SynapseAI 编译框架，Intel Gaudi v2 加速器后端）
  - **Kernel 调度层**：论文未明确说明（标准 MoE FFN 前向传播 kernel，expert selection 通过 KeepTopK 掩码实现）
  - **硬件架构层**：4 × Intel Gaudi v2 加速器，路由选择在 Gaudi 上执行，无法将不同 expert 的激活分配到不同设备上以利用所有 expert

  **Baseline 核心痛点**：
  1. MoE 教师仅有 Top-k expert 参与知识生成，non-activated experts 不参与。但 non-activated experts 的 gate probabilities 总和超过 50%，意味着大量有价值的知识被浪费。
  2. KD 和 GKD 均为 dense-to-dense 场景设计，不感知 MoE 的 expert routing 特性。在 dense teacher 和 MoE teacher 性能相当的情况下，dense teacher 竟然作为更好的教师（学生 ROUGE-L 更高），说明现有方法未能有效利用 MoE 的分布式知识。
  3. Load balancing 使同一输入的不同训练迭代可能激活不同 expert 集合，知识被分散到多个 expert 中，但 conventional KD 每次只取 Top-k，无法覆盖完整知识。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  论文提出两种 MoE 专用的 KD 方法：

  1. **Knowledge Augmentation (KA)**：每次迭代对同一输入进行 M 次教师前向传播，每次以概率 λ 从 gate probability 分布中随机采样 N-1 个 expert（以 1-λ 概率取 Top N-1 个），生成 M 份增广知识。使用 student 生成的 pseudo-target + reverse KL 进行蒸馏。
  2. **Student-Aware Router (SAR)**：先以 student 反馈（reverse KL + auxiliary load balancing loss β=0.01）训练 MoE 教师的路由器，再使用更新后的路由器激活所有 expert 并加权聚合输出进行蒸馏。

  **论文方法全栈执行例子（与 baseline 对比）**：
  - **算法层**：
    KA 模式下：MoE 教师前向 → gate logits H(x) 计算 → softmax → 以 λ=0.05 概率从 gate prob 采样 N-1=15 个 expert（以 0.95 概率取 Top 15）→ 激活 15 个 expert → 加权聚合 → 重复 M=2 次，每次不同 expert 组合（通过采样随机性）→ 每次均与 student on-policy 输出计算 reverse KL → 参数更新。
    SAR 模式下：student 前向生成 pseudo-target → teacher 前向（激活所有 16 个 expert，使用完整 gate prob 加权）→ 计算 reverse KL + β·L_b → 仅更新 router 参数 W_g, W_noise → router 更新后，teacher 再次前向（激活所有 expert，用更新后 gate prob 加权）→ 计算 student reverse KL → 更新 student 参数。

  - **系统框架层**：论文未明确说明（标准训练框架，与 baseline 相同）
  - **编译框架层**：论文未明确说明（与 baseline 相同，使用 SynapseAI 在 Gaudi v2 上执行）
  - **Kernel 调度层**：论文未明确说明。KA 在同一迭代内对同一输入执行 M 次教师前向（M=2），每次选 N-1 个 expert，计算量增加但 non-activated expert 减少。SAR 激活所有 expert（全激活），每次迭代多一次 router 更新前向，计算开销通过 β=0.01 的轻量 load balancing 控制。
  - **硬件架构层**：论文未明确说明（与 baseline 相同，4 × Intel Gaudi v2）

  **三个缺陷的对应解决**：
  | Baseline 缺陷 | 论文解决方案 |
  |---|---|
  | Non-activated experts 知识未被利用（gate prob sum >50% 的 expert 被丢弃） | KA 将激活 expert 从 k 扩展到 N-1，覆盖几乎全部 expert；SAR 激活所有 N 个 expert，100% 覆盖 |
  | Dense teacher 比 MoE teacher 更好（现有 KD 不适用于 MoE） | KA 和 SAR 在 MoE teacher 下均超过 dense teacher + GKD 的效果（如 KA: 25.71 avg vs GKD dense: 24.89 avg） |
  | Load balancing 导致知识分散但 KD 只取单次 Top-k | KA 通过 M 次采样聚合不同 expert 组合的知识，模拟多次 routing 的多样性；SAR 通过训练 router 动态调整 expert 权重，让 router 感知 student 需求 |

  实验效果：(a) KA 和 SAR 在所有 MoE teacher 配置下均超过 KD 和 GKD baselines，最高提升 KA +4.8 ROUGE-L over KD；(b) ALL（直接全激活）优于 KD/GKD 但不如 KA/SAR，验证 router 训练的价值；(c) KA 的 M=2 最佳，M 过大导致过度多样性反而降低性能；(d) SAR 的 KL divergence 随层深增加而增加，说明 student-friendly router 调整在深层累积效果明显；(e) λ=0.05 取得最佳 trade-off（随机采样与确定性选择的平衡），过大 λ 导致性能下降。

## ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference

- baseline方法是什么？
  **Baseline 是现有 MoE offloading 推理系统（Cache-MoE, SE-MoE, Pregated-MoE）**，它们在单 GPU 内存受限场景下将 inactive experts 卸载到 CPU 内存并按需加载。

  **Baseline 全栈执行例子（Cache-MoE 处理两个 batch 的 MoE 推理）**：

  - **算法 Pipeline 层**：Standard MoE 推理 —— router 计算 `softmax(xW_g)` 选 top-k experts，weighted sum 组合 expert FFN 输出。Router 在原模型内，无外部预测器。
  - **Serving 框架层**：Cache-MoE 使用固定 per-layer expert cache + LRU 替换策略。每层独立管理 cache，无视跨层 routing 模式。Token batch 保持原始顺序，不进行重组。Pregated-MoE 用 MLP 逐层预测 expert（sequential dependency），SE-MoE 用 ring-buffer 预取连续两层全部 experts（内存膨胀）。
  - **编译框架层**：论文未明确说明（使用 PyTorch 框架）。
  - **Kernel 调度层**：CPU→GPU expert 加载与 GPU compute 无法充分重叠。LRU 按 recency 驱逐，不感知 routing 模式，导致缓存命中率不稳定。SE-MoE 的 ring-buffer 在 Switch-128 等大 expert 数场景下加载大量 inactive experts 造成带宽浪费。Expert kernel 在少量 token 下 near-constant cost（roofline 模型的 memory-bound 区域），token 稀疏分布导致低计算效率。
  - **硬件架构层**：单 NVIDIA A40 GPU (48GB) + Intel Xeon Gold 6338 CPU。PCIe 带宽限制 CPU-GPU 传输。Mixtral-8×7B 在 All-in-GPU 下 OOM。

  **Baseline 的三个核心缺陷**：
  1. **Inefficient Expert Prediction**：回归式方法（gate score 近似）误差累积需要 fine-tuning 修复；启发式方法（token-expert 统计）无法捕获 input-dependent routing；学习式方法（ProMoE 逐层预测）限制调度灵活性。
  2. **Low Expert Utilization**：decoding 阶段 token 分布极度不均衡，部分 expert 仅收到单 token，expert kernel 在少量 token 下 near-constant cost，GPU 计算效率低。
  3. **Ineffective Expert Caching**：LRU 仅按 recency 驱逐，无视动态 routing pattern；SE-MoE ring-buffer 在大量 expert 场景下内存膨胀且重复加载 inactive experts。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  **论文提出 ExpertFlow，通过三个协同组件解决 Baseline 三大缺陷**。

  **ExpertFlow 全栈执行例子（对比 Baseline）**：

  - **算法 Pipeline 层**：
    RPP（T5-style encoder-decoder）替代 baseline 的无预测/逐层预测/回归预测。在单次前向传播中输出所有 token 在所有层的 expert 激活概率矩阵 (B, S, L, E)。训练使用 binary cross-entropy 多标签分类：`L = (1/LE) * Σ[r*log(p) + (1-r)*log(1-p)]`。模型仅 7.21 MB，in-domain accuracy >90%，跨域下降 5-10%。

    **解决缺陷 1**：一次性全局预测（vs ProMoE 逐层预测）提前暴露完整 routing plan，支持早期 prefetch 和调度；T5 encoder-decoder 捕获输入语义（vs 启发式方法），accuracy 比 TLP/SLP baselines 高 60-80%。

  - **Serving 框架层**：
    (a) TS 用 K-means 聚类将两个 batch 的 token 按 routing path 相似度重新分组为两个新 batch：`min Σ Σ (R1 + R2)`，其中 `R_k = OR(r_i for i in T_k)`。在 CPU 上 <10ms 完成聚类。Dual-Batch Pipeline 将 RPP+TS 与 MoE 执行重叠，消除调度 overhead。
    (b) ECE/PLEC 基于预测自适应分配各层 cache slot（如 layer_1 需求 3 experts、layer_2 需求 2 experts、总 cache 容量 4 → 分配 3:1），预取最可能需要的 experts，early-layer expert 完成后释放 slot 供后续层复用。Real-time Correction 在 GPU compute 期间异步加载误预测遗漏的 expert，重叠 I/O 与 compute。

    **解决缺陷 2**：TS 将相似 routing 的 token 聚集到同一 batch，减少 active expert 数（per-batch），增加 per-expert token 量（从单 token 到多 token），使 expert kernel 从 memory-bound 移向 compute-bound 区域。Switch-128 上 throughput 提升 1.17×。

    **解决缺陷 3**：PLEC 替换 LRU/ring-buffer，预测驱动槽位分配 + 运行时复用 → cache hit ratio 91.90%（CS=16, BS=4），比 LRU 高 15-36%。Real-time Correction 的 async load 与 compute overlap 消除 cache miss 延迟。

  - **编译框架层**：论文未明确说明。
  - **Kernel 调度层**：Dual-Batch Pipeline 将 RPP+TS 的 CPU 计算与 GPU 的 MoE 执行交叠。ECE 的异步 CPU→GPU expert 加载与当前 running expert 的 GPU compute 并行。论文未明确说明是否使用 CUDA stream 或自定义 kernel。
  - **硬件架构层**：单 NVIDIA A40 (48GB)。与 baseline 相同硬件，但通过预测驱动缓存和 token 重排将 GPU 显存从 15.26GB (Switch-128 AIG) 降至 1.03GB（最大 93.72% 降低），Mixtral-8×7B 从 OOM 降至 15.99GB 完成推理。

  **三个缺陷的对应解决**：
  | Baseline 缺陷 | 论文解决方案 |
  |---|---|
  | Inefficient Expert Prediction（逐层预测/回归近似/启发式不准确） | RPP 一次性全局预测所有层所有 expert 激活，T5 encoder-decoder 捕获输入语义，>90% accuracy |
  | Low Expert Utilization（token 分散到不同 expert，per-expert 单 token，kernel 低效） | TS 按 routing 相似度 rebatch token，减少 active expert 数并增加 per-expert token load，1.17× 提升 |
  | Ineffective Expert Caching（LRU 无视 routing，ring-buffer 内存膨胀） | PLEC 预测驱动 adaptive slot 分配 + runtime slot 复用 + async correction，hit ratio 91.90%，比 LRU 高 61.15% |

  实验效果：(a) Switch-128 CS=4 达 9.99× throughput vs SE-MoE；(b) GPU memory 最高降低 93.72%；(c) Mixtral-8×7B AIG OOM → ExpertFlow 15.99GB；(d) Qwen1.5 cross-domain 达 2.21× vs Cache-MoE；(e) RPP 准确率 >95% on Qwen1.5，跨域仅降 5-10%。

## Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference

- baseline方法是什么？
  当前 MoE 推理的 expert parallelism（如 DeepSpeed-MoE）严格要求每个 MoE 层执行 2 次 Alltoall 通信：第一次 Alltoall（token dispatch）将 token 从数据并行 GPU 发送到持有对应 expert 的 GPU，第二次 Alltoall（token gather）将计算完成的 token 拉回原 GPU 以执行下一层的 attention（因为 context 仅在原 GPU 上）。expert 在各 GPU 上的放置按 rank 均匀随机分配，不考虑跨层 routing pattern。

  baseline 全栈执行例子（GPT 350M MoE-32 在 8 GPU 上，每 GPU 4 experts/层）：
  - 算法pipeline层：GPT decoder stack，每层 = self-attention + MoE FFN（Top-1 gating + GShard loss），token 按 gating score 选择 top-1 expert
  - 系统框架层：DeepSpeed-MoE expert parallelism，数据并行组 + 模型并行组（expert 维度），每 GPU 持有 E/P 个 expert per layer
  - 通信层：每 MoE 层 2× Alltoall（dispatch：scatter tokens to expert GPUs；gather：return tokens to data-parallel GPUs），attention 在 token 原 GPU 执行
  - 硬件层：A100 GPU，NVLINK intra-node，IB HDR inter-node。当跨节点时 Alltoall 占端到端时间可达 76%

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出 ExFlow，包含两个核心设计：

  **(1) Context-Coherent Expert Parallelism**（解决 baseline 第二个 Alltoall 的问题）：
  在推理开始时通过 AllGather 使所有 GPU 持有全部 token 的 context，每轮迭代结束后再次 AllGather 新生成的 token。由此 token 可在任意 GPU 上原地执行 attention，不再需要 gather Alltoall 返回原 GPU。每个 MoE 层通信从 2× Alltoall 降为 1× Alltoall + 1× AllGather（仅在迭代结束时）。

  **(2) ILP-based Expert Affinity Placement**（解决 baseline expert 随机放置导致大量跨 GPU routing）：
  离线从 Pile 采样数千 token 记录 routing log，计算 P(E_{p,j+1} | E_{i,j}) conditional probability matrix。将 expert placement 建模为 ILP 最小化 token re-routing cost。通过 Lagrange 对偶将最大化 affinity 问题转化为最小化 disruption 问题。分两阶段优化：stage 1 优先最小化 inter-node routing（利用 NVLINK < InfiniBand 带宽层次），stage 2 在 stage 1 结果上最小化 intra-node routing。

  论文方法全栈执行例子（同一 GPT 350M MoE-32 在 8 GPU 上）：
  - 算法pipeline层：同 baseline，无需修改模型参数或 gating function，无需 fine-tuning
  - 离线优化层：采样 3000 token → 记录 routing → 构建 conditional probability matrix → ILP 求解 expert placement → placement 直接用于模型加载
  - 系统框架层：ExFlow 修改 DeepSpeed 的 expert placement 逻辑，加载时按 ILP 求解结果分配 expert，context coherence 通过推理前后 AllGather 实现
  - 通信层：每 MoE 层仅 1× Alltoall（dispatch），attention 在原地执行；迭代结束 1× AllGather（新 token），当模型层数多时 AllGather 开销摊薄
  - 硬件层：同 A100 集群。通过 intra-GPU affinity → intra-node affinity → inter-node 的 staged 优化，适应 NVLINK > IB HDR 带宽差异

  baseline 缺陷 → 方法设计的映射：
  - **缺陷1**：每层 2× Alltoall → **设计1**：context coherence 消除 gather Alltoall，通信量从 2G·N·L·p 降为 G·N·(L·p* + G)
  - **缺陷2**：expert 随机放置，tokens 每层都需要大量跨 GPU routing → **设计2**：ILP 建模 expert affinity，使 affiliated experts 同 GPU，tokens 有更高概率留在本地
  - **缺陷3**：topo-aware gating（如 FasterMoE/TA-MoE）需要在 training 时加 topology loss，推理时 topology 改变则失效 → **设计3**：离线 ILP 求解，placement 可适应任意 topology，无需 retrain/fine-tune
  - **缺陷4**：之前方法（Lina）仅考虑 consecutive layer 的 top-k popular experts 并创建 replica，局部最优且浪费 GPU memory → **设计4**：全局 ILP 优化无需 expert replica，extreme case（每 GPU 仅 1 expert）仍可通过 intra-node affinity 加速

## eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

- baseline方法是什么？
  Baseline 是 MoE-based LLM 的标准推理系统（如 vLLM、DeepSpeedFastGen），其中 MoE 模型的所有 expert 在推理前全部预加载到 GPU 显存中（static pre-loading）。该方案直接导致 MoE 模型消耗 4×-14× 于同等 dense 模型的 GPU 显存。另一种 naive 方案是动态加载（dynamic loading）：所有 expert 存放在 CPU，推理时 instant transfer 所需 expert 到 GPU，但实验显示这会增加 3.2×-5× 推理延迟。Pre-gatedMoE 和 MoEInfinity 通过 prefetching 在计算当前层时预取下一层 expert 来 overlap 通信，但仍引入 2.5×-3.5× 的额外延迟。

  **Baseline 全栈执行例子**（以 Mixtral-8x7B 在 4× A100 40GB GPU 上为例）：

  **算法 Pipeline 层**：每个 token 通过 router gate 计算 top-2 expert → 仅激活 2/8 experts per layer。MoE layer 输出：`O = Σ_{i∈top-2} g_i · E_i(x)`。但所有 8 experts 的权重矩阵 (W_in, W_out) 均驻留在 GPU HBM 中。

  **系统框架层**：DeepSpeed-FastGen 或 vLLM 接受请求 → continuous batching → 每个 MoE layer 的 forward pass 执行 router gating → all-to-all dispatch → expert FFN → all-to-all combine → attention。所有 32 MoE layers（Mixtral-8x7B）的 8 experts/layer = 256 expert matrices 全部在 GPU 上。

  **编译框架层**：论文未明确说明。baseline 使用 PyTorch eager execution + DeepSpeed 的 MoE kernel 优化。

  **Kernel 调度层**：DeepSpeed-FastGen 的 MoE kernel 执行 all-to-all 通信（NVLink/PCIe between GPUs）+ expert FFN GEMM。每个 expert 的 weight 已固定在 GPU 上，expert 选择 → 直接 GEMM 无传输开销，但所有 expert 权重占用显存。

  **硬件架构层**：4× A100 40GB GPU（NVLink 互联）+ 128GB CPU host memory。Mixtral-8x7B ~47B 参数全部在 GPU 上：attention 权重 + 32 layers × 8 experts × (W_in+W_out) ≈ 96GB。512 tokens prompt 仅 memory 就占用 ~96GB（接近 4×40GB=160GB 的 60%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？

  eMoE 通过四个协同组件将 expert 从"全量常驻 GPU"变为"按需预测加载 + 周期性复用 + 任务感知过滤 + SLO 感知调度"，从而同时优化记忆体消耗和推理延迟。

  **eMoE 全栈执行例子**（以 Mixtral-8x7B，60% experts loaded，p=40 为例）：

  **算法 Pipeline 层**：
  - Expert Prediction 模型（BERT-XLNet, 0.108B params）学习 expert 路由序列的时序依赖。每个 prompt 的 expert 序列是跨层的 top-k 索引序列 `[e_1, e_2, ..., e_m]`，predictor 基于 consecutive layer 间的 cross-correlation（~0.50）和 consecutive prompt 间的 cross-correlation（0.75-0.95）预测未来 expert。
  - eMoE-A：`f([e_1^{r1}, ..., e_m^{r1}]) → [e_1^{r2}, ..., e_m^{r2}]`，用前一条 prompt 的 expert 分布预测当前。
  - eMoE-L：`f(e_{i-1}^{r1}) → e_i^{r1}`，逐层预测。
  - Predictor memory：仅 0.24%-1.3% of MoE model size。
  - 预测错误时：token 被路由到已加载的 next top-k expert（fallback routing）。

  **系统框架层**（DeepSpeed-FastGen 修改）：
  1. **Task Type Extraction**（CPU）：关键词匹配识别任务类型（SUM/CLSFY/QA/COMP/CONV）。
  2. **Task-aware Request Scheduling**（CPU, Algorithm 1）：从等待队列按 SLO stringiness 排序 → 遍历检查 `t_i = ΔE + (W + n_i·G_i)·c + r_i < SLO` → 贪心调度。G_i 为 profiled 任务特定生成 token 数，运行时递减。
  3. **Task-aware Expert Loading**：对每个 MoE 层计算 `N_i = (ΣW_j + T·W_o) · s · f_i`。s=0 时（任务对该层 routing 不敏感）跳过 expert 加载，直接复用已加载 expert。排序 N_i 后仅加载 top L（L 由 memory budget 决定）。
  4. **Periodic Expert Invocation**：每 40 prompts 调用一次 predictor；中间的 39 prompts 复用已加载 expert。Correlation 分析显示 consecutive prompts 间 correlation 为 0.48-0.55，perplexity 在 ≤60 prompts 内基本不变。

  **编译框架层**：论文未明确说明。

  **Kernel 调度层**：
  - Expert 加载：`torch.Tensor.copy_(non_blocking=True)` 异步 CPU→GPU 传输，与 self-attention layer（non-expert）计算重叠。
  - 同步机制：Python multiprocessing lock per MoE layer + CUDA event 防止使用 stale weights。
  - PCIe 带宽管理：当前 MoE 层的 expert 加载以前一层加载完成为条件（conditioned loading），防止多路并发 DMA 饱和 PCIe 通道。
  - Expert 卸载：不在预测列表中的 expert → 从 GPU 移到 CPU（释放显存）。

  **硬件架构层**（同 baseline）：4× A100 40GB GPU + 128GB CPU host memory + PCIe 总线。60% experts 加载时 Mixtral-8x7B 仅占用约 59GB（vs baseline 96GB），可处理 40× longer prompts 和 4.5× larger batches。

  **Baseline 缺陷 → eMoE 方法设计映射**：

  - **缺陷 1**：所有 expert 全量常驻 GPU 导致 4×-14× 记忆体开销 → **设计 1**：Expert Prediction（BERT-XLNet）基于 recurrent routing patterns 预测并仅加载所需 expert，memory 减少 up to 80%。
  - **缺陷 2**：per-prompt 动态加载导致 3.2×-5× 延迟增加 → **设计 2**：Periodic Expert Invocation（p=40），利用 consecutive prompts 的 high correlation（0.48-0.55）复用 expert，amortize 预测/加载开销至 0.24%-3.11%。
  - **缺陷 3**：所有任务无差别对待，导致不必要的 expert 加载 → **设计 3**：Task-aware Expert Loading，发现 Classification/Comparison 任务即使用 random routing 仍保持 >90% similarity，仅对敏感任务（Conversation/Summarization）精确加载，跳过不敏感任务的预测开销。
  - **缺陷 4**：现有调度器（vLLM, Orca, Sarathi-Serve）不考虑 expert loading latency 和 task-specific characteristics → **设计 4**：Task-aware Request Scheduling 联合 SLO + profiled output length + ΔE expert loading latency 做贪心调度，delay 宽松 SLO 请求以减少对 running requests 的干扰。
  - **缺陷 5**：Pre-gatedMoE/MoEInfinity 的 continuous prefetching 导致 CPU-GPU 带宽争抢 → **设计 5**：eMoE 仅周期性加载（而非每层 prefetch），配合 conditioned loading 避免 PCIe 饱和。（Pre-gatedMoE 2.4×-3.5× slower than eMoE）

## CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

- baseline方法是什么？
  - Baseline 方法：LLaVA v1.5 架构（CLIP ViT-L 作为视觉编码器 + 两层 MLP 连接器 + Mistral-7B LLM），所有模块均为 **dense MLP** 结构。
  - Baseline 全栈执行例子（从请求到 token 输出）：
    - **模型推理算法层**：输入图像 → CLIP ViT-L（dense MLP blocks）提取 visual tokens → 两层 dense MLP 将 visual tokens 投影到 word embedding 空间 → Mistral-7B LLM 的 dense MLP blocks 执行自回归解码，每个 token 必须通过全部 MLP 参数（7.1B 激活参数）
    - **系统框架层**：基于 LLaVA 框架，使用标准 PyTorch + HuggingFace Transformers 加载模型；推理时所有 dense 参数常驻 GPU 显存，使用 greedy decoding 策略
    - **编译框架层**：论文未明确说明
    - **kernel 调度层**：使用 flash-attention 加速 attention 计算；dense MLP 使用标准 PyTorch Linear + GELU kernel
    - **硬件架构层**：NVIDIA A100 GPU（8/16/32 卡），ZeRO-2/ZeRO-3/ZeRO-3-offload 分布式策略

  - Baseline 的痛点：
    1. **视觉侧扩展低效**：现有方法通过多编码器、更大 ViT、或复杂连接器（如 Q-Former）来增强视觉能力，但这些方法增加大量额外参数和 visual tokens，导致 LLM 处理负担加重
    2. **dense 模型的参数效率瓶颈**：dense MLP 的每个 token 必须通过全部参数，无法通过条件计算选择性激活相关专家来提升模型容量
    3. **MoE 在 LLM 侧已成熟，但视觉侧的 MoE 探索几乎空白**：MoE-LLaVA 仅在小型 LLM 中采用 MoE，未涉及视觉编码器或连接器

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **论文方法**：CuMo = CLIP-MoE + MLP-MoE + (可选 Pre-trained LLM-MoE)，通过 Co-Upcycling 将 pre-trained dense MLP 权重初始化为 MoE 专家，配合三阶段训练和辅助负载均衡损失。
  - **对应解决**：
    - **缺陷 1（视觉侧扩展低效）** → 将 CLIP ViT 和 MLP 连接器的 dense MLP 替换为 Top-2-in-4 稀疏 MoE 块，仅激活 50% 专家（CLIP-MoE 激活 0.50B/总 0.91B，MLP-MoE 激活 0.05B/总 0.10B），以极少的额外激活参数（7.1B → 7.8B，仅 +9.8%）显著提升视觉理解能力
    - **缺陷 2（dense 参数效率瓶颈）** → 每个 visual token 在 MoE 块中仅通过 Top-K 选中的 2 个 expert MLP（而非全部 4/8 个），通过 Router 的条件计算使模型容量增加而激活 FLOPs 几乎不变
    - **缺陷 3（视觉 MoE 缺乏训练方法）** → 提出 Co-Upcycling：用预训练/预微调 dense MLP 权重初始化 MoE 专家（而非随机初始化），避免训练不收敛；三阶段训练：MLP 预训练 → 全参数预微调（ALLaVA 标注数据温热模型）→ 含 MoE 的指令微调；辅助 bzloss（L_balance + L_z_loss）维持专家负载均衡

  - 论文方法全栈执行例子（对比 baseline）：
    - **模型推理算法层**：
      - 输入图像 → CLIP ViT-L（**每个 transformer 层中 dense MLP → Top-2-in-4 MoE**）：`X_out = Σ(i=1..2) W_K[i] ⊙ MLP_expert[i](X)`，仅 2/4 专家被激活 → 输出 visual tokens
      - MLP 连接器（**dense MLP → Top-2-in-4 MoE**）：visual tokens 通过稀疏门控路由，仅 2/4 专家参与投影 → word embedding tokens
      - LLM 解码（**Mistral-7B dense → 可选 Mixtral 8×7B pre-trained MoE**）：每 token 通过 Top-2 专家（12.9B 激活/46.7B 总）
      - 总激活参数：CuMo Mistral-7B = 7.80B（vs baseline 7.1B）；CuMo Mixtral-8×7B = 13.45B
    - **系统框架层**：基于 LLaVA + CuMo 自定义模块（`cumo/serve/`），Gradio Web UI / CLI 推理，支持 4-bit/8-bit 量化减少显存；训练使用 DeepSpeed ZeRO-3-offload
    - **编译框架层**：论文未明确说明
    - **kernel 调度层**：使用 flash-attention；MoE 的 Top-K 路由和加权求和通过标准 PyTorch scatter/gather 操作实现，无自定义 CUDA kernel
    - **硬件架构层**：NVIDIA A100 GPU（8/16/32 卡），三阶段训练逐步增加 GPU 数量（8→16→32），MoE 引入的额外参数通过 ZeRO-3-offload 卸载到 CPU 内存以节省 GPU 显存

## D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

- baseline方法是什么？
  Baseline 核心是**静态量化 + 按需加载**的 MoE 端侧推理：
  
  **(1) 固定 Bit-Width 量化**：EdgeMoE 通过离线 calibration 对每个 expert 分配固定 bit-width（如 INT2/3/4 混合），推理时不再改变。MC-MoE 根据 expert 激活频率和置信度设计固定分配。这种静态策略忽略 expert 重要性随输入 token 动态变化的特性（Observation #2），导致某些 token 分到不足以保精度的低 bit-width，另一些占用不必要的内存。
  
  **(2) 独立存储多版本权重**：若需支持动态 bit-width 选择，传统量化（GPTQ/AWQ）需独立存储 INT2/3/4 权重。LLaMA-MoE INT4 需 3.81GB，同时存 INT2/3/4 膨胀至 9.62GB。
  
  **(3) 按 Expert ID 顺序 I/O-Compute**：现有方法按 expert ID 升序加载和计算，无法重叠 I/O 与计算。Observation #3：LLaMA-MoE-3.5B 在 32 requests 时 I/O 2.6s + 计算 2.04s，但因 bubble 总延迟达 3.55s（增加 31%）。

  **Baseline 全栈执行例子（LLaMA-MoE-3.5B, RTX 3060, 单 request, 对比 D2MoE 的动态路由+MWQ）**：
  - **算法层**：输入 token → Embedding → Attention → MoE Gating（Top-2 选 expert 3 和 expert 7）→ static bit-width router（固定 INT4）。Expert 3 统计重要度低却被分配 INT4（过保守，内存浪费），Expert 7 统计重要度高也被分配 INT4，但无法利用更低 INT2/3 压缩。
  - **系统框架层**：Serving 按 expert ID 升序：[Expert 3 INT4 → Expert 7 INT4]。先加载 Expert 3 INT4（~320ms I/O），再计算（~3.1ms），此时 Expert 7 I/O 未开始（idle bubble）。计算完 Expert 3 后才加载 Expert 7（~320ms I/O），再计算（~3.1ms）。总时间 ≈ 646.2ms，bubble 3.1ms（Expert 3 计算时无并行 I/O）。
  - **Kernel层**：GPTQ-style dequantization（INT4→FP16 via scale+zero-point）→ cuBLAS GEMM。dequantization 占推理总时间 20%–70%。
  - **硬件层**：NVIDIA RTX 3060 (6GB) 无法容纳全量 INT4 权重（expert 参数占 ~89.9%），须逐 expert 从 NVMe SSD (3.5 GB/s) 加载。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  D2MoE 通过**算法-系统协同设计**的三个组件解决 baseline 缺陷：
  
  **(1) Token-Adaptive Bit-Width Selection → 解决静态 bit-width 无法适应 token 动态性**
  - 在每个 expert 前插入轻量化 bit-width router（额外开销 <0.5% 计算/0.5% 内存/1.7% 延迟），根据 token 表示动态选 bit-width
  - Quantized Expert Capacity：{c_k} 约束每个 bit-width 的 token 容量（如 D2MoE-V1 中 {0.3, 0.4, 0.3} 对应 INT2/3/4），超限 token 随机丢弃防止坍塌
  - Dynamic Bit-Width Selection Loss = CE(p, q) + (α/L) * Σ p_k^l * b_k，CE 保精度，正则项促选低 bit-width 以省内存
  - 解决 Observation #2：不同 token 下同一 expert 的重要度波动大，动态自适应获得更优准确率-内存权衡
  
  **(2) Matryoshka Weight Quantization (MWQ) → 解决多 bit-width 版本存储爆炸**
  - 嵌套量化：Asymmetric Quant 到最低 b_1（如 INT2）→ 对残差逐次 Binary Residual Quantization（+1/-1），每步增加 1 bit
  - b_K = b_1 + Σ_{k=2}^{K} 1-bit residual，高 bit-width 包含低 bit-width，如嵌套娃娃
  - 仅存一份 base INT2 + 若干 1-bit residual + 对应 scale factor，存储量接近 INT4 而非 INT2+INT3+INT4
  - 解决 Challenge #2（多版本高内存开销）
  
  **(3) Bit-Width-Aware I/O-Compute Pipeline + HEBF → 解决 I/O-Compute bubble**
  - MWQ 嵌套允许低 bit-width 权重被多个高 bit-width 请求复用。如 3 个 request 选 Expert 2（1×INT2+2×INT3），所有 3 个共享 INT2 base，仅 2 个 INT3 额外加载 1-bit residual
  - HEBF 按激活频率排序 I/O 队列（Hottest Expert Bit First），优先加载高频低 bit-width 权重
  - Memory Budget Scheduler：配置内存预算 M，保留高频低 bit-width 权重常驻 GPU
  - 解决 Observation #3（大 bubble）和 Challenge #4（轻量化调度）
  
  **D2MoE 全栈执行例子（同场景，LLaMA-MoE-3.5B, RTX 3060, M=1600MB）**：
  - **算法层**：token → Embedding → Attention → MoE Gating（Top-2 选 expert 3 和 expert 7）→ **bit-width router**（动态决策：Expert 3→INT2，Expert 7→INT3）
  - **系统框架层**：HEBF 按激活频率构建队列。假设 Expert 7(INT3) 频率 > Expert 3(INT2)：加载 Expert 7 INT2 base → 加载 Expert 7 +1-bit residual 的同时 Expert 7 INT2 dequant+GEMM 并行 → Expert 3 INT2 加载与 Expert 7 INT3 final 计算并行
  - **Kernel层**：Parallel Loading Dequantization Kernel 用 CUDA streams 重叠 disk→GMEM cudaMemcpyAsync 与 L2→CUDA cores dequantization。MWQ dequant = INT2 reconstruction(scale+zp) + Σ 1-bit residual * s_{b_k}（位操作代替传统 bit-transpose），单 kernel 减少 launch overhead
  - **硬件层**：NVIDIA RTX 3060。I/O 量从 INT4×2（~2.24GB→~640ms）降至 INT2 base + INT3 ≈ 448MB（~128ms），总 latency 从 ~646ms 降至 ~150ms（1.39× 吞吐提升），峰值内存降 33%–53%

