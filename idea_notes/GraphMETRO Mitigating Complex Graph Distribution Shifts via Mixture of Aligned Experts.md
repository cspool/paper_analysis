## GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

- baseline方法是什么？
  - **ERM (Empirical Risk Minimization) 训练的标准 GNN**：在源分布 D_s 上通过最小化经验风险训练 GNN（GCN/GIN/GAT），直接用于目标分布 D_t 的推理。全栈执行例子（以 WebKB node classification 为例）：输入网页节点特征 x_i + 邻接矩阵 A → 3层 GCN（GraphConv → ReLU → GraphConv → ReLU → GraphConv）→ 节点 embedding → MLP classifier → 输出 5-class 概率。训练使用 Adam optimizer + cross-entropy loss。
  - **Invariant Learning 方法（IRM [1], VREx [28], EERM [67]）**：假设存在对环境变量不变的表示或预测器，通过环境划分（environment partition）学习 invariant representations。全栈执行例子（EERM）：将源域数据构造成多个环境 → 对每个环境训练 GNN encoder → 通过正则化项强制 encoder 在不同环境间产生相似的表示 → classifier 在 invariant representation 上训练。IRM 额外约束最优分类器在环境间一致。
  - **Data Augmentation 方法（G-Mixup [22], SRGNN [85], OOD-G-Mixup）**：通过对训练数据进行特定类型 shift 的增广（如 graph size variation、local structure perturbation）来提升分布外泛化能力。全栈执行例子（G-Mixup）：在训练图的图元空间进行 mixup → 生成虚拟 OOD 样本 → 在增强数据集上训练 GNN。
  - **Graph-Specific OOD 方法（DIR [69], GSAT [45], CIGA [6]）**：通过 causal intervention 或 attention stochasticity 学习因果子图/不变子结构。全栈执行例子（DIR）：构建 intervention 分布 → 通过 causal 干预蒸馏 causal subgraph pattern → 在 causal subgraph 上做分类。GSAT：在 attention weights 中注入 stochasticity → 通过 information bottleneck 原则阻断 label-irrelevant 信息。
  - Baseline 痛点：
    1. **单一 shift 假设与现实脱节（核心痛点）**：现有 data augmentation 方法假设目标分布遵循某种特定的 shift 类型（如 graph size [49, 14]、feature noise [26, 8]、degree shift [65, 39]），但真实世界的分布偏移往往由多个 shift 维度融合而成（如 WebKB 的大学域偏移、Twitch 的语言域偏移），且每个维度的统计特性不同。单一维度的合成增广无法覆盖这种复合 shift。
    2. **环境划分的组合爆炸**：Invariant learning 方法依赖将数据划分为不同环境来学习不变性。但在复杂分布偏移下，环境空间 E 是巨大的——环境由不同子集节点和不同 shift 维度的组合构成，环境数量呈组合爆炸（product of subsets × shifts），使标准 invariant learning 不可行。
    3. **忽略 instance-wise heterogeneity（关键痛点）**：同一目标分布中，不同 instance（node/graph）可能经历不同类型和不同程度的分布偏移（如图 1 中 u^1 和 u^2 在 target domain 的内容特征变化程度不同）。现有 invariant learning 关注 group-level patterns，缺乏对 instance 级异质性的建模能力。
    4. **环境变量依赖**：EERM 等方法需要 domain/environment 信息来指导训练，但许多场景下 domain 标签不可得。GraphMETRO 不需要 domain 信息。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GraphMETRO 方法**：基于 Mixture-of-Experts 架构，通过"分解-对齐-聚合"三步策略处理复杂图分布偏移。
    1. **Shift Decomposition via Mixture Modeling**（解决痛点 1 和 2）：代替直接学习不变预测器，将未知分布偏移分解为 K 个 shift components，每个 component 由一种 stochastic transform function τ_i 定义（如 subgraph sampling、feature noise、edge removal）。关键假设（Assumption 1）：任意分布偏移都可以建模为最多 k 个 transform classes 的混合（k ≤ K）。这避免了环境空间的组合爆炸——用 K 个 basis transforms 替代 combinatorial environments，通过连续权重 w ∈ R^{K+1} 实现无限环境表达。
    2. **Instance-Adaptive Gating via MoE Architecture**（解决痛点 3）：Gating model ϕ 对每个 graph/node instance 输出个性化的权重向量 w ∈ R^{K+1}，权重编码了该 instance 的分布偏移中各个 shift component 的贡献度。这使得模型能捕捉 instance-wise heterogeneity——不同 instance 的 w 不同，因而被不同类型和程度的 shift 影响时产生不同的 expert 组合。
    3. **Referential Invariant Representation**（核心创新）：不同于传统 invariant learning 直接优化"表示在环境间不变"，GraphMETRO 设计每个 expert ξ_i 对其对应的 τ_i 产生 referentially invariant 表示：ξ_0(G) ≈ ξ_i(τ_i(G))。ξ_0 作为 reference model 为所有 expert 提供统一的表示空间 anchor。这使得不同 expert 的输出处于相同表示空间，可以在聚合时避免信息丢失。
    4. **Alignment via Frobenius Distance**（解决 aggregation 兼容性）：在 L2 中加入 Frobenius norm distance penalty d = (1/n)·||h(τ^{(k)}(G)) - ξ_0(G)||_F，强制聚合后的表示与 reference model 对齐。若缺少此对齐项（λ=0），WebKB 准确率从 41.11 暴跌至 18.79，验证了 expert 输出空间对齐的必要性。
    5. **τ^{(k)}-invariance 训练目标**：通过联合采样 τ^{(k)}（k 个 transform 的组合）和在 L2 中同时优化分类+对齐，模型学会对组合 shift 产生不变性（Theorem 2: composition of shifts）。
  - 全栈执行例子（GraphMETRO 在 WebKB node classification，与 ERM baseline 对比）：
    - **算法层**：
      - Baseline (ERM GCN)：x_i → 3× GCNConv → node embedding h_i → MLP → prediction。对 domain shift 无任何处理。
      - GraphMETRO：
        1. 对输入 subgraph 应用 5 种 τ_i（noisy_node_feat, add_edge, drop_edge, drop_node, random_subgraph）
        2. Gating GNN ϕ 输出 w ∈ R^6（K=5 + reference expert），表征该节点的分布偏移成分
        3. K+1=6 个 expert GNNs 分别编码 → reference expert ξ_0 在原始图上编码，其他 expert 在 transformed 图上编码
        4. Softmax(w) 加权聚合 → h → MLP classifier → prediction
        5. 训练时 L1（BCE: gating 预测正确 τ_i 组合）+ L2（CE + Frobenius alignment）
    - **系统框架层**：基于 PyG (PyTorch Geometric) 实现，标准 GNN 训练框架。GraphMETRO 训练时每 batch 对每个 graph 采样 τ^{(k)} 并生成变换图。独立 GNN encoder design（每个 expert 一个完整 GNN）更占内存但 expressiveness 更强；共享 encoder design 更省内存但性能降低（WebKB: 31.14 vs 41.11）。
    - **编译框架层**：论文未明确说明。使用 PyTorch 标准编译栈。
    - **Kernel 调度层**：论文未明确说明。标准 PyTorch Scatter/Gather + cuBLAS GEMM，无自定义 CUDA kernel。
    - **硬件架构/芯片设计层**：论文未明确说明。使用 NVIDIA GPU。
  - 关键实验数据：
    - Real-world datasets (Table 1): WebKB 41.11% (vs EERM 24.61%, +67.0%), Twitch 53.50% (vs EERM 51.34%, +4.2%), Twitter 57.24% (vs GSAT 56.40%), SST2 81.87% (vs DIR 81.55%)
    - Synthetic datasets: Average +4.6% over ERM across all shift environments
    - Synthetic DBLP average: GraphMETRO 81.08 vs ERM 77.88 vs ERM-Aug 78.63
    - Ablation: w/o L1 → WebKB 41.11→23.22 (gating 失效导致 expert 选择不准), λ=0 → WebKB 41.11→18.79 (alignment 是核心设计)
    - Invariance Matrix (Fig 4a): 对角线值最小，验证每个 expert 专精于其对应的 shift component
    - Distribution Discovery (Fig 4b): WebKB 主导 shift=add_edge, Twitch 主导 shift=noisy_node_feat+drop_node
    - Gating accuracy: WebKB 92.4%, Twitch 93.8%（多标签二分类）

  - **核心设计洞察**：GraphMETRO 的核心贡献是将"复杂分布偏移的泛化"问题从 invariant learning 的"寻找在所有环境中都不变的表示"重新定义为"将偏移分解为基础成分，智能地组合 expert 的表示来适应每个 instance"。关键在于 MoE 架构天然适合这个范式——gating 负责识别偏移成分（decomposition），expert 负责消除各自对应的偏移（mitigation），weighted aggregation 负责组合输出适应 instance-specific 的偏移（adaptation）。Referential invariant representation 是一个优雅的设计：通过 reference model ξ_0 作为 anchor，解决了不同 expert 独立训练时表示空间不兼容的问题，将 K+1 个表示空间的"翻译"简化为每个空间与同一 reference 空间的"对齐"。这使得聚合操作（weighted sum）在数学上有意义，而非简单地将异质向量拼在一起。

- baseline方法是什么？
  - **GPT-3 (175B Dense Decoder-only)**：标准 dense Transformer，175B 参数全部在每 token 推理时激活 (nact-params=175B)。全栈执行例子（推理 1 token）：Token → Embedding [1, 12288] → 96 层 dense Transformer（每层包含 Multi-head Self-Attention + Standard FFN, ReLU activation, 绝对位置编码）→ 所有 175B 参数参与逐层 matrix multiply → LM head → logits。FLOPs/Token=350G。训练使用 V100 GPU 集群, 训练能耗 1287 MWh, 300B tokens。
  - **GLaM Dense 基线**：同架构但无 MoE 层的 dense decoder-only 模型（0.1B, 1.7B, 8B, 137B），使用与 GLaM MoE 相同的数据集、tokenizer、优化器和超参数训练。
  - Baseline 痛点：
    1. **计算效率**：Dense 模型每 token 激活全部参数，FLOPs 随参数线性增长。训练能耗极高（GPT-3 1287 MWh）。
    2. **参数容量受限于计算预算**：在固定 FLOPs 预算下 dense 模型的参数量即表达能力上限，无法通过增加参数不增加计算的方式扩展容量。
    3. **知识存储效率低**：Dense 模型的知识密集型任务（如 TriviaQA）性能受限于 nact-params，而扩展 nact-params 同时线性增加推理成本。
    4. **数据效率低**：Dense 模型需要更多 training tokens 才能达到与 MoE 模型相同的 downstream 性能。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  - **GLaM 方法**：基于 sparse MoE 的 decoder-only Transformer。每隔一层 Transformer FFN 替换为 64-expert MoE 层，top-2 softmax gating 动态路由每 token 到 2 个 expert。1.2T 总参数，每 token 仅激活 96.6B（8%）。非 MoE 层使用 GLU+GeGLU 提升表达能力。采用 GSPMD 2D sharding 实现 expert 维度 + hidden 维度并行分布。
  - 解决 baseline 缺陷的对应机制：
    1. **计算效率飞跃**：sparse activation 使推理 FLOPs/Token 降至 180G（≈GPT-3 的 51.4%），训练能耗降至 456 MWh（≈GPT-3 的 35.4%）。原因：每 token 仅激活 2/64 experts，计算量主要由 nact-params (96.6B) 决定而非 nparams (1.2T)。
    2. **解耦参数容量与计算量**：通过增加 expert 数量 (1→256) 在固定 FLOPs 预算下指数级增加模型容量 (nparams 从 1.7B→105B, nact-params 仅从 1.700B→1.886B)，实现 "more capacity, same compute"。O(E²) 种 expert 组合为每 token 提供灵活的 sub-network 选择。
    3. **知识存储密度提升**：TriviaQA one-shot 达 75.8%（GPT-3 one-shot 68.0%），超越 fine-tuned SOTA KG-FiD (69.8%)。因 1.2T 参数提供了 7× GPT-3 的知识存储空间，而推理计算量仅为一半。
    4. **数据效率提升**：相同 training token 量下 MoE 模型性能远超 dense（Fig 4），GLaM (64B/64E) 用 280B tokens 训练即匹配/超越 GPT-3 用 300B tokens 的性能。因 expert 专业化使有限数据学习更有效的表示。
  - 全栈执行例子（GLaM 64B/64E 推理单 token）：
    - **算法层**：Token → SentencePiece tokenizer (vocab 256K) → Embedding [1, M=8192] → 64 层 decoder-only Transformer：
      - Layer 0 (Dense Attention): Q/K/V 投影 [M=8192, dhead=128 × nheads=128] → Relative Positional Bias → Softmax(QK^T/√d + rel_bias) → Attn output
      - Layer 1 (MoE FFN): Gating softmax(W_gate·x) 输出 [1, E=64] → top-2 选 expert i,j → dispatch x 到 expert_i, expert_j → 各 expert FFN: Linear[8192→32768] → GeGLU → Linear[32768→8192] → gate 加权 sum → residual add
      - Layer 2-63 交替 Attention/MoE 层... 
      - Non-MoE FFN 层: GLU → gate=GeGLU(x·W_g), value=x·W_v → (gate * value)·W_o → residual add
      → Final LM head [8192→256K] → softmax → next token logits。
    - **系统框架层**：模型权重和计算通过 GSPMD 2D sharding 分布到 1,024 TPU-v4 芯片。Expert tensor [E=64, M=8192, H=32768] 沿 E 和 H 划分。Input activation [B, S=1024, M=8192] 沿 B 和 M 划分。同一个 index 的 expert 跨层驻留同一 device（减少 expert-to-device 映射开销）。MoE 层用 while_loop 包装重复模块以降低 XLA 编译时间。All-to-all 通信用于 token dispatch/combine（expert 并行模式）。
    - **编译框架层**：使用 GSPMD 编译器自动推导非显式标注张量的 sharding 属性。XLA 编译器进行 TPU 后端代码生成和 while_loop 控制流优化。论文在 Section C 中描述了将 expert 按 index 对齐跨层放置以生成 "identical computation graph"，从而复用 while_loop 编译结果。
    - **Kernel 调度层**：论文未明确说明。TPU-v4 使用 TPU-specific matrix multiply unit (MXU) 执行 expert FFN 的 dense matmul。Sparse gating 的 token dispatch/gather 通过 all-to-all collective 通信实现（Lepikhin et al. 2021 GShard protocol）。
    - **硬件架构/芯片设计层**：使用 Google TPU-v4（326W/chip, PUE 1.11），未涉及自定义 RTL 或芯片架构修改。训练总能耗 456 MWh（600B tokens）或 213 MWh（280B tokens）。TPU-v4 的 bfloat16 支持用于激活值，float32 用于权重。
  - 关键实验数据：GLaM (64B/64E) vs GPT-3 (175B)：Zero-shot Avg NLG 54.6 vs 47.6, NLU 66.2 vs 60.8；One-shot NLG 58.4 vs 52.9, NLU 68.6 vs 65.4；Few-shot NLG 61.6 vs 58.8, NLU 71.4 vs 68.4。FLOPs/token: 180G vs 350G (-48.6%)。训练能耗: 456 MWh vs 1287 MWh (-64.6%)。CO₂排放: 40.2 tCO₂e vs 552 tCO₂e。
