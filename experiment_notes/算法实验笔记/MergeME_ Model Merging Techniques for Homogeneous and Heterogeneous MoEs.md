## MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

- 属于算法pipeline的实现是什么？实验比较什么？
  - 提出三类 MoE 合并技术：(1) 同构专家合并：用 Dare/Ties 合并替代 BTX 的简单平均（unweighted averaging）处理参数干扰（parameter interference），减少 post-merge fine-tuning 需求；(2) 无 fine-tuning 路由启发式：PPL（perplexity）路由 + 分离 attention 层替代训练的 router network，仅需一次额外 forward pass 计算 PPL 即完成路由决策；(3) 异构专家合并：通过 Proj-in/Proj-out projector（随机初始化 MLP）+ sequence-level router 将不同架构的专家模型合并为统一 MoE。
  - 实验比较：
    - 同构合并（Table 1）：BTX merging vs Ties merging vs Dare merging vs Random Routing vs Router Fine-tuning vs MoE Upcycling，6 benchmark 评估。
    - 无 fine-tuning 合并（Table 3）：Dare Dense vs Ties Dense vs merge attention + PPL routing vs separate attention + PPL routing vs separate attention + task vector routing，验证分离 attention 层和 PPL 路由的有效性。
    - 异构合并（Table 4）：3-expert MoE vs MoE w/ Math Olmo vs MoE w/ Math TinyLlama，验证 projector-based 异构合并性能。
    - 消融分析：路由概率分析（Figure 5/6/7/8/9）、不同 fine-tuning token 数量下的性能变化（Figure 10）、训练成本对比（Table 6/7/8）。

- 硬件平台是什么，配置是什么。
  - 论文未明确说明训练/评估所用的 GPU 具体型号和数量。论文提到 "limitations of computation resources" 限制在 1B 级别模型实验，BTX 论文（Sukhbaatar et al. 2024）的 MoE fine-tuning 需多 GPU 支持（因 expert 间 GPU 通信开销），但 MergeME 未说明具体集群配置。推测使用了至少支持 4-expert MoE（~3.7B 参数）训练的 GPU 集群。

- 模型是什么。数据集和bench分别是什么。
  - 模型：
    - Base-1B：Llama-2 架构，24 层，hidden_dim=2048，从 RedPajama 数据集（Arxiv, CommonCrawl, C4, StackExchange, 前一半 Wikipedia）random init 预训练 250B tokens。
    - Math Expert：Base-1B 在 OpenWebMath 上 CPT 100B tokens。
    - Code Expert：Base-1B 在 RedPajama GitHub 数据上 CPT 100B tokens。
    - Knowledge Expert：Base-1B 在 RedPajama 后一半 Wikipedia 数据上 CPT 100B tokens。
    - Math TinyLlama：TinyLlama-1.1B（22 层, hidden_dim=2048）在 Math Expert 相同数据上 CPT。
    - Math Olmo：Olmo-1B（16 层, hidden_dim=2048）在 Math Expert 相同数据上 CPT。
    - 同构 MoE：合并 Base-1B + Math Expert + Code Expert + Knowledge Expert，top-2 routing，总参数 ~3.7B。
    - 异构 MoE：合并 Base-1B + Code Expert + Knowledge Expert + (Math TinyLlama 或 Math Olmo)，总参数 ~4B。
  - 数据集/Benchmark：GSM8K (8-shot)、MATH (4-shot)、MBPP (0-shot)、HumanEval (0-shot)、Natural Questions (NQ, 5-shot)、TriviaQA (5-shot)。
  - 训练数据：CPT 用 100B tokens/专家，MoE fine-tuning 用额外 40B tokens（混合所有数据源，按 Table 5 比例采样）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 论文未提供开源链接。经 web search，未发现该论文的公开代码仓库（截至论文阅读时）。论文作者来自 University of Maryland 和 Amazon AGI。
  - 算法pipeline 核心流程（三种合并模式）：

```
输入: l 个 dense expert 模型参数 [θ₁,θ₂,...,θₗ]，base 模型参数 θ_b
输出: 合并后的 MoE 模型 θ_m

// ====== 模式1: 同构专家合并（Dare/Ties 替代平均）======

// 步骤 1: 计算 task vectors
for each expert i in {1..l}:
    τᵢ = θ_b - θᵢ  // task vector = base - expert 参数差

// 步骤 2a: Ties merging
// 参数: 保留比例 p (默认 80%)
for each task vector τᵢ:
    按 magnitude 排序，将 bottom (100-p)% 参数重置为 0  // 剪枝冗余参数
for each parameter position j:
    确定所有 τᵢ[j] 中总 magnitude 最大的符号方向
    τ_m[j] = Σ_{i: sign(τᵢ[j]) == 主导符号} τᵢ[j]      // 仅累加同符号值

// 步骤 2b: Dare merging（替代 Ties）
for each task vector τᵢ:
    随机 drop (100-p)% 参数（重置为 0）
    τᵢ = τᵢ / (0.01 * p)  // rescale 补偿丢弃
τ_m = Σ_{i=1}^{l} τᵢ  // 简单求和

// 步骤 3: 合并回 base model
θ_m = θ_b + λ · τ_m  // λ = 1/3（scaling term）

// 步骤 4: MoE 结构构建
// 非 FFN 层（embedding, attention, normalization, head）用 θ_m
// FFN 层保持各 expert 独立
// 插入 router network: θ_r * v → SoftMax(top-K(·))
// FF_MoE(v) = Σ_{i=1}^{K} SoftMax(top-K(θ_r · v)) · FF_i(v)

// 步骤 5: Fine-tuning
// 在混合数据源（Table 5 比例）上 fine-tune 全部参数（含 router），40B tokens

// ====== 模式2: 无 Fine-tuning MoE（PPL 路由 + 分离 attention）======

// 输入: 推理 prompt x_inf（t 个 tokens）
// 预处理: 各 expert 的 attention 层不合并，保持独立

// PPL 路由（替代 router network）:
for each expert i in {1..l}:
    PPL(x_inf | θᵢ) = exp( -1/t * Σ_{j=1}^{t} log P(xⱼ | x_{<j}, θᵢ) )
confidence_i = 1 / PPL(x_inf | θᵢ)

α = SoftMax(top-K(confidence_1, confidence_2, ..., confidence_l))
// α 即为各 expert 的权重

// Token 处理:
// 输入按 α 权重分配给 top-K expert
// 每个 expert 使用自己的 attention + FFN 处理
// 输出: Σ αᵢ · expert_outputᵢ

// ====== 模式3: 异构专家合并（Projector + Sequence-level Router）======

// 各 expert 架构不同（层数、hidden_dim 不同）
// 设最大 hidden_dim = d_m，各 expert hidden_dim = dᵢ

// 共享层:
// 1. Embedding 层 M_e: V → R^{d_m}（各 expert embedding/head 平均，小维度 padding 0）
// 2. Head 层 M_h: R^{d_m} → R^{|V|}

// 投影层（per expert，随机初始化 MLP）:
Proj-inᵢ:  R^{d_m} → R^{dᵢ}
Proj-outᵢ: R^{dᵢ} → R^{d_m}

// Sequence-level Routing:
// 将输入所有 token 的 embedding 平均
avg_e = 1/t * Σ_{j=1}^{t} eⱼ  // eⱼ = M_e(vⱼ)
α = SoftMax(top-K(θ_r · avg_e))

// Forward Pass:
for each selected expert k:
    e_proj[k] = Proj-in_k(e₁, e₂, ..., e_t)  // 投影到 expert k 的维度
    h_k = Expert_k.forward(e_proj[k])         // 标准 forward（含 attention + FFN）
    r_k = Proj-out_k(h_k)                     // 投影回 d_m

// 组合输出:
combined = Σ_{k in top-K} α_k · r_k
output_logits = M_h(combined)  // head 层输出 token 概率分布

// Fine-tuning: 所有参数（含 projector）在混合数据上 fine-tune
```

  - 关键超参数配置：
    - Dare/Ties: p=80% (retain ratio), λ=1/3 (scaling term)
    - Top-2 routing for all MoE models
    - CPT learning rate=1e-5, weight decay=0.01
    - Fine-tuning: 40B tokens on mixed data sources
    - Inference: temperature=0.0 (greedy decoding), max generated tokens=512
