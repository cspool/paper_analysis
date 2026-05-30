## Mixture of Experts (MoE)

术语解释
MoE是一种神经网络架构范式，将模型容量分布在多个专门的子网络（"专家"）之间，通过可学习的路由机制（门控网络）为每个输入选择性激活相关的专家子集，实现条件计算。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
MoE的核心原理可表达为 y = Σ_{i=1}^{N} G(x)_i · E_i(x)，其中G(x)是门控函数输出，E_i(x)是第i个专家的输出，N是专家总数。

推理过程分为四步：
1. Router计算专家选择概率：θ = Softmax(R(x))，x∈R^d为输入token embedding，R(·)为路由函数，θ∈R^N为选择概率
2. Top-K选择：E_selected = TopK(θ, K)，选出概率最高的K个专家（K ≤ N）
3. 专家并行计算：y_i = E_i(x), ∀i∈E_selected，选中的专家各自独立处理输入
4. 加权聚合：y = Σ_{i∈E_selected} (θ_i / Σ_{j∈E_selected} θ_j) · y_i

在现代LLM中，MoE模块通常替代Transformer中的FFN层（如Mixtral-8x7B、DeepSeek-V2/V3），也有工作将其应用于Attention模块（如MoA、SwitchHead、MoH）。典型配置：总专家数N=8~256，每token激活K=1~8个专家。专家参数占比极高（如Mixtral-8x7B中专家占96%总参数）。

MoE的关键优势：
- 条件计算：仅激活专家子集，相比同等容量的稠密模型节省计算
- 专家专业化：不同专家可专注于输入空间的不同方面
- 动态路由：根据输入复杂度自适应分配计算资源

从算法pipeline角度拆解术语。
对于输入文本序列X=[x_1, ..., x_T]，MoE推理pipeline：

```
# MoE Layer Forward Pass
for each transformer layer l:
    # 1. Attention (dense, all tokens)
    A = MultiHeadAttention(LayerNorm(X))
    X = X + A  # residual
    
    # 2. MoE-FFN (sparse)
    X_norm = LayerNorm(X)
    for each token x_t in X_norm:
        θ = Softmax(R(x_t))           # router probabilities
        E_sel = TopK(θ, K)            # select top-K experts
        y_t = 0
        for i in E_sel:
            w_i = θ_i / sum(θ_j for j in E_sel)  # normalized weight
            y_t += w_i * E_i(x_t)     # expert FFN: W_2·σ(W_1·x_t)
    X = X + y  # residual
```

模型级优化在此pipeline上的改进：
- 量化：将E_i的权重W_1, W_2从FP16→INT4/INT2/INT1
- 剪枝：移除不重要的expert（structured）或其权重（unstructured）
- 动态门控：用自适应阈值替代固定K
- 蒸馏：将MoE教师的知识迁移到更小的学生模型

术语一般如何实现？如何使用？
- PyTorch实现：使用nn.ModuleList存储expert，nn.Linear实现router
- DeepSpeed-MoE：提供MoE层的分布式实现，支持expert parallelism
- HuggingFace Transformers：Mixtral、Qwen-MoE等模型的内置MoE实现
- vLLM：支持MoE模型的推理服务，带expert offloading

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach
- Accelerating Distributed MoE Training and Inference with Lina
- Accelerating MoE Model Inference with Expert Sharding
- Adaptive Gating in Mixture-of-Experts based Language Models
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- Brainformers Trading Simplicity for Efficiency
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models
- ExpertFlow: Optimized Expert Activation and Token Allocation for Efficient Mixture-of-Experts Inference

**Kim et al. (2025) 的 Non-Activated Expert Knowledge 观察**：
在 MoE KD 场景中，通过可视化各层 activated vs non-activated experts 的 gate probability 之和，发现：(1) 在大多数层中，activated experts 的 gate probability 总和低于 50%——即超过一半的"router 信心"分配给了未被激活的 expert；(2) 增加 activated experts 数量（k → N-1）提升 student 性能但不一定提升 teacher 自身性能，说明 non-activated experts 持有对 student 有价值的独特知识；(3) Load balancing 使同一输入在不同训练迭代可能激活不同 expert 组合→知识分散在多个 expert 中→传统 Top-k KD 每次只用到部分知识。这一发现催生了 KA（多次采样增广知识）和 SAR（训练 router 优化 expert 权重聚合）两种 MoE 专用 KD 方法。

**Brainformers 中的 MoE 使用（Non-uniform Block 中的 Sparse Layer）**：
Brainformers 将 MoE（sparsely gated FFN）视为一种通用稀疏化方法嵌入到非均匀 block 中。与 GLaM 的固定交替结构（alternating dense/sparse blocks）不同，Brainformers block 包含 3 种 sub-layer 类型（attention, dense FFN, MoE），演化搜索自动决定它们在 block 内的最佳数量和顺序。关键发现：
- MoE 层使用 Expert Choice gating + capacity factor=1 → 每 token 平均激活 1 expert，达到极致稀疏
- 演化搜索选择更大 model dim (1024) + 更小 MoE hidden dim (2048)，利用多 expert 的宽度替代单层大 FFN
- MoE sub-layer 在 block 中的占比为 3/8（vs GLaM 的 1/2，attention 减少至 2/8）
- 在 8B64E 规模实现 5x step time speedup + 2x training convergence speedup vs GLaM

**BrainMoE 的域特化 MoE 应用（脑 fMRI 基础模型）**：
BrainMoE 将 MoE 范式从 LLM 领域迁移到脑 fMRI 基础模型领域，核心区别在于：(1) 路由粒度不是 per-token 而是 per-sample（每个 fMRI scan 作为一个整体），由 Router 为每个样本选择 top-k expert；(2) Expert 按认知状态（cognitive state）分层预训练——12 种认知状态各训练一个独立 expert，而非随机初始化后联合训练；(3) Expert 输出的是 cognition embedding Z∈R^{C_hid}（脑活动特征表示），而非 token-level hidden state；(4) 下游不使用标准 MoE 的加权求和输出，而是通过 Cognition Adapter（Transformer Decoder with cross-attention）混合 expert embeddings 后做分类。BrainMoE 在 7 个下游数据集（ADNI、ABIDE、PPMI、Taowu、SZ、HCPA、HCPYA）上展示了 MoE 在跨认知状态泛化中的优势，尤其在 small-sample datasets（如 Taowu n=40）上 F1 提升 +43.76 over single-expert baseline。

**Task-MoE 的路由粒度扩展（Kudugunta et al., EMNLP 2021）**：
Task-MoE 将 MoE 路由从 token 级（per-token GATE(x_s)）改为 task 级（per-task GATE(task_id)），使 MNMT 中同一语言对的所有 token 共享相同的 experts。这允许在推理时提取 task-specific sub-network（仅 K=2 experts）直接部署，无需蒸馏。WMT 32 expert 配置下 decoder 221M→25M params (↓88%)；200 language pairs 128 expert 配置下 decoder 6.5B→201M (↓97%)。Peak throughput 提升 1.87x-2.6x。

**AquilaMoE 的 8×16B MoE 配置**：
AquilaMoE 使用 8 experts × 16B params each，top-2 routing（每 token 激活 2/8 experts，约 30B 激活参数），router 参数随机初始化为 N(0, 0.02)。通过 Sparse Upcycling 从 AquilaDense-16B checkpoint 转换而来。训练时加 load balancing loss（α=0.001）和 max z-loss（α=0.01）防止训练崩溃。

**MoEShard 的 Switch Transformer 使用**：
MoEShard 评估使用 Google Switch Transformer (Fedus et al., JMLR 2022) 的 Switch-Base encoder。Switch Transformer 是 top-1 routing（hard routing）MoE 架构，将 T5 encoder 的 FFN 替换为 MoE 层，每 token 仅路由到 1 个 expert（vs Mixtral 的 top-2）。Switch-Base 有 128 个 expert 的配置。MoEShard 仅评估 encoder 部分（非 decoder），因 decoder autoregressive 生成计算量较小且更依赖 fine-grained 优化。

**Lina 的 MoE All-to-All 瓶颈分析**:
在分布式 MoE 中，每个 MoE layer 需两次 All-to-All（dispatch + combine），平均占 step time 34.1%。GPU SM efficiency 在 All-to-All 期间仅 3.7%。Training 端 All-to-All（expert parallelism stream）与 Allreduce（data parallelism stream）在 backward pass 重叠→公平共享带宽→All-to-All 被延长 median 1.83x。Inference 端 expert popularity 倾斜（max/min ratio 4.02x~5.56x），uniform allocation 导致 popular expert device 过载。

**MELD 的独立外部 Router MoE 设计（KDD '24）**：
MELD（Mixture of Experts on Large Language Models for Data Preprocessing）提出一种不同于 Mixtral/Switch Transformer 等内置 MoE layer 的架构：使用**独立的外部 router network**，而非嵌入 Transformer 层内部的 gating 机制。核心区别：
- **Expert 独立性**：每个 expert 是基于同一 base LLM（Mistral-7B）用不同 task data 独立 LoRA fine-tune 的 adapter，而非联合训练的参数子集。Expert 训练和部署完全解耦，可灵活增删。
- **Router 独立性**：Router network 是一个独立的轻量 transformer（共享 sentence-bert 编码层），不嵌入 LLM backbone。Router 用对比学习训练，为每个 query 选择 top-k（k=3）diverse 且 relevant 的 experts。
- **推理机制**：通过 Punica + vLLM 实现 multi-LoRA serving。Router 选定 experts 后，Punica 动态加载对应的 LoRA adapter 到 base model 上，各 expert 独立推理后加权融合输出。单 3090 GPU 可同时 serving 200 个 LoRA experts。
- **与 Mixtral 对比**：Mixtral 的内置 MoE layer 中 experts 和 router 联合训练，load imbalance 严重，且 56B total params 无法在单 3090 部署。MELD 的 total params ≈ 7B（base model）+ N × LoRA params（每个约 10-50MB），deploy 灵活。
- **理论支撑**：Theorem 3 证明 router 能学习按 ITS（Intrinsic Task Subspace）cluster 分配数据；Theorem 2 证明 sparse MoE 比 single expert 的 error bound 更紧（与 sparsity factor s = O(√(k/N·(1+log(n/k)))) 成正比）。

---
