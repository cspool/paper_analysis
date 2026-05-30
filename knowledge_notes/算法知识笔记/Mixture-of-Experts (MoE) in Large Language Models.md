## Mixture-of-Experts (MoE) in Large Language Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture-of-Experts (MoE) 是一种神经网络架构范式，通过将模型的 FFN（前馈网络）层替换为多个并行的"专家"子网络（Expert FFNs），并引入路由机制（Router/Gating Network）动态地为每个输入 token 选择激活哪些专家。与传统 Dense 模型每个 token 激活全部参数不同，MoE 实现了条件计算（conditional computation）：每个 token 仅激活全部专家中的 top-k 个（通常 k=2），从而在保持模型总参数量（capacity）极大的同时，控制实际激活的参数量（compute cost）。MoE LLM 通常由 m 个 Transformer 层组成，每层包含 n 个专家（Expert FFN），Router 为每个 token 输出 n 维的 softmax 概率分布，选择 top-k 个专家。最终输出为 k 个选中专家输出的加权和：$\text{output} = \sum_{i=1}^{k} P_{\text{topk},i} \cdot \text{Expert}_{\text{FFN},i}(x)$。典型 MoE LLM 包括 Mixtral 8x7B、DeepSeek-MoE-16B 等。

MoE 的核心优势在于：通过增加专家数量（扩容量）而非激活参数量（控计算），在固定计算预算下获得更强的表示能力。但代价是：(1) 专家之间的协作机制尚不清晰；(2) 大量专家导致巨大的存储和部署压力；(3) Router 坍塌需要 auxiliary load balancing loss 来缓解。

从算法pipeline角度拆解术语：

MoE 层的计算流程（以 DeepSeek-MoE 为例，normal experts + shared experts）：

```
输入: x in R^{T x d}  # T 个 token 的 hidden states

# Step 1: Router 计算专家分配
logits = x @ W_router                 # W_router in R^{d x (n_normal + n_shared)}
probs = softmax(logits, dim=-1)       # probs in R^{T x (n_normal + n_shared)}

# Step 2: Top-K 选择
topk_probs, topk_indices = top_k(probs[:, :n_normal], k=2)

# Step 3: Shared experts 始终激活
shared_out = sum_{j=1}^{n_shared} SharedExpert_FFN_j(x)

# Step 4: Routed experts 加权输出
routed_out = sum_{i=1}^{k} topk_probs[:,i] * NormalExpert_FFN_{topk_indices[:,i]}(x)

# Step 5: 最终输出
output = shared_out + routed_out + x   # residual connection
```

MoE 模型分析中的关键矩阵——Expert Activation Matrix（专家激活矩阵）：对于 m 层、每层 n 个普通专家的模型（共 $N_e = m \times n$ 个专家），在 $N_s$ 个样本上收集每个 token 的 router 分配权重 $\alpha(i)_{t,j,k}$，按句子聚合为：$v_{i,j,k} = \sum_{t=1}^{T} \alpha(i)_{t,j,k}$，构造 $X \in \mathbb{R}^{N_e \times N_s}$。该矩阵是分析专家协作模式的基础数据。

术语一般如何实现？如何使用？

典型实现：基于 HuggingFace Transformers，MoE 层通过 `MixtralSparseMoeBlock` 或 DeepSeek MoE 模块实现，Router 为 `nn.Linear(hidden_size, num_experts)`。训练时使用辅助负载均衡损失（auxiliary load balancing loss）防止路由坍塌。推理时，MoE 支持专家并行（Expert Parallelism, EP）——将不同专家分布在不同 GPU 上，通过 all-to-all 通信完成 token dispatch 和 combine。

涉及论文标题：
- Unveiling Hidden Collaboration within Mixture-of-Experts in Large Language Models
- Upcycling Large Language Models into Mixture of Experts
- Who Says Elephants Can't Run: Bringing Large Scale MoE Models into Cloud Scale Production
- Uni-MoE Scaling Unified Multimodal LLMs with Mixture of Experts

Uni-MoE 将 MoE 引入统一多模态 LLM 场景，基于 Vicuna-7B（LLaMA-7B），将 LLM 中部分 FFN 层替换为稀疏 MoE 层。每层包含 4~8 个专家（Expert FFN），Router 为线性层 $W \in \mathbb{R}^{d \times M}$，对每个 token 计算 softmax 概率并选择 top-2 专家。配置包括 Uni-MoE-7B×4-Top2（16 层 MoE，4 专家/层，激活 8.9B/总 13.2B）和 Uni-MoE-7B×4-Top2†（32 层 MoE，激活 11.1B/总 19.7B）。Uni-MoE 的特殊之处在于：(1) 每个专家在不同模态数据上分别预训练（阶段二），发展出模态偏好；(2) 使用 LoRA 微调替代全量专家参数更新，rank=8/alpha=16；(3) 支持 expert-level model parallelism 和 modality-level data parallelism；(4) 实验发现 auxiliary balancing loss 在 pure MoE（相同初始专家）中有效，但在 mixture MoE（预训练多样化专家）中不加 aux loss 反而更好——因为专家已自然发展出模态分化。
