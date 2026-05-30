## Expert Selection Prediction with Multi-Feature Bayesian Posterior（基于多特征贝叶斯后验的专家选择预测）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
该术语描述一种在 MoE 推理前预测每个 expert 将处理多少 token 的方法，用于 serverless 平台上预先配置 expert function 的内存大小。传统的 expert 预测方法（如 Lina）仅使用 token ID 作为特征，但论文发现具有相同 token ID 的 token 可能被路由到不同 expert（Fig. 3 所示），因此 token ID 不足以唯一确定 token-to-expert 映射。

该方法设计了三个 token 特征：(1) **token ID (f_1)**：tokenizer 分配的 token 标识，推理前已知；(2) **position ID (f_2)**：token 在输入序列中的位置，假设均匀分布；(3) **attention ID (f_3)**：在每层 self-attention 中与该 token 有最高累积 attention score 的 token 的 token ID，反映 token 间的依赖关系。由于 f_3 在推理前未知，用 f_1 的概率近似 f_3 的概率。

后验概率计算通过 Bayes 定理将三个特征融入：
$$P(N_{e,i}|f_1') = \int_{f_2} \int_{f_3} P^*(N_{e,i}|f_1',f_2,f_3) \cdot \frac{P^*(f_1',f_2,f_3)P'(f_3)}{P^*(f_1',f_2)} \cdot \frac{P^*(f_1',f_2)P'(f_2)}{P^*(f_1')} df_3 df_2$$
其中 P*(·) 是从 profiled data 的 key-value table 计算的概率，P'(·) 是真实请求分布的概率。取 argmax 得预测专家：`î_e = argmax_i P(N_{e,i}|f_1')`。

从算法pipeline角度拆解术语：
专家选择预测在 MoE 推理 pipeline 中的位置和计算流程：

```
// 离线Profiling阶段
for each sample in profiling_dataset (≥100 samples):
    for each MoE layer e:
        for each token t:
            extract features: f1=token_id(t), f2=position(t), f3=attention_id(t)
            observe routing: expert_i = gate_network(t)
            key = (f1, f2, f3, e, i)
            Ω[key] += 1  // 记录token-to-expert映射频次

// 在线预测阶段
for each new token with known f1':
    for each MoE layer e:
        for each expert i:
            // 计算联合概率P*(f1',f2,f3) from key-value table
            // f2: uniform distribution
            // f3: approximated by P(f1)
            compute P(N_{e,i}|f1') via double integral (Eq.1)
        predicted_expert[e] = argmax_i P(N_{e,i}|f1')
    return predicted_expert
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：P*(·) 直接从 key-value table 的频率统计获得；P'(f_2) 取均匀分布 1/seq_len；P'(f_3) 用 P*(f_1) 近似。
- 与 Lina 对比：Lina 只用 token ID 单特征 + MAP 估计；本方法加入 position ID 和 attention ID，在多个模型/数据集上预测差异显著优于 Lina。top-2 routing 时预测准确度进一步提升（另一 expert 可修正预测错误）。
- Profiling 开销：~28.89s（100 个 batch）；预测时间：~20.31s（10 个 batch）。
- 适用场景：serverless 平台等需要提前（部署前）知道 expert 负载的场景；也可用于 CPU/GPU cluster 上的资源预分配。

涉及论文标题：
- Optimizing Distributed Deployment of Mixture-of-Experts Model Inference in Serverless Computing

## Mixture-of-Experts (MoE / 混合专家模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture-of-Experts (MoE) 是一种神经网络架构范式，通过将模型的部分层（典型为 FFN 层）替换为多个并行的 "expert" 子网络，并由一个可学习的 Gate 网络（router）根据每个输入 token 的特征动态选择激活其中少数几个 expert（top-K routing），实现条件计算（conditional computation）。与 densely activated 模型（每个 token 激活全部参数，如 GPT-3）相比，MoE 的参数量可以线性扩展（增加更多 expert），但每个 token 的计算量仅以 sub-linear 方式增长（因只激活 K 个 expert），即"参数规模扩展但计算量近乎不变"。MoE 最早可追溯到 1991 年 Jacobs et al. 的工作，2017 年 Shazeer et al. 的 Sparsely-Gated MoE 首次将 MoE 引入大规模深度神经网络。Switch Transformer (2021) 进一步简化 gating 为 top-1 routing，GLaM (2021) 以 1.2T 参数（每次仅激活 95B 即 8%）验证了 MoE 在大语言模型中的实用价值。

从算法pipeline角度拆解术语：
MoE layer 的计算流程（以 top-K routing 为例，expert = SwiGLU FFN）：
```
# Input: hidden states H [B, d_model]  for B tokens
# Model: E experts, each expert e = {W_gate, W_up, W_down}
# Gate: W_g [d_model, E]

# Step 1: Gating
logits = H @ W_g              # [B, E]  router logits
probs = softmax(logits)        # [B, E]  routing probabilities
weights, indices = top_k(probs, K)  # [B, K]  top-K expert weights & indices

# Step 2: Dispatch tokens to experts
for each token b in 0..B-1:
    for each (w, e) in zip(weights[b], indices[b]):
        dispatch token H[b] to expert e with weight w

# Step 3: Expert computation (SwiGLU FFN)
for each expert e:
    tokens_e = all tokens dispatched to e  # [N_e, d_model]
    gate_out = SiLU(tokens_e @ W_gate[e])  # [N_e, d_ff]
    up_out = tokens_e @ W_up[e]            # [N_e, d_ff]
    expert_out = gate_out * up_out @ W_down[e]  # [N_e, d_model]

# Step 4: Combine (weighted sum)
for each token b:
    output[b] = sum(w * expert_out_e for (w, e) in routing[b])
# 最终: output [B, d_model]
```

关键参数：
- E: 总 expert 数（如 Mixtral-8x7B 有 8 experts）
- K: 每 token 激活的 expert 数（如 Mixtral top-2, Switch Transformer top-1）
- ρ = K/E: 稀疏度（sparsity），越小越稀疏
- α = 每 token 平均激活的 expert 比例（受 batch size 和 routing 分布影响，≤K）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练挑战：(1) load imbalance — 某些 expert 被过多 token 选择导致过载（hot expert），需 auxiliary loss 鼓励均匀 routing；(2) communication — Expert Parallelism 下需 AlltoAll 通信交换 dispatch/combine token 数据，Switch Transformer 每层需 4 次 AlltoAll；(3) memory — 全部 expert 参数的天量存储需求，需 Hierarchical Storage 或 offloading。
- 推理挑战：(1) memory footprint 大（所有 expert 参数需常驻或可访问），但 batch 小时仅少量 expert 被激活；(2) expert 选择的不确定性使 offloading/prefetching 复杂化。
- 主流 MoE 模型：Mixtral-8x7B (8 experts, top-2), Qwen-MoE, DeepSeek-V2/V3, Phi-3.5-MoE, GLaM, Switch Transformer。
- 训练框架：DeepSpeed-MoE、FastMoE、Tutel、Megatron-LM (expert parallelism)、PaddleFleetX + MoESys。

- OEA 论文的补充：MoE decode 阶段的 memory-bound 问题。在 decode 阶段，batch 中每个 token 激活 k 个 expert（共 N 个），平均每 expert 负载仅以 k/N 速度增长（如 Qwen3 的 k=8, N=128，速率仅 1/16）。batch size 为 B 时，唯一激活 expert 数 T 的期望值为 N(1-(1-k/N)^B)。对于 Qwen3，B=16 时 E[T]≈82（vs B=1 时 T=8），增长约 10×。MoE 层延迟 ≈ b·T + a·Bk，在 memory-bound 下 b·T 主导。因此减少 T 是降低 decode 延迟的主要优化目标。

- 补充：2017 Shazeer et al. 的开创性工作。首次将 MoE 引入大规模深度神经网络（之前工作限于小规模图像任务），提出 Sparsely-Gated MoE Layer 并验证在语言建模和机器翻译上的效果。核心创新：(1) Noisy Top-K Gating — 在 Softmax 前加可调高斯噪声（噪声幅度由 W_noise 控制），再 KeepTopK 保留 k 个最大值并抑制其余（设 -∞），实现稀疏专家选择；(2) Convolutional MoE Application — 在 stacked LSTM 之间插入 MoE 层，等前一层所有时间步完成后再对所有时间步一次性应用 MoE，将 seq_len 折叠进 batch dim；(3) Mixed Data+Model Parallelism — 标准层+Gate 网络用数据并行，各 expert 仅保留一份共享副本（模型并行），同一设备双重角色；(4) Two-Level Load Balancing — L_importance = w·CV(Importance)² 和 L_load = w·CV(Load)²（Load 为平滑可微估计器，基于噪声和 CDF）；(5) Hierarchical MoE — 两级门控降低 branching factor。实验在 Tesla K40 GPU (16-128 卡) 上，最大模型含 137B 参数 (131072 experts)，在 1 Billion Word LM 和 100 Billion Word News Corpus 上分别比同计算量 baseline 降低 24% 和 39% Perplexity。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

## MoE Gate Network / Expert Routing (门控网络 / 专家路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gate Network（亦称 Router）是 MoE 模型中决定每个输入 token 被分配到哪些 expert 的核心组件。通常是一个简单的线性层 W_g [d_model, E]（E 为 expert 总数），将 hidden states 映射到 E 维 logits → softmax 得到每个 expert 的 routing probability → top-K 选择概率最高的 K 个 expert → 输出每个选中 expert 的权重和 index。Gate 网络的设计直接影响 MoE 的负载均衡、模型质量和通信模式。

从算法pipeline角度拆解术语：
Gate Network 在 MoESys 训练中的决策链路：
```
# Input: hidden states H [B, d_model]
# Gate weight: W_g [d_model, E]

# Forward:
logits = H @ W_g                     # [B, E]
probs = softmax(logits)               # [B, E]
weights, indices = top_k(probs, K)    # [B, K], [B, K]

# Auxiliary Loss (load balancing, e.g. Switch Transformer):
# f_e = fraction of tokens routed to expert e
# P_e = mean routing probability for expert e
# Loss_aux = E * sum(f_e * P_e)  # encourages uniform routing

# AlltoAll dispatch:
# Each GPU sends token H[b] to GPU hosting expert indices[b][k]
# → triggers 2× AlltoAll (fwd) + 2× AlltoAll (bwd) per MoE layer
```

MoESys 对 Gate 的利用：Gate 的 expert 选择结果在 AlltoAll 通信中自然可获得 → MoESys 的 2D Prefetch 利用此结果决定 sparse 参数的 prefetch 目标，无需额外通信。hash table 中记录的 hits 频率 = 各 expert 被激活的历史频率 → CPU cache 的 LFU 管理依据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 常见 routing 变体：top-1 (Switch Transformer, 简化但需更多 expert 或更高 capacity factor)、top-2 (GShard, Mixtral, 平衡负载和计算)、noisy top-K (添加 Gaussian noise 到 logits 鼓励探索)、random routing (ST-MoE, 随机选 expert 减少 bias)。
- Capacity Factor (CF)：限制每个 expert 能处理的最大 token 数 = CF × (tokens_per_batch / num_experts)。超出 capacity 的 token 被"dropped"（不经过该 expert，由 residual connection 绕过）。CF 引入 trade-off：CF 小 → 更多 dropped tokens → 质量下降；CF 大 → 更多 computation 和 memory → 效率下降。
- Auxiliary Loss 的类型：(1) Load balancing loss (Switch Transformer)：L_aux = E·Σ(f_e·P_e)；(2) Z-loss (ST-MoE)：加在 logits 上防止数值溢出。

- 补充 — Noisy Top-K Gating (Shazeer et al. 2017)：在标准 top-K routing 基础上添加两个关键组件。(1) **Tunable Gaussian Noise**：H(x)_i = (x·W_g)_i + StandardNormal()·Softplus((x·W_noise)_i)，噪声幅度由第二个可训练矩阵 W_noise 控制。噪声在训练中提供随机性，防止 Gate 过早收敛到固定 expert；噪声同时使负载均衡损失可微——P(x,i) = Φ((clean_logits_i - kth_excluding(H,k,i)) / noise_std_i)，其中 Φ 是标准正态 CDF，由此构建平滑的 Load(X) 估计器。(2) **KeepTopK**：保留 H(x) 中最大的 k 个值，其余设 -∞，经 Softmax 后对应 gate 值为 0，实现精确稀疏。与后来的 Switch Transformer (top-1) 和 GShard (top-2 + capacity factor) 不同，该论文使用 k=4（LM）或 k=2×2（hierarchical MT），且未使用 capacity factor（因 k 固定且 load balancing loss 已足够均衡）。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## Embedding Partition in Data Parallelism (数据并行中的Embedding分区)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Embedding Partition in Data Parallelism 是 MoESys 提出的在 Data Parallelism 框架下对大 vocabulary embedding table 做分布式存储和计算的技术。传统 tensor model parallelism 的 embedding 分区沿 vocabulary 维度切分（每个 GPU 持有 V/N 个 token 的 embedding），但在 Data Parallelism 下每个 GPU 处理不同的输入数据，沿 vocab 维度切分会导致部分 token 在本 GPU 没有对应 embedding。MoESys 的方法改为沿 embedding 的 hidden_size 维度做 column-wise partition：每个 worker 持有 [V, H/N] 的 embedding shard（完整 vocabulary 但部分 hidden dimension），这样每个 device 都能访问完整 vocabulary。计算时通过 3 次 AlltoAll 通信完成 embedding lookup：Forward 阶段 AlltoAll 交换 input data → 本地 lookup → AlltoAll 交换结果；Backward 阶段 AlltoAll 交换 gradients。

从算法pipeline角度拆解术语：
Embedding Partition in Data Parallelism 的计算过程（embedding table E[V, H]，N 个 devices）：
```
# 列切分: 每个 device i 持有 E_i[V, H/N]
# 即: E_i 是 E 的第 i 个列切片

# Forward:
# Step 1: AlltoAll 交换 input token IDs
# 每个 device 有 batch token IDs: ids_local[batch_size]
all_ids = AlltoAll(ids_local)  # 每个 device 获得所有 input token IDs

# Step 2: 本地 embedding lookup
for each token_id in all_ids:
    embed_partial = E_i[token_id]  # 形状 [H/N]，仅部分 hidden dim
    embed_results.append(embed_partial)

# Step 3: AlltoAll 交换 partial embedding 结果 (AlltoAll inverse)
# 将归属于原始 device 的 embedding 结果返回
final_embeddings = AlltoAll(embed_results)  # [batch_size, H]

# Backward:
# 只需 1 次 AlltoAll 交换 gradients of E
grads_E_i = backward_pass()
AlltoAll(grads_E_i)  # 交换梯度恢复完整 embedding table gradient
```

关键差异对比：
- 传统 tensor parallelism embedding partition: 沿 vocab 维度切分 → 每 GPU 有 V/N 个 token 的完整 embedding → 需要 AllReduce 同步
- MoESys DP embedding partition: 沿 hidden_size 维度切分 → 每 GPU 有全部 V 个 token 的部分 embedding → 需要 3 次 AlltoAll（替代 AllReduce）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 MoESys 实验中，embedding partition 在 vocab_size=50304 的 GPT MoE 模型上表现突出：4 experts, hidden=8192 时 memory 从 15.81GB 降至 8.63GB，speed 从 80421 tokens/s 升至 91687 tokens/s。
- 该方法的关键优势：在 DP 框架下（每个 device 处理不同 data），vocab 维度切分不可行（每个 device 的 input token 不同），hidden_size 维度切分避免了这一矛盾。
- 相比 EmbRace 的列切分（主要用于 tensor parallelism，针对的是通信均衡）和传统 DP 的 AllReduce embedding 同步，该方法的 3 次 AlltoAll 虽然引入更多通信次数，但避免了 AllReduce 的全局同步开销，在总通信量相近的情况下更灵活。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

## Knowledge Distillation for MoE Inference (MoE推理中的知识蒸馏 / Mixture-of-Students)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Knowledge Distillation for MoE Inference 是将知识蒸馏技术应用于 MoE 模型推理阶段，以在保持推理精度的前提下显著减少模型大小。MoE 模型虽然在训练时通过稀疏激活降低了计算量，但 inference 时需要存储所有 expert 参数（包括未被激活的 expert），参数量远超同规模的 dense 模型，导致显存压力和部署成本高。MoESys 在 Graph Optimization Pipeline 的 Distillation & Compression 步骤中，将 teacher MoE（含大量 expert）蒸馏为 student MoE（含少量 expert），通过 DeepSpeed 提出的 Mixture-of-Students (MoS) 架构提升 student 模型的精度。蒸馏后，student 模型在推理时仅需激活更少的 expert。

从算法pipeline角度拆解术语：
MoESys 中 MoE distillation 的 pipeline 流程：
```
# Teacher: MoE model with E_t experts
# Student: MoE model with E_s experts (E_s << E_t)

# Step 1: Teacher inference on training data
for batch in distillation_dataset:
    teacher_logits = teacher_moe(batch)  # E_t experts, top-K gating

# Step 2: Student training with distillation loss
for batch in distillation_dataset:
    student_logits = student_moe(batch)  # E_s experts
    teacher_logits = teacher_moe(batch)  # pre-computed or on-the-fly
    loss = CE(student_logits, labels) + λ * KL_div(student_logits, teacher_logits)
    
# Step 3: Deploy student for inference
deploy(student_moe)  # 更少的 expert, 更低的 memory 和更快推理
```

MoS (Mixture-of-Students) 的关键改进：传统 KD 用单一 student 模仿 teacher，MoS 用 multiple students（每个 student 是一个子 expert 组）联合学习，student 间通过 gating 机制分工，提升蒸馏后的模型容量和精度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoESys 的蒸馏步骤作为 Graph Optimization Pipeline 的一部分离线执行，在模型部署前完成。
- 类似思路在 GLaM、DeepSpeed-MoE 中也有应用——通过训练阶段的 sparsity 和推理阶段的压缩/蒸馏，在整个 model lifecycle 中保持效率。
- KD 的 trade-off：减少 expert 数会降低模型容量和表达能力，需要通过精心设计的蒸馏策略（如 MoS、task-specific distillation）和量化/剪枝结合来弥补。
- MoESys 论文未详细给出蒸馏的实验结果对比，该方法作为 inference pipeline 中减少模型大小的一个步骤提及。

涉及论文标题：
- MoESys: A Distributed and Efficient Mixture-of-Experts Training and Inference System for Internet Services

## Sub-Expert Decomposition (Expert Partitioning for MoE Elasticity / 子专家分解)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Sub-Expert Decomposition 是 MoE-Prism 提出的后训练（post-training）模型重构技术，将预训练 MoE 模型中每个 monolithic expert（完整 FFN）分解为 N 个细粒度、功能内聚的 "sub-expert"。核心思想基于两个观察：(1) MoE 中每个 monolithic expert 内部存在显著的激活稀疏性——对任意输入 token，expert 内 50% 的 neuron 激活幅度低于 0.0167，75% 低于 0.0391；(2) SwiGLU FFN 中不同列（即不同 neuron）的计算是独立的，因此可以将 neuron 重新分组。分解过程分三步：Neuron Activation Profiler 从校准数据集收集激活矩阵 M(B×C)；Partitioning Optimization Solver 以最小化被停用 sub-expert 的 L1 范数之和为目标，用贪心初始化 + Simulated Annealing (T0=100, α=0.995, 100K 迭代) 求解最优分区 P*；Gating Mechanism Reconstructor 构建新的细粒度路由机制。每个原 expert 划分为 N=4 个子 expert 后，激活控制粒度提升 4 倍，将 MoE 的 "Quality Cliff" 转化为平滑的 cost-quality 权衡曲线。

从算法pipeline角度拆解术语：
MoE-Prism 对每个 MoE layer 中每个 expert 的分解流程：
```
# Step 1: Profiling
for each expert e in MoE_layer:
    M_e = []  # B x C activation matrix
    for token batch in calibration_dataset:
        H = input_hidden_states  # [B, d_model]
        A_gate = SiLU(H @ W_gate)  # [B, C]
        A_up = H @ W_up            # [B, C]
        A = A_gate * A_up           # [B, C], element-wise
        M_e.append(A)

# Step 2: Partition Optimization (SA solver)
def simulated_annealing_partition(M, N_sub_experts, T0=100, alpha=0.995, I=100000):
    P = greedy_init(M, N_sub_experts)  # 按impact降序贪心分配, 维护负载均衡
    T = T0
    best_P, best_cost = P, compute_cost(P, M)
    for i in range(I):
        P_new = swap_random_neurons(P)  # 随机交换两neuron所属sub-expert
        cost_new = compute_cost(P_new, M)
        if cost_new < best_cost or random() < exp((best_cost - cost_new) / T):
            P = P_new
            if cost_new < best_cost:
                best_P, best_cost = P_new, cost_new
        T *= alpha
    return best_P  # {S_1: [neuron_ids], ..., S_N: [neuron_ids]}

# Cost function: sum of L1 norms of K deactivated sub-experts
def compute_cost(P, M):
    cost = 0
    for b in range(B):
        L_b = [||M[b, S_n]||_1 for S_n in P]  # per-sub-expert L1 norms
        cost += sum(smallest_K(L_b))  # top-K smallest norms
    return cost

# Step 3: Gating Reconstruction
C_co = B.T @ B  # co-activation matrix, B = binary_top_k(M)
for S_n in P:  # for each sub-expert
    centrality[n] = sum(C_co[n, j] for j in S_n)  # neuron centrality
    gate_neurons[S_n] = top_r(centrality[S_n])  # r=4 representative neurons
```
流程：校准数据前向传播收集激活→SA 求解器将每个 expert 的 C 个 neuron 划分到 N 个子 expert→选择每个子 expert 中 centrality 最高的 gate neurons→可选微调 router。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 论文在 PyTorch 2.7.0 + CUDA 12.6 上实现。校准使用 Wikitext-2-raw-v1，SA 参数 T0=100, α=0.995, I=100K 迭代。每个 expert 分为 N=4 sub-experts。
- 相关方法：DualSparse-MoE (2025) 也使用 post-training expert partitioning，但侧重于 tensor-level 和 neuron-level 双重稀疏性，在 ~25% drop rate 下仅损失 0.08%-0.28% 准确率。DERN (2025) 通过 expert 剪枝后分解为 neuron-level expert segment 再合并，在 50% expert sparsity 下提升 5% 推理性能。
- 核心价值：使 MoE 模型从 coarse-grained（如 k 只能选 1-2 个整数）升级为 fine-grained（如 k' 可以是 9-32，等效于原模型的 2.25-8 个 expert），提供 4 倍以上可区分操作点。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---

## Co-activation Matrix & Gate Neurons (共激活矩阵与门神经元)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Co-activation Matrix 是 MoE-Prism 中用于量化 MoE expert 内部 neuron 间功能相似性的矩阵 C_co ∈ R^{C×C}（C 为 neuron 总数），定义为 C_co = B^T·B，其中 B ∈ {0,1}^{B×C} 是二值化激活矩阵——对每个 token，激活幅度处于 top-k_a 的 neuron 标记为 1（活跃），其余为 0。C_co[i,j] 表示 neuron i 和 j 在同一 token 下同时活跃的 token 总数，用作功能共激活频率的度量。Gate Neurons 是从每个 sub-expert 中选出的 r 个最具代表性的 neuron，作为该 sub-expert 的"功能中心"——通过计算每个 neuron 在该 sub-expert 内与其它 neuron 的累积共激活次数（centrality），取 top-r 得最高 centrality 的 neuron。Gate neuron 的激活 L1 范数被用作该 sub-expert 整体输出范数的轻量级代理估计，使 router 无需执行所有 sub-expert 即可判断哪些 sub-expert 对当前 token 最有用。

从算法pipeline角度拆解术语：
```
# Co-activation matrix construction
M = activation_matrix  # [B, C], from Neuron Activation Profiler
k_a = C * 3 // 4  # top 3/4 neurons considered "active"
B = zeros(B, C)  # binary activation matrix
for t in range(B):
    threshold = top_k(|M[t,:]|, k_a)
    B[t,:] = (|M[t,:]| >= threshold).astype(int)

C_co = B.T @ B  # [C, C], symmetric

# Gate neuron selection for sub-expert S_n
centrality = []
for neuron_i in S_n:
    co_sum = sum(C_co[neuron_i, neuron_j] for neuron_j in S_n)
    centrality.append((neuron_i, co_sum))
gate_neurons_S_n = top_r(centrality, r=4)  # by centrality desc
```
匹配案例：在 NAACL 2025 论文 "A Closer Look into Mixture-of-Experts in LLMs" 中，研究确认 MoE 中 neuron 行为类似 fine-grained expert，gate projection matrix 负责选择哪些 neuron 激活，且不同 expert 间的 neuron co-activation 模式在深层有更大的多样性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 中 r=4（每个 sub-expert 选 4 个 gate neuron），使用 top 3/4 的激活 neuron 构建 B 矩阵，在 Wikitext-2-raw-v1 校准集上计算 C_co。
- 推理时仅需计算 gate neurons 的中间激活（O(r·B·d_model) vs 完整 sub-expert 的 O(C·B·d_model)），用 gate neuron 的平均 L1 norm 作为代理分数，router 据此选 top-k sub-experts。
- 这一机制是 training-free proxy gating 的基础，无需微调即可实现有效路由，PPL 接近原始模型。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---

## Proxy Gating (代理门控)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Proxy Gating 是 MoE-Prism 提出的一种无需训练的门控机制重建策略。在将 monolithic expert 分解为 N 个子 expert 后，原始 router（为原 expert 设计）失效。朴素的解决方案是对每个 token 执行所有子 expert 以计算其输出范数再做选择，但这完全抵消了细粒度分解的性能收益。Proxy Gating 的解决方案：从每个子 expert 中选择 r 个 gate neuron（通过 co-activation matrix 和 centrality 排序选出），推理时仅计算 gate neurons 的激活，用这些 neuron 的平均 L1 范数作为整个子 expert 输出贡献的廉价代理估计。由于 gate neuron 是其子 expert 的"功能中心"（与子 expert 内其他 neuron 共激活频率最高），其激活模式能有效代表整个子 expert 的行为。

从算法pipeline角度拆解术语：
```
# Inference with Proxy Gating
h = input_hidden_state  # [d_model]
# 仅计算gate neurons的中间激活 (极低成本)
for each sub_expert S_n:
    gate_h = h @ W_gate[:, gate_neurons[S_n]]  # [r]
    gate_up = h @ W_up[:, gate_neurons[S_n]]    # [r]
    proxy_score[S_n] = mean(|SiLU(gate_h) * gate_up|)  # avg L1 norm

# Router选择
top_k_sub_experts = top_k(proxy_score, k)
# 仅执行选中的sub-experts (完整前向)
output = sum(softmax(proxy_score[n]) * execute_sub_expert(S_n, h) 
             for n in top_k_sub_experts)
```
对比 baseline：原始 MoE router 对 N 个 expert 输出 N 维 logits→top-k 选择。Proxy Gating 的等效操作：对每个子 expert 仅计算 r 个 gate neuron 的 L1 范数（O(r·d_model)），相比执行完整子 expert（O(C·d_model)），gate neuron 的开销可忽略不计（r ≪ C）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 中与 training-free Linear Gate (LG w/o FT) 结合使用。Proxy gating 的 proxy score 输入到新的 linear router（Linear(d_model, N_sub_experts)），router 按 softmax 归一化后选 top-k。
- 相关技术：Confidence-Guided Gate (2025) 用 token-level confidence 替代 softmax routing 解决 expert collapse；ASMG (2025) 用 Generalized Hebbian Algorithm 学习 adaptive routing subspace。
- Proxy Gating 是 training-free 的，适合快速部署。若追求最大保真度，可搭配 Low-cost Router Finetuning（仅微调 router，冻结 99.9%+ 参数）。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---

## SwiGLU FFN in MoE (MoE中的SwiGLU前馈网络)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SwiGLU (Swish-Gated Linear Unit) 是现代 LLM（包括 MoE 模型）中最常用的 FFN 激活函数。其计算定义为 FFN(X) = (SiLU(X·W_gate) ⊙ (X·W_up))·W_down，其中 SiLU(x) = x·σ(x)（σ 为 sigmoid）。W_gate, W_up ∈ R^{d_model × d_intermediate} 是两个上投影矩阵，W_down ∈ R^{d_intermediate × d_model} 是下投影矩阵，⊙ 是逐元素乘法。与传统 ReLU-FFN（两层 MLP + ReLU）相比，SwiGLU 用门控机制（gate 通道 + up 通道的逐元素乘）替代简单非线性，提供更好的训练稳定性和模型质量。

从算法pipeline角度拆解术语：
MoE-Prism 利用 SwiGLU 的一个关键数学性质——列独立性——来实现 expert 分解：
```
# SwiGLU FFN 前向计算
X = input  # [B, d_model]
A_gate = X @ W_gate  # [B, C], C = intermediate_size
A_up = X @ W_up      # [B, C]
A = SiLU(A_gate) * A_up  # [B, C], element-wise, 每列独立计算
output = A @ W_down  # [B, d_model]

# 列独立性: output_j 仅依赖于 A[:, j] 和 W_down[j, :]
# 按列分组 = 按neuron分组, 将FFN分解为子专家
for sub_expert S_n containing neurons [j1, j2, ...]:
    A_n = A[:, [j1, j2, ...]]           # 仅相关列
    output_n = A_n @ W_down[[j1, j2, ...], :]  # 仅相关行
    # expert总输出 = sum(output_n for all sub_experts)
```
这个列独立性意味着：每个 neuron（W_gate 的一列 + W_up 的一列 + W_down 的一行）可以独立计算其对最终输出的贡献，因此 SwiGLU FFN 天然可分解为 neuron 组（子 expert），且总输出是各组输出的精确求和。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- SwiGLU 由 Shazeer (2020) 在 "GLU Variants Improve Transformer" 中推广，现已是 LLaMA、Mixtral、DeepSeek、Qwen 等主流 LLM 的默认 FFN 激活。标准实现中 intermediate_size 通常为 d_model 的 8/3 到 4 倍。
- MoE-Prism 中使用的模型 intermediate_size：OLMoE-1B-7B: 1024→256 (per sub-expert)、DeepSeek-V2-Lite: 1408→352、Qwen3-30B-A3B: 6144→1536。
- 列独立性是 MoE-Prism Sub-Expert Decomposition 在数学上保持输出恒等性（不改变 FFN 输出值）的根本保证，无需微调即可重构 expert。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 从训练内存角度分析了 SwiGLU 的内存瓶颈——SwiGLU 需要两次投影（a = x·W1, b = x·W2）→ 逐元素 SiLU(a)·b → W3 下投影，传统 kernel 需在 HBM 中保存 a, b, σ(a), SiLU(a), y_swi 等 5 个中间张量用于反向传播，单个 MoE 层的中间激活可达约 98GB（DeepSeek 规模：L≈2M, h=24576, bf16）。MoEBlaze 提出 fused SwiGLU training kernel：将 W1/W2 两个 GEMM 融合为单 kernel，同时在 register/shared memory 中计算 SiLU 和 element-wise multiply；反向传播时采用 activation checkpoint——不保存 SiLU(a)，仅保存 a, b, y_swi，backward 时 recompute SiLU(a)（element-wise 操作，memory bandwidth bound，recompute 开销极低）。此方法在 SwiGLU 下实现最高 4× 激活内存减少（conf3: 40GB→10GB），训练速度提升 2×–6.2×。

---

## Curriculum-based Router Finetuning for MoE (基于课程学习的MoE路由器微调)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Curriculum-based Router Finetuning 是 MoE-Prism 提出的可选低成本微调策略，用于在 sub-expert 分解后提升 router 的 routing 质量。核心思路：(1) **参数效率极高**：仅微调 gating network/router（线性层，占总参数 <0.1%），冻结所有其他权重（expert FFN、attention、LayerNorm 等）；(2) **课程学习**：训练过程中逐步增加激活 sub-expert 数量 k（如从 k=8 递增到 k=24/32），而非固定 k 值。这使 router 学会在不同资源预算下做出高质量的 routing 决策，而非仅针对单一 k 值过拟合。

从算法pipeline角度拆解术语：
```
# Curriculum Router Finetuning in MoE-Prism
router = Linear(d_model, N_sub_experts)  # 仅此参数可训练
k_min, k_max = 8, 24  # 或 8→32
total_steps = len(dataloader) * epochs
# 冻结所有其他参数
for param in model.parameters():
    param.requires_grad = False
router.weight.requires_grad = True

for step, batch in enumerate(dataloader):
    # 课程调度: k随训练进度线性递增
    k_current = k_min + (k_max - k_min) * (step / total_steps)
    h = model.forward_to_router(batch)  # 冻结部分前向
    router_logits = router(h)  # [B, N_sub_experts]
    top_k_idx, top_k_probs = top_k(softmax(router_logits), k_current)
    # 仅选中的sub-experts计算输出
    output = weighted_sum(probs * sub_expert_ffn(h) for ...)
    loss = cross_entropy(output, labels)
    loss.backward()  # 仅更新router参数
```
关键设计：与标准 MoE 微调（固定 k）不同，课程训练使 router 暴露于多种 k 值，学习到在不同计算预算下的灵活 routing 策略。这与 Chen et al. (2023) "Sparse MoE as the New Dropout" 的渐进式训练理念一致。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 实现：LR=1e-5，训练集为 SlimPajama 的 200K 序列，batch_size=32（Deepseek/Qwen）或 64（OLMoE），k 从 8 线性递增到 24（Deepseek）或 32（OLMoE/Qwen）。
- 微调后 PPL 通常优于原始模型（如 OLMoE K=12: 原模型 15.72, LG w/FT 14.68），且下游任务（Winogrande, ARC-C, SciQ, BoolQ）保持或提升。
- 这一策略与全参数微调或 LoRA 等常见方法不同：它仅微调一个线性层的参数，比 LoRA（通常加 adapter 到 attention + FFN）更轻量。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---

## Residency-Aware Thresholding (MoE Expert Selection / 驻留感知阈值路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Residency-Aware Thresholding 是 MoE-ERAS 提出的第一种 residency-aware expert 选择算法。在标准 MoE gating 的 Softmax 之后、Top-K 选择之前，对已驻留在 GPU HBM（fast memory）的 expert 的概率值统一加上超参数 α，人工提升 on-chip expert 的激活竞争力。核心洞察：gating network 的输出并非总有"绝对赢家"——有时 top-1 expert 仅略微优于第二名。若 top-1 恰好在 CPU 中而第二名在 HBM 中，thresholding 的 α 偏置可以使 on-chip 的第二名在调整后的概率中超过 off-chip 的第一名，从而避免一次 costly 的 CPU→GPU 传输。

从算法pipeline角度拆解术语：
```
# 标准 MoE Gating
Logits = H_i @ W_exp               # [seq_len, num_experts]
Weights = Softmax(Logits)           # [seq_len, num_experts], sum=1 per token

# === Residency-Aware Thresholding ===
# residency[e] = True 表示 expert e 当前在 HBM 中
for e in range(num_experts):
    if residency[e]:
        Weights[:, e] += alpha       # on-chip expert 加 α 偏置

Selected = SelectTopK(Weights, k=2)  # 调整后概率的 Top-K
# 注意：Weights 加 α 后可能 sum > 1，但 Top-K 仅需相对排序，不影响选择
```
超参数 α ∈ {0.05, 0.15, 0.25}。α=0 退化为标准 Top-K routing。α 越大，on-chip expert 越容易被选中 → speedup 越大 → quality 下降越多。

从算法pipeline角度拆解术语，给出具体例子：
以 Mixtral-8x7B layer i，8 experts，当前 3 个 offloaded（expert 0, 3, 7 在 CPU，其余在 HBM）为例：
- Token hidden state 经 gating 得 logits = [2.1, 1.9, 1.5, 2.0, 0.8, 0.5, 1.2, 0.3]
- Softmax → weights = [0.22, 0.18, 0.12, 0.20, 0.06, 0.05, 0.10, 0.07]
- Standard Top-2: expert 0 (权重 0.22, CPU) + expert 3 (权重 0.20, CPU) → 两个都在 CPU！两次传输。
- Thresholding α=0.15: expert 0 在 CPU 不加 (0.22), expert 1 在 HBM 加至 0.33, expert 2 在 HBM 加至 0.27, expert 3 在 CPU 不加 (0.20), expert 4+α=0.21, expert 5+α=0.20, expert 6+α=0.25, expert 7 在 CPU 不加 (0.07)
- 调整后 Top-2: expert 1 (0.33, HBM) + expert 2 (0.27, HBM) → 零传输开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 serving 框架的 gating 输出后插入 residency lookup + 条件加法操作。residency table 由 LRU cache manager 维护。
- 使用：用户通过 α 控制 speedup-quality trade-off。论文实测 α=0.05→perplexity 几乎不变（C4-PPL 8.044→8.062），α=0.15→10-13% latency reduction，α=0.25→更大 speedup 但 PPL 退化至 8.522。
- 优势：实现极简（仅条件加法），零额外参数，推理时生效不改变模型权重。
- 局限：对 all experts 的 on-chip 概率加相同 α，不考虑 expert 热度差异——冷门 on-chip expert 和热门 on-chip expert 获得相同 boost。

涉及论文标题：
- MoE-ERAS: Expert Residency Aware Selection

---

## Residency-Aware Biasing (MoE Expert Selection / 驻留感知偏置路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Residency-Aware Biasing 是 MoE-ERAS 提出的第二种 residency-aware expert 选择算法，是对 thresholding 的改进。在 Softmax 之前对 off-chip expert 的 logits 施加基于激活频率的差异化惩罚：penalty = β(1 - freq(E_i))，其中 freq(E_i) 是从 profiling 阶段（500k tokens）收集的归一化激活频率。核心洞察：(1) 不同 expert 的"热度"差异显著——hot expert 频繁被激活，cold expert 很少被激活；(2) 加载一个冷门 off-chip expert 到 HBM 大概率很快被 LRU evict（两次 swap），比加载热门 off-chip expert 代价更大。Biasing 通过频率加权的惩罚体现这一差异——冷门 expert 惩罚大（几乎不会被选中加载），热门 expert 惩罚小（值得加载因为后续会复用）。

从算法pipeline角度拆解术语：
```
# 标准 MoE Gating
Logits = H_i @ W_exp               # [seq_len, num_experts]

# === Residency-Aware Biasing ===
# freq[e] ∈ [0, 1]: 归一化激活频率（从 profiling 收集）
# residency[e] = True 表示 expert e 在 HBM 中
for e in range(num_experts):
    if not residency[e]:  # expert 在 slow memory (CPU)
        Logits[:, e] -= beta * (1 - freq[e])

Weights = Softmax(Logits)           # 惩罚后重新归一化
Selected = SelectTopK(Weights, k=2)
```
关键与 thresholding 的区别：(1) 操作在 logits 层面（Softmax 前），而非 probabilities 层面；(2) 惩罚是差异化的（考虑 freq），而非统一 α；(3) 热门 off-chip expert（freq ≈ 0.8, penalty ≈ 0.2β）惩罚小，冷门 off-chip expert（freq ≈ 0.05, penalty ≈ 0.95β）惩罚大。

从算法pipeline角度拆解术语，给出具体例子：
以 Mixtral-8x7B layer i，profiling 得 freq = [0.20, 0.18, 0.05, 0.22, 0.08, 0.15, 0.10, 0.02]，当前 expert 0, 3, 7 在 CPU，β=1.0：
- Logits = [2.1, 1.9, 1.5, 2.0, 0.8, 0.5, 1.2, 0.3]
- Biasing 调整：
  - expert 0 (CPU, freq=0.20): Logits -= 1.0×(1-0.20) = Logits[0] -= 0.80 → 1.30
  - expert 3 (CPU, freq=0.22): Logits -= 1.0×(1-0.22) = Logits[3] -= 0.78 → 1.22
  - expert 7 (CPU, freq=0.02): Logits -= 1.0×(1-0.02) = Logits[7] -= 0.98 → -0.68
  - HBM experts (1,2,4,5,6): Logits 不变
- 调整后 Logits = [1.30, 1.9, 1.5, 1.22, 0.8, 0.5, 1.2, -0.68]
- Softmax 后 Top-2 可能为 expert 1 (HBM) + expert 2 (HBM)，避免传输。
- 对比 thresholding：热门 CPU expert 0 (freq=0.20) 惩罚较小，仍可能被选中（若其 logit 显著高于 HBM expert）；冷门 CPU expert 7 (freq=0.02) 几乎不可能被选中。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Profiling 依赖性：需在 serving 前运行 profiling（CNN DailyMail, 139k-500k tokens）收集每层每个 expert 的激活频率。论文用线性回归验证 expert activation predictability（仅用前 4 层预测后 28 层，accuracy > 50%，远超随机 12.5%）。
- 实现：在 serving 框架中维护 freq 查找表（从 profiling 结果加载）。每个 MoE layer 的 gating→TopK 之间插入 biasing 逻辑。
- 使用：β=1.0 时减少 8.0-9.7% 解码延迟。Quality trade-off 优于同 speedup 水平的 thresholding——因为 biasing 的差异化惩罚更智能。
- 局限性：(1) 依赖 profiling 数据质量，distribution shift 可能导致 freq 不准确（论文建议 periodic re-calibration）；(2) 当前仅在 Mixtral-8x7B 上实现。

涉及论文标题：
- MoE-ERAS: Expert Residency Aware Selection

---

## Checkpoint Recycling（检查点回收）

术语是什么？
Checkpoint Recycling 是 MoE Jetpack 框架的核心组件，一种将预训练 Dense 模型（predecessor）的权重直接转化为 MoE 模型（successor）中各 Expert 初始权重的方法。它不复制整个 MLP，而是从前驱模型的 MLP 权重中通过重要性采样等策略选择性地提取 weight 子集来构造 expert，从而将 dense checkpoint 的预训练知识注入 MoE 模型，避免从零训练 MoE 的高昂开销。该过程为一次性离线操作（在 RTX 4090 上对 30000 张图像进行 activation profiling 仅需约 5 分钟）。

从算法pipeline角度拆解术语：
给定 predecessor dense 模型 P（N 层、channel dim d、hidden dim 4d），目标 successor MoE 模型 S（N 层、channel dim d' ≤ d、前半 N/2 dense layers + 后半 N/2 SpheroMoE layers）。Importance-Based Weight Sampling（默认策略）流程：

```
# Step 1: Activation Profiling
images = sample_batch(dataset)        # 一批图像通过 predecessor
for layer l in P:
    A_c[l] = activation_of_channel(c)  # 每层每个 channel 的 activation
    A_h[l] = activation_of_neuron(h)   # 每层每个 hidden neuron 的 activation

# Step 2: Channel Selection（跨层平均，选 top-d'）
for channel c in [0..d-1]:
    A_c = mean([A_c[l] for l in range(N)])
selected_channels = top_k(A_c, d')     # activation 最高的 d' 个 channel

# Step 3: Neuron Sampling（按 activation 概率分布为每个 expert 采样）
P(h|H) = A_h / sum(all_A_h)           # activation → 概率分布
for each expert e:
    expert_neurons[e] = sample(P(h|H), 4d')  # 采样不同 neuron 保证 expert 多样性

# Step 4: Weight Extraction
for each expert e:
    W_expert[e] = W_predecessor[selected_channels][expert_neurons[e]]
```

其他策略：(a) Co-Activation Graph Partitioning：构造 neuron 共激活图，用 Metis 图分割将频繁共激活的 neuron 分入同一 expert；(b) Uniform Selection：等距采样；(c) Random Sampling：随机选。

术语一般如何实现？如何使用？
在 PyTorch 中实现：加载 timm 预训练 dense checkpoint（如 ViT-S/16 ImageNet-21k），通过 forward hook 捕获中间层 activation 值，按策略索引提取权重子矩阵构造 expert 的 Linear 层权重，保存为 MoE checkpoint。使用时作为 MoE 模型初始权重加载，然后执行标准 fine-tuning。开源实现见 https://github.com/Adlith/MoE-Jetpack。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

## SpheroMoE Layer（超球面自适应 MoE 层）

术语是什么？
SpheroMoE（Hyperspherical Adaptive MoE）Layer 是 MoE Jetpack 框架中的 MoE 层架构，专为 fine-tuning 从 dense checkpoint 初始化的 MoE 模型而设计。它由三个子组件构成：(1) SpheroMoE Routing：基于 cross-attention 的超球面路由；(2) Expert Regularization：防止 expert 过度特化的正则化策略；(3) Adaptive Dual-path MoE：双路径 expert 结构（core experts + universal experts）。

从算法pipeline角度拆解术语：
SpheroMoE 层的前向传播流程（对应论文 Algorithm 1）：

```
def spheromoe_layer(X, Q, T, core_experts, univ_experts):
    # X: (b, n, d) 输入 token
    # Q: (e*s, d) 随机初始化的查询向量
    
    # 1. 继承 dense checkpoint 的 LayerNorm，保证分布一致性
    X_norm = inherit_layer_norm(X, dim=-1)           # (b, n, d)
    Q_norm = l2_norm(inherit_layer_norm(Q, dim=-1))   # (e*s, d), 超球面投影
    
    # 2. Key 投影
    K = linear(X_norm, W_k)                           # (b, n, d)
    
    # 3. 超球面相似度（L2-norm Q 与 K 做点积 = cosine similarity）
    S = einsum(K, Q_norm, "b n d, e s d -> b n e s")  # (b, n, e, s)
    
    # 4. Expert Regularization
    S = S + normal_noise(S) * noise_mult               # 加噪声
    dispatch = softmax(S / T, dim=1)                   # token→slot 分配
    combine = softmax(S / T, dim=[-1,-2])               # slot→token 重组
    
    # 5. Token 分发
    X_hat = einsum(dispatch, X_norm, "b n e s, b n d -> b e s d")
    X_core = X_hat[:, :core_num, :, :]                 # core expert slot
    X_univ = X_hat[:, core_num:, :, :]                 # universal expert slot
    
    # 6. 并行 Expert 前向（合并所有权重为单一大矩阵，一次 einsum）
    Y_core = parallel_expert_forward(X_core, core_experts)
    Y_univ = parallel_expert_forward(X_univ, univ_experts)
    Y_hat = concat([Y_core, Y_univ], dim=1)
    
    # 7. Stochastic Expert Dropout + 输出重组
    Y_hat = expert_dropout(Y_hat, p)
    Y = einsum(combine, Y_hat, "b n e s, b e s d -> b n d")
    return Y
```

术语一般如何实现？如何使用？
在 PyTorch 中实现为替换 ViT/ConvNeXt 后半层 MLP 的 MoE 模块。Q 随机初始化，与模型参数共同训练。LayerNorm 直接从 dense checkpoint 继承且固定。并行 expert 前向通过将 e 个 expert 的权重矩阵在 batch 维度合并（shape e×d2×d1），单次 einsum 完成所有 slot 的并行计算。使用 AdamW fine-tuning，总体 FLOPs 与原始 dense 模型相当。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

## Soft MoE（软混合专家）

术语是什么？
Soft MoE（Soft Mixture of Experts）是一种隐式软分配的 MoE routing 方法，由 Puigcerver et al. (ICLR 2023) 提出。与 Top-K routing 的硬选择不同，Soft MoE 将每个 expert 处理一个 "slot"——所有输入 token 的加权平均，每个 slot 是一个 learnable 的 token 组合。具体地，通过 learnable parameters Φ ∈ R^{d×(e·s)} 将 m 个输入 token 映射到 e×s 个 slot：X̃ = softmax(XΦ)^T X。每个 expert 处理 s 个 slot，输出通过 softmax(XΦ) 的转置重新组合回 m 个 token。这种方法避免了 token dropping 和 load imbalance 问题。

从算法pipeline角度拆解术语：
Soft MoE 前向传播：
```
# X: (m, d) 输入 token
# Phi: (d, e*s) learnable routing parameters
# experts: e 个 MLP, 每个处理 s 个 slot

# 1. Token → Slot 映射
slots = X @ Phi                                      # (m, e*s)
dispatch_weights = softmax(slots, dim=1)              # (m, e*s)
X_tilde = dispatch_weights.T @ X                      # (e*s, d) 加权 slot

# 2. 每个 expert 独立处理其 slot
for i in range(e):
    Y_tilde[i*s:(i+1)*s] = expert_i(X_tilde[i*s:(i+1)*s])

# 3. Slot → Token 重组
combine_weights = softmax(slots, dim=0)               # (m, e*s)
Y = combine_weights @ Y_tilde                         # (m, d)
```

术语一般如何实现？如何使用？
作为 ViT 或 ConvNeXt 中 MLP 层的替代。在 timm/MMPretrain 框架中使用。每个 MoE 层的 Φ 与 expert MLP 权重共同训练。slot 数 s 通常设为 1（每个 expert 1 个 slot）。作为 MoE Jetpack 论文的 baseline，直接用 Soft MoE 替换 dense ViT 的后半 MLP 层进行 from-scratch 训练。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

## Sparse Upcycling（稀疏升级回收）

术语是什么？
Sparse Upcycling 是一种将预训练 dense checkpoint 转换为 MoE 模型的方法，由 Komatsuzaki et al. (ICLR 2022) 提出。其核心操作是将 dense 模型的 MLP 层复制多份作为 MoE 的 expert 初始权重（即每个 expert = 原 MLP 的完整复制），然后添加 router 并继续训练。这种方式保证了初始时 MoE 模型输出与 dense 模型等价（所有 expert 相同，router 任意加权结果相同），然后通过后续训练逐步分化 expert。

从算法pipeline角度拆解术语：
Sparse Upcycling 流程：
```
# 输入：预训练 Dense 模型（N 层）
# 输出：MoE 模型（N 层, E 个 expert per MoE layer）

for layer l in dense_model:
    if is_moe_layer(l):                          # 选择部分层转为 MoE
        for i in range(E):                       # 每个 expert
            expert_i[l] = copy(dense_mlp[l])      # 完整复制原 MLP
        router[l] = random_init()                 # 随机初始化 router
    else:
        keep_dense(l)                             # 保留非 MoE 层

# 继续训练：expert 从相同的初始点开始逐步分化
```

术语一般如何实现？如何使用？
在 HuggingFace Transformers 框架中实现：加载 T5 等预训练模型，复制 FFN 层权重为多个 expert，添加 Top-K router，继续预训练。与 Checkpoint Recycling 的关键区别：(1) Sparse Upcycling 完整复制 MLP，expert 大小固定与原模型一致；(2) Checkpoint Recycling 可选择性采样权重，构造不同大小的 expert，更灵活。

**MoE-Pruner 的补充**：MoE-Pruner (Xie et al., 2024) 通过实验验证了 Sparse Upcycling 对 MoE 剪枝策略的影响：(a) Upcycling 初始化的 MoE 模型（如 Mixtral-8x7B、Qwen1.5-MoE-A2.7B、MiniCPM-MoE-8x2B）具有更高的 expert 相似性和更均衡的 expert 激活频率，因此 expert-level pruning 会带来严重性能下降，weight-level pruning 是更好的选择；(b) 从零训练（train from scratch）的 MoE 模型（如 DeepSeek-V2、OLMoE）具有更低的 expert 相似性和更不均衡的激活频率，cold expert 可以被安全剪掉。MoE-Pruner 通过 Load Balancing Score（激活频率的变异系数）量化了这一差异：upcycling 模型的 score 通常更低（更均衡），train-from-scratch 模型的 score 更高（更不均衡）。此发现为"不同初始化的 MoE 需要不同压缩策略"提供了定量依据。

**Nexus 的补充**：Nexus (Gritsch et al., 2024) 进一步推广了 Sparse Upcycling 概念，从**多个独立训练的域特化 dense expert** 而非单一 dense checkpoint 进行 upcycling。其流程：(1) 在 SlimPajama 的各子域（ArXiv, Books, C4, StackExchange, Wikipedia）上分别训练 dense expert；(2) 合并时，seed model 的原始 FFN 作为 shared expert（始终激活），各 dense expert 的 FFN 沿新维度拼接为 routed experts；(3) 非 FFN 参数（attention, norms）通过简单权重平均 merge。Nexus 的关键创新是用基于域嵌入的 adaptive router（见 Adaptive Domain-Embedding Router 术语）替代从零训练的线性 router。这种 upcycling 方式使得 expert 保留域专业化（如 ArXiv expert 对 ArXiv token 的路由概率达 63%），且支持后续高效扩展新 expert。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

## Importance-Based Weight Sampling（基于重要性的权重采样）

术语是什么？
Importance-Based Weight Sampling 是 Checkpoint Recycling 的默认权重选择策略。通过在前驱 dense 模型上跑一批图像获取每层各 channel 和 neuron 的 activation 值，然后按 activation 大小选择最重要的 channel（确定性 top-d'）和按 activation 概率分布采样 neuron（随机性保证 expert 多样性），从 dense checkpoint 中提取对应权重子矩阵构造 MoE expert。

从算法pipeline角度拆解术语：

```
# Channel 选择：跨层平均 activation，确定性取 top-d'
# Activation 收集：在 ImageNet 训练集 30K 子集上推理
for each batch B in calibration_set:
    for each layer l in [0..N-1]:
        A_channel[l] += activation_of_channels(B)  # (d,)
A_avg = mean(A_channel, dim=layer)                  # (d,) 跨层平均
top_channels = argsort(A_avg, descending=True)[:d']  # 选最重要的 d' 个 channel

# Hidden Neuron 选择：按 activation 概率分布独立采样每个 expert
for each layer l:
    A_neuron = activation_of_neurons[l]              # (4d,)
    prob = A_neuron / sum(A_neuron)                  # 概率分布
    for each expert e:
        neurons[e] = sample(prob, size=4d')          # 独立采样保证 diversity
```

术语一般如何实现？如何使用？
在 PyTorch 中通过 forward hook 在 predecessor 模型的每层 MLP 后注册 hook 捕获 activation 值。activation profilling 使用校准集（如 ImageNet 子集）进行一次前向传播。选出的 channel/neuron index 用于从 state_dict 中提取对应权重子矩阵构造 expert。作为 Checkpoint Recycling 默认策略（消融实验表明在所有策略中效果最好），替代 Sparse Upcycling 的 naive 复制。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

## Adaptive Dual-path MoE（自适应双路径 MoE）

术语是什么？
Adaptive Dual-path MoE 是 SpheroMoE Layer 中的双分支 expert 结构，利用 Checkpoint Recycling 继承的 dense 先验知识区分重要/非重要 token。Core path 包含少数（约占比 1/3）大型 expert（完整 hidden dim 4d'），处理高重要性 token；Universal path 包含多数（约占比 2/3）小型 expert（hidden dim ≈ d'，约 1/4 参数），处理低重要性 token。两路径在保持 FLOPs 不变的前提下优化计算资源分配。

从算法pipeline角度拆解术语：
SpheroMoE Routing 的 dispatch weights 自然区分了 token 的重要性（通过相似度 logits S 的 softmax 值）。dispatch 后的 slot 数组按重要性排序，前 core_num 个分配给 core experts，剩余分配给 universal experts。流程见 SpheroMoE Layer 术语中的伪代码 Step 5。

术语一般如何实现？如何使用？
在 MoE 层实现中，core_experts 和 univ_experts 是两个独立的 expert group，各有不同的 hidden dim（core: 4d', univ: d' 或更小）。router 输出的 dispatch weights 自然决定 slot 的重要性排序。Core expert 数量占总 expert 数的最优比例为 1/3（由消融实验确定）。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

## Expert Regularization（专家正则化，MoE 上下文）

术语是什么？
在 MoE Jetpack 的 SpheroMoE Layer 中，Expert Regularization 是一组防止 MoE expert 在 fine-tuning 过程中过度特化（over-specialization）和防止输出过度依赖单一 expert 的正则化技术组合，包括：(1) Learnable Softmax Temperature T：初期 T 大→logits 平滑→expert 均匀分散注意力；逐步减小 T→expert 聚焦特定特征；(2) Gaussian Noise：加在相似度 logits S 上，提升泛化能力；(3) Stochastic Expert Dropout：以概率 p 随机停用 expert，防止任一 expert 成为输出瓶颈。

从算法pipeline角度拆解术语：
```
# 在 SpheroMoE 前向传播的相似度计算后应用
S = einsum(K, Q_norm, "b n d, e s d -> b n e s")     # 原始相似度 logits

# 1. Gaussian Noise
noise = torch.randn_like(S) * noise_multiplier
S = S + noise

# 2. Learnable Temperature（训练过程中动态变化）
# T 初始化为较大值（如 5.0），随训练 epoch 逐步减小
dispatch = softmax(S / T, dim=1)                      # T 大→均匀分布，T 小→尖锐分布

# 3. Stochastic Expert Dropout
mask = torch.bernoulli(torch.ones(e) * (1 - p))      # 每个 expert 以 p 概率被 drop
Y_hat = Y_hat * mask.view(1, e, 1, 1)                # 被 drop 的 expert 输出归零
```

术语一般如何实现？如何使用？
Temperature T 实现为 nn.Parameter，随模型一同训练，初期设为较大值（5.0 或更高），训练中通过标准梯度下降自动调节。Noise multiplier 可设为小值（0.01-0.1），dropout rate p 类似标准 dropout 设为 0.1-0.2。三者组合使用，共同确保 fine-tuning 稳定性。

涉及论文标题：
- MoE Jetpack: From Dense Checkpoints to Adaptive Mixture of Experts for Vision Tasks

## Expert Skipping

术语是什么？
Expert Skipping（专家跳过）是一种针对 Mixture-of-Experts (MoE) 模型的推理加速技术。其核心思想是：MoE 的 top-k router 为每个 token 选择 k 个 expert，但并非所有被选中的 expert 都对当前 token 的输出有实质性贡献。Expert Skipping 通过在推理时动态识别并跳过（deactivate）冗余 expert，减少实际执行的 expert 数量，从而降低计算开销。与 training-aware 的 MoE 效率优化（如 load-balanced routing）不同，Expert Skipping 是 **training-free** 方法，直接应用于已训练好的 MoE 模型，无需重新训练或访问训练数据。

从算法pipeline角度拆解术语：
MoDES 中的 Expert Skipping 全流程（以单 token 经过第 l 层 MoE FFN 为例）：

```
# 离线阶段：校准（per model, 一次执行）
calib_set = 随机采样 1024 条数据 (GQA)
for each MoE layer l in [1..L]:
    prob_orig = model.forward(calib_set)           # 原始输出概率
    prob_skip_l = model.forward(calib_set,         # 跳过第 l 层所有 expert
                                skip_experts_at_layer=l)
    alpha[l] = mean(KL(prob_orig || prob_skip_l))  # Eq.(4): 层全局重要性
alpha_tilde = alpha / sum(alpha)                   # 跨层归一化

# 在线推理阶段：Dynamic Expert Skipping
for each token x:
    modality = "text" if is_text_token(x) else "vision"
    for each MoE layer l:
        r = router(x)                              # (M,) routing logits
        pi = softmax(r)                            # (M,) routing probs
        S = topk_indices(pi, k)                    # top-k expert indices
        for i in S:
            s_i = alpha_tilde[l] * pi[i]           # Eq.(3): importance score
            if s_i < threshold[modality]:          # Eq.(5): DMT
                skip Expert_i
        y = sum(pi[i] * Expert_i(x) for i in kept) # 仅保留的 expert 参与计算
```

关键设计：(1) 浅层 expert 的 $\alpha^{(l)}$ 更大 → 更难被跳过 → 保护关键层；(2) Vision token 阈值 $\tau_v$ > text token 阈值 $\tau_t$ → 更激进跳过 vision expert。

术语一般如何实现？如何使用？
- **离线校准**：使用小规模 calibration set (~1024 样本) 计算层的全局重要性因子 $\alpha^{(l)}$ + 搜索最优阈值，耗时 20 min ~ 4 hr (取决于模型大小和硬件)。
- **在线推理**：$\widetilde{\alpha}^{(l)}$ 和阈值 pair $(\tau_t, \tau_v)$ 预加载，每次 expert skipping 决策仅需对 top-k 个路由概率做 element-wise 乘法 + 比较，无额外推理开销。
- **适用场景**：已训练好的 MoE 模型（LLM 或 MLLM），尤其是 top-k > 2 的场景。无需重新训练、无需访问训练数据。
- **与其他技术结合**：可与模型量化（混合精度量化）正交叠加——MoDES 决定跳过哪些 expert，量化压缩保留 expert 的参数精度。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

## Globally-Modulated Local Gating (GMLG)

术语是什么？
Globally-Modulated Local Gating (GMLG) 是 MoDES 提出的 expert importance 评分机制。传统的 expert skipping 方法（如 NAEE、MC-MoE、DiEP）仅依赖当前层的 local routing probability $\pi_i^{(l)}$ 决定跳过哪些 expert，忽略了不同层 expert 对模型最终输出的全局贡献差异。GMLG 将离线校准得到的层级别全局重要性因子 $\alpha^{(l)}$ 与推理时的 local routing probability 相乘，得到综合考虑全局和局部贡献的 expert importance score：$s_i^{(l)} = \alpha^{(l)} \cdot \pi_i^{(l)}$。

从算法pipeline角度拆解术语：
GMLG 的校准与推理流程：

```
# === 离线：计算 alpha ===
calib_set C = {c_1, ..., c_N}  (e.g., GQA 中 1024 样本)
for each MoE layer l in [1..L]:
    # 前向传播原始模型 → 输出概率分布
    prob_j = model.forward(c_j)  for each c_j in C

    # 前向传播跳过第 l 层所有 expert 的修改模型
    prob_j_l = model.forward(c_j, skip_all_experts_at_layer_l)  for each c_j in C

    # 计算该层的 KL 散度均值作为全局重要性
    alpha[l] = (1/N) * sum_{j=1}^{N} KL(prob_j || prob_j_l)

# 实际使用前归一化：使得 0 < s_i^{(l)} < 1
alpha_tilde[l] = alpha[l] / sum_{l'=1}^{L} alpha[l']

# === 在线推理：计算 importance score ===
for token x at layer l:
    pi = softmax(router(x))
    for i in topk_indices(pi, k):
        s_i = alpha_tilde[l] * pi[i]    # GMLG importance score
```

核心 insight：浅层 expert 被跳过时，其误差会经后续 Transformer 层逐层放大（error explosion），因此浅层的 $\alpha^{(l)}$ 更大，对应的 $s_i^{(l)}$ 也更大，更难被阈值过滤掉。

术语一般如何实现？如何使用？
- 校准数据集对结果不敏感——GQA、COCO、VMMMU 等不同数据集的 $\alpha^{(l)}$ 趋势一致（浅层大、深层小），性能也接近。
- 校准计算量：对 N 个样本的 calibration set，需要 L（MoE 层数）次额外前向传播来计算每层被跳过时的 KL 散度。20-30B 参数模型在 8×H200 上校准耗时约 20 min ~ 4 hr。
- 归一化的 $\widetilde{\alpha}^{(l)}$ 作为标量预加载，推理时仅需一次乘法（$\widetilde{\alpha}^{(l)} \times \pi_i^{(l)}$），零额外推理开销。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

## Dual-Modality Thresholding (DMT)

术语是什么？
Dual-Modality Thresholding (DMT) 是 MoDES 提出的针对多模态 MoE 模型（MLLM）的 modality-aware expert skipping 策略。传统 expert skipping 方法对所有 token 采用统一阈值，忽略了 text token 和 vision token 在 MoE FFN 中的行为差异。DMT 为 text token 和 vision token 分别设置独立的跳过阈值 $\tau_t$ 和 $\tau_v$，允许根据不同 modality 的 expert 冗余度进行差异化跳过。

从算法pipeline角度拆解术语：
DMT 的决策过程：

```
# 离线搜索最优阈值对
B = sorted grid of D=100 candidates in (0, 1)
target_skip_ratio = rho (e.g., 83%)
前端搜索 (Frontier Search) 在 O(ND) 时间内找到最优 (tau_t*, tau_v*)
满足 g(tau_t*, tau_v*) >= rho 且最小化 f(tau_t*, tau_v*)

# === 在线推理：DMT 决策 ===
for token x at layer l:
    modality = is_text_token(x) ? "text" : "vision"
    threshold = (modality == "text") ? tau_t : tau_v

    for i in topk_indices(pi, k):
        s_i = alpha_tilde[l] * pi[i]   # GMLG score
        if s_i < threshold:
            skip Expert_i              # Eq.(5)
```

DMT 的设计依据（论文 Motivation 节 Fig. 3）：
- **(Middle)** Vision token 的 pre-FFN 与 post-FFN cosine similarity 高于 text token → FFN 对 vision token 的更新幅度更小 → vision expert 冗余度更高。
- **(Right)** Vision token 与 FFN 权重的夹角更接近 90°（更正交）→ 向量投影小 → 更新幅度受限。
- 实际运行时，$\tau_v > \tau_t$（vision 阈值更高），使得 vision expert 被更激进跳过。MoDES 在 83% 总 skipping ratio 下，vision token 的 skipping ratio 显著高于 text token（Fig. 8）。

术语一般如何实现？如何使用？
- 阈值对 $(\tau_t, \tau_v)$ 由 Frontier Search 离线确定，推理时作为预加载常量。
- DMT 决策在 router kernel 内部实现：计算 importance score 后，使用 branch-free masked comparison 与 modality-specific threshold 比较，不引入额外 kernel launch。
- 仅适用于 MLLM（同时处理 text + vision token）。对于 text-only LLM，只需单一阈值（退化为 Thresholding baseline）。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

## Frontier Search for Threshold Optimization

术语是什么？
Frontier Search 是 MoDES 提出的一种利用单调性属性加速阈值对搜索的优化算法。在 DMT 中需要确定最优的 $(\tau_t, \tau_v)$ 对，使 skipping ratio 满足目标约束 $\rho$ 的同时，最小化与原始模型输出的 KL 散度。Naive exhaustive search 需要评估所有 $D \times D$ 个阈值对（时间复杂度 $\mathcal{O}(ND^2)$），Frontier Search 利用 $f(\tau_t, \tau_v)$（KL 散度）和 $g(\tau_t, \tau_v)$（skipping ratio）对各自参数的单调非递减性质，将搜索复杂度降至 $\mathcal{O}(ND)$（实际约 45× 加速）。

从算法pipeline角度拆解术语：
算法流程（Alg. 1 简化版）：

```
func FrontierSearch(B[1..D], rho_target):
    # B 是排序后的候选阈值 grid: B[1] < B[2] < ... < B[D]
    frontier = []
    p = D
    for q = 1 to D:               # 遍历 tau_t 候选
        while p >= 1 and g(B[q], B[p]) >= rho_target:
            p = p - 1             # 单调递减：找到满足约束的最小 p
        p_q = p + 1               # 最小可行的 tau_v index
        if p_q <= D:
            compute f(B[q], B[p_q])  # 记录该对的目标值
            frontier.append((q, p_q))
    (q*, p*) = argmin_{frontier} f(B[q], B[p_q])
    return (B[q*], B[p*])         # 最优阈值对
```

单调性保证：
- **Monotonicity of g**: $\tau_t \uparrow$ 或 $\tau_v \uparrow$ → 更多 expert 被跳过 → $g \uparrow$。因此对于固定 q，可行的 p 集合是后缀区间 $[p_{(q)}, D]$。
- **Monotonicity of p(q)**: $q \uparrow$ → $p_{(q)} \downarrow$（$\tau_t$ 增大时，满足约束所需的最小 $\tau_v$ 减小）。因此内循环指针 p 在整个外循环中单调递减，总计至多 2D 次 guard evaluation。
- 对于每个记录到 frontier 的 $(q, p_{(q)})$ 对，只需一次前向传播计算 f。总复杂度 $\mathcal{O}(ND)$。

术语一般如何实现？如何使用？
- D=100 grid 点在 (0,1) 间等间隔采样，经 rectified sigmoid 映射。
- N=1024 calibration samples。一次 f/g 评估 = 在 calibration set 上做一次前向传播。
- 实验表明 D=100 和 D=200 的精度差异可忽略（diminishing returns），N=1024 在精度和校准成本间平衡良好。
- 搜索时间：20-30B 参数模型 < 2 hr（vs naive 搜索的 ~45 hr）。搜索时间与 D 大致线性增长。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping

## Sequence Parallelism (SP) for MoE Training

术语是什么？
Sequence Parallelism (SP) 是一种将 Transformer 层的激活张量沿序列长度（sequence length）维度切分到多个 GPU 的并行策略。与 Tensor Parallelism (TP) 沿 hidden dimension 切分不同，SP 使用 all-to-all 通信替代 TP 的 all-gather/reduce-scatter。MegaScale-MoE 将 DeepSpeed-Ulysses 的 SP attention 首次应用于大规模 MoE 训练场景——SP 将 Q/K/V 的 head 维度分片，通过 all-to-all 交换实现 attention 的分布式计算。SP attention 的通信量为 2bsh(n-1)/n × (2+2/m)/n，其中 m 为 GQA 的 query/key-value head 比。当 m=4 时，SP 通信量仅为 TP 的约 1/4。SP 复制 attention weights 而非切分（TP 切分），带来额外参数量，但在 MoE 中 expert 参数占绝对多数（>90%），内存开销仅 1.2-5.4%。

从算法pipeline角度拆解术语：
MegaScale-MoE 的单层 SP attention forward 计算流程：
```
输入: hidden [b, s/n, h]   // n-way SP, 每个GPU持有 s/n 长度的序列片段
// Step 1: QKV Projection (local)
qkv = MatMul(RMSNorm(hidden), qkv_weight)  // [b, s/n, h(1+2/m)]
q, k, v = split(qkv)
q_rope = RoPE(q)   // [b, s/n, h]
k_rope = RoPE(k)   // [b, s/n, h/m]

// Step 2: All-to-All 将 head 分片转为 sequence 分片
// 输入: 每个 GPU 持有所有 heads 但仅 s/n 的序列
// 输出: 每个 GPU 持有完整序列但仅 1/n 的 heads
qkv_a2a = All-to-All([q_rope, k_rope, v])  // [b, s, h(1+2/m)/n]

// Step 3: Self-Attention (每个 GPU 独立计算其 head 子集)
attn = FlashAttention(qkv_a2a)  // [b, s, h/n]

// Step 4: 反向 All-to-All 恢复为 sequence 分片
attn_a2a = All-to-All(attn)  // [b, s/n, h]

// Step 5: Output Projection (local)
attn_out = MatMul(attn_a2a, out_weight)  // [b, s/n, h]
```
SP 相比 Context Parallelism (CP) 的优势：CP 沿 sequence 切分所有激活，但由于 causal masking（每 token 仅能 attend 前面的 token），后部序列分片计算量少于前部分片，造成负载不均。SP 按 head 维度交换数据，每个 GPU 计算完整序列上的部分 head，天然负载均衡。

术语一般如何实现？如何使用？
- 实现来源：DeepSpeed-Ulysses (Jacobs et al., 2023) 提出，MegaScale-MoE 首次在 MoE 训练中大规模部署。
- 参数同步：SP 复制的 attention weights 需通过 hierarchical communication 同步——intra-node reduce-scatter → inter-node reduce-scatter → inter-node all-gather → intra-node all-gather，在 NVLink/NIC 带宽不对称场景下，inter-node 主导总延迟，与 TP 的参数同步时间差异仅 0.3-3.1%。
- 适用条件：(1) GQA 场景下 m>1 时 SP 通信优势扩大；(2) MoE 场景下 expert 参数占内存主体，SP 的额外参数开销可忽略；(3) 不适合 expert 数量少、attention 参数占比高的模型。

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

## Expert Parallelism Adaptive Communication (AG+RS vs A2A)

术语是什么？
Expert Parallelism (EP) 将 MoE 的 expert FFN 分布到多个 GPU 上。标准的 EP 实现使用两次 all-to-all 通信（一次 dispatch + 一次 combine）。MegaScale-MoE 提出自适应通信模式：当 top-k > n（expert 并行度）时，将 all-to-all 替换为 all-gather + reduce-scatter。其原理是：当每个 token 路由到的 expert 数超过并行度时，每个 GPU 上必然有 expert 被某 token 选中，此时 all-gather 可收集所有 token（无信息损失），然后用 local scatter 丢弃不需要的 token。AG+RS 使用环形通信（仅相邻 GPU 通信），而 A2A 需每个 GPU 与其他所有 GPU 通信（全对全），因此 AG+RS 在大 top-k 场景下更高效。通信量公式：EP 为 2k/n × bsh(n-1)/n，TP 为 2bsh(n-1)/n。EP 的优势在于不切分 expert 的 intermediate dimension，保持完整 GEMM 效率。

从算法pipeline角度拆解术语：
EP 通信模式的自适应选择逻辑：
```
if top_k > n:   // n = EP 并行度
    // 使用 AG+RS 模式（环形通信，更高效）
    // Dispatch 阶段:
    gathered = All-Gather(ln2_out)  // [b, s/n, h] → [b, s, h]，每个GPU获得全部token
    ffn_in = Scatter(gathered, routing_map)  // local过滤，仅保留路由到本地expert的token → [b*s*k/n, h]
    // Expert 计算:
    fc2_out = SwiGLU_GroupedGEMM(ffn_in, expert_weights)  // [b*s*k/n, h]
    // Combine 阶段:
    gathered_out = Gather(fc2_out, routing_map)  // local组装 → [b, s, h]
    ffn_out = Reduce-Scatter(gathered_out)  // [b, s/n, h]，环形通信归约
else:
    // 使用 A2A 模式（标准EP实现）
    dispatched = All-to-All(ln2_out, routing_map)  // [b, s/n, h] → 各GPU收到路由给本地expert的token
    fc2_out = SwiGLU_GroupedGEMM(dispatched, expert_weights)
    ffn_out = All-to-All(fc2_out, reverse_routing_map)  // token归还原位
```
在 Mixtral-8×7B 上的实测显示：当 top-k > 6 时，AG+RS 通信时间低于 A2A（Figure 7）。

术语一般如何实现？如何使用？
- 基于 Megatron-LM 实现，自定义 CUDA scatter/gather 算子替代 torch.scatter_add/torch.gather，预计算 routing→memory mapping 实现高效数据传输。
- 每个 MoE layer 的 EP 限制在单 node 内（利用 NVLink 高带宽），跨 node 使用 PP 扩展。
- 负载均衡：使用 auxiliary loss + token dropping，以 GPU 为粒度（而非单个 expert）计算 balance loss 和 capacity。

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production

## Selective Activation Rematerialization in MoE Training

术语是什么？
Selective Activation Rematerialization (SAR) 是 MegaScale-MoE 提出的一种内存优化策略。在 MoE 训练中，由于 expert 参数数量庞大，GPU 内存压力显著。传统的 gradient checkpointing（activation recomputation）在反向传播时重计算前向的所有中间激活，以时间换空间；而 SAR 采取"选择性"策略——在前向传播中仅保留计算密集的激活（如 GroupedGEMM 的输入），丢弃可由轻量级计算或通信重新获得的激活（如 RMSNorm 的输出、all-gather 后的结果）。反向传播时，丢弃的激活通过与独立计算/通信 operator 重叠重新生成，不增加关键路径延迟。MegaScale-MoE 将单 MoE layer 的激活内存从 (2n+2k+3kf+12+5/m)bsh/n 降至 (2kf+4+2/m)bsh/n，节省约 50%。

从算法pipeline角度拆解术语：
以 Mixtral MoE layer 的 backward pass 为例：
```
// 前向保留的激活（存在GPU内存中）: hidden, ln1_out, attn_out, ln2_in, fc2_out
// 前向丢弃的激活（反向时重新生成）: fc2_in, ffn_in, fc1_out, fc3_out

// 反向传播时的激活恢复（与梯度通信重叠）:
// 1. 重新计算 ffn_in
recomputed_ln2_out = RMSNorm(ln2_in)        // 轻量级，隐藏在其他通信中
recomputed_ln2_out_ag = All-Gather(recomputed_ln2_out)  // 与上一个GEMM的反向计算重叠
recomputed_ffn_in = Scatter(recomputed_ln2_out_ag)

// 2. 重新计算 fc2_in (SwiGLU 的输入)
// fc1_out 和 fc3_out 通过重新执行 fc1 和 fc3 的 GroupedGEMM 获得
// 这些计算与 Δfc2_out 的 gradient 通信同时进行
recomputed_fc1_out = GroupedGEMM(recomputed_ffn_in, fc1_weight)
recomputed_fc3_out = GroupedGEMM(recomputed_ffn_in, fc3_weight)
recomputed_fc2_in = SiLU(recomputed_fc1_out) * recomputed_fc3_out

// 3. 使用恢复的激活完成反向计算
Δfc2_in = GroupedGEMM_backward(Δfc2_out, fc2_weight, recomputed_fc2_in)
```
关键设计：(1) 将 ffn_out 的加权求和立即放在 SwiGLU 激活函数后（而非单独存储 ffn_out），消除该激活的存储；(2) 不跨越非线性边界重排 operator，保证计算一致性。

术语一般如何实现？如何使用？
- 与 Holistic Scheduling 紧密配合：通过手动编排整个 MoE layer 的前向/反向 operator 执行顺序（而非依赖 torch.autograd），使重计算与通信 overlap。
- 实测在 Mixtral-8×7B 上节省 45.5% 激活内存（总内存节省 21.3%），在 Mixtral-8×22B 上节省 57.2% 激活内存（总内存节省 35%），训练 MFU 差异 <0.5%。
- 适用场景：MoE 模型训练（expert 数量多、激活内存压力大），配合 inter-operator overlap 使用效果最佳。

涉及论文标题：
- MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 采用粒度更细的 activation checkpoint 策略——针对特定激活函数（SwiGLU 中的 SiLU）而非整个通信/计算操作进行选择性重计算。策略基于两个观察：(1) SiLU 为 element-wise 操作（仅 point-wise multiply + sigmoid），在现代 GPU（H100）上 memory bandwidth bound；(2) tall-and-skinny 矩阵（L≫d）下 activation 的内存带宽瓶颈尤为显著。因此 forward 中不保存 SiLU(a)，仅保存 a 和 b 用于 GEMM 反向；backward 时从保存的 a recompute SiLU(a) = a·σ(a)，recompute 开销 ≈ 从 HBM 读取 SiLU(a) 的成本（memory bandwidth bound 条件下）。此策略与 kernel fusion 协同设计——fused kernel 中的 epilogue fusion 已消除了 a, b, σ(a), SiLU(a), y_swi 等中间结果的多次 HBM 往返，checkpoint 进一步消除 SiLU(a) 的存储。此策略在 SwiGLU 下贡献约 4× activation memory reduction（vs MegaBlocks baseline）。

---

## Communication Compression for Distributed Training

术语是什么？
Mixture-of-Experts (MoE) 是一种在推荐系统中广泛使用的多任务学习架构。其核心思想是：使用多个独立的"专家网络"（expert networks，通常为 MLP + ReLU 激活）并行处理输入，再通过一个或多个可学习的门控网络（gate network）对专家输出进行软性加权组合，最后送入任务特定的预测塔（prediction tower）。相比 Shared-Bottom 等硬共享方法，MoE 通过软路由机制实现了更灵活的参数共享——不同任务可以从同一组专家中提取不同比例的共性信息，从而缓解任务冲突和负迁移。代表作包括 MMoE（Multi-gate MoE，每任务独立 gate）、PLE（Progressive Layered Extraction，引入 task-specific experts 并多层堆叠）和本论文 M3oE（引入 domain experts 和 task experts 的解耦三模块设计）。

从算法pipeline角度拆解术语：
MMoE 的计算流程（以 D 个域、T 个任务为例）：
```
输入: h_d (域d的embedding)
// Expert forward pass（N个共享专家）
for e in 1..N:
    expert_e = ReLU(LayerNorm(W_e @ h_d + b_e))

// Gate forward pass（每个任务t有独立gate）
for t in 1..T:
    gate_t = softmax(W_gate_t @ h_d + b_gate_t)  // shape: (N,)
    weighted_sum_t = sum_{e=1}^{N} gate_t[e] * expert_e

// Task-specific tower
for t in 1..T:
    y_hat_t = Sigmoid(W2_t @ ReLU(W1_t @ weighted_sum_t + b1_t) + b2_t)
```
M3oE 在此基础上扩展为三个专家模块：共享专家（N个）、域专家（D个）、任务专家（T个），并用两级融合机制替代单层 gate，实现 domain-aspect 和 task-aspect 信息的显式解耦建模。

术语一般如何实现？如何使用？
MoE 在推荐系统中通常通过 PyTorch/TensorFlow 实现：专家网络为单层或两层 MLP（含激活和归一化），gate 为线性层 + softmax。多个专家形成一个 ModuleList，gate 输出与 expert 输出做加权求和（等价于 batch matrix multiplication）。PLE 等变体进一步引入 task-specific experts 和多层 CGC（Customized Gate Control）模块堆叠。在工业界（如快手、字节跳动），MoE 被广泛用于点击率/转化率等多任务预估场景。

在 Transformer 语言模型中，MoE 层通常替换 FFN 层（Shazeer et al. 2017, Fedus et al. 2022）。每个 token 通过 Router 动态选择 top-k（通常 k=1-4）个 expert。标准实现使用 batched GEMM 计算所有 expert，但这引入了 expert capacity 约束和 token dropping/padding 问题。MegaBlocks (MLSys 2023) 通过 block-sparse 重表述将 batched GEMM 替换为 block-sparse 矩阵乘法（SDD/DSD/DDS），从根本上消除 token dropping，实现 dropless-MoE (dMoE)。已被用于训练 Mixtral 8×7B 和 DeepSeek V2。

MegaScale-Infer 从 serving 效率角度分析了 MoE 的 decoding 瓶颈：基于 Roofline Model，dense LLM 的 FFN 利用率 = min(B·F/B, 1)，MoE 的 FFN 利用率 = min(top-k/#experts · B·F/B, 1)。以 Mixtral 8×22B 在 A100（312 TFLOPS, 2 TB/s）为例，batch size 至少需 156 tokens 才能使 dense FFN compute-bound，但 MoE sparsity（top-2/8=25%）使有效 batch per expert 仅 39，MFU 仅 25%。更大的 MoE 模型（更多 experts、更低 top-k/#experts 比）sparsity 退化更严重。解耦 attention-expert 部署通过聚合多个 attention node 的请求增大 expert batch size 来逆转此退化。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism

---

## Multi-Domain Multi-Task (MDMT) Recommendation

术语是什么？
Multi-Domain Multi-Task (MDMT) Recommendation 是指在推荐系统中同时处理多个业务域（如不同 tab、不同终端、不同场景）和多个优化目标（如点击率、点赞率、收藏率、长播率）的问题。与单一的多域推荐（MDR）或多任务推荐（MTR）不同，MDMT 引入了一个更高维度的交叉关系：域-任务交互（domain-task interplay）。即同一域信息传递策略在不同任务上效果不同，同一任务平衡策略在不同域上效果也不同。这是一个尚未被充分研究的实际推荐场景，相比 MDR 或 MTR 更具挑战性。

从算法pipeline角度拆解术语：
MDMT 问题的形式化定义：
- 设用户集 U，物品集 I，D 个域，T 个任务
- 每个样本 (x_d, y_{d,1}, ..., y_{d,T}) 属于某个域 d
- 目标：学习 T × D 个预测函数 f^{d,t}(x_d) → ŷ_{d,t}
- Loss: L = Σ_{d=1}^{D} Σ_{t=1}^{T} BCE(ŷ_{d,t}, y_{d,t})

挑战在于：不同域和任务的最优信息共享和融合策略各不相同。M3oE 通过 α_d/α_t（控制模块间贡献）和 β_d/β_t（控制专家间贡献）两级融合权重实现自适应。PEPNet（快手 KDD 2023）则通过 EPNet 对齐域间 embedding 语义 + PPNet 个性化 tower 参数来解决此问题。

术语一般如何实现？如何使用？
MDMT 推荐通常有两种实现路径：(1) 联合训练（如 M3oE）：一个模型同时处理所有域和任务，通过解耦模块和自适应融合实现信息共享与隔离的平衡；(2) 分离+转移（如 M2M）：用 meta-learning 或迁移学习方法将知识从一个域/任务转移到另一个。实际部署中，工业界（如快手、字节跳动）的推荐系统通常面临 3-5 个域和 3-8 个任务的同时优化需求。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---

## MDMT Seesaw Problem

术语是什么？
MDMT Seesaw Problem（多域多任务跷跷板问题）是 M3oE 论文提出的新概念，描述了在多域多任务推荐中同时出现的两类跷跷板效应的叠加。具体表现为：(1) 同一多域信息传递方法不能泛化到不同任务——例如用域间迁移提升点击率的方法可能损害点赞率；(2) 同一多任务优化平衡策略不能泛化到不同域——例如在域 A 有效的任务权重分配在域 B 可能失效。这一问题综合了已知的 domain seesaw（域间跷跷板，提升一个域的性能可能损害其他域）和 task seesaw（任务间跷跷板，提升一个任务的性能可能损害其他任务，如 PLE 论文所述）。M3oE 是首个明确定义并系统解决此问题的工作。

从算法pipeline角度拆解术语：
以视频平台的例子说明 MDMT seesaw 的具体表现：
- Domain seesaw 场景：用户在 TV 上观看 Sci-Fi 的偏好如何迁移到 Tablet 域
- Task seesaw 场景：用户"观看"行为与"点赞"行为之间的关系建模
- MDMT seesaw 场景：用户在 TV 上"观看"Sci-Fi 的偏好如何迁移并增强 Tablet 域"点赞"的预测——这涉及跨域×跨任务的双重信息传递

M3oE 解决此问题的策略是解耦：shared experts 学习跨域跨任务的共同模式，domain experts 维护域特定信息，task experts 维护任务特定信息，通过 AutoML 自适应两级融合权重为每个 (d,t) 对找到最优的信息组合方式。

术语一般如何实现？如何使用？
解决 MDMT seesaw 的核心思路是解耦（disentanglement）+ 自适应融合（adaptive fusion）。解耦确保不同类型的信息不会互相干扰，自适应融合确保每个 domain-task pair 能按需获取合适比例的信息。其他相关工作如 PEPNet 通过个性化先验信息注入来缓解此问题，MTKDN 通过对比解耦机制分离共享和任务特定表征。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---

## Multi-View Expert Learning (Shared/Domain/Task Expert Modules)

术语是什么？
Multi-View Expert Learning（多视角专家学习）是 M3oE 框架的核心设计，通过三种类型的专家模块从输入中提取不同视角的信息：(1) Shared Expert Module（共享专家模块）——N 个专家网络处理所有域的输入，通过 D×T 个独立 gate 为每对域-任务生成加权组合，捕获跨域跨任务的共性模式（如通用用户兴趣）；(2) Domain Expert Module（域专家模块）——D 个域专属专家，每个专家关联一个特定域，通过带偏置权重 β_d 的融合策略整合域特定视角和其他域的增强信息；(3) Task Expert Module（任务专家模块）——T 个任务专属专家，每个专家关联一个特定任务，通过 β_t 控制当前任务视角和其他任务视角的融合。三类专家共享相同的网络结构（单层 MLP + LayerNorm + ReLU），但具有独立的可学习参数。

从算法pipeline角度拆解术语：
```
输入: h_d (域d的表示)
// Shared Expert Module (N个专家, D×T个gate)
S_{d,t}(h_d) = softmax(W_gate_{d,t} @ h_d) · [expert_1(h_d), ..., expert_N(h_d)]

// Domain Expert Module (D个专家, 偏置融合)
d_out = β_d·expert_d(h_d) + (1-β_d)/(D-1)·Σ_{k≠d} expert_k(h_d)

// Task Expert Module (T个专家, 偏置融合)
t_out = β_t·expert_t(h_d) + (1-β_t)/(T-1)·Σ_{k≠t} expert_k(h_d)

// 两级融合 (Level-2: 模块间平衡)
h̄_d = S_{d,t}(h_d) + α_d·t_out + α_t·d_out
```
其中 α_d, α_t, β_d, β_t ∈ (0,1) 由 AutoML 自适应学习。

术语一般如何实现？如何使用？
所有专家网络共享相同结构但参数独立，可用 PyTorch ModuleList 实现。Shared expert 的每个 gate 是一个线性层后接 softmax；Domain/Task expert 的融合通过标量权重加权求和实现（无需额外网络）。β_d 接近 1 表示仅依赖域自身专家、忽略其他域信息；β_d 接近 0 表示更多依赖其他域知识传递。T-SNE 可视化（论文 Figure 3）验证了该设计确实产生了解耦的嵌入表示——domain expert 的融合嵌入与对应域嵌入分布相似（域特定专家占主导），而 task expert 的融合嵌入在 β_t 最优时取得多个专家间的平衡分布。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---

## Two-Level Fusion Mechanism

术语是什么？
Two-Level Fusion Mechanism（两级融合机制）是 M3oE 框架中用于精确控制多域多任务信息聚合的设计，分为两个层级：(1) 第一级融合：在 Domain Expert Module 和 Task Expert Module 内部，通过 β_d 控制当前域专家与其他域专家输出的加权平衡（域间融合），通过 β_t 控制当前任务专家与其他任务专家输出的加权平衡（任务间融合）；(2) 第二级融合：通过 α_d 和 α_t 控制 Shared Expert、Domain Expert 和 Task Expert 三个模块之间的贡献比例（模块间融合）：h̄_d = S(h_d) + α_d·T(h_d) + α_t·D(h_d)。两级融合权重的乘积（如 α_d·β_d）共同决定了域特定专家最终对预测的贡献比例，实现了对每一对 (d,t) 信息源的精细逐样本调控。

从算法pipeline角度拆解术语：
两级融合的权重建模：
```
// 第一级：专家内部融合（expert-level）
Domain_fused = β_d·expert_d + (1-β_d)/(D-1)·Σ_{k≠d} expert_k
Task_fused   = β_t·expert_t + (1-β_t)/(T-1)·Σ_{k≠t} expert_k

// 第二级：模块间融合（module-level）
h_bar = Shared_gated + α_d·Task_fused + α_t·Domain_fused

// 所有权重由可训练标量生成
w = Sigmoid(e_w)  // e_w ∈ {e_αd, e_αt, e_βd, e_βt} 是一维可训练张量

// 权重语义：
// β_d ∈ (0.5,1) → 当前域专家主导；β_d ∈ (0,0.5) → 其他域知识传递主导
// α_d 大 → 域模块贡献高（相比共享和任务模块）
```

术语一般如何实现？如何使用？
两级融合通过可训练标量 + Sigmoid 激活实现，微分友好，可直接通过梯度下降优化。与使用门控网络（如 fully-gated variant）相比，该方法参数量极小（仅 4 个标量参数），但通过解耦设计实现了比统一 gate 更精准的融合控制。消融实验（论文 Table 3）表明：两级融合优于直接拼接（Concat modules）和全门控融合（Fully gated modules），验证了显式解耦+可控融合优于隐式学习的结论。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---

## Bi-Level Optimization for AutoML in Recommendation Models

术语是什么？
Bi-Level Optimization（双层优化）是一种优化框架，将模型参数和超参数/架构参数分为两个层级交替优化。在 M3oE 的语境中：外层优化（upper-level）在给定融合权重 α, β 的情况下更新模型参数 W = argmin_W L(W, α, β)；内层优化（lower-level）在模型参数更新后，基于一个 mini-batch 的数据优化融合权重 α, β = argmin_{α,β} L(W* , α, β)。这一方法源自 DARTS（Differentiable Architecture Search, Liu et al. 2018），区别在于 M3oE 搜索的不是网络结构（如卷积核大小、层数），而是融合权重——将 α_d, α_t, β_d, β_t 参数化为可训练标量经 Sigmoid 激活，使其可直接通过梯度下降优化。

从算法pipeline角度拆解术语：
M3oE 的 Bi-Level Optimization 训练流程：
```
初始化: 模型参数 W, 可训练标量 e_αd, e_αt, e_βd, e_βt

for epoch in 1..E:
    // Step 1: 外层更新 (模型参数)
    前向传播计算预测 ŷ_{d,t}
    计算 Loss L = Σ_{d,t} BCE(ŷ_{d,t}, y_{d,t})
    反向传播更新 W (固定 α, β)

    // Step 2: 内层更新 (融合权重)
    取一个 mini-batch 数据
    计算当前 W 下的 Loss
    反向传播更新 e_αd, e_αt, e_βd, e_βt (固定 W)
    更新 α_d = Sigmoid(e_αd), β_d = Sigmoid(e_βd), ...
```
内层更新的计算量很小（仅 4 个标量参数），因此额外开销"trivial"（论文原文）。消融实验（Table 3, w/o AutoML）表明：将融合权重固定为 0.5（等价于无差异化融合）会导致 MovieLens AUC 从 77.02 降至 76.37，KuaiRand-Pure AUC 从 66.37 降至 65.41，验证了自适应权重学习的必要性。

术语一般如何实现？如何使用？
在推荐模型中使用 Bi-Level Optimization 通常涉及以下实现细节：(1) 融合权重不参与常规 optimizer 的更新步骤，而是单独用一个 optimizer（如 Adam）在验证 loss 上优化；(2) 由于架构参数少，内层优化通常在一个 mini-batch 上完成即可（不需要完整 epoch）；(3) 实际部署时权重在训练完毕后固定，推理时无额外开销。此方法不仅适用于 M3oE 的融合权重，也可扩展到其他需要自适应权衡的超参数场景（如多任务 loss 权重）。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning (用于优化 Rational Activation 参数 Θ vs LoRA/Router 参数 Ω，inner: Ω=argmin L(D_train,Ω,Θ) lr=1e-4，outer: min L(D_val,Ω*,Θ) lr=1e-6，交替优化；Θ 仅为每层 ~12 scalars)

---

## Gate Fusion in MoE

术语是什么？
Gate Fusion（门控融合）是 MoE 架构中用于聚合多个专家网络输出信息的机制。标准实现为：一个可学习的门控网络（通常为线性层 + softmax）根据输入样本生成 N 维权重向量，对 N 个专家的输出做加权求和：output = Σ_{e=1}^{N} gate[e] · expert_e(input)。门控融合的关键特性是样本自适应（sample-wise）——不同样本可能激活不同的专家组合。在 M3oE 中，Shared Expert Module 使用 D×T 个独立 gate 为每一对域-任务生成专属的专家融合权重（公式 4），确保不同优化目标能从共享知识中获取不同比例的信息。而对于 Domain/Task Expert Module，则采用基于可训练标量的偏置融合而非门控网络，以降低参数量并实现更显式的解耦控制。

从算法pipeline角度拆解术语：
```
// 标准 Gate Fusion (MMoE)
h = input_embedding
gate_logits = W_gate @ h + b_gate     // shape: (batch, N)
gate_weights = softmax(gate_logits, dim=-1)  // 每行和为1
expert_outputs = stack([expert_i(h) for i in 1..N])  // shape: (batch, N, hidden)
fused = sum(gate_weights[:, :, None] * expert_outputs, dim=1)  // 加权求和

// M3oE Shared Module 的 Gate Fusion
// 有 D×T 个独立 gate，每个负责一对 (d,t)
for d in 1..D, t in 1..T:
    gate_{d,t} = softmax(W_{d,t} @ h_d + b_{d,t})
    S_{d,t}(h_d) = gate_{d,t} · [expert_1(h_d), ..., expert_N(h_d)]
```
注意 M3oE 的 Shared gate 与 domain expert / task expert 模块的融合方式不同：后者使用固定标量权重（β_d, β_t）而非输入依赖的门控，因为域/任务归属在样本级别已确定，无需样本自适应融合。

术语一般如何实现？如何使用？
Gate Fusion 在 PyTorch 中通常通过 `nn.Linear(hidden_dim, num_experts)` + `F.softmax(dim=-1)` 实现，结合 `torch.einsum` 或广播乘法完成加权求和。Gate 融合适用于专家选择依赖输入内容的场景（如 Shared Expert，因为不同样本利用跨域共性知识的程度不同），而对域/任务归属已知的场景，使用可训练标量权重更高效且可解释性更强。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---

## Domain Representation Extraction

术语是什么？
Domain Representation Extraction（域表示提取）是 M3oE 框架的底层模块，负责将不同域的异构输入特征映射到统一的表示空间，同时保留域特定信息和跨域共性信息。其核心操作是：(1) 对每个域 d，将域特定权重矩阵 W_d 和共享权重矩阵 W_sh 做 element-wise product：Ŵ_d = W_d ⊙ W_sh，结合了域独特性和跨域共性；(2) 通过共享线性变换 W_c 将所有域的表示映射到同一 embedding 空间；(3) 额外引入 domain-agnostic mapping f_DA（一个多层神经网络）对原始输入做域无关映射并作为残差加入，以调节统一表示空间、抑制来自其他域的噪声。这一设计受 STAR（Sheng et al. 2021, CIKM）中 star topology 的启发——STAR 使用因子化的域共享和域特定网络来处理多域输入。

从算法pipeline角度拆解术语：
```
输入: x_d (域d的原始特征)

// Step 1: 域特定 + 共享权重融合
W_hat_d = W_d ⊙ W_sh                  // element-wise product

// Step 2: 域特定线性变换 + 共享偏置
z_d = W_hat_d @ x_d + b_d + b_sh

// Step 3: 统一空间映射
u_d = W_c @ z_d + b_c

// Step 4: 域无关残差连接
h_d = u_d + f_DA(x_d)                 // f_DA 为多层MLP

输出: h_d (统一表示空间中的域表示)
```
其中 W_sh、W_c 和 f_DA 在所有域上共享参数，学习跨域通用模式。

术语一般如何实现？如何使用？
Domain Representation Extraction 本质上是输入层的特征工程 + 域适配，通常作为模型的第一层。W_d ⊙ W_sh 的 element-wise product 设计在 PyTorch 中可直接用 `*` 运算符实现。f_DA 的域无关映射作为残差连接，其作用类似于正则化项——即使域特定信息有限，也能通过通用映射提供稳定的基准表示。这一层的输出 h_d 被后续的 Multi-View Expert Learning Layer 中各模块共享使用，因此其质量直接决定上层解耦和融合的效果。

涉及论文标题：
- M3oE: Multi-Domain Multi-Task Mixture-of-Experts Recommendation Framework

---

## GPTQ (GPT Post-Training Quantization)

术语是什么？
GPTQ (Frantar et al. 2022) 是一种 training-free 的 LLM 权重量化方法，通过逐层（layer-wise）量化 + Hessian 矩阵引导的误差补偿，在无需额外训练的情况下将 LLM 权重量化到 2-4 bit。核心思想：利用 Hessian 矩阵 H = 2XXᵀ（X 为校准数据的激活输入）衡量各权重的量化敏感度，对 H⁻¹ 做 Cholesky 分解后逐列量化，每量化一列后用 Hessian 信息补偿剩余列的量化误差（optimal brain surgeon 风格）。支持 group-wise quantization（如 128 列一组共享 scale/zero-point）。在 MC-MoE 中，GPTQ 被用作底层量化引擎，对每个 expert 按 PMQ 分配的位宽（1/2/3-bit）执行量化，Mixtral 8×7b 上 90 分钟完成。

从算法pipeline角度拆解术语：
```
输入: W ∈ R^{d_in × d_out}, 校准激活 X, 目标位宽 B
H = 2 * X @ X^T
L = cholesky(H^{-1})
for col in 1..d_out:
    W_q[:,col] = quantize(W[:,col], B)
    error = (W[:,col] - W_q[:,col]) / L[col,col]
    for r in col+1..d_out:
        W[:,r] -= error * L[col,r] / L[col,col]
输出: W_q
```
GPTQ 对 2/3-bit 使用标准线性量化（scale + zero-point）；对 1-bit 使用二值化 B = sign(W)。

术语一般如何实现？如何使用？
- 开源：https://github.com/IST-DASLab/gptq
- 校准数据：通常 128 序列 × 2048 tokens（C4/WikiText2）
- 适用于 training-free 快速量化场景，被 MC-MoE、BSP 等方法复用为底层量化算子
- 局限：≤2 bit 时精度下降显著，需配合混合精度策略才能保持可用精度

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts

**MoE-SpeQ 中的 GPTO 使用方式**：作为 draft model 的量化方法，对 target FP16 MoE 模型的 expert MLP 线性层做 4-bit 对称 INT4 量化（group_size=128），Router/Attention/Shared Experts 保持 FP16（Hybrid-Precision 策略，防止 router 量化误差通过 softmax 放大导致误路由）。量化后的 draft 模型全量驻留 GPU VRAM，配合 Marlin 后端的 fuseMoE kernel 执行。关键数据：INT4 draft 以 90.9% total fidelity 预测 FP16 target 的 expert selection。

---

## Expert Significance in MoE (ϕ·w Multi-Factor Importance)

术语是什么？
Expert Significance（专家重要性）是 MC-MoE 提出的衡量 MoE 中每个 expert 对模型输出贡献程度的多维评估指标。包含三维因子：(1) 访问频率 ϕᵢ = nᵢ/N：expert i 被 Router 选入 Top-K 的频率，反映通用性；(2) 激活权重和 wᵢ = Σσᵢʲ/N：expert i 的 routing score 累计值，反映每次激活的贡献强度——低频但高权 expert 对特定 token 可能极为关键；(3) 量化重构误差 εᵢⱼ = ‖F(θ) − F(θ[eᵢ→Q(eᵢ,j)])‖_F：expert i 被单独量化到 j-bit 后输出 activation 的 F-norm 偏差。三者以 ϕᵢᵅ·wᵢᵝ·εᵢⱼᵞ 组合构成 Integer Programming 损失函数核心项（MC-MoE 消融确定 α=β=1, γ=2），解决"均匀量化忽略重要 expert"和"仅看频率忽略低频高权 expert"两大缺陷。

从算法pipeline角度拆解术语：
```
// 离线: 在 FP16 MoE 模型上用校准数据 C4 做一次前向推理
for each token t in C4:
    routing = softmax(W_gate @ t)
    top_k = TopK(routing, k=2)
    for each selected expert e_i:
        ϕ_i++, w_i += routing[e_i]
ϕ_i /= total_tokens, w_i /= total_tokens

// 量化误差: 单独量化每个 expert 并测输出 F-norm
for each expert e_i, bit j ∈ {1,2,3}:
    ε_{i,j} = ||F(θ) - F(θ[e_i→Q(e_i,j)])||_F

// 综合重要性 = ϕ_i^α · w_i^β · ε_{i,j}^γ
```

术语一般如何实现？如何使用？
- 离线计算：仅需一次 FP16 前向推理（无梯度），计算开销极小
- 应用：(a) expert 位宽分配（MC-MoE PMQ）；(b) expert 静态剪枝（永久移除不重要的 expert）；(c) expert 卸载决策（低频低权 expert 卸载到 CPU/SSD）
- MC-MoE 的发现：ϕ 和 w 的分布可能不一致甚至相反（如 expert[1,3] 低权但高频），验证了多因素评估的必要性
- 局限：重要性依赖校准数据分布，分布外任务可能导致不同的重要性排序

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---

## Dynamic Expert Pruning in MoE-LLMs (Token-Aware)

术语是什么？
Dynamic Expert Pruning（动态专家剪枝）是在 MoE-LLM 推理时对每个 token 动态决定实际激活的 expert 数量的技术。与静态剪枝（永久移除某些 expert）不同，动态剪枝的决策是 per-token 的。MC-MoE 的 ODP（Online Dynamic Pruning）含两个关键组件：(1) Weight-guided pruning：当 Top-2 中次要 expert 的 routing score 远小于主要 expert（w₁/w₀ < μ，μ 为 calibration 数据中位数），跳过该次要 expert；(2) Token protection：基于 token importance Iⱼ = ‖tⱼ‖₁ · mean_attention_score 保护 top 2% 重要 token 的所有 expert 不被剪枝，防止 attention decay 级联效应。平均减少约 15% 激活参数，准确率损失 < 1%。

从算法pipeline角度拆解术语：
```
// 每个 MoE layer 推理时动态执行
I_j = ||t_j||_1 · (Σ_{i≥j} A_{j,i}) / (L - j)  // token importance
is_protected = (I_j in top 2%)

{w_0, w_1} = Top-2{G(t)}
if is_protected:
    y = w_0·E_0(t) + w_1·E_1(t)     // 保护: 完整 top-2
elif w_1/w_0 < μ:                     // μ = 校准集 w₁/w₀ 中位数
    y = w_0·E_0(t)                    // 剪枝: 降为 top-1
else:
    y = w_0·E_0(t) + w_1·E_1(t)     // 正常 top-2
```

术语一般如何实现？如何使用？
- 实现依赖：(a) 校准数据确定 pruning threshold μ；(b) 推理时在线计算 token importance（计算开销 O(n²+mn) FLOPs，远小于 expert 推理的 O(n·m·m₁) FLOPs）
- 适用场景：MoE-LLM 延迟敏感推理（如 real-time chatbot）
- 与静态剪枝互补：静态剪枝减少存储，动态剪枝减少计算
- 创新点：token-aware protection 机制，仅保护 2% token 即消除 attention decay

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---

## Attention Decay under MoE Expert Pruning

术语是什么？
Attention Decay（注意力衰减）是 MC-MoE 发现的一种 expert 剪枝引起的级联效应：在 MoE layer L 中对某 token 的 expert 进行剪枝后，该 token 的 hidden state 表示质量下降，进而在下一层（L+1）的 self-attention 中，该 token 无法吸引其他 token 的注意力（attention score 降低），导致关键上下文信息丢失。具体表现为：未剪枝时 attention map 中有明显垂直高亮列（其他 token 高度关注此 token），weight-only pruning 后该列 score 显著降低。这解释了为什么仅用 routing weight 做剪枝会在 15% 剪枝率下造成 ~10% LM-Eval 精度损失。保护 top 2% 重要 token 即可有效缓解此效应。

从算法pipeline角度拆解术语：
```
Block L: token t_j 的次 expert 被 weight-only pruning 剪枝
  → h_j^L = w_0·E_0(t_j) (vs 正常的 w_0·E_0 + w_1·E_1)
  → h_j^L 信息量降低

Block L+1:
  → Q_j = W_Q @ h_j^L, K_j = W_K @ h_j^L  // 受污染的表征
  → A[:,j] = softmax(Q @ K^T/√d_k)[:,j]   // attention score 降低
  → 其他 token 对 t_j 关注度下降，信息传递受阻
```

术语一般如何实现？如何使用？
- 检测：对比剪枝前后 attention map 特定 token 列的 score 变化
- 缓解方案：(a) Token-aware protection（MC-MoE ODP）；(b) Attention-aware pruning metric（在 pruning 决策中直接考虑 attention map 影响）；(c) Layer-wise adaptive threshold
- 意义：MoE 压缩不能仅关注 expert 层面的精度损失，必须考虑 token 间注意力交互的级联效应

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---

## HQQ (Half-Quadratic Quantization)

术语是什么？
HQQ (Half-Quadratic Quantization, Badri & Shaji 2024) 是一种无需校准数据的 LLM 权重量化方法，通过半二次优化（half-quadratic optimization）直接在权重空间上交替优化求解量化参数（scale s、zero-point z）和量化权重 W_q。与传统 PTQ（需校准数据）不同，HQQ 完全不需要任何校准数据或前向推理。支持 1-8 bit 量化。在 MC-MoE 中，HQQ 被用于两方面：(1) 混合精度权重存储（compact bit-packed storage），将 PMQ 量化后的 1/2/3/4-bit 权重紧凑位压缩保存；(2) 提供 CUDA kernel 执行反量化（dequantization）+ 矩阵乘法，对 1-bit 权重有专门位运算加速。

从算法pipeline角度拆解术语：
```
// 推理路径
W_packed = load_compact_bits(memory, bit_width)  // 位压缩加载
W_dequant = W_packed * scale + zero_point         // 反量化到 FP16
Y = X @ W_dequant                                 // GEMM

// MC-MoE 1-bit 特化优化（位变换）:
// 存储: B̃ = (sign(W)+1)/2 ∈ {0,1}  // ±1 → 0/1
// 推理: s·xB = s(Σ_{B̃_{ij}=1} x_j - Σ_{B̃_{ij}=0} x_j)
// MACs: m (仅 scaling) vs FP16 dm 次乘法
```

术语一般如何实现？如何使用？
- 开源：https://github.com/mobiusml/hqq
- 优势：无需校准数据，推理 CUDA kernel 现成，位压缩存储紧凑
- MC-MoE 中的角色：作为推理部署工具，GPTQ 负责量化、HQQ 负责存储/反量化/内存管理
- 局限：HQQ 自身量化精度通常不如 GPTQ/Omniquant（无校准数据优化），但在极低位宽下差异缩小

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---

## Post-Training Quantization (PTQ) for LLMs

术语是什么？
Post-Training Quantization (PTQ) 是一类无需重新训练或微调即可将预训练模型量化的技术总称。与 QAT (Quantization-Aware Training，需数百 GPU 小时训练) 相比，PTQ 仅需少量校准数据（或无数据如 HQQ）做前向推理确定量化参数（scale、zero-point、bit-width 分配），计算开销极小。常见 PTQ 方法：(1) Round-to-nearest (RTN)：直接四舍五入，简单但极低位宽精度低；(2) GPTQ：Hessian 引导逐列量化 + 误差补偿；(3) AWQ：激活感知权重等效变换（per-channel scaling）；(4) Omniquant：可学习权重裁剪（LWC）+ 校准优化；(5) HQQ：半二次优化无需校准数据。LLM PTQ 的核心挑战是 activation outlier（某些 channel 激活值远大于其他），需 per-channel/group quantization、smooth quantization 或 Hadamard rotation 缓解。

从算法pipeline角度拆解术语：
```
// 阶段1: 校准 (确定量化参数)
for batch in calibration_data:
    activations = model.forward(batch)
    layer.scale = (max(W)-min(W)) / (2^B-1)
    layer.zero_point = round(-min(W) / layer.scale)

// 阶段2: 量化
W_q = clamp(round(W/scale + zero_point), 0, 2^B-1)

// 阶段3: 推理 (融合反量化)
Y = X @ (W_q * scale + zero_point)
```

术语一般如何实现？如何使用？
- 适用场景：(a) 资源受限无法做 QAT；(b) 原型验证不同位宽效果；(c) 一次性批量部署
- 局限：(a) ≤4 bit 精度损失显著（尤其推理/长上下文任务）；(b) 依赖校准数据分布；(c) 无法像 QAT 那样通过训练适应特定任务
- MC-MoE：使用 GPTQ PTQ + expert-wise 混合精度策略，验证 MoE 场景下 PTQ 可接近 QAT 效果

涉及论文标题：
- MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More

---

## Expert Parallelism

术语是什么？
Expert Parallelism（专家并行）是 MoE 模型训练中一种专门针对 expert 参数的并行策略。核心思想：将 MoE layer 中的 E 个 expert 均匀分布到 N 个 GPU 设备上（每个设备持有 E/N 个 expert），每个 token 通过 Router 被分配到目标 expert 所在的设备上进行计算。与 Data Parallelism（每个设备持有完整模型副本）和传统 Model Parallelism（按层切分）不同，Expert Parallelism 按 expert 粒度切分模型参数，配合 All-to-All 通信原语实现 token 的跨设备路由。从 MoE 架构提出（Shazeer et al. 2017）到 Switch Transformer（Fedus et al. 2021），Expert Parallelism 已成为 MoE 训练的标准分布式策略。

从算法pipeline角度拆解术语：
Expert Parallelism 的 MoE layer 前向计算流程（以 top-2 gating 为例，每层 N 个设备）：

```
输入: T_I ∈ R^{B, M}   // B tokens, model dim M
输出: T_O ∈ R^{B, M}

每层执行:
1. Router: G(T_I) = softmax(W_g · T_I) ∈ R^{B, E}
2. Top-K gating: 选择每 token 的 top-2 expert
3. 统计每个 expert 收到的 token 数 → input_split_sizes[N], output_split_sizes[N]
4. // 第一个 All-to-All: Dispatch
   T_DI = All-to-All(T_I, input_split_sizes, output_split_sizes)
5. // Expert FFN 计算（每个 device 独立执行其 local experts）
   for each received chunk c:
       T_M[c] = GeLU(Linear1(T_DI[c]))
       T_DO[c] = Linear2(T_M[c])
6. // 第二个 All-to-All: Combine
   T_O = All-to-All(T_DO, output_split_sizes, input_split_sizes)
```

关键内存分析（MPMoE Equation 1-3）：
- Model States: M_ms = 4 * (E*M + 2*H*M)
- Activations: M_act = 4*B*M + B*H
- Temporary Buffers: M_buf = B*M + B*H

Expert Parallelism 通过分布式存储 expert 参数解决了 model states 的内存瓶颈，但 All-to-All 通信成为新的性能瓶颈。

术语一般如何实现？如何使用？
- 实现框架：FastMoE（PyTorch 原语）、DeepSpeed-MoE（分层 All-to-All + 自定义 CUDA kernel）、FasterMoE（pipeline + expert shadowing）、MPMoE（微批次 pipeline + 内存复用）。
- 关键考量：(a) Router 计算量极小但需全局同步 token 分布；(b) 不均匀的 expert 负载导致 All-to-All 出现 straggler；(c) Expert Parallelism 通常与 Data Parallelism 组合。
- MPMoE 的改进：沿 batch 维度切分 micro-batch 进行 pipeline（保留 NCCL All-to-All 优化），从固定 granularity 升级为自适应 granularity。

MixNet 从网络架构视角进一步揭示了 EP 通信的三个关键特性（基于生产环境 128 H800 GPU 的 Mixtral 8×7B 训练测量）：
1. **时间非确定性**：每个 training iteration 中 token-specific expert activation 导致 all-to-all 通信矩阵在 iterations 间显著变化。即使使用 load balancing loss，traffic matrix 的 sparsity 始终存在。
2. **空间非均匀性**：每个 traffic matrix 是非均匀的——仅有少数 GPU 对（heavy hitters）之间有大流量通信，大部分 GPU 对之间通信量很小或为零。
3. **强局部性**：仅同一 MoE block 内的 expert 层需要 all-to-all 通信——不同 PP stage 的 expert 层不直接通信。

这三项特性是 MixNet 设计区域可重构 OCS 高带宽域的理论基础。EP 的 all-to-all 通信量占比显著：Mixtral 8×7B 中 EP 占 30%（TP 占 60%），LLaMA-MoE 和 Qwen-MoE 中 EP 超过 80%。EP 的 all-to-all 通信占据了 33%-55% 的总训练迭代时间（400 Gbps 网络）。

- **MoE Parallel Folding 中的 EP**：该论文将 EP 从 DP 的子组中解放出来，允许 EP 折叠到 Attention 层的 TP/CP/DP 任意子组中。这使得 EP 的 All-to-All 通信可以限制在 NVLink 域（节点内 450 GB/s）而非 InfiniBand 域（节点间 400 Gbps），显著降低通信开销。同时通过统一 token dispatcher 支持 EP 与 ETP 的任意组合。配置示例（Mixtral 8x22B, 128 H100）：Attention TP=2, CP=1, DP=8, PP=8；MoE ETP=1, EP=8, PP=8（MoE 层纯 EP 不做 TP）。EP=8 时 8 个完整 expert 分布在 8 GPU，GEMM 效率最高。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MixNet: A Runtime Reconfigurable Optical-Electrical Fabric for Distributed Mixture-of-Experts Training
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

**MoESD 的 EP+SD 兼容性分析**：在 EP 配置下，expert 分布到多 GPU，N(t) 和 Texp 不受影响（仅影响每 GPU 持有的 expert 子集），因此 MoESD 的理论分析仍然有效。MoE FFN 仍占显著处理时间 → memory-boundness 效应在端到端性能中可观测。值得注意的是，在大量 EP GPU 配置下，小 batch 时的 SD 低效问题可能消失——因为 EP 提供的额外聚合内存带宽使验证阶段计算增量更容易被吸收。

---

## Dynamic Expert Duplication (动态专家复制)

术语是什么？
Dynamic Expert Duplication 是一种 MoE 推理负载均衡技术。在多 GPU Expert Parallelism 设置中，当 token-to-expert 分布倾斜（skewed）时，部分 GPU 上的热门 expert 处理的 token 远超平均，成为 compute 和 communication 的 bottleneck。Expert Duplication 将热门 expert 的权重复制到 underloaded GPU 上，使多个 GPU 共同处理同一 expert 的 token，从而将集中的负载分散到多 GPU。关键在于"动态"——token 分布随输入 batch 变化，需要 predictor 预先预测每层的 expert 激活分布，然后据此决定哪些 expert 需要被复制到哪些 GPU。

从算法pipeline角度拆解术语：
MoE-GPS 的 Expert Duplication 在每层 Transformer Block 的 Attention 之前执行（Algorithm 1）：
```
输入: token_expert_map f (T tokens → E experts), GPU_memory M, 
     初始 placement P, 最大复制数 C_max
输出: 均衡 placement P, token→GPU dispatch d

1. d(t) = min{g | (f(t), g) ∈ P}     // 将 token 分配到持有其 expert 的 GPU
2. L[g] = |{t | d(t) = g}|            // 每 GPU 负载
3. while max(L) - min(L) > 1:         // 不均衡时迭代
4.   g_h = argmax(L); g_c = argmin(L)
5.   Δ = ceil((L[g_h] - L[g_c]) / 2)
6.   e* = most_popular_expert_on(g_h)  // overloaded GPU 上 token 最多的 expert
7.   if (e*, g_c) ∉ P and copies(e*) < C_max and params(e*) ≤ M[g_c]:
8.     copy_weights(e* → g_c)          // NVLink/PCIe 传输 ~47MB (Mixtral expert FP16)
9.     P = P ∪ {(e*, g_c)}
10.    reassign first Δ tokens of e* from g_h to g_c
11. update L[g_h], L[g_c]
return P, d
```
核心贪心策略：每次迭代取负载最大 GPU 上 token 最多的 expert，将其复制到负载最小的 GPU，并转移一半差值 token 到新 GPU。当所有 GPU token 数差异 ≤1 时停止。Expert 复制通信（~0.1ms per expert over NVLink 3.0）可与 Attention 计算重叠。

术语一般如何实现？如何使用？
已有工作：MoE-Prediction [Cong et al. 2024] 首次提出预测 token-to-expert 分布指导 expert placement；Prophet [Wang et al. 2023] 提出细粒度动态 expert duplication 策略；FlexMoE [Nie et al. 2023] 和 FasterMoE [He et al. 2022] 也使用 expert duplication。实现要点：(1) predictor 频率——从每 batch [He et al., Prophet] 到每 10 分钟 [DeepSeek-V3]，权衡 overhead 和有效性；(2) placement 开销——expert 权重传输可通过与 Attention 计算 overlap 隐藏（batch size≥16, seq_len≥2K 时 PCIe 也可隐藏）；(3) 内存容量——每 GPU 的 expert 复制数受显存限制（C_max 参数）。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

---

## Distribution-Only Prediction (仅分布预测)

术语是什么？
Distribution-Only Prediction 是 MoE-GPS 提出的一种轻量级 expert 预测策略。与预测每个 token 具体路由到哪个 expert（Token-to-Expert Prediction）不同，Distribution-Only Prediction 仅预测 coarse-grained 的 token 分布比例（如 Expert 1 将收到 75% 的 tokens），不指定哪些具体 token 去哪个 expert。它使用 Multinomial Distribution + MLE (Maximum Likelihood Estimation) 建模每层 MoE 的 expert 激活概率：$\hat{p}_i^l = n_i^l / N$，其中 $n_i^l$ 为训练集中第 l 层 expert i 被激活的总次数，N 为总 token 数。该预测是 offline 完成的，运行时 zero overhead。

从算法pipeline角度拆解术语：
```
# Offline 训练阶段
for layer l in 1..L:
    for batch in training_data:
        expert_assignments = MoE_Router(hidden_states, layer=l)
        for expert e in 1..E:
            n_e[l] += count(expert_assignments == e)
    p_hat[e][l] = n_e[l] / total_tokens[l]  # MLE

# 推理阶段（每层）
predicted_distribution = p_hat[:, l]           # 各 expert 预期 token 比例
target_tokens_per_gpu = total_tokens / G      # 均衡目标
P, d = ExpertDuplication(f, predicted_distribution, M, C_max)
# Token 仍通过 All-to-All Scatter 随机分发（通信未优化）
tokens = AllToAllScatter(tokens, d)
output = FFN_Experts(tokens, P)  # compute 已均衡化
```
关键特性：(1) zero prediction overhead（offline MLE 估计，运行时仅查表）；(2) 仅均衡 FFN compute，不减少 All-to-All 通信开销；(3) 高 skewness 时 estimation error 增大（因冷门 expert 训练样本不足），但整体性能仍优于无 prediction baseline；(4) 在 skewness≤1.4 时比 Token-to-Expert 最佳配置快 23%。

术语一般如何实现？如何使用？
MLE 假设 expert selection 是 i.i.d. Multinomial draws（因为 expert activation 主要受 local token features 影响）。Error rate 定义为 $|\hat{p} - p| / (1/E)$。实验在 Mixtral 8×7B 上 MMLU（skewness=1.39, error=1.80%）、Alpaca Eval（skewness=1.40, error=0.98%）、SST2（skewness=1.99, error=16.00%）上验证。适用场景：low skewness 或 high-bandwidth interconnect (NVLink) 的推理，此时通信不是瓶颈。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

---

## Token-to-Expert Prediction (Token级专家预测)

术语是什么？
Token-to-Expert Prediction 是传统 MoE expert 预测策略：直接预测每个 token 将被路由到哪个 expert，即 exact token-to-expert mapping。预测结果用于跳过 All-to-All 的 Scatter 阶段（Direct Routing），同时均衡 FFN compute 和通信。代价是 predictor 本身的 inference overhead。MoE-GPS 将 expert selection 建模为多分类问题，探索三类 predictor：(a) Probability Model——始终选全局频率最高的 expert；(b) Conditional Probability Model——按 token index 或 position index 条件化选择；(c) Neural Networks——FFN（2 层 MLP: 4096→128→64→8 logits）和 LSTM with Sparse Attention（2-layer LSTM, hidden 64 + sparse attention + residual connection）。

从算法pipeline角度拆解术语：
```
# 训练 Predictor（以 FFN 为例）
# 输入: token_embeddings ∈ R^{seq_len × 4096}（Mixtral）
# 输出: expert_logits ∈ R^{seq_len × 8}（每层独立 head）
class FFNPredictor:
    def forward(x):
        h = ReLU(Linear(x, 4096→128))
        h = ReLU(Linear(h, 128→64))
        return Linear(h, 64→8)  # 每层一个 head

# 推理阶段：Predictor 插入每层 Attention 之前
for layer l in 1..L:
    predicted_experts[l] = Predictor[l](hidden_states)  # overhead
    P, d = ExpertDuplication(predicted_experts[l], ...)
    # Direct Route: 跳过 Scatter，token 直接发送到目标 GPU
    tokens = DirectRoute(tokens, predicted_experts[l])
    output = FFN_Experts(tokens, P)
```
Accuracy-overhead trade-off 呈 U 形：更高 accuracy → 更好的 load balancing → 更好系统性能，但也 → 更大 predictor overhead → 降低 net 收益。最优配置通常在 intermediate accuracy 处。高 skewness 时 prediction 更容易（accuracy 更高、overhead 更低），sweet spot 向高 accuracy 方向移动。

术语一般如何实现？如何使用？
Prediction error 建模三种场景：Optimistic（errors 不影响 load balancing）、Typical（errors 均匀分布在各 GPU，最负载 GPU 处理 (1+ε)×avg_tokens）、Pessimistic（errors 集中在单 GPU，最坏 N×(1+ε)×avg_tokens）。默认使用 Typical。适用场景：high skewness（prediction 更容易）+ low-bandwidth interconnect（PCIe）时，节省的 communication 超过 predictor overhead。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing

---

## MoE Load Skewness (MoE负载倾斜度)

术语是什么？
Skewness 是 MoE-GPS 定义的 MoE 推理负载倾斜度量指标：最热门 expert 收到的 token 数除以平均每 expert token 数（均衡分布时）。即 $\text{skewness} = \frac{\#\text{tokens in the most popular expert}}{\#\text{average tokens per expert}}$。Skewness=1 表示 perfect balance，skewness=3 表示最热门 expert 收到 3 倍于平均的 token。Skewness 仅影响 FFN compute 和 communication runtime：bottleneck GPU 的 FFN compute 时间被 scale by skewness；All-to-All 通信时间也被 scale by skewness：$(N-1)·skewness/N^2$（N=GPU 数）。

从算法pipeline角度拆解术语：
Skewness 直接影响 prediction strategy 选择（MoE-GPS Figure 7）：
- Low skewness (1.0-1.5)：Distribution-Only 由于 zero overhead 优势明显
- High skewness (1.5+)：Token-to-Expert 的高 accuracy 优势逐渐超越 predictor overhead
- Skewness 越高 → Distribution-Only 的 estimation error 越大（因冷门 expert 样本少，error 占比高）→ 效果略降
- Skewness 越高 → Token-to-Expert 的 predictor 越容易达到高 accuracy（分布更可预测）→ overhead/accuracy 比更优

实验数据：MMLU skewness=1.39, Alpaca Eval skewness=1.40, SST2 skewness=1.99（Mixtral 8×7B, seq_len=512）。

术语一般如何实现？如何使用？
Skewness 可通过 training data 的 expert activation 统计离线测量。在 MoE-GPS 中，Distribution-Only Prediction 对 skewness 敏感的 error rate 通过 testset 的 empirical probability vs. trainset MLE estimation 的差异计算：$|\hat{p} - p| / (1/E)$。SST2（skewness=1.99）error rate=16%，远高于 MMLU（1.39, error=1.80%），因为高度倾斜导致冷门 expert 训练 token 不足。Skewness 也是 MoE-GPS simulator 选择最优 prediction strategy 的关键输入参数。

涉及论文标题：
- MoE-GPS: Guidelines for Prediction Strategy for Dynamic Expert Duplication in MoE Load Balancing
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

### MoETuner 补充

MoETuner 从硬件 placement 角度处理 skewness：即使 routing 本身就存在 skewness（某些 expert 确实处理更多 token），通过 ILP 求解将高频和低频 expert 混合分配到同一 GPU，使各 GPU 总 token 处理量趋向均衡（min Σ|T_{c,l} - T̄_l|）。这避免了 Megatron-LM contiguous placement 下 skewness 导致的严重 GPU 计算不平衡（如 layer 14 中 GPU0 处理 64% token）。在 Mixtral-8x7B 上单节点 token processing tail latency 减少 36%。

### Orders in Chaos 补充

本论文对 4 个 large-scale MoE 模型 (235B-1000B) 的 >24,000 requests 进行了系统性的 expert activation skewness profiling，发现了以下新特征：
- **量级差异**：部分 expert 被激活的频率是平均值的 16 倍以上（Llama 4 layer 7），远超早期小模型如 Mixtral 的 skewness。
- **Task/Language 特异性**：不同 MMLU subject（biology, history, math 等共 57 个）的 top-10 popular experts 既有共性（horizontal bright lines 表示跨 subject 共同热门 experts），也有明显差异。同一问题使用英文 vs 中文时，即使内容相同，仅 ~5-6 个 experts 保持 popular，且只有 2 个与英文 MMLU 的 top experts 重叠。这揭示了 language 对 expert selection 的显著影响。
- **系统影响**：skewness 导致的 workload imbalance 在 wafer-scale GPU 上尤为严重——热门 expert 所在 die 可能处理 16× 于平均的请求量，而冷门 expert 所在 die 基本空闲。论文提出 Popular Expert Decentralization (Insight 4) 策略——duplicate/replicate 热门 experts 到多个 compute unit 以均衡负载。
- **与 Expert Co-activation 的关系**：top 10% expert pairs 占 60-80% 总激活量（DeepSeek V3, Qwen3），说明 skewness 不仅体现在 single expert 层面，也体现在 expert pair 层面。

---

## Dropless-MoE (dMoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dropless-MoE (dMoE) 是 MegaBlocks (MLSys 2023) 提出的 MoE 层计算方法，核心思想是将 MoE 层的 expert 计算从 batched matrix multiplication 重新表述为 block-sparse matrix multiplication，从而**从根本上消除 token dropping** 的需求。传统 MoE 实现（GShard, Switch Transformer, Tutel）为了满足 batched GEMM 的形状约束（要求所有 expert 分配相同的 token 数量），在 token 分配不均衡时强制丢弃超出 expert capacity 的 token 或 zero-padding 不足的 expert batch。dMoE 将 expert 计算视为 variable-size block diagonal matrix multiplication（图 3C）：每个 expert 的 token batch 被分解为多个 128×128 固定 block，仅计算实际分配的 token 行（sparse non-zero blocks），天然支持负载不均衡的 token 分配。dMoE 从算法层面消除了 capacity_factor 超参数和 token dropping/padding 的 tradeoff，已被用于训练 Mixtral 8×7B 和 DeepSeek V2 等模型。

从算法pipeline角度拆解术语：
dMoE 的 forward pass（图 4）：
```
输入: x (num_tokens, hidden_size)
输出: y (num_tokens, hidden_size)

# (1) Router: Assign tokens to experts (与标准 MoE 相同)
indices, weights = router(x)  # top-k greedy selection

# (2) 构造 block-sparse matrix topology（关键差异）
# 将 variable-size expert batches 分解为 128×128 blocks
topology = make_topology(indices)
# topology 描述图 3C 的 block-sparse matrix:
#   - row_offsets[i]: expert i 的 blocks 在 non-zero list 中的起始偏移
#   - column_idxs[b]: block b 对应的 expert (决定使用 w1 的哪一列)
#   - row_idxs[b]: block b 在输出中的行坐标 (用于 SDD)

# (3) 按 expert 分组 tokens + padding 到 128 倍数
x_permuted = padded_gather(x, indices)  # (total_tokens_padded, hidden_size)

# (4) Expert 计算: Sparse = Dense × Dense (第一层)
# w1.shape: (hidden_size, ffn_hidden_size * num_experts)
intermediate = sdd(x_permuted, w1, topology)  # block-sparse output

# (5) Dense = Sparse × Dense (第二层)
# w2.shape: (ffn_hidden_size * num_experts, hidden_size)
y_permuted = dsd(intermediate, w2)  # dense output

# (6) Un-permute + scaling
y = padded_scatter(y_permuted, indices)
return y * weights
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- dMoE 通过自定义 block-sparse GPU kernels 实现（§5.1）：扩展 CUTLASS 2.5 实现 SDD、DSD、DDS 操作，使用 Hybrid Blocked-CSR-COO 编码（§5.1.3）和 Transpose Indices（§5.1.4）。
- 开源实现：https://github.com/databricks/megablocks (Apache-2.0)，通过 `pip install megablocks` 安装。集成于 Megatron-LM，支持 data/expert/pipeline parallelism。
- 两种计算后端：Sparse MLP（block-sparse via STK，Ampere GPU A100）和 Grouped MLP（grouped GEMM，Hopper GPU H100 推荐）。
- 已被工业界广泛采用：Mistral AI 的 Mixtral 8×7B 训练使用 MegaBlocks，vLLM 集成 MegaBlocks 进行 MoE 推理，DeepSeek V2 训练也基于此技术栈。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MoEBlaze: Breaking the Memory Wall for Efficient MoE Training on Modern GPUs

**MoEBlaze 补充**：MoEBlaze 从不同角度实现 dropless MoE——通过轻量级索引数据结构（expert_token_indices, token_expert_indices 等）替代 per-expert materialized buffer，天然支持 variable-length expert batches（每个 expert 处理的 token 数量任意，由实际路由决定而非固定 capacity 限制）。与 MegaBlocks 的 block-sparse 方法（将 variable-size batch 分解为 128×128 fixed blocks）不同，MoEBlaze 的索引方法无需 block decomposition 和 padding。

---

## Expert Capacity / Capacity Factor / Token Dropping in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert capacity 是标准 MoE 实现中的核心约束概念：为满足 batched matrix multiplication 的等大小输入约束，限制每个 expert 最多处理的 token 数量。具体定义为：expert_capacity = (num_tokens / num_experts) × capacity_factor。capacity_factor 是一个超参数乘数：capacity_factor=1 表示每个 expert 刚好能处理均匀分配下的 token 数；capacity_factor>1 增加容量以降低丢 token 概率。Token dropping 发生在某 expert 被分配超过其 capacity 的 token 时——超出部分直接被丢弃，不参与 expert 计算，依赖 residual connection 传递信息（图 1）。为避免丢 token，Tutel 引入 dynamic capacity factor（运行时设为刚好不丢 token 的最小值），但 MegaBlocks 实验显示可能需要 capacity_factor 高达 11（Hwang et al. 2022），且 capacity_factor 尖峰在训练中不可预测地出现。

从算法pipeline角度拆解术语：
标准 MoE 的 token dropping/padding 流程：
```
输入: indices (num_tokens,)  # 每个 token 的 expert 分配
      capacity_factor       # 超参数
输出: padded_batches        # 可用于 batched GEMM 的 expert inputs

# 1. 计算 capacity
expert_capacity = ceil(num_tokens / num_experts * capacity_factor)

# 2. 按 expert 分组 tokens
for expert e in 1..num_experts:
    batch[e] = tokens[indices == e]

# 3. Dropping & Padding
for expert e in 1..num_experts:
    if len(batch[e]) > expert_capacity:
        batch[e] = batch[e][:expert_capacity]  # Truncate/DROP
        dropped_tokens += len(batch[e]) - expert_capacity
    elif len(batch[e]) < expert_capacity:
        batch[e].pad_to(expert_capacity)  # Zero-padding

# 4. Batched GEMM
# All experts computed with same batch size = expert_capacity
outputs = batched_gemm(batches, expert_weights)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 典型实现：GShard (Lepikhin et al. 2020) 引入 capacity_factor 概念；Switch Transformer (Fedus et al. 2022) 广泛使用 token dropping；Tutel (Hwang et al. 2022) 使用 dynamic capacity factor 在运行时自适应调整容量。
- 核心 tradeoff：(a) capacity_factor 小 → 更多 token 被丢弃 → 模型质量下降（MegaBlocks 实验显示 capacity_factor=1 时 loss 改善仅 0.15 vs dense，而不丢 token 改善 0.26）；(b) capacity_factor 大 → 大量 zero-padding → 计算和内存浪费（某些模型需要 capacity_factor 高达 11，MoE 层计算量增加 >2×）。
- MegaBlocks 通过 block-sparse 重表述从根本上消除 capacity_factor 参数和 token dropping/padding 问题。
- 在分布式 expert parallelism 中，token dropping 还影响 All-to-All 通信效率：不均匀的 expert 负载导致 straggler 问题。

- **Sub-sequence Dropping（子序列丢弃）**：MoE Parallel Folding 论文提出的一种 token dropping 优化策略。在进行 token dropping 决策时，仅基于当前 rank 处理的子序列（sub-sequence）的本地 logits 做决策，而非跨所有 rank 收集完整序列的 logits（full-sequence dropping）。这避免了 AllGather 通信开销。论文经验验证：sub-sequence dropping 不影响模型收敛（training/validation loss 曲线与 MCore v0.9 对齐）。对 token-dropless 训练范式（如 MegaBlocks），Dispatcher 直接按 expert 分配所有 token，无容量约束。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- Mixture-of-Experts with Expert Choice Routing
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

---

## Load Balancing Loss (MoE Auxiliary Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load balancing loss（负载均衡损失）是 MoE 训练中的辅助损失函数，用于激励 Router 在 token 分配时产生均匀的 expert 负载。定义为 L_aux = num_experts × Σ_e f_e × P_e，其中 f_e 是分配给 expert e 的 token 比例，P_e 是 Router 分配给 expert e 的平均概率。最小化此损失鼓励两个目标的均匀性：(1) 实际分配的 token 数（f_e），(2) Router 的 softmax 概率（P_e）。由 Shazeer et al. (2017) 首次提出，Switch Transformer (Fedus et al. 2022) 广泛使用。该损失以权重 α 加到主任务损失上。虽然能改善负载均衡，但 MegaBlocks 论文指出即使使用负载均衡损失，token routing 仍然高度不均衡（Hwang et al. 2022 也证实此事）。

从算法pipeline角度拆解术语：
Load balancing loss 的计算：
```
输入: router_probs (num_tokens, num_experts)  # softmax 输出
      expert_indices (num_tokens,)             # top-k 选择
输出: L_aux (scalar)

# 1. 计算每个 expert 的实际 token 比例
f_e = count(indices == e) / num_tokens

# 2. 计算每个 expert 的平均路由概率
P_e = mean(router_probs[:, e])

# 3. 负载均衡损失
L_aux = num_experts * sum_e (f_e * P_e)

# 4. 总损失
L_total = L_task + α * L_aux  # α 典型值为 0.01
```
L_aux 在负载不均衡时较大（某些 expert f_e 高 P_e 高），在均匀时最小（所有 f_e = 1/E, P_e = 1/E 时 L_aux = 1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 PyTorch 中通过 `torch.scatter_add` 或 `torch.bincount` 统计每个 expert 的 token 数，再计算 f_e 和 P_e 的点积。
- α 通常设为 0.01（Switch Transformer）。过大的 α 会干扰主任务学习，过小则效果不足。
- 除了计算效率价值，负载均衡损失还确保所有 expert 在训练中看到足够的 token 以避免退化——某些 expert 可能长时间收不到 token 而停止接收梯度更新（Zhou et al. 2022）。
- 替代方法：BASE layers (Lewis et al. 2021) 将路由建模为线性分配问题保证完美均衡；Expert Choice Routing (Zhou et al. 2022) 反转路由方向让 expert 选择 token。

**Nexus 的发现：自适应 Router 对负载均衡损失不敏感。** Nexus 的 ablation（Figure 7）对比了 load balancing loss factor α=0.05 和 α=0.0005 的效果。结果：线性 router 的 upcycled MoE 在 α=0.0005 时性能下降约 2%（相对），而 Nexus 的自适应 domain-embedding router 在两个 α 值下性能几乎不变。原因：Nexus router 的 expert embedding 始终基于域表示（e_i = P_r(d_i)），即使负载均衡损失权重极低，域语义本身也能提供稳定的 token 分配——专家嵌入的内在域语义充当了隐式正则化。这使得 Nexus 在实际部署中无需精细调优 load balancing loss 超参数。

**MoLE 的特殊情况：全激活训练的天然均衡性。** MoLE 在所有 experts 始终激活（不做 top-K 稀疏选择）且接收梯度的全激活训练范式下，Router 不会面临 collapse 风险。因此 MoLE 仅使用 language modeling cross-entropy loss，无需任何 auxiliary loss。MoLE 的 ablation（Table 4）显示，添加 load balance loss 和 z-loss 后模型性能反而下降（MoLE-16E 160M: LM loss only AVG 41.9 → +load_balance 41.7 → +z-loss 40.6），因为 auxiliary loss 使优化目标与推理需求不对齐。

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning (per-layer prompt-level routing, f_i = proportion of prompts assigned to expert i, p̂_i = mean probability mass, λ_lb=1e-2)
- Mixture-of-Experts with Expert Choice Routing
- Mixture of Lookup Experts
- MoH: Multi-Head Attention as Mixture-of-Head Attention
- MoLA: MoE LoRA with Layer-wise Expert Allocation (per-layer load balancing loss: L_aux = Σ_j N_j · Σ_e f_e^j · P_e^j, where N_j is layer j's expert count; follows Switch Transformers formulation)
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts (adaptive domain-embedding router is robust to low α = 0.0005, while linear router drops ~2%)

**MoH 中的应用**：MoH 将 load balance loss 应用于 attention head 级别的路由（而非 FFN expert）。L_b = Σ_{i=h_s+1}^{h} P_i · f_i，仅对路由头计算（共享头始终激活无需均衡）。P_i = mean(Softmax(W_r·x_t)[i-h_s])，f_i = mean(token选择head i 的指示函数)。β=0.01 对所有任务（ViT/DiT/LLM）通用。

---

## Micro-Batch Pipeline Parallelism for MoE Training

术语是什么？
Micro-Batch Pipeline Parallelism for MoE 是将 GPipe 风格的微批次流水线引入 MoE 训练层的技术。将 MoE 层的三个阶段（All-to-All Dispatch S → Expert 计算 C → All-to-All Collect R）类比为 GPipe 的模型层，将 mini-batch 沿 batch 维度切分为 n 个 micro-batch，使不同 micro-batch 的三个阶段在多个 CUDA stream 中并行执行。与 FasterMoE 沿 device 维度切分不同，MPMoE 沿 batch 维度切分：(1) 保留 NCCL All-to-All 的集体通信优化；(2) pipeline granularity n 不受 device 数限制；(3) 交替调度 S 和 R stage 增强内存访问局部性。

从算法pipeline角度拆解术语：
MPMoE 的 pipeline 调度（以 n=4 为例）：

```
时间轴 →
Stream_comm:  S(0)---|R(0)---|S(2)---|R(2)---|
Stream_comp:         |C(0)---|C(1)---|C(2)---|C(3)---|
// S(i): dispatch, C(i): expert FFN, R(i): collect
// S 和 R 交替调度（利用 NCCL 双向通信，增强内存局部性）
```

Pipeline granularity n 的最优选择（MPMoE Figure 14）：B < 8k 时 n=2，8k ≤ B ≤ 22k 时 n=4，B > 22k 时 n=8。n 随 B 单调递增——过粗的 pipeline 导致 insufficient overlap，过细的 pipeline 导致 kernel launch overhead 和 GPU under-utilization。

术语一般如何实现？如何使用？
- 实现要点：(a) 沿 batch 维度切分：`torch.split(T_I, B/n, dim=1)`；(b) 每个 micro-batch 的 stage 在不同 CUDA stream 上异步执行；(c) 自适应 granularity：MPMoE-pb 通过 profile-based search（Algorithm 1），MPMoE-pm 通过性能模型。
- 适用场景：MoE 训练中 batch size 较大（>256 tokens/GPU）且通信/计算比高的场景。
- 局限性：n>8 时 kernel launch overhead 超过 overlap 收益；需要足够 GPU 资源支持多 stream 并发。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---

## Memory Reuse Strategy in Distributed Training

术语是什么？
Memory Reuse Strategy 是 MPMoE 提出的通过共享 buffer 减少 MoE 训练 activation memory 占用的技术。核心观察：在 micro-batch pipeline 中，不同 partition 的 tensors（T_DI, T_M, T_DO）在不同时间点激活，产生"memory bubbles"，因此可共享同一个物理 buffer。原本 n 个 partition 各需独立 buffer，共享后仅需 1 份，内存从 O(n·B·M) 降至 O(B·M)。共享 buffer 意味着前向中被覆写的 tensors 需在后向中恢复，MPMoE 提出 4 种恢复策略（S1-S4），组合 CPU offload、通信重放和重计算三种机制。

从算法pipeline角度拆解术语：
4 种策略的恢复方法（Table 2）：

| 策略 | T_DI 恢复方式 | T_M 恢复方式 | 适用场景 |
|------|-------------|------------|---------|
| S1 | CPU offload | CPU offload | N 小（计算瓶颈） |
| S2 | 通信重放 | CPU offload | N 小-中 |
| S3 | CPU offload | 重计算 | N 中-大 |
| S4 | 通信重放 | 重计算 | N 大（通信瓶颈） |

内存节省公式（Equation 5-6）：ΔM_act = ΔM_buf = B * (2M*(n-2)/n + H*(n-1)/n)，n=8 时达 ~38% 节省，最高 vs FasterMoE 节省 53%。

术语一般如何实现？如何使用？
- 类似技术：Gradient Checkpointing（重计算所有 activations）、ZeRO-Offload（offload optimizer states）、vDNN（offload activations）。MPMoE 的创新在于 buffer 共享 + 选择性恢复（组合 offload/recompute/communication replay）联合应用于 MoE pipeline 场景。
- 实现要点：(a) pinned memory 支持异步 D2H/H2D；(b) 不同 CUDA stream 上 overlap；(c) 通信重放依赖原始 T_I 保留在内存中。
- 局限性：(a) PCIe 带宽限制（V100 ~32 GB/s vs HBM ~900 GB/s）；(b) 实际内存节省约理论上限的 95%（Figure 12）。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---

## Activation Memory Footprint in Distributed Training

术语是什么？
Activation Memory Footprint Analysis 是对分布式训练中 activation tensors（前向中间结果，需保留至后向计算梯度）占用 GPU DRAM 的定量分析。MPMoE 将 MoE 训练的内存分解为三部分：(1) Model States M_ms（parameters + gradients + Adam momentum/variance）；(2) Activations M_act（T_I, T_DI, T_M, T_DO, T_O）；(3) Temporary Buffers M_buf（后向中间梯度峰值）。揭示了：随 batch size B 增大，M_act 和 M_buf 占比快速上升（Figure 2），成为限制大 batch size 训练的主要瓶颈。

从算法pipeline角度拆解术语：
内存公式（MPMoE Section 2.2.2）：
- M_ms = 4*(E*M + 2*H*M)：params + grads + momentum + variance（×4 for Adam）
- M_act = 4*B*M + B*H：5 个主要 tensors（T_I, T_DI, T_M, T_DO, T_O 各 (B,M) 或 (B,H)）
- M_buf = B*M + B*H：后向中相邻两个 tensor 的梯度峰值
- Pipeline 后 M_act^pipe = M_buf^pipe = 4*B*M + B*H（总量不变，时间分布改变）
- Memory Reuse 后 ΔM_buf = ΔM_act = B*(2M*(n-2)/n + H*(n-1)/n)
- 总节省率 φ = (ΔM_act + ΔM_buf)/(M_ms + M_act^pipe + M_buf^pipe)

实际案例：1.5B GPT-2 (seq_len=1K, batch_size=32) 约需 60GB GPU 内存，~80% 为 activations。

术语一般如何实现？如何使用？
- ZeRO（Rajbhandari et al. 2020）处理 model states 瓶颈（partitioning across devices），但未处理 activations。
- Gradient Checkpointing（Chen et al. 2016）丢弃中间 activations 后向重计算，以计算换内存。
- vDNN（Rhu et al. 2016）将 activations offload 到 CPU。
- MPMoE 的贡献：将三类方法（partitioning + recomputation + offloading）系统组合应用于 MoE pipeline 场景，并给出量化的内存分析公式指导策略选择。

涉及论文标题：
- MPMoE: Memory Efficient MoE for Pre-Trained Models With Adaptive Pipeline Parallelism
- MPipeMoE: Memory Efficient MoE for Pre-trained Models with Adaptive Pipeline Parallelism

---

## Layer-wise Synchronized Execution in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Layer-wise Synchronized Execution（逐层同步执行）是 MoE 推理的固有执行模型：transformer 的前向传播严格遵守 layer-by-layer 的顺序，每一层内 attention 和 expert 计算之间存在同步屏障。对于每个 layer ℓ：(1) 所有 data-parallel AW 独立执行 attention 计算（生成 token embeddings）；(2) 每个 AW 通过 gating network 选 top-k experts，将 token embeddings 发送到对应 EWs；(3) 所有 AW 等待所有选中 experts 返回输出（同步屏障），加权聚合后才进入 layer ℓ+1。EW 侧也遵守此模式——按 layer-wise batch 聚合同层同 expert 的 tokens，完成当前层后才前进到下一层。这一同步模式是保证 GPU 批处理效率的关键（避免碎片化的 per-request 执行导致 GPU 低利用率），但也意味着任一 worker 故障会导致整条 pipeline 在同步屏障处停滞。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
for layer ℓ in 1..L:                    // L 层 transformer
    // === All AWs (data-parallel) ===
    for each AW in parallel:
        // Attention computation
        Q, K, V = W_Q@h, W_K@h, W_V@h
        attn_out = softmax(Q@K^T/√d_k) @ V
        h_mid = LayerNorm(h + attn_out)
        
        // Gating (select top-k experts)
        gate_scores = softmax(W_gate @ h_mid)
        top_k_experts = TopK(gate_scores, k=2)
        
        // Dispatch to EWs (scatter)
        for each expert e in top_k_experts:
            rdma_send(EW[e], token_embedding=h_mid, weight=gate_scores[e])
    
    // === Synchronization Barrier (ALL AWs wait for ALL experts) ===
    
    // === All EWs (expert-parallel) ===
    for each EW in parallel:
        for each hosted expert e:
            batch = gather_tokens_from_AWs(layer=ℓ, expert=e)
            expert_out = FFN(batch)  // Linear1 → GeLU → Linear2
            for each token in batch:
                rdma_send(token.source_AW, expert_out[token] * token.weight)
    
    // === Synchronization Barrier ===
    
    // === Back to AWs ===
    for each AW:
        h_next = aggregate_expert_outputs(received_outputs)
        h = LayerNorm(h_mid + h_next)
```

关键属性：同步屏障是**全局的**——任一 worker 慢（straggler 或故障）都会阻塞整条 pipeline；层间完全串行——layer ℓ+1 必须等 layer ℓ 完成所有 token 的所有 expert。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 MegaScale-Infer 和 Tarragon 等解耦系统中，同步屏障由 AW-EW 间的请求-响应匹配隐式实现，不依赖显式的 NCCL barrier。
- Tarragon 的创新在于**部分打破**此同步屏障：EW 侧自愈允许 EW 在收到足够 AW 输入时即开始计算（不等所有 AW），将同步条件从 "all" 放松到 "enough"。
- 约束：batch 大小必须至少达到 GPU 效率拐点（NVIDIA A100 约 256-512 tokens），因此不能无限放松。
- 自适应：expert batch threshold 可根据 expert kernel 的 throughput-knee-point 动态配置。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---

## EGNN (E(n) Equivariant Graph Neural Network)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
EGNN（E(n) Equivariant Graph Neural Network，Satorras et al. 2022）是一种对 E(n) 群（n 维欧几里得空间的旋转、平移、反射变换）保持等变性的图神经网络。与标准 GNN 仅学习 node representation 不同，EGNN 同时更新 node embedding hᵢ 和 3D 坐标 xᵢ，且坐标更新满足等变性：对输入坐标施加任何 E(n) 变换（旋转/平移/反射），输出坐标会自动施加相同的变换。核心机制是每层中：(1) 边消息 eᵢⱼ = ϕ(hⱼ, xⱼ, hᵢ, xᵢ) —— 输入包含坐标的 L2 距离差，ϕ 为 MLP；(2) node 表示更新 hᵢᐟ = COM^H(hᵢ, AGG({eᵢⱼ}))；(3) 坐标更新 xᵢᐟ = COM^X(xᵢ, AGG({eᵢⱼ})) —— 等变性的关键在于坐标更新仅使用边消息聚合（不直接操作坐标），因此自然保持 E(n) 等变性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: X⁽⁰⁾ ∈ R^{N×3} (初始坐标), H⁰ = MLP(X⁽⁰⁾) (node embedding), 边集 E
for l in 1..L:                            // L 层 EGNN
    for each edge (i,j) in E:
        // 计算相对距离平方（等变特征）
        d_ij² = ||x_i^{l-1} - x_j^{l-1}||²
        // 边消息网络 ϕ（MLP），输入: h_i, h_j, d_ij²
        e_ij^l = ϕ(h_j^{l-1}, h_i^{l-1}, d_ij²)
    
    for each node i:
        // 聚合邻居边消息
        m_i = Σ_{j∈N(i)} e_ij^l / |N(i)|    // mean aggregation
        // 更新 node embedding
        h_i^l = h_i^{l-1} + MLP_h(m_i)     // 残差更新
        // 更新坐标（等变）
        Δx_i = Σ_{j∈N(i)} (x_i^{l-1} - x_j^{l-1}) · MLP_x(e_ij^l)
        x_i^l = x_i^{l-1} + Δx_i           // 残差更新

输出: H^L, X^L → Decoder → X̂⁽ᵗ⁾
```
坐标更新公式 Δxᵢ = Σ(xᵢ−xⱼ)·MLP_x(eᵢⱼ) 保证等变性：坐标差 (xᵢ−xⱼ) 本身是等变的（平移不变、旋转变换一致），乘以标量 MLP 输出后仍保持等变。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/vgsatorras/egnn（官方 PyTorch 实现）
- 实现要点：ϕ 网络输出可以是 scalar（用于坐标更新权重），node/coordinate 的 COM 操作支持 sum、mean、concat 等多种聚合方式
- 数据集：广泛用于 N-body 物理模拟（Spring, Charged）、分子动力学（MD17, QM9）、人体运动捕捉（Motion Capture）
- 在 LEGO 中的使用：EGNN 作为 Graph MoE 框架的基础 expert 模型。多个 EGNN 专家（同架构不同参数 θ¹...θᴷ）并行预测，由 LLM Judge 选最优。当基础 expert 改为 EGNO 或 Radial Field 时，LEGO 框架保持不变
- 局限：EGNN 仅保证 E(n) 等变性（非 SE(3)），对包含手性的分子场景可能不够严格；坐标更新仅依赖 L2 距离会丢失角度信息
- 相关模型：EGNO（Fourier 神经算子扩展）、Radial Field（仅操作坐标的 E(n) 模型）、SE(3)-Transformers（更严格的 3D 等变）、TFN（Tensor Field Networks）

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Graph Mixture-of-Experts (Graph MoE) for Dynamical Systems

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Graph Mixture-of-Experts（Graph MoE）是将 Mixture-of-Experts 架构应用于图神经网络以提升动态系统建模泛化能力的技术。与 LLM MoE（路由函数通常基于输入 token 的可学习 gate）不同，LEGO 提出的 Graph MoE 使用预训练 LLM 作为 context-aware routing function：K 个同构 GNN experts（如 EGNN）各自处理相同的输入图 G 和初始状态 X⁽⁰⁾，生成 K 个候选预测；LLM 基于环境上下文（系统参数、物体状态、连接关系）选择最合适的 expert。路由权重通过 one-hot + label smoothing（选中 expert α，其他 (1-α)/(K-1)）实现软性选择。Diversity-enhanced contrastive loss 确保不同 expert 学习互补的动力学模式。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: 图 G=(V,E), 初始状态 X⁽⁰⁾, 环境上下文 C, K 个 GNN experts {f₁,...,f_K}
输出: 预测状态 X̂⁽ᵗ⁾

// 1. 所有 experts 并行预测
for k in 1..K:
    H^k = f_k(G, X⁽⁰⁾)                    // GNN 前向（L 层消息传递）
    X̂^k = Decoder(H^k)                     // 各 expert 独立预测

// 2. LLM Judge 选择 expert
prompt = HierarchicalPrompt(C, X⁽⁰⁾, E)   // 系统/物体/边 三层 prompt
chosen = LLM(prompt, {X̂^k}_{k=1}^K)       // LLM 评估并选择

// 3. Label Smoothing 权重
ω(k) = α           if k == chosen          // Eq.7
     = (1-α)/(K-1) otherwise

// 4. 加权组合（Eq.8）
for each node i:
    h̄_i = Σ_k ω(k) · h_i^k
    x̂_i⁽ᵗ⁾ = Decoder({h̄_i})
```

关键设计的独特之处：
- Routing 不是 learnable MLP gate（如标准 MoE），而是预训练 LLM 的 zero-shot 推理。LLM 不需要微调，利用世界知识理解环境语义
- 与环境无关的 MoE 路由仅依赖输入数据 → LEGO 的 LLM 路由额外利用了文本化的环境元信息
- Diversity loss 保证 expert specialization：不同 expert 专家的激活表征被推远（contrastive），同一 expert 的表征被拉近

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/jdp22/LEGO.git
- expert 模型选择：EGNN（默认）、EGNO、Radial Field 均可作为基础 expert，仅需同构 GNN 架构
- 超参数：K=5 experts（默认，经验最优），α label smoothing 系数，τ contrastive loss temperature
- LLM：Llama 3.1 8B（大模型更优但小模型也可用），temperature=0 时推理阶段性能最好
- 训练：交替优化（每隔若干 epoch 更新 LLM routing weights，内部循环更新 expert 参数），Adam optimizer (lr=0.0005)
- 适用场景：动态系统预测、物理模拟、分子动力学、人体运动预测等含环境变化的图结构预测任务
- 与标准 LLM-MoE 的区别：LLM-MoE 中 gate 是小型可学习 MLP → LEGO 中 gate 是预训练 LLM，利用外部知识推理环境
- 局限：(1) LLM 推理成本较高（虽可通过交替优化降低调用频率）；(2) LLM 对专业科学领域（如分子动力学）的理解可能有限；(3) expert 数量过多时 LLM 判断困难

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## LLM-as-a-Judge for Expert Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LLM-as-a-Judge 是一种利用大语言模型的推理能力进行决策/评估而非直接生成内容的使用范式。在 LEGO 中，LLM 不作为预测器（不直接输出动态系统的未来状态），而是作为"裁判"（routing function）：接收系统环境的三层文本化描述和 K 个 GNN expert 的候选预测，经过逐步推理（观察初始条件 → 分析各 expert 预测的物理合理性 → 选择最一致的结果）选出最合适的 expert。这一设计与 LLM-as-Predictor（直接生成预测）形成对比：LEGO 实验（Table 5）显示 LLM Forecasting 的 MSE 为 6.42 而 LEGO 为 0.0072（~890× 差距），且 LLM Forecasting 推理时间更长（1.27s vs 0.44s per sample）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// LLM-as-Predictor (baseline, 不可靠)
prompt = "Predict future positions of 5 balls in 3D space given initial state X⁽⁰⁾..."
output = LLM.generate(prompt)  // 可能输出错误格式/幻觉/不合理数值

// LLM-as-a-Judge (LEGO 方法)
prompt = """
  System: 5 balls connected by springs with k=1.0.
  Object: Ball 0 pos=(0.1,0.2,0.3) vel=(0.01,-0.02,0.01)
  ...
  Edge: ball 2 connects ball 0, ball 1, ball 3.
  
  Expert A prediction: (positions at t=10) ...
  Expert B prediction: ...
  
  Question: Which expert's prediction is most physically plausible?
"""
decision = LLM.reason(prompt)  // step-by-step 推理 → "Expert B because..."
```

LLM Judge 的推理过程（Case Study, Figure 5）：
1. 分析初始条件：各物体的位置、速度、受力方向
2. 检查物理一致性："Are the objects moving in the expected directions?"
3. 评估预测范围："Are the predictions within a reasonable range?"
4. 综合判断：选择与物理规律最一致的 expert 预测

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) LLM 通过 API（如 OpenAI API 或本地 Llama 3.1 推理）调用；(b) prompt 包含环境描述 + 各 expert 预测（数值以 digit token 编码）；(c) LLM 输出选择结果被解析为 one-hot index，经 label smoothing 转为 soft weight
- LLM Judge 的优势：(a) zero-shot 能力——无需在特定环境上微调 LLM；(b) 常识推理——可利用预训练中的物理世界知识；(c) 可解释性——LLM 可输出逐步推理过程（Case Study）
- LLM Judge 的局限：(a) 推理成本（交替优化降低调用频率）；(b) 复杂科学场景的理解深度有限；(c) 大规模 expert 选择退化（K>15 时性能下降）；(d) 对 LLM temperature 敏感（低 temperature 更稳定）
- 其他 LLM-as-a-Judge 应用：代码评审、文本质量评估、RLHF 中的 reward model、多 agent 辩论等
- 与 learnable gate 的对比：learnable gate（如 MoE Transformer）仅依赖输入数据分布 → LLM Judge 利用外部语义知识（世界模型）理解环境变化，泛化能力更强

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Hierarchical Prompt Engineering for Dynamical Systems

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Prompt Engineering for Dynamical Systems 是 LEGO 提出的将动态系统环境信息转化为 LLM 可理解文本的三层提示设计方法。三个层次分别捕获不同粒度的信息：(1) System Level（系统级）：系统的物理背景、参数（如弹簧系数 k、电荷量 q）及高层语义描述（如"The force on the balls are significant"）；(2) Object Level（物体级）：每个物体的初始状态（位置向量 (x,y,z) 和速度向量 (vx,vy,vz)），数值直接作为 digit token（遵循 Gruver et al. 2024 的做法）；(3) Edge Level（边级）：物体间的连接/交互关系，如"ball 2 connects ball 0, ball 1, ball 3"。三层信息构成对环境的完整文本化描述，使 LLM 能理解分布偏移的本质并据此选择合适的 expert。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 *Charged* 数据集为例的三层 prompt 结构（来自论文 Figure 6）：
```
// System Level（系统参数 + 物理背景）
System Description: There are 5 charged particles moving in a 3D space.
The particles interact via Coulomb's law: F = k * q1 * q2 / r².
The charge of each particle is: [1, -1, 1, -1, 1].
The interaction strength k = 1.01.
The system evolves from time step 30 to 40.

// Object Level（逐物体初始状态）
Object 0: initial position (0.12, -0.34, 0.56), initial velocity (0.01, 0.02, -0.01)
Object 1: initial position (-0.23, 0.45, -0.11), initial velocity (-0.02, 0.01, 0.03)
...

// Edge Level（连接/交互关系）
Edge Information: In this charged system, every particle interacts 
with every other particle (fully connected graph).
```

Ablation 实验（Table 4）验证了三层 prompt 的必要性：
- V1（仅 system level）：MSE = 0.761
- V2（system + edge）：MSE = 0.735
- V3（完整三层 prompt）：MSE = 0.728
Edge level 信息（连接关系文本化）的贡献最显著（V1→V2 降幅 > V2→V3）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) 系统级 prompt 由环境参数模板填充（如 k=1.01 → "spring constant = 1.01"）；(b) 物体级 prompt 由初始状态矩阵 X⁽⁰⁾ 逐行转换为文本描述；(c) 边级 prompt 由邻接矩阵转换为自然语言连接描述
- 数值编码策略：数值以 digit 形式作为 token（如"0.12"作为单个 token），而非科学记数法或量化表示。遵循 Gruver et al. (2024) 证明 LLM 可直接处理数值序列
- 设计原则：(a) 环境变化相关信息优先（系统参数、边界条件）；(b) 空间结构显式文本化（连接关系）；(c) 数值直接作为 token（保持精度）
- 扩展性：可适配不同物理系统（将 Coulomb/F=ma 等物理规则替换为对应领域的专业描述）
- 局限：(a) 对大规模系统（数百个物体）prompt 可能过长（超过 LLM context window）；(b) 需要人工设计每类系统的 prompt 模板；(c) 科学领域需要领域知识辅助 prompt 设计

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Diversity-Enhanced Objective (Contrastive Loss for MoE Experts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diversity-Enhanced Objective 是 LEGO 提出的确保 Graph MoE 中不同 expert 学习互补动力学模式的对比学习损失函数。核心思想：对于每个节点 i，同一 expert k 在不同训练样本中产生的激活表征 hᵢᵏ 应当相互靠近（正样本对），而不同 expert 产生的表征应当相互远离（负样本对）。通过此损失，各 expert 被迫专业化于不同的动力学模式（如某些 expert 擅长高能量场景、某些擅长低能量场景），从而为 LLM Judge 提供多样化的候选预测。损失函数为 InfoNCE 的变体（Eq. 9-10）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: K 个 experts 在训练集上所有节点的激活表征
输出: diversity loss ℒ_div

定义: S_i^k = {样本中节点 i 被 expert k 激活的所有表征}
      S_i   = ∪_{k=1}^K S_i^k  (所有 experts 的表征集)

for each node i and expert k:
    选取两个不同的表征 h, h̃ ∈ S_i^k (正样本对)
    sim_pos = exp(h · h̃ / τ)                    // τ: temperature
    
    计算所有表征的相似度和
    sim_all = Σ_{h' ∈ S_i} exp(h · h' / τ)      // 包含正样本和负样本
    
    ℓ_i^k = -(1/C) · log(sim_pos / sim_all)     // Eq.9: 对比损失

ℒ_div = 1/(KN) · Σ_k Σ_i ℓ_i^k                  // Eq.10: 平均所有 node/expert

// 最终损失（Eq.11）
ℒ = ℒ_mse + ℒ_div
```

损失函数的直观解释：
- sim_pos/sim_all 大 → 正样本在语义空间中靠近（同一 expert 的表征一致）→ 损失小
- sim_pos/sim_all 小 → 正样本被负样本淹没关系（不同 expert 的表征混在一起）→ 损失大
- 优化目标：不同 expert 学习到可区分的表征空间，使 LLM Judge 有真正的"选择余地"

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) 在训练过程中维护每个 expert 的激活表征集合 S_i^k（或使用 mini-batch 内的动态对比）；(b) 温度参数 τ 控制表征空间的距离敏感度（小 τ → 严格区分，大 τ → 宽松）；(c) C 为归一化常数
- 理论基础：基于 Contrastive Learning（Chuang et al. 2020, MoCo/SimCLR 等）在多 expert 场景的扩展。类似 MoELora（Luo et al. 2024）中 contrastive learning 引导 expert 专业化
- 在 LEGO 中的作用：没有 diversity loss 时，多个 experts 可能 converge 到类似的解（mode collapse），diversity loss 确保 expert specialization 是真正的多模态覆盖
- 局限：(a) 需要足够大的训练集来构建有意义的正样本对；(b) τ 的选择影响分类粒度；(c) 在极少数 expert（K=2）场景下 diversity 收益有限

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Alternative Optimization for LLM-Guided MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Alternative Optimization（交替优化）是 LEGO 提出的联合训练 LLM routing weights 和 graph expert 参数的优化策略。由于 LLM（Llama 3.1 8B）的推理成本远高于 GNN expert 的梯度下降更新，LEGO 不每步都更新 routing weights，而是每隔若干 epoch（E 个 epoch）才调用 LLM 重新评估 routing，在此期间固定 routing weights 仅优化 GNN expert 参数。此设计将 LLM 调用次数从 per-batch 降低到 per-epoch-interval，大幅降低训练开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
输入: 训练集 D, 预训练 LLM (Llama 3.1 8B), K 个 GNN experts {θ^1,...,θ^K}
输出: 训练好的 expert 参数

初始化所有 expert 参数
while not converged:                        // 外层循环
    // === Step A: 更新 Routing Weights（LLM 推理，低频）===
    for each sample in D (or subset):
        提取 hierarchical prompt
        各 expert 前向生成 candidate predictions
        LLM Judge 推理 → 选择最佳 expert
        // Label Smoothing (Eq.7)
        ω(k) = α if k == chosen else (1-α)/(K-1)
    
    // === Step B: 优化 Expert 参数（梯度下降，高频）===
    for epoch in 1..E:
        for each batch:
            使用当前 routing weights ω 组合 experts
            计算 ℒ = ℒ_mse + ℒ_div
            梯度下降更新 {θ^1,...,θ^K}（Adam, lr=0.0005）
```
关键设计：
- LLM 推理仅在 Step A 执行（低频），Step B 不涉及 LLM
- 间隔 E 的选择：E 过小 → LLM 调用过多成本高；E 过大 → routing weights 过时影响训练
- LLM 不需要微调（保持零样本能力），仅用作 routing function

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现方式：(a) LLM 调用通过 API（如 HuggingFace transformers 加载 Llama 3.1 8B 本地推理）；(b) Routing weights 在每个交替周期存储为哈希表（sample_id → one-hot expert choice）；(c) Step B 中固定 routing weights 使得 expert 训练等价于标准多任务学习
- 计算开销：LLM 推理成本 vs GNN 训练成本的比例决定了最优交替频率
- 类似方法：(a) EM（Expectation-Maximization）算法（固定 routing 优化 expert → 固定 expert 优化 routing）；(b) K-Means 的交替优化（分配聚类 → 更新中心）；(c) GAN 的交替训练（生成器/判别器）
- 优势：(a) 避免 LLM per-step fine-tuning 的高成本；(b) LLM 保持原始的零样本泛化能力（不被动态系统数据 overfit）；(c) 训练稳定性好
- 局限：(a) 论文未具体说明交替间隔 E 的值；(b) Step A 可能需要 sub-sampling 来处理大数据集

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Distribution Shift in Dynamical Systems

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Distribution Shift in Dynamical Systems（动态系统中的分布偏移）指训练和测试环境之间的数据分布不匹配，通常由系统参数 ξ（如弹簧系数 k、电荷量 q、分子类型）或初始状态分布的变化引起。形式化定义：设系统演化由 dX/dt = F(X, ξ) 决定，训练和测试的数据分布分别为 P_train(X⁰, ξ) 和 P_test(X⁰, ξ)。当 P_train ≠ P_test 时（环境参数从 ξ~P_train(ξ) 变为 ξ~P_test(ξ)），传统数据驱动方法（EGNN/EGNO 等）因仅从训练数据隐式学习分布而性能显著下降。LEGO 通过 LLM 显式理解环境参数（文本化 ξ）来选择合适的 model expert，缓解此问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
LEGO 论文中考虑的三种环境变化类型（以 Spring 为例）：
```
// 类型1: Hard environment（更强的物理系数）
训练: strength k = 1.0, start=30, end=40
测试: strength k = 1.10（弹簧更硬，移动更剧烈）
结果: EGNN MSE = 0.112 → EGNN+LEGO MSE = 0.078 (↓30.4%)

// 类型2: Soft environment（更弱的物理系数）
训练: strength k = 1.0, start=30, end=40
测试: strength k = 0.90（弹簧更软，移动更缓慢）
结果: EGNN MSE = 0.118 → EGNN+LEGO MSE = 0.114 (↓3.4%)

// 类型3: Temporal Shift（不同时间窗口）
训练: start=30, end=40
测试: start=20, end=30
结果: EGNN MSE = 0.115 → EGNN+LEGO MSE = 0.072 (↓37.4%)
```

跨分子迁移（OOD，MD17）：
```
训练分子: salicylic acid（9个重原子）
测试分子: naphthalene（10个碳原子，无氧原子）
// 分子拓扑和化学性质完全不同
EGNN MSE = 0.320 → Radial Field+LEGO MSE = 0.186 (↓41.9%)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 传统应对方法：(a) Domain Generalization（域泛化）：从多域数据中学习域不变特征；(b) Domain Adaptation（域适配）：利用目标域无标签数据做分布对齐；(c) Test-Time Adaptation：推理时在线调整模型参数
- LEGO 的创新：(a) 用 LLM 的常识推理替代数据驱动的域泛化——LLM 被告知"k=1.10"可推理出"弹簧更硬"并选择相应的 expert；(b) MoE 的 expert specialization 天然适合多域——不同 expert 可专门适配不同环境模式
- 评估 benchmark：Spring（Hard/Soft/Temporal Shift）、Charged（Hard/Soft/Temporal Shift + 多种 strength）、MD17（跨分子迁移）、Motion（跨受试者/运动类型迁移）
- 局限：(a) LLM 对环境的理解限于 prompt 中的信息；(b) 对于从未见过的全新物理系统类型（如训练集全是弹簧、测试集是电荷），LLM 的判断也可能不准确

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Label Smoothing for MoE Expert Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Label Smoothing for MoE Expert Routing 是 LEGO 在 LLM Judge 选择 expert 后应用的软性权重分配策略。传统做法：LLM 选中的 expert 权重=1，其他 expert 权重=0（硬路由），但这会导致：(1) 错误选择累积——若 LLM 判断错误，整个预测完全依赖错误 expert；(2) 训练不稳定——梯度仅流向被选中的 expert，其他 expert 无信号更新。LEGO 的 label smoothing 方案（Eq. 7）：选中 expert 的权重为 α ∈ (0,1)，其余 (K-1) 个 expert 平分剩余权重 (1-α)/(K-1)。这种软性分配使所有 expert 都接收到一定梯度信号，同时保留选中 expert 的主导地位。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// 硬路由 (hard routing)
if LLM_chosen == k:
    weights = [0, ..., 0, 1, 0, ..., 0]    // one-hot, 仅 expert k 激活
else:
    // 其他 expert 无梯度信号

// Label Smoothing 软路由 (LEGO, Eq.7)
α = 0.8  // 选中 expert 的主导权重
chosen = LLM_choice
for k in 1..K:
    if k == chosen:
        ω(k) = α
    else:
        ω(k) = (1-α) / (K-1)              // 其余 expert 共享

X̂⁽ᵗ⁾ = Σ_k ω(k) · Decoder(h_i^k)         // Eq.8: 软性组合

// 梯度流到所有 expert：∂ℒ/∂θ^k ∝ ω(k)，所有 expert 都被更新
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 理论基础：Labels Smoothing（Müller et al. 2019）原本用于分类任务（将 one-hot label 平滑为 soft label 防止过拟合）。LEGO 将其应用于 MoE routing weights 的 slot
- α 的选择：α 接近 1 → 接近硬路由（更依赖 LLM 判断）；α 接近 1/K → 均匀路由（忽略 LLM 判断）。论文未明确给出 α 的具体值，由实验调参确定
- 作用机制：(a) 缓解 LLM 偶然判断错误的影响（错误 expert 仍有少量权重，不至于完全错误）；(b) 防止 expert collapse（仅部分 expert 持续被更新，其他 expert 停滞）；(c) 与 diversity loss 协同（diversity loss 促进 expert 分化，label smoothing 确保所有 expert 被更新）
- 与 top-k routing 的对比：LLM MoE 常用 top-2 routing（选 2 个 expert 各给部分权重）→ LEGO 的 smoothing 选择所有 expert（K=5），但主导 expert 权重远大于其他
- 局限：α 作为超参数需调优；α 过大则 smoothing 效果弱，α 过小则丢失 LLM Judge 的选择信号

涉及论文标题：
- Marrying LLMs with Dynamic Forecasting A Graph Mixture-of-expert Perspective

---

## Failure Cost Model for MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Failure Cost Model（故障开销模型）是 Tarragon 提出的量化单 worker 故障对 MoE 推理影响的数学模型。模型定义了两个核心指标：(1) **Inference Stall Time (T_stall)**：pipeline 无法产生新 token 的时长，等于 worker 重启时间 + 重放所有 prefill 层 + 重放已产生 decoding 层的时间；(2) **Re-execution Cost (G)**：以 GPU-time（执行时间 × GPU 数量）衡量的浪费计算量。模型以 decoded-token index i 和 frontier layer ℓ 为参数，区分三种场景：monolithic worker failure、decoupled AW failure、decoupled EW failure。核心发现：decoding 阶段 fault 的开销远超 prefill（64 tokens 解码后 ~19× 高于 128-token prefill），因此 decoding 是优化的主目标。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
模型公式（以 decoupled AW failure 为例）：

```
T_stall(ℓ, i) ≈ T_w + L · t_pre + [(i-1)L + ℓ] · t_dec

其中:
  T_w: worker 重启时间 (包含进程启动 + CUDA context init + 权重加载 + 通信栈初始化)
  t_pre: 单层 prefill 平均执行时间
  t_dec: 单层 decoding（单 token）平均执行时间
  L: transformer 层数
  i: 当前正在生成的 token index（1-indexed）
  ℓ: 故障发生时正在执行的 layer（1 ≤ ℓ ≤ L）

G(ℓ, i) ≈ M · [L · g_pre + ((i-1)L + ℓ) · g_dec]

其中:
  M: worker 总数
  g_pre: 单 worker 处理单层 prefill 的 GPU-time
  g_dec: 单 worker 处理单层 decoding 的 GPU-time
```

实测参数（Mixtral-8×7B, MegaScale-Infer 配置, 16 GPUs, 8 AWs + 8 EWs, GCP H200）：
- T_w = 18.5s, t_pre = 2.18ms, t_dec = 0.85ms
- g_pre = 0.006, g_dec = 0.0022

对于 EW failure（stateless）：
```
T_stall ≈ T_w + t_dec       // 仅 worker 重启 + 单层 expert 重算
G ≈ g_dec                   // 仅单 EW 的单层 expert 计算
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 用途：(a) 指导恢复策略设计——既然 decoding 开销远大于 prefill，Tarragon 重点优化 decoding 阶段 recovery（KV cache checkpointing）；(b) 指导 resource provisioning——EW 故障的 G 代价小可用 shadow expert 快速恢复，AW 故障需更重的 checkpointing 机制；(c) 推广到其他 MoE 系统——任一 worker 故障开销可由此模型参数化推广。
- 模型简化假设：忽略 warm cache 效应、通信/计算 overlap、straggler 影响；假设 worker 完全负载均衡。
- 与 Tarragon 设计的对应：D1（worker 级故障域）解决 M· 乘数；D2（self-healing）解决 T_w 等待；D3（KV cache checkpointing）解决 (i-1)L 重放开销。

涉及论文标题：
- Making MoE-based LLM Inference Resilient with Tarragon

---

## BTX (Branch-Train-Mix)

术语是什么？
BTX（Branch-Train-Mix, Sukhbaatar et al. 2024）是一种将多个领域特化的 dense expert LLM 合并为统一 Mixture-of-Experts (MoE) 模型的方法。流程分三步：(1) **Branch**：从 pretrained base model 分支，对各分支在特定领域数据上做 Continual Pretraining (CPT) 得到多个 dense expert；(2) **Train/Merge**：将各 expert 的非 FFN 层（embedding、attention、normalization、head）通过 unweighted averaging 合并为一套共享参数，FFN 层保持独立作为 MoE expert，插入 MLP router 做 token-level top-K routing；(3) **Mix/Fine-tune**：合并后 MoE 在混合所有数据源的数据集上 fine-tune 40B tokens 以训练 router 并恢复因参数干扰导致的性能损失。

从算法pipeline角度拆解术语：
```
输入: base model θ_b, l 个 dense expert [θ₁,...,θₗ]（均从 θ_b 分支 CPT 得到）
输出: fine-tuned MoE model θ_m

// Step 1: 非 FFN 层平均合并
for each non-FFN layer (embedding, attention, norm, head):
    θ_m[layer] = 1/l * Σ_{i=1}^{l} θ_i[layer]     // unweighted averaging

// Step 2: FFN 层保留 + 插入 router
θ_r = random_init_MLP(hidden_dim, l)

// Step 3: MoE layer forward
v = input_token_embedding
FF_MoE(v) = Σ_{i=1}^{K} SoftMax(top-K(θ_r · v)) · FF_i(v)

// Step 4: Fine-tuning 40B tokens on mixed data
```

术语一般如何实现？如何使用？
- 基于标准 PyTorch 分布式训练。BTX 论文使用 8-expert MoE，训练在 64-128 GPUs。
- 局限：(1) 所有 expert 需同架构（从同一 ancestor 分支）；(2) 简单平均忽略参数干扰（sign conflict, magnitude disparity）；(3) fine-tuning 通信开销大。
- MergeME 以 BTX 为 baseline，改进为 Dare/Ties 替代平均、PPL 路由替代 training router、projector 方法支持异构合并。

**Nexus 对 BTX 的改进**：Nexus 在 BTX 的 upcycling 框架基础上做了三处改进：(1) **Router 设计**：BTX 使用标准线性 router W_r ∈ R^{h×n}（从零训练），Nexus 用基于域嵌入投影的自适应 router e_i = P_r(d_i)（P_r 为 2-layer SwiGLU MLP），路由概率 s_i = softmax(x · e_i)；(2) **Shared Expert**：BTX 将 seed model FFN 作为一个普通 routed expert，Nexus 将其作为 always-activated shared expert；(3) **扩展能力**：BTX 不支持扩展，Nexus 可通过计算新域嵌入 → 投影 → appendix FFN 高效添加新 expert。实验显示 Nexus 在 470M/2.8B 规模上分别优于 BTX-style MoE (Linear Router) 2.1%/1.6%（相对），扩展新 Code expert 时相对 gain 18.8%。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

---

## Task Vector in Model Merging

术语是什么？
Task Vector（任务向量）定义为 fine-tuned expert 模型参数与 base 模型参数之间的差值：τᵢ = θ_b − θᵢ，表示从 base 到领域特化 expert 的参数空间位移方向。在模型合并中，通过组合多个 task vector 并加回 base：θ_m = θ_b + λ · Σ τᵢ，可获得多领域能力。MergeME 在 MoE 场景中使用 task vector：(a) 计算各 expert 的 τᵢ → Dare/Ties 处理 → τ_m → θ_m；(b) Task Vector Routing（附录 C）——计算输入梯度 g_inf 与 τᵢ 的余弦相似度作为路由决策。

从算法pipeline角度拆解术语：
```
τ_i = θ_b - θ_i                                // task vector 定义
τ_m = Σ τ_i                                     // 合并
θ_m = θ_b + λ · τ_m                            // 加回 base
// MergeME MoE: 仅非 FFN 层参与，FFN 层保持独立
```

术语一般如何实现？如何使用？
- 开源：mergekit（https://github.com/arcee-ai/mergekit）提供 Ties/Dare/Task Arithmetic。
- Task Vector Routing：g_inf = ∇_{θ_b} L(x_inf)，路由权重 = SoftMax(top-K(Sim(g_inf, τᵢ)))。实验显示 PPL 路由优于 Task Vector Routing（Table 3: 8.08 vs 7.05）。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---

## Parameter Interference in Model Merging

术语是什么？
Parameter Interference（参数干扰）是合并多个 fine-tuned 模型时，因 task vector 间的冲突导致的性能下降。MergeME Figure 2 识别三类干扰：(1) **Sign Conflict**：τ₁[j] > 0, τ₂[j] < 0 → 平均后抵消；(2) **Magnitude Disparity**：大 magnitude 被小值稀释；(3) **Redundancy**：接近零的参数不携带信息但占用参数空间。Dare 和 Ties 通过 drop 和 sign alignment 缓解。MergeME Figure 4 验证 attention 层 task vector 的余弦相似度也较低（~0.1-0.3），说明 BTX 的"attention 层可直接平均"假设不成立。

从算法pipeline角度拆解术语：
```
// Sign Conflict 示例:
τ_math[j] = +0.8, τ_code[j] = -0.6
Average: 0.1（几乎归零）  // Ties 方案: 保留 +0.8, 丢弃 -0.6

// Magnitude Disparity 示例:
τ_math[j] = 0.9, τ_know[j] = 0.01
Average: 0.455（大参数被稀释）  // Dare 方案: drop 0.01 后 rescale
```

术语一般如何实现？如何使用？
- 检测：task vector 余弦相似度分析（MergeME Figure 4）。
- 缓解：(a) Dare: random drop + rescale；(b) Ties: trim + elect sign + disjoint merge。
- MoE 特殊性：仅共享层（非 FFN）受干扰，范围比 dense 合并更受控。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---

## Dare Merging for MoE (Drop and Rescale)

术语是什么？
Dare（Drop and Rescale, Yu et al. 2024）通过随机 drop + rescale 解决参数干扰：(1) 随机将 task vector 中 (100-p)% 参数置零；(2) rescale 保留参数 × 1/(0.01·p) 补偿幅值损失；(3) 求和得到 τ_m。MergeME 首次将 Dare 从 dense 模型合并扩展到 MoE 合并——仅对非 FFN 层应用 Dare，FFN 层保持独立。MergeME 设置 p=80%, λ=1/3。Table 1: Dare merging MoE avg 12.86 vs BTX 11.72 (+9.72%)，尤其显著提升在 TriviaQA（30.68 vs 25.10）。

从算法pipeline角度拆解术语：
```
输入: task vectors [τ₁,...,τₗ], p=80%, λ=1/3
for each τᵢ:
    mask = random_bernoulli(prob=p/100)
    τᵢ[mask==0] = 0                 // random drop 20%
    τᵢ = τᵢ / (0.01 * p)            // rescale
τ_m = Σ τᵢ
θ_m = θ_b + λ · τ_m                // 仅应用于非 FFN 层
```

术语一般如何实现？如何使用？
- 开源：mergekit 库提供 `dare_linear` 方法。随机 drop 的 seed 可固定以保持可复现性。
- 在 MergeME MoE 场景仅应用于非 FFN 层。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---

## Ties Merging for MoE (Trim, Elect Sign, Merge)

术语是什么？
Ties（Trim, Elect Sign, and Merge, Yadav et al. 2024）通过三步解决参数干扰：(1) **Trim**：每个 task vector 中 drop bottom (100-p)% 最小 magnitude 参数（消除冗余）；(2) **Elect Sign**：每个参数位置确定总 magnitude 更大的符号方向为"主导符号"（解决 sign conflict）；(3) **Disjoint Merge**：仅累加与主导符号方向相同的 task vector 值。MergeME 首次将 Ties 应用于 MoE 合并，设置 p=80%, λ=1/3。Table 1: Ties avg 12.52 vs BTX 11.72 (+6.94%)。

从算法pipeline角度拆解术语：
```
// Step 1: Trim — threshold = magnitude_percentile(|τᵢ|, 100-p); 重置小值为 0
// Step 2: Elect Sign — for each j: dom[j] = ±1 based on total magnitude
// Step 3: Disjoint Merge — τ_m[j] += τᵢ[j] only if sign(τᵢ[j]) == dom[j]
θ_m = θ_b + λ · τ_m
```

术语一般如何实现？如何使用？
- 开源：mergekit 库提供 `ties` 方法，参数 `density`(=p/100) 和 `weight`(=λ)。
- Ties vs Dare: Ties 显式解决 sign conflict；Dare 处理 magnitude disparity 更轻量。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---

## Perplexity-based Expert Routing

术语是什么？
Perplexity-based (PPL) Expert Routing 是 MergeME 提出的无需训练的 MoE 路由启发式。利用 perplexity 衡量各 expert 对输入的不确定度：PPL(x|θᵢ) = exp(−1/t · Σ log P(xⱼ|x_{<j}, θᵢ))。PPL 低 → confidence 高 → 路由权重大：α = SoftMax(top-K(1/PPL₁, ..., 1/PPLₗ))。仅需一次额外 forward pass（远小于 inference 时 generate 多 token 的多次 forward）。MergeME Table 2 验证 PPL 路由能有效导向领域专家（GSM8K → Math 43%, HumanEval → Code 43%）。Table 3: separate attention + PPL routing avg 8.08 vs merge attention + PPL routing 7.32 vs Dare Dense 7.11。

从算法pipeline角度拆解术语：
```
输入: prompt x (t tokens), experts [θ₁,...,θₗ]
for each expert i:
    PPL_i = exp(-1/t * Σ log P(x_j | x_{<j}, θ_i))
    conf_i = 1 / PPL_i
α = SoftMax(top-K(conf_1, ..., conf_l))     // routing weights
output = Σ α_i · expert_i.forward(x)          // 加权组合
```

术语一般如何实现？如何使用？
- 一次 no_grad forward pass 计算 log_softmax → PPL。开销 ≈ O(1) forward vs O(generate_tokens) forward。
- 局限性：(a) 跨领域输入可能选错 expert；(b) 所有 expert PPL 接近时区分度差。
- PPL 路由优于 Task Vector Routing（Table 3: 8.08 vs 7.05）。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---

## Heterogeneous MoE Merging with Projectors

术语是什么？
Heterogeneous MoE Merging with Projectors 是 MergeME 首个支持不同架构 expert 合并为 MoE 的方法。核心组件：(1) 共享 Embedding/Head：各 expert embedding/head padding 零对齐到 d_m 后平均；(2) Proj-in/Proj-out：每个 expert 配备一对随机初始化 MLP——Proj-in: R^{d_m}→R^{dᵢ}, Proj-out: R^{dᵢ}→R^{d_m}；(3) Sequence-level Router：因异构 attention 不兼容，整句 token embedding 平均后通过 MLP router 做序列级路由。

从算法pipeline角度拆解术语：
```
d_m = max(d₁,...,dₗ)
emb_shared = avg(padded(emb_i, d_m)), head_shared = avg(padded(head_i, d_m))
Proj_in[i]: d_m → d_i, Proj_out[i]: d_i → d_m  // 随机初始化
// Forward:
avg_e = mean(e₁,...,e_t), α = SoftMax(top-K(θ_r · avg_e))
for selected expert k:
    r_k = Proj_out[k](Expert_k(Proj_in[k](e)))
output = head_shared(Σ α_k · r_k)
```

术语一般如何实现？如何使用？
- 局限：(a) 不合并 attention → 参数多（~4B vs ~3.7B）；(b) 合并 embedding 可能导致 router 偏好同架构 expert（Figure 6）。
- MergeME Table 4: MoE w/ Math TinyLlama avg 13.34 vs 3-expert MoE 10.54。

涉及论文标题：
- MergeME: Model Merging Techniques for Homogeneous and Heterogeneous MoEs

---

## Expert Merging for MoE Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Merging（专家合并）是一种后训练 MoE 模型压缩技术，通过将多个功能相似或冗余的 expert 合并为更少的 expert 来减少 MoE 模型的总参数量和内存占用。核心原理：MoE 层中 N 个独立 expert（每个含 SwiGLU FFN 的三组权重矩阵 W_D, W_U, W_G）被聚类为 M 个组（M < N），每组内的 expert 通过某种合并策略融合为一个新的 expert，路由权重相应聚合。最早由 M-SMoE（Li et al. 2023, ICLR 2024）提出——基于 expert 使用频率识别 dominant experts、按路由 logits 相似度聚类、簇内按使用频率加权平均参数、可选 low-rank decomposition + structural pruning 进一步压缩。MergeMoE（Miao et al. 2025, arXiv 2510.14436）从理论上改进：将合并重新解释为"输出合并"视角下的优化问题，通过最小二乘法优化维度缩减矩阵 T1（Moore-Penrose 伪逆闭式解），并严格证明了使用频率作为合并权重的最优性。

从算法pipeline角度拆解术语：
```
// MergeMoE 完整压缩流程 (N→M experts)
// 1. 频率统计：calibration 数据上前向推理，统计 f_i = count(expert_i 被 top-K 选中)/total
// 2. 聚类：top-M frequency 为 center，其余按 ||[W_Uj||W_Gj] - [W_Uc||W_Gc]|| 分配
// 3. 簇内权重：B_{ji} = f_j / Σ_{k∈C_i} f_k (Theorem 1 证明最优)
// 4. 扩展参数+维度缩减：W'_{Di}=[B_{1i}W_{D1},...]; T2,T3=[B_{1i}I,...] (式4); T1=Q·P^† (式6)
// 5. 最终权重：W^final_Di=W'_{Di}·T1; W^final_Gi=T2·W'_{Gi}; W^final_Ui=T3·W'_{Ui}
// 6. 路由更新：merged_routing = A · original_routing (求和)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- M-SMoE 开源：https://github.com/pppp/M-SMoE；MergeMoE 论文未提供公开代码仓库。
- 实现：PyTorch + HuggingFace Transformers，torch hooks 获取中间激活，BFloat16 精度，逐层反向遍历压缩，每层 <1 分钟。
- 适用场景：将大 MoE 模型压缩到资源受限设备；减少 expert 数量降低推理内存带宽需求。
- 局限：合并后无法恢复原始结构；聚类策略影响最终性能；routing discriminative power 下降（REAP 指出可能导致 functional subspace collapse）。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---

## Expert Output Merging (Output-Merging View)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Output Merging 是 MergeMoE 提出的对 expert merging 的新理论视角。传统"参数合并"视角将 expert 参数加权平均，而输出合并视角将 merging 重新表述为对 expert 输出空间的优化：压缩后 expert E'_i 的输出应逼近原始 experts 输出的线性组合 E'_i(X) ≈ Σ_j B_{ji} E_j(X)，而非简单平均参数。该视角将压缩建模为在前向计算中插入矩阵的线性优化：原始 Y · mask_top_K(...) → 压缩后 Y · B · A · mask_top_K(...)，最小化 ||YBA - Y|| 的 Frobenius 误差。在 MergeMoE 框架中，merged expert 被形式化为 E'_i(X) = W'_Di T1 (σ(T2 W'_Gi X) ⊙ (T3 W'_Ui X))，其中 T1/T2/T3 为维度缩减矩阵。传统 M-SMoE 等价于 T1=[I;I;...;I]（不做优化）、T2/T3 做加权平均的特例，而输出合并视角允许分别优化三个矩阵。

从算法pipeline角度拆解术语：
```
// 矩阵 A (路由求和): A_{ij}=1 if expert j→cluster i, else 0
// 矩阵 B (输出组合): B_{ji}=f_j/Σf_k if j∈C_i, else 0
// 原始 forward: Y · mask_top_K(softmax(W_r X))^T
// 压缩后 forward: Y · B · A · mask_top_K(softmax(W_r X))^T
// 目标: min ||Y(BA - I_N) · mask_top_K(...)||_F^2
// 在独立假设下，问题分解为每个 cluster 的二次优化，使用频率为最优解
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 优势：将 merging 从启发式转变为可量化优化问题；T1 通过采样输入+最小二乘法求解（QP†）直接降低输出误差；解释了 M-SMoE 为何有效以及改进空间。
- 局限：依赖采样输入代表真实分布；T2/T3 仍为启发式（非线性 σ/⊙ 无法联合求解）；样本数 <32 时性能崩溃。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---

## Least-Squares Optimization for Expert Merging (T1 Matrix)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
在 MergeMoE 中，T1 矩阵是用于减少 merged expert 中间维度的线性变换矩阵。当 N 个 expert 被合并为一个时，扩展后的中间维度为 N×E，T1 ∈ R^{E×E}（在合并后 expert 内部）将其压缩回 E。MergeMoE 使用最小二乘法而非简单拼接来求解 T1：固定 T2/T3 后，对采样输入 X̂ 计算 P = σ(T2 W'_G X̂) ⊙ (T3 W'_U X̂)（经压缩路径的中间激活）和 Q = σ(W'_G X̂) ⊙ (W'_U X̂)（原始路径），利用 Moore-Penrose 伪逆求 T1 = Q P^† 的闭式解，最小化 ||T1 P - Q||_F。这与 M-SMoE 形成对比——M-SMoE 等价于 T1 = [I; I; ...; I]（仅拼接不做优化）。

从算法pipeline角度拆解术语：
```
// 步骤1: 获取中间激活 (torch hooks)
P = σ(T2 · W'_Gi · X̂) ⊙ (T3 · W'_Ui · X̂)    // 压缩路径
Q = σ(W'_Gi · X̂) ⊙ (W'_Ui · X̂)               // 原始扩展路径

// 步骤2: 最小二乘闭式解
T1 = Q @ pinv(P)   // Moore-Penrose 伪逆, shape (E, E)

// 步骤3: 构造最终权重
W^final_Di = W'_{Di} @ T1; W^final_Gi = T2 @ W'_{Gi}; W^final_Ui = T3 @ W'_{Ui}
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：PyTorch `torch.linalg.pinv(P)` 或 `torch.linalg.lstsq(P.T, Q.T)`；BFloat16 精度容纳更多样本。
- 样本量临界阈值 ~32 samples，低于此值性能崩溃（≈random guessing）；高于阈值后逐步提升。
- 跨数据集泛化好：即使单一数据集（如 WinoGrande）样本计算的 T1 在其他 benchmark 上表现良好（<1-2% drop）。
- 从后往前逐层压缩：后层压缩不影响前层激活，每层获取 hooks → 计算 T1 → 释放内存。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---

## Expert Clustering for MoE Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Clustering for MoE Compression 是在 expert merging 之前将功能相似的 expert 分组的技术。MergeMoE 的聚类策略：(1) 选取 top-M 使用频率的 expert 作为聚类中心（确保高频 expert 不被稀释）；(2) 距离度量使用拼接矩阵 [W_U || W_G] 的 L2 距离（而非全部参数），因为 T2/T3 仅作用于 W_G/W_U，在这些矩阵上聚类可直接减少加权平均误差。M-SMoE 则使用路由 logits 余弦相似度、dominant experts 作为中心。消融实验（MergeMoE Table 5）表明聚类质量对最终压缩效果至关重要——即使跳过 T1/T2/T3 优化，仅聚类+直接输出合并性能已接近完整流程。

从算法pipeline角度拆解术语：
```
// MergeMoE 聚类
centers = top-M by frequency f_i
for non-center expert j:
    for center k:
        V_j = concat(W_Uj, W_Gj)  // 仅关注影响 T2/T3 的矩阵
        V_k = concat(W_Uk, W_Gk)
        dist(j,k) = ||V_j - V_k||_2
    分配 j 到最近 center
// 簇内权重: w_j = f_j / Σ_{k∈C_i} f_k
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 替代方案：M-SMoE 用 dominant experts + 路由 logits 余弦相似度；Sub-MoE 用 joint SVD 子空间内聚类；DM-MoE 用混合 drop-then-merge。
- 聚类 vs Pruning 的 tradeoff：聚类保留互补信息但可能引入参数干扰；pruning 消除干扰但信息损失更大；混合方法折中。
- REAP (arXiv 2510.13999) 质疑 merging 方法——实验表明 merging 导致 functional subspace collapse，主张 pruning 可能更优。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---

## Usage-Frequency Weighted Expert Merging

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Usage-Frequency Weighted Expert Merging 是以每个 expert 被 router 选中的相对频率作为合并权重的策略。M-SMoE 首次采用此策略但仅基于经验；MergeMoE 通过 Theorem 1 严格证明了在独立假设下（router logits 与 expert 输出独立），使用频率 f_j / Σ f_k 作为簇内权重是 Frobenius 输出误差下界的最优解。证明思路：目标函数 Σ f_j (v_i - e_j)^T W (v_i - e_j) 在每个 cluster 内是独立的二次函数，W = Y_0^T Y_0 为准正定矩阵，设 v_i[j] = f_j/Σf_k 使一阶导数为零，二阶导数 ≥0 保证全局最优。

从算法pipeline角度拆解术语：
```
// 统计频率: calibration 数据前向推理一次，f_i = count(expert_i 被 top-K 选中)/total
// 归一化权重: B_{ji} = f_j / Σ_{k∈C_i} f_k (和为1)
// 在 MergeMoE 框架中: T2/T3 列权重 = B_{ji} (式4), T1 单独最小二乘优化
// Theorem 1 证明: min Σ f_j(v_i-e_j)^T W(v_i-e_j) → v_i[j] = f_j/Σf_k 为全局最优
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 频率统计只需一次无梯度前向推理，开销极小。
- 理论局限：独立性假设在实际模型中可能不完全成立（router 输出与 expert 参数通过训练耦合），但实践效果良好。
- 其他权重方案：均匀权重（Average）、参数幅值加权、路由 logits 加权——MergeMoE 实验验证使用频率加权最优。
- 对比 M-SMoE：两者均使用频率加权，但 MergeMoE 提供了理论最优性证明。

涉及论文标题：
- MergeMoE: Efficient Compression of MoE Models via Expert Output Merging

---

## LoRA (Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA (Low-Rank Adaptation) 由 Hu et al. (2021) 提出，是一种参数高效微调（PEFT）方法。核心思想：对于预训练权重矩阵 W ∈ R^{d1×d2}，不直接微调 W，而是在旁路添加一对低秩分解矩阵 W^A ∈ R^{d1×r} 和 W^B ∈ R^{r×d2}（r << min(d1,d2)），前向计算变为 x' = xW + x·W^A·W^B + b。训练时冻结 W，仅更新 W^A（Kaiming 初始化）和 W^B（零初始化），确保训练起始 ΔW = 0。推理时可将 ΔW = W^A·W^B 与 W 合并（merge），无额外推理开销。典型 r 值：8/16/32/64。在 LLaMA-2 7B 的 Transformer 层中，有 7 个线性模块（Q/K/V/O/G/U/D），每个均可附加 LoRA。LoRA 可调参数通常仅为全模型参数的 <1%。

从算法pipeline角度拆解术语：
LoRA 在 LLaMA-2 Transformer 层中的前向计算：
```
输入: x ∈ R^{batch × seq × d1}
冻结权重: W_m ∈ R^{d1 × d2}  (m ∈ {Q,K,V,O,G,U,D})
LoRA矩阵: W_m^A ∈ R^{d1×r}, W_m^B ∈ R^{r×d2} (r=32)

# 标准前向 + LoRA 修正
output = x @ W_m + x @ W_m^A @ W_m^B + b_m

# 训练: W_m 冻结, W_m^A, W_m^B 可训练
# 参数量: r×(d1+d2), 对 LLaMA-2 7B d=4096: 32×8192=262K/module
# 7模块×32层 ≈ 80M (~1% of 7B)
```
推理场景：(a) 合并模式：W' = W + W^A·W^B，零额外开销；(b) 非合并模式（multi-tenant）：每次 forward 额外计算 7 个 LoRA 模块。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- HuggingFace PEFT: github.com/huggingface/peft, 通过 LoraConfig(r=32, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]) 配置。
- 变体：AdaLoRA（自适应 rank）、DoRA（magnitude+direction 分解）、QLoRA（4-bit 量化）、MOELoRA（sub-rank MoE experts）、MiLoRA（per-module experts + prompt-aware routing）。
- Multi-tenant: 多个任务各自有独立 LoRA weights，共享 frozen backbone。MiLoRA 在此场景下通过 prompt-aware routing 减少生成延迟。MOLE 进一步提出：多个预训练 LoRA 可通过逐层 gating 组合为统一模型，不同层对不同 LoRA 赋不同权重（Hierarchical Weight Control），保持各 LoRA 的个体特征。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning
- Mixture of LoRA Experts
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---

## Dyadic Product / Rank-One Adapter (in LoRA)（并矢积 / 秩一适配器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dyadic Product（并矢积，也称 outer product / 外积）是两个向量的矩阵乘积：给定 u ∈ R^{p×1} 和 v ∈ R^{q×1}，其 dyadic product u ⊗ v = u·v^T ∈ R^{p×q}。在 LoRA 中，ΔW = A·B^T 可以按列分解为 r 个 rank-one dyadic product 之和：ΔW = Σ_{j=1}^r (a_j ⊗ b_j)，其中 a_j 是 down-projection 矩阵 A 的第 j 列，b_j 是 up-projection 矩阵 B 的第 j 列。每个 (a_j ⊗ b_j) 构成一个 rank-one adapter，捕获 weight matrix 在特定 rank 维度上的一个方向性变化。MoDE 论文的关键洞察：这种 dyadic decomposition 允许对每个 rank 维度独立进行混合专家（MoE）路由——即每个 rank j 可以有 m 个备选 up-projection 向量 {b_j^1, ..., b_j^m}，router 为每个 rank 独立选择。

从算法pipeline角度拆解术语：
```
# LoRA update 的 dyadic 分解
# A ∈ R^{P×r}, B ∈ R^{Q×r}
# a_j = A[:, j] ∈ R^{P×1}, b_j = B[:, j] ∈ R^{Q×1}

ΔW = A @ B^T                           # r×(P×Q) 低秩矩阵
    = Σ_{j=1}^r (a_j ⊗ b_j)            # r 个 rank-1 矩阵之和
    = Σ_{j=1}^r (a_j @ b_j^T)           # 每个外积贡献一个秩一更新

# 前向计算（对单个 token x ∈ R^{1×P}）
h = x @ A                               # [1×r]  共享 down-projection
dyadic_sum = 0
for j in range(r):
    h_j = h[0, j]                       # 标量
    dyadic_sum += h_j * b_j^T           # h_j ∈ R, b_j^T ∈ R^{1×Q} → [1×Q]
y = x @ W0 + dyadic_sum
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Rank-one adapter 是 MoDE 的核心组件：每个 dyadic term (a_j ⊗ b_j^i) 构成一个独立可路由的 expert。对 rank r 的 up-projection 矩阵，共有 m×r 个 rank-one expert（m 个备选 per rank j）。
- 广义 rank-p adapter（MoDE m×r×p）：将每 p 列 A 和 B 合并为一个 group adapter A_k ∈ R^{P×p}, B_k^i ∈ R^{Q×p}。MoDE 1×r×r = 标准 LoRA rank r，MoDE m×r×r = LoRA-MoE-SD。
- 实现上可通过标准 PyTorch 矩阵操作完成，无需特殊 kernel。

涉及论文标题：
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---

## Shared Down-Projection Matrix (in LoRA-MoE)（共享下投影矩阵）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Down-Projection Matrix 是 MoDE/LoRA-MoE-SD 的核心设计：在 multi-task LoRA-MoE 中，所有 expert 共享同一个 down-projection 矩阵 A ∈ R^{P×r}，仅 up-projection 矩阵 B^i ∈ R^{Q×r} 保持 expert-specific。设计动机来自 PCA 分析：对 15 个独立训练的 LoRA 模块进行 PCA 可视化，发现不同任务的 down-projection 向量（A 的列向量 a_j）按 rank 维度高度聚类（task-agnostic），而 up-projection 向量（B 的列向量 b_j）分散分布（task-specific）。这意味着为每个 expert 学习独立 A^i 矩阵是参数冗余的——多个任务可以用同一个 A 完成输入特征提取（down-projection），而任务特定性仅通过 B^i 表达。

从算法pipeline角度拆解术语：
```
# LoRA-MoE (传统): m 个 expert，每个有独立 A^i, B^i
# y = x@W0 + Σ_{i=1}^m R^i(x) * (x@A^i@B^{iT})
# 参数量: m × r × (P + Q)

# LoRA-MoE-SD (共享 down-projection): 共享 A, 各自 B^i
# y = x@W0 + Σ_{i=1}^m R^i(x) * (x@A@B^{iT})
# 参数量: r×P + m×r×Q ≈ r×P + m×r×Q (节省 m×r×P - r×P)

# 实际效果 (Gemma 2B, r=4, m=4):
# MoLORA 16×4: 7.62% 额外参数, ROUGE-L 57.77
# MoLORA-SD 16×4: 2.71% 额外参数, ROUGE-L 58.28
# 参数节省 64%, 性能提升 0.88% ROUGE-L
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与标准 LoRA 类似：A ∈ R^{P×r} 唯一，B^i ∈ R^{Q×r} 有 m 个副本。Training 时仅更新 A 和 B^i。
- Merge 到 backbone 时的等效权重：W_eff = W0 + Σ_i R^i(x) · (A @ B^{iT})，即 m 个不同的 ΔW^i 均通过相同的 A 生成。
- 适用场景：multi-task PEFT，任务数多时收益最大（无需为每个新任务分配独立的 down-projection 参数）。

涉及论文标题：
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---

## Fine-Grained Per-Rank Routing（逐秩细粒度路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Fine-Grained Per-Rank Routing 是 MoDE 的核心路由创新。传统 LoRA-MoE 的 router 输出 m 维权重，所有 r 个 rank 维度**绑定**在一起路由——即一次 routing decision 选择整个 up-projection 矩阵 B^i。MoDE 的 router 为每个 rank j 独立输出 m 维权重：W_R ∈ R^{r×P×m}，其中 W_{R;j} ∈ R^{P×m} 负责第 j 个 rank。对输入 x，第 j 个 rank 的 routing 权重为 R_j(x) = softmax(x · W_{R;j})。这允许 "B 的第 1 列选 expert 1，第 2 列选 expert 3，第 3 列选 expert 2，第 4 列选 expert 1" 的细粒度组合，可表达 m^r 种不同的 up-projection 矩阵组合（vs 传统 LoRA-MoE 的 m 种），在同等参数量下极大提升模型表达力。

从算法pipeline角度拆解术语：
```
# 输入: x ∈ R^{1×P}
# 共享 A: a_j ∈ R^{P×1} for j=1..r
# Expert up-projection: b_j^i ∈ R^{Q×1} for i=1..m, j=1..r
# Router: W_{R;j} ∈ R^{P×m} for j=1..r

# 传统 LoRA-MoE routing（所有 rank 绑定）
R(x) = softmax(x @ W_R)  ∈ R^{1×m}      # m 种选择
# 对每个 expert i: 贡献 = R_i(x) * x@(Σ_j a_j⊗b_j^{iT})

# MoDE fine-grained routing（每 rank 独立）
dyadic_sum = 0
for j in range(r):                        # 对每个 rank 维度
    R_j = softmax(x @ W_{R;j})            # [1×m] 独立路由权重
    h_j = x @ a_j                         # 标量
    for i in range(m):
        dyadic_sum += R_j[i] * h_j * b_j^{iT}  # 加权 rank-one 贡献
y = x @ W0 + dyadic_sum

# 组合空间: m^r >> m (例如 m=4,r=4: 256 vs 4)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Router 参数 W_R ∈ R^{r×P×m}，推理时需计算 r 次 softmax（每次输出 m 维），计算开销 O(r×P×m)。
- 典型配置：MoDE 4×4（4 experts × rank 4）即可超越 LoRA 64（ROUGE-L 60.18 vs 56.11），参数量仅 1.90%。
- 局限性：router 计算开销随 rank r 增大而线性增长；论文指出在实时应用场景中路由计算可能成为瓶颈。

涉及论文标题：
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts

---

## Prompt-Aware Routing (in MoE-style PEFT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Prompt-Aware Routing 是 MiLoRA 提出的 LoRA expert 路由机制。与传统 MoE-style LoRA（如 MOELoRA、LoRAMoE）的 token-wise routing 不同，prompt-aware routing 在输入 prompt 首次通过 LLM backbone 时仅计算一次路由决策（在生成第一个新 token 之前），后续所有 auto-regressive token 生成步骤均复用该决策。Router 输入为 prompt hidden states H^l ∈ R^{n_p×d}，经 Self-Attention Pooler → Rational Activation → Linear Router + Softmax → Top-k，每层选出 1 个 LoRA expert（从 Q/K/V/O/G/U/D 7 个模块中）。Router 调用次数从 token-wise 的 O(L×T×7) 降至 O(L×7)，生成阶段无 router 开销。

从算法pipeline角度拆解术语：
```
阶段一: Prompt编码（仅一次，在第一个new token前）
输入: H^l ∈ R^{n_p × d}  (layer l 的输入hidden states)

# Pooling
W_sa ∈ R^{d×1}
U = H^l @ W_sa → [n_p × 1]
A = Softmax(U, dim=0)
h^l = A^T @ H^l → [1 × d]

# Rational Activation（可学习，每层独立）
g^l = Ra(h^l)  # m=6, n=5, 初始化为GeLU近似

# Router
logits = g^l @ W_r^l   # [1×d]@[d×7]→[1×7]
probs = Softmax(logits)
expert_idx = TopK(probs, k=3)  # 选top-1 expert

阶段二: Token生成（复用阶段一路由决策）
for token in 1..T:
    if module == expert_idx:
        output = x@W_m + x@W_m^A@W_m^B + b_m  # LoRA修正
    else:
        output = x@W_m + b_m  # 原始backbone
```
效率：L=32, T=256时，router调用从57344降至224。实测 beam=1 tps 43.7 (vs MOELoRA 35.9, +21.7%)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 HuggingFace Transformers forward 中增加 router 计算。Routing decisions 缓存为 layer-level 属性（如 `self.activated_expert_idx`），供后续 token 复用。
- Pooler: 默认 self-attention pooling（最优），备选 last-token/average/max pooling。
- Load balancing: L_lb = N_mod · Σ f_i·p̂_i，λ_lb=1e-2。
- 场景: multi-tenant LLM serving（每 tenant 独立 LoRA adapter），消除 per-token routing 开销。

涉及论文标题：
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

---

## Rational Activation Function (Learnable Activation in PEFT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Rational Activation Function 由 Molina et al. (2019) 提出，是有理函数形式的可学习激活函数：Ra(x) = Σ_{j=0}^{m} a_j·x^j / (1 + ||Σ_{i=1}^{n} b_i·x^i||)，其中 a_j, b_i 为可学习参数，m, n 为阶数。通过调整参数可逼近 ReLU/GeLU/Swish 等，也可学习全新形态。MiLoRA 将其用于每层 LoRA router 的激活函数（m=6, n=5，初始化为 GeLU 逼近），使不同深度的 Transformer 层学习最适合路由的激活函数。通过 bi-level optimization（DARTS 风格）训练：inner level 优化 LoRA+router 参数 Ω（lr=1e-4），outer level 优化 activation 参数 Θ（lr=1e-6，仅 ~12 scalars/layer）。Ablation 表明 learnable activation 在 BoolQ/PIQA/MMLU 上优于固定 GeLU 或 ReLU+GeLU 混合方案。

从算法pipeline角度拆解术语：
```
# 逐元素计算（per element x in pooled hidden state h^l）
num = a_0 + a_1·x + a_2·x^2 + a_3·x^3 + a_4·x^4 + a_5·x^5 + a_6·x^6
den = 1 + |b_1·x + b_2·x^2 + b_3·x^3 + b_4·x^4 + b_5·x^5|
Ra(x) = num / den

# 每层参数: a_j (7个), b_i (5个) = 12 scalars/layer
# 32层总计: 384 scalars, overhead trivial
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 实现: 自定义 nn.Module，forward 中预计算 x 各次幂避免重复。参数初始化为 GeLU Padé 逼近系数（保证训练初期稳定）。
- 训练: bi-level optimization，alternating 更新（每 step 更新 Ω，每若干 step 更新 Θ）。
- 适用: 任何需要为不同层学习不同激活函数的场景。论文验证在 3 个 benchmark 上一致优于固定激活函数。

涉及论文标题：
- MiLoRA: Efficient Mixture of Low-Rank Adaptation for Large Language Models Fine-tuning

---

## Expert Data Parallelism (EDP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Data Parallelism (EDP) 是 Expert Parallelism 与 Data Parallelism 结合产生的特殊并行模式。当 DP degree > EP degree 时，系统必须跨多个 EP group 使用 DP。EDP group 定义为共享同一 EP rank 的多个设备（不同 EP group 中同一位置的 GPU）。EDP group 内的设备持有相同 expert 的 replica（参数完全一致），各自处理来自不同 EP group 的不同 token。这些 replica 之间通过标准 DP 机制同步参数和梯度（all-reduce）。EDP 是 FineEP 实现 token scheduling 的关键基础——因为同一 expert 在多个 GPU 上有 replica，token 可以选择任一 replica 计算，从而创造了"调度空间"。

从算法pipeline角度拆解术语：
以 DP=8, EP=4 为例：
- 8 GPU 分为 2 个 EP group（各 4 GPU）。
- EP group 0: GPU{0,1,2,3}，EP group 1: GPU{4,5,6,7}。
- Expert 0 的 replica 在 GPU 0 和 GPU 4（EDP group of expert 0 = {0,4}）。
- 各 GPU 0 和 GPU 4 持有 expert 0 的相同参数，但处理不同 EP group 的不同 tokens。
- 传统 EP：token 只能在 assigned EP group 内的 GPU 0（或 GPU 4）计算 expert 0。
- FineEP：合并 EP groups 后，token 可在 EDP group {0,4} 中任一 GPU 计算 expert 0。

术语一般如何实现？如何使用？
- 参数同步：EDP group 内通过 DP all-reduce 同步 expert replica 的 gradients。
- 在 Megatron-LM 中，EDP 由 DP 和 EP 的配置自动形成，无需显式设置。
- FineEP 利用 EDP 创建 token 调度空间：每个 expert 在 |G_FineEP|/EP_degree 个 GPU 上有 replica。
- 约束：所有 replica 必须具有相同的 local expert index（确保 DP synchronization 一致性，避免 deadlock）。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---

## Cayley Graph Expert Placement

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cayley Graph Expert Placement 是 FineMoE 用于构造 symmetric expert placement 的图论方法。Cayley graph 从群 A 及其生成集 S 构造：群中每个元素为 vertex，对每个 a∈A 和 s∈S，存在一条从 a 到 a·s 的边。FineEP 将 GPU 映射为 vertices，expert（edge 连接其 EDP group 中的 GPU）映射为 edges。利用 Cayley graph 的对称性保证 edges 在 vertices 间均匀分布，使得 graph 的 max induced subgraph density（Equation 3 中的最大化项）最小化。这从理论上保证：在未知 load 分布时，任何 GPU 子集的 expert 集中度都有上界。

从算法pipeline角度拆解术语：
构造示例（p GPU 数 exponent, q expert/GPU exponent）：
- 8 GPU, 8 experts (p=3, q=1): group (Z_8, +), generators {1,-1} → cycle graph。每个 vertex 度=2。
- 16 GPU, 32 experts (p=4, q=2): group (Z_4×Z_4, +), generators {(0,1),(0,-1),(1,0),(-1,0)} → 4×4 toroidal grid。每个 vertex 度=4。
- 8 GPU, 16 experts (p=3, q=2): group (Z_2×Z_4, +), generators {(0,1),(0,-1),(1,1),(1,-1)} → isomorphic to K_{4,4}。满足性质: ∀i, 所有 i-vertex induced subgraph 的最大 edge count 最小。
- 8 GPU, 32 experts (p=3, q=3): 先构造 complete graph（28 edges），剩余 4 edges 补为额外配对。

术语一般如何实现？如何使用？
- 论文假设 GPU 数和 expert/GPU 数为 2 的幂（实际常见于数据中心配置）。
- Cayley graph placement 生成在 Placement Manager 中完成，训练开始时 broadcast 到所有 GPU。
- 需要满足同步一致性约束：同一 expert 的所有 replica 必须具有相同 local expert index（Appendix B.3）。
- 仅用于 d=2 场景（hypergraph 退化为常规 graph）；d>2 尚未系统研究。

涉及论文标题：
- FineMoE: Fine-grained Load Balancing for Mixture-of-Experts with Token Scheduling

---

## DoRA (Weight-Decomposed Low-Rank Adaptation / 权重分解低秩适配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
DoRA (Weight-Decomposed Low-Rank Adaptation) 由 Liu et al. (2024) 提出，是 LoRA 的变体。将预训练权重 W 分解为 magnitude（幅度 m）和 direction（W/||W||）分量独立更新：magnitude 通过可训练 scalar vector 直接更新（非低秩约束），direction 通过标准 LoRA（B·A 低秩分解）更新。相比标准 LoRA 仅用 BA 整体模拟 ΔW，DoRA 的 magnitude-direction 分解使学习模式更接近 full fine-tuning。在 MixLoRA 中，DoRA 被用作 expert 基础微调单元（替代 LoRA），构成 MixDoRA 变体。

从算法pipeline角度拆解术语：
```
// DoRA 线性层前向 (per expert)
输入: x [B, N, d_in], 预训练权重 W [d_out, d_in]
可训练参数: m [d_out], B [d_out, r], A [r, d_in]

W_norm = W / ||W||_c                           // column-wise L2 norm → unit direction
ΔW_dir = B · A                                  // [d_out, d_in], rank-r 方向更新
W_dir' = W_norm + ΔW_dir
W_updated = m · W_dir'                          // element-wise m 广播
y = W_updated · x
```
MixLoRA 中的 MixDoRA：每个 expert 的 FFN 权重按上式分解，8 experts, top-2, r=16。MixDoRA 对 load balance loss coefficient 更不敏感（禁用时仅降 ~1% vs MixLoRA 降 ~2.5%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：HuggingFace PEFT `LoraConfig(use_dora=True)`。
- MixLoRA 实验：Gemma 2B 单任务 MixDoRA (71.6%) > MixLoRA (69.9%)；LLaMA-2 7B 多任务 MixLoRA (75.3%) ≈ MixDoRA (74.9%)。MoE 结构的微调多样性使 DoRA 的分解策略效果减弱。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

---

## Top-K Router (Top-K 路由器 for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Top-K Router 是 MoE 为每 token 从 N 个 expert 中选择 K 个最适合 expert 的路由机制。Router 为线性层 W_r ∈ R^{d×N}，将 token hidden state h ∈ R^d 映射为 N 个 expert logits → Softmax → Top-K 选 K 个 expert。MixLoRA 采用 Top-2 Router（K=2），从 8 个 LoRA-based expert 中为每 token 选择 2 个 expert，按路由概率加权输出。

从算法pipeline角度拆解术语：
```
输入: h_i [1, d], W_r [d, N]  (N=8, K=2)
logits = W_r · h_i^T                           // [1, N]
probs = Softmax(logits)                         // [1, N]
r' = KeepTop-2(probs)                           // re-normalize top-2 positions
output = Σ_{k: r'_k>0} r'_k · E_k(h_i)         // 加权求和
```
Router 在每个 MoE layer 有独立参数 W_r^ℓ。MixLoRA 中 router 仅应用于 FFN MoE（非 attention），区别于 MOELoRA/MOLA 将 router 应用于 attention+FFN。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：`nn.Linear(d_model, num_experts)` + Softmax + `torch.topk()`。
- 训练时需 load balance loss 防止 collapse。MixLoRA 验证 Top-2 在单任务和多任务下均有效。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- Mixture of Diverse Size Experts (MoDSE 的 Top-K Router 逻辑与标准 MoE 相同，但 router 将 token 分配给不同尺寸的 expert，困难 token 倾向选择大专家)
- MoLA: MoE LoRA with Layer-wise Expert Allocation (per-layer independent Top-2 router; each layer j has its own W_r^j ∈ R^{d_q × N_j} with layer-specific N_j experts; router output S_i^{jt}(x) = TopK(Softmax(W_r^{jt}x), K)_i / Σ TopK(...)_i)

---

## Load Balance Loss (Auxiliary Loss for MoE / MoE 负载均衡辅助损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load Balance Loss 是 MoE 训练中防止 expert 负载不均衡的辅助损失函数。MoE router 倾向反复选择少数 expert → 其他 expert 欠训练、热门 expert 成为计算瓶颈。MixLoRA 采用 Switch Transformers (Fedus et al. 2022) 公式：L_aux = a·N·Σ F_i·P_i，F_i=batch 中被路由到 expert i 的 token 比例，P_i=router 分配给 expert i 的平均概率，a=scaling coefficient（推荐 1e-3）。N 倍乘使 loss 在 expert 数量变化时恒定。总 loss: L = L_CE + L_aux。

从算法pipeline角度拆解术语：
```
for i in {1..N}:
    F_i = (1/T)·Σ_{x∈B} 𝟙{argmax_k R(x)_k = i}    // token 分配比例
    P_i = (1/T)·Σ_{x∈B} R(x)_i                       // router 概率均值
L_aux = a · N · Σ_{i=1}^{N} F_i · P_i
// a=1e-3 最优; N=8 时 L_aux ≈ 8e-3 · Σ F_i·P_i
```
Ablation：a=1e-3 时 MixLoRA 最佳 accuracy；a=0 (禁用) 降 ~2.5%；a=1e-1 (过大) 降 ~1.5%。验证效果：启用后 expert load std dev 低至 0.0223 (MixLoRA) / 0.0328 (MixDoRA)。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练 loop 中额外 loss 项与 CE loss 直接相加。需获取 router 完整概率分布（不仅是 top-K 结果）。
- 几乎所有 MoE 训练必备（GShard → Mixtral → DeepSeek-V3）。
- MixLoRA 发现 MixDoRA 对 a 更不敏感：禁用 load balance loss 时 MixDoRA 仅降 ~1% vs MixLoRA 降 ~2.5%。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- Mixture of Diverse Size Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**局部路由一致性与负载均衡的 trade-off** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文揭示了 Load Balance 与 Local Routing Consistency 之间的重要区分——(1) Local Load Balance（单个 query 内 expert 激活的均匀程度）：与局部路由一致性存在强 trade-off，高一致性模型路由更集中（expert activation SD 更大），TOY 模型 NoLB（无 load balance loss）SRP 最高 56.42 但 LB SD=13.21 极高，OverLB（loss coefficient=0.1）SRP 最低 36.42 但 LB SD=1.79 极低。(2) Global Load Balance（跨不同 query 的整体 expert 利用率）：可与高局部路由一致性共存——Qwen3 (SRP 54.14, LB SD 3.19) 和 GRIN-MoE (SRP 50.39, LB SD 3.89) 同时具有高 SRP 和适中的全局负载均衡。机制：domain-specialized experts 在匹配其专长领域的上下文中持续激活（高局部一致性），在不同领域被不同 expert 集处理（全局均衡）。论文建议：若 target 场景涉及 expert offloading（如边缘设备部署），可适度牺牲局部负载均衡以换取更高的局部路由一致性。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diverse Size Experts（MoDSE）是 Mixture of Diverse Size Experts 论文提出的 MoE FFN 层新结构。与传统 MoE 中所有 N 个 expert 拥有完全相同的 hidden dimension h 不同，MoDSE 在同一 MoE layer 内设置不同 hidden dimension $\hat{h}_i$ 的 expert：大专家（$\hat{h}_i > h$）具有更强预测能力，处理高难度 token 预测；小专家（$\hat{h}_i < h$）计算量更小，处理低难度 token 预测。Experts 按对分组 $(i_k^1, i_k^2)$，每对满足 $\hat{h}_{i_k^1} + \hat{h}_{i_k^2} = 2h$，保证总参数量与 baseline（所有 expert 尺寸相同）一致。设计动机：预训练语料中 token 预测难度差异巨大——同一短语内 token 极易预测，跨领域推理 token 极难预测——same-size expert 无法区分对待不同难度 token，diverse-size expert 让不同能力 expert 各司其职。

从算法pipeline角度拆解术语：
```
# MoDSE FFN Layer with Diverse Size Experts (N=8, K=2)
# Expert pairs with diverse hidden dims:
#   pair_0: E_{4.5}(h=6912), E_{0.5}(h=768)  -> 6912+768=7680=2×3840
#   pair_1: E_{4.0}(h=6144), E_{1.0}(h=1536) -> 6144+1536=7680
#   pair_2: E_{3.0}(h=4608), E_{2.0}(h=3072) -> 4608+3072=7680
#   pair_3: E_{2.5}(h=3840), E_{2.5}(h=3840) -> 3840+3840=7680
# baseline h=3840, dim=1536

# Input: x [B, S, dim]
# Step 1: Standard gating (same as Switch Transformer)
logits = x @ W_g                          # [B, S, N]
noise = RMSNorm(Softplus(x @ W_n))        # [B, S, N]
H = logits + noise
probs = Softmax(KeepTopK(H, k=2))         # [B, S, N]

# Step 2: Diverse-size expert computation
# Each expert E_i has different hidden dim h_i
#   E_i: w1_i [dim, h_i] -> SiLU -> w2_i [h_i, dim]
#   Parameter count for expert i: 2 * dim * h_i
#   Total params: 2 * dim * Σ_i h_i = 2 * dim * N * h (same as baseline)

output = zeros([B, S, dim])
for each expert i in {0..N-1}:
    mask_i = (expert i in top-2 for each token)
    if mask_i.any():
        tokens_i = x[mask_i]              # [n_i, dim]
        h_i = tokens_i @ w1_i             # [n_i, h_i] -- h_i varies!
        a_i = SiLU(h_i)                   # [n_i, h_i]
        out_i = a_i @ w2_i                # [n_i, dim]
        output[mask_i] += probs[mask_i, i] * out_i

# Step 3: Load balance loss (Switch Transformer style)
f_i = fraction of tokens routed to expert i
P_i = mean router probability for expert i
L_aux = α * N * Σ_i f_i * P_i
```

术语一般如何实现？如何使用？
- 实现：在 PyTorch 中，每个 expert 用不同尺寸的 `nn.Linear(dim, h_i)` 和 `nn.Linear(h_i, dim)`，forward 时根据 router 输出的 expert index 分发 token 到对应尺寸的 expert。需使用诸如 `torch.index_select` 或 scatter/gather 操作处理不同 expert 的不同 batch size。
- 配置：实验中使用 4 对 expert，尺寸比例分别为 (4.5,0.5), (4.0,1.0), (3.0,2.0), (2.5,2.5) 相对于 input dim。更大模型维持相同比例但绝对值更大（如 700M×8 中最大 expert h=9216）。
- 负载均衡：由于 expert 尺寸不同，天然导致计算负载不均衡→需配合 Expert-Pair Allocation 策略将每对 expert 放置在同一 GPU。
- 论文验证：700M×8 MoDSE 在 9 个 benchmark 上全面超越 same-size baseline（如 MMLU 29.9 vs 26.5, SIQA 60.9 vs 42.9），训练 loss 曲线更低，且推理耗时几乎与 baseline 相同（MMLU 3min27s vs 3min26s）。
- 局限性：实验仅在小规模 MoE（300M×8, 700M×8）上验证，大规模 MoE 的可扩展性未知。训练数据为非开源中英双语 100B tokens。

涉及论文标题：
- Mixture of Diverse Size Experts

---

## LoRA-based MoE (LoRA-MoE / 基于 LoRA 的混合专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA-based MoE 是一类将 LoRA（低秩适配）与 MoE（混合专家）架构结合的 PEFT 方法：在预训练 dense 模型的 transformer 子层中插入多个 LoRA adapter 作为 expert，用可训练 router 为每 token 选择激活的 expert 子集，以低成本扩展容量并获得 MoE 的多任务泛化优势。MixLoRA 的关键设计区别于其他 LoRA-MoE 方法：expert = 共享冻结 FFN 权重 + 独立 LoRA adapter（而非LoRA 本身作为 expert），更贴近 Mixtral 等预训练 MoE 架构；attention 层使用独立 LoRA 适配器（非 MoE）。解决标准 LoRA 的多任务 catastrophic forgetting 和容量受限问题。

从算法pipeline角度拆解术语（MixLoRA 与其他 LoRA-MoE 流派对比）：

| 方法 | Expert构造 | 位置 | Router | LB | 特点 |
|------|----------|------|--------|----|------|
| MOELoRA | sub-rank per module | Attn+FFN | 有 | 有 | 对比学习路由 |
| LoRAMoE | 多LoRA+FFN | FFN only | 有 | 有 | 防知识遗忘 |
| MOLA | 层级expert数 | Attn+FFN | 有 | 有 | 高层多expert |
| PESC | dense→sparse | FFN only | 有 | 有 | dense2sparse |
| **MixLoRA** | **共享FFN+LoRA expert** | **FFN(MoE)+Attn(独立LoRA)** | **Top-2** | **有** | **对齐预训练MoE** |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/TUDB-Labs/MixLoRA
- 关键参数：N=8 experts, K=2, r=16（远小于标准 LoRA r=80）。共享 FFN 计算减少 30% token 延迟；多模型 batching 减少 40% GPU memory。
- 适用：consumer GPU（24GB）多任务微调、multi-tenant LoRA serving。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts
- MoDE: Effective Multi-task Parameter Efficient Fine-Tuning with a Mixture of Dyadic Experts
- MoLA: MoE LoRA with Layer-wise Expert Allocation (applies LoRA-MoE to ALL dense weight matrices in Transformer, including attention Wq/Wk/Wv/Wo and MLP Wgate/Wdown/Wup; introduces layer-wise expert allocation — different layers have different numbers of LoRA experts; lower layers have more expert redundancy, middle/upper layers benefit from more experts)

MoDE 论文对 LoRA-MoE 的贡献：(1) 通过 PCA 分析发现 down-projection 向量跨任务聚类，提出共享 down-projection 矩阵 A（LoRA-MoE-SD），节省 ~64% 参数同时提升性能；(2) 将 LoRA 分解为 dyadic sum Σ(a_j ⊗ b_j)，对每个 rank j 独立配置 m 个 rank-one adapter 并 per-rank routing，实现 m^r 种组合空间（vs 传统 m 种）；(3) 泛化为 rank-p adapter（MoDE m×r×p）。

---

## Multi-LoRA Optimization (m-LoRA / 多 LoRA 并行优化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Multi-LoRA Optimization 是在单 GPU 上同时训练/推理多个 LoRA 变体模型的高吞吐技术，由 m-LoRA / Aspen (Ye et al. 2023) 提出。核心原理：多个 LoRA 变体共享同一 frozen backbone，不同模型的 batch 合并避免重复 backbone 计算。MixLoRA 扩展至 multi-MixLoRA 场景：多模型的 multi-task 输入合并为单 batch，共享 W1/W3 FFN 计算，各模型 router 独立路由执行各自 LoRA adapter。per-model GPU memory 降 ~45%（LLaMA-2 7B: 15.1→8.8GB 训练, 13.7→7.2GB 推理）。

从算法pipeline角度拆解术语（Algorithm 1, Appendix A.7）：
```
for t in {1..M}:                               // M 个 MixLoRA 模型
    T_t = T^{l-1}[t]                            // 模型 t 的输入 [B,N,D]
    r_t' = Top2(Softmax(Linear_t(T_t)))         // 模型 t 独立 router
    h_W1, h_W3 = Shared_W1(T_t), Shared_W3(T_t) // 共享 FFN 计算
    for k in {1..K}:
        h_gate = SiLU(h_W1 + LoRA_k^{W1}) ⊙ (h_W3 + LoRA_k^{W3})
        h_k = Shared_W2(h_gate) + LoRA_k^{W2}(h_gate)
        T_t^l += h_k ⊙ r_t'[:,k]                // router 加权累加
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/TUDB-Labs/m-LoRA
- 适用：多下游任务同时微调、multi-tenant LoRA serving。限制：所有模型需共享相同 backbone。

涉及论文标题：
- MixLoRA: Enhancing Large Language Models Fine-Tuning with LoRA based Mixture of Experts

## Product Key Retrieval（产品密钥检索）

术语是什么？
Product Key Retrieval 是一种基于乘积量化（Product Quantization）思想的高效最近邻检索技术，由 Lample et al. (2019) 在 Product Key Memory (PKM) 中首次提出。核心思想是将 N 个 d 维 key 向量分解为两组子密钥的笛卡尔积：K = {[c; c'] | c ∈ C, c' ∈ C'}，其中 C, C' 各含 √N 个 d/2 维子密钥。查询向量 q 同样拆分为 q₁, q₂，分别在两组子密钥中做 top-k 检索，候选集大小为 k²，再从候选中选出最终 top-k。检索复杂度从朴素穷举的 O(Nd) 降至 O((√N + k²)d)，使得从百万级（10⁶）候选中检索成为可能。PEER 论文将 Product Key 用作 MoE router，从超过一百万个 singleton expert 中检索 top-k 专家。

从算法pipeline角度拆解术语：
Product Key Retrieval 在 PEER 中的路由 pipeline：
```
# 输入: query q ∈ R^d，子密钥组 C, C' 各含 √N 个向量
# Step 1: 拆分 query
q₁, q₂ = q[:d/2], q[d/2:]

# Step 2: 分别在两组子密钥中 top-k 检索
I_C = TopK({q₁^T c_i | c_i ∈ C}, k)   # k 个候选子密钥索引
I_C' = TopK({q₂^T c'_j | c'_j ∈ C'}, k)

# Step 3: 候选 product keys 集合 (k² 个)
K' = {(c_i, c'_j) | i ∈ I_C, j ∈ I_C'}

# Step 4: 在候选中最终 top-k（利用内积可加性）
scores = {q₁^T c_i + q₂^T c'_j | (c_i, c'_j) ∈ K'}
final_indices = TopK(scores, k)  # k 个最终 expert 索引
```
数学保证：K 中与 q 内积最大的 k 个 key 一定在候选集 K' 中。证明基础：内积的可加性——q^T [c; c'] = q₁^T c + q₂^T c'。

术语一般如何实现？
标准实现参考 Lample et al. (2021) 的 PKM-layer.ipynb（https://github.com/facebookresearch/XLM/blob/main/PKM-layer.ipynb）。子密钥存储为两组 Embedding 矩阵（各 √N × d/2），query 投影通过线性层映射到 d 维后拆分为两半。get_indices 函数执行上述两步 top-k 检索。PEER 中 expert 的 down/up projection 权重同样存储为 Embedding 层，通过检索到的索引进行 lookup。论文指出高效实现需要 specialized hardware kernels 加速 embedding lookup 与 einsum 的融合。当前 PEER 实现为 JAX 原型（内部代码库，未开源）。

涉及论文标题：
- Mixture of A Million Experts

## PEER (Parameter Efficient Expert Retrieval / 参数高效专家检索)

术语是什么？
PEER 是 Google DeepMind 提出的新型 MoE 层设计，通过 Product Key Retrieval 实现从超过一百万（10⁶）个极小专家（单神经元 MLP）中高效稀疏检索。PEER 层由三部分组成：(1) N 个 singleton expert e_i(x) = σ(u_i^T x) v_i（每个仅一个隐藏神经元，参数为两个 d_model 维向量）；(2) N 个 product key（由两组各 √N 个 d/2 维子密钥的笛卡尔积构成）；(3) h 个独立 query network（multi-head retrieval），每个检索 k 个 expert。PEER 解耦了模型总参数 P 与每 token 激活参数 P_active：P = N × 2d_model 可扩展到百万级，而 P_active = hk × 2d_model 保持恒定。PEER 可直接替换 Transformer 中任意 FFW 层，在 isoFLOP 条件下显著优于 dense FFW、coarse-grained MoE 和 PKM。

从算法pipeline角度拆解术语：
PEER 层前向传播（基于论文 Algorithm 1）：
```
def peer_forward(self, x):                    # x: (batch, tokens, d_model)
    # 多 query 头投影
    queries = self.query_proj(x)               # (b, t, h, d)
    
    # Product Key 检索
    indices, scores = self.get_indices(queries, self.sub_keys, top_k=k)
    # indices: (b, t, h, k), scores: (b, t, h, k)
    
    # Embedding lookup 检索 expert 权重
    w_down = self.w_down_embed(indices)        # (b, t, h, k, d_model)
    w_up = self.w_up_embed(indices)            # (b, t, h, k, d_model)
    
    # Singleton expert 计算
    x = einsum('btd, bthkd -> bthk', x, w_down)  # u_i^T x
    x = activation(x)                              # σ(·)
    x = x * softmax(scores)                        # router 加权
    x = einsum('bthk, bthkd -> btd', x, w_up)     # 输出投影
    return x
```
等效解释：当 k=1 时，PEER 动态组装一个 h 神经元 MLP：f(x) = V σ(W^T x)，其中 W=[u₁,...,u_h], V=[v₁,...,v_h] 从共享 expert pool 中检索得到。

术语一般如何实现？
PEER 层可插入 Transformer backbone 中间（如 12 层 transformer 的第 6 层替换 FFW），也可替换全部 FFW 层。实现使用 JAX 的 Embedding 层存储 expert 权重（类似大词表），通过 einsum 操作执行批量内积计算。默认配置：N=1024²=1,048,576 experts, h=8 heads, k=16 experts/head, query BatchNorm 启用。可扩展至 GLU 变体（添加额外 linear gating 权重）。论文代码为 Google 内部代码库，未开源；参考实现可基于 facebookresearch/XLM 的 PKM-layer.ipynb 修改。

涉及论文标题：
- Mixture of A Million Experts

## Expert Choice Routing（专家选择路由）

术语是什么？
Expert Choice Routing 是 Zhou et al. (2022) 提出的 MoE 路由算法，与传统的 Token Choice Routing（token 选择 expert）相反，Expert Choice 让每个 expert 选择 top-k 个 token 进行处理。具体地，对于 N 个 expert 和 M 个 token，计算 gating score 矩阵 S ∈ R^{N×M}，每个 expert（行）选择 score 最高的 k 个 token。这种方法天然解决了 Token Choice 中的 expert 负载不均衡问题——每个 expert 恰好处理 k 个 token（或按 capacity factor 调整），避免了某些 expert 过载而其他闲置。PEER 论文使用 Expert Choice MoE（128 experts）作为 coarse-grained MoE baseline。然而，Expert Choice 仍需要在整个 N×M 的 gating score 矩阵上操作（通过 top-k），路由复杂度至少 O(N)，限制了 expert 数量通常 < 128。

从算法pipeline角度拆解术语：
Expert Choice Routing 算法流程：
```
# S ∈ R^{N×M}: gating score 矩阵，S[i,j] = expert i 对 token j 的 score
# capacity = k × capacity_factor  (每个 expert 的 token 容量)

for each expert i in {1..N}:
    # expert i 选择 score 最高的 capacity 个 token
    selected_tokens = TopK(S[i, :], capacity)
    # expert i 仅计算被选中 token 的 FFN
    for token j in selected_tokens:
        output[j] += softmax(S[:, j])[i] × ExpertFFN_i(x_j)
```
与 Token Choice 的对比：Token Choice 是每个 token 选 top-k expert（行方向 top-k），Expert Choice 是每个 expert 选 top-k token（列方向 top-k）。Expert Choice 保证了每个 expert 的负载均衡，但需要所有 token 同时可用（训练时 batch 内），且每个 expert 处理的 token 可能来自 batch 中不连续的位置。

术语一般如何实现？
Expert Choice 在训练中应用：batch 内所有 token 共同参与 gating score 计算，每个 expert 选择 score 最高的 ⌈k × M / N⌉ 个 token（或按 capacity factor 调整）。推理时一般仍用 Token Choice（因 token 按流式到达）。PEER 论文中 Expert Choice MoE baseline 使用 128 个 expert，每个 expert 大小等于对应 dense 模型的 FFW 层。与 PEER 的对比：Expert Choice 为 O(N) 路由复杂度，限制 N < 128；PEER 为 O(√N) 复杂度，支持 N ≥ 10⁶。

涉及论文标题：
- Mixture of A Million Experts

## Singleton MLP Expert（单神经元 MLP 专家）

术语是什么？
Singleton MLP Expert 是 PEER 中使用的极小专家设计：每个 expert e_i(x) = σ(u_i^T x) v_i 仅有 1 个隐藏神经元（d_expert = 1），参数为两个 d_model 维向量 u_i（down projection）和 v_i（up projection），总计 2×d_model 个参数。这与传统 MoE 中每个 expert 为完整 FFW（hidden dimension 通常等于 d_ffn，如 4×d_model）形成鲜明对比。Singleton expert 将专家大小推到理论最小值，最大化 MoE 的粒度 G = P_active / P_expert = hk / 1 = hk，使得在固定 P_active 下总 expert 数量 N 达到 O(10⁶) 级别。由于不同 expert 共享 hidden neuron（通过 multi-head retrieval 动态组合），singleton expert 隐式实现知识共享和参数效率。

从算法pipeline角度拆解术语：
Singleton expert 的计算过程（单个 token x ∈ R^{d_model}）：
```
def singleton_expert(x, u_i, v_i):
    # u_i ∈ R^{d_model}, v_i ∈ R^{d_model}
    z = dot(u_i, x)          # 标量: 内积 → 单神经元激活输入
    a = σ(z)                  # 标量: 非线性激活 (ReLU/GELU/SwiGLU)
    output = a * v_i          # R^{d_model}: 标量 × 向量 = 缩放
    return output
```
与标准 MoE expert 的对比：标准 expert FFN(x) = W₂ σ(W₁ x)，W₁ ∈ R^{d_ffn×d_model}, W₂ ∈ R^{d_model×d_ffn}，参数量为 2×d_model×d_ffn。Singleton expert 参数量为 2×d_model（d_ffn = 1），减少 d_ffn 倍（通常为 4×d_model 倍，即减少数千倍）。

H 个 singleton expert 的聚合等价于一个 h 神经元 MLP：
```
output = Σ_{j=1}^{h} σ(u_j^T x) v_j = V σ(W^T x)
# 其中 W = [u₁,...,u_h] ∈ R^{d_model×h}
#      V = [v₁,...,v_h] ∈ R^{d_model×h}
```

术语一般如何实现？
Singleton expert 的权重 u_i, v_i 存储在 Embedding 层中（w_down_embed 和 w_up_embed），通过 product key 检索到的索引进行 lookup。计算通过 einsum 批量完成所有选中的 singleton expert。PEER 使用 h=8 heads × k=16 experts/head = 128 个 active singleton expert，等效于动态组装 128 神经元 MLP。可扩展至 GLU 变体（额外添加 linear gating 权重）。当前实现为 JAX 原型。

涉及论文标题：
- Mixture of A Million Experts

## Fine-Grained MoE Scaling Law / Granularity（细粒度 MoE 缩放定律 / 粒度）

术语是什么？
Fine-grained MoE Scaling Law 由 Krajewski et al. (2024) 提出，将 MoE 模型性能建模为总参数 P、训练 token 数 D 和粒度 G（active expert 数量）的函数：L(P, D, G) = c + (g/G^γ + a)/P^α + b/D^β。其中 G = P_active / P_expert，即每 token 激活的 expert 数量。与 Clark et al. (2022) 的早期 MoE scaling law 不同，该公式引入了粒度 G 作为独立 scaling axis，并证明在 compute-optimal 设置下，更高粒度（更多更小的 expert）一致优于低粒度（更少更大的 expert）。外推预测：持续提升模型容量最终需要极高粒度的大型模型，对应极大量极小专家的架构。PEER 直接将此理论推到极致：d_expert = 1 → G = hk，N = P/G（百万级）。

从算法pipeline角度拆解术语：
Scaling law 的推导逻辑：
```
给定: P (总参数), P_active (每 token 激活参数), P_expert (单个 expert 大小)
G = P_active / P_expert    (粒度 = active expert 数量)
N = P / P_expert = P × G / P_active   (总 expert 数量)

目标: 降低 loss L → 增大 P, D, G
约束: 限制 P_active (控制计算和内存开销)
策略: G ↑ → P_expert = P_active/G ↓ → N = P/P_expert ↑
结论: 需要大量小 expert 而非少量大 expert
```
PEER 的参数化：d_expert=1 → P_expert ≈ 2×d_model, P_active = hk × 2×d_model, G = hk。通过增加 N（product key 规模）仅增加总参数存储，不增加 P_active（每 token FLOPs 不变）。

术语一般如何实现？
PEER 直接基于此 scaling law 设计：(1) d_expert=1 最小化 P_expert；(2) Product Key 支持 N ≥ 10⁶；(3) hk 控制 P_active（固定计算预算）。Ablation 验证了预测：增加 N（128² → 1024²）单调改善 perplexity；增加 hk（32 → 512）改善性能但渐趋饱和，需权衡性能和资源。当前此 scaling law 在 Krajewski et al. (2024) 中为经验性发现，理论上的最优 G 与 P 的关系仍为开放问题。

涉及论文标题：
- Mixture of A Million Experts

## IsoFLOP Analysis（等计算量分析）

术语是什么？
IsoFLOP Analysis 是由 Borgeaud et al. (2022b) 引入的实验方法论：固定总计算预算（FLOPs），联合变化模型大小和训练 token 数，在验证集上绘制"模型大小 vs perplexity"曲线（isoFLOP 曲线）。曲线上的每个点具有相同的总计算成本，曲线最低点对应的模型大小即为该 FLOP 预算下的 compute-optimal 模型。IsoFLOP 分析避免了训练多个 model size 到收敛的高昂成本，用相对较小的 FLOP 预算来表征模型大小的缩放行为。PEER 使用 isoFLOP 分析（6e18 和 2e19 FLOPs）比较 PEER vs Dense FFW vs Coarse-grained MoE vs PKM 的性能-计算 trade-off。

从算法pipeline角度拆解术语：
IsoFLOP 分析流程：
```
固定 FLOP 预算 F:
for model_size in [M_1, M_2, ..., M_n]:
    # 计算 FLOPs per training step: flops_per_step(model_size, batch, seq_len)
    num_steps = F / flops_per_step
    train model with (model_size, num_steps) on C4
    record validation perplexity
plot: x = model_size, y = validation perplexity (isoFLOP curve)
compute_optimal = argmin(perplexity)
```
关键假设：同一 FLOP 预算下训练不同大小的模型，验证 perplexity 是模型大小的 U 形函数（过小欠拟合，过大过拟合/欠训练）。PEER 中 isoFLOP 曲线显示：稀疏替代方案（MoE/PKM/PEER）将曲线向下和向右移动——引入更大 P 但使用更少或相等的 P_active。

术语一般如何实现？
PEER 的具体参数：FLOP 预算 = 6e18 和 2e19，batch size = 128，sequence length = 2048，训练步数 = FLOP 预算 / 每步 FLOPs。对于每个方法（Dense/MoE/PKM/PEER），从同一 dense backbone 开始，取不同 model size（通过变化层数、attention heads、d_model），中间一层替换为对应方法，训练至相同 FLOP 预算后在 C4 验证集评估 perplexity。IsoFLOP 曲线在双对数坐标中呈现 U 形。

涉及论文标题：
- Mixture of A Million Experts

## Multi-Head Expert Retrieval（多头专家检索）

术语是什么？
Multi-Head Expert Retrieval 是 PEER 层中的核心设计：使用 h 个独立的 query network（类似于 Transformer 的 multi-head attention 和 PKM 的 multi-head memory），每个 head 独立计算 query 向量并从共享的 N 个 singleton expert 池中检索 k 个 expert。不同 head 的检索结果直接求和：f(x) = Σ_{i=1}^h Σ_{j ∈ I^i} g_j(x) e_j(x)。Multi-head 设计的核心价值：(1) 增加模型表达能力——h 个 head × k 个 expert/head = hk 个 active expert，动态组装等效 hk 神经元 MLP；(2) 共享 expert pool 实现参数复用——不同 head 可检索相同或不同的 expert，隐式实现 hidden neuron 共享；(3) 每个 head 的 router 可学习不同的检索偏好，增强 expert 池的利用多样性。

从算法pipeline角度拆解术语：
Multi-Head Expert Retrieval 的计算过程（替换单头版）：
```
# 单头版本: h=1
q¹(x) = query_net₁(x)           # R^d
I¹ = ProductKeyRetrieve(q¹, C, C', k)  # k 个 expert 索引
output = Σ_{j∈I¹} gⱼ(x) · σ(uⱼ^T x) vⱼ

# 多头版本: h>1
for i in 1..h:
    qⁱ(x) = query_net_i(x)       # h 个独立 query network
    Iⁱ = ProductKeyRetrieve(qⁱ, C, C', k)  # 各自检索 k 个 expert
    # 共享相同的 N 个 expert 和 product keys C, C'
output = Σ_{i=1}^h Σ_{j∈Iⁱ} gⱼ(x) · σ(uⱼ^T x) vⱼ
```
等效关系：当 k=1 时，PEER 的 h 个 head 各检索 1 个 expert ≡ 1 个 h 神经元 MLP。

术语一般如何实现？
PEER 默认配置：h=8, k=16, hk=128 active experts。不同 head 的 query 通过 h 个独立线性层（或一个批量线性层）投影。get_indices 函数对每个 head 分别执行 product key 检索（当前实现未做 head 间检索共享优化）。每个 head 从共享的 N 个 expert（Embedding 层存储）中独立检索，检索到的索引可能重叠（不同 head 选到同一 expert）。Ablation 研究了 h 和 k 的最优组合：给定固定 hk，最优 h 随 hk 增大而增大。

涉及论文标题：
- Mixture of A Million Experts

## Query Batch Normalization for MoE Router（MoE 路由器的查询批归一化）

术语是什么？
Query Batch Normalization 是 PEER（继承自 PKM Lample et al. 2019）中应用于 query network 输出上的 Batch Normalization 层，目的是提升 expert 使用的均匀性。由于 product key 的子密钥在训练中可能形成不均匀分布，某些子密钥被频繁选中，导致部分 expert 过载而其他闲置。在 query vector 上添加 BN 层后，query 分布更加均匀（零均值、单位方差），使得 product key 检索到的候选集覆盖更广的 expert 空间。PEER 实验表明：使用 query BN 后，expert 使用率接近 100%（甚至对 N=1M），unevenness（KL 散度，衡量 expert 分布与均匀分布的偏离程度）显著降低，perplexity 也有所改善。

从算法pipeline角度拆解术语：
Query BN 在 PEER 前向传播中的位置：
```
# 无 BN 版本
q = query_proj(x)                           # 可能分布不均
indices, scores = product_key_retrieve(q, ...)

# 有 BN 版本
q = query_proj(x)
q = BatchNorm(q)                            # 标准化 query 分布
indices, scores = product_key_retrieve(q, ...)
```
BN 的作用机制：在训练 batch 上计算 query 的均值和方差，归一化后使各维度 scale 一致。这使得 product key 检索中 q₁^T c_i 和 q₂^T c'_j 的分布更均匀，不同子密钥被选中的概率差异减小。

术语一般如何实现？
标准 BatchNorm1d 应用于 query 向量的 d 维度。PEER 实验中默认启用 query BN。消融实验对比了 16K-1M experts 下使用/不使用 BN 的 expert usage 和 unevenness（表 2）：使用 BN 时 unevenness 约降低 30-50%（如 1M experts: 1.52→1.06），perplexity 从 20.73 降至 20.64（1M experts, C4）。Query BN 在 isoFLOP 最优区域附近改善最明显（Fig. 4）。

涉及论文标题：
- Mixture of A Million Experts

## Expert Usage / Unevenness（专家使用率 / 不均衡度）

术语是什么？
Expert Usage 和 Unevenness 是 PKM (Lample et al., 2019) 提出的 MoE 专家利用评估指标，PEER 沿用这些指标评估百万级 expert 的利用效率。给定验证集上所有 token 的路由分数累积：z_i = Σ_x g_i(x)（expert i 在所有 token 上的 router score 之和），定义：(1) Expert Usage = 被至少一个 token 选中的 expert 比例：#{i | z_i ≠ 0} / N；(2) Unevenness = z 分布与均匀分布之间的 KL 散度：log(N) + Σ_i (z_i / ||z||₁) log(z_i / ||z||₁)。Unevenness 越小表示 expert 使用越均衡，0 表示完全均匀。

从算法pipeline角度拆解术语：
评估计算过程：
```
# 遍历验证集所有 token，累积每个 expert 的 router score
z = zeros(N)                     # 初始化
for each token x in validation_set:
    indices, scores = peer_forward.get_indices_and_scores(x)
    for (i, s) in zip(indices, scores):
        z[i] += s                # 累积 router score

p = z / sum(z)                   # 归一化为概率分布

# Expert Usage: 被使用的 expert 比例
usage = count(p > 0) / N

# Unevenness: KL(p || uniform)
uniform = ones(N) / N
unevenness = KL(p || uniform) = sum(p_i * log(p_i / (1/N)))
           = log(N) + sum(p_i * log(p_i))
```

术语一般如何实现？
PEER 在 C4 验证集上评估所有 expert usage 指标。实验表明：即使 N=1M，expert usage 接近 100%（使用 query BN 时 100%，不使用 BN 时 99.8%）。Unevenness 随 N 增大而上升（16K: 0.30→1M: 1.06 with BN），表明更大 expert 池中负载均衡更具挑战性，但 BN 可将 unevenness 控制在可接受范围。这些指标仅评估覆盖度（哪些 expert 被使用），不评估 expert 是否学到了有意义的专业化功能。

涉及论文标题：
- Mixture of A Million Experts

## LoRA Composition（LoRA 组合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
LoRA Composition（LoRA 组合）是指将多个分别独立训练好的 LoRA adapter（每个适配特定任务/概念，如服装风格、面部特征、语言翻译能力等）组合为一个统一模型，使组合后模型同时具备各 LoRA 的特化能力。核心挑战：(1) 直接叠加多个 LoRA（$\hat{W} = W + \sum \Delta W_i$）在 N≥3 时会破坏预训练模型的生成能力（参数偏移过大）；(2) 归一化加权叠加（$\hat{W} = W + \sum w_i \Delta W_i, \sum w_i=1$）虽保护了生成能力，但会稀释每个 LoRA 的独有特征（各 w_i ≈ 1/N）。LoRA 组合的两大流派：(a) Linear Arithmetic Composition（线性算术组合）——直接对 LoRA 权重矩阵做加权求和；(b) Reference Tuning-based Composition（参考调优组合）——用小规模参考数据重训练整个模型以融合 LoRA 输出。

从算法pipeline角度拆解术语：
Linear Arithmetic Composition 的前向计算：
```
# 对于每个 transformer block 的每个线性层:
W ∈ R^{d×k}              # 预训练权重（冻结）
ΔW_i = B_i @ A_i         # 第 i 个 LoRA 的增量权重, B_i∈R^{d×r}, A_i∈R^{r×k}
w_i                       # 第 i 个 LoRA 的组合权重 (Σ w_i = 1)

# NLA (Normalized Linear Arithmetic) 组合权重:
W_hat = W + Σ_{i=1}^{N} w_i · ΔW_i

# 前向: y = W_hat @ x = W @ x + Σ_{i} w_i · (B_i @ A_i @ x)
```

关键特性：w_i 是全局标量，所有 transformer 层共享同一组 {w_i}。组合权重可在推理前一次性 merge（W_hat 预计算），推理时无额外开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Linear Arithmetic：最简单的实现是加载所有 LoRA weights → 按 w_i 缩放 → merge 到 base model（peft.merge_and_unload）。PEMs (Zhang et al. 2023) 定义 LoRA 算术算子（加法/减法），LoRAHub (Huang et al. 2023) 用 gradient-free 优化（CMA-ES）在 few-shot 样例上自动搜索 {w_i}。
- Reference Tuning-based：Mix-of-Show (Gu et al. 2023) 使用梯度融合 + 可控采样 + 位置 mask，但需要全模型重训练，灵活性差。
- 适用场景：V&L 域的多概念图像生成（同时生成多个视觉主体）、NLP 域的多任务能力组合（翻译+NLI+QA 一次推理）。实际部署中常见于 Stable Diffusion 生态（Civitai 上的 LoRA 组合）和 LLM 多任务 serving。

涉及论文标题：
- Mixture of LoRA Experts
- PEMs: Composing Parameter-Efficient Modules with Arithmetic Operations
- LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
- Mix-of-Show: Decentralized Low-Rank Adaptation for Multi-Concept Customization of Diffusion Models
- SVDiff: Compact Parameter Space for Diffusion Fine-Tuning

## Mixture of LoRA Experts (MOLE / LoRA 专家混合)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture of LoRA Experts (MOLE) 由 Wu et al. (Microsoft Research Asia, 2024) 提出，是一种用于组合多个预训练 LoRA adapter 的方法。核心设计：(1) 将每个已训练 LoRA 的每一层视为一个独立 expert（而非整个 LoRA 为一个 expert），即 N 个 LoRA × M 个 transformer block 产生 N×M 个 expert；(2) 在每个 transformer block 层级嵌入一个可学习的 gating function $\mathcal{G}(\cdot)$，接收该层所有 LoRA 的输出 $\{E_{\Delta\theta_i}(x)\}_{i=0}^{N-1}$，输出 N 维 softmax 分布作为组合权重；(3) 训练时仅优化 gating 参数（e 向量和 τ 温度标量），冻结所有 LoRA 和预训练模型权重，极低训练开销；(4) 推理时支持双模式：全专家模式（所有 LoRA 参与，gating 自动分配权重）和 mask 模式（手动排除不需要的 LoRA，gating 按比例重新分配剩余权重，无需重训练）。

从算法pipeline角度拆解术语：
MOLE 单 transformer block 的前向计算（§3.2 Eq.5-13）：
```
输入: x ∈ R^{L×d}, 预训练 block θ, N 个 LoRA {Δθ_i}

# Step 1: 预训练 block 前向
F_θ(x) = x + f_Attn(LN(x)|θ) + f_FFN(LN(x + f_Attn(LN(x)|θ))|θ)

# Step 2: 每个 LoRA expert 独立前向（可并行）
for i in range(N):
    E_Δθi(x) = x + f_Attn(LN(x)|Δθ_i) + f_FFN(LN(...)|Δθ_i)

# Step 3: Gating 计算组合权重
E_Ω(x) = Normalize(concat([E_Δθ0(x), ..., E_Δθ{N-1}(x)]))  # [N·L·d]
ε = flatten(E_Ω(x))^T @ e                                    # e ∈ R^{N·L·d × N}
G_i = exp(ε_i / τ) / Σ_j exp(ε_j / τ)                        # τ learnable

# Step 4: 加权组合 + 残差融合
O(x) = F_θ(x) + Σ_i G_i · E_Δθi(x)
```
与 NLA 的关键区别：MOLE 在"block 输出空间"（而非"权重空间"）组合，每个 LoRA 需独立计算完整 block 输出。

MOLE 与 LoRA-based MoE（如 MixLoRA、MOELoRA）的本质区别：
| 维度 | LoRA-based MoE | MOLE |
|------|---------------|------|
| Expert 定义 | LoRA adapter 作为 expert | 整个 LoRA block 输出作为 expert |
| Router 位置 | 每个 token 路由到 expert | 每层 gating 加权组合所有 expert |
| 训练方式 | 联合训练 router + LoRA | 仅训练 gating（LoRA 预训练并冻结） |
| LoRA 来源 | 同一训练任务中学习 | 独立预训练的多个 LoRA |
| 应用场景 | 多任务微调 | 多 LoRA 组合/融合 |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 论文声明开源在 github.com/yushuiwx/MoLE.git（2026年已 404）。
- V&L 域训练：基于 DreamBooth + Stable Diffusion V2.1，400 iterations，lr=1e-5，batch=2，α=0.5（L = L_CLIP + α·L_balance）。CLIP 提供 local+global guidance 作为无监督训练目标。
- NLP 域训练：基于 FLAN-T5，800 iterations，lr=1e-5，batch=12，α=0.5（L = L_task + α·L_balance）。
- 适用：需要将多个独立获取的 LoRA adapter（如社区发布的角色/风格 LoRA）组合使用的场景。推理灵活性（mask 模式）使其适合交互式 LoRA 组合（用户手动选择保留哪些 LoRA 特征）。

涉及论文标题：
- Mixture of LoRA Experts

## Normalized Linear Arithmetic (NLA) Composition（归一化线性算术组合）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NLA (Normalized Linear Arithmetic) Composition 是 LoRA 组合中最简单、最广泛使用的 baseline 方法（MOLE Eq.2）。对 N 个预训练 LoRA adapter 的增量权重做加权求和：$\hat{W} = W + \sum_{i=1}^{N} w_i \Delta W_i$，其中约束 $\sum w_i = 1$。归一化约束防止了直接叠加（Eq.1: $\hat{W} = W + \sum \Delta W_i$）在 N 增大时导致的权重膨胀和生成能力退化。但代价是每个 LoRA 的有效贡献被压缩到约 1/N，导致个体 LoRA 的区分性特征被稀释。

从算法pipeline角度拆解术语：
```
# NLA 的权重合并（推理前一次性操作）:
for each linear layer with weight W ∈ R^{d×k}:
    ΔW_merged = zeros(d, k)
    for i in 1..N:
        ΔW_merged += w_i * (B_i @ A_i)   # w_i ∈ [0,1], Σw_i=1
    W_hat = W + ΔW_merged

# 推理时: y = W_hat @ x  (与标准推理完全相同，零额外开销)
```

NLA 的关键局限（MOLE Observation 1 & 2）：
1. 所有层共享同一组 {w_i}：忽略了不同层编码不同特征
2. w_i 由人工指定或 heuristic 搜索，缺乏数据驱动的逐层自适应
3. 当 N≥3 时稀释效应显著，个体 LoRA 特征被噪声淹没

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现最简单：加载所有 LoRA weights → 按 w_i 缩放 → merge_to_base_model。PEMs 和 LoRAHub 均以 NLA 为基础。
- 适用：快速原型、LoRA 数量少（≤2）的场景。常见于 Stable Diffusion WebUI 中的 LoRA weight slider。

涉及论文标题：
- Mixture of LoRA Experts
- LoRAHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition
- PEMs: Composing Parameter-Efficient Modules with Arithmetic Operations

## Gating Balancing Loss（门控平衡损失）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gating Balancing Loss ($\mathcal{L}_{\text{balance}}$) 是 MOLE 提出的辅助损失函数，用于防止可学习 gating function 在训练过程中坍塌到仅激活少数 LoRA expert。问题根源：训练初期表现较好的 LoRA 会获得越来越高的 gating 概率，形成正反馈循环 → gating 熵持续下降 → 最终 68% 权重集中在单个 LoRA。该损失鼓励 gating 分布在所有 block × 所有 LoRA 上的联合分布尽可能均匀。

从算法pipeline角度拆解术语：
```
# 输入: M 个 block 的 gating 输出, N 个 LoRA
for i in 1..N:                    # 对每个 LoRA
    q_i = (1/M) * Σ_{k=1}^{M} exp(ε_i^k / τ) / Σ_j exp(ε_j^k / τ)

L_balance = -log(Π_{i=0}^{N} q_i)  # Eq.14
          = - Σ_i log(q_i)         # 等价形式
```

数学性质：当所有 q_i = 1/N 时 L_balance 最小。对数积形式对极端不平衡施以极强惩罚（某个 q_i → 0 时 log → -∞ 使 L_balance → +∞）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 总训练目标：L = L_D + α·L_balance，α=0.5（论文所有实验统一取值）。L_D：V&L 用 CLIP guidance，NLP 用 FLAN-T5 cross-entropy。
- 替代方案对比（Table 7）：仅调大 τ 虽可缓解不平衡但会丧失 gating 区分能力——MOLE^{τ1/τ2/τ3}（温度递增）性能单调下降（78.07→77.45→76.71→76.35），均低于带 L_balance 的 MOLE（78.07）。
- MOLE w/o L_balance 在 NLP NLI 任务上平均 77.57 vs MOLE 78.07（-0.50）。

涉及论文标题：
- Mixture of LoRA Experts

## Hierarchical Weight Control in LoRA（LoRA 层级权重控制）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical Weight Control（层级权重控制）是 MOLE 的核心设计理念：不同于 NLA 对所有 transformer 层使用统一组合权重，MOLE 在每层独立学习一组 gating 分布，实现"不同层对不同 LoRA 赋予不同权重"。其理论基础是 MOLE Observation 2：单个 LoRA 的不同层编码了不同特征——V&L 域中浅层控制"毛色/耳朵形状"而深层控制"背景风格"；NLP 域中 0%-20% 层擅长 QNLI，80%-100% 层擅长 ANLI-R1。因此理想的 LoRA 组合应在不同层给不同 LoRA 不同权重。

从算法pipeline角度拆解术语：
MOLE 的 coarse-to-fine gating 层级划分（Table 9）：
```
n-MoLE (network-wise):   1 个 gating，全局统一权重  → 最粗粒度
b-MoLE (block-wise):     每 transformer block 1 个 gating
l-MoLE (layer-wise):     每 transformer sub-layer 1 个 gating
m-MoLE (matrix-wise):    每个参数矩阵 1 个 gating  → 最细粒度

V&L text-alignment 实验结果:
n-MoLE: 0.722 → 粒度过粗，无法区分层间差异
m-MoLE: 0.731 → 粒度过细，过度控制破坏 LoRA 参数间的内在关联
l-MoLE: 0.760
b-MoLE: 0.766 → 最佳，block 级在灵活性和稳定性间取得平衡
```
论文默认使用 b-MoLE（= "MoLE"），即每 transformer block 独立 gating。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在选定层级的每个位置插入 gating 模块（concat → normalize → flatten → dot-product → softmax）。对于 L 层、N 个 LoRA，b-MoLE 有 L 个独立 gating。
- 可视化验证（Fig. 7）：MOLE 确实学习到非均匀的层间权重分布——LoRA A 在浅层权重 45%、深层 12%；LoRA C 相反（浅层 8%、深层 52%），自动复现 Observation 2 的层特异性。
- 粒度选择是性能-灵活性的 trade-off：更细粒度 = 更强表达能力但可能过拟合。

涉及论文标题：
- Mixture of LoRA Experts

## Learned Index Structure for Neural Routing（基于学习索引的神经路由）

术语是什么？
Learned Index Structure 是 Kraska et al. (2018) 提出的一种数据库索引替代方案：使用机器学习模型（如 B-Tree 的神经网络替代品）来预测数据位置，而非使用传统数据结构（如 B-Tree、Hash Table）。PEER 论文声称首次将 learned index structure 应用于 MoE 路由——product key 可视为一种可学习的索引结构：两组子密钥 C, C' 是可学习的参数，查询网络 q 学习如何将输入映射到最相关 expert 对应的索引位置。与传统数据库索引不同，这里的"索引"不仅要考虑查找效率，还要根据输入语义选择最优 expert。PEER 路由器（product key + query network）的复杂度为 O(√N)——亚线性于 expert 数量 N，与可学习索引的目标（替代 O(log N) 的传统索引）一致。

从算法pipeline角度拆解术语：
Product Key 作为 learned index 的工作方式：
```
传统 MoE router（Token Choice / Expert Choice）：
    类似暴力扫描：计算所有 N 个 expert 的 score → TopK
    复杂度: O(N) — "无索引"，必须扫描所有 expert

Hash-based MoE router（Hash Layers, MoWE）：
    类似 Hash 索引：hash(token) → expert index
    复杂度: O(1) — 但 hash 函数固定（不可学习）

PEER router（Product Key）：
    类似 Learned Index：训练子密钥 C, C' 和 query net q
    q(x) 预测 x 在 product key 空间中的位置 → 检索附近 expert
    复杂度: O(√N) — 可学习的亚线性索引
```

术语一般如何实现？
Product Key 本身尚未在数据库领域作为 learned index 的标准实现；PEER 论文首次将其用于神经网络路由。子密钥 C, C' 存储为可学习参数（Embedding 矩阵），通过梯度下降端到端训练。与数据库 learned index 的区别：目标不是最小化查找延迟，而是在高维语义空间中检索语义相关的 expert。未来可能的扩展方向：更复杂的 learned index（如 RMI, PGM Index）应用于更大规模 expert 池。

涉及论文标题：
- Mixture of A Million Experts

---

## Expert Choice Routing (Expert-Choice MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Choice Routing（专家选择路由）是 Zhou et al. (2022) 提出的一种 MoE 路由策略，与传统的 Token-Choice Routing（每个 token 独立选择 top-k 个专家）相反，该方法让每个**专家**独立选择 top-k 个**token**。具体流程：(1) 计算 token-to-expert affinity matrix S = Softmax(X · W_g) ∈ R^{n×e}（与传统方式相同）；(2) 转置得到 S^T ∈ R^{e×n}，对每个专家（每一行）取 TopK，k = n×c/e（c 为容量系数，e 为专家数），得到 G, I = TopK(S^T, k)；(3) 通过排列矩阵 P = OneHot(I) 将 token 按专家分组：X_in = P · X ∈ R^{e×k×d}；(4) 各专家独立计算 FFN；(5) 反排列回原始顺序：X_out = Σ P[i,j,l] · G[i,j] · X_e[i,j,d]。核心创新：(a) 每个专家恰好处理 k 个 token，负载天然完美均衡，无需 auxiliary load balancing loss；(b) 每个 token 可被 0~e 个专家选中（实际分布：~77% tokens 被 1-2 个专家选中，~23% 被 3-4 个），实现可变计算分配。训练收敛速度比 GShard top-2 gating 快 2× 以上，每步 latency 快 20%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Choice Routing 的完整计算流程：
```
输入: X ∈ R^{n×d}  (n = batch_size × seq_len)
      W_g ∈ R^{d×e}  (gate 参数)
      c (容量系数), e (专家数)
输出: X_out ∈ R^{n×d}

# Step 1: affinity 计算（与 token-choice 相同）
S = Softmax(X @ W_g)  ∈ R^{n×e}

# Step 2: 专家选 token（核心区别——对 S^T 的每行即每个专家取 top-k）
k = n × c / e
G, I = TopK(S^T, k)
# G ∈ R^{e×k}: 门控权重，G[i,j] = S[I[i,j], i]
# I ∈ R^{e×k}: I[i,j] = 第 i 个专家选的第 j 个 token 的全局索引

P = OneHot(I)  ∈ R^{e×k×n}  # 排列矩阵

# Step 3: shuffle — 按专家分组 token
X_in = P @ X  ∈ R^{e×k×d}

# Step 4: 专家 FFN（每个专家批量处理 k 个 token）
for i in 1..e:
    X_e[i] = GeLU(X_in[i] @ W_1[i]) @ W_2[i]^T
# X_e ∈ R^{e×k×d}

# Step 5: unshuffle — 反排列回原始 token 顺序，门控加权
X_out[l, d] = Σ_{i=1..e} Σ_{j=1..k} P[i,j,l] × G[i,j] × X_e[i,j,d]
```
与 Token-Choice 的关键区别：TopK 应用于 S^T 的行（专家维度）而非 S 的行（token 维度），k 由全局容量决定（k = n×c/e），而非每个 token 固定选 k 个专家。可选的约束版本 EC-CAP：通过熵正则化线性规划 + Dykstra 交替投影算法限制每个 token 最多 b 个专家，λ=0.001，max 100 iterations。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 官方实现：Google Research `sparse_mixers/routing.py`（JAX/Flax）包含 `ExpertsChoose` 路由；Flaxformer/T5X 也集成此机制。
- 第三方 PyTorch 实现：`pytorch-mixtures` 提供 `ExpertChoiceRouter`。
- 训练配置：c=2 匹配 GShard top-2 计算量，c=1 匹配 Switch Transformer top-1。无需 load balancing loss。最大模型 8B/64E 使用 512 TPU V4 chips，GSPMD 2D sharding。
- 局限：不直接适用于 auto-regressive 生成（需要 future tokens）；小 batch inference 时需改用 global top-k + cap 策略。

涉及论文标题：
- Mixture-of-Experts with Expert Choice Routing

## Token-Dropless MoE Training（无 Token 丢弃 MoE 训练）

术语是什么？
Token-dropless MoE Training 是一种 MoE 训练范式，指不通过 capacity_factor 限制丢弃 token 来达到负载均衡，而是处理 Router 分配给每个 expert 的**全部** token（包括极端不均衡的情况）。由于不丢弃 token，dropless 训练的信息保留更完整，模型质量更高（MegaBlocks 实验：dropless 的 validation loss 改善 0.26 比 capacity_factor=1 的 0.15，1.73× 改善）。MoE Parallel Folding 框架通过灵活的 token dispatcher 同时支持 dropless 和 token-dropping 两种训练范式。

从算法pipeline角度拆解术语：
Token-dropless 训练的流程（以 Mixtral 8x22B, EP=8, ETP=1 为例）：

```
1. Router: probs, indices = Router(local_input)  # 每个 token 分配 top-k expert
2. 无 capacity 检查：不丢弃任何 token
3. Permutation: 将同一 expert 的 token 紧凑排列（变长 batch per expert）
4. All-to-All-V: 跨 EP 组发送变长 token batch 到对应 expert 所在 rank
5. Expert GEMM: 各 expert 独立计算（batch size 可变）
6. All-to-All-V: token 返回原 rank
7. Unpermutation + weighted sum

注意：因 batch size 可变，无法使用统一 batched GEMM。
Megatron-Core/MegaBlocks 使用 GroupedGEMM 或 block-sparse kernel 处理变长 expert batch。
```

术语一般如何实现？如何使用？
- Megatron-Core 支持 dropless 训练作为默认范式（benchmark 时可用 token-dropping 避免负载不均的性能抖动）
- 实现需变长 token 处理能力：All-to-All-V（可变消息长度）、GroupedGEMM（不等大小 batch）、block-sparse kernel
- Token-dropless 训练可能因负载不均导致某些 expert 处理极大量 token，造成 straggler 问题。此时可结合 MoE Parallel Folding 优化通信，缓解瓶颈

涉及论文标题：
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

## Model FLOPs Utilization (MFU)（模型 FLOPs 利用率）

术语是什么？
Model FLOPs Utilization (MFU) 是衡量大模型训练硬件效率的核心指标，定义为模型实际达到的 FLOPs 与硬件理论峰值 FLOPs 的比值：MFU = Actual TFLOPS / Peak TFLOPS。MFU 考虑了训练中所有开销（通信、kernel launch、pipeline bubble、重计算等），是端到端训练效率的综合度量。Megatron-LM (Narayanan et al. 2021) 引入这一指标并给出 dense LLM 的 MFU 上限约 52-57%（A100 上）。

从算法pipeline角度拆解术语：
MFU 的计算过程：
```
1. 模型每步 FLOPs：基于模型参数和 micro-batch 配置的理论计算量
   - MoE 模型：仅计算激活的 expert FLOPs（稀疏计算），而非全部参数 FLOPs
   - MFU = 实际 TFLOPS / 峰值 TFLOPS → 其中"峰值"通常取 BF16 理论峰值（H100: 989.5 TFLOPS/GPU）

2. 通信模型对 MFU 的衰减：
   - TP 通信: 2 × bsh (n-1)/n 字节（AG + RS），恒定占比
   - EP 通信: 2 × k/n × bsh (n-1)/n 字节（A2A × 2），随 n 增大而减小
   - MFU ≈ T_compute / (T_compute + T_comm + T_bubble)
   其中 T_compute ∝ FLOPs / Peak, T_comm ∝ Communication_vol / Bandwidth

3. 本论文结果：
   - Mixtral 8x22B (w/ Folding): 49.3% MFU，128 H100 GPU
   - Qwen2-57B-A14B (w/ Folding): 39.0% MFU，64 H100 GPU
   - Fine-grained MoE 的 MFU 低于 coarse-grained: 更小的 expert hidden size → 更低 GEMM 效率 + 更高通信占比
```

术语一般如何实现？如何使用？
- 开源计算脚本：https://github.com/NVIDIA/Megatron-LM 中的 MFU 计算逻辑
- 使用 benchmark 模式（token-dropping, CF=1）消除负载不均的性能抖动，获取稳定的 MFU 读数
- MFU 用于指导并行策略选择：比较不同 (tp, ep, cp, pp) 配置下的 MFU，选最优配置
- 对 MoE 模型，MFU 远低于 dense 模型（因稀疏激活 + all-to-all 通信开销）

涉及论文标题：
- Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM
- MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core

## Zero-Computation Expert (零计算专家)

术语是什么？
Zero-Computation Expert 指在推理阶段不执行任何矩阵乘法或激活函数计算的 expert。已有两种不同的实现路径：

**路径一（MoE++）：异构零计算专家。** 三类零参数的专家类型：(1) Zero Expert：输出恒为零 E_zero(x)=0，使 Top-2 退化为 Top-1；(2) Copy Expert：输出等于输入 E_copy(x)=x，相当于残差 shortcut；(3) Constant Expert：输出为 α1·x + α2·v，其中 [α1,α2]=Softmax(W_c·x)。三种专家参数极少（zero/copy 零参数，constant 仅 O(D) 参数），与标准 FFN 专家混合部署。

**路径二（MoLE）：重参数化零计算专家。** 训练时 expert 是标准 FFN，以 embedding tokens 为输入，所有 experts 同时激活。推理前将所有 expert 输出预计算为 Lookup Table (LUT)：LUT_l = {{v_j^i}_{j=1..N}}_{i=1..|V|}，其中 v_j^i = FFN_j(Embedding(i))。推理时 expert 计算被替换为 LUT lookup：仅按 input_ids 检索预计算的 v_j^i，然后通过 router 加权求和 h' = Σ_j g_j·v_j^i + FFN_shared(h) + h。Routed experts 的推理 FLOPs 从 4dND_r 降至 0（仅 lookup + weighted sum）。

**MoLE vs MoE++ 核心区别：** MoE++ 的 zero-computation expert 是设计时就确定的特殊类型（zero/copy/constant），训练和推理结构一致；MoLE 的 expert 在训练时是正常 FFN，通过 training-inference decoupling 和 reparameterization 在推理前转换为 LUT。MoLE 的 expert 总参数量远大于 MoE++（LUT size = dN|V|），但 per-token 加载量仅 dN。

从算法pipeline角度拆解术语：
**MoE++ 路径：**
```
selected_experts, probs = top_k_router(logits, k=2, capacities=C)
y = 0
for idx, p in zip(selected_experts, probs):
    if type[idx] == FFN:
        out = FFN(x)
    elif type[idx] == ZERO:
        out = torch.zeros_like(x)
    elif type[idx] == COPY:
        out = x  # identity
    elif type[idx] == CONST:
        alpha = softmax(W_c @ x)
        out = alpha[0] * x + alpha[1] * self.v
    y += p * out
```
计算复杂度：O(τ·N_FFN·T/(τ·N_FFN + N_ZC))。

**MoLE 路径（训练→重参数化→推理）：**
```
# 训练: Expert 接受 embedding tokens 输入，全激活
e = Embedding(input_ids)           # [b, s, d]
for j in 1..N:
    routed_output += g_j * FFN_j(e)  # g_j = SoftMax(Router(h))

# 重参数化 (推理前一次性):
for j in 1..N:
    V_j = FFN_j(W_emb)             # [|V|, d], W_emb = embedding 权重
LUT = stack([V_1, ..., V_N])       # [|V|, N, d]

# 推理: Expert 计算替换为 LUT lookup
v = LUT[input_ids]                 # [b, s, N, d] — 零 FLOPs
routed_output = Σ_j g_j * v[:,:,j,:]
```
MoLE 训练时 FFN_j 接受 embedding tokens e = Embedding(input_ids) 而非中间特征 h。因为 e 仅由离散 input_ids 决定，输入空间从连续 R^d 收缩为有限集 |V|（vocab size），使得 LUT 预计算成为可能。

术语一般如何实现？如何使用？
- MoE++ 代码：https://github.com/SkyworkAI/MoE-plus-plus（Apache 2.0，ICLR 2025），在 Megatron 中定义 FFN/ZERO/COPY/CONST 四种专家类型
- MoLE 代码：https://github.com/JieShibo/MoLE（ICML 2025），训练使用 modeling_mole.py（embedding as expert input + 全激活），推理使用 modeling_mole_rep.py（LUT lookup 替代 expert 计算）
- MoLE 的 LUT 可进一步压缩：NF4 量化将 LUT 从 3.5GB 降至 0.9GB，NF3 降至 0.7GB，性能几乎无损（Table 8）
- MoLE 关键 trade-off：LUT 存储开销大（dN|V|），但 per-token 传输量极小（dN），适合大容量存储设备 offloading

涉及论文标题：
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts
- Mixture of Lookup Experts

## Gating Residuals (门控残差) / Pathway-Aware Router (路径感知路由器)

术语是什么？
Gating Residuals 是 MoE++ 提出的路由增强机制：将前一层 MoE++ 的路由分数（softmax 概率分布）通过可训练变换矩阵 W_g∈R^{N×N} 融入当前层的路由计算中。具体公式：G(x^j) = W^j·x^j + W_g^j·G(x^{j-1})（j>1 时），首层仅使用 W^1·x^1。这种设计使每个 token 在选择当前层专家时能"记住"前一层走过的路径，保证异构专家架构下的路由稳定性。带有 Gating Residuals 的 Router 称为 Pathway-Aware Router。

从算法pipeline角度拆解术语：
```
# 第 j 层 MoE++ routing
logits = W @ x  # [B, S, N], 当前层的基础路由分数
if j > 1:
    logits += W_g @ prev_gating_scores  # [N×N]@[B,S,N]→[B,S,N]
gating_scores = softmax(logits, dim=-1)  # 存为下一层的 prev_gating_scores
selected_indices, selected_probs = topk_with_capacity(gating_scores, k=2, capacities)
```

W_g 是可学习的 N×N 矩阵，显式建模层间专家选择的相关性。实验证明（Fig.6）：Gating Residuals 降低了路由分数的方差，但不改变均值和值域范围，因此稳定了异构专家架构的路由。

术语一般如何实现？如何使用？
- 实现为 Megatron 中 MoE Router 的扩展：在 forward 时传入 prev_gating_scores，用额外的线性层 W_g 做变换后加到 logits 上
- W_g 初始化为小值或零，在训练中学习层间路由关联
- 消融实验（Tab.6）显示 Gating Residuals 在 1B 模型上提升 average benchmark 约 0.2 个百分点（47.5→47.7）
- 适用场景：任何多层 MoE 架构，特别是异构专家结构（不同层可能有不同专家类型分布时更有价值）

涉及论文标题：
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

## Heterogeneous Load Balance Loss (异构负载均衡损失)

术语是什么？
Heterogeneous Load Balance Loss 是 MoE++ 为异构专家（FFN vs. 零计算专家）设计的负载均衡损失函数。标准 MoE 负载均衡损失将所有专家视为等价，但 MoE++ 中 FFN 专家和零计算专家的参数量和计算量差异巨大，统一分配 token 不合理。该损失引入超参数 τ 控制零计算专家与 FFN 专家的 token 分配比例：L_b = Σ η_i·f_i·P_i，其中 f_i 为专家 i 被选中频率，P_i 为平均路由概率，η_i=1（FFN 专家）或 τ（零计算专家）。较小的 τ 将更多 token 分配给零计算专家（更高 throughput），较大的 τ 将更多 token 分配给 FFN 专家（通常更高性能）。

从算法pipeline角度拆解术语：
```
# 异构负载均衡损失计算
f_i = mean(Indicator(token selects expert i))  # [N], 每个专家的选中频率
P_i = mean(Softmax(G(x))_i)                     # [N], 每个专家的平均路由概率
eta_i = 1.0 if is_ffn[i] else tau               # 权重：FFN=1, ZC=τ
L_b = sum(eta_i * f_i * P_i for i in range(N))
L_total = L_ce + beta * L_b  # beta=0.01
```

配套的异构专家容量公式：C_i_FFN = γ·τ·T/(τ·N_FFN+N_ZC)，C_i_ZC = γ·T/(τ·N_FFN+N_ZC)。当 τ=1 时退化为标准均匀分配；τ<1 时零计算专家获得更高容量。默认 τ=0.75，capacity factor γ=1.1。

术语一般如何实现？如何使用？
- 在 Megatron 训练代码的 MoE 层中对 load balance loss 计算做修改：按专家类型分组统计 f_i 和 P_i，分别乘 η_i
- τ 的选择是 throughput-accuracy trade-off：τ=0.10 时 expert forward throughput 提升最高（164.5%）但 accuracy 下降；τ=0.75 时平衡（~25% throughput 提升，accuracy 持平或略优于 baseline）
- 预训练时 τ 可以保持固定，未来工作可探索自适应 τ 策略（如训练早期大 τ 保证学习、后期小 τ 加速）

涉及论文标题：
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts

## MoE Router (Top-K Gating Network)

术语是什么？
MoE Router（也称Gating Network/门控网络）是稀疏MoE模型中的核心路由组件，决定每个输入token被分配到哪些expert进行计算。Router通常是一个小型线性层（W_g ∈ R^{d_model × n_expert}），对每个token的hidden state计算logits，经Softmax后得到每个expert的概率分布，然后选择概率最高的top-K个expert（K通常为1-4）。MoE-CAP分析了多种路由变体：(1) 纯Top-K路由（如Mixtral-8x22B, K=2）；(2) Top-K+Shared Experts路由（如Qwen1.5-MoE, K=4+4 shared，DeepSeek-R1, K=8+1 shared）；(3) 高expert数+低K路由（如Switch-C, 128 experts, K=1）。Router的设计直接影响模型稀疏性（activated/total参数比）和负载均衡。

从算法pipeline角度拆解术语：
MoE Router在单个MoE layer中的计算流程：
```
# 输入: hidden_state h ∈ R^{d_model}（单个token）
# W_g ∈ R^{d_model × n_expert}: router权重
# 输出: selected_experts 和对应的gate weights

logits = h @ W_g                          # [n_expert] 原始logits
probs = Softmax(logits)                    # [n_expert] 概率分布
topk_vals, topk_indices = TopK(probs, K)   # 选top-K expert索引和概率

if has_shared_experts:                     # 若有shared experts
    # shared experts始终激活（如Qwen1.5-MoE: 4 shared experts）
    selected_experts = topk_indices ∪ shared_expert_indices

# 归一化gate weights（仅对选中的expert）
gate_weights = Softmax(topk_vals)          # re-normalize

for each selected expert i:
    expert_output_i = gate_weights[i] * FFN_i(h)

output = Σ expert_output_i                 # 加权求和
```
MoE-CAP通过追踪Router输出的topk_indices来记录每层𝟙[l,i]（expert i是否被激活），从而计算S_activated。S-MFU中的k_expert参数也直接来自模型配置的K值和shared expert数。

术语一般如何实现？
HuggingFace Transformers中MoE Router通常实现为`nn.Linear(d_model, n_expert)`后接Top-K选择。主流serving框架（vLLM, SGLang）在MoE layer实现中集成fused routing kernel以降低router开销。MoE-CAP在router后植入probe记录激活模式用于S-MBU计算。Router的负载均衡是主要挑战——部分expert可能被过度使用（load imbalance），导致GPU闲置。

**MoE-Pruner 的补充**：MoE-Pruner (Xie et al., 2024) 首次将 Router 权重显式纳入 MoE expert 层的剪枝度量。其核心洞察是：Router 输出的 Gate 权重向量（经 Softmax 归一化后）反映了"该 expert 对当前 token 的重要性"——若某 expert 对一批 token 的 Gate 权重极低，其权值即使 magnitude 大也应优先剪除。MoE-Pruner 的剪枝度量 S = |W_ij| * ||X_j * Gate_j|| 在 Wanda（S = |W_ij| * ||X_j||）基础上多乘了一个按元素广播的 Gate 权重项，使被 Router 抑制的 expert（Gate ≈ 0）的权值重要性被降低，被 Router 强激活的 expert（Gate ≈ 1）的权值重要性被保留。这是首次将 MoE routing 信息从"选择哪些 expert"扩展到"在 expert 内部哪些权值更重要"的粒度。

**Nexus 的 Adaptive Domain-Embedding Router 变体**：Nexus 提出了一种从根本上不同于线性 router 的路由机制——不学习 W_r ∈ R^{h×n}，而是学习投影函数 P_r 将域嵌入映射为 expert embedding：
```
# Nexus Router（每个 MoE block）
domain_embeddings: [m, n_experts]  # 预计算的域嵌入（如 Cohere Embed v3 编码）
expert_embeddings = P_r(domain_embeddings)  # [h, n_experts], P_r = 2-layer SwiGLU MLP
router_probs = softmax(inputs @ expert_embeddings)  # [batch, seq, n_experts]
index, gate = topk(router_probs, k=1)
```
与线性 router 的关键区别：(1) Router 参数不随 expert 数量增长——P_r 是固定的投影函数，输入为域嵌入 d_i 而非 token x；(2) 路由基于 token 与各域 expert embedding 的点积相似度，具有域语义归纳偏置；(3) 新 expert 加入时仅需计算 d_new → P_r(d_new) = e_new，无需修改 router 参数维度。Nexus 的实验显示此 router 对 load balancing loss 因子不敏感（低至 0.0005 时仍稳定），而线性 router 下降约 2%。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
- Opportunistic Expert Activation: Batch-Aware Expert Routing for Faster Decode Without Retraining

## Shared Experts in MoE

术语是什么？
Shared Experts（共享专家）是MoE架构的一种变体：在MoE layer中设置一组始终激活的expert（shared experts），所有token都必须经过它们计算，再结合一组通过router选择性激活的routed experts。典型配置：Qwen1.5-MoE每个MoE layer有60个routed experts（每token选top-4）+ 4个shared experts；DeepSeek-R1有256个routed experts（每token选top-8）+ 1个shared expert。Shared experts的引入使得MoE层既保持了sparsity的计算优势，又通过shared experts保证了基础表示能力——所有token都经过shared部分处理，避免了某些token因routing不当而完全遗漏重要特征。

从算法pipeline角度拆解术语：
含Shared Experts的MoE layer计算流程：
```
h = input_hidden_state                    # 当前token

# Shared Experts（始终激活，对所有token）
shared_output = Σ_{i=1}^{n_shared} FFN_shared_i(h)

# Routed Experts（选择性激活）
logits = h @ W_g                         # Router计算
probs = Softmax(logits)
topk_vals, topk_indices = TopK(probs, K)  # K=4 routed experts

routed_output = 0
for each selected expert i:
    routed_output += gate_weights[i] * FFN_routed_i(h)

output = shared_output + routed_output    # 合并
```
从S-MBU角度：shared experts对应的𝟙[l,i]恒为1（i=1..4），routed experts的𝟙[l,i]需通过profiler追踪。因此vanilla MBU的高估程度在含shared experts的模型上相对较低（batch size=1时高估约1.5×而非3×），因为shared部分始终计入S_model。

**MoLE 中的 Shared Expert：** MoLE 的 shared expert 保持标准 FFN 计算（不接受 embedding tokens 输入，接受中间特征 h），推理时执行标准 SwiGLU 计算（FLOPs = 4dD_s）。MoLE 的 shared expert 承担了 routed experts 被 LUT 化后缺失的"上下文相关"计算能力——因为 LUT-based routed experts 的输入不含上下文信息（仅 input_ids），shared expert 仍从中间特征 h 中提取上下文信息。这种"shared expert（有计算）+ routed LUT experts（无计算）"的组合实现了 FLOPs 等同于同大小 dense model（FLOPs_MoLE = 4dD_s = FLOPs_dense）。

术语一般如何实现？
Shared experts在模型config中以独立参数组存在（如Qwen的`shared_expert_intermediate_size`），HuggingFace Transformers在MoE layer forward中先计算shared experts再计算routed experts，最后合并。DeepSeek-MoE论文[10]首次系统提出shared experts设计，后续Qwen1.5-MoE[4]和DeepSeek-R1采用。MoLE codebase (https://github.com/JieShibo/MoLE) 中 shared expert 即为标准 MLP(config)，与 attention 权重一同常驻 VRAM，不参与 LUT offloading。

涉及论文标题：
- MoE-CAP: Cost-Accuracy-Performance Benchmarking for Mixture-of-Experts Systems
- Mixture of Lookup Experts
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**共享专家对局部路由一致性的影响** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文发现 Shared Experts 是降低 MoE 模型局部路由一致性（Local Routing Consistency）的重要因素。在所有 REAL 模型中，高 SRP 的 Group 1 和 Group 2 模型均不使用 shared experts；TOY 模型的 1ShrExp/2ShrExp 变体在相近 PPL 水平下 SRP 显著低于 Baseline。双重机制：(1) Bypass effect——更多信息由 shared 处理，使 routed expert 相对不重要；(2) 减小 expert combination space——从 C(64,8) 降至 C(62,6) 约 72×，限制了 router 在相邻 token 间做局部调整的能力。论文结论：若目标部署场景涉及 expert offloading，架构设计时应权衡 shared experts 的好处（保留通用能力）与 local routing consistency 的下降。

**Nexus 中 Shared Expert 的使用**：Nexus 将 seed model 的原始 FFN 作为 shared expert（而非像 BTX 那样作为普通 routed expert），目的是"更好地保留 upcycling 前 seed model 的通用语言能力"。在 Nexus 的 470M MoE 中：1 shared expert + 6 routed experts（top-1）= 每 token 激活 2 experts；2.8B MoE 中：1 shared + 4 routed。Shared expert 始终激活确保模型不会因 routing 失误而丢失基础能力——这在扩展新 Code expert 时尤为关键：shared expert 保留了非 Code 域的知识，防止 catastrophic forgetting。实验显示 Nexus 扩展 Code expert 后通用任务性能仅下降 1.9%（相对），而 shared expert 是实现这一稳定性的关键组件。

## Block Coordinate Descent (BCD) for MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Block Coordinate Descent (BCD) 是一种用于大规模非凸优化的迭代算法，核心思想是每次迭代仅更新一个参数块（block），保持其他参数固定，从而大幅降低单步内存和计算开销。在 MoE 训练语境下，MoE-DisCo 将 BCD 应用于 expert 级：每个 training step 仅更新一个 expert 及其共享 backbone，其余 expert 全部冻结。每次训练仅需维护等价于单 expert 分支的 dense 子模型，内存需求从 O(E) 降至 O(1)，使 MoE 训练可在低内存设备（如 24GB RTX 4090）上进行。各 expert 的训练完全并行、零通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 传统 MoE 训练：全部参数同时加载和更新
Θ = (θ_shared, θ_1, ..., θ_E)    # O(E) 内存
for batch in D:
    loss = M(Θ, batch).forward()  # 前向经过 gating + Top-K experts
    loss.backward()               # 反向遍历所有 expert 路径
    optimizer.step()              # 全部参数同时更新

# BCD for MoE (MoE-DisCo)：分块独立训练
for k in 1..E:                    # 可完全并行，零通信
    Θ_k = (θ_shared^(k), θ_k)    # 仅 1/E expert + 1 份共享参数
    for batch in D_k:
        loss = M(Θ_k, batch).forward()  # 无 gating，固定 single expert
        loss.backward()                 # 反向仅遍历 expert k
        optimizer.step()               # 仅更新 Θ_k
```

训练阶段移除 gating 机制，子模型退化为标准 dense Transformer。关键优势：(1) 子模型参数量远小于完整 MoE；(2) 无跨设备通信开销（无 gradient/parameter 交换）；(3) 每个子模型可独立放入低成本 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 MoE-DisCo 中，BCD 通过四阶段 pipeline 落地：(1) Model Decoupling：完整 MoE 分解为 E 个 dense 子模型；(2) Data Decoupling：K-Means 聚类产生 E 个语义区分的数据子集；(3) Independent Parallel Training：各子模型在低成本 GPU 上独立训练；(4) Reintegration + Fine-Tune：expert 直接拼接，共享参数加权平均（WP-SGD），短期全局微调恢复 gating 协调性。实验验证：Qwen1.5-MoE-2.7B 上，BCD 策略将 69.5% 训练成本从 A100 移至 RTX 4090，且最终 PPL 和 downstream 性能不降反升。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

## SimulParallel SGD

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SimulParallel SGD（Simultaneous Parallel SGD，Zinkevich et al., 2011）是一种分布式训练优化框架，在多个计算节点上独立训练模型副本，每个副本使用互不相交的数据子集，训练完成后通过参数平均聚合。MoE-DisCo 受此启发，将其视为 MoE 的极端情况（uniform gating + All-K averaged output），并据此设计 expert 级分块训练。两个关键洞见被采用：(1) 最大化数据子集间分布差异可加速收敛并提升集成效果——通过 K-Means 聚类实现；(2) 数据均衡时简单参数平均可逼近全局最优，不均衡时需加权平均（即 WP-SGD）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SimulParallel SGD 核心流程
for k in 1..K:                      # K 个 worker，完全并行
    Θ^(k) = Θ_init                  # 复制初始参数
    for batch in D_k:               # 互不相交的数据子集
        Θ^(k) = Θ^(k) - η · ∇L(Θ^(k), batch)

# 聚合：简单平均（数据均衡）
Θ_final = (1/K) · Σ_{k=1}^{K} Θ^(k)

# 聚合：WP-SGD 加权（数据不均衡）
Θ_final = Σ_{k=1}^{K} (|D_k|/|D|) · Θ^(k)
```

MoE-DisCo 将此框架映射到 MoE：每个 worker 对应一个 expert 子模型 + 其 K-Means 数据子集。共享参数 θ_shared 按 WP-SGD 加权平均融合，expert 参数直接拼接（保持专业化差异）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SimulParallel SGD 为 MoE-DisCo 的两阶段训练设计提供理论保证：不同的 K-Means 数据簇天然最大化分布差异，实现 expert 专业化。消融实验确认聚类必要性——随机数据分配使 fine-tune 性能退化至 Full-Parameter 水平。框架的完全去中心化特性确保子模型训练期间零通信开销，仅需本地操作。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

## WP-SGD (Weighted Parallel SGD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
WP-SGD（Weighted Parallel SGD，Cheng et al., 2020）是 SimulParallel SGD 在数据不均衡场景下的扩展。当各 worker 分配的数据子集大小不一致时，简单参数平均产生有偏梯度估计。WP-SGD 引入样本数加权系数 γ_k = |D_k|/|D|，对各 worker 参数做加权平均以保持梯度无偏性。MoE-DisCo 将其用于共享 backbone 参数的融合阶段：θ_shared* = Σ γ_k · θ_shared^(k)。当 K-Means 产生平衡簇时 γ_k ≈ 1/E，退化为简单平均。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# WP-SGD 在 MoE-DisCo Reintegration 阶段
total = Σ_{k=1}^{E} |D_k|
for k in 1..E:
    γ_k = |D_k| / total

θ_shared* = Σ_{k=1}^{E} γ_k · θ_shared^(k)   # 加权平均
θ_exp* = Concat(θ_1, ..., θ_E)                # expert 直接拼接
Θ = (θ_shared*, θ_exp*)                       # 组装完整 MoE
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
WP-SGD 在 Reintegration 阶段执行一次（离线），权重 γ_k 由 K-Means 聚类后自然得到。在 MoE-DisCo 实验中，K-Means 通常产生大小相近的簇（γ_k ≈ 1/E），但 WP-SGD 作为理论保障确保聚类不均衡时不引入偏差。该框架可推广到任意分布式训练中数据量不一致的场景。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

## K-Means Data Partitioning for MoE Expert Specialization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE-DisCo 提出的基于无监督聚类的训练数据划分方法，将原始数据划分为 E 个语义区分的子集，各分配给一个 expert 以促进专业化。流程：(1) 预训练 embedding 层编码句子所有 token，mean pooling 得到固定维度句子向量 h_x；(2) K-Means（K=E）聚类最小化簇内平方距离和；(3) 每个簇映射为一个数据子集 D_k。目标函数：min Σ_{k=1}^{K} Σ_{h_x∈C_k} ||h_x - μ_k||²。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# K-Means 数据划分（离线，一次性）
for sentence x in D:
    for token x_i in x:
        e_i = Embedding(x_i)              # token embedding [d_embed]
    h_x = (1/n) · Σ e_i                   # mean pooling [d_embed]

{C_1, ..., C_E} = KMeans({h_x}, K=E)     # 聚类
for k in 1..E:
    D_k = {x | h_x ∈ C_k}                # 簇→数据子集映射
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
聚类在训练前离线执行一次。消融实验（Figure 6）验证：随机分配替代 K-Means 后 fine-tune 性能退化至 Full-Parameter 水平，证明语义区分的分配对 expert 专业化至关重要。该方法与 LRP（Latent Prototype Routing）和 domain-adaptive pre-training 的动机一致——通过无监督发现数据内在结构指导训练。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

## MoE Model Decomposition (MoE Model Decoupling)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE Model Decomposition（模型解耦）是 MoE-DisCo 的核心操作——将完整 MoE Θ = (θ_shared, θ_1, ..., θ_E) 分解为 E 个独立 dense 子模型 Θ_k = (θ_shared^(k), θ_k)。每个子模型包含：(1) 完整共享 backbone（embedding、attention、LayerNorm），参数被复制 E 份；(2) 仅一个 expert，所有 MoE 层移除 gating，固定使用该 expert。分解后子模型退化为标准 dense Transformer，参数量远小于完整 MoE。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 Qwen1.5-MoE-2.7B（E=4）为例：

```
# 原始 MoE 每层：Input → Attn → LayerNorm → Gating → Top-K Experts → Output
# 子模型 k 每层：  Input → Attn → LayerNorm → Expert_k FFN → Output
#                                           ↑ gating 移除，固定 expert k

# 分解
原始: Θ = (θ_shared, θ_1, θ_2, θ_3, θ_4)
分解: Θ_1 = (θ_shared^(1), θ_1), ..., Θ_4 = (θ_shared^(4), θ_4)

# 重组（Reintegration）
θ_exp* = Concat(θ_1, ..., θ_4)             # expert 拼接
θ_shared* = Σ γ_k · θ_shared^(k)           # WP-SGD 加权平均
Θ = (θ_shared*, θ_exp*)                     # 完整 MoE
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
关键优势：(1) 子模型约 1/E expert 参数 + 1 份共享参数，可放入 RTX 4090 24GB；(2) 训练完全独立无需分布式框架（NCCL/GLOO）；(3) embarrassingly parallel——expert 数量增加时边际成本近常数。Qwen1.5-MoE-2.7B 上 S-phase 成本仅 $1.79-$4.37（4×RTX 4090），远低于 baseline $6.93-$29.91（A100）。

涉及论文标题：
- MoE-DisCo: Low Economy Cost Training Mixture-of-Experts Models

## MoLE (Mixture of Lookup Experts)

术语是什么？
MoLE（Mixture of Lookup Experts）是一种训练-推理结构解耦的新型 MoE 架构。核心思想：训练时 expert 是标准 FFN，但以 embedding tokens（Embedding 层输出）而非中间特征为输入，且所有 experts 同时激活；推理前将所有 expert 输出预计算为 Lookup Table (LUT)，推理时 expert 计算被 LUT lookup 替代，实现零计算开销。MoLE 由 Jie et al. 在 ICML 2025 提出，代码开源：https://github.com/JieShibo/MoLE。

MoLE 的三个关键设计（对应 MoE 的三项缺陷）：
1. **Embedding as Expert Input**：将 routed expert 的输入从中间特征 h 改为 embedding tokens e = Embedding(input_ids)。e 仅由离散 input_ids 决定，输入空间从连续 R^d 收缩为有限集 |V|，使 LUT 预计算成为可能。代价：expert 无法直接访问上下文信息（由 shared expert 和 attention 层补偿）。
2. **全激活训练**：所有 N 个 routed experts 同时激活并接收梯度（不做 top-K 稀疏选择）。Router 输出全 N 维 SoftMax 权重。因为无稀疏性带来的 collapse 风险，无需 auxiliary loss，仅使用 LM cross-entropy loss。
3. **推理前重参数化**：训练后将每个 expert FFN_j 对 embedding 权重 W_emb ∈ R^{|V|×d} 做单次 forward pass，得到 LUT = {FFN_j(Embedding(i)) for j=1..N, i=1..|V|}。推理时 h' = Σ_j g_j·LUT[input_ids]_j + FFN_shared(h) + h。

从算法pipeline角度拆解术语：

**训练阶段（MoLE Decoder Layer）：**
```
输入: hidden_states (b,s,d), input_ids (b,s)
embedding_states = Embedding(input_ids)        # [b, s, d]

# 1. Self-Attention（标准）
h = RMSNorm(hidden_states)
h = Attention(h) + hidden_states

# 2. Shared Expert（接受中间特征，标准 SwiGLU FFN）
residual = h
h = RMSNorm(h)
shared_out = FFN_shared(h)                     # FLOPs: 4dD_s

# 3. Routed Experts（接受 embedding tokens, 全激活）
g = SoftMax(Router(h))                         # [b, s, N]
e = RMSNorm(embedding_states)
routed_out = Σ_{j=1}^N g_j * FFN_j(e)         # FLOPs: 4dND_r

# 4. 合并输出
h = residual + shared_out + routed_out
```
训练 FLOPs = 4d(D_s + ND_r)，包含所有 expert 计算。

**重参数化（训练后、推理前，一次性）：**
```
W_emb = Embedding.weight                        # [|V|, d]
for j in 1..N:
    V_j = FFN_j(W_emb)                          # [|V|, d]
LUT = {V_j}_{j=1..N}                            # size: N × |V| × d
```

**推理阶段（LUT lookup 替代 expert 计算）：**
```
# 1-2. Attention + Shared Expert：同训练
# 3. Routed "Experts" (零计算)
g = SoftMax(Router(h))                          # [b, s, N]
v = LUT[input_ids]                              # [b, s, N, d] — O(1) lookup
routed_out = Σ_j g_j * v[:,:,j,:]               # 仅加权求和, 零 FFN FLOPs
```
推理 FLOPs = 4dD_s（同 dense model）。Per-token 加载参数量：仅 dN（LUT lookup results）。

术语一般如何实现？如何使用？
- 开源：https://github.com/JieShibo/MoLE（ICML 2025），含 modeling_dense.py / modeling_moe.py / modeling_mole.py / modeling_mole_rep.py
- HuggingFace checkpoints: JieShibo/MoLE-{160M,410M}-{4E,16E}
- 训练配置：Pythia 架构、bf16、Adam(β1=0.9,β2=0.95)、100B Pile tokens、GPT-NeoX tokenizer (|V|=50k)、cosine LR decay
- 适用场景：VRAM 受限的推理部署，LUT 可 offload 到 CPU/disk，per-token 通信量可忽略（~KB 级 vs MoE 的 ~MB 级）
- 局限：(1) LUT 存储开销大（dN|V|），可通过 NF4/NF3 量化压缩至 20-25%；(2) expert 无法直接访问上下文信息；(3) 仅适用于有固定 vocabulary 的语言模型

涉及论文标题：
- Mixture of Lookup Experts

## Embedding as Expert Input（Embedding 令牌作为专家输入）

术语是什么？
Embedding as Expert Input 是 MoLE 的关键设计选择：将 routed expert 的输入从 Transformer 中间层的 hidden states h（含上下文信息的连续向量）改为 embedding 层的输出 e = Embedding(input_ids)。这一修改的根本目的是限制 expert 的输入空间——从无限连续的 R^d 收缩为有限离散集 |V|（vocabulary size），从而使 expert 可以被重参数化为 Lookup Table。直接代价是 expert 丧失了直接访问上下文信息的能力（补偿：shared expert 和 attention 层仍处理中间特征）。

从算法pipeline角度拆解术语：
```
# 标准 MoE (expert 接受中间特征)
h = Attention(LN(x)) + x                # 中间特征，含上下文
g = SoftMax(Router(h))                  # Routing 基于上下文
expert_out = FFN_j(h)                   # Expert 看到完整上下文表示

# MoLE (expert 接受 embedding tokens)
e = Embedding(input_ids)                # embedding 输出，不含上下文
h = Attention(LN(x)) + x                # 中间特征仍用于 Attention + Router
g = SoftMax(Router(h))                  # Routing 基于上下文（context-aware）
expert_in = RMSNorm(e)                  # Expert 只看到词级别信息
expert_out = FFN_j(expert_in)           # 对纯词 embedding 做变换
output = Σ_j g_j * expert_out + FFN_shared(h) + h
```

关键：Router 的输入仍是中间特征 h（含上下文），因此 routing 决策本身是 context-aware 的。MoLE ablation 显示此修改仅造成 0.7 point AVG 下降（Table 7），但带来 expert 可重参数化 + 全激活的收益。

术语一般如何实现？如何使用？
- PyTorch 实现：`embedding_states = self.expert_layernorm(embedding_states)` → `routed_output = torch.stack([expert(embedding_states) for expert in self.routed_expert], dim=2)`
- expert_layernorm（RMSNorm）确保 embedding 输入的 scale 与中间特征一致
- 适用条件：必须存在离散输入空间（如 language token vocabulary），|V| 不能过大（否则 LUT 存储不可接受）
- 不适用：视觉模型（连续像素输入）、语音模型（连续频谱输入）等无固定离散 vocabulary 的模态

涉及论文标题：
- Mixture of Lookup Experts

## FP8 Quantization for MoE Inference（MoE 推理的 FP8 量化）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
FP8（8-bit Floating Point）量化是一种将神经网络模型的权重和激活从 FP16/BF16（16-bit）降低到 8-bit 浮点精度的压缩技术。与 INT8 量化不同，FP8 保留了浮点数的指数位（exponent bits），因此具有更大的动态范围（dynamic range），特别适合需要同时表示极大值和极小值梯度的 LLM 推理场景。FP8 格式有两种常见变体：E4M3（4-bit exponent + 3-bit mantissa，精度更高，适合前向传播的权重和激活）和 E5M2（5-bit exponent + 2-bit mantissa，动态范围更大，适合梯度）。NVIDIA H100 GPU 的第四代 Tensor Core 原生支持 FP8 计算，FP8 Tensor Core 的峰值算力是 FP16/BF16 的 2 倍（H100 SXM5 的 FP8 峰值达 1979 TFLOPS vs FP16 的 989.5 TFLOPS）。在 MoE 推理场景中，FP8 量化可显著减少显存占用（将 expert 权重参数从 FP16 的 2 bytes/param 降至 1 byte/param），使更多 expert 参数可同时驻留在 GPU 显存中，或支持更大的 batch size。MoE-Inference-Bench 使用 GPTQ 和 AWQ 等 post-training quantization 方法实现 FP8 量化，并通过 vLLM 在 H100 上评估。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
FP8 量化 MoE 推理的算法 pipeline（以 Mixtral-8x7B on H100 + vLLM with FP8 为例）：

```
# MoE Layer with FP8 Quantization (pseudocode, per MoE layer)
# Input: hidden_states [batch_size, seq_len, hidden_dim=4096], dtype=FP16

# Step 1: Router (kept in FP16 for routing accuracy)
router_logits = fp16_matmul(hidden_states, W_gate_fp16)  # [B, S, num_experts=8]
gate_weights, topk_indices = topk(softmax(router_logits), k=2)

# Step 2: FP8 Quantized Expert FFN (quantized weights loaded as FP8)
for expert_id in range(8):
    tokens_for_expert = hidden_states[topk_indices == expert_id]  # FP16 input

    # FP8 matmul: input (FP16) × weight (FP8) → accumulate in FP32 → output (FP16)
    # H100 Tensor Core: FP8 E4M3 weights, FP16 inputs auto-promoted
    gate_out = fp8_matmul(tokens_for_expert, W_gate_fp8[expert_id])  # [n, 14336]
    gate_out = silu(gate_out)

    up_out   = fp8_matmul(tokens_for_expert, W_up_fp8[expert_id])    # [n, 14336]
    expert_hidden = gate_out * up_out                                  # element-wise

    expert_out = fp8_matmul(expert_hidden, W_down_fp8[expert_id])    # [n, 4096]

    # Weighted sum accumulation (FP32 for numerical stability)
    output[topk_indices == expert_id] += gate_weights[expert_id] * expert_out
```

MoE-Inference-Bench 的 FP8 vs FP16 关键发现（Section 6.1）：(a) FP8 在 batch size=64 时提供 25-30% 更高吞吐量（batch size 越大，FP8 优势越明显，因为显存节省允许更大的有效 batch）；(b) 在不同 sequence length 下 FP8 保持 20-25% 吞吐量优势（鲁棒于 context length 变化）；(c) FP8 在 compute-bound 和 memory-bound 两种场景下均有效——compute-bound 场景受益于 2× Tensor Core 算力，memory-bound 场景受益于减半的权重显存带宽需求。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FP8 量化在 MoE 推理中的实现方法：
- **Post-Training Quantization (PTQ)**：不需要重新训练，通过对已训练的 FP16 模型应用量化算法（GPTQ/AWQ）直接转换为 FP8。GPTQ 基于逐层 Hessian 矩阵的最小二乘优化，AWQ 基于 activation 感知的权重重要性缩放。MoE-Inference-Bench 使用这些方法。
- **框架支持**：vLLM 通过 PyTorch 的 `torch.fp8` 支持和自定义 FP8 kernel 进行 MoE 的 FP8 推理。TensorRT-LLM 支持 MoE 模型的 FP8 推理（NVIDIA blog 报告 Mixtral 8x7B 在 2×H100 上 FP8 streaming 达 38.4 req/s）。
- **硬件要求**：需要 H100 (SM90) 或更新架构（FP8 Tensor Core 支持）。A100 及以下不支持 FP8 Tensor Core。
- **精度保持**：MoE-Inference-Bench 中 FP8 量化不显著影响模型质量（Section 8 的准确率实验在 FP16 下运行），论文未报告 FP8 下的准确率对比。一般经验：FP8 E4M3 对推理精度影响极小（<0.1% perplexity 退化）。
- 局限：Router 和 attention 层通常保持 FP16/BF16（量化 router 会显著影响 routing 决策精度）；需要 per-tensor 或 per-channel 的 scaling factor 管理 FP8 的有限动态范围。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models

## Inter-Expert Pruning / Intra-Expert Pruning in MoE（MoE 的专家间剪枝 / 专家内剪枝）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE 剪枝是针对 Mixture of Experts 模型的两类结构化压缩策略：(a) **Inter-Expert Pruning（专家间剪枝）**：移除整个 expert 子网络及其对应的 router 权重行，减少总 expert 数量但保持 active experts 数量不变。例如，Mixtral-8x7B 有 32 层 × 每层 8 experts = 256 个 expert 实例，12.5% inter-expert pruning 移除每层 1 个 expert（即 32 个 expert 实例）。(b) **Intra-Expert Pruning（专家内剪枝）**：在不改变 expert 数量的前提下，缩减每个 expert 内部的 FFN dimension（intermediate/hidden dimension）。例如，25% intra-expert pruning 将每个 expert 的 FFN intermediate dimension 从 14336 缩减至约 10752，降低每 expert 的计算量但不减少 expert 数量。这两种剪枝策略有不同的内存-计算 trade-off：inter-expert pruning 直接减少模型总参数量（移除整列参数），intra-expert pruning 减少 per-expert 计算量。MoE-I²（Yang et al., EMNLP 2024）是该方向代表性工作，结合 inter-expert pruning（移除低重要性 expert）和 intra-expert low-rank decomposition（对保留 expert 进行低秩分解压缩）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
MoE 剪枝后的推理 pipeline（以 Mixtral-8x7B, 50% intra-expert pruning, FFN dim: 14336→7168）：

```
# Pruned MoE Layer Forward Pass (pseudocode)
# Pruning applied offline; inference uses pruned dimensions directly

# Pre-processing (offline): Pruning decision
# inter-expert: compute importance_score[i] = mean(|W_gate[i,:]|) + mean(|W_expert_i|)
#    → remove expert j if importance_score[j] < threshold (e.g., bottom 50%)
# intra-expert: compute per-channel importance of FFN weight columns
#    → remove columns with lowest importance_score (e.g., bottom 50%)

hidden_states = input  # [B, S, 4096]

# Router: #experts reduced if inter-pruning applied
router_logits = hidden_states @ W_gate_reduced   # [B, S, num_experts_surviving]
topk_weights, topk_indices = topk(softmax(router_logits), k=2)

for expert_id in surviving_expert_ids:  # fewer experts if inter-pruning
    tokens = hidden_states[topk_indices == expert_id]

    # FFN with REDUCED intermediate dim (if intra-pruning)
    # W_gate: [4096, ffn_dim_reduced=7168]  (originally 14336)
    gate_out = silu(tokens @ W_gate[expert_id])   # [n_tokens, 7168]
    up_out   = tokens @ W_up[expert_id]            # [n_tokens, 7168]
    hidden   = gate_out * up_out
    expert_out = hidden @ W_down[expert_id]        # [n_tokens, 4096]

    output[topk_indices == expert_id] += topk_weights[:, expert_id] * expert_out
```

MoE-Inference-Bench 的核心发现（Section 6.2）：(a) **50% aggressive pruning 反而显著提高吞吐量**（因为减少的总参数和计算量超过负载不平衡带来的损失）；(b) **12.5%/25% 低比例剪枝可能降低吞吐量**——因为剪枝引入了负载不均衡（某些 expert 成为瓶颈），但节省的计算量不足以补偿；(c) **OLMoE-1B-7B 对 intra-expert 剪枝容忍度高**（结构设计使其计算分布更均匀）；**Qwen1.5-MoE-A2.7B 更敏感**（高剪枝比例在低 TopK 时吞吐量显著退化）。这表明剪枝策略需要模型特定的调优。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MoE 剪枝的实现方法：
- **重要性评估**：常见方法包括 weight magnitude（权重大小）、activation-based importance（激活值感知）、Taylor expansion（梯度×权重近似精度损失）、以及 expert 使用频率（inference 时激活次数）。MoE-Inference-Bench 具体使用的方法论文未详细说明，但引用了 [29]（Lu et al., 2024, "Not all experts are equal"）和 [48]（Yang et al., 2024, "MoE-I²"）。
- **剪枝时机**：post-training one-shot pruning（无需重新训练，直接基于预训练模型的权重/激活统计做剪枝决策）。
- **框架集成**：剪枝在模型加载前完成（weight matrix 直接缩减），推理框架无需感知剪枝过程——只需加载剪枝后的较小权重矩阵。
- 局限：(a) inter-expert pruning 移除 expert 后 router 也要同步更新（移除对应输出维度），可能改变 routing 行为；(b) 剪枝比例的选择是模型和任务特定的；(c) 激进的剪枝可能导致某些 token 的 routing 选择范围受限（所有被路由到的 expert 都被剪枝）。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

## Expert Activation Frequency in MoE（MoE 中的专家激活频率）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Activation Frequency（专家激活频率）是衡量 MoE 推理过程中每个 expert 被 gating network 选择（激活）次数的指标。在推理时，每个 token 经过 router 后选择 top-k 个 expert，统计所有 token 在整个推理过程中对每个 expert 的选择次数，即得到每 expert 的 activation frequency。这个指标反映了：(a) 哪些 expert 被频繁使用（热门 expert），哪些几乎不被使用（冷门 expert）；(b) expert 负载分布的均匀程度；(c) 训练时负载均衡损失（auxiliary load balancing loss）的实际效果。MoE-Inference-Bench (Section 8.3) 通过 activation frequency heatmap 可视化每层每个 expert 的激活次数，对比了 DeepSeek-VL2 系列（经过精心训练的负载均衡）和 MolmoE-1B 模型的 activation pattern。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Expert Activation Frequency 的计算过程：

```
# Activation frequency counting during MoE inference
# Method 1: Inference-time counting (used in MoE-Inference-Bench)
activation_counts = zeros(num_layers, num_experts_per_layer)  # [L, E]

for each batch in evaluation_dataset:
    hidden_states = model.forward_first_layers(batch)  # up to first MoE layer

    for layer in moe_layers:
        router_logits = layer.router(hidden_states)     # [B, S, E]
        _, topk_indices = topk(softmax(router_logits), k)

        for e in range(num_experts):
            activation_counts[layer][e] += count(topk_indices == e)

        hidden_states = layer.forward(hidden_states, topk_indices)

# Normalize to activation frequency (per 1000 tokens or absolute counts)
activation_freq = activation_counts / total_tokens * 1000
```

MoE-Inference-Bench 的关键发现（Section 8.3, Figure 15）：(a) DeepSeek-VL2 系列模型显示相对均匀的 activation pattern，各 expert 和 layer 间的激活分布接近——这是因为 DeepSeek-V2 在训练时加入了 auxiliary loss 来平衡 expert 利用率；(b) MolmoE-1B 显示更稀疏的 activation pattern，某些 expert 的激活次数远超其他（最高达 1M 次，而 DeepSeek-VL2 最高约 290K），形成明显的"热门-冷门"expert 分布；(c) **关键洞察**：Activation frequency 在 well-balanced 模型中不是评估 expert 重要性的可靠指标——因为均匀分布下所有 expert 被激活次数接近，无法通过 frequency 区分哪些 expert 对模型质量更重要。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **离线统计**：在代表性 benchmark 或 validation dataset 上运行一次完整推理，记录路由决策。MoE-Inference-Bench 使用 MME benchmark dataset 对 VLM 模型进行 activation frequency 统计。
- **在线监控**：在 production inference 中持续监控 activation frequency，用于检测：(a) 某些 expert 长期不被激活（可考虑剪枝）；(b) expert 负载漂移（可能由于 data distribution shift）。
- **可视化**：通常以 heatmap 形式呈现，x 轴为 layer index，y 轴为 expert index，颜色深度代表激活次数/频率。MoE-Inference-Bench 的 Figure 15 是典型例子。
- **与 expert importance 的关系**：在负载不均的模型中，activation frequency 低可能意味着 expert 可被安全剪枝；在 well-balanced 模型中（如 DeepSeek-VL2），activation frequency 不能单独作为重要性指标，需要结合 weight magnitude、gradient-based importance 等多维指标（参见 MoE-I² 的 ϕ·w multi-factor importance）。
- 局限：(a) 依赖于具体的 evaluation dataset（不同数据集的 token 分布不同，activation pattern 也不同）；(b) 需要额外的 hook/callback 机制来捕获每层 router 的 topk_indices 输出。

涉及论文标题：
- MoE-Inference-Bench: Performance Evaluation of Mixture of Expert Large Language and Vision Models
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

**MoESD 的理论扩展**：MoESD 从概率角度推导了激活专家数的闭式表达式 N(t) = E × (1 - ((E-K)/E)^t)，其中 E 为总 expert 数，K 为每 token 激活的 expert 数，t 为输入 token 数。推导假设各 expert 激活独立同分布（均匀路由），MoESD 实验验证与实际模型行为高度一致。进一步定义全激活阈值 T_thres = ⌈log_{(1-ρ)}(1-τ)⌉（τ 通常取 0.95，ρ=K/E 为 sparsity），当 batch size B ≥ T_thres 时几乎所有 expert 同时激活。此时每 expert 平均处理 token 数 Texp(t;ρ) = ρt/(1-(1-ρ)^t)，证明 ρ 越小（越稀疏）→ Texp 越小 → 系统更 memory-bound → SD 验证的计算增量近乎免费。该分析是 MoESD 证明"中等 batch size 下 SD 对稀疏 MoE 更有效"的理论基础。

## Roofline Model (Classical / 经典Roofline模型)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Roofline Model 是 Williams, Waterman, Patterson (2009) 提出的可视化性能分析模型，用于评估给定应用在特定硬件上的性能上限和优化方向。模型核心关联两个参数：Operational Intensity I（FLOPs/Byte，即计算量/内存访问量，作为 X 轴）和 Achievable Performance P（FLOPs/sec，作为 Y 轴）。模型画出两条"屋顶"：(1) Memory Roof——斜线 P ≤ B_peak × I，由峰值内存带宽 B_peak 决定，表示数据供给速率对性能的上限；(2) Compute Roof——水平线 P ≤ P_peak，由处理器峰值算力决定，表示计算能力对性能的上限。两条屋顶的交点称为 Ridge Point，对应的 operational intensity 为 critical intensity Ī = P_peak / B_peak。应用若 I ≥ Ī → compute-bound（黄色区域）；若 I < Ī → memory-bound（蓝色区域）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 LLM 推理场景中，不同算子的 operational intensity 差异巨大：(1) Attention softmax：I ≈ 1-2 FLOPs/Byte（对每个 KV pair 仅做乘加和 exp），memory-bound——受限于 HBM BW；(2) Linear projection / FFN GEMM：I ≈ 100-500 FLOPs/Byte（大矩阵乘法，数据复用率高），compute-bound——受限于 GPU FLOPS。以 A100 为例，B_peak ≈ 2TB/s, P_peak ≈ 312 TFLOPS (FP16)，Ī = 312T/2T = 156 FLOPs/Byte。Attention 的 I 远低于 156 → memory-bound；FFN GEMM 的 I 可在 100-500 → 可能 compute-bound。

Roofline 分析流程：
```
1. Profile or calculate: Ops = total FLOPs of kernel, Bytes = total DRAM bytes accessed
2. I = Ops / Bytes
3. If I >= P_peak / B_peak: compute-bound → optimize via better algorithm, mixed precision
   Else: memory-bound → optimize via kernel fusion, data reuse, quantization
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 经典 Roofline 分析工具：Intel Advisor、NVIDIA Nsight Compute（自动计算 kernel 的 I 并绘制 roofline chart）、Empirical Roofline Toolkit (ERT)。
- MoE-Lightning 将 Roofline Model 扩展为 Hierarchical Roofline Model (HRM)，引入多层内存层次（CPU DRAM、GPU HBM、PCIe）和多处理器 compute roof，用于指导 CPU-GPU 混合推理的 operator placement 和 resource allocation。
- 局限：经典 Roofline 仅适合同构单处理器场景——对于 CPU-GPU 混合系统，需要 HRM 等扩展来建模跨层数据传输。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs
- MoE-SpeQ: Speculative Quantized Decoding with Proactive Expert Prefetching and Offloading for Mixture-of-Experts
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

**MoESD Roofline 应用**：MoESD 将 Roofline Model 应用于 SD speedup 性能建模（Algorithm 1）。核心设计 G(t; λRP, s) 函数——将 Ridge Point 的过渡区域建模为指数增长段后接线性段，λ<1 修正实际内存带宽利用率。MoE 专家部分的 modeling 引入 N(t)（激活专家数）和 Texp(t;ρ)（每专家平均 token 数）两个因子：N(t) 控制参数加载时间（memory access volume），Texp(t;ρ) 替代原始 t 作为 G() 的输入（因为每个 expert 仅处理分配到的 tokens 子集）。该设计解释了为何稀疏度 ρ 越小→Texp 越小→系统更 memory-bound→SD 加速窗口更宽。

**Amortization Roofline 变体**：MoE-SpeQ 将 Roofline 思想扩展到 speculative offloading 场景。X 轴改为 Amortization Intensity I_amort(k) = E[Accepted Tokens] / E[Synchronous I/O Bytes]（有用工作每字节同步 I/O），Y 轴为 Effective Throughput Θ(k)。两 Roof：Compute Roof（I/O 完美隐藏时上限）和 I/O Roof（斜率=B_PCIe）。在线 argmax_k Θ(k) 确定最优 draft length，受离线 SLO 约束 k_SLO。与 HRM 区别：HRM 指导 operator placement；Amortization Roofline 指导 speculation degree。

## Operational Intensity (计算强度/算术强度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Operational Intensity I（计算强度/算术强度）是 Roofline Model 的核心参数，定义为单位内存访问所执行的计算量：I = FLOPs / Bytes Transferred（单位：FLOPs/Byte）。它衡量计算的"数据复用率"——I 越高，意味着每次从内存取数据后做了更多计算，对内存带宽的依赖越小。Operational Intensity 是判断 kernel 瓶颈类型（compute-bound vs memory-bound）的关键指标：将 I 与硬件 critical intensity Ī = P_peak / B_peak 比较——I ≥ Ī 则为 compute-bound，I < Ī 则为 memory-bound。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Operational Intensity 的计算取决于具体算子：
- **GEMM (M×K · K×N)**：Ops = 2×M×N×K FLOPs, Bytes = (M×K + K×N + M×N) × sizeof(dtype)。当 M,N ≫ K 时，I ≈ min(M,N) / sizeof(dtype)（高数据复用）。
- **Attention (Q·K^T)**：Ops = 2×B×H×S×d FLOPs, Bytes = B×H×(S+d)×sizeof(dtype)。对 decode (S ≫ 1, d ≈ 128)，I ≈ d / sizeof(dtype) × 2S/(S+d) ≈ 常数（极低复用，GEMV 模式）。
- **LayerNorm**：Ops = 5×B×d FLOPs, Bytes = 2×B×d × sizeof(dtype)（每个元素访问一次计算一次），I ≈ 2.5 / sizeof(dtype) ≈ 1.25（FP16，极低）。

在 MoE-Lightning 的 HRM 中，定义了 General Operational Intensity I_x^i——计算任务 x 在内存层次 level i 的操作强度。MoE FFN 的 I 随 batch size N 增大而增大（N 增大意味着更多 token 共享同一组 weights），因此更大的 batch 可以增加 I 从而跨越 PCIe bandwidth roof。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 计算方式：理论分析（从模型配置计算 FLOPs 和 bytes）或 profiling（NVIDIA Nsight Compute 自动报告 kernel 的 arithmetic intensity via Metrics: sm__throughput, dram__bytes）。
- 在 policy 搜索中的应用：MoE-Lightning 使用理论 Operational Intensity（而非 profiling）构建 HRM 性能模型，仅需硬件峰值参数——因为理论计算足以比较不同策略之间的相对效果。

涉及论文标题：
- MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

## Wanda (Pruning by Weights and Activations)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Wanda（Pruning by Weights and Activations）是 Sun et al. (ICLR 2024) 提出的一种简单且高效的 LLM 后训练剪枝方法。其核心思想是：对线性层权重 W ∈ R^{d_row×d_col}，用一小批校准数据前向传播得到输入激活 X ∈ R^{b×d_col}，定义每个权值的剪枝重要性度量为 S_ij = |W_ij| * ||X_j||_2（即权重绝对值乘以对应输入维度的 L2 范数），然后在每个输出神经元内比较，保留重要性最高的 (1-p%) 个权值，其余置零。Wanda 不需要权重更新（no weight update），不需要计算 Hessian 逆矩阵（不像 SparseGPT），计算复杂度为 O(d_hidden²)，仅需一次前向传播。其度量近似来源于 OBS 框架的简化：丢弃 Hessian 非对角项，用激活协方差对角近似替代完整 Hessian 逆。

从算法pipeline角度拆解术语：
```
# Wanda 逐层剪枝
# 输入: 预训练 LLM, 校准数据 (128 seqs), 目标稀疏度 p%
for layer in model.layers:
    X = forward_until(layer, X_calib)       # 收集该层输入激活
    for W_name in [W_gate, W_up, W_down]:
        W = layer.W_name                     # W ∈ R^{d_row × d_col}
        col_norm = ||X||_2 along dim=0       # [d_col] 每维 L2 范数
        S = |W| * col_norm.unsqueeze(0)      # [d_row × d_col] 重要性
        for row in range(d_row):
            thresh = top_k(S[row,:], k=d_col*(1-p%))
            mask[row,:] = (S[row,:] >= thresh)
        W = W * mask                         # 不重要权值置零
    X = forward_layer(layer, X)              # 传递到下一层
```
Wanda vs SparseGPT：(1) 无需 O(d_hidden³) Hessian 逆计算；(2) 不做权重更新补偿重构误差；(3) 小校准集下更鲁棒。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch forward hook 捕获激活 → 列 L2 norm → sort → mask。官方代码：https://github.com/locuslab/wanda。支持 LLaMA、OPT 等 50% unstructured 和 2:4 semi-structured。
- 局限：对 MoE 缺乏 router 感知——所有 expert 用相同度量。MoE-Pruner 增加 router 权重项（S = |W_ij| * ||X_j * Gate_j||）改进此局限。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

## Expert-wise Knowledge Distillation (专家级知识蒸馏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert-wise Knowledge Distillation 是 MoE-Pruner (Xie et al., 2024) 提出的剪枝后性能恢复方法。与标准 KD（仅在 logits 层计算 KL/MSE）不同，它在 MoE 所有 l 层、每层 n 个 expert 的输出层面逐 expert 计算 teacher（未剪枝 pretrained）和 student（剪枝后）的 MSE。损失 L_KD = L_CE + λ * Σ_{j=0}^{l-1} Σ_{i=0}^{n-1} MSE(E_it^j, E_is^j)，λ 初始化为 L_CE / L_expert 以平衡两条损失。

从算法pipeline角度拆解术语：
```
for batch in data (1000 C4 samples):
    # Teacher forward (no_grad)
    for layer j: teacher_out[j][i] = expert_i_j(x) for i in 0..n-1
    # Student forward
    L_CE = CrossEntropy(student_logits, labels)
    L_expert = Σ_j Σ_i MSE(teacher_out[j][i], student_expert_i_j_output)
    λ = L_CE.item() / L_expert.item()
    L_total = L_CE + λ * L_expert
    optimizer.step()  # sparsity mask 保持
```
Mixtral-8x7B 50% 稀疏度：Expert-wise KD 将 zero-shot 准确率从 67.23 恢复到 68.40（原始 69.16），维持 99% 性能。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 在 Llama-Factory 中实现，full-parameter fine-tuning 保持 sparsity mask。lr=2e-5, cosine scheduler, 3 epochs, 16×H100-80GB, ~1小时。仅需 1000 条 C4 训练样本。
- Pretrained model 是天然 teacher（同结构直接对应蒸馏）。局限：需 16×H100 同时加载 teacher+student；full-parameter fine-tuning 计算不减；λ 仅启发式初始化。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

## N:M Semi-structured Sparsity (N:M 半结构化稀疏)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
N:M Semi-structured Sparsity 是 NVIDIA Ampere 架构引入的稀疏模式：每 M 个连续权值中保留 N 个非零。最常见 2:4——每 4 个权值保留 2 个（50% 稀疏），Tensor Core 直接跳过零值，接近 2× 加速。相比 unstructured sparsity（零值随机分布、难硬件加速），N:M 在硬件友好性和精度间取得平衡。

从算法pipeline角度拆解术语：
```
for row in d_row:
    for g in range(0, d_col, M=4):
        idx = argtop2(S[row, g:g+4])  # 保留最重要的 2 个
        mask[g:g+4] = 0; mask[g+idx] = 1
W_sparse = W * mask  # 2:4 pattern
```
MoE-Pruner 将 unstructured 度量 S 的 comparison group 从"每行"改为"每 M 个连续权值"扩展为 N:M。Mixtral-8x7B 2:4 下 perplexity 5.88（SparseGPT 7.09, Wanda 6.98）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Ampere+ Tensor Core 通过 cuBLAS/cuSPARSE 支持。局限：精度损失高于同比例 unstructured；非所有 GEMM 受益。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

## Load Balancing Score in MoE (MoE 负载均衡分数)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load Balancing Score 是 MoE-Pruner 提出的量化 MoE expert 激活均衡度的指标：s = σ/μ = sqrt(Σ(f_i - μ)²/n) / μ，f_i 为 batch 内 routed 到 expert i 的 token 数。所有 layer 取平均得模型整体分数。分数越低 → expert 越均衡；越高 → 越不均衡。

从算法pipeline角度拆解术语：
```
for layer in 1..l:
    for token in batch: f[TopK(Softmax(x @ W_g), k=2)] += 1
    s[layer] = std(f) / mean(f)  # 变异系数
S = mean(s)
```
发现：(a) Upcycling 模型（Mixtral, Qwen1.5-MoE, MiniCPM-MoE）分数低 → expert 均衡 → 适合 weight-level pruning；(b) Train-from-scratch 模型（DeepSeek-V2, OLMoE）分数高 → cold expert 可被安全 expert-level prune；(c) Qwen1.5-MoE 例外——虽用 upcycling 但打乱 expert 参数，行为似 train-from-scratch。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- C4 子集一次前向 hook 收集 topk_indices。用于指导 MoE 剪枝策略选择和评估训练质量。局限：仅计数未考虑 router 权重幅度。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

## One-Shot Post-Training Pruning (一次性后训练剪枝)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
One-Shot Post-Training Pruning 是一类无需重训练的模型压缩方法：在预训练模型上用少量校准数据（128 条 C4）一次前向计算权值重要性，生成并应用稀疏 mask，不更新权重。代表方法：SparseGPT（Hessian 逆+权重更新）、Wanda（仅激活范数）、MoE-Pruner（router 感知激活范数）。

从算法pipeline角度拆解术语：
```
X = sample(C4, 128)
for layer in model.layers:
    X = forward_until(layer, X)
    S = pruning_metric(W, X)       # Wanda: |W|*∥X∥; MoE-Pruner: |W|*∥X*Gate∥
    mask = row_topk_mask(S, p%)    # 每输出神经元保留 top-(1-p%)
    W = W * mask                   # 无权重更新
    X = forward_layer(layer, X)
```
MoE-Pruner 的 one-shot 特性：无需 retraining/weight update，O(d_hidden²) 复杂度，128 seqs 校准。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 单卡 H100 数分钟至数十分钟。MoE-Pruner 剪枝后用 Expert-wise KD 恢复（1小时/1000样本）。局限：对校准数据质和量敏感；非结构化稀疏无硬件加速；剪枝决策不可逆。

涉及论文标题：
- MoE-Pruner: Pruning Mixture-of-Experts Large Language Model using the Hints from Its Router

## Target Efficiency (目标效率 / SD系统瓶颈度量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Target Efficiency 是 MoESD 提出的一个新系统指标，用于衡量目标模型架构和工作负载（而非 draft 模型算法质量）对 Speculative Decoding speedup 的影响。定义为 Target Efficiency = T_T(B, 1) / T_T(B, γ)，即目标模型单 token 前向时间与多 token（γ 个 draft tokens）验证前向时间的比值。该比值越接近 1，说明验证多 token 的时间与单 token 解码时间接近，SD 的系统开销越低。与之互补的传统指标 acceptance rate α 衡量 draft 模型的算法准确性，但无法解释系统瓶颈——即使 α 相同，在不同 batch size 或不同模型架构下 SD speedup 可能差异巨大。Target Efficiency 将系统因素（batch size、模型架构、memory-bound vs compute-bound）从算法因素中解耦，帮助研究者独立理解"目标模型和 workload 对 SD 是否友好"。

从算法pipeline角度拆解术语：
Target Efficiency 直接取值于 SD 执行过程中的实测时间，计算简单但含义丰富：

```
# SD 一轮的组成
T_SD_round = γ × T_D(B, 1) + T_T(B, γ) + T_reject

# Speedup 公式（MoESD Eq. 4）
Speedup = σ × (γ + 1) / (γ × T_D(B,1)/T_T(B,1) + T_T(B,γ)/T_T(B,1) + T_reject/T_T(B,1))

# Target Efficiency 定义
Target_Efficiency = T_T(B, 1) / T_T(B, γ)
```

Target Efficiency 反映两种导致 T_T(B,γ)/T_T(B,1) 增大的因素：
- **(1) Compute-boundness**：大 batch 下模型进入 compute-bound，T_T(B,γ) ∝ γ → Target Efficiency → 1/γ（低）
- **(2) Extra memory loads**：小 batch 下 MoE 验证 γ tokens 激活更多 expert → 参数加载增加 → T_T(B,γ) > T_T(B,1) → Target Efficiency 下降

在中等 batch size（所有 expert 已激活但未 compute-bound）下，Target Efficiency ≈ 1（最优）。MoESD 实验（Fig. 2）验证 Target Efficiency 与 end-to-end SD speedup 趋势高度一致，而 acceptance rate 仅在小范围内波动，无法解释 speedup 大幅变化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：从 vLLM runtime logs 中提取 T_T(B,1) 和 T_T(B,γ)（分别来自单 token AR 解码和多 token SD 验证的计时），直接计算比值。vLLM 的 cudagraph optimization 和详细 timing report 使这一提取可行。
- 使用场景：(a) 评估 SD 在不同 batch size 下的适用性——Target Efficiency 曲线与 speedup 曲线高度相关，无需实际运行完整 SD 即可预测趋势；(b) 比较不同模型架构对 SD 的友好程度——MoE vs dense 的 Target Efficiency 对比揭示系统性优势；(c) 指导 SD 部署决策——若 Target Efficiency 持续 < 0.5，SD 可能不适合当前 workload。
- 局限：仅反映 target model 的时间比例，不捕获 T_D（draft model 开销）和 acceptance rate 的影响——完整 speedup 仍需三者结合。论文旨在通过 Target Efficiency 补充（而非替代）acceptance rate，提供更全面的 SD 加速理解。

涉及论文标题：
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

## Speculative Decoding (SD / 投机解码)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Speculative Decoding（投机解码）是一种无损的 LLM 推理加速技术，最初由 Leviathan et al. (ICML 2023) 和 Chen et al. (2023) 独立提出。核心思想：用一个小型 draft model（通常比 target model 小 10-100×）快速自回归生成 γ 个候选 tokens，然后用大 target model 并行验证这些候选 tokens 的正确性（一次 forward pass 处理所有 γ 个 tokens），通过 rejection sampling 丢弃 draft model 预测错误的 tokens。加速原理：target model 验证 γ 个 tokens 的计算时间 ≈ 单 token 解码时间（memory-bound 时），但可接受 σ×(γ+1) 个 tokens，因此 speedup ≈ σ×(γ+1)（理想情况）。MoESD 将此分析扩展到 MoE：指出验证时间 T_T(B,γ) 的额外开销来自 (1) compute-bound 导致的逐 token 计算增加和 (2) 验证多 token 时额外激活 expert 导致的参数加载增加。

从算法pipeline角度拆解术语：
```
# SD 一轮（per decoding round）
# 输入: prefix tokens P, target model M_T, draft model M_D, draft length γ

# Step 1: Draft（自回归）
draft_tokens = []
for i in 1..γ:
    logits_D = M_D.forward(prefix + draft_tokens)     # T_D(B, 1)
    next_token = sample(logits_D[-1])
    draft_tokens.append(next_token)

# Step 2: Verify（并行）
logits_T = M_T.forward(prefix + draft_tokens)          # T_T(B, γ)
# 一次 forward 同时处理所有 draft tokens

# Step 3: Rejection Sampling
accepted = []
for i in 1..γ:
    p_D = softmax(logits_D[i])
    p_T = softmax(logits_T[i])
    if random() < min(1, p_T[draft_tokens[i]] / p_D[draft_tokens[i]]):
        accepted.append(draft_tokens[i])
    else:
        accepted.append(sample_from_residual(p_T - p_D))
        break  # 后续 tokens 全部丢弃

# 本轮产出: len(accepted) 个新 tokens
# Speedup = (total_accepted_tokens) / (R × (γ×T_D + T_T(B,γ) + T_reject))
```

变体：(a) **Eagle**：用集成在 target model 内的 trained speculation head 替代独立 draft model，利用 feature-level uncertainty 提升 acceptance rate；(b) **Tree-structured SD**（SpecInfer, Medusa, Eagle-2/3）：一次生成多分支 draft token tree 而非单链，扩大候选空间；(c) **Self-speculative SD**：target model 自身早期层作为 draft（无需额外模型）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 框架支持：vLLM（batched SD + cudagraph）、SGLang、TensorRT-LLM（Medusa）、HuggingFace Transformers（assisted generation API）。
- Acceptance rate α 是核心算法指标——α 越高，每轮产出 tokens 越多。MoESD 补充指出 α 无法解释系统瓶颈（如 MoE 的 expert 激活开销或 batch size 效应），需结合 Target Efficiency。
- MoESD 的关键新发现：传统观点认为 SD 对大 batch 和 MoE 无效（T_T(B,γ) 显著增长），但中等 batch size 下 MoE 所有 expert 已激活 → 验证不增加参数加载 → SD 反而对 MoE 加速效果优于 dense 模型（尤其在稀疏度高的 MoE 上）。最长 speedup 2.29×（Qwen2-57B-A14B, γ=4, humaneval, temperature=0, 2xGPU-B）。

涉及论文标题：
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE

## Inter-layer Expert Affinity (跨层专家亲和性 / Token Routing Dependency)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Inter-layer Expert Affinity（跨层专家亲和性）是 MoETuner 发现并利用的 MoE 模型关键特性：token 在相邻 MoE 层之间的路由决策不是独立的，而是遵循可预测的依赖模式。当 token 在层 l 被路由到 expert e_1 时，该 token 在层 l+1 更倾向于被路由到特定的少数几个 expert（而非均匀分布到所有 E 个 expert）。度量指标 R_{e_1,e_2,l}：层 l→l+1 间从 expert e_1 路由到 expert e_2 的 token 数量。Mixtral-8x7B 分析表明此路由模式跨 batch 高度一致，可使用数据集采样子集准确近似整体行为。

从算法pipeline角度拆解术语：
MoE 模型逐层推理时，对每个 token 追踪跨层路由路径：
```
for l in range(L-1):
    gate_l = Softmax(h_l @ W_gate[l])       # [E] router logits
    top2_l = TopK(gate_l, 2)                # 层 l 选中的 2 个 expert
    gate_l1 = Softmax(h_{l+1} @ W_gate[l+1])
    top2_l1 = TopK(gate_l1, 2)              # 层 l+1 选中的 2 个 expert
    # 统计跨层 expert 对
    for e_src in top2_l:
        for e_dst in top2_l1:
            R[e_src][e_dst][l] += 1         # 累加路由计数
```
MoETuner 利用此属性：在 ILP 2 中，若 R_{e_1,e_2,l} 很大（expert e_1 和 e_2 频繁被同一 token 连续层激活），则将其放置在同一 GPU → 消除该 token 在层 l→l+1 间的跨 GPU 通信。这是一种数据驱动的编译优化：将运行时 token 路由模式转化为离线 expert placement 决策。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 发现方法：在任务数据集采样子集上运行推理，收集逐 token 跨层路由路径，构建频率表。MoETuner 证明采样子集足以捕获整体路由模式。
- 应用：(1) Expert Placement——高亲和性 expert 对放同一 GPU。(2) Expert Prefetching——预测下层可能激活的 expert 提前加载。(3) Network Scheduling——预分配高通信量 GPU pair 带宽。
- 相关研究：ExFlow (IPDPS 2024) 同样利用 inter-layer expert affinity 做 locality-aware expert placement（graph-based, 非 ILP）。
- 局限：依赖特定任务数据集，切换任务需重新 profiling。

涉及论文标题：
- MoETuner: Optimized Mixture of Expert Serving with Balanced Expert Placement and Token Routing

## Gumbel-Top-K Routing (Gumbel-Top-K 随机路由)

术语解释
一种基于 Gumbel-Max Trick 的无放回随机采样方法，在 MoE 推理时通过向 router logits 添加 Gumbel 噪声来实现受控的 expert 选择随机化，等价于从 router 定义的 categorical 分布中无放回采样 k 个 expert。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gumbel-Top-K Routing 是将 Gumbel-Max Trick 的 TopK 扩展版本应用于 MoE expert routing 的机制。其数学基础为：

1. **Gumbel-Max Trick**（Gumbel, 1954）：给定 logits $\phi_1, ..., \phi_E$，若 $G_i \sim \text{Gumbel}(0,1)$ 独立同分布，则 $\arg\max_i(\phi_i + G_i)$ 等价于从 $\text{Categorical}(\text{softmax}(\phi))$ 中采样一次。
2. **Gumbel-Top-K 扩展**（Kool et al., ICML 2019）：$\text{TopK}(\phi + G, k)$ 等价于从该 categorical 分布中**无放回**顺序采样 k 个元素。
3. **RoE 中的应用**：在 MoE router logits $\mathbf{R} \in \mathbb{R}^E$ 上添加缩放 Gumbel 噪声后做 TopK 选择：$\text{Indices} = \text{TopK}(\mathbf{R} + \tau \cdot \mathbf{G}, k)$，其中 $\tau \geq 0$ 为温度参数控制随机性程度。

温度参数 $\tau$ 的作用：
- $\tau = 0$：退化为标准确定性 TopK routing
- $\tau$ 中等：高 logit expert 仍更可能被选中（Gumbel-Max 性质保证），但低 logit expert 也有机会被激活
- $\tau \to \infty$：退化为纯均匀随机选择，预测质量下降

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 RoE 单 token 生成中，Gumbel-Top-K Routing 的 pipeline 位置如下：

```
# 单 MoE 层的 Gumbel-Top-K Routing Forward
输入: hidden_state h ∈ R^d, router weight W_r ∈ R^{E×d}, 温度 τ, top-k 数 k
输出: expert 输出 y ∈ R^d

# Step 1: Router logits
R = W_r @ h          # (E,)  router logits per expert

# Step 2: Add scaled Gumbel noise
U = rand(E)          # Uniform(0,1) i.i.d.
G = -log(-log(U))    # Gumbel(0,1) via inverse CDF: F^{-1}(u) = -log(-log(u))
noisy_R = R + τ * G  # (E,)  perturbed logits

# Step 3: Top-K expert selection (无放回采样)
topk_values, topk_indices = topk(softmax(noisy_R), k)

# Step 4: Weighted expert aggregation (standard MoE)
y = Σ_i topk_values[i] * Expert_FFN_i(h)

return y
```

在整个 RoE pipeline 中，上述过程在每层 MoE 对 batch 中的每个 sample 独立执行一次，产生 n 条不同的内部计算路径。batch 内第一个 sample（index 0）在启用 Clean Cache 时使用 τ=0 确定性路由作为"clean path"。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现要点：
- **Gumbel 噪声生成**：标准实现为 $G = -\log(-\log(U))$，其中 $U \sim \text{Uniform}(0, 1)$。需加 $\epsilon$（如 1e-20）防止 $\log(0)$。
- **PyTorch 等效**：`torch.distributions.Gumbel(0, 1).sample()` 或手动 `-torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)`。
- **温度调优**：$\tau$ 为逐层超参数，通过 Optuna TPE 在验证集上搜索最佳值。搜索空间约束为 $[0, 0.5]$（论文经验观察：$\tau > 0.5$ 引入过多噪声导致性能下降）。
- **首尾层保护**：前 k 层和最后 k 层固定 $\tau = 0$（确定性路由），仅中间层参与 Gumbel-Top-K 随机化。论文实验表明初始层处理 raw embedding、最终层整合输出信息，对路由扰动更敏感。
- **跨任务差异**：不同任务（数学/常识/代码）的最优 $\tau$ 分布差异显著，需分别调优。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

## Hyper-Parallel Scaling (超并行扩展)

术语解释
一种新的推理时扩展范式，通过在每 token 层面增加模型内部计算量和计算路径多样性，直接提升模型的内在 next-token 预测质量，与传统的 sequential scaling（如 CoT，生成更长的推理步骤）和 parallel scaling（如 Self-Consistency，生成多条完整序列后投票）正交。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hyper-Parallel Scaling 是论文 Introducing 的第三类推理时扩展范式，其核心定义与其他两类范式的区别如下：

| 范式 | 操作对象 | 聚合粒度 | 代表性方法 |
|------|---------|---------|-----------|
| Sequential Scaling | 输出生成过程（生成更长步骤） | 序列级（更长的输出链） | Chain-of-Thought, Tree-of-Thoughts |
| Parallel Scaling | 输出生成过程（多次采样） | 序列级（多条完整序列投票） | Self-Consistency, Beam Search |
| **Hyper-Parallel Scaling** | **模型内部计算路径** | **Token 级（每次 next-token 预测内）** | **RoE (本文)** |

关键区别：
- Sequential/Parallel Scaling 把模型当作黑盒，在外层操作输出序列
- Hyper-Parallel Scaling 进入模型内部，在每 token 预测层面多样化计算路径并聚合
- Hyper-Parallel 是 Sequential/Parallel 的**正交补充**，可与两者同时使用

在 MoE 模型中，Hyper-Parallel Scaling 利用"每 token 仅激活 k 个 expert"的稀疏性——E−k 个 inactive expert 闲置——通过随机激活不同的 expert 子集来释放模型的全部潜力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hyper-Parallel Scaling 的执行流程（以 RoE 实现为例）：

```
# 单 token 生成中的 Hyper-Parallel Scaling
输入: prefix tokens, MoE model M
输出: 下一个预测 token

# 传统 greedy decoding（单路径）
h = M.forward(prefix)           # 单次确定性 forward
logits = M.lm_head(h)           # 单组 logits
next_token = argmax(logits)     # 确定性预测

# Hyper-Parallel Scaling with RoE（多路径）
candidate_logits = []
for i in range(n):              # n = sample count (e.g., 32)
    h_i = M.forward(prefix, routing_mode="gumbel_top_k", tau=τ)
    logits_i = M.lm_head(h_i)
    candidate_logits.append(logits_i)

# Token 级聚合（非序列级！）
final_probs = mean(softmax(candidate_logits), dim=0)  # probability averaging
next_token = argmax(final_probs)
```

关键点：聚合发生在 **logits/probability 层面**，而非序列层面。这使 Hyper-Parallel Scaling 适用于开放生成任务（如代码生成），而 Self-Consistency 的 majority voting 只在可验证答案的任务上有效。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现 Hyper-Parallel Scaling 需要三个组件：
1. **内部路径多样化机制**：论文在 MoE 中使用 Gumbel-Top-K routing；对于 dense 模型，论文提出可用 dropout-based variation（Shelmanov et al., 2021）或 recurrent re-computation（Lin et al., 2022）。
2. **高效执行**：通过 batched inference 将多次 forward 合并，利用 GPU sub-linear batch scaling。
3. **缓存优化**：Clean Cache 策略避免维护 N 份 KV-cache。

论文指出 Hyper-Parallel Scaling 是 domain-agnostic 的，可扩展到 vision、audio、video 等模态。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

## Roster of Experts (RoE) (专家花名册)

术语解释
一种无需训练的 MoE 推理算法，将单个 MoE 模型视作一个动态专家集成（dynamic ensemble of MoEs），通过 Gumbel-Top-K 随机路由在每 token 生成多条内部计算路径，聚合多路径输出以提升预测质量。RoE 是 Hyper-Parallel Scaling 概念在 MoE 模型上的具体实例化。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RoE 由三个核心组件构成：

1. **Gumbel-Top-K 随机路由**：向 router logits 添加温度控制 Gumbel 噪声 → 从 router 隐含分布中无放回采样 k 个 expert → 产生 expert 组合多样性。
2. **多路径聚合**：对同一 token 执行 n 次独立 Gumbel-Top-K forward → 得到 n 组 logits → softmax 后概率平均（probability averaging）→ argmax 得最终 token。
3. **Clean Cache**：batch 中 sample 0 使用 deterministic routing (τ=0) → 其 KV-cache 作为共享"clean"缓存 → 其余 sample 复用此 cache → KV-cache 内存 = 单样本内存。

RoE 的关键性质：
- **Training-free**：不修改模型参数，直接应用于任何预训练 MoE 模型
- **Post-hoc**：可在部署后动态启用/禁用，按需 trade compute for quality
- **Orthogonal**：与 sequential/parallel scaling 可同时使用（论文仅用 greedy decoding 评估以隔离 RoE 收益）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
RoE 完整算法（基于论文 Algorithm 和描述）：

```
# RoE 单步 Token 生成
输入: 当前 token embedding e, 逐层温度数组 τ[l], 样本数 n, 共享 KV-cache K
输出: 下一 token

def roe_generate_step(e, τ, n, K, model):
    # 准备 batched input
    batch_e = e.expand(n, -1)            # (n, d_model)

    for l, layer in enumerate(model.layers):
        # Attention: sample 0 计算，其余共享 KV-cache
        if is_clean_cache_enabled:
            attn_out_0, K_new = layer.attn(batch_e[0:1], K)
            K = K_new                       # 更新共享 KV-cache
            attn_out_i = layer.attn_shared_kv(batch_e[1:], K)  # 复用 K
            attn_out = cat([attn_out_0, attn_out_i], dim=0)
        else:
            attn_out = layer.attn(batch_e)

        # MoE Layer: per-sample Gumbel-Top-K
        if layer.is_moe:
            R = layer.router(attn_out)     # (n, E) router logits
            for i in range(n):
                τ_eff = 0.0 if (i == 0 and is_clean_cache) else τ[l]
                G = sample_gumbel(E)
                noisy_R = R[i] + τ_eff * G
                topk_idx, topk_w = topk(softmax(noisy_R), k)
                ff_out = sum(w * layer.experts[idx](attn_out[i])
                           for idx, w in zip(topk_idx, topk_w))
                batch_e[i] = attn_out[i] + ff_out  # residual
        else:
            batch_e = layer.ffn(attn_out) + attn_out

    # Logit 聚合: probability averaging
    logits = model.lm_head(batch_e)        # (n, vocab_size)
    probs = softmax(logits, dim=-1)        # (n, vocab_size)
    avg_probs = probs.mean(dim=0)          # (vocab_size,)
    next_token = argmax(avg_probs)

    return next_token, K
```

超参数配置（来自论文 Table 1）：
- OLMoE-7B 数学任务：n=32, τ_max=0.5, skip=1 首尾层
- Mixtral-8x7B 数学任务：n=64, τ_max=0.25, skip=5 首尾层
- 温度通过 Optuna TPE 在验证集上逐层搜索

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
使用 RoE 的典型流程：
1. **温度调优**（离线）：在目标任务验证集上用 Optuna TPE 搜索每层最优 τ_l（~50 trials）
2. **推理部署**（在线）：加载调优后的 τ 配置 → 对每个生成 token 执行 batched Gumbel-Top-K forward → Clean Cache 控制内存
3. **计算-质量 trade-off**：通过调整 n（样本数）控制——n 越大质量越高但计算开销越大（论文显示 n=32 已有显著收益）

论文关键结果：OLMoE-7B + RoE (n=32) ≈ 10.5B 标准 MoE 的性能，内存减少 25%，延迟降低 30%。

涉及论文标题：
- MoEs Are Stronger than You Think: Hyper-Parallel Inference Scaling with RoE

## MoE Modality Extension / Expert Addition for New Modalities (MoE 模态扩展 / 为新增模态添加专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MoE Modality Extension 是一种将预训练的 MoE LLM（原仅处理文本模态）扩展到新模态（如视觉、语音）的方法。核心思想是：**不修改原有 MoE 模型的任何参数（包括 expert FFN、router、attention 层），仅在部分 MoE 层中新增 expert 和对应的 router 参数，仅训练这些新增部分来学习新模态知识**。这与 LLaVA 式的全参数微调形成根本区别——全参数微调修改所有 MLP/FFN 权重来桥接模态间隙，而 MoE Modality Extension 通过为不同模态提供专用 expert，让原有专家继续服务文本模态，新增专家专门处理新模态 token。

在 MoExtend 中的具体实现：为 Mixtral 8x7B 的 50% MoE 层（由 Extender 选出的层）各添加 1 个新 expert FFN_{m+1}，router 拓展为 W_new = [W; v_new] ∈ R^{D×(m+1)}，每 token 仍选 top-2 expert。

从算法pipeline角度拆解术语：
扩展前后的 MoE 层前向计算对比：

**扩展前（原始 MoE，m experts）：**
```
# Router 计算
logits = H @ W_g     # [B, m]
probs = softmax(logits)  
# Top-K selection (K=2)
weights, indices = top_k(probs, K)  
# Expert 加权输出
MoE(x) = Σ_{j=1}^{K} s(x)_j · FFN_{idx_j}(x)
```

**扩展后（MoE + 新模态 expert，m+1 experts）：**
```
# Router 扩展为 m+1 列
W_new = concat(W_g, v_new)  # W_g [D, m] → W_new [D, m+1]
logits = H @ W_new           # [B, m+1]
probs = softmax(logits)
weights, indices = top_k(probs, K)  # 仍选 K=2
# Calibration 校正后加权输出
MoE(x) = Σ_{j=1}^{K} s(x)_j · [1 + s_c(x)] · FFN_{idx_j}(x)
```

关键洞察：新 expert 权重初始化为复制该层原有最活跃 expert（对视觉数据响应最大的），router 列 v_new 同理。这使得新 expert 从"最接近新模态理解"的参数空间出发训练，避免"冷启动"导致的选中概率过低问题。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **训练成本**：MoExtend 仅训练新增参数（~3B），全参数冻结（~46.7B），训练时间约 45 小时（Alignment ~15h + Fine-tuning ~30h），对比全参数微调约 200 小时，加速约 6 倍（8×A800-80G）。
- **推理成本**：无额外开销——推理时 MoE layer 仍执行 top-K routing，仅 router 从 m 列扩为 m+1 列（对 Mixtral 从 8 选 2 变为 9 选 2），新增参数仅在扩展层存在（50% 层）。
- **防遗忘机制**：原有 expert FFN 参数完全冻结，文本 token 仍优先被原有 expert 选中处理，因此文本性能几乎不降（Avg. drop 仅 0.41 vs 全参数微调 3.30）。
- **扩展性**：方法不限于视觉模态——替换 vision encoder 为语音/其他模态 encoder 即可扩展，也不限于 MoE LLM 文本→视觉扩展场景。
- **已知局限**：1）视觉任务外未验证（论文因 GPU 资源限制仅验证视觉模态）；2）需要足够的专家容量——如果原有 expert 已经过度专门化，简单复制可能不够。

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension

## Calibration Module in MoE Extension (MoE 扩展中的校准模块)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Calibration Module 是 MoExtend 在新增 expert 后为保持原有 MoE 输出分布一致性而引入的轻量校正网络。当 MoE 层从 m 个 expert 扩展为 m+1 个 expert 后，softmax 概率分布会因分母增大而整体缩小：s(x)_j' = e^{f(x)_j} / (Σ_{h=1}^m e^{f(x)_h} + e^{f(x)_{m+1}}) ≤ s(x)_j。这意味着原有 expert 的输出权重被"稀释"，即使原有 expert 参数未变，前向传播的特征分布也会漂移，导致一定程度的已有知识遗忘。Calibration Module 通过为每个 expert 添加一个可学习的校正因子 s_c(x) 来修正此效应。

MoExtend 中采用的 Type2(a) 结构：两个线性层 + GELU 激活函数组成的轻量网络，输出作为每个 expert 的校正因子（加法模式）。初始化策略：第一层正态初始化，第二层零初始化——确保训练初期 s_c(x)=0，模型输出与未加 calibration 时一致。

从算法pipeline角度拆解术语：

**带 Calibration 的 MoE 前向计算：**
```
# 输入 x，原有 m experts + 1 new expert
logits = x @ W_new          # [B, m+1]
probs = softmax(logits)       # [B, m+1]

# 获取 top-K expert 的权重和索引
weights, indices = top_k(probs, K)

# 对每个选中的 expert 计算输出，并施加 calibration
output = 0
for j in range(K):
    expert_idx = indices[j]
    w = weights[j]
    expert_out = FFN[expert_idx](x)
    
    # Calibration: 每个 expert 有独立的 calibration 模块
    calib = sc(x)  # 两层 GELU 网络，输出标量
    
    # 加法校正模式（Type2 a）
    output += w * (1 + calib) * expert_out

# output 即为 MoE 层的最终输出
```

注释：
- s_c(x) = W_1(GELU(W_2(x)))，其中 W_1 零初始化，W_2 正态初始化
- 初始状态 s_c(x)=0，MoE(x) = Σ s(x)_j · FFN(x)_j（与原始一致）
- 训练后 s_c(x) 学习出对概率缩放的补偿

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **结构设计空间**（MoExtend 实验结论）：
  - Type1（简单可学习参数 1×m）：加法模式用 Zero 初始化、乘法模式用 One 初始化
  - Type2（两层 GELU 网络）：加法模式用 Zero+Normal 初始化（最优）、乘法模式导致梯度爆炸
  - **Type2(a) 加法模式为最优**：POPE 84.3, MME 1571.0, SQA 73.4, VQA^T 55.7
- **关键设计原则**：初始化必须使 s_c(·) 初始输出为零（加法模式）或一（乘法模式），确保训练初期不干扰模型前向输出，避免异常 loss
- **可扩展性**：Calibration 的概念不仅限于 MoE 扩展——任何涉及模型结构修改后需要"输出分布对齐"的场景都可用类似设计（如 model merging、架构搜索后的 fine-tuning）

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension

## Distribution Shift-based Expert Layer Selection / Extender (基于分布偏移的专家层选择 / 扩展器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Extender 是 MoExtend 中用于自适应决定"在哪些 MoE 层添加新专家"的机制。直接在所有 MoE 层添加新专家会增加参数和过拟合风险，但手动设计插入策略（如前半层、间隔层等）需要大量实验调优。Extender 通过度量各层在新模态数据下 expert 选择分布的偏移程度，自动选出最需要扩展的层。

从算法pipeline角度拆解术语：

**Extender 的完整流程：**
```
# 输入：
#   κ: Alignment stage 后得到的模型（router 未微调）
#   S_t: 子训练集（LLaVA 1.5-mix-665k，除去验证集后的数据）
#   S_e: 验证集（10,000 条随机抽样）

# Step 1: 微调 router（仅使 router 可训练，其余冻结）
κ' = copy(κ)
for step in range(1000):
    κ' = train_step(κ', S_t, trainable={all_routers})

# Step 2: 统计 expert 被选次数
R_κ  = count_expert_selections(κ, S_e)   # [m, L] 矩阵
R_κ' = count_expert_selections(κ', S_e)  # [m, L] 矩阵

# Step 3: 归一化为概率分布
R̄_κ  = normalize_by_column(R_κ)   # 每列（每层）归一化
R̄_κ' = normalize_by_column(R_κ')

# Step 4: 逐层计算分布差异
for j in range(L):
    diffs = [R̄_κ'[i,j] - R̄_κ[i,j] for i in range(m)]
    d_j = std(diffs)  # 标准偏差度量分布偏移程度

# Step 5: 选 top-⌊pL⌋ 层添加 expert（p=0.5）
# 对 Mixtral 8x7B：L=32, ⌊0.5×32⌋=16 层
selected_layers = top_k_by_d(layers, k=16)
```

注释：
- d_j 小 → MoE 层 j 对新模态数据响应变化小 → 无需新 expert
- d_j 大 → MoE 层 j 在新模态下路由分布发生显著变化 → 该层需新 expert 专门处理新模态
- p=0.5 是基于消融实验的默认值（16 层与手动最佳策略 First-half/Interval 性能相当，但训练时收敛更快）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **与手动搜索的对比**：消融实验显示，全层加（All layer, 32 layers）与最优手动策略（First-half/Interval, 16 layers）性能几乎相同（POPE 84.0 vs 84.5 vs 83.5），但 Extender 自动选出的 16 层在训练时收敛更快
- **选出的层分布**（Mixtral 8x7B）：集中在模型中部（layer 3-28），深层和极浅层变动小。具体为层 3,4,6,7,9,10,11,13,14,15,17,18,20,21,26,28
- **计算开销**：Extender 仅需 1,000 步 router 微调 + 一次验证集前向统计，相对于后续 Fine-tuning Stage (30h) 可忽略
- **推广性**：Extender 的设计理念（通过分布偏移度量"哪个组件对新数据最敏感"）可推广到任何"在预训练模型中选择性添加新组件"的场景，如增量学习、持续学习的模块扩展

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension

## Catastrophic Forgetting in Multimodal MoE Fine-tuning (多模态 MoE 微调中的灾难性遗忘)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Catastrophic Forgetting（灾难性遗忘）是指神经网络在学会新任务后，显著丧失在旧任务上已获得能力。在多模态 LLM 的上下文中，指 LLM 在全参数微调学习视觉理解等新模态时，遗忘了原有的文本理解、推理、代码、数学等能力。MoExtend 揭示了 **MoE 架构对全参数微调导致的遗忘比 dense LLM 更敏感**——这是因为 MoE 的 expert 专业化特性使每个 expert 存储更集中的知识块，全参数更新时被整体覆盖。

从算法pipeline角度拆解术语：

**全参数微调导致 MoE 遗忘的机制：**
```
# 原始 Mixtral 8x7B：8 experts/层 × 32 layers
# 全参数微调时所有 expert FFN 参数更新
for epoch in training_epochs:
    for batch in multimodal_data:  # 视觉+文本指令数据混合
        for x in concatenated_tokens:
            indices, weights = router(x)  
            loss = cross_entropy(pred, target)
            # 所有被选中的 expert FFN 参数都参与更新
            W_gate -= lr * ∂loss/∂W_gate   # ← 文本知识所在权重被覆盖
            W_up   -= lr * ∂loss/∂W_up     # ← 同前
            W_down -= lr * ∂loss/∂W_down   # ← 同前
```

**遗忘程度对比（基于 Mixtral 8x7B，来自 MoExtend Table 2）：**
```
方法              | Avg. drop ↓（7 个文本 benchmark 平均）
LLaVA-1.5-7B     | -0.81（dense LLM，遗忘轻微）
LLaVA-1.5-13B    | -0.27（dense LLM，遗忘轻微）
MoExtend-Full     | -3.30（MoE LLM 全参数微调，遗忘显著）
MoE-LLaVA         | -7.86（MoE LLM 全参数微调，遗忘严重）
MoExtend          | -0.41（仅训练新增 expert，几乎无遗忘）
```

关键发现：MoE 的 expert 专业化使全参数微调时知识覆盖更严重——每个 expert 存储的特定领域知识被视觉相关梯度整体"冲刷"，不像 dense FFN 通过部分更新即可适应新模态。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **缓解策略分类**：
  1. **参数冻结 + 新增模块**（如 MoExtend）：原有参数不变，仅训练新增 expert/模块；对 MoE 尤其有效
  2. **正则化方法**（如 EWC, SI）：约束重要参数的更新幅度
  3. **数据回放**（Replay）：在微调数据中混入原有文本数据
  4. **参数高效微调**（如 LoRA, Adapter）：仅训练少量新增参数
- **MoE 特定的脆弱性根源**：MoE 的 expert 专业化 + sparse activation 导致：(a) 每个 expert 知识集中化，(b) 只有部分 expert 被 token 激活，(c) 被激活的 expert 通过梯度传播完整覆盖。相比之下 dense FFN 的分布式表示使知识覆盖更渐进
- **评估指标**（MoExtend 采用）：7 个纯文本 benchmark——ARC-Easy（常识推理）、HellaSwag（常识推理）、PIQA（物理常识）、Winogrande（常识推理）、MBPP（代码）、MMLU（聚合知识）、GSM8K（数学），使用 OpenCompass 评估框架，计算 Avg. drop

涉及论文标题：
- MoExtend: Tuning New Experts for Modality and Task Extension


## Mixture-of-Head Attention (MoH / 混合头注意力)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Mixture-of-Head Attention (MoH) 是一种将 Mixture-of-Experts (MoE) 机制引入 multi-head attention 的新型注意力架构。MoH 将每个 attention head 视为 MoE 框架中的 expert，通过一个可学习的 router 为每个输入 token 动态选择 Top-K 个 attention head 进行激活，并将标准 MHA 中的等权求和替换为加权求和。MoH 不增加注意力头数量，参数总量与标准 MHA 可比（router 带来的额外参数量极小，约为 O(h·d_in)）。MoH 在 ViT（图像分类）、DiT（图像生成）和 LLM 上均验证有效，可在仅激活 50%~90% 注意力头的情况下达到或超越标准 MHA 的性能。代码开源：https://github.com/SkyworkAI/MoH（Apache 2.0），发表于 ICML 2025。

从算法pipeline角度拆解术语：
MoH layer 的计算流程（h 个 head，h_s 个共享 head，K 个路由 head 激活）：
```
# Input: X ∈ R^{T×d_in}, X' ∈ R^{T'×d_in}
# 参数: W_Q^i, W_K^i, W_V^i, W_O^i (per head i)
# Router: W_s ∈ R^{h_s×d_in}, W_r ∈ R^{(h-h_s)×d_in}, W_h ∈ R^{2×d_in}

# Step 1: Router 计算共享head分数
s_s = Softmax(W_s @ x_t)           # [h_s] per token

# Step 2: Router 计算路由head分数
s_r = Softmax(W_r @ x_t)           # [h-h_s] per token

# Step 3: Top-K 选择路由head
topk_indices = TopK(s_r, K)        # 选择分数最高的K个路由head

# Step 4: 两阶段系数
[α_1, α_2] = Softmax(W_h @ x_t)   # 平衡共享/路由head贡献

# Step 5: 组装routing score g_i
for i in 1..h_s:     g_i = α_1 * s_s[i]                    # 共享head
for i in h_s+1..h:   g_i = (i ∈ topk_indices) ? α_2 * s_r[i-h_s] : 0  # 路由head

# Step 6: 仅激活g_i≠0的head计算attention
for i where g_i ≠ 0:
    Q_i = X @ W_Q^i, K_i = X' @ W_K^i, V_i = X' @ W_V^i
    H^i = Softmax(Q_i @ K_i^T / sqrt(d_k)) @ V_i

# Step 7: 加权求和输出
MoH(X, X') = Σ_{i=1}^{h} g_i · H^i · W_O^i

# Step 8: Load Balance Loss（仅对路由head）
P_i = mean(s_r[i-h_s])              # token选择head i的平均概率
f_i = mean(1[token选择head i])      # head i被选择的实际比例
L_b = Σ_{i=h_s+1}^{h} P_i · f_i     # 鼓励均匀路由

# 总loss: L = L_task + 0.01 * L_b
```

关键参数：
- h: 总注意力头数
- h_s: 共享头数（始终激活）
- K: 每 token 激活的路由头数（Top-K）
- 激活比例 = (h_s + K) / h
- 激活预算在各层可不均匀分布：浅层激活较少 head，深层激活较多 head（论文中 TransNeXt 设置）
- β = 0.01（load balance loss 权重）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 训练方式：(1) 从头训练——MoH 可直接替换标准 MHA 进行训练，router 与模型其他参数同时学习，需添加 Load Balance Loss 防止 routing collapse；(2) Continue-Tuning——预训练 MHA 模型（如 LLaMA3-8B）可转换为 MoH 模型，关键技巧包括：参数无关 router（用 ℓ₂ norm of query 作为 routing score）、straight-through estimator 量化 routing score 保持输出分布稳定、两阶段训练（第一阶段适配数据分布，第二阶段切换为 MoH）。
- 推理加速：将 Q/K/V 特征通过 router mask 转为稀疏矩阵，用稀疏矩阵乘法替代 dense 矩阵乘法。序列越长优势越大（seq=512 时 50% head 激活比 MHA 快 37.3%）。
- 代码开源：https://github.com/SkyworkAI/MoH，基于 Skywork-MoE 训练框架。预训练模型在 HuggingFace：Chat-UniVi/MoH-ViT-*、Chat-UniVi/MoH-DiT-*、Chat-UniVi/MoH-LLaMA3-8B。
- 与 MoA（Mixture of Attention Heads, Zhang et al. 2022）的区别：MoH 不增加参数、引入 shared heads 和 two-stage routing、支持 continue-tuning、在 ViT/DiT/LLM 多框架验证。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention

## Shared Heads in MoH（MoH 中的共享注意力头）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Shared Heads 是 MoH 架构中始终激活的注意力头子集。在 MoH 中，h 个注意力头被分为两类：前 h_s 个为共享头（shared heads），始终对所有 token 激活；剩余 h − h_s 个为路由头（routed heads），由 router 动态选择 Top-K 激活。共享头的设计动机是：某些注意力头可能捕获跨上下文的通用知识（如语言中的语法规则、视觉中的基础纹理特征），使这些 head 始终激活可减少其他路由头之间的冗余。论文消融实验（Tab.5）表明，添加共享头能将 ViT Acc 从 75.6% 提升到 78.3%（75% 激活，100 epoch）。

从算法pipeline角度拆解术语：
```
# 共享头路由分数计算
s_s = Softmax(W_s @ x_t)          # W_s ∈ R^{h_s×d_in}
# 共享头始终激活，routing score 非零
for i in 1..h_s:
    g_i = α_1 * s_s[i]            # α_1 来自两阶段路由的 head-type 系数

# 共享头计算 attention
for i in 1..h_s:
    H^i = Attention(X @ W_Q^i, X' @ W_K^i, X' @ W_V^i)

# 共享头输出参与加权求和
output += Σ_{i=1}^{h_s} g_i · H^i · W_O^i
```
共享头可视为 Soft MoE (Puigcerver et al., 2024) 的一种形式——所有 token 都经过这些 head，但通过 routing score 加权。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 共享头比例：论文消融（Tab.6）表明，共享头占激活头比例在 13.9%~74.0% 范围内性能稳定（ViT-S, 75% 激活, 100 epoch, Acc 在 78.4%~78.6% 之间），推荐使用较高比例（>40%）。
- LLaMA3-8B Continue-Tuning 中：简单选择每层前 16 个注意力头作为共享头。
- 共享头与 DeepSeekMoE 的 shared experts 概念类似但应用领域不同：DeepSeekMoE 的 shared experts 在 FFN 层，MoH 的 shared heads 在 attention 层。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention

## Two-Stage Routing in MoH（MoH 中的两阶段路由）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Stage Routing 是 MoH 中用于动态平衡共享头和路由头权重的路由策略。与标准 MoE 的单一 router 不同，MoH 的路由分两阶段：(1) Stage 1——分别计算每个 head 的 token 级分数：共享头用 W_s 投影，路由头用 W_r 投影；(2) Stage 2——通过 W_h 投影产生 α_1 和 α_2 两个 head-type 级别的系数，动态调整共享头和路由头对最终输出的贡献比例。消融实验（Tab.5）表明，两阶段路由在共享头基础上进一步提升性能（ViT Acc 78.3%→78.6%, DiT FID 69.54→69.42）。

从算法pipeline角度拆解术语：
```
# Input: token x_t ∈ R^{d_in}

# Stage 1: 逐头分数
s_s = Softmax(W_s @ x_t)          # 共享头分数 [h_s]
s_r = Softmax(W_r @ x_t)          # 路由头分数 [h-h_s]

# Stage 2: Head-type 系数
[α_1, α_2] = Softmax(W_h @ x_t)  # W_h ∈ R^{2×d_in}, α_1+α_2=1

# 最终 routing score
for i in 1..h_s:     g_i = α_1 * s_s[i]          # 共享头
for i in h_s+1..h:   g_i = (Top-K选中的i) ? α_2 * s_r[i-h_s] : 0
```
两阶段路由的设计直觉：α_1/α_2 让模型根据 token 内容决定是更多依赖共享头（通用知识）还是路由头（专用知识）。例如，对于简单/常见 token，α_1 可能更大；对于需要特定领域知识的 token，α_2 可能更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- W_h 仅为 2×d_in 的投影矩阵，参数量极小。
- 两阶段路由与标准 MoE routing 的主要区别：标准 MoE 仅有一个 router 产生所有 expert 的分数；MoH 的两阶段路由显式分离了"head 选择"和"head-type 权重平衡"两个决策层次。
- 论文未提供 α_1/α_2 在不同任务/类别下的详细分析（仅在 Appendix D 中可视化了 routing score 分布，指出共享头的 routing score 在不同类别间变化更大）。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention

## Straight-Through Estimator (STE / 直通估计器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Straight-Through Estimator (STE) 是一种用于训练含不可微离散操作（如量化、二值化、Top-K 选择）的神经网络的梯度估计技术。当网络前向传播包含不可微操作（如将实数值量化为 0/1 或整数）时，STE 在反向传播中将该操作的梯度直接"直通"（即梯度恒等映射 ∂ŷ/∂y = 1），使得梯度可以流过离散操作到达上游参数。STE 由 Bengio et al. (2013) 首次系统提出，广泛用于量化神经网络（QAT）、二值网络、VQ-VAE、Gumbel-Softmax 等场景。

从算法pipeline角度拆解术语：
MoH 在 LLaMA3-8B Continue-Tuning 中使用 STE 量化 routing score：
```
# Forward: 将 routing score 量化为 0/1
g_i^q = 1[token x 选择 head i]       # 离散值 0 或 1
# 即 g_i^q = 1 (if head i activated), 0 (otherwise)

# Backward: STE 将梯度直通
∂L/∂g_i = ∂L/∂g_i^q                   # 梯度直接赋值（恒等映射）
# 等价于把 g_i^q 在反向传播中视为 g_i 处理

# 作用: g_i^q ∈ {0,1} 保持输出分布与原始 MHA 一致（等权求和）
# 同时通过 STE 让 g_i (实值 routing score) 仍能接收梯度更新
```
论文采用此设计的动机：加权 routing score 会显著改变 attention 层输出分布，需要大量训练数据恢复性能。量化 routing score 为 0/1 使 MoH 输出接近原始 MHA 的等权求和，配合 STE 保持 router 可训练。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 实现：`(quantized - original).detach() + original`，forward 时 quantized 生效，backward 时梯度流向 original。
- 典型应用场景：(1) QAT (Quantization-Aware Training) — 将浮点权重/激活量化为 INT8/INT4，STE 保持可微；(2) VQ-VAE — 将连续 latent 量化为离散 codebook entry，STE 训练 encoder；(3) Binary Neural Networks — 权重二值化为 ±1；(4) Gumbel-Softmax 的硬采样模式；(5) MoE routing 的离散化（如 MoH 的 continue-tuning）。
- 局限性：STE 引入 biased gradient estimate（梯度与实际前向操作不匹配），可能导致训练不稳定或收敛到次优点。Wang et al. (2024, Q-Sparse) 指出 STE 可显著缓解梯度消失问题。

涉及论文标题：
- MoH: Multi-Head Attention as Mixture-of-Head Attention

---

## Layer-wise Expert Allocation (层级专家分配)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Layer-wise Expert Allocation（层级专家分配）是 MoLA 论文提出的核心创新：在结合 LoRA 和 MoE 进行参数高效微调时，不为 Transformer 每一层分配相同数量的 LoRA expert，而是根据各层的表示特性和冗余程度，灵活分配不同数量的 expert。其理论基础是：Transformer 底层处理 token-level 特征（词义、语法），expert 间高度相似（冗余大），不需要太多 expert；中层/高层处理抽象推理和任务特定模式，需要更多 expert 学习细粒度特征。对于有 m 层的 Transformer，每层 j 分配 N_j 个 expert，总 expert 数 ΣN_j 固定（与 baseline 等量分配相同参数量），仅分配方式不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子：
MoLA 提出五种基于不同假设的层级分配配置（以 LLaMA-2-7B 32 层为例，总 config sum=20）：

| 配置名称 | 分配 (层1-8, 9-16, 17-24, 25-32) | 假设 |
|------|------|------|
| MoLA-△ (Triangle) | 8, 6, 4, 2 | 底层需更多 expert 处理细粒度 token 特征 |
| MoLA-▽ (Inverted Triangle) | 2, 4, 6, 8 | 高层需更多 expert 处理抽象推理和任务特定模式 |
| MoLA-▷◁ (Hourglass) | 8, 2, 2, 8 | 底层和高层需更多 expert，中层做特征聚合/映射 |
| MoLA-✸ (Diamond) | 2, 8, 8, 2 | 中层表示学习最关键（effective representation learning） |
| MoLA-□ (Rectangle) | 5, 5, 5, 5 | 各层 expert 数相同，传统 MoE baseline |

伪代码（每层使用不同 N_j）：
```
# 配置: expert_config = [2]*8 + [4]*8 + [6]*8 + [8]*8  # MoLA-▽ 2468
for layer_j in 1..m:
    N_j = expert_config[layer_j]
    for module in [Wq, Wk, Wv, Wo, Wgate, Wdown, Wup]:
        W_r = Linear(d_model, N_j)               # 该层 router
        probs = Softmax(W_r @ x)                  # [B, L, N_j]
        topk_vals, topk_idx = TopK(probs, K=2)
        topk_vals /= sum(topk_vals)
        h = frozen_W0 @ x
        for idx, w in zip(topk_idx, topk_vals):
            h += w * (B[idx] @ A[idx] @ x)       # LoRA delta
        f_e = mean(indicator(token to expert e))
        P_e = mean(probs[:, e])
        L_aux += N_j * sum(f_e * P_e)            # per-layer LB loss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 开源：https://github.com/GCYZSL/MoLA，基于 HuggingFace PEFT + Transformers。通过 expert_config 列表指定每层 expert 数量。
- 关键超参数：LoRA rank=8, top-K=2, LoRA alpha=16, LoRA dropout=0.05。总可训练参数 ~105.6M（LLaMA-2-7B 的 ~1.5%）。
- 最优配置因 base model 而异：LLaMA-2 和 Gemma 倾向 MoLA-▽（高层多 expert），Mistral 倾向 MoLA-✸（中层多 expert）。与各模型预训练层级质量（HT-SR PL Alpha Hill metric）高度相关（Pearson r=0.91 LLaMA-2, r=0.74 Mistral）。
- 核心发现：在固定总 expert 预算下，减少底层 expert（冗余高）、增加中高层 expert（冗余低），可提升性能且不增加参数。MoLA-▽ (2468) 以 62.5% 的参数量超越等量 MoLA-□ (8888)。

涉及论文标题：
- MoLA: MoE LoRA with Layer-wise Expert Allocation

---

## Expert Redundancy in MoE (MoE 专家冗余)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Redundancy 指 MoE 架构中多个 expert 学习到相似或重叠的表示，导致 expert 之间无法有效分工，浪费参数和计算预算。Chen et al. (2023) 发现 Sparse MoE 可作为 Dropout 的替代，暗示许多 expert 确实冗余；Zoph et al. (2022, ST-MoE) 报告 routing policy overfitting 导致少数 expert 被过度使用。MoLA 论文给出 quantitative definition: Expert Redundancy measures the layer-wise difference between expert modules, 通过 Frobenius Norm of pairwise expert weight differences 量化：值越小 → expert 越相似 → 冗余越高。

从算法pipeline角度拆解术语（MoLA 的 Expert Redundancy 分析流程）：
```
# 对每层 j 的每个 attention module
for layer_j in 1..m:
    for module in [Wq, Wk, Wv, Wo]:
        # 1. 合并 LoRA: W_full = B_e @ A_e
        # 2. 计算 pairwise Frobenius Norm
        norms = []
        for (p, q) in combinations(range(N_j), 2):
            diff = W_p - W_q                        # [d_p, d_q]
            norms.append(sqrt(sum(diff ** 2)))      # Frobenius Norm
        redundancy[layer_j] = mean(norms)           # 该层平均冗余
```

数值示例（LLaMA-2-7B MoLA-□ 8888, instruction tuning 后）：
- Layers 1-8（底层）: Frobenius Norm ~0.1-0.2 → 高冗余
- Layers 9-24（中层）: ~0.3-0.4 → 中等冗余
- Layers 25-32（高层）: ~0.5-0.6 → 低冗余（expert 差异化大）

所有 MoLA 配置（▽, △, ▷◁, □ 各种变体）均呈现底层→高层 Frobenius Norm 单调递增，证实底层冗余是普遍现象而非特定配置导致。

极端配置实验验证：
- MoLA (10-2-2-2): Expert 集中在底层 → AVG 83.0%（LLaMA-2）
- MoLA (2-2-2-10): Expert 集中在高层 → AVG 84.2%（LLaMA-2）
底层 10 expert 不如高层 10 expert 有效，证明底层冗余 > 高层。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 定量工具：Frobenius Norm（MoLA）、Cosine Similarity、CKA (Centered Kernel Alignment)、SVCCA 均可度量 expert 相似度。
- 实际应用：(1) Expert Pruning — 剪除冗余 expert 减参数；(2) Layer-wise Allocation — 在冗余高的层减 expert（MoLA）；(3) Expert Merging — 合并相似 expert。
- Router 层面分析补充：大部分 expert 被选中的平均融合权重 ~0.5（重要性相近），大部分 expert 被选择频率较高且均匀 → 冗余主要来自 expert 表示而非 routing collapse。

涉及论文标题：
- MoLA: MoE LoRA with Layer-wise Expert Allocation
- Sparse MoE as the New Dropout (Chen et al. 2023, ICLR)
- ST-MoE: Designing Stable and Transferable Sparse Expert Models (Zoph et al. 2022)

---

## Frobenius Norm for Expert Similarity (Frobenius 范数量化专家相似度)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Frobenius Norm（Frobenius 范数）在 MoLA 中被创新性地用作量化 MoE 架构中不同 expert 之间差异（等价于相似度/冗余度）的度量工具。定义为：$\|A - B\|_F = \sqrt{\sum_{i,j} |a_{ij} - b_{ij}|^2}$，即逐元素差的平方和的平方根。MoLA 计算每层所有 expert pair（任意两个不同 expert 的合并 LoRA 权重矩阵 B_e@A_e）的 Frobenius Norm 均值，值越小 → expert 越相似 → 冗余越高。

从算法pipeline角度拆解术语：
```
# 输入: 每层 j 所有 expert 的 LoRA 矩阵 {A_e, B_e}_{e=1..N_j}
# 输出: 每层的 mean Frobenius Norm

for layer_j in 1..m:
    W_list = []  # expert 等效权重
    for e in 1..N_j:
        W_list.append(B[e] @ A[e])  # [d_p, r] @ [r, d_q] → [d_p, d_q]
    
    norms = []
    for p in 1..N_j:
        for q in p+1..N_j:
            diff = W_list[p] - W_list[q]            # [d_p, d_q]
            fn = sqrt(sum(diff ** 2))               # Frobenius Norm
            norms.append(fn)
    
    layer_mean_fn[layer_j] = mean(norms)
    # 值越大 → expert 越多样化 → 该层越受益于更多 expert
```

MoLA 关键数值发现（LLaMA-2-7B, 32 层, MoLA-□ 8888, instruction-tuned）：
- Layers 1-8: mean FN ~0.1-0.2 → 高冗余 → 可减少 expert
- Layers 9-24: mean FN ~0.3-0.4 → 中等
- Layers 25-32: mean FN ~0.5-0.6 → 低冗余（专家差异化大）→ 应分配更多 expert

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch: `torch.norm(W_p - W_q, p='fro')` 或手动 `torch.sqrt(torch.sum((W_p - W_q) ** 2))`
- 适用场景：(1) MoE expert redundancy 定量分析；(2) 指导 expert pruning/allocation 决策；(3) 与其他相似度量（Cosine Similarity, CKA）互补。
- 注意事项：Frobenius Norm 受矩阵 scale 影响——MoLA 中所有 expert 从相同初始化开始（A: randn, B: zeros），scale 差异天然反映功能分化程度，因此适用。若 expert 经历不同训练动态导致 scale 差异，建议同时使用 Cosine Similarity。

涉及论文标题：
- MoLA: MoE LoRA with Layer-wise Expert Allocation

## Muon Optimizer (MomentUm Orthogonalized by Newton-Schulz)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Muon (MomentUm Orthogonalized by Newton-Schulz) 是由 Keller Jordan et al. (2024) 提出的一种专用于神经网络中 2D 矩阵参数（如 attention 投影矩阵、FFN 权重矩阵）的优化器。其核心思想：将标准 SGD momentum 累积的梯度动量矩阵通过 Newton-Schulz 迭代进行近似正交化（≈ (M M^T)^(-1/2) M = U V^T，即用 momentum 矩阵的左右奇异向量之积替代逐元素更新），使得每次更新的奇异值全部近似为 1，消除更新在少数主导方向上的过拟合，迫使参数在所有奇异向量方向上等强度学习。非矩阵参数（如 RMSNorm 的 gamma/bias、embedding、LM head）仍用 AdamW 处理。Muon 仅维护 1 个动量 buffer（vs AdamW 的 m 和 v 两个），内存开销减半。Moonshot AI (Liu et al. 2025) 将 Muon 扩展到大规模 LLM 训练，提出三项关键技术：weight decay、Consistent Update RMS 和 Distributed Muon，训练了 16B MoE 模型 Moonlight，证明 Muon 在 compute-optimal 设置下仅需 AdamW 约 52% 的训练 FLOPs 即可达到相同性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Muon 优化器在 LLM 训练中的完整 pipeline（结合 Liu et al. 2025 的扩展）：

```
# 每个训练 step，对每个矩阵参数 W ∈ R^{A×B}：
# Step 1: 计算梯度 G = ∇L(W_{t-1})  (标准反向传播)
# Step 2: Nesterov-style momentum
M_ext = μ * M_{t-1} + G              # M_{t-1} 来自上一步的动量

# Step 3: 准备 Newton-Schulz 输入 (Nesterov 外推)
X = μ * M_ext + G                     # 注意: M_ext = μ*M_{t-1} + G 后再加 G
X = X / ||X||_F                       # Frobenius norm 归一化，确保 |X|_F = 1

# Step 4: Newton-Schulz 迭代 (N=5, 系数 a=3.4445, b=-4.7750, c=2.0315)
for k = 1 to 5:                       # 在 bf16 精度下执行
    X_tmp = X @ X^T                   # [A, B] × [B, A] → [A, A]
    X = a*X + b*(X_tmp @ X) + c*(X_tmp @ X_tmp @ X)
    # 等价于 f(x) = ax + bx³ + cx⁵ 作用于奇异值
    # 结果 X ≈ U V^T (SVD 中 M = U Σ V^T 的左右奇异向量乘积)

O_t = X                               # 正交化后的更新方向

# Step 5: Consistent Update RMS + Weight Decay
update = 0.2 * O_t * sqrt(max(A,B))   # 缩放因子匹配 AdamW 的 update RMS ~0.2
                                       # sqrt(max(A,B)) 抵消 Lemma 1 的 shape 效应
W_t = W_{t-1} - lr * (update + λ * W_{t-1})  # λ = 0.1

# Step 6: 保存动量用于下一步
M_t = M_ext                           # 注意：存的是不带 Nesterov 外推的动量
```

关键超参数：lr 复用 AdamW 的 optimal lr（因 update RMS 已匹配），μ = 0.95，λ = 0.1，N=5。对于非矩阵参数（RMSNorm gamma/bias、embedding table、LM head），直接使用 AdamW 更新，共享相同的 lr 和 λ。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 原始实现：Keller Jordan 的 GitHub 仓库 [KellerJordan/Muon](https://github.com/KellerJordan/Muon)，提供 `zeropower_via_newtonschulz5` 函数，系数 a=3.4445, b=-4.7750, c=2.0315 由手工调优得到，确保多项式在 [0.5, 1.5] 范围内有界且零点处导数最大
- Moonshot AI 扩展：分布式 Muon 实现将以 PR 形式贡献给 Megatron-LM 开源项目；预训练 checkpoint、SFT checkpoint 已发布
- HuggingFace 社区实现：`Motif-Technologies/optimizer` 仓库的 `torch-ext/optimizer/muon.py` 提供了完整可复现代码；`bird-of-paradise/muon-distributed` 提供了带注释的 CPU 友好版本
- 使用时需注意：Muon 仅用于矩阵参数（≥2D），非矩阵参数（bias、norm、embedding）必须用 AdamW；Newton-Schulz 迭代在 bf16 下计算以利用 GPU tensor core，通信开销 <1% of total training FLOPs
- Newton-Schulz 系数可通过 Chebyshev-type 多项式加速（CANS, arXiv:2506.10935），或使用 AuON (arXiv:2509.24320) 以 O(n) 替代 O(n²) 的 Newton-Schulz

涉及论文标题：
- Muon is Scalable for LLM Training

## Newton-Schulz Iteration (牛顿-舒尔茨迭代 / Matrix Orthogonalization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Newton-Schulz 迭代是一种通过矩阵多项式迭代逼近矩阵极分解（polar decomposition）中正交因子的数值方法。给定矩阵 M，其极分解为 M = U P（U 正交，P 半正定），Newton-Schulz 迭代通过纯矩阵乘法（无 SVD/QR 分解）逼近 U = M (M^T M)^(-1/2)。在 Muon 优化器中，Newton-Schulz 用于将梯度动量矩阵 M_t 近似正交化为 O_t ≈ U V^T（即 M_t = U Σ V^T 的奇异向量乘积），用 5 次迭代逼近 (M M^T)^(-1/2) M。使用 5 阶多项式 f(x) = a x + b x³ + c x⁵（a=3.4445, b=-4.7750, c=2.0315），通过对动量矩阵在 Frobenius 归一化后反复应用矩阵乘法实现。该迭代的核心优势：(1) 仅需矩阵乘法——可在 GPU tensor core 上高效执行（bf16），<1% 总训练 FLOPs 开销；(2) 远快于 SVD（O(n³) 且 GPU 不友好）；(3) 5 步迭代足够产生良好正交近似。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Newton-Schulz 迭代在 Muon 中的计算过程：

```
输入: 梯度动量矩阵 M ∈ R^{A×B} (经过 Nesterov 外推)
输出: 近似正交化的更新矩阵 O ≈ (M M^T)^{-1/2} M

# 1. Frobenius 归一化
X = M / ||M||_F                    # 确保 X 的谱范数 ≤ 1，迭代稳定

# 2. 5 阶 Newton-Schulz 迭代 (quintic polynomial)
# 系数: a=3.4445, b=-4.7750, c=2.0315
for step in range(5):
    X_tmp = X @ X^T                 # [A, B] × [B, A] → [A, A]
    B = b * X_tmp + c * (X_tmp @ X_tmp)  # bX² + cX⁴
    X = a * X + B @ X               # aX + bX³ + cX⁵ (等价形式)

return X                            # X ≈ U V^T
```

迭代的数学原理：设 M 的奇异值为 σ₁, σ₂, ..., σ_min(A,B)，则第 k 次迭代后 X 的奇异值变为 f^(k)(σ_i / ||M||_F)，其中 f(x) = ax + bx³ + cx⁵。系数被设计为使 f 在零点导数最大且值域限制在 [0.5, 1.5] 内，这使得 X 的奇异值被推向 1（"半正交化"），而非完全正交化（精确正交化在实际训练中反而性能更差）。N=5 是精度-效率平衡点：N=10 产生更精确的正交化但无性能提升。

与经典 Newton-Schulz（3 阶 X_{k+1} = 1.5 X_k - 0.5 X_k X_k^T X_k）的区别：5 阶多项式通过待定系数法手工调优，在零点附近收敛更快（对小奇异值放大效果更好）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- Muon 原版实现：`zeropower_via_newtonschulz5` 函数，约 10 行 PyTorch 代码，所有矩阵运算在 bf16 下执行
- 加速变体：CANS (Chebyshev-Accelerated Newton-Schulz, arXiv:2506.10935) 使用 Chebyshev 交替定理和 Remez 算法推导最优系数，改进收敛速度
- 替代方案：AuON (arXiv:2509.24320) 提出 O(n) 的 hyperbolic-cosine RMS 缩放替代 O(n²) 的 Newton-Schulz，在保持性能的同时降低复杂度
- NVIDIA NeMo 的 Scion 优化器实现了可配置系数的 Newton-Schulz（"simple" / "quintic" / "polar_express"）
- 使用时需注意：若 A < B（矩阵更"瘦"），转置后计算可减少计算量（因 X @ X^T 的维度为 min(A,B) × min(A,B)）；N 通常设 5，更多迭代无额外收益

涉及论文标题：
- Muon is Scalable for LLM Training

## Consistent Update RMS (一致更新均方根 / Per-Parameter Update Scale Adjustment)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Consistent Update RMS 是 Muon 扩展到大规模 LLM 训练时的关键技术之一（Liu et al. 2025）。其核心问题来自 Lemma 1：shape 为 [A, B] 的满秩矩阵经 Muon 更新后，理论更新 RMS = √(1/max(A,B))。这意味着不同 shape 的矩阵参数（如 attention QKV 的 [H, H] vs MLP 的 [H, 2.6H] vs 独立 KV head 的小矩阵）会有差异极大的更新尺度：(1) 大矩阵（max(A,B) 大，如 MLP up-projection）更新过小，限制模型容量；(2) 小矩阵（max(A,B) 小，如 GQA/MLA 中独立 KV head）更新过大，导致训练不稳定。解决方案：对每个矩阵参数按 √(max(A,B)) 缩放其 Muon 更新，再乘以 0.2 因子以匹配 AdamW 的经验更新 RMS 范围（0.2~0.4），使所有矩阵参数在不同 shape 下具有一致的更新尺度，且可直接复用 AdamW 调优的 lr 和 weight decay。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Consistent Update RMS 在 Muon 更新中的具体实现（Adjusted LR 方法）：

```
# 对每个矩阵参数 W，其 shape = [A, B]
O_t = Newton-Schulz(M_t)                        # 正交化后的更新方向
                                                 # O_t 的 RMS ≈ √(1/max(A,B)) (Lemma 1)

# Adjusted LR: 按 shape 缩放以取消 Lemma 1 的效应
scale = 0.2 * sqrt(max(A, B))                   # 使最终 update RMS ≈ 0.2
update = scale * O_t                             # 现在 update RMS ≈ 0.2，与 AdamW 一致

W_t = W_{t-1} - lr * (update + λ * W_{t-1})     # lr 和 λ 直接复用 AdamW 的 optimal 值
```

与其他方法的对比（消融实验，Table 1）：
1. Baseline: scale = 0.2 * √H（H=hidden size），对大矩阵 [H, 4H] 更新不足，validation loss 2.812
2. Update Norm: O_t / RMS(O_t) * 0.2，直接归一化更新 RMS 到 0.2，对所有矩阵一视同仁，但忽略了不同 shape 应有的不同行为，validation loss 2.789
3. Adjusted LR: √(max(A,B)) 缩放，既保持不同 shape 矩阵的自然差异，又使 RMS 与 AdamW 一致，validation loss 2.789（与 Update Norm 相当但计算开销更低）

Adjusted LR 被选为最终方案，因其在 MLP 权重上有效提升 RMS（相对 Baseline 翻倍），同时在 attention QKV 矩阵上保持与 Baseline 一致的 RMS（因 max(H,H)=H，√(max)/√H=1）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 在优化器的 `step()` 函数中，对每个矩阵参数组计算 `max(param.shape[0], param.shape[1])`，乘以 0.2 作为缩放因子
- 该缩放是纯标量操作，计算开销可忽略不计
- 需与 weight decay 配合使用，缩放仅作用于正交化更新 O_t，不作用于 weight decay 项 λW_{t-1}
- 设置 0.2 的理由来自消融实验（Table 8）：在 [0.05, 0.1, 0.2, 0.4, 0.8] 范围内，0.2 和 0.4 表现相当且明显优于其他值，0.2 被选择以与 AdamW 经验范围的下限对齐
- 原始 Muon (Keller Jordan) 的缩放方式为 √(max(1, A/B))，在矩阵 second dimension 相同时等价于本文方案

涉及论文标题：
- Muon is Scalable for LLM Training

## Weight Decay in Muon (Muon中的权重衰减 / L2正则化在矩阵正交化优化器中的应用)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Weight Decay（权重衰减）是 AdamW (Loshchilov & Hutter, 2019) 中提出的 decoupled weight decay 机制：W_t = W_{t-1} - η_t (∇L_effective + λ W_{t-1})，即直接将 λW 作为独立项加入更新而非通过 L2 正则化嵌入损失函数。在原始 Muon (Keller Jordan et al. 2024) 中未包含 weight decay。Liu et al. (2025) 发现在大规模训练中，vanilla Muon 的权重 RMS 和层输出 RMS 持续增长超出 bf16 表示范围（图 2：初期收敛快，但长期被 AdamW 超越），引入 AdamW 风格的 weight decay (λ=0.1) 后解决了此问题——Muon + weight decay 在 over-train 区间持续优于 AdamW。更新公式变为：W_t = W_{t-1} - η_t (0.2·O_t·√(max(A,B)) + λ W_{t-1})。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Weight Decay 在 Muon 中的更新流程：

```
# 对矩阵参数 W ∈ R^{A×B}：
O_t = Newton-Schulz(M_t)                        # 正交化 momentum

# 两项独立相加 (与 AdamW 风格一致)
gradient_update = 0.2 * O_t * sqrt(max(A, B))   # 正交化更新 (RMS ~0.2)
weight_decay_term = λ * W_{t-1}                  # λ = 0.1

W_t = W_{t-1} - lr * (gradient_update + weight_decay_term)
```

关键效果 (图 2)：
- 无 weight decay 的 vanilla Muon：初期收敛最快（红色曲线），但约 40B tokens 后被 Muon+weight decay（蓝色）超越，最终高于 AdamW（绿色）
- Muon + weight decay：全程优于 AdamW，在 100B tokens（~5× optimal）时仍保持优势
- Weight decay 有效抑制了大矩阵（如 MLP [H, 2.6H]）在长期训练中权重 RMS 的发散问题

论文还特别指出对 RMSNorm 的 gamma 参数施加 weight decay 对训练稳定性至关重要——防止每层输出 RMS 过高。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 在 PyTorch 中，通过优化器的 `param_groups` 为不同参数组设置不同的 weight_decay 值，矩阵参数组与非矩阵参数组可共享相同的 λ
- λ 的典型值：0.1（论文所有训练阶段一致使用），与常见 AdamW 设置兼容
- 与 L2 正则化的区别：Weight Decay 直接作用在参数上（λW），而 L2 正则化作用于梯度（∂(λ||W||²)/∂W = 2λW，与自适应学习率交互后不等价）
- 对于 Muon 中的 weight decay，由于 Newton-Schulz 正交化已标准化了梯度方向，weight decay 是主要控制参数范数增长的机制
- 原始 Muon 仓库在论文发表后的 commit (e0ffefd) 中同步添加了 weight decay 支持

涉及论文标题：
- Muon is Scalable for LLM Training

## SVD Entropy (SVD熵 / 奇异值分解熵 / Spectral Entropy Analysis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SVD Entropy（奇异值分解熵）是一种量化矩阵奇异值分布"平坦度"的度量。给定权重矩阵 W 的奇异值 σ = (σ₁, σ₂, ..., σ_n)（降序排列），SVD entropy 定义为：
$$H(σ) = -\frac{1}{\log n} \sum_{i=1}^{n} \frac{\sigma_i^2}{\sum_{j=1}^{n} \sigma_j^2} \log \frac{\sigma_i^2}{\sum_{j=1}^{n} \sigma_j^2}$$
值域 [0, 1]。H=1 表示所有奇异值相等（分布最平坦、最均匀），H→0 表示仅少数奇异值主导（分布最集中、rank 最低）。在深度学习优化中，SVD entropy 用于评估优化器是否使模型权重矩阵学习到更多样化、更平坦的奇异值谱——高 SVD entropy 意味着权重在更多方向上具有表达能力，而非集中在少数主导方向（后者可能导致过拟合或容量浪费）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 Muon vs AdamW 的谱分析实验中 (Sec 3.4)：

```
# 对每个权重矩阵 W：
U, Σ, V^T = SVD(W)                     # 计算奇异值分解
σ = diag(Σ)                             # 提取奇异值向量

# 归一化为概率分布
p_i = σ_i² / Σ_j σ_j²                   # 用平方奇异值（对应能量/方差）

# 计算归一化 SVD entropy
H = - Σ_i p_i * log(p_i) / log(n)       # n = min(A, B)，归一化到 [0,1]

# 分组平均
groups = {AttnQO, AttnKV, Experts, SharedExperts, Router, Dense}
H_group = mean(H over all matrices in group)
```

实验结果 (图 4)：
- 在 1.2T tokens 训练过程中的所有 checkpoint、所有 6 组权重矩阵上，Muon 的 SVD entropy 均高于 AdamW
- Router 权重的差异最大（Muon 显著高于 AdamW），说明 MoE 模型受益更大——更平坦的路由器权重谱意味着更差异化的专家选择
- 超过 90% 的独立权重矩阵在 Muon 下 SVD entropy 更高（Appendix F, 图 9-10）
- Singular value 分布可视化显示 Muon 训练的权重奇异值曲线更平坦（更少在少数大奇异值处集中）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 计算工具：任何支持 SVD 的数值库（PyTorch `torch.linalg.svdvals()`, NumPy `np.linalg.svd()`）
- 在训练中通常仅对 checkpoint 做离线分析（全 SVD 开销大），不必每步计算
- 应用场景：(a) 优化器对比——衡量不同优化器产生的权重多样性（如 Muon vs AdamW）；(b) 权重初始化质量评估——高 SVD entropy 的初始化可能更有利于训练；(c) 模型压缩——ARSVD (Adaptive-Rank SVD, Cherukuri & Lala 2025) 用 SVD entropy 指导每层 rank 分配
- 变体：某些工作用 σ_i 而非 σ_i² 计算 p_i，或使用非归一化 entropy H = -Σ σ_i log σ_i（此时非归一化到 [0,1]）
- 注意事项：SVD entropy 仅反映奇异值分布，不直接度量模型性能；高 entropy 不等于好性能，但结合 AdamW 和 Muon 的实验，高 entropy 与更好的下游性能相关

涉及论文标题：
- Muon is Scalable for LLM Training

## Steepest Descent under Norm Constraints (范数约束下的最陡下降 / Spectral Norm Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Steepest Descent under Norm Constraints 是 Bernstein & Newhouse (2024) 提出的优化理论框架。核心观点：深度学习的每一步优化可视为在某种范数约束下寻找使损失下降最快的方向（最陡下降方向）。形式化地：ΔW = argmin_{||Δ|| ≤ η} ⟨∇L, Δ⟩，其中范数 ||·|| 的选择决定了优化器的行为。在此框架下：
- Adam/AdamW 可解释为 Max-of-Max norm 约束下的最陡下降（动态调整的逐元素范数约束）
- Muon 可解释为 spectral norm（或大 p 的 Schatten-p norm）约束下的最陡下降——当 Newton-Schulz 精确计算时，Muon 的谱范数约束意味着更新矩阵的奇异值被限制为 1，即更新在所有方向上等强度
- 从数学角度看，权重矩阵作为输入/隐空间上的 operator，其自然范数应为 induced operator norm（spectral norm），因此 Muon 的 norm constraint 比 AdamW 的逐元素约束更合理

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
该理论对 Muon 的设计提供了数学解释：

```
# 一般的最陡下降框架：
给定 norm ||·||，每一步求解：
ΔW* = argmin_{||Δ|| ≤ η} ⟨∇L(W), Δ⟩

# 不同的 norm 选择对应不同的优化器：
# - ||Δ||_∞（element-wise max norm）→ 符号梯度下降 (signSGD)
# - Max-of-Max norm（动态自适应）→ Adam/AdamW
# - ||Δ||_2（spectral norm）→ Muon (当 Newton-Schulz 精确时)
# - ||Δ||_{S_p}（Schatten-p norm, p 大）→ Muon 近似实现

# Muon 如何实现 spectral norm 约束下的最陡下降：
M = momentum(∇L)                           # 先累积动量
O = Newton-Schulz(M)                        # O ≈ U V^T
                                            # O 的奇异值 = 1 (精确时) 或 ≈ 1 (近似)
                                            # ||O||_2 = 1, ||O||_F = √r
ΔW = -η * O                                # 谱范数 = η (受约束)
```

该视角还揭示了 Muon 与 Shampoo 的关系：当去掉 Shampoo 中的 preconditioner accumulation 后，Shampoo 的更新退化为 W_{t+1} = W_t - η U V^T（即无动量的 Muon = spectral descent）。移除 preconditioner 等价于将优化问题拉伸为各向同性——这正是矩阵正交化所做的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现方式：
- 该理论框架本身不是可执行的算法，而是理解已有优化器行为的概念工具
- 实践含义：(a) 选择优化器 = 选择 norm constraint 类型，应根据参数结构选择（矩阵参数 → spectral norm，向量参数 → Euclidean norm）；(b) Muon 的设计由此推广到 Schatten norm——未来工作方向是在 Muon 框架中引入 Schatten-p norm 支持（论文 Sec 4 讨论），可能通过调整 Newton-Schulz 多项式实现不同的奇异值变换；(c) 该框架解释了为什么 Muon 与 AdamW 结合使用是合理的——非矩阵参数（norm、bias、embedding）的适当范数是 Euclidean/逐元素范数（AdamW），而矩阵参数的适当范数是 spectral norm（Muon）
- 相关代码：Bernstein & Newhouse 的分析在 [arXiv:2409.20325](https://arxiv.org/abs/2409.20325)；Cesista (2024) 的博客提供了可视化解释

涉及论文标题：
- Muon is Scalable for LLM Training

## Expert Residual Inlining（专家残差内联）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Residual Inlining 是 NetMoE 为支持 Dynamic Sample Placement 而提出的计算顺序重排技术。在标准 Transformer MoE 层中，残差连接（residual connection）在 All-to-All Gather 之后执行：`output = gather(expert_outputs) + residual_input`。但在 NetMoE 中，All-to-All Gather 阶段会改变 token 的放置位置（按优化后的 SmpDev 重分配），若残差在原位置执行将导致计算错误。Expert Residual Inlining 将残差加法从 gather 之后移到 scatter 之后、gather 之前：`output_on_expert_device = scatter_input + expert_output`，然后 gather 只需将结果传输到新的 sample 位置。这样保证：不论 token 最终被 gather 到哪个 GPU，其残差连接的计算结果都是正确的——因为残差已经在 expert 所在的 GPU 上被加到了 expert 输出中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
**标准 MoE 层前向传播（无 sample placement 调整）**：
```
input: x ∈ R^{S×H}  (tokens on device d)
route = gating_network(x)  # K selected experts per token
x_scattered = all_to_all_scatter(x, route)  # dispatch to expert GPUs
x_expert = experts(x_scattered)  # FFN on each expert's GPU
x_gathered = all_to_all_gather(x_expert, reverse(route))  # return to original GPU
output = x_gathered + x  # residual connection
```

**NetMoE 的 Expert Residual Inlining**（Algorithm 1, lines 10-12）：
```
input: x ∈ R^{S×H}  (tokens on device d)
route = gating_network(x)
x_scattered = all_to_all_scatter(x, route)  # dispatch to expert GPUs
x_expert = experts(x_scattered)  # FFN on expert's GPU
x_inlined = x_expert + x_scattered  # EXPERT RESIDUAL INLINING: residual added HERE
# CPU 后台求解最优 SmpDev
x_gathered = all_to_all_gather(x_inlined, SmpDev_optimized)  # gather to NEW positions
# output = x_gathered  # 无需再 + x，已内联
```
注意：残差加法的输入是 `x_scattered`（scatter 到 expert GPU 上的 token 数据）而非原始的 `x`（原始 GPU 上的数据），因为两者在 scatter 前后是等价的（只是位置不同），且 inlining 后不再需要 gather 回原 GPU——这是实现零额外通信开销 place 调整的关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现要求：修改 MoE 层的 forward 函数，将 `x_residual = x` 的保存和 `output = gathered + x_residual` 的执行逻辑替换为 `gathered = all_to_all_gather(x_expert + x_scattered, new_placement)`。
- 正确性保证：数学上等价——残差加法的交换律和结合律保证无论在哪台 GPU 上执行 `expert_output + original_input` 结果相同。
- 与标准 Transformer 的差异：标准实现中残差独立于 MoE 层（在 Transformer block 级别），而 Expert Residual Inlining 将残差嵌入 MoE 层内部。
- 适用范围：仅在需要改变 token 返回位置（如动态 sample placement）时才需要。若不做 placement 调整，标准残差方式更简单。
- 限制：论文未讨论对 gradient checkpointing（重计算）和混合精度训练（FP16/BF16）的具体影响，这些是实际部署中需要验证的问题。

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement

## Gating Network in MoE（MoE 门控网络 / Router）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Gating Network（也称 Router 或 Gate）是 Mixture of Experts 模型中的核心路由组件，负责为每个输入 token 选择应激活的 expert(s)。标准实现为：`g(x) = softmax(W_g · x)`，其中 `W_g ∈ R^{H×E}` 是可训练参数，输出每个 expert 的得分。然后通过 Top-K 选择（K 通常为 1 或 2）确定每个 token 的路由目标：`route = topk(g(x), K)`。最终输出为各选中 expert 输出的加权和：`y = Σ_k g(x)_k · expert_k(x)`。Gating Network 的训练通过 auxiliary load balancing loss（如 GShard 的 `L_aux = E·Σ_e f_e·P_e`，其中 f_e 为 expert e 被选中的比例，P_e 为 gate 分配给 expert e 的平均概率）来防止 expert 崩溃（所有 token 路由到少数 expert）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
**MoE 层前向传播（含 Gating Network）伪代码**：
```python
def moe_layer_forward(x):  # x: [batch, seq_len, hidden_dim]
    # Step 1: Gating
    gate_logits = gate_linear(x)  # [B, S, E], E = num_experts
    gate_probs = softmax(gate_logits, dim=-1)
    topk_weights, topk_indices = topk(gate_probs, K)  # [B, S, K]
    topk_weights = softmax(topk_weights, dim=-1)  # re-normalize
    
    # Step 2: Dispatch tokens to experts via All-to-All Scatter
    dispatched = all_to_all_scatter(x, topk_indices)
    
    # Step 3: Expert computation
    expert_outputs = []
    for e in range(E):
        expert_tokens = dispatched[e]  # tokens routed to expert e
        expert_out = expert_ffn[e](expert_tokens)  # FFN forward
        expert_outputs.append(expert_out)
    
    # Step 4: Combine via All-to-All Gather
    combined = all_to_all_gather(expert_outputs, reverse(topk_indices))
    
    # Step 5: Weighted sum
    output = sum(topk_weights[k] * combined[k] for k in range(K))
    return output
```

NetMoE 中 Gating Network 的关键角色：
- routing 结果 `route ∈ N^{I×L×K}` 是 NetMoE 优化的输入——基于 routing 计算 `num_{i,e}`，进而构建二分图边权重，求解最优 sample placement。
- NetMoE 需要下一层 routing 结果来计算 `c^{(l+1,scatter)}`，通过将当前层输入传入下一层 router 提前预测（Eliseev & Mazur, 2023; Tang et al., 2024）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 标准实现：`torch.nn.Linear(hidden_dim, num_experts)` + Top-K + Softmax。在 FastMoE/FasterMoE/Tutel 等框架中封装为 MoE layer 的一部分。
- Top-K 选择策略变体：Top-1（Switch Transformer）、Top-2（GShard/Mixtral）、expert choice routing（让 expert 选择 top tokens）、随机 routing（训练早期增加探索）。
- 负载均衡：auxiliary loss（GShard）、capacity factor（Switch Transformer，限制每个 expert 处理的最大 token 数）、z-loss（ST-MoE，防止 logits 过大）。
- Router 预测下一层路由：Eliseev & Mazur (2023) 提出在 MoE 推理 offloading 中用当前层输入预测下一层路由；NetMoE 将此技术用于训练中以获取 `c^{(l+1,scatter)}` 信息。
- NetMoE 中的 Gating 不受修改——routing 机制完全保持原样，保证了模型收敛不受影响（与修改 routing 的 topology-aware 方法如 TA-MoE、SCoMoE 形成对比）。

涉及论文标题：
- NetMoE: Accelerating MoE Training through Dynamic Sample Placement

---

## Adaptive Domain-Embedding Router for MoE（基于域嵌入的自适应 MoE 路由器）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Adaptive Domain-Embedding Router 是 Nexus (Gritsch et al., 2024) 提出的 MoE 路由器设计。与传统线性路由器 W_r @ x 不同，Nexus router 由两部分组成：(1) 投影层 P_r（2-layer SwiGLU MLP）：将预计算的域嵌入 d_i ∈ R^m 投影为 expert embedding e_i ∈ R^h；(2) 相似度路由：通过 token hidden state x 与各 e_i 的点积计算路由概率 s_i = softmax(x · e_i)。P_r 是 hypernetwork——它以域嵌入 d_i 为条件生成路由器参数（expert embeddings），而非直接存储固定参数。域嵌入 d_i 通过外部 embedding model（如 Cohere Embed v3）对该域训练数据编码后取平均得到，也可通过无监督聚类的 centroid 获得。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Nexus Adaptive Domain-Embedding Router（每个 Transformer block 一个）
# 输入: pre-computed domain_embeddings [m, n_experts], hidden states x [s, h]
# 输出: routed expert indices, gate weights

# Step 1: 投影层将域嵌入映射为专家嵌入
# P_r = 2-layer MLP with SwiGLU: W1 in R^{2h×m}, W2 in R^{h×h}
for i in range(n_experts):
    e_i = P_r(d_i)                          # d_i: [m] → e_i: [h]
    # = W2 @ SwiGLU(W1 @ d_i)
    # SwiGLU(x) = SiLU(W_gate @ x) ⊙ (W_up @ x)

# Step 2: 计算 token 与各专家嵌入的相似度
router_logits = x @ E                        # [s, n_experts], E = [e_1,...,e_n]

# Step 3: Top-K 路由
router_probs = softmax(router_logits)
topk_gates, topk_indices = topk(router_probs, k=1)

# Step 4: MoE 输出
output = shared_expert_ffn(x)               # seed model FFN, always active
output += topk_gates * routed_expert_ffns[topk_indices](x)
```
**添加新 expert（Extension）**：新域 d_new → e_new = P_r(d_new) → 直接 append FFN_new 到 expert 数组。P_r 不变，无需重新训练 router 维度。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：投影层 P_r 在每个 Transformer block 中独立训练一个 2-layer SwiGLU MLP。域嵌入在训练前预计算并存储，推理时可预计算 expert embedding 缓存（与输入 x 无关）。
- **与超网络的关系**：P_r 是 hypernetwork——它以域嵌入 d_i 为条件生成路由器参数（e_i），而非直接存储固定路由参数。
- **优势**：(a) Router 参数与 expert 数量解耦——新增 expert 不需要扩大 router；(b) 域语义归纳偏置——即使负载均衡损失很低，域嵌入的内在语义也能维持稳定的路由（Nexus ablation：α=0.0005 时性能不变，线性 router 下降 2%）；(c) 保留域间关系——P_r 投影后 Books-C4、GitHub-SE 的高 cosine similarity 关系被保留。
- **局限**：需要外部 embedding model 预计算域嵌入（论文使用 Cohere Embed v3）；当前仅验证了基于预定义数据源的域划分。
- **代码**：论文未开源，但提供了完整的 PyTorch 伪代码（Figure 2）。

涉及论文标题：
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

---

## Expert Specialization in MoE（MoE 中的专家专业化度量）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Specialization（专家专业化）是指 MoE 模型中不同 expert FFN 对特定数据域或任务形成差异化处理能力的现象。一个"专业化"的 expert 意味着它在处理其专长域的数据时被 router 频繁选择。衡量专业化程度的标准方法是计算路由概率矩阵：对每个域采样的 token，统计其在各 Transformer block 中被路由到各 expert 的平均概率。DeepSeek-MoE (Dai et al., 2024) 将"ultimate expert specialization"作为核心目标。Nexus 系统性地量化了 upcycled MoE 中的 expert specialization：通过计算每个域 token 跨所有 Transformer block 的平均路由频率矩阵（Figure 5）来验证。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Expert Specialization 度量（Nexus Figure 5 方法）
for domain_d in [ArXiv, Books, C4, SE, Wiki, Code]:
    samples = sample_from_domain(domain_d, n=512)
    routing_count = zeros(n_experts)
    for each token in samples:
        for each MoE layer l:
            expert_chosen = router.forward(token)  # top-1 index
            routing_count[expert_chosen] += 1
    routing_freq[domain_d] = routing_count / sum(routing_count)
# 理想专业化: routing_freq[domain_d][expert_d] → 1.0
# Nexus 结果: ArXiv→ArXiv: 63.0%, Books→Books: 64.7%, Wiki→Wiki: 69.8%,
#             C4→C4: 40.9% (C4 覆盖广), Code→Code: 69.1% (新增后)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **度量工具**：通过 hook router 的 top-k 选择（记录每个 token 的 selected expert index），在评估集上统计路由频率矩阵。
- **影响因素**：(a) 域数据的覆盖范围（C4 因覆盖广导致路由分散到其他 expert）；(b) 训练数据采样策略（均匀采样 vs 比例采样——Nexus 发现均匀采样将 C4 路由精度从 27.6% 提升至 71.1%）；(c) load balancing loss factor（过高会强制分散路由，降低专业化显示度）。
- **与 vanilla MoE 的对比**：standard MoE training 中专家通常不展示明确域专业化（Jiang et al., 2024; Zoph et al., 2022），因为 router 仅基于 token hidden state 选择 expert，无域语义信息作为路由依据。Nexus 通过域嵌入 router 的归纳偏置实现并保持了专业化——这是区分"语义专业化"（由数据驱动）和"统计均衡"（由 load balancing 驱动）的关键洞见。

涉及论文标题：
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

**局部路由一致性视角下的专家专业化** (来自 "Not All Models Suit Expert Offloading", ICLR 2026)：论文将 Expert Specialization 区分为两种类型：(1) Domain Specialization——expert 对不同领域数据的激活频率差异（用 Coefficient of Variation across domains 量化）；(2) Vocabulary Specialization——expert 对特定 token ID 的激活频率差异（分为 input/predicted output/ground-truth 三种）。关键发现：Domain-specialized experts 对局部路由一致性的贡献显著大于 vocabulary-specialized ones；高 SRP 且 global load balance 良好的模型（如 Qwen3, GRIN-MoE, OLMoE）同时具有强 domain specialization。机制：domain-specialized expert 在匹配其专长领域的上下文中持续激活（高局部一致性），而在不相关领域则保持 inactive（实现全局负载均衡）。Paper Figure 7 展示了 SRP 与 domain specialization 的正相关，而与 input vocabulary specialization 的负相关或无显著相关。

---

## Domain Embedding for MoE Routing（用于 MoE 路由的域嵌入）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Domain Embedding for MoE Routing 是 Nexus 提出的用预计算的域级语义嵌入替代随机初始化作为 MoE router 输入的技术。传统 MoE router 用随机初始化的线性层参数 W_r 处理每个 token，与域语义无关；Nexus 则用外部 embedding model（论文使用 Cohere Embed v3）对每个域对应的训练数据集编码，将编码向量平均得到 d_i ∈ R^m 作为该域的"域嵌入"，然后通过投影层 P_r 将 d_i 映射为该域 expert 的 expert embedding e_i。域嵌入在训练前一次计算并存储，训练和推理期间不更新。替代方案：如果使用无监督聚类划分域（如 c-BTM 的 Gururangan et al. 2023），centroid 可代替 embedding model。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 域嵌入预计算（离线，一次完成）
for domain_i in [ArXiv, Books, C4, SE, Wiki]:
    domain_data = load_domain_data(domain_i)
    embeddings = []
    for doc in domain_data:
        emb = embed_model.encode(doc)        # [m]
        embeddings.append(emb)
    d_i = mean(embeddings, dim=0)            # [m]

# 投影后 expert embedding 的域间关系（Nexus Figure 8）:
# 投影前 cosine similarity: Books-C4 ≈ 0.6, GitHub-SE ≈ 0.7
# 投影后: 相对关系保持但整体 pushed apart（lower inter-expert similarity）
# P_r 的学习目标: 在保持域关系的条件下增大 expert embedding 间距
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现**：(a) 外部 embedding model：需要与 LLM 独立的 embedding service；(b) 预计算：对每个域的整个训练集编码 → 平均 → 存储为 [m, n_domains] 张量；(c) 在 router 训练中作为固定输入（不参与梯度更新）。
- **适用条件**：需要预先划分的数据域结构（如 SlimPajama 的 sub-dataset）；域嵌入质量依赖 embedding model 的表征能力。
- **灵活性**：P_r 投影保持域间相对关系（Figure 8），使得语义相近的域（Books & C4）的 expert embedding 也相近——token 可能被交叉路由——这一特性实现了隐式的跨域知识共享，同时避免了同一 token 被完全不相关的 expert 处理。

涉及论文标题：
- Nexus: Specialization meets Adaptability for Efficiently Training Mixture of Experts

---

## Post-training Expert Pruning via Reconstruction Loss Minimization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Post-training Expert Pruning（后训练专家剪枝）是一种无需训练的 MoE LLM 压缩技术，通过永久移除 MoE 层中贡献较小的 expert 来减少模型参数总量和部署内存需求。该方法属于 expert-level sparsity（专家级稀疏），与传统的 weight-level 剪枝（如 Wanda 的非结构化稀疏、SparseGPT 的 2:4 结构化稀疏）正交。

核心思想：给定一个训练好的 MoE 模型（如 Mixtral 8x7B，每层 8 个 expert），对每一层独立地枚举所有可能的保留 r 个 expert 的组合 C（从 n 个中选 r 个），对每个组合计算"token 重建损失"——即 prune 后 MoE 层输出 F'(x, C) 与原始层输出 F(x) 之间的 Frobenius 范数差异。选择使重建损失最小的组合 C*，丢弃其余 n−r 个 expert。

公式化为逐层优化问题（受 He et al. 2017 channel pruning 启发）：

$$\min_{\mathbf{C}} \|\mathcal{F}'(\boldsymbol{x}, \mathbf{C}) - \mathcal{F}(\boldsymbol{x})\|_F \quad \text{s.t.} \quad \mathbf{C} \subseteq \{\text{expert}_0, \dots, \text{expert}_{n-1}\}, |\mathbf{C}| = r$$

其中 x 为校准集缓存的输入，F(x) 为原始输出，F'(x, C) 为仅保留 C 中 expert 及对应 routing weight 后的输出。

关键实现细节：
- 校准集：task-agnostic 使用 C4（与 Wanda 一致，pre-training 数据覆盖面广）；task-specific（如数学）切换到 MATH training set
- 枚举复杂度：每层 C(n, r) 种组合。Mixtral 8x7B 的 n=8，r=6 时 C(8,6)=28 种；r=4 时 C(8,4)=70 种。Pruning 耗时 r=6 约 30 分钟，r=4 约 90 分钟
- 逐层独立剪枝（layer-wise），而非逐层渐进式剪枝（progressive layer-by-layer）：实验表明 progressive 方式在高速剪枝率下会过拟合小校准集（Tab. 6）
- 剪枝后仅需修改 HuggingFace model config 的 num_experts 字段即可加载部署

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ============ Step 1: 校准数据缓存 ============
calibration_set = sample(C4, num_sequences=128, seq_len=2048)
for each MoE_layer l in [0, 1, ..., L-1]:
    X_l = []   // 输入 token hidden states
    Y_l = []   // 原始输出
    for x in calibration_set:
        x_l = forward_to_layer_l(x)           // 到达第 l 层的 hidden state
        y_l = MoE_layer_l(x_l)                // 8-expert 完整推理
        X_l.append(x_l)
        Y_l.append(y_l)

// ============ Step 2: 逐层枚举搜索 ============
for each layer l:
    best_loss = inf
    best_subset = None
    for each subset C of {expert_0, ..., expert_7} with |C| == r:
        total_loss = 0
        for each (x, y) in zip(X_l, Y_l):
            y_hat = pruned_MoE_layer(x, C)    // 仅 C 中 expert 参与 top-k
            total_loss += ||y_hat - y||_F^2
        avg_loss = total_loss / len(X_l)
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_subset = C
    keep[l] = best_subset

// ============ Step 3: 拼接并保存 ============
for each layer l:
    experts[l] = experts[l][keep[l]]
    router_weights[l] = router_weights[l][keep[l]]
model_config.num_experts = r
save_pruned_model(model)
```

**Pruned MoE Layer 推理（推理阶段）**：
```
Input: token x ∈ R^d
Router: l = W_router @ x                // logits ∈ R^r (仅保留的 r 个 expert)
        w = Softmax(l)                  // routing weights ∈ R^r
        {e0, e1} = TopK(w, k=2)        // 从 r 个中选 top-2
        w̃_e0 = w_e0 / (w_e0 + w_e1)
        w̃_e1 = w_e1 / (w_e0 + w_e1)
Expert e0: z0 = W_down_e0 @ (SiLU(W_gate_e0 @ x) ⊙ W_up_e0 @ x)
Expert e1: z1 = W_down_e1 @ (SiLU(W_gate_e1 @ x) ⊙ W_up_e1 @ x)
Output: z = w̃_e0 * z0 + w̃_e1 * z1
```

Annotations:
- 逐层搜索空间：C(n, r) 种组合，Mixtral n=8, r=6 → 28 种/层，32 层共 896 次评估
- 校准数据量：128 sequences × 2048 tokens = 262,144 tokens
- Pruning 后 expert 权重直接删除（deleted），不是 mask（zero-out），因此内存实际减少
- 关键设计选择：layer-wise 独立（非 progressive）是为了避免小校准集导致的过拟合

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **开源实现**：https://github.com/Lucky-Lance/Expert_Sparsity（论文承诺开源）
- **部署方式**：Prune 后仅需修改 HuggingFace model config 中 num_experts 字段，无需修改推理代码
- **Task-Agnostic vs Task-Specific**：通用任务用 C4 校准，领域任务（如数学）切换到领域数据集（如 MATH）。两种校准仅在 Fig. 4 中 4/32 层的 expert 选择相同，差异显著
- **Fine-tuning 恢复精度**：Task-specific prune 后在 MetaMathQA 上 fine-tune 900 steps（lr=2e-5, cosine scheduler, 16×A100），可将 r=6 模型的 GSM8K 从 51.25 恢复到 79.53，接近原模型 81.35
- **限制**：枚举复杂度为 O(C(n, r))，n=32 expert 时 C(32,6)=906,192 组合，不可行。论文承认此局限
- **与 weight pruning 的关系**：正交且可叠加（Sec. A.6），expert pruning 减少 expert 数量，weight pruning 减少单个 expert 内部参数，quantization 进一步减少 bit-width

涉及论文标题：
- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models

---

## Dynamic Expert Skipping in MoE Inference

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Expert Skipping（动态专家跳过）是在 MoE LLM 推理时，对每个 token 动态决定是否跳过 routing weight 较小的 expert，从而减少实际激活的 expert 数量以加速推理。与 Expert Pruning（永久删除 expert 权重）不同，Dynamic Skipping 不修改模型参数，仅在推理时基于 routing weight 比例做在线决策。

核心机制（top-2 场景，k=2）：对于每个 token x，Router 计算 routing weights w = {w_{e0}, w_{e1}}（w_{e0} ≥ w_{e1}）。如果次要 expert 的权重远小于主要 expert，即 w_{e1} < β · w_{e0}，则跳过 e1，仅使用 e0 计算输出。β 是每层独立的超参数，通过在校准集上前向推理并取该层所有 token 的 w_{e1}/w_{e0} 的中位数来确定（使跳过概率约 50%）。

理论依据（Sec. A.2）：假设不同 expert 输出向量间的 L2 距离 D 近似恒定，则跳过次要 expert(s) 的重建损失上界为：

$$\mathcal{L} \leq \frac{\sum_{m=i+1}^{k} w_m}{\sum_{m=1}^{k} w_m} \cdot D$$

在 top-2 特例下，条件简化为 w_{e1} ≤ β · w_{e0}（β = H/(D−H)），其中 H 为允许的损失上界。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// ============ Calibration: 确定每层 β ============
for each MoE_layer l:
    ratios = []
    for x in calibration_set:
        w = Softmax(Router_l(x))            // routing weights
        {e0, e1} = TopK(w, k=2)            // w_e0 ≥ w_e1
        ratios.append(w_e1 / w_e0)
    beta[l] = median(ratios)               // 每层独立中位数

// ============ Inference: 动态跳过 ============
for each token x in input_sequence:
    for each MoE_layer l:
        w = Softmax(Router_l(x))
        {e0, e1} = TopK(w, k=2)
        if w_e1 < beta[l] * w_e0:
            // 跳过 e1，仅使用 top-1 expert
            z = E_{e0}(x)                   // 未归一化，因仅一个 expert
        else:
            // 正常 top-2
            w̃_e0 = w_e0 / (w_e0 + w_e1)
            w̃_e1 = w_e1 / (w_e0 + w_e1)
            z = w̃_e0 * E_{e0}(x) + w̃_e1 * E_{e1}(x)
```

**推广到 top-k 场景（k > 2）**：
```
// 保留 top-i* expert，其中 i* = min i 满足：
// Σ_{m=i+1}^k w_m ≤ β · Σ_{m=1}^k w_m
i_star = 1
cumsum = w_e1 + w_e2 + ...   // 从第2大开始累加
total = w_e0 + w_e1 + ... + w_{k-1}
while i_star < k and cumsum > beta * total:
    cumsum -= w_{e_{i_star}}
    i_star += 1
// 使用 top-i_star expert 计算加权输出
```

Annotations:
- β 确定逻辑：取中位数使 calibration 集上约 50% token 跳过次要 expert
- 当 w_{e1} ≈ 0（次要 expert 几乎无贡献）时跳过收益最大，w_{e0} ≈ w_{e1}（两 expert 同等重要）时不跳过
- 跳过不影响内存使用（expert 权重仍在 GPU 显存中），但减少计算 FLOPs
- 与 Expert Pruning 正交叠加：r=6 pruning + skipping 可得 1.23× 加速 + 62.91 LM-eval；而 r=4 pruning alone 仅 59.57

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- **实现方式**：在 HuggingFace Transformers 的 MoE layer forward 中插入条件判断，无需修改模型权重
- **β 校准**：对 Mixtral 8x7B + C4 校准集，第一层 β 值从 0.402 到最后一层 0.535 不等（见 Sec. A.7），各层差异显著，说明 expert 选择倾向在不同深度变化
- **适用场景**：(a) 与 Expert Pruning 组合使用以进一步加速；(b) 单独用于 8-expert 原模型：仅 skipping 可得 1.08× 加速，LM-eval 从 67.58 降至 66.37
- **性能 trade-off**（Tab. 5）：Mixtral 8x7B Instruct + r=4 pruning + skipping → 1.33× speedup, LM-eval 62.33（原模型 69.98 的 ~89%）
- **领域特化**（Tab. 8）：数学任务中 skipping 导致稍大性能下降（β 用 MATH 校准），但 r=6 pruning + skipping 仍优于 r=4 pruning alone
- **与动态剪枝的区别**：MC-MoE 的 ODP 也使用 w₁/w₀ < μ 跳过，但额外引入 token protection（保护 top 2% 重要 token）。Dynamic Expert Skipping 更简洁，无保护机制，与 permanent pruning 组合是核心贡献

涉及论文标题：
- Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language Models
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

## Local Routing Consistency in MoE（MoE 中的局部路由一致性）

术语是什么？
Local Routing Consistency（局部路由一致性）是 MoE 模型中连续 token 倾向于激活相同或相似 experts 的程度属性。当 MoE 模型具有高局部路由一致性时，在一定长度的 token segment 内 router 选择的 expert 集合保持相对稳定。该属性由论文 "Not All Models Suit Expert Offloading" (ICLR 2026) 首次系统定义。高局部路由一致性使 expert offloading 系统的 GPU expert cache 获得更高命中率，减少从 CPU 加载 expert 的慢路径。大多数模型在短 segment（m=4）内展示相似的短期一致性，但长期（m≥16）差异显著——仅 Group 1 模型（LLaMA-MoE-v2, OLMoE 等）维持高 SRP (>0.5)。

从算法pipeline角度拆解术语：
```
# 局部路由一致性分析 pipeline
# Phase 1: 收集路由决策
for each sequence T, layer l, token t:
    A[T][l][t] = Router_l.top_k(hidden_state)  # 激活专家索引

# Phase 2: 统计 per-segment 专家激活频率
for each expert e, segment length m, start position p:
    f[e,p,m] = Σ A[T][p:p+m] where e is activated

# Phase 3: SRP 计算 (Eq.4,6 — 专家固有属性，无参数)
for α in [0,m]:
    F1[α] = 2*Σ_{f>=α} f / Σ[m·I(f>=α) + f]
SRP = max_α F1[α]

# Phase 4: SCH 计算 (带 cache 容量约束)
for scenario with cache ratio ρ:
    simulate oracle cache evicting least-future-used experts
    SCH = hit_count / total_accesses
```

论文关键发现：(1) 局部路由一致性与局部负载均衡存在 trade-off——高一致性模型路由更集中（expert activation SD 大），但全局负载均衡可通过 domain-specialized experts 与局部一致性共存；(2) Shared experts 通过减小 expert combination space 降低局部一致性；(3) Domain-specialized experts 对局部一致性的贡献大于 vocabulary-specialized ones；(4) Cache size ≈ 2× active experts 在大多数模型上取得最佳性价比。

术语一般如何实现？如何使用？
部署前评估：对候选 MoE 模型调用论文代码 (https://github.com/ljcleo/moe-lrc) 计算 SRP/SCH，选择高一致性模型部署到 memory-constrained 设备。架构设计指导：避免 shared experts、增大 expert combination space（更多 total experts 和适当 k）、重视 domain specialization 的训练。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

## Segment Routing Best Performance (SRP / 分段路由最优性能)

术语是什么？
SRP 是衡量 MoE 模型中 expert 或 expert group 局部路由一致性的无参数量化指标 (ICLR 2026)。定义：segment-based estimator R_e^m 对长度 m 的 segment 统一预测（全激活或不激活），在所有可能 segment 上的最大 F1 分数。数学证明 F1 最大化当且仅当 estimator 对所有 f ≥ α_e^m 的 segment 给出激活预测，α_e^m ∈ [0,m] 是 expert e 和 m 的唯一函数——因此 SRP 是 expert 的固有属性，与具体路由方法无关。辅助指标 ρ̂（segment routing size ratio）= 最佳预测所需激活 expert 数 / 原始激活数，越小说明局部一致性越强。

从算法pipeline角度拆解术语：
```
# Single-expert SRP (Eq.4)
for α in [0..m]:
    TP = Σ f  for {f >= α};  FP = Σ (m-f) for {f >= α}
    FN = Σ f  for {f < α}
    F1 = 2*TP / (2*TP + FP + FN)
SRP(e, m) = max_α F1

# Expert group SRP (Eq.5-6, 联合优化所有 expert)
# 决策空间: ∀e ∈ E, 对 f[e,T,p,m] >= α_e^m 的 segment 激活 expert e
# α_e^m 由 group E 和 m 联合决定
```

术语一般如何实现？如何使用？
论文实现：收集 20 个 MoE 模型在 22,528 样本上的路由决策，统计 per-expert per-segment 激活频率，搜索 α 计算 SRP。由于不同位置 segment 的 SRP 几乎恒定 (Appendix E.2)，所有位置统一计算。代码开源 https://github.com/ljcleo/moe-lrc。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

## Segment Cache Best Hit Rate (SCH / 分段缓存最优命中率)

术语是什么？
SCH 是衡量 MoE 模型在带 cache size 限制 (ρ = cache_size / active_experts) 的 expert offloading 场景下的理论最大 hit rate (ICLR 2026)。模拟 oracle segment cache：缓存大小为 ρ·k，驱逐未来 m 个 token 间激活次数最少的 expert。SCH 桥接 SRP 与实际 offloading：无容量限制时用 F1 (SRP)，有容量限制时用 hit rate (SCH)。实验表明 SCH 与 LRU/LFU hit rate 高度正相关 (m=64 时 r > 93%)，且在中等 ρ 下接近 Belady 最优 (ρ=2 时可达 90.55% of optimal)。

从算法pipeline角度拆解术语：
```
cache = empty, cache_size = ρ*k
for each segment [p, p+m):
    for token t in segment:
        demanded = top_k(router(t))
        missed = demanded \ cache
        if missed:
            # oracle: evict experts least activated in remaining segment
            future_counts = count_activations(t+1, p+m)
            evict = bottom_k(cache, future_counts, |missed|)
            cache = (cache \ evict) ∪ missed
        # record hit/miss per expert
SCH = hits / total_accesses
```

术语一般如何实现？如何使用？
与 SRP 使用相同路由决策数据。SCH 用于确定模型的最佳 GPU cache 大小——ρ=2 是大多数模型的 sweet spot（此后收益递减）。代码开源 https://github.com/ljcleo/moe-lrc。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

## Expert Combination Space in MoE（MoE 中的专家组合空间）

术语是什么？
Expert Combination Space 指 MoE 模型中 router 可选择的 expert 组合总数 C(E_routed, k_routed) (ICLR 2026)。与局部路由一致性正相关：组合空间越大，router 在相邻 token 间做局部微调的灵活性越高。Shared experts 缩减组合空间的双重机制：(1) 占用激活 quota（k_routed = k - shared）；(2) bypass effect——更多信息由 shared expert 处理，routed expert 重要性降低。

从算法pipeline角度拆解术语：
```
无 shared: C(64, 8) ≈ 4.4×10^9
2 shared (DeepSeekMoE): C(62, 6) ≈ 6.1×10^7 (缩小 ~72×)
TOY 实验验证: ActMore (C(64,16)=4.9e14) 提升 SRP; ActFewer (C(64,2)=2016) 降低 SRP
```

术语一般如何实现？如何使用？
架构设计指导：使用更多 total experts + 适中 k（不超过半数），避免 shared experts 过度占用配额。虽然此因素对 SRP 影响弱于 load balance 和 shared experts，但确实存在正相关。论文代码 https://github.com/ljcleo/moe-lrc。

涉及论文标题：
- Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models

---

## Oracle-Space-Based Routing (Oracle-MoE 路由)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Oracle-Space-Based Routing 是 Oracle-MoE (ICML 2025) 提出的 MoE routing 机制，用基于高层语义的"Oracle Space"替代传统的 token embedding 空间做专家路由。其核心洞察：连续 token 具有语义局部性（semantic locality），但 token-level embedding 被 token-identity 特征主导（如"the" vs "cat" 的 token ID 差异淹没语义相似性），导致相邻 token 被路由到不同专家，产生频繁的 expert swapping。Oracle-MoE 利用注意力分数（Q·K^T 内积）挖掘高层语义相关性：attn 分数高的 token 共享相似高层语义 → 归入同一语义组（Semantic Group） → 计算组嵌入作为 token 的语义表示 → 在 Oracle Space 上用 K-means 聚类（k=专家数）→ 每个聚类中心对应一个专家 → 同一组内所有 token 路由到同一专家。由于语义组嵌入比 token embedding 方差低得多（理论证明 Var(z_S) = (Σ_s + Σ_j)/n < Var(t_t)），连续 token 路由变化极小，CSD_oracle << CSD_token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Oracle-Space-Based Routing 的完整 pipeline（5 阶段）：

```
# ===== Stage 1: Warm-up Training =====
# 对 token-level MoE 进行短期预训练，获取合理的 token embeddings

# ===== Stage 2: Oracle Space Initialization =====
oracle_space = []
for each sampled_sequence in N_samples:  # N=8192
    # 2a: 计算注意力分数矩阵（使用 KV cache 中的 Q,K）
    A = Q @ K.T / sqrt(d)  # [T, T] lower-triangular
    A = softmax(A, dim=-1)

    # 2b: 贪心语义组划分 (Minimum Clique Cover on DAG, polynomial-time greedy)
    semantic_groups = []
    for t in range(T):
        merged = False
        for group in reversed(semantic_groups):
            if all(A[t][k] > epsilon for k in group):
                group.append(t)
                merged = True
                break
        if not merged:
            semantic_groups.append([t])

    # 2c: 计算语义组嵌入（组内 token embedding 均值）
    for group in semantic_groups:
        z_S = mean(token_embeddings[group])  # z_S ∈ R^d
        oracle_space.append(z_S)

# 2d: SVD 降维（保留 top-r 奇异值，提高计算效率）
U, S, Vt = SVD(oracle_space)
W_svd = Vt[:r, :]  # 降维变换矩阵 r << d

# 2e: K-means 聚类（k = num_experts）
reduced_embeddings = [W_svd @ z for z in oracle_space]
cluster_centers = KMeans(reduced_embeddings, k=num_experts)

# ===== Stage 3: Training/Prefill Routing =====
def oracle_moe_forward(token_embeddings, attention_scores):
    groups = partition_semantic_groups(attention_scores, epsilon)
    for group in groups:
        z_S = mean(token_embeddings[group])
        z_reduced = W_svd @ z_S
        expert_id = argmin(||z_reduced - c_k|| for k in range(num_experts))
        for token in group:
            output += expert_ffn[expert_id](token_embeddings[token])
    return output

# ===== Stage 4: Decode Routing =====
def oracle_moe_decode(new_token_embedding, kv_cache):
    q = W_Q @ new_token_embedding
    attn_scores = [q @ k_i / sqrt(d) for k_i in kv_cache.K]
    for group in existing_groups:
        if all(attn_scores[0][k] > epsilon for k in group):
            group.append(new_token_idx)
            z_S = mean(token_embeddings[group])
            z_reduced = W_svd @ z_S
            return argmin(||z_reduced - c_k||)
    new_group = [new_token_idx]
    z_reduced = W_svd @ new_token_embedding
    return argmin(||z_reduced - c_k||)

# ===== Stage 5: Expert Prediction Optimization =====
for layer in range(1, num_layers):
    pred_expert[layer] = expert_predictor[layer](hidden_states[0])
# 预加载预测的专家以隐藏 I/O 延迟（准确率 85%-95%，减少 10%-15% latency）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 计算开销：路由计算从 token-level 的 W_g · t_t（矩阵乘法, 1e-4s）变为降维后的欧氏距离计算 ||W_svd · z_S - c_k||（2.5e-4s），相比单次 forward-backward pass（3.5s）可忽略。语义组划分利用 KV cache（decode 时已存在），无额外内存开销。
- 使用场景：(1) 内存受限的边缘设备（手机、Jetson 等）上部署 MoE LLM 推理；(2) 单用户 batch_size=1 场景（连续 token 语义局部性强）；(3) 需要减少 GPU memory footprint 但不可接受 token-level MoE 的高 swapping latency 的场景。
- 限制：(1) 需要 warm-up 阶段建立 Oracle Space（每层聚类 ~4 min，相对 tens of hours 预训练可忽略）；(2) 语义局部性假设在极端随机 token 序列中减弱（但实验显示即使跨数据集拼接的 diverse data，Oracle-MoE 仍每 100 token 仅换 12.2 次 vs Switch 90.54 次）；(3) 当前仅验证于 GPT-2 架构 MoE。
- 开源：论文未明确说明（ICML 2025 proceedings 无 GitHub 链接）。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

---

## Semantic Group (语义组, in MoE Routing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic Group 是 Oracle-MoE 中基于注意力分数的因果图（DAG）定义的一组最大化互连 token 集合，用于捕捉 token 序列中的高层语义局部性。定义：将 token 序列建模为有向无环图 G = (V, E)，V 包含所有 token，E 由注意力分数矩阵 A = [a_ij] 的 lower-triangular 部分加权（a_ij 存在仅当 i > j，因果注意力约束）。若 token 组 S = {t_k1, ..., t_km}（k1 < ... < km）满足：(1) 所有 i > j 均有 a_ij > ε；(2) 不存在包含 S 的真超集也满足条件(1)，则 S 为一个语义组。这本质上是 DAG 上的 Minimum Clique Cover 问题的重构。由于注意力矩阵具有块结构（block structure），可用多项式时间贪心算法求解（从左到右扫描 token，尝试合并到已有组中）。

从算法pipeline角度拆解术语：

语义组划分贪心算法：
```
def partition_semantic_groups(attention_matrix_A, epsilon, seq_len):
    """
    A: [T, T] causal attention score matrix (lower-triangular)
    epsilon: 注意力分数阈值，决定"语义相关"的最小分数
    """
    groups = []
    for t in range(seq_len):
        merged = False
        for group in reversed(groups):
            if all(A[t][k] > epsilon for k in group):
                group.append(t)
                merged = True
                break
        if not merged:
            groups.append([t])
    return groups
```

关键性质：(1) 同一语义组内的 token 共享相似的高层语义（由 attention Q·K^T 内积保证）；(2) 1024 token 的序列通常仅产生 < 5 个语义组；(3) 同一序列/用户交互的语义组在 Oracle Space 中倾向于属于同一 K-means 聚类；(4) 组内平均操作将 token-identity 方差从 Var(t_t) 降至 (Σ_s + Σ_j)/n。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 依赖因果注意力分数（decode 时利用 KV cache 可自然获取）。
- 阈值 ε 控制语义组粒度，论文未明确给出具体值。
- 复杂度 O(T × G)，G 为组数（通常 < 5），近乎线性。
- 仅适用于 causal attention 的 auto-regressive 模型。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

---

## Consecutive Semantic Difference (CSD, 连续语义差异)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSD 是 Oracle-MoE 提出的衡量 MoE routing 中连续 token 之间 expert 选择变化程度的指标。定义：CSD = Σ_{t=2}^{T} Δe_t，其中 Δe_t = |e_t \ e_{t-1}| 为连续 token 的激活专家集合的对称差。在固定硬件和 swapping 算法下，总延迟 L_total = Σ(L_compute + l_swap · Δe_t)，因此 CSD 直接决定总延迟。

从算法pipeline角度拆解术语：

Token-level MoE：CSD_token ≈ Σ C(W_g, k) ||t_t - t_{t-1}||，由于 token embedding 受 token-identity 主导、方差大，CSD_token 高。
Oracle-MoE：CSD_oracle ≈ Σ ||z_{S(t)} - z_{S(t-1)}||，语义组嵌入在 Oracle Space 中平滑变化，CSD_oracle 低。

Theorem 1: 以高概率 CSD_token > CSD_oracle。实验验证：Oracle-MoE 激活不一致性 4-6 per 100 tokens，Switch Transformer 为 53-82。

术语一般如何实现？如何使用？
- CSD 用作 memory-constrained MoE 推理中 latency 的代理指标，无需实际测量 I/O 即可评估 routing 策略的 swapping 友好程度。
- 将 latency 优化转化为语义空间连续性问题，可从理论上分析和比较不同 routing 策略。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

---

## Semantic Group Embedding (语义组嵌入)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic Group Embedding 是语义组内所有 token embedding 的均值向量：z_S = (1/|S|) Σ t_j。通过将 token embedding 分解为 t_i = s_i + u_i（s_i: 高层语义, u_i: token-identity）后，组内平均压制 token-identity 噪声 u_i 同时保留高层语义 s_i。理论保证：Var(z_S) = (Σ_s + Σ_j)/n < Var(t_t)，n = |S|。类似 GNN 的均值聚合和句子元嵌入。

从算法pipeline角度拆解术语：

```
# 输入: token embeddings T = [t_1, ..., t_n] for semantic group S
z_S = (1/n) * Σ_{i=1}^{n} t_i           # d 维均值
z_S_reduced = W_svd @ z_S               # 可选 SVD 降维到 r 维
expert_id = argmin_k ||z_S_reduced - c_k||  # 路由到最近聚类中心
```

方差分析：Var(z_S) = Var((1/n)Σ s_i) + Var((1/n)Σ u_i) = Σ_s/n + Σ_j/n < Var(t_t)，因为 1/n < 1 for n > 1。

术语一般如何实现？如何使用？
- Streaming 增量更新：z_S_new = (|S|*z_S_old + t_new) / (|S|+1)，O(d)。
- SVD 降维加速 K-means 聚类和 routing 时的距离计算（r << d）。
- 与 token embedding 的关键区别：语义组嵌入消除了单个 token 的 identity 噪声，保留语义上下文的整体特征。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

## Expert Co-activation Affinity (专家共激活亲和性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Co-activation Affinity 是指在 MoE 模型的同一层中，某些 expert pairs 被同时激活的概率远超随机选择理论值的现象。在 top-k routing（k>1）的 MoE 模型中，每层有 n 个 expert，每个 token 选择 k 个 expert。如果 expert 选择完全随机，任意 expert pair (i,j) 被同时选中的理论概率为 $p = \frac{k(k-1)}{n(n-1)}$。但实际 profiling 发现，某些 expert pairs 的共激活频率是理论值的 20-40 倍，形成显著的"共激活亲和性"。这种亲和性在不同模型间表现不同：DeepSeek V3 的 heatmap 中频繁共激活的 pairs 形成 bright squares（受其 routing restriction 影响——token 仅路由到相邻 node），而 Qwen3 表现出更分散的 bright dots 模式。量化分析显示 top 10% 的 expert pairs 占据了 60-80% 的总激活量。

从算法pipeline角度拆解术语：
Expert Co-activation Affinity 直接影响 MoE 层计算的并行度和负载分布。在 EP 场景下，如果两个高频共激活的 expert 被分配到同一 GPU，则该 GPU 在该层将承受不成比例的计算负载（两个 expert 同时被大量 token 选中），而其他 GPU 可能空闲。反之，如果将共激活 expert pairs 分离到不同 GPU，可以最大化并行度，但引入跨 GPU 通信开销。

论文提出的 Expert-pair separation insight (Insight 5)：separate frequently co-activated expert pairs to maximize parallelism，但需要 trade-off communication costs。

专家共激活分析流程（基于论文 profiling methodology）：
```
输入: expert selection traces D (per-layer per-token expert IDs)
输出: co-activation heatmap H, top co-activated pairs

for each layer l:
    # 初始化 n×n 矩阵
    co_act_count = zeros(n, n)
    
    for each token t in traces[l]:
        selected_experts = traces[l][t]  # top-k expert IDs
        for each pair (i, j) in combinations(selected_experts, 2):
            co_act_count[i][j] += 1
            co_act_count[j][i] += 1  # symmetric
    
    # 归一化到理论随机概率
    total_tokens = len(traces[l])
    random_prob = k*(k-1) / (n*(n-1))
    H[l] = co_act_count / (total_tokens * random_prob)
    
    # H[l][i][j] > 1 表示高于随机期望的共激活
    # H[l][i][j] = 20-40 表示比随机高 20-40 倍
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Co-activation heatmap 通过离线 profiling 构建：使用 SGLang 等 serving framework 收集 expert selection traces（论文在 8×H100 上收集了 >24,000 requests，>150 GB JSON traces）。
- 在系统优化中，共激活 affinity 信息用于：(1) Expert placement——将高频共激活的 expert pairs 分配到不同 compute unit 以均衡负载（但需与通信开销 trade-off）；(2) Expert 复制策略——若共激活无法通过 separation 解决（通信开销过高），可在多个 unit 复制高频共激活 expert pair 中的专家。
- 注意：Llama 4 每层只选 1 个 expert（k=1），因此不存在 co-activation 关系。论文仅分析 DeepSeek V3 (top-8) 和 Qwen3 (top-8) 的共激活模式。

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

---

## Expert Selection Temporal Correlation (专家选择时间相关性)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Selection Temporal Correlation 是指 MoE 模型在推理过程中，expert 选择在不同时间尺度上表现出的可预测模式。论文通过 >24,000 requests 的 profiling 识别出三个时间尺度的相关性：(1) **Layer-level correlation (Ob1)**：相邻两层之间 expert 选择的条件概率分布——给定 layer N 选择了 expert i，layer N+1 选择 expert j 的概率 $P(e_j^{N+1} | e_i^N)$，top 20% 候选 expert 覆盖了 50-77% 的条件概率质量；(2) **Token-level correlation (Ob2)**：同一层相邻两个 token 之间 expert 选择的条件概率——高层（17, 43）出现明显的对角线模式（同一 expert 在相邻 token 被反复选中，即 temporal locality），而低层（1, 3）不明显；(3) **Prefill-decode-level correlation (Ob3)**：prefill 和 decode 阶段的 expert 选择模式高度相似——cross-layer 和 cross-token heatmap 形状相似（Spearman's ρ ≥ 0.7 for most layers），top-5 prefill experts 覆盖 ~60% 的 top-5 decode experts，top-20 覆盖 ~90%。

从算法pipeline角度拆解术语：
三种时间相关性对应不同的 reuse distance 和优化机会：

```
时间尺度层次:
Pattern        | Reuse Distance         | 优化目标          | Memory Tier
Layer-level    | 短（相邻层连续执行）     | LLC/prefetch      | 快速小容量
Token-level    | 长（遍历所有层后）       | DRAM cache        | 大容量
Prefill-decode | 跨阶段（不同机器可能分离）| 初始 placement    | 静态/半静态
```

论文用 Conditional CDF 量化相关性：
- 对 layer-level：$F(x) = P(\text{top } x\% \text{ candidates cover } \ge y\% \text{ of conditional probability})$
- 结果：top 20% next-layer candidates cover 50% (DeepSeek), 65% (Qwen3), 77% (Llama4), 56% (Kimi K2) 的条件概率
- 对 token-level：top 20% next-token candidates cover 47% (DeepSeek), 62% (Qwen3), 80% (Llama4), 53% (Kimi K2)

Llama 4 的相关性最强，DeepSeek V3 最弱——这与模型架构差异相关（Llama 4 在 MoE layers 之间插入 dense FFN layers）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- Temporal correlation 通过离线 profiling 建立：使用 SGLang 部署模型 → 收集每层每 token 的 expert selection → 构建 heatmap（条件概率矩阵）→ 存储为 lookup table。
- 在 serving 系统中应用：(1) **Cross-hierarchy memory management** (Insight 2): layer-level correlation 指导 LLC/快速 memory tier 的 prefetch，token-level correlation 指导 DRAM/大容量 tier 的 cache；(2) **Data-driven predictor**：用 cross-token heatmap 预测下一 token 的 expert 选择；(3) **Prefill-guided placement**：用 prefill traces 预测 decode 阶段的 expert 需求，指导初始 placement。
- 论文开源了所有 heatmap 和 traces：https://huggingface.co/datasets/core12345/MoE_expert_selection_trace

涉及论文标题：
- Orders in Chaos: Enhancing Large-Scale MoE LLM Serving with Data Movement Forecasting

## Conditional Computation (条件计算)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Conditional Computation 是一种神经网络计算范式，指对每个输入样本（token/example），仅激活网络的部分参数进行计算，而非激活全部参数。其核心思想是"按需计算"——不同输入使用不同的网络子结构，从而在增加模型容量（总参数数）的同时控制计算量增长。在传统 dense 模型中，所有参数对每个样本均被激活，计算量 ≈ O(#params)；Conditional Computation 通过路由机制将计算量降至 O(#activated_params)，使得 #params >> #activated_params，即"参数规模与计算量解耦"。该概念最早由 Bengio et al. (2013, 2015)、Davis & Arel (2013) 等在理论上提出，Shazeer et al. (2017) 首次在大规模深度神经网络中实现 >1000× 的容量提升。

从算法pipeline角度拆解术语：
Conditional Computation 在 MoE 中的实现流程（以 Shazeer et al. 2017 为例，LSTM+MoE 语言模型）：

```
# 模型结构: Embed -> LSTM1 -> MoE -> LSTM2 -> Softmax
# MoE 含 n 个 expert，每个 expert = FFN(1024 ReLU -> 512)

# 对每个位置 t:
# Step 1: 标准层 (全激活)
h_t = LSTM1(embed(x_t))          # 所有 token 经相同 LSTM1

# Step 2: Gate 路由 (条件分支)
gate_logits_t = h_t @ W_g        # [1, n]
noise_t = StandardNormal() * Softplus(h_t @ W_noise)
H_t = gate_logits_t + noise_t    # noisy logits
topk_vals, topk_idx = KeepTopK(H_t, k)  # 仅保留 k=4 个 expert
G_t = Softmax(topk_vals)         # [1, k] 稀疏 gate 权重

# Step 3: 条件计算 (仅 k 个 expert 执行)
for each selected expert i:
    E_i_out = W_out_i @ ReLU(W_in_i @ h_t)  # 仅被选中的 expert 执行
# 其余 n-k 个 expert 不参与计算 (条件不激活)

# Step 4: 加权合并
moe_out_t = sum(G_t[j] * E_selected[j]_out for j in range(k))
```

关键参数关系：
- 总参数 ≈ n × params_per_expert + params_dense
- 计算量 ≈ k × ops_per_expert + ops_dense (≈ O(k)，而非 O(n))
- 稀疏度 = 1 - k/n，越大模型的效果越明显

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现挑战：(1) **Shrinking Batch Problem** — 每个 expert 仅收到 kb/n 的 batch，n 增大时单个 expert 吞吐急剧下降。Shazeer 2017 的解决方案：混合数据并行+模型并行使 expert batch 放大 d 倍（d=设备数）；卷积式应用 (将所有时间步折叠进 batch)；增加总 batch size。(2) **Load Imbalance** — Gate 倾向塌缩到少数 expert（self-reinforcing），需辅助损失函数强制均衡。(3) **Network Bandwidth** — expert 输入/输出在网络间传输，需保证 compute-to-IO ratio 超过设备能力比。通过增大 expert hidden layer 提高该比值（如 1024 → 2048 → 8192）。
- 除 MoE 外，Conditional Computation 的其他形式包括 Early Exit / Dynamic Depth（根据输入难度在不同层提前输出），以及 token pruning/sparsity 技术。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

## Load Balancing Loss for MoE (MoE 负载均衡损失)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Load Balancing Loss 是 MoE 训练中用于防止 Gate 网络收敛到仅激活少数几个 expert（expert collapse / routing collapse）的辅助损失函数。没有负载均衡损失时，Gate 网络会形成 self-reinforcing 循环：被频繁选中的 expert 梯度更新更多 → 更早学会 → Gate 更倾向选它 → 负载进一步集中。Shazeer et al. (2017) 提出两级负载均衡损失：L_importance 和 L_load，分别控制 expert 的重要性分布和负载分布。两者作为辅助损失加入总 loss：Total Loss = CrossEntropy + L_importance + L_load。

从算法pipeline角度拆解术语：
两级负载均衡损失的计算过程（Shazeer et al. 2017）：

```
# 给定 batch X，n 个 expert，gate 输出 G(x) [1, n] (稀疏)

# === Level 1: Importance Loss ===
# Importance(X)[i] = Σ_{x∈X} G(x)_i  # expert i 的 batch-wise gate sum
Importance = sum(G(x) for x in X)    # [n]

# CV (Coefficient of Variation) = σ / μ
CV_importance = std(Importance) / mean(Importance)

# Importance Loss: 鼓励所有 expert 的重要性相等
L_importance = w_importance * CV_importance^2

# === Level 2: Load Loss ===
# 问题: Importance 均衡不代表每个 expert 接收的样本数均衡
# (一个 expert 可能收少量大权重样本，另一个收大量小权重样本)

# Smooth Load Estimator (利用 noise 的可微性):
# P(x,i) = 概率(G(x)_i > 0 | 重新采样 expert i 的 noise, 保持其他 noise 固定)
P(x,i) = Φ((clean_logits_i - kth_excluding(H,k,i)) / Softplus(noise_std_i))
# Φ = 标准正态 CDF

# Load(X)[i] = Σ_{x∈X} P(x,i)
Load = sum(P(x, i) for x in X)  # [n]

CV_load = std(Load) / mean(Load)
L_load = w_load * CV_load^2

# === 组合 ===
Total_Loss = CrossEntropy + L_importance + L_load
```

关键设计要点：
- **Noise 的双重作用**：(1) 训练中提供探索随机性，防止过早收敛；(2) 利用 noise 分布构造平滑可微的 Load(X) 估计器，使负载均衡可反向传播。
- **初始化**：W_g 和 W_noise 初始化为全零 → 初始状态每个 expert 被均匀选中（仅有 noise 驱动选择），避免训练初期的 OOM。
- **权重调参**：w=0.1/0.1 可达到良好平衡（Test PPL 35.6 vs 无 loss 的 39.8），过大值(1.0/1.0) 不进一步改善质量但可进一步降低最大 expert 负载（max/mean ratio 1.07 vs 1.47）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 后续变体：(1) Switch Transformer — L_aux = E·Σ(f_e·P_e)，其中 f_e = fraction of tokens routed to expert e, P_e = mean routing probability；(2) GShard — auxiliary loss = α·Σ(f_e - 1/E)²；(3) Z-loss (ST-MoE) — 加在 logits 上的正则化项。
- 替代/补充方案：Capacity Factor (CF) — 硬限制每个 expert 最多处理 CF×(total_tokens/n) 个 token，超出部分丢弃并由 residual connection 绕过。CF 与辅助 loss 常联合使用。
- 工程要点：importance loss 和 load loss 在各 GPU 上的梯度同步方式需适配并行策略——在 Expert Parallelism 下每个 GPU 只计算本地 expert 的 loss 分量。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer

## Hyper-Connections (HC)

术语是什么？

Hyper-Connections (HC) 是一种对标准残差连接的扩展范式。标准残差连接公式为 $\mathbf{x}_{l+1} = \mathbf{x}_l + \mathcal{F}(\mathbf{x}_l, \mathcal{W}_l)$，残差流宽度固定为 C（模型隐藏维度）。HC 将残差流宽度扩展 n 倍（称为 n-stream residual），输入变为 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$，引入三个可学习线性映射：
- $\mathcal{H}_l^{\text{pre}} \in \mathbb{R}^{1 \times n}$：将 n-stream 特征聚合为 C 维层输入
- $\mathcal{H}_l^{\text{post}} \in \mathbb{R}^{1 \times n}$：将层输出映射回 n-stream
- $\mathcal{H}_l^{\text{res}} \in \mathbb{R}^{n \times n}$：在残差流内混合 n 个 stream 之间的特征

完整前向：$\mathbf{x}_{l+1} = \mathcal{H}_l^{\text{res}} \mathbf{x}_l + \mathcal{H}_l^{\text{post}^\top} \mathcal{F}(\mathcal{H}_l^{\text{pre}} \mathbf{x}_l, \mathcal{W}_l)$。每个映射由输入依赖的动态映射（线性投影+tanh+gating factor）和全局静态映射（可学习 bias）组成。扩展率 n（如 4）远小于 C，FLOPs 开销可忽略。核心缺陷：$\mathcal{H}_l^{\text{res}}$ 无约束，跨层复合映射可能信号爆炸/消失（Amax Gain Magnitude 可达 ~3000 vs 理想值 1），导致训练不稳定。同时 n-stream 导致显存 I/O 和 pipeline 通信开销增大约 n 倍。

从算法pipeline角度拆解：

```
def HC_forward(x_l):  # x_l: (n, C)
    x_norm = RMSNorm(x_l)
    H_pre  = alpha_pre  * tanh(x_norm @ theta_pre)  + b_pre   # (1, n)
    H_post = alpha_post * tanh(x_norm @ theta_post) + b_post  # (1, n)
    H_res  = alpha_res  * tanh(x_norm @ theta_res)  + b_res   # (n, n)
    layer_in = H_pre @ x_l                          # (C,)
    layer_out = F(layer_in, W_l)                    # (C,)
    x_next = H_res @ x_l + H_post.T * layer_out     # (n, C)
    return x_next
```

术语一般如何实现？如何使用？

HC 插入 Transformer 每层（Attention 子层和 FFN 子层），n 通常取 2-4。参数总量约 $C \times (n^2 + 2n) + (n^2 + 2n)$ 每层。动态映射需读取完整 $\mathbf{x}_l$，n-stream 显存 I/O 约为标准残差的 $(5n+1)$ 倍读和 $(3n+1)$ 倍写。需配合 kernel fusion、recomputing 等系统优化才能实用部署。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---

## Manifold-Constrained Hyper-Connections (mHC)

术语是什么？

mHC 是 HC 的改进框架，将 HC 的 $\mathcal{H}_l^{\text{res}}$ 通过 Sinkhorn-Knopp 算法约束到 Birkhoff polytope（双随机矩阵流形），恢复 identity mapping 稳定性，同时保留 HC 的多流特征混合能力。三个关键约束：(1) $\mathcal{H}_l^{\text{res}}$ 双随机（行和=列和=1，元素 ≥ 0），使 $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 成为 n 个 stream 的凸组合，谱范数 ≤ 1 非膨胀，乘法封闭性保证跨层稳定；(2) $\mathcal{H}_l^{\text{pre}}$ 经 Sigmoid，$\mathcal{H}_l^{\text{post}}$ 经 $2\sigma(\cdot)$ 约束为非负，防止正负系数抵消；(3) n=1 时退化为标量 1，完全恢复标准残差连接。与 HC 相比，复合映射 Amax Gain Magnitude 从 ~3000 降至 ~1.6（降低 3 个数量级）。

从算法pipeline角度拆解：

```
def mHC_forward(x_l):  # x_l: (n, C)
    x_flat = flatten(x_l); x_norm = RMSNorm(x_flat)    # (1, nC)
    H_pre_raw  = alpha_pre  * (x_norm @ phi_pre)  + b_pre   # (1, n)
    H_post_raw = alpha_post * (x_norm @ phi_post) + b_post  # (1, n)
    H_res_raw  = alpha_res  * reshape(x_norm @ phi_res, (n, n)) + b_res  # (n, n)
    # Manifold projection (key difference from HC)
    H_pre  = sigmoid(H_pre_raw); H_post = 2 * sigmoid(H_post_raw)
    H_res  = SinkhornKnopp(H_res_raw, t_max=20)  # doubly stochastic
    layer_in = H_pre @ x_l; layer_out = F(layer_in, W_l)
    x_next = H_res @ x_l + H_post.T * layer_out  # (n, C)
    return x_next
```

术语一般如何实现？如何使用？

需配合大量系统优化：5 个融合 kernel、选择性重计算、DualPipe 通信重叠。n=4 时额外训练时间仅 6.7%。flatten 操作展平 $\mathbf{x}_l$ 为 $\vec{\mathbf{x}}_l \in \mathbb{R}^{1 \times nC}$ 保留完整上下文。gating factor $\alpha$ 初始化为 0.01。基于 DeepSeek-V3 MoE 架构验证（3B/9B/27B 参数），8 个下游 benchmark 全面超越 baseline 和 HC。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---

## Sinkhorn-Knopp Algorithm

术语是什么？

Sinkhorn-Knopp 算法将正矩阵迭代投影为双随机矩阵（行和=列和=1）。给定初始正矩阵 $\mathbf{M}^{(0)}$（通过 $\exp(\cdot)$ 保证正性），交替行归一化 $\mathcal{T}_r$ 和列归一化 $\mathcal{T}_c$：$\mathbf{M}^{(t)} = \mathcal{T}_r(\mathcal{T}_c(\mathbf{M}^{(t-1)}))$。当 $t \to \infty$ 时收敛到唯一双随机矩阵，形式为 $\mathbf{M} = \text{diag}(\mathbf{u}) \cdot \mathbf{M}^{(0)} \cdot \text{diag}(\mathbf{v})$。在 mHC 中 $t_{\text{max}} = 20$ 为实际近似值。

从算法pipeline角度拆解：

```
M = exp(H_res_raw)            # element-wise exp for positivity
for t in 1..20:
    M = M / sum(M, axis=1, keepdim=True)   # row normalize
    M = M / sum(M, axis=0, keepdim=True)   # col normalize
return M  # ~doubly stochastic, used as H_res
```

术语一般如何实现？如何使用？

广泛用于最优传输中求解熵正则化问题。在 mHC 中前向和反向均实现为单一 GPU kernel——反向在片上重计算整个迭代过程而非保存 20 次迭代的中间矩阵。20 次迭代已产生近双随机矩阵（行列和接近 1），复合映射 Amax Gain Magnitude 仅 ~1.6。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---

## Doubly Stochastic Matrix / Birkhoff Polytope

术语是什么？

双随机矩阵是 $n \times n$ 非负矩阵，满足每行之和 = 1 且每列之和 = 1。所有 $n \times n$ 双随机矩阵的集合构成 Birkhoff polytope $\mathcal{M}^{\text{res}}$。Birkhoff-von Neumann 定理表明其顶点恰好是所有 $n \times n$ 置换矩阵，因此任何双随机矩阵可表示为置换矩阵的凸组合。关键性质：(1) 谱范数 ≤ 1（非膨胀）；(2) 对矩阵乘法封闭（乘积仍为双随机）；(3) 作用于向量时保持均值和范数界限。

从算法pipeline角度拆解：

在 mHC 中 $\mathcal{H}_l^{\text{res}}$ 约束为双随机矩阵 → $\mathcal{H}_l^{\text{res}} \mathbf{x}_l$ 每个输出 stream 是 n 个输入 stream 的凸组合 → 信号均值全局保持。跨 L 层后复合映射 $\prod_{i=1}^{L} \mathcal{H}_{L-i}^{\text{res}}$ 仍为双随机 → 深层信号稳定。n=1 退化为标量 1，完全恢复 identity mapping。

术语一般如何实现？如何使用：

通过 Sinkhorn-Knopp 算法或 Bregman 投影进行约束优化。除 mHC 外，也广泛用于最优传输（耦合矩阵）、图匹配和 ranking 问题。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---

## Identity Mapping Property (in Residual Connections)

术语是什么？

Identity Mapping 是残差连接的设计原则：shallower layer 信号直接、不经修改地映射到 deeper layer。标准残差连接递归展开为 $\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$，$\mathbf{x}_l$ 项体现 identity mapping。前向保证浅层信号 norm 不因残差结构本身变化；反向保证梯度有直接路径 $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \supset \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}$，避免梯度消失（He et al., 2016b）。

从算法pipeline角度拆解：

HC 破坏了 identity mapping——递归展开中的 $\mathbf{x}_l$ 被 $(\prod \mathcal{H}_{L-i}^{\text{res}}) \mathbf{x}_l$ 替代，无约束的 $\mathcal{H}^{\text{res}}$ 乘积可能极大（~3000×）或极小，导致信号爆炸/消失。mHC 通过双随机约束恢复：乘积仍为双随机 + 谱范数 ≤ 1 + 凸组合保持均值。

术语一般如何实现？如何使用？

微架构实现方式：Pre-Norm Transformer（Layer Norm 在 sublayer 之前）、ReZero（零初始化残差分支）、mHC（流形约束投影）。核心目标是保证梯度能无阻碍回传。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---

## Residual Stream Expansion (n-stream Residual)

术语是什么？

将 Transformer 标准一维残差流 $\mathbf{x}_l \in \mathbb{R}^{C}$ 扩展为 n 个并行流 $\mathbf{x}_l \in \mathbb{R}^{n \times C}$。动机：残差流信息容量受限于 C，而 C 与 FLOPs 强相关——扩展 n 在不增加每层 FLOPs 前提下提升信息容量，提供独立于模型尺寸/数据量之外的第三条扩展路径。

从算法pipeline角度拆解：

```
# layer_in: aggregate n streams, layer_out: standard F, x_next: update with mixing
layer_in = sum_i(H_pre[i] * x_l[i, :])            # (C,)  n→1
layer_out = F(layer_in, W_l)                       # (C,)  standard
x_next = H_res @ x_l + H_post.T * layer_out        # (n, C)  1→n+ mix
```

在 mHC 中 $\mathcal{H}^{\text{res}}$ 双随机 → 每个新 stream 是旧 stream 的凸组合（$\sum_j H^{\text{res}}_{ij} = 1, H^{\text{res}}_{ij} \geq 0$），在保持信号均值的同时实现信息交换。

术语一般如何实现？如何使用？

n 通常取 2-4。系统开销：显存 I/O 增加约 n 倍、pipeline 通信增加 n 倍、中间激活增加 n 倍——需通过 kernel fusion、recomputing 和通信重叠缓解。mHC 中 n=4 时额外时间开销仅 6.7%。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

## Hierarchical Mixture of Experts (层次化混合专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Hierarchical MoE 是 Shazeer et al. (2017) 提出的一种两级 MoE 结构，用于在 expert 总数极大时（数千至数万）降低计算 branching factor。在 flat MoE 中，Gate 网络为每个 token 从 n 个 expert 中选择 k 个，当 n 很大时 (如数万)，Gate 网络的计算量 (x@W_g: [d, n]) 本身成为一个瓶颈。Hierarchical MoE 将选择分为两级：第一级主 Gate (G_primary) 从 a 个 group 中选择 k1 个，第二级 Gate (G_i) 在每个 group 内的 b 个 expert 中选择 k2 个。总 expert 数 n = a × b。论文将第一级 branching factor 设为 GPU 数量，使次级 expert 无需跨设备通信。

从算法pipeline角度拆解术语：
Hierarchical MoE 的计算流程：

```
# 参数: a 个 group (第一级, 对齐 GPU 数), 每组 b 个 expert (第二级)
# 超参数: k1 (主 Gate 选的 group 数), k2 (次级 Gate 每组内选的 expert 数)

# Gate 1: 主 Gate
logits_primary = x @ W_g_primary        # [1, a]
H_primary = logits_primary + noise_1 * StandardNormal()
topk_vals_1, topk_idx_1 = KeepTopK(H_primary, k1)
G_primary = Softmax(topk_vals_1)        # [1, k1]

# Gate 2: 次级 Gate (在对应 GPU 上本地执行, 无跨设备通信!)
for i in topk_idx_1:
    logits_secondary = x @ W_g_secondary_i  # [1, b]
    H_secondary = logits_secondary + noise_2 * StandardNormal()
    topk_vals_2, topk_idx_2 = KeepTopK(H_secondary, k2)
    G_secondary_i = Softmax(topk_vals_2)    # [1, k2]
    
    for j in topk_idx_2:
        expert_out_{i,j} = Expert_{i,j}(x)  # 本地 FFN 计算

# 合并:
output = sum(G_primary[i] * G_secondary_i[j] * Expert_{i,j}(x) 
             for i in topk_idx_1 for j in topk_idx_2[i])
# 总激活 expert = k1 × k2 (如 2×2=4)
```

与 flat MoE 的关键区别：
- **Branching Factor**: flat 需 look up n 个 logits → Hierarchical 仅需 a + b (a+b << n)
- **通信优势**: 次级 expert 全在同一 GPU → 无跨设备通信
- **负载均衡扩展**:
  - Importance_H(X)_{i,j} = Σ_x G_primary(x)_i · G_i(x)_j
  - Load_H(X)_{i,j} = Load_primary(X)_i · Load_i(X^{(i)})_j / |X^{(i)}|

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 第一级 branching factor = GPU 数是有意设计——将 hierarchical 结构的物理意义与分布式拓扑绑定。
- 论文实验配置 (100B Word Corpus)：256×32, 1024×32, 4096×64, 16384×128, 65536×256, 131072×256 expert×branching factor。第一级 k1=2，第二级 k2=2。
- 论文指出"未发现需要更深层级"——两级已足够。
- 后续 MoE 模型（GShard, Switch Transformer, Mixtral）多采用 flat MoE，主要因现代 GPU 的 compute/memory ratio 更高且 flat 结构更简单，但 hierarchical 的设计思路影响了后续 Expert Parallelism 的拓扑感知设计。

涉及论文标题：
- Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer
## Post-Training Quantization (PTQ / 后训练量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Post-Training Quantization (PTQ) 是一种在模型训练完成后对权重（有时也包括激活）进行低精度压缩的方法，无需重新训练或微调。典型流程包括：（1）用少量校准数据（calibration set）前向传播模型，收集各层的激活分布或权重统计信息；（2）基于这些统计信息计算量化参数（scale s 和 zero-point z）；（3）将 FP16/FP32 权重量化到 INT4/INT8 等低精度格式。PTQ 的核心优势是快速、不需大量计算，适合部署场景。

与 Quantization-Aware Training (QAT) 的区别：PTQ 不需要训练，校准数据仅需几百到几千条序列，量化过程只需一次前向传播，总耗时通常在分钟级；QAT 在训练时模拟量化噪声并通过反向传播调整权重，需要完整训练 pipeline 和大规模数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以 per-channel 对称均匀量化的 PTQ 为例：

```
# 输入：FP16 权重矩阵 W ∈ R^{o×c}，校准数据 X ∈ R^{b×c}
# 输出：量化权重 W_q ∈ Z^{o×c}，scale s ∈ R^o

# Step 1: 确定量化参数 (per-channel)
for i in range(o):
    s[i] = max(abs(W[i,:])) / (2^{bit-1} - 1)  # symmetric, max range

# Step 2: 量化
W_q = clamp(round(W / s), q_min, q_max)  # q_min = -2^{bit-1}, q_max = 2^{bit-1}-1

# Step 3: 反量化（推理时）
W_hat = W_q * s  # 近似原始权重

# Step 4: 推理计算
output = X @ W_hat  # 或使用 INT4/FP16 dequant kernel
```

在 GPTQ 中，量化不是独立逐行完成的，而是逐列量化并使用 Hessian 补偿误差：H = X X^T，对每列 j，量化 W[:,j] → 计算误差 ΔW[:,j] = W_q[:,j] * s - W[:,j] → 用 H^{-1} 将误差按比例补偿到剩余列的权重上。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

主流实现：
- **GPTQ (Frantar et al., 2022)**：基于 OBQ 的 Hessian 误差补偿方法，每次量化一列后用 Hessian 逆补偿剩余列，支持 4-bit/3-bit 量化。开源：https://github.com/IST-DASLab/gptq
- **AWQ (Lin et al., 2023)**：利用激活分布选择平滑系数和剪枝权重，量化损失为 L = ||WX - W_hat X||_F^2。开源：https://github.com/mit-han-lab/llm-awq
- **SmoothQuant (Xiao et al., 2022)**：通过数学等效变换将量化难度从 activation 迁移到 weight，支持 W8A8 量化。
- **QuaRot (Ashkboos et al., 2024)**：使用 Hadamard 变换消除 outlier 实现免旋转的量化。

PTQ 在 MoE 模型上的挑战：由于每个 expert 只处理部分 token，校准集分布不均会导致部分 expert 校准不足（inter-expert imbalance）；同时 MoE 的 gating coefficient 使不同 token 对同一 expert 具有不同重要程度（intra-expert imbalance），这些因素是 MoEQuant 论文的核心动机。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## Per-channel Symmetric Uniform Quantization (逐通道对称均匀量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Per-channel Symmetric Uniform Quantization 是一种对权重矩阵按输出通道（行）独立计算量化参数的对称均匀量化方法。对称（symmetric）意味着 zero-point = 0，量化范围关于原点对称；均匀（uniform）意味着量化步长恒定；per-channel 意味着每行（输出通道）有独立的 scale 因子。

数学表达：对权重矩阵 W ∈ R^{o×c}，每行 i 的 scale s_i = max(|W[i,:]|) / (2^{bit-1} - 1)，量化后 W_q[i,j] = clamp(round(W[i,j] / s_i), q_min, q_max)，反量化 W_hat[i,j] = W_q[i,j] * s_i。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Per-channel Symmetric Uniform Quantization (4-bit)
# W: FP16 weight matrix of shape [out_features, in_features]
bit = 4
q_min = -2**(bit-1)      # -8
q_max = 2**(bit-1) - 1   # 7

scales = []
for row in W:
    max_abs = max(abs(row))
    scale = max_abs / q_max  # 或 max_abs / (2^{bit-1}-1)
    scales.append(scale)
    
W_q = clamp(round(W / scales.reshape(-1,1)), q_min, q_max)
# W_q ∈ Z^{o×c}, scales ∈ R^o

# Dequantization at runtime:
# W_hat[i,:] = W_q[i,:] * scales[i]
# Then compute: output = input @ W_hat^T (or fused dequant+matmul)
```

相比 per-tensor 量化（整个矩阵一个 scale），per-channel 量化为每行独立选择 scale，更好地适应不同输出通道的权重分布差异；相比 group quantization（每 32/128 个元素一个 scale），per-channel 的粒度更粗但内存开销更小。论文 MoEQuant 采用 per-channel 对称均匀量化（见 Equation 5-6）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPTQ、AWQ 等主流 LLM PTQ 方法默认使用 per-channel 对称均匀量化。在 GPU 推理时，dequantization 由 GEMM kernel 内的向量指令完成：每个 warp/block 加载对应的 scale 值，在 INT4→FP16 dequant 后执行 FP16 matmul。llama.cpp 的 GGUF 格式中，Q4_0/Q4_1/Q5_0 等为 group-wise 量化（比 per-channel 更精细），Q8_0 为 per-channel 对称量化。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## Hessian-based Error Compensation (基于Hessian的量化误差补偿)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hessian-based Error Compensation 是 GPTQ 的核心机制，源自 Optimal Brain Quantization (OBQ，基于 LeCun 的 Optimal Brain Damage)。基本思想：量化一个权重列 w_j 后产生的 output error Δy = (w_j_hat - w_j) * X[j,:]，该误差可通过对剩余未量化列按 Hessian 逆矩阵 H^{-1} 的比例进行补偿来消除，从而最小化总体输出误差。

Hessian 矩阵 H = X X^T ∈ R^{c×c} 编码了输入激活的二阶统计信息（c 是输入通道数）。H^{-1} 的 (j,k) 元素表示：对第 j 列的量化误差应该以多大比例传导到第 k 列的权值补偿。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GPTQ Algorithm (simplified)
# W: weight matrix ∈ R^{o×c}, X: calibration input ∈ R^{b×c}
# H = X^T X ∈ R^{c×c}  (Hessian)

H = X.T @ X  # 或 X @ X.T，取决于定义
H_inv = inverse(H)  # 实际使用 Cholesky 分解以提高稳定性

W_q = zeros_like(W)  # 量化后的权重
E = zeros(o, b)      # 累积误差

for j in range(c):  # 逐列量化
    # 量化第 j 列
    for i in range(o):
        W_q[i,j] = clamp(round(W[i,j] / s[i]), q_min, q_max) * s[i]
    
    # 计算量化误差（输出空间）
    err = (W_q[:,j] - W[:,j]).reshape(o, 1)  # [o, 1]
    
    # 用 Hessian 逆补偿剩余列
    for k in range(j+1, c):
        delta = H_inv[j,k] / H_inv[j,j]
        W[:,k] -= err * delta  # 补偿到权重上
    
    # 更新 Hessian 逆
    # 移除第 j 行/列后重新计算逆（实际用 Cholesky 更新更高效）
```

MoEQuant 的 AGQ 改进：传统 Hessian 对所有 token 等权，而 AGQ 将 gating coefficient c_i 引入 Hessian 计算：H = (X ⊙ √c)(X ⊙ √c)^T，使高亲和力 token 对 Hessian 的贡献更大，从而在误差补偿时更准确地保护关键 token 的表达质量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPTQ (Frantar et al., 2022) 是这一方法的标准实现。其关键优化包括：（1）对 H 做 Cholesky 分解以提高求逆稳定性；（2）对权重列做随机顺序或按 Hessian 对角线排序的贪心量化顺序；（3）对较大矩阵分 block 量化（每 block 128 列）以减少内存。Hessian 的规模为 c×c（c 为 hidden size，如 4096），因此内存开销较大但仍在可接受范围。在 MoE 场景下，每个 expert 的 FFN 矩阵可以独立计算 Hessian。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## Expert-Balanced Self-Sampling (EBSS / 专家均衡自采样)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Balanced Self-Sampling (EBSS) 是 MoEQuant 论文提出的专家均衡校准集生成方法。传统 PTQ 使用固定校准集（如 WikiText2），由于 gating 路由机制，不同 expert 收到的 token 数量极不均匀（长尾分布），导致欠载 expert 校准不足。EBSS 利用 LLM 自身的自采样能力生成校准数据，同时优化两项指标：（1）perplexity（低 PPL 保证与预训练分布一致）；（2）expert balance（σ，即各层 expert 使用频率的标准差）。

EBSS 将目标形式化为联合优化：D* = argmin_D {PPL(M, D) · exp(σ(M, D)/τ)}，其中 τ 控制专家均衡的重要性权重（论文取 τ=1.2）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# EBSS: Expert-Balanced Self-Sampling
# Input: MoE model M, beam width w, seq length n, temperature τ

# Initialize w empty sequences
beams = [""] * w
R = [0.0] * w  # cumulative log-probability per beam

for step in range(n):
    candidates = []
    for b in range(w):
        # Get next-token probability distribution from M
        probs = M.forward(beams[b])  # P(v|S^b) for all v in V
        expt_dist = M.get_expert_distribution()  # expert usage stats
        
        for v in top_k_by_prob(probs, top_k_prune):
            cum_log_prob = R[b] + log(probs[v])
            ppl = exp(-1/(step+2) * cum_log_prob)
            sigma = std(expert_usage_frequencies)
            score = -ppl_log + sigma / τ  # Eq. 13
            
            candidates.append((beams[b] + v, cum_log_prob, v, score))
    
    # Select top-w candidates by score
    candidates.sort(key=lambda x: x[3], reverse=True)
    beams = [c[0] for c in candidates[:w]]
    R = [c[1] for c in candidates[:w]]

D_star = beams  # w sequences, each n tokens
```

关键设计：（1）**Deferred Expert Imbalance Calculation**：候选 token v 不参与当前步的 expert balance 评估，而是使用当前序列 S 的已知 expert 分布（因为遍历词汇表计算每个 token 的 expert 分布开销过大），这实际上做的是 beam 级剪枝而非 token 级剪枝；（2）**Probability-Guided Path Pruning**：只对 vocab 中概率最高的部分 token 展开搜索，忽略低概率分支；（3）**复杂度**：从暴力搜索的 O(m^n) 降至 O(wn)，w=4 即可取得最优效果。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EBSS 在 MoEQuant 中用于生成替代 WikiText2/C4 的校准集。生成的校准数据有两个特性：（1）低 PPL——与模型预训练分布高度一致（甚至低于 WikiText2 和 C4 的 PPL）；（2）专家均衡——各 expert 分配到的 token 数量接近均匀分布。EBSS 生成的校准集可直接替代 GPTQ/AWQ 的原始校准数据输入，实现插件式集成。论文实验设定 w=4 branches，τ=1.2，sequence length=512（与 WikiText2 校准集相同），在 DeepSeek-MoE-16B 上 EBSS 单独使用带来约 1.3% 的平均分提升。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## Affinity-Guided Quantization (AGQ / 亲和力引导量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Affinity-Guided Quantization (AGQ) 是 MoEQuant 论文提出的方法，通过在 PTQ 的量化误差计算中引入 token-expert 亲和力（即 gating coefficient c_i），解决 MoE 中不同 token 对同一 expert 贡献不均的问题。在 MoE 架构中，经过 gating network 的 softmax 后，每个 token 对其路由到的 expert 有一个权重 c_i，该权重链式传导到 expert FFN 的所有线性层（see Equation 17）。

传统 PTQ 假设所有 token 同等重要（量化损失为 L = Σ_i ||W x_i - W_hat x_i||_F^2），而 AGQ 重定义为 L = Σ_i c_i · ||W x_i - W_hat x_i||_F^2，使高亲和力 token 的量化误差惩罚更大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# AGQ: Affinity-Guided Quantization for MoE
# X: input activations to expert e [b, c]  (b tokens routed to this expert)
# c: gating coefficients for these tokens [b]  (from softmax after routing)
# W: weight matrix of expert e [o, c]

# ---- For AWQ-style methods (error-based) ----
# Original AWQ loss:
# L = ||WX - W_hat X||_F^2  = sum_i ||W x_i - W_hat x_i||^2

# AGQ-modified loss (Equation 18):
L_agq = 0
for i in range(b):
    error = ||W @ x_i - W_hat @ x_i||^2  # output error for token i
    L_agq += c[i] * error  # weight by gating coefficient

# ---- For GPTQ-style methods (Hessian-based) ----
# Original Hessian: H = X @ X^T
# Shape: [c, c]

# AGQ-modified Hessian (Equation 19):
sqrt_c = sqrt(c)  # [b]
X_weighted = X * sqrt_c.reshape(-1, 1)  # [b, c], broadcast
H_agq = X_weighted.T @ X_weighted  # [c, c]
# Equivalent to: H_agq = X.T @ diag(c) @ X
```

物理含义：c_i 体现了 token i 与 expert e 的"相关度"。对 router 高度信任的 token（c_i 大），AGQ 赋予更大的量化误差权重，确保这些关键 token 的输出质量得到更好保护。反之，对 c_i 极小的 token，量化误差的影响也较小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AGQ 在 MoEQuant 框架中作为插件式模块实现，与 GPTQ 或 AWQ 无缝集成。对于 AWQ，AGQ 修改量化损失函数中的 per-token 权重；对于 GPTQ，AGQ 修改 Hessian 矩阵计算。论文实验表明 AGQ 单独使用带来约 2% 的平均分提升（DeepSeek-MoE-16B 上），而 EBSS + AGQ 结合使用提升约 2.6%。在 Mixtral-8x7B 上 AGQ 单独效果不如 baseline GPTQ，但 EBSS 和 AGQ 联合使用仍是最优配置，说明了两种方法的互补性。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## MoEQuant

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoEQuant 是 ICML 2025 发表的针对 Mixture-of-Experts (MoE) LLM 的后训练量化框架，由 Houmo AI 和东南大学联合提出。框架包含两个核心组件：EBSS（Expert-Balanced Self-Sampling）和 AGQ（Affinity-Guided Quantization），二者均为插件式设计，可与 GPTQ、AWQ 等现有 PTQ 方法无缝集成。

MoEQuant 解决的核心问题是：现有 LLM PTQ 方法（GPTQ、AWQ）在 MoE 模型上性能严重下降，原因是忽略了 MoE 架构的两个关键特性——(1) inter-expert imbalance：校准集中不同 expert 负载极不均衡；(2) intra-expert imbalance：不同 token 对同一 expert 的贡献权重不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MoEQuant Complete Pipeline
# Phase 1: EBSS — Generate balanced calibration set
calib_data = EBSS(
    model=M,           # MoE LLM (e.g., Qwen-MoE-14B)
    beam_width=4,      # w
    seq_length=512,    # n
    temperature=1.2    # τ
)
# → D*: expert-balanced calibration sequences

# Phase 2: AGQ — Affinity-guided quantization
for layer in M.moe_layers:
    X = forward_and_collect_activations(D*, layer)
    
    for expert in layer.experts:
        # Get tokens routed to this expert and their gating weights
        X_e, c_e = get_expert_inputs(X, expert)
        
        # AGQ-modified Hessian (for GPTQ)
        H = (X_e * sqrt(c_e)).T @ (X_e * sqrt(c_e))
        
        # Standard GPTQ with AGQ Hessian
        for weight_matrix in [W_gate, W_up, W_down]:
            GPTQ_columnwise_quantize(weight_matrix, H, bits=4)
            # Uses H for error compensation
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoEQuant 的实现基于 GPTQ 和 AWQ 官方仓库修改。关键配置：4-bit/3-bit per-channel 对称均匀量化，EBSS 参数 w=4 / τ=1.2，AGQ 与 GPTQ 集成时使用改进的 Hessian H = (X⊙c)X^T。硬件平台为 NVIDIA A6000 GPU。实验覆盖 Qwen-MoE-14B、DeepSeek-MoE-16B、Mixtral-8x7B 及其 instruction-tuned 变体。性能：4-bit MoEQuant++（基于 GPTQ）在三个模型上的平均分分别比 GPTQ 提升 0.59/1.00/2.16 分，3.2x 内存节省，1.2x 推理加速。代码发布于 https://anonymous.4open.science/r/MoEQuant-DDFD/README.md。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## GPTQ (GPT Post-Training Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GPTQ (Frantar et al., 2022) 是一种基于 Hessian 误差补偿的后训练权重量化方法，专门针对 GPT 系列 LLM 设计。核心原理源自 Optimal Brain Quantization (OBQ)：逐列量化权重矩阵，量化每个列后计算输出误差，使用 Hessian 逆矩阵将误差按比例补偿到剩余的未量化列上，从而保证最终输出的近似精度。

GPTQ 的关键技术：（1）H = 2X^T X 作为近似 Hessian 矩阵；（2）对 Hessian 做 Cholesky 分解提高数值稳定性；（3）在量化时对权重列按 Hessian 对角线大小排序（贪心顺序）；（4）对较大矩阵分 block 量化以控制内存；（5）所有列量化完后的总误差 = Σ_j (H^{-1})_{jj} · ε_j^2，被 H^{-1} 缩小。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GPTQ (simplified for one linear layer)
# W ∈ R^{o×c}, X ∈ R^{b×c}
H = 2 * X.T @ X  # [c, c], Hessian approximation
H_inv = inverse(H + lambda*eye(c))  # damped inverse for stability

W_q = zeros_like(W)
perm = argsort(diag(H_inv))  # order columns by sensitivity
W = W[:, perm]
H_inv = H_inv[perm][:, perm]

for j in range(c):
    # Quantize column j
    for i in range(o):
        s_i = max(abs(W[i,j])) / q_max
        W_q[i,j] = clamp(round(W[i,j] / s_i), q_min, q_max)
    
    # Error in weight space
    err = W_q[:,j] - W[:,j]  # [o]
    
    # Compensate remaining columns (k > j)
    for k in range(j+1, c):
        W[:,k] -= (H_inv[j,k] / H_inv[j,j]) * err

# Reorder back
inv_perm = argsort(perm)
W_q = W_q[:, inv_perm]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GPTQ 的开源实现在 https://github.com/IST-DASLab/gptq，支持 2/3/4/8-bit 量化，默认使用 128 个 2048-token 序列作为校准集，序列来自 C4 数据集。推理时配合 HuggingFace Transformers 或 vLLM 等框架的量化 kernel。在 MoEQuant 论文中，GPTQ 作为核心 baseline 使用，配合 QuaRot 的 Hadamard 变换预处理（消除权重 outlier）但不使用在线变换。GPTQ 在 MoE 模型上的问题：Hessian 未考虑 token-expert 亲和力（AGQ 的改进方向），且校准集未针对 MoE 架构做专家均衡（EBSS 的改进方向）。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance

## AWQ (Activation-aware Weight Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AWQ (Lin et al., 2023, MLSys 2024 Best Paper) 是一种基于激活感知的 LLM 权重量化方法。核心观察：LLM 权重中并非所有通道同等重要——对应较大激活幅度的权重通道对模型输出贡献更大。AWQ 通过分析校准数据的激活分布，识别重要权重通道，并对其施加更小的量化步长（等价于 per-channel scaling），同时将保护这些通道的代价以数学等效的方式转移到其他通道。

AWQ 的核心技技术：（1）计算激活的 per-channel 幅值（如 L2 norm 或 max），识别显著通道；（2）为显著通道搜索最优缩放因子 s（放大权重、缩小对应激活，或反之），目标是 min_{s} ||Q(W·diag(s)) · (diag(s)^{-1}·X) - WX||_F；（3）基于量化误差的 grid search 确定 s；（4）量化后不保存 s 的副本——s 被融合到量化前后的权重中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# AWQ (simplified)
# W ∈ R^{o×c}, X ∈ R^{b×c} (calibration activations)

# Step 1: Compute per-channel activation statistics
for j in range(c):
    act_norm[j] = mean_abs(X[:,j])  # average activation magnitude

# Step 2: Identify salient channels (top 1%)
salient_mask = act_norm > percentile(act_norm, 99)

# Step 3: Search optimal scaling factors
for j in range(c):
    if salient_mask[j]:
        # Grid search for optimal scale s
        best_s = 1.0; best_loss = inf
        for s in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            W_scaled = W * s  # scale weight
            X_scaled = X / s  # inverse scale activation
            loss = ||Q(W_scaled) @ X_scaled - W @ X||_F
            if loss < best_loss:
                best_s = s; best_loss = loss
        
        W[:,j] *= best_s  # apply scaling to weight
        # (activations are scaled inversely at runtime)

# Step 4: Standard per-channel quantization on scaled weights
W_q = per_channel_symmetric_quantize(W, bits=4)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

AWQ 开源实现在 https://github.com/mit-han-lab/llm-awq。默认使用 Pile 数据集中的 128 条序列作为校准集。推理时配合 TinyChat/tinychat 或 vLLM AWQ kernel 使用。在 MoEQuant 论文中，AWQ 作为第二个核心 baseline 使用。AWQ 也可被 AGQ 增强——原始 AWQ 对所有 token 等权计算损失，AGQ 引入 gating coefficient c_i 使量化损失变为 L = Σ_i c_i · ||W x_i - W_hat x_i||_F^2。

涉及论文标题：
- MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance
