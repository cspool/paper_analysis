## Unimportance Degree Score (Expert Importance Estimation for Mixed Precision Offloading)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Unimportance Degree Score 是 HOBBIT 提出的动态评估 MoE expert 重要性的方法，用于决定 cache-miss 时加载高精度还是低精度 expert。基于 MoE 输出公式 y = Σ G(x)_{e_i} E_{e_i}(x)，expert e_i 的贡献为 G(x)_{e_i}E_{e_i}(x)。由于无法在加载权重前计算 E_{e_i}(x)，使用 gating output ||G(x)_{e_i}|| 作为代理——实验验证二者 Pearson 相关系数为 0.99。将所有 top-K experts 按 ||G(x)_{e_i}|| 降序排列后，计算累积 unimportance score：s_{e_i} = Σ_{j=0}^{i-1} ||G(x)_{e_j}||（i>0），top-1 expert 始终 s=0。双阈值 T1/T2 划分三组：高精度（≤T1）、低精度（≤T2）、跳过（>T2）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert Importance Scoring Pipeline (per token, per MoE layer)
输入: x ∈ R^M (hidden state), W_gate ∈ R^{E×M} (gating weights), K=2

# Step 1: 标准 gating
gate_logits = W_gate @ x               # [E]
gate_probs = softmax(gate_logits)       # [E]
topk_vals, topk_ids = topk(gate_probs, k=K)  # top-2

# Step 2: 归一化 gate weights
gate_norm = topk_vals / sum(topk_vals)  # [K], 确保累积 ≤1

# Step 3: 按 gate weight 降序排列
sorted_order = argsort(gate_norm, descending=True)

# Step 4: 计算 unimportance degree score
scores = zeros(K)
cumulative = 0.0
for rank in range(K):
    e_idx = sorted_order[rank]
    scores[e_idx] = cumulative        # 累积前面所有 expert 的 weight
    cumulative += gate_norm[e_idx]

# Step 5: 双阈值决策 (T1=0.6, T2=0.9)
for i, e in enumerate(topk_ids):
    if scores[i] <= 0.6:
        load_high_precision(e)        # FP16/INT8
    elif scores[i] <= 0.9:
        load_low_precision(e)         # INT4/INT2
    else:
        skip_expert(e)                # 不加载

# 结果分布 (Mixtral-8x7B): 67% high / 30% low / 3% skip
# top-1 expert 始终 score=0，保持高精度 (50% 选择)
```

关键设计点：
- Pearson r=0.99 验证 ||G(x)|| 是 ||G(x)E(x)|| 的有效代理，避免计算 E(x) 的开销
- s_{e_i} 的累积性质确保：排位越低的 expert（gate weight 越小）得分越高，更可能被降精度或跳过
- top-1 expert 始终得分为 0，确保最重要的 expert 始终高精度
- 阈值通过 profiling 一次 ||G(x)|| 分布确定，无需逐样本调整

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 计算开销极小：||G(x)|| 就是 softmax 输出本身，无需额外计算
- 归一化：所有 ||G(x)_{e_i}|| 除以 Σ||G(x)_{e_j}|| 使 score 在 [0,1] 范围内
- 阈值设定：在 calibration dataset 上运行一次推理，收集所有 ||G(x)|| 值，按分位数确定 T1/T2。例如 Mixtral-8x7B 的 T1=0.6 覆盖 67% expert，T2=0.9 覆盖 97% expert
- 精度保持：GSM8K accuracy 从 0.52 降至 0.51（FP16→FP16+INT4），TruthfulQA 基本不变
- 变体（MoE-APEX）：相同公式但 cache policy 改用 LCU (Least Costly Used) 替代 LHU

涉及论文标题：
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- MoE-APEX: An Efficient MoE Inference System with Adaptive Precision Expert Offloading

## Module Decomposition for MoE Inference Parallel Strategy

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Module Decomposition for MoE Inference Parallel Strategy 是 HAP 提出的将 MoE 模型按计算特征分解为 Attention 模块和 Expert 模块两个独立计算单元进行并行策略选择的方法。核心洞察：Attention 模块和 Expert 模块在推理时具有截然不同的计算特征（FLOPs 量、参数规模、通信模式敏感性），因此需要不同的并行策略。Attention 模块参数量小但包含 KV cache 内存需求，适合 DP/TP/DP+TP；Expert 模块占总参数约 90%，适合 EP/TP/EP+TP（排除 DP 以节省显存）。每个模块配备专用的推理延迟仿真模型（计算仿真基于 FLOPs，通信仿真基于数据量和带宽），支持对任意策略组合的端到端延迟进行精确估计（计算误差 <10%，通信误差 <5%）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MoE Transformer 层的 Module Decomposition 结构：

```
# MoE Transformer Layer: Module Decomposition View
# 每层包含两个独立模块，可分别选择并行策略

# ┌─ Attention Module ─────────────────────────────────────┐
# │  可选策略: DP, TP, DP+TP                                │
# │  参数量: ~d_model² × 4 (Q/K/V/O) + KV cache            │
# │  计算特征: O(b × s × d_model²)，prefill 计算量大        │
# │  约束: prefill 和 decode 必须使用相同策略 (KV cache)     │
# │                                                        │
# │  Input: h [b, s, d_model]                              │
# │  Q/K/V = h @ W_qkv                                     │
# │  Attention(Q, K, V) = softmax(QK^T/√d_k) × V           │
# │  Output = Attn_out @ W_o                                │
# └────────────────────────────────────────────────────────┘
#                          ↓ h_attn
# ┌─ Expert Module ────────────────────────────────────────┐
# │  可选策略: EP, TP, EP+TP (排除 DP)                      │
# │  参数量: ~(d_model × d_intermediate × 3) × num_experts  │
# │  计算特征: O(b × s × top_k × d_model × d_intermediate)   │
# │  约束: prefill 和 decode 可使用不同策略                  │
# │                                                        │
# │  gate_logits = h_attn @ W_gate  [b×s, num_experts]     │
# │  topk_idx, topk_w = topk(softmax(gate_logits), k)      │
# │  for expert in topk_idx:                               │
# │      expert_out += topk_w × SwiGLU(h_attn, expert)     │
# │  Output = h_attn + expert_out                           │
# └────────────────────────────────────────────────────────┘
#                          ↓
#                     h_next = LayerNorm(h)

# Module Decomposition 的优势:
# 1. Expert Module 在 prefill/decode 可独立切换策略
# 2. Attention Module 不受 Expert 策略影响 (KV cache 独立)
# 3. 仿真模型可按模块粒度分别校准
```

在推理全流程中，每层延迟 = T_attn + T_experts + T_comm。T_attn 取决于 Attention 模块的并行策略（DP=无通信全独立计算, TP=局部计算+AllReduce 聚合），T_experts 取决于 Expert 模块的并行策略（EP=All-to-All dispatch/combine+本地计算, TP=局部计算+AllReduce 聚合），T_comm 是两个模块通信的总和（可能重叠）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Module Decomposition 在 HAP 中实现为 DeepSpeed-FastGen 的扩展。每个模块的可行并行策略由硬件配置（GPU 数、显存、带宽）和模型配置（hidden dim、expert 数、层数）决定。策略空间的构建规则：(1) Attention: DP degree A_d 必须整除 batch size；(2) TP degree A_t 必须整除 hidden dim 和 KV head 数；(3) 总设备数 N = A_t × A_d = E_d × E_t × E_e；(4) EP degree E_e 必须整除 expert 数，TP degree E_t 必须整除 expert 中间维度。内存约束检查包含 KV cache（与 A_d 相关）、Attention 权重（DP 时复制 d 倍）、Expert 权重（各策略下 per-device 相同）、activation（EP 时按 2× TP 保守估计）。Module Decomposition 的设计使得搜索空间从单一策略（TP or EP）扩展为组合空间（Attn DP+Exp EP、Attn TP+Exp TP、Attn DP+Exp TP 等），ILP 在更大的空间中寻找到真正的最优解。

涉及论文标题：
- HAP: Hybrid Adaptive Parallelism for Efficient Mixture-of-Experts Inference

## SiLU Activation and Sparsity Challenge in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SiLU (Sigmoid Linear Unit，也称 Swish) 是一种神经网络激活函数：SiLU(x) = x · σ(x) = x / (1 + e^{-x})。与 ReLU（Rectified Linear Unit: max(0, x)）不同，SiLU 对所有输入（包括负值）都产生非零输出——负值输出为负的小值，正值输出为正，零点附近平滑过渡。SiLU 是现代 LLM（包括 Mixtral-8x7B, LLaMA 等）中广泛使用的激活函数。

Fiddler 论文分析了 SiLU 对 MoE 推理中稀疏性利用的影响：ReLU 的稀疏性（大量零输出）使得某些优化方法可以利用激活稀疏性跳过大比例计算，但 Mixtral-8x7B 使用 SiLU 而非 ReLU，导致激活值几乎全部非零（<2% 的激活值绝对值 < 0.001）。然而，FloE 论文发现 SiLU 并不完全阻止稀疏性利用——通过 magnitude-based 阈值剪枝，SiLU 输出中许多**小幅值激活可以被截断为零**，尤其是 SiLU(gate) 的输出在大量接近 -0.28（SiLU 最小值）的输入下，对应的输出幅值非常小。FloE 的实验表明：在 up projection 输出上做幅值剪枝（90% 稀疏度）仅带来 ~5% perplexity 退化，而 SiLU(gate) 输出剪枝在 70% 稀疏度下 perplexity 已突破 7。理论分析（Theorem 3.1）证明：L_down ≤ L_up < L_gate，即对 down projection 输入做剪枝误差最小，对 gate projection（SiLU 输出）做剪枝误差最大。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mixtral-8x7B 中 SiLU 在 expert FFN 中的作用（SwiGLU 结构），FloE 的上下文稀疏化修改：

```
// 标准 SwiGLU Expert FFN:
gate = x @ W_gate                  // [s, 14336]
up   = x @ W_up                    // [s, 14336]
act  = SiLU(gate)                  // SiLU(x) = x * sigmoid(x)
fused = act * up                   // Hadamard product
output = fused @ W_down

// FloE Contextual Sparsification (基于 up projection 输出):
up = x @ W_up
mask = (|up| >= t)                 // t 由目标稀疏率 k 确定
// 仅保留 |up| >= t 的通道
sparse_gate = SiLU(x @ W_gate[mask])
sparse_fused = sparse_gate * up[mask]
sparse_output = sparse_fused @ W_down[mask]
// W_down 转置为列主序 W_down^T，与 W_gate 列对齐
```

三种投影矩阵输出的激活值分布（FloE Figure 2）：
- W_gate 的 SiLU 输出：大量值聚集在 -0.28（SiLU 最小值）附近，呈 shift-exponential 分布
- W_up 的输出：近似高斯分布 N(0, σ²)，零均值对称
- W_down 的输入（= SiLU(gate)⊙up）：两种分布组合，但仍以零为中心

FloE 证明的核心不等式：在相同稀疏率下，up projection 输出剪枝的恢复误差严格小于 gate projection（SiLU 输出）的误差，这源于 up 的高斯对称性使得阈值剪枝的信息损失最小。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- SiLU 在 PyTorch 中为 `F.silu()` 或 `nn.SiLU()`（自 PyTorch 1.7+）
- SwiGLU 将 SiLU 与 gated 结构结合，是 LLaMA/Mixtral 等模型的标准 FFN 结构
- 关键优化洞见：虽然 SiLU 不产生严格零值，但可通过幅值阈值实现有效的上下文稀疏化——FloE 选择剪枝 up projection 输出（而非 SiLU 输出），因为 up 的线性+高斯分布使剪枝误差可控
- FloE 在不同 MoE 模型（Mixtral-8×7B, Phi-3.5-MoE, DeepSeek-V2, DeepSeek-MoE-16B, Qwen1.5-MoE）和 dense LLM（LLaMA-3-8B）上验证了 up projection 对稀疏化最不敏感的一致性结论

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Expert Popularity in MoE Models

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Popularity（专家热门度）是 MoE 模型中各 expert 被不同输入 token 激活的频率分布。在 Mixtral-8x7B 等 MoE 模型中，不同 expert 学习不同的语言模式或 token 特征，导致某些 expert（如学习常见句法结构的 expert）被显著更频繁地激活。Fiddler 通过 offline profiling 量化了这种分布：在 256 个 expert 中 popularities 均值=0.71（相对于最热门 expert 的比值），std=0.08，25th percentile=0.67，75th percentile=0.76，最低值=0.22。分布相对均衡但存在足够差异，使得热门度导向的 GPU 放置比随机放置提升 3-5 个百分点的 hit rate。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fiddler 的 expert popularity profiling 流程：

```
// Offline profiling (一次性, 使用 calibration data)
// 输入: MoE model, calibration dataset (ShareGPT)
// 输出: popularity[layer][expert] ∈ [0, 1]

for each sample in calibration_data:
    hidden_states = model.embed(sample)
    for layer in 0..31:
        gate_scores = softmax(W_gate[l] @ hidden_states)  // [tokens, 8]
        top2_indices = topk(gate_scores, k=2)              // [tokens, 2]
        for token_t in range(num_tokens):
            for idx in top2_indices[token_t]:
                activation_count[layer][idx] += 1

// 归一化 (vs 最热门 expert)
for layer in 0..31:
    max_count = max(activation_count[layer])
    for expert in 0..7:
        popularity[layer][expert] = activation_count[layer][expert] / max_count

// 全局排序 (所有 256 experts 统一比较)
all_experts = [(l, e, popularity[l][e]) for l in 0..31 for e in 0..7]
all_experts.sort(key=lambda x: -x[2])  // 降序

// GPU placement: select top-N_gpu
gpu_experts = set(all_experts[:N_gpu])
```

Fiddler 的 heat map 可视化（Figure 8, Appendix C）显示：
- 大部分 expert 的 popularity 在 0.6-0.9 之间（相对均衡）
- 仅 15/256 expert 的 popularity < 0.6
- 27/256 expert 的 popularity > 0.8
- 最低 popularity=0.22（某 expert 激活次数仅为最热门 expert 的 22%）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **Profiling data**：Fiddler 使用 ShareGPT 对话数据集；论文在 LMSYS-Chat-1M 上验证了跨数据集的鲁棒性
- **Calibration 效率**：offline profiling 仅需一次（每个 calibration sample forward pass 一次），不增加推理运行时开销
- **假设依据**：Expert 选择基于 token 语义/句法特征，该假设在 Mixtral 论文和 OpenMoE 论文中得到验证——expert popularity 在不同下游 domain 间近乎 universal
- **与其他 expert 选择方法的对比**：
  - Expert LRU Cache（Mixtral-Offloading）：runtime 动态调整，适合 expert locality 较强的场景
  - Expert Popularity（Fiddler）：init-time 固定，无 runtime 维护开销，适合 popularity 分布稳定的场景
  - 两者正交互补——可同时使用
- **局限性**：若模型 weight 更新且 expert specialization 改变，需重新 profiling

HarMoEny 的关键发现：Expert popularity skew 是**动态的（dynamic）**且**batch 间剧烈波动**的。对 Qwen MoE（60 experts）和 Switch128（128 experts, bookcorpus）的分析（Figure 1）表明：(1) 偏斜随层深累积——层 0 仅 3/128 expert 接收平均 19% token，最后层 3 expert 接收 60%；(2) 偏斜随输入 domain 变化——medical vs programming prompts 产生完全不同的 expert activation pattern；(3) Batch 间波动——连续 batch 间 throughput 可下降 37.6%。这一动态性使得 profiling-based 方案（ExFlow 需 8.5min profiling for Switch128, 45min for Qwen）完全失效——profiling 时间远超 batch 处理时间（289ms）。HarMoEny 通过 **online token rebalancing**（无需 profiling，per-batch adaptation）和 **async expert prefetching** 解决此动态偏斜。

涉及论文标题：
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- HarMoEny: Efficient Multi-GPU Inference of MoE Models

## MoE (Mixture-of-Experts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixture-of-Experts (MoE) 是一种神经网络架构，将传统 dense 模型中的前馈网络（FFN）替换为多个并行的"专家"子网络和一个可学习的门控机制（gate）。对于每个输入 token x，gate 计算一个分数向量 g(x) ∈ R^{|E|}，表示该 token 与每个专家 E_i 的亲和度，然后选择 top-k 个专家处理该 token，各专家输出加权求和得到最终结果：

$$MoE(x) = \sum_{i \in \tau} g(x)_i \cdot E_i(x)$$

其中 gate 通常是一个线性变换后接 softmax：g(x) = softmax(W_g · x)。专家网络 E_i 通常实现为标准 FFN。MoE 的核心优势是 sparse activation——每个 token 只激活 k 个专家（k << |E| 总数），因此在增大模型总参数量（capacity）的同时，计算量仅随 k 线性增长而非随专家总数增长。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 FOLDMOE 论文中，GPT-MoE 模型每隔一个 Transformer block 将 FFN 替换为 MoE 层（使用 top-1 GShard gate）。一个 Transformer-MoE block 的算法 pipeline：

```
# 输入: sequence X = [x_1, x_2, ..., x_n]
# 阶段1: Attention
for t in 1..n:
    q_t, k_t, v_t = W_q(x_t), W_k(x_t), W_v(x_t)
    z_t = softmax(q_t @ K_{1:t-1}^T / sqrt(d_k)) @ V_{1:t-1}

# 阶段2: MoE with top-1 gating
for t in 1..n:
    g(z_t) = softmax(W_g @ z_t)         # gate scores
    expert_idx = argmax(g(z_t))          # top-1 routing
    y_t = E_{expert_idx}(z_t) * g(z_t)[expert_idx]
```

在分布式训练中，专家分布在多张 GPU 上（Expert Parallelism），gate 计算后需要 all-to-all dispatch 将 token 发送到对应专家所在 GPU，计算完成后 all-to-all combine 收集结果回原 GPU。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoE 的实现通常基于现有训练框架：
- **Megatron-LM**: 提供 Megatron-MoE 实现，支持 EP + TP + DP 混合并行
- **Tutel**: 提供自适应 MoE，支持 token-level overlapping
- **DeepSpeed-MoE**: 微软的 MoE 实现，集成在 DeepSpeed 中
- **FairScale / PyTorch FSDP**: 通过专家并行方式支持 MoE

FOLDMOE 基于 Megatron-LM 框架，将每层 MoE 隔一层插入 Transformer block（alternating dense-MoE pattern），使用 top-1 GShard gate 和 capacity factor=1.0。

FSMoE 通过模块化设计将 MoE 层分解为 6 个子模块（Gate、Order、I-Order、Dispatch、Combine、Expert），预实现 4 种路由函数（GShard、Sigmoid、X-MoE、Expert Choice）。这种非侵入式模块化使得新增路由函数或通信算法只需继承对应基类，无需修改调度器代码。FSMoE 的在线 profiler 对各子模块执行时间建模，为调度器提供性能数据。

Fair-MoE 提出 FO-MoE（Fairness-Oriented MoE）变体，在医疗 VLM 中将 embedding-based MoE 和 feature-based MoE 两级结构集成到 CLIP encoder 中，通过 expert capacity 过滤偏置 patch embedding 以提取公平特征。

FasterMoE（PPoPP'22）进一步从分布式训练角度分析了 MoE 的动态特性：(1) 训练数据的偏斜分布导致 expert 热度高度不均衡且随 iteration 动态变化（图 4 可视化），热门 expert 可接收 3.2× 平均的 tokens；(2) MoE 允许在不增加计算量的前提下增大模型参数量——weight matrices 沿特定维度切分，每部分仍产生同尺寸输出，但 GeMM 计算量保持较小；(3) 在 transformer 中，MoE 层通常替换 MLP 层中的密集 FC 层，gate 是一个小型 FC 层（计算 fit score 并取 top-k）。

Fiddler 从资源受限环境推理角度利用了 MoE 的两个关键特性：(1) MoE 的参数量-计算量不对称性——模型总参数可极大（>90GB for Mixtral-8x7B FP16），但每 token 仅激活 top-2 expert（~12.5% 参数），使得 CPU 在 small-batch 场景下的低计算能力仍可承接部分 expert 计算；(2) Expert 的独立可分离性——每个 expert 的权重和计算是完全独立的，可以被独立分配到不同设备（GPU 或 CPU）执行，无需跨 expert 通信。这是 Fiddler 的 per-expert CPU/GPU 动态调度策略成立的前提——若 expert 之间共享权重或有数据依赖，则无法独立决策每个 expert 的执行后端。

Flex-MoE 将 SMoE 应用于**多模态 missing modality** 场景：将 Transformer 的 FFN 替换为 SMoE layer，expert 索引按 modality combination 分配（如 "IGCB"=0, "IGC"=1, ..., "B"=14），剩余 index 作为 buffer expert。核心创新是两阶段训练：(1) Generalization——全模态样本通过 G-Router 训练通用 expert 知识；(2) Specialization——S-Router 通过 cross-entropy loss 将 top-1 gate 绑定到目标 modality combination expert。batch 内 samples 按可用模态数降序排列（课程学习），encoder 仅用对应 modality 的 observed 样本训练，缺失嵌入从 Missing Modality Bank 查找。

GLaM (Google, 2022) 进一步展示了 MoE 在 decoder-only LLM 上的大规模实践。GLaM (64B/64E) 拥有 1.2T 总参数，64 个 expert，每 token 通过 top-2 softmax gating 仅激活 2 个 expert（96.6B 活跃参数，占总参数 8%）。GLaM 的架构特征：(1) 每隔一层 Transformer FFN 替换为 MoE 层（alternating pattern），MoE 层 expert 的 hidden dim H=32768；(2) top-2 gating：gating_logits = softmax(x @ W_gate)，选 top-2 expert 并归一化 gate 值，加权组合输出；(3) GShard auxiliary load balancing loss 系数 0.01；(4) GSPMD 2D sharding 将 expert 权重 [E, M, H] 沿 E 维和 H 维划分到 TPU-v4 集群的 2D device mesh 上。GLaM 证明了 sparse MoE 在 few-shot in-context learning 任务上超越同等计算量的 dense 模型：at similar FLOPs/token, MoE (64B/64E) 在 29 个 NLP benchmark 上平均 zero/one/few-shot 性能均高于 GPT-3 (175B dense)，而推理 FLOPs/Token 仅为 GPT-3 的 51.4%，训练能耗仅为 GPT-3 的 1/3。

Hecate 进一步揭示了 MoE 训练中的 **expert load 动态性**：gate 的频繁演化导致 expert load 快速波动和不平衡（图 3 可视化，不同 expert 的 token 比例在 iteration 间显著变化），导致 EP 的严重 straggler 效应（最坏情况下性能下降 5.18×）。为解决此问题，Hecate 提出 FSSDP 范式：将 MoE layer 的 parameters 和 optimizer states 完全分片到所有 device，每次 iteration 用 SparseAllGather 从 shards 零构建临时 expert placement，用 SparseReduceScatter 同步 gradients 回 source device，消除 traditional expert rearrangement 的 memory/timeliness 两难困境。

LExI (Chitty-Venkata et al., 2025) 进一步从 layer sensitivity 角度丰富了 MoE 的理解：(1) 不同层的 expert 冗余程度差异显著——某些层对减少 active expert 数量高度不敏感（低 Frobenius 范数扰动），而其他层则高度敏感；(2) 传统的固定 top-k 假设所有层需要相同数量的 active expert 是不合理的，layer-adaptive top-k 可以更高效地分配计算资源；(3) MoE 推理中 load imbalance 不仅来自 token-to-expert 路由的不均匀，也来自固定 top-k 导致的跨层计算冗余——即使每层 expert 负载均衡，统一 top-k 仍可能在低敏感层浪费计算；(4) Expert 冗余可以通过仅使用模型权重的 data-free sensitivity profiling 量化，无需 calibration 数据集。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Layerwise Recurrent Router for Mixture-of-Experts
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

Hunyuan-Large 是目前最大的开源 Transformer-based MoE 模型（389B 总参数, 52B 激活参数, 256K 上下文），采用 64 layers、1 shared expert + 16 specialized experts（top-1 激活）、GQA (8 KV groups) + CLA (每 2 layers 共享 KV)、SwiGLU 激活、RoPE、128K tokenizer。预训练 7T tokens（含 1.5T 合成数据），后训练 SFT + DPO。证明了 MoE 在大规模开源模型上的有效性——52B 激活参数在 MMLU (88.4)、MATH (69.8)、HumanEval (71.4) 等 benchmark 上超越 LLama3.1-405B。

Joint MoE Scaling Laws (Ludziejewski et al., 2025) 进一步从 Scaling Laws 角度研究了 MoE 的 compute/memory efficiency。使用 Switch MoE (top-1 routing) 训练 280+ 模型（最高 5B total params, E∈{1,2,4,8,16,32}），推导 joint scaling law L(N_act, D, Ê) = aÊ^δ·N_act^(α+γ·ln(Ê)) + bÊ^ω·D^(β+ζ·ln(Ê)) + c，证明 MoE 在 memory-constrained 场景下可超越 dense 模型，打破"MoE memory-inefficient"的传统认知。

LYNX (Gupta et al., 2025) 从 MoE 推理角度揭示了 batch 级别 expert 选择的关键特性：(1) Training 时的 load-balancing loss 导致 inference 时 batch 级 expert activation 存在系统性冗余——虽然 aggregate 分布均匀（变异性 ~1.2%），batch 级分布显著偏斜（变异性 ~15-20%）；(2) Decode 阶段的 arithmetic intensity = B×k/N，在 moderate batch size 下 MoE decode 是 memory-bandwidth-bound——42% decode latency 花在 HBM 加载 expert 权重上；(3) Top-1 expert 主导输出质量，lower-ranked experts 高度冗余，这种 Expert Rank Hierarchy 跨 tasks 一致；(4) Prefill 和 Decode 对 expert fidelity 的敏感度存在根本性不对称——prefill 需严格 fidelity（建立 context），decode 因 attention/residual/context 补偿而高度容错。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models
- Fiddler: CPU-GPU Orchestration for Fast Inference of Mixture-of-Experts Models
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Layerwise Recurrent Router for Mixture-of-Experts
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

## FSSDP (Fully Sharded Sparse Data Parallelism)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FSSDP (Fully Sharded Sparse Data Parallelism) 是 Hecate 系统提出的 MoE 训练新范式，受 FSDP (Fully Sharded Data Parallelism) 启发但针对 MoE layer 的稀疏性重新设计。FSSDP 分为两个阶段：
(1) **Sharding Phase**：将每个 MoE layer 的 parameters 和 optimizer states 划分为 |𝒟| 个不相交的 MoE shards，每个 shard 包含一组 expert 的完整参数和优化器状态，唯一分配给一个 device。全局仅保留一份 optimizer states 副本，实现最小且均衡的内存占用。
(2) **Materialization Phase**：每次 iteration，用 SparseAllGather 从 shards 稀疏物化 (sparsely materialize) 一个临时的 expert placement ——即"从零构建"而非"从上一个 placement 迁移"。Forward 后释放物化参数（可选 re-materialization 在 backward 重新物化），backward 后用 SparseReduceScatter 将 replicated expert 的 gradients reduce 回持有对应 MoE shard 的 device。每次 iteration 都能工作在针对当前 expert load 分布最优的 placement 下，无需在 iteration 间迁移 expert 状态，因此不存在 rearrangement 系统的 memory/timeliness trade-off。

FSSDP 与 rearrangement 系统的关键区别：对于同一 expert placement 𝒫'，FSSDP 的 spAG(𝒫, 𝒫') + spRS(𝒫', 𝒫) 的通信量上界 O(2λS) 与 rearrangement 系统为同步 replicated expert gradients 的 AllReduce 总通信量完全相同（λ 为需跨 device 通信的 expert 比例），但 FSSDP 消除了 rearrangement 系统在 iteration 间 expert 参数+优化器状态迁移的额外通信开销。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSSDP 在单 MoE layer l 一个 iteration 中的完整执行流程：

```
=== SHARDING PHASE (每 100 iterations 左右低频执行) ===
// 输入所有 MoE layers 的 expert load 分布 F^g
// Heterogeneous Sharding (Algorithm 2):
1. J ← 各层 top-t overloaded experts (可被 sparse materialization 选中)
2. J' ← E^g - J  // underloaded experts
3. 每 device 分配 |E^g|/|D| 个 slots
4. 先放置 J' (underloaded experts): layer by layer,
   优先最 overloaded 的 layer, 每 expert 选 least-loaded node/device
5. 再填充 J (overloaded experts): 任意分配到剩余 slots
// 输出: P^g = {P_0, ..., P_L} 各层 sharding plan

=== MATERIALIZATION PHASE (每 iteration 执行) ===
// Sparse Materialization (Algorithm 1):
Input: P (当前 sharding plan), F (预测的 expert load),
       t (overlap degree), m (memory capacity per device)
t = T_attn_fwd * bw / expert_size  // 可在 attention 时间内
                                   // 隐藏通信的最大 expert 数
m = 每 device 可额外容纳的 expert 参数数

if t <= m:
    // 物化 top-t overloaded expert 到所有 device
    P' ← P ∪ (D × E^topT)
else:
    totSlots ← |D| * m
    for e in sorted overloaded experts (by load descending):
        n ← 按负载比例分配 replica slots
        P^e ← 分配 n 个 replica 到 nodes/devices (优先有空闲 slots 的 node)
        P' ← P' ∪ P^e

=== FORWARD PASS ===
// Communication-Overlap:
[Attention Forward] ← spAG(P, P') 与此重叠
[MoE Gate] → Calibration (可选): 用实际 token assignment
    重新运行 Algorithm 1, 若收益>通信开销则追加额外的 spAG
[Token Dispatch: Topology-aware All-to-All]
    - 同 node 内有 expert replica → 优先 intra-node dispatch
    - 无同 node replica → 跨 node, 均匀分配到 replica devices
[Expert FFN Computation on materialized parameters]
[Release materialized parameters (for re-materialization mode)]

=== BACKWARD PASS ===
[Attention Backward] ← spRS(P',P) (layer l 的梯度 reduce)
                    ← spAG(P, P') (layer l+1 的 re-materialize)
                    两者与此重叠 (backward 约 2× forward 时间)
[Expert Backward Computation on re-materialized parameters]
[spRS(P',P) for this layer (若未在 attention backward 中完成)]

=== OPTIMIZER STEP ===
各 device 在其 MoE shards 上用 reduced gradients 更新 optimizer states
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSSDP 的实现需要以下组件：
- **稀疏通信原语**：SparseAllGather 和 SparseReduceScatter。在 Hecate 的 prototype 中，用 NCCL group calls 实现（spAG = 一组 Broadcast, spRS = 一组 Reduce），每组包含多个同步调度的 point-to-point 或 collective 操作。更高效的实现可利用数据稀疏性和网络拓扑（留作 future work）。
- **Scheduler**：基于滑动窗口 (w=5) 预测 expert load，在 overlap degree t 和 memory capacity m 约束下搜索近似最优 placement。
- **Dispatcher**：拓扑感知的 token 路由，优先 intra-node 通信。
- **Communicator**：管理稀疏 collective 和 All-to-All 的通信队列。
- FSSDP 适用于大规模 MoE 训练场景（64+ experts × 32+ GPUs），expert load imbalance 越严重收益越大（低带宽 inter-node 环境下加速比更显著）。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## Heterogeneous Sharding (for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Heterogeneous Sharding 是 Hecate 在 FSSDP 的 sharding phase 中使用的跨层 MoE shard 分配算法 (Algorithm 2)。与传统的 homogeneous sharding（每个 device 等量分配 expert——如 64 experts / 32 devices = 每 device 2 experts）不同，heterogeneous sharding 允许每个 MoE shard 包含 0 到 |ℰ| 个任意数量的 expert，且不同 MoE layer 可以有不同的分配方案，只要所有 layer 的 shards 在每 device 上的总内存需求均衡。

设计动机：sparse materialization 主要帮助 overloaded expert（物化到多个 device 分散负载），但 underloaded expert 的 placement 也需要优化。例如，若某 node 上的所有 MoE shards 只包含 underloaded experts，则该 node 的入站带宽可能在 All-to-All 中被这些 crowded underloaded experts 的 token 淹没，因为该 node 可能是这些 token 的唯一目的地。Heterogeneous sharding 通过跨层统一调度 underloaded experts 来缓解这一问题。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Algorithm 2 的核心思路：
1. 将各层 experts 分为 J (overlappable/top-t overloaded) 和 J' (underloaded/其余) 两个不相交集
2. 先放置 J' (underloaded experts)：layer by layer，优先最 overloaded 的 layer（因为 overloaded expert 更多的 layer 面临的 All-to-All congestion 更严重），每个 expert 选 least-loaded node → least-loaded device on that node
3. 再填充 J (overloaded experts)：任意分配到剩余 slots

```
Algorithm 2: Heterogeneous Sharding
Input: F^g (所有层 expert load), t (overlap degree)
Output: P^g = {P_0, ..., P_L} (各层 sharding plan)

J ← top-t experts by load for each layer
J' ← E^g - J
slots_per_device ← |E^g| / |D|

// Phase 1: Place underloaded experts
L ← {E_l ∩ J' for l = 0..L}  // 各层的 underloaded expert set
for each E'_l in sortByMaxLoadDescending(L):
    P_l ← ∅
    for each e in sortByLoadDescending(E'_l):
        n ← least-loaded node (优先剩余 slots 少的)
        d ← least-loaded device on node n (同上优先级)
        P_l ← P_l ∪ {(d, e)}
        S_d ← S_d - 1
    P^g ← P^g ∪ P_l

// Phase 2: Fill remaining slots with overlappable experts
update P^g by arbitrarily placing J into remaining slots
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Heterogeneous sharding 涉及跨 MoE layer 的状态迁移（re-sharding），会引入 critical path 上的延迟。但 Hecate 论证 re-sharding 可以低频触发（每 100 iterations），因为 underloaded experts 的梯度更新幅度小（处理的 token 少），其 load 变化缓慢（图 3 证实的 temporal locality）。实验显示 heterogeneous sharding 在不同 re-sharding 间隔（10-100 iterations）下均能提供一致的 1.34-1.42× speedup，证明对频率不敏感。
- Re-sharding 仅在 shard 确实发生变化时才执行实际的数据迁移，进一步摊销开销。
- 与 sparse materialization 的组合使用是关键：单独使用 heterogeneous sharding 或 sparse materialization 的效果有限，两者结合能实现 3.32× speedup。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## Sparse Materialization (for MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse Materialization 是 Hecate 在 FSSDP 的 materialization phase 中使用的 expert placement 搜索算法 (Algorithm 1)。其目标是在两个系统约束下搜索近似最优的 expert placement：(1) overlap degree t——可在 attention computation 时间内隐藏通信的 expert 物化数上限；(2) memory capacity m——每 device 可额外容纳的 expert 参数数。基于预测的 expert load distribution（滑动窗口平均，w=5），算法决定哪些 expert 需要 replica（物化到多个 device）以及 replica 的分布。

算法是拓扑感知的：overlap degree t 的计算使用 inter-node bandwidth（异构网络时）或 uniform bandwidth（同构网络时），优先避免跨 node 通信。当 t > m 时（高 overlap degree 但内存受限），按 expert load 比例分配 replica slots，优先有空闲 slots 的 node。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Algorithm 1: Sparse Materialization
Input: P (sharded placement), F (expert load distribution),
       t (overlap degree), m (memory capacity)
Output: P' (materialization plan)

t ← min(t, |E|), m ← min(m, t)
P' ← P

if t <= m:
    // Case 1: 内存充裕，可将 top-t expert 物化到所有 device
    E^topT ← Top t experts by load F
    P' ← P' ∪ (D × E^topT)  // 所有 device 都接收这些 expert
else:
    // Case 2: 内存受限，按负载比例分配 replica
    totSlots ← |D| * m
    for each e in sortByLoadDescending(E^topT):
        n ← assignSlotsByLoad(e, totSlots, F)
        // 如 e 有 30% 总负载 → 分配 totSlots * 30% 个 slots
        P^e ← Distribute n replicas of expert e across
              nodes and devices, prioritizing nodes with
              more available slots
        P' ← P' ∪ P^e

// Calibration (optional, on critical path):
// 在 MoE gate 输出实际 token assignment 后，重新运行 Algorithm 1
// 若 calibrated placement 的延迟收益 > 额外通信开销 → 接受
return P'
```

overlap degree t 的计算：`t = T_non-MoE * bw / expert_size`，其中 T_non-MoE 是 attention layer 的计算延迟（可 profiled 或实时获取），bw 是 inter-node bandwidth（异构网络）/ uniform inter-device bandwidth（同构网络），expert_size 是单个 expert 参数的字节数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Sparse materialization 在每次 iteration 的 forward pass 中执行（与 attention computation 重叠），不在 critical path 上。
- Calibration 阶段可选地在 critical path 上额外执行（MoE gate 输出后、token dispatch 前），仅当追加的 SparseAllGather 带来的 load balance 改善超过其通信延迟时才执行。
- 结合 re-materialization：forward 后释放物化参数，backward 前重新执行 sparse materialization（通过 spAG 重新物化）。

涉及论文标题：
- Hecate: Unlocking Efficient Sparse Model Training via Fully Sharded Sparse Data Parallelism

## DDL-Roofline Model (Distributed Deep Learning Roofline)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

DDL-Roofline（Distributed Deep Learning Roofline）是 FasterMoE（PPoPP'22）提出的面向分布式训练的性能分析模型，将经典单设备 Roofline 模型扩展到分布式场景。X 轴为计算-通信比 R_CC = Lat_comp / Lat_comm，Y 轴为平均计算吞吐量 P̄ = (总 FLOPs) / (N × Lat_e2e)。模型定义两条理论上界：(1) 理想曲线 P̄_ideal = P_w · min{1, R_CC}（通信与计算完全重叠执行），(2) 半理想曲线 P̄_semi = P_w · R_CC/(R_CC+1)（同步执行模式）。不同并行策略（数据并行、模型并行、专家并行）在图中占据不同区域，可直接反映其效率特征和优化方向。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# DDL-Roofline 分析流程（以训练一个 MoE MLP 层为例）

# Step 1: 预测计算延迟 (Eq. 1)
Lat_comp = max_{w in workers} { 4 * B_w * α * H² / P_w }
# B_w: worker w 的 batch size, H: embedding 维度
# αH: MLP 中间层维度, P_w: GeMM 吞吐（通常为峰值的 90%+）

# Step 2: 预测通信延迟 (Eq. 2)
Lat_comm = max_{l in links} { T_l / W_l }
# T_l: 链路 l 上的流量（基于路由策略和拓扑计算）
# W_l: 链路带宽（有向图，两个方向分别建模）

# Step 3: 计算 R_CC 和 P̄
R_CC = Lat_comp / Lat_comm
P̄ = (12 * α * H² * ΣB_w) / (N * Lat_e2e)

# Step 4: 在 DDL-Roofline 图上定位并分析
# 数据并行: R_CC 极小（all-reduce 同步梯度通信量大）→ 左侧，低于半理想曲线
# 模型并行: R_CC 较大（同步 embedding 通信量小但不可重叠）→ 半理想曲线上
# 专家并行: R_CC 大但 P̄ 低（负载不均衡）→ 远低于半理想曲线
# 优化方向: (1) 减少 Lat_comp（影子化）→ 向左移; (2) 重叠执行（智能调度）→ 向上跃升
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DDL-Roofline 在 FasterMoE 中的实现：(1) 计算模型基于 GeMM 吞吐（测量数据显示 V100 可达 90%+ 峰值），(2) 通信模型基于有向网络拓扑图（考虑 NVLink、PCIe、Infiniband 的带宽不对称性），(3) 不同集合通信操作（all-to-all-v、all-reduce、broadcast/reduce）使用不同的流量模型——all-to-all 按 pair-wise 路径累加流量，all-reduce 使用 ring 算法（2(n-1)/n·S 总发送量），(4) 在 *johnny* 和 *trevor* 两个集群上验证，端到端预测 R² 分别为 0.987 和 0.967。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

## Dynamic Shadowing (in MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Shadowing 是 FasterMoE 提出的运行时负载均衡策略，用于解决 MoE 分布式训练中因 skewed expert selection 导致的动态负载不均衡问题。核心思想是：将热门 expert 的模型参数广播复制到所有 worker（"影子化"），使得原本需要远程发送的大量 input tokens 被替换为少量的模型参数传输，热门 expert 的计算在各 worker 本地执行。影子化决策基于性能模型在每 iteration 动态判断——当 token 传输开销大于模型传输开销，或减少的计算延迟大于增加的通信开销时启用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Algorithm 1: SelectShadowExperts (每 iteration 每 worker 执行)
# 输入: B[N] - 每 worker 的 batch size (token 数)
# 输出: E_s - 需影子化的 expert 集合

def SelectShadowExperts(B):
    B_max = max(B)
    c_min = Lat_imbl(B_max)        # 当前不均衡配置的延迟
  
    # Lat_imbl(B) = max_w{3·4B_wαH²/P + 4·B_wH/W_net}   (Eq. 7)
    #   3×GeMM (1 forward + 2 backward) + 4×all-to-all
  
    E_s = []
    for i, B_i in sorted(B, key=lambda x: -x[1]):  # 降序遍历
        B_i = T[i][i]           # 保留本地 tokens
        for j != i:
            B_i += T[i][i]      # 影子化后在其他 worker 本地执行
  
        B_max_prime = max(B)    # 影子化后的最大 batch
        c = Lat_shadow(len(E_s)+1, B_max_prime)
  
        # Lat_shadow(r, B') = max_w{3·4B'_wαH²/P} + 2r·2αH²/W_net   (Eq. 8)
        #   第一项: 均衡后的 computation; 第二项: 广播 r 个 expert 参数的开销
  
        if c < c_min:           # 影子化改善延迟则采纳
            c_min = c
            E_s.append(i)
        else:
            return E_s          # 一旦不改善即停止

# 影子化启用条件 (简化):
# 条件1: B_max > rαH     → token 传输开销 > 模型传输开销
# 条件2: 3(B_max-B'_max)αH/(rαH-B_max) > P/W_net → 减少的计算 > 增加的通信
```

执行流程：(1) Forward: broadcast expert 参数到所有 worker → 各 worker 本地计算影子化 expert 的 GeMM → 非影子化 expert 仍通过 all-to-all 远程计算；(2) Backward: 各 worker 本地计算影子化 expert 的梯度 → reduce 梯度到原 worker → 原 worker 更新参数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 实现中，动态影子化基于 FastMoE 扩展，决策逻辑位于 `fastermoe/fmoe/transformer.py:34`。矩阵 T（token-to-expert 分配）在所有 worker 间共享，无需额外通信。实验显示平均 19% 的 experts 被影子化，单 expert 影子化最大加速 1.97×。在 *johnny*（16 GPU）上单独启用影子化加速 1.95×，在 *trevor*（64 GPU）上加速 4.74×——更大规模下负载不均衡问题更严重，影子化收益更大。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

## Topology-aware Gate (in MoE Training)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Topology-aware Gate 是 FasterMoE 提出的网络拓扑感知的 expert 选择策略。标准 MoE gate 仅基于 fit score 选择 top-k experts，导致大量跨节点通信在树形拓扑的上层链路上产生拥塞。Topology-aware Gate 限制跨节点 token 数量上限 L = (W_net / (M·W_local))·B，超出 L 的 token 在本地节点内重新选择 expert，从而将跨节点通信量降低至与节点内通信等时。同时保留 fit score 最高的 token-expert 对，减少对模型质量的影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 树形拓扑下的 Topology-aware Gate
# N 个节点, 每节点 M 个 worker
# W_net: 跨节点带宽, W_local: 节点内带宽

def TopologyAwareGate(tokens, scores, B, M, W_net, W_local):
    # 计算跨节点 token 上限
    L = (W_net / (M * W_local)) * B
  
    # 收集所有希望跨节点的 tokens
    remote_candidates = []
    for t in tokens:
        best_expert = argmax(scores[t])
        if expert_node[best_expert] != local_node:
            remote_candidates.append((t, scores[t][best_expert]))
  
    # 按 fit score 降序排序
    remote_candidates.sort(key=lambda x: -x[1])
  
    # 仅允许 fit score 最高的 L 个跨节点
    allowed = set(t for t, _ in remote_candidates[:L])
  
    # 其余 token 在本地节点内重新选择
    for t in tokens:
        if t not in allowed:
            # 限制 expert 选择范围为本地节点
            local_experts = [e for e in experts if expert_node[e] == local_node]
            t.selected_expert = argmax(scores[t][local_experts])
  
    # 结果: 跨节点流量从 M(N-1)/N · BH 降至 W_net/W_local · BH
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 FasterMoE 中作为 FastMoE 的 custom gate 实现，与环境变量控制开关配合。实验显示（MoE-GPT, johnny 集群）：启用拓扑感知门控后 per-iteration 延迟减少 9.4%，但需额外 18% 的 iteration 才能收敛（因部分 token 被重新分配到次优 expert）。整体收敛时间比 GShard 快 1.37×，比 BASE Layer 快 2.19×。FasterMoE 强调这是一种 co-design 方法论——对于不同网络拓扑，应设计对应的专用 gate。

涉及论文标题：
- FasterMoE modeling and optimizing training of large-scale dynamic pre-trained models

## Expert LRU Cache

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert LRU Cache 是一种针对 MoE 推理的 GPU 显存管理策略。在 MoE offloading 场景中，expert 参数大部分存储在 host RAM 中，每次推理仅加载当前 token 所需的 top-k expert 到 GPU。LRU Cache 利用相邻 token 间 expert 使用的局部性（expert locality），在 GPU 显存中为每个 MoE 层维护 k 个最近使用过的 expert 作为缓存。处理新 token 时，若所需 expert 已在 cache 中（cache hit），则无需 host-to-device 传输直接使用；若不在（cache miss），则从 host RAM 加载，并淘汰 cache 中最久未使用的 expert（若 cache 已满）。对于 Mixtral-8x7B，k=2（12GB GPU）或 k=4（16GB GPU）。

该策略的核心洞察来自对 MoE 模型 expert 激活模式的观测（图 1）：某些 expert 在 2-4 个连续 token 上反复使用，另一些则以"间隔"模式被复用。LRU 是最简单的缓存替换策略——不考虑 expert 激活频率、不同 MoE 层 cache 大小的变化或 expert 激活的序列模式——但即使如此简单的策略也能显著加速 MoE 推理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Per-layer Expert LRU Cache for Mixtral-8x7B
# 每层维护 C_l: OrderedDict, max size=k (LRU order via move_to_end)

for token t in generate():
    for layer l in 0..31:
        h = attention_block[l](h)             # attention 常驻 GPU
        gate_scores = W_gate[l] @ h           # gate 常驻 GPU
        top2_idx = topk(gate_scores, k=2)    # [e_a, e_b]
        
        output = zeros_like(h)
        for e_id in top2_idx:
            if e_id in C_l:
                # Cache hit: expert 已在 GPU 显存
                expert_w = GPU_expert_buf[l][e_id]
                C_l.move_to_end(e_id)          # 标记为 most recently used
            else:
                # Cache miss: 从 host RAM 加载
                if len(C_l) >= k:
                    evict_id, _ = C_l.popitem(last=False)  # 淘汰 LRU
                    # 若 host RAM 不足, 将 evicted expert 写回 host
                    copy GPU_expert_buf[l][evict_id] → host_pinned[l][evict_id]
                # 加载新 expert
                copy host_pinned[l][e_id] → GPU_expert_buf[l][e_id]
                C_l[e_id] = True               # 加入 cache
            
            gate_w = gate_scores[e_id] / sum(gate_scores[top2_idx])
            output += gate_w * expert_ffn(GPU_expert_buf[l][e_id], h)
        
        h = output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现细节（论文 Section 3.3）：
- Expert 参数在 pinned memory 中以连续 buffer 存储，单次 `cudaMemcpyAsync` 完成 host-to-device 传输
- GPU 侧预分配 b=4 个临时 device buffer 用于异步 expert 交换，所有 MoE 层共享以减小内存足迹
- 当 host RAM 也无法容纳完整模型时（如 Google Colab），expert 在 host RAM 和 GPU 之间按 LRU 策略换入换出，换出时写回 host
- 实现代码开源在 https://github.com/dvmazur/mixtral-offloading
- **FloE 中的扩展**：FloE 同样受益于 expert locality——相邻 token 倾向于激活相同或相近的 expert。FloE 的 inter-expert predictor (MLP) 学习捕获这种时序关联性，但预测失败时仍需 fallback 到 LRU cache 机制。FloE 的 VRAM 消融实验（Figure 8）表明，随 VRAM 增加，可缓存更多 MoE 层的 expert，减少 misprediction reload overhead。

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Speculative Expert Loading (Expert Prefetching in MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Expert Loading 是一种 MoE 推理的通信-计算重叠技术。传统 offloading 中，MoE 层必须等待 gate 计算完成后才知道需要哪些 expert——这意味着 expert 加载必须串行在 gate 之后，无法像 dense 模型那样预先加载下一层。论文发现可以利用 Transformer 残差连接的归纳偏置来预测下一层 expert：当前层的 hidden states 是下一层 hidden states 的合理近似（因残差连接逐层累加而非重算），因此将**下一层 MoE gate 函数应用于当前层 hidden states**可得到下一层 expert 选择的近似估计。系统在当前层计算期间异步预取预测的 expert，若预测正确则消除下一层加载延迟；若错误则仅浪费带宽不影响正确性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Speculative Expert Loading per token per layer
# 在处理 layer l 时预测 layer l+1 需要的 expert

# 当前层 hidden states (pre-MoE): h_l
# 当前层 gate (正常):
gate_l = softmax(W_gate[l] @ h_l)
top2_l = topk(gate_l, k=2)

# 投机预测下一层 gate (利用残差归纳偏置):
pred_gate_l1 = softmax(W_gate[l+1] @ h_l)    # W_gate[l+1] 应用到 h_l
pred_top1_l1 = argmax(pred_gate_l1)           # 最可能的 expert
pred_top2_l1 = argmax_second(pred_gate_l1)    # 次可能的 expert

# 异步预取 (独立 CUDA stream, 与当前层 expert 计算重叠):
async_stream.load_expert(l+1, pred_top1_l1)   # 后台 host→device copy
async_stream.load_expert(l+1, pred_top2_l1)   # 后台 host→device copy

# 继续当前层 expert 计算 (在 compute stream):
output_l = expert_compute(top2_l, h_l)

# 进入 layer l+1 时:
# - 若 pred_topk 正确: 即时命中，跳过加载延迟
# - 若 pred_topk 不正确: 重新加载正确 expert (仅浪费带宽)
```

论文评测了 1 层、2 层和 10 层 ahead 的预测 recall（图 2 right panel）。结果：1 层 ahead 时 recall 较高（因残差连接的归纳偏置最准确），2 层和 10 层 ahead 的 recall 显著下降——隐藏状态距离增加导致预测退化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 投机预取在当前层所有 expert 加载完成后立即触发
- 预取 1-2 个最可能 expert，使用独立 CUDA stream 异步执行
- 预取的 expert 不替换当前层的 LRU cache，而是暂存于共享 device buffer
- 若预测正确，该 expert 后续替换目标层 LRU cache 中最久未使用的 expert
- 与 LRU cache 正交互补：LRU 减少平均加载时间，投机预取尝试消除剩余加载延迟
- **FloE 的扩展——双预测器架构**（Section 3.3）：FloE 在投机预取基础上引入两个专门的预测器：
  1. **Inter-Expert Sparsity Predictor（学习型）**：用一个小型 MLP（32K~2M 参数，随层深自适应）预测下一层激活的 expert 索引。输入为当前层 hidden state + 历史 expert 选择轨迹，平均 precision 0.88。该 MLP predictor 相比简单 gate reuse 的优势在于可以利用跨层的历史轨迹信息。
  2. **Intra-Expert Sparsity Predictor（复用型，参数免费）**：用当前层 hidden state 与下一层复用的 W_up 矩阵直接做矩阵乘法，近似估计 up projection 输出激活，预计算下一层的稀疏掩码。平均 recall 0.95，零额外内存开销（对比学习型方法如 PowerInfer 需 9GB 额外参数）。
  3. **关键洞察**：相邻 MoE 层的 hidden state 相似度 >0.95（FloE Figure 4），使得当前层 hidden state 可以准确预测下一层的 expert 选择和稀疏分布。双预测器配合 prefetching 实现了 DRAM→VRAM 传输与 GPU 计算的流水线重叠。

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference

HOBBIT 的 Adaptive Expert Prefetching 扩展：
- **预测原理**：利用 MoE 层间 gating input 的高余弦相似度（因 Transformer 残差连接，相邻层 hidden state 高度相似）。Mixtral-8x7B 上相邻层 top-1 expert 预测准确率平均 96%，跳 2-3 层仍约 90%。
- **混合精度预取**：关键创新——即使预测错误，低精度 expert 的错误加载惩罚仅为高精度的 1/4（INT4 vs FP16），使预取在任何精度下都产生正向收益。对比：纯 FP16 预取在 Phi-MoE 上可能因误预测导致性能退化（<1.0× speedup）。
- **Stacking Computer**：将所有后续层的 gating 权重矩阵堆叠成 [N_layers_ahead, d_model, num_experts] 张量，与 hidden state 做一次矩阵乘，利用 GPU 并行性实现与单层 gating 几乎相同的计算速度。
- **预测深度**：从当前层开始逐层预测，若所有预测 expert 已在 cache 中则继续预测下一层，直到遇到 cache miss 或达到最大预测层数（建议 1-3 层）。预取的 expert 被 mask 保护不被 evict。
- 效果：prefill 阶段 latency 降低约 10%（因 prefill 激活所有 expert），decode 阶段约 5% speedup。

HarMoEny 的 Asynchronous Expert Prefetching（Section 4.3）采用了不同于基于预测的方法：
- **触发方式**：非预测驱动，而是 **rebalancing-driven**——token rebalancing 可能将 token 分配到当前不持有对应 expert 的 GPU，此时通过独立 CUDA stream 从 system memory 异步预取所需 expert 权重。
- **Overwrite-based loading**：直接覆写已完成 expert 的内存位置（无需先写回 system memory）。关键洞察——expert 权重仅需加载（推理中不变），overwrite 比 "write-back + load" 快 5.5×（11ms→2ms on V100）。
- **与计算重叠**：预取发生在当前 expert 计算期间（独立 CUDA stream），当 computation time > transfer time 时 transfer 完全被隐藏。由 token threshold q 保证（q > φ·d_type/(2β)，Section 4.4）。
- **适用条件**：要求至少 2 个 expert 可同时驻留 GPU memory（大多数 MoE serving 系统已满足）。
- 效果：在 token rebalancing 基础上进一步降低 layer latency 8.6%（Switch128）和 13.8%（Qwen）。

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU
- HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference
- HarMoEny: Efficient Multi-GPU Inference of MoE Models

## HQQ (Half Quadratic Quantization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HQQ（Half Quadratic Quantization，半二次量化）是一种 data-free 的模型权重量化算法（Badri & Shaji, 2023），无需校准数据即可将模型权重压缩到低比特（4-bit、3-bit、2-bit）。与需要校准数据的 GPTQ、AWQ 不同，HQQ 通过半二次优化直接从权重分布求解量化参数，避免了校准数据依赖带来的部署复杂性。论文选择 HQQ 仅因为其对 Mixtral 模型已有良好测试，且算法选择不影响方法的核心结论。

Mixtral offloading 论文中的 HQQ 配置：
- 4-bit: group size 64, scale group size 256（用于 attention 层）
- 3-bit: group size 64, scale group size 128（用于 expert 层）
- 2-bit: group size 16, scale group size 128（用于 expert 层，实际约 2.6 bits/param 因大量 scale/zero-point overhead）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HQQ 对 expert 权重的量化流程：
```
# 对于每个 expert 权重矩阵 W ∈ R^{M×H}:
# Step 1: 按 group size G 分组
#   W 沿行维分成 M/G 个 group，每 group 独立量化

# Step 2: 对每个 group g 的半二次优化
#   min_{W_q, s, z} ||W_g - (W_q - z) * s||²  
#   其中 W_q 为 INT2/3/4 整数权重，s 为 scale，z 为 zero point
#   通过迭代交替优化求解

# Step 3: 存储格式
#   expert_weight_int[expert_id]  # INT3 packed
#   expert_scale[expert_id]       # FP16, per group
#   expert_zero[expert_id]        # FP16, per group

# Step 4: 推理时 dequantize
#   W_fp16 = (W_int.to(fp16) - zero) * scale
#   output = input @ W_fp16  # 或 fused dequant+GEMM
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 开源实现：https://github.com/mobiusml/hqq
- Data-free 特性优势：无需准备校准数据集、无校准分布偏差风险、部署即用
- 与 GPTQ/AWQ 的关键区别：GPTQ 逐列贪心量化需校准数据（128 样本），AWQ 需校准数据确定 per-channel scaling，HQQ 纯优化求解零校准
- 论文作者表示若换用 GPTQ 或 AWQ 结论应类似（因量化选择与 offloading 策略正交）
- 子 1-bit QMoE 在 Mixtral-8x7B 上导致过大的 perplexity 退化，不适用

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Contextual Sparsification (Contextual Activation Sparsity)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Contextual Sparsification（上下文化稀疏化）是一种 training-free 的推理时激活稀疏化技术。在不重新训练模型的前提下，对每个输入 token 的激活值按**上下文相关**的幅值阈值进行剪枝：对给定输入 x 和投影矩阵 W 产生的激活向量 a(x) = x·W，仅保留 |a_i| ≥ t 的元素，将其余置零（S_t(a_i) = a_i if |a_i| ≥ t else 0）。阈值 t 不是全局常量，而是根据目标稀疏率 k（如 90%）从采样数据集的激活幅值经验 CDF 反向确定：t = min{t': F(t') ≥ k}。与全局固定阈值（如 ReLU 的 max(0,x)）或结构化的通道级剪枝不同，contextual sparsification 的剪枝模式**每个输入 token 都不同**——对当前 token 不重要的激活通道被完全跳过。

FloE 将 contextual sparsification 应用于 MoE expert 内部：基于 up projection 的输出激活 a_up = x·W_up 的幅值决定剪枝 mask，然后用该 mask 同时剪枝 W_gate 的对应列和 W_down 的对应行（转置后为列），实现计算量和传输量的双重减少。关键理论贡献：证明了在相同稀疏率下，三种激活（a_down, a_up, a_gate）剪枝的 L2 恢复误差满足 L_down ≤ L_up < L_gate，即剪枝 a_up 的误差严格小于剪枝 SiLU(gate) 输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Contextual Sparsification 的阈值确定（offline, per-expert）
// 输入: calibration dataset (e.g., C4), target sparsity k
// 输出: threshold t for each expert

for each expert E_ij in model:
    activations = []
    for each sample in calibration_data:
        h = model_forward_to_layer_i(sample)
        a_up = h @ W_up_ij                    // up projection 输出激活
        activations.extend(|a_up|.flatten())  // 收集幅值
    // 经验 CDF 反函数
    activations.sort()
    idx = int(k * len(activations))           // k=0.9 → 90% 稀疏
    t_ij = activations[idx]                   // 阈值

// 推理时的 sparse forward pass (Algorithm 1):
v = x @ W_up                                  // 全精度 up projection
mask = (|v| >= t_ij)                          // bool mask, ~10% True at 90% sparsity
x_prime = SiLU(x @ W_gate[mask]) ⊙ v[mask]    // 仅加载被选中的 gate 列
y = (W_down^T[mask] @ x_prime)^T              // 仅加载被选中的 down 列
```

FloE Figure 3(a) 的稀疏化敏感度对比（WikiText-2 perplexity, Mixtral-8×7B）：
| 稀疏率 | Down input pruning | Up output pruning | SiLU(gate) output pruning |
|--------|-------------------|-------------------|--------------------------|
| 50% | PPL +0.5% | PPL +3% | PPL +12% |
| 70% | PPL +3% | PPL +16% | PPL +40% |
| 90% | PPL +27% | PPL +77% | PPL +259% |

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 相关系统：CATS (Context-Aware Thresholding for Sparsity, Lee et al. 2024a) 是首个 training-free contextual sparsification 方法，应用于 dense LLM；TEAL (Liu et al. 2024) 扩展了训练无关激活稀疏化
- FloE 的差异化贡献：(1) 将 contextual sparsification 首次应用于 MoE expert 内部（而非 dense FFN），(2) 仅对 up projection 输出做剪枝并联动剪除 gate/down 对应通道，(3) 理论证明了 L_down ≤ L_up < L_gate 的误差排序
- 阈值由 calibration dataset 离线确定，推理时无额外计算开销
- 与量化技术正交：FloE 将 contextual sparsification（gate/down）与 INT2 量化（up）结合形成 hybrid compression
- 局限：极端稀疏率（>90%）下精度退化显著，且对 SiLU(gate) 输出的剪枝不可行（因误差太大）

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Expert Hybrid Compression (Sparsity-Quantization Hybrid)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Hybrid Compression 是 FloE 提出的 MoE expert 差异化压缩策略。与传统的 uniform 压缩（所有投影矩阵用同一量化位宽或同一稀疏策略）不同，FloE 利用 expert 内部三组投影矩阵（W_gate, W_down, W_up）对压缩方法的**差异化敏感性**，对不同矩阵采用不同压缩方法：(1) W_up 对 ultra-low-bit 量化最不敏感但在稀疏化上中等敏感——使用 INT2 HQQ 量化；(2) W_gate 和 W_down 对量化极其敏感（INT2 时 perplexity 暴涨 100×+）但对 contextual sparsification 可接受——使用基于 up projection 输出的幅值剪枝（90% 稀疏度下保留 ~10% 通道）。这种混合策略在 Mixtral-8×7B 上实现 9.3× 总体压缩比，同时将精度退化控制在 4.4%~7.6%。

设计依据：(1) 量化敏感度实验（FloE Figure 3b）显示各投影矩阵在 INT2 下的 WikiText-2 perplexity: W_down=14.36, W_gate=6.245, W_up=6.177，即 W_up 对量化最鲁棒；(2) 稀疏化敏感度实验显示各投影矩阵在 90% 稀疏度下 perplexity: W_gate(SiLU)=18.53, W_up=9.13, W_down(input)=6.55，即 W_down 对稀疏化最鲁棒；(3) 理论证明 L_down ≤ L_up < L_gate，支持选择 up 输出剪枝作为误差-效率最佳平衡点。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FloE Expert Hybrid Compression 的 DRAM 存储布局
// 每个 expert E_ij 存储:
//   W_up_ij:    INT2 packed [4096, 14336], ~3.6MB (vs FP16 ~117MB)
//   W_gate_ij:  仅保留被 sparsity mask 选中的列, FP16 [4096, 1434] (~10%)
//   W_down_ij^T: 仅保留被 sparsity mask 选中的行(转置为列), FP16 [4096, 1434]
// 总大小: ~3.6MB + 11.7MB + 11.7MB ≈ 27MB vs FP16 全量 ~351MB → 13×

// 实际压缩比:
// 稀疏 gate: 10% × FP16 = 等效 ~1.6 bits/element
// 稀疏 down: 10% × FP16 = 等效 ~1.6 bits/element
// 量化 up:  INT2 = 2 bits/element
// 总体: (1.6+1.6+2)/48 ≈ 10.8% → 9.3× 压缩 (考虑 scale/zero-point overhead)

// 推理时的 decompress + compute 流水线:
// Step 1: CPU 从 DRAM 读取 INT2 W_up + FP16 sparse W_gate[cols], W_down^T[cols]
// Step 2: AVX-512 解量化 W_up (INT2→FP16) + 打包到 pinned memory
// Step 3: 多 CUDA stream 异步传输到 GPU
// Step 4: GPU 执行 sparse GEMV kernel
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 与 Mixed MoE Quantization（attention 高精度 + expert 低精度）不同，Hybrid Compression 是 expert **内部**的差异化压缩
- W_up 选择 INT2 而非 INT1：FloE 未明确说明，但 Table 7 显示 INT1 时 W_up perplexity=520（已不可用），INT2 时仍可控（6.177）
- 稀疏化与量化引入的误差近似独立且可加（FloE Figure 9b），便于分别建模和控制
- 实现依赖：HQQ library（https://github.com/mobiusml/hqq）用于 W_up INT2 量化；Triton kernel 用于 sparse GEMV
- 通用性：FloE 在 Mixtral-8×7B, Phi-3.5-MoE, DeepSeek-V2, DeepSeek-MoE-16B, Qwen1.5-MoE 上均验证了 up projection 对量化和稀疏化的低敏感性

涉及论文标题：
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Mixed MoE Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Mixed MoE Quantization 是一种针对 MoE 模型的差异化量化策略：对不同组件使用不同的量化精度以在模型质量和内存占用之间取得最优权衡。核心发现是 MoE 模型的 expert 参数占总体参数的绝大多数（Mixtral-8x7B 中 expert 占 96.6%），但 attention 层对量化更敏感。因此最优策略是 attention 层保持较高精度（4-bit 或 FP16），expert 层可激进量化到 2-3 bit。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Mixtral-8x7B 的量化方案组合及 perplexity 对比（Table 1）：

| Attn quant | Expert quant | Model size | WikiText2 perplexity |
|------------|-------------|-----------|---------------------|
| FP16 | FP16 | 86.99 GB | 3.59 |
| FP16 | 4-bit | 25.82 GB | 3.67 |
| FP16 | 3-bit | 23.21 GB | 3.96 |
| FP16 | 2-bit | 19.33 GB | 4.52 |
| 4-bit | 4-bit | 23.99 GB | 3.76 |
| 4-bit | 3-bit | 21.37 GB | 4.05 |
| 4-bit | 2-bit | 17.54 GB | 4.61 |

论文选择的两种方案（绿色标注）：
- **4-bit attention + 3-bit experts**: 21.37 GB, Wiki2=4.05, MMLU=68.47%
- **4-bit attention + 2-bit experts**: 17.54 GB, Wiki2=4.61, MMLU=65.58%

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Embedding、logits head、MoE gate 和 normalization 层保持 FP16（参数少，对精度关键）
- 混合量化的内存计算：针对 12-16GB GPU + 8-16GB/s PCIe 带宽，模型必须压缩到可放入 host RAM 且加载延迟可接受
- 所有量化使用 HQQ 算法，但因策略与算法选择无关，可替换为 GPTQ/AWQ 等同效果

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- FloE: On-the-Fly MoE Inference on Memory-constrained GPU

## Expert Locality in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Locality 是 MoE 语言模型中观察到的一种 token 序列模式：在处理连续 token 时，模型倾向于复用部分 expert，而非每 token 随机选择全新 expert。论文对 Mixtral-8x7B-Instruct 的分析（图 1）发现两类局部性：(1) 某些 expert 在 2-4 个连续 token 上持续激活（连续复用模式）；(2) 另一些 expert 以"间隔"方式复用——在非相邻 token 之间反复出现。这种局部性是 MoE offloading 中 LRU cache 策略有效的基础——若无局部性，cache hit rate 将接近随机水平。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert locality 示例 (Mixtral-8x7B, 某层 8 experts, top-2 routing):
# Token:    t0    t1    t2    t3    t4    t5    t6    t7
# Experts:  E2,E5 E2,E5 E2,E7 E7,E3 E3,E0 E0,E1 E1,E6 E6,E2
# 
# 观察:
# - E2: t0,t1,t2 → 3 连续 token (连续复用)
# - E5: t0,t1 → 2 连续 token
# - E7: t2,t3 → 2 连续 token, 间隔复用 (t2 和 t3)
# - E3: t3,t4 → 2 连续 token
# - E0: t4,t5 → 2 连续 token
#
# LRU cache (k=2):
# t0: cache=[E2,E5], miss=2, load E2,E5
# t1: cache=[E2,E5], hit=2, 直接使用
# t2: cache=[E2,E5], hit=1 (E2), miss=1 (E7), load E7, evict=无(k=2, 仅 E2/E5 在 cache)
#     实际实现中 t2 需加载 E7 并 evict less recently used
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 论文在 OpenAssistant 对话数据上评测了不同 cache size k 的 LRU cache hit ratio（图 2 left panel）
- Cache hit ratio 随 k 增大而单调增长但边际递减
- MoE 的 expert 局部性源于专家专业化——某些 expert 学习特定语言模式（如介词、概念表达），在相关主题的连续 token 上被反复激活
- 该模式最早由 Shazeer et al. (2017) 观察到 interpretable expert specializations，论文首次将其用于 offloading 优化

涉及论文标题：
- Fast Inference of Mixture-of-Experts Language Models with Offloading
- LocMoE: A Low-overhead MoE for Large Language Model Training

**LocMoE 的 Locality Loss (局部性损失)**：

LocMoE 将 Expert Locality 从一种观察现象提升为主动优化的训练目标。其 Locality Loss $L_{loc}$ 鼓励 token 优先路由到同节点（本地）的 expert：

$$L_{loc} = \mu \cdot KL(D_c || D_l) = -\mu \int D_c(x) \ln[\frac{D_l(x)}{D_c(x)}] dx$$

其中 $D_c$ 为当前 batch 中 token 在各节点各 expert 的实际分配分布，$D_l$ 为完全局部化的理想分布（token 仅分配给本地 expert），$\mu$ 为超参数。

Localitiy Loss 与 Auxiliary Load Balance Loss ($L_{aux}$) 联合作为软约束：

$$L_{task} = L_{aux} + L_{loc} + L_{cross}$$

作用机制：
- Load balance ($L_{aux}$) 保证 token 在各 expert 间均匀分配（统计均衡）
- Locality ($L_{loc}$) 在负载均衡前提下，将跨节点 All-to-All 通信转为节点内高带宽通信（如 HCCS 256GB/s），降低 All-to-All 时间 5.13%
- 同时 locality 软约束避免 SwitchMoE 的 "winner-take-all"——更多 expert 参与早期训练

局限性：当节点数 > expert 数时（如 256N 下 16 experts 分布在 32 节点），部分节点无本地 expert，locality 策略失效，性能不如单纯负载均衡的 HashMoE。

## Gating Function / Token Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gating Function（门控函数/路由函数）是 MoE 层中决定每个 token 被分配到哪个（些）expert 的核心组件。输入为 token hidden state x ∈ R^M，输出为路由决策——包括每个 expert 的选中概率和 top-k 索引。不同路由函数在 MoE 训练稳定性和模型质量上有显著差异。FSMoE 预实现了 4 种主流路由函数：

1. **GShard Routing** (Lepikhin et al., 2020): 使用带噪声的 Top-k Gate——g(x) = softmax(KeepTopK(x·W_g + N(0,1)·Softplus(x·W_noise), k))，噪声帮助训练初期探索不同的 expert 分配。
2. **Sigmoid Routing** (Lewis et al., 2021 / BASE): 使用 sigmoid 替代 softmax，expert 输出按 σ(x·W_g)_i 缩放——若输出有益于训练目标则 gate 值增大，形成正反馈。
3. **X-MoE Routing** (Chi et al., 2022): 对 hidden state 做低秩投影 W_proj 后与 expert embedding W_g 做余弦相似度 s_i = cos(W_proj·x, W_g)，缓解表示坍缩问题。
4. **Expert Choice (EC) Routing** (Zhou et al., 2022): 从 expert 视角独立选择 top-k token，即 g(x) = softmax(KeepTopK((x·W_g)^T, k))，与 token-choice 路由对称。
5. **Top-P Routing** (Huang et al., 2024 / HMoE): 动态激活不同数量的 expert per token，而非固定 k。将 router 输出概率 P 从高到低排序，若最高概率 $P_{\max} \ge p$（threshold, e.g. 0.6），仅激活 1 个 expert；否则逐步累加直到累积概率 $\ge p$ 为止：$t = \operatorname{argmin}_k \sum_{j \le k} \tilde{P}_j \ge p$。核心优势：简单 token 可能仅需 1 个 expert（省计算），复杂 token 可激活更多 expert（保证质量），与 HMoE 的异构设计天然协同——二者都旨在按 token 复杂度差异化分配计算资源。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 Gate 子模块的抽象接口和执行流程：

```
# Gate 子模块处理流程（以 GShard routing 为例）
输入: hidden_states = [B, L, M]  # batch × seqlen × d_model
Gate 参数: W_g [M, E], W_noise [M, E]

# Step 1: 计算干净 logits
logits = hidden_states @ W_g              # [B, L, E]

# Step 2: 添加可学习噪声 (GShard 特有)
noise = randn(B, L, E) * softplus(hidden_states @ W_noise)
logits_noisy = logits + noise

# Step 3: KeepTopK + Softmax
logits_topk = KeepTopK(logits_noisy, k)   # 非 top-k 位置置 -inf
gate_probs = softmax(logits_topk)         # [B, L, E]

# Step 4: 输出路由索引和概率
topk_idx = argtopk(gate_probs, k)         # [B, L, k]
topk_prob = gather(gate_probs, topk_idx)  # [B, L, k]
```

FSMoE 的 Gate 模块支持即插即用切换——调用 `LinearGate(gate_type="gshard")` 即可选择路由函数，无需修改下游的 Order/Dispatch/Expert 模块。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Gating Function 在主流 MoE 系统中通常作为 MoE block 的一部分实现。HuggingFace Transformers 的 `MixtralSparseMoeBlock` 使用标准 softmax top-k gate。FSMoE 通过 `GateBase` 抽象基类统一各种路由实现，用户可通过继承 `GateBase` 并实现 `forward()` 方法添加新路由函数，调度器通过在线 profiler 自动适配。FSMoE 在 Testbed-B（32×RTX2080Ti）上验证 4 种路由的端到端训练时间：相比 DeepSpeed-MoE，FSMoE 在 GShard/X-MoE/Sigmoid/EC 四种路由上分别获得 1.37×/1.42×/1.37×/1.33× 加速。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Layerwise Recurrent Router for Mixture-of-Experts
- LocMoE: A Low-overhead MoE for Large Language Model Training
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

**RMoE 的 GRU-based Cross-Layer Routing**：
RMoE 引入一种全新的路由范式——跨层循环路由。与标准 router 每层独立计算 gating scores 不同，RMoE 在每层 router 前插入跨层共享的 GRU 单元（state dim p=128），将路由决策从独立逐层计算改为跨层循环依赖。核心流程：$x_i' = \mathrm{Proj}_i(x_i)$（逐层独立投影降维），$h_i = \mathrm{GRU}(x_i', h_{i-1})$（共享 GRU 结合历史路由状态），$\mathrm{score}_i = \mathrm{softmax}(h_i \cdot G_i)$（基于 GRU 输出计算 gating）。关键设计：(1) 逐层独立 Proj_i（因为不同层 hidden state 分布差异大）；(2) GRU 跨层共享以引入跨层信息；(3) GRU 额外提供 Recurrent Gradient 路径优化 router 训练；(4) 该设计正交于现有路由方法，可与 XMoE/DeepSeekMoE 无缝组合。RMoE 仅引入额外 ~3.5M 参数（相对于 0.91B 模型），训练速度降低 <1%。代码开源：https://github.com/qiuzh20/RMoE。

**LocMoE 的 GrAP (Grouped Average Pooling) Routing**：

GrAP 是一种固定正交权重的门控计算方式，替代传统可学习 Dense 层。核心思想是将 token hidden state x_m ∈ R^d 均分为 n 组（n = expert 数量），每组取均值作为对应 expert 的门控值：

$$\text{gate\_logits}_i = \text{mean}(x_m[i \cdot d/n : (i+1) \cdot d/n]), \quad i \in [0, n-1]$$

等价于固定正交权重矩阵 ω 与 x_m 的内积，其中 ω_{i,j} = 1{i·d/n ≤ j < (i+1)·d/n} else 0。

GrAP 的关键特性：
- **正交性**：不同 expert 的 gating weight 相互正交（〈ω_i, ω_j〉= 0 for i ≠ j），使不相关 token 更可能被路由到不同 expert，增强语义区分能力。正交性也是 LocMoE 理论推导 expert capacity 下界的必要前提（满足 Lemma 2：各 expert 等概率被选）。
- **计算效率**：GrAP 仅需 O(d) 的均值计算，而 Dense 层门控需要 O(d·n) 的矩阵乘法，无需可学习参数（ω 固定为 0/1 矩阵）。
- **Top-1 路由**：i* = argmax_i(softmax(gate_logits))，仅激活概率最大的 expert。

GrAP 本质是将 Dense 门控简化为空间池化，在 PanGu-Σ 的 1.085T 参数 MoE 模型上验证了与 SwitchMoE (Dense gate) 和 HashMoE (无学习参数 hash) 的对比。

**Hunyuan-Large 的 Recycle Routing**：Hunyuan-Large 使用 mixed routing strategy——1 个 shared expert（所有 token 消费）+ 16 个 specialized experts（top-1 激活）。为解决 top-1 路由中 token dropping 问题，提出 Recycle Routing：对因 expert capacity overflow 被丢弃的 token，随机重新分配到未满 capacity 的其他 specialized experts，从而保留关键信息、提升训练稳定性。相比直接丢弃，recycle routing 确保每个 token 都参与梯度更新。

Flex-MoE 提出了两种独特的 Router 设计：**G-Router (Generalized Router)** 和 **S-Router (Specialized Router)**。G-Router 在 warm-up 阶段使用全模态样本训练，遵循标准 top-k gating + load/importance balancing loss；S-Router 在 specialization 阶段通过 cross-entropy loss $L_{ce} = -\sum_j MC(x_j) \log(\max(S\text{-Router}(x_j)))$ 将 top-1 强制绑定到目标 modality combination expert index，其余 top-(k-1) expert 继续使用 load/importance balancing。这种设计使每个 expert 同时具备通用知识（来自全模态样本）和专有知识（来自特定 modality combination 样本）。

## Token Ordering (Order/I-Order) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Ordering（token 排序/重排）是 MoE 层中在 AlltoAll dispatch 之前对 token 张量进行布局变换的操作。Gate 计算完成后，每个 token 被分配到一个或多个 expert，但此时 token 仍按原始序列顺序存储。Ordering 函数将张量 layout 从 (B, L, M) 变换为 (E, T, M)，其中 T 是 expert 能处理的最大 token 数（T = k×f×B×L/E，f 为 capacity factor）。这个变换使每个 expert 的数据在内存中连续排列，便于后续 Dispatch 和 Expert 计算。

I-Ordering 是 Ordering 的逆操作——在 Expert 计算和 AlltoAll Combine 完成后，将 expert-layout 张量 (E, T, M) 恢复为原始序列 layout (B, L, M)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FSMoE 中 Order/I-Order 的执行流程：

```
# Ordering: (B, L, M) → (E, T, M)
# 输入: hidden_states=[B,L,M], gate_idx=[B,L,k], gate_prob=[B,L,k]

# GShard ordering: 使用 einsum + matmul
# 创建 sparse routing matrix R = [B*L, E] (one-hot per token)
R = scatter_nd(gate_idx, gate_prob, shape=[B*L, E])
# 将 tokens 按 expert 聚合
ordered = einsum("be,blm->etm", R, hidden_states.reshape(B*L, M))
# ordered shape: [E, T, M] where T = capacity * B*L/E

# Tutel ordering: 使用 SIMT-efficient sparse 操作
# 直接按 gate_idx 做 gather/scatter，避免 dense einsum
for expert_id in range(E):
    mask = (gate_idx == expert_id)          # [B, L, k]
    indices = mask.nonzero()                 # N tokens 的索引
    ordered[expert_id, :len(indices), :] = hidden_states[indices]

# I-Ordering: (E, T, M) → (B, L, M)
# 将 expert 计算后的结果 scatter 回原始序列位置
output.zero_()
for expert_id in range(E):
    output[indices[expert_id]] += expert_output[expert_id] * gate_prob
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FSMoE 预实现了两种 Ordering 实现：GShard ordering（einsum+matmul, 适合小规模）和 Tutel ordering（SIMT-efficient sparse ops, 适合大规模和负载不均衡场景）。Order 子模块通过 `OrderBase` 抽象，与 Gate/Dispatch/Expert 解耦，用户可替换而不影响调度器。Capacity factor f 控制每个 expert 能处理的最大 token 数——f=* 表示不丢弃 token（但可能导致显存溢出），f=1.2 表示允许 20% overfill。

**Lancet 的 Gating 约束分区范围分析**（Lancet, MLSys 2024）：

Lancet 发现 gating 方法限制了算子分区的可行范围：(1) **Switch Gate** (Fedus et al., 2022) 和 **Random Gate** (Zuo et al., 2022)：expert assignment 可从部分 batch 决定（每个 token 独立路由），因此可将分区扩展到 MoE layer 之前和之后的 non-MoE 计算（Fig. 4d）；(2) **Batch-Prioritized Routing** (Riquelme et al., 2021)：按整个 batch 内 token 的 importance score 排序后分配 expert（低分 token 先被 drop），沿 batch 维度分区会导致不同 micro-batch 的 token dropping 不同（破坏了数学等价性），因此只能扩展到 MoE layer 之后的 non-MoE 计算（Fig. 4c）。Lancet 的 DP partition range selection 自动感知 gating 类型，对无法分区 before-MoE 的 gating，P(i,n,k) 被设为 ∞。

涉及论文标题：
- FSMoE: A Flexible and Scalable Training System for Sparse Mixture-of-Experts Models
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

## Attention-MoE Pipelining

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Attention-MoE Pipelining 是 FOLDMOE 论文提出的核心创新——将 token-level 的通信-计算重叠从仅 MoE 层扩展到整个 Transformer block（同时包含 attention 层和 MoE 层）。传统 MoE-only overlapping（如 Tutel）仅在 MoE 层内部做 token-level pipelining，但 expert computation 计算量小，无法充分隐藏 A2A 通信延迟（32K seqlen 时 expert 仅占 21% 执行时间）。FOLDMOE 利用 attention 层的 O(n²) 计算量（随序列长度平方增长）覆盖 A2A 通信，将 Transformer block 重组为四级流水线：

Stage 1: Attention computation (Token micro-batch i)
Stage 2: A2A dispatch (Token micro-batch i)
Stage 3: Expert computation (Token micro-batch i)
Stage 4: A2A combine (Token micro-batch i)

通过 causal attention 的 KV cache 累积特性（计算 token t 只需前 t-1 个 token 的 K/V），可在 sequence 维度上对 attention 做微批次划分，使不同微批次的 attention 计算和 MoE 通信/计算在分离的 CUDA stream 上并行执行。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Attention-MoE Pipelining 在一个 Transformer block 内的执行伪代码：

```
# 序列 X[0..L-1], 切片方案 S={l1,...,ld}, Token Buffer B
K_prev, V_prev = [], []
start = 0

for j in 0..d-1:                         # attention 按时间均匀切片
    l_j = S[j]
    X_mb = X[start : start+l_j]
    # === Stage 1: Attention (Compute Stream) ===
    K_mb, V_mb = W_k(X_mb), W_v(X_mb)
    K_all = [K_prev; K_mb], V_all = [V_prev; V_mb]
    Z_mb = FlashAttn(Q=W_q(X_mb), K=K_all, V=V_all, causal=True)

    B.enqueue(Z_mb)                      # 存入 token buffer

    # === Stages 2-4 (Comm Stream, 可与 Stage 1 重叠) ===
    while B.size >= ceil(L/d):           # MoE 侧按 token 数量均匀取
        Z_moe = B.dequeue(ceil(L/d))
        Z_disp = A2A_dispatch(Z_moe)     # Stage 2
        Y_exp = Experts(Z_disp)          # Stage 3
        Y_moe = A2A_combine(Y_exp)       # Stage 4
        Y.append(Y_moe)

    K_prev, V_prev = K_all, V_all
    start += l_j

# Drain buffer (cool-down phase)
while B not empty:
    ...  # 同上 Stages 2-4
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FOLDMOE 基于 Megatron-LM 框架实现，修改了 Transformer block 的执行流程。与 FlashAttention 兼容（因 micro-batch causal attention 与全序列 causal attention 的 mask pattern 一致），与 TP 正交（TP 切分算子，FOLDMOE 切分序列），与 SP 兼容（SP 仅操作 layernorm/dropout 等非 attention/MoE 区域）。配置参数为 overlap degree d（微批次数量），通过 runtime profiling 确定最优 d（d 过小则 bubble 大，过大则 kernel launch overhead 超重叠收益）。论文未开源代码。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

## Causal Attention / Masked Self-Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Causal Attention（因果注意力，也称为 Masked Self-Attention 或 Autoregressive Attention）是 decoder-only Transformer 的核心注意力机制，确保每个 token 只能 attend 到自身及之前的 token，不能看到未来 token。对于位置 t 的 token x_t，其查询向量 q_t 只能与位置 1..t 的键向量 k_i 和值向量 v_i 交互：

$$Attn(x_t; x_1, ..., x_{t-1}) = \sum_{i=1}^{t} softmax(\frac{q_t^T k_i}{\sqrt{d_k}}) v_i$$

这种因果约束是实现自回归语言建模（autoregressive language modeling）的关键——模型逐 token 预测下一个 token：P(x_t | x_1, ..., x_{t-1})。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Causal attention 的关键性质——token t 只依赖前 t-1 个 token 的 K/V——是 FOLDMOE 能实现 attention-MoE pipelining 的基础。具体来说：
- 将序列切分为微批次 X_{1:m}, X_{m+1:2m}, ...
- 计算 X_{m+1:2m} 的 attention 时，只需已缓存的 K_{1:m}, V_{1:m} 加上自身的 K_{m+1:2m}, V_{m+1:2m}
- 这使得 attention 层可以沿 sequence 维度流水线化，在计算后续微批次的同时，前序微批次已可进入 MoE 层的 A2A 通信

```
# 微批次间的 KV 累积
for mb in micro_batches:
    K_mb, V_mb = proj_kv(X_mb)
    K_cache = concat(K_cache, K_mb)   # 逐步累积
    V_cache = concat(V_cache, V_mb)
    Z_mb = attention(Q_mb, K_cache, V_cache, causal_mask)  # 只 attend 到前缀
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Causal attention 的标准实现方式：
1. **Naive**: 计算完整 N×N attention 矩阵后应用上三角 mask（设为 -∞），O(n²) 内存
2. **FlashAttention**: fused kernel，分 tile 计算，IO-aware，将 softmax 在线计算融入 tile 循环，避免物化完整 attention 矩阵
3. **PagedAttention (vLLM)**: 用于推理的 KV cache 管理，将 KV cache 分页存储

在 FOLDMOE 中，使用 FlashAttention 作为 attention 实现，每个 micro-batch 内的 causal attention 计算与原全序列 causal attention 产生相同的输出（因 mask pattern 一致）。

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

## GShard Gate / Top-1 Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GShard Gate（来自论文 GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding, Lepikhin et al., 2020）是 MoE 中最常用的门控路由机制。Gate 是一个可学习的线性变换 W_g，将每个 token 的 d_model 维表示映射到 |E| 维（专家数量），经 softmax 后得到该 token 对各专家的亲和度分数。Top-1 routing 选择分数最高的专家处理该 token。Top-k routing 选择分数最高的 k 个专家。

$$g(x) = softmax(W_g \cdot x), \quad \tau = top\text{-}k(g(x))$$

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 FOLDMOE 使用的 GPT-MoE 模型中：
- 使用 top-1 GShard gate（k=1，每个 token 只路由到一个专家）
- 每隔一个 Transformer block 替换 FFN 为 MoE 层（alternating pattern）

```
# MoE layer forward with top-1 GShard gate
def moe_layer_forward(x):  # x: [num_tokens, d_model]
    gate_logits = Linear(d_model, num_experts)(x)  # [num_tokens, num_experts]
    gate_probs = softmax(gate_logits, dim=-1)
    expert_idx = argmax(gate_probs, dim=-1)         # top-1

    # Expert Capacity 约束
    for e in 0..num_experts-1:
        tokens_for_e = x[expert_idx == e][:capacity]  # 截断到 capacity
        if len(tokens_for_e) > 0:
            output[expert_idx == e] = expert_e(tokens_for_e) * gate_probs[e]

    return output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GShard gate 是 MoE 训练的标准选择：
- **Auxiliary loss**: 除主任务 loss 外，通常加辅助 load balancing loss 鼓励 token 均匀分配到各专家，避免某些专家过载或闲置
- **Expert capacity**: 限制每个专家每步最多处理的 token 数（Capacity = CF * B * L / |E|），超出部分被丢弃（token dropping）或用 residual connection 绕过
- **Top-2 routing**: 某些模型（如 Mixtral 8x7B）使用 top-2 gate，每个 token 路由到 2 个专家，增加模型容量但增加计算和通信开销
- FOLDMOE 使用 top-1 + capacity factor=1.0，EP 为每个 GPU 分配 1 个专家

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining

## Expert Capacity / Capacity Factor

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Capacity（专家容量）是 MoE 训练中控制每个专家每步最多处理 token 数量的约束机制。定义为：

$$Expert\ Capacity = CF \cdot \frac{B \cdot L}{|\mathcal{E}|}$$

其中 CF 是 Capacity Factor（容量因子），B 是 batch size，L 是序列长度，|E| 是专家总数。CF=1.0 表示每个专家容量等于均匀分配时的期望 token 数。CF>1 提供冗余容量以减少 token dropping，但增加内存和计算开销。超出容量的 token 被丢弃（token dropping），即那些 token 跳过该 MoE 层的专家计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Expert Capacity 在 MoE 层 forward 中的作用：

```
def moe_forward_with_capacity(x, gate, experts, capacity_factor=1.0):
    B, L, D = x.shape
    num_experts = len(experts)
    capacity = int(capacity_factor * B * L / num_experts)

    gate_scores = gate(x)           # [B*L, num_experts]
    expert_idx = argmax(gate_scores, dim=-1)  # top-1

    # 每个专家的 token 计数器
    expert_counts = zeros(num_experts)
    output = zeros_like(x)

    for t in range(B * L):
        e = expert_idx[t]
        if expert_counts[e] < capacity:
            output[t] = experts[e](x[t]) * gate_scores[t, e]
            expert_counts[e] += 1
        # else: token dropped (output[t] remains 0, or use residual)

    return output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- CF 的选择是 memory vs. quality 的 trade-off：CF=1.0 节省内存但可能有较多 token dropping（尤其当 gate 路由不均衡时）；CF=1.25-2.0 更安全但增加计算和通信
- **Auxiliary load balancing loss**: 通常与 capacity 协同使用——loss 惩罚路由不均衡，capacity 作为硬件约束的硬上限
- FOLDMOE 使用 CF=1.0，配合 EP=16（每 GPU 1 个专家），在保证训练收敛（通过 Figure 12 验证 loss curve 一致性的同时）最小化内存
- Joint MoE Scaling Laws 在评估阶段使用 dropless 模式（CF→∞ / 移除 capacity 限制），确保所有 token 均被处理以避免 capacity-induced dropping 影响 loss 评估

涉及论文标题：
- FOLDMOE: Efficient Long Sequence MoE Training via Attention-MoE Pipelining
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Llama 3 Meets MoE: Efficient Upcycling
- LocMoE: A Low-overhead MoE for Large Language Model Training
- Lancet: Accelerating Mixture-of-Experts Training via Whole Graph Computation-Communication Overlapping

**Lancet 的不规则 Expert Capacity**（Lancet, MLSys 2024）：

Lancet 提出了 expert capacity 在 micro-batch 分区场景下的不规则使用机制。当沿 batch 维度将 MoE layer 的输入分为 k 个 micro-batch 时，直接等比例缩小每个 micro-batch 的 expert capacity（C/k）会导致额外 token dropping——因为 token 分布不均匀（如第一个 micro-batch 有 3/4 C 个 token 指向某 expert，超过了 C/k）。Lancet 实现 special gating operator 在各 partition 间传递容量信息：第一个 partition 使用多少容量后，调整剩余 capacity 传给后续 partition。保证所有 partition 的 token-to-expert mapping 和 token dropping 与不分区的原版完全一致（数学等价性）。但这导致每个 partition 可以向每个 expert 发送 0 到 C 之间任意数量的 token，引入不规则 all-to-all 通信（Irregular All-to-Allv）。

**LocMoE 的 Expert Capacity 下界理论**：

LocMoE 首次将 pMoE (Chowdhury et al., 2023) 在 CV 领域的 expert capacity 下界结论推广到 NLP 领域，并结合网络结构分析：

前提假设：
1. Gating weight 范数 ‖ω_i‖ 对所有 expert 等价
2. Token 均匀分布在高维单位球面 (‖x_m‖ = 1)
3. GrAP 的正交 gating weight 满足 Lemma 2：各 expert 被等概率选择 P{i_j = i'} = 1/n

基于高维球面几何推导：
- token 应分配给 expert i 的概率 $p_\delta = 1 - I_{\delta^2}(1/2, (d-1)/2)$，其中 $\delta = \cos(\theta)$ 为 token 与 gating weight 夹角余弦
- 当 d 很大且 $\delta = \Theta(1/\sqrt{d})$ 时，$p_\delta \approx 0.3$
- 当 $\delta$ 增大（token 与 expert 更匹配，夹角变小），$p_\delta$ 快速衰减至 0——仅少量 token 为 class-discriminative

由此得到 expert capacity 下界：

$$ec_{min} \ge \frac{1}{n \cdot \operatorname{erfc}(\sqrt{\frac{\delta^2 d}{2 - \delta^2}})} > \frac{1}{n} \exp(\frac{\delta^2 d}{2 - \delta^2})$$

实验测得 δ ≈ 0.03，可据此计算安全的 expert capacity 下界，在保证模型精度前提下降低 capacity 以减少冗余计算。

**Llama 3 Meets MoE 对 Capacity Factor 的扩展分析**：

论文通过 CF∈{1, 2, 4, Dropless} 消融实验（Table 4）量化了 CF 对 training MFU 和 downstream accuracy 的 trade-off：
- CF=1: MFU=46.8%（最高），MMLU 0-shot=63.7
- CF=2: MFU=39.2%，MMLU 0-shot=64.0（最高 accuracy）
- CF=4: MFU=39.4%，MMLU 0-shot=63.5（最终主实验选择）
- Dropless (CF=∞): MFU=39.6%，MMLU 0-shot=63.3

关键发现：(1) CF 隐式引入了正则化——Dropless 的 MMLU 准确率反而不及 CF=2/4，因为缺少 token dropping 的正则化效果；(2) CF=1 时 MFU 显著高于高 CF，因为更少 token 被处理且内存 footprint 更小，允许更小的模型并行度；(3) CF=4 为 accuracy-MFU 最佳平衡点。

训练配置补充：主实验使用 100B tokens, bfloat16, 512 H100 GPU, CF=4。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FO-MoE（Fairness-Oriented Mixture of Experts）是 Fair-MoE 论文提出的面向公平性的 MoE 架构变体，专为医疗 Vision-Language Model 设计，用于过滤偏置 patch embedding 并提取公平的任务相关特征。FO-MoE 包含两级 MoE：

1. **Embedding-based MoE**：替换图像和文本 encoder 最后一个 attention block 中的 MLP 层。输入为所有 patch embeddings I^1 ∈ R^(N+1)×D（N 个 patch + 1 个 [CLS] token）。Gate 输出 W^1 = softmax(G^1(I^1))，然后通过两级稀疏化：Ŵ^1 = Top_c(Top_r(W^1, k^1), α)。Top_r 保留每行（每个 patch）权重最高的 k^1 个 expert；Top_c 通过 expert capacity C 限制每列（每个 expert）可处理的 patch 数，仅保留 α = C(N+1)k^1/M^1 个最高权重。被清零的权重对应的 expert 输出被丢弃，实现**偏置 patch 的主动过滤**——包含敏感属性信息（如肤色、性别特征）的 patch 对应的 expert 输出权重被清零。

2. **Feature-based MoE**：放置在 encoder 之后，取 [CLS] token 对应的特征向量 I^2_0 ∈ R^D 作为输入，通过 M^2 个 experts 做进一步 sparse gating：W^2 = Top_r(softmax(G^2(I^2_0)), k^2)。最终公平特征 I^3 = Σ_{b=0}^{M^2-1} Ŵ^2_b · E^2_b(I^2_0)。

Expert 结构为标准两层 MLP：E_b(x) = T̃_b · σ(W̃_b · x)，其中 σ 为激活函数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

在 Fair-MoE 的 CLIP-based pipeline 中，FO-MoE 的图像侧前向流程：

```
# 图像侧 FO-MoE 流程
I_image = ViT_patch_embed(fundus_image)       # (N+1)×D
I_enc = attention_blocks[0..K-2](I_image)     # 前 K-1 个 block

# === Embedding-based MoE (替换最后一个 attention block 的 MLP) ===
I^1 = I_enc                                    # 输入
W^1 = softmax(G^1(I^1))                       # Gate: R^{(N+1)×D} → R^{(N+1)×M^1}
Ŵ^1 = Top_r(W^1, k^1)                         # 保留每行 top-k^1 权重
Ŵ^1 = Top_c(Ŵ^1, α)                           # capacity filtering: 每列仅保留 α 个
# α = C·(N+1)·k^1 / M^1, C 为 expert capacity
I^2_a = Σ_{b=0}^{M^1-1} Ŵ^1_{a,b} · E^1_b(I^1_a)  # 加权聚合各 expert 输出

# === [CLS] token 作为特征向量 ===
I_feat = I^2_0                                # R^D

# === Feature-based MoE (encoder 之后) ===
W^2 = Top_r(softmax(G^2(I_feat)), k^2)        # R^{M^2}, 保留 top-k^2
I^3 = Σ_{b=0}^{M^2-1} Ŵ^2_b · E^2_b(I_feat)   # Fair image feature

# 文本侧对称执行相同流程 → T^3 (Fair text feature)

# 对比学习损失
similarity = cosine(I^3, T^3)
L = contrastive_loss(similarity) + FOL
```

**与标准 Sparse MoE 的关键区别**：
- 标准 MoE 的 Top_c 仅用于 load balancing（防止个别 expert 过载），Fair-MoE 的 Top_c 用于**公平性过滤**——通过 capacity 约束使偏置 patch 的 expert 权重被清零
- 两级 MoE 设计：patch 级（过滤空间偏置）+ feature 级（过滤语义偏置）
- 同时应用于图像和文本两侧 encoder

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FO-MoE 基于标准 CLIP (ViT-B/16 或 ViT-L/14) 架构修改。实现方式：在 PyTorch 中替换 CLIP encoder 最后一个 Transformer block 的 MLP 为 embedding-based MoE，并在 encoder 输出后插入 feature-based MoE。Gate 使用标准线性层 + Softmax，sparse 操作通过 mask tensor 实现（单 GPU 训练，无 EP 通信开销）。论文代码开源在 https://github.com/LinjieT/Fair-MoE-Medical-Fairness-Oriented-Mixture-of-Experts-in-Vision-Language-Models。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

## Fairness-Oriented Loss (FOL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FOL（Fairness-Oriented Loss）是 Fair-MoE 论文提出的公平性损失函数，创新地将 MoE load balance 中使用的方差（variance）度量同时用于公平性优化。FOL 由五个组件组成：

$$FOL = F_{EI} + F_{ET} + F_{FI} + F_{FT} + L_{distance}$$

其中：
- **F_EI**：图像 embedding-based MoE 的方差损失
- **F_ET**：文本 embedding-based MoE 的方差损失
- **F_FI**：图像 feature-based MoE 的方差损失
- **F_FT**：文本 feature-based MoE 的方差损失
- **L_distance**：Sinkhorn distance loss（继承自 FairCLIP）

以 F_EI 为例，核心公式：

$$F_{EI} = \sum_{p \in P} \sum_{j=0}^{M^1-1} (Var(O_{N_j}) - Var(O_{N|p_j}))^2$$

其中 O_N 是从整个数据集采样的 N 个样本的 gate weight 矩阵（所有 expert 的权重），O_{N|p} 是从特定受保护属性组 p 采样的 gate weight 矩阵，Var(·) 计算每列（每个 expert）的方差，P 是某属性的所有组集合（如 race 的 {White, Black, Asian}）。

**核心设计思想**：FOL 同时优化两个维度的公平性：(1) L_distance 最小化不同属性组分布之间的**距离**（位置对齐）；(2) 四个方差项最小化不同属性组分布的**离散度差异**（形状对齐，即方差对齐）。方差优化同时服务于 load balancing（防止 MoE 训练中 expert 使用不均衡导致的训练不稳定），从而让 Fair-MoE 能更好地利用 MoE 的学习能力提取公平特征。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

FOL 的计算流程：

```
# 输入: 训练一个 batch 的数据 (image, text, protected_attr_labels)
# 对于四个 MoE 模块分别计算方差差异

# 以图像 embedding-based MoE 为例:
# 从全数据集和每个属性组分别累积 gate weights
O_N = []      # 全数据集的 weights
O_N_race_0 = []  # race=White 的 weights
O_N_race_1 = []  # race=Black 的 weights
# ... 类似地累积其他属性组

for batch in dataloader:
    W^1 = FO_MoE_image_emb.gate(batch.images)  # gate weights
    O_N.append(W^1)
    for p in protected_groups:
        mask = (batch.attr == p)
        O_N_p.append(W^1[mask])

# 计算 F_EI
F_EI = 0
for p in protected_groups:        # 遍历每个属性组
    for j in range(M^1):          # 遍历每个 expert
        var_all = Var(O_N[:, j])
        var_group = Var(O_N_p[:, j])
        F_EI += (var_all - var_group)^2

# 类似地计算 F_ET, F_FI, F_FT
# L_distance 使用 Sinkhorn distance (最优传输距离)
# 最终 FOL = F_EI + F_ET + F_FI + F_FT + L_distance
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FOL 在训练过程中作为辅助 loss 与 CLIP 的对比学习 loss 联合优化：L_total = L_CLIP + λ · FOL。方差通过 PyTorch 的 `torch.var()` 在累积的 gate weight 矩阵上计算（需要采样足够多数据以获得稳定的方差估计）。FOL 适用于任何使用 MoE 架构且需要考虑公平性的场景，特别是医疗影像分析中多个受保护属性（race, gender, ethnicity, language）共存的情况。消融实验证明：移除 FOL 导致 Race AUC 下降 2.56%，Gender ES-AUC 下降 2.34%，验证了方差优化对公平性和有效性的双重贡献。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

## Sinkhorn Distance for Fairness

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sinkhorn Distance（Sinkhorn 距离）是熵正则化的最优传输（Optimal Transport, OT）距离。标准 Wasserstein 距离的最优传输问题求解复杂度为 O(n³ log n)，Cuturi (2013) 提出通过添加熵正则化项 H(P) 使问题变为强凸，从而可通过 Sinkhorn 算法（迭代矩阵缩放）高效求解，复杂度降至 O(n²)。在机器学习公平性领域，Sinkhorn distance 被用于衡量和最小化不同受保护属性组（如不同种族、性别）的特征分布之间的差异——将公平性问题建模为最优传输问题：寻找将一组分布传输到另一组的最小代价方案，用 Sinkhorn distance 作为 fairness regularization term。

数学定义：

$$Sinkhorn_{\epsilon}(\mu, \nu) = \min_{P \in \Pi(\mu,\nu)} \langle P, C \rangle + \epsilon \sum_{i,j} P_{ij} \log P_{ij}$$

其中 μ, ν 为两组分布，C 为代价矩阵，ε 控制正则化强度，P 为传输计划矩阵。通过 Sinkhorn 算法迭代更新缩放因子 u, v 使 P 满足行/列边际约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

在 FairCLIP 和 Fair-MoE 中，Sinkhorn distance 用于在对比学习 loss 层面实现公平性：

```
# Sinkhorn distance 在 fairness loss 中的使用
# 输入: 两组样本的 embeddings Z_a, Z_b (来自属性 a 和 b)

# Step 1: 计算代价矩阵 C (通常用 cosine distance)
C[i][j] = 1 - cosine_similarity(Z_a[i], Z_b[j])

# Step 2: Sinkhorn 算法迭代
K = exp(-C / epsilon)          # Gibbs kernel
u = ones(n) / n                # 初始化缩放因子
v = ones(m) / m
for t in 1..T:
    u = a / (K @ v)            # a, b 为边际分布 (通常均匀)
    v = b / (K^T @ u)
P = diag(u) @ K @ diag(v)      # 最优传输计划

# Step 3: Sinkhorn distance = ⟨P, C⟩
L_sinkhorn = sum(P * C)
```

在 Fair-MoE 的 FOL 中，L_distance 即为 Sinkhorn distance，用于最小化不同受保护属性组特征分布之间的距离，与方差优化项互补。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sinkhorn distance 通过 Python OT 库（如 POT: Python Optimal Transport）或 PyTorch 自定义实现。在 fairness 应用中通常作为辅助 loss 项：L_total = L_task + λ · L_sinkhorn。FairCLIP 使用 Sinkhorn distance 作为唯一的 fairness constraint。Fair-MoE 将其保留为 FOL 的一个子项，同时引入方差优化项以增强公平性。适用于需要对齐不同组特征分布的场景（如医疗 VLMs、面部识别、推荐系统）。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

## Demographic Parity Difference (DPD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Demographic Parity Difference（DPD，人口均等差异）是机器学习公平性的基础度量指标，衡量不同受保护属性组（如不同种族、性别）获得正向预测结果的概率差异。DPD 越小表示模型越公平（理想值为 0）。

在 Fair-MoE 论文中，对于受保护属性 s 的所有组 a 和 b：

$$DPD_s = |\max_a P(\hat{y}=1|G=a, y=1) - \min_b P(\hat{y}=1|G=b, y=1)|, \quad a \neq b$$

其中 ŷ=1 表示正向预测（诊断患病），G=a 表示属于属性组 a，y=1 表示真实患病。DPD 关注的是"在真实患病的人群中，不同组获得正确诊断的概率是否一致"。DPD < 0.1 通常被认为公平。

在 Harvard-FairVLMed 青光眼诊断任务中，CLIP/b16 的 Race DPD=14.57（高度不公平），FairMoE/l14 的 Race DPD=2.63（接近公平）。DPD 适用于任何存在受保护属性的分类任务，是 most widely used group fairness metric。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

DPD 在模型评估 pipeline 中的计算（以二分类医学诊断为例）：

```
# 输入: predictions ŷ, true labels y, protected attribute groups G
def compute_DPD(y_pred, y_true, groups):
    # 仅考虑真实患病人群 (y=1)
    positive_mask = (y_true == 1)
    group_probs = {}
    for g in unique(groups):
        group_mask = positive_mask & (groups == g)
        group_probs[g] = mean(y_pred[group_mask] == 1)
    # DPD = 最大组概率 - 最小组概率
    return max(group_probs.values()) - min(group_probs.values())
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DPD 在 FairLearn（Microsoft）、AIF360（IBM）、FairTorch 等公平性工具包中均有标准实现。在训练中，DPD 通常不作为直接优化目标（不可微），而是作为评估指标或通过代理 loss（如 adversarial debiasing、contrastive fairness loss）间接优化。Fair-MoE 未直接优化 DPD，而是通过 FOL 优化 gate weight 的方差差异，间接降低 DPD。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

## Equal Opportunity Difference (EOD)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Equal Opportunity Difference（EOD，机会均等差异）是比 DPD 更细粒度的公平性度量，同时考虑真正例率（TPR）和假正例率（FPR）在不同受保护属性组之间的差异。EOD 是 Equalized Odds 的差异版本，确保模型在各组上的分类错误类型分布一致。

在 Fair-MoE 论文中：

$$EOD_s = \max_{a,b \in s, a \neq b} (|P(\hat{y}=1|G=a,y=1)-P(\hat{y}=1|G=b,y=1)|, |P(\hat{y}=1|G=a,y=0)-P(\hat{y}=1|G=b,y=0)|)$$

第一项为 TPR 差异（真实患病者中获得正确诊断的概率差异），第二项为 FPR 差异（健康人中被误诊的概率差异）。取两者中的最大值。EOD 比 DPD 更严格，因为它要求模型在真正例和假正例两个维度上都公平。

在 Harvard-FairVLMed 上，CLIP/b16 的 Race EOD=18.47，FairMoE/l14 的 Race EOD=4.25（↓77%）。EOD 特别适用于医疗诊断场景——确保模型不会系统性地对某组产生更多假阳性（过度诊断）或更多假阴性（漏诊）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
def compute_EOD(y_pred, y_true, groups):
    tpr = {}  # True Positive Rate per group
    fpr = {}  # False Positive Rate per group
    for g in unique(groups):
        mask = (groups == g)
        pos = mask & (y_true == 1)   # 该组真实患病
        neg = mask & (y_true == 0)   # 该组真实健康
        tpr[g] = mean(y_pred[pos] == 1)
        fpr[g] = mean(y_pred[neg] == 1)
    max_tpr_diff = max(tpr.values()) - min(tpr.values())
    max_fpr_diff = max(fpr.values()) - min(fpr.values())
    return max(max_tpr_diff, max_fpr_diff)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

EOD 在 FairLearn、AIF360 中作为标准指标实现。与 DPD 相同，EOD 通常是评估指标而非直接优化目标。在医疗 AI 公平性研究中，EOD 是最受关注的指标之一——因为它能同时暴露过度诊断（高 FPR）和漏诊（低 TPR）的组间不公平。Fair-MoE 通过 FO-MoE 和 FOL 间接优化 EOD。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

## Equity-Scaled AUC (ES-AUC)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Equity-Scaled AUC（ES-AUC，公平缩放 AUC）是哈佛医学院团队提出的性能-公平性联合度量指标，用于量化模型在整体性能和组间公平性之间的权衡。与单独使用 AUC（仅衡量性能）或 DPD/EOD（仅衡量公平性）不同，ES-AUC 将两者统一为单一指标。

定义（Fair-MoE 论文中的表述）：

$$ES\text{-}AUC_s = \frac{AUC_s}{1 + \sum_a |AUC_s - AUC_{s,a}|}$$

其中 AUC_s 是属性 s 上的整体 AUC，AUC_{s,a} 是属性 s 中组 a 的组内 AUC。分母中的惩罚项 Σ|AUC_s - AUC_{s,a}| 衡量各组 AUC 偏离整体 AUC 的程度——偏离越大，惩罚越重，ES-AUC 越低。当所有组 AUC 完全相等时，ES-AUC = AUC（无惩罚）；各组差距越大，ES-AUC 衰减越多。

在 Fair-MoE 论文中，ES-AUC 是 primary evaluation metric（因它同时衡量 effectiveness 和 fairness）。FairMoE/l14 在 Race 上 ES-AUC=72.53（+5.00% vs FairCLIP/l14）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
def compute_ES_AUC(y_scores, y_true, groups):
    """
    y_scores: 模型预测分数 [N]
    y_true: 真实标签 [N]
    groups: 受保护属性组标签 [N]
    """
    from sklearn.metrics import roc_auc_score
    
    # 整体 AUC
    auc_overall = roc_auc_score(y_true, y_scores)
    
    # 各组 AUC
    auc_groups = {}
    for g in unique(groups):
        mask = (groups == g)
        auc_groups[g] = roc_auc_score(y_true[mask], y_scores[mask])
    
    # 惩罚项 = 各组 AUC 偏离整体 AUC 的绝对值之和
    penalty = sum(abs(auc_overall - auc_g) for auc_g in auc_groups.values())
    
    # ES-AUC
    es_auc = auc_overall / (1 + penalty)
    return es_auc
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

ES-AUC 最初在 Fair Identity Normalization (FIN) 和 Fair Adaptive Scaling (FAS) 论文中提出，用于青光眼和糖尿病视网膜病变筛查。ES-AUC 的优势在于它直接反映"提升整体性能是否会以牺牲某组性能为代价"——如果某组性能下降，即使整体 AUC 提升，ES-AUC 也会因惩罚项增大而不升反降。这使得 ES-AUC 成为多组公平性场景下的首选联合评估指标。Fair-MoE 在所有受保护属性和 backbone 上均以 ES-AUC 为主要对比指标。

涉及论文标题：
- Fair-MoE: Fairness-Oriented Mixture of Experts in Vision-Language Models

## MoE Data Shuffling (Token Dispatch-Combine)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Data Shuffling 是 Expert Parallelism 中由 token-to-expert routing 触发的全局数据重排过程。在 EP 下专家分布在不同 GPU 上，每个 token 经 router 分配给 top-k 专家后经历完整的 shuffle pipeline：token 按 destination rank 重排（permute）→ all-to-all 跨设备交换（dispatch）→ 按 expert layout 再次重排 → expert FFN 计算 → 对称的反向 all-to-all（combine）→ 恢复原始 token order。FUSCO 的 profiling 显示 shuffling 占端到端运行时的 22%–61%（随 EP degree 增长），其中 rearrangement（permute/repack）占 intra-node 延迟的 68.8%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 传统 MoE Data Shuffling (NCCL baseline，per MoE layer):
# Stage 1: permute by rank → Stage 2: A2A dispatch
# → Stage 3: permute by expert → Stage 4: FFN
# → Stage 5: inverse permute → Stage 6: A2A combine → Stage 7: inverse permute
# 共 5 次 memory passes（permute 各 read+write） + 2 次网络传输

# FUSCO fused approach:
descriptors = planner.build(token_expert_matrix)  # 一次性构建两级 descriptor
dispatched = dcomm.dispatch(tokens, descriptors)  # gather→ringbuf→RDMA，一步完成
expert_outputs = experts(dispatched)              # 直接消费已排列好的数据
output = dcomm.combine(expert_outputs, descriptors)  # 对称反向 fused 操作
# 仅 1 次 memory pass + 1 次 pipelined 网络传输（per dispatch/combine）
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Expert-major layout（模型执行所需）和 device-major layout（通信操作所需）的 mismatch 是 root cause——每次 all-to-all 都需要一对对称的逆排列。Token size 4-14KB，足够大以 amortize per-unit transformation cost。FUSCO 通过 fused approach 消除所有显式 permute，在 16K seqlen real-world traffic 下比 NCCL 快 1.66×。

涉及论文标题：
- FUSCO: High-Performance Distributed Data Shuffling via Transformation-Communication Fusion

## FarSkip-Collective

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FarSkip-Collective 是一种修改 MoE Transformer 模型残差连接（residual connectivity）的架构方法，通过在通信进行期间使用"过时"（outdated）或"部分"（partial）激活值作为下一子块的输入，消除分布式执行中的阻塞通信模式。核心思想是：标准 Transformer 中 `o_k = o_{k-1} + f_k(o_{k-1})`，即下一子块 `f_{k+1}` 必须等待 `f_k` 的完整输出（包括其通信结果）。FarSkip 改为 `o_k = o_0 + f_1(o_0) + f_2(o_1^*) + ... + f_k(o_{k-1}^*)`，其中 `o_k^*` 是不依赖当前子块通信结果的可用激活值。

两种 `o_k^*` 选择：
- **(8a) Outdated**：`o_k^* = o_{k-1}`，使用上一层的完整输出
- **(8b) Partial**：`o_k^* = o_{k-1} + f_k^*(o_{k-1}^*)`，使用当前子块中不依赖通信的部分计算结果

对于 MoE 层的具体应用：
- Attention 子块输入（partial）：`attn-in_k = o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1}`（省略 routed-exp-out_{k-1}），使 Combine 通信可与 Attention 重叠
- MLP 子块输入（outdated）：`mlp-in_k = o_{k-1}`，使 Dispatch 通信可与 Attention 重叠

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FarSkip-Collective 修改后的 MoE 层前向执行（训练，EP=8）：

```
# 原始 MoE 层前向:
# attn_out = Attention(LN(o_{k-1}))
# o_k_attn = o_{k-1} + attn_out
# gate = Router(LN(o_k_attn))
# dispatched = AllToAllDispatch(o_k_attn, gate)  ← 阻塞通信气泡
# routed_out = RoutedExperts(dispatched)
# combined = AllToAllCombine(routed_out)           ← 阻塞通信气泡
# o_k = o_k_attn + SharedExperts(o_k_attn) + combined

# FarSkip-Collective 前向:
# 1. MLA q,k,v 准备 (attn-in_k 使用 partial activation)
q, k, v = MLA_prepare(o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1})

# 2. 同步上一层的 Combine (此时 Combine 已被重叠)
WaitCombineHandle(prev_combine_handle)

# 3. MoE gating
gate = Router(LN(o_{k-1}))

# 4. 异步 Dispatch (async_op=True, 立即返回)
dispatch_handle = AllToAllDispatchAsync(tokens, gate)

# 5. Core attention + output projection (与 Dispatch 重叠!)
attn_out = MLA_core(q, k, v)

# 6. 同步 Dispatch
WaitDispatchHandle(dispatch_handle)

# 7. Routed experts
routed_out = RoutedExperts(dispatched_tokens)

# 8. 异步 Combine (与 shared experts 重叠)
combine_handle = AllToAllCombineAsync(routed_out)
shared_out = SharedExperts(o_{k-1})  # 与 Combine 并行
```

重叠窗口条件（Eq. 9）：`T_Dispatch + T_Combine ≤ T_layer - (T_RoutedExperts + T_Gate)`

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **训练侧**：在 Megatron-LM 中实现，使用 `torch.dist.all_to_all(async_op=True)` 异步启动通信，通过 backward hook 和 Sequence Number hijacking 实现反向传播的通信重叠。前向重叠率 87.6%-92.9%，反向重叠率 84.1%-89.0%。
- **推理侧**：在 vLLM/SGLang 中实现，将 EP all-reduce 改为 `async_op` 模式，通过 CUDA Stream 分离通信与计算，使用 PyNCCL 兼容 CUDA graphs。All-reduce 重叠率 95.3%-97.6%。
- **适用范围**：任何 MoE 模型（训练和推理），不改变模型参数形状，仅修改连接性。与 TP、PP、DP 正交兼容。
- **限制**：routed experts 和 gating 的计算不可重叠（它们依赖通信的输入/输出），是重叠窗口的下界。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

## FCSD (FarSkip-Collective Self-Distill)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FCSD（FarSkip-Collective Self-Distill）是将预训练 MoE 模型转换为 FarSkip-Collective 架构连接性的知识蒸馏方法。以原始模型（未修改连接性）作为 teacher，FarSkip 修改后的模型作为 student，使用 KL 散度 loss 训练约 10B tokens 即可恢复原始模型约 97.5-99% 的下游任务准确率。直接加载原始权重到 FarSkip 架构中会导致性能崩溃（MMLU 降至随机基线，HumanEval+ 降至 0%），因为模型接收到的输入激活值分布与训练时完全不同（OOD）。

FCSD 配方：
- **Loss**：`L_KD(θ) = E_x[Σ_t KL(q(·|x, y_<t) || p_θ(·|x, y_<t))]`，以原始模型 q 为 teacher
- **优化器**：AdamW + cosine-annealing LR scheduler + 1000-step warmup
- **Batch-size**：从 {2^16, 2^17, 2^18} 中 sweep 选择
- **Learning rate**：从 {2e-5, 4e-5, 8e-5} 中 sweep 选择
- **训练数据**：GenQA + Infinity Instruct SFT 数据
- **Early stopping**：使用 MBPP+ 每 1000 steps 评估，patience=20, delta=2%

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FCSD 训练流程：

```
teacher = load_original_moe_checkpoint()  # 冻结
student = convert_to_farskip_architecture(teacher)  # 仅修改 skip connections

# Hyperparameter sweep (各 2000 steps)
for bs in [2^16, 2^17, 2^18]:  # batch-size sweep, lr=2e-5
    train(student, teacher, bs, lr=2e-5, steps=2000)
    select by training loss
for lr in [2e-5, 4e-5, 8e-5]:  # lr sweep with best bs
    train(student, teacher, best_bs, lr, steps=2000)
    select by training loss

# Full training with best config, up to 10B tokens
for step in range(max_steps):
    x = next_batch(best_bs)  # SFT data
    with torch.no_grad():
        teacher_logits = teacher(x)
    student_logits = student(x)
    loss = KL(teacher_logits || student_logits)
    loss.backward()
    optimizer.step()

    if step % 1000 == 0:
        mbpp_score = evaluate(student, "MBPP+")
        if early_stop(mbpp_score, patience=20, delta=0.02):
            break
```

消融发现（Tab. 2, Qwen-3-30B MoE, 500M tokens）：
- KL alone 效果最好（Avg-11: 68.2 → 原始 75.9）
- KL + Intermediate L2 反而更差（65.4），可能是 intermediate 对齐过于刚性
- SFT only 显著劣于 KL（58.1），尤其在生成任务上（HEval+ 仅 1.2）
- 冻结 embedding/LM-head 无明显影响（67.6）
- 仅转换部分层（如 50%/75%/90%）使任务更容易，但减少了重叠机会

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

FCSD 的优势：
- **高效**：仅需 < 10B tokens（vs 从头预训练需数万亿 tokens），约 100-1000× 更便宜
- **通用**：适用于任何 MoE 模型，不依赖强 teacher 模型（self-distillation）
- **鲁棒**：KL loss 提供细粒度训练信号，即使 SFT 数据质量不高也能恢复模型表征
- **稳定性挑战**：训练后期可能出现 mode collapse（teacher-student 差异导致大梯度），通过 MBPP+ early stopping 解决

FCSD 对比 SFT 的关键洞察：KL 散度匹配 teacher 的完整概率分布，提供比 one-hot SFT label 更丰富的训练信号。尤其在 FarSkip 场景中，student 的权重已在 teacher 的大部分任务上训练好，主要需要适应新的连接性——KL distillation 正是为此设计的"软对齐"方法。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

## Fine-grained MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fine-grained MoE（细粒度专家混合）是一种 MoE 架构变体，相比传统 MoE（如 GShard、Mixtral）使用更多但更小的 expert。传统 MoE 通常每层配备 8-16 个 expert（每个 expert FFN 隐藏维度与 dense 模型相同或接近），而 fine-grained MoE 将 expert 数量增加到 64-256 个，每个 expert 的参数量相应缩小。典型代表为 DeepSeek-V2/V3 系列（每层 1 shared expert + 256 routed experts，top-8 激活）、Qwen2-57B-A14B（64 experts，top-6 激活）、Deepseek-Lite（64 experts）。

设计动机：(1) 更多 expert 意味着更细粒度的知识分工，每个 expert 可专精于更窄的知识领域，提升专家专业化程度；(2) 更小的 expert 使单次前向计算量降低（虽然激活更多 expert 以保持总参数量），训练成本降低；(3) 路由灵活性更高——64 选 6 的组合数（C(64,6) ≈ 7.5×10^7）远超 8 选 2（C(8,2)=28）。

推理挑战：(1) 激活 expert 数增多（如 top-6 vs top-2），GroupedGEMM 的 expert 数量增加导致 memory-bound 延迟上升；(2) 共享参数（Attention、Norm、Shared Expert）在传统 EP 下每 GPU 复制，expert 数量增多使共享参数内存占比相对降低但绝对值仍然可观；(3) 负载均衡在训练阶段通常良好（expert 小而多使 token 分布更均匀），但推理时 batch size 增大导致几乎所有 expert 被激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Fine-grained MoE 的解码 pipeline（以 IFMoE 论文中 Deepseek-Lite-Chat 为例）：

```
# Fine-grained MoE (64 experts/layer, top-6 routing, 1 shared expert)
For each layer:
    hidden = Norm(input)
    
    # 1. Attention (dense)
    attn_out = Attention(hidden)
    
    # 2. Router
    gate_logits = Router(hidden)           # [B, 64]
    gate_probs = Softmax(gate_logits)      # [B, 64]
    topk_weights, topk_indices = TopK(gate_probs, k=6)  # 选 6 个 experts
    
    # 3. Routed Experts (sparse via GroupedGEMM)
    routed_out = GroupedGEMM(hidden, topk_weights, topk_indices, expert_weights)
    
    # 4. Shared Expert (dense, all tokens)
    shared_out = SharedExpertMLP(hidden)
    
    output = attn_out + routed_out + shared_out
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Fine-grained MoE 的典型实现：DeepSeek-V2 使用 256 routed experts（top-8）+ 1 shared expert，每 expert FFN 维度约为 dense 模型的 1/16-1/32。训练使用 DeepSpeed-MoE 或 Megatron-LM 的 EP 并行。推理方面，IFMoE 提出用 EP+TP hybrid parallelism 减少共享参数内存浪费，并用 self-draft speculative decoding（减少激活 expert 数从 6→2）降低 GroupedGEMM 延迟。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE
- Deepseekmoe: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models

## Speculative Decoding

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Speculative Decoding (SD, Leviathan et al., 2023) 是一种 LLM 推理加速技术。核心思想：用一个较小/较快的 draft model 自回归生成多个候选 token，然后由原始大模型（target model）在一次前向中并行验证这些 token。如果 draft token 与 target model 的预测一致（acceptance rate 高），则等效于一次前向生成了多个 token，实现 wall-clock 加速。

标准 SD 流程：(1) Draft model 自回归生成 γ 个候选 token；(2) Target model 一次前向输入 [prefix + γ 个候选 token]，输出 γ+1 个 logits；(3) 逐 token 比较 draft 和 target 的预测分布，通过 rejection sampling 接受匹配的 token，在第一个不匹配处截断并重新采样；(4) 接受的 token 追加到输出，继续下一轮。

IFMoE 的 Self-Draft 变体不同于标准 SD：(1) Draft model 和 target model 是同一个 fine-grained MoE 模型，区别在于激活的 expert 数（draft 用 2 experts，target 用 6 experts）；(2) 不做逐 token rejection sampling，接受所有 draft token；(3) 通过 KV-cache revision（用全量 experts 重算 KV）补偿 draft 阶段的信息损失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

标准 Speculative Decoding 算法：

```
# Standard SD (Leviathan et al. 2023)
Input: prefix p, draft model M_q, target model M_p, draft length γ
while not EOS:
    # Draft phase (auto-regressive, small model)
    draft_tokens = []
    for i in 1..γ:
        q_i ~ M_q(p + draft_tokens)
        draft_tokens.append(q_i)
    
    # Verification phase (parallel, large model)
    p_1..p_{γ+1} = M_p(p + draft_tokens)  # single forward
    
    # Rejection sampling
    for i in 1..γ:
        if random() < min(1, p_i(x_i)/q_i(x_i)):
            output.append(draft_tokens[i])  # accept
        else:
            output.append(resample from p_i - q_i)  # reject, break
            break
    p = p + output
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SD 的有效性依赖：(1) Draft model 的 acceptance rate 足够高（通常 >70% 才有加速收益）；(2) Draft model 推理速度显著快于 target model；(3) Target model 的验证前向（并行处理 γ 个 token）比 draft model 的 γ 次自回归前向更快。典型配置：draft model 参数量约为 target 的 1/10-1/100。

IFMoE 的 self-draft 方法不需要额外 draft model，通过减少激活 expert 数（6→2）自然获得约 3× 的草稿加速，无需额外模型部署和内存开销。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE

## Self-Draft Speculative Decoding with KV-cache Revision

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IFMoE 提出的 Self-Draft Speculative Decoding 是一种专门针对 fine-grained MoE 的投机解码变体。核心洞察：fine-grained MoE 模型在激活更少 expert 时（如 top-2 vs top-6）仍能保持较好的输出质量，因此无需额外 draft model——MoE 模型自身在"少 expert 模式"下就能作为 draft model。

与标准 SD 的关键区别：(1) 不用额外的 draft model，而是同一模型的不同配置（Dk=2 vs Ek=6）；(2) 接受所有 draft token（不做 rejection sampling），信任 fine-grained MoE 在小 expert 数下的输出质量；(3) KV-cache revision——每 α 步后用全量 experts 重算 KV-cache，因为 draft 阶段 attention 看到的 KV 是基于 2 个 expert 的 residual stream 产生的，与全量 6 experts 的 KV 存在偏差，revision 补偿这个偏差以保证后续 decode 的 attention 质量。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

IFMoE Self-Draft 算法（来自论文 Algorithm 1）：

```
Input: α=10 (draft steps), encode_topk Ek=6, decode_topk Dk=2, MoE model M
Initialize: terminate = False, buffer = []

while not terminate:
    # Draft phase: α steps with fewer experts
    for step in 1..α:
        token = M.decode(topk=Dk)   # 仅激活 top-2 experts
        buffer.append(token)
    
    # KV-cache Revision: recompute KV with full experts
    M.encode(buffer, topk=Ek)       # 全量 top-6 experts 重算
    
    terminate = detect_terminate()
    buffer = []
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

超参选择：α=10（每 10 步 draft 后 revision），Ek=6（全量 expert），Dk=2（draft expert）。IFMoE 在 Qwen2-57B-A14B 和 Deepseek-Lite-Chat 上验证，下游性能（XSum, GSM8K, TruthfulQA, IFEval）与 full model 接近（如 Qwen2 GSM8K: 75.4→71.1），推理延迟和吞吐均提升 >30%。

论文列为 Future Work：(1) 在高要求任务（如代码生成）中引入 logits-based rejection 和 rollback 机制；(2) 动态选择 expert 数——探索何时可减少 expert 数、何时需全量 expert。

涉及论文标题：
- IFMoE: An Inference Framework Design for Fine-grained MoE

## Shared Experts in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shared Experts（共享专家）是现代 MoE 架构（如 DeepSeek-V2/V3）中的设计组件：除了通过 router 稀疏激活的 routed experts 外，额外设置一组对所有 token 都激活的 shared expert MLP 层。MoE 层的完整输出为：`MoE(A) = Σ_j G(A)_j · MLP^j(A) + MLP_shared(A)`，即 routed experts 的稀疏输出加上 shared expert 的全量输出。

Shared experts 的设计动机：(1) 保证所有 token 至少经过一定量的通用处理（不依赖 router 决策），提高训练稳定性；(2) routed experts 侧重专业化知识，shared experts 捕获跨 token 的共享知识/模式；(3) 允许 MoE 层在 routed expert 不可用或质量不佳时仍有一定的基线处理能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 DeepSeek-V2/V3 架构中，每层包含 1 个 shared expert + 多个 routed experts（如 2/8/256 个）：

```
# DeepSeek MoE layer forward:
hidden = RMSNorm(input)

# Routed experts (sparse, top-k routing)
gate_scores = Router(hidden)               # [B, num_routed_experts]
topk_weights, topk_indices = TopK(gate_scores, k=8)
routed_out = RoutedMoE(hidden, topk_weights, topk_indices)
# 仅 top-k experts 被激活

# Shared expert (dense, all tokens)
shared_out = SharedExpertMLP(hidden)       # SwiGLU FFN, 所有 token 都经过

# Combine
output = input + routed_out * router_scale + shared_out
```

在 FarSkip-Collective 中的作用：
Shared experts 的计算不依赖 routed experts 的输出（不依赖 Dispatch/Combine 通信），其输出可以在 Combine 通信期间立即用于下一层的计算。具体来说，FarSkip 的 partial activation `attn-in_k = o_{k-2} + attn-out_{k-1} + shared-exp-out_{k-1}` 中包含了 shared expert 输出但省略了 routed expert 输出，使得 Combine all-to-all 可与 Attention 计算重叠。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Shared expert 通常实现为标准 SwiGLU FFN，与 routed experts 结构相同但无 routing——所有 token 无条件经过。在 EP 配置下 shared expert 权重通常与 routed experts 在同一 rank 上（或复制到所有 rank），不引入额外通信。典型配置：DeepSeek-V2/V3 每层 1 个 shared expert + 256 个 routed experts（top-8 激活）。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- IFMoE: An Inference Framework Design for Fine-grained MoE

在 Hunyuan-Large 中，shared expert 与 16 个 specialized experts 配合使用（而非 DeepSeek 的 top-8 routed + 1 shared）。每个 token 经过 shared expert（捕获通用知识）和 1 个 top-1 激活的 specialized expert（捕获领域特定知识）。这与 expert-specific learning rate scaling 协同——shared expert 使用较大学习率 ε_opt(B)，specialized experts 使用缩小后的学习率 ε_opt(B/16) ≈ 0.31 × ε_opt(B)。

## Knowledge Distillation for LLM Architecture Conversion

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Knowledge Distillation（知识蒸馏）是将大模型（teacher）的知识迁移到小模型或架构修改后的模型（student）的训练方法。在 LLM 上下文中有两类主要蒸馏形式：(1) **Logit-based KD**：匹配 teacher 和 student 的输出概率分布（KL 散度），`L_KD = E[Σ_t KL(q(·|x, y_<t) || p_θ(·|x, y_<t))]`；(2) **Feature-based KD**：对齐中间层 hidden states，`L_L2 = Σ_i ||o_i(θ) - t_i||²`。

**Self-distillation** 是 KD 的特殊形式，teacher 和 student 共享相同的参数空间（或 student 从 teacher 初始化），teacher = 原始模型，student = 修改后的模型。FCSD（FarSkip-Collective Self-Distill）就是典型的 self-distillation 应用——teacher 和 student 的权重形状完全相同，仅连接性不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FarSkip FCSD 使用的 logit-based self-distillation：

```
# Teacher: 原始 MoE 模型, Student: FarSkip 修改后模型
teacher = load_checkpoint("original_moe")  # 冻结，eval mode
student = FarSkipModel(teacher_config)     # 从 teacher 权重初始化
student.load_state_dict(teacher.state_dict())  # 同一参数空间，直接拷贝

for batch in dataloader:
    # Forward
    with torch.no_grad():
        teacher_logits = teacher(batch)  # [B, seq_len, vocab_size]
    student_logits = student(batch)

    # KL divergence loss (temperature=1)
    # p_teacher = softmax(teacher_logits)
    # p_student = softmax(student_logits)
    loss = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction='batchmean'
    )
    loss.backward()
    optimizer.step()
```

FCSD 消融发现：
- KL vs SFT：KL 显著优于 SFT（DeepSeek-V2-Lite: 62.0 vs 55.0 Avg），因为 KL 提供完整的概率分布信号而非 one-hot label
- KL vs KL + Intermediate L2：仅 KL 更好（68.2 vs 65.4），intermediate L2 的刚性约束可能阻碍模型适应新连接性
- 训练稳定性：KL distillation 后期可能出现 mode collapse（小的 teacher-student 差异产生大梯度），early stopping 是最简单有效的解决方案

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LLM 架构转换场景中（如 FarSkip、ladder-residual、Kraken），self-distillation 是对比 SFT 微调更优的选择——因为 student 权重已经在 teacher 的大部分能力上预训练好，主要任务是适应新的连接性（而非学习新知识）。KL loss 让 student 的每步输出"模仿"teacher，保留了 teacher 的生成行为和内部表征。实践中 batch-size 和 learning rate 的 sweep 至关重要（影响收敛速度和稳定性）。

涉及论文标题：
- FarSkip-Collective: Unhobbling Blocking Communication in Mixture of Experts Models

## Personalized Federated Learning (PFL)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Personalized Federated Learning (PFL) 是联邦学习的一个子领域，旨在为每个参与客户端训练定制化的个性化模型，而非为所有客户端训练单一的全局模型。传统 FL（如 FedAvg）假设一个全局模型可以服务所有客户端，但在实际场景中，不同客户端的数据分布（non-IID）、任务类型（跨任务）和资源能力（异构硬件）差异巨大，单一模型难以同时满足所有客户端的需求。

PFL 的核心思路是允许不同客户端拥有不同的模型参数或结构。实现策略包括：(1) 正则化方法——在本地训练 loss 中加入 proximal term（如 FedProx）或修正梯度方向（如 SCAFFOLD）；(2) 模型拆分——将模型分为共享层和个性化层（如 FedPer）；(3) 知识蒸馏——用全局模型蒸馏个性化小模型；(4) 模型剪枝——为不同客户端剪裁不同子网络；(5) MoE 方法——利用 expert 并行结构为不同客户端选择不同的 expert 子集（如 FedMoE）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 FedMoE 中，PFL 通过 MoE 架构的 expert 级个性化实现——不是为每个客户端训练独立模型，而是从共享的全局 MoE expert 池中为每个客户端 sub-sample 最优 expert 子集：

```
# PFL via MoE Expert Sub-Sampling (FedMoE)
# 全局模型: 32 experts/layer
# 客户端 k 的个性化子模型: 平均 65 experts 从 32 experts 中选出

# 个性化子模型构建
for client k in selected_clients:
    for layer_i in 1..L:
        # 从 client-expert map 中提取该客户端该层的 expert 子集
        kept = expert_map[k][layer_i]
        W_k.experts[layer_i] = {W_global.experts[layer_i][j] for j in kept}
        W_k.router[layer_i] = W_global.router[layer_i][kept]  # 仅保留相关维度
    W_k.dense = W_global.dense  # dense 层全员共享

    # 本地个性化训练 (仅训练子模型参数)
    W_k* = TRAIN(W_k, D_k)  # D_k 为客户端 k 的私有数据

# 知识聚合: Modular Aggregation 将个性化知识吸收回全局 expert 池
for expert_j in all_experts:
    clients_using = {k: expert_j in W_k.experts}
    if |clients_using| == 0:  W_global[j] unchanged
    elif |clients_using| == 1:  W_global[j] = W_k[j]  # 直接更新
    else:  W_global[j] = weighted_avg(W_k[j] for k in clients_using)
```

MoE-PFL 的独特优势：个性化体现在 expert 选择层面而非整个模型空间——不同客户端共享同一个 expert 池但训练不同子集，既能充分个性化（不同 expert 处理不同数据/任务），又能在重叠部分共享知识（使用相同 expert 的客户端通过聚合相互学习）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

PFL 的实现框架包括：
- **Flower (flwr)**：提供 FedProx、FedAvgM 等策略的内置实现
- **FedML**：支持 PFL 算法库
- **FATE**：企业级 FL 框架，支持多种 PFL 策略
- **自定义实现**：FedMoE 基于 PyTorch + HuggingFace Transformers 自建 FL 模拟框架

FedMoE 的方法适合：(1) 底层使用 MoE 架构模型（如 Switch Transformers），(2) 客户端数据/任务异构性强，(3) 客户端资源有限需要个性化轻量模型，(4) 需要兼顾协作学习和个性化。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

## Modular Aggregation (in Federated Learning with MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modular Aggregation 是 FedMoE 提出的联邦学习模型聚合策略，专门针对 MoE 架构设计，用于替代传统 FL 中的 FedAvg。其核心思想是按"模块粒度"（即每个 expert）独立决定聚合方式，而非对所有参数执行统一的加权平均。

三种聚合模式：(1) Unactivated experts——未被任何客户端使用的 expert，保持不变；(2) Single-client experts——仅被单个客户端使用的 expert，直接替换为该客户端的更新；(3) Multi-client experts——被多个客户端共享的 expert，使用 FedAvg 加权聚合。Router 参数按对应 expert 维度同步更新。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Modular Aggregation (FedMoE)
def ModularAggregation(w_global, client_models, data_sizes):
    # Dense 层: 标准 FedAvg
    for param in dense_params:
        w_global[param] = weighted_avg(client_models[k][param], data_sizes[k])

    # Sparse (expert) 层: 按模块粒度
    for layer_i in 1..L:
        for expert_j in 1..E_i:
            clients_using = [k for k in client_models if expert_j in w_k]
            if |clients_using| == 0:
                continue  # 未激活 → 不变
            elif |clients_using| == 1:
                w_global[layer_i][expert_j] = w_{clients_using[0]}[layer_i][expert_j]
            else:
                w_global[layer_i][expert_j] = weighted_avg(
                    w_k[layer_i][expert_j] for k in clients_using, weights=data_sizes[k])
```

相比 FedAvg，Modular Aggregation 防止不相关客户端相互干扰（负迁移），保留个性化 expert 的专用性，在共享 expert 上实现协作学习，天然支持异构子模型。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

## Expert Recommendation in Federated Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Recommendation 是 FedMoE Stage 2 中的动态子模型结构调整机制。当客户端连续多轮性能无提升（达到瓶颈），云端利用其他客户端作为参考，推荐增加高效 expert 或裁剪低效 expert。

核心流程：(1) 基于 expert 激活概率向量的 cosine similarity 找到 top-K 最相似客户端；(2) 若参考组平均 expert 数多于当前客户端，按参考组加权激活概率（Eq. 6）排序推荐引入外部 expert；否则推荐裁剪低效 expert；(3) 调整后验证，不改善则回退并固定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert Recommendation
for client u_k:
    if acc[u_k] stalled:
        # Cosine similarity (Eq. 4)
        sim(u_k, u_a) = Σ_i Σ_j p_{i,j}(u_k)·p_{i,j}(u_a) / (||p(u_k)||·||p(u_a)||)
        S' = top_K_by_similarity(sim, K)
        n = AVG(n_expert(S')) - n_expert(u_k)

        if n > 0:  # 增加 expert
            for expert e NOT in w_k:
                p_hat(e) = Σ_{a∈S'} sim(u_k,u_a)·p_e(u_a) / Σ_{a∈S'} sim(u_k,u_a)
            add top_n experts sorted by p_hat descending
        elif n < 0:  # 裁剪 expert
            for expert e in w_k:
                p_hat(e) = Σ_{a∈S'} sim(u_k,u_a)·p_e(u_a) / Σ_{a∈S'} sim(u_k,u_a)
            remove top_|n| experts sorted by p_hat ascending

        if adjusted model worse:
            revert and fix structure
```

Expert Recommendation 利用群体智慧指导个体结构调整，是一种"试探-验证-回退"的安全机制。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

## Load Balance Loss in Mixture of Experts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load Balance Loss 是 MoE 训练中的辅助损失函数，防止 gate 将所有 token 路由到少数 expert 导致利用率极度不均衡。Switch Transformers 中定义为：

$$L_{LB} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 N 为 expert 总数，$f_i$ 为路由到 expert i 的 token 比例，$P_i$ 为 gate 分配给 expert i 的平均路由概率。该 loss 最小化 $f_i$ 和 $P_i$ 之间的差异。

FedMoE 客户端本地训练中：$\mathcal{L}_k = \mathcal{L}_{CE} + \alpha \mathcal{L}_{LB}$，$\alpha$ 通常取 0.01。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Load Balance Loss (Switch Transformers)
def load_balance_loss(gate_logits, top1_indices):
    E = gate_logits.shape[1]  # expert 数
    P = softmax(gate_logits, dim=-1).mean(dim=0)   # (E,) 平均路由概率
    f = one_hot(top1_indices, E).float().mean(dim=0)  # (E,) token 分配比例
    return E * (f * P).sum()  # N·Σ f_i·P_i
```

FedMoE 的子模型仅包含部分 expert，load balance loss 仅在子模型包含的 expert 之间作用。

Flex-MoE 中，Load Balance Loss 有独特变体：在 Expert Specialization 阶段，top-1 gate 已被 S-Router 通过 cross-entropy loss 强制绑定到目标 modality combination expert，因此 load/importance balancing loss 仅对**剩余 top-(k-1) 个 expert** 计算（$E \setminus e_{\text{top-1}}$），以避免对已锁定 expert 重复施加平衡约束。公式为：

$$\mathcal{L}_{\text{balance}} = \text{CV}^2\left(\sum_{j}^{N} \text{importance}_j\right) + \text{CV}^2\left(\sum_{j}^{N} \text{load}_j\right)$$

$$\text{importance}_e = \sum_{i}^{N} g_{ie}, \quad \text{load}_e = \sum_{i}^{N} \delta(g_{ie} > 0), \quad \forall e \in E \setminus e_{\text{top-1}}$$

其中 $\text{CV}^2(x) = (\sigma(x) / \mu(x))^2$ 为变异系数平方。loss coefficient 设为 0.01，与 task classification loss 和 cross-entropy loss 联合优化。

GatePro 揭示了 Load Balance Loss 的局限性——它仅解决 token 分配的统计均衡，但不解决 expert 选择的功能多样性（diversity）问题。功能相似的 expert 仍可被同时激活（只要 token 分配均衡），产生冗余计算。GatePro 通过局部竞争机制补充了 diversity 维度，与 LBL 形成互补：LBL 保证"资源利用效率"，GatePro 保证"资源利用质量"。实验证明 GatePro + LBL 的组合收敛最快。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- Layerwise Recurrent Router for Mixture-of-Experts
- LocMoE: A Low-overhead MoE for Large Language Model Training

**RMoE 对 Load Balance Loss 梯度的分析**：RMoE 论文通过分析训练过程中 router 梯度的两个来源（LM loss 和 LB loss），揭示了 linear router 与 GRU router 的行为差异：对于 linear router，LB loss 在训练早期主导梯度（LB grad: 0.433 vs LM grad: 0.625 at step 0.1k），随后 LB grad 迅速衰减（10k 步后 LB grad 仅 0.001-0.011），表明 linear router 过早收敛于 LB loss 的次优解。对于 GRU router，LB loss 梯度在训练早期较稳定（0.337→0.014→0.003），且 LM loss 梯度持续下降，表明 GRU router 更优地优化了 LM loss 与 LB loss 的权衡。结论：跨层 recurrent router 能有效控制 LB loss 的影响，避免其过早主导训练。

**LocMoE 的 Load Balance Loss 用法**：LocMoE 使用与 Switch Transformer 相同的 aux loss 公式 ($L_{aux} = \alpha \cdot n \cdot \sum f_i \cdot P_i$)，α=0.01。区别在于 LocMoE 将其与 locality loss 联合使用：$L_{task} = L_{aux} + L_{loc} + L_{cross}$，同时约束负载均衡和局部性。

**LLEP 对 Load Balance Loss 的替代视角**：

LLEP 指出了 Load Balance Loss（及 moving-average routing bias）的局限性：这些方法在训练过程中强制统计均衡，但在 post-training 或推理阶段不可行（会改变预训练的 routing behavior，破坏模型完整性）。LLEP 提出了系统级的替代方案——在 dispatch-combine 通信阶段动态重新分配 token 到 GPU，而不修改 gate network 的输出或 expert FFN 计算。这使得 LLEP 适用于 post-training（SFT、RLHF）、推理、甚至训练（支持 backward pass），而 Load Balance Loss 仅适用于预训练。LLEP 的 exact computation 属性（保证数学输出与传统 EP 完全一致）是其区别于所有修改模型行为方案的核心优势。

## Non-IID Data Heterogeneity in Federated Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Non-IID 数据是 FL 的核心挑战，指不同客户端数据不满足独立同分布。类型包括 Label skew（标签分布偏斜）、Feature skew（特征分布偏斜）、Quantity skew（数据量偏斜）、Task-level heterogeneity（任务级异构——FedMoE 重点场景）。

Non-IID 导致 client-drift：各客户端本地训练的梯度方向不一致，聚合后的模型在不同目标间摇摆。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

FedMoE 设置 4 种 FL 模拟：(1) Standard-Hetero-T——30 客户端各异构任务，(2) Standard-Hetero-TD——额外 label-skewed non-IID，(3) Enforced-Hetero-T——强制选不同任务客户端制造冲突，(4) Enforced-Hetero-TD——Enforced + label-skewed。FedMoE 通过 expert 级个性化使不相关任务的梯度互不干扰：client 1（分类）的 expert 3 不受 client 2（阅读理解）的梯度影响。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

处理 non-IID 的主流方法：FedProx（proximal term）、SCAFFOLD（control variates）、FedMoE（expert 级个性化解耦参数空间）。FedMoE 在 Enforced 设置下优势更显著，验证了 MoE 个性化对强异构场景的有效性。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

## FedProx (Federated Proximal Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FedProx 是 Li et al. (2020) 提出的 FL 优化算法，在客户端本地 loss 中添加 proximal term 限制模型偏离全局模型：

$$\min_w h_k(w; w^t) = \mathcal{L}_k(w) + \frac{\mu}{2}\|w - w^t\|^2$$

其中 $\mu$ 控制正则化强度。还引入 $\gamma$-inexactness 允许不同客户端执行不同数量的本地更新。

在 FedMoE 中作为 baseline，全局模型 8 experts/layer（受限于内存 18-24GB），内存 24.71GB，通信 2.30GB。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# FedProx 客户端本地训练
w_k = w_global
for epoch in 1..E:
    for batch in D_k:
        L_CE = cross_entropy(model(w_k, x), y)
        proximal_term = (mu/2) * ||w_k - w_global||^2
        loss = L_CE + proximal_term
        w_k = w_k - lr * gradient(loss)
# 上传 → 服务器 FedAvg 聚合
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Flower 内置 `FedProx` 策略（`proximal_mu` 参数），FedML 和 FATE 也支持。μ 通常 0.001–0.1。FedMoE 实验中 FedProx 在跨任务场景不如 FedMoE——proximal term 只能约束参数距离但无法从根本上解决不同任务需要不同参数的问题。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

## SCAFFOLD (Stochastic Controlled Averaging for Federated Learning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SCAFFOLD（Karimireddy et al., ICML 2020）使用控制变量修正本地梯度方向以解决 client-drift。维护全局控制变量 c（全局梯度估计）和本地控制变量 c_k（客户端梯度估计），本地更新规则：$w_k \leftarrow w_k - \eta \cdot (\nabla \mathcal{L}_k(w_k) - c_k + c)$。

在 FedMoE 中作为 baseline，因控制变量传输额外开销，通信量最大（4.61GB）。Enforced-Hetero-T 下 TC accuracy 仅 36.17（FedMoE 94.85），验证了 task-level heterogeneous 场景下仅修正梯度方向不足以弥合不同任务的根本差异。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SCAFFOLD 每 round
c_k_new = c_k - c + (w_global - w_k) / (T * lr)  # Option II 更新控制变量
# 上传 (w_k, Δc_k)  →  额外通信开销约为模型大小 2 倍
w_global = weighted_avg(w_k)
c = c + (|S|/N) * avg(c_k_new - c_k)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

SCAFFOLD 理论收敛速度最优但实际挑战：(1) 控制变量通信量翻倍，(2) 控制变量存储需与模型相同内存，(3) task-level heterogeneous 场景表现差——不同任务的最优梯度方向本身不一致，仅靠修正无法弥合。

涉及论文标题：
- FedMoE Personalized Federated Learning via Heterogeneous Mixture of Experts

## Missing Modality Bank

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Missing Modality Bank 是 Flex-MoE 提出的用于多模态 missing modality 场景的 learnable embedding bank。给定模态集合 $\mathcal{M}$，缺失模态的所有可能组合数为 $2^{|\mathcal{M}|}-1$（不包括全模态组合）。bank $\mathbf{B} \in \mathbb{R}^{(2^{|\mathcal{M}|}-1) \times |\mathcal{M}| \times d}$ 为每种 observed modality combination 下的每个 missing modality 存储一个可学习的 embedding（维度为 d）。

其核心设计原则是：缺失模态的 embedding 不应是全局统一的 learnable vector，而应**依赖于当前样本有哪些模态被观测到**。例如，同一个缺失的 biospecimen 模态，在样本有 {Image, Clinical} 时的补充 embedding 与样本有 {Image, Genetic, Clinical} 时应不同——因为观测组合提供的上下文信息不同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 输入: modality_set M = {I, C, B, G}, hidden_dim d=128
# Missing modality bank 初始化:
B = torch.randn(2^|M|-1, |M|, d)   # (15, 4, 128) for |M|=4

# MC_index: 将观测模态组合映射到 bank 行索引
# e.g., MC_index(I=1, C=1, B=0, G=0) → "IC" → index 6
#       MC_index(I=1, G=1, B=1, C=0) → "IGB" → index 3

def forward(sample_i, observed_modalities, encoders):
    embeddings = []
    mc_idx = MC_index(observed_modalities)  # 观测组合 → bank 行
    for m in M:
        if m in observed_modalities:
            e_i^m = encoders[m](sample_i[m])     # 使用真实数据编码
        else:
            e_i^m = B[mc_idx][m]                 # 从bank查找缺失embedding
        embeddings.append(e_i^m)
    return concat(embeddings)                    # (4*d,) 的完整多模态表示
```

Flex-MoE 在 ADNI 数据集上验证了 bank 的有效性：cosine similarity 分析显示"共享更多观测模态的组合有更相似的缺失 embedding"——full "ICBG" 与 "ICB" 相似度 0.56，与 "IC" 仅 0.46。去除 embedding bank 使得 ACC 从 66.11 降至 63.87。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Bank 是一个 PyTorch `nn.Parameter` 张量，随模型端到端训练。bank 索引使用位掩码：将 modality combination 编码为位掩码（如 [I=1, G=1, C=0, B=0] → 二进制 1100），转为整数索引。编码器**仅用对应 modality 被 observed 的样本训练**——避免了 traditional zero-padding/imputation 对 encoder 训练质量的破坏。bank 的参数随下游任务 loss 一起优化，学习"当特定模态缺失时，基于已有观测信息应该补充什么"。

涉及论文标题：
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

## Expert Generalization and Specialization in Multimodal MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Generalization and Specialization 是 Flex-MoE 提出的两阶段 SMoE 训练策略，用于在 missing modality 场景下让每个 expert 同时具备通用知识和专有知识。灵感来自课程学习——先用"简单"样本（全模态）学习通用知识，再用"困难"样本（部分模态）学习专有知识。

- **Generalization 阶段（warm-up epochs）**：仅使用全模态样本（all modalities observed），G-Router 执行标准 top-k gating + load/importance balancing loss，让所有 expert 学习从完整多模态信息中提取的通用知识。
- **Specialization 阶段（剩余 epochs）**：使用所有 modality combination 的样本，S-Router 通过 cross-entropy loss 将 top-1 gate 强制绑定到目标 modality combination expert index，其余 top-(k-1) expert 继续做 load/importance balancing。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === Phase 1: Expert Generalization (warm-up epochs) ===
# 输入: 仅全模态样本 batch (all 4 modalities observed)
for sample_i in full_modality_batch:
    h_i = concat([e_i^I, e_i^C, e_i^B, e_i^G])  # 所有模态真实编码
    gate_logits = g(h_i)                         # G-Router (1-2 layer MLP)
    gate_vals = TopK(softmax(gate_logits), k)    # k=4 for ADNI
    y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)
    L = L_CE(y_i, label) + 0.01 * L_balance     # 全 expert 参与 balancing

# === Phase 2: Expert Specialization (remaining epochs) ===
# 输入: 任意 modality combination 的样本
for sample_i in batch:  # batch 按可用模态数降序排列
    h_i = flex_moe_encode(sample_i)              # 含 missing modality bank
    gate_logits = g(h_i)                         # S-Router
    top1_pred = argmax(gate_logits)
    target_exp = MC_index(observed_modalities(i))
    L_ce = -sum_j one_hot(MC(x_j)) * log(softmax(gate_logits))  # 绑定 top-1
    gate_vals = TopK(softmax(gate_logits), k)
    y_i = sum_{e in top-k} gate_vals[e] * f_e(h_i)
    # L_balance 仅计算 E \ {e_top1} 的 expert
    L = L_CE(y_i, label) + 0.01 * (L_balance + L_ce)
```

Flex-MoE 在 ADNI 数据集上验证：去除 Expert Specialization 后 ACC 从 66.11 降至 62.75；同时去除 ES+EG 后降至 62.49。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 样本排序——训练开始前按可用模态数降序排列所有样本，warm-up 阶段自然只包含全模态样本；(2) Expert index 分配——每种 modality combination 对应固定 expert index（如 "IGCB"=0, "IGC"=1, ..., "B"=14），剩余 index 为 buffer expert；(3) S-Router 的 cross-entropy loss 直接作用于 gate logits 的 softmax，不打断梯度流；(4) warm-up epochs 后使用 shuffled 样本增强泛化性；(5) top-k 选择在 specialization 阶段偏大（k=4 for ADNI），因为 top-1 已固定，剩余 3 个 expert 提供跨模态组合的交互。

涉及论文标题：
- Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts

## Low-Rank Adaptation (LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Low-Rank Adaptation (LoRA) 由 Hu et al. (2022, ICLR) 提出，是当前最主流的 Parameter-Efficient Fine-Tuning (PEFT) 方法之一。核心原理基于大语言模型的参数更新具有内在低秩性（intrinsic low-dimensionality）：全参数微调中权重更新矩阵 ΔW ∈ R^{m×n} 的实际有效自由度远低于 mn，可由两个低秩矩阵 B ∈ R^{m×r} 和 A ∈ R^{r×n} 的乘积近似，其中 r ≪ min(m, n)。正向传播：W'·x = W₀·x + (α/r)·B·(A·x)，其中 W₀ 冻结，A 使用 Kaiming 均匀初始化，B 零初始化。推理时可将 ΔW 合并到 W₀ 中无额外延迟。

FlyLoRA 论文基于 LoRA 的标准公式 (Eq. 1-2) 构建改进：标准 LoRA 中不同 rank（即 A 的不同行和 B 的不同列）之间存在参数耦合——梯度在所有 rank 间密集计算，导致 rank 间梯度协方差非零 (intra-task interference)；多任务 LoRA 合并时可训练的 A 之间无正交性保证 (inter-task interference)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// 标准 LoRA 训练流程:
Forward:
  h_proj = A @ x               // [r×n] @ [n] → [r]
  delta = (α/r) * B @ h_proj  // B 的 r 个列加权组合
  output = W₀ @ x + delta     // W₀ 冻结 (无梯度)

Backward:
  grad_B = (α/r) * grad_output @ h_proj^T   // 所有 r 列有梯度
  grad_A = (α/r) * B^T @ grad_output @ x^T  // 所有 r 行有梯度

// LoRA rank-wise 展开 (Eq. 6, FlyLoRA):
// f_LoRA(x) = W₀·x + (α/r)·Σ_{i=1}^r b_i·(a_i·x)
// 每对 (a_i, b_i) 是一个 rank-1 组件, 类比一个 expert
// 标准 LoRA 中所有 r 个 rank-1 组件始终全激活 (k=r)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 标准实现：HuggingFace PEFT (`peft.LoraConfig`)，指定 target_modules（通常为 q/k/v/o/gate/up/down_proj）、rank r (8/16/32)、alpha scaling。保存为 adapter_model.bin。
- LoRA 变体：LoRA-FA (冻结 A 节省激活内存)、AsymmetryLoRA (固定 A 为随机投影)、AdaLoRA (自适应 rank 分配)、DoRA (幅值-方向解耦)、MoE-based LoRA (多 expert + router)。
- FlyLoRA 代码：https://github.com/gfyddha/FlyLoRA

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

## MoE-based LoRA (Mixture-of-Experts based Low-Rank Adaptation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE-based LoRA 将 Mixture-of-Experts 架构集成到 LoRA 框架中。与标准 LoRA 使用单对低秩矩阵 (A, B) 不同，MoE-based LoRA 将适配器分解为 N 个 expert 对 {(A_i, B_i)}_{i=1}^N，每 expert 独立低秩分解，由 router G(x) 动态选择每个 token 激活的部分 expert。正向传播 (Eq. 3)：f_MoE-LoRA(x) = W₀·x + (α/r)·Σ_{i=1}^N G(x)_i·(B_i·A_i·x)，其中 G(x)=top-k(W_g·x)。核心动机：通过 expert 专门化实现 task decoupling，缓解 LoRA 的 intra-task parameter interference。

FlyLoRA 将 Split-LoRA (N×r_i) 作为代表基线。关键洞察 (Sec 2.3)：将 expert 粒度推至 rank-wise (N=r, 每 expert 仅 1 rank) 获最佳 decorrelation (图 1a)，但显式 router W_g ∈ R^{N×n} 随 N 线性增长 (图 1b)，造成参数效率劣化。这驱动 FlyLoRA 提出隐式 router。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// MoE-based LoRA (Split-LoRA(4×8)):
Forward:
  gate_logits = W_g @ x                              // W_g ∈ R^{N×n}
  gate_scores = sigmoid(topk(gate_logits, k_act))    // 选 top-k experts
  delta = Σ_i gate_scores[i] * (B_i @ (A_i @ x))
  output = W₀ @ x + (α/r) * delta

// 激活参数分析 (Table 9, d=hidden_dim, r=total rank, k=activated rank):
// LoRA(r):         param=2dr,   grad=4dr,     optim=24dr
// Split-LoRA(N×r): param=2dk+dN, grad=4dk+2dN, optim=24dk+12dN
// FlyLoRA(k):      param=dk,    grad=2dk,     optim=12dk
//
// Split-LoRA 额外开销来源于 router W_g ∈ R^{N×n}:
// Forward 多 dN 参数, Backward 多 2dN gradient + 12dN optimizer
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Split-LoRA 使用 sigmoid + top-k gate (FlyLoRA Appendix C.3)
- 代表性方法：MoLA (Wu et al. 2024), MixLoRA (Li et al. 2024), HydraLoRA (Tian et al. 2024, 非对称), LoRAMoE (Dou et al. 2023)
- FlyLoRA 改进方向：消除显式 W_g (用冻结 A 替代), 消除 A 的训练开销, 引入跨任务正交性

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

## Sparse Random Projection as Implicit Router (in LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlyLoRA 的核心创新：将 LoRA 的 A 矩阵替换为**冻结的、稀疏的、随机初始化矩阵**，每行仅 p < n 个非零元素 (p≪n)，采样自 N(0, 1/r²)。A 同时承担两个角色：(1) 下投影 (传统 LoRA A 功能)；(2) 隐式 router——通过 top-k(Ax) 幅值选择应激活的 B 列。灵感来自果蝇嗅觉回路：projection neurons 通过稀疏随机连接投射到 Kenyon cells，然后 winner-take-all 选择性地激活 mushroom body output neurons。

由于 A 的稀疏随机投影近似保持 pairwise 距离 (Theorem 3.1, Johnson-Lindenstrauss 延伸)，语义相似 token 被路由到相似 expert。不同 task 的独立随机 A_i, A_j 天然近似正交 (Theorem 3.4: E[A_i·A_j^T] = 0, P(||A_i·A_j^T||₂ ≥ εr) ≤ p²/(nr²ε²))，实现 inter-task decoupling。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// FlyLoRA 稀疏随机投影 + 隐式路由:
// A ∈ R^{r×n}: 冻结, 每行 p 个非零 ~N(0,1/r²), sparsity ρ=p/n=k/r
// B ∈ R^{m×r}: 可训练, d ∈ R^r: 负载均衡偏置

Forward (Eq. 7-11):
  y = A @ x                               // 稀疏投影: O(r·p) vs LoRA O(r·n)
  y_biased = y + d
  I_topk = argtopk(y_biased, k)           // 隐式路由决定
  delta = (α/r) * sum_{i ∈ I_topk} b_i * y[i]
  output = W₀ @ x + delta

// A 初始化策略对比 (Appendix B.7, MMLU, Llama-3.1-8B):
// Gaussian (默认):      Before 40.88±1.61, Δ after merge -2.02
// Rademacher:           Before 40.42±0.23, Δ -2.35
// FJLT (结构化):         Before 40.57±1.34, Δ -2.50
// Two-Phase (可训练预热): Before 40.76±1.04, Δ -4.86 (破坏正交性!)
//
// 默认配置: total rank r=32, activated rank k=8, sparsity ρ=k/r=0.25
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 与 hash router (Roller et al. 2021) 关系：FlyLoRA 的固定随机投影 A 类似 hash router 的固定映射，通过距离保持性提供更强理论保证
- A 冻结消除：(1) router 参数 W_g；(2) A 的梯度计算和优化器状态；(3) A 相关的激活值存储
- 实现：使用 PyTorch `nn.Linear` weight 冻结 + `requires_grad=False`
- 代码：https://github.com/gfyddha/FlyLoRA

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

## Rank-Wise Expert Allocation (in MoE-based LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlyLoRA 提出的极限 expert 分解策略：将 LoRA 的 r 个 rank 每个作为独立 expert——共有 r 个 rank-1 expert，每 token 仅 top-k 个被激活 (k<r)。f(x) = W₀·x + (α/r)·Σ_{i∈I_topk} b_i·(a_i·x)，其中 a_i=A[i,:]∈R^{1×n}, b_i=B[:,i]∈R^{m×1}。

理论依据 (Theorem 3.3)：top-k 稀疏性使 off-diagonal 梯度协方差按 k²/r² 缩减。k=8, r=32 时 off-diagonal 协方差约为 dense 的 1/16；k=1 时几乎完全去耦合。图 3(b-c) 梯度相关热力图验证：LoRA-FA(r=32) 密集相关，FlyLoRA(k=8) 显著稀疏。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Rank-Wise Expert Allocation:
// 将 B 矩阵按列分解为 r 个 rank-1 "expert"
// 每个 expert_i = b_i · a_i · x

Forward:
  y = A @ x                             // [r]
  I_topk = argtopk(y + d, k)            // k 个 rank-1 expert 被激活
  mask = zeros(r); mask[I_topk] = 1
  delta = (α/r) * (B ⊙ mask) @ y       // 仅 k 列计算

// 梯度去耦合 (Theorem 3.3):
// E[g_i^sparse · g_j^sparse] ≈ (k²/r²) · E[g_i^dense · g_j^dense]
//
// 消融 (图 4b): 固定 r=32, 变 k:
// k 太小 → 信息不足; k 太大 → interference 增加
// 最优 k=8~12; 默认 k=8
//
// 消融 (图 4c): 固定 k=8, 增加 r:
// 性能持续改善 (更多 capacity 无额外 interference)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch 实现：A ∈ R^{r×n} 作为 frozen nn.Linear weight, B ∈ R^{m×r} 作为 trainable nn.Linear weight
- Top-k selection：torch.topk(y, k).indices, boolean mask 乘 B forward
- Backward：grad 通过 mask 自动归零 (PyTorch autograd)

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

## Model Merging (LoRA Weight Averaging Fusion)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model Merging (模型合并) 是 training-free 多任务能力集成技术：将不同下游任务上微调的 LoRA adapters 通过权重平均合并到同一 base model。标准方法 (Eq. 12)：W' = W₀ + (1/t)·Σ_{i=1}^t B_i·A_i。高级方法：TIES-MERGING (sign consensus + trimming)、DARE (dropout + rescale)。

核心痛点：不同任务 LoRA 参数 B_i·A_i 在参数空间中重叠——可训练 A_i, A_j 之间无正交性，合并造成 destructive interference (e.g., ScienceQA Δ=-60.34%)。FlyLoRA 通过冻结随机投影 A_i (近似正交) 解决。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Weight Averaging (FlyLoRA Eq. 12):
// 输入: task_i 的 (B_i, A_i), base W₀
W' = W₀ + (1/t) * Σ_{i=1}^t B_i·A_i

// FlyLoRA 优势: A_i, A_j 近似正交使:
// <B_i·A_i, B_j·A_j>_F ≈ 0 (Corollary 3.5)
// ||Σ w_i·B_i·A_i||²_F ≈ Σ w²_i·||B_i·A_i||²_F
//
// 合并性能降 Δ% (Llama-3.1-8B, weight averaging):
//               MMLU   ScienceQA  GSM8K   HumanEval
// LoRA(r=8):    -6.48  -60.34     -30.15  -13.04
// LoRA(r=32):   -4.91  -59.66     -31.48  -11.43
// Split-LoRA:   -4.86  -54.74     -28.30  -9.92
// FlyLoRA:      -2.02  -43.05     -21.81  -4.27
//
// CKA (Table 16): FlyLoRA 0.85/0.53/0.71/0.84 vs LoRA 0.78/0.39/0.58/0.75
// 更高 CKA → 合并后与单任务输出对齐更好
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- HuggingFace PEFT `add_weighted_adapter()`, mergekit (https://github.com/arcee-ai/mergekit)
- TIES-MERGING: trim (移除低幅值) → elect sign (多数决定) → disjoint merge
- DARE: delta params 随机 dropout p% → rescale 1/(1-p) → merge
- FlyLoRA 与 TIES/DARE 兼容 (plug-and-play), 叠加效果更佳 (Table 12)

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

## Loss-Free Load Balancing (in MoE-based LoRA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

FlyLoRA 采用的 MoE expert 负载均衡策略 (基于 DeepSeek-V3 auxiliary-loss-free 策略)。维护 expert-wise bias d ∈ R^r，训练期间手动更新 d_i ← d_i + u·sign(ē_i - c_i)，其中 ē_i 为期望分配频率 (均匀 1/r)、c_i 为实际计数、u 为小步长。当 expert i 过度使用时 d_i 减小 (抑制)，使用不足时 d_i 增大 (鼓励)。bias 在 top-k 前加到 Ax 上 (Eq. 10)：I_topk = argtopk(Ax + d, k)，直接改变 expert 选择而非通过 loss 间接影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Loss-Free Load Balancing (Eq. 9-10):
// d ∈ R^r: bias, 初始化为 0; u ≈ 1e-3

每训练步更新:
  expected = total_tokens * (1/r)
  for i in 0..r-1:
    actual[i] = count(I_topk == i)
    d[i] += u * sign(expected - actual[i])

Forward 中使用:
  I_topk = argtopk(Ax + d, k)  // bias 影响 top-k

// 消融 (Table 3, MMLU, Llama-3.1-8B):
// Loss-Free:       40.88±1.61 (默认)
// Loss-Controlled: 40.59±0.51
// No Balancing:    37.56±2.87 (↓3.32, 方差大)
//
// 对比 Loss-Controlled (Switch Transformer):
// L_aux = α·N·Σ_i f_i·P_i, L_total = L_task + L_aux
// 需额外超参数 α, 与主 loss 梯度竞争
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 来源：DeepSeek-V3 (Liu et al. 2024) 首次提出，FlyLoRA 适配到 MoE-based LoRA
- d 不参与梯度计算——手动更新；u 适当设置避免 bias 震荡
- 开销：r=32 时仅 32 个 float，r 步 O(r) 操作，可忽略
- 代码：https://github.com/gfyddha/FlyLoRA

涉及论文标题：
- FlyLoRA: Boosting Task Decoupling and Parameter Efficiency via Implicit Rank-Wise Mixture-of-Experts

## GLU / GeGLU (Gated Linear Unit / GELU-gated Linear Unit)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gated Linear Unit (GLU) 是一种门控前馈网络结构，由 Dauphin et al. (2017) 在语言建模中首次提出。GLU 计算两个线性投影的 component-wise 乘积，其中一个经过激活函数作为门控：

$$GLU(x) = (xW_1 + b_1) \otimes \sigma(xW_2 + b_2)$$

其中 $\otimes$ 表示逐元素乘积，$\sigma$ 为 sigmoid 激活函数。

Shazeer (2020) "GLU Variants Improve Transformer" 系统研究了用不同激活函数替代 sigmoid 的变体，发现 GELU-gated (GeGLU) 和 Swish-gated (SwiGLU) 表现最佳：

$$GeGLU(x) = (xW_g + b_g) \otimes GELU(xW_v + b_v)$$

其中 GELU (Hendrycks & Gimpel, 2016) 为 $GELU(x) = x \cdot \Phi(x)$，$\Phi$ 是标准正态分布的 CDF。

在 GLaM 中，非 MoE 层使用 GLU + GeGLU 激活替代标准 ReLU+Linear：先计算 gate = GeGLU(x·W_g) 和 value = x·W_v，逐元素乘积 gate * value，最后通过 W_o 映射回模型维度。MoE expert FFN 内部也使用 GeGLU 激活。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GLaM 非 MoE FFN 层（使用 GLU + GeGLU）前向传播
# 输入: x [B, S, M=8192]
# 权重: W_g [M, H=32768], W_v [M, H=32768], W_o [H, M]

gate_logits = x @ W_g    # [B, S, H]
value = x @ W_v          # [B, S, H]

# GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
gate = GELU(gate_logits)  # GeGLU 激活

gated_output = gate * value  # 逐元素门控
output = gated_output @ W_o  # 输出投影

# 注：为保持参数量对等，H 约为标准 FFN hidden dim 的 2/3
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GLU 变体已成为现代 LLM 的标准组件：LLaMA 系列使用 SwiGLU（SiLU/Swish gate），GLaM 使用 GeGLU。实现时 hidden dim 缩减为 2/3 以补偿第三个 weight matrix 的参数开销。GELU 通过近似公式高效计算：`0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))`。在 HuggingFace 中 LLaMA 的 SwiGLU 实现为 `down_proj(act_fn(gate_proj(x)) * up_proj(x))`。

涉及论文标题：
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

**SwiGLU 变体**：SwiGLU 使用 SiLU (Swish) 作为门控函数替代 GELU：`SwiGLU(x) = (xW_g + b_g) ⊗ SiLU(xW_v + b_v)`，其中 `SiLU(x) = x · σ(x)`。Hunyuan-Large 在所有 FFN（包括 shared/specialized expert FFN）中使用 SwiGLU 作为激活函数。SwiGLU 已成为现代 LLM 最常用的激活函数，LLaMA、Qwen、DeepSeek 系列均采用。

## Relative Positional Bias (Transformer-XL Style)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Relative Positional Bias 是 Transformer 中的一种位置编码方法（Dai et al. 2019, Transformer-XL），在 attention score 计算中直接添加一个仅依赖于相对位置 (i-j) 的可学习偏置项，替代绝对位置编码嵌入到 input 中的方式：

$$Attention(Q, K, V) = softmax\left(\frac{QK^T}{\sqrt{d_k}} + B_{rel}\right)V$$

其中 $B_{rel}[i, j] = b_{i-j}$，$b$ 是一个可学习的 bias table。相对距离通常 clip 到 [-k, k] 范围。

在 GLaM 中，每层维护独立的 per-layer Relative Positional Bias，替代标准绝对位置编码。这使模型能更好地处理变长序列和长距离依赖。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GLaM Attention + Relative Positional Bias
# 输入: x [B, S, M=8192], nheads=128, dhead=128
# bias_table: [2k+1, nheads] 可学习参数

scores = Q @ K^T / sqrt(dhead)  # [B, nheads, S, S]

# 为每个 (i, j) 查相对位置 bias
for i, j in range(S):
    dist = clip(i - j, -k, k)
    bias[i, j] = bias_table[dist + k]  # [nheads]

scores += bias
output = softmax(scores) @ V
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：预计算相对距离矩阵 [S, S]，查 bias table 后 broadcast 到 attention scores。优势是天然支持任意长度外推（超出范围的相对距离被 clip）。T5 使用简化的 bucket 版本（32 个 bucket），GLaM 使用 per-layer 独立 table。在现代框架中通常通过 `torch.nn.Embedding` 存储 bias table。

涉及论文标题：
- GLaM: Efficient Scaling of Language Models with Mixture-of-Experts

## Expert Affinity / Co-Activation Pattern in SMoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Affinity（专家亲和力）是 Sparse MoE 中两个 expert 被同一 token 同时激活的频率。在 top-k routing 下，每个 token 每层选择 k 个 expert。Affinity matrix $A \in \mathbb{R}^{n \times n}$ 记录所有 token 上 expert pair (i, j) 的共激活次数。GRACE-MoE 在 offline profiling 阶段通过 calibration data 构建 per-layer affinity matrix，发现 expert 间存在强 co-activation pattern——某些 expert 几乎总是一起被选中（处理特定领域知识），另一些几乎从不同时出现。C2R (Zhang et al. 2025a) 也独立发现此模式用于 collaboration-constrained routing。GRACE-MoE 将其作为 spectral clustering 的输入来指导 expert grouping——高 affinity expert 对放在同 GPU/同节点以减少跨设备 All-to-All 通信。Cross-dataset transfer 实验表明 affinity pattern 在数据集间稳定（最差 latency 增加 ≤4.52%），意味着 offline placement 可跨 dataset 复用。

从算法pipeline角度拆解术语：

```
# Profiling 构建 affinity matrix (per layer)
A[l] = zeros(n_experts, n_experts)
for token t in calibration_data:
    topk = router[l](h_t)  # k=6 or 8
    for i in topk:
        for j in topk:
            if i != j: A[l][i][j] += 1

# Affinity 指导 grouping:
# 高 A[i][j] → 同组 → 减少跨设备通信
# 低 A[i][j] → 可分到不同组 → 保持负载灵活
# Cross-node: fully non-uniform grouping (无 size 约束)
# Intra-node: controlled non-uniform (ratio r 约束 size deviation)
C = SpectralClustering(A, D)
```

术语一般如何实现？如何使用？

- Calibration data 通常数千到数万 token，从训练/验证集采样
- Affinity matrix 可直接作为 spectral clustering 的 weighted adjacency（无需额外归一化）
- 混合 dataset profiling 获得最鲁棒的 affinity estimation
- C2R 用 affinity 限制 routing（token 只能选组内 expert），GRACE-MoE 用 affinity 指导 placement 且保持 routing 不变（lossless）

### Expert-Expert Collaboration (ECC) as Dropping Criterion (Jaiswal et al. 2025)

MC-Suite 中的 ECC (Expert-Expert Collaboration) 准则从 pruning 视角利用 co-activation 模式：给定 calibration data，定义 collaboration matrix C_{p,q} = Σ 1[K_i ∩ {E_p, E_q} == {E_p, E_q}]（两 expert 共同被路由到同一 token 的次数）。高 collaboration 的 expert pair → 一个可被丢弃（因为另一个可覆盖相同 token 的处理任务）。具体丢弃决策：从 collaboration matrix 中选 min/max 值的 expert pair，再结合 EUF（Expert Usage Frequency）选择使用频率更低的那个丢弃。这与 GRACE-MoE/HD-MoE 用 affinity 做 placement grouping 不同——ECC 将 co-activation 信息用于压缩而非通信优化。

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference
- HD-MoE: Hybrid and Dynamic Parallelism for Mixture-of-Expert LLMs with 3D Near-Memory Processing

### HD-MoE 中的 Co-Activation
HD-MoE 独立发现并利用了 expert co-activation 模式（图 3c 的 Expert Routing Affinity heatmap，值 (i,j) 表示给定 expert i 被激活时 expert j 也被激活的条件概率）。HD-MoE 将此模式用于通信模型：定义 expert group g，f_g 为 group co-activation 频率，t̂_comm = (4/BW)·max_c{ Σ_g (Π_{i∈g} ⌈P_ic⌉)·f_g·B·h }。Co-activation 量化使 LP 能准确估计不同 placement 方案的通信开销。

## Non-Uniform Hierarchical Expert Grouping for Distributed MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Non-Uniform Hierarchical Expert Grouping 是 GRACE-MoE 的 expert placement 策略，替代传统的 uniform grouping（每设备等量 expert）。分层设计：跨节点层面使用 fully non-uniform（无 size 约束）最大化 intra-node affinity 以减少昂贵的跨节点通信；节点内层面使用 controlled non-uniform（ratio r 约束 group size deviation $\delta = E \cdot r$）保留 affinity 的同时限制负载倾斜。r 通过绘制 U(r)（intra-group affinity utilization）vs S(r)（size deviation）曲线取 knee point 确定。r=0 即 uniform grouping，r=1 即 fully non-uniform。

从算法pipeline角度拆解：

```
E = floor(n_experts / D); delta = max(1, round(E * r))
{C_d} = SpectralClustering(A, D)
for each oversized C_d: trim to num_max, push overflow to Omega
for e in Omega: assign to group maximizing intra-group affinity
for undersized groups: move weakest-affinity experts from oversized groups
# U(r) = sum_{C} sum_{i,j in C} A[i,j] / sum_{i<j} A[i,j]
# S(r) = sqrt(1/D * sum (|C_d| - E)^2)
# Select r at knee point of (S(r), U(r))
```

Table 2 验证：controlled non-uniform (r=0.15) 实现 end-to-end 5698ms vs uniform 6328ms vs fully non-uniform 5747ms。Fully non-uniform 通信最优（2826ms All-to-All）但 GPU idle 从 502ms 增至 617ms，controlled 实现最佳平衡。

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

## Dynamic Expert Replication based on Load Skew

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Dynamic Expert Replication (DR) 是 GRACE-MoE 的自适应 expert 复制策略，用于补偿 affinity-based grouping 的计算负载倾斜。定义 load skew factor $\rho = W_{\max} / \bar{W}$，由 $n_{\text{replica}} = \min(\max(1, \lfloor \rho \rfloor), n_{\text{gpu}} - 1)$ 确定副本数。仅复制 heaviest group 中最热的 expert（cumulative load > $W_{\max} \cdot n_{\text{replica}}/(1+n_{\text{replica}})$），作为 secondary copies 放置到最少负载的 GPU。对比 fixed replication（1 replica always）：GPU idle 仅 −1.59%；DR：−19.71%，且 GPU load std 从 +90.03%（HG only）降至 +31.92%（HG+DR+WRR）。

从算法pipeline角度拆解：

```
rho = W_max / W_mean; n_replica = min(max(1, floor(rho)), n_gpu - 1)
# In heaviest group, sort experts by load, select hot experts:
#   cumulative_load > W_max * n_replica / (1 + n_replica)
# Post-replication load prediction (for routing weights):
W_p = W_max / (n_replica + 1)  # evenly split assumption
W'_max = W_max - W_r + W_p; W'_i = W_i + W_p
# routing weights ∝ 1/W' (inverse proportional)
```

涉及论文标题：
- GRACE-MoE: Grouping and Replication with Locality-Aware Routing for Efficient Distributed MoE Inference

## GatePro

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GatePro 是 ByteDance Seed 提出的一种无参数（parameter-free）MoE gating 优化方法，通过局部竞争机制直接提升 expert 选择的多样性（diversity），而非仅关注 token 分配的负载均衡。核心思路：(1) 计算 gating weight matrix W_g 各行向量间的 cosine similarity matrix S_{ij} = ⟨w_{g,i}, w_{g,j}⟩ / (|w_{g,i}|·|w_{g,j}|) 来识别功能相似的 expert 对；(2) 对每个 expert i 找到最相似的 j*(i) = argmax_{j≠i} S_{ij}；(3) 在 token 级根据 logit 比较决定竞争 winner，对 loser 施加固定惩罚 λ=10^{-4} 抑制其激活。该方法无额外可学习参数，可 hot-swappable（训练中途启用/禁用），计算开销极小（cosine similarity O(N²d)，per-token competition O(N)）。GatePro 与辅助平衡损失（LBL）互补而非替代——LBL 保证 token 分配的统计均衡，GatePro 保证 expert 选择的功能多样性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

GatePro 在标准 MoE 的前向传播中插入 gate-level competition：

```
# === GatePro MoE Forward Pass ===
# Input: token x, gating weights W_g, penalty λ=1e-4

# Step 1: Original logits
logits = W_g @ x  # [N], N=128/256

# Step 2: Gate similarity (periodically updated, not per-token)
S = cosine_similarity(W_g)  # [N, N], diagonal set to -inf

# Step 3: For each expert i, find most similar counterpart
j_star = argmax(S, dim=1)  # [N]

# Step 4: Localized competition — loser gets penalty
l_competitor = gather(logits, j_star)  # competitor logits
mask = (logits < l_competitor)         # loser positions
logits_tilde = logits + mask * (-lambda)  # apply penalty

# Step 5-8: Standard top-k + softmax + weighted combination
topk_idx = topk(logits_tilde, k=6)
alpha = softmax(logits_tilde[topk_idx])
output = sum(alpha[j] * E[topk_idx[j]](x) for j in range(6))
```

实验数据：Seed-MoE-0.7B/7B (128 experts): MMLU-Pro 21.8% vs baseline 20.5% (500B tokens); GSM8K 45.0% vs 43.0%。Seed-MoE-1.3B/13B (1.2T tokens): MMLU-Pro 31.6% vs 30.6%, BBH 50.7% vs 49.8%。OLMoE-1B/7B (400B tokens): Overall 62.5% vs 61.8%。Expert utilization: Layer 14 零激活 convergence 从 3000 steps 缩短至 1500 steps。256 experts 下深层的加速优势更显著。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GatePro 以 hook 形式注入现有 MoE 代码的 gating 模块：在 top-k 选择前插入 competition penalty 计算，维护 gating similarity buffer 周期性更新 S 矩阵，使用 boolean flag 控制 hot-swap。适用场景：MoE pretrain（N≥64 experts）、continuous training、深层 MoE 层的 diversity 增强。可与任何 top-k routing 方案（softmax、sigmoid）兼容。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

## Expert Selection Diversity in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Selection Diversity 指 MoE 模型中被同一 token 选中的 top-k experts 在功能上的互补性——即它们是否学习了不同的、互补的知识而非冗余的、相似的功能。这不同于 load balance（token 分配的数量均衡）。当 expert i 和 expert j 的 gating weight vectors w_{g,i} 和 w_{g,j} 高度相似（cosine similarity → 1）时，它们对相似类型的 token 产生高 logit，倾向于被同时激活（co-activation），从而学习到冗余的功能。GatePro 通过三个指标量化 diversity：Average Cosine Similarity（越低越好）、Average Angle（越大越好）、Spectral Entropy（越高越好，表示 expert 选择的分布更均匀）。GatePro 实验显示其在所有层上都持续维持更优的 diversity metrics，特别是在深层（Layer 16）中 baseline 的 similarity 持续上升而 GatePro 保持稳定低位。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Diversity 计算的三个指标：

- Average Cosine Similarity: $\frac{2}{N(N-1)} \sum_{i<j} |S_{ij}|$，衡量 all expert pairs 的平均 gating weight 对齐程度
- Average Angle: $\frac{2}{N(N-1)} \sum_{i<j} \arccos(S_{ij})$，互补于 cosine similarity，角度越大表示 expert 越正交
- Spectral Entropy: $-\sum_i \tilde{\sigma}_i \log \tilde{\sigma}_i$，其中 $\tilde{\sigma}_i$ 是 similarity matrix S 的标准化奇异值，反映 expert 行为模式的整体分散度

低 diversity 场景的例子（baseline）：
```
Expert 3 和 Expert 17 的 w_{g,3} ≈ w_{g,17} (S_{3,17}=0.92)
Token x: logits[3]=0.8, logits[17]=0.79 → both in top-6
→ 两个几乎等价的 FFN 被同时激活，浪费计算资源
```

高 diversity 场景（GatePro）：
```
Expert 17 被惩罚: logits[17] -= 1e-4 → 0.79-1e-4 < logits[25]=0.08
→ Expert 25 取代 Expert 17 进入 top-6
→ 6 个功能互补的 expert 执行不同计算
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Diversity 提升可通过：GatePro 的 localized competition（最直接的方法），diversity regularization loss（在 training objective 中加入高 S_{ij} 对的惩罚项），正交初始化策略，或动态 expert merging/pruning。GatePro 的优势在于 parameter-free 且不影响 loss landscape。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

## Localized Competition Mechanism in MoE Gating

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Localized Competition Mechanism 是 GatePro 的核心操作——不在所有 expert 之间引入全局竞争，而是仅在最相似的 expert 对之间进行 targeted pairwise competition。设计原理：(1) 通过 cosine similarity 找到每个 expert i 的最相似 counterpart j*(i)；(2) 对每个 token，比较 pair 中两方的 logits；(3) 仅对 loser 施加固定惩罚 λ=10^{-4}。关键优势：仅干扰功能冗余的 expert 对（dissimilar experts 不受影响），竞争粒度是 pairwise 而非 global（每个 expert 最多一个直接对手），惩罚是 token-specific 的（同一 expert 在不同 token 上可能赢也可能输，保持路由灵活）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Vectorized localized competition
l_competitor = logits[j_star]       # gather competitor logits
loser_mask = (logits < l_competitor) # True for losers
logits_tilde = logits + loser_mask * (-1e-4)

# Example (N=8, λ=1e-4):
# logits:        [0.1,  0.05, 0.8,  0.79, 0.3,  0.2,  0.15, 0.4]
# j_star:        [2,    0,    1,    0,    7,    3,    0,    4]
# l_competitor:  [0.8,  0.1,  0.05, 0.1,  0.4,  0.79, 0.1,  0.3]
# loser_mask:    [T,    F,    F,    F,    T,    T,    F,    F]
# logits_tilde:  [0.0,  0.05, 0.8,  0.79, 0.2,  0.1,  0.15, 0.4]
# Note: expert 0 penalized (0.1→0.0), expert 4 (0.3→0.2), expert 5 (0.2→0.1)
```

设计选择：(1) λ=10^{-4} — 足够改变 top-k 排序（logit 差异通常在 10^{-3} 到 10^{-1} 量级），但对数值稳定性无影响；(2) 每对 expert 最多惩罚一个 — 避免 both penalized 导致 none selected；(3) penalty 在 logit space — 比 probability space 更稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

适用于 N≥64 的 MoE 架构（larger pool → more redundancy → 更大收益），深层 MoE 层（specialization 更难），pretrain 和 continuous training 阶段。可与任何 top-k routing（softmax/sigmoid）集成。256 experts 下 GatePro 的优势比 128 experts 更显著。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

## Hot-swappable Training in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hot-swappable Training 是 GatePro 的特性——localized competition 机制可在训练期间随时启用或禁用，无需额外参数、架构修改或学习率调整。与 auxiliary loss 不同（其启用/禁用改变 loss landscape），GatePro 的 hot-swap 在 gate logit 层面操作，不影响 loss 计算。论文验证了"训练遗产效应"（training legacy effect）：GatePro 训练阶段建立的 expert diversity 对后续标准 MoE 训练持续产生正面影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
use_gatepro = True  # hot-swap flag

for step, batch in enumerate(dataloader):
    for layer in model.layers:
        x = layer.moe_gate(x, use_gatepro=use_gatepro)
    ...

    if step == hotswap_step:  # e.g., at 400B tokens
        use_gatepro = False   # disable, continue as standard MoE
```

Hot-swap 实验（Table 3, 0.7B/14B, 256 experts, 500B tokens）：
- 100B GatePro → 400B MoE: MMLU-Pro 28.7%
- 400B GatePro → 100B MoE: MMLU-Pro 30.0%
- 500B GatePro (Full): MMLU-Pro 30.1%
越长的 GatePro 训练产生越好的最终性能，400B+100B 接近 Full GatePro。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际价值：(1) 资源优化——计算密集的早期训练用 GatePro 建立 diversity，后期切换 standard 节省开销；(2) 灵活部署——不同阶段按需切换，无需重新初始化或修改 checkpoint；(3) 实验探索——研究者可测试不同时间窗口的 diversity 提升效果。与 auxiliary loss 相比，hot-swap 不影响 loss landscape，切换对训练稳定性无影响。

涉及论文标题：
- GatePro Parameter-Free Expert Selection Optimization for Mixture-of-Experts Models

## Graph Distribution Shift

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Graph Distribution Shift 是指图数据中训练分布（源分布 D_s）与测试分布（目标分布 D_t）之间的差异，源于图结构、节点特征、边特征或其组合的自然变化。与标准 ML 中的 covariate shift 不同，图分布偏移具有独特性质：(1) **多维性**——偏移可来自图大小变化、节点度变化、特征噪声、边密度变化、子图结构变化等多种维度，且可组合形成复合偏移；(2) **实例级异质性**——同一目标分布中不同节点/图实例可能经历不同程度和类型的偏移；(3) **非 IID 传播**——偏移通过邻居关系和消息传递机制在图结构中传播。GraphMETRO 将这些偏移建模为多个 shift component 的混合（Assumption 1: 任意分布偏移可建模为 ≤k 个 transform classes 的混合），通过 gating model 识别成分、expert models 缓解各成分的影响。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GraphMETRO 中分布偏移的分解与处理
stochastic_transforms = {
    'drop_edge':    随机删除边 (p ∈ [0.3, 0.5]),
    'add_edge':     随机添加边 (p),
    'drop_node':    随机删除节点 (p),
    'noisy_node_feat': 向节点特征加 Gaussian noise,
    'random_subgraph': 随机 k-hop 子图采样,
}

# τ^{(k)} = τ_{i1} ∘ τ_{i2} ∘ ... ∘ τ_{ik} 模拟 G 在 D_t 中的表现
# Gating: w = ϕ(G) → 识别各 τ_i 对当前 instance 的贡献
# Expert: ξ_i(τ_i(G)) ≈ ξ_0(G) → 消除对应 τ_i 的影响
# Aggregate: h = Σ Softmax(w)[i] · ξ_i(G) → 对组合偏移不变
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际使用：(1) 根据领域知识选择 transform 函数集合——通用图变换覆盖多数场景，特定领域（分子图）需定制 transforms；(2) 可通过目标域样本在 embedding 空间中测量变换后数据集与目标样本的距离来筛选相关 transforms；(3) transform 数量影响性能——过多引入噪声降低性能，过少无法充分覆盖偏移空间。GraphMETRO 代码：https://github.com/Wuyxin/GraphMETRO。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

## Referential Invariant Representation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Referential Invariant Representation（参照不变表示）是 GraphMETRO 的核心概念（Definition 1）：给定图 G、随机变换 τ 和 reference model ξ_0，函数 ξ* 产生的表示被称为 referentially invariant w.r.t. τ，当且仅当 ξ_0(G) ≈ ξ*(τ(G))，∀G ∈ supp(D_s)。本质是通过 reference model ξ_0 的表示空间作为"锚点"来对齐所有 expert 的输出——每个 expert ξ_i 学习对 τ_i(G) 编码以匹配 ξ_0(G)（原图在 reference model 中的表示），而非要求 ξ_i(τ_i(G)) = ξ_i(G)（自不变性）。这解决了多个独立 expert 表示空间不兼容的问题——所有 expert 通过同一 reference space 间接对齐，使 weighted sum aggregation 在数学上有意义。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Referential Invariant Representation 的训练
for G in D_s:
    z_0 = ξ_0(G)              # reference "anchor" 表示（原图）
    G_trans = τ^{(k)}(G)      # 应用组合 shift transforms
    z_i = ξ_i(G_trans) for i=0..K  # 各 expert 对变换后图的编码
    w = ϕ(G_trans)            # gating 预测 shift mixture
    h = Σ Softmax(w)[i] · z_i # 加权聚合
    d = (1/n)·||h - z_0||_F   # Frobenius norm distance
    L_align = λ · d           # λ=1 (所有实验)
    # h(τ_i(G)) = h(G)  (Theorem 1, 单 shift 不变性)
    # h(τ^{(k)}(G)) ≈ h(G) (Theorem 2, 组合 shift 近似不变性)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键点：(1) ξ_0 在源数据 D_s 上正常训练，作为"in-distribution expert"；(2) alignment term 权重 λ 至关重要——λ=0 时 WebKB 41.11%→18.79%，验证 alignment 的必需性；(3) Frobenius norm 简单有效：d(z₁,z₂) = (1/n)·||z₁ - z₂||_F = (1/n)·√(Σ(z₁ᵢ - z₂ᵢ)²)；(4) alignment 同时在 τ^{(k)} 组合上执行，保证对复合偏移的不变性；(5) L2 不反向传播到 gating model，避免 gating 和 alignment 目标冲突。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

## GOOD (Graph Out-of-Distribution) Benchmark

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

GOOD (Graph Out-of-Distribution) Benchmark 是 NeurIPS 2022 提出的 GNN 分布外泛化基准（Gui et al. 2022）。区别于标准 ML benchmark 的随机数据划分，GOOD 提供基于真实 covariate shift 的数据集划分（如按大学域名、用户语言域、分子 scaffold 划分 train/val/test），模拟真实部署中的自然分布偏移。GraphMETRO 使用四个 GOOD 数据集：WebKB（5-class 节点分类，按大学域名划分，target=Washington domain）、Twitch（二分类，按用户语言域划分，metric=ROC-AUC）、Twitter（grammar tree graph 分类，不同 domain 的句子结构形成 shift）、GraphSST2（sentiment tree graph 分类）。GOOD 提供标准化 encoder/classifier 和评估协议。代码：https://github.com/divelab/GOOD/tree/GOODv1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GOOD benchmark 使用 PyG 接口
from GOOD import get_dataset
dataset = get_dataset(root='./data', dataset_name='WebKB', domain='domain')
# train: Cornell/Wisconsin/Texas, val/test: Washington (natural domain shift)
# 评估不提供 domain 标签（GraphMETRO 不需要 domain info）
# GraphMETRO 使用 GOOD 的统一架构：GCN for node tasks, GIN+VirtualNode for graph tasks
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GOOD 提供统一的数据加载、encoder、classifier 和评估协议，确保不同方法间的公平比较。GraphMETRO 使用 GOOD 的标准 GCN/GIN 作为 backbone，与 ERM、IRM、EERM、DIR、GSAT 等 baseline 在同一框架下比较。训练/验证/测试划分固定，减少划分随机性对结果的影响。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

## Invariant Learning for Graph Out-of-Distribution

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

图分布外泛化中的不变性学习（Invariant Learning for Graph OOD）是一类假设存在跨环境不变的图结构或图表示的方法。代表方法：DIR（因果干预蒸馏 causal subgraph）、EERM（环境划分学习 invariant representations）、GSAT（stochastic attention + information bottleneck）、CIGA（因果不变性表示）。核心范式：将源数据划分为伪环境 {E_1,...,E_K}，学习满足 P(Y|f(G), E=e_i) ≈ P(Y|f(G)), ∀e_i 的 encoder f。GraphMETRO 指出该范式的三方面局限：(1) 环境空间因组合爆炸不可行（环境 = 节点子集 × shift 类型组合）；(2) 忽略 instance-wise heterogeneity（关注 group-level patterns）；(3) 依赖 domain/environment 标签。GraphMETRO 通过 shift component decomposition（K 个 base transforms + continuous weight vector → 无限环境）和 MoE 架构的 instance-adaptive gating 替代传统 env-based 方法。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 标准图不变性学习 (EERM-style) vs GraphMETRO
# Standard:
environments = [E_1, ..., E_K]  # 从源数据构建伪环境
for G in D_s, E_k in environments:
    z = encoder(G in E_k)
    L_k = CE(classifier(z), y)
L_inv = var(best_classifier_params across environments)
L = Σ L_k + β · L_inv

# GraphMETRO (no environment dependency):
w = ϕ(G)  # continuous weight ∈ R^{K+1}, 替代 discrete environments
h = Σ Softmax(w)[i] · ξ_i(G)  # instance-adaptive expert combination
L = CE(μ(h), y) + λ·||h - ξ_0(G)||_F  # task + alignment
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实际使用的主要挑战：(1) 环境构建质量直接影响不变性——划分不当导致 spurious invariance；(2) 需要 domain/environment 标签，但许多场景不可得。GraphMETRO 的替代方案不需要 domain 标签，gating model 从图数据自动推断偏移成分。当限制 gating 输出为 binary 时，GraphMETRO 可退化为传统 finite-environment invariant learning。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

## Instance-wise Heterogeneity in Graph Distribution Shifts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Instance-wise Heterogeneity 是图分布偏移中的关键现象：同一目标分布的图/节点实例经历不同类型和程度的偏移。在 WebKB 数据集中，两个不同网页节点在目标域的特征变化程度截然不同——尽管都经历了 source→target 的偏移，各自的 shift pattern 不同。标准 invariant learning 关注 group-level patterns，缺乏对实例间差异的建模。GraphMETRO 通过 gating model ϕ 对每个 instance 输出个性化权重 w ∈ R^{K+1} 来编码该 instance 的偏移成分分布——w 连续且 instance-dependent，支持无限种偏移组合，实现 instance-adaptive 处理。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Instance-wise 异质性处理
# 数学表达: h(G) = Σ_{i=0}^K Softmax(ϕ(G))[i] · ξ_i(G)
# w 为 instance-dependent → 不同 instance 得到不同 expert 组合

# WebKB 例子:
# Node u¹ (内容大变化/结构不变): w=[0.05,0.05,0.5,0.1,0.2,0.1]
#  → noisy_node_feat expert (idx 2) 主导
# Node u² (内容小变化/结构变化): w=[0.05,0.4,0.1,0.3,0.1,0.05]
#  → add_edge (idx 1) + drop_node (idx 3) 主导

# 连续 w → 无限种 expert 组合 → 任意粒度偏移自适应
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现依赖 MoE 的 softmax routing——gating model 使用 BCE loss（多标签二分类）训练，使 ϕ 对每个 τ_i 的敏感性独立于其他 τ_j，确保 w 各分量独立反映对应 shift component 的存在与否。Distribution shift discovery（Figure 4b）验证了 gating model 能准确识别目标分布的全局偏移类型（WebKB: add_edge 主导，Twitch: noisy_node_feat+drop_node 主导），gating accuracy 达 92.4%/93.8%。

涉及论文标题：
- GraphMETRO Mitigating Complex Graph Distribution Shifts via Mixture of Aligned Experts

## Expert-Specific Operators (ESMM, ESS, ESTMM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Specific Operators 是 HEXA-MoE 提出的替代传统 GeMM/grouped GeMM 的 MoE 计算范式。核心洞察：传统 MoE 使用 GeMM 接口时，由于各 expert 的 workload 动态变化，必须通过 token padding（填充到 capacity）或 discarding（丢弃超出 capacity 的 token）来构造规整的 batch，这产生了冗余 FLOPs 和冗余内存。Expert-Specific Operators 将 MoE 计算从 "GeMM 视角"（先重排为规则 batch 再调 GeMM）重新定义为 "Expert-Specific 视角"（不重排 token，直接做 expert-wise 计算）。三个基本算子：

- **ESMM (Expert-Specific Matrix Multiplication)**：给定输入 x [N, D_i]、权重 W [E, D_i, D_o]、偏置 b [E, D_o] 和 routing choice R(x) [N]，输出 y [N, D_o]，其中 y_i = x_i @ W_{R(x_i)} + b_{R(x_i)}。每个 token 仅与其路由 expert 的权重做矩阵乘法，无需 padding。
- **ESS (Expert-Specific Summation)**：给定输入 x [N, D] 和 routing choice R(x) [N]，输出 y [E, D]，其中 y[e] = Σ_{i: R(x_i)=e} x_i。按 expert 分组累加，用于 backward 中计算 bias 梯度。
- **ESTMM (Expert-Specific Transposed Matrix Multiplication)**：给定两个输入 x1 [N, D1]、x2 [N, D2]（共享 routing choice R(x)），输出 y [E, D1, D2]，其中 y[e, i, j] = Σ_{m: R(x_m)=e} x1[m,i] · x2[m,j]。expert-wise 外积累加，用于 backward 中计算权重梯度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

以 top-1 routing 的 MoE 层为例，对比传统 GeMM 方法和 Expert-Specific Operators 方法：

```
# === 传统 GeMM 方法（Tutel）===
# Forward: 需要 dispatch + token padding
for each expert e:
    tokens_e = dispatch(x, R, e)              # 收集路由到 expert e 的 token
    tokens_e_padded = pad(tokens_e, capacity)  # padding 到 expert capacity
    y1_e = GeMM(tokens_e_padded, W1[e])       # [capacity, D_mid]
    y2_e = F(y1_e)                             # 激活
    y_e = GeMM(y2_e, W2[e])                    # [capacity, D_o]
y = combine({y_e})                            # 按原始顺序重组，丢弃 padding 部分

# === Expert-Specific Operators 方法（HEXA-MoE）===
# Forward: in-place 计算，无需 padding/dispatch/combine
y1 = ESMM(x, W1, b1, R(x))                   # [N, D_mid]
y2 = F(y1)                                     # 激活函数（如 GELU）
y  = ESMM(y2, W2, b2, R(x))                   # [N, D_o]

# Backward: auto-diff 提供 ∂ℓ/∂y
∂ℓ/∂b2 = ESS(∂ℓ/∂y, R(x))                     # [E, D_o]
∂ℓ/∂W2 = ESTMM(y2, ∂ℓ/∂y, R(x))               # [E, D_mid, D_o]
∂ℓ/∂y2 = ESMM(∂ℓ/∂y, W2^T, null, R(x))        # [N, D_mid]
∂ℓ/∂y1 = ∂ℓ/∂y2 ⊙ F'(y1)                      # element-wise
∂ℓ/∂b1 = ESS(∂ℓ/∂y1, R(x))                    # [E, D_mid]
∂ℓ/∂W1 = ESTMM(x, ∂ℓ/∂y1, R(x))               # [E, D_i, D_mid]
∂ℓ/∂x  = ESMM(∂ℓ/∂y1, W1^T, null, R(x))       # [N, D_i]
```

Top-k routing 扩展：对 k 个 routing choice 分别执行 ESMM，输出为 k 个 ESMM 结果的累加。中间结果 tensor 的内存分配仅扩展为 k 倍。使用 atomicAdd 聚合各 expert 对同一 token 的贡献。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Expert-Specific Operators 在 HEXA-MoE 中通过 CUDA kernel 实现（也提供 Triton 实现）。核心依赖 **re-index vector**（按 routing choice 重排 token indices）作为 I/O 指导，使同 expert 的 token 在内存中逻辑连续，提高 GPU 访存局部性。ESMM kernel 使用 nvcuda::wmma 接口调用 Tensor Core 做 16×16×16 矩阵乘法。开源实现：https://github.com/UNITES-Lab/HEXA-MoE。

使用方式：替代 PyTorch 中 MoE 层的标准 nn.Linear + routing 组合。HEXA-MoE 提供 `hexa_moe.moe` 模块，通过 `MoE_Cascaded` 类构建 MoE 层，内部自动使用 ESMM/ESS/ESTMM 替代 GeMM。

涉及论文标题：
- HEXA-MoE: Efficient and Heterogeneous-aware MoE Acceleration with ZERO Computation Redundancy
- HMoE: Heterogeneous Mixture of Experts for Language Modeling

## HMoE (Heterogeneous Mixture of Experts)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

HMoE 是一种 MoE 变体，核心区别在于同一 MoE 层中的不同 expert 具有**不同的参数量/容量**（不同的 FFN hidden dimension），而非传统 MoE 中所有 expert 大小相同。每个 expert 仍沿用标准 LLaMA-style FFN 设计：$e_i(\mathbf{x}) = \mathbf{W}_{o,i} \cdot (\text{SiLU}(\mathbf{W}_{g,i} \cdot \mathbf{x}) \odot (\mathbf{W}_{p,i} \cdot \mathbf{x}))$，但关键差异在于不同 expert 的 $\mathbf{W}_{g,i} \in \mathbb{R}^{h_{\text{input}} \times h_{\text{ffn},i}}$ 中的 hidden dim $h_{\text{ffn},i}$ 各不相同。例如在 HMoE-3B 主实验中，8 个 expert 的 hidden dim 按 arithmetic progression 设置为 {2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888}，large expert (5888 dim) 的参数量约为 small expert (2304 dim) 的 2.5×。异构设计使不同 expert 天然具有不同的表示容量和处理能力——大 expert 处理复杂 semantic token（如需要深度推理的后缀词），小 expert 处理简单 token（如冠词、介词）。

HMoE 面临的核心挑战：(1) 训练中 router 自然偏好激活大 expert（容量更强），导致小 expert 被闲置，总激活参数量不降反升；(2) 异构 expert 的不规则形状（不同 GEMM dim）给批量计算带来工程挑战，需使用 Megablocks 等 block-sparse kernel 替代传统统一形状 GEMM。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HMoE 层的 forward pass 伪代码（与 homogeneous MoE 的前向传播结构相同，差异在于各 expert FFN 的 hidden dim 不同 + 额外的 P-Penalty loss）：

```python
# HMoE Layer Forward (Top-P routing, arithmetic distribution)
# x: [B, S, h_input], 如 h_input=4096
# W_r: [h_input, N], N=8 experts
# expert_dims: [2304, 2816, ..., 5888]  # 异构 hidden dim

# Step 1: Router
P = softmax(x @ W_r)  # [B, S, N]

# Step 2: Top-P Routing (adaptive expert selection)
P_sorted, indices = sort(P, descending=True, dim=-1)
if P_sorted[0] >= p_threshold:  # p=0.6
    n_selected = 1
else:
    n_selected = min_k_where(cumsum(P_sorted) >= p_threshold)
selected_experts = indices[:, :, :n_selected]

# Step 3: Heterogeneous Expert Computation
output = zeros([B, S, h_input])
for e_idx in selected_experts:  # 每个 expert 的 hidden dim 不同
    mask = (tokens routed to expert e_idx)
    x_e = x[mask]                    # [n_e, h_input]
    gate_e = P[mask, e_idx]          # [n_e], 归一化后

    W_g = expert_weights[e_idx].W_g  # [h_input, h_ffn,e]
    W_p = expert_weights[e_idx].W_p  # [h_input, h_ffn,e]
    W_o = expert_weights[e_idx].W_o  # [h_ffn,e, h_input]

    # LLaMA-style SiLU-gated FFN (各 expert 的 h_ffn 不同)
    gate_out = SiLU(x_e @ W_g)       # [n_e, h_ffn,e]
    up_out = x_e @ W_p               # [n_e, h_ffn,e]
    hidden = gate_out * up_out       # element-wise
    expert_out = hidden @ W_o        # [n_e, h_input]

    output[mask] += gate_e.unsqueeze(-1) * expert_out

# Step 4: P-Penalty Loss (训练时)
# 对比传统 load balancing loss
L_pp = N * sum_i(M_i * P_hat_i)
# M_i = (1/T) * sum_t(indicator(e_i activated for t) * h_ffn,i)
# P_hat_i = (1/T) * sum_t(P_i,t)
# 激活大 expert 时 M_i 更大 → loss 更高 → 引导使用小 expert
```

异构 expert 的 token 分配行为：实验表明 smaller experts (2304 dim) 最常被激活的 top tokens 为简单冠词/代词 (the, such, your, these, most)，medium experts (3328-3840 dim) 处理具象语义词汇 (tables, valley, sun, day, war, water)，large experts (5376-5888 dim) 处理后缀/模糊 token (_ly, _zen, _icker, _decom, _inf)，验证了异构容量驱动的差异化 token 分配。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

HMoE 的实现基于 PyTorch，训练使用 DeepSpeed Zero2 + gradient checkpointing。关键实现考量：(1) 异构 expert 的不规则 GEMM shape 使用 Megablocks block-sparse kernel 高效批量计算；(2) P-Penalty loss 在每个 MoE 层的 forward 中计算 M_i（expert 激活次数 × hidden dim），系数设为 0.1；(3) 三种 expert 大小分布策略可通过配置 expert_dims 列表切换：arithmetic {9,11,13,15,17,19,21,23}（归一化比例）性能最优，geometric {1,2,4,8,16,32,64,128} 因过大的容量差距导致小 expert 训练不足，hybrid {1,1,1,1,2,2,4,4} 次优。代码尚未开源（论文声明 "Codes will be released upon acceptance"）。HMoE 使用 A800/H800 (80GB) GPU，AdamW optimizer (β1=0.9, β2=0.999), LR=1e-4 with 1000-step warmup, context=4096, batch=640, seed=12345。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling

## P-Penalty Loss (Parameter Penalty Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

P-Penalty Loss（Parameter Penalty Loss）是 HMoE 提出的训练辅助损失函数，专为解决异构 MoE 中 router 过度偏好大 expert 的问题设计。传统 MoE 使用 load balancing loss $L_{lb} = N \sum_{i=1}^{N} \mathcal{T}_i \cdot \hat{\mathcal{P}}_i$ 鼓励 expert 被均匀使用，但在异构 MoE 中这一目标不适用——因为 expert 大小不同，"均等使用"不等于"经济使用"。P-Penalty loss 将 expert 的大小（hidden dim $h_{\text{ffn},i}$）直接纳入损失函数：

$$L_{\text{P-Penalty}} = N \sum_{i=1}^{N} \mathcal{M}_i \cdot \hat{\mathcal{P}}_i$$

$$\mathcal{M}_i = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}\{e_i \in E^t\} \times h_{\text{ffn},i}$$

$$\hat{\mathcal{P}}_i = \frac{1}{T} \sum_{t=1}^{T} P_{i,t}$$

其中 $\mathcal{M}_i$ 是 expert i 的"加权激活计数"——激活次数 × expert 的 hidden dim，$P_{i,t}$ 是 router 分配给 expert i 对 token t 的门控概率。关键特性：(1) 激活大 expert 时 $\mathcal{M}_i$ 更大（因 $h_{\text{ffn},i}$ 更大），P-Penalty 更高，驱动模型优先使用小 expert；(2) 若所有 expert 大小相同（$h_{\text{ffn},i}$ 均等），P-Penalty 退化为标准 load balancing loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# P-Penalty Loss 计算（per MoE layer）
def p_penalty_loss(gate_probs, expert_assignments, expert_dims, N):
    """
    gate_probs: [B, S, N] 每个 token-expert 的 softmax 概率
    expert_assignments: [B, S, N] boolean mask, 标记各 expert 是否被激活
    expert_dims: [N] 各 expert 的 hidden dim
    """
    # P_hat_i: 各 expert 的平均门控概率
    P_hat = gate_probs.mean(dim=(0, 1))  # [N]

    # M_i: 各 expert 的加权激活计数
    # 对每个 token，如果 expert i 被激活，累加 h_ffn,i
    is_activated = expert_assignments.float()    # [B, S, N]
    M = is_activated.mean(dim=(0, 1)) * expert_dims  # [N], T 归一化

    # P-Penalty = N * Σ M_i * P_hat_i
    loss = N * (M * P_hat).sum()
    return loss

# 训练: L_total = L_lm + alpha * L_pp (alpha=0.1)
# Top-P 额外: L_total = L_lm + alpha * L_pp + beta * L_entropy (beta=3e-2)
```

P-Penalty vs Load Balancing Loss 的效果对比（Figure 7）：(1) Load balancing loss 无法阻止大 expert 被过度激活——虽然 expert 激活次数趋于均匀，但大 expert 每次激活的计算量更大；(2) P-Penalty loss 成功逆转激活比例——训练后期小 expert 激活率持续上升，大 expert 激活率下降；(3) 最终 HMoE-3B 的激活参数量从 1.23B（homogeneous MoE Top-P）降至 0.68B（HMoE Top-P），同时平均 benchmark 得分从 45.62 提升至 46.53。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

P-Penalty 在每个 MoE 层 forward 时作为辅助 loss 计算，与 language modeling loss 加权求和后反向传播。HMoE 中系数 α=0.1。使用方式：在 PyTorch 训练循环中，每个 MoE 层 forward 后收集 gate_probs 和 expert_assignments，计算各层 P-Penalty 并累加到总 loss。P-Penalty 替换（而非补充）传统 load balancing loss——当所有 expert 大小相同时等价，但对异构 MoE 更有效。局限：仅对 Top-K/Top-P 路由有效，不适用于 expert-choice routing（此时无 token-gate probability 概念）。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling

## Router Entropy Loss in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router Entropy Loss 是 MoE 中用于控制路由稀疏性的辅助损失函数，定义为 gate 概率分布的负熵：

$$L_{\text{entropy}} = N \sum_{i=1}^{N} P_i \cdot \log(P_i)$$

其中 N 为 expert 总数，$P_i$ 为 router 对每个 token 分配到 expert i 的平均 softmax 概率。该 loss 惩罚 router 输出过于均匀的概率分布（即接近 uniform 的高熵状态），从而**抑制**模型激活过多 expert。在 HMoE 的 Top-P routing 场景中，router entropy loss 尤为重要——Top-P routing 允许动态激活任意数量的 expert，训练中可能逐步增加激活数量（router 倾向于输出更均匀的概率以"安全地"激活更多 expert），导致稀疏性退化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```python
# Router Entropy Loss 计算
def router_entropy_loss(gate_probs, N):
    """
    gate_probs: [B, S, N] router 输出的 softmax 概率（未做 top-k/top-p masking）
    """
    # P_i: 各 expert 的平均路由概率
    P = gate_probs.mean(dim=(0, 1))  # [N]

    # L_entropy = N * Σ P_i * log(P_i)
    # 当 P_i 接近 uniform (1/N) 时 loss 最大 → 不期望
    # 当 P_i 接近 one-hot (某个 expert 概率近 1，其余近 0) 时 loss 最小 → 期望
    loss = N * (P * P.log()).sum()
    return loss
```

HMoE 的最终 training loss（Top-P routing 时）：
```
L_final = L_lm + α * L_P-Penalty + β * L_entropy
# α = 0.1 (P-Penalty coefficient)
# β = 3e-2 (Entropy loss coefficient)
```

Router entropy loss 的效果：(1) 防止 Top-P routing 在训练中激活 expert 数量无限制增长（entropy 高时 router 对 expert 的区分度低，Top-P 可能需累加更多 expert 才达到 threshold p）；(2) 鼓励 router 对少数 expert 给出高置信度（低熵），使 Top-P 更准确地按需选择真正相关的 expert。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 HMoE 中实现为每个 MoE 层 forward 后对 router softmax 输出（未经 top-k/top-p masking 的原始概率分布）计算 entropy loss，累加到总 loss。仅用于 Top-P routing（Top-K 固定激活数量，无需 entropy 控制）。系数 β 需权衡——太小则无法抑制 expert 数量增长，太大则 router 过度集中于少数 expert 导致负载不均衡。HMoE 使用 β=3e-2。首次由 Huang et al. (2024) 在 Top-P routing MoE 论文中提出。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling

## Expert Size Distribution Strategy in HMoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Size Distribution Strategy 是 HMoE 中定义各 expert 的 hidden dimension 相对大小的策略，直接决定异构 MoE 中 expert 之间的容量差异程度。HMoE 提出三种分布策略：

1. **Arithmetic Strategy（算术级数）**：expert 大小按等差序列分布，如相对比例 {9, 11, 13, 15, 17, 19, 21, 23}。特点：相邻 expert 容量差恒定，总差异相对温和（最大/最小≈2.5×），小 expert 仍有足够能力参与训练。HMoE 主实验采用此策略，训练最稳定。

2. **Geometric Strategy（几何级数）**：expert 大小按等比序列分布，如 {1, 2, 4, 8, 16, 32, 64, 128}。特点：容量差异极大（最大/最小=128×），突出关键 expert 的作用。但实验表现最差——极小 expert 缺乏足够容量，即使 P-Penalty loss 也无法充分激活它们（Figure 8 right）。

3. **Hybrid Strategy（混合策略）**：结合同构与异构，如 {1, 1, 1, 1, 2, 2, 4, 4}。特点：部分 expert 共享相同大小（形成"功能组"），组间有容量差异。假设某些场景需要多个相似能力的 expert 协同工作。实验表现优于 arithmetic（Figure 8 left），说明适量的"相似 expert 组"结合"组间异构"可能是最优方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

三种策略的 expert hidden dim 计算（以 HMoE-3B 为例，总 hidden=32768，8 experts）：

```python
# 归一化比例 → 实际 hidden dim 映射
total_sum = sum(ratio_list)  # e.g., arithmetic: 9+11+...+23 = 128
scale = total_hidden_dim / total_sum  # 32768 / 128 = 256
expert_dims = [r * scale for r in ratio_list]

# Arithmetic Strategy: 等差
# {9,11,13,15,17,19,21,23} × 256
# = {2304, 2816, 3328, 3840, 4352, 4864, 5376, 5888}

# Geometric Strategy: 等比 (ratio=2)
# {1,2,4,8,16,32,64,128} × (32768/255)
# ≈ {128, 257, 514, 1028, 2056, 4112, 8224, 16448}
# 问题：expert_0 (128 dim) 容量过小，几乎无建模能力

# Hybrid Strategy: 分组
# {1,1,1,1,2,2,4,4} × (32768/16) = 
# {2048, 2048, 2048, 2048, 4096, 4096, 8192, 8192}
```

HMoE 进一步实验了 arithmetic 策略在不同方差下的表现（Figure 11）：改变最大/最小 expert dim ratio 从 1:1（完全同构）到约 2.5:1（主实验设置）再到更大的比例，发现 loss 先上升后下降，存在一个最优异构度——ratio 约 2.5:1 时 loss 最低，验证了适度异构优于极端异构和完全同构。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Expert 大小分布在模型初始化时设定，通过定义各 expert 的 FFN 权重矩阵维度（W_g, W_p 的 out_dim, W_o 的 in_dim）实现。HMoE 基于 LLaMA 架构，每层 8 个 expert。使用时根据任务场景选择策略：(1) 需要最大化大 expert 能力且能接受训练不稳定 → geometric；(2) 需要稳定训练且平衡所有 expert 的参与度 → arithmetic；(3) 需要"功能组"协作（部分 expert 冗余）→ hybrid。未来方向可能包括可学习的异构度（训练中自适应调整 expert 大小）。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling

## Representation Collapse in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Representation Collapse（表示坍塌）是 MoE 训练中的一种退化现象：多数输入 token 被路由到少数几个 expert，导致这些 expert 负载过高而其他 expert 被闲置。这不仅是负载均衡问题，更导致 expert 的表征多样性丧失——闲置 expert 的梯度更新极少，逐渐失去有效的知识表示。Chi et al. (2022) 系统分析了这一现象：随着训练进行，gate 对部分 expert 的路由概率趋近于 0，这些 expert 接收的 token 越来越少，形成正反馈循环（越少被激活 → 梯度越少 → 能力越差 → 越不被选择）。在 homogeneous MoE 中，representation collapse 表现为少数"赢家 expert"占据大部分 token，其余 expert 退化；在 heterogeneous MoE 中，collapse 更严重——大 expert 天然能力更强，router 倾向于"赢家通吃"，小 expert 完全被边缘化（HMoE Section 3.2, Figure 3）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Representation collapse 的正反馈循环机制：

```
Training step t:
  1. Router: P = softmax(x @ W_r)
     # 此时 expert_a 和 expert_b 的概率已极化
  2. Token assignment: 80% tokens → expert_a, 15% → expert_b, 5% → rest
  3. Expert_a 接收大量 token → 梯度估计准确 → 能力提升
  4. Expert_c 几乎无 token → 梯度噪声大/无梯度 → 能力停滞
  5. Step t+1: Router 观察到 expert_a 能力更强 → 分配更多 token
     → 正反馈循环加速 → collapse 加剧
```

HMoE 通过两种机制缓解 representation collapse：
- **异构容量**：不同大小的 expert 天然具有不同的能力范围。小 expert 处理简单 token 时表现不逊于大 expert（因简单 token 不需要深层表示），因此 router 可以被 P-Penalty 引导将简单 token 分配给经济的小 expert，打破"赢家通吃"逻辑。
- **P-Penalty Loss**：显式惩罚激活大 expert（M_i 包含 h_ffn,i），使得"激活大 expert"比"激活小 expert"有更高的 loss 代价，打破 collapse 的正反馈。

实验验证（Figure 13）：homogeneous MoE 中 expert 的 Wasserstein distance 显示 expert 形成 2 个聚类（a/b/c 高度相似），而 HMoE 中 expert 按大小形成差异化分组（相似大小 expert 聚为一类），表明异构设计有效促进了 expert 分化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Chi et al. (2022) 提出 X-MoE routing 缓解 collapse：对 hidden state 做低秩投影后与 expert embedding 做余弦相似度，避免 softmax 的 winner-take-all 效应。HMoE 从另一个角度——通过异构容量 + P-Penalty 创建"积极差异化"而非"消极缓解"——让 collapse 的正反馈在异构建模空间中自然收敛到各 expert 按 token 复杂度分化的均衡状态。HMoE 实验显示 representation collapse 被有效缓解：训练后期各 expert 的激活频率保持稳定的差异化分布（而非 collapse 到 1-2 个 expert）。

RMoE (Qiu et al., 2025) 从 router 设计的角度分析了 representation collapse 的另一个侧面：(1) 单线性层 router 的局限性——token hidden states 通过 softmax 计算 gating score 时，embedding 容易 collapse 到 expert embedding 附近，导致 softmax 输出近乎 one-hot（gate entropy 极低），Top-k 退化为 Top-1；(2) RMoE 通过在 router 中引入 GRU + 逐层投影 Proj_i 间接缓解了 collapse——Proj_i 将 hidden state 与 expert embedding 分离（类似 XMoE 的低维投影策略），GRU 提供跨层路由信息使 router gate score 分布适度平坦（高熵但非随机）。实验验证：RMoE 的 gate entropy 分布比 SMoE 更均匀但不像 RandomMoE 那样完全扁平，在 exploration vs exploitation 之间取得更好平衡。

涉及论文标题：
- HMoE: Heterogeneous Mixture of Experts for Language Modeling
- Layerwise Recurrent Router for Mixture-of-Experts

## Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hierarchical Token Deduplication AlltoAll (HierD-AlltoAll) 是 HierMoE 提出的分层 token 去重 AlltoAll 算法，用于减少 MoE 训练中 Expert Parallelism 下的通信冗余。核心原理：在 GPU 集群的分层拓扑结构（如 4 层：Inter-Node/IB → Inter-QPI → Inter-NVLink → Intra-NVLink）中，不同层级的 AlltoAll 操作将 experts 划分为不同大小的 group（如 Inter-Node 按 4 nodes 分为 4 groups，Intra-GPU 按 32 GPUs 分为 32 groups）。当 top-K 中多个 expert 位于同一 group 时（如 K=8, R=4 → 55% 重复率），同一 token 在 AlltoAll 中被冗余传输。HierD-AlltoAll 在每层 AlltoAll 前执行 token 去重：将 routing mask I_route ∈ R^{T×E} 按 expert group 聚合为 I_route ∈ R^{T×U[i]}，同一 group 内多 expert 选中 → 仅传输一份 token 副本。通过线性性能模型 t_d = Σ(Inter-level time) + Intra-level time（公式 3），自动选择使总通信时间最小的维度 d*。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

HierD-AlltoAll 在 MoE layer 的 dispatch 阶段执行：

```
Input: I_route ∈ R^{T×E} (boolean routing mask), M (embedding dim),
       G (GPUs), E (experts), D (topology levels), U[D] (group sizes per level)
Output: Optimal d*, executing token dispatch with deduplication

Step 1 -- Compute HD1 (standard AlltoAll) time:
  m ← E/G                              // experts per GPU
  for j=1..G:
    I_route_g[i,j] ← OR_{j1=(j-1)m+1}^{j·m} I_route[i,j1]
  p[j] ← Σ_i I_route_g[i,j]            // dedup token count per GPU group
  n_a2a ← G · max(p) · M · v           // total comm bytes (v=2 for FP16)
  t1 ← α + n_a2a · β

Step 2 -- Compute HDd for d=2..D:
  for k=1..d-1:                         // Inter-level layers
    m ← E/U[k]
    I_route_k[i,j] ← OR_{j1=(j-1)m+1}^{j·m} I_route[i,j1]
    p_k[j] ← Σ_i I_route_k[i,j]        // dedup tokens in U[k] expert groups
    n_inter_k ← (U[k]/U[k-1]) · max(p_k) · M · v
    Update I_route to reflect post-Inter-level-k token distribution
  n_intra ← (G/U[d-1]) · max(p_d) · M · v
  td ← Σ_{i=1}^{d-1} (n_inter_i · β_inter(i) + α_inter(i))
       + n_intra · β_intra(d-1) + α_intra(d-1)

Step 3 -- Select optimal dimension:
  d* ← argmin_{1≤d≤D} td
  Complexity: O(D·T·K) where T=tokens, K=top-K
```

关键权衡：高层（小 group 数 → 大去重收益，如 R=4 时 K=8 重复率 55%）+ 低带宽链路（IB 200Gb/s）→ 去重后大幅减少 Inter-node 通信量。低层（大 group 数 → 小去重收益）+ 高带宽链路（NVLink 112.5GB/s）→ 去重收益有限但带宽充裕。d* 自动权衡这两者。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现于 Megatron-LM 的 MoE token dispatcher 中，替换标准 NCCL AlltoAll
- 集群启动时用 nccl-tests 一次性测量 7 种 AlltoAll 变体的 α, β 参数（r² > 0.997, <300s 测量 + <10ms 拟合）
- 每 iteration 在 CPU 控制逻辑上计算 d* (O(D·T·K)，微秒级)
- HierMoE 在 32-GPU 集群上实现 vs Megatron-LM AlltoAll 1.99×-2.72× 加速

涉及论文标题：
- HierMoE: Accelerating MoE Training with Hierarchical Token Deduplication and Expert Swap

## Expert-Specific Learning Rate Scaling in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Specific Learning Rate Scaling 是 Hunyuan-Large 提出的 MoE 训练策略：为 shared expert 和 specialized experts 分配不同的学习率（而非统一学习率），以解决不同 expert 处理的 token 数量不平衡导致的 effective batch size 差异问题。核心思想：MoE 中 shared expert 处理所有 token，而每个 specialized expert 仅处理约 1/n 的 token（n=specialized experts 数量），因此它们的 effective batch size 不同，需要不同的最优学习率。

基于 AdamW 的最优学习率公式（Li et al., 2024a）：

$$\epsilon_{opt}(B) = \frac{2\epsilon_{max}}{\sqrt{\frac{B_{noise}}{B} + \sqrt{\frac{B}{B_{noise}}}}}$$

其中 ε_max 是 AdamW 的最大学习率，B_noise 是训练速度与数据效率的 trade-off 点。shared expert 使用 ε_opt(B)，specialized expert 使用 ε_opt(B/n)。在 Hunyuan-Large 中（n=16, B=实际 batch size），比例 ε_opt(B)/ε_opt(B/n) ≈ 0.31。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Hunyuan-Large 的 Expert-Specific LR 设置
# 全局参数: 最大学习率 ε_max, 噪声 batch size B_noise, 实际 batch size B
# specialized experts 数量 n=16

# Step 1: 计算 shared expert 最优学习率
B_shared = B                                          # shared expert 处理所有 token
shared_lr = 2*ε_max / (sqrt(B_noise/B) + sqrt(B/B_noise))

# Step 2: 计算 specialized expert 最优学习率
B_specialized = B / 16                                # 每个 specialized expert 仅 1/16 token
specialized_lr = 2*ε_max / (sqrt(B_noise/B_specialized) + sqrt(B_specialized/B_noise))

# Step 3: 分配学习率
# optimizer param groups:
#   - shared_expert.params: lr = shared_lr
#   - specialized_experts.params: lr = specialized_lr = shared_lr * 0.31 (approximately)
#   - other params (attention, embedding, etc.): lr = shared_lr
```

Hunyuan-Large 使用 AdamW optimizer。SFT 阶段的学习率从 2e-5 衰减到 2e-6（3 epochs）。Annealing 阶段在最后 5% tokens 将学习率降至 peak 的 1/10。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 PyTorch 中通过 optimizer 的 `param_groups` 实现：为 shared expert 参数和 specialized expert 参数创建不同的 param group，设置不同的 `lr` 值。关键在于计算合理的比例因子——该因子取决于 `B_noise`（需通过小规模实验估计）和 specialized experts 数量 n。此方法适用于所有使用 shared + specialized experts 架构的 MoE 模型（如 DeepSeek-V2/V3 也可受益）。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## MoE Scaling Laws

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Scaling Laws 是 Hunyuan-Large 提出的针对 MoE 模型的计算预算-性能关系。不同于传统 dense 模型的 C = 6ND（Kaplan et al., 2020）或 Chinchilla 的 C ≈ 6ND（Hoffmann et al., 2022），MoE 模型由于稀疏激活和长序列 attention 复杂度，计算预算公式需要修正：

$$C \approx 9.59ND + 2.3 \times 10^8 D$$

其中 N 为激活参数量（非总参数量），D 为训练 token 数。常数项 2.3×10^8 D 来自 attention 计算的开销。

考虑 batch size 影响，通过临界 batch size B_crit(L) 修正为最小计算预算：

$$C_{min} = \frac{C}{1 + \frac{B}{B_{crit}(L)}}$$

通过拟合 isoFLOPs 曲线，得到：
- N_opt = 5.9×10^{-3} × C_min^{0.5305}（最优激活参数量-计算预算关系）
- D_opt = 3.2 × C_min^{0.50}（最优训练数据量-计算预算关系）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hunyuan-Large 的 MoE Scaling Law 探索流程：

```
# 1. 训练系列 MoE 模型（10M-1B 激活参数）
for N_active in [10M, 50M, 100M, 300M, 1B]:
    for D in [10B, 30B, 50B, 100B]:
        C = 9.59 * N_active * D + 2.3e8 * D     # 计算预算
        C_min = C / (1 + B/B_crit(L))            # 修正 batch size
        train_model(N_active, D)                  # 记录 loss L(N, D)

# 2. 拟合 isoFLOPs 曲线
# N_opt = N_c * C_min^α  →  N_c = 5.9e-3, α = 0.5305
# D_opt = D_c * C_min^β  →  D_c = 3.2, β = 0.50

# 3. 确定最优配置
# N_opt ≈ 58.1B activated → 选择 52B (smooth curve trade-off)
# D_opt ≈ 5.6T tokens → 选择 7T (maximize within optimal range)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

MoE Scaling Laws 的实现需要：(1) 训练多组不同规模的 MoE 模型（小模型代理），(2) 记录每个配置的 loss 和 FLOPs，(3) 拟合参数化公式 N_opt = N_c × C_min^α 和 D_opt = D_c × C_min^β，(4) 根据计算预算选择最优的激活参数量和训练数据量。Hunyuan-Large 受 Llama 3 启发，利用抛物线在最优值附近的平滑特性，从理论最优值 58.1B 调整到 52B（工程可行范围）。此方法适用于任何 MoE 模型的预训练规模规划。

---

**Joint MoE Scaling Laws (Ludziejewski et al., 2025)** 提出了统一的 Dense+MoE scaling law，将 expert 数 E 纳入 Chinchilla 形式：

$$\mathcal{L}(N_{\text{act}}, D, \hat{E}) = a\hat{E}^{\delta}N_{\text{act}}^{\alpha + \gamma \ln(\hat{E})} + b\hat{E}^{\omega}D^{\beta + \zeta \ln(\hat{E})} + c$$

其中 Ê 是 E 的单调变换（Eq.4: 1/Ê = 1/(E-1+(1/E_start-1/E_max)^(-1)) + 1/E_max），α,β,γ,ζ,δ,ω 为拟合系数。核心洞察是：exponent μ(E)=α+γ·ln(Ê) 和 ν(E)=β+ζ·ln(Ê) 中的对数项捕捉了 E 与 N_act 和 D 的交叉效应——γ>0 意味着更多 expert 会使 N_act 的 exponent 更负（更大模型时 MoE 收益递减），ζ<0 意味着更多 expert 会使 D 的 exponent 更负（需要更多 data）。

该 scaling law 的关键推导和发现：
1. **Compute optimality with E**：给定 budget F=6·N_act·D，compute-optimal 配置为 N_act_opt = G·(F/6)^(ν/(μ+ν))，D_opt = G^(-1)·(F/6)^(μ/(μ+ν))。E 增加 → 应减少 N_act、增加 D（Table 1: E=1→16 时 N_act 降 52%, D 增 113%）。
2. **Memory optimality**：引入 total params 约束 N_total ≤ M（含 KV-cache），在 3D 空间 {N_act, D, E} 求 argmin L。发现 E≤8 的 MoE 用 E× tokens 训练可超越 compute-optimal dense——Rule of Thumb。
3. **Inference optimality**：将 inference cost 2·N_act·D_inf 纳入 joint budget，揭示 MoE 的 inference 优势（每 token FLOPs = dense 的 39-64%）。
4. **LR scaling for MoE**：LR = exp(8.39 - 0.81·ln(N_act\e) - 0.25·ln(E))，更多 expert → 更低的 optimal LR（E 系数为负）。

拟合方法：LBFGS 优化 Huber loss (δ=0.01) on log-space，280+ 模型 runs（N_act 最高 2.7B, N_total 最高 5B, E∈{1,2,4,8,16,32}），RMSE_v=0.0039。该 scaling law 的特点是将 dense (E=1) 和 MoE 统一在同一框架下，使得跨 E 的公平比较成为可能。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## GQA (Grouped-Query Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Grouped-Query Attention (GQA) 是 Ainslie et al. (2023, EMNLP) 提出的 attention 变体，介于 Multi-Head Attention (MHA) 和 Multi-Query Attention (MQA) 之间。GQA 将 query heads 分为多个组，每个组共享一组 key-value heads：

- MHA：H 个 query heads, H 个 KV heads（KV cache = 4 × H × d_h × l）
- GQA：H 个 query heads, G 个 KV heads, G < H（KV cache = 4 × G × d_h × l）
- MQA：H 个 query heads, 1 个 KV head（KV cache = 4 × 1 × d_h × l）

GQA 在保持接近 MHA 表达能力的同时大幅减少 KV cache。在 Hunyuan-Large 中，设置 80 个 query heads、8 个 KV groups（G=8），相比 MHA 的 80 个 KV heads 将 KV cache 减少 10 倍。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# GQA forward: H query heads, G KV groups
# 输入: x [B, L, d_model]
# KV heads 数: G (每组有 H/G 个 query heads 共享 KV)

# 投影
Q = x @ W_q  # [B, L, H*d_k]  — 标准 MHA query
K = x @ W_k  # [B, L, G*d_k]  — 仅 G 组 KV
V = x @ W_v  # [B, L, G*d_k]

# Reshape
Q = reshape(Q, [B, L, H, d_k])
K = reshape(K, [B, L, G, d_k])
V = reshape(V, [B, L, G, d_k])

# 计算 attention: query head h 使用 KV group h // (H/G)
for h in range(H):
    g = h // (H // G)                        # 确定 KV group
    score = Q[:,:,h,:] @ K[:,:,g,:].T / sqrt(d_k)
    attn = softmax(score + mask)
    out[:,:,h,:] = attn @ V[:,:,g,:]
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GQA 已被广泛采用——LLaMA 2/3、Mistral、Gemma、Hunyuan-Large 等主流 LLM 均使用 GQA。实现时可以复用 MHA 的代码框架，仅需调整 KV head 数量和 repeat KV 的维度。在 HuggingFace Transformers 中，`num_key_value_heads` 参数即指定 GQA 的 KV group 数。GQA 也可与 FlashAttention 结合使用，通过 `flash_attn_func` 的 GQA 模式支持。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## CLA (Cross-Layer Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Layer Attention (CLA) 是 Brandon et al. (2024) 提出的 KV cache 压缩技术：相邻的 Transformer layers 共享同一组 KV cache，而非每层独立维护。CLA 从 "layer 维度" 压缩 KV cache，将 KV cache 大小从 O(l) 降至 O(l/s)，其中 s 为共享步长（通常 s=2，即每 2 层共享）。与 GQA 从 "head 维度" 压缩互补。

KV cache 内存比较（bf16 bytes）：
- MHA: 4 × H × d_h × l
- CLA (s=2): 2 × H × d_h × l（减半）
- GQA+CLA (G=8, s=2): 2 × G × d_h × l（Hunyuan-Large 最终方案, ~5% MHA）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# CLA with share interval=2 (每2层共享KV)
# 64 layers → 32 组共享 KV pairs

for block_idx in range(32):  # 32 个 CLA pairs
    # Layer 2*block_idx 和 Layer 2*block_idx+1 共享 KV
    for layer_offset in [0, 1]:
        layer_id = 2 * block_idx + layer_offset
        if layer_offset == 0:  # first layer in pair: 计算并缓存 KV
            K, V = self_attn_proj(hidden[layer_id])
            kv_cache[layer_id] = (K, V)      # 写入共享 KV cache slot
        else:  # second layer: 复用 KV
            K, V = kv_cache[layer_id - 1]    # 读取前一层（同一pair)的 KV
        
        # attention 计算
        Q = self.query_proj(hidden[layer_id])
        output[layer_id] = attention(Q, K, V)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

CLA 实现相对简单：(1) 在 attention 模块中，根据 layer_id 判断该层是否需要自己计算 KV（如 layer_id % share_interval == 0）或复用前一层的 KV。需要计算 KV 的层才分配 KV cache 空间。(2) 推理时只需维护共享的 KV cache entries。Hunyuan-Large 在 64 layers 中每 2 层共享，仅需 32 组 KV cache。CLA 带来的细微性能损失在实践中可忽略，而内存节省显著（~50% KV cache 减少，与 GQA 叠加后 ~95%）。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## KV Cache Compression (GQA+CLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

KV Cache Compression 是 Hunyuan-Large 联合使用 GQA 和 CLA 两种技术，从两个维度压缩 KV cache 内存占用的策略：

1. **Head 维度（GQA）**：将 KV heads 数从 80 (MHA) 压缩到 8 (GQA)，10× 压缩
2. **Layer 维度（CLA）**：每 2 层共享 KV cache，2× 压缩
3. **联合效果**：KV cache 从 4×H×d_h×l 降至 2×G×d_h×l，仅为 MHA 的 `G/(2H) = 8/160 = 5%`

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hunyuan-Large 每层 MoE block 的 KV cache 使用模式：

```
# 64 layers, GQA (G=8), CLA (share every 2 layers)
# KV cache slots: 32 (每2层1个)

for layer_id in range(64):
    hidden = RMSNorm(input)
    
    # CLA: 判断是否需要计算 KV
    if layer_id % 2 == 0:
        K, V = proj_kv(hidden)  # GQA: 仅8组KV
        cache_slot = layer_id // 2
        kv_cache[cache_slot] = (K, V)  # 存储KV
    else:
        K, V = kv_cache[layer_id // 2]  # 复用前一层KV
    
    # Attention with GQA (8 KV groups × 80 query heads = 10:1 ratio)
    Q = proj_q(hidden)        # [B, L, 80×d_k]
    attn_out = gqa_attention(Q, K, V, num_kv_groups=8)
    
    # MoE FFN
    moe_out = shared_expert(hidden) + top1_specialized_expert(hidden, router)
    
    input = input + attn_out + moe_out
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

GQA+CLA 联合压缩无需特殊硬件支持——可在标准 PyTorch 中实现。GQA 通过修改 attention 的 KV 投影矩阵维度实现（`nn.Linear(d_model, G*d_head)` 替代 `nn.Linear(d_model, H*d_head)`）；CLA 通过在各层间共享 KV cache buffer 实现。两者组合的工程实现要点：(1) 验证 GQA 的 KV head 数量选择（太小编码能力下降），(2) 验证 CLA 的共享步长（太大影响层间表示多样性），(3) 与 FlashAttention 兼容。Hunyuan-Large 选择了保守的参数（G=8, s=2），在 ~95% KV cache 节省下无显著性能损失。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## RoPE (Rotary Position Embedding)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Rotary Position Embedding (RoPE) 是 Su et al. (2024, Neurocomputing) 提出的位置编码方法，通过旋转矩阵将位置信息编码到 attention 的 query 和 key 向量中。核心思想：对 Q 和 K 的每对维度施加基于绝对位置 m, n 的旋转，使 attention score Q_m^T K_n 仅依赖于相对位置 (m-n)：

$$f_Q(x_m, m) = R_{\Theta,m} W_Q x_m$$
$$f_K(x_n, n) = R_{\Theta,n} W_K x_n$$
$$f_Q(x_m, m)^T f_K(x_n, n) = x_m^T W_Q^T R_{\Theta,m-n} W_K x_n$$

旋转矩阵 R_{\Theta,m} 是分块对角矩阵，每块施加二维旋转：`[cos(mθ_i), -sin(mθ_i); sin(mθ_i), cos(mθ_i)]`，其中 θ_i = base^{-2i/d}, i=0..d/2-1。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# RoPE: 对 Q 和 K 施加位置依赖旋转
# 输入: q, k [B, L, H, d], 位置索引 position_ids [L]

def rope_forward(q, k, position_ids, base=10000):
    d = q.shape[-1]
    # 计算频率: θ_i = base^{-2i/d}
    theta = 1.0 / (base ** (torch.arange(0, d, 2).float() / d))
    
    # 计算 cos, sin 表
    pos = position_ids.unsqueeze(-1)  # [L, 1]
    freqs = pos * theta               # [L, d/2]
    cos = freqs.cos().repeat(1, 2)    # [L, d]
    sin = freqs.sin().repeat(1, 2)
    
    # 旋转: 每对维度 (2i, 2i+1)
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot

def rotate_half(x):
    # x = [..., x0, x1, x2, x3, ...]
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x_rot = torch.stack([-x2, x1], dim=-1).flatten(-2)
    # x_rot = [..., -x1, x0, -x3, x2, ...]
    return x_rot
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

RoPE 已成为现代 LLM 的标准位置编码：LLaMA、Qwen、DeepSeek、Hunyuan-Large 等均使用。Hunyuan-Large 在长上下文预训练阶段（256K）将 RoPE base frequency 从标准 10000 扩展到 1 billion (10^9)（参考 Xiong et al., 2023），以支持更长的上下文长度。实现上，HuggingFace Transformers 的 `LlamaRotaryEmbedding` 类可作为参考。RoPE 的效率优化包括：(1) 使用预计算的 cos/sin 表，(2) 与 FlashAttention 结合时在 kernel 内部完成旋转。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## DPO (Direct Preference Optimization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Direct Preference Optimization (DPO) 是 Rafailov et al. (2024, NeurIPS) 提出的离线偏好对齐算法，被 Hunyuan-Large 用于 RLHF 阶段。DPO 直接从偏好数据中优化策略模型，无需显式训练 reward model，也无需在线采样：

$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{ref}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma\left(\beta \log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

其中 y_w 是 chosen response，y_l 是 rejected response，π_ref 是参考模型（SFT 模型），β 控制偏离参考模型的程度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Hunyuan-Large 的 DPO 采用单阶段训练策略，结合离线与在线数据：

```
# Hunyuan-Large 的 DPO 训练流程
# 输入: SFT 模型 π_SFT, 偏好数据集 D_pref

for batch in training_data:
    # 1. 离线数据: 预编译的偏好对
    x, y_w, y_l = batch["offline"]
    loss_offline = DPO_loss(π_θ, π_SFT, x, y_w, y_l, β)
    
    # 2. 在线数据: 当前策略模型生成多个 response
    y_candidates = [π_θ.generate(x_i) for x_i in batch["online_prompts"]]
    y_w_online, y_l_online = reward_model.select_best_worst(y_candidates)
    loss_online = DPO_loss(π_θ, π_SFT, x, y_w_online, y_l_online, β)
    
    # 3. 添加 SFT loss 项 (stabilization)
    loss_sft = -log π_θ(y_w | x)  # 防止 chosen prob 下降
    
    # 4. Total loss
    loss = loss_offline + loss_online + λ * loss_sft
    
    # 5. EMA (exponential moving average) 防 reward hacking
    π_ema = EMA(π_θ, decay=0.999)
```

Hunyuan-Large 还使用 EMA（指数移动平均）策略减少 reward hacking 和 alignment tax。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

DPO 实现：(1) 需要 reference model（通常为 SFT 模型，frozen）；(2) 每步计算两个 forward pass（π_θ 在 chosen 和 rejected 上的 log prob）+ 一个 reference forward；(3) β 典型值 0.1-0.5。Hunyuan-Large 结合离线（预编译 preference data）和在线（当前策略生成+reward model 评分）数据。常见实现库：TRL (`DPOTrainer`)、HuggingFace TRL。DPO 相比 RLHF (PPO) 的优势：不需要 reward model 训练、不需要在线采样（纯离线）、训练更稳定。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## Synthetic Data Pipeline for LLM Pre-training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Synthetic Data Pipeline 是 Hunyuan-Large 提出的四步合成数据生成流程，用于补充自然语料中欠缺的高质量训练数据（数学、代码、低资源语言、高教育价值领域）。流程包含四个步骤：

1. **Instruction Generation**：使用高质量种子数据源（网页、QA数据、代码仓库、书籍等）+ 多样化 instruction 生成 prompt → 生成覆盖多领域、多风格、多复杂度的 instructions
2. **Instruction Evolution**：通过三方面改进初始 instructions：(a) 增强清晰度和信息量，(b) 自指导增强低资源领域，(c) 提升难度层级
3. **Response Generation**：使用多个不同大小的专门模型为 evolved instructions 生成专家级 answers
4. **Response Filtering**：使用 critique model + self-consistency 检查（多答案一致性过滤）去除低质量或不一致的数据

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Hunyuan-Large 四步合成数据 pipeline
# 总产出：约 1.5T tokens 高质量合成数据

# Step 1: Instruction Generation
seeds = [web_pages, qa_data, code_repos, books, ...]
instructions = []
for seed in seeds:
    prompt = instruction_gen_template.format(seed=seed)
    instructions.append(llm.generate(prompt))  # 覆盖多领域+多风格

# Step 2: Instruction Evolution
evolved = []
for inst in instructions:
    inst = enhance_clarity(inst)               # (a) 清晰度提升
    if is_low_resource(inst.domain):
        inst = self_instruct_augment(inst)     # (b) 自指导增强
    inst = increase_difficulty(inst)           # (c) 难度提升
    evolved.append(inst)

# Step 3: Response Generation
synthetic_pairs = []
for inst in evolved:
    expert_model = select_expert_model(inst.domain, inst.complexity)
    response = expert_model.generate(inst)
    synthetic_pairs.append((inst, response))

# Step 4: Response Filtering
filtered_pairs = []
for inst, resp in synthetic_pairs:
    score = critique_model.score(inst, resp)    # 多维质量评分
    if score < threshold:
        continue
    if is_objective_qa(inst):
        responses = [model.generate(inst) for model in ensemble]
        if not self_consistency_check(responses):
            continue
    filtered_pairs.append((inst, resp))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：(1) 需要多个专门模型作为 generator 和 critique model（Hunyuan-Large 使用 70B dense critic）；(2) instruction 多样性依赖种子数据的覆盖面和 prompt 设计的多样性；(3) self-consistency 筛选对客观 QA 任务有效（生成多个答案投票），主观任务需要人审。Hunyuan-Large 的 1.5T 合成数据覆盖数学、代码、低资源语言和高教育价值领域。实际生成时需配合分类标签系统灵活调整各类数据比例。合成数据 pipeline 已在 LLaMA 3/3.1、Phi-3 等主流工作中广泛验证，是提升模型能力的关键手段。

涉及论文标题：
- Hunyuan-Large: An Open-Source MoE Model with 52 Billion Activated Parameters by Tencent

## Expert-Level Sparsification (Expert Dropping)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Level Sparsification（专家级稀疏化，亦称 Expert Dropping/Expert Pruning）是专门针对 SMoE 架构的模型压缩技术，通过识别和移除整个 expert 子网络（包括其权重和 router entry）来减少模型总参数量和内存占用。核心理念：SMoE 中存在显著的 expert redundancy——部分 expert 对模型性能至关重要（dominant experts），而另一些则高度冗余。Jaiswal et al. (2025) 的实验显示 Mixtral-8×7B 中某些 expert 被单独丢弃后 perplexity 急剧上升，另一些几乎无影响。与传统 weight pruning 不同，Expert Dropping 丢弃的是整个结构化计算单元（FFN expert），不改变剩余 expert 的内部参数结构，因此可直接获得实际推理加速（50% sparsity → 1.27× speedup, ≤0.55× memory usage）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert Dropping Pipeline
def expert_drop_per_layer(M, l, s, criterion, X_calib):
    """
    M: SMoE model, l: layer index, s: droprate
    criterion: MC-Suite criterion function
    """
    scores = [criterion(M, l, e, X_calib) for e in range(n)]
    n_drop = int(n * s)
    drop_set = argsort(scores)[:n_drop]  # lowest importance
    
    # Remove from router: W_G^{d×n} → W_G^{d×(n-n_drop)}
    keep_mask = ones(n, dtype=bool)
    keep_mask[drop_set] = False
    M.layers[l].router.W_G = M.layers[l].router.W_G[:, keep_mask]
    for e in drop_set:
        del M.layers[l].experts[e]
```

**与 Weight Pruning 的对比** (Mixtral-8×7B @ 50% sparsity, zero-shot avg):
- Random Weight Pruning (2:4): 27.27 (Base) / 31.94 (Instruct)
- Wanda Weight Pruning (2:4): 52.91 (Base) / 62.28 (Instruct)
- Min-EAN Expert Pruning (r=4): 56.62 (Base) / 63.95 (Instruct)
- Full Mixtral (r=8): 70.32 (Base) / 76.31 (Instruct)

Expert-level sparsification 整体优于 weight pruning，尤其在 ARC-c 上提升 ~16.2%。Base model 上的表现优于 Instruct model（建议在 instruction tuning 前做 expert dropping）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Per-layer uniform dropping：每层丢弃相同比例 expert 以避免 bottleneck layers
- 必须同步修改 router gating 函数（删除对应 expert 入口），否则 router 可能将 token 路由到已删除 expert
- 丢弃后 router 矩阵直接修改导致负载分布偏斜，需要 finetuning 或 load rebalancing 校正
- 相关方法：REAP (Lasby et al., 2025) 用 router gate-value × activation norm；DERN (Zhou et al., 2025) 重组合 pruned expert 的神经元到 retained expert

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

## MC-Suite (MoE Experts Compression Suite)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MC-Suite（MoE Experts Compression Suite）是 Jaiswal et al. (2025) 提出的首个全面的 MoE expert 重要性估计 benchmark，从四个维度（Weight/Inference/Activation/Gradient）系统化设计了 16 种 task-agnostic 准则来识别"最可丢弃"的 expert。实验验证最优准则为 **Min-EAN**（最小激活范数）和 **Min-EGE**（最小梯度熵），因为它们同时考虑了 input tokens 和 weight parameters，比仅基于 expert usage frequency 的传统准则（EUF, ECC 等）更精确。

四个维度：
- **Weight-Guided（4种）**：仅需 expert weights 本身，无需 calibration data。Expert Weight Similarity (EWS)、Router Weight Norm (RWN)、Expert Weight Stable Rank (WSR)、Expert Weight Norm (EWN)。最佳为 Max-RWN（Table 1 中 50% sparsity pp=10.70）
- **Inference-Guided（4种）**：依赖 calibration data forward pass 统计 routing 行为。Expert Usage Frequency (EUF)、Expert-Expert Collaboration (ECC)、Expert Vocabulary Coverage (EVTC)、Expert Input Token Similarity (ETS)
- **Activation-Guided（4种）**：calibration data forward pass + hooks 收集 expert 输出。Expert Activation Similarity (EAS)、Expert Activation Entropy (EAE)、Expert Activation Outliers (EAO)、Expert Activation Norm (EAN)。**Min-EAN 为最优准则**（50% sparsity pp=9.99 vs full 7.82）
- **Gradient-Guided（4种）**：forward + backward pass 收集梯度。Expert Gradient Similarity (EGS)、Expert Gradient Entropy (EGE)、Expert Gradient Outliers (EGO)、Expert Gradient Norm (EGN)。**Min-EGE 为次优准则**（50% sparsity pp=10.45）

从算法pipeline角度拆解：

```
# Min-EAN (Minimum Expert Activation Norm) — 最优准则
def min_ean_score(M, l, e, X_calib):
    A_e = []
    for batch in X_calib:
        h = M.embed(batch)  # hidden states
        topk = topk(softmax(h @ M.layers[l].W_G), k=2)
        mask = (topk == e).any(dim=-1)  # tokens routed to expert e
        if mask.any():
            x_e = h[mask]  # (t_e, d)
            # SwiGLU FFN: SiLU(gate) * up → down
            a = silu(x_e @ W_gate) * (x_e @ W_up) @ W_down
            A_e.append(a)
    A_all = concat(A_e, dim=0)  # (total_tokens_e, d)
    return sum(norm_l2(A_all, dim=0))  # ||A_e||_2, lower→more droppable

# Min-EGE (Minimum Expert Gradient Entropy) — 次优准则
def min_ege_score(M, l, e, X_calib):
    for batch in X_calib:
        loss = cross_entropy(M(batch), batch_labels)
        loss.backward()  # accumulate gradients
    grad_W = M.layers[l].experts[e].weight.grad
    # H ∝ Σ_j log[σ(W_grad^j)]
    stds = [std(grad_W[j,:]) for j in range(grad_W.shape[0])]
    return sum(log(s) for s in stds if s > 0)
```

**实验发现**：Activation entropy 和 gradient entropy 强正相关；dominant expert 具有较高 entropy（信息量大，适合 downstream adaptation）；layers 1-2 中有 2 个 expert 的 gradient entropy 极高，丢弃它们导致 abrupt 性能崩溃；activation entropy 跨层逐渐增长（initial→terminal layers）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Calibration data：256 C4 validation samples, max_seq_len=2048 即可获得稳定估计
- Activation criteria 需 forward hooks（开销低），gradient criteria 需 backward pass（开销较高）
- 推荐 Min-EAN：最优性能 + 最低开销（仅 forward pass）
- 准则选择不敏感于 calibration dataset 选择（cross-dataset robust）

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

## MoE Lottery Subnetworks (Iterative Estimate-Prune-Finetune)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Lottery Subnetworks 是 Jaiswal et al. (2025) 提出的 expert-level sparsification 方法，受 Lottery Ticket Hypothesis (Frankle & Carbin, 2018) 启发，将传统 one-shot expert pruning 改进为迭代 **estimate-prune-finetune** 三阶段循环。核心创新：(1) 用 k 轮迭代替代 one-shot pruning，每轮重估 MC-Suite 准则；(2) 每轮 pruning 后插入 task-agnostic budget finetuning（next-token prediction on C4, 仅需 ~1M training tokens），校正 expert 丢弃导致的 sub-optimal 状态（负载偏斜 + 性能骤降）。

从算法pipeline角度拆解：

```
def moe_lottery(M, s, k, criterion, X_calib):
    droprate = s / k           # e.g., 50%/4=12.5%
    tokens = 0.2M              # finetuning budget round 1
    for r in range(k):
        # ESTIMATE: per-layer importance scoring
        drop_sets = {}
        for l in M.moe_layers:
            scores = [criterion(M, l, e, X) for e in remaining[l]]
            n_drop = int(n_experts * droprate)
            drop_sets[l] = argsort(scores)[:n_drop]
        
        # PRUNE: remove from router + delete weights
        for l in M.moe_layers:
            W_G.keep_mask[drop_sets[l]] = False
            for e in drop_sets[l]: del experts[e]
        
        # FINETUNE: task-agnostic next-token prediction
        opt = AdamW(M.parameters(), lr=1e-6)
        for batch in X_calib:
            if tokens_used >= tokens: break
            loss = cross_entropy(M(batch.inp), batch.lbl)
            loss.backward(); opt.step()
        tokens *= 2  # progressive: 0.2M→0.4M→0.8M→1.6M
    return M
```

**性能对比** (Mixtral-8×7B Base, C4 pp, full=7.44):

| % Dropped | One-shot Min-EAN | Iterative Min-EAN | MoE Lottery Min-EAN |
|-----------|-----------------|-------------------|---------------------|
| 12.5% | 7.95 | 7.90 | 7.89 |
| 50.0% | 14.74 | 10.44 | **9.76** |
| 75.0% | 30.59 | 17.39 | **13.05** |

关键发现：(a) MoE Lottery ≥ One-shot 3× 更优；(b) Finetuning 收益在 ~1M tokens 后饱和（Table 5）；(c) Finetuning 不显著改变 expert 选择（Figure 5b，MoE Lottery 与 Iterative 选择高度一致），但重调 router weights 实现负载 rebalance（Figure 6）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 配置：AdamW, cosine LR, max lr=1e-6, batch=8, 8×A100
- Progressive token schedule 最小化总计算；每轮重置 optimizer
- 建议在 Base model 上做 MoE Lottery pruning 再 instruction tuning（Instruct 上 finetuning 收益更小）
- 不修改 Serving 框架，pruned 模型可直接推理

涉及论文标题：
- Finding Fantastic Experts in MoEs: A Unified Study for Expert Dropping Strategies and Observations

## Expert-Combine Operation

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert-Combine Operation（专家组合操作）是 MoE layer 中将多个 activated expert 的输出融合为单个 token 输出的操作。在 Top-K routing 机制下，每个 token 被路由到 K 个 expert，各 expert 产生独立输出 h_i^k，Combine 操作将这些输出按 gate affinity score 加权求和（FlashMoE 式 2-3）：

$$C_i = \sum_{j=1}^k g_{i,e}, \quad \mathbf{h}_{i} = \sum_{j=1}^{k} \frac{g_{i,e}}{C_{i}} \cdot \mathbf{h}_{i}^{k}$$

其中 g_{i,e} 为 gate affinity score，C_i 为归一化因子，h_i^k 为第 k 个选定 expert 的输出。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Expert-Combine 在 MoE pipeline 中:
# ... Dispatch → Expert FFN → Combine → ...

# Top-2 routing by FlashMoE Task 统一抽象:
# Combine task: t = (M, ⊙, identity)
# F_t(A, S, C, C) := C ← A ⊙ S + C
# A=expert output, S=gate weight, C=accumulator

for each token i:
    C_norm = sum(g_i[e] for e in selected_experts[i])
    output[i] = zeros(H)
    for k in range(top_k):
        e = selected_experts[i][k]
        output[i] += (g_i[e]/C_norm) * expert_outputs[e][i]
```

在分布式 MoE 中，不同 expert 可能位于不同 GPU，Combine 需要跨 GPU 通信——这是第二轮 AlltoAll 或 one-sided transfer 的触发点。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- GShard-style: 加权求和，所有 K 个 expert 输出参与
- Switch Transformer: Top-1, Combine 退化为 copy
- DeepSeek-V3: Top-K + shared expert, K routed + 1 shared
- FlashMoE: Combine 统一到 Task 抽象，在 Processor actor 内与 GEMM task 交错调度
- 分布式场景中，Combine 触发的跨 GPU 通信传统上用 AlltoAll，FlashMoE 用 NVSHMEM one-sided put（每 GPU individually 回传 GEMM1 结果）

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel

## SR-Based Expert Compression (Shared-Residual Expert Compression)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SR-Based Expert Compression（共享-残差专家压缩）是 HybridEP 提出的用于在 MoE 跨 DC 训练中大幅减少 expert 参数传输量的压缩算法。核心思想是将 expert 参数分解为 shared expert（共享专家，所有 expert 的平均值，学习 expert 间的共性/冗余知识）和 residual expert（残差专家，expert - shared_expert，捕捉每个 expert 特有的知识差异），仅传输压缩后的残差。关键动机来自两个观察：(1) Expert 权重分布比 activation data 更集中、outlier 更少（Figure 4），具有更高的可压缩性；(2) Expert 间的主要差异集中在少数参数上（Figure 9a），残差的分布比原始权重更集中和稀疏。压缩算法分两阶段——SREncode（编码：计算残差 → Top-k 保留绝对值最大的 k 个元素 → value-index 稀疏格式存储）和 SRDecode（解码：从稀疏格式 scatter 恢复残差 → 与 shared_expert 相加恢复完整 expert，其中恢复和加法被 fused 以减少 overhead）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# SR-Based Expert Compression Pipeline (HybridEP)
# 配置: E experts, 每个 expert 参数量 P_E, 压缩比 CR, k = P_E // CR

# === 算法初始化 ===
shared_expert = mean(expert_0, expert_1, ..., expert_{E-1})  # 各专家均值
# shared_expert 通过 backward All-Reduce 在每次 iteration 同步梯度

# === Phase 1: SREncode (与 optimizer.step() 融合执行) ===
def SREncode(expert, shared_expert, CR):
    # Step 1: 残差分解
    residual = expert - shared_expert    # shape: [P_E]
    # residual 分布更集中、更稀疏 (Figure 9a "res" suffix)
    
    # Step 2: Top-k 稀疏压缩
    k = P_E // CR                       # 例如 P_E=4.7M, CR=50 → k≈94k
    abs_residual = abs(residual)
    _, topk_indices = topk(abs_residual, k)  # 选绝对值最大的 k 个位置
    topk_values = residual[topk_indices]     # 保留对应的值(含符号)
    
    # Step 3: 稀疏格式存储
    compressed = (topk_values, topk_indices)  # 存储为 value-index pairs
    return compressed
    # 压缩后数据量: k * (sizeof(FP16) + sizeof(INT32))
    # 例如: 94k * (2B + 4B) ≈ 0.56 MB vs 原始 4.7MB → 8.4× (与带宽和 CR 相关)

# === Phase 2: SRDecode (与 expert FFN computation 融合执行) ===
def SRDecode(compressed, shared_expert):
    values, indices = compressed
    
    # Step 1+2: 恢复残差 + 加回共享专家 (fused)
    expert_recovered = shared_expert.clone()        # [P_E]
    expert_recovered.scatter_(indices, values)      # 将 values 写入 indices 位置
    # 等价于: expert = shared_expert + residual_recovered
    #   其中 residual_recovered[i] = values[j] if i == indices[j] else 0
    
    return expert_recovered
    # 融合 overhead: SRDecode + expert FFN 融合可减少 ~45% overhead (Figure 15b)

# === 训练 iteration 中的使用 ===
# 前一步: SREncode
for expert in local_experts:
    compressed = SREncode(expert, shared_expert, CR)
    send_queue.push(compressed)

# 当前步: AG 通信 → SRDecode → Expert FFN
for layer_experts in send_queue:
    all_compressed = AllGather(layer_experts, domain_group)  # 域内收集
    for c in all_compressed:
        expert = SRDecode(c, shared_expert)
        output += gate_weight * expert_ffn(expert, tokens)
```

关键词: 为什么用 shared + residual 而非直接压缩？Figure 14 对比显示：HybridEP w/o S（直接 Top-k 压缩）的 loss 显著高于 baseline，而 HybridEP w/ S（shared expert + residual Top-k）的 loss 与 baseline 几乎一致（50× CR），证明 shared expert 对维护精度至关重要——shared expert 捕获了 expert 间的共性知识，residual 仅编码微小的专家差异。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 实现依赖于 PyTorch 的 scatter 操作和 CUDA 优化的 Top-k kernel。编码通过 `torch.topk(abs(residual), k)` 实现，解码通过 `tensor.scatter_(dim, indices, values)` 融合完成。
- Shared expert 占用的额外 GPU memory 通过 offloading 到 CPU（ZeRO-Offload 兼容策略）管理——local experts 被 offload 到 CPU memory 而 shared experts 保留在 GPU memory。
- SREncode 与 optimizer step 融合的关键：在 Adam optimizer 更新参数后，立即对更新后的 expert 执行 SREncode，利用 GPU 已经在更新 expert 参数时的高计算利用率，减少额外 kernel launch 开销。实验显示融合可减少 ~30% 编码 overhead（Figure 15a）。
- SRDecode 与 expert FFN 融合的关键：SRDecode 的 scatter 操作可与 FFN 的第一个 GEMM（gate projection）通过 CUDA stream 或 kernel fusion 重叠，减少 ~45% overhead（Figure 15b）。
- 压缩比 (CR) 是一个超参数，论文在 50× 下验证无精度损失（Figure 14），但更高压缩比下的行为未充分探索（论文未展示 >50× CR 的结果，仅注明 page limit）。

涉及论文标题：
- HybridEP: Scaling Expert Parallelism to Cross-Datacenter Scenario via Hybrid ExpertData Transmission

## a_max (Maximum Activated Expert Count)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

a_max 是 JANUS 论文定义的核心 MoE 性能度量，表示一次 MoE layer forward pass 中，所有 MoE instance (GPU) 上 distinct activated expert 数量的最大值：

$$
a_{\max} = \max_{i \in \{1,\dots,n_e\}} a_i
$$

其中 a_i 是 MoE instance i 在本次 layer forward 中被分配到的 distinct expert 数量（注意：是 distinct expert count，而非 token count）。JANUS 的核心发现是：在在线 decode 场景下，MoE 层是 memory-bound 的，其延迟主要由 a_max 决定（即最慢的 instance 决定整层延迟），而非 total token count 或 routing probabilities。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

a_max 与 MoE layer latency 的关系 (JANUS Eq. 1c, Roofline 推导):

```
MoE layer latency model:
  T_moe = β · a_max + c_e

Roofline analysis:
  Expert arithmetic intensity: I_e ≈ 2b·d_h·d_e / 2d_e·d_h = b
  (b = per-expert batch size, d_h = hidden dim, d_e = expert intermediate dim)

  Compute-bound condition: I_e ≥ π/β
  For H100 (π=989 TFLOPs, β=3.35 TB/s):
    B ≥ π·n/(β·k) ≈ 989×256/(3.35×8) ≈ 9.4k tokens per layer
  For A100 (π=312 TFLOPs, β=2.0 TB/s):
    B ≥ 312×256/(2.0×8) ≈ 5k tokens

  Online decode: per-instance batch size typically < 100
  → MoE layers are MEMORY-BOUND in online serving
  → Latency ∝ number of expert weights to load = ∝ distinct activated experts

  Therefore: T_moe ∝ a_max (not ∝ total_token_count)
```

a_max 的 theoretical bound (JANUS Appendix A, balls-into-bins model):
```
Uniform activation: p_e = K/E (K=top-k, E=total experts)
Expected activated experts per instance (容量C的instance):
  E[a_g] ≤ C · [1 - (1 - K/E)^B]

Bottleneck instance: ā_max = max_g E[a_g]

Tail bound (Bernstein + union bound over n_e instances):
  a_max ≤ min(C, ā_max + sqrt(2·ā_max·ln n_e)) + 1

Three regimes (Fig. 17):
  Sparse (B ≲ 10): â_max ≤ 4, insensitive to placement
  High-Leverage (B ∈ [10,100]): steepest slope, 30-60% of C
    → Online decode operates here → placement + scheduling matter most
  Saturation (B ≥ 100): â_max plateaus near min(C, E/n_e)
    → Structural ceiling, no scheduling policy can push below it
```

a_max Monte Carlo estimation:
```
For each candidate (n_e, B):
  Sample B tokens from recent activation trace
  Apply AEBS scheduling strategy
  Record â_max = max_i(distinct experts on instance i)
  Build lookup table â_max^(ℓ)(n_e, B) for each MoE layer ℓ
Rebuild periodically (e.g., every 15 min)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- a_max 是分析指标，不在系统中直接显式计算
- Monte Carlo estimator 用于 JANUS 的 SLO-aware scaling 决策（Algorithm 2 TPOT evaluation）
- AEBS scheduling 的目标函数是 minimize a_max（greedy heuristic）
- a_max bound 公式用于快速 pruning 明显 infeasible 的 (n_e, B) 配置
- 可以推广到任何 expert-count-balanced 而非 token-count-balanced 的 MoE 系统

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference

## MoE Roofline Analysis (Memory-Bound vs Compute-Bound)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

MoE Roofline Analysis 是 JANUS 用来分析 MoE 层性能瓶颈的方法论。通过将单个 expert 的 arithmetic intensity（算术强度）I_e 与硬件 roofline 拐点 π/β 比较，判断 MoE 层在当前 workload 下是 memory-bound 还是 compute-bound。

单个 expert（含 2 个 GEMM）的 arithmetic intensity：
$$
I_e \approx \frac{2b \cdot d_h \cdot d_e}{2 \cdot d_e \cdot d_h} = b
$$

即 expert 的算术强度近似等于其 batch size b = B·k/n（B = layer-wise batch size, k = top-k, n = experts per GPU）。在在线 decode 场景下，b 通常远小于使 expert compute-bound 所需的阈值，因此 MoE 层是 memory-bound 的，延迟由需要从 HBM 加载的 expert weight 数量决定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Roofline Analysis for MoE Layer:

Given:
  π: peak FLOPs (e.g., H100: 989 TFLOPs/s, A100: 312 TFLOPs/s)
  β: memory bandwidth (e.g., H100: 3.35 TB/s, A100: 2.0 TB/s)
  Roofline ridge point: π/β

For single expert with batch size b:
  FLOPs = 2 · b · d_h · d_e  (2 GEMMs: gate+up projection, down projection)
  Bytes = 2 · d_h · d_e  (expert weight loading, assuming weights loaded once)
  Arithmetic Intensity I_e = FLOPs / Bytes ≈ b

For MoE layer with n experts per GPU, top-k routing:
  Per-expert batch size b = B · k / n  (expected, uniform routing)
  Minimum B for compute-bound: B ≥ π·n/(β·k)

Numerical Examples (JANUS):
  DeepSeek-V3 on H100 (n=256, k=8):
    B_min = 989 × 256 / (3.35 × 8) ≈ 9,400 tokens
  DeepSeek-V3 on A100 (n=256, k=8):
    B_min = 312 × 256 / (2.0 × 8) ≈ 5,000 tokens

  Online decode: per-instance B typically < 100
  → MoE layers are firmly MEMORY-BOUND in online serving

Implication for latency modeling (JANUS Eq. 1c):
  T_moe = β · a_max + c_e
  (linear in distinct activated expert count, not in token count)

Validation (JANUS Fig. 2 right, Fig. 3):
  Fix B=64, vary activated expert count → latency ~linear
  Fix B, vary activation distribution (uniform vs skewed) → nearly identical latency
  Vary B from 64 to 512 → latency changes marginally
```

JANUS TPOT Model (Eq. 1, integrating roofline):

```
TPOT = Σ_{ℓ=1}^{L} [T_attn^(ℓ) + T_moe^(ℓ) + T_comm^(ℓ)]

T_attn^(ℓ) = max(c_a^(ℓ), α^(ℓ)·b + c_kv^(ℓ)·b·S_ctx)
  // Attention follows roofline: memory-bound plateau (c_a) dominates at small b
  // then transitions to compute+KV-cache regime at large b

T_moe^(ℓ) = β^(ℓ) · a_max^(ℓ)(n_e, B) + c_e^(ℓ)
  // MoE is memory-bound in online serving → linear in a_max

T_comm^(ℓ) = profiled two-phase communication cost
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- HPC 标准方法：roofline model 由 Williams, Waterman, Patterson (2009) 提出
- JANUS 使用 roofline 指导：(1) TPOT 性能模型结构选择；(2) 解释为什么 AEBS minimize a_max 而非 token count；(3) 辨识 high-leverage batch size range [10, 100]
- 系数 (α, β, c_a, c_kv, c_e) 通过一次性 offline profiling 获得
- 可推广到任何需要在不同 batch size regime 下分析 MoE 层性能的方法

涉及论文标题：
- JANUS: Disaggregating Attention and Experts for Scalable MoE Inference
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

**LatentMoE 的 Roofline 扩展（Section 2.1-2.2）**：

LatentMoE 在 JANUS 的基础上扩展了 roofline analysis 到两个维度：(1) Memory bandwidth analysis for latency-critical serving（考虑完整 memory traffic: weights + inputs + intermediate activations）；(2) Communication-computation ratio analysis for throughput-oriented serving。

Memory BW Roofline (LatentMoE, GB200, FP4):
- Ridge point: F/BW_HBM = 10 PFLOPs / 8 TB/s = 1250 FLOPs/byte
- Per-expert compute: C_exp = 2·t_exp·d·m
- Per-expert memory traffic: M_exp = d·m + t_exp·(d+m)
- Arithmetic intensity: I = 2·t_exp·d·m / [d·m + t_exp·(d+m)]
- For Qwen3-235B (d=4096, m=1536): t_exp ≥ 1418 for compute-bound
- Typical latency-critical: t_exp ~ hundreds → firmly memory-bound

Communication Roofline (LatentMoE, GB200 NVL72):
- All-to-All volume per GPU: M_comm = 2.5·(N/EP)·t_exp·d (mixed precision: FP4+BF16)
- Compute time: t_comp = 4·t_exp·d·m/F
- Communication time: t_comm = 5·t_exp·d/BW_NVL
- Ratio t_comm/t_comp = 5·F/(4·m·BW_NVL) ≈ 9 for Qwen3-235B
- → Communication is ~9× more expensive than computation in throughput regime

Key insight from LatentMoE roofline: d (hidden dimension) is the only parameter that affects BOTH memory BW (via d·m) and communication cost (via K·d), while m (intermediate dim) only affects memory BW. This motivates compressing d→ℓ as the primary optimization target.

## Router Z-Loss

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router Z-Loss 是 Switch Transformer (Fedus et al., 2022) 引入的一种辅助损失函数，用于稳定 MoE 训练中 router（门控网络）的 logits 输出。其数学形式为：
$$L_z = \frac{1}{B} \sum_{i=1}^{B} \left(\log \sum_{j=1}^{E} \exp(x_i \cdot W_r)_j\right)^2$$
其中 B 为 batch token 数，E 为 expert 数，x_i 为第 i 个 token 的 hidden state，W_r 为 router 权重矩阵。实质是对 router logits 的 log-sum-exp 值（即归一化前的 softmax 分母的对数）施加 L2 惩罚，鼓励 router 输出值保持较小，防止 logits 漂移过大导致训练不稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 Joint MoE Scaling Laws 论文中，Router Z-Loss 的计算流程（每 MoE layer 前向）：

```
# 输入: x [B, L, d_model]
# Router: W_r [d_model, E]
# z_loss_coefficient = 0.001

# 1. Router logits 计算
router_logits = x @ W_r  # [B*L, E]

# 2. Z-Loss 计算
log_z = log(sum(exp(router_logits), dim=-1))  # [B*L], softmax 分母的对数
z_loss = (1 / (B*L)) * sum(log_z ** 2)         # scalar

# 3. 总 loss = cross_entropy + load_balancing_loss + z_loss_coefficient * z_loss
```

Z-Loss 与 Load Balancing Loss 的区别：Load Balancing Loss 惩罚的是 expert 间 token 分配的不均衡，Z-Loss 惩罚的是 router logits 幅值的过大增长。两者协同：Load Balancing 确保 token 均匀分布，Z-Loss 确保 router 数值稳定。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Z-Loss 在 MoE 训练中的典型使用：
- Switch Transformer 原论文：z_loss_coefficient = 0.01（推荐值）
- Joint MoE Scaling Laws：z_loss_coefficient = 0.001（更保守，配合 truncated normal initialization scale=0.1）
- ST-MoE (Zoph et al., 2022)：进一步提出 router z-loss 的变体和改进
- 通常与 Load Balancing Loss 共同作为辅助损失，系数通过网格搜索确定
- 对训练稳定性的影响：无 z-loss 可能导致 router logits 发散 → softmax 退化为 one-hot → 训练崩溃

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## Load Balancing Loss (MoE Auxiliary Loss)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load Balancing Loss（负载均衡损失）是 MoE 训练中的核心辅助损失函数，用于鼓励门控网络（router）将输入 token 均匀分配给所有 expert，防止出现"rich get richer"现象——少数 expert 接收绝大多数 token 而其余 expert 被闲置。其标准形式（GShard / Switch）为：
$$L_{LB} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e$$
其中 f_e = (1/B) Σ_{i=1}^{B} 1{token i routed to expert e}（实际路由到 expert e 的 token 比例），P_e = (1/B) Σ_{i=1}^{B} p_i(e)（token i 对 expert e 的 gate probability 平均值）。L_LB 的最小值为 1（完美均衡），最大值为 E（所有 token 路由到同一 expert）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在 Switch Transformer 的 top-1 routing 场景中：

```
# 输入: router_logits [B*L, E]
# P [B*L, E] = softmax(router_logits, dim=-1)

# 1. 计算 P_e (每个 expert 的平均 gate probability)
P_mean = P.mean(dim=0)  # [E]

# 2. 计算 f_e (每个 expert 的实际 token 分配比例)
expert_id = argmax(P, dim=-1)    # [B*L], top-1 路由
f_e = zeros(E)
for e in range(E):
    f_e[e] = count(expert_id == e) / (B*L)

# 3. 计算 Load Balancing Loss
L_LB = E * sum(f_e * P_mean[e] for e in range(E))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 默认系数：Switch Transformer 和 Joint MoE Scaling Laws 均使用 0.01
- 与 capacity factor 协同：Load Balancing Loss 是 soft constraint（通过梯度施加），capacity factor 是 hard constraint（token 硬上限）
- Router 结构影响：top-1 routing (Switch) 的 load imbalance 通常比 top-2 routing (GShard) 更严重，因此 L_LB 在 Switch MoE 中更为关键
- 效果验证：论文 280+ 实验中 L_LB 确保所有 expert 均被充分训练
- **SIMBAL 论文的批判**：LBL 强制 expert 分布接近均匀 → 不同 expert 在训练中接触到相似 token 集合 → expert 间产生知识冗余。训练早期 embedding 变化大 + near-uniform 分配 → 微小输入扰动可导致 token 被重新分配给不同 expert → 路由不稳定 → 进一步加剧冗余。LBL 还对 batch size 敏感，需要分布式同步最大化 batch size 以改善 specialization
- **与 SIMBAL 的对比**：SIMBAL 使用正交 Router（L_orth = ||R^T R - I||_1）保持 token 间成对角度关系，使相似 token 得到相似 expert 分布，从根源上避免冗余，而非仅让分布均匀

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient
- Load Balancing Mixture of Experts with Similarity Preserving Routers

## Switch Transformer / Switch MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Switch Transformer (Fedus et al., 2022) 是 Google 提出的简化 MoE 架构，核心设计：每个 token 仅路由到单个 expert（top-1 routing），而非 GShard 的 top-2。通过减少每 token 的 expert 计算量和通信量，Switch Transformer 以更简单的设计实现更大规模的稀疏模型。关键组件：(1) Top-1 gating (softmax → argmax)，(2) Load Balancing Loss (系数 0.01)，(3) Router Z-Loss，(4) Capacity Factor 控制每 expert 最大 token 数，(5) Truncated Normal Initialization (scale=0.1)。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Joint MoE Scaling Laws 使用的 Switch MoE 层前向流程：

```
# 输入: x [B*L, d_model]
# 超参数: E=experts数, CF=capacity_factor (训练), CF=inf (dropless eval)

# 1. Router
router_logits = x @ W_r           # [B*L, E]
router_probs = softmax(router_logits)  # [B*L, E]
expert_idx = argmax(router_probs, dim=-1)  # [B*L]

# 2. Expert FFN (SwiGLU, hidden=3*d_model)
for e in range(E):
    mask = (expert_idx == e)
    tokens_e = x[mask]  # [n_e, d_model]
    # FC1: gate = tokens_e @ W_gate[e], up = tokens_e @ W_up[e]
    # SiLU(gate) * up → h [n_e, 3*d_model]
    # FC2: out_e = h @ W_down[e] [n_e, d_model]
    y[mask] = router_probs[mask, e] * out_e

# 3. 辅助损失
L_aux = 0.01 * LoadBalancingLoss + 0.001 * RouterZLoss
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Switch Transformer 的实现要点：
- Expert FFN 与 dense FFN 同尺寸（expert granularity=1.0），保持简单
- 可用 standard GeMM 或 GroupedGEMM 实现 expert 批量计算
- Joint MoE Scaling Laws 在 280+ 模型中使用 Switch MoE，E∈{1,2,4,8,16,32}（E=1 退化为 dense）
- 评估时设置 capacity factor 为无穷大（dropless），确保所有 token 被处理
- 优势：每 token 仅计算 1 个 expert = 最低计算成本；劣势：load imbalance 比 top-2 更严重

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## Learning Rate Scaling Law for MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Learning Rate (LR) Scaling Law for MoE 是 Joint MoE Scaling Laws 论文提出的 MoE 训练超参数指导公式，根据模型规模和 expert 数自动选择最优 peak learning rate：

$$LR(N_{act \setminus e}, E) = \exp(8.39 - 0.81 \ln(N_{act \setminus e}) - 0.25 \ln(E))$$

其中 N_act\e 为不含 embedding 的 active parameters。该公式揭示两个关键趋势：(1) 更大模型 → 更低 LR（系数 -0.81），(2) 更多 expert → 更低 LR（系数 -0.25）。此前文献对 MoE LR 方向存在分歧——Dai et al. (2024) 用更低 LR，Zoph et al. (2022) 用更高 LR——该公式通过实验证据解决了这一分歧：更低的 LR 有利于 MoE 训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LR Scaling Law 的拟合和使用流程：

```
# 1. 网格搜索：针对不同 (N_act\e, E) 组合训练小模型
configs = [(N_1, E_1), (N_2, E_2), ...]  # 覆盖参数空间
candidate_LRs = [1e-5, 3e-5, 1e-4, ...]

for (N, E) in configs:
    for lr in candidate_LRs:
        train_model(N, E, lr, D=small_dataset)
        record loss(N, E, lr)

# 2. 确定每个 config 的 optimal LR
LR_opt[N, E] = argmin_lr loss(N, E, lr)

# 3. 最小二乘拟合（log-log 空间）
# ln(LR) = c0 + c1·ln(N_act\e) + c2·ln(E)
# → c0=8.39, c1=-0.81, c2=-0.25

# 4. 使用：给定 N, E 直接计算 optimal LR
lr = exp(8.39 - 0.81*ln(N_act\e) - 0.25*ln(E))
```

验证：E={1,8} 上拟合，E=4（插值）和 E=32（外推）上验证。Ablation 移除 E 项后 E=32 外推明显 suboptimal。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 公式使用 ln(LR) 的线性预测而非直接预测 LR，避免 Kaplan et al. (2020) 公式在 N>10^10 时预测负 LR 的问题——exp(负值) 始终为正
- 论文验证了 E 项的必要性（Fig.8 ablation）
- 适用场景：任何需要在不同 MoE 配置间做公平比较的训练实验
- 限制：公式基于 ≤5B total params 的实验拟合，大模型外推需谨慎

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## IsoFLOP Profiles

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

IsoFLOP Profiles（等计算量曲线）是 Scaling Laws 研究中用于可视化和分析模型性能的核心工具。在固定的训练 FLOPs budget F 下，绘制 loss L 随 model size N_act 的变化曲线——曲线上每一点的 (N_act, D) 满足 6·N_act·D = F。曲线的最低点对应 compute-optimal 配置 (N_act_opt, D_opt)。多条曲线的包络线展示 scaling 的整体趋势。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Joint MoE Scaling Laws 中 IsoFLOP Profiles 的生成：

```
给定: FLOPs budgets F ∈ {10^19, 10^20, 10^21, 10^22}
      scaling law L(N_act, D, E) = m(E)·N_act^μ(E) + n(E)·D^ν(E) + c

for each F:
    for N_act in reasonable_range:
        D = F / (6·N_act)  # token 数自动确定
        L = m(E)·N_act^μ(E) + n(E)·D^ν(E) + c
        plot(N_act, L)
    # compute-optimal point:
    G = (μ·m/(ν·n))^(1/(μ+ν))
    N_act_opt = G·(F/6)^(ν/(μ+ν)), D_opt = G^(-1)·(F/6)^(μ/(μ+ν))
```

关键观察（Fig.2a）：E=1 在较大 N_act 处最优，E=8 在较小 N_act 处最优——反映 ν(E) 更负时数据需求更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- IsoFLOP Profiles 可跨 E 值比较，揭示 FLOPs savings（Fig.2b: E=4 在 10^20 FLOPS 节省 40% vs dense）
- 曲线形状受 μ(E) 和 ν(E) 影响：μ 决定 small-model 侧陡峭度（underfitting），ν 决定 large-model 侧陡峭度（overtraining）
- 在 memory-constrained 场景中被 N_total ≤ M 截断，截断后最低点对应 memory-optimal 配置

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## Batch Size Ramp-up

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Batch Size Ramp-up 是一种训练策略，在训练初期使用较小的 batch size，随着训练进行逐步增大到目标值。动机来自 McCandlish et al. (2018) 的发现：过大的 batch size 会损害优化过程（尤其训练初期），存在临界 batch size B_crit 随训练 loss 降低而增大。Joint MoE Scaling Laws 采用直接 grid search 选择过渡点，而非使用 noise scale 做 B_crit 预测。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

论文使用的具体 ramp-up 策略：

```
# Phase 1: batch_size=64K tokens,  training tokens 0→0.5B
# Phase 2: batch_size=128K tokens, training tokens 0.5B→1.0B
# Phase 3: batch_size=256K tokens, training tokens 1.0B→end

# 实现
if tokens_trained < 0.5B:
    batch_size = 64K
elif tokens_trained < 1.0B:
    batch_size = 128K
else:
    batch_size = 256K
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 被当代 LLM 训练广泛采用：Gopher, LLaMA 3 等
- 实现方式：修改 DataLoader 的 batch size 参数，在指定 token 计数后触发 ramp
- 与 LR schedule 配合：batch size 增大时 linear scaling rule (lr ∝ batch_size) 建议同步调高 LR，但 ramp-up 通常不调 LR（已在 peak）
- 价值：早期小 batch 帮助模型快速学习基本模式，后期大 batch 最大化硬件利用率

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## Trapezoidal Learning Rate Schedule

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Trapezoidal Learning Rate Schedule（梯形学习率调度）是介于 constant 和 cosine 之间的 LR 调度策略，由 Hägele et al. (2024) 提出。形状为：warmup → constant plateau → linear decay → 0。核心优势：中间 checkpoint 可复用于不同训练时长的实验——短训练在 plateau 阶段结束时取 checkpoint，无需像 cosine schedule 那样因 LR 值不同而引入 bias。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Joint MoE Scaling Laws 的具体配置：

```
total_tokens = D_total  # 计划训练的总 token 数

lr_schedule(tokens_trained):
    if tokens_trained < 130M:           # warmup phase
        lr = peak_lr * (tokens_trained / 130M)  # linear warmup
    elif tokens_trained < 0.8 * D_total:  # plateau: 前 80% 训练
        lr = peak_lr                       # constant
    else:                                  # decay: 最后 20%
        progress = (tokens_trained - 0.8*D_total) / (0.2*D_total)
        lr = peak_lr * (1 - progress)      # linear decay to 0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 在 Scaling Laws 研究中特别有价值：需要训练多个不同 D 的模型时，短的 runs 可复用长的 runs 的中间结果，大幅降低总计算量
- Hoffmann et al. (2022) 指出 cosine schedule 的中间 checkpoint 会在 scaling law fitting 中引入系统性 bias（不同 D 的 checkpoint 处于不同 decay 阶段），trapezoidal 的 plateau 阶段避免了此问题
- Hägele et al. (2024) 证明 trapezoidal 的性能与 cosine 相当，但为 scaling law 研究提供了显著的实验效率优势

涉及论文标题：
- Joint MoE Scaling Laws: Mixture of Experts Can Be Memory Efficient

## Goal-Conditioned Markov Decision Process (Goal-Augmented MDP)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Goal-Conditioned MDP（目标条件马尔可夫决策过程）是标准 MDP 的扩展，将目标 $g \in \mathcal{S}$ 作为额外输入条件注入到策略、价值函数和奖励函数中。标准 MDP 由 $(\mathcal{S}, \mathcal{A}, p, r, \gamma)$ 定义，而 Goal-Conditioned MDP 将奖励函数扩展为 $r(s_t, a_t, g): \mathcal{S} \times \mathcal{A} \times \mathcal{S} \to \mathbb{R}$，策略扩展为 $\pi(a_t|s_t, g)$。这使得同一个 agent 可以泛化到训练时未见过的目标状态，而无需为每个目标重新训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Goal-Conditioned MDP 定义了 RL 训练的形式化框架。在本论文中，目标 $g = [x_g, y_g, \theta_g, 0, 0, 0]^T$ 定义为期望的位姿状态，cost 函数 $c(s,a,g) = ||W e||_{0.25}$（其中 $e = [e_{xy}, e_\theta, \beta, \dot{\beta}, v_f]^T$ 为误差向量），$p=0.25$ 使得 cost 呈稀疏特性——未收敛到目标的代价近乎恒定，只有到达目标时显著降低，从而鼓励时间最优行为。

伪代码（Goal-Conditioned RL 训练循环）：
```
# 每 episode 采样新目标
g = sample_goal()
s = env.reset()
for t = 1..T:
    a = policy(s, g)          # goal-conditioned policy
    s' = dynamics(s, a)
    c = cost(s, a, g)         # goal-conditioned cost
    buffer.store(s, a, c, s', g)
    # 从 buffer 采样更新 policy 和 critic
    update_policy_and_critic(buffer, g)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 通用实现：在 actor/critic 网络中将 goal 与 state 拼接后输入，或使用编码器将绝对坐标转换为相对目标的坐标（如本论文将 $(x_f, y_f, \theta_f)$ 编码为 $(x_f - x_g, y_f - y_g, \sin(\theta_f - \theta_g), \cos(\theta_f - \theta_g))$）。
- Hindsight Experience Replay (HER, Andrychowicz et al. 2017) 是 Goal-Conditioned RL 的关键训练技巧：将失败 episode 中实际到达的状态作为"假想目标"重新标记，大幅提高样本效率。
- 在 LLM/推理系统中，Goal-Conditioned 思想可应用于将请求的 SLO 目标作为 condition 注入调度策略。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

## Actor-Critic Reinforcement Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Actor-Critic 是强化学习中的一种混合架构，结合了 Policy-based（Actor）和 Value-based（Critic）方法的优势。Actor（策略网络 $\pi_\phi(a|s)$）直接输出动作，决定"做什么"；Critic（价值网络 $V_\psi(s)$ 或 $Q_\psi(s,a)$）评估当前策略的好坏，为 Actor 提供低方差的梯度信号。Actor 按照 Critic 的建议方向更新策略参数，Critic 则通过 TD 学习逼近真实价值函数。这种架构的优势是：Actor 可以在连续动作空间中学习（Policy Gradient 的优势），而 Critic 通过 Bootstrapping 减少梯度方差（Value-based 的优势）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在本论文中，Actor 参数化一个 Gaussian 动作分布（经 tanh 限幅后缩放至最大加速度），Critic 构造为 $L_\psi(s,a,g) = Q_\psi Q_\psi^T$（确保正输出满足 Lyapunov 下界条件）。Critic 通过最小化与 target critic $L_{target} = c + \gamma \bar{L}_{\bar{\psi}}$ 的 MSE 训练，Actor 通过最小化 $J(\phi) = \lambda_e (\log \pi_\phi + \mathcal{H}) + \lambda_l \Delta \mathcal{L}_\psi$ 训练（Lyapunov 条件违反量 + 熵正则）。

伪代码（通用 AC 算法骨架）：
```
# 初始化 Actor π_φ, Critic V_ψ
for each episode:
    s = env.reset()
    for each step:
        a = π_φ(s) + noise          # Actor: 动作选择
        s', r = env.step(a)
        δ = r + γ·V_ψ(s') - V_ψ(s)  # Critic: TD error
        V_ψ ← V_ψ + α_c·δ·∇V_ψ     # Critic 更新
        π_φ ← π_φ + α_a·δ·∇log π_φ # Actor 更新 (policy gradient)
        s = s'
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 最广泛使用的实现：stable-baselines3（PyTorch），提供 A2C、PPO、SAC、TD3 等 AC 算法。
- 本论文使用 ALAC（Lyapunov-based AC），基于 stable-baselines3 实现。Actor/Critic 均为前馈 NN（层结构 48→96→144→96→48，SoftPlus 激活），在仿真环境中训练至 $\lambda_l$ 收敛到 0.8。
- 在 AI 系统中，AC 可用于学习请求调度策略（state=队列状态，action=调度决策，reward=latency/SLO violation）。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

## Sampling-based Lyapunov Function (ALAC Critic)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sampling-based Lyapunov Function 是 Adaptive Lyapunov-based Actor-Critic (ALAC, Wang et al. 2023) 算法训练的 Critic 网络所满足的稳定性证书。传统 Lyapunov 函数要求对整个状态空间严格满足衰减条件，而 Sampling-based Lyapunov 仅要求在经验采样分布 $\mathcal{S}_\pi$ 上满足。具体条件（Theorem 3.1）：存在 $L(s,g)$ 满足 (a) $k_l c_\pi \le L \le k_u c_\pi$（上下界），(b) $L(s,g) \ge c_\pi + \lambda \mathbb{E}[L(s',g)]$（期望衰减），(c) 特定不等式约束在稳态分布上成立。当 Critic 满足这些条件时，系统满足 mean cost stability（Definition 3.1: $\lim_{t\to\infty} \mathbb{E}[c_\pi(s_t,g)] = 0$）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ALAC 训练过程中，Critic 通过最小化 Bellman-like loss $J_c(\psi) = \mathbb{E}[(L_\psi - (c + \gamma \bar{L}_{\bar{\psi}}))^2]$ 来逼近 sampling-based Lyapunov function。Actor 则通过 Lagrange 乘子 $\lambda_l$ 引导 Critic 满足条件 (c)。当 $\lambda_l$ 自适应下降到 < 1 时（本论文训练至 0.8），Critic 被认证为有效 Lyapunov 函数。

Lyapunov 条件验证流程：
```
训练循环中:
1. Critic target: L_target = c(s,a,g) + γ·L̄(s', a', g)  // TD-style bootstrap
2. Critic loss: Ĵ_c = E[(L - L_target)²] + ρ·E[(1-||∇L||₂)²]  // + gradient penalty
3. 计算条件(c)违反量: ΔL = L(s',π(s')) - L(s,a) + k(L(s,a) - λL(s',π(s')))
4. Actor loss: J_φ = λ_e·(log π_φ + H) + λ_l·ΔL    // 引导满足条件(c)
5. Lagrange 更新: λ_l ← λ_l + α·ΔL                 // 收敛至0.8→Lyapunov有效
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- ALAC 论文（Wang et al. 2023, CoRL）开源: https://github.com/ShengjieWang00/ALAC
- 与标准 Actor-Critic 的关键区别：Critic 构造为正定形式 $L=QQ^T$（满足条件a），额外的 Lagrange 乘子自适应机制保证条件(c)。
- 在 LLM 推理系统中的应用潜力：将 Lyapunov 稳定性用于保证请求队列的 long-term stability（mean latency bounded），但尚未有相关工作。
- 本论文创新：在 critic loss 中加入 gradient penalty 项 $\rho\mathbb{E}[(1-||\nabla L||_2)^2]$（Eq. 16），鼓励 critic 的 1-Lipschitz 性质，缓解下游 MPC 优化困难。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

## Gradient Penalty (1-Lipschitz Regularization for Neural Networks)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Gradient Penalty（梯度惩罚）是一种正则化技术，最早由 Gulrajani et al. (2017) 在 WGAN-GP 中提出。其核心思想是在损失函数中加入惩罚项 $\rho \cdot \mathbb{E}[(||\nabla_x f(x)||_2 - 1)^2]$，强制神经网络的梯度范数接近 1，从而实现 1-Lipschitz 连续性：$||f(x_1) - f(x_2)|| \le ||x_1 - x_2||$。在 WGAN 中，这是为了满足 Kantorovich-Rubinstein 对偶要求；在本论文中，用途完全不同——为下游 MPC 优化器提供平滑的优化景观。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在本论文中，gradient penalty 被加入 critic 训练损失（Eq. 16）：
$$\hat{J}_c(\psi) = J_c(\psi) + \rho \cdot \mathbb{E}_{\mathcal{D}}[(1 - ||\nabla L_\psi(s,a,g)||_2)^2]$$

其中 $\nabla L_\psi$ 是 critic 输出对输入的梯度（通过 autograd 计算），$\rho \ge 0$ 控制惩罚强度。这使得 critic 在训练数据分布上近似 1-Lipschitz，从而其梯度 bounded：$||\nabla L_\psi||_2 \approx 1$。当这个 critic 被用作 MPC 的 cost function 时，SQP-RTI solver 在优化非凸 NN-based cost 时更稳定。

伪代码（Critic 训练时施加 Gradient Penalty）：
```
for batch (s, a, s') in replay_buffer:
    s.requires_grad = True
    L = critic(s, a, g)
    grad_L = autograd(L, s, create_graph=True)  # ∇_s L
    gp = ((grad_L.norm(2) - 1) ** 2).mean()     # gradient penalty
    
    L_target = cost + γ * target_critic(s', π(s'), g)
    critic_loss = F.mse(L, L_target) + ρ * gp
    critic_loss.backward()
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch 实现：`torch.autograd.grad(outputs, inputs, create_graph=True)` 计算梯度，然后施加二次惩罚。
- 在 GAN 训练中（WGAN-GP），gradient penalty 作用于 discriminator 对插值样本的梯度。
- 在本论文的创新用途中：gradient penalty 使 critic 成为"MPC 友好的" cost function，解决了 AC4MPC (Reiter et al. 2024) 中观察到的"因 NN critic 高度非线性导致优化困难"问题。
- 与 Spectral Normalization 的关系：两者都实现 Lipschitz 约束，但 gradient penalty 是软约束（通过 loss），Spectral Normalization 是硬约束（通过权重归一化）。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC

## Model Predictive Control (MPC) with Learned Cost Function

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model Predictive Control (MPC) 是一种基于模型的优化控制方法。在每个控制周期，MPC 求解一个有限 horizon 的约束优化问题（通常为 NLP），得到最优控制序列，但仅执行第一步控制，下一周期重新求解（receding horizon）。标准 MPC 的 cost function 通常由人工设计（如二次型跟踪误差），而本论文中的 Actor-Critic MPC 使用 RL 训练的 critic $L_\psi$ 作为 cost function——terminal cost $l_f(x_N, g) = L_\psi(x_N, 0, g)$，stage cost $l(x_n, u_n, g) = \Delta t \cdot \tilde{L}(x_n, u_n, g)$（critic 在上一解处的二阶 Taylor 近似）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

本论文的 Actor-Critic MPC 每步执行流程（Jetson AGX Orin, <100ms/iter）：
```
1. 输入延迟补偿: x_init ← propagate(x_init, 200ms)  # 匹配执行器延迟
2. 构建 NLP (Eq. 20):
   min Σ_{n=0}^{N-1} Δt·L̃(x_n, u_n, g) + L_ψ(x_N, 0, g)  # stage + terminal cost
   s.t. x_{i+1}=f(x_i, u_i)  # 运动学模型 (4阶 Runge-Kutta)
        状态/控制/障碍物约束
3. Stage cost L̃: critic 在上一解处的二阶 Taylor 近似
   L̃(z_n) ≈ ∂L/∂z·(z* - z_n) + 0.5 ∂²L/∂z²·(z* - z_n)²
   （比直接 NN forward 更轻量）
4. SQP-RTI solver (HPIPM QP) → 最优轨迹
5. 取 x₁ 的 (β̇, v_f) → 发送给低层控制器
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 标准 MPC 工具链：CasADi（符号建模）+ Acados（嵌入式优化框架）+ HPIPM（高性能 QP solver）。本论文使用 CasADi + Acados + L4CasADi（将 PyTorch NN 转换为 CasADi 符号表达式）。
- Learned Cost MPC 的关键挑战：(a) NN 的高度非线性使 NLP 难以优化 → 通过 gradient penalty (1-Lipschitz) 缓解；(b) NN 推理开销 → 通过 Taylor 近似（仅需前向一次 NN 计算 terminal cost，stage cost 复用）。
- 在 LLM 推理系统中，MPC 可用于动态调整 batch size、request admission control 等资源分配问题（如以 latency 和 throughput 为 cost，GPU memory 和 compute 为约束），但目前尚无相关工作将 Learned Cost MPC 用于 LLM serving。

涉及论文标题：
- Autonomous Wheel Loader Navigation Using Goal-Conditioned Actor-Critic MPC


## LatentMoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LatentMoE 是 NVIDIA 提出的一种新 MoE 架构，核心思想是将 routed experts 的输入从原始 hidden dimension d 压缩到低维 latent space ℓ，利用节省的 memory bandwidth 和 communication 成本来扩展 expert 数量 N 和 top-K，在 iso-inference-cost 下提升模型精度。这是首次从 hardware-software co-design 角度系统性重新思考 MoE 架构设计的工作。

LatentMoE 包含两个变体：(1) ℓ-MoE_eff：压缩 hidden dim (d→ℓ)、扩展 N→αN、K 不变，在保持精度的同时降低 inference cost；(2) ℓ-MoE_acc（推荐）：同时扩展 N→αN 和 K→αK，在 iso-inference-cost 下提升精度。α = d/ℓ 为压缩比，论文通过 sweep 验证 α=4 为 safe compression ratio。

LatentMoE 被 NVIDIA Nemotron-3 Super/Ultra 模型采用，已通过 TensorRT-LLM v1.2.0+ 部署支持。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LatentMoE ℓ-MoE_acc 的算法 pipeline：

```
# LatentMoE ℓ-MoE_acc MoE Layer Forward
# 配置: d=4096, ℓ=1024, α=4, N=128 → N'=512, K=6 → K'=24, m=2688

Input: x ∈ R^{B×S×d}

# 1. Router（原始空间 d）
gate_logits = x @ W_r'.T              # W_r' ∈ R^{N'×d}
gate_probs = softmax(gate_logits)      # [B,S,N']
topk_vals, topk_ids = topk(gate_probs, K'=24)

# 2. Shared Down-Projection（所有专家共享）
z = W_↓ @ x                            # W_↓ ∈ R^{ℓ×d}, z ∈ R^{ℓ×B×S}

# 3. All-to-All Dispatch（在 latent space ℓ 中）
# 通信量 per token: ℓ (1024), 共 K'×ℓ = 24×1024 = 24576 bytes
# vs Standard MoE: K×d = 6×4096 = 24576 bytes (相同)

# 4. Expert FFN（在 latent space ℓ 中，per expert）
for each expert e (among 512 routed experts):
    z_e = tokens routed to e            # [n_e, ℓ]
    # W_gate ∈ R^{m×ℓ}, W_FC1 ∈ R^{m×ℓ}, W_FC2 ∈ R^{ℓ×m}
    h_gate = activation(z_e @ W_gate.T)  # [n_e, m]
    h_up   = z_e @ W_FC1.T              # [n_e, m]
    h      = h_gate * h_up              # [n_e, m] (SwiGLU/Squared-ReLU)
    e_out  = h @ W_FC2.T                # [n_e, ℓ]  FC2 down-projection

# 5. All-to-All Combine + Up-Projection
z_combined = aggregate across experts    # [B×S, ℓ]
routed_out = W_↑ @ z_combined            # W_↑ ∈ R^{d×ℓ}, [d, B×S]

# 6. Shared Experts（原始空间 d，S=2）
shared_out = Σ E_j(x; d) for j=1..S

# 7. Final output
output = routed_out + shared_out
```

关键对比：Standard MoE 中每个 expert 的权重总大小为 3×d×m（FC1 gate + FC1 up + FC2 down），LatentMoE 中每个 expert 的权重总大小为 3×ℓ×m，减少 α=4× per expert。但由于 K'=αK=24 vs K=6，总激活参数 K'×3×ℓ×m = K×3×d×m（在 ℓ-MoE_acc 中相同）。

Expert 组合空间：C(512,24) >> C(128,6)，指数级增长 token 级别的 expert combination diversity。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LatentMoE 通过以下方式实现：
- 共享的 W_↓ 和 W_↑ 对所有 routed experts 复用，仅增加 modest 的计算开销（~9% at trillion scale）
- Expert 权重在 latent space 中，每个 expert 的 weight loading memory BW 降低 α×
- All-to-All 通信在 latent space 中进行，per-token message size 降低 α×（但 K'=αK 补偿总通信量）
- TensorRT-LLM v1.2.0+ 原生支持 LatentMoE 部署
- 训练使用 DeepSeek-v2-lite 超参和架构，配合 aux-loss-free load balancing
- 可通过分离 CUDA streams for routed/shared experts 和专门的小 inner-dimension GEMM kernels（CUTLASS）进一步优化

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

## Combinatorial Sparsity in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Combinatorial Sparsity（组合稀疏性）是 MoE 模型中 token 级专家组合多样性的理论度量。由于每个 token 通过 top-K routing 从 N 个 expert 中选择 K 个激活，共有 C(N,K) 种可能的专家组合。这一组合数随 N 和 K 的增长呈指数级（或超指数级）增长，构成了 MoE 模型表达能力的一个重要来源。

LatentMoE 论文首次系统性地将 combinatorial sparsity 纳入 MoE 架构设计的理论框架（Design Principle V）。核心数学关系：

$$\begin{pmatrix} \alpha N \\ \alpha K \end{pmatrix} \ge \left( \begin{pmatrix} N \\ K \end{pmatrix} \right)^{\alpha}$$

即同时将 N 和 K 缩放 α 倍后，组合数以原组合数的 α 次幂增长。例如 C(512,24) >> C(128,6)，不仅仅是线性增加。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
MoE Expressivity = Router Selection Freedom + Expert Specialization + Combinatorial Diversity

Standard MoE (N=128, K=6):
  Expert combinations: C(128,6) = 5.5 × 10^9
  Each token selects from ~5.5 billion possible expert subsets

LatentMoE ℓ-MoE_acc (N=512, K=24, α=4):
  Expert combinations: C(512,24)
  C(512,24) ≥ C(128,6)^4 = (5.5 × 10^9)^4

Pipeline impact:
  - More N → finer-grained expert specialization (each expert covers narrower domain)
  - More K → each token benefits from more expert perspectives
  - C(N,K) diversity → better coverage of input distribution modes
```

Combined with Barron function theory: MoE layer 的有效非线性预算 U_eff ∝ K·m。LatentMoE 的 K'=αK 将 U_eff 提升 α 倍（在 ℓ-MoE_acc 中），同时保持 FLOPs 不变或降低。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式：
- 通过增加 expert 数量 N 和 top-K 来实现更大组合空间（LatentMoE 的核心贡献）
- 必须在 iso-inference-cost 约束下进行（通过 latent space projection 补偿增加的 K 带来的 memory/communication cost）
- 实践中的实现：Training 时使用 aux-loss-free load balancing (Wang et al., 2024) + load balancing loss coefficient=10^-4 确保 token 均匀分布
- 组合稀疏性在推理时自动生效：每个 token 选择不同的 K'=24 experts subset，产生动态的专家组合

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models (Dai et al., 2024)

## Feature Rank (r_eff) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Feature Rank（r_eff，有效特征秩）是 LatentMoE 论文引入的信息论概念，定义为：对于给定的推理任务，保留任务相关信息所需的最小自由度（degrees of freedom）。r_eff 构成了 MoE routed expert 输入维度 d 的信息论下界——将 d 压缩到低于 r_eff 会导致任务相关信息的不可逆丢失，从而造成精度塌缩。

r_eff 的作用是为 latent space compression 提供理论边界：只要压缩后的 latent dim ℓ ≥ r_eff，信息损失 negligible。LatentMoE 通过压缩比 sweep 实验（α=1,2,4,8）验证 r_eff ≈ d/4（即 α=4 为 safe compression ratio）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Feature Rank Determination Pipeline (Design Principle IV + Empirical Validation):

1. Hypothesis:
   For a given task T and model dimension d:
   ∃ r_eff(T) such that:
   - Compression to ℓ ≥ r_eff → information loss negligible
   - Compression to ℓ < r_eff → accuracy collapse

2. Empirical Validation (LatentMoE 16BT-2BA ablation):
   Fix architecture, sweep compression ratio α = d/ℓ:
   
   α=1 (baseline, ℓ=d=2048): validation loss = L_base
   α=2 (ℓ=1024):          validation loss ≈ L_base (marginal)
   α=4 (ℓ=512):           validation loss ≈ L_base (acceptable)
   α=8 (ℓ=256):           validation loss >> L_base (collapse!)

   → r_eff ≈ d/4 = 512 for this model/task configuration

3. Pipeline Integration:
   if ℓ = d/α ≥ r_eff:
       use LatentMoE with compression ratio α
   else:
       reduce α until ℓ ≥ r_eff (safety margin)

4. Task Dependence:
   - r_eff is task-specific (different tasks extract different information)
   - Larger models may have larger r_eff
   - More diverse training data may require larger r_eff
```

LatentMoE 进一步验证 r_eff 在更大规模（95B）上仍然有效——即对于更大的模型，α=4 的压缩仍然 safe。这表明 r_eff 可能更多是任务驱动的而非模型规模驱动的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

r_eff 本身是一个理论概念，实践中通过以下方式使用：
- 决定 LatentMoE 的压缩比 α：α_max = d / r_eff（最大安全压缩比）
- 通过实验 sweep 确定：对给定模型/任务配置 sweep α=1,2,3,4,6,8... 找到精度开始下降的拐点
- 类似于 quantization 中的 effective bit-width 概念：不同 tensor/operator 需要不同精度
- 当前需要 per-configuration 实验验证，未来可通过理论分析/小规模 proxy 实验预测

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts

## Effective Parameter Multiplier (EPM)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Effective Parameter Multiplier (EPM) 是一种用于衡量模型改进效果的方法论指标。给定一个基线的 scaling law f(N)（将参数数量 N 映射到任务精度），对于一个经过某种优化（如 LatentMoE、量化、剪枝）的模型，EPM 定义为达到相同精度所需的 baseline 参数量与优化模型物理参数量的比值：

$$N_{eff} = f^{-1}(S_{treat})$$

$$\lambda = \frac{N_{eff}}{N_{treat}}$$

其中 S_treat 是优化模型在目标 benchmark 上的得分，f 是 baseline 模型的 scaling law（将参数映射到精度），f^{-1} 是其反函数。λ > 1 表示优化模型相当于拥有更多有效参数。

EPM 的核心作用是建立 iso-accuracy baseline，用于公平比较不同架构的 inference efficiency。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
EPM Construction Pipeline (LatentMoE, Section 4.3.1):

1. Establish Baseline Scaling Law:
   Use Qwen-3-Dense model family (0.6B, 1.7B, 4B, 8B, 14B, 32B)
   Measure MMLU accuracy for each size
   Fit: f(N) = a·log(N) + b  (log-linear fit)

2. Measure Treated Model:
   Kimi-K2-1T-LatentMoE MMLU score = S_treat

3. Compute Effective Parameters:
   N_eff = f^{-1}(S_treat) = exp((S_treat - b) / a)

4. Compute EPM:
   λ = N_eff / N_treat = exp((S_treat - b)/a) / 1.0T
   For Kimi-K2-1T-LatentMoE: λ ≈ 1.35

5. Construct Iso-Accuracy Baseline:
   N_iso = λ · N_treat = 1.35T
   → Kimi-K2-1.35T: 61 layers → 80 layers (d=4096, standard MoE)

6. Compare Inference Efficiency:
   Kimi-K2-1T-LatentMoE (1T physical params) 
   vs Kimi-K2-1.35T (1.35T, iso-accuracy)
   Result: 1.24×-3.46× speedup for LatentMoE
```

EPM 基于 Frantar et al. (2025) 的 "Effective Parameter Count" 框架（Compression Scaling Laws: Unifying Sparsity and Quantization）。原始框架用于量化/稀疏模型的压缩效率评估，LatentMoE 将其推广到架构改进的效率评估。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实践使用步骤：
1. 选择一组同架构、不同规模的 baseline 模型（如 Dense 或 Standard MoE）
2. 在目标 benchmark 上测量各 scale 的精度
3. 拟合 scaling law f(N)（通常 log-linear）
4. 测量优化模型在相同 benchmark 上的精度
5. 反推 N_eff 并计算 λ

局限性：
- 依赖于 scaling law 的拟合质量（需要足够多的 baseline scale points）
- 外推超出 baseline 训练范围可能不准确
- λ 是 benchmark-dependent（不同 benchmark 可能得出不同的 λ）
- 假设 baseline 和 treated model 在 scaling behavior 上可比

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
- Compression Scaling Laws: Unifying Sparsity and Quantization (Frantar et al., 2025)

## RMoE (Layerwise Recurrent Router for MoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

RMoE (Recurrent Router for Mixture-of-Experts) 是一种在 MoE 路由过程中引入跨层循环依赖的 router 设计。核心思想：当前 MoE 中不同层的 router 独立工作，各自仅基于本层 hidden state 做出路由决策，缺乏跨层协调。RMoE 在每层 router 前插入一个跨层共享的轻量级 Gated Recurrent Unit (GRU)，将各层路由决策串联为一个序列——第 i 层的路由结果依赖于第 i-1 层及之前所有层的 GRU 隐状态：

1. **逐层投影**：x_i' = Proj_i(x_i)，将 hidden state x_i ∈ R^h 降维到 GRU 状态维度 R^p（p=128），每层使用独立的 Proj_i（因为不同层的 hidden state norm 和方差差异大）
2. **跨层 GRU**：h_i = GRU(x_i', h_{i-1})，其中 GRU 跨层共享参数，h_i 携带前 i-1 层路由决策的历史信息
3. **路由决策**：score_i = softmax(h_i @ G_i)，top-k gating 选择 experts
4. **标准 MoE 计算**：y_i = sum_n g_n(h_i; G_i, k) * E_n(x_i)

RMoE 的四个核心特性：(1) 跨层信息共享——GRU 显式传递历史路由决策，使当前层 router 知道 token 在之前层被分配到哪些 experts；(2) 额外梯度路径——GRU 提供跨层反向传播的 Recurrent Gradient，辅助 router 优化；(3) 正交兼容——GRU 路由作为一个新计算阶段，可与 XMoE、DeepSeekMoE 等现有方法组合；(4) 开销可忽略——对 0.91B 模型仅增加 ~3.5M 参数，训练速度仅降低 0.4%，GPU 内存仅增加 1.4%。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

RMoE 在 decoder-only transformer 中的完整前向流程：

```
# 初始化：跨层共享 GRU 参数 W_s, U_s, W_z, U_z, W_h
# 每层独立：Proj_i (linear, h→p), Router G_i (linear, p→N)
h_0 = zeros(p)  # p=128

for layer i in 1..L:
    # Step 1: Attention (标准)
    x_i = Attention_i(LayerNorm_attn(x_{i-1})) + x_{i-1}

    # Step 2: RMoE (替代标准 FFN/MoE)
    x_i_norm = LayerNorm_moe(x_i)

    # 2a: 逐层投影降维
    x_i_prime = Proj_i(x_i_norm)  # [B, L, h] → [B, L, p]

    # 2b: GRU 跨层循环
    s_i = sigmoid(W_s @ x_i_prime + U_s @ h_{i-1})   # reset gate
    z_i = sigmoid(W_z @ x_i_prime + U_z @ h_{i-1})   # update gate
    h_tilde = tanh(W_h @ x_i_prime + s_i * (W_h @ h_{i-1}))
    h_i = (1 - z_i) * h_tilde + z_i * h_{i-1}       # [B, L, p]

    # 2c: 基于 GRU 输出的 routing
    gating_scores = softmax(h_i @ G_i)               # [B, L, N]
    topk_val, topk_idx = topk(gating_scores, k)

    # 2d: 稀疏 Expert 计算
    y_i = zeros_like(x_i_norm)
    for n in topk_idx:
        y_i += topk_val[n] * Expert_n(x_i_norm)

    # Step 3: 残差连接
    x_i = y_i + x_i
```

**消融变体**：
- **RMoE + NP (Not Passing)**：将 h_i = GRU(x_i', h_0)，取消跨层 recurrence（GRU 变为 stateless），与 RMoE 参数量相同但性能大幅下降（Enwiki8 BPC 1.141→1.150），验证跨层 recurrence 是主要贡献者
- **RMoE + detach h_{i-1}**：detach h_{i-1} 阻止其梯度计算，仅保留前向信息共享。性能比 RMoE-NP 更差（1.159 vs 1.150），证明 Recurrent Gradient（反向梯度流）比前向信息共享更重要
- **RMoE + NP + r-α**：将上一层的 routing logits 作为残差加到当前层：g_i = softmax(h_i @ G_i) + α * softmax(h_{i-1} @ G_{i-1})。性能接近 RMoE-NP，不能提供有效的跨层信息。detach 后性能大幅下降，再次验证额外梯度的重要性

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **代码开源**：https://github.com/qiuzh20/RMoE
- **小模型实现**：基于 PyTorch 原生，8 层 decoder-only transformer (hidden=352, 16 experts top-2)，1×A100 约 21 小时训练
- **大模型实现**：基于 Megablocks 框架（block-sparse kernel），24 层 Llama-style (hidden=1280, RoPE + SwiGLU + RMSNorm, 16 experts top-4 fine-grained)，8×A100 约 5 天 pre-training
- **关键超参数**：GRU hidden dim p=128（p=256/512 在大规模设置下性能下降）；使用逐层独立 Proj_i 而非共享投影器（共享 Proj 性能更差，因为不同层 hidden state 分布差异大）；GRU 优于 RNN 和 LSTM
- **SFT 时冻结策略**：冻结 GRU 和线性 router 层，或仅冻结 router
- **Load Balance Loss 权重**：0.01（与标准 SMoE 相同）

涉及论文标题：
- Layerwise Recurrent Router for Mixture-of-Experts

## Expert Load Imbalance in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Load Imbalance 是 MoE 模型的固有特征：由于 gate network 是动态可训练的，不同 expert 被路由到的 token 数量存在显著差异。Lazarus 论文（Figure 2）展示了 GPT-L (16 experts) 训练过程中，up to 87% tokens 被路由到最热门的 2 个 experts，而最冷门的 experts 几乎不被激活。这种不平衡在不同层之间、以及同一层的不同训练迭代之间动态变化（gate network weights 随时间更新）。

在传统 EP 中，所有 experts 被等分到不同 GPU，load imbalance 直接导致 GPU 间计算不均衡——持有 popular experts 的 GPU 处理远超其他 GPU 的 token 数，其他 GPU idle waiting。这不仅降低了训练吞吐（straggler effect），也使得故障恢复更困难（如果 cold expert 的唯一 replica 所在 GPU 故障）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

MoE 层中 expert load 的计算过程：

```
# MoE layer forward pass (per layer, per iteration)

# Step 1: Gate network computes routing probabilities
gate_logits = Linear_gate(hidden_states)  # [B, S, E]
gate_probs = softmax(gate_logits)          # [B, S, E]

# Step 2: Top-k selection + auxiliary load balancing loss
# Load balancing loss (Switch Transformer style):
# f_e = (1/T) * Σ_{tokens} 1{top-k includes expert e}     # fraction of tokens
# P_e = (1/T) * Σ_{tokens} softmax(gate_logits)[e]         # avg gate prob
# L_balance = E * Σ_e f_e * P_e                             # scalar loss
# Total loss = LM_loss + α * L_balance

# Step 3: Expert load t_e for iteration
t_e = Σ_{tokens} 1{e in top-k(token)}   # actual token count per expert

# Lazarus observation: t_e varies significantly
# e.g., 16 experts: t_1 ≈ t_2 ≈ 43% each, t_15 ≈ t_16 ≈ 0.5% each
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

缓解 Expert Load Imbalance 的常见策略：(a) **Load Balancing Loss** (Switch Transformer, GShard)：在 training loss 中加入辅助项惩罚不均衡路由；(b) **Expert Capacity**：设置每个 expert 的最大 token 容量，超出则 drop tokens；(c) **Dynamic Parallelism Switching** (Tutel, SmartMoE)：根据 expert load 动态切换 parallelism 策略；(d) **Expert Replication** (FasterMoE, FlexMoE, Lazarus)：为 popular experts 分配多个 replicas 增加计算容量。Lazarus 使用策略 (d) 在弹性训练环境下——根据运行时收集的 routing history 的 t_e，用 Eq. 1 计算自适应 replica 分配，使 r_e ∝ t_e。

Lazarus 的消融实验（single MoE layer with 8 experts）显示：当 load ratio 从 1:1 (balanced) 变为 4:1 (imbalanced)，DS baseline 吞吐急剧下降（因 straggler GPU），而 Lazarus 通过 adaptive expert allocation 保持恒定吞吐。

涉及论文标题：
- Lazarus: Resilient and Elastic Training of Mixture-of-Experts Models with Adaptive Expert Placement
- Least-Loaded Expert Parallelism: Load Balancing An Imbalanced Mixture-of-Experts

**LLEP 对 Expert Load Imbalance 的洞察**：

LLEP 从系统而非算法角度重新框定了 expert load imbalance。论文实证分析 gpt-oss-20b (32 experts, 8-way EP) 在数学数据集上的路由模式：(a) 特定 expert (E11) 持续接收最多 token（up to 20% load vs 3% balanced）；(b) 某些 GPU 整体过载（GPU 0 有 30-35% vs 12.5% balanced）；(c) 不均衡程度 per-batch 动态变化。

LLEP 的核心论点：mild imbalance 是训练良好的 MoE 的自然属性（专家专业化），而非需要算法层面修正的缺陷。强制均衡路由（如 auxiliary load balancing loss）会破坏已学习的专家专业化模式。因此 LLEP 采取系统级负载均衡——在 dispatch 阶段动态将超载 GPU 的 excess load 溢出到欠载 GPU，保持 exact MoE computation。α 容量因子、m 最小 GEMM token 数、λ 自适应阈值构成超参调优空间，允许用户根据硬件配置权衡均衡度与通信开销。

## Sparse Upcycling

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparse Upcycling 是一种将预训练 dense checkpoint 转换为稀疏激活 MoE (Mixture-of-Experts) 模型的训练初始化技术。由 Komatsuzaki et al. (ICLR 2023) 首次提出，核心思想是：将 dense 模型中某些 FFN (Feed-Forward Network) 层的权重复制 N 次，初始化 N 个 expert，同时添加一个随机初始化的 router（门控网络），其余层（embedding、attention、norm 等）直接从 dense checkpoint 复制。随后仅需少量继续训练（<1% 原始预训练 compute），router 学会将不同 token 路由到不同 expert 组合，expert 在 fine-tuning 中逐渐分化。该方法避免从头训练 MoE 的高昂成本（数据需求大、训练不稳定、expert collapse 等），同时复用已投入的预训练 GPU 小时。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Sparse Upcycling 的算法 pipeline（以 Llama 3-8B → E8T2 为例）：

```
# 输入: dense checkpoint Θ_dense (Llama 3-8B), N=8 experts, K=2 (Top-K)
# 输出: initialized MoE checkpoint Θ_moe

def sparse_upcycling(Θ_dense, N, K, moe_layer_indices):
    Θ_moe = deep_copy(Θ_dense)  # 复制所有非 MoE 权重

    for layer_idx in moe_layer_indices:  # 每隔一层替换 FFN
        W_ffn = Θ_dense[layer_idx].ffn    # 原始 FFN 权重

        # Step 1: 复制 FFN N 次初始化每个 expert
        for i in range(N):
            Θ_moe[layer_idx].expert[i] = deep_copy(W_ffn)

        # Step 2: 随机初始化 router
        Θ_moe[layer_idx].router.W_g = random_init()      # gating weights
        Θ_moe[layer_idx].router.W_noise = random_init()   # noise weights

    return Θ_moe
```

Upcycling 后继续训练的 MoE 前向传播（Mixtral-type router）：

```
# Router: KeepTopK → Softmax (确保初始输出 = dense 输出)
H(x) = x @ W_g + StandardNormal() * Softplus(x @ W_noise)
G(x) = Softmax(KeepTopK(H(x), k=K))

# Expert FFN (SiLU-gated):
for token x routed to experts (i1, i2):
    gate_i1 = G(x)[i1], gate_i2 = G(x)[i2]
    y = gate_i1 * E_i1(x) + gate_i2 * E_i2(x)

# Expert Capacity 硬约束:
capacity = (tokens_per_batch / N) * CF
# 溢出 token 跳过该 MoE 层，直接传递 residual
```

Router 类型选择：Mixtral-type (KeepTopK→Softmax) 优于 ST-type (Softmax→KeepTopK)，因为 upcycling 后初始前向输出与 dense 模型完全一致，训练初始 loss 更低、收敛更快。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现方式（基于 NVIDIA NeMo + Megatron-Core）：
1. Online Upcycling：按并行训练配置（TP/EP/PP/DP）分片 dense checkpoint，各设备独立完成权重复制和 router 初始化，无需跨设备通信
2. 训练配置：CF=4, EP=8, TP=2, PP=4, VPP=8, DP with ZeRO-1, bfloat16 精度
3. 学习率调度：初始 LR=3e-5，余弦退火至 3e-7，100 warmup steps
4. 仅需 100B tokens 训练（<1% 预训练 compute），512 H100 GPU 上消耗 11K GPU hours
5. 适用场景：已有预训练 dense checkpoint，希望在有限 compute budget 下提升模型性能（如 MMLU 0-shot +2%）

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling

## Noisy Top-k Gating

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Noisy Top-k Gating 是 MoE 模型中 router（门控网络）的核心机制，通过注入可学习的 Gaussian 噪声到 router logits 中，再从 N 个 expert 中选 Top-K 个进行激活。数学形式（Shazeer et al., ICLR 2017）：

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$

其中 W_g 是 gating 权重矩阵，W_noise 是可训练的噪声权重矩阵。StandardNormal() 注入随机性，Softplus 保证噪声非负。噪声的作用：(1) 鼓励探索——防止 token 始终走相同路径；(2) 辅助负载均衡——打破"富者愈富"的正反馈循环；(3) 防止 expert collapse——少数 expert 支配训练而其余退化。

KeepTopK(v, k)_i = v_i if v_i in top-k else -∞。Softmax 将 -∞ 映射到 0，实现真正的稀疏激活——仅 K 个 expert 有非零权重，其余完全不被计算。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Noisy Top-k Gating 在 MoE 层 forward 中的流程
def moe_router_forward(x, W_g, W_noise, N, K):
    # Step 1: 计算 clean logits
    clean_logits = x @ W_g  # [B*S, N]

    # Step 2: 注入可学习噪声
    noise_std = softplus(x @ W_noise)  # [B*S, N]，保证非负
    noise = torch.randn_like(clean_logits) * noise_std
    noisy_logits = clean_logits + noise

    # Step 3: KeepTopK → 仅保留 Top-K 的 logits，其余设为 -inf
    topk_vals, topk_indices = torch.topk(noisy_logits, K, dim=-1)
    mask = torch.full_like(noisy_logits, float('-inf'))
    mask.scatter_(-1, topk_indices, topk_vals)

    # Step 4: Softmax （在 Mixtral-type 中 KeepTopK 在前）
    gate_weights = F.softmax(mask, dim=-1)  # 仅 K 个非零

    return gate_weights, topk_indices
```

Load balancing 辅助损失（GShard 风格）：
$$L_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 f_i = fraction of tokens dispatched to expert i, P_i = average gating probability for expert i, α 为损失系数。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

现代框架实现（Megatron-Core）支持：
- switch_load_balancing_loss_func：GShard 风格辅助损失
- Sinkhorn routing：基于最优传输的负载均衡
- Group-limited top-k routing：限制 expert 选择范围在 device/node 子集内（减少 All-to-All 通信）
- Mixtral-type (KeepTopK→Softmax) 和 ST-type (Softmax→KeepTopK) 两种路由顺序。Upcycling 场景下 Mixtral-type 更优，因为初始输出与 dense 模型一致

涉及论文标题：
- Llama 3 Meets MoE: Efficient Upcycling

## SIMBAL (Similarity-Preserving Router for MoE Load Balancing)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

SIMBAL（SIMilarity-preserving routers for MoE load BALancing）是一种新的 MoE 负载均衡方法，通过鼓励 Router 权重矩阵 R ∈ R^{D_M×E} 保持 token 间成对相似性来替代传统的 uniform-distribution-based 负载均衡损失。核心思想：如果 Router 矩阵 R 是正交的（R^T R = I），则 Router 的前向映射 x → xR 保留输入 token 间的点积（即角度/相似性）：(x1 R)·(x2 R) ≈ x1·x2。这意味着相似 token 会获得相似的 expert 分布，不同 token 会获得不同分布，从而自然地实现负载均衡（而非强制均匀分布）。

SIMBAL 的辅助损失：L_orth = ||R^T R - I_E||_1（L1 norm of Gram matrix deviation from identity）。该方法属于 loss-based soft constraint，而非显式正交参数化（如 QR 分解）。优势：(1) 不需要 float32→bfloat16 的精度转换，(2) 不需要昂贵的重正交化步骤，(3) 与标准 AdamW 训练兼容。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

SIMBAL 在 MoE Transformer 训练中的完整流程：

```python
# ===== 每 training step =====

# 1. Standard MoE forward pass (Token Choice, top-A routing)
x = attention_output                                # [B, S, D_M]
router_logits = x @ R                                # [B, S, E], R ∈ R^{D_M×E}
router_probs = softmax(router_logits, dim=-1)        # [B, S, E]
topk_vals, topk_ids = topk(router_probs, k=A)        # [B, S, A]

# 2. Expert computation (unchanged from standard MoE)
output = compute_moe_experts(x, topk_vals, topk_ids)  # [B, S, D_M]

# 3. SIMBAL auxiliary loss computation
# Only requires the router weight matrix R
w = R.weight                                          # [E, D_M] (或 [D_M, E])
gram = w @ w.T                                        # [E, E], Gram matrix
L_orth = ||gram - I_E||_1                             # L1 deviation from identity

# 4. Combined loss
L_total = L_lm + lambda_simbal * L_orth  # lambda_simbal typically 0.1 (insensitive)

# 5. Backward: L_orth gradient pulls R toward orthogonality
```

关键特性：
- L_orth 仅依赖 Router 权重，与数据分布无关 → 对 batch size 不敏感
- lambda_simbal 不敏感 (0.01/0.1/1.0 下 perplexity 差异 < 0.03)
- 配合正交初始化 (Saxe et al. 2014) 加速收敛；也可以仅执行少量 router-only SGD steps
- 比显式 QR 分解参数化更高效：后者需要 float32 计算 + 每次迭代重正交化

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- Router 维度特征：D_M >> E（如 D_M=1536, E=32）→ R 是 tall matrix → Gram matrix E×E 极小 → L_orth 计算开销可忽略
- 与 LBL 的对比：LBL 计算每个 expert 的 f_i·P_i 聚合统计量（需要 batch token routing info）→ SIMBAL 只需要 Router weight matrix
- 在 OLMo 开源代码库 (https://github.com/allenai/OLMo) 上实现，loss 通过 AddAuxiliaryLoss autograd trick 或直接相加集成
- 效果：比 LBL 快 36% 收敛（相同 loss 所需 token），expert 冗余（PES）降低 5-8x
- 与推理时 expert pruning 的协同：SIMBAL 产生 less uniform routing → 低 weight expert 更可安全丢弃 → 7.4% throughput improvement

涉及论文标题：
- Load Balancing Mixture of Experts with Similarity Preserving Routers

## Router Orthogonalization in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router 正交化是指对 MoE 模型的 Router 权重矩阵施加正交约束（R^T R ≈ I），使 Router 的前向映射 x → xR 成为近似等距变换（保留 token 间的成对角度/点积）。在 MoE 中，这意味着相似 token 被 Router 映射后仍保持相似的 routing scores，从而获得相似的 expert 分配。正交化可以通过两种方式实现：

1. **显式参数化（Parametric）**：使用 QR 分解或 Cayley 变换将 Router 权重约束在 Stiefel 流形上。PyTorch 提供 `torch.nn.utils.parametrizations.orthogonal`。缺陷：需要 float32 计算，bfloat16 下数值不稳定；频繁重正交化开销大；不适用于大规模训练。

2. **损失-based 软约束（Loss-based）**：SIMBAL 采用的方法——将 L_orth = ||R^T R - I||_1 作为辅助损失项，通过梯度下降软性地将 R 推向正交。可在 bfloat16 中直接训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

正交 Router 如何保持 token 间关系：

给定两个 token: x1, x2 ∈ R^{D_M}
Router 输出: s1 = x1·R, s2 = x2·R ∈ R^E

若 R 是正交矩阵 (R^T R = I):
  s1·s2 = (x1·R)·(x2·R) = x1·(R·R^T)·x2 ≈ x1·x2
  
因此 cos(x1, x2) ≈ cos(s1, s2) → 相似 token 获得相似 routing scores

SIMBAL 论文 Table 2 验证：在 1536×32 的 Router 上 100 步优化后，Gram matrix 与 I 的 L1 distance 为 ~1×10^-5（vs QR 参数化的 ~2×10^-4），即 loss-based 方法比显式参数化更接近正交。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **初始化**：使用 Saxe et al. 2014 正交初始化（对权重矩阵做 SVD 后替换奇异值为 1），或仅训练 Router 数步即可达到接近正交
- **loss coefficient**：SIMBAL 论文 lambda=0.1，但 0.01-1.0 均有效
- **训练稳定性**：正交 Router 对输入扰动更鲁棒（角度保持），训练早期不会出现频繁的 routing shift
- **与前人工作的区别**：OMoE (Liu et al. 2024) 正交化 Expert 权重（在 optimizer 中更新方向正交），MOORE (Hendawy et al. 2024) 正交化 Expert 表示（Gram-Schmidt）。SIMBAL 是首个对 Router 做正交化的方法——Router 参数极少（<0.02% total params）但编排 billions of parameters

涉及论文标题：
- Load Balancing Mixture of Experts with Similarity Preserving Routers

## Pairwise Expert Similarity (PES)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Pairwise Expert Similarity (PES) 是 SIMBAL 论文提出的轻量级 expert 冗余度量指标。定义为所有 expert 输出之间成对余弦相似度的 batch 平均值：

C_expert(x) = (2/(N(N-1))) * Σ_i Σ_{j>i} cos(f_i(x), f_j(x))

PES = (1/|B|) Σ_{x∈B} C_expert(x)

其中 N 为 expert 数，f_i(x) 为 expert i 对输入 x 的输出向量，cos(u,v) = u·v/(||u||·||v||)。PES 越低表示 expert 输出越多样化（越不冗余），expert 专精化程度越高。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

PES 计算流程：

for each batch sample x:
    outputs = []  # compute all N expert outputs
    for expert_id in range(N):
        h = silu(x @ W_gate[i]) * (x @ W_up[i])
        out = h @ W_down[i]
        outputs.append(out)
    similarities = []
    for i in range(N):
        for j in range(i+1, N):
            sim = cosine_similarity(outputs[i], outputs[j])
            similarities.append(sim)
    C_expert = mean(similarities)
PES = mean(C_expert over batch)

计算开销：需要在每个 expert 上做一次 forward pass → FLOPs 约为 full model 的 3.6-4.9x（但仅对验证集子集做一次），远少于 dropout-based 评估需要数百次 full model 验证。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **vs Dropout-based 评估**（Dai et al. 2024 的 expert dropout 方法）：dropout 方法需对每种 expert 组合做 full validation → 计算量巨大且缺乏粒度。PES 只需一次 full expert inference → 可做 per-layer、per-checkpoint 的细粒度分析
- **使用方式**：训练过程中定期计算 PES on validation set（4M tokens），监控 expert 冗余度变化
- **SIMBAL 结果**：MoE-L 上 SIMBAL min PES = 0.0028 vs LBL 0.0241（约 8.6x 更低），表明 SIMBAL 训练的 expert 专精化程度显著更高
- **局限性**：需对所有 expert 做 forward → 仅适用于离线分析，不能作为在线训练指标；对个别 layer 的 outlier spikes 敏感

涉及论文标题：
- Load Balancing Mixture of Experts with Similarity Preserving Routers

## Token-to-Expert Routing (TER) in Vision-Language MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token-to-Expert Routing (TER) 是 MoE 架构中 Router（门控网络）决定每个输入 token 分配给哪个/哪些 expert 处理的核心机制。Router 是一个小型可学习线性层 `W_g ∈ R^{D × K}`（D = hidden dim, K = num_experts），对每个输入 token x 计算 `logits = x @ W_g`，经 Softmax 归一化得到 routing probabilities `P(x) ∈ R^K`，Top-k 选择后对被选中 expert 输出加权求和：`MoE(x) = Σ P(x)_i * Expert_i(x)`。

在 LVLM 场景下，TER 同时处理两类模态 token：CLIP visual encoder 提取的 vision tokens（~576 per image）和 language tokens（输入 text sequence）。传统 TER 对所有 token 统一施加 load balancing 约束 `L_balancing = K * Σ F_i * G_i` 以鼓励均匀路由分配，但 LTDR 发现 vision tokens 服从 long-tailed distribution（少量高信息 foreground + 大量低信息 background），load balancing 将 critical foreground tokens 打散到不同 expert，阻碍 expert 专业化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**统一 TER Pipeline（Baseline MoE-LLaVA）**：
```
# Vision + Language token concatenation
x = [v_1,...,v_M, t_1,...,t_N]  # (M+N)×D

# Router forward: all tokens share same W_g
logits = x @ W_g                  # (M+N)×K
probs = Softmax(logits)           # (M+N)×K
selected = TopK(probs, k=2)       # select k=2 experts per token
norm_probs = probs[selected] / sum(probs[selected])

# Load balancing: uniform constraint on ALL tokens
for i in 1..K:
    F_i = fraction of (M+N) tokens routed to expert i
    G_i = mean(probs[:, i])
L_bal = K * Σ F_i * G_i  # applied to vision + language equally

# Expert computation
output = Σ norm_probs[j] * ExpertFFN_j(x)
```

**LTDR TER Pipeline（Modal-aware + Long-tailed aware）**：
```
# Step 1: Router forward (same as baseline)
logits_v = V @ W_g  # M×K, vision
logits_t = T @ W_g  # N×K, language

# Step 2: MsDaR - Language-only load balancing
for i in 1..K:
    F_i = count(T routed to i) / N    # only language tokens
    G_i = mean(softmax(logits_t)[:, i])
L_bal = K * Σ F_i * G_i  # vision tokens excluded!

# Step 3: VsDEA - Vision tail token identification
RPV = Variance(softmax(logits_v), dim=1)  # per-vision-token RPV
threshold = Mean(RPV)                       # dynamic threshold
is_tail = RPV > threshold                   # ~13% of vision tokens

# Step 4: Differentiated expert activation
for each token x:
    if x is vision tail token:
        # activate a=4 (LTDR) > k=2 (baseline) experts
        indices = TopK(probs(x), a=4)
    else:
        indices = TopK(probs(x), k=2)
    weights = Softmax(logits(x)[indices])  # renormalize
    output = Σ weights[j] * ExpertFFN_j(x)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **标准实现**：`nn.Linear(D, K)` + Softmax + TopK。Mixtral-8x7B 用 Top-2 gating，DeepSeek-V3 用 Sigmoid 替代 Softmax 扩展至 256 experts
- **LVLM MoE 实现**：MoE-LLaVA 将指定层 FFN 替换为 MoE layer（每 2 个 Transformer block 中 1 个），4 experts Top-2；Molmo 使用 64 experts Top-8
- **LTDR 改动**：对现有 MoE 框架改动极小——仅修改 `L_balancing` 计算范围（排除 vision tokens）和 tail token 的 TopK 参数（k→a）。与 HuggingFace Transformers + PyTorch 完全兼容，不需额外框架支持
- **训练配置**：epoch=1, LR=2e-5 cosine, weight decay=0, batch size=16/GPU, FP16, L_balancing coefficient=0.01, a=4 (MoE-LLaVA) / a=12 (Molmo)

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

## Load Balancing Loss in Mixture-of-Experts

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load Balancing Loss（负载均衡损失）是 MoE 训练中防止 expert 负载不均衡的辅助损失函数。由于 Router 在训练中可能收敛到总是选择少数几个 expert（导致其他 expert 不被训练而导致 collapse），Load Balancing Loss 约束每个 expert 处理大致等量的 tokens。标准形式为 `L_balancing = K * Σ_{i=1..K} F_i * G_i`，其中 F_i 是分配给 expert i 的 token 比例，G_i 是 expert i 的平均 routing probability。该损失通过鼓励均匀的 token 分配来避免 expert 过载/空闲问题。

LTDR 论文揭示了 Load Balancing Loss 在 multi-modal（vision-language）场景中的关键缺陷：vision tokens 服从 long-tailed distribution（大量低信息 background + 少量高信息 foreground），load balancing 会将 sparse 但关键的 foreground (tail) tokens 均匀分散到不同 expert，阻止 expert 对视觉关键信息进行专业化学习。实验数据显示移除 vision TER 的 load balancing 直接提升性能。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Standard Load Balancing (Switch Transformer, GShard)
# Applied to ALL tokens uniformly
for batch in training_data:
    # ... router forward ...
    for i in 1..K:  # for each expert
        F_i = count(tokens routed to expert i) / total_tokens
        G_i = mean(softmax(logits)[:, i])
    L_balancing = K * Σ_{i=1..K} F_i * G_i
    L_total = L_task + α * L_balancing  # α = coefficient (0.01)

# LTDR Modality-specific Load Balancing
# Only applied to language tokens
for batch in training_data:
    # ... router forward ...
    for i in 1..K:
        F_i = count(language_tokens routed to i) / N_lang
        G_i = mean(softmax(lang_logits)[:, i])
    L_balancing = K * Σ F_i(T) * G_i(T)  # vision tokens excluded!
    L_total = L_task + α * L_balancing
```

**消融实验验证**：
- Vision load balancing coefficient: 0.01 (standard) vs 0.001 (reduced) vs 0 (LTDR, removed)
- 结果：reduced (0.001) 不如 complete removal (0)，因为即使系数缩小 10x 仍然对 vision tail tokens 的 distribution 产生约束
- Strategy-swap 实验：将 MsDaR（移除 load balancing）分别应用于 vision 和 language 侧。Language+MsDaR 导致性能波动（因为语言确实服从 uniform distribution），Vision+MsDaR 稳定提升

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现方式**：在训练 loss 中作为 additive auxiliary loss，系数通常设为 0.01（Switch Transformer）。PyTorch 中作为额外 loss term 加到 `L_total.backward()`
- **使用方式**：MoE-LLaVA / GShard / Switch Transformer 均使用此机制。LTDR 的改动是在计算 F_i 和 G_i 时 filter 掉 vision tokens（仅在 token dispatch 时标记 modality type）
- **作用范围**：影响 expert 的 token 分配分布，从而影响 expert 专业化程度。对 language 有效（uniform distribution 适配），对 vision 有害（long-tailed distribution 不适用）

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

## Routing Probability Variance (RPV)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Routing Probability Variance (RPV) 是 LTDR 论文提出的衡量每个 token 的 routing distribution 离散度的指标。定义：对于 token x 的 routing probabilities `P(x) ∈ R^K`（softmax 归一化后的 K 维向量），`RPV(x) = Variance(P(x)) = (1/K) * Σ_i (P(x)_i - μ)²`，其中 μ = (1/K) * Σ_i P(x)_i。RPV 反映 Router 对 token 的"路由置信度"：低 RPV 表示 token 被均匀分配给各 expert（router 不确定该由谁处理），高 RPV 表示 token 集中分配给少数 expert（router 有明确偏好）。

LTDR 利用 RPV 实现了两个功能：(1) 通过 RPV distribution 分析 vision token 的 long-tailed 特性；(2) 用 Mean(RPV) 作为动态阈值区分 vision head tokens（低 RPV）和 tail tokens（高 RPV）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# RPV 计算 (per vision token)
def compute_RPV(probs_v):  # probs_v: [M, K]
    mu = mean(probs_v, dim=1)        # [M], mean routing prob
    var = mean((probs_v - mu)^2, dim=1)  # [M], variance per token
    return var  # [M] = RPV for each vision token

# Tail/Head Classification via Mean RPV
RPV_v = compute_RPV(softmax(V @ W_g))  # [M]
threshold = Mean(RPV_v)                 # scalar, dynamic
is_tail = RPV_v > threshold             # ~13% of vision tokens
is_head = RPV_v <= threshold            # ~87% of vision tokens

# RPV-L2 Norm Analysis (验证 RPV 与信息量的关联)
# Top-13% RPV tokens:  mean L2 norm = 0.3158
# Top-13%-26% tokens:   mean L2 norm = 0.2124  
# Top-26%-39% tokens:   mean L2 norm = 0.1475
# → Higher RPV correlates with richer vector representations
```

**RPV 分析结果**：
- Language tokens: RPV 接近均匀分布（load balancing 适配）
- Vision tokens without load balancing: long-tailed RPV distribution
- Vision tokens with load balancing: 高 RPV token 数量被抑制（biased long-tailed）
- LTDR 显著提升 vision tail tokens 的 mean RPV，head tokens 不受影响

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：在每次前向传播中计算 softmax(logits) 后直接求 per-token variance，计算开销极小（O(M*K) per MoE layer）
- **用途 1 — Tail token 识别**：用 Mean(RPV) 作为动态阈值，避免固定比例阈值（fixed 10%/15%/20%）对不同数据分布不鲁棒。实验表明 adaptive mean-RPV 优于所有 fixed-ratio 方案
- **用途 2 — 分布分析**：通过 RPV 分布可视化验证 TER 策略效果（training steps evolution、cross-router comparison）
- **替代方案对比**：VsDEA 中用 Instruction-Aware Tokens (IATs, attention-based selection) 替代 RPV-based selection → 效果较差（跨模态 attention 噪声干扰）

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

## Long-Tailed Distribution in Vision MoE Token Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Long-Tailed Distribution in Vision MoE Token Routing 是 LTDR 论文揭示的 LVLM MoE 中 vision token 路由分布的核心特性。Vision tokens（CLIP 编码器输出，~576/image）天然包含大量低信息背景 patches（head, ~87%）和少量高信息前景 patches（tail, ~13%），导致 vision TER 的 routing 呈现 long-tailed 分布。具体表现：大部分 vision tokens 的 RPV 较低（router 不确定分配，token 含信息量低 → 类似 long-tail 分类问题中的 head classes），少数 vision tokens 的 RPV 较高（router 明确偏好某些 expert，token 含信息量高 → 类似 tail classes）。这与 language tokens 的近似 uniform TER 分布形成对比。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**Vision Token Long-Tailed 特性分析流程**：
```
# Step 1: 计算 vision token RPV distribution
for each image in batch:
    V = CLIP_encoder(image)  # [576, D] vision tokens
    probs = Softmax(V @ W_g)  # [576, K] routing probabilities
    rpv = Variance(probs, dim=1)  # [576] per-token RPV

    # Step 2: 绘制 RPV 直方图 (Fig. 1(b) in paper)
    # x-axis: RPV range (0.00-0.01, 0.01-0.02, ...)
    # y-axis: token count in each bin
    # Result: long-tailed shape
    #   - 442 tokens in 0.00-0.01 bin (head)
    #   - rapidly decreasing counts in higher RPV bins (tail)

# Step 3: Load balancing impact analysis
# with L_balancing:  高 RPV token 数量减少 → expert specialization 受阻
# without L_balancing: 保留 long-tailed shape → tail tokens 获得专业 expert

# Step 4: Modality comparison
# Language RPV distribution: near-uniform (各 RPV 区间 count 接近)
# Vision RPV distribution: long-tailed (集中低 RPV, 长尾高 RPV)
```

**关键发现**：
- Vision 的 long-tailed 特性来自于视觉内容的固有结构——大多数图像区域是背景/纹理（低信息量），少数是目标/文本/关键细节（高信息量）
- Load balancing 的正则化效果类似 undersampling tail tokens → 使 critical foreground tokens 被分散到各 expert，降低 expert 对视觉关键信息的特化能力
- GMoE 实验：移除 load balancing 后性能提升，验证了 load balancing 对 vision TER 的负面影响

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **识别方法**：在 MoE 训练的每个 batch 中计算 per-token RPV，绘制分布直方图确认 long-tailed 特性
- **应对策略**：
  1. 移除 vision TER load balancing（MsDaR）→ 保持 long-tailed 天然分布
  2. 增强 tail token expert activation（VsDEA）→ 补偿 tail token 的稀少性
- **与其他 long-tailed 研究的关系**：传统 long-tailed classification（RIDE, BBN, LDAM）主要处理 sample-level class imbalance，LTDR 是首次在 token-level TER distribution 层面处理 long-tailed 问题

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

## Modality-specific Distribution-aware Router (MsDaR)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Modality-specific Distribution-aware Router (MsDaR) 是 LTDR 的第一个核心模块。基于 vision tokens 服从 long-tailed distribution、language tokens 服从 uniform distribution 的观察，MsDaR 修改了 MoE 的 load balancing 策略：保留 language TER 的 load balancing（适配其 uniform distribution），移除 vision TER 的 load balancing（让 vision tokens 按天然 long-tailed 分布路由到专业化 expert）。具体实现是将 L_balancing 公式中的 F_i 和 G_i 计算限定为：`L_balancing = K * Σ F_i(T) * G_i(T)`，仅对 language tokens T 计算，vision tokens 完全不受负载均衡约束。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MsDaR: Modality-specific Distribution-aware Router
def moe_layer_forward(V, T, W_g, experts):
    # V: [M, D] vision tokens, T: [N, D] language tokens
    x = concat([V, T], dim=0)  # [(M+N), D]

    # Shared router forward
    logits = x @ W_g            # [(M+N), K]
    probs = Softmax(logits)     # [(M+N), K]

    # MsDaR: Language-only load balancing
    probs_t = probs[M:, :]      # [N, K], language portion
    F_t = argmax(probs_t, dim=1)  # which tokens go where
    for i in 1..K:
        F_i = sum(F_t == i) / N    # fraction of language tokens to expert i
        G_i = mean(probs_t[:, i])  # mean routing prob to expert i
    L_balancing = K * Σ F_i * G_i  # ONLY language tokens

    # Vision tokens: NO load balancing constraint
    # → naturally follow long-tailed distribution
    # → tail tokens get higher RPV → route to specialized experts

    return moe_output, L_balancing
```

**与 Modality-aware MoE 的区别**：
- Modality-aware MoE（MoMa, Eve）：将 experts 硬性划分为 vision group 和 language group → 损失模型容量
- MsDaR：保持所有 experts 共享，仅通过 routing 策略实现模态差异 → 更灵活、不损失 expert pool 容量
- 实验验证：MoE-LLaVA-v2Top1-t2Top1 (modality-split experts) 性能 57.7 vs baseline 57.6（几乎无提升）；添加 MsDaR 后提升至 58.2

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现复杂度**：极低。只需在 L_balancing 计算前添加 modality mask（根据 token 在 concatenation 中的位置区分 V 和 T）
- **兼容性**：与任何基于 load balancing 的 MoE 训练框架兼容（HuggingFace, Megatron, DeepSpeed-MoE）
- **训练 overhead**：计算 F_i 和 G_i 时少处理 vision tokens → 略微减少计算量
- **独立贡献**：MsDaR alone 提供 0.6%（StableLM-1.6B）和 0.5%（Phi2-2.7B）的平均提升

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

## Vision-specific Dynamic Expert Activation (VsDEA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Vision-specific Dynamic Expert Activation (VsDEA) 是 LTDR 的第二个核心模块。基于 RPV 将 vision tokens 分类为 head tokens（RPV ≤ mean RPV, ~87%）和 tail tokens（RPV > mean RPV, ~13%），对 tail tokens 激活更多 experts（Top-a, a > k），采用 renormalized softmax 权重。本质是一种 data-augmentation 策略：tail tokens 数量少但信息密度高，通过让更多 experts 联合处理来降低错误路由的影响，确保 tail tokens 得到充分学习。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# VsDEA: Vision-specific Dynamic Expert Activation
def vsdea_expert_activation(x, logits, is_vision_tail):
    probs = Softmax(logits)  # [K]

    if is_vision_tail:
        # Tail token: activate a=4 experts (k=2 baseline)
        k_active = a  # a = 4 (MoE-LLaVA) / 12 (Molmo)
    else:
        # Head token or language token: normal activation
        k_active = k  # k = 2 (MoE-LLaVA) / 8 (Molmo)

    indices = TopK(probs, k_active)
    # Renormalize weights over selected experts
    weights = Softmax(logits[indices])
    output = Σ_{j=1..k_active} weights[j] * ExpertFFN_j(x)
    return output

# Tail token identification (before VsDEA)
RPV_v = Variance(Softmax(V @ W_g), dim=1)  # [M]
threshold = Mean(RPV_v)                       # dynamic
is_tail = RPV_v > threshold                   # boolean [M]
```

**Tail Token Selection 消融对比**：
| Selection Method | % Tokens Selected | Avg |
|---|---|---|
| VHTs (Vision Head Tokens, low RPV) | ~87% | 58.0 |
| IATs (Instruction-Aware Tokens, attention-based) | 15% (fixed) | 57.7 |
| VTTs 10% (fixed threshold) | 10% | 57.9 |
| VTTs 15% (fixed threshold) | 15% | 57.9 |
| VTTs 20% (fixed threshold) | 20% | 57.4 |
| VTTs mean-RPV (LTDR, adaptive) | ~13% | **58.8** |

- Selecting head tokens (VHTs) 也有提升但远不如 tail tokens → 验证了 tail tokens 的高信息密度
- Fixed ratio thresholds 不如 adaptive mean-RPV → 动态阈值对不同数据分布更鲁棒
- IATs 效果最差 → 跨模态 attention 的噪声干扰 vision token 重要性判定

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- **实现**：在 MoE layer 的 TopK 选择后添加 conditional branching（if is_tail: k→a）
- **Inference overhead**：几乎为零。All-to-All 通信速度由最慢 expert 决定（all-to-all barrier），VsDEA 不显著增加最慢 expert 负载。实测 V100 avg 1100s vs 1108s，A800 avg 846s vs 917s
- **Memory**：9.44G vs 9.44G baseline（V100-30G）→ 无额外内存开销
- **GPU Utilization**：59.29% vs 59.57% baseline → 几乎无变化
- **a 值选择**：MoE-LLaVA 上 a=4（vs k=2），Molmo 上 a=12（vs k=8）→ 经验性翻倍

涉及论文标题：
- Long-Tailed Distribution-Aware Router For Mixture-of-Experts in Large Vision-Language Model

## Zero-Computation Experts (零计算专家)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Zero-Computation Experts 是 LongCat-Flash / MoE++ 提出的一种动态计算资源分配机制。在标准 MoE 的 FFN expert pool 中额外引入 Z 个"零计算专家"（Zero-Computation Experts），其输出定义为恒等映射 `E_i(x_t) = x_t`（输入直接通过，不经过 FFN 计算）。Router 从 N+Z 个 experts 中选 top-K 个，如果 token 选中 zero-computation expert，则该 expert 不引入额外 FLOPs。这样，模型可根据 token 的上下文重要性自适应分配计算资源：简单 token（如标点、虚词、冠词）更多激活 zero-computation experts 节省计算，困难 token（如语义关键词）更多激活 FFN experts 获取更强表达能力。

公式：$$\begin{aligned} \operatorname{MoE}(x_t) &= \sum_{i=1}^{N+Z} g_i E_i(x_t), \\ E_i(x_t) &= \begin{cases} \operatorname{FFN}_i(x_t), & 1 \leq i \leq N \\ x_t, & N < i \leq N+Z \end{cases} \end{aligned}$$ 其中 $g_i$ 为 router 输出（softmax + top-K selection），K 为每 token 选中的总 expert 数，实际激活的 FFN expert 数在 [0, K] 之间动态变化。

LongCat-Flash 配置：N=512 FFN experts, Z=256 zero-computation experts, K=12, 期望 FFN 激活数 $K_e=8$，实际激活参数范围 18.6B-31.3B（平均 27B）。训练后观察：平均 FFN 激活数在 20B tokens 后收敛到 ~8（波动 <1%），而标准差持续在 ~3，表明 token 间计算资源分配存在显著差异。浅层（Layer 1）中 function words、标点、数字持续获得较少计算资源；深层（Layer 28）资源分配更动态。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Zero-Computation Experts Forward Pass (per token, per MoE layer)

输入: x_t [d_model]  # MoE layer input
参数: FFN_experts = [FFN_0, ..., FFN_{N-1}]  # N 个标准 FFN experts
      zero_experts = [Identity, ..., Identity]  # Z 个零计算 experts (恒等映射)
      router: Linear(d_model, N+Z)
      expert_bias: [N+Z]  # PID-controlled bias

# Step 1: Router 计算
logits = router(x_t)  # [N+Z]
probs = softmax(logits + expert_bias)  # 加上 expert bias 后 softmax

# Step 2: Top-K 选择
scores, indices = topk(probs, k=K)  # 从 N+Z 个中选 K 个

# Step 3: Expert 聚合
output = zeros_like(x_t)
for g, idx in zip(scores, indices):
    if idx < N:
        output += g * FFN_experts[idx](x_t)  # FFN expert: 实际 FLOPs
    else:
        output += g * x_t  # Zero-comp expert: 零 FLOPs (identity)

# Step 4: PID Bias Update (训练时, 仅更新 FFN experts)
# Δb_i = μ * (K_e/K * 1/N - T_i/(K*T_all))  for i in [0, N-1]
# b_i = b_i + Δb_i

输出: output [d_model]
```

Forward pass 的计算复杂度由实际激活的 FFN expert 数量决定，而非固定的 K。Training loss 实验显示：动态激活 4.2B-7.0B 参数（平均 6B）的 zero-expert 变体，validation loss 持续低于固定 top-k=8（固定 6B）的 baseline（Figure 3a）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **Zero-comp expert 不参与任何矩阵乘法**：在 MoE GEMM kernel 中直接跳过，返回输入引用。在 Distributed EP 场景下，zero-comp expert 的 dispatch/combine 通信也可省略（DeepEP 修改支持）。
2. **PID Controller 控制平均负载**：仅对 FFN experts 更新 bias（Eq. 2），zero-comp experts 的 bias 固定为零。PID 确保 $K_e/F$ 比例的 token 流向 FFN experts。在 LongCat-Flash 中，$K_e=8, F=K=12$，意味着平均 8/12=66.7% 的选中 expert 为 FFN expert。
3. **Load Balance 适应**：在 device-level load balance loss 中，将所有 zero-comp experts 划入一个单独的 group (D+1)，其 coefficient 保证 loss 收敛时 FFN:zero 比例为 $K_e:(K-K_e)$。
4. **Kernel 集成**：MoE permute/unpermute kernels 需集成 zero-comp expert 处理逻辑——识别选中 zero-comp expert 的 token，对这些 token 跳过通信/expert GEMM，直接累加 gating weight × input。

与相关工作对比：AdaMoE [Zeng et al., 2024] 提出 "null experts"，与 MoE++ [Jin et al., 2024] 同期独立提出类似概念。LongCat-Flash 在此基础上增加了 PID 控制器的精细计算预算控制和 EP-aware load balance。

涉及论文标题：
- LongCat-Flash Technical Report
- MoE++: Accelerating Mixture-of-Experts Methods with Zero-Computation Experts
- AdaMoE: Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

## Shortcut-Connected MoE (ScMoE)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Shortcut-Connected MoE (ScMoE) 是 LongCat-Flash / [Cai et al., 2024] 提出的 MoE 架构创新。核心思想：在 Transformer layer 中引入跨层 shortcut 连接——将同一层第一个 Multi-head Latent Attention (MLA) block 的输出直接连接到该层的 MoE block，使前一层的 Dense FFN 计算可以与当前层 MoE 的 dispatch/combine 通信并行执行。

传统 MoE execution paradigm（如 DeepSeek-V3 的 interleaved MoE+Dense FFN）中，Expert Parallelism 要求先完成 all-to-all 通信（token dispatch）才能开始 expert 计算，通信延迟成为串行瓶颈。Shared-expert 架构尝试用单个 expert 的计算时间与通信重叠，但重叠窗口受限于单个 expert 的计算量。ScMoE 将 Dense FFN 从 MoE 之后移到 MoE 之前（通过 shortcut 连接），利用 Dense FFN 较大的 intermediate size（12288 vs expert 2048）创造更大的 computation-communication overlap 窗口。

LongCat-Flash 验证了 ScMoE 在四种模型配置下（2.4B-16B MLA, 3B-20B MHA, 15B-193B GQA）training loss 与 baseline 几乎相同（Figure 4），证明 ScMoE 是 quality-neutral 的架构优化。进一步提升：将 MoE layer 沿 token 维度分为两个 chunk，实现 chunk 间互相重叠 + 与 Dense FFN 重叠。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ScMoE layer 结构（单层包含 2 个 MLA + Dense FFN + MoE）：

```
# ScMoE Layer Forward Pass (per token batch)

输入: hidden_states [batch, seq_len, d_model]

# Stage 1: 第一个 MLA (独立执行)
h1 = MLA_0(hidden_states)  # 产生 attention output + KV cache

# Stage 2: Dense FFN (可与当前层 MoE 通信并行)
dense_out = DenseFFN(h1_chunk_a)  # chunk_a 的 dense FFN
MLA_0_qkv = QKV_Projection(h1_chunk_a)  # chunk_a 的 QKV 投影
# 同时: All-to-All Dispatch(h1_chunk_b → experts)  # chunk_b 的 token dispatch

# Stage 3: MoE GEMM (独立执行)
moe_out_b = MoE_GEMM(dispatched_chunk_b_tokens)  # chunk_b 的 expert 计算

# Stage 4: 第二个 MLA + Dense FFN + All-to-All Combine (并行)
attn_out_a = CoreAttention(MLA_0_qkv_a) + OutputProjection
dense_out_b = DenseFFN(h1_chunk_b)
# 同时: All-to-All Combine(moe_out_b → original GPUs)

输出: attn_out + dense_out + moe_out (残差累加)
```

ScMoE 的关键特征：(1) Shortcut 从 MLA_0 直连到 MoE block，使 Dense FFN 在 MoE 之前执行（而非之后），创造重叠窗口；(2) Token chunking 分两个 chunk 交替执行，chunk_a 的 dense FFN+attention 与 chunk_b 的 all-to-all 通信并行；(3) architecture 与 attention 机制（MLA/MHA/GQA）正交——Figure 4 证明 loss 曲线在三种 attention 下均几乎相同。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. **架构层面**：每层内 MLA_0 → Dense FFN + MoE 平行路径（shortcut 连接），替代传统 MLA_0 → MoE → Dense FFN 串行或 MoE ↔ Dense FFN interleaved。
2. **训练层面**：non-overlapping dispatch/combine 时间从 25.3% 降至 8.4%。ScMoE 与 expert parallelism group (EP=32) 和 V-ZB pipeline 协同使用。
3. **推理层面（SBO）**：ScMoE 是 Single Batch Overlap 的基础——Dense FFN 计算可与 all-to-all dispatch 重叠，Attention Core 可与 all-to-all combine 重叠。TPOT 理论值降低近 50%（vs DeepSeek-V3 TBO）。
4. **通信层面**：Dense FFN 的 intra-node NVLink 通信（TP 的 all-gather/reduce-scatter）可与 MoE 的 inter-node RDMA 通信（EP 的 all-to-all）通过 GPUDirect RDMA 并发执行。

涉及论文标题：
- LongCat-Flash Technical Report
- Shortcut-Connected Expert Parallelism for Accelerating Mixture-of-Experts

## PID Controller for Expert Bias (MoE 专家偏置 PID 控制器)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

PID Controller for Expert Bias 是 LongCat-Flash 在 aux-loss-free load balancing [Wang et al., 2024a] 基础上提出的改进。核心思想：使用比例-积分-微分 (PID) 控制器动态调整 expert bias $b_i$，使每个 FFN expert 的 token 分配精确收敛到目标比例，同时不干扰 LM 训练目标。相比原方案使用的固定 bias increment，PID 控制器提高了 softmax router 在大规模 expert 数量下的概率分布鲁棒性。

Bias 更新公式：$$\Delta b_i = \begin{cases} \mu \left( \frac{K_e}{K} \cdot \frac{1}{N} - \frac{T_i}{K T_{\text{all}}} \right), & 1 \le i \le N \\ 0, & N < i \le N + Z \end{cases}$$ 其中 $\mu$ 为 bias adaptation rate（decay schedule），$T_{\text{all}}$ 为 global batch 的 token 总数，$T_i$ 为路由到 expert i 的 token 数，$K_e$ 为期望的 FFN expert 激活数。关键设计：(1) Zero-computation experts 的 bias 固定为零（不参与更新），因为它们的 identity 性质只需要全局约束——当所有 FFN experts 达到目标比例时自动满足；(2) 大 batch size + decay schedule for $\mu$ 提高 budget control 的稳定性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# PID Expert Bias Update (每 training step)

参数:
  N: 512 FFN experts, Z: 256 zero-comp experts
  K: 12 (top-K), K_e: 8 (期望 FFN 激活数)
  mu: bias adaptation rate (根据 global batch size 和 schedule 衰减)
  b: [N+Z]  # expert bias, 初始化为 0

# 每个 step 执行:
T_all = 0  # global batch 总 token 数
T = [0] * (N+Z)  # 每个 expert 接收的 token 数

# 收集统计 (跨所有 EP group)
for each micro_batch:
    for each token x_t:
        probs = softmax(router(x_t) + b)  # 加上当前 bias
        topk_indices = topk(probs, k=K)
        for idx in topk_indices:
            T[idx] += 1
            T_all += 1

# 更新 bias (仅 FFN experts)
for i in range(N):
    target_ratio = (K_e / K) * (1.0 / N)  # 每 expert 目标占比
    actual_ratio = T[i] / (K * T_all)
    delta = mu * (target_ratio - actual_ratio)
    b[i] += delta

# zero-comp experts (N 到 N+Z-1): b 保持 0，不更新
```

LongCat-Flash 观察：大 batch size 和 μ 的 decay schedule 提高收敛稳定性；小 batch size 可能需要降低更新频率。经过约 20B tokens 训练后所有层平均 expert 数收敛至目标值（波动 <1%）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 原方案 [Wang et al., 2024a] 使用固定增量的 bias 更新，PID 改进在于用 error 比例驱动更新而非固定步长。
2. μ 使用 decay schedule（如 cosine decay），从较大初始值开始确保快速收敛，后期降低避免振荡。
3. 与 device-level load balance loss (Eq. 3-5) 互补：PID 控制 corpus-level 平均负载，load balance loss 防止 sequence-level 极端不均衡。
4. 依赖大 global batch size——LongCat-Flash 使用 tens of thousands of accelerators，global batch 足够大以保证统计稳定。

涉及论文标题：
- LongCat-Flash Technical Report

## Variance Alignment for MLA (Scale-Correction in Multi-head Latent Attention)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Variance Alignment for MLA 是 LongCat-Flash 针对 MLA 低秩分解中的方差不对齐问题提出的修复。问题根源：MLA 的 query 压缩维度 $d_q$、KV 压缩维度 $d_{kv}$ 和模型维度 $d_{\text{model}}$ 通常在 scaling 时独立变化。在初始化时，query 分量 $q_t^C$ 和 $q_t^R$ 的方差分别 $\propto d_q$，key 分量 $k_t^C$ 的方差 $\propto d_{kv}$，而 rotary key 分量 $k_t^R$ 的方差 $\propto d_{\text{model}}$。维度间的方差不匹配导致注意力分数在初始化时不稳定（某些维度的分量主导 attention score），small scale 下表现良好的 MLA 配置在 scaling up 时性能退化。

解决方案：在低秩路径分量上应用 scale-correction 因子 $\alpha_q = \sqrt{\frac{d_{\text{model}}}{d_q}}$ 和 $\alpha_{kv} = \sqrt{\frac{d_{\text{model}}}{d_{kv}}}$，将它们缩放后的最终方差对齐到 $d_{\text{model}}$ 参考尺度。

修正后的 MLA 公式：$$c_t^Q = \alpha_q W^{DQ} h_t, \quad c_t^{KV} = \alpha_{kv} W^{DKV} h_t$$ 使得初始化时 $q_t^C, q_t^R, k_t^C, k_t^R$ 的方差均匀贡献到 attention score，确保模型 scaling 时的稳定性和可预测性。LongCat-Flash 实验（Figure 5a）显示在 1B activated MoE 上 scale-correction 带来更低 validation loss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LongCat-Flash 的 MLA 完整 forward pass：

```
输入: h_t [batch, seq_len, d_model=6144]

# Hyper-params: d_q=1536, d_kv=512, n_h=64, per-head dim=128 on the up-projected side

# Stage 1: Latent Compression (with scale-correction)
alpha_q = sqrt(d_model / d_q)      # sqrt(6144/1536) = 2.0
alpha_kv = sqrt(d_model / d_kv)    # sqrt(6144/512) ≈ 3.464

c_Q  = alpha_q  * W_DQ  @ h_t    # [batch, seq, d_q=1536], query latent
c_KV = alpha_kv * W_DKV @ h_t    # [batch, seq, d_kv=512], KV latent

# Stage 2: Up-Projection
q_C = W_UQ @ c_Q                 # [batch, seq, n_h * dim_per_head=128] for compressed part
q_R = W_QR @ c_Q                 # [batch, seq, n_h * d_rope] for RoPE part
k_C = W_UK @ c_KV                # [batch, seq, n_h * dim_per_head=128] (non-RoPE part)
v   = W_UV @ c_KV                # [batch, seq, n_h * dim_per_head=128]
k_R = W_KR @ h_t                 # [batch, seq, d_rope] shared across heads

# Stage 3: RoPE
q_R_rope = RoPE(q_R)
k_R_rope = RoPE(k_R)  # broadcast to all heads

# Stage 4: Concatenation
q = concat([q_C, q_R_rope], dim=-1)
k = concat([k_C, k_R_rope.expand(-1, n_h, -1)], dim=-1)

# Stage 5: Attention
o = Attention(q, k, v)

# Stage 6: Output Projection
u = W_O @ concat([o_1, ..., o_{n_h}], dim=-1)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Scale-correction 的实现极简：在模型初始化/forward 时将低秩投影权重乘以上述 scaling factor（或直接在 forward 中对 c_Q, c_KV 乘 factor）。α 值仅依赖于架构选择的 d_q, d_kv, d_model 三个超参数，无需额外训练或调参。LongCat-Flash 配置：d_model=6144, d_q=1536, d_kv=512 → α_q=2.0, α_kv≈3.464。

与 DeepSeek-V2/V3 的 MLA 对比：DeepSeek 原始 MLA 未使用 scale-correction，可能在特定 d_q/d_kv 比例下表现良好但缩放时性能退化。Scale-correction 提供了保证任意维度配置下注意力机制都能稳定运行的通用解决方案。

涉及论文标题：
- LongCat-Flash Technical Report

## Multi-Token Prediction (MTP) / 多 Token 预测

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Multi-Token Prediction (MTP) 是一种辅助训练目标，使模型在每个位置预测多个 future token（而非仅 next token），在推理时可用作 speculative decoding 的 draft model。LongCat-Flash 采用 dense layer（非 MoE layer）作为 MTP head，在训练中期（而非全程）引入 MTP 训练。

LongCat-Flash 的 MTP 设计要点：
1. **Single dense layer head**：使用单个 dense FFN layer（非 ScMoE/MoE layer）作为 MTP head，参数量仅为主模型的 1.41%，接受率 92.1%（vs ScMoE head 的 4.17% params, 92.9% accept rate）。以微小的接受率损失换取大幅减少的 draft 计算开销。
2. **Late-phase training**：MTP head 在训练的中间阶段引入（非从零开始），因为 MTP loss 收敛极快。过早引入可能干扰主模型训练。
3. **Speculative decoding integration**：MTP head 作为 draft model，接受率 >90%，配合 C2T (Classifier-based Tree Construction) 过滤低接受概率 token，实现 expected accept length $\Omega(\gamma, \alpha)$ 约 1.8x。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# MTP Training (per training step)

输入: hidden_states [batch, seq_len, d_model]
      主模型: 28-layer ScMoE Transformer

# Forward:
h = MainModel(hidden_states)  # [batch, seq_len, d_model]

# MTP head 预测
logits_next1 = MTP_Head(h[:, :-1])    # 预测下一个 token
logits_next2 = MTP_Head(h[:, :-2])    # 预测下下个 token (如果 MTP depth=2)

# Loss
loss_next1 = CrossEntropy(logits_next1, tokens[:, 1:])
loss_next2 = CrossEntropy(logits_next2, tokens[:, 2:])  # 如果 MTP depth=2
total_loss = main_loss + lambda_mtp * (loss_next1 + loss_next2)

# MTP Head 结构 (LongCat-Flash 选择 single dense layer):
# MTP_Head(x) = LayerNorm(x) @ W_mtp.T
# W_mtp: [d_model, vocab_size=131072]
```

推理时 MTP + C2T Speculative Decoding：
- Draft stage: Target model forward → MTP head → 生成 γ 个 draft tokens
- Filter stage: C2T classification model → 过滤低接受概率 token
- Verify stage: Target model forward with all draft tokens → 接受/拒绝每个 token

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. MTP 最初由 Gloeckle et al., 2024 提出，DeepSeek-V3 将其从 independent output heads 改进为 sequential prediction（每个 MTP head 间有因果依赖），但 LongCat-Flash 采用 simpler single dense layer head。
2. LongCat-Flash 选择 dense layer 而非 MoE layer 的关键 tradeoff：dense head 参数量少但 GPU 利用率高（decode batch 小），MoE head 接受率略高但 draft cost 大（需要 all-to-all 通信）。
3. C2T (Huo et al., 2025) 是 classifier-based tree construction——训练一个轻量分类器判断 draft token 是否可能被 target model 接受，提前过滤可减少 verification 开销。
4. TVD fusion: Target forward + Verification + Draft forward 融合为单个 CUDA Graph 减少 kernel launch overhead。

涉及论文标题：
- LongCat-Flash Technical Report
- Better & Faster Large Language Models via Multi-Token Prediction

## Model Growth Initialization (Layer Stacking / 模型增长初始化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Model Growth Initialization 是 LongCat-Flash 使用的大模型参数初始化策略。核心思想：先训练一个小规模的"前身模型"（predecessor model），然后通过 layer stacking（层堆叠）将小模型扩展为目标大模型。具体地，LongCat-Flash 先训练一个 14-layer（half-scale）的模型（与目标模型架构完全一致），然后用 expansion rate r=2 将 14 layers 堆叠为 28 layers，作为 560B target model 的初始化。

公式：$$L_{\text{target}} = \underbrace{L_{\text{small}} \circ L_{\text{small}} \circ \cdots \circ L_{\text{small}}}_{r}$$ 其中 $L_{\text{small}}$ 是从 token embedding 到 final hidden states 的变换，$L_{\text{target}}$ 是堆叠 r 份复本后的大模型变换。

LongCat-Flash 实验（Figure 5b）显示 model growth 初始化的典型 loss 轨迹：初期 loss 短暂上升（因参数翻倍导致的不一致）→ 随后加速收敛 → 最终 outperform random initialization baseline。推测两个因素：(1) 小模型收敛更快→提供更高质量的初始参数；(2) Growth 操作作为 implicit regularization 防止参数坍塌。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Model Growth Pipeline

# Step 1: 训练 Predecessor Model
small_model = ScMoE(n_layers=14, d_model=6144, ...)
small_model.train(data=first_segment_of_20T_tokens)
checkpoint = save(small_model)  # 保留 optimizer state, LR schedule, sample counter

# Step 2: 构造 Target Model (Layer Stacking, r=2)
target_model = ScMoE(n_layers=28, d_model=6144, ...)  # 架构与 small_model 一致，仅 depth 翻倍
for i in range(14):
    target_model.layers[2*i] = copy(small_model.layers[i])
    target_model.layers[2*i + 1] = copy(small_model.layers[i])
# Embedding/Unembedding: 直接继承

# Step 3: 恢复状态继续训练
target_model.load_optimizer_state(checkpoint.optimizer_state)  # optimizer state 被扩展
target_model.load_lr_schedule(checkpoint.lr_schedule)
target_model.load_sample_counter(checkpoint.sample_counter)  # 从 predecessor 的训练进度继续
target_model.train(data=remaining_tokens)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. 与相关方法（Net2Net, bert2BERT, LiGO, SOLAR）相比，LongCat-Flash 使用最简单的 layer stacking (Du et al., 2024; Kim et al., 2023)，即直接复制层而非训练更复杂的 growth operator。
2. 关键实践：(1) 保留所有训练状态（optimizer states, LR schedule, sample counter）而非仅保留模型参数；(2) Expansion rate r=2 (depth doubling) 而非更大的 r——过大的 r 可能导致更严重的初期性能退化；(3) Over-optimizing predecessor 会降低 target model 的 token efficiency——需在适当时间点执行 growth（LongCat-Flash 在 tens of billions tokens 后执行）。
3. Predecessor 架构必须与 target 一致（相同的 d_model, expert 数量, MLA 配置等），仅 depth 不同。

涉及论文标题：
- LongCat-Flash Technical Report

## Hidden z-loss (隐层 z-loss / 抑制 Massive Activation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hidden z-loss 是 LongCat-Flash 提出的训练稳定性正则化技术，用于抑制 LLM 训练中出现的 massive activation 现象（某些 hidden state 元素幅度极大，达到 10^4-10^6 量级）。Massive activations 与训练中的 loss spike 强相关——Sun et al. [2024] 观察到这种相关性，LongCat-Flash 进一步确认并通过 hidden z-loss 解决。

公式：$$\mathcal{L}_Z = \frac{\lambda}{T} \sum_{t=1}^{T} \left( \log \sum_{i=1}^{|z_t|} \exp(\operatorname{abs}(z_t^i)) \right)^2$$ 其中 $\lambda$ 为极小的 loss coefficient，$z_t$ 为 final layer 输出（在 final layer norm 之前），$|z_t|$ 为 hidden state size，abs(\*) 为绝对值函数。

设计原理：通过在 LogSumExp(abs(z)) 上施加 L2 penalty，抑制 hidden state 中个别元素的 extreme magnitude。LogSumExp 近似 max(abs(z))——平滑且可微的 max 函数——因此 $\mathcal{L}_Z$ 惩罚 hidden state 的最大绝对值。内层 exp 放大极端值的影响，外层 log 平滑，平方使 penalty 随 max magnitude 超线性增长。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Hidden z-loss 计算 (per training step)

输入: z [batch, seq_len, d_model]  # final layer output, BEFORE final LayerNorm

# z-loss 计算:
abs_z = abs(z)                           # [batch, seq_len, d_model]
logsumexp_z = log(sum(exp(abs_z), dim=-1))  # [batch, seq_len], 近似 max(|z|)
z_loss_per_token = logsumexp_z ** 2       # [batch, seq_len]
L_Z = lambda * mean(z_loss_per_token)     # scalar, lambda 极小 (e.g., 1e-6)

# 总 loss:
total_loss = L_LM + alpha * L_LB + L_Z   # LM loss + Load Balance loss + z-loss
```

LongCat-Flash 实验（Figure 6）：使用极小的 λ（coefficient negligible）即可显著抑制 massive activation 现象（L2 norm of final layer hidden states 趋于稳定），且不 degrade training loss。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
1. λ 需极小（如 1e-6 或更小）以避免干扰主训练目标。即使 λ 很小，massive activation 的 magnitude 极大（10^4-10^6），loss contribution 仍足以驱动优化。
2. 与 Router z-loss [Zoph et al., 2022] 的区别：hidden z-loss 作用于 hidden states（所有层的最终输出），Router z-loss 作用于路由 logits。二者的共同点在于都用 LogSumExp 惩罚极端值。
3. 对于 BF16 训练，BF16 的动态范围有限（max ~3.4e38），massive activations 虽未直接溢出但增大数值误差风险。Hidden z-loss 降低 hidden state 的 magnitude，提高数值稳定性。

涉及论文标题：
- LongCat-Flash Technical Report

## Hyperparameter Transfer (μTransfer / 超参数迁移)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Hyperparameter Transfer 是 LongCat-Flash 使用的大模型超参数选择策略。核心思想：在小 proxy model 上搜索最优超参数（初始化方差 σ² 和学习率 η），然后通过理论推导的 scaling rules 将这些超参数迁移到 target 大模型，避免在大模型上直接进行昂贵的超参数搜索。

LongCat-Flash 采用 Standard Parameterization (SP) 下的 "Adam LR Full Align" scaling rules [Everett et al., 2024]。对于 width scaling factor $s = n_{\text{target}}/n_{\text{proxy}} = 6144/768 = 8$，迁移规则：
- Embedding layer: $\sigma^2$ 和 $\eta$ 直接迁移（不变）
- Hidden/Unembedding layers: $\sigma^2_{\text{target}} = \sigma^2_{\text{proxy}}/s$, $\eta_{\text{target}} = \eta_{\text{proxy}}/s$
- 所有其他属性（depth, sparsity, batch size）在迁移中保持不变

LongCat-Flash 选择 proxy width=768（s=8），认为这个比例在计算效率和迁移精度间取得了最佳平衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LongCat-Flash 中：(1) 用 s=8 的 proxy 模型搜索最优 σ² 和 η；(2) 按 Table 1 的规则逐层映射到 target model；(3) 在 proxy 上训练极小规模（<1B activated params）即可完成搜索，计算开销远小于 target-scale 搜索。

涉及论文标题：
- LongCat-Flash Technical Report

## Fully Differentiable MoE (Expert Merging in Parameter Space)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Fully Differentiable MoE (Expert Merging in Parameter Space) 是一种完全不依赖离散路由决策的混合专家架构。传统稀疏 MoE 使用 top-k 离散选择（argmax/top-k）决定每个 token 激活哪些专家，路由决策不可微，需要辅助负载均衡损失和复杂的分配算法。相反，Fully Differentiable MoE 在参数空间对所有专家进行软合并（soft merging）：给定路由权重 e_i = Softmax(R(h))，计算所有专家参数的加权平均 θ̄ = Σ_i e_i · θ_i，然后用合并后的 FFN 处理输入 o = FFN(h; θ̄)。整个过程端到端可微，梯度通过合并操作和路由网络全程回传，无需辅助损失。

该方法首先由 SMEAR (Muqeeth et al., 2023) 提出，在 BERT 编码器的文本分类下游微调中验证。Lory (Zhong et al., 2024, COLM) 首次将该架构扩展到自回归语言模型预训练。与 SMEAR 使用池化表示对整个输入序列做一次路由不同，Lory 设计了 causal segment routing 以保留自回归因果性。

关键优势：(1) 无需离散路由，端到端梯度回传；(2) 无需负载均衡辅助损失，减少超参数调优；(3) 合并后的 FFN 在推理时等价于单个 Dense FFN，推理效率与 Dense 模型相同。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# === 参数空间专家合并（Lory 的 moe_ffn 核心） ===
# 输入: seg_x (B*N, T, d), e (B*N, E) 路由权重
# experts: E 个 FFN, 每个 θ_i = (W_gate_i, W_up_i, W_down_i)

# 步骤1: 在参数空间合并所有专家
merged_W_gate = sum_{i=1}^{E} e[:, i] · W_gate_i  # shape: (d, d_ffn)
merged_W_up   = sum_{i=1}^{E} e[:, i] · W_up_i    # shape: (d, d_ffn)
merged_W_down = sum_{i=1}^{E} e[:, i] · W_down_i  # shape: (d_ffn, d)

# 步骤2: 用合并后的 FFN 处理输入（SwiGLU）
gate_out = SiLU(seg_x @ merged_W_gate)   # (B*N, T, d_ffn)
up_out   = seg_x @ merged_W_up            # (B*N, T, d_ffn)
output   = (gate_out ⊙ up_out) @ merged_W_down  # (B*N, T, d)
```

与稀疏 MoE 的关键张量流对比：

**稀疏 MoE (top-k 路由)**：
1. Router: h → W_r·h → softmax → top-k → gate_weights, expert_indices
2. Token dispatch: 将每个 token 路由到选中的 k 个 expert（通过 all-to-all 通信）
3. Expert compute: 每个 expert 独立计算 FFN(h; θ_i)
4. Token combine: 将 expert 输出加权聚合（通过 all-to-all 通信）

**Fully Differentiable MoE (参数合并)**：
1. Router: h̄ → W_r·h̄ → softmax → routing weights e (E-dim)
2. Parameter merge: θ̄ = Σ_i e_i · θ_i（参数空间操作，无需 token dispatch）
3. Merged FFN compute: FFN(h; θ̄)（等价于单次 Dense FFN GEMM）

关键区别：稀疏 MoE 在激活空间路由（dispatch-combine tokens），Fully Differentiable MoE 在参数空间路由（merge parameters）。合并操作的计算开销为 O(E · d · d_ffn) per merge，若每 T 个 token 合并一次，则额外 FLOPs 为 E/T × (FFN FLOPs)，对 T=256, E=32 约 12.5%。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键：
- **合并操作**：纯 PyTorch tensor ops，无需自定义 CUDA kernel。加权求和在参数张量上执行，然后调用标准 cuBLAS GEMM。
- **Expert 架构**：每个 expert 与 Dense FFN 结构完全相同（SwiGLU: W_gate + W_up + W_down），因此合并后的 FFN 也是标准 SwiGLU FFN，仅权重不同。
- **分布式训练**：Lory 使用 data parallelism + ZeRO。但专家参数量大时，可通过 expert-wise model parallelism 按 hidden dim 分片所有专家到不同设备（Section 6）。
- **推理**：Prompt-only routing——用 prompt 的平均隐藏表示计算路由权重，合并 FFN，后续所有 token 使用合并后的 FFN 生成。推理与 Dense 模型完全相同（无额外通信或计算开销）。
- **转换到稀疏推理**：Lory 模型可微调为 hard-decision routing（top-k），在推理时恢复稀疏激活以节省 GPU 内存（但论文未实现）。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

## Causal Segment Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Causal Segment Routing 是 Lory 提出的段级路由策略，用于在自回归语言模型中实现可微 MoE 的高效训练。核心思想：将 token 级路由替换为段级路由——将输入序列分为固定长度 T=256 的段，每段仅做一次路由决策和专家合并，使用前一段的隐藏表示计算当前段的路由权重（因果性），避免信息泄露。

动机：如果对每个 token 做一次专家合并（naive extension of SMEAR），合并计算开销为 O(L · E · d · d_ffn)，对于 L=4096 训练序列不切实际。段级路由将合并次数从 L 降为 L/T（对 T=256 为 16 次），额外 FLOPs 仅 E/T。

Causal shift 机制确保自回归因果性：
- Segment S_k (k>1)：使用 S_{k-1} 的隐藏表示平均值 h̄_{k-1} 计算路由权重，然后合并 FFN 处理 S_k
- Segment S_1：使用自身表示 h̄_1 计算路由，但施加 stop-gradient 防止信息泄露
- 推理时：仅用 prompt 做一次路由决策，后续生成使用同一合并 FFN

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Causal Segment Routing 的 PyTorch 风格伪代码（论文 Algorithm 1）：

```python
# input: x (B, L, d), segment size T
# R: routing network (linear layer)
N = L // T  # number of segments

# Step 1: Split into segments and compute segment representations
seg_x = x.view(B*N, T, d)            # (B*N, T, d)
repr = mean(seg_x, dim=1)            # (B*N, d) avg per segment

# Step 2: Compute routing weights for ALL segments (non-causal)
e = softmax(R(repr), dim=-1)         # (B*N, E)

# Step 3: Make routing causal by shifting
e_first = e.view(B, N, E)[:, 0]      # first segment routing
e = roll(e, 1)                        # shift: segment k uses segment k-1's routing
e = e.view(B, N, E)
e[:, 0] = stop_grad(e_first)         # first segment uses own repr (no leakage)
e = e.view(B*N, E)

# Step 4: Expert merging + FFN computation
seg_y = moe_ffn(seg_x, e)            # merged FFN per segment (see Fully Differentiable MoE entry)
y = seg_y.view(B, L, d)              # reshape back
```

**推理时的 Prompt-only Routing**：
```python
# input prompt: x_prompt (1, L_prompt, d)
repr_prompt = mean(x_prompt, dim=1)  # (1, d)
e = softmax(R(repr_prompt), dim=-1)  # (1, E) single routing decision
theta_merged = sum(e[i] * expert[i].params for i in range(E))
# All subsequent generated tokens use theta_merged (same as Dense FFN)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键：
- **段大小 T=256**：论文通过实验选定，平衡合并效率（更大 T 减少合并次数）和路由粒度（更小 T 使路由更细粒度）。T=256 在 L=4096 下产生 16 段。
- **Stop-gradient**：第一段使用自身表示的路由权重来自 stop_grad(e_first)，防止该段 token "看到未来"信息。这是 causal 属性的关键技术细节。
- **Segment representation**：使用段内所有 token 的 hidden state 平均值，而非 [CLS] 或其他聚合方式。均值操作使模型在推理时能适应不同长度的 prompt。
- **与 Prefix Routing 对比**：Prefix routing 仅用第一个段路由整个序列（类似 SMEAR），性能显著差于 causal segment routing（图 3），证明每个段提供路由训练信号的重要性。
- **推理 train-test gap**：segment-level routing 和 prompt-only routing 在下游任务上差异不显著（Table 9），前者为训练设计，后者为推理简化。
- **与 Token-level MoE routing 对比**：Token-level routing（如 Expert Choice）学到的是浅层词法特征（标点、冠词），segment routing 学到的是领域级特征（arXiv, Python, Books 等）。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

## Domain-level Expert Specialization with Segment Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Domain-level Expert Specialization 是 Lory 训练后观察到的专家行为模式：使用 segment-level routing 训练的 MoE 模型，其专家自动学习按领域（domain）而非按词法特征（token-level features）进行专业化。具体表现为：不同层级的专家对特定领域（arXiv 学术论文、Python 代码、Books、Wikipedia）展现出不同的路由偏好权重（图 6），且低层专家的领域偏好较平坦，中高层的领域分化更明显。

与 token-level MoE routing（如 Switch Transformer, Expert Choice）的对比：
- Token-level MoE 专家学到的是浅层特征：某些专家专门处理标点、冠词、介词，某些专门处理数字、动词等（Zoph et al., 2022; Jiang et al., 2024; Xue et al., 2024）
- Segment-level MoE 专家学到的是深层语义领域特征：专家按主题/领域分化，在训练数据（CommonCrawl）中低频的领域（如 Python code）也能获得专业化专家

这种领域级专业化的驱动力来自 similarity-based data batching：通过将语义相似的文档拼接为训练实例，相邻段来自同一领域，段级路由因此学习到领域感知的路由策略。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**专家专业化分析流程（论文 Section 5.4）**：

```python
# 输入: 0.3B/8E 模型，来自 4 个领域的评估数据（Books, arXiv, Python, Wikipedia）
# 对每层 l 和每个领域的数据 D_domain：
for domain in [Books, arXiv, Python, Wikipedia]:
    for segment in domain_data:
        h_bar = segment.mean_repr              # segment avg hidden state
        e = softmax(R(h_bar))                  # routing weights (E-dim)
        routing_weights[l][domain] += e        # accumulate
    routing_weights[l][domain] /= len(domain_data)  # average

# 可视化: 热力图 (expert × domain) 每层一个
# 观察:
# - Layer 0: routing weights flat across domains (所有专家权重相似)
# - Layer 11: clear domain specialization (expert 7 for arXiv, etc.)
# - Layer 23: clear domain specialization (distinct patterns per domain)
```

关键发现：
- **中间层和高层的领域分化最明显**：低层路由权重跨领域均匀分布，中高层专家权重呈现清晰的领域偏好模式
- **arXiv 和 Python code 的路由权重更相似**：可能因为 LaTeX 代码和 Python 代码都与自然语言有距离
- **与 token-level 专家行为完全不同**：token-level MoE 学到的模式是"专家 3 处理冠词、专家 5 处理标点"（浅层），Lory 学到的是"专家 7 处理 arXiv 论文、专家 2 处理 Python 代码"（深层语义）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现和使用：
- **无需领域标签**：专家在没有领域监督的条件下自动学习领域专业化，完全通过 self-supervised language modeling + similarity-based data batching 驱动。
- **Similarity-based batching 是必要条件**：random batching 下段级路由学到的专业化程度不足（图 4），因为相邻段来自不相关的文档，路由信号被稀释。
- **互补性应用前景**：论文建议将 segment-level 的领域特征与 token-level 的语法特征结合，构建更强模型。例如，在同一 MoE 层中同时使用两种粒度的路由。
- **Out-of-domain 泛化**：在训练数据中低频的领域（如 Python code 在 CommonCrawl 中占比小），segment-level routing 通过领域级特化专家提供更强的 out-of-domain 性能（Python perplexity 12.5 vs EC 的 14.1/13.6）。
- **Expert Utilization**：无辅助负载均衡损失仍能实现高专家利用率（图 9），专家之间的自然领域专业化自发实现负载均衡（不同领域 token 自然路由到不同专家组）。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

## Similarity-based Data Batching for MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Similarity-based Data Batching 是 Lory 用于构造 MoE 训练实例的数据准备策略。标准预训练做法是将随机文档拼接成固定长度实例，可能导致相邻段（segment）来自无关文档。当 segment routing 使用前一段的表示路由当前段时，不相关的相邻段会削弱路由信号的语义一致性，阻碍专家专业化。

Lory 的解决方案：使用 Contriever (Izacard et al., 2022) 计算文档语义相似度，通过贪心搜索将相似文档拼接为训练实例，使相邻段大概率来自相同或相关领域。该方法启发自 In-context Pre-training (Shi et al., 2024, ICLR)，但目标不同——后者旨在增强跨文档边界的推理能力，Lory 旨在促进专家按领域专业化。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

**数据批处理 Pipeline（论文 Section 3.2, Appendix C）**：

```python
# 输入: 文档集合 D = {d1, d2, ..., dM}
# C: Contriever encoder

# Step 1: 计算文档 embedding
embeddings = {d: C(d) for d in D}

# Step 2: 构建相似度图（使用 FAISS 近似搜索）
N = {}  # adjacency
for d in D:
    topk = FAISS.search(embeddings[d], k=top_k)  # top-k most similar
    N[d] = {d_j: cosine_sim(embeddings[d], embeddings[d_j]) for d_j in topk}

# Step 3: Greedy concatenation (Shi et al., 2024 的算法)
instances = []
remaining = set(D)
while remaining:
    current = random.choice(list(remaining))
    instance = [current]
    remaining.remove(current)
    while instance_length(instance) < L:  # L = 4096 tokens
        candidates = [d for d in N[current] if d in remaining]
        if not candidates:
            break
        # 选择与 current 相似度最高且未被使用的文档
        next_doc = argmax(candidates, key=lambda d: N[current][d])
        instance.append(next_doc)
        remaining.remove(next_doc)
        current = next_doc
    instances.append(concat(instance))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键：
- **Contriever**：无监督对比学习训练的 dense retriever (Izacard et al., TMLR 2022)，用于编码文档为固定维度向量。Lory 使用预训练 Contriever 不做微调。
- **FAISS (Johnson et al., 2019)**：Facebook 的高效近似最近邻搜索库，Lory 使用 FAISS GPU 版本进行十亿级文档的相似度搜索。
- **贪心搜索**：从随机文档开始，每次选择与当前文档最相似且未被使用的文档追加到实例，直到实例达到 token 预算（4096）。如果当前文档没有可用相似文档，重启新实例。
- **与 Standard Random Batching 的对比**：Random batching 将随机文档拼接，相邻段可能来自无关领域（如医学论文 + 餐厅评论）。Similarity-based batching 确保相邻段语义相关，提供更一致的路由信号。
- **对 MoE 训练的具体影响**：Similarity batching 下 Lory MoE 相对 Dense 的 loss 改善显著大于 random batching（图 4 right），且差异随训练数据量增加而放大。
- **性能开销**：Contriever encoding + similarity graph construction 是一次性预处理，不影响训练 throughput。FAISS 搜索和贪心拼接在数据预处理阶段完成。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

## AffinityBinning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

AffinityBinning 是 LYNX 提出的用于 MoE 推理中 batch 级别动态专家选择的离散化技术。核心思想：将每个 token 对每个 expert 的 router 置信度（通过 log-ratio 到 top-1 expert 衡量）离散化为有限数量的 bin，bin 的宽度和数量仅由模型架构的 sparsity ratio (k/N) 决定，而非 workload 或 task。这使得 LYNX 成为 self-calibrating 系统——自动适配任何 MoE 架构，无需 profiling 或 tuning。

具体实现：对于每层的每个 token，计算相对于 top-1 expert 的 log-ratio：log_ratio(e) = logit[e] - logit[top1]（即 softmax 概率比的对数等价形式）。然后将这些值按 α（bin width 的倒数控制参数）和 β（最大 bin 数限制参数）离散化：bin[e] = clamp(floor(log_ratio[e] * α), -β, 0)。bin=0 表示与 top-1 亲和力最高，bin 越负表示亲和力差距越大。对于 sigmoid-based router（如 DeepSeek），使用 pre-sigmoid scores 的差值代替 log-ratio。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

AffinityBinning 在 LYNX MoE 推理 pipeline 中的位置和伪代码：

```
# 输入：batch size B, router logits L[B][N], N experts, top-k
# 输出：per-token bin assignments B[B][k]

# Step 1: Router 前向（标准 MoE）
for t in 0..B-1:
    probs[t] = softmax(L[t])          # [N]
    topk_idx[t], topk_prob[t] = topk(probs[t], k)

# Step 2: AffinityBinning（LYNX 核心）
α = compute_alpha(k/N)   # 由 sparsity ratio 决定
β = compute_beta(k/N)    # 通常 5-8

for t in 0..B-1:
    top1_logit = max(L[t])
    for each expert e in topk_idx[t]:
        log_ratio = L[t][e] - top1_logit     # router logits 之差
        bin[t][e] = clamp(floor(log_ratio * α), -β, 0)

# Step 3: Batch-level Adaptive Scoring（使用 bin 值）
for each expert e:
    score[e] = 0
    for t in 0..B-1:
        if e in topk_idx[t]:
            score[e] += B^{bin[t][e]}  # batch_size 为底数的指数加权

# 效果：高置信度 token (bin=0) 贡献 B^0=1
#       低置信度 token (bin=-5) 贡献 B^{-5}≈0.00006 (B=16)
#       被多个高置信度 token 偏好的 expert 得分指数级更高
```

关键参数：
- Qwen2-57B (k=8, N=64, k/N=0.125): α 产生约 6 个 bin (β=5)
- Mixtral-8x7B (k=2, N=8, k/N=0.25): 更宽的 bin (更少的划分)
- DeepSeek-V3 (k=8, N=256, k/N=0.03): 更细的 bin (更多的划分)

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 将 AffinityBinning 实现为 Triton fused kernel（Kernel 1 — Token-wise Binning），拦截 vLLM 每层 MoE router 输出后执行。kernel 对 batch 中所有 token 并行执行 log-ratio 计算和 discrete binning，融合了原本需数百个 PyTorch element-wise ops（subtract, floor, clamp）。α 和 β 在模型加载时根据 sparsity ratio 计算一次，随后所有 forward pass 重用。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Router Confidence Score / Log-Ratio to Top-1

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Router Confidence Score 是 LYNX 提出的衡量 MoE router 对每个 token-expert assignment "置信度"的度量。定义为 token 对某个 expert 的 router logit 相对于 top-1 expert logit 的差值（log-ratio）：confidence(t, e) = logit[t][e] - logit[t][top1]。由于 softmax 概率比的对数等价于 logit 差（log(P(e)/P(top1)) = logit[e] - logit[top1]），log-ratio 直接反映了 router 在各 expert 间的区分度。

LYNX 通过实验验证了该度量的有效性：high-confidence tokens（router 强烈偏好某个 expert，各 expert 分数差异大）的 expert assignment 必须保留；low-confidence tokens（各 expert 分数接近）可以安全地重映射到其他 expert 而不影响输出质量。该区分能力源于 MoE 训练中 load-balancing loss 的副作用——训练时强制均匀利用 expert，导致 router 对许多 token 产生弱偏好（各 expert 分数接近），这些弱偏好在 inference 时是冗余的。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Router Confidence 计算（单 token，MoE layer l）
输入: router_logits z ∈ R^N  (N experts)

# 标准 softmax
p_i = exp(z_i) / Σ_j exp(z_j)    # i ∈ [1, N]

# Top-k selection
topk_idx = argsort(p, descending=True)[:k]

# Confidence: log-ratio to top-1
top1_logit = z[topk_idx[0]]
for e in topk_idx[1:]:
    conf[e] = z[e] - top1_logit   # ≤0, 越接近 0 表示越 confident

# 与概率比的关系
# conf[e] = log(p_e / p_top1)  因为 log(p_e/p_top1) = log(e^{z_e}/e^{z_top1}) = z_e - z_top1
```

LYNX 实验发现 (Figure 6)：随着 confidence threshold 提高，high-confidence 和 low-confidence token 在 remapping 后的 accuracy impact 差异显著扩大。这表明 log-ratio 确实是区分 critical vs redundant token-expert mapping 的可靠信号。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 LYNX 实现中，Confidence Analyzer (Kernel 1) 直接拦截 router 输出的未归一化 logits（而非 softmax 后概率），计算 log-ratio 并做 AffinityBinning 离散化。使用 logits 而非概率避免了 softmax 的数值稳定性问题，且在 GPU 上 logit 差值比概率除法更高效。对于使用 sigmoid-based routing（如 DeepSeek-V2/V3）的模型，使用 pre-sigmoid scores 的差值。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Expert Rank Hierarchy in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Rank Hierarchy 是 LYNX 通过实验揭示的 MoE router top-k 选择中不同 rank 位置的 expert 对输出质量的贡献不对称性。核心发现：top-1 (rank-0) expert 对输出质量具有决定性影响——deny top-1 expert 会导致 catastrophic accuracy drop；而 lower-ranked experts (rank 1, 2, ..., k-1) 高度冗余——deny 它们仅造成 minimal accuracy degradation。此外，当 cumulatively restore experts（先 restore top-1, 再加 top-2, ...）时，恢复 3-4 个 expert 后 accuracy 迅速接近 baseline，之后 diminishing returns。

这一层级结构是 LYNX 在设计时保留所有 token 的 top-1 expert（as "anchor"）、仅重映射 lower-ranked experts 的理论依据。发现该 hierarchy 在 GSM8K 和 HumanEval 两个性质完全不同的 task 上一致成立，表明它是 MoE computation 的结构属性而非 task-specific artifact。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LYNX Expert Rank Hierarchy 的验证实验（以 Qwen2-57B, k=8 为例）：

```
# 实验 1: Deny expert by rank（Figure 7）
for rank in range(k):  # rank 0..7
    # 强制将 batch 中所有 token 的 rank-r expert 替换为下一个候选
    for token in batch:
        deny expert at position rank in sorted top-k
        replace with next-best expert
    measure accuracy drop

结果: rank-0 (top-1) deny → catastrophic drop (>50% accuracy loss)
      rank-1 to rank-7 deny → <5% drop each

# 实验 2: Cumulative restore（Figure 8）
for n_keep in range(1, k+1):
    for token in batch:
        keep top n_keep experts, remap rest
    measure accuracy recovery

结果: 保持 3-4 experts → 恢复 ~95% baseline accuracy
      保持 5+ experts → 几乎完全恢复
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 利用 Expert Rank Hierarchy 设计 Adaptive Expert Scorer：高置信度 token 始终保留其 top-1 expert（确保 minimal accuracy impact），low-confidence token 的 lower-ranked experts 被安全地 remap 到 batch 的 reduced active expert set。这比 naive voting schemes（对所有 top-k 选择等权重投票）更有效地减少 active expert 数量而不损失 accuracy。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Sparsity Ratio (k/N) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Sparsity Ratio (k/N) 是 MoE 模型架构的关键参数，表示每个 token 激活的 expert 数 (k) 与总 expert 数 (N) 之比。它决定了 MoE 的稀疏激活程度：k/N 越小，稀疏性越高，每个 token 的计算量越低，但每个 expert 接收到的 token 越少。代表性值：Mixtral-8x7B k/N=2/8=0.25、Qwen2-57B k/N=8/64=0.125、DeepSeek-V3 k/N=8/256≈0.03。近期模型趋势向更低的 k/N 发展（保持 k 不变、增大 N），以获得更好的 accuracy-FLOPs trade-off。

LYNX 揭示了 sparsity ratio 的另一维度：它决定了 batch 级别 expert activation 的饱和速度。例如 Qwen2-57B (k/N=0.125)，每个 token 选 8 个 expert，batch 仅需 8 个 diverse requests 即可饱和全部 64 个 expert。对于更低的 k/N（如 DeepSeek-V3 的 0.03），batch 级 expert 饱和更慢，MoE 的稀疏性优势在更大的 batch size 下仍可保持。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Sparsity Ratio 对 decode 阶段 arithmetic intensity 的影响
arithmetic_intensity ∝ (batch_size × k) / N

# 例：Mixtral-8x7B, B=16, k=2, N=8
AI = 16×2/8 = 4  → moderate memory-bound

# 例：Qwen2-57B, B=16, k=8, N=64
AI = 16×8/64 = 2  → strongly memory-bound

# 例：DeepSeek-V3, B=16, k=8, N=256
AI = 16×8/256 = 0.5 → extremely memory-bound
```

LYNX 中 sparsity ratio 还决定了 AffinityBinning 的 α 和 β 参数——k/N 越小（稀疏性越高），bin 划分越细（更大的 β），因为更高的稀疏性意味着更少的 tokens 竞争每个 expert，需要更精细的 confidence 区分。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Sparsity ratio 是模型架构设计时的固定 choice，在模型训练前确定。当前 MoE 模型通过增大 N（而非减小 k）来降低 k/N——因为 k 太小会导致 training instability and expert collapse，而增大 N 通过 load-balancing loss 可以稳定训练。LYNX 利用 sparsity ratio 仅在模型加载时读取一次来确定 binning 参数，无需运行时调整。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Load-Balancing Loss in MoE (Training Side Effect on Inference)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Load-Balancing Loss 是 MoE 训练中为防止 expert collapse（所有 token 路由到少数 expert）而引入的辅助损失函数。标准公式为 L_aux = α · N · Σ_i f_i · P_i，其中 f_i 为路由到 expert i 的 token 比例，P_i 为 gate 分配给 expert i 的平均概率，α 为 loss 系数（通常 0.01）。该 loss 强制 router 将 token 均匀分布到所有 expert，确保训练过程中所有 expert 都得到充分训练。

LYNX 揭示了 load-balancing loss 在 inference 时的关键副作用：虽然它成功防止了 expert collapse，但也迫使 router 在 confidence 较低时仍将 token 分配到 less-preferred experts——产生 "forced diversification"。结果是 inference 时许多 token-to-expert assignment 是 training regularization 的产物，而非 genuine token-expert affinity。这造成了 batch 级别 expert activation 的系统性冗余，正是 LYNX 通过 AffinityBinning 利用的机会。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Load-Balancing Loss（GShard 风格）
f_i = (1/T) Σ_t 𝟙[argmax(g(x_t)) = i]   # expert i 的 token 比例
P_i = (1/T) Σ_t g(x_t)_i                    # gate 分配给 expert i 的平均概率
L_balance = α · N · Σ_i (f_i · P_i)

# 训练 total loss
L_total = L_task + L_balance

# 效果：训练时 P_i 趋向均匀 → 推理时 router 为每个 token 产生的
#       expert probability distribution 也趋向均匀（各 expert 分数接近）
#       → 产生 low-confidence token-expert assignments（LYNX 利用的冗余）

# LYNX 观察（Figure 3）：
# - 数据集级别（aggregate）: expert activation frequency uniform（~1.2% 变异性）
# - Batch 级别: expert activation frequency skewed（~15-20% 变异性）
# - 原因：load-balancing 在 aggregate level 起作用，但每次 iteration 的
#         batch composition 不同 → 产生 batch-level heterogeneity
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Load-balancing loss 在训练框架中作为 auxiliary loss 实现（如 DeepSpeed-MoE, Megatron-MoE, Tutel）。常见变体：
- **GShard-style**：α·N·Σ f_i·P_i，同时考虑 dispatch 比例和 gate 概率
- **Switch Transformer-style**：α·N·Σ f_i·P_i，加 capacity-based expert overflow handling
- **DeepSeek-V3-style**：expert-level balance loss + device-level balance loss (for expert parallelism)
- **Auxiliary-loss-free**：一些近期工作探索不依赖 auxiliary loss 的 load balancing（如 expert choice routing）

LYNX 不修改 load-balancing loss，而是利用其产生的 inference 时副作用——这是一种纯 inference-time optimization，与训练解耦。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Prefill-Decode Asymmetry in MoE Expert Sensitivity

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Prefill-Decode Asymmetry 是 LYNX 发现并利用的 MoE 推理阶段非对称属性：prefill 和 decode 阶段对 expert selection fidelity 的敏感度存在根本性差异。在 prefill 阶段，expert reassignment 会显著降低模型性能（特别是在 code generation 和 complex reasoning 任务上）；在 decode 阶段，相同的 expert modification 仅产生 minimal accuracy impact。这种不对称性跨 task types（code, math, reasoning）一致成立，暗示它是 auto-regressive inference 的根本属性——prefill 建立 context 指导所有后续计算，而 decode 受益于 attention、residual connections 和累积 context 的补偿机制。

LYNX 利用此不对称性设计 Phase-Aware Optimizer：仅在 memory-bound decode iteration 中启用 expert remapping，prefill 和其他 compute-bound 阶段直接绕过。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Phase-aware expert reduction policy（基于 Prefill-Decode Asymmetry）

def moe_forward_with_lynx(batch, phase, memory_bound):
    if phase == "prefill":
        # Prefill: 严格保留所有 router 选择的 expert
        # 原因：prefill 建立 full context, expert fidelity critical
        return standard_moe_forward(batch)
    
    elif phase == "decode" and memory_bound:
        # Decode: 可以安全地 remap low-confidence experts
        # 原因：attention/residual/accumulated context 补偿 suboptimal selection
        return lynx_expert_remapping(batch)
    
    else:
        # Compute-bound decode (rare): skip LYNX overhead
        return standard_moe_forward(batch)

# Arithmetic intensity 差异（§2.1）:
#   Prefill: AI high (many tokens) → compute-bound → remapping 无益
#   Decode: AI = B × k / N → memory-bandwidth-bound → remapping 直接减少 HBM 流量

# LYNX Figure 4 实验验证:
#   Prefill expert reassignment → HumanEval accuracy 显著下降
#   Decode expert reassignment → accuracy minimal impact
#   跨 GSM8K 和 HumanEval 一致性 → 结构属性而非 task artifact
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LYNX 的 Phase-Aware Optimizer 集成在 vLLM batch scheduler 中，支持三种常见 serving policy：(1) Co-located prefill/decode：识别 pure-decode batches 为 memory-bound；(2) Disaggregated serving：直接标记 decode 实例为 memory-bound；(3) Chunked prefill：标记仅含 decode tokens 的 batch 为 memory-bound。含 prefill chunks 的混合 batch 被认为是 edge case，留给 future work。

涉及论文标题：
- LYNX: Enabling Efficient MoE Inference Through Dynamic Batch-Aware Expert Selection

## Expert Map（专家概率图）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Map 是 FineMoE 提出的核心数据结构，用于在 **iteration-level**（而非 request-level）追踪 MoE 模型中 gate network 对所有 experts 的选择偏好。每个 expert map 记录一次 inference iteration 中所有 L 个 MoE 层的 gate network 输出的完整概率分布 P_l^{(i)} ∈ R^J（而非 binary activation 或 hit count），其中 map_i = {P_1^{(i)}, ..., P_L^{(i)}}，P_l^{(i)} = {p_{l,1}^{(i)}, ..., p_{l,J}^{(i)}}, Σp = 1。直观上，expert map 不仅记录"哪些 experts 被选择"，更捕获了 gate network 对每个 expert 的 confidence/preference 分布——包含 "expert A 以 0.65 概率被选，expert B 以 0.20 概率被选" 等细粒度 confidence 信息。

与 MoE-Infinity Expert Activation Matrix 的关键区别：
- Activation Matrix: request-level, binary hit count（“expert_3 activated 5 times”）
- Expert Map: iteration-level, full probability distribution（“iteration i, layer l: expert_3 p=0.65”）
Expert Map 可通过 top-K selection + iteration aggregation 退化恢复 Activation Matrix，因此是 Activation Matrix 的 generalization。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Expert Map 构建流程（单 iteration，Mixtral-8×7B, L=32, J=8, K=2）：

for l in range(L):
    # Step 1: self-attention
    attn_out = self_attention_layer[l](hidden_states)

    # Step 2: gate network 输出 probability distribution
    logits = gate_network[l](attn_out)           # R^{J=8}
    P_l = softmax(logits)                         # R^{J=8}, Σp = 1
    # 例: P_l = [0.02, 0.45, 0.03, 0.01, 0.38, 0.05, 0.04, 0.02]

    # Step 3: top-K expert selection (用于实际计算)
    top_k_experts = topk(P_l, K=2)               # 例: [1, 4] (expert_1:0.45, expert_4:0.38)

    # Step 4: expert computation
    expert_out = sum(expert[e](attn_out) for e in top_k_experts)

    # Step 5: 记录 expert map 条目
    map[l] = P_l  # 完整概率分布，不只是 top-K

# 最终 expert_map = {P_0, P_1, ..., P_31} ∈ R^{32×8}
```

Expert Map 的关键优势：
1. Fine granularity: per-iteration 而非 per-request → Shannon entropy 低 → 可预测性高
2. Probability information: 不仅知道哪些 experts 被选，还知道 gate network 对各 expert 的 confidence
3. Degradability: top-K + 聚合 = 退化恢复 Activation Matrix，保证向后兼容
4. Trajectory comparability: probability distributions 向量可直接 cos_sim 比较

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 用 PyTorch/NumPy ndarray 存储 expert maps。每个 map 包含：(1) L×J float32 概率值，(2) 1×d_model semantic embedding，(3) 可用于 trajectory comparison 的 flattened probability vector。Map Store 容量 1K maps（<200MB CPU memory）。去重：通过 unified redundancy score 计算 pairwise redundancy，保留覆盖更多 pattern 空间的 map 集合。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Semantic-based Expert Map Search（基于语义的专家图搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Semantic-based Expert Map Search 是 FineMoE 的两种 expert map 检索方式之一，利用 MoE 模型中 embedding layer 输出的 semantic embedding 与 Expert Map Store 中历史 semantic embeddings 的 cosine similarity，检索最相似的 historical expert map。核心假设：语义相似的 prompts 具有相似的 expert 选择模式（此假设经 Pearson correlation 验证，semantic similarity 与 expert hit rate 正相关）。主要用于前 d 层（prefetch distance 内，尚无足够 trajectory history 可用时）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Semantic-based Expert Map Search 流程：

Input: new_prompt_tokens ∈ R^{B×seq_len}
        Expert Map Store: {sem_old ∈ R^{C×h}, map_old ∈ R^{C×L×J}}
        prefetch_distance d

# Step 1: 提取 semantic embedding
sem_new = embedding_layer(new_prompt_tokens)  # R^{B×h}, h=4096 for Mixtral

# Step 2: pairwise cosine similarity
score_sem ∈ R^{B×C} = (sem_new · sem_old^T) / (||sem_new|| · ||sem_old||)
# 每个 batch 元素与 C 个历史 prompts 的 pairwise similarity

# Step 3: 选择最相似 historical iteration
for b in range(B):
    best_iter[b] = argmax(score_sem[b, :])  # 第 y 个历史 iteration

# Step 4: 使用 best_iter 的 expert map 指导前 d 层 prefetch
for l in range(1, d+1):
    P_l = map_old[best_iter, l, :]  # 第 y 个历史 iteration 第 l 层的概率分布
    prefetch_experts_with_similarity_aware_selection(P_l, score_sem)

# 仅用于 l ∈ [1, d] 的层 → 之后切换为 trajectory-based search
```

关键设计决策：使用模型的原始 embedding layer 而非额外训练的 encoder，因为 "words that appear in similar contexts will have similar embeddings" (Mikolov et al., 2013)。Embedding layer 的输出天然捕获了 prompt 的语义特征，且无需额外计算开销。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中语义搜索用 PyTorch native cosine_similarity 实现。sem_old 和 map_old 均为预先填充的历史数据（70% prompts 用于 Expert Map Store）。语义搜索的有效性由 Pearson correlation analysis 验证（图 9）：所有 6 个 model-dataset 组合中，semantic similarity 与 expert hit rate 的 Pearson coefficient > 0，表明正相关。搜索开销极小（<50ms），与 expert prefetching 均为异步执行，不进入 critical path。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Trajectory-based Expert Map Search（基于轨迹的专家图搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Trajectory-based Expert Map Search 是 FineMoE 的第二种 expert map 检索方式，利用已观察到的前 (l-d) 层 expert probability distributions（称为 "expert trajectory"）与 Expert Map Store 中历史 expert maps 对应层的 cosine similarity，检索最匹配的 historical expert map。Expert trajectory 定义为 "从 Layer 1 到当前 visible layer 的 gate network probability distributions 序列"。用于第 l ∈ [d+1, L] 层（prefetch distance 之后，已有足够的 trajectory history）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Trajectory-based Expert Map Search 流程：

Input: current_trajectory[tokens] ∈ R^{(l-d)×J} (前 l-d 层的 gate probability)
        Expert Map Store historical maps ∈ R^{C×L×J}
        target_layer l ∈ [d+1, L]

# 对于每次 inference iteration 的每个 layer l:
for l in range(d+1, L+1):
    # Step 1: 收集前 (l-d) 层的 expert trajectory
    traj_new = concat([P_1, ..., P_{l-d}])  # R^{(l-d)×J} flattened to R^{(l-d)·J}

    # Step 2: pairwise cosine similarity with historical
    traj_old = map_old[:, :(l-d), :].reshape(C, -1)  # R^{C×(l-d)·J}
    score_traj ∈ R^{B×C} = cos_sim(traj_new, traj_old)

    # Step 3: 选择最相似 historical iteration
    best_iter = argmax(score_traj, dim=-1)

    # Step 4: 提取该 iteration 的第 l 层 expert map
    P_l = map_old[best_iter, l, :]  # R^{J}
    
    # Step 5: similarity-aware expert prefetching for layer l
    prefetch_experts_with_similarity_aware_selection(P_l, score_traj)

# 特点：随着 l 增大，(l-d) 增大 → trajectory 信息量增加 → prediction 更准确
# 例：l=d+1 时仅用 1 层 trajectory →; l=L 时用 L-d 层 trajectory（最多信息）
```

与 Semantic Search 的协同关系：
- Semantic search: 适应初始层（无 trajectory history），利用 prompt 全局语义
- Trajectory search: 适应后续层（轨迹越长越准），利用 expert selection 的序列依赖性
- 两者通过 unified redundancy score 统一为单一 map store

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch native cosine_similarity 计算。traj_new 按需拼接前序层的 gate outputs（每层 gate 输出被缓存）。Pearson correlation analysis 表明 trajectory similarity 与 expert hit rate 正相关（所有 model-dataset 组合）。随着 l 增大（trajectory 信息增加），trajectory-based prediction 准确度持续提高，弥补了 semantic-based 在后期层的不足。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Similarity-Aware Expert Selection（相似度感知专家选择 / δ-threshold）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Similarity-Aware Expert Selection 是 FineMoE 的 expert prefetching 决策机制：根据检索到的 historical expert map 与当前 context 的 cosine similarity score 动态决定预取多少 experts。核心公式：δ_l = clip(1 - similarity_score, 0, 1)，从 searched expert map P_l 中按概率从高到低选择 experts，直到累积概率 Σp ≥ δ_l 且至少选择 K 个（MoE 模型每层需激活 K 个 experts）。直观逻辑：高 similarity → 高 confidence → 低 δ → 选少量 high-probability experts → 节省 GPU cache；低 similarity → 低 confidence → 高 δ → 选更多 experts → 增大 coverage 防 miss。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Similarity-Aware Expert Selection 算法：

Input: searched probability distribution P_l ∈ R^J  (layer l, from best-match historical map)
       similarity_score ∈ [-1, 1]                  (cosine similarity from search)
       K ∈ Z+                                      (top-K required by MoE model, e.g., K=2 for Mixtral)

Algorithm:
δ_l = clip(1 - similarity_score, 0, 1)
# 例: score=0.9 → δ=0.1 (高 confidence, 少选)
#     score=0.3 → δ=0.7 (低 confidence, 多选)
#     score=-0.5 → δ=1.0 (极低 confidence, 全选)

sorted_experts = argsort(P_l, descending=True)  # 按概率降序
E_prefetch = []
cum_prob = 0

for j in sorted_experts:
    E_prefetch.append(j)
    cum_prob += P_l[j]
    if cum_prob >= δ_l and len(E_prefetch) >= K:
        break

# 约束条件 (Eq. 6-8):
#   Σ_{j ∈ E_prefetch} p_{l,j} ≥ δ_l   (累计概率达到阈值)
#   |E_prefetch| ≥ K                   (至少选 K 个)

return E_prefetch
```

该机制在 hit rate（减少 expert miss）和 GPU memory（减少 prefetch 量）之间实现连续可调的 trade-off。消融实验（图 14a）表明 Map(T+S+δ) 比 Map(T+S)（静态 top-K selection）进一步提升 expert hit rate。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中以 PyTorch ops 实现，与 Expert Map Searcher 集成。每个 inference iteration 的每个 target layer 都执行动态 selection，确保 prefetch 策略随 context 变化自适应调整。对比 baseline 的固定 stride（ProMoE）或 LFU（MoE-Infinity）策略，similarity-aware 方式使 GPU cache 容量利用更高效：高 confidence 时省出 cache 空间给更多 KV cache/batch tokens，低 confidence 时增大 coverage 保 latency。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading

## Expert Map Deduplication / Redundancy Score（专家图去重 / 冗余分数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Map Deduplication 是 FineMoE Expert Map Store 的容量管理策略：当 Expert Map Store 达到容量上限 C（默认 1K）时，通过计算新 iteration data 与历史 data 的 pairwise redundancy score 来判断哪些 historical expert maps 是冗余的（即新 data 已能覆盖其 expert selection pattern 空间），并剔除冗余 maps 以维持 store 的 pattern diversity。Redundancy score 统一了 semantic similarity 和 trajectory similarity：

RDY_{x,y} = (d/L) × score^{sem}_{x,y} + ((L-d)/L) × score^{traj}_{x,y}

其中 d 是 prefetch distance，L 是总层数，x 是新 batch iteration index，y 是历史 iteration index。权重 (d/L) 和 ((L-d)/L) 对应 semantic search 和 trajectory search 在 overall matching 中的贡献比例。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
Expert Map Deduplication 流程：

Input: new_batch_context (B 个新 iterations 的 semantic + trajectory data)
        Expert Map Store with C historical maps (at capacity)
Output: Updated Expert Map Store (≤ C maps, 去重后)

# Step 1: 计算所有 pairwise redundancy scores
for x in range(B):           # 新 iterations
    for y in range(C):       # 历史 iterations
        score_sem[x,y] = cos_sim(sem_new[x], sem_old[y])
        score_traj[x,y] = cos_sim(traj_new[x], traj_old[y])
        RDY[x,y] = (d/L) × score_sem[x,y] + ((L-d)/L) × score_traj[x,y]

# Step 2: 对于每个新 iteration x，找到与之最冗余的历史 iteration y
for x in range(B):
    best_y = argmax(RDY[x, :])  # 最低 redundancy → 最不相似 → 最值得保留
    
# Step 3: 用新 iteration 替换与之最相似（冗余）的历史 iteration
# 注意：保留最少 redundancy 的历史 maps → 维持 pattern diversity

# 理论保证 (Minimum Sphere Covering):
#   保持 2LJ expert maps → ≥75% similarity lower bound (任意新 iteration 可找到 ≥75% 相似的 map)
#   保持 (1/2)LJ·ln(LJ) maps → ≥98% similarity lower bound
#   对于现代 MoE: L∈[8,128], J∈[24,96] → 需求 < 50K maps → < 200MB
```

理论分析：expert map deduplication 可被形式化为 Minimum Sphere Covering Problem（每个 expert map 是向量空间中的一个点，要去重后仍能覆盖尽可能多的 pattern 空间）。Dumer (2007) 和 Rankin (1947) 给出覆盖球面的 number-of-centers lower bound。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FineMoE 中以 PyTorch pairwise cosine similarity 计算 redundancy scores。去重在每次 Expert Map Store 满容时触发（而非每次 iteration）。权重比例 (d/L) vs ((L-d)/L) 直接映射 semantic search 和 trajectory search 在整体 expert map matching 中的贡献。实验表明 C=1K 已足够（similarity scores 在 >1K 后 quickly diminishing returns），对应 ≤50MB 内存开销。

涉及论文标题：
- Taming Latency-Memory Trade-Off in MoE-Based LLM Serving via Fine-Grained Expert Offloading


## Top-k Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Top-k Routing 是 MoE 模型中的核心路由机制，决定每个输入 token 激活哪些 expert。给定输入 x，gate network 计算所有 expert 的匹配分数 g(x) ∈ R^N（N 为 expert 总数），通过 Softmax(TopK[x·W_g]) 选出得分最高的 k 个 expert，其余 expert 被 mask 为零。最终输出为选中 expert 输出的加权和：y = Σ_{i=1}^{k} G(x)_i · E_i(x)。k 是固定的超参数（如 Mixtral 的 k=2，DeepSeek-V2 的 k=6），在传统 MoE 架构中所有层使用相同的 k 值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LExI 论文中 top-k routing 的伪代码：

```
# 标准 Top-k Routing (per MoE layer, per token)
输入: x ∈ R^H (hidden state), W_gate ∈ R^{N×H} (gate weights), k (top-k)

# Step 1: Gate 计算
gate_logits = x @ W_gate.T             # [N], raw scores
gate_scores = Softmax(gate_logits)      # [N], 概率分布

# Step 2: Top-k 选择
topk_vals, topk_idx = TopK(gate_scores, k)  # 选最高 k 个

# Step 3: 归一化选中权重
topk_weights = topk_vals / sum(topk_vals)   # [k], 归一化

# Step 4: Expert 计算 + 加权求和
output = zeros(H)
for i in range(k):
    e_idx = topk_idx[i]
    expert_out = Expert_FFN[e_idx](x)  # W1→Act→W2
    output += topk_weights[i] * expert_out
```

LExI 的关键发现：固定 top-k 在不同层引入不同程度的计算冗余。通过 Frobenius 范数测量每层在不同 k 值下的输出扰动，发现浅层和深层对 top-k 变化的敏感度差异显著（Mixtral 浅层低敏感、深层高敏感；Qwen 反之）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Top-k routing 在 HuggingFace Transformers 中通过 `MixtralSparseMoeBlock` 实现，gate 为 `nn.Linear(hidden_size, num_experts)`。推理框架 vLLM 使用 FusedMoE kernel 将 routing 和 expert 计算融合执行，减少 kernel launch overhead。LExI 通过离线计算最优的逐层 k 值，在推理时调用 `set_topk(model.moe_layers[j], k_j)` 修改每层的 k 参数——无需修改 routing 逻辑本身。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference


## Expert Pruning (Inter-Expert / Intra-Expert Pruning)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Expert Pruning 是 MoE 模型的后训练压缩方法，通过删除冗余 expert 或其内部维度来减少模型大小。分为两类：(1) **Inter-Expert Pruning**（如 NAEE）：删除整层中不重要的 expert，保留的 expert 继续被 router 选择，但 top-k 不变——导致剩余 expert 需处理更多 token，造成负载不均衡；(2) **Intra-Expert Pruning**（如 MoE-I²）：缩减每个 expert 内部 FFN 的 intermediate 维度（如从 14336 缩减到 10752），保留所有 expert 但每个 expert 计算量减少。两种方法都依赖 calibration 数据集评估 expert 重要性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Inter-Expert Pruning Pipeline (NAEE-style)
# 目标: 每层删除 p% 的 expert

# Step 1: Calibration-based importance scoring
for layer in moe_layers:
    for expert in layer.experts:
        # 在 calibration set 上计算移除该 expert 后的 loss 增加量
        importance[expert] = ΔLoss when removing expert

# Step 2: 逐层删除最低 importance 的 experts
for layer in moe_layers:
    prune_count = int(num_experts * p)
    prune_experts = bottom_k(importance[layer], prune_count)
    remove_from_model(prune_experts)  # 永久删除

# Step 3: 推理时仍用原始 top-k
# 问题: 剩余 expert 数量减少，但 k 不变 → 每个 expert 处理更多 token
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NAEE (Lu et al., 2024) 从 calibration set 计算 expert 对输出的贡献度来排序，并额外提出 token-aware dynamic expert skipping，可在推理时跳过某些 token 的 expert 计算，但仅支持 top-k=2。MoE-I² (Yang et al., 2024) 进一步结合 inter-expert pruning + intra-expert low-rank decomposition (SVD)，用遗传算法搜索最优的逐层剪枝比例。LExI 实验表明，vLLM 上 pruning 的吞吐量提升有限甚至退化——因为稀疏 routing 不变，剩余 expert 负载增加导致长尾 latency。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference


## LExI (Layer-Adaptive Active Expert Allocation)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

LExI 是一种 data-free 的 post-training MoE 推理优化技术，核心思想是为预训练 MoE 模型的每一层静态分配不同的 active expert 数量（top-k_j），替代传统所有层统一的 top-k。LExI 通过两阶段 pipeline 实现：(1) Monte Carlo 敏感性分析：使用随机 Gaussian 输入计算每层在不同 top-k 下的 Frobenius 范数输出扰动；(2) 进化搜索：以扰动损失为 proxy，在总 active expert budget B 约束下搜索全局最优的逐层 k_j 分配。LExI 不删除任何 expert 参数，仅通过减少低敏感层的 active expert 数量来减少 FFN 计算量、inter-GPU 通信和 memory bandwidth 使用。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# LExI 两阶段 Pipeline

# === Stage 1: Per-Layer Sensitivity Profiling ===
输入: pretrained MoE model M, target top-k list T = [1, 2, ..., k_base]
输出: D[layer][k] = average Frobenius norm perturbation

for layer in range(L):
    for k in T:
        perturbations = []
        for iter in range(N_iter):
            X = randn(B, L_seq, H)  # ~ N(0,1)
            set_topk(M, k_base)
            Y_base = moe_forward(M, X)   # 只计算当前层
            set_topk(M, k)
            Y_k = moe_forward(M, X)
            Δ = ||Y_k - Y_base||_F       # Frobenius norm
            perturbations.append(Δ)
        D[layer][k] = mean(perturbations)

# === Stage 2: Evolutionary Search ===
输入: D, budget B, k_min, k_max
输出: k* = (k_1, ..., k_L)  # 每层最优 top-k

population = rand_feasible(N_pop, L, B)  # 满足 Σk_j = B
for gen in range(G_max):
    p1, p2 = tournament_select(population)  # min Σ D_j(k_j)
    offspring = uniform_crossover(p1, p2)   # 每层随机选父代
    offspring = mutate(offspring)            # ±1, ΣΔ = 0
    offspring = project(offspring, B)        # 保证 budget 约束
    population.append(offspring)
k* = argmin_{k in population} Σ_j D_j(k_j)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LExI 在 vLLM 推理框架上使用：加载预训练模型 → 运行一次 LExI profiling（Stage 1）+ search（Stage 2）→ 得到 k* → `set_topk(layer_j, k_j)` 修改每层路由参数 → 正常 vLLM 推理。LExI 是 data-free 的（仅用随机噪声 + 模型权重），不需要任何 calibration 数据集或微调。Budget B 是可控参数：B 越小吞吐越高但精度越低，B 越大越接近 baseline 精度。LExI 不减少模型显存占用，但可与 expert pruning 方法结合以实现 memory + computation 的联合优化。限制：(1) 不减少 memory footprint；(2) 对 k_base=1 的模型（如 Llama-4）不适用。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference


## Frobenius Norm Perturbation Profiling (Monte Carlo Sensitivity Analysis)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Frobenius Norm Perturbation Profiling 是 LExI 提出的用于评估 MoE 每层对 top-k 变化敏感度的 data-free 方法。核心原理：对于某一层，用不同的 top-k 值计算同一批随机 Gaussian 输入的输出，用 Frobenius 范数 ||Y_k - Y_base||_F 量化输出偏差。偏差越大，说明该层对该 k 值越敏感（减少 expert 会导致输出变化大）。重复 N_iter 次取平均以获得统计稳健的估计。整个过程仅需模型权重，无需真实数据。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Frobenius Norm Perturbation Profiling
# 对每个 MoE layer 独立执行

# 输入生成
X = randn(32, 128, 4096)  # [Batch=32, Seq=128, Hidden=4096]
                           # 标准正态分布 N(0,1)

# Baseline 输出 (pretrained k_base = 2)
set_topk(this_layer, 2)
Y_base = moe_forward(X)    # [32, 128, 4096]

# 扰动输出 (k=1)
set_topk(this_layer, 1)
Y_k1 = moe_forward(X)

# Frobenius 范数计算
# ||A||_F = sqrt(Σ_{i,j} A_{i,j}²)
Δ_k1 = ||Y_k1 - Y_base||_F
      = sqrt(sum((Y_k1[b,s,h] - Y_base[b,s,h])² 
                 for b in 0..31, s in 0..127, h in 0..4095))
# Δ_k1 是一个标量，反映 top-1 vs top-2 的输出总偏差

# 重复 N_iter > 1M 次，取平均获得稳健估计
D[layer][1] = mean(Δ_k1 over N_iter iterations)
```

LExI 实验显示：Mixtral-8x7B 浅层在 k=1 vs k=2 时扰动小（低敏感），深层扰动大（高敏感）；Qwen1.5-MoE 呈现相反模式，浅层更敏感；OLMoE 和 DeepSeekV2 呈钟形曲线（中间层最稳定）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在 PyTorch 中实现：`torch.norm(Y_k - Y_base, p='fro')`。选择 Frobenius 范数而非其他度量（如 L1/L∞/cosine similarity）的原因：Frobenius 范数在欧几里得空间中精确捕捉高维输出的 magnitude shift，Monte Carlo 采样确保对 diverse input 的泛化性。计算开销：仅需前向传播（无反向传播），且每层独立执行可并行化。对于 Mixtral-8x7B（32 MoE layers × ~8 k-values），profiling 在单 H100 上 <30 分钟完成。

涉及论文标题：
- LExI: Layer-Adaptive Active Experts for Efficient MoE Model Inference

## Locality-Sensitive Hashing (LSH) in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Locality-Sensitive Hashing (LSH) 是一种概率性降维方法，主要用高维空间的近似最近邻搜索。其核心思想是：将高维数据通过一组hash函数映射到低维"桶"（buckets）中，使得相似的数据以高概率被映射到同一个桶，而相异的数据以高概率被映射到不同桶。数学上，LSH hash函数h满足：P[h(x)=h(y)] = 1 − d(x,y)/D，其中d(x,y)是x和y之间的距离，D是空间的直径。

在MoE训练中，LSH被LSH-MoE论文创新性地用作一种在线聚类压缩技术：在all-to-all通信前，对每个expert接收的token集合使用LSH快速聚类，将相似的token归入同一cluster，然后仅传输各cluster的中心（centroid）而非全部token，从而大幅减少all-to-all通信量。压缩率由hash函数数量控制——hash函数越多，bucket越多，cluster越细，压缩率越低（即传输的数据越接近原始量）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

LSH-MoE中LSH在MoE层forward pass中的具体流程：

```
# 输入: X_i = {x_1, x_2, ..., x_N} 为分配给第i个expert的token集合
# LSH参数: hash函数数量H, 旋转矩阵R (d×d随机矩阵)

# Step 1: LSH聚类 — 将每个token映射到bucket
for each token x in X_i:
    # Cross-Polytope Hashing:
    # 将x用随机旋转矩阵R变换，然后映射到最近cross-polytope顶点
    hash_code = argmax_{j in {±1,...,±d}} |(R @ x)[j]|
    # hash_code ∈ [0, 2d-1] 标识cross-polytope的顶点

# Step 2: 按hash_code分组，同一bucket的token归为一个cluster
clusters = group_by_hash(X_i, hash_codes)

# Step 3: 计算每个cluster的中心（替代传输数据）
for each cluster_j:
    centroid_j = mean(cluster_j)                 # 聚类中心
    residuals_j = {x - centroid_j | x in cluster_j}  # 残差（本地保存）

# Step 4: 仅传输centroid集合替代完整token
C_i = {centroid_j for j=1..m}                   # m个centroid << N个token
send_via_alltoall(C_i)                            # 通信量: m*h << N*h

# Step 5: Expert对中心计算
E_centroids = Expert(C_i)

# Step 6: 接收结果并用残差恢复每个token的近似输出
receive_via_alltoall(E_centroids)
for each cluster_j, for each token k in cluster_j:
    Y_jk = E_centroids[j] + residuals_j[k]      # 残差补偿
```

压缩率 = m/N（cluster数量/原始token数量），由hash函数数量H控制。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

LSH在MoE训练中的实现：
- 通过PyTorch实现LSH聚类模块，hash计算为GPU上的矩阵运算（旋转矩阵乘法+argmax）
- 必须高效在线执行（因为待压缩数据是动态实时生成的，无法预压缩或重叠处理），因此选择cross-polytope hashing（O(d)复杂度）
- LSH聚类替代传统的K-Means等迭代聚类算法，因为K-Means的迭代特性不适合在线实时场景
- 默认使用6个hash函数（约20%压缩率时精度无损），可通过调整hash函数数量控制压缩率
- 计算开销远小于通信节省——LSH的矩阵运算相比all-to-all通信可忽略

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing


## Cross-Polytope Hashing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Polytope Hashing (CPH) 是Locality-Sensitive Hashing (LSH)的一种具体hash函数族，专为角距离（angular distance / cosine similarity）设计。其核心思想是将输入向量通过随机旋转后映射到cross-polytope（交叉多面体）的最近顶点。Cross-polytope是一个在d维空间中由2d个顶点组成的几何体，顶点为各坐标轴正负方向上的单位向量：{±e_1, ±e_2, ..., ±e_d}。

数学表示：
$$LSH(\mathbf{x}) = \operatorname{argmax}_{i \in \{\pm 1, \pm 2, \dots, \pm d\}} |\mathbf{R}\mathbf{x}|_{i}$$

其中R是随机旋转矩阵（或使用Fast Johnson-Lindenstrauss Transform加速到O(d log d)），|Rx|_i是旋转后向量第i个分量的绝对值。hash结果为选中分量的索引i（带正负号），共2d个可能的bucket。

CPH的渐近最优敏感度（sensitivity）ρ = 1/c²（c为近似因子），在理论上是角距离下LSH能达到的最优值。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

在LSH-MoE训练pipeline中，Cross-Polytope Hashing的执行流程：

```
# 输入: X = [N, h]  — N个token，每个h维
# 参数: R = [d, h]  — 随机旋转矩阵（或Fast JL transform）

# Step 1: 随机旋转（矩阵乘法）
Rx = X @ R.T                            # [N, d], O(N·h·d)

# Step 2: 取每个分量绝对值，找最大值索引
abs_Rx = |Rx|                           # [N, d]
hash_indices = argmax(abs_Rx, dim=-1)   # [N], 值域 [0, 2d-1]
# argmax返回的是维度的索引，正负号由Rx值的符号隐含决定

# Step 3: 按hash_indices分组
# 相同hash_indices的token属于同一cluster
clusters = group_by(hash_indices)
```

CPH的计算复杂度为O(N·h·d)（旋转矩阵乘法），相比迭代聚类（如K-Means的O(N·k·h·iter)）在在线场景下效率显著更高。

LSH-MoE在消融实验中比较了Cross-Polytope Hashing与Spherical-Plane Hashing (SP)，发现CP在相同压缩率下达到更好的模型收敛质量。原因：CP基于n维cross-polytope编码数据，对多种复杂数据模式更有泛化能力；而SP依赖球面和平面之间的几何关系，更适合球面分布特征的数据。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- PyTorch实现：旋转矩阵R可以是一个随机初始化的固定矩阵（不需要学习/训练），LSH本身没有可训练参数
- Fast Cross-Polytope LSH (Kennedy & Ward, 2016): 使用subsampled randomized Hadamard transform将矩阵乘法从O(d²)加速到O(d log d)，适合极高维场景
- 在LSH-MoE中，R矩阵在训练开始时随机初始化并固定，不参与梯度计算
- CPH也用于Reformer（Kitaev et al., 2020）利用attention的稀疏性——使用LSH将query和key分桶，仅计算同一桶内的attention

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing


## Residual-based Error Compensation for Communication Compression

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Residual-based Error Compensation（基于残差的误差补偿）是LSH-MoE框架中用于缓解通信数据压缩对模型精度影响的补偿方案。其核心思想是：在压缩阶段（all-to-all通信前）记录每个token与其所属cluster center的差异（残差），在解压后（expert计算完成后）将残差加回到expert对center的输出中，从而近似恢复每个token的完整输出。

数学公式：
1. 残差记录：$$\Delta \text{cluster}_j = \{x - \overline{\text{cluster}}_j \mid x \in \text{cluster}_j\}$$
2. 残差恢复：$$Y_{ij} = \{E(\overline{\text{cluster}}_j) + \Delta \text{Cluster}_{jk} \mid k = 1, 2, \dots, N_j\}$$

关键洞察：该方案利用expert FFN计算的近似线性性——对于cluster内相似的token，其expert输出也高度相似，因此用central的输出加token-specific残差可以很好地近似完整输出。

LSH-MoE实验证明，不使用error compensation时，在相同训练时间下perplexity高出0.3个点；使用error compensation后，模型质量与无压缩训练几乎一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# 在MoE layer forward pass中

# 阶段1: 压缩与残差记录
for each expert i:
    tokens_i = {x_1, x_2, ..., x_N}  # 分配给expert i的token
    clusters = LSH(tokens_i)          # LSH聚类
    centroids_i = []
    residuals_i = []
    for cluster_j in clusters:
        c_j = mean(cluster_j)                          # 聚类中心
        resid_j = {x - c_j for x in cluster_j}          # 残差
        centroids_i.append(c_j)
        residuals_i.extend(resid_j)

# 阶段2: 仅传输centroids（通信量：m·h << N·h）
E_centroids = Expert(all_to_all(centroids_i))  # expert计算
results = all_to_all(E_centroids)               # 传回

# 阶段3: 残差补偿恢复
for each cluster_j, for each k in 1..N_j:
    Y_jk = results[j] + residuals_j[k]           # 加上各自残差
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- 残差计算和存储都在本地GPU上进行，不增加额外通信
- 残差的存储开销为N·h（与原始token相同），但仅在本地保存，不通过网络传输
- 误差补偿的有效性源于"token similarity"——cluster内部的token非常相似，它们的expert输出也相似，所以线性残差近似精度足够
- 需要注意：该补偿是对中间激活值进行压缩的误差补偿，而非梯度压缩。激活压缩对误差的容忍度更低（因为误差会在后续层累积放大），因此误差补偿对保持模型质量至关重要
- 本质上是一种有损压缩的误差控制技术，类似压缩感知中的残差编码思想

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing


## Token Similarity in MoE Training

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Token Similarity（Token相似性）是指在MoE训练的all-to-all通信中，发送到同一个expert的token在hidden representation空间中呈现高度相似的现象。LSH-MoE论文通过Principal Component Analysis (PCA)降维可视化发现，输入到all-to-all通信的token形成明显的聚类结构（clustering phenomenon）。

Token相似性的来源被归因为两个主要因素：
1. 数据因素：真实世界数据遵循Zipf's Law，导致某些数据元素比其他元素更频繁出现，形成token表示的偏斜分布。
2. 模型结构因素：Transformer的attention机制会捕获和整合token间的上下文信息，从而在句子级别均质化（homogenize）token表示，增强共享语义关系。

这一观察是LSH-MoE方法的核心动机：因为token高度相似，所以可以用聚类中心替代完整token进行all-to-all传输，且仅损失少量信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Token相似性在MoE pipeline中的表现：

```
# 在MoE layer中，gate网络将token分配给不同expert
# 分配到同一expert的token集合X_i存在内在相似性

# 可视化token分布（论文Figure 4）:
# 对all-to-all通信中的token做PCA降维到2D
# 结果：token自然聚类成若干个group
# 每个cluster内的token在语义上相似（如共享类似的上下文）

# 利用token相似性压缩通信:
# 不传输所有N个token，而是:
# 1) 将N个token聚成m个cluster (m << N)
# 2) 仅传输m个cluster center
# 3) 接收端用center + 保存的残差恢复近似token
# 压缩率 = m/N ≈ 20% (6个hash函数时)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

- Token相似性不需要显式实现——它是被观测到的数据特征
- 利用方式：使用LSH等快速聚类方法将相似token分组，以group-level信息替代instance-level信息
- 在实践中，token相似性不仅存在于NLP模型（RoBERTa, GPT, T5），也存在于CV模型（Swin-MoE），说明这是MoE架构中通信数据的普遍特性
- 论文通过PCA可视化提供了token相似性的实验证据，但不需要在实际训练中对token做PCA分析——直接使用LSH聚类即可隐式利用此特性

涉及论文标题：
- LSH-MoE Communication-efficient MoE Training via Locality-Sensitive Hashing

## Recurrent Gradient in MoE Router

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Recurrent Gradient 是 RMoE 论文提出的概念，指引入跨层 GRU 后为 MoE router 提供的额外反向传播梯度路径。标准 SMoE 中 router 梯度仅来源于 expert weight score g_n 对 LM loss 的偏导和 load balance loss 的偏导。引入 GRU 后，第 i 层 GRU hidden state h_i 通过跨层连接 (h_{i-1} -> h_i -> h_{i+1}) 传递梯度，形成 Recurrent Gradient。消融实验验证：(1) RMoE + detach h_{i-1}（切断梯度但保留前向信息）：test BPC 从 1.116 退化到 1.133，甚至差于完全无跨层连接 RMoE-NP (1.123)，表明仅有前向信息不足；(2) RMoE-NP + routing logits residual（无 GRU 但有 logits 残差的梯度路径）：test BPC 1.124-1.126，优于纯 NP 但不如完整 RMoE；(3) 更深模型上 RMoE vs NP 的 gap 随深度增大，支持 Recurrent Gradient 缓解深层 router 梯度消失。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Recurrent Gradient 的梯度流分解

# 标准 SMoE router 梯度（仅 per-layer）
dL/dG_i = dL/dy_i * dy_i/dscore_i * dscore_i/dG_i

# RMoE router 梯度（含 Recurrent Gradient）
dL/dG_i = dL/dy_i * dy_i/dscore_i * dscore_i/dh_i * dh_i/dG_i   # 直接路径
         + dL/dy_{i+1} * dy_{i+1}/dscore_{i+1} * dscore_{i+1}/dh_{i+1}
           * dh_{i+1}/dh_i * dh_i/dG_i                            # Recurrent Gradient
         + ...  (更后层继续反向传播)

# 消融设置对比
# (a) RMoE:              h_i = GRU(x_i', h_{i-1})          # 完整
# (b) RMoE + detach:     h_i = GRU(x_i', h_{i-1}.detach()) # 有前向无梯度
# (c) RMoE-NP:           h_i = GRU(x_i', h_0)              # 无前向无梯度
# (d) RMoE-NP + r-α:     g_i += α * g_{i-1}                # logits残差梯度
```

结果 (Enwiki8 test BPC): SMoE=1.128, RMoE=1.116, RMoE+detach=1.133, RMoE-NP=1.123, RMoE-NP+r-0.5=1.124。关键洞察：仅有前向信息而无 Recurrent Gradient (detach) 甚至不如完全无跨层连接 (NP)，说明 Recurrent Gradient 是核心贡献者。本质原理与 ResNet 残差连接类似——GRU 为深层 router 创建了直接的梯度传播路径。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

Recurrent Gradient 不是手动实现的，而是 PyTorch autograd 引擎通过 GRU 跨层 hidden state 连接自动构建计算图并反向传播。实现要求：(1) forward 时不 detach h_{i-1}；(2) 跨层共享 GRU 参数（使 RNN cell 权重梯度从所有层累积）。该技术可推广：任何跨层路由连接（routing logits residual、attention-based cross-layer routing）都可能通过类似机制提供额外梯度路径，关键是确保跨层连接不被 detach 且连接权重可学习。

涉及论文标题：
- Layerwise Recurrent Router for Mixture-of-Experts

## Cross-Layer Mutual Information in MoE Routing

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Cross-Layer Mutual Information (MI) in MoE Routing 是 RMoE 论文提出的分析工具，用于量化 MoE 模型中不同层 router 之间共享的路由信息量。基于各层 router 输出的 gating probability distribution（softmax 后 N 个 expert 的概率分布）计算 pairwise MI：对每对层 (i,j)，将 token 的 routing distribution 离散化（100 bins）后使用 sklearn 的 mutual_info_score 计算 MI，形成 L×L 的 MI 矩阵。高 MI 值表示两层 router 决策高度相关。论文使用 MI 矩阵验证了 RMoE 的设计直觉：标准 SMoE/XMoE/HyperMoE 的跨层 MI 均很低（<0.1），router 层间独立决策；RMoE 的 MI 显著高于所有 baseline，验证 GRU 促进了跨层信息共享。该框架可推广用于评估任何跨层路由方法的有效性。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
# Cross-Layer MI 计算
import numpy as np
from sklearn.metrics import mutual_info_score

def discretize(prob_dist, bins=100):
    return np.digitize(prob_dist, bins=np.linspace(0, 1, bins))

def calc_cross_layer_mi(all_gate_probs):
    # all_gate_probs: shape (L, N_tokens, N_experts)
    L = all_gate_probs.shape[0]
    mi_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            mi_vals = [mutual_info_score(
                discretize(all_gate_probs[i, t]),
                discretize(all_gate_probs[j, t])
            ) for t in range(N_tokens)]
            mi_matrix[i, j] = np.mean(mi_vals)
    return mi_matrix
```

MI 矩阵解读（L×L heatmap）：(1) SMoE/XMoE/HyperMoE: 全体 MI < 0.1 → router 独立决策；(2) RMoE: 对角线附近 MI 高（邻近层信息共享强），远距离层 MI 逐渐衰减但仍 > 0；(3) RMoE-NP-r1.0: 仅对角线高，远距离层 MI 快速衰减 → logits 残差难以保证长程信息共享。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

该分析框架适用于：(1) 验证跨层路由方法是否建立了真正的信息共享（而非仅增加参数量）；(2) 对比不同跨层机制的传递效率（GRU vs logits residual vs attention-based）；(3) 诊断路由训练问题（异常高 MI 可能表示 representation collapse）。计算复杂度 O(L²·T)，在验证集上采样即可。注意 MI 受 expert 数 N 影响（N 越大基线越低），应在相同 N 下跨方法比较。

涉及论文标题：
- Layerwise Recurrent Router for Mixture-of-Experts

