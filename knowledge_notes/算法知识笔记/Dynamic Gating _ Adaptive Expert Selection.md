## Dynamic Gating / Adaptive Expert Selection

术语解释
动态门控是对固定Top-K路由的改进，根据每个token的输入特征自适应决定激活的专家数量，而非对所有token使用统一的K值。

术语是什么？
动态门控的核心思想是不同token的复杂度不同，应分配不同的计算资源：
- Li et al.：基于累积概率的自适应门控，当累加到一定阈值时停止激活更多专家
- DynMoE：top-down gating + 动态expert数量决定（per-token阈值）
- XMoE：使用更多但更小的专家，以阈值替代固定K
- AdapMoE：使用Fisher信息矩阵计算专家重要性，自适应跳过不重要的专家
- DA-MoE：从attention机制衍生token重要性预测，指导专家分配

从算法pipeline角度拆解术语。
```
# Dynamic Gating (累积概率阈值法)
def dynamic_gating(x, router_weight, threshold=0.9):
    logits = x @ router_weight
    probs = sort(softmax(logits), descending=True)
    cumsum = cumsum(probs)
    # 选择累积概率刚好超过阈值的expert集合
    K_dynamic = argmin(cumsum >= threshold) + 1
    return topk(probs, K_dynamic)

# DynMoE: per-expert threshold
def dynmoe_gating(x, router_weight, thresholds):
    # thresholds: learnable per-expert threshold
    logits = x @ router_weight
    probs = softmax(logits)
    # 每个expert独立判断是否激活
    active = probs > thresholds
    return active  # 变长expert集合
```
结果：FLOPs减少9%-75%，加速比1.32x-1.37x，同时保持模型性能。

术语一般如何实现？如何使用？
- 训练期间引入门控稀疏性损失以鼓励动态稀疏
- 推理时替换forward中的TopK为动态选择逻辑
- 需要处理变长expert输出的聚合（masked weighted sum）
- 可结合expert prefetching和cache管理使用

Ada-K 引入了一种全新的动态门控范式——基于强化学习的可学习 allocator。与基于阈值的方法不同，Ada-K 使用 PPO 训练一个独立的 allocator 模块（线性层）来采样每个 token 的最优专家数量 k*，并通过 activation regularization loss 直接最小化期望激活数。该方法在 4 个主流 MoE 模型上实现了 30%-40% 的专家激活减少同时提升性能，训练开销极小（<2M 参数，<8 GPU-hours）。

AdaMOE 引入了另一种动态门控范式——**null experts（空专家）**。与基于阈值或强化学习的方法不同，AdaMOE 保持传统 top-k 路由机制不变，但在 expert set 中引入 m 个零 FLOPs 的 null experts，并将 k 值增大。由于 null expert 不消耗任何计算资源，token 实际使用的 true expert 数量自然自适应（0~k 个），平均 true expert 负载通过 load balancing loss 中 null expert 的目标使用率和 annealing α 来间接控制。该方法无需额外训练模块或 RL，实现极其简单，在 Mixtral-8x7B 上实现 FLOPs 减少 14.5% 同时 accuracy 提升 1.69%。

**Adaptive Gating（阈值差值门控，Li et al. EMNLP 2023）**：
Li et al. (EMNLP 2023) 提出了第四种动态门控范式——**基于概率差值的阈值门控**。与上述三种方法不同，Adaptive Gating 不需要额外的训练模块（无 allocator、无 null expert、无阈值网络），仅利用现有 gate network 的 softmax 输出：

引入固定阈值 T（默认 0.1）。对每个 token，计算 router 输出 R = Softmax(x · W_G)，比较 top-1 与 top-2 概率：
- 若 R_1 - R_2 ≤ T → top-1 与 top-2 概率接近，token 需要双专家 → 激活 2 个 expert
- 若 R_1 - R_2 > T → top-1 概率显著偏斜，token 仅需单专家 → 激活 1 个 expert

$$\text{dispatch}(x) = \begin{cases} \{i, j\} & \text{if } R_1 - R_2 \leq T \\ \{i\} & \text{otherwise} \end{cases}$$

关键实证发现：≥55% 的 token 中 top-1 概率显著偏斜，这些 token 仅需单 expert。该方法的独特优势在于：
- **零额外训练参数**：无 allocator、null expert 或阈值网络，仅需一个超参数 T
- **实现极简**：仅修改 gate forward 中的 dispatch 逻辑（if-else 判断）
- **训练即推理一致**：training 和 inference 使用相同门控策略（vs 传统 top-2 train / top-1 infer）
- **配合 curriculum learning**：按 token 复杂度重排训练数据，减少 batch 内 top-2 token 比例不均造成的等待

实验：6 个 NLP 任务（BERT-Base/BART-Large/GPT-2/DialoGPT/FSMT on SST-2/WMT19/SQuAD/CNN-DM/WikiText/SODA），T=0.1 或 0.2 最优，最多减少 22.5% 训练时间。Sentiment analysis 仅 11.3% token 使用 top-2，FLOPs 3.28G→2.30G（↓30%）。Dialogue response 最高 23.4% token 使用 top-2。

**四种动态门控范式对比**：

| 方法 | 动态机制 | 额外参数 | 训练方式 | 实现复杂度 |
|------|---------|---------|---------|-----------|
| Ada-K | RL allocator 采样 k* | W_alloc (~1M) | PPO RL + warm-start | 中 |
| AdaMOE | Null experts + 增大 k | Router 维度扩展 (m) | Standard + annealing α | 低 |
| AdaMoLE | 可学习阈值 τ(x) | W_τ, b_τ (single layer) | Standard BP | 中 |
| Adaptive Gating | 固定阈值 T 比较 top-1 vs top-2 | 0 | Standard | 最低 |

**AdaMoLE 的 Dynamic Threshold Network（动态阈值网络）**：
AdaMoLE 引入第三种动态门控范式——**learnable per-token threshold**。与 Ada-K 的 RL allocator 和 AdaMOE 的 null expert 不同，AdaMoLE 在 MoE router 旁增加一个轻量级阈值网络（单层 Linear + Sigmoid），为每个 token 生成自适应阈值 τ ∈ [0, τ_max]：

$$\tau = \tau_{max} \cdot \sigma(W_{\tau} x + b_{\tau})$$

其中 τ_max = 1/N（N 为专家数）。然后激活所有 p_i ≥ τ 的专家。关键创新在于使用 (p_i - τ) 替代原始 p_i 进行加权，使阈值网络可通过反向传播端到端学习：

$$y = \sum_{i=1}^{N} \frac{\mathbb{1}(p_i \ge \tau)(p_i - \tau)}{\sum_{j=1}^{N} \mathbb{1}(p_j \ge \tau)(p_j - \tau)} \cdot E_i(x)$$

**AdaMoLE 与 LoRA 的结合**：在 fine-tuning 场景中，每个 expert E_i 是一个 LoRA adapter (B_i A_i)，基础权重 W_0 冻结：

$$h = W_0 x + \sum_{i=1}^{N} \frac{\mathbb{1}(p_i \ge \tau)(p_i - \tau)}{\sum_{j=1}^{N} \mathbb{1}(p_j \ge \tau)(p_j - \tau)} \cdot B_i A_i x$$

**AdaMoLE 的三个关键行为**：
1. **简单 token → 高 τ → 少专家激活**：τ 接近 τ_max，仅 0-2 个专家满足 p_i ≥ τ
2. **复杂 token → 低 τ → 多专家激活**：τ 接近 0，可能 4-8 个专家被激活
3. **极端情况**：τ 太高导致无专家激活时，由于 Σ p_i = 1，必存在 p_i ≥ 1/N = τ_max，保证至少 1 个专家激活

**实验结果**：Llama-2-7B + AdaMoLE (τ∈[0, 1/N]) vs MoLE top-2：CommonsenseQA 78.71% vs 77.15% (+1.56%)，平均激活 3.46 vs 2.0 专家。在 Gemma-7B 和 Llama-2-13B 上也一致超越 baseline。

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models
- AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts
- Adaptive Gating in Mixture-of-Experts based Language Models
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

**DSMoE 的 Sigmoid 门控（第五种动态门控范式）**：
DSMoE 引入了第五种动态门控范式——**Sigmoid 独立门控 + STE 梯度**。与上述四种方法不同，DSMoE 完全放弃 softmax 归一化，改用 sigmoid 使每个 expert 独立判断是否激活：

$$gate_i(x) = \sigma(x \cdot Y_i) = \frac{1}{1 + e^{-x \cdot Y_i}}$$

$$active_i(x) = \begin{cases} gate_i(x) & \text{if } gate_i(x) > \tau \\ 0 & \text{otherwise} \end{cases}$$

关键设计特征：
1. **非归一化门控**：sigmoid 各 expert 决策互不依赖（vs softmax 的竞争性选择），允许 0~n 任意个 expert 同时激活
2. **STE 确保梯度流**：S(x) = sg(G(x)) + x - sg(x) 使未通过阈值的 expert 门控参数 Y_i 在反向时仍接收梯度 ∂h/∂Y_i = (ĥ)^T · (o_i · σ'(ĥY_i))，解决死 expert 问题
3. **无 load balancing loss**：不引入传统 MoE 的负载均衡约束，让模型自由学习稀疏模式
4. **稀疏损失替代**：使用 L1 正则 Σ G(σ(ĥY_n)) 施加稀疏压力，与门控梯度形成对抗博弈

vs 其他范式的对比：
- vs Ada-K（RL allocator）：DSMoE 无需额外 RL 训练模块，端到端反向传播
- vs AdaMOE（null experts）：DSMoE 不引入占位 expert，直接使用硬阈值控制激活
- vs Adaptive Gating（固定阈值 T）：DSMoE 使用 STE 保证梯度流，训练和推理的 gate 行为不一致（训练用 STE，推理用纯硬阈值）

---
