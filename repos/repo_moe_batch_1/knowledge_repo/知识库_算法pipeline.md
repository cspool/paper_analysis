# 知识库_算法pipeline

## Token Condensation (令牌凝聚)

术语解释
Token Condensation 是 LUFFY 分布式 MoE 训练系统中提出的通信优化技术：在 MoE 的 dispatch phase 中，识别被路由到同一 expert 的高度相似 token，仅传输 representative token，其余 token 在 combine phase 使用 representative 的 expert 输出替代，从而消除冗余的跨 GPU 通信。

术语是什么？
Token Condensation 基于两个关键观察：(1) 在 MoE 训练中，被路由到同一 expert 的 token 之间存在显著相似性——MoE-TransformerXL 中约 62% 的 token 对相似度超过 0.75，MoE-BERT-Large 第六 block 中约 57% 的 token 对相似度超过 0.55；(2) token 相似度在通过 expert 后高度保留——约 95% 的 token 对在 expert 计算后相似度变化小于 0.2。

核心流程：
1. 将 token 建模为全连接图（node=token, edge weight=similarity）
2. 通过 Fast Similarity Measurement 快速计算边权重
3. 根据 Adaptive Threshold h_t 删除低相似度边
4. 在每个连通分量中保留 degree 最高的代表性 token
5. Dispatch 阶段仅传输 representative tokens
6. Expert 计算仅处理 representative tokens（减少计算量）
7. Combine 阶段使用 representative 的输出替代被凝聚 token 的输出

从算法pipeline角度拆解术语：
Token Condensation 在 MoE 训练 pipeline 中的位置和执行流程：

```
=== MoE Layer Forward Pass with Token Condensation ===

输入: token embeddings X [N, d] after self-attention

Step 1 - Gate Routing (标准):
    gate_logits = X @ W_gate        # [N, num_experts]
    gate_probs = SoftMax(gate_logits)
    expert_ids = TopK(gate_probs, k=2)  # 每 token 选 top-2 experts

Step 2 - Token Graph Construction:
    G = (V, E) where V = {tokens}, E = all pairs
    for each edge (u, v):
        # 2a: 不同 expert → 边权重=0 (直接跳过)
        if expert_ids[u] != expert_ids[v]:
            weight[(u,v)] = 0
        
        # 2b: 历史相似度查找 (O(1))
        elif s_prev[(u,v)] > S1: weight[(u,v)] = 1  # 极端相似
        elif s_prev[(u,v)] < S2: weight[(u,v)] = 0  # 极端不相似
        
        # 2c: 计算真实余弦相似度 (仅不确定的少量对)
        else:
            weight[(u,v)] = cosine(X[u], X[v])

Step 3 - Adaptive Threshold:
    l_norm = (loss_ini - loss_prev) / loss_ini
    h_t = 1.0 / (1.0 + exp(l_norm))
    # 早期: h_t ≈ 0.73 (保留大部分 token)
    # 后期: h_t ≈ 0.27 (凝聚更多 token)

Step 4 - Graph Pruning & Component Selection:
    删除 weight < h_t 的边 → 稀疏图
    对每个连通分量:
        rep = argmax(degree(node))  # 保留连接最多的 token
        token_to_token[node] = rep  # 其他 token 映射到代表

Step 5 - Condensed Dispatch:
    for expert in experts:
        tokens_to_send = {rep}  # 仅 representative tokens
        all_to_all_send(tokens_to_send, target_gpu)

Step 6 - Reduced Expert Computation:
    # 更少的 tokens → 更少的 FLOPs
    expert_out = expert_ffn(received_tokens)  # [N' << N, d]

Step 7 - Expanded Combine:
    for token in all_tokens:
        if token in token_to_token:
            output[token] = expert_out[token_to_token[token]]
        else:
            output[token] = expert_out[token]
```

术语一般如何实现？如何使用？
- 实现基于 DGL (Deep Graph Library) 构建 token 图，利用 GPU 加速图操作
- Fast Similarity Measurement 将 O(N²·d) 的 naive pairwise 计算降低到仅少量不确定对需真实余弦计算
- Adaptive Threshold 通过 sigmoid 函数将 loss 下降量映射为 [0, 1] 区间的阈值
- 参数 S₁ 和 S₂ 控制历史相似度判断的激进程度：减小 S₁ 增加凝聚率但可能影响收敛，增大 S₂ 保留更多 token 但减少通信节省
- 局限性：仅适用于训练阶段（需访问 token embeddings 和 gate 输出）；token 相似度假设在推理时可能不成立

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---

## Fast Similarity Measurement (快速相似度测量)

术语解释
Fast Similarity Measurement 是 LUFFY 系统中 Token Condensation 的核心子算法，用于在 MoE 训练中高效识别被路由到同一 expert 的相似 token。它通过三层过滤策略将 O(N²·d) 的全对余弦相似度计算降低到仅少量不确定 token 对需要真实计算。

术语是什么？
Naive 方法的计算复杂度为 O(N²·d)，其中 N 可达数千，d 为 token embedding 维度（如 1024），在每次 training iteration 的每个 MoE block 都执行是不可行的。Fast Similarity Measurement 利用两个关键观察：
1. 被路由到不同 expert 的 token 极不可能相似（不同 expert 设计为处理不同类型输入）
2. 在连续 block 间，极端相似（s_{b-1} > S₁）或极端不相似（s_{b-1} < S₂）的 token 对会维持其模式——约 90% 的高相似 token 对在后续 block 保持相似

三步过滤策略：
- **Step 1: Expert Activation Filter** → 过滤掉约 (E-1)/E 的 token 对（被路由到不同 expert 的）
- **Step 2: Historical Similarity Lookup** → O(1) 查找前一 block 的相似度缓存，极端情况直接判定
- **Step 3: Real Cosine Calculation** → 仅对剩余高度不确定的 token 对计算 cos(u,v) = (u·v) / (||u||·||v||)

从算法pipeline角度拆解术语：
```
Algorithm: Fast Similarity Measurement

Input:  tokens X ∈ R^{N×d}, expert_ids ∈ Z^N, 
        prev_sim_cache (from block b-1), params S1, S2

Output: similarity graph G with edge weights

1. 初始化全连接图 G: N nodes, N(N-1)/2 edges

2. for each edge (i, j) in parallel:
     # Layer 1: Expert Activation Filter
     if expert_ids[i] != expert_ids[j]:
         G[i][j].weight = 0
         continue  # ~(E-1)/E 的边在此过滤
     
     # Layer 2: Historical Similarity Lookup
     s_prev = prev_sim_cache.get((i, j))
     if s_prev is not None:
         if s_prev > S1:  # 极端相似 (e.g., S1=0.8)
             G[i][j].weight = 1
             continue
         if s_prev < S2:  # 极端不相似 (e.g., S2=0.2)
             G[i][j].weight = 0
             continue
     
     # Layer 3: Real Cosine Similarity (仅 ~10-20% 的剩余边)
     u, v = X[i], X[j]
     G[i][j].weight = dot(u, v) / (norm(u) * norm(v))

3. return G
```

术语一般如何实现？如何使用？
- 每个 GPU 维护独立 CUDA stream 运行相似度计算，与 expert computation 并行
- 历史相似度缓存在 GPU 内存中，跨 block 复用
- S₁ 和 S₂ 参数需根据模型调整：减小 S₁→更多 token 被标记为相似→更多凝聚但可能误判；增大 S₂→更多 token 被标记为不相似→更保守
- 适合 MoE 训练场景，因 expert activation 提供了天然的 token 分组信号

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---

## Adaptive Token Condensation Strategy (自适应令牌凝聚策略)

术语解释
Adaptive Token Condensation Strategy 是 LUFFY 中用于动态调整 Token Condensation 相似度阈值 h_t 的策略。它根据训练收敛状态自动平衡通信效率与训练稳定性：训练早期使用高阈值保留更多 token 以保证收敛，训练后期降低阈值以最大化通信节省。

术语是什么？
固定阈值的问题：h=0.3 时 MoE-BERT-Large 的 F1 从 90.82 降至 85.41（显著精度损失），h=0.8 时 F1 为 88.29（仍低于 Vanilla 的 90.82）。自适应策略使用 sigmoid 函数将归一化 loss 下降量映射为阈值：

$$h_t = \frac{1}{1 + \exp(l_{norm})}, \quad l_{norm} = \frac{l_{ini} - l_{t-1}}{l_{ini}}$$

其中 l_{ini} 是第一个 training iteration 的 loss，l_{t-1} 是前一个 iteration 的 loss。

从算法pipeline角度拆解术语：
```
Algorithm: Adaptive Threshold Computation

在每个 training iteration t:
    l_norm = (loss[0] - loss[t-1]) / loss[0]
    h_t = 1.0 / (1.0 + exp(l_norm))
    
    # 行为分析:
    # t=0:    l_norm ≈ 0    → h_t ≈ 0.73  (保留 ~73% token)
    # t=mid:  l_norm ≈ 0.5  → h_t ≈ 0.38  (凝聚更多)
    # t=late: l_norm ≈ 2.0  → h_t ≈ 0.12  (大量凝聚)

    # 在 Token Condensation 中使用:
    for each subgraph in G:
        # 删除 weight < h_t 的边
        keep edges where weight >= h_t
        # 连通分量分析
        for each component:
            rep = token with max degree
            condense all others into rep
```

术语一般如何实现？如何使用？
- 使用指数函数（sigmoid）确保当 loss 下降趋于平缓（训练稳定期）时阈值变化也是平滑的
- 无需额外超参数调优，完全由 loss 信号驱动
- 可与任何基于相似度阈值的 token 选择/剪枝策略结合
- 适用于训练阶段；推理阶段无 loss 信号，需替代策略

涉及论文标题：
- Communication-Efficient Sparsely-Activated Model Training via Sequence Migration and Token Condensation

---

## Multimodal Native Model (多模态原生模型)

术语解释
Multimodal Native Model 是 ARIA 提出的可量化定义：指一个单一模型在多模态输入（文本、代码、图像、视频）上具有强大的理解能力，且其性能匹配或超过同规模的单模态专用模型。核心特征是用户无需区分不同模态的输入，模型无缝处理和整合多模态信息。

术语是什么？
ARIA 给出了 multimodal native 的量化标准：一个 multimodal native model 在所有输入模态上的性能应匹配或超过类似容量的 modality-specialized models。这不同于简单的"多模态模型"（可能在不同模态上性能不均衡），也不同于通过 upcycling 从 dense model 转成 multimodal MoE 的方法。

关键设计原则：
1. **从零开始的多模态预训练（not upcycling）**：不依赖 dense checkpoint 初始化，language 和 multimodal 数据混合从头训练
2. **Modality-Generic Architecture**：不设计 modality-specific expert，所有 expert 对所有模态通用，expert specialization 在训练中自然涌现
3. **统一的 next-token prediction objective**：visual tokens 和 text tokens 使用相同的自回归 loss

从算法pipeline角度拆解术语：
ARIA 的 multimodal native pipeline：

```
Stage 1 - Language Foundation:
  MoE decoder only, 6.4T language tokens, 8K context
  → 建立通用知识和语言理解

Stage 2 - Multimodal Integration:
  Visual encoder + MoE decoder 联合训练
  1T language + 400B multimodal tokens
  → 多模态理解能力，维护语言能力

Stage 3 - Long Context Extension:
  33B tokens (69% long sequences)
  RoPE theta: 100K → 5M, context: 8K → 64K
  → 长视频/多页文档理解

Stage 4 - Instruction Following:
  20B high-quality QA data, LR annealing
  → 指令遵循和对齐
```

术语一般如何实现？如何使用？
- 开源实现：Aria (github.com/rhymes-ai/Aria), Apache 2.0
- 模型变体：Aria-Base-8K, Aria-Base-64K, Aria-Chat
- 推理：HuggingFace Transformers 或 vLLM，单 A100 80GB 即可 bf16 全模型推理
- 微调：支持 LoRA（单 GPU）和 Full parameter（8×A100 + DeepSpeed ZeRO）
- 对比 proprietary multimodal native models (GPT-4o, Gemini-1.5)，关键区别是训练配方的透明度

涉及论文标题：
- Aria An Open Multimodal Native Mixture-of-Experts Model

---

## United Experts (United Expert Model / 联合专家模型)

术语解释
United Experts 是 BrownoutServe 提出的一种 MoE 模型蒸馏机制：通过知识蒸馏将每层多个原始 expert 的知识合并到单个同参数规模的 "united expert" 中，减少推理时的 expert 访问次数，提升 GPU 利用率和降低延迟。

术语是什么？
United Experts 的核心思想：在 MoE 推理中，大量 token 被路由到少数 hot experts，而多数 cold experts 只处理极少量 token，GPU SM 大量空闲。United Experts 将冷门 experts 的 token 聚合到 united expert 中批量处理，增大 effective batch size。

给定每 Transformer 层有 m 个原始 routed experts，按 way=k 分组得 ⌈m/k⌉ 组。每组训练一个 united expert：以组内 k 个原始 experts 为 teacher（输出 hidden states），united expert 为 student，MSE loss 最小化两者的 hidden states 差异：

$$\mathcal{L}_{\text{MSE}}^{j} = \frac{1}{k} \left( \sum_{i=0}^{k-1} \left\| H_{u}^{j} - H_{o}^{j \times k+i} \right\|^{2} \right)$$

其中 H_u^j 是第 j 个 united expert 的 hidden states，H_o^{j×k+i} 是第 j 组中第 i 个原始 expert 的 hidden states。训练完成后，每个 united expert 具备等价于 k 个原始 experts 的综合知识表达。

从算法pipeline角度拆解术语：
United Experts 的完整 pipeline 分为离线训练和在线推理两阶段：

**离线训练（以 Qwen1.5-MoE-A2.7B, m=60, k=8 为例）**：
```
# 每层 60 个原始 experts → ⌈60/8⌉ = 8 个 united experts
for each transformer layer:
    for group_j in [0, 7]:   # 8 groups
        UE_j = init_expert(hidden_dim)  # 与原 expert 同参数规模
        
        for batch in training_data:
            # Teacher: k=8 个原始 experts
            H_o_list = []
            for i in [0, 7]:
                expert_idx = j * 8 + i   # expert 0-7 for group 0
                H_o_list.append(Expert_{expert_idx}(batch))
            
            # Student: 1 个 united expert
            H_u_j = UE_j(batch)
            
            # MSE Loss
            loss = (1/8) * sum(||H_u_j - H_o||^2 for H_o in H_o_list)
            loss.backward()
            optimizer.step()

# 保存每层的 ⌈m/k⌉ 个 united expert 权重
```

**在线推理 - Partial-Brownout 调用**：
```
# BrownoutMoE forward path
输入: token hidden states x_t, threshold=0.7

# 1. Gate routing
for each token t:
    scores[t] = softmax(TopK(x_t @ E_centroids, K=2))

# 2. Token 统计与划分
expert_counts = [count tokens per expert]  # 60 个 experts
A = sort experts by count descending
S1 = experts with cumulative count >= total_tokens * threshold  # hot experts
S2 = experts for united expert processing                       # cold experts

# 3. S1: 原始 experts FFN
for e in S1:
    outputs += FFN_e(tokens_e)

# 4. S2: United experts FFN
for group in group_experts(S2, k=8):  # 按 index 分组
    concat_tokens = concat(all tokens in this group)
    outputs += UE_{group_id}(concat_tokens)
```

术语一般如何实现？如何使用？
- **训练方式**：知识蒸馏（teacher=原始 experts，student=united expert），使用 MSE loss。训练数据与下游任务相关。
- **存储**：United expert 权重与原 expert 等参数规模，需常驻 GPU 显存或在 CPU memory/disk 间切换。way=k 越大，united expert 数越少但每个 united expert 集成的知识越多（精度可能越低）。
- **代码**：BrownoutServe 开源（https://github.com/beyondHJM/BrownoutServe），但预训练 United Expert 权重需联系作者获取。
- **适用场景**：MoE 模型（Qwen1.5-MoE 等）、多 expert（≥8/layer）的模型效果更好。

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

## BrownoutMoE Architecture (BrownoutMoE 架构)

术语解释
BrownoutMoE 是 BrownoutServe 提出的 MoE 模块变体，在标准 MoE（含 shared experts 和 routed experts）基础上引入 united experts 和 brownout token routing，通过动态选择部分 token 走原始 experts、部分 token 走 united experts 来平衡精度和延迟。

术语是什么？
BrownoutMoE 的输出公式为（Eq. 5）：

$$\mathbf{h}_{t} = \mathbf{x}_{t} + \sum_{i=1}^{N_{s}} \text{FFN}_{i}^{(s)}(\mathbf{x}_{t}) + \sum_{i=1}^{N_{r}} p_{i,t} \text{FFN}_{i}^{(r)}(\mathbf{x}_{t}) + \sum_{i=1}^{N_{u}} q_{i,t} \text{FFN}_{f(i)}^{(u)}(\mathbf{x}_{t})$$

其中 N_s、N_r、N_u 分别为 shared experts、routed experts（原始）、united experts 的数量。p_{i,t} 为 token t 属于 S1（原 experts 处理）时对 expert i 的 routing weight，q_{i,t} 为 token t 属于 S2（united experts 处理）时的 routing weight。

与标准 MoE 的区别：第三项 Σq_{i,t}·FFN_{f(i)}^{(u)}(x_t) 是新增的 united expert 路径，f(i) 将原始 expert index 映射到其所在的 united expert group。这种设计使得即使是 cold expert 的 token 也能得到近似处理而非被忽略。

从算法pipeline角度拆解术语：
BrownoutMoE 的完整 forward 路径：
```
输入: x_t (token hidden state)

# 1. Gate 计算（与标准 MoE 相同）
for each token t, expert i:
    s_{i,t} = x_t^T @ e_i               # affinity score
g_{i,t} = softmax(TopK({s_{j,t}}, K))  # routing weight

# 2. Brownout 划分（新增）
expert_token_counts = count tokens routed to each expert
A = sort experts by token_count descending
T = total_tokens * threshold
S1 = top experts whose cumulative token count ≥ T
S2 = remaining experts

# 3. Shared expert（同标准 MoE）
h_shared = Σ FFN_i^{(s)}(x_t)

# 4. Routed experts - 分两路（Brownout 关键创新）
# Path A: S1 tokens → 原始 experts
h_original = Σ p_{i,t} * FFN_i^{(r)}(x_t)   # p_{i,t} = g_{i,t} if i∈S1 else 0

# Path B: S2 tokens → united experts
h_united = Σ q_{i,t} * FFN_{f(i)}^{(u)}(x_t) # q_{i,t} = g_{i,t} if i∈S2 else 0

# 5. 输出
h_t = x_t + h_shared + h_original + h_united
```

术语一般如何实现？如何使用？
- 实现约 5.5k 行 Python，基于 PyTorch。MoE 相关算子使用 Triton 重写。
- 与 DeepSeekMoE 等架构的区别：DeepSeekMoE 使用 shared experts + fine-grained routed experts，BrownoutMoE 在此基础上增加 united experts 作为第三类 expert 组件。
- 适用场景：可集成到任何 MoE transformer 模型中。

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs

## Knowledge Distillation for MoE Expert Merging (MoE 专家知识蒸馏合并)

术语解释
通过知识蒸馏将多个 MoE expert 的知识压缩到一个参数规模更小的模型中，使得推理时只需访问合并后的 expert，减少计算开销。BrownoutServe 的 united expert 训练是这一范式的典型应用。

术语是什么？
MoE expert 知识蒸馏合并的核心流程：
- **Teacher**: 一组原始 experts（k 个），各自拥有独立参数
- **Student**: 一个 united expert，参数规模与单个原始 expert 相同
- **Distillation target**: 最小化 student 与各 teacher 输出的 hidden states 之间的 MSE
- **结果**: Student 学会近似 k 个 teachers 的综合行为，推理时用 1 次 expert 访问替代 k 次

与标准知识蒸馏（Hinton et al. 2015）的区别：标准 KD 通常用于压缩一个大模型到小模型，而 MoE expert merging 是在模型内部横向合并多个同规模的 experts 到一个等价规模的 expert，保持推理路径的参数规模不变，但减少 expert 访问次数。

从算法pipeline角度拆解术语：
```
# 训练阶段（offline）
for each expert group in each transformer layer:
    # group = {Expert_a, Expert_b, Expert_c, Expert_d}  (k=4)
    UE = init_expert(same_param_size_as_one_expert)
    
    for x in distillation_dataset:
        # 收集所有 teacher 输出
        teacher_outputs = [Expert_i(x) for Expert_i in group]
        
        # Student 前向
        student_output = UE(x)
        
        # MSE 蒸馏损失
        loss = mean(||student_output - teacher_outputs[i]||^2 for i in range(k))
        loss.backward()
    
    save(UE)  # 保存 trained united expert

# 推理阶段（online）
# 原路径：token → expert_a(少量token) + expert_b(少量token) + ...
#          → 多次小batch kernel launch，GPU利用率低
# 蒸馏后：token → UE(合并后的token batch) → 1次大batch kernel，GPU利用率高
```

术语一般如何实现？如何使用？
- 训练数据：需准备蒸馏数据集，包含模型需要处理的典型输入分布
- 损失函数：MSE（BrownoutServe）、KL divergence（标准 KD）、或组合 loss
- 与相关方法的区别：MoDE（mutual distillation among experts）、KDEM（KD-enhanced expert merging）等进一步探索了 expert 间的相互蒸馏
- 限制：united expert 的参数容量固定，way=k 越大（合并越多 experts）精度损失越大

涉及论文标题：
- BrownoutServe: SLO-Aware Inference Serving under Bursty Workloads for MoE-based LLMs


## Task-level Routing / Task-MoE (任务级路由)

术语解释
Task-level Routing 是 MoE 路由的一种粒度策略，根据 task identity（如 multilingual NMT 中的语言对或目标语言）而非输入 token 内容来决定 expert 选择，使同一 task 的所有 token 路由到相同的 expert 子集。由 Kudugunta et al. (EMNLP Findings 2021) 在"Beyond Distillation"中提出。

术语是什么？
Task-level routing 将 MoE 的 gating function 从 GATE(x_s)（per-token）改为 GATE(task_id_s)（per-task）。在 MNMT 中，task_id 可以是 target language（French→English 和 German→English 共享 "English" experts）或 language pair（各自独立）。公式：

$$\mathcal{G}_{s,E} = \mathrm{GATE}(\mathrm{task\_id}_s)$$

这与 token-level routing GATE(x_s) 和 sentence-level routing GATE(mean(x_{1:S})) 形成三种路由粒度。核心价值在于**推理效率**：task-level routing 使每个 task 仅需加载 K 个 experts（K=2 for top-2），而非全部 E 个 experts，从而避免了 token-level MoE 的模型并行和 all-to-all 通信开销。

从算法pipeline角度拆解术语。
```
# Task-level MoE Forward (MNMT, decoder)
def task_moe_decoder_forward(x_s, task_id):
    # task_emb 是可学习的 task embedding table
    task_emb = task_embedding_table[task_id]  # e.g., "French"
    logits = router(task_emb)                 # GATE(task_emb), NOT GATE(x_s)
    G = TopK(Softmax(logits), k=2)            # 所有 token 共享相同 G
    y_s = sum(G[e] * FFN_e(x_s) for e in top_k_indices)
    return y_s

# Hybrid strategy (best performer in paper):
# Encoder: token-level routing (flexibility for source language processing)
# Decoder: task-level routing (decoder dominates inference cost, 200x per step)
```

推理时子网络提取：server 仅预加载 task-specific 的 K 个 experts（如 expert 5 + expert 17 for French），不同 task 可在不同设备上独立并行解码。WMT: decoder params 221M→25M (↓88%); large-scale: 6.5B→201M (↓97%)。

术语一般如何实现？如何使用？
- 仅适用于 task boundary 明确的 multi-task 场景（如 multilingual NMT），不适用于通用单任务 LLM
- Hybrid 策略（Token encoder + Task decoder）效果最佳：encoder 保持 per-token 灵活性处理多语言源输入，decoder 用 task-level 路由降低推理成本
- Task boundary 选择：target language（同一目标语言的所有源语言共享 experts，最大化 transfer）vs language pair（各自独立，最大化 specialization）
- 实现基于 GShard 框架（TensorFlow/Lingvo），在 router 输入侧用 task_embedding 替代 token_embedding

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

---

## Sub-network Extraction from MoE

术语解释
Sub-network Extraction 是从大型稀疏 MoE 模型中按任务或条件提取仅包含部分 experts 的子网络直接用于推理部署的方法，与知识蒸馏（将 MoE 压缩为稠密模型）形成对比。

术语是什么？
与传统 MoE 推理（需加载全部 E 个 experts）不同，sub-network extraction 利用路由策略（如 task-level routing）使特定任务仅需要少量 experts，从而提取 sub-network 独立部署。核心公式：推理时 decoder 参数从 ΣE（全部 experts）降至 K（每 task 激活的 experts）。

Kudugunta et al. (2021) 的关键发现：蒸馏 token-MoE→dense 仅保留 32% BLEU 增益，而 task-MoE sub-network extraction 保留 **100%** BLEU 增益（且 decoder 参数量更小：25M vs 142M distilled dense model）。

从算法pipeline角度拆解术语。
```
# Token-MoE Inference (baseline): 需全部 E experts
for each decoding step:
    y_s = sum(TopK(Softmax(GATE(x_s)), k=2)[e] * FFN_e(x_s))
    # 不同 token → 不同 experts → 需加载全部 E experts + all-to-all 通信

# Task-MoE Sub-network Extraction (proposed):
# Step 1: 根据 task_id 确定 sub-network experts
task_experts = TopK(Softmax(GATE(task_emb[task_id])), k=2)  # e.g., {5, 17}

# Step 2: 仅加载 sub-network experts 到加速器
load_experts({FFN_5, FFN_17})  # K=2 vs E=32/128

# Step 3: 解码，所有 token 使用相同 experts
for each decoding step:
    y_s = G[5] * FFN_5(x_s) + G[17] * FFN_17(x_s)
    # 无 all-to-all，无跨设备通信
```

术语一般如何实现？如何使用？
- 适用于 task boundary 明确的 multi-task 场景
- 实现需修改 MoE router 接受 task_id 作为输入（而非 token embedding）
- 多 task 并行：不同 task 的 sub-networks 分配到不同设备，独立解码
- 与蒸馏的关系：sub-network extraction 是蒸馏的上位替代（当 task boundary 已知时），保留 100% MoE 增益，而蒸馏仅保留 ~32%

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

---

## Multilingual Neural Machine Translation (MNMT)

术语解释
MNMT 是用单一神经网络模型同时翻译多个语言对的范式，本质上是 multi-task learning 问题。参数共享的程度决定正迁移（positive transfer）的程度，过度共享则导致任务干扰（task interference）因容量瓶颈。

术语是什么？
在 MNMT 中，参数可以在不同语言对之间完全共享（如 Johnson et al., 2017 的 Google Multilingual NMT），也可以部分共享、部分专用。MoE 模型天然适合 MNMT：不同 experts 可以学习不同语言的专业知识，router 根据输入语言动态分配计算资源。Kudugunta et al. (2021) 的 Task-MoE 利用 MNMT 的 task boundary 天然先验，将 "翻译到 French" 和 "翻译到 German" 定义为不同 task。

从算法pipeline角度拆解术语。
MNMT 中参数共享的程度谱系：
- 完全共享（all-shared）：单一 encoder-decoder，所有语言使用相同参数 → 最大 transfer 但容量瓶颈
- 语言特定（language-specific）：每语言独立 encoder/decoder → 无 transfer 但无干扰
- MoE（本论文）：共享非 expert 参数 + task-specific experts → 在 transfer 和 specialization 之间平衡
- Task-level MoE：task 级 expert 选择 → 推理时可提取 task-specific sub-network

实际设置：15-102 种语言，to/from English，温度采样（T=5）处理数据不平衡（150k—64M 句对）。

术语一般如何实现？如何使用？
- SentencePiece 共享词汇表（64k tokens），源句前 prepend `<2xx>` token 指示目标语言
- BLEU (SacreBLEU) 评估
- 数据采样策略（温度 T 控制高/低资源语言平衡）
- Adafactor optimizer，inverse sqrt LR schedule

涉及论文标题：
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

---

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

## Top-K Routing / Gating Mechanism

术语解释
Top-K路由是MoE模型中最常用的专家选择策略，对每个token选择门控概率最高的K个专家进行处理，其余专家不参与计算。

术语是什么？
路由（Gating/Routing）机制决定每个输入token应该激活哪些专家。Top-K路由的具体计算过程：
1. 门控网络：θ = Softmax(R(x))，R通常是线性层W_r·x
2. Top-K选择：E_selected = TopK(θ, K)，保留概率最高的K个
3. 稀疏化：将未选中专家的权重置零，仅K/N的专家参与计算

K的典型取值为1~8。K=1（top-1 routing）最大稀疏性但可能性能不足；K=2（top-2 routing）是常用折中（如Mixtral-8x7B）。更细粒度的MoE（如DeepSeek-V2）使用更小的专家和更大的K和N。

从算法pipeline角度拆解术语。
```
def topk_gating(x, router_weight, K, N):
    # x: [batch, seq_len, d_model]
    # router_weight: [d_model, N]
    logits = x @ router_weight           # [batch, seq_len, N]
    probs = softmax(logits, dim=-1)      # [batch, seq_len, N]
    topk_vals, topk_idx = topk(probs, K, dim=-1)
    # 归一化选中的权重
    topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
    # 未选中的专家权重置零
    mask = zeros_like(probs)
    mask.scatter_(-1, topk_idx, topk_vals)
    return mask
```
固定Top-K的问题：所有token使用相同数量的专家，无法根据token复杂度自适应分配——简单token浪费计算，困难token可能计算不足。

术语一般如何实现？如何使用？
- 实现：通常为nn.Linear(d_model, N)，后接Softmax+TopK
- 辅助负载均衡损失：防止某些expert被过度使用或完全不被使用
- 容量因子：限制每个expert处理的token数上限，防止热点

Adaptive Gating (Li et al., EMNLP 2023) 从实证角度揭示了固定 top-2 的浪费：≥55% 的 token 其 top-1 与 top-2 概率差异显著，这些 token 仅需单 expert。固定 top-2 为所有这些 token 浪费了 1 个 expert 的 FLOPs，且训练时 all-to-all 通信量也翻倍。

Ada-K 论文定量分析了固定 Top-K 的具体缺陷：Mixtral-8x22B 降低 k=2→1 导致平均准确率下降 15.80 点，Mixtral-8x7B 降低 7.68 点。Ada-K 通过动态路由在减少 34.4% 专家激活的同时提升性能 +0.77。

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs
- Adaptive Gating in Mixture-of-Experts based Language Models
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

**Task-MoE 的路由粒度分类（Kudugunta et al., EMNLP 2021）**：
路由决策可在三个粒度级别：
1. **Token-level**（标准）：GATE(x_s)，每 token 独立选择 experts → 推理时需全部 experts
2. **Sentence-level**：GATE(mean(x_{1:S}))，整句共享相同 experts → 效果较差
3. **Task-level**（提出）：GATE(task_id)，同 task 所有 token 共享 experts → 允许 sub-network extraction

Hybrid 策略（Token encoder + Task decoder）在 WMT 上 BLEU 最高（23.6 vs Token/Token 22.6），decoder 推理成本占 200x encoder step time。

---

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

## Expert-Shift Problem

术语解释
Expert-Shift 是 MoE-LLM 低比特量化后出现的路由器输出偏移问题：量化引入的噪声导致 MoE router 的 expert 选择概率分布发生偏移，使模型选错 expert，造成显著的性能退化。这是 MoE 量化中除权重重构误差之外的第二个关键退化因素。

术语是什么？
在 EAC-MoE 论文中，作者通过受控实验分离了两个退化因素：(1) 量化本身导致的权重重构误差；(2) 量化引起的 expert-shift。实验将 FP16 模型和 3-bit 量化模型在 WikiText-2 上记录所有输入的 expert 选择及对应分数，交叉施加条件后测量 PPL：

| 条件 | Mixtral-8x7B PPL | Deepseek-moe-16b PPL |
|------|-----------------|---------------------|
| FP16 + 正确选择 | 3.84 | 6.51 |
| FP16 + Expert-Shift | 4.17 | 6.76 |
| 3-bit + 正确选择 | 4.21 | 6.81 |
| 3-bit + Expert-Shift | 4.65 | 7.17 |

结果表明：(1) 单纯的权重重构误差使 PPL 从 3.84 升至 4.21（+0.37）；(2) Expert-Shift 更进一步恶化至 4.65（+0.44，占总退化的 ~54%）；(3) 即使 FP16 模型被强制使用量化模型的错误 expert 选择，PPL 也显著退化（3.84→4.17）。

Expert-Shift 的根因：量化后的 MHSA 和已量化 expert 的激活值（x̂）与原全精度激活值（x）存在偏差，导致 router 计算 W_r·x̂ 偏离 W_r·x，Softmax 后的概率分布改变，top-K 选择结果发生变化。由于逐层传播，expert-shift 会在深层累积放大。

从算法pipeline角度拆解术语：
```
=== Expert-Shift 如何发生（逐层传播）===
输入: token x (FP16)
Layer l:
    1. x_out = Quantized_MHSA(x)        # MHSA 被量化，输出激活有偏差
    2. logits = Router_W @ x_out         # [num_experts]，但 x_out 已含噪声
    3. probs = Softmax(logits)            # 概率分布因噪声偏移
    4. selected = TopK(probs, K)          # 可能选错 expert（shifted experts）
       # 95.9% 的 shifted expert 仍在 top-16 概率内（64 expert 中）
    5. output = Σ probs[i] * Quantized_Expert_i(x_out)  # 用错误的 expert 计算
    → 第 l+1 层继承错误的 hidden state，expert-shift 继续传播
```

术语一般如何实现？如何使用？
- Expert-Shift 是 MoE-LLM 量化特有的退化机制，dense LLM 不存在此问题
- 量化位宽越低，expert-shift 越严重（2-bit >> 3-bit >> 4-bit）
- EAC-MoE 的 QESC 方法通过 TopK-MSE Loss 逐层校准 router 来缓解 expert-shift
- 在量化校准中保持 router 全精度（router 仅占 <0.03% 参数，不增加显著内存开销）
- 量化 MHSA 的位宽从 2→4→8 bit 提升会显著降低 expert-shift rate（MHSA 4-bit 以上变化趋缓）

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

## TopK-MSE Loss

术语解释
TopK-MSE Loss 是 EAC-MoE 中用于校准 MoE router 的损失函数，在计算 router 输出的 MSE 时，仅对 top-K 最高概率的 expert 计算损失，而非所有 N 个 expert。其核心动机是：量化后 shifted expert（全精度选中但量化后未选中）95.9% 仍排名在 top-16 概率内（64 expert 中），但 top-16 的 MSE 损失仅占全部 N 个 expert MSE 损失的 29.25%。如果对所有 expert 计算 MSE，损失会被大量低概率 expert 的噪声主导，优化过程难以聚焦于真正重要的 expert 对齐。

术语是什么？
TopK-MSE Loss 的公式：

$$\mathcal{L} = \frac{1}{K} \sum_{i \in \mathrm{top-}K(W_r x)} ((W_r x)_i - (W_r \hat{x})_i)^2$$

其中 $W_r$ 是 router 权重矩阵，$x$ 是全精度模型输入，$\hat{x}$ 是量化后模型的输入（经过量化 MHSA 及已量化 expert 的激活值）。通过仅对全精度 router 输出的 top-K 高的 expert 计算损失，优化器聚焦于对齐"更可能被选中的 expert"的 router 输出。

K 值通过网格搜索确定：对 Phi3.5-moe（16 expert, top-2）K=8；对 Deepseek-moe-16b-base（64 expert, top-6）K=20；对 Qwen1.5-MoE-A2.7B（60+4 expert, top-4）K=20。K 值过小（接近 per-token selection count）会过拟合；过大则退化为全量 MSE（噪声主导）。

从算法pipeline角度拆解术语：
```
=== TopK-MSE Loss 计算与 Router 校准 ===
输入: router权重 W_r, 全精度输入 x, 量化后输入 x_hat, 超参数 K
输出: 校准后的 router 权重

# 1. 前向计算
logits_full = W_r @ x        # [num_experts], 全精度参考
logits_quant = W_r @ x_hat   # [num_experts], 量化后实际输出

# 2. 确定 top-K 索引（基于全精度参考，非量化后）
topK_indices = arg_top_k(logits_full, K)  # 仅关注最重要 K 个 expert

# 3. TopK-MSE Loss
loss = 0
for i in topK_indices:
    loss += (logits_full[i] - logits_quant[i])^2
loss = loss / K

# 4. 反向传播更新 W_r，对齐量化后 router 输出与全精度参考
# 低概率 expert 的偏差被忽略，避免噪声主导优化
```

术语一般如何实现？如何使用？
- 在逐层量化校准过程中使用：量化每层 MHSA 后计算 TopK-MSE Loss 校准该层的所有 MoE router
- K 值选择需网格搜索（在 MMLU 上评估不同 K 值的效果）
- 相比全量 MSE Loss：Phi3.5-moe 2.06-bit 下准确率 65.03%（TopK-MSE）vs 64.52%（MSE）；Deepseek-moe 57.05% vs 55.91%
- 高量化位宽（如 3.03-bit）时 K 值敏感性低（expert-shift rate 本身较低）
- 可与任何基于 GPTQ 的 MoE 量化方法正交结合

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

## QESC (Quantization with Expert-Selection Calibration)

术语解释
QESC 是 EAC-MoE 提出的 MoE-LLM 静态量化方法，核心思想是在标准 GPTQ 权重量化之外，逐层校准 MoE router 以缓解低比特量化引起的 expert-shift 问题。与 PMQ/BSP 等基于 expert 使用频率分配混合精度的策略不同，QESC 不依赖静态校准集确定 expert 重要性（避免跨任务过拟合），而是直接对齐量化前后 router 的输出，确保模型仍能为当前任务选对 expert。

术语是什么？
QESC 的量化流程：
1. **逐层处理**：从第 0 层到第 L-1 层顺序量化
2. **MHSA 量化**：每层 MHSA 量化为 4-bit（group-wise asymmetric, group_size=128, GPTQ）
3. **Router 校准**：使用 WikiText2 校准集（128 条 × 2048 tokens）前向传播，记录全精度 router 输出的 top-K expert 分布作为标签；获取通过量化 MHSA 和已量化 expert 的激活值作为输入；用 TopK-MSE Loss 更新 router 权重对齐输出
4. **Expert 量化**：将该层所有 expert 量化为 B-bit（GPTQ, group-wise, asymmetric）
5. **Router 保持精度**：Router 权重保持 FP16（仅占 <0.03% 参数）

位宽配置：MHSA 4-bit，expert 2/2.5/3-bit，最终平均位宽 2.06/2.54/3.03-bit。2.5-bit 设置下前半层 expert 分配 3-bit，后半层分配 2-bit。

从算法pipeline角度拆解术语：
```
=== QESC 逐层量化与校准 ===
For layer l in [0..L-1]:
    # Step 1: 量化该层 MHSA
    W_attn_q = GPTQ_quantize(W_attn, bits=4, groupsize=128)
    
    # Step 2: 获取校准输入
    for each calibration sequence:
        x_l = Forward(model_quantized[:l], input)   # 到当前层的 hidden state
    
    # Step 3: 对当前层的每个 MoE router 进行校准
    for each MoE_router at this layer:
        y_full = router_W @ x_l                     # 全精度参考
        x_hat_l = Forward_with_quantized_MHSA(x_l)  # 量化后激活
        
        # TopK-MSE Loss
        topK = arg_top_k(y_full, K_l)  # K_l 通过网格搜索确定
        loss = mean((y_full[i] - (router_W @ x_hat_l)[i])^2 for i in topK)
        router_W = optimizer_step(router_W, loss)
    
    # Step 4: 量化该层所有 expert
    for each expert e at layer l:
        W_expert_q[e] = GPTQ_quantize(W_expert[e], bits=B, groupsize=128)

输出: 量化后 MoE 模型（MHSA 4-bit, experts B-bit, router FP16）
```

术语一般如何实现？如何使用？
- 使用 GPTQ 作为底层量化框架，QESC 在 GPTQ 基础上增加 router 校准步骤
- 量化过程在单张 A100 40G GPU 上执行；router 校准开销仅 ~2% 总时间（如 Mixtral-8x7B: GPTQ 1.30h + Calibration 0.02h）
- 使用 BitBLAS 处理量化后权重的混合精度 BLAS 操作实现 GPU 加速
- 在 3.03-bit 下，Mixtral-8x7B 和 Deepseek-moe-16b-base 的准确率几乎无损（<0.5%），可实际部署
- QESC 理论上与其他减少量化误差的方法（如 QuaRot、SmoothQuant）正交兼容
- 相比 BSP/PMQ：QESC 不依赖静态 expert 频率分配位宽，跨任务泛化性显著更好（详见论文 Table 9 过拟合分析）

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

## PESF (Pruning based on Expert-Selection Frequency)

术语解释
PESF 是 EAC-MoE 提出的 MoE-LLM 动态专家剪枝方法，在推理时基于当前输入序列中每个 expert 被选中的频率，动态剪枝不频繁被选的 expert，直接跳过其全部计算。与 EES/ODP 等逐 token 剪枝低权重 expert 的方法不同，PESF 从 expert 粒度（而非 token 粒度）进行剪枝，可实现更显著的加速。

术语是什么？
PESF 的核心机制：
- **剪枝阈值**：对每层 MoE（N 个 expert，每 token 选 K 个），序列长度 l，剪枝阈值 $c < \frac{l \times K}{N} \times \alpha$，其中 $\alpha \in (0, 1]$ 是超参数
- **直觉**：如果某 expert 被选中的次数低于"均匀选择期望值 × α"，说明它对当前任务不重要
- **动态性**：基于当前序列实时统计（非静态先验），适应不同任务类型的 expert 偏好
- **限制**：仅在 prefill 阶段使用（需要多个 token 的统计信息），不适用于逐 token 的 generate 阶段

两个操作点：(1) α=0.3（保守）：几乎无损准确率（<0.5%），加速 1.08-1.14×；(2) α=0.7（激进）：加速 1.30-1.47×，准确率下降~1.5%。Mixtral-8x7B 对激进剪枝敏感（expert 选择更均衡，稀疏性弱），仅适合 α=0.3。

从算法pipeline角度拆解术语：
```
=== PESF 动态 Expert 剪枝 ===
输入: 输入序列 seq[l], MoE 模型, 阈值 α
超参数: N (expert数量), K (per-token选择数)

For each MoE layer:
    # Phase 1: 统计阶段（prefill）
    c = [0] * N                           # expert 选择计数
    for each token t in seq:
        logits = router_W @ h_t
        probs = Softmax(logits)
        selected = TopK(probs, K)
        for expert_id in selected:
            c[expert_id] += 1
    
    # Phase 2: 剪枝决策
    threshold = (l * K / N) * α           # 均匀期望 × α
    active_experts = []
    for i in range(N):
        if c[i] >= threshold:
            active_experts.append(i)
        # else: expert i 被跳过
    
    # Phase 3: 仅计算未剪枝的 expert
    for each token t:
        logits = router_W @ h_t
        probs = Softmax(logits)
        selected = TopK_over_active(probs, K, active_experts)  # 仅在 active set 中选
        output = Σ norm_probs[i] * ExpertFFN_i(h_t) for i in selected
```

术语一般如何实现？如何使用？
- 完全在线、无训练：仅需一次额外的遍历统计 expert 选择计数，延迟开销可忽略
- 可与 QESC 量化组合使用（EAC-MoE = QESC + PESF），在 3.03-bit 量化基础上额外获得 1.09-1.13× 加速
- 核心依据：同一任务类别内 expert 选择频率高度相似（cosine similarity >0.8），因此序列级统计能准确反映任务偏好
- 对比 EES/ODP：PESF 从 expert 角度剪枝，直接跳过整个 expert 计算（而非仅减少某 expert 的部分输入），加速比更显著
- 局限：仅适用 prefill（generate 阶段仅单个 token，无法统计频率）；Mixtral-8x7B expert 选择分布较均匀，不适合激进剪枝

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

## Expert Selection Frequency

术语解释
Expert Selection Frequency 是 MoE 模型推理期间统计的各 expert 被 router 选中的频率分布，用于分析 MoE 的任务偏好、sparsity 特性，以及指导 expert 压缩策略（量化位宽分配、剪枝决策）。

术语是什么？
EAC-MoE 论文的核心发现：
1. **任务内相似性**：同一任务类别（QA/CR、Math、Code、特定语言）的不同数据集，expert 选择频率 pairwise cosine similarity >0.8
2. **任务间差异性**：不同任务类别之间 expert 选择频率 cosine similarity 显著较低
3. **稀疏性**：少数 expert 被高频选中（>30%），多数 expert 很少被选（<1%），但"重要 expert"的身份因任务而异
4. **结论**：不能用单一静态校准集确定 expert 重要性——对 QA/CR 重要的 expert 可能对 Code 不重要

计算方式：对 MoE layer m，数据集 d，统计 expert i 被选中次数 C(m,d,i)，归一化后展平为向量 P(d)，计算 cosine similarity。

从算法pipeline角度拆解术语：
```
=== Expert Selection Frequency 统计 ===
输入: 数据集 D, MoE 模型 (L 个 MoE layer, N 个 expert/layer)
输出: 每层每个 expert 的选择频率分布

For each layer m in [0..L-1]:
    C[m] = [0] * N                              # selection counts
    For each sequence in D:
        For each token t:
            logits = router_W @ h_t
            selected = TopK(Softmax(logits), K)  # K = top-K per token
            for expert_id in selected:
                C[m][expert_id] += 1
    # 归一化
    total = sum(C[m])
    P[m] = C[m] / total                         # P[m][i] = freq of expert i

# 展平所有层为一维向量用于 similarity 计算
P_flat(D) = concat([P[0], P[1], ..., P[L-1]])
Sim(D_i, D_j) = cosine_similarity(P_flat(D_i), P_flat(D_j))
```

术语一般如何实现？如何使用？
- 在 MoE 推理时自然产生（只需记录 router 的 TopK 选择），无额外计算开销
- 应用：(1) 指导 PESF 动态剪枝阈值；(2) 分析模型对任务的 specialization 程度；(3) 检测 expert 负载均衡情况
- PMQ/BSP 用此频率决定混合精度位宽分配，但 EAC-MoE 证明了这会导致跨任务过拟合
- Mixtral-8x7B 的 expert 选择分布更均匀（稀疏性弱），Deepseek-moe-16b-base 有 64 expert 则稀疏性更强

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

## Expert Pruning (MoE)

术语解释
MoE专家剪枝是减少MoE模型中专家参数量的技术，通过移除不重要的专家（structured pruning）或专家权重（unstructured pruning）来降低模型大小和计算量。

术语是什么？
由于专家占MoE模型参数的绝大部分（如Mixtral-8x7B中占96%），专家剪枝是MoE压缩的首要目标：
- **结构化剪枝（Structured Pruning）**：直接移除整个专家，减少专家数量
  - TSEP：针对下游任务移除非专业专家，只保留和微调专业专家
  - NAEE：在小校准集上评估专家组合，最小化精度损失
  - SEER-MoE：使用heavy-hitters计数法进行专家剪枝
- **非结构化剪枝（Unstructured Pruning）**：剪除专家内部的权重
  - MoE-Pruner：基于magnitude × activation × router weight的剪枝准则
- **专家合并（Expert Merging）**：将多个相似专家合并为一个
  - MC-SMoE：基于路由策略分组后合并
  - HC-SMoE：层次聚类合并，无需重训练
- 混合方法：STUN结合结构化+非结构化剪枝；MoE-Compression统一框架
- **C-PRUNE (Cluster-driven Expert Pruning)**：两阶段自适应剪枝框架。Phase 1 在每层内用 hierarchical agglomerative clustering 按 expert 参数相似度（cosine affinity）将功能冗余的 expert 分组成 cluster；Phase 2 跨所有层建立 unified importance score 进行全局剪枝，考虑深层 expert 更同质的趋势（depth penalty）。剪枝后 expert 通过 **parameterized expert merging**（affinity-weighted averaging）合并，路由权重同步更新。20% pruning rate 下 DeepSeek-V2-Lite 15.7B→13.0B，MMLU 仅降 1.4%。
- **DiEP (Differentiable Expert Pruning)**：将专家选择从离散搜索转化为连续优化。定义 intra-layer importance α 和 inter-layer importance β，通过交替梯度优化（3:1 比例）学习 per-expert 和 per-layer 的重要性权重。全局排序 s_i^(l) = α·β 后统一剪枝，自动实现 non-uniform pruning。仅需 128 calibration samples（C4），额外参数 ~0.01%。Mixtral 8×7B 50% sparsity 下保留 92% 性能，pruning time 0.23h（vs NAEE 1.31h exhaustive search）。NeurIPS 2025。

从算法pipeline角度拆解术语。
```
# Structured Expert Pruning (TSEP-style)
def prune_experts(model, task_dataset, target_sparsity):
    # 1. 评估每个expert对下游任务的重要性
    for each expert e in MoE layers:
        importance[e] = evaluate_on_task(model \ {e}, task_dataset)
    # 2. 移除最不重要的expert
    keep = topk(importance, ceil(N * (1-sparsity)))
    # 3. 微调保留的expert
    finetune(model, task_dataset, keep)
    return model

# Unstructured Expert Pruning (MoE-Pruner)
def moe_pruner_weight_score(W_expert, x_activation, router_weight):
    # W: [d_ffn, d_model]
    # score = |W| ⊙ mean(|x|) ⊙ router_weight (per output neuron)
    score = abs(W_expert) * mean(abs(x_activation), dim=0) * router_weight
    mask = score > percentile(score, sparsity)
    return W_expert * mask

# C-PRUNE: Cluster-Driven Expert Pruning
def c_prune(model, D_calib, K_layer, K_global, R_target):
    # Phase 1: Layerwise Expert Clustering
    for layer l in model.moe_layers:
        # Step 1: Compute expert embeddings φ(f_i)
        phi = []  # shape: [N, d]
        for expert f_i in layer.experts:
            # Average expert output over calibration samples
            phi_i = mean([f_i(x_k)/K for x_k in D_calib], dim=0)
            phi.append(phi_i)
        # Step 2: Affinity matrix A (cosine similarity)
        A = zeros(N, N)
        for i, j in pairs:
            A[i,j] = sigmoid(alpha * cos_sim(phi[i], phi[j]))
        # Step 3: Hierarchical agglomerative clustering
        clusters = agglomerative_cluster(A, n_clusters=K_layer)
        # Step 4: Parameterized merging within clusters
        for cluster C_k in clusters:
            omega = softmax([gamma * A[i, center] for i in C_k])
            theta_merged = sum(omega_i * theta_i)
    # Phase 2: Global Cluster Pruning
    scores = []
    for layer l, cluster c:
        # Depth penalty: deeper layers get lower scores
        score = importance(c) / (1 + beta * depth_penalty(l))
        scores.append((l, c, score))
    prune = bottom_k_percentile(scores, R_target)
    for (l, c) in prune:
        remove_cluster_from_layer(l, c)
    return model
```
结果：C-PRUNE 在 20% pruning rate 下显著优于 Random（MMLU avg 16.28→44.94）、Seer Prune（28.76）和 Group&Merge（32.03），接近 Base 模型（45.58）。50%-96.875%的稀疏率，结构化剪枝可直接减少expert数量从而减少激活参数。

术语一般如何实现？如何使用？
- 通常需要校准数据集或下游任务数据指导剪枝
- 部分方法需要微调恢复精度（如TSEP），部分无需微调（如EEP、HC-SMoE）
- C-PRUNE 使用 task-specific calibration data 计算 expert embedding（φ(f_i)），剪枝后可选择 task-specific fine-tuning 恢复精度
- 剪枝后的MoE模型可直接在标准框架中推理（structured pruning减少expert数，unnstructured需稀疏kernel支持）
- C-PRUNE 开源：https://github.com/Fighoture/MoE_unsupervised_pruning

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

**补充（来自 EAC-MoE 的 PESF）**：PESF 是一种在线动态 expert 剪枝方法，从 expert 粒度（而非 token 粒度）剪枝。推理时统计当前序列的 expert 选择频率，将选择次数低于阈值 (l×K/N)×α 的 expert 完全跳过。α=0.3 时几乎无损（<0.5% 准确率），α=0.7 时加速 1.3-1.47×。关键创新：基于"同一任务类别内 expert 偏好高度相似"的观察动态统计，而非使用静态先验。仅适用 prefill 阶段。详见 PESF 和 Expert Selection Frequency 词条。

**补充（来自 EEP 的 Gradient-Free Evolutionary Expert Pruning）**：EEP (Efficient Expert Pruning) 引入了一种全新的 expert 剪枝范式——使用无梯度进化策略搜索最优剪枝模式，无需任何参数更新即可在仅支持推理的设备上执行。核心创新包括：

- **参数化搜索空间**：引入 Router Mapping 矩阵 WRM ∈ R^{E'×E} 和 Expert Merging 矩阵 WEM ∈ R^{E'×E}，将剪枝决策转化为矩阵元素搜索问题。两矩阵在 Pruning Phase 中约束为 one-hot rows（每行仅一个元素为1），且 WRM = WEM，确保仅选择/保留 experts。Router 变换：G' = WRM·softmax(ZW_G)（E→E' 维路由权重降维）。
- **进化搜索流程**：(a) 随机初始化 one-hot 矩阵构成初始种群 P；(b) 每轮按 fitness F(W·Θ)（下游任务准确率）排名，选 Top M_CP 个体进入候选父代集 CP；(c) Crossover：随机组合两个父代的 expert 维度；(d) Mutation：随机替换 pruned expert 为其他 expert（Pruning Phase）或加 Gaussian noise（Merging Phase）；(e) 迭代 Epochs 轮。
- **两阶段设计**：Pruning Phase（40 iterations，仅选择最佳 expert 子集，不更新参数）→ Expert Merging Phase（160 iterations，WRM/WEM 解耦后从离散 0/1 过渡到连续值，通过 weighted sum 合并 knowledge）。
- **反直觉发现**：剪枝 50% experts 在 SQuAD 上准确率从 53.4% 升至 75.4%（不做任何参数更新）。原因：Router 从 8 个 experts 的复杂划分简化为 4/2 个 experts 的决策，re-normalized routing weights 使路由更精准。
- **双重使用场景**：减少 total experts（节省显存：8→4 减 47%，8→2 减 71%）和减少 active experts（Top-2→Top-1：prefill 加速 1.63×）。
- 代码开源：https://github.com/imagination-research/EEP

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

---

## Expert Quantization (MoE)

术语解释
MoE专家量化是将MoE模型中expert的高精度权重（FP16/FP32）转换为低精度表示（INT8/INT4/INT2/INT1）的技术，以显著减少内存占用和数据传输量。

术语是什么？
MoE量化的核心挑战是expert之间重要性不均，需为不同expert分配不同的量化策略：
- **MC-MoE**：基于expert的访问频率φ_i、激活权重w_i和量化损失ε_ij构建整数规划模型，为每个expert分配最优位宽（1/2/3 bit）。目标函数：min Σ φ_i^α · w_i^β · (ε_ij · x_ij)^γ
- **MoE-CSP**：将expert权重量化为4或8 bit，设计专用CUDA kernel处理量化权重+浮点计算
- **MoQE**：观察到expert的FFN层量化到2 bit对模型质量影响小，而self-attention量化显著损害性能
- **QMoE**：极致压缩至1 bit，实现可扩展压缩算法和自定义GPU kernel
- **CMoE**：二值权重网络（1 bit权重）+ 4 bit激活量化
- **HOBBIT**：动态精度选择——根据gating输出计算expert重要性分数，低于阈值用低精度版本，高于阈值用高精度版本
- **EdgeMoE**：通过校准数据集统计分析确定每个expert的最优位宽

从算法pipeline角度拆解术语。
```
# MC-MoE: Adaptive Bit-width Allocation
for each expert e_i in layer:
    φ_i = n_i / N                    # 访问频率
    w_i = sum(σ_j) / N               # 平均激活权重
    for bit j in {1, 2, 3}:
        W_q = quantize(W_i, j)       # 量化到j bit
        ε_ij = ||W_i - W_q||_F       # Frobenius范数量化损失

# 整数规划求解最优位宽分配
min Σ_i Σ_j φ_i^α · w_i^β · (ε_ij · x_ij)^γ
s.t. Σ_j x_ij = 1, x_ij ∈ {0, 1}
```
结果总结（Table 3）：
- 内存减少：4x-150x（取决于位宽和方法）
- 精度损失：0%-23.81%
- 推理加速：0.95x-26x（取决于是否有专用kernel）

术语一般如何实现？如何使用？
- 训练后量化（PTQ）：在小型校准集上确定量化参数，无需重训练
- 量化感知训练（QAT）：训练中模拟量化效果，精度更高但成本大
- 需要专用反量化kernel才能真正实现加速（否则仅节省内存，不加速计算）
- 结合offloading使用效果更佳——低精度expert加载延迟更低

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

**补充（来自 EAC-MoE）**：EAC-MoE 的 QESC 方法提出 MoE 量化特有的 **expert-shift** 退化机制——量化后 router 因激活噪声而选错 expert，其退化程度可达与权重误差同量级。QESC 通过逐层 TopK-MSE Loss 校准 router 而非依赖静态 expert 频率分配位宽，避免了 PMQ 等方法的跨任务过拟合问题（详见 Expert-Shift Problem 和 QESC 词条）。

**补充（来自 Compression Error Sensitivity Analysis）**：该论文将量化作为 baseline 对比方法，汇总了 MC-MoE/MoE-CSP/MoQE/QMoE/CMoE/MoE-MPTQS/HOBBIT/EdgeMoE 等八种 MoE 量化方案的性能数据（Table 1），指出低比特量化（1-4 bit）的共同缺陷是引入不可控、不可预测的误差，导致生成质量不稳定。这一观察直接驱动了用 error-bounded lossy compression（SZ3/CuSZp）替代量化来压缩 MoE expert 的提议——error-bounded 方法通过有界误差保证实现精度可控的压缩。

---

## Expert Distillation (MoE)

术语解释
MoE知识蒸馏是将大型MoE教师模型的知识迁移到更小的学生模型（可以是更小的MoE或稠密模型）的技术，以在保持性能的同时减少模型大小和推理成本。

术语是什么？
MoE蒸馏的两种主要范式：
1. **MoE→MoE蒸馏**：保持MoE结构但减少专家数/参数
   - DeepSpeed-MoE：使用阶段式知识蒸馏创建PR-MoE的蒸馏版本MoS
   - LLaVA-MoD：结合MoE结构和两阶段蒸馏（mimic distillation + preference distillation）训练小多模态模型
2. **MoE→Dense蒸馏（Sparse to Dense）**：将稀疏MoE转换为稠密模型
   - OneS：两阶段——Knowledge Gathering（求和/平均/top-k/SVD四种聚合方法合并专家）+ Knowledge Distillation
   - Switch Transformers：压缩97%参数后，稠密模型仍保留30%+性能
   - ELSM：稠密学生模型可匹配甚至超越稀疏教师性能
   - MoE-KD：用最频繁使用的expert初始化学生FFN，然后逐层蒸馏

从算法pipeline角度拆解术语。
```
# MoE → Dense Distillation (OneS)
# Step 1: Knowledge Gathering - 合并所有expert
W_merged = aggregate({E_1, E_2, ..., E_N})
# 聚合方法选择：sum, average, top-k, SVD

# Step 2: Knowledge Distillation
for each training sample x:
    y_teacher = MoE_teacher(x)   # 原MoE模型输出
    y_student = Dense_student(x) # 合并后的稠密模型
    
    # KL散度损失
    L_KD = KL(softmax(y_teacher/T), softmax(y_student/T))
    # 也可结合task loss
    L_total = α * L_KD + (1-α) * L_task
```

术语一般如何实现？如何使用？
- 温度系数T控制softmax平滑度，典型值T=2~10
- mimic distillation阶段通常只匹配输出分布，preference distillation进一步优化
- 蒸馏数据集可以是通用语料或任务特定数据
- 可将MoE模型部署到资源受限环境（移动端、边缘设备）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference

**Task-MoE vs Distillation 对比（Kudugunta et al., EMNLP 2021）**：
蒸馏 token-MoE (533M) → dense Transformer-Base (142M) 仅保留 **32.25%** BLEU 增益（与 Fedus et al., 2021 Switch Transformer 结果一致：蒸馏 large sparse model 仅保留 small fraction 质量增益）。Task-MoE sub-network extraction 保留 **100%** BLEU 增益（decoder 仅 25M params vs distilled 142M），BLEU 29.0 vs 26.9 (+2.1)。核心原因：蒸馏过程中引入 undesirable artifacts (Freitag et al., 2019)，而 sub-network extraction 直接使用原始 MoE expert 权重。

---

## Expert Merging

术语解释
专家合并是将MoE模型中功能相似的多个专家合并为一个专家的技术，以减少专家总数、降低参数量和推理计算量，同时尽量保持模型性能。

术语是什么？
专家合并基于观察：MoE中不同专家可能学到相似的功能或特征，可以被合并而不显著损害性能。
- **Branch-Train-Merge**：在不同数据子集上独立训练模型的不同部分，避免了传统大模型训练中的大规模多节点同步
- **Branch-Train-Mix**：异步并行训练多个种子LLM以专精于不同领域，然后合并MoE层的参数创建统一模型，经过二次微调提升性能
- **MC-SMoE**：基于路由策略将专家分组，每组合并为一个专家（加权和），然后对合并后专家使用低秩分解
- **HC-SMoE**：层次聚类合并，无需重训练，任务无关
- **MEO**：drop-in replacement算法——先合并选中的专家参数，再高效计算
- **DEK**：在特征空间识别并分组相似专家，在权重空间合并
- **C-PRUNE Parameterized Expert Merging**：在 hierarchical clustering 分组后，每个 cluster 内专家通过 affinity-weighted averaging 合并为一个。权重 ω_i = exp(γ·A_ik) / Σ exp(γ·A_jk)，其中 A_ik 是 expert i 与 cluster center 的 cosine affinity score，温度 γ 控制融合锐度（γ 越大越接近 hard selection）。同时路由权重通过均值 + exploration noise 更新：Ŵ_k = mean(W_i) + ε·N(0,I)
- **HyperMoE**：利用未选中专家的上下文信息补偿迁移到特定专家的性能损失
- **LiteMoE**：基于应用特征保留最关键的专家，合并次要专家，获得最终稀疏模型——适用于移动设备

从算法pipeline角度拆解术语。
```
# Expert Merging on Forward Pass (MEO drop-in replacement)
def moe_forward_with_merge(x, experts, router, K):
    # 标准Top-K选择
    θ = Softmax(R(x))
    selected = TopK(θ, K)
    
    # 合并选中的专家（而非分别计算）
    W_merged = sum(θ[i]/sum(θ[j] for j in selected) * experts[i].weight 
                   for i in selected)
    # 单次FFN计算（替代K次）
    y = FFN_with_weight(x, W_merged)  # σ(x @ W_1) @ W_2
    return y

# 静态合并（预处理阶段）
def static_expert_merging(experts, similarity_threshold):
    groups = hierarchical_clustering(experts, 
              distance=cosine_similarity(weight_flattened))
    merged_experts = []
    for group in groups:
        W_merged = weighted_average([experts[i].weight for i in group])
        merged_experts.append(W_merged)
    return merged_experts
```

术语一般如何实现？如何使用？
- 静态合并：离线完成，推理时直接使用合并后的模型
- 动态合并（MEO）：推理时在线合并选中的专家，节省K次FFN计算的kernel启动开销
- 合并可在权重空间（参数平均）或特征空间（聚类）进行
- 通常需要微调恢复精度

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

**BTS 论文中的 "Expert Merging" 范畴**：BTS 论文将 Expert Merging 定义为一类方法的统称（与 Expert Upcycling 相对）：在合并阶段 Seed 和 Expert 模型参数**保持冻结**，仅训练少量新参数（如 stitch layers 或 adapters）。具体包括以下变体：

- **Model Soup (Wortsman et al., 2022)**：均匀平均 Seed 和所有 Expert 的权重，无需任何训练。形式：$\theta_{\text{soup}} = \frac{1}{n+1}\sum_{i=0}^n \theta_i$
- **BTM (Li et al., 2022)**：对 Seed 和 Expert 的输出 logits 使用 Bayes 规则加权 ensemble。无需训练。每位专家的权重 $w_i \propto P(\text{input} | \text{expert}_i)$ 由均匀先验下的 Bayes 规则估计。
- **Expert Routing**：训练一个线性路由器 $R \in \mathbb{R}^{\dim \times n}$，基于 prompt 平均 embedding 选择单个模型处理整个序列。路由器决策即模型选择，所有后续 token 路由到同一模型。
- **BAM with Adapters (Zhang et al., 2024)**：在 MoE/MoA 架构中，每个 Attention/FFN Expert 输出后插入线性 adapter $W_{\text{proj}_i} \in \mathbb{R}^{\dim \times \dim}$，仅训练 router 和 adapters。
- **BTS**：通过插入 Stitch Layer 在 Seed（Hub）和 Expert（Spoke）之间建立双向可学习连接（详见 Stitch Layer 条目）。

BTS 论文的关键对比维度：Expert Merging（264M 可训练参数、11B 总参数）vs Expert Upcycling（7.2B+ 可训练参数），前者保持模块性和可解释性。

**补充（来自 EEP 的 Continuous Expert Merging via Evolutionary Strategy）**：EEP 将 Expert Merging 作为一种无梯度 post-pruning knowledge recovery 方法，与 pruning phase 组合为一个统一的进化搜索框架。

- **连续合并矩阵**：Expert Merging Matrix WEM ∈ R^{E'×E} 的元素从 Pruning Phase 的 one-hot (0/1) 过渡到 Merging Phase 的连续实数值。第 j 个新 expert: θ'_j = {Σ_i ω_ji W₁i, Σ_i ω_ji W₂i, Σ_i ω_ji W₃i}，其中 ω_ji 来自 WEM 第 j 行。
- **与 Router 解耦**：Merging Phase 中 WRM ≠ WEM（不再相等），允许路由权重和 expert 权重独立优化。WRM 也变为连续值，实现了更灵活的路由映射。
- **与 Model Soup 的区别**：Model Soup 对所有模型做均匀平均，EEP 通过进化搜索学习最优的非均匀加权系数，可包含负值（负系数表示某些 expert 的知识对下游任务无益）。
- **进化参数**：160 iterations，Mutation=element-wise Gaussian noise added to merging coefficients，Crossover=沿 retained expert 维度组合父代 merging coefficients。
- **关键结果**：Merging 在 Pruning 基础上额外提升 5%-7% 准确率（如 SQuAD: 75.2%→80.6%），整个过程不需要梯度计算，可在仅支持推理的设备上完成。
- **权重分组**：为减少优化参数数量，expert weights 按深度均匀分为 4 groups（或 32 groups per dataset），组内共享 merging coefficients。

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

---

## Branch-Train-MiX (BTX)

术语解释
由 Sukhbaatar et al. (FAIR at Meta, 2024) 提出的一种三阶段 LLM 继续预训练方法，将 embarrassingly parallel 的领域专家训练与 Mixture-of-Experts 架构结合，使多个领域专用 LLM 合并为一个统一的 MoE LLM。

术语是什么？
BTX 包含三个阶段：(1) **Branch**：从预训练 Seed LLM（如 Llama-2 7B）复制 N 份，每份作为领域专家初始化；(2) **Train**：各副本在对应领域数据（如 Math、Code、Wikipedia）上独立继续预训练，完全无同步通信（embarrassingly parallel），训练吞吐线性 scaling；(3) **MiX**：将所有 expert 的 FFN 子层组合为 MoE 层（每个 expert FFN 成为 MoE 的一个 expert），self-attention 和 embedding 等参数直接平均，随机初始化 Router W_l，在全部数据混合上进行 MoE finetune 学习 token 级路由。

```
# === BTX 三阶段算法 ===

# 阶段一: Branch
seed_model = Llama-2-7B
experts = [copy(seed_model) for _ in range(N)]  # N=3 for Math/Code/Wiki

# 阶段二: Train (embarrassingly parallel, 无同步)
for expert_i, data_i in zip(experts, domain_datasets):
    # 各 expert 在不同 GPU 组上完全独立训练
    for batch in data_i:
        loss = CrossEntropy(expert_i(batch), labels)
        loss.backward()
        optimizer_i.step()  # 无 all-reduce

# 阶段三: MiX
# 3a. 组合 FFN 为 MoE 层
for layer_l in range(L):
    # Attention: 对所有 expert 平均
    W_attn[l] = mean([expert_i.attn[l] for i in range(N)])
    # FFN: 构建 MoE
    moe_ff[l] = MoELayer(
        experts = [expert_i.ffn[l] for i in range(N)],  # 4 experts
        router  = Linear(4096, N, init=random)           # 唯一新增参数
    )
    # Router: Top-2 + Load Balancing
    logits = x @ router.weight        # [seq, N]
    top2_vals, top2_idx = TopK(logits, k=2)
    weights = SoftMax(top2_vals)
    output = sum(weights[i] * experts[i](x) for i in top2_idx)

# 3b. MoE Finetune (学习路由)
L_total = CrossEntropy(output, labels) + α * N * sum(u_i * p_i)
# 其中 u_i = mean(g_i(W_l x)), p_i = mean(SoftMax_i(W_l x)), α=0.01
moe_model.train(all_data_mixture, tokens=80B)
```

BTX 泛化两个特例：(1) BTM = 100% expert training + 0% MoE finetune；(2) Sparse Upcycling = 0% expert training + 100% MoE finetune。BTX 在两者之间分配 compute（expert training 512B tokens + MoE finetune 80B tokens），取得最优 accuracy-efficiency tradeoff。

BTX 路由分析关键发现：无 load balancing 时 Code expert 成为 "dead expert"（不被激活），load balancing 使其 "back to life" 并在 code/math domain 成为主导 expert；freeze FFN experts 在 MoE finetune 时对性能几乎无影响（34.7 vs 34.7），说明 domain knowledge 已在 expert training 阶段获得，MoE finetune 主要训练 router 和调优平均的 attention 权重。

术语一般如何实现？如何使用？
- Seed 模型：Llama-2 7B（32 layers, hidden 4096, FFN 11008, 32 heads）
- Expert 训练：Math 48k steps/201B tokens, Code 50k steps/210B tokens, Wiki 42B tokens
- MoE Finetune：80B tokens（Math 30.16%, Code 40.31%, Wiki 10.30%, Llama-2 data 19.23%）
- 最终 BTX 有 4 expert（Math + Code + Wikipedia + original Llama-2 7B generalist）
- 路由选择：Top-2 routing with load balancing α=0.01 为默认；Sample Top-1 更高效（可训练 160B tokens）
- 激活参数：Top-2 11.1B, Sample Top-1 6.7B（vs seed 7B）
- 论文未公开开源代码

涉及论文标题：
- Branch-Train-MiX Mixing Expert LLMs into a Mixture-of-Experts LLM
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
## Expert Decomposition (Low-Rank)

术语解释
专家分解是利用低秩分解技术（如SVD、MPO）将MoE中较大的expert权重矩阵分解为更小的矩阵乘积，从而减少参数量，同时保持计算表达能力。

术语是什么？
低秩分解的直觉：expert权重矩阵通常存在冗余，可以用低秩近似表示：
- **MPOE**：使用矩阵乘积算子（MPO）——一种源自量子多体物理的张量分解技术——将expert权重矩阵分解为中心张量（保留大部分参数和核心信息）+ 若干辅助张量（较小，作为中心张量的补充）。同一层所有expert共享相同的中心张量，大幅减少每层总参数。
- **MC-SMoE**：先合并专家分组，再对合并后专家使用低秩分解（基于合并后专家秩更低的观察）
- **MoE-I²**：识别每个expert的重要性I_{i,j}，为重要expert分配更高秩、不重要expert分配更低秩。秩分配公式：r_{i,j}=⌊(I_{i,j}+ε)^α / Σ(I_{i,j}+ε)^α · R_a · M_i⌋

从算法pipeline角度拆解术语。
```
# Low-Rank Expert Decomposition (SVD-based)
def decompose_expert(W, rank):
    # W: [d_out, d_in]
    U, S, V = SVD(W)
    # 保留前rank个奇异值
    U_r = U[:, :rank]          # [d_out, rank]
    S_r = S[:rank]             # [rank]
    V_r = V[:rank, :]          # [rank, d_in]
    # 分解为两个小矩阵
    A = U_r @ diag(sqrt(S_r))  # [d_out, rank]
    B = diag(sqrt(S_r)) @ V_r  # [rank, d_in]
    # FFN: σ(x @ B^T) @ A^T   (替代 x @ W^T)
    return A, B

# 参数量：d_out*d_in → 2*d_out*rank
# 压缩比 ≈ d_in / (2*rank)
```

术语一般如何实现？如何使用？
- 适用于参数量大的expert进行分解
- 可结合其他压缩技术（量化+分解，剪枝+分解）
- MPO分解在量子物理领域成熟，应用于NN是一种跨学科迁移
- 需要注意分解后的精度恢复（可能需要微调）

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

**DeRS-LM (Low-rank Matrix-based DeRS Upcycling)**：DeRS 论文中的 DeRS-LM 采用低秩矩阵 A∈R^{d×r} 和 B∈R^{r×d_h}（A 随机初始化，B 零初始化）表示专家专属增量权重 Δ_i：
- 专家权重合成为 W_i = W_shared + A_i·B_i，其中 W_shared 从原始 FFN 权重初始化
- 训练参数从 N·d·d_h 降至 d·d_h + N·r·(d+d_h)
- rank r=1 时仅增加 ~2.4M 参数（vs Vanilla Upcycling 的 ~2.5B），实现 1041× 参数减少
- 关键设计：B 零初始化确保初始 Δ=0（即初始专家权重等于原始 FFN 权重，保持 upcycling 的 warm-start 特性）
- 与传统的 expert 低秩分解不同：DeRS-LM 分解的是 delta 权重 Δ_i 而非完整的 expert 权重 W_i；base weight W_shared 保持完整 dense
- 当 pretrained dense model 未经过先验微调时，推荐使用 DeRS-LM（低秩矩阵能进行全局修改，即使 rank 很低也能有效调整所有元素）

---

## Sparse to Dense Conversion

术语解释
稀疏到稠密转换是将MoE稀疏模型转换为同等结构的稠密模型的技术，以消除动态路由开销和expert管理的复杂性，在推理时将MoE模型"压缩"为高效的稠密模型。

术语是什么？
稀疏到稠密转换适用于稠密模型部署更优的场景（如缺少MoE优化框架支持的设备）：
- **XFT**：生成sparse-upcycled MoE模型，再通过可学习的合并机制转换回同等大小和结构的稠密LLM
- **Switch Transformers**：蒸馏将稀疏模型转稠密，97%参数压缩后保留30%+性能
- **OneS**：Knowledge Gathering（求和/平均/top-k/SVD聚合）→ Knowledge Distillation两阶段
- **EWA**：训练时用MoE替代FFN，推理时恢复为稠密ViT
- **AdaMoLE**：结合LoRA结构的专用网络，根据不同任务复杂度调整激活阈值

从算法pipeline角度拆解术语。
```
# Sparse → Dense Pipeline
# Stage 1: Knowledge Gathering
W_dense = aggregate_experts_to_dense(MoE_model)
# 聚合方式：
# - Sum: W_dense = Σ_i W_i
# - Average: W_dense = mean(W_i)
# - Top-K: W_dense = Σ_i topk_weight(i) * W_i
# - SVD: W_dense = SVD_reconstruct([W_1, ..., W_N])

# Stage 2: Knowledge Distillation
for x in dataset:
    y_moe = MoE(x)      # 原MoE教师
    y_dense = Dense(x)  # 聚合后的稠密学生
    loss = KL_div(y_moe, y_dense) + task_loss(y_dense, labels)
    update(Dense)
```
转换后的稠密模型参数量更小（相当于原MoE的激活参数量级），推理路径更简单（无router、无expert选择开销）。

术语一般如何实现？如何使用？
- Knowledge Gathering阶段将专家知识合并到单一FFN
- 蒸馏阶段微调稠密模型以恢复MoE性能
- 适用于需要简单部署的场景（移动端、边缘设备）
- 牺牲了MoE的"无限"扩展能力，但获得了部署简便性

涉及论文标题：
- A Survey on Inference Optimization Techniques for Mixture of Experts Models
- A Survey on Mixture of Experts in Large Language Models

---

## Expert-Choice Gating (EC Routing)

术语解释
Expert-Choice Gating 是 Zhou et al. (NeurIPS 2022) 提出的门控策略，反转传统 Token-Choice Gating 逻辑：由每个专家独立选择 top-k tokens 来处理，而非每个 token 选择 top-k 专家。

术语是什么？
传统 MoE 路由（Token-Choice）令每个 token 从 N 个专家中选 top-k 个，导致负载不均和 token 丢弃。Expert-Choice 则令每个专家从 T 个 tokens 中选 top-k 个。具体：
- 计算 token-expert affinity 矩阵 S ∈ R^{T×N}
- Token-Choice: TopK(S, k) 沿 expert 维度（每行选 k 列）
- Expert-Choice: TopK(S^T, k) 沿 token 维度（每行/每专家选 k 个 tokens）
- 每个专家恰好处理 k 个 tokens，天然负载均衡，无需 auxiliary loss

Expert-Choice 允许变长专家激活：简单 token 可能只被 0-1 个专家选中，复杂 token 可能被 3-4 个专家选中。实验显示 74% tokens 路由到 1-2 个专家，23% 到 3-4 个，3% 到 >4 个。

从算法pipeline角度拆解术语。
```
# Expert-Choice Gating
affinities = x @ W_gate.T                    # [T, N]
topk_vals, topk_idx = TopK(affinities.T, k)  # [N, k], 沿 token 维度

# 每个专家 i 处理 topk_idx[i, :] 指定的 tokens
for i in range(N):
    selected_tokens = topk_idx[i, :]
    y[selected_tokens] += FFN_i(x[selected_tokens]) * gate_weights[selected_tokens, i]
```
与 Token-Choice 的对比：
- Token-Choice: 需要 capacity factor（1.25x~2x）和 auxiliary load balancing loss（w=0.01）
- Expert-Choice: 天然负载均衡，无需 auxiliary loss，训练 2x 更快收敛

术语一般如何实现？如何使用？
- 训练时 2x 更快的收敛（达到相同 perplexity），~20% 更快的 step time
- GLUE + SuperGLUE 11 任务上平均比 Switch top-1 和 GShard top-2 高 ~2%
- 被 Brainformers 等多种后续架构采用，可从 16 扩展到 128 专家

**Brainformers 中的 Expert Choice Gating 使用**：
Brainformers 的演化搜索将 gating function 作为搜索维度之一（搜索空间包含 Top-2 和 Expert Choice），搜索到的 Brainformer Block 1 选择 Expert Choice gating + capacity factor=1。相比于 Top-2 routing，Expert Choice 在 Brainformers 中的优势：
- **Perfect load balance**：无需 auxiliary load balancing loss，训练更稳定
- **更稀疏的专家激活**：capacity factor=1 时每 token 平均路由至 1 个 expert（vs Top-2 的固定 2 个）
- **更快的 step time**：通信量减半（top-1 dispatch vs top-2 dispatch），配合更小的 expansion ratio 实现 5x step time speedup at 8B scale
- 但需要训练时可访问全部 token（双向 attention 或 encoder 场景），不适合 decoder-only 自回归推理的 naive 实现

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Brainformers Trading Simplicity for Efficiency
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

**ETR 论文中的 Expert-Choice Gating 使用**：
ETR 将 ECR 与 TCR 组合为双向路由。在 ETR 中，ECR 不是独立使用的，而是作为 TCR 之后的第二阶段：TCR 先让 token 选 top-ℓ experts，然后 ECR 让每个 expert 从已分配的 token 中按 affinity score δ 选择 top-C tokens（Bottom-C 保留最高分 token）。ETR 的 ECR 与传统 Expert-Choice 的一个关键区别：传统 EC 每个 expert 固定选 k 个 token；ETR 的 EC 使用自适应容量 C = max(C_min, s/n)，C 随训练进度动态调整（后期降低 ~40%）。ETR 理论证明 (Theorem 5)：在 expert 获得判别能力后 (q_i << 1)，ECR 成功率 ≥ 1-e^{-3C/16}，接近 1，而 TCR 仍受限于 C/s。

---

## Soft MoE (Token Merging & Expert Merging)

术语解释
Soft MoE 是一类保持完全可微的 MoE 变体，避免离散的 top-k 专家选择。分为 Token Merging（Puigcerver et al. 2023，仅 vision）和 Expert Merging（SMEAR 2023 → Lory 2024，支持自回归 LM）。

术语是什么？
离散门控的主要问题是不可微且负载不均。Soft MoE 通过软合并避免这些问题：
1. **Token Merging**: 计算所有 tokens 的加权平均（权重依赖于 token 和专家），每个专家处理一个"合并 token"
2. **Expert Merging (SMEAR/Lory)**: 对所有专家参数做加权平均得到单一"合并专家"，然后执行单次 FFN 前向

关键公式（Expert Merging）：y = FFN(x; Σ_i e_i · θ_i)，替代 y = Σ_i e_i · FFN(x; θ_i)

从算法pipeline角度拆解术语。
```
# Expert Merging (SMEAR/Lory)
def soft_merged_moe(x, experts, router):
    logits = router(x)                                  # [batch, N]
    weights = softmax(logits, dim=-1)                   # [batch, N]
    # Merge experts in parameter space
    W1_merged = sum(w[b,i] * experts[i].W1 for i, b)
    W2_merged = sum(w[b,i] * experts[i].W2 for i, b)
    h = activation(x @ W1_merged.T)                      # [batch, d_ffn]
    y = h @ W2_merged.T                                  # [batch, d_model]
    return y
```

术语一般如何实现？如何使用？
- Lory: 首个扩展到自回归 LM 预训练（150B tokens, 32 experts），+13.9% perplexity vs dense
- 因果分段路由：用上一段 hidden state 计算本段路由权重，保持自回归性
- 专家自然学习到 domain-level specialization
- 局限：token 级离散路由 MoE 仍有性能优势

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models

---

## Shared Expert (Residual MoE)

术语解释
Shared Expert 是 MoE 中预留一部分对所有 tokens 始终激活的固定专家，与动态路由专家配合。DeepSpeed-MoE (2022) 的 Residual-MoE 首次提出，DeepSeekMoE (2024) 推广为多共享专家的 fine-grained 设计。

术语是什么？
核心动机是解决"知识冗余"：多个 routed expert 可能重复学习通用知识。Shared Expert 捕获共性知识，让 routed experts 专注于细粒度专业化。

DeepSpeed-MoE: 1 fixed + 1 routed = top-1 通信开销获得 top-2 精度。
DeepSeekMoE: 多个 shared experts（如 2 shared + 64 routed），shared:routed ≈ 1:3。

从算法pipeline角度拆解术语。
```
def moe_with_shared_expert(x, shared_experts, routed_experts, router, K):
    y_shared = sum(SE_i(x) for SE_i in shared_experts)  # 固定激活
    logits = router(x)                                    # [batch, N_routed]
    topk_vals, topk_idx = TopK(softmax(logits), K)       # 稀疏激活
    y_routed = sum(topk_vals[e] * routed_experts[e](x) for e in topk_idx)
    return y_shared + y_routed
```
使用模型：DeepSpeed-MoE, DeepSeekMoE/V2/V3, OpenMoE, Qwen1.5-MoE, MoCLE, ARIA。

涉及论文标题：
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA 每 MoE 层有 2 shared experts (always active) + 64 routed experts (Top-6 per token)；shared experts 捕获通用跨模态知识，routed experts 发展出 modality-specific specialization）

术语一般如何实现？如何使用？
- 减少专家间知识冗余，提升参数效率
- 减少 All-to-All 通信量（shared expert 固定本地），有利于通信-计算 overlap
- 典型配置：总参数量的 10-20% 分配给 shared experts

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework（DeepSeek-MoE-16B 使用 2 shared + 64 routed 残差 MoE 架构；发现 shared experts 相比 routed experts 更不可压缩——pruning 不包含 shared experts 提升 Wanda 平均精度 +3.6%、SparseGPT +1.5%，因为 shared experts 对所有 token 激活，承载更关键和更通用的知识，对压缩更敏感）
- Efficient MoE Inference with Fine-Grained Scheduling of Disaggregated Expert Parallelism（FinDEP 在 DEP 架构下对 Shared Expert 的调度视角：Shared Expert 在 DEP 中置于 Attention Group(AG)因需对所有 token 计算。关键发现：Shared Expert 与 A2E 通信无数据依赖——A2E 仅需 attention 输出即可发送，无需等 Shared Expert 完成。FinDEP 支持两种调度策略：AASS(All Attention then All Shared)使 A2E 最早启动，ASAS(Alternating Attention-Shared)使 AG GPU 利用率最高；通过 Algorithm 1 自适应选择最优顺序，解决原 PP-Pipe 将 Shared Expert 与 Attention 串行导致 GPU 空闲的问题。）
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs（Ling 采用单个 Shared Expert 配合 Fine-Grained Routed Experts，公式为 o_t' = o_t + E_share(h_t)，其中 o_t 为 routed experts 的加权输出。Shared Expert 无需路由，所有 token 均通过其计算，提供通用语言能力，使得 routed experts 可专注于专业化。Paper 指出仅靠 fine-grained experts 不足以同时发展通用和专用能力，Shared Expert 是必要补充。）
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model（2 shared experts + 160 routed experts per MoE layer, shared experts 捕获通用知识，routed experts 通过 fine-grained segmentation 实现专业化）

---

## Fine-grained Expert Segmentation

术语解释
由 DeepSeekMoE (2024) 提出，将标准 MoE 中每个专家的 FFN 中间维度 (d_ffn) 切分为更小的粒度，增加专家数量同时减小每个专家的尺寸，提升知识分解精度。

术语是什么？
标准 MoE 中 d_expert = d_ffn。Fine-grained 将 d_expert 缩小为 d_ffn / m（如 1/8），专家数扩大 m 倍。DeepSeekMoE-145B: d_expert = 1/8 d_ffn, 16→128 experts, top-2→top-16。

核心优势：(1) 更精细的知识分解 (2) 更灵活的专家组合 (3) 解决"知识混杂"问题。
LLAMA-MoE 验证：激活 4/16 experts (d_expert=688) 优于 2/8 (d_expert=1376)。

术语一般如何实现？如何使用？
- DeepSeekMoE, Qwen1.5-MoE, DBRX 均采用此策略
- 需配合 shared experts 使用以补偿单个专家容量不足
- DeepSeek-V3: d_expert=2048, shared=1, routed=256, top-8

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA: 24.9B total / 3.5B activated per text token, 66 experts/layer = 2 shared + 64 routed, expert FFN dim=1664, hidden dim=2560, 每 token 激活 6 routed + 2 shared；所有 expert 为 modality-generic，expert specialization 在 multimodal 预训练中自然涌现）
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts（使用 32c4 fine-grained MoE 配置：32 total experts × 4 active, 1.96B total params, 565M active params, hidden dim=1024）
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

## Device-Limited Routing (设备限制路由)

术语解释
Device-Limited Routing 是 DeepSeek-V2 提出的一种 MoE 路由约束机制，在细粒度专家分割（大量小专家）场景下，限制每个 token 的目标专家最多分布到 M 个设备上，从而控制 expert parallelism 下的 all-to-all 通信开销。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
在 DeepSeek-V2 的 160 routed experts 配置中，采用 8-way expert parallelism (D=8)，每个设备部署 20 个 routed experts。若不做限制，top-K (K_r=6) 选择可能将 6 个专家分布在多达 6 个设备上，导致每 token 的 MoE 通信量翻倍。Device-Limited Routing 分两步：(1) 先按 token-to-expert affinity 选出 M 个最受青睐的"设备"（而非直接选 expert）；(2) 再在这 M 个设备的 expert 子集中做 top-K 选择。DeepSeek-V2 设置 M=3，实验表明 M≥3 时性能与无限制 top-K 路由接近对齐。

为什么需要？MoE 通信频率与目标专家覆盖的设备数成正比。DeepSeekMoE 的细粒度专家分割导致激活专家数多（K_r=6），若不加设备限制，expert parallel all-to-all 通信量会严重拖累训练效率。Device-Limited Routing 将通信量从 O(K_r) 限制到 O(M)，当 M=3 时通信量减少约 50%。

从算法pipeline角度拆解术语：
```
=== Device-Limited Routing (per token) ===

Input: u_t (token hidden state), {e_i} (expert centroids), D=8 devices, M=3, K_r=6

// Step 1: Compute token-to-expert affinity
for i in 1..160:
    s_{i,t} = Softmax_i(u_t^T · e_i)

// Step 2: Aggregate affinity per device (20 experts per device)
for device d in 1..8:
    S_d = max_{i in experts(d)} s_{i,t}     // or sum/top-k aggregation

// Step 3: Select top-M devices
Devices_selected = TopK_devices({S_d | d=1..8}, M=3)

// Step 4: Top-K among experts on selected devices only
Experts_selected = TopK({s_{i,t} | expert i is on device d ∈ Devices_selected}, K_r=6)

// Step 5: Compute gating and FFN output (standard)
g_{i,t} = s_{i,t} if i in Experts_selected else 0
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
在 HAI-LLM 训练框架中实现，与 expert parallel all-to-all 通信层紧密集成。训练时：每 token 最多与 M=3 个设备通信（而非 naive 的 K_r=6 个）。DeepSeek-V2-Lite 因为所有 expert 部署在同一设备上（无 expert parallelism），不需要 Device-Limited Routing。DeepSeek-V3 继用此设计（M=4, D=8, K_r=8）。

涉及论文标题：
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

## Token-Dropping Strategy for MoE Training (MoE训练中的Token丢弃策略)

术语解释
Token-Dropping Strategy 是 DeepSeek-V2 在 MoE 训练中提出的负载均衡补充机制：当 balance loss 无法保证严格负载均衡时，在每个设备上按计算预算（capacity factor=1.0）丢弃 affinity score 最低的多余 token，避免因负载不均导致的计算资源浪费。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
尽管 expert-level balance loss、device-level balance loss 和 communication balance loss 三层辅助损失鼓励均衡负载，但它们无法保证严格均衡——某些 expert/device 可能仍收到超过平均值的 token 分配。Token-Dropping Strategy 在每个 device 上强制执行硬性计算预算：(1) 计算每 device 的平均计算预算 = total_tokens / D (capacity factor=1.0)；(2) 若某 device 收到的 token 超过预算，按 affinity score 从低到高丢弃超出的 token；(3) 保留约 10% 的训练序列永远不丢 token（保证训练-推理一致性）。推理时可灵活选择是否启用 token dropping。

为什么需要？MoE 训练中负载不均的两个后果：(1) routing collapse——某些 expert 训练不足；(2) 计算效率下降——expert parallel 下过载 device 成为瓶颈。Balance loss 是软约束，token dropping 提供硬约束兜底。

从算法pipeline角度拆解术语：
```
=== Token-Dropping Strategy (per MoE layer, per training step) ===

Input: tokens assigned to device d: {(t, s_t_expert, g_t)}, capacity C = T/D

// Step 1: Sort tokens by affinity score (descending)
tokens_sorted = sort_by_affinity(tokens_to_device_d, descending=True)

// Step 2: Keep top-C tokens, drop the rest
kept = tokens_sorted[:C]
dropped = tokens_sorted[C:]     // lowest affinity tokens

// Step 3: Guarantee ~10% sequences never drop
// Mark 10% sequences as "protected" before sorting
// Protected tokens always in kept set regardless of affinity

// Step 4: Forward only kept tokens through experts
for (t, s, g) in kept:
    expert_output += g * FFN_expert(h_t)

// Dropped tokens: skip expert computation (output = shared experts only)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
训练时启用加速，评估时不丢 token 以保证结果确定性。推理时：根据效率需求灵活选择——offline batch 推理可启用以提升吞吐，online serving 通常不启用以保证质量。DeepSeek-V2 训练中 capacity factor=1.0，略低于 Riquelme et al. (2021) 的 1.25-1.5（更激进）。与 GShard 的 capacity factor 机制核心区别：GShard 对 expert 做 capacity limit，DeepSeek-V2 对 device 做 capacity limit（因 device-limited routing 后 device 是计算的实际分配单位）。

涉及论文标题：
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

---

## Mixture of Parameter-Efficient Experts (MoPE)

术语解释
MoPE 将 MoE 的门控机制与参数高效微调（PEFT，如 LoRA、Adapter、Prefix-tuning）结合，每个专家本身是一个 PEFT 模块，实现多任务微调下的参数效率和任务隔离。

术语是什么？
将 MoE 的多专家架构与 PEFT 的低参数量结合：专家由 LoRA 矩阵/Adapter 块/(IA)^3 向量构成，仅更新专家和门控，base model 冻结。按放置位置分为：FFN-level, Attention-level, Transformer Block-level, Every Layer-level。

从算法pipeline角度拆解术语。
```
# LoRAMoE (FFN-level MoPE)
def loramoe_forward(x, base_ffn, lora_experts, router):
    y_base = base_ffn(x)  # frozen FFN
    weights = softmax(router(x))
    y_lora = sum(weights[i] * (x @ A_i @ B_i) for i in range(N))
    return y_base + y_lora
```

术语一般如何实现？如何使用？
- LoRAMoE: 专家分为两组（任务学习 + 知识保持），局部平衡约束
- MoV: (IA)^3 向量专家，MoV-10 仅用 ~40 个向量
- MoLA: 不同层使用不同数量的专家
- 适用场景：多任务微调、领域适配、指令调优

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach

### AT-MoE 的任务特定 LoRA 专家训练方法

AT-MoE 提出了一种与标准 MoPE 不同的训练策略：先在各任务数据上**分别独立训练**任务特定 LoRA 专家（冻结 LLM 权重 W_0），确保每个专家有明确的任务领域属性，再冻结所有 LoRA 专家，训练路由模块。这与标准 MoPE（专家和路由器在混合数据上联合训练）的根本区别在于：
- **任务级专业化**：每个 LoRA 专家 ΔW_j = B_j A_j（B_j ∈ R^{d×r}, A_j ∈ R^{r×k}）在特定任务领域上训练，而非混合数据
- **专家可解释性**：专家有明确的任务标签（如医学中：诊断专家、处方专家、分诊专家、外科专家、放射科专家）
- **可控融合**：路由器可根据复合指令的多个意图，按层次（群组→组内）分配专家权重

此外，AT-MoE 还训练了一个在所有任务混合数据上训练的"预合并 LoRA 专家"W_p 作为通用后备专家，通过平衡参数 λ 控制任务特定专家与通用专家的融合比例：y_i = (λ·F_G(W̄_e) + (1-λ)·W_p)x_i + W_0·x_i。

---

## Sparse Upcycling

术语解释
由 Komatsuzaki et al. (ICLR 2023) 提出，从预训练 dense checkpoint 初始化 MoE：将 FFN 复制为多个专家，随机初始化门控，继续训练使专家分化，避免从头训练 MoE 的高昂成本。

术语是什么？
流程：(1) 加载 dense checkpoint (2) 每层 FFN 复制 N 份作为 N 个专家 (3) 随机初始化门控 (4) 继续训练。Skywork-MoE 发现 from-scratch 训练最终优于 upcycled，expert similarity 可作为训练诊断指标。

```
def sparse_upcycling(dense_checkpoint, N):
    model = load(dense_checkpoint)
    for layer in model.layers:
        experts = [copy(layer.FFN) for _ in range(N)]
        layer.FFN = MoELayer(experts, Gate(N))
    return train(model)  # 专家逐步分化
```

术语一般如何实现？如何使用？
- 前提：已有 quality dense checkpoint
- Skywork-MoE, MoE-LLaVA 使用此方法
- 节省 30-50% FLOPs vs from-scratch
- 局限：最终性能可能不及 from-scratch

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Branch-Train-MiX Mixing Expert LLMs into a Mixture-of-Experts LLM
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

**CuMo 中的 Co-Upcycling 扩展**：CuMo 扩展了 Sparse Upcycling 为 Co-Upcycling——同时 upcycle 视觉编码器（CLIP ViT）和 MLP 连接器的 dense MLP → MoE，而非仅 upcycle 单一模块。不同于标准 upcycling（所有 expert 从同一 MLP 复制），Co-Upcycling 的每个 expert 从同位置预训练/预微调后的 MLP 权重复制。CuMo 还发现 upcycling LLM 效果远不如使用 pre-trained MoE LLM（Mixtral 8×7B），因此最终仅对视觉侧做 Co-Upcycling。

**DeRS 论文中的 Upcycled MoE 冗余发现**：DeRS 首次定量分析了 upcycled MoE 专家的冗余机制——训练后专家权重 W_i 与初始 FFN 权重 W_base 之间的余弦相似度 > 0.999，专家间余弦相似度也 > 0.999。这意味着 W_i = W_base + Δ_i 中的 Δ_i 是微小且高度冗余的调整量。基于此观察，DeRS 提出将 N 个专家重构为 1 个共享基础权重 + N 个轻量 delta 权重（通过稀疏化或量化压缩，或从训练开始就用稀疏/低秩矩阵表示），实现 MoE 层参数减少 65% 或新增训练参数减少 2270×。

**BTS 论文中的 "Expert Upcycling" 范畴**：BTS 论文将 Expert Upcycling 定义为与 Expert Merging 相对的一类方法——将 Seed 和 Expert 模型的参数用于初始化 MoE，但**所有参数都参与训练**，Expert 不再保持完整独立。这丧失了模块性（增删 Expert 需重训大量参数）。具体变体：

- **BTX (Branch-Train-MiX; Sukhbaatar et al., 2024)**：将 Seed 和 Expert 的 FFN 拷贝为 MoE Expert。两种 routing 变体：(1) BTX Sample — Gumbel-Softmax top-1 routing，激活 2.9B/7.2B 参数；(2) BTX Soft — 所有 4 个 Expert 始终激活（7.2B/7.2B）。全部 7.2B 参数参与训练。
- **BAM (Zhang et al., 2024)**：将 Attention 和 FFN 都改为 MoE/MoA 结构，使用 soft-routing（所有 Expert 始终激活）。8.4B 训练参数，8.4B 总参数。

与 BTS (264M 训练参数，仅 stitch 层) 形成鲜明对比。

---

## Mixture of Attention Heads (MoA)

术语解释
由 Zhang et al. (EMNLP 2022) 提出，将 MoE 思想应用于多头注意力：每 token 通过门控动态选择注意力头而非使用全部注意力头，实现 attention 层的条件计算。

术语是什么？
两套 experts（Q projection + O projection），共享门控，K 和 V 在所有 experts 间共享。仅对选中的 attention experts 计算 Q 和 O。DS-MoE 发现 Attention layers 稀疏度远低于 FFN layers（80% vs 20% active ratio），因此保持 attention experts 全激活。

```
def moa_forward(x, Wk, Wv, q_experts, o_experts, gate, K):
    K, V = x @ Wk, x @ Wv  # shared
    topk_idx = TopK(softmax(gate(x)), K)
    y = sum(gate_w[e] * softmax(x@q_e @ K.T/sqrt(d)) @ V @ o_e.T for e in topk_idx)
    return y
```

术语一般如何实现？如何使用？
- 减少 Attention 计算量（仅激活部分 heads）
- 后续工作：JetMoE, ModuleFormer 沿用此设计
- 适用于长序列场景以节省 attention 计算

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- BTS Harmonizing Specialized Experts into a Generalist LLM
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

**DS-MoE 中的 MoA 使用**：DS-MoE (Pan et al., 2024) 将 MoA 与 Grouped-Query Attention (GQA) 结合。每个 MoA expert 负责计算 N_head 个 query vectors Q_i = W_q_i @ X，其中 W_q_i ∈ R^{N_head × d_head × d_h}。K、V 在所有 expert 间共享（通过 GQA 机制共享 KV heads）。最终输出：O = Σ_{k=1..K} S_{A_k} Σ_{j=1..N_head} O_{A_k, j}，其中 O_{A_k, j} = Softmax(Q_{A_k, j} @ K^T) @ V @ W_o_j。DS-MoE 的 1B 模型使用 N_att=16, N_head=2；3B/6B 模型使用 N_att=8, N_head=4。训练阶段所有 MoA expert 全激活（dense training），推理阶段仅激活 top-K 或超阈值 expert。论文发现 Attention 层 sparsity 低于 MLP 层（active ratio >60% vs <30%），因此推理时保持 Attention 层使用 dense inference。

**BTS/BAM 中的 MoA 使用**：BAM (Zhang et al., 2024) 同时使用 MoE 和 MoA 模块，均采用 soft-routing（所有 Expert 始终激活）。MoA 的软路由输出为：
$$y_{\text{MoA}} = \sum_{i \in \mathcal{M}} q_i(x) W_{\text{attn proj}_i} (\text{Attention}_i(x))$$
其中 $q_i(x)$ 为 attention router 输出的软权重，$W_{\text{attn proj}_i}$ 为 BAM Adapters 变体中可训练的线性 adapter。BAM 在 BTS 论文中作为 Expert Upcycling baseline（全参数训练，8.4B 训练参数）。

---

## Expert Capacity

术语解释
Expert Capacity 是限制每个专家每批次最大处理 token 数的阈值，防止 MoE 训练中某些专家被过度使用导致计算热点和内存溢出。由 GShard (2020) 引入。

术语是什么？
capacity = (tokens_per_batch / N_experts) × capacity_factor (CF, 典型值 1.0-2.0)。超过 capacity 的 tokens 被丢弃或随机路由至备选专家。BPR 按 gate score 高→低分配优先级。

OpenMoE 发现的关键现象：(1) "Drop-towards-the-End": 序列后部 tokens 更易被丢弃 (2) "Context-independent Specialization": 专家按 token ID 专业化 (3) "Early Routing Learning": 路由模式在预训练早期固定。

术语一般如何实现？如何使用？
- CF 越大→dropped tokens 越少→计算量越大
- Expert-Choice Gating 不需 capacity（天然均衡）
- 推理时通常不启用 capacity（batch 更小）

从算法pipeline角度拆解术语：
SYMI 重新定义了 Expert Capacity 在 adaptive replication 下的行为。在传统静态系统中：
capacity(e_i) = capacity_factor × tokens_per_batch / E = slot_capacity × r

在 SYMI 的 adaptive replication 下：
capacity_SYMI(e_i) = slot_capacity × r_i

其中 r_i 随 iteration 动态变化（r_i ∝ popularity_i）。当 replication 精确匹配 popularity 时，capacity_factor 变得无关——热门 expert 因更多 replica 而自动获得更大总 capacity，冷门 expert 减少 replica 但不影响其处理能力。这使得 SYMI 在所有 auxiliary loss coefficient 下均保持约 10% token drops（vs DeepSpeed 的 40%+）。

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)
- Accelerating MoE Model Inference with Expert Sharding
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

**MoEShard 对 Capacity Factor 的批判**：
MoEShard 指出 CF 方法的根本缺陷：(1) 超限 token 被丢弃直接损害模型精度；(2) CF 增大虽减少丢 token 但导致显存问题（实验中将 DeepSpeed CF 固定为 min(|E|, 50)，再增大即 OOM）；(3) DeepSpeed 的 CF 限制使得当 expert 数超过 50 时开始丢 token，此时 MoEShard 的 dropless 优势更明显（256 expert 时 MoEShard 仍保持 2.39× 加速）。MoEShard 通过 expert tensor sharding 从根本上避免对 CF 的依赖——所有 token 全程保留、无需 capacity 限制。

**Capacity-Aware Token Drop：推理时的 Expert Capacity 应用**：
Capacity-Aware Inference 首次将 Expert Capacity 系统性地应用于**推理阶段**（而非仅训练阶段），解决 Expert Parallelism 下的 Straggler Effect。核心设计：

1. **Capacity 定义**：C = γN̄ = γ(tk/n)，其中 γ 为容量因子（典型值 1.0-2.0），t = batch_size × seq_len 为总 token 数，k 为 top-k 值，n 为 expert 总数。

2. **Score-based Dropping**：使用 softmax 后的 gating score 作为 token 重要性度量，对超载 expert 丢弃 score 最低的 token。论文验证 Score 优于 Order、Reverse Order、Random（Table 1: Score Avg 61.1 vs Random 53.1 at γ=1.0）。

3. **效率-精度权衡**：γ=1.5 时 OLMoE 获得 30% speedup 仅损失 0.9% 性能（64.0→63.1）；Mixtral-8×7B γ=1.5 时 Token Drop 获 1.87× 加速。

4. **Dropped Token 比例**：DT = Σ ReLU(N_i - γN̄) / Σ N_i。丢弃 12% token 可获 85% 加速（Mixtral）。

5. **与 Expert Parallelism 的交互**：每 GPU 托管 expert 越少（如 Mixtral 1-2E/GPU）效果越显著，因单个 straggler expert load 占比大。托管 8E/GPU 时加速减弱。

---

## Mixture-of-Depths (MoD)

术语解释
由 Raposo et al. (2024) 提出，将 MoE 条件计算原理应用于 Transformer 深度维度：通过二值门控决定每个 token 是否被某层处理，动态分配 FLOPs 到不同序列位置。

术语是什么？
不同于标准 MoE 在宽度维度的条件计算（选择专家），MoD 在深度维度进行条件计算（选择层）。每层有一个二值 router，若输出为 0 → token 走 skip connection 直通下一层。

```
def mod_layer(x, router, layer_fn):
    selected = router(x) > 0       # [seq] bool
    y = x
    y[selected] = layer_fn(x[selected])
    return y  # 未选中的 tokens 直接跳过
```

术语一般如何实现？如何使用？
- 可与标准 MoE 组合（MoD + MoE FFN）
- FLOPs footprint 低于 vanilla Transformer 或 pure MoE
- 在固定 FLOPs budget 下提升性能

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models

---

## BASE Layer (Balanced Assignment of Sparse Experts)

术语解释
由 Lewis et al. (ICML 2021) 提出，将 MoE 的 token-to-expert 分配建模为线性分配问题（Linear Assignment），在约束条件下最大化 token-expert affinity 总和，保证每个专家处理等量 tokens。

术语是什么？
给定 affinity S ∈ R^{T×N}，约束每个 expert 恰好 B 个 tokens，每个 token 恰好 1 个 expert，目标 max Σ S_{i, assigned(i)}。解法：Hungarian algorithm (O(T^3)) 或 Sinkhorn (S-BASE)。

```
assignment = linear_sum_assignment(-S)     # [T] -> expert_idx
# 每个专家恰好 B tokens，每个 token 恰好 1 expert
# S-BASE: Sinkhorn normalization → soft assignment → harden during training
```

术语一般如何实现？如何使用？
- 严格保证负载均衡（数学约束），不需要 auxiliary loss
- S-BASE 通过 Sinkhorn 迭代提供可微训练版本
- 局限：每个 token 仅 1 个 expert（vs top-K 可使用多个）
- 适用场景：对负载均衡有严格要求的分布式训练

涉及论文标题：
- A Survey on Mixture of Experts in Large Language Models

## Expert Popularity (专家热度) / Expert Activation Distribution

术语解释
Expert Popularity 描述 MoE 模型中不同 expert 被输入 token 路由激活的概率分布。由于 gate network 的 top-k 选择机制和训练数据特性，expert 激活通常呈现严重偏斜（skewed）分布，少数 expert 承担大部分 token，大量 expert 仅被极少 token 激活。

术语是什么？
在 MoE 模型中，对于第 l 层第 e 个 expert，其热度可量化为 p_l^e（该 expert 被分配的 token 占总 token 的比例）。APTMoE 定义全局 expert 热度指标 G = (1/L) Σ_l Σ_e (p_l^e)²，G 越大表示分布越不均。Expert 热度存在三个时间维度：
- **历史热度（Historical Popularity）**：少数 expert 跨时间持续高激活，先前 iteration 的热度分布可指导当前 inter-stage loading
- **预测热度（Predicted Popularity）**：通过 predictor 提前预测目标层的 expert 激活分布，用于 inter-layer loading
- **实时热度（Real-time Popularity）**：gate operation 执行后得到的确切 expert 路由结果，用于 inter-expert loading

从算法pipeline角度拆解术语。
Expert Popularity 在 MoE 推理中的利用流程：
```
# Forward: token x_t 流经第 l 层 MoE 层
h = Attention(LayerNorm(x_t))
# Step 1: 提前获得预测热度（predictor 在 l-δ 层插入）
pred_probs = Predictor_l(h_early)      # [num_experts], 预测的激活概率

# Step 2: 计算 real-time 热度
gate_logits = Router(LayerNorm(h))      # 路由 logits
real_probs = Softmax(gate_logits)       # [num_experts]
topk_idx, topk_weights = TopK(real_probs, k=2)

# Step 3: 基于热度做计算分配（Equation 1）
sorted_experts = sort_by_popularity(all_experts)
R = sum(CPU_time[low_pop_experts]) / (Load_MHA + Load_Gate + sum(Load_time[high_pop_experts]))
if R < 1:  GPU_execute(high_pop_experts); CPU_execute(low_pop_experts)
```
在 APTMoE 评估中，NLLB-MoE (128 experts) 的 G≈0.05 但因 expert 数量多热度偏斜反而比 Mixtral-8x7B (8 experts, G≈0.2) 更显著。预测器对 least activated 32 experts（n=32）的预测准确率达 94%（m=48 时），历史热度跨 iteration 保持率为 73.3%。

术语一般如何实现？如何使用？
- 通过 gate network 的 Softmax 输出获取 expert 概率分布
- Predictor 结构：与 gate operation 相同（linear + softmax），初始化权重从 target gate 复制，通过少量微调步骤训练
- 离线 profiling：记录各 expert 在不同 token 数量下的 GPU/CPU 执行时间以计算 affinity
- 用于 prefill/prefetch 决策、GPU/CPU 计算分配、expert cache 替换策略

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

## Computation Affinity (计算亲和性)

术语解释
Computation Affinity 是 APTMoE 提出的概念，描述一个 expert 的计算工作负载对 GPU 或 CPU 的"适配程度"。由于 MoE 的 expert 热度偏斜，不同 expert 的输入 token 数量差异巨大，导致其计算强度（computational intensity）不同——高 token 数的 expert 适合 GPU 的并行计算能力（compute-bound），低 token 数的 expert 在 CPU 上执行时间与 GPU 可比甚至更优（memory-bound 时 CPU 更友好）。

术语是什么？
计算亲和性的核心判断依据：对于给定数量的输入 token，比较 expert 在 GPU 上的端到端时间（计算+数据移动）vs CPU 上的计算时间（无需数据移动，因 expert 权重留在 host memory）。当 token 数较少时，GPU 的计算优势被 kernel launch overhead 和 PCIe 数据传输时间抵消，CPU 就地计算反而更高效。

从算法pipeline角度拆解术语。
```
# Profiling 阶段（离线）
for each expert_config:
    for num_tokens in [1, 2, 4, 8, ..., max_tokens]:
        t_gpu = profile(expert.forward, device='cuda', input_tokens=num_tokens)
        t_cpu = profile(expert.forward, device='cpu', input_tokens=num_tokens)
        t_load = profile(cudaMemcpy, host_to_device, size=expert_size)
        lookup_table[expert][num_tokens] = {gpu_time, cpu_time, load_time}

# Runtime 决策（Equation 1）
def decide_allocation(experts, predicted_popularity):
    sorted_experts = sort(experts, key=predicted_popularity, reverse=True)
    cumulative_cpu_time = 0
    cumulative_load_time = Load_MHA + Load_Gate
    for expert in sorted_experts:  # 从高热度到低热度
        cumulative_load_time += lookup_table[expert][num_tokens].load_time
    for expert in reversed(sorted_experts):  # 从低热度到高热度
        cumulative_cpu_time += lookup_table[expert][num_tokens].cpu_time
        R = cumulative_cpu_time / cumulative_load_time
        if R >= 1:
            break  # 从此处及以上热度的 expert 在 GPU 执行
        allocate_to_cpu(expert)
```
关键发现（Figure 6）：当 token 数量 < ~64 时，A800 GPU 的 expert 计算时间与 Intel Xeon Gold 6348 CPU (28 cores) 可比；token > 256 时 GPU 优势显著。且低热度 expert 无法 saturate CPU 核心（减少核心数影响小），适合 C1+G4（7核/进程）场景。

术语一般如何实现？如何使用？
- 通过 PyTorch profiler 记录单层单 expert 在不同 token 数下的 forward/backward 时间
- Lookup table 在 static 阶段生成，runtime 阶段查表
- 适用于 batch size 固定、sequence length 固定的 fine-tuning 场景
- 需要权衡：expert 尺寸越大（如 MoE-M vs MoE-S），CPU 执行效率越低，affinity 效益递减

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes
- Accelerating Distributed MoE Training and Inference with Lina

**Lina 的 Expert Popularity Estimation 方法**：
与 APTMoE 使用 Predictor 模块（learned model）不同，Lina 采用 **profiling-based statistical estimation**：
1. **Expert Selection Pattern**: 发现 tokens 在相邻 MoE layers 中选择同一 expert 的倾向性——选定同一 expert 的 tokens 在下一层中选择 top-1 same expert 的比率达 41.94%（k=2 时 54.59%）
2. **Sample Path Profiling**: 在 training 阶段采集 load balancing loss 稳定后的 expert selection results，按 sample path（从 layer i-l 到 layer i 的 expert 序列）分组，为每个 sample path j 计算到 layer i+1 的 expert 分布 `Ψ_j^{i+1}`
3. **Online Estimation**: inference 时对每个 batch，从 layer l 开始，根据每个 token 的 sample path j(t) 查找对应的 `Ψ_{j(t)}^{i+1}`，取 top-k expert 的概率 `P(e)` 作为 popularity estimate
4. **Resource Allocation Formula**: `n_e = N × Σ_t P_{j(t)}(e) / N_t`（expert e 应占设备数比例）
5. **Accuracy**: path length l=3 时 estimation accuracy 60.4%（Transformer-XL）和 63.5%（BERT-Large），l=6 时可达 71.4%

---

## Expert Popularity Predictor (专家热度预测器)

术语解释
APTMoE 提出的一种轻量级预测模块，在 MoE 层的 gate operation 之前若干层插入，结构与该层 gate operation 相同（linear + softmax），通过微调训练提前预测目标层的 expert 激活分布，为 inter-layer loading 提供决策依据。

术语是什么？
预测器是 gate operation 的"影子"副本，放置在目标层之前（如 Mixtral-8x7B 中提前 1 层，NLLB-MoE 中提前 4 层）。它接收当前层的中间 hidden states 作为输入，输出目标层各 expert 的预测激活概率。预测器不修改原模型结构，不影响原始 MoE 的计算结果。

从算法pipeline角度拆解术语。
```
# 假设 target_layer = l, predictor 插入在第 l-δ 层
# δ=1 for Mixtral (全 MoE layers)
# δ=4 for NLLB-MoE (每4层一个 MoE layer)

# Training (predictor fine-tuning)
predictor = copy_gate_structure(layer_l.gate)       # 复制结构
predictor.weights = layer_l.gate.weights.clone()    # 复制权重初始化
for step in range(training_steps):
    hidden_early = model.forward_up_to(layer_{l-δ})  # 提前若干层的 hidden states
    hidden_target = model.forward_up_to(layer_l)     # 目标层的 hidden states
    pred_probs = predictor(hidden_early)              # 预测的 expert 概率
    real_probs = layer_l.gate(hidden_target)          # 真实的 expert 概率
    loss = CrossEntropy(pred_probs, real_probs)       # 监督信号
    loss.backward(); optimizer.step()

# Inference (预测用于 inter-layer loading)
pred_probs = predictor(hidden_early)
sorted_experts = sort_by_predicted_popularity(pred_probs)
for expert in sorted_experts:
    if should_load_to_gpu(expert, Equation_1):
        interlayer_queue.add(expert)
```
预测器 overhead：FLOPs = 2sdE（gate 同结构），expert FLOPs = 8sdh，h >> E（Mixtral h=14336, E=8），因此 predictor 额外计算可忽略。训练收敛时间：Mixtral ~0.93s (700 steps)，NLLB ~0.18s。

术语一般如何实现？如何使用？
- PyTorch nn.Linear + Softmax 实现，与 gate 结构一致
- 使用 KL Divergence 或 Cross Entropy 作为训练损失
- 预测准确率取决于：expert 数量（越少越高）、预测提前量（越近越准）
- Expert 级准确率比 token 级准确率更有实际意义（只需知道哪些 expert 是 low-demand）
- 在 Mixtral-8x7B 上 least 2/8 experts 预测准确率 100%，在 NLLB-MoE 上 least 32/128 experts 准确率 94%

涉及论文标题：
- APTMoE Affinity-Aware Pipeline Tuning for MoE Models on Bandwidth-Constrained GPU Nodes

## Adaptive Grouped Routing (自适应分组路由)

术语解释
Adaptive Grouped Routing 是 AT-MoE 提出的两层层次化 MoE 路由机制，先用群组级路由（Group-level Routing）在专家类别间分配全局权重，再用组内路由（Within-group Routing）在组内进行局部专家权重归一化，实现对复杂多意图复合指令的层次化、可解释的专家权重分配。

术语是什么？
标准 MoE 路由仅使用单层 top-K 门控（Linear→Softmax→TopK），无法区分复合指令中不同子任务的重要性。Adaptive Grouped Routing 通过两层矩阵解决此问题：

**第一层 - 群组路由（Group Routing）**：使用群组路由向量 W_G ∈ R^{N_dim × N_G}，将输入嵌入 x 映射为跨组权重向量：
```
M_G = x @ W_G                       # [1, N_G]
W'_G = SoftMax(M_G / τ_g)          # 温度 SoftMax，得各组权重
```
N_G 为专家组的数量（如医学场景分为功能类、领域知识类、风格类三个组）。

**第二层 - 组内路由（Within-group Routing）**：使用组内路由矩阵 W_D ∈ R^{N_G × N_M}，其中 N_M 为每组最多专家数。对组内专家做逐列 SoftMax：
```
M_D = W'_G @ W_D                    # 用群组权重加权
W'_D = col_wise_SoftMax(M_D / τ_d)  # 每列（每组）独立 SoftMax
```
不足 N_M 专家的组用 -inf padding，使其不参与 SoftMax 计算。

最终路由函数 F_G(W̄_e) = Σ_j W'_D[j] · LoRA_j(x)，各任务特定 LoRA 专家按层次化权重加权求和。

从算法pipeline角度拆解术语。
```
# Adaptive Grouped Routing Forward Pass
def adaptive_grouped_routing(x, W_G, W_D, lora_experts, group_ids, tau_g, tau_d):
    """
    x: token embedding [d]
    W_G: group routing vector [d, N_G]
    W_D: within-group routing matrix [N_G, N_M]
    lora_experts: list of LoRA modules, each has group_id and expert_id
    """
    # Step 1: Group-level routing
    M_G = x @ W_G                                    # [N_G]
    W_prime_G = softmax(M_G / tau_g)                 # [N_G], 跨组权重

    # Step 2: Within-group routing
    M_D = W_prime_G @ W_D                            # [N_M], 组内专家logits
    # 对不足N_M的组pad -inf
    for g in range(N_G):
        actual_experts = count_experts_in_group(g)
        M_D[actual_experts:] = -inf
    W_prime_D = softmax(M_D / tau_d)                 # [N_M], 组内专家权重

    # Step 3: 将所有专家按层次权重加权
    expert_out = 0
    for j, (lora, group_id) in enumerate(zip(lora_experts, group_ids)):
        weight = W_prime_G[group_id] * W_prime_D[j]  # 层次权重 = 群组权重 × 组内权重
        expert_out += weight * lora(x)
    
    return expert_out
```

以医学复合查询"四肢无力+开中药方"为例，路由过程：
1. 群组路由 → 功能组 0.6, 领域组 0.3, 风格组 0.1
2. 功能组内路由 → 诊断 0.5, 处方 0.4, 分诊 0.1
3. 领域组内路由 → 消化内科 0.5, 中医 0.4, 放射科 0.1
4. 风格组内路由 → 严谨型 0.8
5. 最终诊断 LoRA 权重 = 0.6 × 0.5 = 0.3（可追溯、可解释）

术语一般如何实现？如何使用？
- 训练阶段：先在各任务数据上训练 LoRA 专家（冻结 LLM + 路由器未训练），再冻结所有 LoRA 专家，训练 W_G 和 W_D 路由矩阵
- 推理阶段：路由模块根据输入动态计算 W'_G 和 W'_D，加权融合多个 LoRA 的输出
- 不同 Transformer 层有独立的 W_G^(l) 和 W_D^(l)（layer-wise routing）
- 通过预合并通用 LoRA W_p 和平衡参数 λ 融合任务特定专家与通用专家
- 适用于需要可解释性和可控性的多任务场景（如医学诊断、法律咨询）
- 目前无开源代码，论文未提供实验验证

涉及论文标题：
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach

## Layer-wise Routing Matrix in MoE (层级路由矩阵)

术语解释
Layer-wise Routing Matrix 指 MoE 模型中不同 Transformer 层使用**独立的路由参数矩阵**，而非所有层共享同一个路由器。AT-MoE 论证了不同层关注不同抽象级别特征（低层偏基础领域知识，高层偏功能性和风格性特征），因此需要为每层训练独立的路由矩阵。

术语是什么？
在标准 MoE 中，每层虽各有独立的路由权重 W_r^(l) ∈ R^{d×N}（Switch Transformer 等主流架构均如此），但路由逻辑相同：均为 Linear→Softmax→TopK，未针对不同层的特征偏好进行差异化设计。AT-MoE 明确将 layer-wise 路由作为设计原则：对于有 N_T 个 Transformer block 的模型，使用 N_T 组独立的路由矩阵对 (W_G^(l), W_D^(l))，l = 1...N_T。其假设是：
- **低层（1~N_T/3）**：关注基础领域知识特征，路由偏向领域知识类专家组
- **中层（N_T/3~2N_T/3）**：关注综合分析特征，路由在各组间均衡分配
- **高层（2N_T/3~N_T）**：关注功能性和风格性特征，路由偏向功能类和风格类专家组

从算法pipeline角度拆解术语。
```
# Layer-wise Routing: 不同层使用不同路由参数
# Layer 1 (低层): 偏重领域知识
W_G^(1) → 领域组 0.5, 功能组 0.3, 风格组 0.2
# Layer 16 (中层): 均衡
W_G^(16) → 领域组 0.35, 功能组 0.35, 风格组 0.3
# Layer 32 (高层): 偏重功能和风格
W_G^(32) → 领域组 0.15, 功能组 0.5, 风格组 0.35

# 每层的完整 forward:
for l in range(N_T):
    x = Attention_LayerNorm(x) + MHA(x)
    F_G_l = adaptive_grouped_routing(x, W_G^(l), W_D^(l), ...)
    y = (λ * F_G_l + (1-λ) * W_p) @ x + W_0 @ x
    x = x + y  # residual
```

这种设计的理论基础来自 Gao et al. (2024) 的发现：高层学习更抽象和高级的信息，这些特征用于下游任务 ("Higher layers need more LoRA experts")。

术语一般如何实现？如何使用？
- 实现为 N_T 个独立的路由参数矩阵集合 {(W_G^(l), W_D^(l)) | l = 1...N_T}
- 训练时对每组路由矩阵分别训练（所有 LoRA 专家冻结）
- 路由矩阵的总参数量 = N_T × (N_dim × N_G + N_G × N_M)，相比 LLM 总参数量可忽略
- 可与 grouped routing 结合使用，也可独立应用
- 目前无开源实现，论文未提供实验验证
- 相关研究：PathMoE (2026) 共享相邻 block 的路由参数以减少路径空间；UniPool (2026) 共享全局专家池但保留独立路由；Omni-Router (2025) 全层共享路由器

涉及论文标题：
- AT-MoE: Adaptive Task-planning Mixture of Experts via LoRA Approach

---

## Expert Selection Pattern / Sample Path (专家选择模式/样本路径)

术语解释
Expert Selection Pattern 是 Lina 论文实证发现的 MoE 跨层关联规律：在 token 流经各 MoE layer 时，相邻层之间 expert 选择呈现可预测模式——在 layer i 中选择同一 expert 的 tokens，在 layer i+1 中倾向于再次选择同一 expert。Sample Path 是 token 连续穿过 l 层时选择的 expert 序列，用于估计下一层的 expert popularity distribution。

术语是什么？
Lina 的实验发现：tokens 在相邻层中选择同一 top-1 expert 的概率为 41.94%（k=1），选择同一 top-2 中任一 expert 的概率为 54.59%（k=2）。更深层该模式更明显。原因：Gate network 架构简单，路由决策主要基于 token 局部特征（POS、词义等）；Expert 专注于局部句法信息（非跨序列依赖）；特征固定于 token，导致相似 token 在各层被相同 expert 处理。

样本路径（Sample Path）定义：token 从 layer i-l 到 layer i 所经过的 expert 序列 `[e_{i-l}, ..., e_i]`。路径长度 l 控制 accuracy-overhead tradeoff。

从算法pipeline角度拆解术语。
```
# Expert Selection Pattern Profiling (Training阶段)
def profile_expert_selection_patterns(model, dataset, path_length=3):
    patterns = defaultdict(dict)  # {layer: {sample_path: distribution}}
    for batch in dataset:
        for layer_i in range(path_length, model.num_layers):
            sample_paths = collections.defaultdict(list)
            for token in batch.tokens:
                path = tuple(token.expert_history[layer_i-path_length:layer_i+1])
                sample_paths[path].append(token)
            # 计算每个 sample path 对应的 layer i+1 expert 分布
            for path, tokens in sample_paths.items():
                next_experts = [token.expert_history[layer_i+1] for token in tokens]
                dist = compute_distribution(next_experts)  # P(e) for each expert
                patterns[layer_i + 1][path] = dist
    return patterns

# Online Estimation (Inference阶段)
def estimate_expert_popularity(batch, current_layer, patterns, path_length=3):
    estimated = defaultdict(float)
    for token in batch.tokens:
        path = tuple(token.expert_history[-path_length:])
        dist = patterns[current_layer + 1].get(path, uniform_dist)
        for expert_id, prob in dist.items():
            estimated[expert_id] += prob / batch.num_tokens
    return {e: N * p for e, p in estimated.items()}  # n_e = N * P(e)
```

关键发现：
- k=1: 41.94% tokens 在相邻层中选择同一 expert
- k=2: 54.59%
- 更深层 pattern 更强（later layers → higher ratio）
- 不是 token 级精准预测，但提供 batch 级 expert popularity estimation

术语一般如何实现？如何使用？
- Profiling: 在 training 阶段 load balancing loss 稳定后采集 expert selection results
- 存储: unordered_map per layer（key: sample path tuple, value: distribution vector）
- 路径长度 l=3 为默认（l=1 accuracy 31.6%→l=3 60.4%→l=6 71.4%）
- 配合 Two-Phase Scheduling: Phase 1 用估算做预分配，Phase 2 偏差大时微调
- 局限性: 需要 training-stage profiling，每个 task 需独立 profile

涉及论文标题：
- Accelerating Distributed MoE Training and Inference with Lina

---

## Adaptive Expert Replication (自适应专家复制)

术语解释
Adaptive Expert Replication 是一种 MoE 分布式训练策略，根据每个 expert class 的动态 token popularity 非均匀地调整 expert 的复制份数（replication degree），使热门 expert 获得更多 replica 以处理更多 token、冷门 expert 减少 replica 以避免 GPU 资源闲置。与传统的 uniform static replication（所有 expert 分配相同数量 replica）不同，adaptive replication 直接解决 MoE 训练中的 convergence-latency tradeoff。

术语是什么？
在传统 MoE 训练中，每个 expert class 被复制固定次数 r = sN/E（s=每 rank slots 数, N=rank 数, E=expert class 数）。由于 expert popularity 高度偏斜且快速变化（SYMI 论文 Figure 2 显示 16× fluctuation 在 3 iterations 内），静态复制导致热门 expert 成为 latency bottleneck 并被迫丢弃超出 capacity 的 token，冷门 expert 的 GPU 资源闲置。

Adaptive Expert Replication 的核心公式：
- 每个 expert class e_i 被复制 r_i 次，其中 r_i ∝ popularity_i
- Σ r_i = sN（总 expert instances 数量不变）
- Effective capacity(e_i) = slot_capacity × r_i（而非固定的 capacity_factor × tokens_per_batch / E）

从算法pipeline角度拆解术语：
SYMI 的 Expert Placement Scheduler (Algorithm 1) 实现 adaptive replication：
```
def compute_placement(popularity, E, G, S):
    # popularity: [E] array, per-expert token counts from previous iteration
    # G: world size, S: slots per rank
    goal = (popularity / sum(popularity)) * G * S  # proportional allocation
    exp_counts = maximum(floor(goal), [1] * E)      # at least 1 replica each
    # Rounding correction to match total slots G*S
    while sum(exp_counts) > G * S:
        i = argmax(exp_counts - goal)
        if exp_counts[i] > 1: exp_counts[i] -= 1
    while sum(exp_counts) < G * S:
        i = argmin(exp_counts - goal)
        exp_counts[i] += 1
    # Contiguous assignment (same-class experts grouped together)
    placement = flatten([[exp_id] * count for exp_id, count in enumerate(exp_counts)])
    return placement  # length = G*S, contiguous same-expert blocks
```
流程：前次迭代的 global popularity（通过 all-reduce 聚合）→ 归一化为比例 → floor + rounding correction → contiguous assignment 优先同 rank 内同 expert replica。

术语一般如何实现？如何使用？
- SYMI 基于 DeepSpeed 实现，以 previous iteration popularity 为 proxy（simple yet effective），per-iteration 更新 placement
- 更复杂的策略可使用历史统计、预测模型、或基于数据集特征的 static replication
- 关键前提：需要 Model-Optimizer State Decoupling 来消除 rebalancing 时的 optimizer state 迁移开销
- 与 Top-k gating、Expert Choice routing、auxiliary-loss-free load balancing 等路由策略正交，可组合使用
- LLama 4 和 DeepSeek-V3 使用 shared + routed experts 混合架构，SYMI 可应用于 routed experts 部分

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---

## Model-Optimizer State Decoupling (模型与优化器状态解耦)

术语解释
Model-Optimizer State Decoupling 是 SYMI 提出的核心系统设计原则：将 MoE 训练中 expert 的模型参数（weights, 2B/param in fp16）与其优化器状态（Adam optimizer: fp32 param + momentum + variance = 12B/param, 总计 16B/param including gradients）在存储和放置上完全分离。Optimizer state 被均匀静态分片到所有 N 个训练节点的 host memory，永不迁移；而 expert weights 在 GPU HBM 上按 popularity 动态调整放置。

术语是什么？
传统系统（DeepSpeed ZeRO-1, FlexMoE）将 optimizer state 与 expert instance 绑定——optimizer shard 仅分布在持有该 expert 的节点上。当 expert 需要 rebalance 到新 GPU 时，必须同时搬运 optimizer state（8× weight size），导致 rebalancing iteration 延迟为正常的 2.46×-4.10×（FlexMoE 数据），迫使系统只能粗粒度 rebalance（每 50-100 iterations）。

SYMI 的解耦设计：
- **Static optimizer**: optimizer[e_i] 均匀切分为 N 份，分布在所有 N 个节点的 host memory 中，永不迁移
- **Dynamic expert placement**: expert weights 在 GPU slot 上的分配每 iteration 可任意变化
- **No-overhead rebalancing**: Weight Communication Phase 的数据量 = sNW（与 static baseline 完全相同），因为每个 slot 始终接收一个完整的 expert weight，无论 expert class 是否改变

从算法pipeline角度拆解术语：
```
# SYMI Training Iteration (per layer, per rank):
# === State Layout ===
# optimizer_state[e_i] partitioned across ALL N nodes (host memory, static)
# expert_weights[slot_j] on GPU HBM (dynamic, changes per iteration)

# Forward: Router aggregates popularity → all-reduce → store in Layer Metadata Store
popularity[t] = allreduce(count_tokens_per_expert())

# Optimizer Step: Gradient Communication Phase (same data volume as static)
for each optimizer_partition on node_k:
    grad_shard = collect_from_source_expert_instances()  # Algorithm 2
    optimizer.step(grad_shard)  # Adam update → updated weight shard

# Optimizer Step: Expert Placement Scheduling (for iteration t+1)
placement[t+1] = compute_placement(popularity[t])  # Algorithm 1

# Optimizer Step: Weight Communication Phase (key insight - no extra data!)
for each slot_j:
    expert_id = placement[t+1][slot_j]
    # Send updated weights to slot_j - SAME data volume regardless of expert_id!
    send(updated_weights[expert_id], to=slot_j)
    # If expert_id changed: slot_j receives DIFFERENT expert's weights (same size W)
    # If expert_id unchanged: slot_j receives SAME expert's weights (same size W)
    # Data volume per slot = W bytes, ALWAYS.
```

通信量不变性证明：
- Grad Phase: D_G^SYMI = Σ r_i × G/N × N = sNG = D_G^static
- Weight Phase: D_W^SYMI = Σ N × W/N × r_i = sNW = D_W^static
- 仅 locality shift 引入约 1.52% 额外通信时间（N=2048, E=64, s=2）

术语一般如何实现？如何使用？
- SYMI 基于 DeepSpeed 实现，optimizer offload 至 CPU host memory (ZeRO-1 风格)
- 解耦设计不强制 optimizer 必须在 host memory——也可均匀分片在 GPU HBM 中（Appendix A.5），仅 locality 略变
- 与 tensor parallelism、pipeline parallelism、expert-sharding parallelism 正交兼容
- 关键约束：需要高效的跨节点梯度收集和权重分发通信（SYMI 使用 batch point-to-point + pre-registered NCCL groups）

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)

---

## Auxiliary Load-Balancing Loss (辅助负载均衡损失)

术语解释
Auxiliary Load-Balancing Loss 是 MoE 训练中添加到主语言模型损失上的辅助损失项，用于鼓励 gate network 将 token 均匀分配到各 expert，避免某些 expert 过载（overloaded）而其他 expert 闲置（underutilized）。首次由 Shazeer et al. (2017) 在 Sparsely-Gated MoE 中引入，后由 Switch Transformer (Fedus et al., 2022) 推广。

术语是什么？
Load-balancing loss 的标准形式（Switch Transformer）：
$$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

其中 f_i = 分配给 expert i 的 token 比例，P_i = router 分配给 expert i 的平均 gate probability，N = expert 总数，α = auxiliary loss coefficient（典型值 10^-2 ~ 10^-5）。

符号化辅助损失的变体（DeepSeek-V3, Wang et al., 2024）直接在 router scores 中注入 expert-level bias，而非修改 loss function，实现 auxiliary-loss-free load balancing。

从算法pipeline角度拆解术语：
MoE 训练 loss 组成：
```
L_total = L_LM + α × L_aux
# L_LM: cross-entropy language modeling loss
# L_aux: load balancing penalty
# α: coefficient controlling the tradeoff

L_aux = E × Σ_i (f_i × P_i)
# f_i = (1/T) × Σ_t 1[token t routed to expert i]  # fraction of tokens
# P_i = (1/T) × Σ_t softmax(gate_logits[t])[i]      # avg routing probability
```

术语一般如何实现？如何使用？
- **α 调优至关重要**：SYMI 论文 Figure 11 显示 DeepSpeed 需要高 α (~10^-1) 才能将 token drop 从 ~40% 降至 ~10%，但高 α 干扰主 loss 收敛
- **SYMI 的发现**：有了 adaptive expert replication 后，SYMI 在任何 α 下均保持 ~10% token drops，auxiliary loss 从"系统必需项"降级为"质量调节旋钮 (quality knob)"
- 替代方案：Expert-Choice routing (Zhou et al., 2022) 天然负载均衡无需 auxiliary loss；BASE Layers (Lewis et al., 2021) 用线性分配保证均衡；DeepSeek 的 auxiliary-loss-free 策略 (Wang et al., 2024) 直接在 router 中注入 bias；ARIA 的 group-level load balancing（见 Group-Level Load Balancing Loss）将 per-expert 约束松弛为 per-group 约束；**DSMoE 刻意不引入 load balancing loss**（见下文）
- **DSMoE 的"无 Load Balancing"设计**：DSMoE 明确不引入 load balancing loss，因为其目标不是 expert 均匀使用，而是学习输入自适应的稀疏激活模式。DSMoE 的 sigmoid 门控（非 softmax）使 expert 激活决策互不依赖，配合 L1 sparse loss 施加稀疏压力，形成"STE 让所有 expert 保持可训练 + sparse loss 鼓励选择性激活"的对抗训练机制。论文在 10B tokens 继续预训练后未观察到严重的 expert 负载不均，且 W 形层间激活模式表明不同层自然形成不同的激活水平
- **AquilaMoE 实践**：AquilaMoE 训练 8×16B MoE 时使用 α=0.001 的 load balancing loss + α=0.01 的 max z-loss，两者均以乘法系数形式施加于最终训练目标，用于防止训练崩溃并维持 expert 负载均衡。Scale-Out 阶段每 token 激活 top-2/8 experts（约 30B 激活参数）

**Micro-Batch vs Global-Batch LBL（Demons in the Detail, 2025）**：

Qiu et al. (2025) 揭示了 LBL 计算粒度对 MoE 性能和 expert specialization 的关键影响：

- **Micro-batch LBL (LBL_micro)**：主流框架（DeepSpeed-MoE, Tutel, MegaBlocks, Megatron-Core）的默认行为。每个 parallel group（即每个 GPU 的 micro-batch）内独立计算 f_i 和 P_i，然后 all-gather 平均。在大模型训练中 micro-batch 仅含极少序列（数千 tokens），LBL 退化到序列级均衡——强制每个序列内的 token 均匀分配到所有 expert。这抑制了 domain-level expert specialization。
- **Global-batch LBL (LBL_global)**：跨并行组同步专家选择频率 f_i（仅 N_E 维向量），用全局 f̄_i 替换本地 f_i 计算 LBL。约束从"每序列内均匀"放松为"全语料库均匀"。额外通信开销 <1%。
- **Buffer 近似机制**：当计算节点有限（微批总和 < 全局批大小），在 GA 各步缓冲累积同步后的 c_i 逐步逼近 global f̄_i。
- **Balance BSZ**：论文引入的度量指标，表示计算专家选择频率时考虑的总 token 数。实验证明 Balance BSZ 从 2 增加到 512，PPL 持续下降 (~0.185)。
- **Shuffle LBL_micro 消融**：通过 all-gather token-expert selection matrix G 并随机抽取等量 token 计算 LBL（保持 token 数与 micro-batch 相同但分布等同 global-batch），证实性能提升来自 **token 多样性**而非 token 数量（方差降低）。
- **缓解局部负载不均**：Global-batch LBL 可能导致局部计算不均（~5.8% slowdown）。加微量 micro-batch LBL（1% weight of global-batch）可将速度恢复至仅 2.6% 慢于 baseline，性能损失极小。

涉及论文标题：
- SYMI: Accelerating Mixture-of-Experts Training with Adaptive Expert Replication
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- Continual Pre-training of MoEs How robust is your router（CPT 中 PBTk routing 使用 α=0.01 的 aux loss + λ=0.001 的 z-loss。CPT 分布偏移时 PBTk 经历短暂 MRI spike 后 ~500 steps 恢复，aux loss 学习新的负载均衡模式。与 SBTk 的显式均衡相比，PBTk 的 penalty-based 方法恢复后的 MRI 更低）

涉及论文标题：
- Accelerating Mixture-of-Experts Training with Adaptive Expert Replication (SYMI)
- A Survey on Mixture of Experts in Large Language Models
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models
- AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts
- Adaptive Gating in Mixture-of-Experts based Language Models
- Beyond Distillation Task-level Mixture-of-Experts for Efficient Inference
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models

**Task-MoE 的负载均衡（Kudugunta et al., EMNLP 2021）**：
Task-MoE 使用 task-level routing，所有 token 按 task 预先分组 → 同 task 的 token 必然路由到相同 experts → task 内的 expert load 天然均衡。负载均衡仅在跨 task 之间需要（确保不同 task 不过度集中到少数 experts）。论文使用 standard auxiliary load balancing loss (α=0.001) with top-2 gating。128 expert 配置下 decoder 仅 2/128 experts per task，专家负载天然跨 task 分散。

Li et al. (EMNLP 2023) 对标准 load balancing loss 做了最简洁的修改以适应 flexible expert count：

**核心修改**：由于 adaptive gating 中 token 可能使用 1 或 2 个 expert，load balancing loss 仅对 top-1 gating 决策施加软约束，top-2 gating 决策完全自由：

$$L_i = E_i \sum_{e \in E} f_e^1 \cdot p_e$$

其中 $f_e^1$ 为 top-1 gating token 中分派到 expert e 的比例（而非所有 token），$p_e$ 为所有 token 对 expert e 的平均门控概率，$E_i$ 为第 i 层 expert 数量。

**设计理由**：top-2 决策代表 token 确实需要双专家处理的"困难"情况，对这些 token 施加负载均衡约束会干扰其学习。仅约束 top-1 决策在保证基本负载均衡的同时，给予 router 在困难 token 上完全的路由自由度。

与其他负载均衡变体的对比：
- Standard (Switch Transformer): 对所有 token 施加约束 → 不适用灵活 expert 数
- AdaMOE ℓ_null: null experts 间不做负载均衡 → 适用于 null expert 范式
- Adaptive Gating: 仅 top-1 决策施加约束 → 最简修改，配合阈值门控

### AdaMOE 的 Null Expert Load Balancing Loss (ℓ_null)

AdaMOE 对标准 load balancing loss 做了关键修改以适应 null experts：

**修改 1 — 不对 null experts 之间做负载均衡**：
由于所有 m 个 null experts 在功能上完全相同（均为 zero mapping），对它们之间做负载均衡区分会施加不必要的约束。AdaMOE 将 null experts 的负载因子 f_j 替换为均值：

$$\tilde{f}_i = \begin{cases} f_i & \text{if } i \leq n \text{ (true expert)} \\ \frac{1}{m} \sum_{j=n+1}^{n+m} f_j & \text{if } i > n \text{ (null expert)} \end{cases}$$

$$\ell_{null} = \alpha \cdot (n+m) \cdot \sum_{i=1}^{n+m} \tilde{f}_i \cdot P_i$$

实验验证 ℓ_null 显著优于对所有 null experts 做负载均衡的 ℓ_bal：RTE accuracy 67.51 vs 56.68, COLA 85.01 vs 83.68。

**修改 2 — α annealing 策略**：
- Epoch 1: α=0.02（大 α）→ 严格负载均衡，确保 tokens 不全部涌向 true experts
- Epoch 2: α=0.0001（小 α）→ 释放 token 自由度，让 router 根据任务需求自由分配
- 效果: WINO accuracy 从 epoch 1 的 76.24 提升至 epoch 2 的 81.93 (+5.69%)，Load 几乎不变 (1.65→1.66)

**修改 3 — Normalization 策略**：
仅对 top-k 中选中的 true experts 做 Softmax normalization（option 2），而非对所有 k 个选中的 expert（含 null）做 Softmax（option 1）。保证加权输出与 vanilla MoE 数值尺度一致。SIQA: option 2 accuracy 81.27 vs option 1 80.19。

---

## Ada-K Routing

术语解释
Ada-K 是一种基于强化学习的动态 MoE 路由策略，通过可学习的轻量级 allocator 模块为每个 token 动态决定激活的专家数量，替代传统固定 Top-K 路由。

术语是什么？
Ada-K Routing 的核心架构：
1. **Allocator 模块**：在每个 MoE layer 插入一个与 router 同结构的轻量级线性层 W_alloc ∈ R^{C×N}，输入 token embedding x_i，输出专家数量概率分布 P_alloc(x_i) = Softmax(W_alloc · x_i)，通过采样 k* ~ P_alloc(x_i) 动态决定该 token 应激活的专家数量。
2. **PPO 训练框架**：由于采样不可微分，使用 Proximal Policy Optimization 端到端训练 allocator。每个 MoE layer 的 allocator 作为 agent (policy π_θ)，token 的 hidden state 作为 state，采样的专家数量作为 action，仅最后一层接收 reward = log P(x_i|x_1,...,x_{i-1})（语言模型对数似然）。Advantage 函数使用 reinforce with baseline 形式，以默认 Top-K 路由输出为 baseline。
3. **Warm-Start (P-Warm)**：使用 Top-P nucleus sampling 生成伪标签预训练 allocator，避免随机初始化导致的训练不稳定。
4. **可插拔设计**：allocator 与原始 router 独立，LLM 主干参数完全冻结，训练仅更新 allocator 参数。

从算法pipeline角度拆解术语。
```
# Ada-K Forward Pass (per token x_i, per layer l)
def adak_forward(x_i, W_router, W_alloc, experts):
    # x_i: [d_model], W_alloc: [d_model, N], W_router: [d_model, N]
    
    # Step 1: Allocator 决定专家数量
    P_alloc = Softmax(W_alloc @ x_i)        # [N] 各 k 值的概率
    k_star = Categorical(P_alloc).sample()   # 采样 (不可微分!)
    
    # Step 2: Router 选择 top-k* 专家
    P_router = Softmax(W_router @ x_i)      # [N] 专家概率
    top_indices = TopK(P_router, k_star)
    top_weights = Softmax(P_router[top_indices])
    
    # Step 3: 加权聚合专家输出
    output = sum(w * expert_j(x_i) for w, expert_j in zip(top_weights, top_indices))
    return output, k_star, log_prob

# Ada-K PPO Training
def ppo_training_step(token_batch, allocators, baseline_model):
    # Forward: 收集 actions 和 log_probs
    for layer in layers:
        for token in token_batch:
            out, k_star, log_prob_old = adak_forward(token, ...)
            save(k_star, log_prob_old)
    
    # Reward: 仅最后一层 (L = total layers)
    R = log P(token | context)                    # LM 对数似然 (Ada-K)
    R_baseline = log P_baseline(token | context)  # 默认 Top-K 输出
    
    # Advantage (reinforce with baseline)
    for layer l in 1..L:
        A_l = gamma^{L-l} * (R - R_baseline)
    
    # PPO Loss (2 PPO epochs)
    for layer l in 1..L:
        r = pi_theta(k_star | x_i) / pi_theta_old(k_star | x_i)
        L_RL = -min(r * A_l, clip(r, 1-eps, 1+eps) * A_l)
        
        # Regularization: 最小化期望专家数量
        L_reg = (1/L) * sum(n * P_theta_l(n) for n in 1..N)
        
        L_total = L_RL + lambda * L_reg
        theta_l = AdamW(L_total, lr=1e-3)
```

术语一般如何实现？如何使用？
- Allocator 等价于一个线性层 + SoftMax（与 router 同规模，约 C×N 参数），训练仅需 1M-3M 参数（vs 140B+ 总参数）
- 训练数据仅需 10k 样本，1 epoch，16 GPU (A800) 最慢 8 小时
- λ=3e-3 作为性能与效率的平衡点；调整 λ 可灵活控制 trade-off
- 完全可插拔：allocator 独立于原始 router 和 LLM 主干，可应用到任何 routing-based MoE 模型（包括 shared expert 架构）
- 保持负载均衡：router 冻结确保专家负载分布不变
- Allocator 可选择性部署（如仅 50% 层），以进一步减少训练开销
- 代码和 checkpoint 将发布于 https://github.com/ivattyue/Ada-K

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---

## MoE Allocator (Expert Count Allocator)

术语解释
MoE Allocator 是 Ada-K 路由中引入的轻量级可学习模块，负责为每个 token 动态决定应激活的最优专家数量，与原始 router 协同工作实现自适应专家分配。

术语是什么？
Allocator 是一个可训练的线性层 W_alloc ∈ R^{C×N}（C = hidden_dim, N = num_experts），输入 token 的 hidden state，输出该 token 应激活 1 到 N 个专家的概率分布。结构上与 MoE router 完全同构——两者都是线性投影 + SoftMax，但功能不同：router 决定"激活哪些专家"，allocator 决定"激活多少个专家"。

与 router 的关键区别：
- Router: 输出在 N 个专家上的概率分布 P_router ∈ R^N，Top-K 选择哪 K 个专家
- Allocator: 输出在 N 个可能的 k 值上的概率分布 P_alloc ∈ R^N，采样决定 K = k*
- Allocator 先于 router 执行：k* = sample(Softmax(W_alloc · x_i)) → Router 再执行 TopK(P_router, k*)

从算法pipeline角度拆解术语。
```
# Allocator = 一个小型线性层 + SoftMax + 采样
class MoEAllocator(nn.Module):
    def __init__(self, d_model, num_experts):
        self.linear = nn.Linear(d_model, num_experts)  # W_alloc
        # 输出维度 = num_experts (每个可能的 k 值一个 logit)
    
    def forward(self, x):
        # x: [batch, seq, d_model]
        logits = self.linear(x)              # [batch, seq, N]
        probs = F.softmax(logits, dim=-1)     # 概率分布 over k=1..N
        k_star = torch.multinomial(probs, 1)  # 采样 (不可微分!)
        log_prob = torch.log(probs.gather(-1, k_star))
        return k_star, log_prob

# Allocator 集成到 MoE Layer
class MoELayerWithAllocator(nn.Module):
    def __init__(self, d_model, num_experts):
        self.router = nn.Linear(d_model, num_experts)   # 冻结
        self.allocator = MoEAllocator(d_model, num_experts)  # 可训练
        self.experts = nn.ModuleList([FFN() for _ in range(num_experts)])  # 冻结
    
    def forward(self, x):
        k_star, log_prob = self.allocator(x)  # 先 allocator
        router_probs = F.softmax(self.router(x), dim=-1)
        topk_vals, topk_idx = torch.topk(router_probs, k_star, dim=-1)
        # ... expert computation ...
```

术语一般如何实现？如何使用？
- 实现为一个简单的 `nn.Linear(d_model, num_experts)` + SoftMax + 多项分布采样
- 训练时：采样操作的前向不可微分，需通过 PPO/REINFORCE 类 RL 算法优化
- 推理时：可直接取 argmax (k* = argmax(P_alloc))，无需采样
- 训练参数量极小：Mixtral-8x22B 仅 2.75M 可训练参数（每层 allocator ~49k 参数）
- 可选择性部署：可根据 layer ratio 决定在多少层插入 allocator（实验显示 ratio=1.0 即每层部署最优）
- 训练无关数据域：使用 10k pretrain 或 SFT 数据均可获得相近效果

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---

## PPO for MoE Routing Optimization

术语解释
将 Proximal Policy Optimization (PPO) 强化学习算法应用于 MoE 路由策略优化，通过端到端训练路由 agent（allocator）来最大化语言模型预测质量与计算效率的加权目标，绕过路由决策的非可微分问题。

术语是什么？
在 MoE 路由场景中，allocator 的采样操作 k* ~ P_alloc(x_i) 是不可微分的，无法通过标准反向传播优化。PPO 将路由建模为 sequential decision-making 问题：
- **Agent**: 每层的 allocator (policy π_θ_l)
- **State s_l**: token 在第 l 层的 hidden state x_i^(l)
- **Action ĉ_l**: 采样得到的专家激活数量 k*
- **Reward**: 仅最后一层 (l=L) 的 agent 接收 reward = log P(x_i|x_1,...,x_{i-1})（即语言模型的对数似然，等价于 NLP caption loss 的负值）
- **Discount factor γ**: 在训练中控制远期 reward 的折扣

关键设计：
1. **Reward 仅分配给最后一层**：因为语言模型的预测质量仅在最终 output token 中体现，中间层的"贡献"通过 advantage 传播
2. **Advantage = reinforce with baseline**：A_l = γ^{L-l}[R(ĉ_L, s_L) - R(c*_L, s*_L)]，其中 baseline 为默认 Top-K 路由的 reward。baseline 减除了方差，使训练更稳定
3. **无需额外的 value network**：与标准 PPO 不同，Ada-K 不需要 value function 来估计 advantage，直接使用 reinforce with baseline 形式
4. **仅需 2 PPO epochs**：因为 action space 较小（最多 N 个选择），训练快速收敛

从算法pipeline角度拆解术语。
```
# PPO Training for Ada-K Allocators
# 仅优化 allocator 参数 θ = {θ_1, ..., θ_L}，LLM 主干冻结

for epoch in 1..2:  # 2 PPO epochs
    for batch in dataloader:
        # === Forward Pass (收集 experience) ===
        for layer l in 1..L:
            for token x_i:
                P_alloc = Softmax(W_alloc[l] @ x_i)
                k* ~ Categorical(P_alloc)           # action
                save_old_prob(π_θ_old(k* | x_i))    # old policy log prob
                # Router (frozen) -> TopK(k*) -> Expert FFN -> hidden state
        
        # === Reward 计算 ===
        R = cross_entropy_loss(logits, labels)      # 负的 language modeling loss
        R_baseline = compute_baseline_reward(x, labels)  # 默认 Top-K 输出
        
        # === Advantage (reinforce with baseline) ===
        for layer l in L..1:  # 从后往前
            A[l] = γ^{L-l} * (R[l] - R_baseline[l])
        
        # === PPO Loss + Regularization ===
        for layer l in 1..L:
            π_θ_new = Softmax(W_alloc[l] @ x_i)[k*]
            r = π_θ_new / π_θ_old                # importance sampling ratio
            L_clip = min(r * A[l], clip(r, 1-ε, 1+ε) * A[l])
            
            # Activation Regularization (期望 k 最小化)
            L_reg = Σ_n n * P_alloc[n]           # 每层期望专家数
            
            L = -L_clip + λ * L_reg
        
        # === Update ===
        θ = AdamW(L)  # 仅更新 allocator 参数
```

术语一般如何实现？如何使用？
- 使用 AdamW optimizer，learning rate = 1e-3，batch size = 64
- PPO clip 参数 ε 通常设 0.2（论文未明确说明具体值）
- 训练 1 epoch over 10k 样本，仅 2 PPO epochs
- λ = 3e-3 作为性能与效率的平衡参数
- 与 RLHF 中的 PPO 有根本区别：RLHF 需要一个独立的 reward model (RM) 评估生成质量；而 Ada-K 的 reward 直接从 language modeling cross-entropy loss 派生，无需外部 RM
- 硬件：Mixtral-8x22B 使用 16×A800-80G，其他模型 8×A800-80G

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---

## P-Warm Strategy (Top-P Warm Start for MoE Routing)

术语解释
P-Warm 是 Ada-K 提出的 allocator 预训练策略，利用 nucleus sampling (Top-P) 从原始 router 的专家概率分布中生成伪标签，warm-start 训练 allocator，避免随机初始化导致 RL 训练初期的采样不稳定。

术语是什么？
P-Warm 的核心思想：
1. **Top-P Nucleus Sampling**: 对每个 token x_i，按 router 输出概率降序排列专家，选择最小子集使其累积概率 ≥ p，该子集大小 n_i(p) 作为"应激活专家数量"的伪标签
2. **最优 p 选择**: n_i(p) = argmin_{k} Σ_{j≤k} P_{i,j}^↓ ≥ p。在所有可能的 p 值中，选择使平均 n_i(p) 最接近默认 Top-K 值 k 的 p*：
   $$p^* = \operatorname{argmin}_p \left|\frac{1}{T}\sum_{i=1}^{T} n_i(p) - k\right|$$
3. **伪标签监督训练**: 使用 n_i(p*) 作为 cross-entropy loss 的目标，预训练 allocator 使其输出分布接近 Top-P 导出的伪标签

从算法pipeline角度拆解术语。
```
# P-Warm 伪代码
def p_warm_pretrain(router, allocator, tokens, k_default):
    # Step 1: 选择最优 p
    best_p, best_diff = None, inf
    for p in [0.1, 0.2, 0.3, ..., 0.9]:
        labels = []
        for token in tokens:
            P_router = Softmax(router(token))
            P_sorted = sort_descending(P_router)
            n = argmin_k(cumsum(P_sorted)[:k] >= p)
            labels.append(n)
        avg_n = mean(labels)
        diff = abs(avg_n - k_default)
        if diff < best_diff:
            best_p, best_diff = p, diff
    
    # Step 2: 使用伪标签训练 allocator
    for epoch in warm_start_epochs:
        for token in tokens:
            P_router = Softmax(router(token))
            P_sorted = sort_descending(P_router)
            pseudo_label = argmin_k(cumsum(P_sorted)[:k] >= best_p)
            
            P_alloc = Softmax(allocator(token))
            L = CrossEntropy(P_alloc, pseudo_label)
            optimizer.step(L)
    
    # Step 3: 初始化完成，进入 PPO 训练
```

术语一般如何实现？如何使用？
- 对每个 baseline model，独立计算最优 p* 值（论文中使用的 threshold p=0.3）
- 在 moderate amount of tokens (T) 上计算 n_i(p) 的平均值来确定 p*
- P-Warm 仅需少量 warm-start epochs
- 消融实验显示：P-Warm (Acc=55.13) > K-Warm (Acc=54.97) > Random (Acc=54.18)，证明灵活的 Top-P 伪标签优于固定 K 值伪标签
- 核心优势：让 allocator 初始输出接近"自然"的专家分布（router 自身的专家概率累积），加速 PPO 训练收敛

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs

---

## Activation Regularization (MoE Expert Count)

术语解释
Activation Regularization 是 Ada-K 训练中用于控制专家激活数量的正则化损失，通过最小化所有层 allocator 输出分布的期望值，直接减少平均激活专家数量，在训练中与 PPO loss 共同优化。

术语是什么？
Activation Regularization Loss 的公式：
$$\mathcal{L}^{reg}(\theta) = \frac{1}{L}\sum_{l=1}^{L}\sum_{n=1}^{N} n \cdot \mathcal{P}_{\theta_l}(n)$$

其中：
- L 为 MoE layer 数量
- N 为每层的专家总数
- P_θ_l(n) 为第 l 层 allocator 输出激活 n 个专家的概率
- n · P_θ_l(n) 为第 l 层激活专家数量的期望值

该损失项**可微分**，因为它直接优化 allocator 输出的概率分布期望，而非通过采样操作。因此可以同时参与标准梯度反向传播（"As Loss"模式），或作为 reward 项合并到 PPO objective 中（"As Reward"模式）。

从算法pipeline角度拆解术语。
```
# Activation Regularization: "As Loss"模式 (default)
# 直接计算 allocator 输出分布的期望值并反向传播

def activation_regularization(allocator_outputs, L):
    # allocator_outputs: list of [batch, seq, N] for each layer
    # L: number of MoE layers
    total_reg = 0
    for l in range(L):
        P = allocator_outputs[l]           # [batch, seq, N]
        n_range = arange(1, N+1)            # [1, 2, ..., N]
        expectation = sum(n_range * P, dim=-1)  # [batch, seq]
        total_reg += expectation.mean()    # average over batch & seq
    return total_reg / L

# 总损失
L_total = L_RL + λ * L_reg
# λ = 3e-3 as default trade-off coefficient
```

术语一般如何实现？如何使用？
- "As Loss"模式（直接优化期望）比"As Reward"模式（将期望纳入 reward）表现略优：Acc=55.13 vs 54.64
- λ 控制性能与效率的 trade-off：更大的 λ → 更强的激活压缩 → 更高的效率但可能降低性能
- 在 activation reduction rate 达 44% 前，Ada-K 性能始终高于 baseline（见图 2 trade-off curve）
- λ 的扫描过程：为每个模型单独扫描 λ 值生成 trade-off curve，选择最优平衡点（论文统一使用 λ=3e-3）
- 该正则化项的梯度直接通过 allocator 输出概率反向传播，不经过采样操作，因此与 PPO loss 互补：PPO 优化采样决策质量，regularization 优化期望激活数量

涉及论文标题：
- Ada-K Routing Boosting the Efficiency of MoE-based LLMs
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

---

## Null Experts (空专家)

术语解释
Null Experts（空专家）是 AdaMOE 提出的核心机制，指在 MoE layer 的 expert set 中引入的固定数量的"空操作"专家。Null expert 定义为一个消耗 **零 FLOPs** 的空操作（默认 zero mapping: E_null(x) = 0，也可选 identity mapping: E_null(x) = x），在 top-k 路由中被选中时不执行任何计算。通过将 null experts 与 true experts 混合，并增大 top-k 的 k 值，使不同 token 可选择不同数量的 true experts，实现 token-adaptive routing。

术语是什么？
Null experts 的关键特性：
1. **零 FLOPs**: E_null(x) = 0（constant zero mapping），不消耗任何计算资源。可选 identity mapping E_null(x) = x 也消耗零 FLOPs，但论文未探索此方案。
2. **等质无差别**: 所有 m 个 null experts 在功能上完全相同，因此在 load balancing loss 中不对 null experts 之间做负载均衡区分。
3. **token bypass 能力**: 若某 token 在 top-k 中全部选中 null experts，该 token 完全绕过此 MoE layer，实现类似 Mixture-of-Depths (MoD) 的 layer skipping 效果。
4. **计算预算可控**: 通过调整 m（null expert 数量）和 k（top-k 值），可精确控制平均 true expert 负载。Load = k × (n/(n+m))（无 load balancing loss 时的理论值）。

从算法pipeline角度拆解术语。
```
# Null Expert 定义
class NullExpert:
    """空专家: 零 FLOPs 空操作"""
    def forward(self, x):
        # Zero mapping (default): output = 0, 0 FLOPs
        return torch.zeros_like(x)
        # Identity mapping (alternative): output = x, 0 FLOPs
        # return x

# AdaMOE Layer with Null Experts
class AdaMOELayer:
    def __init__(self, n_true_experts, m_null_experts, k):
        self.true_experts = [FFN() for _ in range(n_true_experts)]  # E_1...E_n
        self.null_experts = [NullExpert() for _ in range(m_null_experts)]  # E_{n+1}...E_{n+m}
        self.router = Linear(d_model, n_true_experts + m_null_experts)  # W_g
        self.k = k  # top-k selection, k > vanilla MoE's k
    
    def forward(self, x):  # x: [d_model]
        # Step 1: Router 计算所有 expert (含 null) 的 logits
        logits = self.router(x)  # [n+m]
        
        # Step 2: Top-K 选择
        top_logits, top_indices = topk(logits, self.k)
        
        # Step 3: 分离 true experts 和 null experts
        true_mask = top_indices < self.n_true_experts
        null_mask = ~true_mask
        
        if true_mask.sum() == 0:
            # 全部选中 null experts → token bypass this layer
            return torch.zeros_like(x)
        
        # Step 4: 仅对 true experts 做 Softmax (option 2)
        true_logits = top_logits[true_mask]
        true_weights = softmax(true_logits)
        
        # Step 5: 仅 true experts 贡献计算，null experts 贡献 0
        output = sum(
            true_weights[i] * self.true_experts[idx](x)
            for i, idx in enumerate(top_indices[true_mask])
        )
        # null experts: weight * 0 = 0, 无 FLOPs
        return output
```

术语一般如何实现？如何使用？
- **对 vanilla LLM (Mo-LoRA)**: 在 Mo-LoRA 架构中，每个 layer 的 LoRA experts 作为 true experts（n=4），添加 m=5~9 个 null experts（实现为不执行任何操作的占位符），k=2~4。使用 mola-moe 框架实现。
- **对 MoE-LLM (Mixtral)**: 原始 router gate 输出 n=8 → 新增 gate2 module 输出 m=8~48 → 拼接为 n+m 维 router → k=3~8。gate2 参数可从 gate 复制推导。
- **Load Balancing**: ℓ_null = α·(n+m)·[Σ_{i≤n} f_i·P_i + Σ_{j>n} (avg_f_null)·P_j]，不对 null experts 间做负载均衡。
- **Training**: α annealing: epoch 1 用大 α (0.02) 建立负载均衡，epoch 2 用小 α (0.0001) 释放 token 自由度。
- **关键结果**: Mixtral-8x7B fine-tuning: Load 从 2.00→1.66，FLOPs ↓14.5%，accuracy +1.69% on ARC-C。

涉及论文标题：
- AdaMOE Token-Adaptive Routing with Null Experts for Mixture-of-Experts Language Models

## Low-Rank Adaptation (LoRA)

术语解释
LoRA (Low-Rank Adaptation) 是一种参数高效微调方法，通过在预训练权重旁添加低秩分解矩阵实现任务适配，而不修改原模型参数。

术语是什么？
LoRA 由 Hu et al. (2021) 提出，核心公式为 h = W_0 x + ΔW x = W_0 x + BA x，其中 W_0 ∈ R^{d×k} 为冻结的预训练权重，A ∈ R^{r×k} 和 B ∈ R^{d×r} 为可训练的低秩矩阵，r ≪ min(d,k) 为秩。推理阶段 B 和 A 可与 W_0 合并：W' = W_0 + BA，不引入额外延迟。A 通常用 Kaiming 初始化，B 用零初始化以保证训练起始 ΔW = 0。输出缩放因子 α/r 控制 LoRA 更新的幅度。

关键属性：
1. **参数效率**：仅训练 2·r·(d+k) 参数，相比全量 fine-tuning 的 d·k 参数大幅减少
2. **可插拔性**：LoRA 模块独立于 base model 存储，支持灵活任务切换
3. **多目标**：可作用于 attention 的 Q/K/V/O 投影矩阵（Wq, Wk, Wv, Wo），也可扩展到 FFN 层

从算法pipeline角度拆解术语。
```
# LoRA Linear Layer (替换普通 nn.Linear)
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha):
        self.W0 = nn.Linear(in_features, out_features, bias=False)
        self.W0.weight.requires_grad = False  # 冻结
        self.A = nn.Parameter(torch.randn(rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base = self.W0(x)                       # W_0 x
        delta = (x @ self.A.T) @ self.B.T       # B A x
        return base + delta * self.scaling
```

AdaMoLE 将 LoRA 作为 MoE 的 expert 使用：每个 expert E_i 是一个独立的 LoRA adapter (B_i A_i)，共享基础权重 W_0，路由器选择最合适的 LoRA expert 组合应用于每个输入。N 个 rank-r/N 的 LoRA expert 在参数量上等价于一个 rank-r 的单 LoRA。

术语一般如何实现？如何使用？
- HuggingFace PEFT 库（`peft.LoraConfig`）：配置 target_modules、rank r、alpha、dropout
- 推理加载：`model.load_adapter()` 或合并权重 `model.merge_and_unload()`
- LoRA 变体：DoRA（方向-幅度解耦）、AdaLoRA（自适应 rank）、QLoRA（量化+LoRA）
- 在 AdaMoLE 中：N=8 个 rank=4 LoRA expert，总秩=32，作用于 Wq/Wk/Wv/Wo

涉及论文标题：
- AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts

---
## Mixture of LoRA Experts (MoLE)

术语解释
Mixture of LoRA Experts (MoLE) 是将 Mixture of Experts 架构与 LoRA 结合的参数高效微调范式，在每层用多个 LoRA expert 替代单个 LoRA adapter，通过路由机制为不同输入动态选择和组合 LoRA 专家。

术语是什么？
MoLE 的核心思想：在 Transformer 层的目标权重矩阵（如 attention Q/K/V/O）上，创建 N 个独立的 LoRA adapter（各 rank = r_total / N），一个可训练的路由器计算 Softmax 选择概率，Top-K 选出最相关的 LoRA 专家：
$$h = W_0 x + \sum_{i \in \text{TopK}(p, K)} \frac{p_i}{\sum_{j \in \text{TopK}(p, K)} p_j} \cdot B_i A_i x$$

其中 p = Softmax(W_g x) 为路由器输出。

**与标准 MoE 的差异**：MoLE 的 expert 是 LoRA adapter（仅修改权重更新 ΔW 而非完整 FFN），路由作用于 fine-tuning 时的参数更新而非模型前向计算路径。

**关键设计维度**：
1. **专家粒度**：每 expert 作用于单个权重矩阵，N 专家共享总秩（参数等价于单 LoRA）
2. **门控策略**：top-k（k=1,2,3）、固定阈值 τ=1/N、动态阈值 τ(x)（AdaMoLE）
3. **层级位置**：通常应用于 self-attention 四矩阵（Wq, Wk, Wv, Wo）

从算法pipeline角度拆解术语。
```
# MoLE Forward (top-2 gating on self-attention Q projection)
def mole_forward(x, W_q, lora_experts, router):
    base = x @ W_q.T                       # pretrained output
    logits = router(x)                     # [batch, seq, N]
    probs = F.softmax(logits, dim=-1)
    topk_probs, topk_indices = torch.topk(probs, k=2, dim=-1)
    topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
    
    delta = torch.zeros_like(base)
    for k in range(2):
        for expert_idx in topk_indices[:,:,k].unique():
            mask = (topk_indices[:,:,k] == expert_idx)
            delta[mask] += topk_probs[:,:,k][mask].unsqueeze(-1) * lora_experts[expert_idx](x[mask])
    return base + delta
```

AdaMoLE 将 top-k 替换为动态阈值：激活所有 p_i ≥ τ(x) 的 expert，τ(x) 由阈值网络生成。

术语一般如何实现？如何使用？
- 配置：N × r = total_rank（如 N=8, r=4 → total=32）
- 门控变体：top-k（MoLE/MoLA）、固定阈值 τ=1/N、动态阈值 τ(x)（AdaMoLE）
- 训练：frozen W_0 + load balancing loss（λ=1e-3），防止 expert 坍塌
- 推理：所有 expert 矩阵常驻 GPU 显存，router 每次 forward 动态门控

涉及论文标题：
- AdaMoLE Fine-Tuning Large Language Models with Adaptive Mixture of Low-Rank Adaptation Experts

---

## Curriculum Learning for MoE Training (课程学习用于 MoE 训练)

术语解释
Curriculum Learning 在 MoE 训练中特指基于 token 复杂度对训练数据重新排序的策略。由 Li et al. (EMNLP 2023) 在 Adaptive Gating 论文中首次应用于 MoE 训练场景，用于解决 adaptive gating 中不同 token 使用不同数量 expert 导致的 batch 内计算时间不均问题。

术语是什么？
在 adaptive gating 中，虽然多数 token 仅需 top-1 expert（计算量减半），但 Attention 层需要完整序列输入，导致训练 step 时间由 batch 中最慢的 top-2 token 决定。即使 80% token 已提前完成 MoE 计算，仍需等待剩余 20% top-2 token。课程学习通过将相似复杂度的训练样本分组，减少同 batch 内 top-2 token 比例的方差，缓解"快 token 等待慢 token"问题。

复杂度度量：对每个训练样本 d，定义复杂度向量 C_d = [r_0^d, r_1^d, ..., r_L^d]，其中 L 为 MoE 层数，r_i 为第 i 层中由 top-2 expert 处理的 token 占比。

从算法pipeline角度拆解术语。
```
# Curriculum Learning: Training Data Reordering
def reorder_training_data(all_samples, model, T):
    # Step 1: 计算每个样本的复杂度向量
    C_samples = []
    for sample in all_samples:
        C = []  # complexity vector
        for layer in model.moe_layers:
            gate_output = layer.gate(sample.embeddings)
            R = softmax(gate_output, dim=-1)
            top1_prob, top2_prob = R.topk(2, dim=-1).values[:, 0], R.topk(2, dim=-1).values[:, 1]
            prob_diff = top1_prob - top2_prob
            # r = 需 top-2 expert 的 token 比例
            r = (prob_diff <= T).float().mean().item()
            C.append(r)
        C_samples.append(C)

    # Step 2: 找到最简样本作为参考
    ref_idx = argmin([sum(C) for C in C_samples])
    ref_vec = C_samples[ref_idx]

    # Step 3: 按余弦相似度降序排列
    similarities = [cosine_sim(C, ref_vec) for C in C_samples]
    return [all_samples[i] for i in argsort(similarities, descending=True)]
```

术语一般如何实现？如何使用？
- 第一个 epoch 使用随机数据顺序让 model 产生初始 gate 决策
- 每 epoch 结束后重新计算复杂度向量并重排数据
- 排序依据：以最少 top-2 token 的样本为参考，余弦相似度降序排列
- 实验效果：去除 curriculum learning 后，训练时间平均膨胀 13.7%，推理性能最大下降 0.21 F1
- 适用于 adaptive gating 场景；固定 top-k MoE 训练不需要此策略
- 复杂度向量计算需完整前向传播一次，overhead 与一个 eval epoch 相当

涉及论文标题：
- Adaptive Gating in Mixture-of-Experts based Language Models

---

## EfficientScale

术语解释
EfficientScale 是 BAAI 在 AquilaMoE 中提出的两阶段 MoE 高效训练方法，通过 Scale-Up（小模型权重初始化大模型）和 Scale-Out（dense 模型转换为 MoE）两个阶段，用已有预训练权重引导大模型训练，避免从头训练的高昂计算成本。

术语是什么？
EfficientScale 由三个阶段的 pipeline 组成：
1. **Preparation Phase**: 从头训练小 dense 模型（或加载已有预训练权重），准备训练数据
2. **Scale-Up Phase**: 使用小模型的 weights 通过 AKI-Pro 初始化大 dense 模型，大幅降低初始 validation loss，然后连续预训练
3. **Scale-Out Phase**: 使用 Sparse Upcycling 将大 dense 模型转换为 MoE（每个 MLP 层复制为 8 个 expert + 随机初始化 router），再连续预训练 MoE

实际案例：AquilaDense-7B (3.6T tokens) → Scale-Up → AquilaDense-16B (1.2T tokens) → Scale-Out → AquilaMoE 8×16B (545B tokens)。相比从头训练 32B MoE (5345B tokens, 213.8 GPU-days)，EfficientScale 仅需 51.84 GPU-days，时间节省 4.12×，算力节省 3.35×。

从算法pipeline角度拆解术语：
```
# EfficientScale Pipeline
# Phase 1: Preparation
small_dense = train_from_scratch("M(32,4096)", tokens=3.6T)  # AquilaDense-7B

# Phase 2: Scale-Up
# 2a: AKI-Pro 初始化
large_dense = AKI_Pro_init(small_dense, target="M(40,5120)")
# 宽度: AKI 利用相邻层权重打破对称性
# 深度: Interpolation W'_l = floor(l * L_2 / L_1)
# GQA: 将每个 group 视为独立 MHA block 扩展

# 2b: 连续预训练
large_dense = train(large_dense, tokens=1.2T, lr=4.0e-4)  # AquilaDense-16B

# Phase 3: Scale-Out
moe_model = deepcopy(large_dense)
for layer in moe_model.layers:
    experts = [copy(layer.mlp) for _ in range(8)]  # 复制 dense MLP
    layer.moe = MoELayer(experts, router=Linear(hidden_dim, 8, N(0, 0.02)))
moe_model = train(moe_model, tokens=545B, lr=1.5e-4)  # AquilaMoE 8×16B
```

术语一般如何实现？如何使用？
- 前提：有高质量小模型 checkpoint；适用于从零开始训练成本极高的场景
- Scale-Up 阶段验证 loss 显著降低：AKI-Pro initialization loss 7.81 vs random init 12.22 at M(32,4096)
- Scale-Out 使用 Sparse Upcycling，experts 初始化为 dense MLP 复制，router 随机初始化
- 训练期间加 load balancing loss (λ=0.001) 和 max z-loss (λ=0.01) 防止崩溃
- 硬件：Preparation 阶段 480 × ~990 GFLOPS GPU，Scale-Up/Scale-Out 阶段 1024 × 240 GFLOPS accelerators
- 代码开源：https://github.com/FlagAI-Open/Aquila-MoE，模型权重：https://huggingface.co/BAAI/AquilaMoE

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---

## Function Preserving Initialization (FPI)

术语解释
FPI 是一种神经网络权重扩展方法，在扩展模型宽度（hidden dims）时使大模型的输出与输入的关系与小模型完全一致，从而保留小模型学到的知识。最初由 Net2Net (Chen et al., ICLR 2016) 提出，bert2BERT (ACL 2022) 将其扩展到 Transformer 语言模型的训练。

术语是什么？
FPI 的核心思想：扩展 MLP 层 y = U^T · W^T · x 的维度时，将新添加的神经元复制已有神经元的值，通过缩放保持输出等值。当输入 dim 从 2→3、intermediate dim 从 3→4、输出 dim 从 2→3 时：(1) Input Dim Expansion: 复制输入神经元，拆分权重；(2) Output Dim Expansion: 复制 hidden 神经元；(3) MLP Expansion: 复制输出神经元。

但 FPI 存在内在缺陷：复制操作导致对称权重（W'_1 = W'_2），梯度在训练中始终相同，有效参数量减半。具体来说，如果 y = w1x + w2x 且 w1=w2 初始化后，w1 和 w2 的梯度完全相同，永远无法分化。

从算法pipeline角度拆解术语：
```
# FPI 宽度扩展（MLP y = U^T · W^T · x）
# 源模型：d_in=2, d_inter=3, d_out=2
# 目标模型：d_in=3, d_inter=4, d_out=3

# Step 1: Input Dim Expansion
W_new = FPI_expand(W, d_in_new=3)
# w'_1 = w_1/2,  w'_2 = w_2/2,  w'_3 = w_1/2  # 复制并缩放

# Step 2: Output Dim Expansion (upsampling linear)
U_new = FPI_expand(U, d_inter_new=4)
# u'_1 = u_1/2,  u'_2 = u_2/2,  u'_3 = u_3/2,  u'_4 = u_1/2  # 复制

# 结果: 大模型 = 小模型在相同输入下有相同输出
# 问题: 对称权重 → 梯度永远相同 → 有效参数减半
```

术语一般如何实现？如何使用？
- 适用于 Transformer 架构的宽度扩展：Embedding layers、QKV projections、MLP 等
- 对 MHA (Multi-Head Attention) 将每个 attention head 视为一个"神经元"
- 不能扩展深度——bert2BERT 使用 StackBERT 的 stacking 方法扩展层数
- LN (Layer Normalization) 在非整数倍扩展时输出不完全相同，但论文指出这对最终 loss 影响不大
- 在 AquilaMoE 中 FPI 作为对比 baseline：FPI-Stacking validation loss 4.30 vs FPI-Interpolation 3.31

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---

## Advanced Knowledge Initialization (AKI)

术语解释
AKI 是 bert2BERT (ACL 2022) 提出的改进版权重初始化方法，通过利用相邻层的权重打破 FPI 的对称性问题，在扩展模型宽度时保持有效参数量不减少。

术语是什么？
AKI 的核心改进：FPI 扩展时从同层权重复制（如 W'_new = W_1/2），而 AKI 从相邻层的权重复制（如 W'_new = W_next_1）。因为相邻层学到不同的特征映射，这样打破对称性。以两层 MLP 为例：y1 = U1^T · W1^T · x, y2 = U2^T · W2^T · y1。FPI 扩展 W1 为 [w1/2; w2/2; w3/2; w1/2]，AKI 扩展 W1 为 [w1/2; w2/2; w3/2; w2_1]（w2_1 是第二层第一个权重），从而打破第一层内部的对称性。

从算法pipeline角度拆解术语：
```
# AKI 宽度扩展（利用相邻层权重）
# 源模型 Layer L: W_L ∈ R^{d_in×d_inter}
# 源模型 Layer L+1: W_{L+1} ∈ R^{d_in×d_inter}
# 扩展 W_L 到 d_inter_new

# FPI (同层复制):
W'_L = concat(W_L[:d_inter], W_L[:d_inter_new-d_inter]复制)
# → 对称性：复制的权重梯度永远相同

# AKI (相邻层构建):
W'_L = concat(W_L[:d_inter], W_{L+1}[:d_inter_new-d_inter])
# → 来自不同层的权重已有不同学习模式 → 打破对称性
```

术语一般如何实现？如何使用？
- 需同时有相邻两层的权重信息（L 和 L+1 层）
- 仅支持 MHA (Multi-Head Attention)，直接不支持 GQA (Group Query Attention)
- 在 AquilaMoE 中作为 baseline：AKI-Stacking validation loss 9.56 at M(32,4096)
- bert2BERT 实验：AKI 在训练一定步数后超过 FPI 的性能
- AquilaMoE 在此基础上改进得到 AKI-Pro

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---

## AKI-Pro

术语解释
AKI-Pro 是 AquilaMoE 对 bert2BERT AKI 方法的改进版初始化策略，包含两个关键改进：(1) 深度扩展使用 Interpolation 替代 Stacking；(2) 适配 Group Query Attention (GQA) 架构。

术语是什么？
AKI-Pro 的两个改进：
1. **Depth Growing via Interpolation**: 原 bert2BERT 使用 StackBERT 的 stacking（层堆叠）扩展深度，即 W'_l = W_{l mod L_1}。但 stacking 导致 L_1-1 层的输出空间与第 0 层输入空间不匹配，训练初期不稳定。AKI-Pro 改为 interpolation：W'_l = floor(l × L_2 / L_1)，相邻层在输出/输入空间上平滑过渡，连续训练更稳定。
2. **GQA Compatibility**: 原 AKI 仅支持 MHA。AKI-Pro 在源和目标模型 GQA group 数一致的约束下，将每个 group 视为独立 MHA block 进行 AKI 扩展，QKV projection 的扩展操作与 MHA 完全一致。

从算法pipeline角度拆解术语：
```
# 1. Depth Interpolation (L1=3, L2=6):
# 源层: W[0], W[1], W[2]
# Stacking: W'[0]=W[0], W'[1]=W[1], W'[2]=W[2], W'[3]=W[0], W'[4]=W[1], W'[5]=W[2]
#   → W'[2].out != W'[3].in: 输出空间不匹配
# Interpolation: 
#   W'[0]=W[0], W'[1]=W[1], W'[2]=W[1] (插值), W'[3]=W[2], W'[4]=W[2] (插值), W'[5]=W[2]
#   → 平滑过渡，训练更稳定

# 2. GQA 扩展（8 groups, heads_per_group=5）:
for group in range(8):  # 每组独立
    for head in range(5):  # 组内 head 扩展同 MHA
        W_qkv_large[group, head] = AKI_expand(W_qkv_small[group, head])
# 约束: num_groups_small == num_groups_large
```

术语一般如何实现？如何使用？
- 验证结果：AKI-Pro-Interpolation loss 7.81 vs AKI-Stacking loss 9.56 at M(32,4096)——interpolation 显著降低初始 loss
- 用于 AquilaMoE 的 Scale-Up 阶段：1.3B→7B (24→32 layers, 2048→4096 hidden), 7B→16B (32→40 layers, 4096→5120 hidden)
- GQA 适配假设源和目标模型 group 数相同；无法处理 group 数不同的情况
- Depth Interpolation 参考文献：Pan et al. (2024) "Preparing Lessons for Progressive Training on Language Models"

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies

---

## Group Query Attention (GQA)

术语解释
GQA (Group Query Attention) 是多查询注意力 (MQA) 和多头注意力 (MHA) 之间的折中方案，将 Query heads 分组共享 K、V heads，在减少 KV cache 大小和保持模型质量间取平衡。首次由 Ainslie et al. (2023) 在 "GQA: Training Generalized Multi-Query Transformer Models" 中提出。

术语是什么？
MHA 每 head 有独立 Q、K、V 投影 → KV cache = num_heads × d_head × 2。MQA 所有 head 共享 K、V → KV cache = 1 × d_head × 2。GQA 折中：将 heads 分成 groups，每组内共享 K、V。例如 40 heads + 8 KV groups = 5 Q heads per KV head。KV cache = num_groups × d_head × 2。

AquilaMoE 的模型配置演变：1.8B/7B 的 KV group=32 (实际是 MHA)，16B/MoE 的 KV group=8 (8 KV groups for 40 heads = GQA)。AKI-Pro 的 GQA 兼容性改造基于此设计：保持源和目标模型的 group 数一致，将每个 group 作为独立 MHA block 进行扩展。

从算法pipeline角度拆解术语：
```
# GQA Attention (40 heads, 8 groups, heads_per_group=5)
Q = x @ W_Q  # [seq, 40*d_head] — 每个 head 独立 Q
K = x @ W_K  # [seq, 8*d_head]  — 每组共享 K
V = x @ W_V  # [seq, 8*d_head]  — 每组共享 V

# Q: reshape → [seq, 8, 5, d_head]
# K, V: reshape → [seq, 8, 1, d_head] → expand → [seq, 8, 5, d_head]

for group in range(8):
    Q_group = Q[:, group, :, :]  # [seq, 5, d_head]
    K_group = K[:, group, :, :]  # [seq, 5, d_head] — 原为 [seq, 1, d_head] 扩展
    V_group = V[:, group, :, :]
    attn_group = Softmax(Q_group @ K_group^T / sqrt(d_head)) @ V_group

output = concat([attn_group for group in range(8)], dim=-1) @ W_O
```

术语一般如何实现？如何使用？
- 典型配置：Llama 2 70B (8 KV groups for 64 heads, i.e., MQA), Llama 3 70B (GQA, 8 groups)
- 相较于 MHA，GQA 的 KV cache 减少 num_heads/num_groups 倍
- GQA 在推理中降低 KV cache 显存占用，对长 context 场景尤为关键
- 训练中 GQA 略微降低收敛速度但仍能收敛到相似质量
- PyTorch 实现：SDPA 的 `torch.nn.functional.scaled_dot_product_attention` 支持 GQA 广播机制

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

**DS-MoE 中的 GQA 使用**：DS-MoE 在 MoA (Mixture of Attention Head) 中使用 GQA 机制：每个 MoA expert 计算 N_head 个 query vectors，但 K、V 由所有 expert 共享（通过 GQA 的 shared KV heads）。1B 模型使用 2 shared KV heads，3B/6B 模型使用 4 shared KV heads，与 N_head（2/4）数量相等，退化为 MHA 的 KV pattern。

---

## Heterogeneous Mixture-of-Experts (Heterogeneous MoE)

术语解释
Heterogeneous MoE 是一种打破传统 MoE 均匀设计的架构范式，允许在 Transformer 的不同层中使用不同数量的专家（variable expert count）、不同大小的专家（variable expert FFN size），以及不同的专家放置策略（variable expert placement），形成非均匀、异质的 MoE 架构。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
传统 MoE（如 Switch Transformer, GShard）采用 homogeneous 设计：所有层中 expert 数量相同（如每层 4 个或每隔一层 4 个），所有 expert 的 FFN 尺寸相同（如 intermediate size = 3072），专家放置位置采用 ad-hoc 规则（每隔一层、每四层、或最后几层）。

Heterogeneous MoE 打破这些约束，搜索空间包含：
1. **每层可变 expert 数量**：第 i 层可有 1 到 M 个 expert，不同层可以不同
2. **每 expert 可变 FFN 尺寸**：同一层内的每个 expert 可有不同的 intermediate FFN size（如 1024, 2048, 3072），不同层的 expert 也可不同
3. **可变 decoder 层数**：对于 encoder-decoder 架构，decoder 层数可以少于 encoder 层数
4. **可变非 expert 模块**：attention heads 数量、hidden size、QKV dimension 也可变

这种异构设计使模型可以实现 **adaptive computation**：不同 token 通过 routing 自然分配到不同大小的 expert——简单 token 走小 expert（节省 FLOPs），复杂 token 走大 expert（保证质量）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。
在一个 6-layer encoder-decoder Transformer 中，AutoMoE 搜索到的最优异构配置（WMT'14 En-De）：

```
# Encoder: 6 layers, 每层 expert 分布 [5, 1, 1, 1, 2, 1]
#   Layer 1: 5 experts, FFN sizes [3072, 3072, 3072, 2048, 3072]
#   Layer 2: 1 expert,  FFN size  [3072]
#   Layer 3: 1 expert,  FFN size  [3072]
#   Layer 4: 1 expert,  FFN size  [3072]
#   Layer 5: 2 experts, FFN sizes [3072, 2048]
#   Layer 6: 1 expert,  FFN size  [3072]

# Decoder: 4 layers, 每层 expert 分布 [1, 1, 1, 1]
#   每层: 1 expert, FFN size 3072

# Forward Pass (per token x):
for l in encoder_layers:
    logits = x @ W_router[l]              # W_router[l]: 每层不同维度
    # Layer 1 router: [d, 5]; Layer 2 router: [d, 1]; etc.
    routed_expert = argmax(logits)
    # Expert FFN: W_1 ∈ R^{ffn_size[e] × d}, W_2 ∈ R^{d × ffn_size[e]}
    # 不同 expert 的 ffn_size 可能不同
    h = ReLU(x @ W_1^T)
    out = h @ W_2^T                       # 输出维度始终为 d
```

异构设计的关键约束：所有 expert 的输入/输出维度保持相同（均为 d），仅中间 FFN 维度可变，因此 expert 输出可直接聚合。

术语一般如何实现？如何使用？
- 通过 NAS 自动搜索而非手动设计：在异构搜索空间中用演化算法找到最优配置
- AutoMoE 使用 Supernet（最大 MoE 配置）+ weight sharing 训练 + 演化搜索，在 latency constraint 下找到 Pareto 最优架构
- 搜索发现的一般规律：encoder 承担 71% 专家（中间层最多），decoder 首层 expert 最多、逐层递减
- 适用于 encoder-decoder Transformer（NMT），也可扩展到 decoder-only 架构
- 异构设计使 FLOPs 和 active parameters 大幅减少（AutoMoE: 4× FLOPs reduction vs dense Transformer）

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---

## Supernet Training (for MoE NAS)

术语解释
Supernet Training 是 Neural Architecture Search (NAS) 中的核心技术，通过构建一个包含搜索空间中所有可能子架构的"超级网络"（Supernet），并通过权重共享（weight sharing）联合训练所有子架构。AutoMoE 首次将 Supernet training 扩展到 MoE 架构的搜索空间。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Supernet = 搜索空间中的最大模型配置。在 AutoMoE 中，Supernet 是：
- 每层 M 个 expert（M = 最大 expert 数）
- 每个 expert 的 FFN 中间维度 = 最大可选值（3072）
- Decoder 层数 = 最大值（6）
- Attention heads = 最大值（8）
- Hidden/Embedding size = 最大值（640）

Supernet 训练过程：
1. 随机从搜索空间采样一个子架构（subnet）
2. 从 Supernet 中提取对应子架构的权重（通过 weight sharing 机制）
3. 用提取的权重前向传播 + 反向传播（仅更新被提取的部分）
4. 重复采样和训练直至训练 budget 耗尽

训练收敛后，任何子架构的性能可通过从 Supernet 提取权重并在验证集上评估来快速估计，无需单独训练。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Supernet 构建（搜索空间中的最大配置）
Supernet = Transformer(
    encoder_layers=6, decoder_layers=6,
    embedding_size=640,
    experts_per_layer=M,           # 每层 M 个 expert
    expert_ffn_size=3072,          # 最大 FFN 中间维度
    attention_heads=8
)

# Supernet 训练循环
for step in range(total_steps):
    # Step 1: 随机采样一个子架构
    subnet_config = {
        'dec_layers': random_choice([1,2,3,4,5,6]),
        'emb_size': random_choice([512, 640]),
        'attn_heads': random_choice([4, 8]),
        'enc_experts': [random_choice(1..M) for _ in range(6)],
        'dec_experts': [random_choice(1..M) for _ in subnet_config['dec_layers']],
        'enc_ffn_sizes': [[random_choice([1024,2048,3072]) for _ in range(e)] 
                          for e in subnet_config['enc_experts']],
        'dec_ffn_sizes': [[random_choice([1024,2048,3072]) for _ in range(e)] 
                          for e in subnet_config['dec_experts']]
    }
    
    # Step 2: 从 Supernet 提取子架构权重（front rows/columns）
    subnet_weights = extract_subnet_weights(Supernet, subnet_config)
    
    # Step 3: 前向 + 反向传播
    loss = forward(subnet_weights, batch)
    loss.backward()
    optimizer.step()  # 仅更新被提取的权重部分

# 训练后：评估任意子架构（无需额外训练）
def evaluate_subnet(config):
    weights = extract_subnet_weights(Supernet, config)
    val_loss = forward(weights, val_set)
    return val_loss
```

术语一般如何实现？如何使用？
- 基于 fairseq toolkit 实现（AutoMoE）
- Supernet 训练 40K steps，与最终模型训练相同步数（fair comparison）
- Weight sharing 使搜索效率极高：单个 Supernet 涵盖数千个子架构，搜索 + 训练仅需 224 GPU-hours（vs Evolved Transformer 的 2,192,000 GPU-hours）
- 局限：Supernet 中所有子架构共享权重可能导致子架构性能估计有偏；Sandwich sampling 和 inplace knowledge distillation 可改进 Supernet 训练质量
- 后续工作：Mixture-of-Supernets (MoS, ACL 2024 Findings) 用 MoE 增强 Supernet 表达力

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---

## Weight Sharing in MoE Supernet

术语解释
Weight Sharing in MoE Supernet 是 AutoMoE 提出的专用权重共享技术，使 Supernet（最大 MoE 配置）和其子架构之间可以通过"提取前 rows/columns"的方式共享权重。这是将 Supernet training 从 dense Transformer 扩展到稀疏 MoE 架构的关键技术。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
核心操作：给定 Supernet 中某层的 router 权重 W_super ∈ R^{M×d_max}（M 个 expert，d_max 最大 embedding 维度），子架构需要 W_sub ∈ R^{n×d}（n < M 个 expert，d < d_max embedding 维度）。提取方式为：
- W_sub = W_super[:n, :d]（取前 n 行、前 d 列）

对 Expert FFN 权重也是类似操作。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Supernet Router: W_super ∈ R^{M × d_max}
# Subnet Router (n experts, d embed): W_sub ∈ R^{n × d}
W_sub = W_super[:n, :d]  # front n rows, front d columns

# Supernet Expert FFN Layer 1: W_ffn1_super ∈ R^{d_ff_max × d_max}
# Subnet Expert FFN Layer 1 (d_ff=2048, d=512):
W_ffn1_sub = W_ffn1_super[:2048, :512]  # front 2048 rows, front 512 columns

# Supernet Expert FFN Layer 2: W_ffn2_super ∈ R^{d_max × d_ff_max}
# Subnet Expert FFN Layer 2 (d=512, d_ff=2048):
W_ffn2_sub = W_ffn2_super[:512, :2048]  # front 512 rows, front 2048 columns

# 异构 expert sizes 示例：
# Layer 有 4 个 expert，FFN sizes 分别为 [3072, 2048, 2048, 1024]
expert_0_W1 = W_ffn1_super[:3072, :512]  # 最大 expert
expert_1_W1 = W_ffn1_super[:2048, :512]  # 中等 expert
expert_2_W1 = W_ffn1_super[:2048, :512]  # 中等 expert（可共享相同位置）
expert_3_W1 = W_ffn1_super[:1024, :512]  # 最小 expert

# 关键约束：子架构中不存在的 expert 不提取权重
# 若子架构只有 2 个 expert，第 3、4 个 expert 的 Supernet 权重不参与该步训练
```

这种设计的核心假设：Supernet 的前几行/列权重经过了最充分的训练（因为几乎所有子架构都包含它们），因此提取的前行/前列权重质量最高。

术语一般如何实现？如何使用？
- 适用于任何可通过"维度截断"表达的搜索维度（expert count, FFN size, embedding size, hidden size）
- 在 training loop 中，每次采样后动态提取对应权重，训练后权重更新回 Supernet
- 这种机制自然地支持异构 expert 尺寸：每个 expert 提取不同数量的 rows/columns
- 无法处理非单调的搜索维度（如"是否使用某类 attention head"），需其他机制
- AutoMoE 代码：https://aka.ms/AutoMoE（基于 fairseq）

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---

## Evolutionary Search for Neural Architecture Search

术语解释
Evolutionary Search（演化搜索）是 NAS 中基于生物进化原理的搜索算法，通过种群初始化、选择、突变、交叉等操作迭代优化架构种群，以找到满足约束条件的最优架构。AutoMoE 采用演化搜索在异构 MoE 搜索空间中寻找 Pareto 最优架构。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
演化搜索的核心流程（Algorithm 1 in AutoMoE/HAT）：
1. **初始化**：随机采样 num_population 个架构
2. **评估**：对每个架构评估 validation loss（通过 Supernet 快速估计）和 latency（在目标设备上实测）
3. **选择**：选出 top num_parents 个架构作为父代
4. **突变**：对种群中随机选择的架构以 mutate_prob 概率随机修改一个搜索维度，生成 num_mutations 个子代（需满足 latency constraint）
5. **交叉**：随机选择两个架构交换部分维度，生成 num_crossover 个子代（需满足 latency constraint）
6. **新一代**：population = parents ∪ mutations ∪ crossovers
7. 重复 2-6 直到迭代结束，返回最优架构

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# AutoMoE 演化搜索（Algorithm 1）
popu = random_sample(search_space, n=125)  # 125 个随机架构

for iter in range(15):
    # Step 1: 评估所有架构
    for arch in popu:
        arch.val_loss = supernet.evaluate(arch)    # Supernet 快速估计
        arch.latency = measure_latency(arch, device=CPU, passes=100)  # partial gold
    
    # Step 2: 选择父代
    parents = top_k(popu, key=val_loss, k=25)
    
    # Step 3: Mutation (50 offsprings)
    mutations = []
    for _ in range(50):
        parent = random_choice(popu)
        child = mutate(parent, prob=0.3)  # 随机修改 1 个维度
        if child.latency <= latency_constraint:
            mutations.append(child)
    
    # Step 4: Crossover (50 offsprings)
    crossovers = []
    for _ in range(50):
        p1, p2 = random_choice(popu, 2)
        child = crossover(p1, p2)  # 交换部分维度
        if child.latency <= latency_constraint:
            crossovers.append(child)
    
    # Step 5: 新一代
    popu = parents + mutations + crossovers  # 125 个

# 返回最优
return top_1(popu, key=val_loss)
```

Mutation 操作实例：随机选择以下维度之一进行修改：
- 某层的 expert 数量：random(1, M)
- 某 expert 的 FFN 尺寸：random_choice([1024, 2048, 3072])
- Decoder 层数：random_choice([1-6])
- Attention heads：random_choice([4, 8])

术语一般如何实现？如何使用？
- AutoMoE 搜索参数：population=125, parents=25, mutation=50 (prob=0.3), crossover=50, 迭代=15
- 搜索耗时：~224 GPU-hours（含 Supernet 训练），远低于 Evolved Transformer 的 2,192,000 GPU-hours
- 搜索空间大小：M^L × NML 种可能配置（极大的搜索空间），演化搜索通过 guided exploration 而非穷举
- 关键优化：Partially gold latency（100 passes vs 300 passes 的 gold latency）加速搜索过程中的评估
- HAT (Wang et al., 2020) 首次将演化搜索用于 NLP NAS，AutoMoE 将其扩展到 MoE 空间

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---

## Hardware-Aware NAS

术语解释
Hardware-Aware NAS（硬件感知神经架构搜索）是在 NAS 搜索过程中将目标部署硬件的实际性能指标（如 latency、energy、memory）作为约束条件或优化目标，而非仅使用与硬件无关的代理指标（如 FLOPs、参数数量），确保搜索到的架构在目标硬件上真正高效。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
传统 NAS 通常使用 FLOPs 或参数量作为效率指标，但这些指标与实际硬件延迟存在 gap——不同操作在硬件上的执行时间可能差异巨大（如 memory-bound vs compute-bound 操作）。Hardware-Aware NAS 在搜索循环中直接测量每个候选架构在目标设备上的实际 latency。

AutoMoE 的实现：
- 每个候选架构在 Intel Xeon CPU 上实测推理 latency（前向传播 + beam search 翻译）
- 测量方法：batch translation 重复 100 次（partial gold）或 300 次（gold），去除 top/bottom 10% 异常值，取 truncated mean
- Latency constraint 作为硬约束：只有满足 latency ≤ threshold（如 600ms）的架构才会被纳入 population
- 也可使用 FLOPs constraint（但 latency constraint 提供更严格的硬件控制）

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Hardware-Aware NAS: Latency 测量与约束

# Latency 测量（在目标设备上实测）
def measure_latency(arch_config, device, num_passes=100):
    model = build_model(arch_config)
    times = []
    for _ in range(num_passes):
        t_start = time()
        # 模拟推理：source sentence → target sentence
        output = model.translate(source_sent, beam=5, max_len=30)
        t_end = time()
        times.append(t_end - t_start)
    # truncated mean: 去除 top/bottom 10%
    times_sorted = sorted(times)
    trim = int(num_passes * 0.1)
    return mean(times_sorted[trim:-trim])

# 在演化搜索中使用 latency constraint
def filter_by_latency(architectures, constraint_ms=600):
    valid = []
    for arch in architectures:
        if arch.latency <= constraint_ms:
            valid.append(arch)
    return valid

# Latency vs FLOPs constraint 对比（Table 6 in AutoMoE）
# Latency constraint ≤ 200ms (GPU): BLEU 41.23, FLOPs 2.9G, Latency 176ms
# FLOPs constraint ≤ 3G:          BLEU 41.09, FLOPs 3.0G, Latency 216ms
# 结论: Latency constraint → 更严格控制，FLOPs 和 latency 均更优
#        FLOPs constraint → 模型"用完"FLOPs budget 但 latency 偏高
```

术语一般如何实现？如何使用？
- 需要在目标硬件上进行实际测量（profiling），而非使用代理模型预测
- AutoMoE 在搜索中使用 partially gold latency (100 passes) 加速，最终报告用 gold latency (300 passes)
- HAT (Wang et al., ACL 2020) 首次将 hardware-aware NAS 应用于 NLP Transformer
- Look-up table (LUT) 是另一种常用方法：预先测量每个基础操作在目标硬件上的 latency，然后求和估计架构总 latency
- SCAN-Edge (ICLR 2024), PEL-NAS, MicroNAS 等后续工作在 edge device 上进一步推进了 hardware-aware NAS

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation

---

## Adaptive Computation (via Heterogeneous MoE)

术语解释
Adaptive Computation（自适应计算）指模型根据不同输入的复杂度动态分配不同量的计算资源：简单输入使用较少计算，复杂输入使用较多计算。在异构 MoE 中，自适应计算通过不同大小的 expert 自然实现——routing decisions 将 token 发送到不同大小的 expert，简单 token 走小 expert，复杂 token 走大 expert。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
与 traditional MoE 的 conditional computation（条件计算）的区别：
- **Conditional computation**：仅激活 expert 子集（稀疏激活），但所有被激活的 expert 大小相同 → 每个被处理的 token 获得相同计算量
- **Adaptive computation**：不同 token 不仅激活不同的 expert 子集，还被路由到不同大小的 expert → 不同 token 获得不同计算量

AutoMoE 通过异构 expert 设计（variable expert FFN size）实现 adaptive compute：
- 搜索空间允许 FFN intermediate size ∈ {1024, 2048, 3072}
- 同一层中可有不同大小的 expert（如 3 个 expert: sizes [3072, 2048, 1024]）
- Router 的 top-1 路由自然将不同 token 分配到不同大小的 expert
- 该设计等价于 early-exit 风格的 adaptive compute 但通过路由而非层级别实现

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# Adaptive Compute via Heterogeneous MoE
# Layer with 4 heterogeneous experts: sizes [3072, 2048, 2048, 1024]

def adaptive_moe_forward(x):
    # x: [batch, d_model], d_model = 512
    
    # Router
    logits = x @ W_router                # [batch, 4]
    expert_idx = argmax(logits, dim=-1)   # [batch], top-1 routing
    
    output = zeros_like(x)
    flops_per_token = []
    
    for token_i, e_idx in enumerate(expert_idx):
        e_ffn_size = expert_sizes[e_idx]  # 3072, 2048, 2048, or 1024
        
        # FFN: W1 ∈ R^{ffn_size × 512}, W2 ∈ R^{512 × ffn_size}
        h = ReLU(x[token_i] @ W1[e_idx].T)  # [ffn_size] — FLOPs 正比于 ffn_size
        out = h @ W2[e_idx].T               # [512]        — FLOPs 正比于 ffn_size
        output[token_i] = out
        
        flops = 2 * 512 * e_ffn_size  # 2 × d × ffn_size
        flops_per_token.append(flops)
    
    # 示例：简单 token "the" → expert 3 (ffn_size=1024) → FLOPs = 2×512×1024 = 1.05M
    #       复杂 token "photosynthesis" → expert 0 (ffn_size=3072) → FLOPs = 3.15M
    # 计算量差异: 3× reduction for simple tokens
    return output, flops_per_token
```

术语一般如何实现？如何使用？
- 异构 expert 尺寸在搜索阶段确定，推理时 router 动态分配
- 也可通过 identity/dummy expert（FFN size=0）实现极端自适应——某些 token 完全跳过 FFN 计算
- AutoMoE 的关键实证发现：70% expert layers 有 ≥2 experts，>75% 含可变 expert 尺寸
- 另一种 adaptive computation 实现：Ada-K（RL-based dynamic K）和 AdaMOE（null experts）
- 与 Mixture-of-Depths (MoD) 互补：MoD 在深度维度自适应（跳过层），异构 MoE 在宽度维度自适应（不同大小 expert）

**Duo-LLM 的自适应计算视角**：Duo-LLM 在每层 FFN 中并排放置一个 big FFN（inner_dim=10240）和一个 small FFN（inner_dim=640，16x smaller），两者共享 attention。训练时以 random routing（p=0.5）确保两个模块互换。推理时的自适应计算通过 router 在 per-token per-layer 粒度决定走 big 还是 small（甚至 skip），实现 token 级别的计算弹性。关键发现：Oracle 最优路由下仅使用 1 个 big layer 的 perplexity 低于所有 12 层都用 big module，证明了精细粒度自适应计算的巨大潜力。

涉及论文标题：
- AutoMoE: Heterogeneous Mixture-of-Experts with Adaptive Computation for Efficient Neural Machine Translation
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---

## Oracle Routing (Oracle-Guided Optimal Routing)

术语解释
Oracle Routing 是 Duo-LLM 框架中提出的最优路由发现方法：对每条输入序列，穷举所有可能的 per-layer 路由选择（big/small/skip），在给定计算预算约束下选择最小化 perplexity 的路由路径，作为自适应计算的理论性能上界。

术语是什么？
给定 L 层模型，当仅考虑 big vs small 二选一时有 2^L 条可能路径，加入 skip 选项时有 3^L 条。Oracle 对每条路径执行完整 forward pass 并计算 cross-entropy loss，选择 loss 最低的路径。对于计算预算约束的场景（如仅允许 4/12 层使用 big module），仅搜索满足预算的路径子集。

Oracle 的发现揭示了现有 MoE router 训练的次优性：
- 最优 big layer 数量为 6/12（非 12/12），因为 C(12,6)=924 候选路径最多，增大了选到优质路径的概率
- 仅用 1 个 big layer 的 oracle 路由 perplexity 低于所有层都用 big module
- 预算有限（4 big layers）时优先将 big 分配给后层；预算充足（8 big layers）时优先给前层
- 后层存在"容量阈值"——满足后才值得给前层增加计算

从算法pipeline角度拆解术语：
```
# Oracle: Exhaustive Enumeration with Budget Constraint
# Model: L layers, each with big FFN and small FFN
# Budget B: max number of big layers per token

def oracle_optimal_route(x, labels, budget_B):
    L = 12  # number of layers
    best_loss = inf
    best_route = None
    
    # Enumerate all routes with exactly B big layers
    for route in combinations(range(L), budget_B):
        # route: indices of layers using big module
        mask = [1 if l in route else 0 for l in range(L)]
        
        # Forward pass with this route
        h = x
        for l in range(L):
            h_attn = Attention_l(h)
            if mask[l] == 1:
                h_ffn = BigFFN_l(h_attn)    # inner_dim=10240
            else:
                h_ffn = SmallFFN_l(h_attn)  # inner_dim=640
            h = h + h_ffn
        
        loss = CrossEntropy(h @ W_vocab, labels)
        if loss < best_loss:
            best_loss = loss
            best_route = route
    
    return best_route, best_loss
```

术语一般如何实现？如何使用？
- Oracle 需要 ground truth labels（计算 CE loss），因此仅适用于 holdout 评估，无法在生产环境使用
- 需要穷举所有可能路径（2^L 或 3^L），计算复杂度为 O(2^L × L)，实际仅适用于小规模研究（L=12 时约 4096 条路径）
- Oracle 的核心价值是作为理论上界：衡量 learned router 与最优路由之间的差距
- Duo-LLM 发现 trained router perplexity 更接近 fixed pattern 而非 oracle，gap 巨大
- 未来方向：训练 surrogate model 近似 oracle 的决策，避免穷举和 ground truth 依赖

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---

## Relative Token Difficulty

术语解释
Relative Token Difficulty 是 Duo-LLM 提出的概念，定义为 token 从额外计算资源中获益的潜力——通过比较 small model loss 与 oracle/large model loss 的差距来衡量，而非仅看 token 的绝对 loss 值。

术语是什么？
传统观点认为高 loss token 是"困难"token，应路由到大模型处理。Duo-LLM 发现这不足够——某些 token 的 loss 虽高，但切换到 large model 或 oracle 后 loss 几乎不降（如"relationship"在"This can be a"之后，上下文可预测性差导致高 loss 但无法通过更多计算改善）。真正值得更多计算的 token 是那些 small model 与 oracle 之间存在显著 loss gap 的 token。

从算法pipeline角度拆解术语：
```
# Relative Token Difficulty: Loss Gap Method

# For each token in holdout set:
def compute_relative_difficulty(token, context):
    # 1. Small model loss
    loss_small = forward_loss(token, context, all_small=True)
    
    # 2. Big model loss (or oracle)
    loss_big = forward_loss(token, context, all_big=True)
    # or: loss_oracle = oracle_optimal_loss(token, context, budget)
    
    # 3. Absolute difficulty (traditional)
    abs_difficulty = loss_small  # — Higher = harder
    
    # 4. Relative difficulty (Duo-LLM's proposal)
    rel_difficulty = loss_small - loss_big  # Loss gap
    
    return abs_difficulty, rel_difficulty

# 代码示例 (Python):
# "names = [...]" 后紧跟 "iterator = filter(...)"
# loss_small("iterator") = 高 (~5.2)  ← 传统认为是困难 token
# loss_big("iterator")  = 高 (~5.0)  ← gap 仅 0.2, 不值得更多计算
# → 该 token 上下文不确定(新行开头), 计算无法帮助
# 
# "filter(is_big_name, names)" 中的 "len":
# loss_small("len") = 中 (~3.8)
# loss_oracle("len") = 低 (~2.1)  ← gap 1.7, 值得更多计算
# → 上下文有助于推断 "len", 额外计算显著降低 loss
```

术语一般如何实现？如何使用？
- 论文在 C4 validation set 和 Python code holdout set 上计算了所有 token 的 relative difficulty
- 可视化：蓝色=低 difficulty，红色=高 difficulty，发现代码中行首 token 的 small loss 高但 gap 小（不值得额外计算），而 len, filter, None 等关键字 gap 大（值得额外计算）
- 应用：可作为 router 训练的辅助信号——优先将 big 模块分配给 relative difficulty 高的 token，而非仅凭绝对 loss 值
- 该概念仍在早期研究阶段，论文建议进一步探索作为 router 训练的 surrogate metric

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---

## Budget Loss (Global Budget-Constrained Routing Loss)

术语解释
Budget Loss 是 Duo-LLM 提出的替代传统 per-layer Load Balancing Loss 的路由训练损失。它对所有层的 big 模块使用比例施加全局约束，允许 router 跨层灵活分配计算，而非强制每层内均匀使用。

术语是什么？
标准 MoE load balancing loss 在每层内独立计算：L_aux = α·N·Σ f_i·P_i（per-layer）。这强制每层内各 expert 使用率相似，但限制了跨层的灵活计算分配。Budget Loss 改为跨所有层的全局约束：

$$L_{budget} = \left(\frac{\sum_{i=1}^{L} P_{i,\text{big}}}{L} - \text{budget}\right)^2$$

其中 P_{i,big} 是第 i 层 router 分配给 big 模块的 softmax 概率，budget 是目标 big 模块使用比例（如 0.33 表示 33% 的层用 big）。总训练损失 L_total = L_CE + α·L_budget。

从算法pipeline角度拆解术语：
```
# Budget Loss Training (Duo-LLM Stage 3)
# L layers, each layer has W_{r,l} router weight

def budget_loss_training_step(x, labels, budget=0.33, alpha=0.01):
    total_ce_loss = 0
    P_big_all = []
    
    for l in range(L):
        # Router: learnable linear layer per layer
        logits_l = x @ W_r[l]                    # [batch, 2]: big/small
        P_l = softmax(logits_l / tau)            # [batch, 2]
        P_big = P_l[:, 0].mean()                  # scalar
        P_big_all.append(P_big)
        
        # Soft combination of big and small FFN outputs
        H_big = BigFFN_l(x)
        H_small = SmallFFN_l(x)
        # x_{out} = P_big * H_big + P_small * H_small  (per-token)
        # NOTE: 论文使用 soft combination 或 hard routing
        x = layer_forward(x, P_l, H_big, H_small)
    
    # Cross-entropy loss
    ce_loss = CrossEntropy(x @ W_vocab, labels)
    
    # Budget loss: global constraint across all layers
    avg_p_big = mean(P_big_all)
    budget_loss = (avg_p_big - budget) ** 2
    
    total_loss = ce_loss + alpha * budget_loss
    return total_loss
```

术语一般如何实现？如何使用？
- Budget 参数控制全局计算量：如 0.33 表示约 4/12 层使用 big 模块
- α 控制 budget 约束的强度，论文未给出具体值
- Soft routing 使用温度参数 τ：训练初期 τ 较小（soft routing），逐渐增大使 P 趋近 one-hot（hard routing）
- 对比 per-layer load balancing：Budget Loss 允许某些层 80% 用 big、另一些层 10% 用 big，只要全局均值满足 budget
- 优势：释放 layer-level routing flexibility，理论上可接近 oracle 的跨层不均匀分配模式

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---

## Duo FFN Module (Heterogeneous Per-Layer Expert Pair)

术语解释
Duo FFN Module 是 Duo-LLM 的核心架构设计：在 Transformer 的每层 FFN 中并排放置一个大 FFN 和一个小 FFN（16x 尺寸差异），两者共享同一个 attention 模块。这是实现 token 级别自适应计算的最小异构 MoE 单元。

术语是什么？
每层包含：(1) 共享的 Multi-Head Attention；(2) Big FFN（如 inner_dim=10240，SwiGLU）；(3) Small FFN（inner_dim=640，16x smaller，SwiGLU）。两个 FFN 的输入/输出维度相同（d_model=2560），仅中间维度不同。训练时以 p=0.5 随机路由 token 到 Big 或 Small FFN，使两者学习互换性。

从算法pipeline角度拆解术语：
```
# Duo FFN Module per layer
# d_model = 2560
# Big FFN:   inner_dim = 10240, params ≈ 2560×10240×3 ≈ 78.6M/layer
# Small FFN: inner_dim = 640,   params ≈ 2560×640×3   ≈ 4.9M/layer

def duo_ffn_layer_forward(x, route_decision):
    # x: [batch, d_model]
    
    # 1. Shared Attention (RMSNorm + MHA + residual)
    x_norm = RMSNorm(x)
    attn_out = MultiHeadAttention(x_norm)
    x = x + attn_out
    
    # 2. Duo FFN (RMSNorm + FFN choice + residual)
    x_norm = RMSNorm(x)
    if route_decision == 'big':
        # Big FFN: gated SwiGLU
        gate = x_norm @ W_gate_big    # [batch, 10240]
        up   = x_norm @ W_up_big      # [batch, 10240]
        act  = SiLU(gate) * up        # [batch, 10240]
        ffn_out = act @ W_down_big    # [batch, 2560]
        flops = 5 * 2560 * 10240      # ≈ 131M FLOPs
    else:
        # Small FFN: gated SwiGLU (16x smaller)
        gate = x_norm @ W_gate_small  # [batch, 640]
        up   = x_norm @ W_up_small    # [batch, 640]
        act  = SiLU(gate) * up        # [batch, 640]
        ffn_out = act @ W_down_small  # [batch, 2560]
        flops = 5 * 2560 * 640        # ≈ 8.2M FLOPs
    
    x = x + ffn_out
    return x, flops  # Big: 131M, Small: 8.2M (16x reduction)
```

术语一般如何实现？如何使用？
- Duo-LLM 使用 12 层该架构，总计 1.399B 参数（big FFN: 944M, small FFN: 59M, attention: 314M, embedding: 82M）
- 训练策略：random routing (p=0.5) 从零训练优于 freeze big + fine-tune small
- 训练数据：300B tokens (FineWeb, Wiki, Flan/Dolma, Python Stack-v2)
- 推理时 router 动态选择每层每个 token 的 FFN 路径
- 论文提及 Megablocks 的 block-sparse matmul 可高效执行此架构，但未提供具体实现
- 该架构是一种简化的异构 MoE：仅有 2 个 expert（big/small），但允许 fine-grained per-layer per-token routing

涉及论文标题：
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models

---

## Max Z-Loss

术语解释
Max Z-Loss 是 MoE 训练中用于辅助稳定的正则化损失项，用于抑制 Router logits 的过大幅度，防止因 router 输出的 softmax 分数过于极端导致的训练不稳定。首次由 Zoph et al. (2022) 在 ST-MoE 中提出。

术语是什么？
Z-loss 惩罚 router logits 的最大值：
$$\text{max_z_loss} = \lambda_z \cdot \max_j(\text{router_logits})^2$$

router logits 过大 → softmax 后概率分布过于集中 → 负载严重不均 + 梯度不稳定。Max z-loss 与 load balancing loss 配合使用：load balancing loss 促进均匀分配，z-loss 防止 logit 过大。

在 AquilaMoE 的训练中，load balancing loss 乘以 0.001，max z-loss 乘以 0.01，共同施加于 LM loss 上：
$$L_{total} = L_{LM} + 0.001 \cdot L_{aux} + 0.01 \cdot L_{z-loss}$$

从算法pipeline角度拆解术语：
```
# MoE Router forward with Z-Loss
router_logits = x @ W_router  # [batch, num_experts]
router_probs = Softmax(router_logits)

# Z-Loss: 惩罚 logits 最大值
z_loss = max(router_logits, dim=-1)^2  # max over experts
L_z = 0.01 * z_loss.mean()

# 总 loss
L = L_lm + 0.001 * L_aux + L_z
```

术语一般如何实现？如何使用？
- ST-MoE (Zoph et al., 2022) 首次提出，称为 "router z-loss"
- PaLM (Chowdhery et al., 2023) 中也使用
- 通常与 load balancing loss 配合使用
- 系数需谨慎调优：AquilaMoE 使用 0.01，过大会过度约束 router 学习
- 是防止大规模 MoE 训练崩溃的关键技术之一

涉及论文标题：
- AquilaMoE Efficient Training for MoE Models with Scale-Up and Scale-Out Strategies
- Aria An Open Multimodal Native Mixture-of-Experts Model（ARIA 在 fine-grained MoE 训练中使用 z-loss 与 group-level load balancing loss 配合稳定训练）
- Continual Pre-training of MoEs How robust is your router（PBTk CPT 实验使用 z-loss coeff=0.001 与 aux loss coeff=0.01 组合，均与 DeepSeekMoE 和 ST-MoE 保持一致。Z-loss 在 CPT 分布偏移期间继续发挥作用，防止 router logits 因新分布而爆炸）
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts（CuMo 使用 router z-loss α_z=0.01 与 load balancing loss α_b=0.1 组合，分别应用于 MLP connector、CLIP vision encoder 和 LLM 的每个 MoE 块，称为 "bzloss"）

---

## Group-Level Load Balancing Loss

术语解释
Group-Level Load Balancing Loss 是 ARIA 对标准 per-expert load balancing loss 的松弛变体：将 fine-grained MoE 的大量 expert 按固定大小分组，在组级别而非单个 expert 级别施加负载均衡约束，避免过强的负载均衡压制 expert specialization。

术语是什么？
标准 MoE load balancing loss 惩罚每个 expert 的负载不均：$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{E} f_i \cdot P_i$。当 expert 数量很大（如 ARIA 的 64 routed experts）时，per-expert 约束过于严格，会阻止 expert 发展出有意义的 specialization。ARIA 将 64 个 routed experts 分为 8 组（每组 8 experts），负载均衡 loss 在组级别计算：

$$L_{balance} = \sum_{g} \alpha \cdot (\text{fraction\_of\_tokens\_routed\_to\_group}_g)^2$$

这样组内 expert 的负载可以自然不均（允许 specialization），但跨组的负载保持相对均衡。

从算法pipeline角度拆解术语：
```
# Group-Level Load Balancing (ARIA style)
num_routed_experts = 64
group_size = 8
num_groups = 8

# Router forward
gate_scores = softmax(W_router @ x)  # [batch, 64]
topk_probs, topk_indices = topk(gate_scores, k=6)

# Group-level load balancing loss
L_balance = 0
for g in range(num_groups):
    group_mask = (topk_indices // group_size == g).any(dim=-1)
    fraction_g = group_mask.sum() / batch_size
    L_balance += fraction_g ** 2
L_balance *= alpha
```

术语一般如何实现？如何使用？
- ARIA 首次在 fine-grained multimodal MoE 中使用 group-level load balancing
- 适用于 expert 数量大（64+）的场景，per-expert balancing loss 过于严格时
- 与 z-loss 配合使用以稳定训练
- 组大小选择是 trade-off：太小组 = 接近 per-expert（约束过强），太大组 = 负载可能严重不均

涉及论文标题：
- Aria An Open Multimodal Native Mixture-of-Experts Model

---

## Branch-Train-Stitch (BTS)

术语解释
BTS 是由 Qizhen Zhang et al. (Meta, 2025) 提出的三阶段训练算法，将多个独立训练的领域专用 LLM Expert 合并为一个通用（generalist）模型。BTS 在保持 Expert 参数完全冻结的前提下，通过插入并训练轻量 Stitch Layer（264M 参数 vs 总 11B 参数）建立 Expert 之间的可学习连接，实现比 Expert Upcycling 和 Expert Merging baseline 更好的平均下游任务性能。

术语是什么？
BTS 算法包含三个阶段：
1. **Branch**：复制预训练的 Seed 模型 $m_0$ 为 $n$ 份副本 $m_1, ..., m_n$。
2. **Train**：每个 Expert $m_i$ 在领域专用数据 $\mathcal{D}_i$（如 Code、Math、Multilingual）上独立继续预训练。完成后冻结所有参数。
3. **Stitch**：在 Seed（Hub）和 Expert（Spoke）模型之间每 $\lfloor L/K \rfloor$ 层插入 Stitch Layer（共 $K$ 个，本文 $K=4$），仅训练 stitch 层参数 15B tokens。

BTS 的推理流程：输入 → Hub+Expert Layer 1-4 → Stitch 1 (Hub-into-Experts) → Layer 5-9 → Stitch 2 (Experts-into-Hub) → Layer 10-14 → Stitch 3 (Hub-into-Experts) → Layer 15-19 → Stitch 4 (Experts-into-Hub) → Hub 输出 → LM head。

从算法pipeline角度拆解术语。
BTS 论文使用的配置：2.7B Seed 模型（20 层，dim 3072，FFN dim 12288，24 heads，GQA=1 KV head，SwiGLU，RoPE θ=500000）。3 个 Expert（Code / Math / Multilingual），各继续训练 200B tokens。4 个 Stitch Layer（交替 Experts-into-Hub 和 Hub-into-Experts）。Stitch 训练 15B tokens（batch 2M，7000 steps，LR warmup 0→5e-6 cosine decay）。

关键性质：
- **模块性**：Expert 完全冻结 → Expert 可随时增删，仅需重训 stitch 层
- **Token 级路由**：每个 token 重新计算 gate → 支持 context-switching（同一 prompt 内不同任务自动切换 Expert）
- **Cross-capability**：交替双向 stitch 使 Expert 间信息流动 → 产生超越任何单个 Expert 的跨领域能力（如 Russian Math）

术语一般如何实现？如何使用？
- 实现方式：基于标准 PyTorch Transformer + 插入自定义 StitchLayer 模块（见 Stitch Layer 条目）
- 适用场景：需要将多个领域专用 LLM 合并为通用模型，同时保持模块性和可解释性
- 局限：总参数较多（11B = 4 × 2.7B），推理时需要前向传播所有 Expert（无稀疏激活）
- 论文未开源，来自 Meta FAIR 研究

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---

## Stitch Layer

术语解释
Stitch Layer 是 BTS 算法核心的轻量级可学习模块，插入在 Seed（Hub）模型和 Expert（Spoke）模型之间，负责加权合并多个模型的隐藏表示（hidden states）。每个 Stitch Layer 包含两种可学习参数：线性投影 $\{w_{\text{proj}_i} \in \mathbb{R}^{\dim \times \dim}\}_{i=1}^n$ 用于跨模型空间映射，以及线性门控 $w_{\text{gate}} \in \mathbb{R}^{\dim \times \dim \times (n+1)}$ 用于计算各模型表示的贡献权重。BTS 中交替使用两种 Stitch Layer 类型。

术语是什么？
两种 Stitch Layer：

1. **Experts-into-Hub Stitch Layer**（将 Expert 信息合并到 Hub）：
   - Gate 采用 softmax（dropout 后）归一化到 [0,1] 且 sum=1
   - Expert hidden states 投影到 Hub 空间：$\tilde{h}_i = w_{\text{proj}_i}(h_i)$
   - Hub 更新为各表示的加权和：$\tilde{h}_0 = h_0 * g_0 + \sum_{i=1}^n g_i * \tilde{h}_i$
   - Expert hidden states 不变

2. **Hub-into-Experts Stitch Layer**（将 Hub 信息注入 Expert）：
   - Gate 采用 sigmoid（dropout 后）输出 [0,1] 独立权重
   - Hub 保持不变：$\tilde{h}_0 = h_0$
   - 每个 Expert 混入 gated Hub 信息：$\tilde{h}_i = (1 - g_i) * h_i + g_i * w_{\text{proj}_i}(h_0)$

从算法pipeline角度拆解术语。
```
def StitchLayer(xs, merge_into_hub=True):
    x_hub = xs[0]
    x_experts = xs[1:]
    g = dropout(w_gate(x_hub))  # [bs, seq_len, dim, n+1]
    
    if merge_into_hub:  # Experts-into-Hub
        g = g.softmax(dim=-1)
        h_experts = [w_proj[i](x_experts[i]) for i in range(n)]
        h_hub = (g * stack([x_hub] + h_experts, dim=-1)).sum(-1)
    else:  # Hub-into-Experts
        g = g.sigmoid()
        h_experts = [(1-g[...,i+1])*x_experts[i] + g[...,i+1]*w_proj[i](x_hub) 
                     for i in range(n)]
        h_hub = x_hub
    
    return stack([h_hub] + h_experts, dim=-1)
```

Softmax vs Sigmoid gate 选择的设计理由：Experts-into-Hub 使用 softmax 使所有 Expert 的贡献归一化后竞争性地加权（总和为 1），适合 Hub 侧的信息聚合；Hub-into-Experts 使用 sigmoid 使每个 Expert 独立决定融入 Hub 信息的比例（各 Expert 门控值互不影响），适合 Expert 侧的信息吸收。

术语一般如何实现？如何使用？
- 实现：PyTorch nn.Module，w_gate 为 nn.Linear，w_proj 为 nn.ModuleList of nn.Linear
- 参数总量：$K \times (n \times \dim^2 + \dim \times (n+1))$。BTS 中 K=4, n=3, dim=3072 → 约 264M 可训练参数
- 训练仅针对 Stitch Layer，使用 next-token prediction loss from Hub output
- 消融结果：(1) 4 层 vs 10 层性能相近，1 层明显不足；(2) 交替架构对 cross-capability 至关重要；(3) Seed 作 Hub 明显优于 Expert 作 Hub

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---

## Hub-and-Spoke Model Architecture

术语解释
Hub-and-Spoke（中心-辐条）架构是 BTS 采用的多模型合并组织模式：一个中心 Hub 模型（通常为 Seed 模型 $m_0$）通过 Stitch Layer 与多个 Spoke Expert 模型（$m_1, ..., m_n$）双向连接，Expert 之间无直接连接。Hub 的最终输出作为 BTS 模型的输出。

术语是什么？
设计原理：Seed 模型在通用数据上预训练，其表示空间与所有 Expert（均从 Seed 初始化）更对齐，因此 Seed 作为 Hub 可更有效地整合来自不同 Expert 的信息。消融实验证实：用 Math Expert 作 Hub 时平均分数从 28.1 降至 26.2（MMLU 35.8→33.9, GSM8K 20.2→15.6, MATH 10.6→5.73）。

Hub-and-Spoke 的替代方案（论文讨论但未采用）包括：
- 全连接：所有 Expert 之间两两连接 → 参数过多
- 仅 Hub→Expert 单向：非交替架构 → Cross-capability 退化（Ru-MGSM 16.0→11.6）

从算法pipeline角度拆解术语。
BTS Hub-and-Spoke 的数据流模式（以 4 Stitch Layer、3 Expert 为例）：

```
Input → [Hub L1-L4 || Expert_i L1-L4]  → Stitch1(Hub→Experts)
     → [Hub L5-L9 || Expert_i L5-L9]    → Stitch2(Experts→Hub)
     → [Hub L10-L14 || Expert_i L10-L14] → Stitch3(Hub→Experts)
     → [Hub L15-L19 || Expert_i L15-L19] → Stitch4(Experts→Hub)
     → Hub.L20 Output → LM Head → token
```

设计选择：最后一个 Stitch Layer 始终为 Experts-into-Hub 类型，确保最终输出来自 Hub（已融合所有 Expert 信息）。

术语一般如何实现？如何使用？
- 适用于 n 个 Expert + 1 个 Seed 模型的全冻结合并场景
- 推理时所有 Expert 均需前向传播（与 MoE 的稀疏激活不同）
- 可推广至其他 Hub 选择（论文验证了 Seed 作为 Hub 最优）
- 适用条件：要求 Hub 模型与 Expert 共享相同架构（层数、维度）

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---

## Cross-Capability

术语解释
Cross-capability（交叉能力）是 BTS 论文引入的评估概念，指合并后的模型在多个 Expert 专长领域交集处展现的能力——超越任何单个 Expert 在交集任务上的表现。例如：Russian Expert（专精俄语）+ Math Expert（专精数学）→ 合并模型在 Russian Math（俄语数学题）上表现优于两者。

术语是什么？
论文通过实验定义和验证 Cross-capability：
- **设置**：仅合并 Russian Expert + Math Expert（移除 Code Expert 以避免语言污染），训练 2B tokens Russian Math 数据
- **评估**：Russian MGSM（GSM8K 俄语翻译子集，8-shot）
- **发现**：(1) 无 in-domain 训练数据时，所有合并方法无法产生 cross-capability（Russian MGSM 表现不超 Seed）; (2) 加入少量 in-domain 数据后，BTS 在 Expert Merging 方法中表现最佳（Russian MGSM 16.0 vs BAM Adapters 15.6 vs BTM 9.60）; (3) Expert Upcycling 方法（BAM 18.4, BTX 17.6）因更大训练容量略优于 BTS
- **关键消融**：交替 stitch 架构对 cross-capability 至关重要——全 Experts-into-Hub 架构使 Russian MGSM 从 16.0 降至 11.6

从算法pipeline角度拆解术语。
Cross-capability 的机制解释：Hub-into-Experts stitch layer 允许 Math Expert 的表示受 Russian Expert 信息影响（通过 Hub 作为中介），使得 Math Expert 在处理俄语数学题时能利用 Hub 中融合的俄语理解能力；Experts-into-Hub stitch layer 反之将融合结果回流到 Hub 输出。

术语一般如何实现？如何使用？
- 评估方法：选择两个正交 Expert 领域，构建交集任务的 benchmark（需 in-domain 训练和评估数据）
- 关键条件：需要少量 in-domain 交叉训练数据（无此数据则合并模型与 Seed 无显著差异）
- 适用场景：验证模型合并方法的表达能力和泛化边界

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---

## Brain Foundation Model (脑基础模型)

术语解释
Brain Foundation Model 是将深度学习中"基础模型"（Foundation Model）范式迁移到脑神经影像领域的模型类别：在大规模 fMRI 数据上通过自监督预训练学习通用的脑活动特征表示，再微调到下游任务（疾病诊断、行为识别、年龄预测等）。

术语是什么？
类似于 LLM 在通用文本上预训练后适配多种 NLP 任务，Brain Foundation Model 在多个数据集的 resting-state 和 tasking-state fMRI 数据上预训练，学习 BOLD 信号或 FC 矩阵的通用表示。核心流程：
1. 预训练：在大规模 fMRI 数据（UKB ~50k scans, HCP ~15k scans, OpenNeuro）上，通过自监督目标（mask reconstruction, JEPA prediction, contrastive learning）学习脑区活动模式的 latent representation
2. 微调：在目标下游数据集（如 ABIDE Autism, ADNI Alzheimer's）上，用少量标注数据微调预训练 encoder + 分类头
3. 推理：输入新 subject 的 fMRI → 预训练 encoder 提取特征 → 分类器输出诊断/预测

代表性模型：BrainLM (2023, MAE on BOLD, 650M params)、BrainJEPA (2024, JEPA masking, 307M)、BrainMass (2024, FC reconstruction + pseudo-FC, 34M)、BrainMoE (2025, MoE with 12 experts, 709M)。

从算法pipeline角度拆解术语。
```
# 通用 Brain Foundation Model Pipeline
# Phase 1: Pre-training (self-supervised)
for each fMRI_scan in large_scale_dataset:  # 来自多个数据源
    # Step 1: 预处理
    T1w_MRI → FSL segmentation → atlas parcellation (AAL/Schaefer)
    BOLD_4D → regional_mean_timeseries → FC_matrix (Pearson corr)
    
    # Step 2: 自监督预训练 (以 BrainMass 为例)
    FC = compute_FC(BOLD)           # [M, M] functional connectivity
    FC_masked = random_mask(FC)     # mask 部分 brain region pairs
    Z = Encoder(FC_masked)          # bottleneck → latent representation
    FC_hat = Decoder(Z)
    L = ||FC_hat - FC||²            # reconstruction loss

# Phase 2: Fine-tuning (supervised)
for each subject in downstream_dataset:
    Z = frozen_encoder.extract(FC)  # 提取预训练特征
    y_pred = classifier(Z)          # 分类/回归头 (SVM, MLP, Transformer)
    L = CrossEntropy(y_pred, y_true)
```

术语一般如何实现？如何使用？
- 框架：基于 PyTorch，使用 FSL (FMRIB Software Library) 做 MRI 预处理
- 脑图谱：常用 AAL (116 ROIs)、Schaefer (400 ROIs)、C200 等，将全脑分为 ~100-400 个区域
- 输入模态：BOLD timeseries → FC 矩阵、或直接使用 BOLD latent features
- 预训练数据：UK Biobank (UKB)、Human Connectome Project (HCP)、OpenNeuro 等公开数据集
- 下游应用：疾病早期诊断（Alzheimer's, Parkinson's, Autism, Schizophrenia）、性别/年龄预测、行为识别、fMRI-EEG 多模态融合

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---

## Cognition Adapter (认知适配器)

术语解释
Cognition Adapter 是 BrainMoE 中用于将多个脑认知专家（brain expert）的 cognition embeddings 适应到下游任务的 Transformer Decoder 模块，通过 multi-head self-attention 混合 expert embeddings 和 task queries，再通过 cross-attention 将原始 FC 矩阵信息注入，最终输出下游分类预测。

术语是什么？
Cognition Adapter 是一个专门设计的 Transformer Decoder 架构，输入为两部分拼接的 token vectors：
- Z̄_{:k} = Z ⊙ P：top-k 个 cognition embeddings（expert 输出 × Router 权重），维度 [k, C_hid]
- Z̄_{k:(k+P)}：随机初始化的 task query embeddings，维度 [P, C_hid]（P=下游分类数）

每层 adapter 执行两个 attention 操作：
1. Multi-head Self-Attention (MHSA)：Q=Z̄α_h, K=Z̄β_h, V=Z̄γ_h，在 expert embeddings 和 task queries 之间混合信息
2. Multi-head Cross-Attention：Q=Iα̂_h (FC matrix), K=Z̄β̂_h, V=Iγ̂_h，将原始脑连接组信息注入到 task representations
3. FFN（MLP）

最后通过 Linear(Z̄[k:]) 仅取 task query 部分输出分类 logits。

从算法pipeline角度拆解术语。
```
# Cognition Adapter Forward
def cognition_adapter(Z_experts, FC_matrix, P_classes, k_top):
    # Z_experts: [N, C_hid] - N个expert的cognition embeddings
    # FC_matrix: [M, M] - functional connectivity矩阵
    
    # Step 1: Router选择top-k experts
    P = Softmax(Linear_router(Z_experts))       # [N]
    topk_idx = TopK(P, k_top)
    Z_topk = Z_experts[topk_idx] * P[topk_idx]  # [k, C_hid]
    
    # Step 2: 拼接task query embeddings
    Q_task = Parameter(randn(P_classes, C_hid))  # [P, C_hid]
    Z_bar = concat([Z_topk, Q_task])             # [k+P, C_hid]
    
    # Step 3: Adapter layers
    for layer in range(num_layers):
        # Self-Attention: expert-task混合
        Z_bar = Z_bar + MHSA(Z_bar, Z_bar, Z_bar)  # Eq(2)
        
        # Cross-Attention: FC矩阵→task表示
        Q_cross = FC_matrix @ alpha_hat           # [M, C_hid]
        K_cross = Z_bar @ beta_hat                # [k+P, C_hid]
        V_cross = FC_matrix @ gamma_hat           # [M, C_hid]
        Z_bar = Z_bar + CrossAttn(Q_cross, K_cross, V_cross)  # Eq(3)
        
        # FFN
        Z_bar = Z_bar + MLP(Z_bar)
    
    # Step 4: 输出分类
    y_pred = Linear(Z_bar[k_top:])               # [P] logits
    return y_pred
```

术语一般如何实现？如何使用？
- 架构选择：BrainMoE 选择 Transformer Decoder 而非简单 MLP，因为 MLP adapter 在高维 latent space（C_hid=2048）下不可扩展
- Cross-attention 设计：将 FC matrix [M,M] 作为 Q 和 V 的 source，使 adapter 能直接访问原始脑连接组信息，避免信息瓶颈
- Router+Adapter 联合训练：expert 参数冻结，仅训练 Router 和 Adapter 参数
- 输入灵活性：不依赖 expert 的内部架构，支持 FC-based 和 BOLD-based expert 混合使用（All-in-one BrainMoE 36 experts）

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---

## Functional Connectivity (FC, 功能连接)

术语解释
Functional Connectivity (FC) 是 fMRI 数据分析的核心表示，通过计算不同脑区 BOLD 时间序列之间的 Pearson 相关系数，量化脑区之间的功能同步程度。FC 矩阵是脑网络分析的基础数据结构。

术语是什么？
给定 fMRI 数据经脑图谱（如 AAL 116 ROIs）分区后，每个脑区提取一条 BOLD 时间序列 x_i(t)。两个脑区 i 和 j 之间的 FC 定义为：

$$FC_{ij} = \frac{\sum_t (x_i(t) - \bar{x}_i)(x_j(t) - \bar{x}_j)}{\sqrt{\sum_t (x_i(t) - \bar{x}_i)^2 \sum_t (x_j(t) - \bar{x}_j)^2}}$$

结果 FC ∈ R^{M×M}，M=脑区数（通常 116~400），是对称矩阵，值域 [-1, 1]。FC 矩阵反映了全脑功能网络拓扑结构——高 FC 表示两个脑区在时间上高度同步，可能参与相同的认知过程或属于同一功能网络（如 Default Mode Network, DMN）。

从算法pipeline角度拆解术语。
```
# FC 计算 Pipeline
# 输入: 4D fMRI (x, y, z, t) + T1w MRI
# 输出: FC matrix [M, M]

# Step 1: 组织分割 (T1w MRI)
tissue_seg = FSL_FAST(T1w_MRI)    # 白质、灰质、脑脊液

# Step 2: 脑图谱配准与分区
atlas = AAL116  # 或 Schaefer400、C200
for region_r in atlas.regions:    # r = 1..M
    # Step 3: 提取区域BOLD时间序列
    bold_r[t] = mean(fMRI_4D[x,y,z,t] for (x,y,z) in region_r)
    # bold_r: [T] 时间序列, T = 扫描时间点数

# Step 4: 计算Pearson相关 → FC矩阵
for i in range(M):
    for j in range(M):
        FC[i,j] = pearson_corr(bold_i, bold_j)
```

术语一般如何实现？如何使用？
- 预处理工具：FSL (FMRIB Software Library)，用于组织分割、运动校正、空间标准化
- 脑图谱选择：AAL (116 ROIs) 基于解剖标志；Schaefer (400 ROIs) 基于功能边界；C200 基于连接组
- FC 应用：作为 BrainMass 等 brain foundation model 的输入进行自监督预训练（mask reconstruction）；作为 BrainMoE Cognition Adapter 中 cross-attention 的 Key-Value source
- BOLD vs FC：BOLD 保留时间维度信息（timeseries），FC 通过相关性压缩为静态网络。小数据集上 FC 常优于 BOLD 作为输入特征（更高的 SNR）

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---

## BOLD Signal (血氧水平依赖信号)

术语解释
BOLD (Blood-Oxygen-Level Dependent) 信号是 fMRI 测量的核心生理信号，反映神经元活动引起的局部血氧浓度变化。当脑区激活时，局部血流增加带来的氧合血红蛋白变化导致 MR 信号强度改变，形成 BOLD 对比度。

术语是什么？
BOLD 信号是 fMRI 的基础测量量：(1) 神经元活动→局部代谢需求增加→血管扩张→脑血流增加→氧合血红蛋白（diamagnetic）相对脱氧血红蛋白（paramagnetic）比例上升→MR T2* 信号增强→fMRI 记录的信号变化；(2) BOLD 响应具有 hemodynamic delay（约 4-6 秒延迟，~12 秒恢复基线）；(3) 4D fMRI 数据：3D 空间体积 × 时间维度 = [x, y, z, t]，每个 voxel 的时间序列即为 BOLD 信号。

从算法pipeline角度拆解术语。
```
# BOLD 信号的使用
# 输入: raw 4D fMRI [x, y, z, time_points]
# 预处理步骤:
1. motion_correction(fMRI)         # 头动校正
2. slice_timing_correction(fMRI)   # 层时间校正
3. spatial_normalize(fMRI, MNI)   # 空间标准化到MNI模板
4. spatial_smooth(fMRI, FWHM=6mm) # 高斯平滑, 提高SNR
5. bandpass_filter(fMRI, 0.01-0.1Hz) # 保留神经活动频段

# 两种使用路径:
# Path A: BOLD → FC
for region in atlas:
    bold_ts[region] = mean(BOLD[voxels_in_region], axis=space)
FC = corr(bold_ts)  # [M, M]

# Path B: 直接使用 BOLD latent features
# (如 BrainJEPA, BrainLM)
Z = ViT_encoder(BOLD_4D_masked)  # spatiotemporal encoding
```

术语一般如何实现？如何使用？
- BOLD 作为 brain foundation model 的输入：BrainLM 和 BrainJEPA 直接对 BOLD timeseries 做 spatiotemporal masking 和 reconstruction，保留时间动态信息
- BOLD vs FC 权衡：BOLD 含时间信息但维度高（噪声大），FC 压缩为静态矩阵但丢失时序动态
- 预处理管线差异：不同研究使用不同预处理 pipeline（FSL, SPM, DPABI），导致 BOLD/FC 特征分布差异，影响模型跨数据集泛化——这也是 BrainMoE 需要鲁棒设计的原因之一

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---

## Cognition Embedding (认知嵌入)

术语解释
Cognition Embedding 是 BrainMoE 中各 brain expert 输出的特征表示 Z ∈ R^{C_hid}，代表在特定认知状态下脑活动的 latent representation。不同 expert 产出的 cognition embeddings 在同一 latent space 中表示不同认知视角下的脑活动模式。

术语是什么？
Cognition Embedding 是 brain expert f_i(·) 的前向输出：Z_i = f_i(X)，X 为输入 fMRI 数据（FC 或 BOLD），Z_i ∈ R^{C_hid} 为压缩后的特征向量。关键性质：
1. 认知状态特异性：Rest expert 产出的 Z_rest 编码 resting-state 下的脑网络模式；Emotion expert 产出的 Z_emotion 编码情绪加工时的脑激活模式
2. 架构无关性：与 expert 内部架构（BrainMass/brainJEPA/classifier）和输入类型（FC/BOLD）无关，只要求输出维度统一
3. 正交性：不同认知状态的 embeddings 相关性低（绝对 Pearson correlation < 0.5），表明 expert 间互补而非冗余
4. 下游任务适配性：某些认知状态的 expert 对特定疾病有天然的诊断优势（如 Language expert 对 Alzheimer's、Working Memory expert 对 Parkinson's）

从算法pipeline角度拆解术语。
```
# Cognition Embedding 的生成和使用
# 预训练阶段
for cognitive_state in [resting, emotion, WM, language, ...]:
    data_cog = fMRI_data[cognitive_state == state_label]
    expert_i = train(data_cog, objective=recon/classif)
    # expert_i 输出: Z_i = f_i(X), Z_i ∈ R^{2048}

# 微调阶段
X_downstream = fMRI_subject  # 来自任意下游数据集
Z_all = []                    # 收集所有expert的cognition embeddings
for i in range(N_experts):   # N=12
    Z_i = expert_i(X_downstream)  # frozen expert forward
    Z_all.append(Z_i)             # [12, 2048]

# Router 混合
P = Softmax(Linear(concat(Z_all)))   # [12] expert weights
Z_weighted = Z_all * P               # [12, 2048] weighted cognitions
# 送入 Cognition Adapter 继续处理
```

术语一般如何实现？如何使用？
- 相当于 MoE 中 expert 的"中间产品"而非最终输出——不直接用于分类，而是输入 Router 获得权重后进一步由 Adapter 加工
- 不同于 LLM MoE 中 expert 输出直接加权求和产生 token hidden state——BrainMoE 的 cognition embedding 是 sample-level 表示，需要 cross-attention 与 FC 信息融合后才产生分类结果
- 维度 C_hid=2048，对应 BrainMass 的 bottleneck 维度

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---

## Cross-Attention for Multi-Modal Fusion (交叉注意力多模态融合)

术语解释
Cross-Attention 是 Transformer 架构中的一种注意力机制变体，其中 Query 来自一个模态/序列，Key 和 Value 来自另一个模态/序列，实现跨模态信息交互。在 BrainMoE 中，Cross-Attention 用于将原始 FC 矩阵（结构信息）融合到 cognition embeddings（语义信息）中。

术语是什么？
标准 Self-Attention: Q, K, V 来自同一输入。Cross-Attention: Q 来自一个源，K, V 来自另一个源。

在 BrainMoE Cognition Adapter 中：
- Q = I · α̂_h：来自 FC 矩阵 I∈R^{M×M}（脑连接组结构），通过线性投影到 C_hid 维
- K = Z̄ · β̂_h：来自 cognition embeddings + task queries 的混合表示
- V = I · γ̂_h：同样来自 FC 矩阵

Cross-attention 使每个 task query 能根据 brain connectivity 模式自适应地 attend 到相关的 cognition embedding 部分。本质上将"脑区之间如何连接"（FC 矩阵）作为上下文来解读"脑在特定认知状态下如何活动"（cognition embedding）。

从算法pipeline角度拆解术语。
```
# BrainMoE Cross-Attention 的具体操作
# 输入: Z_bar [(k+P), C_hid], FC [M, M]
# 输出: updated Z_bar

for head h in range(num_heads):
    # Query 来自FC矩阵（brain structure）
    Q_h = FC @ alpha_hat_h       # [M, C_hid]
    
    # Key 来自混合表示（cognition semantics）
    K_h = Z_bar @ beta_hat_h     # [(k+P), C_hid]
    
    # Value 来自FC矩阵（brain structure）
    V_h = FC @ gamma_hat_h       # [M, C_hid]
    
    # Scaled Dot-Product Attention
    A_h = Softmax(Q_h @ K_h^T / sqrt(C_hid))  # [M, (k+P)]
    O_h = A_h @ V_h              # [M, C_hid]

# 合并多头 + 残差
O = concat([O_1, ..., O_H]) @ W_out
Z_bar = Z_bar + O[:k+P]         # 取前(k+P)行作为残差
```

核心张量维度：
- α̂_h, γ̂_h ∈ R^{M×C_hid} (M=116 regions)
- β̂_h ∈ R^{C_hid×C_hid}
- Attention: M brain regions attend to (k+P) token queries

术语一般如何实现？如何使用？
- 通用使用场景：多模态融合（文本-图像：Q=text, K=V=image；音频-文本等）、编码器-解码器 attention（Q=decoder, K=V=encoder）
- BrainMoE 的特殊用法：以脑结构信息（FC）为 Q 和 V，以认知语义信息（cognition embeddings）为 K，实现"structure-aware"的认知融合
- 不同于标准 cross-attention 中 Q 来自 decoder（目标模态）——这里 Q 来自 FC 矩阵，起到"通过脑连接组结构的镜头来理解认知状态"的作用

涉及论文标题：
- BrainMoE Cognition Joint Embedding via Mixture-of-Expert Towards Robust Brain Foundation Model

---

## Regularized Evolutionary Search for Transformer Architecture (正则化演化架构搜索)

术语解释
Regularized Evolutionary Search 是 Brainformers (Zhou et al., ICML 2023) 提出的自动化 Transformer 架构搜索方法，在固定训练时间预算和 inference step time 约束下，通过演化算法联合搜索层类型序列、层宽度、gating 机制、routing 策略和激活函数，发现非均匀的 MoE Transformer block 架构。

术语是什么？
与标准 NAS（Neural Architecture Search）不同，Regularized Evolutionary Search 的关键特征：
1. **Block-wise 搜索空间**：不搜索单个算子，而是搜索整个 block 的 sub-layer 序列（F_attn, F_moe, F_ffn）以及各层的维度配置
2. **Training time constrained**：每个搜索 trial 固定 wall clock time（而非固定 training steps），更快收敛的模型自动获得更多 training steps
3. **Early stopping regularization**：在 25% max training steps 时检查，违反 perplexity 或 inference time 约束的模型提前淘汰（R=-1）
4. **Proxy training + scale-up evaluation**：小规模 proxy model（100M, block 堆叠 3 次）搜索 block → ScaleModelDim (2x/4x) + StackNTimes 扩展到目标规模（1B/8B）

从算法pipeline角度拆解术语。
```
# Regularized Evolutionary Search Algorithm
population_size = p

for generation t = 1 to T0:
    for each block_arch B^(i) in SamplePopulation(B, p):
        # Proxy model: stack block 3 times → ~12 sub-layers, 100M scale
        G^(i) = StackThreeTimes(B^(i))
        
        if EarlyStopping(G^(i)):     # at 25% T_max
            R^(i) = -1               # prune: poor perplexity or slow step time
        else:
            A_i, T_i = Train(G^(i), fixed_wall_clock_time)
            R^(i) = f(A_i, T_i)      # reward = accuracy + step time

# Top-k architectures evaluated at target scales
G_topk = TopK({G^(i), R^(i)})
for G^(i) in G_topk:
    G^(i) = ScaleModelDim(G^(i))    # 2x/4x dim scaling
    G^(i) = StackNTimes(G^(i))      # N = target_activated / activated_per_block
    A_i, T_i = Train(G^(i))         # full-scale evaluation
```

术语一般如何实现？如何使用？
- 基于 GLaM 框架（Google 内部），512 TPU V4 运行 1 周完成搜索
- 搜索在 500 trials 内即能发现显著优于 GLaM baseline 的架构
- 相比 Evolved Transformer（2,192,000 GPU-hours），计算开销大幅降低
- 可用于优化 MoE 模型（gating 策略、expert 数量、routing 粒度、层类型分布）
- 局限：大模型规模搜索仍昂贵；block structure 在不同尺度上的可迁移性未被充分验证

涉及论文标题：
- Brainformers Trading Simplicity for Efficiency

---

## Non-uniform Transformer Block Architecture (非均匀 Transformer 块架构)

术语解释
Non-uniform Transformer Block Architecture 是打破标准 Transformer 中 attention 和 FFN 严格交替排列的架构设计范式，允许 block 内以任意顺序排列多种层类型（attention、dense FFN、sparsely gated FFN/MoE），通过自动搜索或手动设计优化层序列和维度配置。

术语是什么？
标准 Transformer (Vaswani et al., 2017): 每层 = Attention + FFN, uniform 重复。GLaM 引入稀疏性但保持交替：每层 = Attention+FFN 或 Attention+MoE, 交替重复。Non-uniform 架构完全打破此约束，一个 block 可包含多个 sub-layer 且顺序不受限：

$$\mathcal{N} = \mathcal{F}_k \odot ... \odot \mathcal{F}_2 \odot \mathcal{F}_1(X_1) = \bigcup_{j=1...k} \mathcal{F}_j(X_1)$$

其中 $\mathcal{F}_i \in \{\mathcal{F}_{\text{attn}}, \mathcal{F}_{\text{moe}}, \mathcal{F}_{\text{ffn}}\}$。

从算法pipeline角度拆解术语。
以 Brainformer Block 1 的 forward 为例（8 sub-layers, 搜索得到的最优非均匀架构）：

```
# Brainformer Block 1: 8 sub-layers 非均匀组合
# 架构: Attn → MoE(EC) → FFN → Attn → MoE(EC) → FFN → MoE(EC) → FFN
# 特征: Attention 仅 2/8 (vs uniform 4/8), MoE 3/8, FFN 3/8

def brainformer_block_forward(X_1):  # [B, L, d=1024]
    X = X_1
    X = X + MultiHeadAttention(LayerNorm(X))        # SL1: Attn, 20 heads
    X = X + ExpertChoiceMoE(LayerNorm(X))           # SL2: MoE, 64E, cap=1
    X = X + FFN_GeLU(LayerNorm(X))                  # SL3: FFN, hidden=2048
    X = X + MultiHeadAttention(LayerNorm(X))        # SL4: Attn
    X = X + ExpertChoiceMoE(LayerNorm(X))           # SL5: MoE
    X = X + FFN_GeLU(LayerNorm(X))                  # SL6: FFN
    X = X + ExpertChoiceMoE(LayerNorm(X))           # SL7: MoE
    X = X + FFN_GeLU(LayerNorm(X))                  # SL8: FFN
    return X
```

术语一般如何实现？如何使用？
- 通过 Evolutionary Search 自动发现最优 sub-layer 序列
- Ablation 发现：层类型比例（attention:MoE:FFN ratio）对质量至关重要，层顺序（order）相对不重要
- 搜索倾向于减少 attention 频率（attention 计算昂贵，尤其在长序列上）
- 搜索倾向于增大 model dim 同时减小 expansion ratio（利用 MoE 多 expert 替代单层大 FFN）
- 相关工作：Sandwich Transformer (Press et al., 2019) 重排但保持 uniform；EfficientNet (Tan & Le, 2019) per-layer scaling for CNN

涉及论文标题：
- Brainformers Trading Simplicity for Efficiency

---

## Training Time Constrained Search (训练时间约束搜索)

术语解释
Training Time Constrained Search 是 Brainformers 提出的公平模型比较和搜索方法：在固定的训练 wall clock time 预算（芯片数 × 训练时间）下进行架构搜索和评估，替代传统的"固定 training steps + 固定 params"比较范式。更快 step time 的模型自动获得更多 training steps，从而在架构搜索中天然偏向训练效率高的设计。

术语是什么？
传统模型比较的局限：
1. **Fixed params + fixed steps**：歧视总参数多但 activated params 少的稀疏模型
2. **Fixed activated params + fixed tokens**（GLaM 方法）：忽略 Chinchilla 定律——小模型可从更多数据受益
3. **Compute-efficient scaling**（Chinchilla）：固定 FLOPs budget，但未考虑架构变化的影响

Brainformers 扩展 Chinchilla 的理念到架构搜索：固定 wall clock budget，允许搜索算法在 model capacity 和 training steps 之间 trade off。

从算法pipeline角度拆解术语。
```
# Training Time Constrained Search 的优化框架

# 给定:
#   budget_chips = N_chips × training_hours  (固定)
# 约束:
#   step_time(architecture) ≤ baseline_step_time
# 目标:
#   minimize validation_perplexity at end of budget

# Step time 影响 training steps:
training_steps = budget_chips / step_time

# 例如 8B scale on 512 TPU V4:
# GLaM: step_time = 2.56s → 33,750 steps in 24h
# Brainformer-1: step_time = 0.51s → 169,412 steps in 24h
# → Brainformer-1 可在相同 wall clock time 内训练 5x 更多 steps

# 搜索 reward:
R = α × (1/perplexity) + β × (1/step_time)
# 或约束优化: R = -log(perplexity) s.t. step_time ≤ T_baseline
```

术语一般如何实现？如何使用？
- 适用于跨模型架构的公平比较（dense vs sparse, uniform vs non-uniform）
- 需要准确的 wall clock time 测量（包括所有通信、数据加载、同步开销）
- Early stopping 在 25% budget 时淘汰表现差的架构
- 局限性：对硬件平台敏感（不同硬件上的 step time ranking 不同）；FLOPs 不能替代 step time 测量（memory/communication 开销不可忽略）

涉及论文标题：
- Brainformers Trading Simplicity for Efficiency


## Expert Functional Redundancy in MoE (MoE 中的专家功能冗余)

术语解释
Expert Functional Redundancy 是 BuddyMoE 中提出的核心观察：大型 MoE 模型中多个 expert 存在功能相似性（functional similarity），即不同 expert 学习到相似或重叠的函数映射能力，这一观察构成了 buddy expert substitution 的理论基础。

术语是什么？
MoE 模型的 expert 功能冗余表现在两个层面：(1) **输出相似性**：多个 expert 对同一输入的 hidden states 输出高度相似；(2) **共激活模式**：特定 expert pairs 经常被同一 token 同时选中。实证证据包括 BuddyMoE 论文 Figure 4 的 expert similarity heatmap（大量 bright 区域表示 high similarity）和 Figure 7 的 co-activation heatmap（sparse but high-intensity patterns）。Prior work 确认 MoE 模型可容忍 aggressive pruning to 4 bits with minimal quality loss，间接验证了 expert 冗余。

C-PRUNE 论文进一步揭示了两个结构化的冗余层次：(1) **Intra-layer Expert Homogeneity（层内专家同质性）**：同一 MoE 层内的 expert 因训练动态发展出功能重叠，在参数空间中表现为高 cosine similarity（Figure 1 的 layer-specific heatmaps）；(2) **Inter-layer Similarity Patterns（跨层相似模式）**：深层（deeper layers）的 expert 比浅层更同质——rightmost heatmap 显示全局相似度随层深度递增。C-PRUNE 利用此观察在全局剪枝中施加 depth penalty，对深层 expert 给予更高的剪枝概率。

从算法pipeline角度拆解术语：
BuddyMoE 利用 expert 功能冗余的四阶段 pipeline：(1) 离线 profiling 量化冗余——在 calibration corpus 上记录 pairwise co-activation count M[i][j]；(2) 条件共激活分布计算 q_{j|i} = M[i][j] / Σ M[i][j']；(3) CFT buddy list 构建——对每个 pivot i，按 q_{j|i} 降序排列，选前缀覆盖 α 比例累积激活 mass；(4) 运行时替代——缺失 expert 被 GPU-resident buddy 替代，~0ms vs ~10ms CPU→GPU 传输。Layer-wise heterogeneity：早期层呈现 broader redundancy（更 diffuse），后期层更 specialized（tighter clusters）。

术语一般如何实现？如何使用？
- 冗余度量方法：co-activation frequency、output similarity（cosine/MSE）、或组合
- Profiling dataset 需匹配部署领域以准确反映路由行为
- 冗余是 BuddyMoE、expert pruning、expert merging 等技术的共同前提
- 关键限制：冗余程度因模型架构（expert 数量、gating 策略、训练数据）而异，需 per-model profiling

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

---

## Buddy Expert Substitution (Buddy 专家替代)

术语解释
Buddy Expert Substitution 是 BuddyMoE 的核心运行时机制：在 MoE 推理中，当 router 选择的 expert 不在 GPU 显存（cache miss/prefetch miss），不等待同步 CPU→GPU 传输（~10ms），而是用 GPU 显存中功能相似的 "buddy expert" 即时替代（~0ms），以极小精度损失换取显著吞吐提升。

术语是什么？
Buddy Expert 通过离线共激活模式分析和 CFT 构建。运行时替代流程（Algorithm 1）：对于每个 token 的每个 CPU-resident expert e_id，按 B_ℓ[e_id] 的 buddy ranking 查找 GPU-resident 的 buddy b_id，通过 atomic CAS 操作确保 uniqueness constraint 后替换到 S'。替代通过三个 safety gate 控制：TAE Gate（token 级敏感度）、Distribution Gate（batch 级 CPU expert 比例）、Buddy Priority Score Ψ（全局相似性 × 局部兼容性 × 拓扑感知）。CUDA kernel 并行化：grid(T,1,1) × block(K,1,1)，shared memory U_t 维护 token 的已分配 expert set，atomicCAS 保证无锁唯一性。

从算法pipeline角度拆解术语：
```
for token t in batch:
    S = Router(x_t).TopK(k)
    for e_id in S where M[e_id] == false:  # CPU-resident
        for r in range(H):
            b_id = B[e_id][r]
            if M[b_id] and b_id not in S':
                S'[e_id] = b_id; break
    output = Σ weight_i * FFN_i(x_t)  # with S'
```

术语一般如何实现？如何使用？
- 集成到 llama.cpp serving 框架，作为 router 和 expert execution 间的中间层
- Buddy profile 离线生成并序列化随 model checkpoint 加载，O(K_max · E_ℓ) 存储可忽略
- 与现有 prefetching 互补：prefetch 成功时正常工作，失败时用 buddy 避免 stall
- 最大收益在极端内存约束下：cache rate=0.375 时 +10.3% t/s vs original baseline

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

---

## Expert Co-activation Matrix (专家共激活矩阵)

术语解释
Expert Co-activation Matrix M_ℓ(i,j) 是 BuddyMoE 中量化 MoE expert 间功能关系的数据结构，记录在 profiling corpus 中 expert i 和 j 被同一 token 同时选中的频率。从 M 导出的条件共激活分布 q_{j|i} = M_ℓ(i,j) / Σ M_ℓ(i,j') 是 buddy selection 的基础。

术语是什么？
给定 layer ℓ 的 E 个 experts：对 profiling corpus 中的每个 token x，router 选中 top-k expert set S_ℓ(x)，对于 (i,j) ∈ S_ℓ(x)×S_ℓ(x), i≠j，M_ℓ[i][j]++。条件分布 q_{j|i} 量化"给定 expert i 被选中时 j 也同时被选中的条件概率"。高 q_{j|i} 暗示 i 和 j 处理相似的 token 子流形（功能相似）。关键实证性质：(1) A_ℓ(·) heavy-tailed——少数 popular expert 占多数激活；(2) q_{j|i} mass 集中在少数 peers——top-r peers (r≪E) 覆盖大量 co-activation；(3) layer-wise heterogeneity——早期层 diffuse，后期层 tight clusters。

从算法pipeline角度拆解术语：
```
M = zeros(E, E)
for x in profiling_data:
    S = TopK(Router(x), k)
    for i in S:
        for j in S, j != i:
            M[i][j] += 1
q[i][j] = M[i][j] / sum(M[i])
buddy_ranking[i] = argsort_descending(q[i])
B[i] = buddy_ranking[i][:t] where cumsum(q) >= alpha
```

术语一般如何实现？如何使用？
- 可选概率加权变体：Σ 𝟙{i,j∈S} · min(p̃(i), p̃(j)) 利用 router probability 作为 soft weight
- Laplace smoothing (M←M+ε) 防零概率，down-weight early warm-up steps 减冷缓存 artifact
- 是 buddy construction、expert pruning、expert merging 的通用预处理
- **CoE 的 intra-layer 视角**：CoE 中的 co-activation 矩阵记录同一 token 在不同 iteration（t vs t+1）的 expert 选择对，衡量 expert transition pattern。对角线低 = flowing nature（token 倾向于跨步切换 expert）；非对称分布 = role differentiation

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models
- Continual Pre-training of MoEs How robust is your router（CPT 分析：Granular PBTk MoE 在 CPT 过程中 layers 0-1 的 co-activation 变化最大，layer 18 出现一致的 spike。0% replay checkpoint 的 co-activation 变化最大且遗忘最严重 → 更显著的 co-activation 变化与更高遗忘相关。SB Granular MoE 在 pre-training 时 co-activation 高度集中在 expert 15，CPT 后分散化）

---

## Token Activating Entropy (TAE, Token 激活熵)

术语解释
Token Activating Entropy (TAE) 是 BuddyMoE 的第一个 safety gate 指标，量化 token 对 expert 替代的敏感度：TAE_ℓ(x) = -Σ_{i∈S} p̃_ℓ(i|x)·log(p̃_ℓ(i|x)) / log(k) ∈ [0,1]。低 TAE=peaked routing（敏感，禁止替换），高 TAE=diffuse routing（容忍，允许替换）。当 TAE ≤ τ 时禁止替换。

术语是什么？
TAE 复用 router 的 top-k softmax 输出，在 renormalized top-k 概率上计算归一化信息熵。归一化因子 log(k) 使值域为 [0,1]。三个实现细节：(1) renormalize over top-k only 避免尾部 artifact；(2) optional temperature smoothing p̃(x;T)=softmax(z(x)/T) with T∈[0.8,1.2] 稳定跨层 TAE；(3) percentile calibration——τ 按 per-layer TAE 分布的 p-th percentile (p∈[10,20]) 选择，跨模型/领域鲁棒。可选与 probability margin (p̃_max - p̃_2nd ≥ γ) 组合增强安全性。

从算法pipeline角度拆解术语：
```
topk_probs = Router(x).softmax()[topk_indices]
p_tilde = topk_probs / sum(topk_probs)
TAE = -sum(p_tilde * log(p_tilde)) / log(k)
if TAE <= tau: forbid replacement for this token
```

术语一般如何实现？如何使用？
- 计算开销 O(k) 可忽略（复用已有 router 输出）
- τ 选择影响 accuracy-throughput trade-off：较高 τ 更保守，较低 τ 更激进
- 论文 τ=0.95 获最佳平衡（c=0.75: Acc=0.695, t/s=36.75, +7.4% vs original）
- TAE 是三个 safety gate 中最先执行的（overhead 最低）

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

---

## Cumulative Frequency Threshold (CFT, 累积频率阈值)

术语解释
Cumulative Frequency Threshold (CFT) 是 BuddyMoE 中从共激活数据构建紧凑 buddy expert 列表的算法参数。对于 pivot expert i，按 q_{j|i} 降序排列 peers，选最小前缀 t 使累积覆盖 ≥ α，构成 buddy list B_ℓ(i;α)。α∈(0,1] 是唯一超参数：α 越大 → 更大 buddy list → 更高 GPU 命中率；α 越小 → 更紧相似性 → 更小 buddy list。

术语是什么？
CFT 公式：t_i(α) = min{t | Σ_{r=1..t} q_{π_i(r)|i} ≥ α}，B_ℓ(i;α) = {π_i(1), ..., π_i(t_i(α))}。Capped at K_max（如 16）以控制 metadata。支持 per-layer α_ℓ 或 monotone schedule 适应不同层的冗余模式差异。Stabilization techniques：同时累积 binary + probability-weighted co-activation；Laplace smoothing M←M+ε；可选 down-weight early warm-up steps。

从算法pipeline角度拆解术语：
```
buddy_list = []
cumsum = 0
for peer in argsort_descending(q[i]):
    cumsum += q[i][peer]
    buddy_list.append(peer)
    if cumsum >= alpha: break
return buddy_list[:K_max]
```

术语一般如何实现？如何使用？
- 离线一次性计算，无运行时开销
- Profiling data 需匹配部署领域以确保 co-activation 模式代表性
- Verifying compactness：报告 |B_ℓ(i;α)| 分布确保 buddy lists 紧凑

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference

## Test-Time Expert Re-Mixing (测试时专家重混合)

术语解释
Test-Time Expert Re-Mixing 是 C3PO 提出的 MoE LLM 测试时自适应范式：在推理阶段，不修改任何模型参数，仅通过优化 expert routing weights（专家路由权重）来重新混合各层的 expert 贡献比例，使每个测试样本获得定制化的 expert pathway。核心优势：优化变量极少（几十到几百个 routing weights vs prompt tuning 的数千维 token embeddings），且部分方法无需反向传播。

术语是什么？
给定预训练的 MoE LLM，其每层有一个 router（gate）计算 expert 的选择概率。Test-Time Expert Re-Mixing 在推理时将 routing weights 视为可优化变量 ω ∈ R^{L×E}（L 层数，E 专家数），通过最小化 surrogate objective 来调整 ω，使得模型在测试样本 x 上的输出 f(x, ω) 更准确。

C3PO 发现预训练的 end-to-end router 存在严重的次优性（sub-optimality）：base model 与 Oracle（使用 ground truth 找到的最优 routing）之间存在 10-20% accuracy gap。这表明仅靠 pretraining 的 router 无法为每个样本找到最优 expert 组合，尤其是对于困难样本或分布外样本。

三种 Re-Mixing 方法：
1. **Mode Finding (Meanshift)**: 在 pathway 权重空间中找到邻居样本 pathway 的最密集区域，将当前 ω 向该区域移动
2. **Kernel Regression**: 用邻居样本 pathway 的核加权平均作为估计值，与原始 ω 插值
3. **Neighborhood Gradient Descent (NGD)**: 用邻居样本 loss 的加权平均作为 surrogate objective，梯度下降优化 ω（性能最强）

从算法pipeline角度拆解术语：
Test-Time Expert Re-Mixing 的完整流程（以 NGD 为例）：

```
# 输入: 测试样本 x, MoE模型 f, 参考集 {(x_i, y_i, ω_i)}
# 超参数: k=3 (邻居数), steps=10, lr cosine 1e-2→1e-5, Gaussian kernel

# Phase 1: 嵌入与检索
emb_x = NV-Embed-V2(task_description(x))     # 用任务描述获取嵌入
emb_ref = {NV-Embed-V2(task_description(x_i))} # 参考集样本嵌入
N = kNN(emb_x, emb_ref, k=3)                 # 检索 k=3 个最近邻

# Phase 2: 初始 pathway 提取
ω_0 = f.get_routing_weights(x)               # shape: [L, E] 或 [L, E, T]
ω = ω_0[last_5_layers, top_20_experts]       # Critical-Layer + Core-Expert 裁剪

# Phase 3: NGD 迭代优化
for step in range(10):
    total_loss = 0
    total_weight = 0
    for i in N:
        K_val = exp(-||emb_x - emb_i||^2 / (2 * σ^2))  # Gaussian kernel
        logits_i = f.forward(x_i, routing_override=ω)   # 替换 routing weights
        loss_i = cross_entropy(logits_i, y_i)
        total_loss += K_val * loss_i
        total_weight += K_val
    
    surrogate_loss = total_loss / total_weight
    grad = ∇_ω surrogate_loss
    lr = cosine_schedule(step, 10, 1e-2, 1e-5)
    ω = ω - lr * grad

# Phase 4: 推理
output = f.forward(x, routing_override=ω)
```

关键设计选择：
- 仅优化最后 1 个 token 的 routing weights（而非所有 token），因最后一个 token 承载最多的任务决策信息
- 仅优化最后 5 层的 routing weights（而非全部 16 层），深层负责任务特定的精炼
- 仅优化 top-20 experts 的 routing weights（而非全部 64 个），覆盖最终激活的 top-8 experts 的 99.8%

术语一般如何实现？如何使用？
- **实现**: 替换 HuggingFace transformers 中的 `olmoe_modeling.py`，在 MoE 层的 forward 中注入优化后的 routing weights。主优化逻辑在 `olmoe_optimizer.py` 中。
- **依赖**: PyTorch, CUDA 12.3, Python 3.10, NV-Embed-V2 (embedding model)
- **参考集构建**: 需要为每个 benchmark 准备参考集——收集模型输出正确的样本及其对应的 expert pathway。参考集与 benchmark 不同但领域相关（如 MMLU 用 BIG-Bench + SuperGLUE 作参考集）
- **适用场景**: 任何 MoE LLM（OLMoE, DeepSeekMoE 等），只需替换 routing weights 的注入逻辑
- **开源**: https://github.com/tianyi-lab/C3PO (Apache-2.0, COLM 2025)

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Expert Pathway / Routing Weights in MoE (MoE中的专家路径/路由权重)

术语解释
Expert Pathway（专家路径）是 MoE LLM 中一个样本在逐层经过所有 MoE 层时，每层被选中的 experts 及其对应权重的序列。形式上，对于 L 层、每层 E 个 experts 的 MoE，pathway 是一个矩阵 ω ∈ R^{L×E}，其中 ω_{l,e} 表示第 l 层第 e 个 expert 的路由权重。Pathway 决定了每个 token 在各层中由哪些 expert 处理以及各 expert 的贡献比例。

术语是什么？
在标准 MoE 架构中，每层的 router（gate）计算：

$$s_{l,e} = x \cdot W_{\mathrm{gate}}[e] \quad \text{(affinity score)}$$

$$w_{l,e} = \frac{\exp(s_{l,e})}{\sum_{j \in \mathrm{TopK}(s_l)} \exp(s_{l,j})} \quad \text{(routing weight after TopK softmax)}$$

最终 MoE 层输出：

$$h_l = x + \sum_{e \in \mathrm{TopK}(s_l)} w_{l,e} \cdot \mathrm{Expert}_e(x)$$

Pathway 矩阵 ω 收集所有层的 {w_{l,e}}。Router 在 pretraining 阶段与模型参数端到端训练，推理时冻结。

C3PO 的发现：预训练的 router 产生的 pathway 存在严重次优性。在 OLMoE 上，base model 的 pathway 与 Oracle pathway 之间存在 15.3% accuracy gap（69.9% vs 85.2%）。

从算法pipeline角度拆解术语：
Pathway 在 MoE 推理中的流转：

```
输入 token x → Layer 1:
  gate_logits = x @ W_gate[1]            # [E] = [64]
  topk_weights, topk_idx = topk(softmax(gate_logits), k=8)
  h_1 = x + Σ_{j in topk_idx} topk_weights[j] * Expert_{1,j}(x)
  → 记录 ω[1, topk_idx] = topk_weights

→ Layer 2: ... (重复)
→ Layer L: 记录 ω[L, topk_idx]

最终 pathway ω ∈ R^{L×E}，其中非 top-k 位置为 0
```

C3PO 对 pathway 的操作：
- **读取**: 从模型 forward pass 中 hook 各层的 gate_logits 或 softmax 后的 routing weights
- **修改**: 在 gate_logits 层面加偏移 Δω（直接修改 softmax 前的 logits），或直接替换 routing weights
- **裁剪**: 只保留 Critical Layers（最后 5 层）和 Core Experts（top-20）的 routing weights

术语一般如何实现？如何使用？
- **获取 pathway**: HuggingFace 模型的 forward hook 机制，在 MoE 层的 router 输出处注册 hook 捕获 routing weights
- **注入 pathway**: 替换 `olmoe_modeling.py` 中的 MoE 层实现，在 forward 时接受外部 routing weights 参数覆盖内部 router 输出
- **优化粒度**: C3PO 实验表明优化单个 last token 的 pathway 效果最好（vs 多个 token），因为 last token 承载了最丰富的任务决策信号

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Collaborative Pathway Optimization (CPO, 协同路径优化)

术语解释
Collaborative Pathway Optimization 是 C3PO 的核心优化范式：利用参考集中多个"成功样本"的 expert pathway 来协同优化测试样本的 pathway。与传统的 prompt tuning（优化连续 prompt embeddings）或 ICL（拼接示例到输入）不同，CPO 直接操作低维的 routing weights 空间，且利用邻居样本之间的协作（collaboration）而非孤立优化。

术语是什么？
CPO 的"协同"体现在三个层面：
1. **邻居协同**: 不是用单个最相似样本的 pathway，而是用 k 个邻居的 pathway 加权融合（kernel weighting）
2. **跨层协同**: pathway 矩阵同时编码了所有层的 routing weights，优化时跨层信息自然交互
3. **参考集协同**: 参考集中的样本互不重叠，但通过 kernel 函数为不同测试样本提供不同的协同信号

CPO 的三种具体形式：
- **Mode Finding (Meanshift)**: 梯度自由，在 ω-space 中找到邻居 pathway 的最密集模式
- **Kernel Regression**: 梯度自由，用邻居 pathway 的核加权平均作为目标
- **NGD**: 梯度方法，用邻居 loss 的加权平均作为 surrogate objective

从算法pipeline角度拆解术语：
CPO 的通用框架伪代码：

```
def CPO(test_sample x, model f, reference_set D_ref, method="NGD"):
    # Step 1: 嵌入检索
    emb = embed(x)
    neighbors = knn(emb, D_ref.embeddings, k=3)
    
    # Step 2: 提取当前 pathway
    ω_curr = extract_routing_weights(f, x, layers=last_5, experts=top_20)
    
    # Step 3: 协同优化
    if method == "ModeFinding":
        ω_new = meanshift(ω_curr, neighbors.omegas, kernel="gaussian")
    elif method == "KernelRegression":
        ω_hat = sum(K(x, xi) * ω_i for xi, ω_i in neighbors) / sum(K(x, xi))
        α = argmin_α surrogate_loss(α*ω_curr + (1-α)*ω_hat)
        ω_new = α*ω_curr + (1-α)*ω_hat
    elif method == "NGD":
        for step in range(10):
            loss = weighted_average([loss(f(xi, ω), yi) for xi, yi in neighbors])
            ω_curr -= lr * ∇_ω loss
    
    return f.forward(x, routing_override=ω_new)
```

CPO vs 其他 test-time adaptation 方法：

| 方法 | 优化变量 | 变量维度 | 需要反向传播 | 参考集需求 |
|------|---------|---------|------------|----------|
| ICL | 无（拼接示例） | 0 | 否 | 大（few-shot examples） |
| Prompt Tuning | Soft prompt tokens | d×len(prompt) | 是 | 全量参考集 |
| Prefix Tuning | Prefix vectors | L×d×len(prefix) | 是 | 全量参考集 |
| CPO (Kernel Reg) | Routing weights | L×E (subset) | 否 | 仅 kNN 邻居 |
| CPO (NGD) | Routing weights | L×E (subset) | 是 | 仅 kNN 邻居 |

术语一般如何实现？如何使用？
- 实现与 Test-Time Expert Re-Mixing 共享代码框架
- 关键超参数：k=3（kNN 邻居数），steps=10（NGD 优化步数），Gaussian kernel（核函数选择），cosine annealing LR（学习率调度）
- NGD 收敛快：前 6 步贡献 +11.6% accuracy gain，10 步后 plateau
- 仅 5.1% 的初始正确预测在优化后被翻转为错误（稳定性好）

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Neighborhood Gradient Descent for MoE Pathway (邻域梯度下降MoE路径优化)

术语解释
Neighborhood Gradient Descent (NGD) 是 C3PO 中性能最强的 pathway 优化方法。它不直接使用测试样本的 ground truth（未知），而是用参考集中 kNN 邻居样本的 loss 加权平均作为 surrogate objective，对 routing weights 做梯度下降。NGD 达到 Oracle 性能的 85-95%，是 C3PO 三种方法中唯一需要反向传播的方法。

术语是什么？
NGD 的核心公式：

$$L(\omega) = \frac{\sum_{i \in \mathcal{N}(x)} K(x_i, x) \cdot \ell(f(x_i, \omega), y_i)}{\sum_{i \in \mathcal{N}(x)} K(x_i, x)}$$

其中 N(x) 是 x 的 k=3 个最近邻（基于 embedding 相似度），K(x_i, x) = exp(-||E(x_i) - E(x)||^2 / (2σ^2)) 是 Gaussian kernel，ℓ(f(x_i, ω), y_i) 是邻居样本在当前优化中的 ω 下的 cross-entropy loss。

关键洞察：虽然测试样本 x 的 ground truth 未知，但邻居样本的 ground truth 已知。如果 ω 能让邻居样本的 loss 降低，那么它很可能也能让 x 的输出变好。

从算法pipeline角度拆解术语：
```
def NGD_optimize(x, model, ref_set, k=3, steps=10):
    emb_x = embedding_model(x)
    neighbors = ref_set.knn(emb_x, k=3)
    
    ω = model.get_routing_weights(x)[last_5_layers][:, top_20_experts]
    K_vals = [exp(-||emb_x - emb_xi||^2 / (2*h^2)) for xi in neighbors]
    K_sum = sum(K_vals)
    
    optimizer = SGD([ω], lr=1e-2)
    scheduler = CosineAnnealing(optimizer, T_max=10, eta_min=1e-5)
    
    for step in range(steps):
        total_loss = 0
        for (xi, yi), K_val in zip(neighbors, K_vals):
            logits_i = model.forward(xi, routing_override=ω)
            total_loss += (K_val / K_sum) * cross_entropy(logits_i, yi)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
    return ω
```

术语一般如何实现？如何使用？
- 需要对 MoE 模型的 routing weights 可微（梯度可以从 loss 回传到 ω）
- 在 HuggingFace 实现中，需修改 MoE 层的 forward 使 routing weights 成为可优化的 nn.Parameter
- 每个测试样本独立运行 NGD，不共享状态
- FLOPs 开销：主要在邻居样本的前向传播（k=3 × steps=10 = 30 次前向）

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Mode Finding / Meanshift for MoE Routing (MoE路由的模式发现/均值漂移)

术语解释
Mode Finding (Meanshift) 是 C3PO 中的梯度自由 pathway 优化方法。它借鉴均值漂移（Mean Shift）聚类算法的思想：在 expert pathway 的权重空间（ω-space）中，将测试样本的 pathway 向邻居样本 pathway 的最密集区域（mode）迭代移动。不需要反向传播，计算开销最小。

术语是什么？
Mode Finding 的更新公式：

$$\bar{\omega} = \frac{\sum_{i \in \mathcal{N}(\omega)} K(\omega_i, \omega) \cdot \omega_i}{\sum_{i \in \mathcal{N}(\omega)} K(\omega_i, \omega)}$$

$$\omega \leftarrow \alpha \cdot \omega + (1-\alpha) \cdot \bar{\omega}$$

关键区别：NGD 在 x-space 中定义邻居并在 loss space 中优化；Mode Finding 在 ω-space 中定义邻居并在 ω-space 中做 meanshift。Kernel Regression 在 x-space 中定义邻居并在 ω-space 中做加权平均。

从算法pipeline角度拆解术语：
```
def mode_finding(ω_curr, ref_pathways, bandwidth, alpha=0.5, max_iter=5):
    ω = ω_curr.clone()
    for t in range(max_iter):
        distances = [||ω - ω_i||^2 for ω_i in ref_pathways]
        K_vals = [exp(-d^2 / (2 * bandwidth^2)) for d in distances]
        ω_bar = sum(K_i * ω_i for K_i, ω_i in zip(K_vals, ref_pathways)) / sum(K_vals)
        ω = alpha * ω + (1 - alpha) * ω_bar
    return ω
```

| 特性 | Mode Finding | Kernel Regression | NGD |
|------|-------------|-------------------|-----|
| 需要梯度 | 否 | 否 | 是 |
| 邻居空间 | ω-space | x-space | x-space |
| 优化空间 | ω-space | ω-space | loss space |
| 性能 (OLMoE avg) | 72.4% | 76.9% | 79.2% |

术语一般如何实现？如何使用？
- 梯度自由，适合资源受限场景或快速原型验证
- 需为参考集中每个样本存储其成功的 pathway 矩阵（存储开销：|D_ref| × L × E × 4 bytes）
- bandwidth 参数控制 ω-space 中邻居的影响范围

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Kernel Regression for MoE Pathway Estimation (MoE路径的核回归估计)

术语解释
Kernel Regression for Pathway 是 C3PO 中的梯度自由 pathway 优化方法。它在样本嵌入空间（x-space）中用 Gaussian kernel 加权平均邻居的 pathway 矩阵，得到目标 pathway 的估计值，然后通过最优插值系数 α* 平衡估计值与原始 pathway。

术语是什么：

$$\hat{\omega} = \frac{\sum_{i \in \mathcal{N}(x)} K(x_i, x) \cdot \omega_i}{\sum_{i \in \mathcal{N}(x)} K(x_i, x)}$$

$$\omega \leftarrow \alpha^* \cdot \omega + (1-\alpha^*) \cdot \hat{\omega}$$

其中 α* 通过在邻居样本上最小化 surrogate loss 搜索得到：

$$\alpha^* = \arg\min_{\alpha} L(\alpha \cdot \omega + (1-\alpha) \cdot \hat{\omega})$$

核心思想：如果测试样本 x 与参考样本 x_i 在语义上相似，那么它们的 optimal pathway 也应该相似。

从算法pipeline角度拆解术语：
```
def kernel_regression_pathway(x, model, ref_set, k=3):
    emb_x = embedding_model(x)
    neighbors = ref_set.knn(emb_x, k=3)
    K_vals = [exp(-||emb_x - emb_xi||^2 / (2*h^2)) for xi in neighbors]
    
    # 核加权平均
    ω_hat = sum(K_i * ω_i for K_i, ω_i in zip(K_vals, neighbors.omegas)) / sum(K_vals)
    
    # 搜索最优 α
    ω_curr = model.get_routing_weights(x)
    best_alpha = min(range(0, 11), key=lambda a:
        sum(K_i * loss(model(xi, a*0.1*ω_curr + (1-a*0.1)*ω_hat), yi) 
            for xi, yi, K_i in zip(neighbors.x, neighbors.y, K_vals)))
    
    return best_alpha*0.1 * ω_curr + (1-best_alpha*0.1) * ω_hat
```

Kernel 选择的影响（OLMoE）：Linear 69.95%, Polynomial 73.33%, Matern 76.28%, Gaussian 79.20%。

术语一般如何实现？如何使用？
- 梯度自由，计算成本低于 NGD
- 性能介于 Mode Finding (72.4%) 和 NGD (79.2%) 之间
- α* 搜索粒度 0.1，影响最终精度

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Critical-Layer Optimization in MoE (MoE中的关键层优化)

术语解释
Critical-Layer Optimization 是 C3PO 的优化策略之一：在 pathway 优化时只修改 MoE 模型中部分"关键层"的 routing weights，而非全部层。实验发现只优化最后 5 层的 routing weights 不仅节省计算，而且性能反超全 16 层优化（OLMoE: L5 79.2% vs All16 77.7%）。

术语是什么？
C3PO 的层重要性分析揭示三层 hierarchy（OLMoE 16 层）：
- **深层 (Late/L)**: 最重要，负责任务特定的高层语义理解
- **浅层 (Early/F)**: 次重要，编码基础特征表示
- **中层 (Middle/M)**: 过渡角色，对最终预测影响最小

规律：M1 < F1 < L1, M2 < F2 < L2, M5 < F5 < L5

从算法pipeline角度拆解术语：
```
def extract_critical_layers(all_routing_weights, strategy="last_5"):
    critical_layers = {"last_5": [12,13,14,15,16], "first_2_last_3": [1,2,14,15,16]}
    ω_opt = {l: all_routing_weights[l] for l in critical_layers[strategy]}
    ω_frozen = {l: all_routing_weights[l] for l not in critical_layers[strategy]}
    return ω_opt, ω_frozen
```

术语一般如何实现？如何使用？
- 层选择策略需在目标 MoE 模型上通过 ablation 验证
- OLMoE 最优: 最后 5/16 层；DeepSeekMoE (28层) 论文未单独报告
- 原则：深层 > 浅层 > 中层，组合时应优先包含深层

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Core-Expert Selection in MoE (MoE核心专家选择)

术语解释
Core-Expert Selection 是 C3PO 的优化策略之一：在 pathway 优化时只修改部分"核心专家"的 routing weights。实验发现只优化 top-20 experts 即可覆盖最终 top-8 的 99.8%，性能与全 64 expert 优化持平，但优化变量减少 68.75%。

术语是什么？
稀疏 MoE（如 OLMoE）每层 64 experts 但只激活 top-8。C3PO 策略：
1. 按预训练 router 的初始 routing weights 对 64 experts 排序
2. 只保留 top-n experts 作为可优化变量
3. 被排除的 experts 的 routing weights 保持为 0

覆盖率实验：top-8 覆盖 71.3%, top-12 提高, top-20 覆盖 99.8%

从算法pipeline角度拆解术语：
```
def select_core_experts(gate_logits, n_core=20):
    sorted_indices = argsort_descending(gate_logits)
    core_indices = sorted_indices[:n_core]
    ω_core = gate_logits[core_indices]       # 可优化
    return ω_core, core_indices
```

总体节省（Critical-Layer + Core-Expert）：16×64=1024 → 5×20=100，优化变量减少 90.2%

术语一般如何实现？如何使用？
- n_core 需在目标模型上通过 ablation 确定
- 不同 MoE 架构的 top-k 激活数不同，需要的 n_core 也不同
- 可结合 expert 激活频率统计动态确定每层的 n_core

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Reference Set for Test-Time MoE Adaptation (MoE测试时自适应的参考集)

术语解释
Reference Set 是 C3PO 测试时自适应的数据基础：一个预先准备的数据集，包含模型输出正确的样本及其对应的 expert pathway。测试时，对每个新样本从参考集中检索最相似的 k 个邻居，利用其 successful pathway 来指导 routing weights 的优化。

术语是什么？
参考集构建要求：
- **正确性**: 样本在 base model 上的输出正确（f(x_i, ω_i) = y_i）
- **无重叠**: 参考集与测试 benchmark 不重叠（使用领域相关但不同的数据集），过滤问题相似度 > 0.95 的样本
- **覆盖度**: 覆盖多种任务类型，提供多样化 pathway 模式

C3PO benchmark-参考集配对：MMLU→BIG-Bench+SuperGLUE, HellaSwag/PIQA→CommonsenseQA+SocialIQA, ARC-C/E→OpenBookQA+SciQ, WinoGrande→KnowRef

从算法pipeline角度拆解术语：
```
def build_reference_set(base_model, reference_data, embedding_model):
    ref_set = []
    for (x_i, y_i) in reference_data:
        ω_i = base_model.get_routing_weights(x_i)
        if argmax(base_model.forward(x_i)) == y_i:
            emb_i = embedding_model(task_description(x_i))
            ref_set.append({'x': x_i, 'y': y_i, 'omega': ω_i, 'embedding': emb_i})
    return ref_set
```

术语一般如何实现？如何使用？
- 存储开销: |D_ref| × (|x| + |y| + L×E×4 bytes + |emb|)
- Embedding 模型质量是关键——更好的 embedding → 更相关的邻居 → 更好的优化
- 参考集可跨任务共享，离线构建后序列化存储

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing

## Top-k Gating / Token-Choice Routing in MoE (MoE中的Top-k门控/Token选择路由)

术语解释
Top-k Gating（Token-Choice Routing）是 MoE 模型中最主流的路由策略：每个 token 独立选择 top-k 个 experts，仅被选中的 experts 参与该 token 的前向计算。这是稀疏 MoE 实现"条件计算"的核心——每个 token 只激活总 expert 数的一小部分（8/64 = 12.5%）。

术语是什么：
$$s_{t,e} = x_t^\top \cdot W_{\mathrm{gate}}[e], \quad \mathcal{K}_t = \mathrm{TopK}(\{s_{t,e}\}, k)$$
$$w_{t,e} = \frac{\exp(s_{t,e})}{\sum_{j \in \mathcal{K}_t} \exp(s_{t,j})} \text{ if } e \in \mathcal{K}_t \text{, else } 0$$
$$h_t = x_t + \sum_{e \in \mathcal{K}_t} w_{t,e} \cdot \mathrm{FFN}_e(x_t)$$

C3PO 不改变 gating 机制本身，而是通过修改 routing weights w_{t,e}（或 gate logits s_{t,e}）来 re-mix expert 贡献。

从算法pipeline角度拆解术语：
```
class SparseMoELayer:
    def forward(self, x, routing_override=None):
        gate_logits = self.gate(x)
        if routing_override is not None:
            gate_logits = gate_logits + routing_override  # C3PO 注入点
        
        topk_weights, topk_indices = torch.topk(
            torch.softmax(gate_logits, dim=-1), k=self.top_k, dim=-1)
        
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = (topk_indices == e).any(dim=-1)
            if mask.any():
                output[mask] += topk_weights[mask][...].unsqueeze(-1) * self.experts[e](x[mask])
        return output
```

术语一般如何实现？如何使用？
- k 是关键超参数：OLMoE k=8/E=64, Mixtral k=2/E=8, DeepSeekMoE k=6/E=64+2 shared
- C3PO 利用 last token 的 routing weights 做优化——last token 在自回归生成中承载最多决策信号
- CoMoE 在 LoRA-based MoE PEFT 中基于 top-k routing 构建对比学习：激活的 top-k expert 作为正样本，其余 n-k 个非激活 expert 作为负样本，通过 InfoNCE loss 促进 expert 专业化

涉及论文标题：
- C3PO Critical-Layer, Core-Expert, Collaborative Pathway Optimization for Test-Time Expert Re-Mixing
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

## Straggler Effect in MoE (MoE中的掉队者效应)

术语解释
Straggler Effect 是 Expert Parallelism 下 MoE 推理中由 token-to-expert 分配不均衡导致的延迟瓶颈现象：高负载 expert 处理大量 token 耗时最长，低负载 expert 提前完成计算后必须等待 All-to-All barrier 同步，导致 GPU 利用不均和端到端延迟由最繁忙 expert 决定。

术语是什么？
在 Expert Parallelism 下，expert 分布在不同 GPU，每 expert 的计算时间由其分配的 token 数决定。MoE 层的延迟 L ∝ max({N_i})——由最繁忙 expert 的 token 数决定（N_i 为第 i 个 expert 分配的 token 数）。推理时 token-to-expert 分布极为不均衡：以 OLMoE 为例，最高负载 expert 收到超过 7× 平均负载的 token（Figure 1, 2），导致延迟瓶颈。延迟范围：max({N_i}) ∈ [N̄, nN̄/k]，N̄=tk/n 为期望 token 数。从 scratch 训练的 MoE（OLMoE, DeepSeek-V2）比 upcycling 模型（Mixtral, Qwen1.5-MoE）不均衡更严重（peak >5N̄ vs <3N̄）。

从算法pipeline角度拆解术语：
Straggler Effect 的成因链：Router 产生 skewed token distribution → 少数 expert 聚集大量 token → EP 下持有这些 expert 的 GPU 计算时间异常长 → 其他 GPU 完成计算后在 All-to-All barrier 空闲等待 → 端到端延迟 = max GPU compute time。

以 Mixtral-8×7B-Instruct 一个 MoE 层为例（8 experts, 8 GPU EP, batch 8K × seq 512）：
```
N̄ = (8000×512×2)/8 = 1,024,000 (期望)
实际: expert_3=3,500,000 tokens, expert_7=150,000 tokens
→ GPU_3 (expert_3) compute ∝ 3.5M → 最慢, GPU_7 (expert_7) ∝ 0.15M → 最快
→ GPU_7 提前完成 → idle 等待 All-to-All barrier
→ MoE 层延迟 ∝ max(3.5, 0.15, ...) = 3.5M = GPU_3 决定
```

术语一般如何实现？如何使用？
Capacity-Aware Inference 通过 Token Drop 限制 max(N_i) ≤ γN̄ 来缓解：γ=1.5 时 Mixtral 获 1.85× 加速。MoEShard 通过 expert tensor sharding 使所有 GPU 计算量均等，从根本上消除 Straggler Effect。

涉及论文标题：
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

## Capacity-Aware Expanded Drop

术语解释
在 Token Drop 施加 expert 容量约束前，将 token 的候选 expert 集从 top-k 扩展为 top-k+m（m 为本地设备 expert 数），使溢出 token 能被有剩余容量的低负载 expert 吸收处理，在容量约束内同时提升负载均衡和模型表示能力。

术语是什么？
Expanded Drop 利用低负载 expert 的剩余容量：Token Drop 仅丢弃超载 expert 的溢出 token → 低负载 expert 容量未被利用 → Expanded Drop 先扩展候选集再施加容量约束 → 溢出 token 被重分配到有容量的 expert 而非丢弃。扩展仅限本地设备 expert（无跨设备 All-to-All 通信开销），利用 gating score 分布的长尾平坦特性（Figure 8：top-k 外 expert 的 score 与 top-k 内末尾 expert 接近）。不强制限制每 token 最多 k 个 expert（w/o max 优于 w/ max, Table 11）。

从算法pipeline角度拆解术语：
```python
def expanded_drop(x, k, gamma, local_ids):
    scores = softmax(gate(x))                       # [N, E]
    topk_scores, topk_idx = scores.topk(k, dim=1)   # [N, k]
    
    # 扩展候选: top-k + 本地所有 expert
    local_idx = local_ids.repeat(N, 1)               # [N, m]
    exp_idx = cat([topk_idx, local_idx], dim=1)      # [N, k+m]
    local_scores = scores[:, local_ids]              # [N, m]
    exp_scores = cat([topk_scores, local_scores], dim=1)
    
    exp_mask = scatter(zeros(N,E), 1, exp_idx, 1)   # [N, E]
    masked_scores = scores * exp_mask
    
    # 逐 expert 容量约束
    cap = int(gamma * (N * k) / E)
    _, keep_idx = masked_scores.topk(cap, dim=0)     # per-expert top-cap
    cap_mask = scatter(zeros(N,E), 0, keep_idx, 1)
    
    final_map = exp_mask * cap_mask
    return scores * final_map, final_map
```

术语一般如何实现？如何使用？
在 Megatron-LM MoE forward 中，Gate 之后、All-to-All dispatch 之前插入。本地 expert ID 从 EP group 获取。与 Token Drop 相比，Expanded Drop 在 Mixtral 提升 Avg 0.7 点（74.5 vs 73.8, Table 2）；多模态场景下 Image First 策略 + Expanded Drop 在 γ=0.5 时性能接近 baseline。

涉及论文标题：
- Capacity-Aware Inference Mitigating the Straggler Effect in Mixture of Experts

---

## Chain-of-Experts (CoE, Expert 链式处理架构)

术语解释
Chain-of-Experts (CoE) 是 Wang et al. (2025) 提出的一种新型 MoE 架构，将传统 MoE 层内 expert 的并行独立激活改为 C 步迭代顺序处理，引入 intra-layer expert communication。每步使用独立 Router 基于前一步的中间表示重新选择 expert，配合 inner residual connection 稳定多步训练。CoE 在相同 expert compute budget 下实现更高的 expert 组合多样性和更低的 validation loss。

术语是什么？
传统 MoE 层中，Router 一次性为 token x 计算 gating scores，TopK 选择 K 个 expert，所有 expert 并行独立处理同一输入 x，输出加权求和：$y = \sum g_i \cdot E_i(x)$。Expert 之间无交互。

CoE 修改为 C 步迭代：初始 $x^{(0)} = x$，每一步 $t = 1,...,C$ 执行：
$$x^{(t)} = \sum_{i=1}^{N} g_{t,i} \cdot E_i(x^{(t-1)}) + \mathbb{I}_r \cdot x^{(t-1)}$$

其中 $g_{t,i}$ 由第 t 步的独立 Router 计算（TopK 选择 K/C 个 expert），$\mathbb{I}_r = 1$ 为 inner residual connection。最终输出 $y = x^{(C)}$。

关键设计要素：
1. **Iteration-based Independent Routing**：每步有独立的 Router 参数（而非所有步骤共享同一 Router），使模型基于精炼后的中间表示动态调整路由
2. **Inner Residual Connection**：每一步将前一步的输出直接加到当前步输出，稳定多步训练（消融：inner=1.12 vs outer=1.21 vs init=1.18 loss）
3. **Sparsity Preservation**：每步仅选 K/C 个 expert，总计算量 = C×(K/C) = K，与标准 MoE 相同

从算法pipeline角度拆解术语：
```
# CoE Layer Forward (pseudocode)
def coe_forward(x, experts[1..N], routers[1..C], K, C):
    x_cur = x
    for t in range(1, C+1):
        logits_t = x_cur @ W_router[t]         # [d], N per step
        topk_scores, topk_idx = TopK(Softmax(logits_t), K/C)
        expert_out = sum(topk_scores[i] * experts[i](x_cur) for i in topk_idx)
        x_cur = expert_out + x_cur              # inner residual
    return x_cur
```

与标准 MoE 的关键差异：标准 MoE 为单步并行（所有 expert 看到同一 x），CoE 为多步迭代（后续 expert 的输入是 predecessors 精炼后的中间表示）。总 FLOPs 相同（K experts × 1 pass vs K/C experts × C passes）。组合空间从 C(N, 2K) 扩展到 C(N, K)^C（N=64, K=4, C=2 时 823×）。

术语一般如何实现？如何使用？
- 实现：PyTorch + veRL FSDP Trainer (https://github.com/volcengine/verl)，扩展 multi-round expert execution
- 模型配置（论文）：DeepSeek-V2-Lite 缩小版，544M params，4 layers，63 routed + 1 shared expert/layer，C=2, K=4/step
- 训练：AdamW，lr=3e-4，10% warmup，H100 单 GPU，<1 GPU hour/run
- 性能：val loss 1.20→1.12 (MetaMathQA)；C=2,L=4 ≈ MoE L=12 (-42% memory)；N=48,C=2 ≈ MoE N=64 (-17.6% memory)
- 局限：C>2 diminishing returns；单设备；sequential 减少单步 GEMM 并行度
- 开源：https://github.com/ZihanWang314/coe

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

---

## Iteration-based Independent Routing (基于迭代的独立路由)

术语解释
Iteration-based Independent Routing 是 CoE 架构的核心路由机制：在 C 步 expert 迭代处理的每一步，使用独立的 Router 参数 $W_{router}[t] \in \mathbb{R}^{d \times N}$（而非跨步共享同一 Router）。共享路由的消融变体（所有步骤复用同一 Router 和 gating）validation loss plateau 在 ~1.5，远差于独立路由的 1.12。

术语是什么？
形式化：第 t 步 $g_{t,i} = \text{TopK}(\text{Softmax}(e_{t,i}^\top x^{(t-1)}), K/C)$，每步 Router 参数独立。与 token-level 动态路由（Ada-K, DynMoE）的区别：后者是同一层内不同 token 使用不同 k 值但 Router 参数固定；CoE 是同一 token 在不同 iteration 使用不同 Router 参数。

从算法pipeline角度拆解术语：
```
# 独立Router（CoE默认，性能好）
for t in 1..C:
    logits = x_cur @ W_router[t]    # 每步不同W_router

# 共享Router（消融变体，性能差~1.5 loss）
logits = x @ W_router               # 仅第一步计算
topk_idx = TopK(Softmax(logits), K)
for t in 1..C:
    expert_out = sum(g[i] * experts[i](x_cur) for i in topk_idx)
    x_cur = expert_out + x_cur      # gating固定不变
```

术语一般如何实现？如何使用？
- 参数量增加 C 倍 Router 参数（通常可忽略，C=2 且 Router 参数占总参数比例极低）
- Co-activation 矩阵验证：不同 iteration 的 expert 选择集合高度非对称，证明路由决策确实随 iteration 变化
- 适用于需要 multi-step reasoning 的任务
- 是 CoE 中两个不可消融的组件之一（与 inner residual 并列）

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

---

## Inner Residual Connection in Iterative MoE (迭代式MoE中的内部残差连接)

术语解释
Inner Residual Connection 是 CoE 中用于稳定多步 expert 迭代训练的关键设计：在每一步 expert 输出后立即加入残差连接 $x^{(t)} = \text{expert\_out} + x^{(t-1)}$，而非只在最后一步之后加一次 outer residual（$y = x^{(C)} + x^{(0)}$）或每步都加初始输入（init residual: $x^{(t)} = ... + x^{(0)}$）。

术语是什么？
三种 residual 设计对比（CoE 论文）：
- **Inner Residual（默认，loss=1.12）**：$x^{(t)} = \text{expert\_out}^{(t)} + x^{(t-1)}$，每一步加入前一步输出
- **Outer Residual（loss=1.21）**：$y = x^{(C)} + x^{(0)}$，仅在最终输出加残差
- **Init Residual（loss=1.18）**：$x^{(t)} = \text{expert\_out}^{(t)} + x^{(0)}$，每步都加原始输入

从算法pipeline角度拆解术语：
```
# Inner Residual（CoE默认，最佳）
for t in 1..C:
    out = sum(g[t,i] * E_i(x_cur))
    x_cur = out + x_cur              # 残差来自上一步

# Outer Residual（消融，差）
for t in 1..C:
    out = sum(g[t,i] * E_i(x_cur))
    x_cur = out                      # 无中间残差
y = x_cur + x_0                      # 仅最后加一次

# Init Residual（消融，居中）
for t in 1..C:
    out = sum(g[t,i] * E_i(x_cur))
    x_cur = out + x_0                # 残差始终来自原始输入
```

术语一般如何实现？如何使用？
- Element-wise addition，计算开销可忽略
- 作用：为梯度提供从 $x^{(C)}$ 到 $x^{(t)}$ 的直接路径，稳定 credit assignment
- 与标准 Transformer residual connection 的区别：不是跨层（layer-to-layer），而是跨 iteration（iteration-to-iteration within single layer）

涉及论文标题：
- Chain-of-Experts: Unlocking the Communication Power of Mixture-of-Experts Models

---

## Expert Embedding for MoE Pruning (MoE剪枝中的专家嵌入)

术语解释
Expert Embedding for MoE Pruning 是将每个 MoE expert 映射为一个固定维度的特征向量的技术，通过将 expert 的功能行为编码为可比较的向量表示，使相似度计算和聚类成为可能。核心思想是用 expert 的实际输出（而非 router logits 或参数值）作为其"功能签名（functional signature）"。

术语是什么？
C-PRUNE 提出的 expert embedding 计算方法：φ(f_i) = E_{x~D}[1/K Σ_{k=1}^K f_i(x_k)] ∈ R^d。具体而言：在 task-specific calibration 数据集 D_calib 上对每个 sample 做 forward pass，对每个 expert f_i，取 K 个 token 的输出向量的平均值作为该 expert 的功能嵌入。维度 d = hidden_dim（如 DeepSeek-V2-Lite 的 d=2048）。这一定义基于如下观察：两个 expert 如果对相同输入产生相似的输出分布，则它们在功能上是冗余的，可以被合并或剪除。

与 router-based 方法（如 Seer Prune 使用 gate activation frequency）的区别：router-based 方法间接通过"哪些 expert 被选中"来判断重要性，而 expert embedding 直接衡量 expert 的"计算行为"——两个不同 expert 可能都被频繁激活（高 gate frequency），但计算输出几乎相同（功能冗余），router-based 方法无法检测这种情况。

从算法pipeline角度拆解术语。
```
# Expert Embedding Computation
Input: MoE model M, calibration dataset D_calib
       samples_per_expert K (e.g., K=100)

expert_embeddings = {}  # layer_id -> list of phi vectors

for each MoE layer l in range(L):
    phi_list = []
    for each expert f_i in layer l (i in 1..N):
        outputs = []
        for each batch x in D_calib:
            # Forward pass through this expert only
            h = f_i(x)  # expert FFN output: [batch, seq, d_model]
            # Average over K randomly selected token positions
            indices = random.sample(range(seq_len * batch), K)
            outputs.append(mean(h.view(-1, d_model)[indices], dim=0))
        
        # phi_i: average expert output over all calibration samples
        phi_i = mean(outputs, dim=0)  # shape: [d_model]
        phi_list.append(phi_i)
    
    expert_embeddings[l] = phi_list  # shape: [N, d_model]

# Use: pairwise cosine similarity -> affinity matrix
for layer l:
    phi = expert_embeddings[l]  # [N, d]
    for i, j in pairs:
        sim = phi[i] @ phi[j] / (norm(phi[i]) * norm(phi[j]))
        A[i,j] = sigmoid(alpha * sim)
```
注解：
- φ(f_i) 的维度 = d_model（expert 输出维度），与 expert 内部 FFN 维度无关
- K 是超参数，控制采样的 token 数量；论文中 K 的值"论文未明确说明"
- Calibration 数据集 D_calib 使用 task-specific samples
- expert embedding 仅需一次 offline computation，计算量 O(L × N × |D_calib| × K × d_ffn²)

术语一般如何实现？如何使用？
- **Offline Computation**：在剪枝前一次性计算所有 expert 的 embedding，存储为 [L, N, d] 张量
- **相似度计算**：基于 cosine similarity 或 Euclidean distance。Cosine similarity 对输出 scale 不敏感，更适合
- **聚类输入**：embedding 矩阵直接作为 hierarchical/k-means clustering 的输入特征
- **与其他方法的关系**：HC-SMoE (ICML 2025) 也使用 expert output-based similarity；Mosaic Pruning 使用 expert performance profile 作为 embedding
- 限制：(1) embedding 质量依赖 calibration 数据的 representativeness；(2) 大模型时 embedding 计算开销显著；(3) 不同 task domain 可能需要不同的 embedding

涉及论文标题：
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

---

## Hierarchical Agglomerative Clustering for MoE Expert Grouping (MoE专家的层次凝聚聚类)

术语解释
Hierarchical Agglomerative Clustering (HAC) 是一种自底向上的聚类算法，在 MoE expert grouping 场景中用于将功能相似的 expert 分组为 cluster。C-PRUNE、HC-SMoE 和 Mosaic Pruning 等近期工作均采用此方法，因其不需要预设 cluster 数量（可通过 pruning rate 自适应确定）且能保留 expert 间的层次相似结构。

术语是什么？
HAC 的基本流程：(1) 初始化：每个 expert 作为一个 singleton cluster；(2) 迭代合并：在每步选择 affinity 最高的两个 cluster 合并；(3) 终止：cluster 数量达到目标值。合并标准使用 Ward's linkage 或 average linkage。

C-PRUNE 中的 HAC 实现特点：
- 亲和矩阵 A_ij = σ(α · cos(φ(f_i), φ(f_j)))，基于 expert embedding 的 cosine similarity
- Cluster 合并后的新 affinity 通过 weighted average 更新（average linkage）
- 聚类在每层独立执行（layerwise），每层可有不同的 cluster 数量
- 聚类阈值 τ^(l) 自适应层深度：τ^(l) = mean_deviation + δ·σ^(l)

从算法pipeline角度拆解术语。
```
# HAC for MoE Expert Grouping (per-layer)
Input: expert_embeddings phi[N][d] for layer l
       target_clusters K (derived from pruning rate)

# Step 1: Build affinity matrix
A = zeros(N, N)
for i in 1..N, j in i+1..N:
    cos_sim = phi[i] @ phi[j] / (norm(phi[i]) * norm(phi[j]))
    A[i,j] = A[j,i] = sigmoid(alpha * cos_sim)

# Step 2: Initialize N singleton clusters
clusters = [{i} for i in range(N)]

# Step 3: Iteratively merge until reaching K clusters
while len(clusters) > K:
    (u, v) = argmax(A[u][v] for all u < v)
    new_cluster = clusters[u] ∪ clusters[v]
    clusters.remove(u), clusters.remove(v)
    clusters.append(new_cluster)
    # Update affinity (average linkage)
    for each remaining cluster c:
        A[new_idx, c] = (|C_u|*A[u,c] + |C_v|*A[v,c]) / (|C_u| + |C_v|)

# Step 4: Merge experts within each cluster
for cluster C_k in clusters:
    omega = softmax([gamma * A[i, center] for i in C_k])
    merged_params = sum(omega_i * expert_params[i])
```
注解：
- α: 相似度敏感度。α 越大，聚类越激进
- K = ceil(N × (1 - pruning_rate_layer))
- 时间复杂度 O(N² log N)，N 为 expert 数量（通常 ≤ 128）
- 各层的 HAC 可完全并行执行

术语一般如何实现？如何使用？
- **Linkage 选择**：C-PRUNE 使用 average linkage；HC-SMoE 验证 ward/average 均有效
- **与 K-means 的比较**：C-PRUNE Table 3 显示 HAC (avg 0.449) 显著优于 K-means (avg 0.405)，因 HAC 不假设球形 cluster
- **并行化**：各层 HAC 可完全并行执行
- **在 C-PRUNE 中的角色**：Phase 1 的核心，分组结果决定后续 global pruning 和 expert merging 质量
- 局限：(1) O(N² log N) 在超大 N 时可能成为瓶颈；(2) 聚类对 affinity matrix 构建方式敏感

涉及论文标题：
- Cluster-Driven Expert Pruning for Mixture-of-Experts Large Language Models

---

## Contrastive Representation for MoE (CoMoE) / InfoNCE Contrastive Loss in MoE

术语解释
Contrastive Representation for MoE (CoMoE) 是在 MoE-based PEFT 训练中引入基于 InfoNCE 的对比学习辅助损失，利用 top-k routing 将专家分为正负样本以促进专家专业化和模块化的方法。其核心创新在于将非激活专家（routing 中未被选中的 expert）从"浪费的计算资源"重新定义为"对比学习的负样本"。

术语是什么？
CoMoE 在标准 supervised fine-tuning 损失 L_CE 基础上添加对比辅助损失 L_con：

$$L_{\text{total}} = L_{CE} + \lambda \cdot L_{\text{con}}$$

其中对比损失基于 InfoNCE（Oord et al., 2018）：

$$L_{\text{con}} = \sum_{i=1}^{k} -\log\left(\frac{\exp(q_i \cdot k_i^+/\tau)}{\exp(q_i \cdot k_i^+/\tau) + \sum_{k_i^-} \exp(q_i \cdot k_i^-/\tau)}\right)$$

具体构造：
- **Query (anchor)**：从 k 个激活 expert 中随机选一个的输出表示 E_a(x) 作为 query q
- **Positive keys**：其余 k-1 个激活 expert 的输出表示（同属激活集的正样本）
- **Negative keys**：n-k 个非激活 expert 的输出表示（路由未选中的负样本）
- **Score function**：指数余弦相似度 h(x,e) = exp(q·e)/τ，τ 为温度超参数

CoMoE 可无缝集成到任何 top-k routing 的 MoE 架构中（LoRA-MoE、DoRA-MoE 等），不需要预训练，仅作为辅助目标添加。推理时无需对比损失——仅标准 top-k routing 前向。

从算法pipeline角度拆解术语：
CoMoE 的完整训练 pipeline（基于论文 Algorithm 1）：

```
# CoMoE Training Forward Pass (single token x)
输入: x, experts {E_j}_{j=1}^n, router g, top-k=2, τ, λ

# Step 1: Standard MoE forward
g(x) = softmax(W_g · x)                    # router logits [n]
T = topk(g(x), k)                           # activated expert indices
ŷ_i = g_i / Σ_{j∈T} g_j  if i∈T else 0     # renormalized weights
y' = W_0·x + Σ_{i∈T} ŷ_i · E_i(x)          # residual output

# Step 2: Expert representations (所有 expert 的中间输出)
e_j = E_j(x)  for j = 1..n                # 每个 expert 的输出表示 [D]

# Step 3: Contrastive loss construction
r = randint(1, k)                          # 随机 anchor 位置
a = T[r]                                   # anchor expert index
q = Normalize(e_a)                         # query [D]
P = {Normalize(e_{T[j]}) | j ≠ r}          # positive set, size k-1
N = {Normalize(e_j) | j ∉ T}              # negative set, size n-k

# Step 4: Similarity scores
s_pos = (q · P^T) / τ                      # [k-1]
s_neg = (q · N^T) / τ                      # [n-k]

# Step 5: InfoNCE loss
logits = [s_pos, s_neg]
L_con = -log( Σexp(s_pos) / (Σexp(s_pos) + Σexp(s_neg) + ε) )

# Step 6: Total loss
L_total = L_CE(y', y_true) + λ · L_con
L_total.backward()
```

CoMoE 还可以应用固定大小负采样策略：从 n-k 个非激活 expert 中随机采样固定数量作为负样本，将复杂度从 O(n) 降至 O(1)，训练时间与 expert 数量解耦（论文验证无性能损失）。

术语一般如何实现？如何使用？
- **实现框架**：基于 HuggingFace PEFT + transformers，在 MixLoRA / OMoE 等现有 MoE-LoRA 实现之上添加对比损失模块
- **关键超参数**：λ = 0.01（对比损失权重，通过消融确定），τ = 论文未明确说明（典型值 0.07-0.1），n = 4 experts，k = 2 (top-2 routing)
- **适用范围**：任何 top-k routing 的 MoE 架构（LoRA-MoE, DoRA-MoE 等），LLaMA-2 7B / Gemma 2B 已验证
- **训练成本**：n=4 时 3.5h on A6000（multi-task），固定采样策略下 O(1) 复杂度
- **推理效率**：对比损失仅用于训练，推理时无额外开销。CoMoE 推理延迟 3,789ms（vs MixLoRA 4,217ms，降低 10%）
- **论文未开源独立代码仓库**，但方法简单可基于标准 MoE-LoRA 实现复现
- **性能**：multi-task avg +1.3 accuracy (LLaMA-2 7B)，1.45% 可训练参数（vs MixLoRA 2.9%）

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

---

## Mutual Information Gap for Expert Routing in MoE (MI Gap / 专家路由互信息间隙)

术语解释
MI Gap (Mutual Information Gap) 是 CoMoE 提出的量化 MoE 中 expert 专业化和冗余程度的信息论度量：定义为输入 token x 与 top-k routing 下激活专家 M⁺ 之间的互信息，减去 x 与非激活专家 M⁻ 之间的互信息。

术语是什么？
给定输入 token x 和专家集合 M，在 top-k routing 下定义：

$$\Delta I = I_{\text{top-}k}(x, M^{+}) - I_{\neg \text{top-}k}(x, M^{-})$$

其中 $I(x; M) = \mathbb{E}_{x, \mathcal{M} \sim \mathcal{D}}[\log \frac{p(M|x)}{p(M)}]$ 为标准互信息。

MI Gap 的直观含义：
- **最大化 I(x; M⁺)**：促进激活 expert 对高度匹配的输入做出响应，鼓励专业化；同时作为信息瓶颈过滤无关噪声
- **最小化 I(x; M⁻)**：抑制非激活 expert 对无关输入的响应，防止多个 expert 学习相似表示

通过 Jensen 不等式，可得到 I(x; M) 的下界：
$$I(x; M) \ge \mathbb{E}_{x,e,\mathcal{M}}\left[\log \frac{p(e|x)}{p(e)}\right]$$

其中 e = E(x) 为 expert 的输出表示。

从算法pipeline角度拆解术语：
MI Gap 不直接计算，而是通过对比学习进行估计。CoMoE 的理论核心是 **InfoNCE 定理**：

**Theorem (InfoNCE)**：MI Gap ΔI = I_top-k(x, e⁺) - I_¬top-k(x, e⁻) 可通过对比目标下界估计：

$$\Delta I \ge \log(N) - \mathcal{L}_{\text{NCE}}$$

```
# MI Gap 估计流程
# 1. 收集样本
(x, e⁺) ~ D_top-k           # 激活 expert 的表示（正样本分布）
(x, e⁻) ~ D_¬top-k          # 非激活 expert 的表示（负样本分布）

# 2. 计算得分函数（信息密度比估计）
h₁(x,e⁺) ∝ p(e⁺|x)/p(e⁺)   # 激活 expert 的信息密度比
h₂(x,e⁻) ∝ p(e⁻|x)/p(e⁻)   # 非激活 expert 的信息密度比

# 3. InfoNCE 对比损失
L_NCE = -E[log( h₁(x,e⁺) / (h₁(x,e⁺) + Σ h₂(x,e⁻)) )]

# 4. MI Gap 下界
ΔI ≥ log(N) - L_NCE        # N 为负样本数，随 N 增大越紧
```

当 expert 专业化程度高时：每个 expert 仅对特定 token 子集产生高互信息（I_top-k 大），同时不同 expert 之间知识冗余最小化（I_¬top-k 趋近于 0），MI Gap 达到最大值。

术语一般如何实现？如何使用？
- **实践中的近似**：不分别估计 I_top-k 和 I_¬top-k，而是统一对比目标直接估计 ΔI。通过将 h₁ 和 h₂ 合并为单一得分函数 h(x,e) = exp(E⁺(x)·e)/τ，使正负样本形成双向样本对
- **评分函数选择**：指数余弦相似度是最常用的选择，温度 τ 控制对比分布的锐度
- **理论保证**：InfoNCE 提供的是 MI Gap 的紧下界（tight lower bound），随负样本数 N 增加而收紧
- **与 Load Balance 的区别**：Load balance loss 强制 expert 使用频率均匀（量的平衡），MI Gap 强制 expert 功能差异化（质的专业化），两者互补但 CoMoE 实验表明仅 MI Gap 即可自然产生负载均衡

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

---

## MoE-based LoRA (LoRA-MoE / Mixture of LoRA Experts / MoLE)

术语解释
MoE-based LoRA（又称 LoRA-MoE、Mixture of LoRA Experts / MoLE）是将 Mixture-of-Experts 架构与 Low-Rank Adaptation (LoRA) 结合的参数高效微调方法：将单一 LoRA 模块替换为 n 个并行的 LoRA expert，通过 Router 稀疏激活部分 expert 来处理异构数据。

术语是什么？
标准 LoRA 对预训练权重 W₀ 引入低秩矩阵 A, B：
$$y' = W_0 x + BA x$$

LoRA-MoE 将 BA 模块替换为 n 个 expert {E_i = B_i A_i}，通过 Router g(x; G) 控制激活：
$$y' = W_0 x + \sum_{i=1}^{n} g_i(x; G) \cdot E_i(x)$$

其中 g_i(x; G) 经 top-k routing 稀疏化后，仅 top-k 个 expert 的权重非零。

LoRA-MoE 的核心优势：
1. **容量-计算权衡**：n 个 expert 提供 n× 参数容量，但每次仅激活 k 个（通常 k=2），计算量接近 k/n 倍
2. **异构数据处理**：不同 expert 可隐式专门化于不同任务/数据分布，适合 multi-task fine-tuning
3. **与全量 MoE 的区别**：LoRA-MoE 的 expert 是轻量低秩矩阵（r × (d_in + d_out) 参数），而非完整 FFN 层

从算法pipeline角度拆解术语：
```
# LoRA-MoE Layer Forward (以 CoMoE 配置: n=4, k=2, r=16)
class LoRAMoELayer:
    def __init__(self, W_0, n_experts=4, rank=16):
        self.W_0 = W_0                    # frozen pretrained weights
        self.experts = []                 # n LoRA experts
        for i in range(n_experts):
            A_i = nn.Linear(d_in, rank, bias=False)
            B_i = nn.Linear(rank, d_out, bias=False)
            self.experts.append((A_i, B_i))
        self.router = nn.Linear(d_in, n_experts)  # gating network
    
    def forward(self, x):
        # 1. Router: 每个 token 对所有 expert 的 affinity score
        gate_logits = self.router(x)      # [batch, seq, n]
        
        # 2. Top-k selection
        topk_weights, topk_idx = torch.topk(
            torch.softmax(gate_logits, dim=-1), k=2, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        # 3. Expert computation (sparse)
        output = self.W_0 @ x             # frozen path
        for i in range(self.n_experts):
            mask = (topk_idx == i).any(dim=-1)  # tokens routed to expert i
            if mask.any():
                A_i, B_i = self.experts[i]
                expert_out = B_i(A_i(x[mask]))  # LoRA: B·A·x
                weight = topk_weights[mask][topk_idx[mask] == i]
                output[mask] += weight.unsqueeze(-1) * expert_out
        
        # 4. (CoMoE 额外) 收集所有 expert 表示用于对比损失
        expert_reprs = [B_i(A_i(x)) for i in range(n_experts)]
        return output, expert_reprs, topk_idx
```

术语一般如何实现？如何使用？
- **实现框架**：HuggingFace PEFT（peft library）提供 MoE-LoRA 支持；MixLoRA (https://github.com/TUDB-LAB/MixLoRA) 是最常用的开源实现之一
- **关键变体**：
  - **MixLoRA** (Li et al., 2024)：resource-efficient sparse MoE，top-k routing + load balance loss
  - **MOELoRA** (Liu et al., 2023)：面向 multi-task medical applications
  - **MiLoRA** (Zhang et al., 2024)：prompt-aware routing 降低延迟
  - **OMoE** (Feng et al., 2025)：orthogonal fine-tuning 强制 expert 多样性
  - **LoRAMoE** (Dou et al., 2024)：token-based routing 缓解知识遗忘
  - **HydraLoRA** (Tian et al., 2024)：非对称 LoRA 结构
- **典型配置**：LLaMA-2 7B 上 r=16, n=4~8, k=2，应用在 Q/K/V/O/Up/Down/Gate 投影层
- **可训练参数占比**：约 0.7%~3%（取决于 n, r, 应用层数）
- **局限**：expert 功能冗余（缺乏专业化约束导致 expert 学到相似功能）、负载不均（部分 expert 过度使用/闲置）

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning

---

## Expert Specialization and Modularization in MoE (MoE中的专家专业化与模块化)

术语解释
Expert Specialization（专家专业化）和 Modularization（模块化）是 MoE 架构的理想属性：每个 expert 应专注于不同的表示子空间和语义技能，不同 expert 之间功能互补、知识冗余最小化，从而协同增强模型的整体表示能力。

术语是什么？
在 MoE 中，理想情况下：
- **专业化 (Specialization)**：每个 expert 对特定类型的输入 token 产生高响应（高互信息），对不同类型 token 产生低响应。类似"领域专家"，每个 expert 掌握独特的知识子集
- **模块化 (Modularization)**：不同 expert 之间的功能边界清晰，知识重叠最小化。类似"模块化设计"，每个模块独立负责一部分功能

专业化与模块化的关系：模块化是专业化的空间结构表现——只有当 expert 的表示在高维空间中分散且互不重叠时，每个 expert 才能实现真正的专业化。

从算法pipeline角度拆解术语：
CoMoE 通过对比学习强制实现专业化与模块化。其核心机制：

```
# Expert specialization 的量化与促进
# 以 4 个 expert 处理 3 个 task 为例

# 无专业化约束（vanilla MoE）：
# 所有 task 的 token 都倾向于激活 expert 1 和 2
Task_A: expert_1=45%, expert_2=40%, expert_3=10%, expert_4=5%
Task_B: expert_1=42%, expert_2=43%, expert_3=8%,  expert_4=7%
Task_C: expert_1=48%, expert_2=38%, expert_3=9%,  expert_4=5%
# → 专家功能重叠严重，容量利用不足

# CoMoE 施加专业化约束后：
# 不同 task 自然分配到不同 expert 组合
Task_A (ARC-c):  expert_1=52%, expert_3=46%, expert_2=1%,  expert_4=1%
Task_B (BoolQ):  expert_1=40%, expert_4=55%, expert_2=3%,  expert_3=2%
Task_C (OBQA):   expert_2=35%, expert_3=60%, expert_1=3%,  expert_4=2%
# → 每 task 有独特的 expert 组合，"协作专业化"自然涌现
```

专业化通过对比损失实现：
1. **正信号（拉近）**：同一输入激活的 expert 被鼓励产生相似的输出表示（因为它们共同解决同一任务）→ 形成"协作组"
2. **负信号（推远）**：非激活 expert 被推离当前输入的表示空间（因为它们不应参与此任务）→ 减少冗余

可视化验证（Figure 5, CoMoE）：加入 contrastive loss 前，所有 expert 表示在降维空间中高度重叠（无专业化）；加入后，expert 表示显著分散（模块化形成）。

术语一般如何实现？如何使用？
- **量化方法**：
  - MI Gap（CoMoE）：ΔI = I(x; M⁺) - I(x; M⁻)，通过对比损失最大化
  - Orthogonality（OMoE）：对 expert 权重矩阵施加正交约束 ||E_i^T E_j - I||
  - Load balance（MixLoRA, LoRAMoE）：通过 auxiliary loss 强制 expert 使用频率均匀
- **CoMoE 的实现优势**：不需要显式 load balance loss（无额外超参数调优），负载均衡作为专业化的"副作用"自然涌现（Figure 4）
- **评估方法**：
  1. Expert activation 分布可视化：不同 task 下每个 expert 的激活频率
  2. Expert representation 降维可视化（t-SNE/PCA）：检查 expert 输出表示的空间分散度
  3. Multi-task 性能：专业化程度越高，multi-task vs single-task 性能差距越小
- **局限**：(1) 过强的专业化约束（大 λ）反而损害性能（Figure 3, CoMoE: λ>0.1 时性能显著退化）；(2) 专业化效果在复杂 multi-task 场景中更显著，简单 single-task 中收益有限

涉及论文标题：
- CoMoE: Contrastive Representation for Mixture-of-Experts in Parameter-Efficient Fine-tuning
- Aria An Open Multimodal Native Mixture-of-Experts Model（Section 4.2 可视化 multimodal MoE 中的 modality-specific expert specialization：对 natural image/video/PDF 三种视觉域计算每个 expert 的 $R_v/R_t$ 比率，发现多层的单个 expert 对所有三种视觉域均 specialized；specialization 在 modality-generic architecture 下自然涌现）
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models

**Global-Batch LBL 如何促进 Domain Specialization（Demons in the Detail, 2025）**：

Qiu et al. (2025) 发现 micro-batch LBL 是阻碍现有 MoE 模型展现 domain-level expert specialization 的关键因素：
- Micro-batch LBL 下：专家选择频率在不同 domain 间无明显差异，同一 domain 内各专家选择频率近似均匀（最高 <0.15），仅能观察到 token-level routing pattern，无法形成 domain-level specialization（与 Mixtral、OpenMoE 的观察一致）。
- Global-batch LBL 下：出现显著的高频专家——如 SFT-Math domain 中多专家选择频率 >0.2，形成可解释的 domain specialization pattern。相近 domain（如 SFT-ZH, ZH-Law, ZH-Literature）共享高频专家（dashed box），而中文 domain 与 SFT-Code 的高频专家几乎不重叠。通用 content（SFT-EN）中个体专家高激活实例较少。
- TopK score sum 分析：Global-batch LBL 下各层 topK sum 更高。因为 LBL 和 z-loss 鼓励路由分数均匀化，只有 language modeling loss 鼓励路由分数集中——更高的 topK sum 表明 routing 更与 language modeling 任务对齐。Expert specialization 促进专家分数集中。Micro-batch LBL 下 topK sum 较低且跨 domain 无差异，对应现有工作中 MoE routing 的不确定性（Wu et al., 2024）。
- 结论：micro-batch LBL 的"序列级均匀"约束本质上是 anti-specialization 的正则化——超过一定约束强度后，LBL 与 language modeling loss 冲突，不仅损害性能，更阻止专家专业化。

---

## Weight-Decomposed Experts (WD Experts)

术语解释
Weight-Decomposed Experts 是将 MoE 中每个 expert 的 FFN 权重矩阵替换为低秩分解（Low-Rank Decomposition），以在保持模型质量的同时减少总参数量的一种技术。由 CoSMoEs (Huber et al., 2025) 提出，应用于端侧设备（on-device）的 MoE 预训练。

术语是什么？
Weight-Decomposed Experts 的核心思想：每个 expert 旨在"专门化"处理约 1/E 的 token 子集（E = 总 expert 数），因此不需要 full-rank 的 FFN 权重矩阵来捕获其专业知识。将每个 expert 的权重矩阵 M ∈ R^{n×m} 替换为两个低秩矩阵的乘积：

$$M_{n \times m} \approx L_{n \times r} \times R_{r \times m}$$

其中 r ≪ n 且 r ≪ m。CoSMoEs 论文采用 r = n/2（half hidden dimension）作为最优 trade-off。每个 expert 的三个 FFN 子矩阵（gate_proj, up_proj, down_proj in SwiGLU）均被低秩分解替代。

WD 与 LoRA (Hu et al., 2021) 相似但应用于预训练阶段：LoRA 在冻结的预训练权重上添加低秩适配器（W + ΔW = W + BA），WD 从零开始以低秩形式训练 expert 的权重。

从算法pipeline角度拆解术语。
WD Expert 在 MoE layer 中的前向计算（以 SwiGLU FFN 为例）：

```
# 标准 Expert FFN:
# Expert_i(x) = W_down_i @ (SiLU(W_gate_i @ x) * W_up_i @ x)

# WD Expert FFN (三个权重矩阵各分解为 L×R):
def wd_expert_forward(x, expert_idx):           # x ∈ R^{seq × d_model}
    # Gate: W_gate ∈ R^{d_ff × d_model} → L_gate ∈ R^{d_ff × r}, R_gate ∈ R^{r × d_model}
    h_gate = SiLU(x @ R_gate[expert_idx].T @ L_gate[expert_idx].T)
    # Up: W_up ∈ R^{d_ff × d_model} → L_up ∈ R^{d_ff × r}, R_up ∈ R^{r × d_model}
    h_up = x @ R_up[expert_idx].T @ L_up[expert_idx].T
    # Gated activation
    h = h_gate * h_up                              # [seq, d_ff]
    # Down: W_down ∈ R^{d_model × d_ff} → L_down ∈ R^{d_model × r}, R_down ∈ R^{r × d_ff}
    out = h @ R_down[expert_idx].T @ L_down[expert_idx].T  # [seq, d_model]
    return out
```

参数量对比（Phone-sized WD MoE, d_model=1600, d_ff≈6400, r=800）：
- 标准 expert: 3 × 6400 × 1600 = 30.7M params per expert
- WD expert: 3 × (6400×800 + 800×1600) = 3 × 6.4M = 19.2M params per expert
- 节省约 37% per expert

不同 r 的 trade-off（CoSMoEs 初步实验）：
- r = n (full rank): baseline 性能，无参数节省
- r = n/2: 最佳 trade-off，参数显著减少 + 性能提升（WD > standard MoE by 1.1%）
- r = n/4: 参数更少但性能下降，表达能力不足
- r = n/8: 参数极少但性能大幅退化

术语一般如何实现？如何使用？
- 实现：在 HuggingFace Transformers 中，修改 expert 的 `nn.Linear` 层为两个串联的 `nn.Linear(d_model, r)` + `nn.Linear(r, d_ff)`
- 训练：WD Experts 从零开始预训练，需要端到端训练 L 和 R 矩阵
- 适用场景：端侧设备部署（内存受限），减少每个 expert 的存储和加载开销；expert 数量较多时（E≥4）积累的参数节省显著
- 与 BlES loss 正交组合：WD 减少每个 expert 大小（Memory），BlES 减少 expert 切换频率（Latency）
- 局限：低秩分解增加矩阵乘法次数（3→6），但 batch=1 时小矩阵乘法更 cache-friendly；r 的选择需要针对模型规模调优

涉及论文标题：
- CoSMoEs Compact Sparse Mixture of Experts

---

## Block-wise Expert Selection (BlES) Loss

术语解释
Block-wise Expert Selection (BlES) Loss 是 CoSMoEs (Huber et al., 2025) 提出的训练阶段辅助损失函数，鼓励 MoE 模型在连续 token 上选择相同的 expert 集合，从而在端侧推理时减少 expert offloading 次数，降低延迟。

术语是什么？
BlES Loss 的核心洞察：标准 MoE 训练中，router 对每个 token 独立选择 expert，导致连续 token 间 expert 频繁切换。在端侧 offloading 场景下（GPU 仅保留 active experts），每次 expert 切换触发 CPU↔GPU 数据传输，引入显著延迟（4-20×）。BlES 通过在训练时惩罚 expert 切换，使模型学会在连续 token 保持一致的 expert 选择。

损失函数由两部分乘积构成：
1. **Hard Expert Replacement (H_norm)**：统计连续 token 间 top-k expert selection 的实际变化次数（不可微分）
2. **Soft Expert Selection Difference (L_norm)**：计算连续 token 间 softmax 路由概率的 L1 变化（可微分）

从算法pipeline角度拆解术语。
```
# Input: R ∈ R^{B×T×E} (router logits), τ: temperature, K=2

# Soft routing weights (differentiable)
W = softmax(τ * R)                                # [B, T, E]

# Hard expert selection (non-differentiable, for H computation only)
S = top_k(W, K)                                    # [B, T, K]

# Hard expert replacement count
# For each expert e, count transitions between active/inactive
H_e = Σ_{b=1}^{B} Σ_{t=1}^{T-1} |(S[b,t+1]==e) - (S[b,t]==e)|
H = Σ_{e=1}^{E} H_e                               # each switch counted 2x
H_norm = floor(H/2) / (B * K * (T-1))              # ∈ [0, 1]

# Soft expert selection difference (differentiable)
L = Σ_{b} Σ_{t=1}^{T-1} Σ_{e=1}^{E} |W[b,t+1,e] - W[b,t,e]|
L_norm = L / (B * T)

# BlES loss = product of hard and soft signals
loss_BlES = H_norm * L_norm
```

需要配合 **Sequence-Level Load Balancing**：标准 load balancing 在 model level 计算（所有层总和），可被 exploit。例如 2 experts, 2 layers：layer 0 只用 expert 0、layer 1 只用 expert 1 → model level 50:50 完美均衡 + minimal BlES（每层内无切换）。改为 sequence level 后消除此 exploit。

```
L_total = L_NLL + α * L_load_balancing(seq_level) + β * L_BlES
```

术语一般如何实现？如何使用？
- 实现：在标准 MoE 训练循环的 loss 计算中添加 BlES loss 项，需要访问每层的 routing weights 计算连续 token 间差异
- 效果：Expert Replacement Ratio 43.82% → 6.55%（6.7× reduction），生成速度 15.02 → 23.10 tok/s（1.54× speedup）
- 质量 trade-off：Phone-sized -0.43% avg，Wearable-sized -1.87% avg（小模型更易受影响）
- 与推理时优化正交：BlES 是训练时优化，可与 MoE-Infinity、EdgeMoE 等推理时预取/缓存方法叠加
- 局限：对极小模型质量影响更明显；效果与 batch size 相关；必须与 sequence-level load balancing 配合

涉及论文标题：
- CoSMoEs Compact Sparse Mixture of Experts

## Collaboration-of-Experts (CoE)

术语解释
Collaboration-of-Experts (CoE) 是一种多专家模型协作推理范式：将多个独立训练的专家模型（experts）通过路由模块（Routing Module）集成，协同完成推理任务。每个 expert 是独立训练的模型（可以架构不同），Routing Module 决定输入应由哪个/哪些 expert 处理以及 expert 间的调用顺序。

术语是什么？
CoE 与 MoE（Mixture of Experts）是两类不同的多专家模型范式：
1. **训练方式**：CoE 的 experts 各自独立训练/微调，Routing Module 可手动配置；MoE 的 experts 和 router 需联合训练。
2. **路由可分析性**：CoE 的路由规则可离线分析——用户可预定义路由规则，从而提前计算每个 expert 的使用概率和依赖关系。MoE 的 router 在推理时动态输出，无法提前获知。
3. **专家管理**：CoE 可独立增删 expert，更灵活；MoE 的 experts 在训练时固定。
4. **精度优势**：CoE 通过集成多个专业化 expert 可达比单一模型更高的精度。例如电路板检测从单模型 92%→CoE 99.9%。

CoE 推理流程（以电路板缺陷检测为例）：
```
输入: 组件图像 I
1. Routing Module(I) → 选择分类 expert E_class
2. E_class(I) → 输出: (缺陷类型, 是否需要进一步检测)
3. if 需要进一步检测:
     Routing(output) → 选择目标检测 expert E_detect
4.   E_detect(I) → 最终结果: (对齐点, 焊接方向)
```

从算法pipeline角度拆解术语：
CoE 的算法 pipeline 特点：
- 与 MoE token-level routing（每个 token 独立选 expert）不同，CoE 是 request-level 路由——一个请求整体被路由到一系列 expert 组成的 pipeline
- Expert 之间存在依赖链（后续 expert 依赖前置 expert 的输出）
- 这种依赖关系可在推理前通过路由规则分析获取，为系统优化提供了 MoE 不具备的先验信息

术语一般如何实现？如何使用？
- 实现框架：PyTorch（CoServe）、SambaNova SN40L 数据流架构（Samba-CoE）
- Routing Module：用户预定义规则（如组件类型→expert 映射）或独立训练
- Expert 模型：可使用多种架构（ResNet、YOLO、Llama 等）
- 典型应用：电路板缺陷检测（300+ experts, 60GB）、Qihoo 360 CoE（多领域 LLM 协作）
- 局限：论文验证仅限智能制造成本场景；CoE 需提供 routing module 和 expert models

涉及论文标题：
- CoServe: Efficient Collaboration-of-Experts (CoE) Model Inference with Limited Memory

## Speculative Decoding in MoE Offloading（MoE Offloading 中的投机解码）

术语是什么？通过联网搜索让回答具体和精准。
Speculative Decoding in MoE Offloading 是 SpecMoEOff 提出的首个将 speculative decoding 应用于 MoE 模型 CPU-GPU offloading 推理场景的系统方法。核心动机：MoE offloading 中，CPU-GPU 的 expert weight 传输（I/O bottleneck）和 MoE 的稀疏激活导致 GPU 利用率极低——batch=1 方案仅 0.76% GPU 利用率，throughput-oriented 方案也仅 3.13%。SpecMoEOff 通过 speculative decoding 增大每次 target model forward 处理的 token 数（从 1 token 变为 k+1 tokens），从而在相同的 expert loading 开销下完成更多计算，隐藏 offloading 延迟。

在该场景中，speculative decoding 使用的 draft model 为 EAGLE（利用 target model 的 hidden state 作为输入，仅含 1 层 attention + FFN，<2GB 参数）。关键适配：(1) draft model KV cache 也需要 offloading（在 large batch 下超过 GPU HBM 容量）；(2) verification 阶段的 chunked attention 在 CPU 执行（避免 CPU→GPU 传输 KV cache）；(3) hyperparameter 需要自动优化（draft length k 与 batch size/micro-batch size 交互影响性能）。

从算法pipeline角度拆解术语：
```
# SpecMoEOff: 单次 Speculative Decoding Iteration
输入: prefix tokens x_1:l, draft model M_d (EAGLE), target model M_t (Mixtral-8x7B)

# Step 1: Draft Phase (EAGLE)
# M_d 参数全在 GPU HBM (<2GB), KV cache: GPU Part + CPU Part
for i = 1 to k:  # k 由 Hyperparameter Optimizer 确定
    h = target_model_hidden_state  # 从 target model 获取 feature
    x_tilde_{l+i} = M_d.generate(h, prefix=x_1:l+i-1)
    # GPU Part: attention+FFN 均在 GPU
    # CPU Part: attention on CPU, FFN on GPU

# Step 2: Target Model Verification
# Q ∈ R^{k×d}, K,V ∈ R^{(l+k)×d} from CPU DRAM
extended_x = concat(x_1:l, x_tilde_{l+1:l+k})
# CPU Chunked Attention (Intel MKL):
scores = Q @ K^T / sqrt(d) + mask  # mask 仅存储 n×n draft 部分
attn_out = softmax(scores) @ V
# GPU MoE: expert weights CPU→GPU HBM → FFN
p_1:k = M_t.forward(extended_x)

# Step 3: Probabilistic Acceptance
n_accepted = verify_and_accept(p_1:k, x_tilde, q_1:k)
# n_accepted = a(k), acceptance rate function from profiling
```

术语一般如何实现？如何使用？
SpecMoEOff 基于 SGLang 框架实现，采纳 MoE-Lightning 的 FFN/expert cache 设计，增加 20,000+ 行 Python/C++/CUDA。Draft model 使用 EAGLE 框架（利用 target model hidden state，仅 1 层 attention + FFN）。在 MoE offloading 场景下与标准 speculative decoding 的关键区别：(1) target model verification 使用 CPU chunked attention（而非 GPU）；(2) draft model KV cache 也需要 offloading；(3) 需要 hyperparameter optimizer 自动确定最优 draft length k。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

## Roofline Analysis for MoE Offloading（MoE Offloading 的 Roofline 分析）

术语是什么？通过联网搜索让回答具体和精准。
Roofline Analysis for MoE Offloading 是 SpecMoEOff 提出的对 MoE 模型 CPU-GPU offloading 推理的分层 Roofline 模型分析。该分析将 MoE 推理中的两种主要计算——MoE layer 和 Attention layer——分别映射到不同的硬件资源组合上：(1) MoE layer 的计算在 GPU，内存访问涉及 GPU HBM 和 CPU-to-GPU transfer（两个不同的带宽约束）；(2) Attention layer 的计算在 CPU，内存访问为 CPU DRAM。通过对比各算子的 arithmetic intensity（Operational Intensity = FLOPs / Bytes accessed）与硬件 peak compute/memory bandwidth，可视化性能瓶颈。

分层 Roofline 模型（Hierarchical Roofline Models）的核心发现：
- MoE layer (GPU compute + GPU HBM): arithmetic intensity 位于 compute-bound 区域，但仅 3.13% GPU peak 利用率
- MoE layer (GPU compute + CPU-GPU transfer): 位于 memory-bound 区域，transfer bandwidth 近乎完全利用
- Attention layer (CPU compute + CPU DRAM): memory-bound, CPU memory bandwidth 是瓶颈
- 结论：CPU-GPU transfer 是 MoE layer 的主要瓶颈，CPU memory bandwidth 是 Attention 的主要瓶颈

从算法pipeline角度拆解术语：
```
# Roofline 成本分析 (Table 1)

# MoE Layer:
computation = 3 × n_activate × b × e   # 3×矩阵乘法, e = h × h_i
memory_access_GpuHBM = n_expert × e      # 大batch: 所有expert可能激活
memory_transfer_CPU_GPU = n_activate × e × r_miss  # r_miss = cache miss rate
# Operational Intensity (GPU HBM axis):
OI = 3 × n_activate × b / n_expert      # b 小时 OI 低 → memory-bound

# Attention Layer:
computation = 2 × b × s × h             # GEMV operations
memory_access = 2 × b × s × h / g       # g = GQA group size factor
# CPU-based attention 优于 GPU-based（因 B_CPU >> B_CPU-GPU）:
# 条件: B_CPU > B_CPU-GPU → attention on CPU 更优 ✓

# Large batch is better when:
# b ≥ n_expert / (n_activate × r_miss)
```

术语一般如何实现？如何使用？
SpecMoEOff 中的 Roofline 分析用于：(1) 确定 MoE layer 和 Attention layer 的最优执行位置（GPU/CPU）；(2) 计算大 batch 与小 batch 方案的最优切换条件；(3) 论证 speculative decoding 的必要性——增大 b 和 k 提升 operational intensity。分析的硬件参数来自实际硬件配置（Table 1: A30/4090D 的 peak TFLOPS、HBM bandwidth、CPU-GPU bandwidth）。

涉及论文标题：
- Accelerating Mixture-of-Experts Inference by Hiding Offloading Latency with Speculative Decoding

## Fine-Grained MoE (细粒度专家混合)

术语解释
Fine-Grained MoE 是 DeepSeekMoE (Dai et al. 2024) 提出的 MoE 架构变体：在总参数量不变的前提下，将传统 MoE 的少量大 expert 拆分为大量小 expert（更多 expert 数量、更小的每个 expert 参数），并增大 top-k 值以激活更多 expert。核心动机是提升 expert specialization——更细粒度的 expert 可以更精准地学习不同知识子域。

术语是什么？
传统 MoE：每层 8 个 expert，每个 expert FFN hidden dim ~4h，top-2 激活 → 粗粒度，expert 难以充分 specialization。
Fine-Grained MoE：每层 64 个 expert，每个 expert FFN hidden dim ~h/4，top-8 激活 → 细粒度，每个 expert 专注更窄的知识域。

关键数量关系：总参数量 ≈ N_experts × d_expert_size。Fine-Grained 通过增大 N、减小 d 保持总参数量不变，但 expert 的专业化能力因更细的分工而增强。已被 DeepSeek-V2、Qwen2-57B-A14B 等模型采用。

然而，Fine-Grained MoE 面临严重的 All-to-All 通信瓶颈——需要激活更多 expert（top_k 更大），导致 All-to-All 通信量与 top_k 线性增长。BigMac 论文 Table 1 显示：top_k=8 时 All-to-All 占训练时间 91.8%、推理时间 90.6%。

从算法pipeline角度拆解术语：
给定同一个 MoE 层，fine-grained vs vanilla 的对比如下：

```
# Vanilla MoE (粗粒度，如 Mixtral)
E = 8           # 8 个 expert
top_k = 2       # 每 token 激活 2 个
d_ff = 5632     # 每个 expert 的 FFN intermediate dim

# Fine-Grained MoE (细粒度，如 DeepSeekMoE/BigMac)
E = 64          # 64 个 expert
top_k = 8       # 每 token 激活 8 个
d_ff = 704      # 每个 expert 的 FFN intermediate dim (约 1/8)

# Fine-Grained MoE forward:
x = input_token               # [h]
logits = x @ W_gate           # [64]
probs = TopK(SoftMax(logits), k=8)
output = sum(probs[i] * Expert_i(x) for i in selected_experts)
# 8 个 expert 各执行 FFN(x) = W_down · σ(W_gate · x) ⊙ (W_up · x)
```

术语一般如何实现？如何使用？
- DeepSeekMoE: 2 shared experts（always active）+ 64 routed experts（fine-grained），top-6 routed
- DeepSeek-V2: fine-grained 扩展到 160 experts，routed top-6 + shared top-2
- Qwen2-57B-A14B: 继承 fine-grained MoE 设计
- BigMac: 在 fine-grained MoE 基础上叠加 DCCA 通信优化

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs（Ling-Lite 16.8B/2.75B active 和 Ling-Plus 290B/28.8B active 均采用 Fine-Grained Experts 策略：扩展 expert 数量同时等比缩小每个 expert 的 intermediate size，搭配 Shared Expert 解决单个 expert 在有限容量下难以同时发展通用能力和专业能力的问题。Ling 引入 Stochastic Routing Warmup 防止 fine-grained 下的训练初期 router 崩溃）
- Continual Pre-training of MoEs How robust is your router（Granular MoE CPT 研究：E=31 routed + 1 shared, K=3 active per token, FFN intermediate=704（dense 的 1/4）。Granular MoE 在 CPT 中显著优于 Switch MoE (E=8, K=1)：更低 validation loss、更好 benchmark、更低的 MRI。早期层（0-6）MRI 在 Granular MoE 中远低于 Switch MoE，表明细粒度架构有利于 CPT 场景下的负载均衡稳定性）

---

## DCCA (Descend-Communicate-Communicate-Ascend)

术语解释
DCCA 是 BigMac 论文提出的低维通信策略，将 fine-grained MoE 的执行顺序从 CDAC（先通信后降维）重新排列为先降维后通信，使 All-to-All 通信在压缩后的低维空间进行，从而大幅减少通信量。DCCA 是 BigMac 的核心创新。

术语是什么？
DCCA 将一个 MoE 层的执行拆分为四个连续阶段：

1. **Descend (降维)**：$x' = xW'_{\downarrow}$，将输入 token x ∈ R^h 通过 descending projection $W'_{\downarrow} \in \mathbb{R}^{h \times (r \cdot h)}$ 压缩到低维空间 x' ∈ R^{r·h}。论文设 downscaling factor r = 0.25。

2. **Communicate (All-to-All Dispatch)**：将压缩后的 token x'（维度 r·h）通过 All-to-All 分发到各 expert 所在设备。通信量 = $2 \times top\_k \times \frac{ep-1}{ep} \times b \times s \times (r \cdot h)$，是 CDAC 的 r 倍（-75%）。

3. **Communicate (All-to-All Combine)** & Expert Computation：各 expert 执行 BigMac Expert 的计算后，All-to-All 将输出汇集回源设备（同样在低维 r·h 进行）。

4. **Ascend (升维)**：$y = y'W'_{\uparrow}$, 将 combined output y' ∈ R^{r·h} 通过 ascending projection $W'_{\uparrow} \in \mathbb{R}^{(r \cdot h) \times h}$ 恢复到原始维度 h。

Gate 路由仍使用降维前的 full-dimension x（而非压缩后的 x'），以保证路由精度。

关键：DCCA 仅增加了两个 projection 矩阵（$W'_{\downarrow}$ 和 $W'_{\uparrow}$），仅带来 +1.35% 参数和 +4.54% FLOPs 的额外开销，却换来 -75% 的通信量削减。

从算法pipeline角度拆解术语：

```
# DCCA MoE layer forward (BigMac)
def dcca_moe_forward(x):              # x: [batch, seq, h]
    # Step 1: Gating at FULL dimension
    gate = SoftMax(x @ W_gate)
    topk_w, topk_idx = TopK(gate, k=top_k)

    # Step 2: DESCEND — compress to r·h
    x_low = x @ W_down_prime           # [batch, seq, h] → [batch, seq, r·h]

    # Step 3: All-to-All DISPATCH (low-dim: r·h)
    dispatched = alltoall_scatter(x_low, topk_idx)

    # Step 4: Expert computation + All-to-All COMBINE (low-dim: r·h)
    # Each expert E_i: σ(x @ W_i↑) @ W_i↓  (先升后降)
    combined = expert_compute_and_alltoall_gather(dispatched)

    # Step 5: ASCEND — restore to h
    y = combined @ W_up_prime          # [batch, seq, r·h] → [batch, seq, h]
    return y
```

给定 GPT3-XL (h=2048, r=0.25, ep=32, top_k=8)：
- CDAC: All-to-All = 1,488 GB, 占总时间 91.8%
- DCCA: All-to-All = 372 GB (-75%), FLOPs +4.54%

术语一般如何实现？如何使用？
- 实现为模型结构变更（添加 projection 层并调整 expert 内部结构），无需修改通信框架
- 与 Megatron、Tutel、DeepSpeed-Inference 等现有框架兼容
- r 是超参数（论文设 0.25），需根据通信/计算 trade-off 调整
- 适合 expert parallelism degree 较大的大规模 MoE 训练和推理

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

---

## CDAC (Communicate-Descend-Ascend-Communicate)

术语解释
CDAC 是传统 fine-grained MoE（如 DeepSeekMoE）的默认执行顺序：先进行高维 All-to-All 通信，再在 expert 内部进行降维-升维投影。BigMac 论文通过分析 CDAC 的通信瓶颈，提出了 DCCA 作为替代。

术语是什么？
在 CDAC 方式下，各 expert 内部的 FFN 计算为 $E_i(x) = \sigma(xW_{i,\downarrow})W_{i,\uparrow}$（先降维后升维）。由于 All-to-All 在 expert 计算之前/之后进行，通信始终在 token 的 full hidden dimension h 上进行。

通信量：$C = 2 \times top\_k \times \frac{ep-1}{ep} \times b \times s \times h$，与 h 成正比。对于 large hidden dimension（如 DeepSeek-V2 的 5120），通信开销极大。

从算法pipeline角度拆解术语：

```
# CDAC MoE layer forward (Fine-Grained MoE baseline)
def cdac_moe_forward(x):              # x: [batch, seq, h]
    gate = SoftMax(x @ W_gate)
    topk_w, topk_idx = TopK(gate, k=top_k)

    # Step 1: All-to-All DISPATCH (HIGH dim: h, 通信瓶颈!)
    dispatched = alltoall_scatter(x, topk_idx)

    # Step 2: Expert computation (内部先降后升)
    for each expert i:
        h_down = dispatched @ W_i_down     # h → h_ff (DESCEND)
        h_act = σ(h_down)                  # activation
        h_out = h_act @ W_i_up             # h_ff → h (ASCEND)
        output += topk_w[i] * h_out

    # Step 3: All-to-All COMBINE (HIGH dim: h, 通信瓶颈!)
    y = alltoall_gather(output)
    return y
```

CDAC 的核心缺陷：All-to-All 始终在最高维度 h 上进行 → 通信量巨大 → 尤在 top_k 大时成为主导延迟（高达 91.8%）。

术语一般如何实现？如何使用？
- 是 DeepSeekMoE、Qwen2-MoE 等 fine-grained MoE 模型的默认结构
- 需要 expert parallelism 支持（All-to-All dispatch/combine）
- 在 small model 或无 EP 时通信不是瓶颈，CDAC 无劣势

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference


## Error-Bounded Lossy Compression

术语解释
Error-Bounded Lossy Compression（有界误差有损压缩）是一类保证重建数据与原始数据之间绝对误差不超过预设阈值（error bound ê）的有损压缩算法。区别于传统量化方法产生不可控、不可预测的误差，error-bounded 压缩在压缩比和重建精度之间提供可配置、可预测的权衡。

术语是什么？
Error-bounded lossy compression 的核心 pipeline：原始数据 → 预测（利用空间/时间相关性预测每个值）→ 量化（基于 error bound ê 控制量化步长，保证 |重建值 - 原始值| ≤ ê）→ 编码（Huffman/变长编码压缩量化残差）。代表性实现包括 SZ3（CPU）、CuSZp（GPU）、ZFP 等。关键特性：
- **有界误差保证**：对任意参数 θ_i 和其重建值 θ'_i，有 |θ_i - θ'_i| ≤ ê（绝对误差界）或 |θ_i - θ'_i|/|θ_i| ≤ ê_rel（相对误差界）
- **高压缩比**：利用数据空间相关性（如相邻参数值相近），通过预测器去除冗余，可实现远超简单量化的压缩比。例如 CuSZp 在 A100 上可实现 ~300 GB/s 端到端吞吐
- **可控制的精度-压缩权衡**：增大 ê 获得更高压缩比但更大误差；减小 ê 获得更高精度但更低压缩比

从算法pipeline角度拆解术语：
```
# Error-Bounded Lossy Compression Pipeline (SZ3 风格)
输入: 原始数据张量 X ∈ R^n, error bound ê (绝对误差界)

# Step 1: 预测 (Prediction)
X_pred = Predictor(X)  # Lorenzo predictor / linear regression / spline
                        # 利用相邻数据点的值预测当前点

# Step 2: 计算残差 (Residual)
R = X - X_pred  # 预测误差

# Step 3: 有界量化 (Bounded Quantization)
# 量化步长 q 的选择保证解量化后误差 ≤ ê
q = 2 * ê  # 线性量化器步长与 error bound 的关系
Q = round(R / q) * q  # 量化残差，每个值误差 ≤ ê

# Step 4: 编码 (Encoding)
compressed_data = Encode(Q)  # Huffman / 变长编码

# 解压过程 (逆过程)
Q' = Decode(compressed_data)
X_reconstructed = X_pred + Q'
# 保证: |X_i - X_reconstructed_i| ≤ ê for all i
```

在 MoE 推理 offloading 场景中的应用：
```
# Expert 压缩 offloading 流程
原始 expert 权重 W ∈ R^{d_model × d_ff}

# CPU 端压缩
compressed_W = SZ3_compress(W, error_bound=ê)

# PCIe 传输（数据量 = 原始 size / 压缩比 CR）
transfer_size = |W| * sizeof(float) / CR

# GPU 端解压
W_reconstructed = CuSZp_decompress(compressed_W)
# 保证: |W_ij - W_reconstructed_ij| ≤ ê

# GPU 上 FFN 推理（使用含 bounded error 的权重）
output = W_reconstructed @ input
```

术语一般如何实现？如何使用？
- **科学数据压缩场景**：SZ3/CuSZp 最初设计用于 HPC 科学模拟数据压缩，数据量巨大且有损压缩可接受
- **ML 模型压缩场景**：MoE expert 参数压缩是新应用方向，利用 error bound 保证推理精度可控
- **常见实现**：SZ3（C++，CPU 多线程）、CuSZp（CUDA，GPU）、ZFP（C，CPU/GPU）
- **配置参数**：error bound mode（ABS/REL/VR_REL/PW_REL）、error bound value ê、预测器类型（Lorenzo/Linear/Polynomial）
- **开源链接**：SZ3 (github.com/szcompressor/SZ3), CuSZp (github.com/szcompressor/cuSZp)
- **与量化的对比**：量化固定位宽（如 4-bit）→ 误差由位宽和数据分布决定，不可控；Error-bounded 压缩固定误差界 → 位宽和压缩比自适应，误差可保证

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

---

## SZ3 Compressor

术语解释
SZ3 是模块化的 CPU 端 error-bounded lossy compression 框架，由 Argonne National Laboratory 等机构开发，是 SZ 系列压缩器（SZ → SZ2 → SZ3）的最新主版本。发表于 IEEE Transactions on Big Data (2022)。

术语是什么？
SZ3 采用"预测-量化-编码"三阶段管线，核心模块化设计允许独立替换各阶段组件：
- **预测器（Predictor）**：Lorenzo predictor（利用多维数据空间梯度预测）、线性回归 predictor、样条 predictor 等可插拔
- **量化器（Quantizer）**：基于用户指定的 error bound ê 进行有界量化，支持 Absolute/Relative/PW_REL 等多种 error bound 模式
- **编码器（Encoder）**：Huffman 编码、变长编码等
- **关键优化**：自适应量化索引预测（IPDPS 2025 最新工作），可提升压缩比最高 95%

从算法pipeline角度拆解术语：
SZ3 在 MoE expert 压缩中的使用：
```
# 假设 expert 权重矩阵 W 形状: [d_model, d_ff] = [4096, 14336]
# 使用 SZ3 压缩（ABS mode, error bound ê = 0.01 * mean(|W|)）

# API 调用
sz3_config = SZ3_Config(
    error_bound_mode="ABS",     # 绝对误差模式
    error_bound=ê,              # 误差界
    predictor="Lorenzo",        # 预测器类型
    encoder="Huffman"           # 编码器类型
)
compressed_data = sz3_compress(W.flatten(), sz3_config)

# 压缩比计算
CR = W.size * 4 / len(compressed_data)  # 假设 FP32 (4 bytes)

# 解压
W_reconstructed = sz3_decompress(compressed_data, W.shape)
```

术语一般如何实现？如何使用？
- 语言：C++，提供 C API
- 平台：CPU (x86/ARM)，多线程并行
- 数据格式：支持 1D/2D/3D float/double 数组
- 典型使用场景：HPC checkpoint 压缩、科学数据传输、Federated Learning 模型压缩
- 开源链接：github.com/szcompressor/SZ3
- 2025 年仍作为 SOTA error-bounded compressor 被广泛使用和扩展

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

---

## CuSZp Compressor

术语解释
CuSZp 是超快速的 GPU 端 error-bounded lossy compression 框架，发表于 SC'23。核心创新是将完整的压缩/解压阶段融合到单个 CUDA kernel 中，消除 kernel launch 和数据移动开销，在 NVIDIA A100 上实现约 300 GB/s 的端到端吞吐。

术语是什么？
CuSZp 的核心设计特点：
- **单 kernel 设计**：压缩和解压各自融合为一个 CUDA kernel 函数，避免多次 kernel launch 和中间数据传输
- **多种编码模式**：Fixed（固定长度编码）、Plain（维度感知 delta 编码 + 固定长度编码）、Outlier（delta 编码 + 异常值保留）——适用于不同数据特征
- **支持 1D/2D/3D 数据**，FP32 和 FP64
- **极高性能**：SC'23 论文报告在 A100 上平均压缩吞吐 93.63 GB/s，解压吞吐 120.04 GB/s；比 cuSZ 快 95.53×，比 cuSZx 快 55.18×。
- **后续版本**：cuSZp2 (SC'24)、cuSZp3 (SC'25)，持续优化编码模式和压缩比

从算法pipeline角度拆解术语：
CuSZp 在 MoE expert GPU 端解压中的使用：
```
# GPU 端使用 CuSZp 解压 expert 权重
# 输入: compressed_expert_data (从 CPU 通过 PCIe 传输)
# 输出: expert_weights (用于 GPU FFN 计算)

# CuSZp 单 kernel 解压伪代码
__global__ void cuszp_decompress_kernel(
    uint8_t* compressed_data,
    float* output_weights,
    int n_elements,
    float error_bound
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // 单 kernel 完成: 解码 → 反量化 → 反预测
    // 输出含 bounded error 的权重
}

# 调用示例
cuszp_decompress(compressed_data, d_expert_weights, n_params, ê);
# 之后直接使用 d_expert_weights 进行 GEMM 计算
output = expert_ffn(d_expert_weights, input_activation);
```

术语一般如何实现？如何使用？
- 语言：CUDA C/C++，提供 C/C++ API 和 Python API
- 平台：NVIDIA GPU (A100/H100 tested)
- 开源：github.com/szcompressor/cuSZp
- 使用模式：CPU 端压缩（SZ3）→ PCIe 传输 → GPU 端解压（CuSZp）→ GPU 计算
- 在 MoE offloading 场景中：压缩由 CPU 端 SZ3 完成（因非激活 expert 存储在主存），GPU 端 CuSZp 负责快速解压以最小化推理延迟

涉及论文标题：
- Compression Error Sensitivity Analysis for Different Experts in MoE Model Inference

## GPTQ (Generative Pre-Trained Transformer Quantization)

术语解释
GPTQ 是一种基于近似二阶（Hessian-based）信息的 one-shot post-training weight-only 量化方法，能够在 3-4 比特精度下将 LLM 权重压缩且保持较高精度，广泛用于 MoE 模型的 expert 量化以降低参数传输和计算开销。

术语是什么？
GPTQ 的核心思想是逐列（column-wise）量化权重矩阵，利用 Hessian 矩阵的逆（用 calibration 数据的激活值估计）作为重要性度量，依次量化每一列权重并补偿之前列的量化误差，在保证整体 MSE 最小的前提下完成所有列的量化。相比于 naive rounding (RTN)，GPTQ 能在 3/4-bit 下保持接近 FP16 的 perplexity。

MoE 推理中的 GPTQ 应用场景：
- **Expert 量化减小传输**：在 expert offloading 场景中，将 expert 权重量化为 3/4-bit 后，PCIe 传输数据量减少 4-5×
- **NDP 计算加速**：在 GPU-NDP 系统中，NDP 设备使用量化 expert 权重执行 FFN 计算，降低 NDP 计算压力
- **多精度缓存**：预计算每个 expert 的 1/2/3/4-bit GPTQ 量化版本，推理时根据重要性动态选择精度

从算法pipeline角度拆解术语：
GPTQ 在 context-aware MoE-NDP 推理中的使用流程：

```
=== 离线阶段：多精度 GPTQ 量化 ===
输入: expert weights W_{l,e} ∈ R^{d_ff × d} (FP16), calibration data D_cal
输出: 预缓存的 1/2/3/4-bit 量化 replicas + per-bitwidth loss table L_{l,e}(b)

for each expert e in MoE model:
    for b in {1, 2, 3, 4}:
        # GPTQ column-wise quantization
        H = 2 * X_cal^T @ X_cal       # Hessian from calibration activations
        H_inv = Cholesky(H)^(-1)       # Inverse Hessian
        
        for col_j in range(d):
            # Quantize column j to b bits
            w_q[j] = quantize(W[:,j], b)  # round to nearest b-bit level
            error = (W[:,j] - w_q[j]) / H_inv[j,j]
            
            # Compensate remaining columns
            for col_k in range(j+1, d):
                W[:,k] -= error * H_inv[j,k] / H_inv[j,j]
        
        W_q(b) = group_and_scale(w_q)  # 应用 per-group scaling
        L_{l,e}(b) = MSE(W_q(b)(X_cal), W_fp16(X_cal))
```

```
=== 在线阶段：根据重要性选择精度 ===
解码前:
    # 从 pre-cached replicas 中按分配的 bitwidth 选择
    for expert e in NDP_resident_experts:
        b_e = bitwidth_assignment[l][e]  # 1/2/3/4 bit
        W_active[e] = load_cached(W_q_{l,e}, b_e)

解码时:
    # NDP 使用量化权重执行 FFN
    for each selected expert e on NDP:
        output = quantized_ffn(activation, W_active[e], bitwidth=b_e)
        # 低 bitwidth: 更少的内存访问和计算，但精度更低
```

术语一般如何实现？如何使用？
- 开源实现: https://github.com/IST-DASLab/gptq
- 基于 PyTorch，支持 OPT, LLaMA, BLOOM 等模型
- 量化过程约 4 GPU-hours for 175B model
- 典型配置: group_size=128, 4-bit weight, 使用 calibration dataset (如 C4 或 WikiText-2)
- 优化: GPTQ 使用 GPU kernel 加速，量化后模型可通过定制 CUDA kernel 实现 3-4× 推理加速
- 局限性: 极限低位 (1-2 bit) 下精度下降显著，需配合补偿方法 (如 LoRC, QLoRA)；不均匀的 per-expert 敏感性导致统一精度不够高效

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models

## Prefix-Structured Mixed-Precision Allocation (前缀结构混合精度分配)

术语解释
Prefix-Structured Mixed-Precision Allocation 是一种基于 expert 重要性排名的逐 expert bitwidth 分配策略，在固定每层平均 bitwidth budget 下，按重要性降序（前缀结构）将更高精度分配给更重要 experts，以最大化量化增益（减少 MSE vs FP16 reference）。

术语是什么？
在 GPU-NDP MoE 推理中，NDP 设备的计算吞吐有限，FP16 expert 执行会成为瓶颈。Prefix-Structured Allocation 的核心假设：更重要的 expert（激活频率高、路由评分高）需要更高的量化精度，次要 expert 可承受更粗量化。通过枚举所有满足 budget 的前缀结构 bitwidth 分配 (n4, n3, n2, n1)（即让最重要的 n4 个 expert 用 4-bit、其次 n3 个用 3-bit、n2 个用 2-bit、其余用 1-bit），选择累积损失降低最大的分配方案。

从算法pipeline角度拆解术语：

```
=== 符号说明 ===
E_ndp: NDP-resident experts per layer
b_bar: target average bitwidth (e.g., 2, 3)
R = E_ndp * (b_bar - 1): bitwidth increment budget
L_i(b): pre-measured MSE loss of expert i at bitwidth b

=== Step 1: 按重要性降序排列 ===
importance_scores = {S_{l,e} = α·P̃_{l,e} + (1-α)·W̃_{l,e}}
idx = argsort(importance_scores, descending=True)  # 最重要的在前

=== Step 2: 预计算前缀累积增益 ===
# 从 1-bit baseline 升级到更高精度的 loss 降低
Δ_i(2) = L_i(1) - L_i(2)
Δ_i(3) = L_i(1) - L_i(3)
Δ_i(4) = L_i(1) - L_i(4)

# 前缀累积: C_b(k) = sum_{i=1..k} Δ_i(b)
C_2 = prefix_sum(Δ(2))  # 前 k 个 upgrades to 2-bit 的增益
C_3 = prefix_sum(Δ(3))  # 前 k 个 upgrades to 3-bit 的增益
C_4 = prefix_sum(Δ(4))  # 前 k 个 upgrades to 4-bit 的增益

=== Step 3: 枚举最优分配 ===
best_gain = -∞
for n4 in 0..E_ndp:
    if 3*n4 > R: break
    for n3 in 0..(E_ndp - n4):
        if 3*n4 + 2*n3 > R: break
        n2 = R - 3*n4 - 2*n3
        if n2 < 0 or n4 + n3 + n2 > E_ndp: continue
        n1 = E_ndp - n4 - n3 - n2
        
        # 前缀结构增益
        gain = C_4(n4)                          # n4 个 most important → 4-bit
             + (C_3(n4+n3) - C_3(n4))           # 其次 n3 个 → 3-bit
             + (C_2(n4+n3+n2) - C_2(n4+n3))     # 再次 n2 个 → 2-bit
             + 0                                 # 其余 n1 个 → 1-bit (baseline)
        
        if gain > best_gain:
            best_gain = gain
            best = (n4, n3, n2, n1)
    end
end

=== Step 4: 分配 bitwidth ===
# 前缀结构: 最重要的 → 高 bitwidth
b[idx[0:n4]] = 4
b[idx[n4:n4+n3]] = 3
b[idx[n4+n3:n4+n3+n2]] = 2
b[idx[n4+n3+n2:]] = 1
```

复杂度：每层 O(E_ndp²)，总共 O(L·E_ndp²)，远小于推理成本。

术语一般如何实现？如何使用？
- 离线预计算 per-expert per-bitwidth loss table（使用 calibration data 如 C4）
- 运行时 prefix search 枚举所有可行分配，使用 prefix sums 在 O(1) 评估每个配置
- 约束：固定每层平均 bitwidth（如 b_bar=2 或 3），控制 NDP 总计算量
- 关键参数：重要性 mixing coefficient α（0.5 平衡 activation frequency 和 routing score）
- 适用场景：NDP/边缘设备的混合精度 expert 量化，多精度硬件（FP16+INT4+INT2+INT1）
- 实测效果：Ours-2bit with prefix selector vs without: +3.2% avg accuracy on 8 benchmarks

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems

## Expert Importance Score for MoE (MoE 专家重要性评分)

术语解释
Expert Importance Score 是基于 prefill 阶段 expert 激活统计计算的重要性度量，用于驱动 GPU-NDP 系统中的 expert placement 和 bitwidth allocation。定义为归一化的激活频率和路由评分的加权混合。

术语是什么？
对于 MoE 模型第 l 层的 expert e，在 prefill 阶段收集两个统计量：
- P_{l,e}：expert e 在所有 prefill tokens 中被选中的次数（激活频率）
- W_{l,e}：expert e 被选中时的累计 routing score（Softmax 后的门控权重之和，反映激活的"置信度"）

重要性分数：S_{l,e} = α · P̃_{l,e} + (1-α) · W̃_{l,e}
其中 P̃ 和 W̃ 是归一化后的值（除以该层所有 experts 的总和），α ∈ [0,1] 控制两个信号的权重。

设计动机：激活频率反映 expert 的"热度"（被用得越多越重要），routing score 反映激活的"质量"（高 score 表示 router 对该选择的置信度高）。两个信号互补——有时高频 expert 的 routing score 不高（被广泛但不强烈需要），有时低频但有高置信度 score（对特定上下文重要）。

从算法pipeline角度拆解术语：

```
=== Prefill 阶段统计收集 ===
def prefill_with_importance_stats(tokens):
    for each MoE layer l:
        P[l] = zeros(E)    # activation counts
        W[l] = zeros(E)    # cumulative routing scores
        
        for each token x in tokens:
            scores = Softmax(W_gate[l] @ x)  # [E]
            top_k = TopK(scores, k=2)
            
            for e in top_k:
                P[l][e] += 1
                W[l][e] += scores[e]
    
    return P, W

=== Importance Score 计算 ===
def compute_importance_scores(P, W, alpha=0.5):
    S = {}
    for layer l:
        # 归一化到 [0, 1] (per-layer)
        P_tilde = P[l] / sum(P[l])
        W_tilde = W[l] / sum(W[l])
        S[l] = alpha * P_tilde + (1-alpha) * W_tilde
    return S
```

```
=== 基于 Importance 的 Expert Placement ===
K = GPU_expert_budget  # 由 GPU HBM 容量决定

for each layer l:
    ranked_experts = argsort(S[l], descending=True)
    H[l] = ranked_experts[:K]   # GPU (FP16, hot)
    C[l] = ranked_experts[K:]   # NDP (quantized, cold)
```

关键性质：
- Prefill-decode similarity (cosine sim ~0.89) → prefill importance 可预测 decoding 行为
- Per-sequence 计算 → 捕捉 context-dependent expert 重要性
- Once-per-sequence → decoding 期间零额外迁移

术语一般如何实现？如何使用？
- 收集开销：每层 E 个计数器（如 Mixtral 8 experts），2 个指标 × 32 layers = 512 values，metadata 开销可忽略
- 归一化：per-layer normalization 保证层间可比，适应不同层 expert 分布的差异
- α 选择：论文使用 α=0.5，可通过 minimal grid search 调优
- 应用：(1) expert placement to GPU/NDP；(2) bitwidth allocation ordering；(3) 可扩展到 expert pruning/caching 决策
- 替代方案：仅用 frequency (simple but ignores routing confidence)、仅用 routing score (captures confidence but misses volume)

涉及论文标题：
- Context-Aware Mixture-of-Experts Inference on CXL-Enabled GPU-NDP Systems
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework（使用更简单的 Expert Drop 重要性评分：S(E_i) = (1/|X|) * Σ_{x∈X} G_i(x)，即批数据上的平均路由分，用于 Expert Drop 的 layer-wise 和 global dropping 策略——layer-wise 每层保留相同数量 experts，global 跨层全局选择 Top experts；发现 score distribution 影响 MoE 对 Expert Drop 的鲁棒性：DeepSeek-MoE-16B 左偏分布（多数 expert 低分→可 drop 更多），Mixtral-8×7B 右偏分布（仅少数不重要→drop 代价高））

---

## Straight-Through Estimator (STE) in MoE Expert Routing

术语解释
Straight-Through Estimator (STE) 是一种使梯度能够穿过前向传播中的不可微离散操作的启发式技术，最初由 Bengio et al. (2013) 提出。DSMoE 将 STE 应用于 MoE 专家路由中的硬阈值门控，使未通过激活阈值的 expert 也能在反向传播中接收梯度信号，从而解决"死 expert"问题。

术语是什么？
STE 的核心思想是**前向传播使用离散/硬阈值操作，反向传播用可微的替代函数（通常是恒等函数 identity）传递梯度**。标准形式（量化场景）：
$$\nabla_x (l \circ q)(x) \approx \nabla l(q(x))$$
即把不可微函数 q 的导数近似为 1，让梯度"直通"（pass straight through）。

DSMoE 对 STE 的关键扩展：定义 S(x) = sg(G(x)) + x - sg(x)，其中 G(x) 为硬阈值阶跃函数（输出仅 0 或 x），sg(·) 为 stop_gradient 算子。公式展开：
- 前向：S(x) = G(x)（因 sg(G)-sg(x) 在前向中不贡献数值，与 G(x) 等价）
- 反向：∂S/∂x = 1（sg(G) 和 sg(x) 的梯度均被阻断，仅 x 项贡献梯度）

在 DSMoE 的 MoE 路由场景中，这意味着：
- 前向：仅 σ(xY_i) > τ 的 expert 参与 FFN 计算（保持稀疏性）
- 反向：所有 expert 的门控参数 Y_i 均接收梯度 ∂h/∂Y_i = (ĥ)^T · (o_i · σ'(ĥY_i))

从算法pipeline角度拆解术语：
DSMoE 中 STE 的工作流程（以 LLaMA-7B、8 experts、阈值 τ=0.5 为例）：

```
=== STE Forward + Backward in DSMoE Expert Routing ===

# 符号:
# x: [B, d] hidden states, Y: [d, n] gate parameters
# o_i: expert i 的 FFN 输出 [B, d]
# tau: 激活阈值 (0.5)

def dsmo_e_gate_ste(x, Y, tau, training):
    gate_raw = x @ Y                          # [B, n]
    gate_prob = sigmoid(gate_raw)             # [B, n], all in (0, 1)
    
    if training:
        # Forward: hard threshold for sparse computation
        gate_hard = gate_prob.clone()
        gate_hard[gate_hard <= tau] = 0.0     # G(gate_prob)
        
        # STE trick: sg(G(x)) + x - sg(x)
        gate_ste = gate_hard.detach() + gate_prob - gate_prob.detach()
        # 前向值 = gate_hard (零值 expert 不参与计算)
        # 反向梯度 = ∂gate_prob/∂Y (所有 expert 均接收梯度)
    else:
        gate_ste = gate_prob.clone()
        gate_ste[gate_ste <= tau] = 0.0
    
    # 加权求和: h = Σ o_i · gate_ste[:, i]
    h = sum(expert_outputs[i] * gate_ste[:, i:i+1] for i in range(n))
    
    # 激活数归一化: h *= n / num_active
    num_active = (gate_prob > tau).float().sum(dim=1, keepdim=True)
    h = h * (n / num_active.clamp(min=1))
    
    return h, gate_ste, num_active
```

关键梯度性质：
- 对门控参数 Y_i 的梯度（无论 expert i 是否激活）：∂h/∂Y_i = (ĥ)^T · (o_i · σ'(ĥY_i))
- 梯度方向取决于 o_i 是否有助于降低 loss：若 o_i 输出有益 → Y_i 增大 → 未来更可能激活；若 o_i 输出有害 → Y_i 减小 → 未来更可能抑制
- vs 无 STE（仅用 G(x)）：无 STE 时 ∂h/∂Y_i = 0 when gate_prob ≤ τ → 导致死 expert，STE 使 ∂h/∂Y_i ≠ 0 regardless → 所有 expert 持续学习

术语一般如何实现？如何使用？
- **PyTorch 实现**：使用 `.detach()` 实现 stop_gradient 操作，`(hard - prob.detach()) + prob` 是标准 STE 模式
- **适用场景**：量化感知训练（QAT）、二值化网络、MoE 动态路由、稀疏激活训练等任何需要对连续参数施加离散约束的训练场景
- **常见 STE 变体**：Identity STE（导数=1）、ReLU STE（导数=1[x≥0]）、Clipped ReLU STE（导数=1[0≤x≤1]）
- **DSMoE 的创新用法**：STE 不仅用于梯度传递，还配合 sparse loss 形成对抗训练机制——STE 允许所有 expert 门控接收梯度更新，sparse loss 提供抑制不重要 expert 的压力，两者博弈使模型自主学习稀疏激活模式
- **局限性**：STE 是一种有偏梯度估计（biased gradient estimator），前向离散操作与反向连续梯度的不匹配在极端稀疏场景（如 τ 值很高）可能导致训练不稳定

**DefaultMoE 对 STE 的关键扩展（Dense Backpropagation）**：DefaultMoE 将 STE 应用于标准 TopK routing（而非阈值门控），通过 EMA default vector 填充未激活 expert 的输出。与 DSMoE 的差异：
- DSMoE 使用 sigmoid 阈值门控，STE 使所有 expert 在反向时接收梯度
- DefaultMoE 使用 Softmax + TopK 门控，STE 替代不可微的 TopK 选择操作，将所有 N 个 expert 的输出纳入梯度计算：∂y/∂π = [E_1(x) or Ê_1, ..., E_N(x) or Ê_N]^T，其中 Ê_i 为未激活 expert i 的 EMA default vector
- 关键洞察：若直接使用 STE（identity），Router 梯度为 ∂y/∂π = [E_i(x) for all i]，但这需要所有 expert 前向计算，丧失了稀疏性。DefaultMoE 通过 EMA 近似绕过此限制

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

---

## SwiGLU (Swish-Gated Linear Unit) Activation for FFN

术语解释
SwiGLU 是现代大语言模型（LLaMA、PaLM、Mistral、Qwen 等）中最广泛使用的 FFN 激活函数，结合了 Swish (SiLU) 激活和 Gated Linear Unit (GLU) 门控机制。它使用三个权重矩阵（gate_proj、up_proj、down_proj）替代传统 FFN 的两个矩阵，通过可学习的门控实现输入自适应的特征过滤。

术语是什么？
SwiGLU FFN 前向计算：
$$FFN_{SwiGLU}(x) = (SiLU(xW_{gate}) \odot (xW_{up})) W_{down}$$

其中 SiLU(x) = x · σ(x)（即 Swish 激活，σ 为 sigmoid），⊙ 为逐元素乘法（Hadamard product）。三步计算过程：
1. **Gate 投影**：h_gate = xW_gate，经 SiLU 激活 → [B, D_intermediate]
2. **Up 投影**：h_up = xW_up → [B, D_intermediate]
3. **门控融合 + 降维**：(SiLU(h_gate) ⊙ h_up)W_down → [B, d]

三个权重矩阵维度：W_gate ∈ R^{d×D}、W_up ∈ R^{d×D}、W_down ∈ R^{D×d}，其中 d 为 hidden dimension，D 为 intermediate/expansion dimension。

从算法pipeline角度拆解术语：
SwiGLU 在 Transformer FFN 层中的计算流程：

```
=== SwiGLU FFN Forward Pass ===

输入: x [B, d] — attention output hidden states

Step 1 — Gate Projection + SiLU Activation:
    gate = W_gate @ x         # [B, D]
    gate_act = gate * sigmoid(gate)  # SiLU activation, [B, D]
    # SiLU 性质: 光滑、非单调、负值区域有非零梯度

Step 2 — Up Projection (value path):
    up = W_up @ x             # [B, D]

Step 3 — Gated Feature Selection:
    hidden = gate_act * up    # element-wise, [B, D]
    # 关键: gate_act 的每个元素控制 up 对应元素的通过量
    # gate_act ≈ 0 → 该维度被抑制
    # gate_act ≈ up → 该维度保持
    # gate_act < 0 → 该维度被反转（SiLU 负值区域）

Step 4 — Down Projection:
    output = hidden @ W_down  # [B, d]

输出: output [B, d]
```

vs 传统 FFN（ReLU/GELU）：
- ReLU FFN：FFN(x) = ReLU(xW1 + b1)W2 + b2（2 个权重矩阵）
- SwiGLU FFN：3 个权重矩阵，但 D 通常缩放为原来的 2/3 以保持参数量可比
- 关键差异：SwiGLU 的"门控"是数据依赖的（data-dependent），不同输入产生不同的特征过滤模式

在 DSMoE 中的应用：DSMoE 将 SwiGLU FFN 的三个矩阵沿 intermediate dimension D 均等划分为 n 组，每组构成一个 expert，数学上保证了划分后所有 expert 输出之和等价于原始 FFN 输出。

术语一般如何实现？如何使用？
- **主流 LLM 框架**：HuggingFace Transformers 中的 LLaMA 系列默认使用 SwiGLU；PyTorch 中 `F.silu()` 为 SiLU 激活的原生实现
- **Fused Kernel 优化**：部分框架提供 fused SwiGLU kernel，将 gate projection + SiLU + element-wise multiply 融合为单次 kernel launch，减少 HBM 读写
- **GLU 变体对比**：SwiGLU（SiLU gate）、ReGLU（ReLU gate）、GEGLU（GELU gate）中 SwiGLU 在语言建模任务上经验表现最优（Shazeer, 2020, "GLU Variants Improve Transformer"）
- **维度选择惯例**：LLaMA-7B 使用 D=11008（≈ 8/3 × 4096 × 2/3 的取整）

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

---

## FFN Matrix Partitioning for Expert Construction

术语解释
FFN Matrix Partitioning 是 DSMoE 的核心技术：将预训练 Dense 模型的 SwiGLU FFN 层的三个权重矩阵（W_gate、W_up、W_down）沿 intermediate/expansion 维度均等划分为 n 组，每组构成一个独立的 expert FFN，全部参数保留、知识零损失地在数学上保证等价性。这是 Dense-to-Sparse MoE 转换中最简洁的参数复用策略。

术语是什么？
给定预训练 SwiGLU FFN 的输出公式：
$$h = (SiLU(xW_{gate}) \odot (xW_{up})) W_{down}$$

将三个矩阵沿 intermediate dimension D 等分为 n 段：
$$W_{gate} = [W_1^{gate} \| \cdots \| W_n^{gate}],\quad W_{up} = [W_1^{up} \| \cdots \| W_n^{up}],\quad W_{down} = [V_1 \| \cdots \| V_n]^T$$

其中每组 W_i^{gate} ∈ R^{d×D/n}、W_i^{up} ∈ R^{d×D/n}、V_i ∈ R^{D/n×d} 构成 expert i 的参数。

**等价性证明**：通过矩阵分块乘法的分配律：
$$h = \sum_{i=1}^{n} (SiLU(xW_i^{gate}) \odot (xW_i^{up})) V_i$$

所有 expert 输出之和在数学上严格等于原始 Dense FFN 输出。这是 DSMoE "零知识损失"转换的理论基础。

从算法pipeline角度拆解术语：
Partitioning 的完整流程（以 LLaMA-7B, d=4096, D=11008, n=8 为例）：

```
=== Pre-trained Dense FFN → Partitioned Experts ===

原始 Dense FFN:
    W_gate: [4096, 11008]  → 划分为 8 个 [4096, 1376]
    W_up:   [4096, 11008]  → 划分为 8 个 [4096, 1376]
    W_down: [11008, 4096]  → 划分为 8 个 [1376, 4096]

Expert i (i = 0, ..., 7):
    Expert_i.W_gate = W_gate[:, i*1376 : (i+1)*1376]  # [4096, 1376]
    Expert_i.W_up   = W_up[:,   i*1376 : (i+1)*1376]  # [4096, 1376]
    Expert_i.W_down = W_down[i*1376 : (i+1)*1376, :]  # [1376, 4096]


=== Expert FFN Forward (per expert i) ===

def expert_i_forward(x, expert_i):
    # x: [B, 4096]
    gate = silu(x @ expert_i.W_gate)  # [B, 1376]
    up   = x @ expert_i.W_up          # [B, 1376]
    out  = (gate * up) @ expert_i.W_down  # [B, 4096]
    return out


=== Full Partitioned FFN (all experts active) ===

def partitioned_ffn_full(x, experts):
    # 所有 expert 激活时 = 原始 Dense FFN（等价性保证）
    outputs = [expert_i_forward(x, exp) for exp in experts]
    h = sum(outputs)  # [B, 4096] ≡ 原始 FFN(x)
    return h


=== Sparse Partitioned FFN (DSMoE inference) ===

def dsmo_e_ffn_sparse(x, experts, Y, tau=0.5):
    gate_probs = sigmoid(x @ Y)  # [B, 8]
    active_mask = gate_probs > tau
    num_active = active_mask.sum()
    
    h = zeros_like(x)  # [B, 4096]
    for i in range(8):
        if active_mask[:, i].any():
            h += expert_i_forward(x, experts[i]) * gate_probs[:, i:i+1]
    
    # 激活数归一化: 保持输出范数稳定
    h = h * (8 / num_active.clamp(min=1))
    return h
```

术语一般如何实现？如何使用？
- **划分策略**：等分为最简单的方案，论文未探索非均匀划分（如根据 expert 重要性分配不同大小的 intermediate dimension slice）
- **expert 数量选择**：论文固定使用 n=8（LLaMA-1B: D=1024×8, LLaMA-7B: D=1376×8），更多 expert → 更细粒度的激活控制但每个 expert 容量更小
- **与其他 Dense-to-MoE 方法的对比**：
  - LLaMA-MoE (Zhu et al., 2024)：类似的分区方案但使用传统 top-k softmax 路由
  - MoEfication (Zhang et al., 2022)：基于 ReLU 激活的 expert 分区，需要额外转换步骤适配 SiLU/GeLU
  - FactorLLM (Zhao et al., 2024)：多阶段训练（teacher-student），路由器先训练后冻结
  - DSMoE 的优势：最简分区方案 + 端到端训练 + 数学等价性保证
- **与 Expert Decomposition (Low-Rank) 的区别**：FFN Matrix Partitioning 是沿 intermediate dimension 的结构化切分（preserves full rank per expert），而非低秩近似
- **局限性**：等分策略隐含假设 intermediate dimension 各部分的"知识"是均匀分布的，如果预训练模型的 FFN 神经元存在显著的功能聚类，等分可能破坏这些功能单元

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

---

## DSMoE (Dynamic Sparse Mixture-of-Experts)

术语解释
DSMoE (Dynamic Sparse Mixture-of-Experts) 是一种将预训练 Dense LLM 的 FFN 层转化为输入自适应的稀疏 MoE 的方法，由 Lv et al. (2025) 提出。其核心创新在于：(1) 通过矩阵分区保留全部预训练知识；(2) 使用 sigmoid 门控替代 softmax top-k 实现每个 expert 独立激活决策；(3) 通过 Straight-Through Estimator 和稀疏损失实现端到端的动态稀疏模式学习。与剪枝（永久丢弃参数）和传统 MoE（固定 top-k 激活 + load balancing loss）有本质区别。

术语是什么？
DSMoE 由三个正交模块组成，总损失函数为 L = L_LM + (1/(L·N)) · Σ_l Σ_n G(σ(ĥY_n))，无可选的 load balancing loss：

**模块 1 — FFN Partitioning**：将 SwiGLU FFN 的三个权重矩阵 (W_gate, W_up, W_down) 沿 intermediate dimension 划分为 n 个 expert，全部参数保留，所有 expert 输出之和数学等价于原始 FFN 输出。

**模块 2 — STE-Enhanced Sigmoid Gating**：使用 sigmoid 激活（非 softmax）使每个 expert 独立判断是否激活（σ(xY_i) > τ）。与 softmax 的关键区别：sigmoid 输出非归一化，各 expert 决策互不依赖，允许变长 expert 集合。STE 确保所有 expert 门控参数在反向传播中接收梯度。

**模块 3 — L1 Sparse Loss**：惩罚门控激活值，与 STE 门控梯度形成对抗，鼓励模型为不同输入学习不同的稀疏激活模式。

从算法pipeline角度拆解术语：
DSMoE 训练和推理的完整流程：

```
=== DSMoE Training Pipeline ===

# 初始化:
# 1. 加载预训练 Dense LLaMA 模型
# 2. 对每层 FFN 执行 Matrix Partitioning → n 个 expert
# 3. 初始化门控参数 Y ∈ R^{d×n}（随机初始化）
# 4. 不引入 load balancing loss

for each training step:
    for each Transformer layer l:
        # Step 1: Self-Attention（不变）
        x = attention(layer_norm_1(h_prev))
        h_hat = residual(x, h_prev)
        
        # Step 2: DSMoE FFN (替代原 Dense FFN)
        gate_probs = sigmoid(h_hat @ Y)           # [B, n]
        gate_ste = STE(gate_probs, threshold=0.5) # 前向稀疏 + 反向全梯度
        
        # Step 3: Expert 计算（仅激活 expert 参与）
        outputs = []
        for i in range(n):
            if gate_ste[:, i].any():
                o_i = expert_i_swiglu_ffn(h_hat)
                outputs.append(o_i * gate_ste[:, i:i+1])
        
        h = sum(outputs) * (n / num_active)  # 归一化
        
        # Step 4: 收集门控值用于 sparse loss
        layer_gate_values.append(gate_ste)
    
    # Step 5: Loss 计算
    lm_loss = cross_entropy(logits, targets)
    sparse_loss = (1/(L*N)) * sum(g.sum() for g in layer_gate_values)
    total_loss = lm_loss + sparse_loss  # 无 load balancing loss!
    
    # Step 6: 反向传播
    total_loss.backward()
    # STE 确保所有 expert Y_i 接收梯度（即使未激活）
    optimizer.step()
```

```
=== DSMoE Inference Pipeline (per token) ===

输入: token embedding x

for each layer:
    h_hat = attention(x)
    gate_probs = sigmoid(h_hat @ Y)   # [1, 8]
    
    # 硬阈值推理
    active = gate_probs > 0.5         # e.g., [1, 0, 1, 0, 1, 0, 1, 0]
    num_active = active.sum()         # 4 of 8 active
    
    # 仅计算激活 expert
    h = zeros(d)
    for i in range(8):
        if active[i]:
            h += expert_i_ffn(h_hat) * gate_probs[i]
    
    h = h * (8 / num_active)          # 归一化
    x = layer_norm(h + residual)

# 层间激活模式: 形成 "W 形" —— 首尾层高激活、中间层突起、其余层低激活
```

术语一般如何实现？如何使用？
- **训练配置**：论文使用 lr=2e-5, batch_size=32, seq_len=1024, 10B tokens 继续预训练，threshold τ=0.5（可通过 sweep τ∈[0.2,0.8] 调节稀疏度，τ 越大越稀疏）
- **激活参数比例**：LLaMA-7B DSMoE 在 τ=0.5 时激活约 58.46% 参数（3.93B/6.74B），τ=0.8 时降至 52.54%
- **推理加速机制**：结构化 expert 跳过（未激活 expert 的矩阵乘法可直接跳过，无需稀疏计算库）vs SparseGPT 的非结构化稀疏需要专用硬件
- **vs 传统 MoE 的关键差异**：
  | 维度 | 传统 MoE (Switch/LLaMA-MoE) | DSMoE |
  |------|---------------------------|-------|
  | 路由函数 | Softmax top-k | Sigmoid + 阈值 |
  | Expert 激活数 | 固定 k | 变长 (1~n) |
  | Load balancing | 需要 auxiliary loss | 不需要 |
  | Expert 初始化 | 随机初始化 | 从预训练模型分区继承 |
  | 训练目标 | L_LM + L_aux (load balance) | L_LM + L_sparse (L1) |
  | 路由策略 | 竞争性（归一化后选择） | 独立性（每个 expert 独立决策） |
- **层间激活模式（W 形）**：底层（高激活，多维特征处理）→ 中层（突起激活，关键特征转换区）→ 中上层（低激活，特化处理）→ 顶层（高激活，综合决策）。不同输入激活模式不同，体现输入自适应特性
- **局限性**：仅验证到 7B 参数（受计算资源限制），更大模型的扩展行为未知；门控参数 Y 需从随机初始化训练（可能导致训练初期路由不稳定）；训练和推理的 gate 行为不一致（训练 STE vs 推理纯硬阈值）；开源代码未公开

涉及论文标题：
- DSMoE Matrix-Partitioned Experts with Dynamic Routing for Computation-Efficient Dense LLMs

---

## DeRS (Decompose, Replace, Synthesis) Paradigm

术语解释
DeRS 是由 Huang et al. (2025) 提出的针对 upcycled MoE 模型的参数效率提升范式。核心思想是将 N 个 MoE 专家重构为 1 个专家共享基础权重 + N 个轻量专家专属增量权重，通过消除专家间冗余参数实现极高参数效率。包含三种操作：Decompose（分解为 W_base + Δ_i）、Replace（用稀疏化/量化/低秩矩阵替换 Δ_i）、Synthesis（按需合成 Ŵ_i = W_base + F(Δ_i)）。

术语是什么？
1. **Decompose**：利用 upcycled MoE 专家共享同一初始权重 W_base，将训练后专家分解为 W_i = W_base + Δ_i。余弦相似度 > 0.999 表明 Δ_i 是微小冗余调整。
2. **Replace**：用轻量表示 F(Δ_i) 替换 Δ_i——后处理稀疏化/量化（DeRS Compression）或从训练开始使用稀疏矩阵/低秩矩阵（DeRS Upcycling）。
3. **Synthesis**：推理/训练时按需合成 Ŵ_i = W_base + F(Δ_i)。

从算法pipeline角度拆解术语。
```
def DeRS_forward(x, router, W_base, compact_deltas, k):
    scores = softmax(x @ W_R)
    selected = TopK(scores, k)
    y = 0
    for i in selected:
        Δ_i = reconstruct(compact_deltas[i])  # 稀疏解压/反量化/低秩乘积
        W_i = W_base + Δ_i                    # 加法合成
        y += scores[i] * FFN(x, W_i)
    return y

# Compression: F(Δ_i) = (1-M)⊙Δ_i/(1-p), M~Bernoulli(p) 或 Quant(Δ_i,k)
# Upcycling-SM: F(Δ_i) = torch.scatter(I_i, V_i), I_i固定 V_i训练
# Upcycling-LM: F(Δ_i) = A_i@B_i, [d,r]×[r,d_h]
```

术语一般如何实现？如何使用？
- 稀疏化适合 dense model 经过先验微调（delta 冗余极高，可承受 0.99 drop rate）
- 量化/低秩适合未经过先验微调（需要全局修改能力）
- 仅适用于 upcycled MoE（需共享 W_base），from-scratch MoE 不适用
- 效果：MoE-LLaVA-Phi 上 DeRS-SM 增加 1.11M 参数（2270× 减少）性能 61.1 vs 60.8

涉及论文标题：
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

---

## Expert-Specific Delta Weight in Upcycled MoE

术语解释
在 upcycled MoE 中，专家权重 W_i = W_base + Δ_i。Δ_i 是训练中学到的相对于共享初始权重的微小偏移。DeRS 论文发现 Δ_i 高度冗余（余弦相似度 > 0.999），可通过稀疏化（drop 90% 元素）或量化（至 2-bit）几乎无损失地压缩。

术语是什么？
- Δ_i = W_i - W_base ∈ R^{d×d_h}，表示专家 i 相对共享知识的偏移
- 冗余程度取决于 dense model 是否经过先验微调：微调过的 delta 冗余度高（可 99% drop / 1-bit），未微调的冗余度低（需保守压缩）
- 独立压缩 Δ_i 而非 W_i 可保护 W_base 中的预训练知识

从算法pipeline角度拆解术语。
```
def analyze_and_compress_deltas(experts, W_base):
    for W_i in experts:
        cos_sim = cosine_similarity(flatten(W_i), flatten(W_base))
        # > 0.999, delta幅值远小于base
        Δ_i = W_i - W_base
    
    for each Δ_i:
        M ~ Bernoulli(p); compact = (1-M)⊙Δ_i/(1-p)  # 稀疏化
        # 或 compact = Quant(Δ_i, k_bits)               # 量化
    return W_base, compact_deltas

def synthesize(W_base, compact_delta_i):
    return W_base + reconstruct(compact_delta_i)
```

术语一般如何实现？如何使用？
- 观察方法：计算 flatten(W_i) 与 flatten(W_base) 的余弦相似度
- 与 LoRA 的关系：LoRA 的 ΔW = A·B 可视为 delta weight 的低秩形式，但应用于不同场景

涉及论文标题：
- DeRS Towards Extremely Efficient Upcycled Mixture-of-Experts Models

## DeepSeekMoE with Auxiliary-Loss-Free Load Balancing (无辅助损失负载均衡的DeepSeekMoE)

术语解释
DeepSeekMoE 是 DeepSeek 系列模型采用的 Mixture-of-Experts 架构（Dai et al. 2024），使用细粒度专家分割（fine-grained expert segmentation）和共享专家隔离（shared expert isolation）。DeepSeek-V3 在此基础上引入 Auxiliary-Loss-Free Load Balancing（Wang et al. 2024a），通过每个专家的可学习 bias 项 b_i 动态调整路由决策，替代传统 auxiliary loss，在保证负载均衡的同时消除 auxiliary loss 对模型性能的负面影响。

术语是什么？
DeepSeekMoE 的核心设计：(1) **细粒度专家分割**：使用大量小规模 routed experts（256 个，每个 intermediate dim=2048）而非粗粒度大专家，提升专家特化程度；(2) **共享专家隔离**：设置 1 个 shared expert 处理通用知识，routed experts 专注特定领域，减少知识冗余；(3) **Auxiliary-Loss-Free 路由**：为每个 expert 引入 bias b_i，仅在 Top-K 路由选择时加到 affinity score s_{i,t} 上（s_{i,t}+b_i 决定 Top-K），但 gating value 仍使用原始 s_{i,t}（Sigmoid 归一化）。训练每步结束时动态调整 b_i：过载 expert 的 b_i -= γ(0.001)，欠载 expert 的 b_i += γ(0.001)。配合极小的 complementary sequence-wise balance loss（α=0.0001）防止单序列极端不均衡。(4) **Node-Limited Routing**：每 token 最多路由到 M=4 个节点的 expert，平均每节点 3.2 个 expert，实际选择 K_r=8。

从算法pipeline角度拆解术语：
```
=== DeepSeekMoE Forward Pass (per token, per MoE layer) ===

Input: u_t ∈ R^d  (FFN input after attention, d=7168)

// 1. Gate Computation
for i in 1..256:
    s_{i,t} = Sigmoid(u_t^T · e_i)       // token-to-expert affinity

// 2. Aux-Loss-Free Routing (bias adjustment at step boundary)
selected = TopK({s_{j,t} + b_j | j=1..256}, K_r=8)  // bias only for routing!
g_{i,t}' = s_{i,t} if i in selected else 0          // gating from raw affinity
g_{i,t} = g_{i,t}' / sum_j(g_{j,t}')                // normalize

// 3. Node-Limited Constraint
// Ensure selected experts are on at most M=4 nodes

// 4. Expert Computation
h_t' = u_t
     + sum_{i=1}^{1} FFN_i^{(s)}(u_t)               // shared expert (always active)
     + sum_{i in selected} g_{i,t} · FFN_i^{(r)}(u_t) // 8 routed experts

// 5. Post-Step Bias Update (at end of each training step)
for i in 1..256:
    load_i = tokens_routed_to_expert_i / expected_tokens
    b_i += γ * (1 - load_i)  // γ=0.001, drives toward balanced load
```

Auxiliary-loss-free vs batch-wise auxiliary loss 对比：实验表明 batch-wise balancing 比 sequence-wise balancing 更灵活，允许专家在不同 domain 上特化。1B MoE 模型验证：sequence-wise loss=2.258, aux-loss-free=2.253, batch-wise loss=2.253。Pile-test 上 aux-loss-free 模型展现更强的 domain-specific expert specialization patterns。

术语一般如何实现？如何使用？
DeepSeek-V3 使用 61 层 Transformer，前 3 层为 dense FFN，后 58 层为 MoE。每个 MoE 层：1 shared expert + 256 routed experts（intermediate dim=2048）。K_r=8，M=4（node-limited）。Sigmoid gating with top-K affinity normalization。训练时：γ=0.001（first 14.3T tokens）→ 0.0（last 500B tokens），α=0.0001。推理时：bias 固定（不再更新），shared expert 在 decoding 阶段视为 always-selected routed expert，实际激活 9 experts/token。无 token dropping（训练和推理均不丢 token）。

涉及论文标题：
- DeepSeek-V3 Technical Report
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

## Multi-Token Prediction (MTP / 多令牌预测)

术语解释
Multi-Token Prediction (MTP) 是一种训练目标扩展技术，使 LLM 在每个 position 不仅预测下一个 token，还额外预测后续多个未来 token，从而稠化训练信号、提升数据效率。DeepSeek-V3 采用 D=1 depth 的 MTP 模块，不同于 Gloeckle et al. (2024) 的并行预测，DeepSeek 使用顺序预测并保持完整因果链。MTP 模块在推理时可丢弃或用于 speculative decoding（第二 token 接受率 85-90%，1.8× TPS 加速）。

术语是什么？
MTP 在 DeepSeek-V3 中的实现：(1) 1-depth MTP 模块（D=1），每个 position 额外预测第 2 个未来 token；(2) 每个 MTP 模块包含：shared embedding layer Emb(·)、shared output head OutHead(·)、独立 Transformer block TRM_k(·)、projection matrix M_k ∈ R^{d×2d}；(3) 保持完整 causal chain：h_i'^k = M_k[RMSNorm(h_i^{k-1}); RMSNorm(Emb(t_{i+k}))]；(4) 训练 loss：λ/D * Σ_k CrossEntropy(P_k, t)，其中 λ=0.3 (first 10T tokens) → 0.1 (last 4.8T tokens)。推理时可直接丢弃 MTP 模块，或保留用于 speculative decoding。

从算法pipeline角度拆解术语：
```
=== MTP Training Forward Pass (D=1) ===

Main Model:
  h_{1:T}^0 = MainTransformer(input[1:T])     // standard representation

MTP Module k=1:
  for i in 1..T-1:
    h_i'^1 = M_1 @ [RMSNorm(h_i^0); RMSNorm(Emb(t_{i+1}))]  // [d×2d] concat projection
  h_{1:T-1}^1 = TRM_1(h_{1:T-1}'^1)          // independent Transformer block
  P_{i+2}^1 = Softmax(OutHead(h_i^1))         // shared output head (with main model)

Loss:
  L_main = CrossEntropy(P_main[2:T+1], t[2:T+1])
  L_MTP^1 = CrossEntropy(P_{2+k:T+1}^1, t_{2+k:T+1})  // predict 2nd-next token
  L_total = L_main + (λ/D) * L_MTP^1
```

术语一般如何实现？如何使用？
消融实验（Table 4）：Small MoE (15.7B) 和 Large MoE (228.7B) 上 MTP 一致提升 benchmark 性能。推理时：(a) 直接丢弃 MTP 模块，主模型独立推理——MTP 的受益已融入主模型训练；(b) 保留 MTP 模块用于 speculative decoding——主模型预测 t_{n+1}，MTP 模块预测 t_{n+2}，接受率 85-90%，实现 1.8× TPS 加速。MTP 的思想也见于 EAGLE (Li et al. 2024b)，但 EAGLE 主要用于 speculative decoding，而 DeepSeek-V3 的 MTP 主要用于改善训练质量。

涉及论文标题：
- DeepSeek-V3 Technical Report

## FP8 Mixed Precision Training for Large-Scale LLMs (FP8混合精度训练)

术语解释
FP8 Mixed Precision Training 是 DeepSeek-V3 提出的低精度训练框架，首次在超大规模模型（671B）上验证 FP8 训练的有效性。核心方案：将大部分 GEMM 操作（Fprop/Dgrad/Wgrad）在 FP8 E4M3 格式下执行，通过 fine-grained quantization（activation: 1×128 tile-wise, weight: 128×128 block-wise）和 CUDA Core FP32 promotion 解决 H800 Tensor Core 仅 14-bit 累积精度的硬件限制。BF16 → FP8 训练的 relative loss error <0.25%。

术语是什么？
DeepSeek-V3 FP8 框架的关键技术：(1) **Fine-grained quantization**：activation 按 1×128 tile 分组缩放（per token per 128 channels），weight 按 128×128 block 分组缩放，优于传统 tensor-wise scaling；(2) **Increased accumulation precision**：每 N_c=128 个 Tensor Core WGMMA 结果拷贝到 CUDA Core 做 FP32 完整精度累积+dequantization（scaling factor 乘法融合），两个 warpgroup 交替执行；(3) **E4M3 for all tensors**：不使用 E5M2，通过 fine-grained scaling 弥补动态范围不足；(4) **Online quantization**：每 tile/block 实时计算 max absolute value 确定 scaling factor，不使用历史值；(5) **Low-precision storage**：BF16 optimizer states（first/second moments），FP8 cached activations（E5M6 for attention inputs），FP8 dispatch activations。

从算法pipeline角度拆解术语：
```
=== FP8 GEMM Forward with Fine-Grained Quantization ===

Input: X [M, K] in BF16, W [K, N] in BF16

// Quantization
for tile i in 1..(K/128):            // 1×128 tile-wise for activation
    scale_X[i] = max(|X[:, i*128:(i+1)*128]|) / 448.0  // FP8 E4M3 max
    X_FP8[:, i*128:(i+1)*128] = X[:, i*128:(i+1)*128] / scale_X[i]

for block (i,j) in (K/128)×(N/128):  // 128×128 block-wise for weight
    scale_W[i,j] = max(|W[i*128:(i+1)*128, j*128:(j+1)*128]|) / 448.0
    W_FP8[block] = W[block] / scale_W[i,j]

// MMA with CUDA Core Promotion (alternating warpgroup pairs)
for k_step in 0..(K/128-1):
    partial_k = WGMMA(X_FP8_k, W_FP8_k)   // Tensor Core, ~14-bit accumulation
    if (k_step+1) % 4 == 0:  // every N_c=128 elements (4 WGMMAs)
        C += FP32_promote(partial_sum) * scale_X * scale_W  // CUDA Cores
    else:
        accumulate_in_tensor_core(partial_sum)  // limited precision

// High-precision retained for: embedding, output head, MoE gating,
// normalization, attention operators
```

术语一般如何实现？如何使用？
训练速度理论 2× over BF16。DeepSeek-V3 每 trillion tokens 仅需 180K H800 GPU hours。FP8 通信优化：MoE up-projection 前 activation 量化为 FP8 → dispatch → FP8 Fprop。组合通信保留 BF16。H800 FP8 Tensor Core GEMM 默认累积精度仅 ~14 bits（K=4096 时最大相对误差近 2%），通过 CUDA Core promotion 解决。与 NVIDIA TransformerEngine 不同，不使用 delayed scaling。未来硬件建议：Tensor Core 原生支持 tile/block-wise group scaling + FP32 累积精度。与 microscaling 格式（MXFP）理念一致。

涉及论文标题：
- DeepSeek-V3 Technical Report

## Group Relative Policy Optimization (GRPO / 组相对策略优化)

术语解释
Group Relative Policy Optimization (GRPO) 是 DeepSeek 系列使用的 LLM 强化学习对齐方法（Shao et al. 2024, DeepSeekMath），无需训练与 policy model 同规模的 critic model。对每个 question q，从旧策略采样一组 G 个输出 {o_i}，以组内奖励的均值和标准差归一化得到 advantage A_i = (r_i - mean(r))/std(r)，使用 PPO 风格的 clipped objective 优化。

术语是什么？
GRPO 的 objective：J(θ) = E[ (1/G) Σ_i min(ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i) - β * D_KL(π_θ||π_ref) ]，其中 D_KL = π_ref/π_θ - log(π_ref/π_θ) - 1（unbiased estimator）。

从算法pipeline角度拆解术语：
```
=== GRPO Training Step ===

for each batch of questions Q:
    for each question q in Q:
        {o_1, ..., o_G} ~ π_θ_old(·|q)         // sample G outputs
        
        for each o_i:
            r_i = RM(q, o_i)                     // rule-based or model-based RM
        
        A_i = (r_i - mean({r})) / std({r})       // group-relative advantage
    
    θ = θ + η * ∇_θ J_GRPO(θ)                   // PPO-style update without critic
```

术语一般如何实现？如何使用？
DeepSeek-V3 使用 GRPO 进行 post-training RL 对齐，结合 rule-based RM（数学确定性答案、LeetCode compiler feedback）和 model-based RM（从 DeepSeek-V3 SFT checkpoints 训练，含 chain-of-thought reward reasoning）。RL 数据覆盖 coding、math、writing、role-playing、QA 等多 domain。GRPO 优势：(1) 消除 critic model 的显存和训练开销（critic 通常与 policy 同规模）；(2) group-relative advantage 自动归一化奖励尺度，无需额外 reward normalization。DeepSeek-V2 和 DeepSeekMath 也使用 GRPO。

涉及论文标题：
- DeepSeek-V3 Technical Report
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model

## Multi-head Latent Attention (MLA / 多头潜在注意力)

术语解释
Multi-head Latent Attention (MLA) 是 DeepSeek-V2 首次提出的注意力机制创新，核心思想是将传统 MHA 的 KV cache 通过低秩压缩为 latent vector，大幅减少推理时的 KV cache 内存占用，提升长序列推理的吞吐量。MLA 是 DeepSeek-VL2 语言模型侧的核心效率优化技术之一。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MLA 的核心原理是将传统 MHA 中"每个 head 独立存储完整 K/V"替换为"共享低维 latent vector + 按需上投影恢复 K/V"。推理时只需为每层存储一个 latent vector c_KV（如 rank=512）而非所有 head 的完整 K/V（维度 = head_count × head_dim）。具体来说：
- **Down-Projection**: c_KV = W_DKV · h_t（将 d_model 的 hidden state 压缩到 latent_dim，如 512 维）。DeepSeek-VL2-Small (d=2048) 和 DeepSeek-VL2 (d=2560) 均使用 rank=512，压缩比约 4-5×。
- **Up-Projection**: K_C = W_UK · c_KV, V_C = W_UV · c_KV（从 latent 恢复各 head 的 K/V 表示）。
- **Decoupled RoPE**: 由于 MLA 的低秩压缩与 RoPE 位置编码不兼容（RoPE 作用于 key 的每个 head 维度，但 latent KV 压缩后无法直接施加 per-head RoPE），MLA 采用解耦设计——额外开辟小维度 k_R = RoPE(W_KR · h_t) 作为带位置信息的 key 补充，通过 dot-product 与压缩后的 K_C 组合。
- **Matrix Fusion Trick**: 推理时通过矩阵乘法结合律将 W_Q 与 W_UK 预融合，避免显式 up-project K。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== MLA Forward Pass (per layer, per token) ===
Input: h_t ∈ R^d  (d=2048 Small / d=2560 DeepSeek-VL2, or d=5120~7168 DeepSeek-V2/V3)

// Stage 1: Latent Compression
c_KV = W_DKV · h_t       // [la'tent_dim=512], down-projection
c_Q  = W_DQ · h_t        // optionally compress Q too (DeepSeek-V2/V3)

// Stage 2: Up-Projection for K and V
K_C = W_UK · c_KV        // [head, dimeqk], up-projected compressed key
V_C = W_UV · c_KV        // [head, dimv], up-projected value

// Stage 3: Decoupled RoPE (hybrid design)
K_R = RoPE(W_KR · h_t)   // [d_rope], small dimension, per-head positional key
Q = W_Q · h_t            // [head, dimeqk + d_rope], query (combined)
// Split Q into Q_C (matching K_C) and Q_R (matching K_R) for dot-product

// Stage 4: Attention with combined scores
scores = Q_C · K_C^T + Q_R · K_R^T   // decoupled dot-product
scores = Softmax(scores / sqrt(dimqk))
output = scores · V_C

// KV Cache: only store c_KV (latent_dim=512) + K_R (d_rope small)
// vs MHA: store full K and V per head (head * dim per head)
```
MLA 的 KV cache 节省比例约为 latent_dim / (head_kv × head_dim)。对 DeepSeek-VL2 (d=2560, head=32, latent=512)，MLAvs MHA 的 KV cache 节省约 512/(32×80)≈20%。在长序列推理（如 VLM 处理高分辨率图像产生大量 visual tokens) 场景下尤为关键。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
FlashMLA（DeepSeek 官方，1.7k 行 CUDA）专门针对 H100/H800 优化 MLA 解码 kernel，利用 TMA 异步加载和 Tensor Core MMA；Triton 也有 MLA 实现。MetaAttention 框架通过 Parallel Pattern + 自定义 Q/K/V shape 支持 MLA（~90 行代码）。开源：https://github.com/deepseek-ai/FlashMLA。

在 DeepSeek-VL2 中，Small 和 DeepSeek-VL2 变体使用 MLA (rank=512)，而 Tiny (d=1280) 仍使用标准 MHA，因为 Tiny 的 hidden dim 较小，MLA 的压缩收益有限。MLA 在 VLM 场景中的关键价值在于：高分辨率图像动态 tiling 产生大量 visual tokens（可达 3000+），MLA 可显著减少 visual tokens 对应的 KV cache 压力。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
- DeepSeek-V3 Technical Report

## Dynamic Tiling Vision Encoding (动态分块视觉编码)

术语解释
Dynamic Tiling Vision Encoding 是 DeepSeek-VL2 提出的高分辨率图像编码策略，将不同宽高比的高分辨率图像动态切分为多个固定大小（384×384）的 local tiles，配合一个全局缩略图 tile，通过共享的 SigLIP 视觉编码器处理所有 tile，在保持图像细节的同时控制视觉 token 总数。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Dynamic Tiling 的核心替代了 DeepSeek-VL 的 hybrid encoder（SigLIP-384 粗粒度 + SAM-B-1024 细粒度，仅支持固定两种分辨率），通过以下步骤处理任意宽高比的图像：(1) 候选分辨率选择：从 C = {(m·384, n·384) | 1≤m,n≤9} 中选择最小 padding 面积的 (m*, n*) 作为目标分辨率；(2) 图像 resize + tile 切分：resize 到目标分辨率保持宽高比后 padding，切分为 m*×n* 个 384×384 local tiles + 1 个 384×384 global thumbnail tile；(3) 共享编码器：所有 tile 通过同一个 SigLIP-SO400M-384 编码（27×27=729 visual embeddings × 1152 dim/tile）；(4) Token 压缩：2×2 pixel shuffle 将每 tile 从 27×27 压缩到 14×14=196 tokens；(5) 序列构建：通过 <tile_newline> 和 <view_separator> special tokens 组织 global 和 local tiles 的 2D 空间结构。最大 tile 数 9×9+1=82，最大 visual tokens 约 82×196≈16,000（实际远小于此，因大多数图像不需要 9×9 tiling）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== Dynamic Tiling Algorithm ===
Input: image I (H, W), base_res=384, max_grid=9

// Step 1: Select best resolution minimizing padding
best_pad = inf
for m in 1..9, n in 1..9:
    scale = min(m*384/H, n*384/W)
    rH, rW = H*scale, W*scale
    pad = m*384 * n*384 - rH * rW
    if pad < best_pad:
        best_pad, m*, n* = pad, m, n

// Step 2: Resize and pad
I_resized = resize(I, (m*·384, n*·384), keep_aspect=True)

// Step 3: Tile generation
thumbnail = resize(I, (384, 384))  // global view
local_tiles = split_into_grid(I_resized, m*, n*)  // m*×n* tiles of 384×384

// Step 4: Encode each tile (shared SigLIP)
tiles = [thumbnail] + local_tiles  // 1 + m*·n* tiles
for tile in tiles:
    v = SigLIP(tile)               // output: 27×27×1152 = 729 embeddings
    v = PixelShuffle_2x2(v)        // 27×27 → 14×14, 196 tokens × 4608

// Step 5: Build visual token sequence
// Global thumbnail: append <tile_newline> per row → 14×15 = 210 tokens
// Local grid: (m*·14) × (n*·14) grid + n*·14 <tile_newline> row separators
// Full: [global_210] + <view_separator> + [local_grid]
// Multiple images (>2): disable dynamic tiling (use thumbnail only)
```
与 DeepSeek-VL 的 hybrid encoder 对比：dynamic tiling 统一使用单一 SigLIP 编码器（而非两个编码器融合），支持 1-81 个 tile 动态自适应（而非固定两种分辨率），视觉 token 数随分辨率线性增长（而非平方，因 SigLIP 使用 local attention）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Dynamic tiling 的实现思路广泛出现于高分辨率 VLM（如 LLaVA-NeXT, InternVL2, Qwen2-VL, NVLM）。共性：(1) 需要 vision encoder 支持 variable input chain（tile 数量可变）；(2) 训练时需做 image tile load balancing（不同图像 tile 数差异大，需在不同 data parallel rank 间均衡负载）；(3) 推理时需考虑 tile 数的性能影响（更多 tile=更多 visual tokens=更慢解码）。DeepSeek-VL2 多图场景（>2）禁用 dynamic tiling 正是出于 context length 和计算效率的考量。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

## Vision-Language Adaptor (视觉-语言适配器)

术语解释
Vision-Language Adaptor (VL Adaptor) 是 LLaVA-style VLM 中的连接模块，负责将 vision encoder 的视觉特征投影到 LLM 的文本 embedding 空间，实现视觉和语言两种模态的特征对齐。DeepSeek-VL2 的 VL Adaptor 由 2×2 pixel shuffle 压缩 + 2-layer MLP 组成。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
VL Adaptor 的核心功能：(1) 维度对齐：将 vision encoder output（如 SigLIP 的 1152/4608 dim）投影到 LLM embedding dimension（如 1280/2048/2560）；(2) Token 压缩：通过 pixel shuffle 将视觉 token 数减少 4×（27×27→14×14=196 tokens/tile），降低后续 LLM 的计算负担；(3) 模态桥接：MLP 学习从视觉特征空间到语言特征空间的非线性映射。在 DeepSeek-VL2 中，VL Adaptor 还负责插入 special tokens (<tile_newline>, <view_separator>) 来编码 tile 的 2D 空间结构信息。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== VL Adaptor Forward Pass (per tile) ===
Input: v ∈ R^{27×27×1152}  (SigLIP output per tile)

// Step 1: 2×2 Pixel Shuffle (token compression)
// Reshape spatial dimensions: 27×27×1152 → 14×14×4608
v_compressed = PixelShuffle_2x2(v)   // 196 tokens, dim=4608

// For global thumbnail tile:
//   Append <tile_newline> after each row → 14×15=210 tokens
// For local tiles grid:
//   Append <tile_newline> after final column to mark row endings

// Step 2: MLP Projection
for each visual token t_i in visual_sequence:
    h_i = MLP_2layer(t_i)   // 4608 → d_LLM (1280/2048/2560)

// Step 3: Combine with text tokens
full_sequence = [h_visual | <view_separator> | h_text]  // ready for LLM
```

VL Adaptor 变体：(a) 简单线性投影（LLaVA-1.5）；(b) Q-Former / Perceiver Resampler（BLIP-2, InstructBLIP, Qwen-VL）——使用可学习的 query tokens 对视觉特征做交叉注意力，输出固定数量的 tokens；(c) MLP 投影 + pixel shuffle（InternVL2, DeepSeek-VL2）；(d) MLP 投影 + convolution compression（Qwen2-VL）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VL Adaptor 在训练的三阶段中扮演不同角色：(1) VL Alignment 阶段——LLM 冻结，仅训练 vision encoder + VL Adaptor，快速建立模态间的 embedding 空间对齐；(2) VL Pretraining 阶段——全参数训练，VL Adaptor 随整体模型一起调优；(3) SFT 阶段——继续全参数训练。这种 staged training 是 LLaVA-style VLM 训练的标准范式。训练 loss 仅计算在文本 token 上（包括 visual token 后的 answer token 和 special token），不计算在 visual token 上。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

## Visual Grounding in VLMs (VLM中的视觉定位)

术语解释
Visual Grounding 在 VLM 中指模型基于自然语言描述（类别名、特征描述或指代表达），在图像中定位并输出对应目标物体的 bounding box 坐标的能力。DeepSeek-VL2 将视觉定位作为新增能力引入，通过 special tokens (<|ref|>, <|/ref|>, <|det|>, <|/det|>, <|grounding|>) 在文本序列中编码定位信息。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Visual Grounding 使 VLM 不仅能"看"图像和回答文字问题，还能"指"出图像中的具体位置，是通往 embodied AI 和 visual agent 应用的关键能力。DeepSeek-VL2 实现三类 grounding：(1) Referring Expression Comprehension (REC)——给定 "cat" 或 "the leftmost person"，输出 bounding box；(2) Grounded Conversation——在对话回复中引用具体目标位置（如 "Two <|ref|>dogs<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|> are running"）；(3) In-context Visual Grounding——给定第一张图中参照目标（可能被 visual prompt 如红框高亮），在第二张图中定位同类目标。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== Visual Grounding Data Format (DeepSeek-VL2) ===
// Format A: Object Localization
Prompt:  "Locate <|ref|>car<|/ref|> in the given image."
Target:  "<|ref|>car<|/ref|><|det|>[[x1,y1,x2,y2],...]<|/det|>"

// Format B: Grounded Conversation
Prompt:  "<|grounding|>Can you describe the content of the image?"
Target:  "Two <|ref|>dogs<|/ref|><|det|>[[120,80,340,350]]<|/det|> are running..."

// Format C: In-context Grounding (2 images)
Prompt:  "<|grounding|>The first image shows an object within the red 
          bounding box. Please identify the same category in the 2nd image."
Target:  "<|ref|>cat<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>"

// Coordinate Normalization: [x1,y1,x2,y2] ∈ [0,999] 
// (top-left, bottom-right), normalized to image resolution
```

Grounding 输出的 bounding box 坐标归一化到 [0, 999]（共 1000 个 bin），模型通过 next-token prediction 学习生成坐标数字。训练时引入 negative samples（图中不含目标物体时 model 不应输出任何 box）增强鲁棒性。DeepSeek-VL2 实现了 emergent generalization——训练主要来自自然场景图像，却能在 meme、动漫等域做 grounding。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
VLM 中实现视觉定位有几种主流范式：(a) 文本坐标范式（DeepSeek-VL2, Qwen2-VL, Kosmos-2, Shikra, Ferret）——将 bounding box 表示为 special token 包裹的文本坐标，利用 LLM 的 next-token prediction 自然生成；(b) 额外检测头范式（Grounding DINO, Florence-2）——额外训练 detection head 输出 box regression；(c) 定位 token 范式（Groma, Molmo）——引入额外的 <loc> 等定位专用 token。文本坐标范式的优势是无需额外参数量，但与 LLM 的自然分布（自然语言而非数字序列）存在 gap，需要大量 grounding 数据 training。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

## Pixel Shuffle for Visual Token Compression (像素重排用于视觉Token压缩)

术语解释
Pixel Shuffle（也称为 sub-pixel convolution 或 depth-to-space）是一种将空间维度上采样/下采样与通道维度变换结合的操作。在 VLM 中，Pixel Shuffle 被用于压缩 vision encoder 输出的视觉 token 数量，减少 LLM 需处理的 token 总数。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Pixel Shuffle 将 shape 为 [H, W, C·r²] 的 tensor 重排为 [H·r, W·r, C]（上采样模式）或反向操作 [H, W, C] → [H/r, W/r, C·r²]（下采样/压缩模式，即 inverse pixel shuffle 或 space-to-depth）。DeepSeek-VL2 使用 2×2 inverse pixel shuffle：将 SigLIP 输出的 27×27×1152 feature map 压缩为 14×14×(1152×4)=14×14×4608，token 数减少 4×（729→196 per tile），但每个 token 的维度增加 4×，保持了总信息量。操作本质是：将 2×2 空间邻域的 4 个像素各 1152 维"折叠"到通道维度，得到 4608 维的单个 token。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
=== 2×2 Inverse Pixel Shuffle (Space-to-Depth) ===
Input:  X [H=27, W=27, C=1152]  (SigLIP output, H,W must be even-ish)
Output: Y [H'=14, W'=14, C'=4608]  (after discarding last row/col if odd)

// NumPy/PyTorch equivalent
// Step: reshape → transpose → reshape
H', W' = H//2, W//2   // 27//2=13, but 27-1=26→13; actually paper says 14×14
// For odd dimensions, discard or pad first
X_crop = X[:26, :26, :]  // crop to 26×26 (nearest even)
X_reshaped = X_crop.reshape(13, 2, 13, 2, 1152)
X_transposed = X_reshaped.transpose(0, 2, 1, 3, 4)  // (13, 13, 2, 2, 1152)
Y = X_transposed.reshape(13, 13, 4608)  // notation: 14×14 in paper

// Equivalent PyTorch op:
Y = torch.nn.functional.pixel_unshuffle(X.permute(0,3,1,2), downscale_factor=2)
```

在 VLM pipeline 中，pixel shuffle 的位置位于 vision encoder 之后、VL Adaptor MLP 投影之前（Intermediate compression）。更激进的方法（如 VisionZip, VisionSelector）进一步压缩到更少 token，如 2-64 tokens，用于极大降低计算量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch 内置：torch.nn.PixelShuffle(r) 用于上采样，torch.nn.PixelUnshuffle(r) 用于下采样。TensorFlow 等价：tf.nn.depth_to_space / tf.nn.space_to_depth。因子 r=2 最为常用（4× token 压缩），过大可能损失空间信息。Pixel shuffle 的优势是零参数、零额外计算（纯内存重排），非常适合 VLM 的压缩需求。InternVL2, MiniCPM-V 等也使用类似操作。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

## SigLIP Vision Encoder（SigLIP 视觉编码器）

术语解释
SigLIP (Sigmoid Loss for Language-Image Pre-training) 是 Google DeepMind 提出的视觉-语言预训练方法，使用 sigmoid loss 替换 CLIP 的 contrastive softmax loss，在 batch 规模上更鲁棒且对 batch size 不敏感。DeepSeek-VL2 使用 SigLIP-SO400M-384 作为 vision encoder。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SigLIP 的核心创新：将 CLIP 的多分类 softmax loss（需在 batch 内对所有 negative pairs 做归一化）替换为独立的二分类 sigmoid loss —— 每对 (image, text) 独立判断是否匹配，其他对作为独立的 negative。SigLIP Loss = -(1/(|B|)) Σ [log σ(z_ii · t_exp) + Σ_j≠i log(1-σ(z_ij · t_exp))]，其中 z_ij = f_img(I_i)·f_txt(T_j)/τ。优势：(1) 不依赖 large batch size（CLIP 需 32k+ batch，SigLIP 可小 batch 训练）；(2) 每个 negative pair 独立处理，对于 noisy image-text pairs 更鲁棒；(3) 对 batch 内负样本分布不敏感。SigLIP 训练出的 vision encoder 在 VLM 任务（尤其是 OCR/文档理解）上表现优异。DeepSeek-VL2 使用的 "SO400M" 变体是 SigLIP 中最大的公开模型之一（~400M params）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
SigLIP 在 VLM 中的应用：作为 frozen 或 finetuned vision encoder，将图像编码为 visual tokens 供给 LLM。DeepSeek-VL2 在所有训练阶段（Alignment/Pretraining/SFT）均对 SigLIP encoder 进行 finetuning（vision encoder LR multiplier = 0.1× LLM LR），而非保持 frozen。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源实现：Google 官方 tf/keras 仓库 + HuggingFace models（google/siglip-so400m-patch14-384）。常见替代方案包括：OpenAI CLIP-ViT, EVA-CLIP, DFN-CLIP, InternViT（InternVL 自研）。VLM 选择 vision encoder 的考量：(1) 分辨率支持（SigLIP-384 固定 384×384，需配合 dynamic tiling 实现高分辨率）；(2) 输出 token 数（384/14=27×27=729 tokens）；(3) OCR 能力（SigLIP 在 OCR 任务上表现较好）。SigLIP 的一个变体 SigLIP2 进一步改进了多分辨率支持和训练效率。

涉及论文标题：
- DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding

## PR-MoE (Pyramid-Residual Mixture-of-Experts)

术语解释
PR-MoE 是 DeepSpeed-MoE 提出的混合专家架构，将 Pyramid-MoE（深层更多专家）和 Residual-MoE（固定 MLP + 可变专家作为残差修正）结合，在不牺牲模型质量的前提下减少 MoE 参数至 3x。

术语是什么？
PR-MoE 由两个独立验证的设计观察推动：

- **Phenomenon-I（Pyramid-MoE 的基础）**：在 CV 中已知浅层学通用特征、深层学任务特定特征，但在 NLP/MoE 中未被验证。论文通过对比 First-Half-MoE（前一半层含 MoE）和 Second-Half-MoE（后一半层含 MoE），发现后者性能显著优于前者，证明**深层使用 MoE 带来的收益更大**。由此提出 Pyramid-MoE：深层 MoE 层使用更多专家（如 350M+PR-MoE-32/64：前 10 层 32 experts/层，后 2 层 64 experts/层）。

- **Phenomenon-II（Residual-MoE 的基础）**：增加 expert capacity（每 token 激活更多专家，如 Top-2 gating）能提升精度，但 all-to-all 通信量翻倍。论文发现将 Top-2 gating 改为固定 dense MLP + 1 个可变 expert（残差相加），精度等价于 Top-2，但通信量等价于 Top-1（因为仅需传输 1 个 expert 的 token）。这种设计将专家视为对固定 MLP 输出的"误差修正项"。

PR-MoE 是 Pyramid-MoE 和 Residual-MoE 的组合：所有标准 MoE 层替换为 PR-MoE 层。

从算法pipeline角度拆解术语：
```
# PR-MoE Layer 前向计算（per token）
Input:  hidden_states h ∈ R^{M}

# Step 1: Attention（与标准 Transformer 相同）
h = SelfAttention(h)

# Step 2: Residual-MoE
h_mlp = W2_fixed @ GeLU(W1_fixed @ h)           # 固定 dense MLP（所有 token 共享此路径）
                                                    # W1_fixed: [4M, M], W2_fixed: [M, 4M]
gate_logits = W_gate @ h                           # [num_experts]，当前层的 expert 数（可变）
expert_id = argmax(Softmax(gate_logits))           # Top-1 gating
h_expert = W2[expert_id] @ GeLU(W1[expert_id] @ h) # 选中的 expert FFN
h = h + (h_mlp + h_expert)                         # 残差连接：固定 MLP 输出 + 专家输出

# Pyramid-MoE 配置示例（350M+PR-MoE-32/64, 24 layers, 12 MoE layers）：
# Layers  1-10 (MoE layers 1-10): 32 experts per layer
# Layers 11-12 (MoE layers 11-12): 64 experts per layer
# 深层使用 2x 专家
```

关键训练设计：
由于不同层有不同 expert 数，传统单一 expert parallelism degree 不再高效。DeepSpeed 实现 multi-expert + multi-data parallelism：128 GPUs 上，32-expert 层使用 {EP=32, DP=4}，64-expert 层使用 {EP=64, DP=2}，128-expert 层使用 {EP=128, DP=1}。每 GPU 始终保持恰好 1 expert，避免 load imbalance 和 batch size 降低。

术语一般如何实现？如何使用？
- 开源：https://github.com/microsoft/DeepSpeed
- 替换标准 MoE layer 为 PR-MoE layer，API 与 DeepSpeed MoE API 兼容
- 训练时需配置 multi-expert + multi-data parallelism（自动通过 DeepSpeed runtime 处理）
- 推理时 PR-MoE 的固定 MLP 路径可与 attention 等 non-expert 操作一起通过 tensor-slicing 并行

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---

## Mixture-of-Students (MoS)

术语解释
Mixture-of-Students (MoS) 是 DeepSpeed-MoE 提出的 MoE-to-MoE 知识蒸馏方法，通过 Staged Knowledge Distillation 将大 MoE 教师模型蒸馏到小 MoE 学生模型（减少层数，保持 MoE 架构），保留稀疏推理优势。与 MoE-to-dense 蒸馏不同，MoS 学生仍为 MoE 结构。

术语是什么？
传统 KD 大多用于 dense 模型或将 MoE 蒸馏为 dense（丢失稀疏推理加速）。MoS 的核心创新：(1) 教师和学生均为 MoE（PR-MoE），学生仅减少深度（如 24→21 层），保留专家结构；(2) 发现全程 KD 在预训练后期伤害精度（学生容量不足导致 underfitting：无法同时最小化 CE loss 和 KD loss），提出 **Staged KD**：前 400K steps 使用 KD + CE loss，之后停用 KD 仅优化 CE loss。

从算法pipeline角度拆解术语：
```
# MoS Staged KD 训练流程
# Teacher: 1.3B+PR-MoE+L24 (31B params, 24 layers)
# Student: 1.3B+PR-MoE+L21 (27B params, 21 layers, 12.5% depth reduction)

For step = 1 to total_steps:
    x, y = next_batch()
    
    teacher_logits = Teacher(x)                    # teacher inference (no grad)
    student_logits = Student(x)                    # student forward
    
    L_CE = CrossEntropyLoss(student_logits, y)     # 标准语言模型损失
    L_KD = KLDivergence(student_logits, teacher_logits)  # 蒸馏损失
    
    if step <= 400K:
        L = L_CE + α * L_KD                        # Staged KD Phase 1: 使用蒸馏
    else:
        L = L_CE                                    # Staged KD Phase 2: 仅标准 LM loss
    
    L.backward()
    optimizer.step()

# 关键发现：Full KD（全程使用 KD）最终精度低于 No KD 的 PR-MoE baseline
# Staged KD 解决了 underfitting，学生保留 99.1-99.5% 教师性能
```

术语一般如何实现？如何使用？
- 开源：https://github.com/microsoft/DeepSpeed (DeepSpeed-MoE 组件)
- 知识蒸馏损失：L = L_CE + α·L_KD，其中 L_KD 为 KL 散度
- 关键超参数：KD 停止步数（如 400K steps）、权重 α
- 学生模型深度减少需配合 Staged KD 才能保持精度；直接减少深度不进行 KD 会导致显著精度损失（LAMBADA 下降 1.3 点，BoolQ 下降 7.5 点）

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---

## Staged Knowledge Distillation (阶段式知识蒸馏)

术语解释
Staged Knowledge Distillation 是 DeepSpeed-MoE 在 MoS 中发现的 KD 策略：蒸馏仅在预训练初期（如前 400K steps）进行，后期停用 KD 仅优化标准 LM loss，解决学生模型容量不足导致的 underfitting 问题。

术语是什么？
标准 KD 在整个训练过程中同时优化任务损失和蒸馏损失（L = L_CE + α·L_KD）。DeepSpeed-MoE 发现对于 MoE 预训练的蒸馏，全程 KD 在后期反而伤害精度——因为 PR-MoE 学生已经通过减少层数降低了容量，进一步最小化 KD loss 迫使学生放弃对标准 LM loss 的优化（underfitting）。Staged KD 通过在训练后期停用 KD loss，允许学生在预训练后期专注于标准 LM loss，避免 underfitting。

从算法pipeline角度拆解术语：
公式：
$$\mathcal{L}_{\text{staged}} = \begin{cases} \mathcal{L}_{CE} + \alpha \cdot \mathcal{L}_{KD}, & \text{if step } < K \\ \mathcal{L}_{CE}, & \text{otherwise} \end{cases}$$
其中 K=400K 为 KD 停止步数，α 为加权系数。

Staged KD 有效性验证（Table 5, 350M+PR-MoE）：
- No KD (PR-MoE+L21): LAMBADA 62.33, BoolQ 52.35（baseline 无蒸馏直接减层）
- Full KD (全程): LAMBADA 61.56, BoolQ 57.89（全程蒸馏，LAMBADA 更差）
- Staged KD (MoS): LAMBADA 63.46, BoolQ 58.07（Staged KD 最佳，接近教师 63.65/59.88）

术语一般如何实现？如何使用？
- 实现于 DeepSpeed-MoE 训练流水线中
- K（停止步数）需要根据模型规模和训练总步数调参
- 适用于预训练阶段的 MoE 知识蒸馏场景
- 核心洞察：当学生容量不足时，"少即是多"——减少 KD 的干扰

涉及论文标题：
- DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale

---

## Dropless Routing (无丢弃路由 / Capacity-Free MoE Routing)

术语解释
Dropless Routing 是一种 MoE token 路由策略，在 token-to-expert 分配过程中不通过 capacity factor 强制丢弃超出 expert 容量上限的 token。与标准 GShard 风格 routing（设置 capacity factor，超限 token 被 drop 或通过 residual connection 绕过 expert）相反，dropless routing 保证每个 token 都被其选中的 top-K expert 处理，无需 token dropping。

术语是什么？
传统 Token-Choice Routing（GShard, Switch Transformer）为每个 expert 设置 capacity = capacity_factor × (total_tokens / N_E)，当路由到某 expert 的 token 数超过 capacity 时，多余的 token 被丢弃（不经过该 expert 计算，仅通过 residual connection 传递）。这虽然保证了计算负载可控（无 straggler），但丢弃 token 会损失信息。

Dropless Routing（由 MegaBlocks/dMoE, Gale et al., 2023 引入）移除 capacity 限制，所有 token 都被其选中的 expert 处理。实现方式不是通过 padding（padding 会导致无效计算），而是通过 block-sparse matrix multiplication——将 t token × E expert 的分配矩阵按固定 block size 划分，仅计算非空 block，避免冗余 FLOPs。

从算法pipeline角度拆解术语：
```
# Standard Routing with Capacity Factor (GShard)
for each expert e:
    capacity_e = capacity_factor * total_tokens / N_E
    tokens_e = argtopk(gate_scores, k=min(topK, capacity_e))
    # 超出 capacity 的 token 被丢弃
    output[tokens_e] += expert_e(tokens[tokens_e])

# Dropless Routing (dMoE / MegaBlocks)
# 无 capacity 限制，依赖 block-sparse GEMM
token_expert_map = topK(gate_scores, k)  # 所有 token 都被路由
# 使用 block-sparse matrix multiply: [t, h] x [E, h, 4h]
# 仅计算非零 block，避免 padding 开销
output = block_sparse_gemm(input, experts, token_expert_map)
```

术语一般如何实现？如何使用？
- **MegaBlocks (dMoE)**：Gale et al. (2023) 的 dropless MoE 实现，通过自定义 block-sparse GEMM kernel 高效执行 token-expert 间的稀疏计算。训练时无需 capacity factor 调参，token 不被丢弃。
- **Demons in the Detail (Qiu et al., 2025)**：论文采用 dropless routing 策略（类似 dMoE），以避免 token drop 对不同 Balance BSZ 方法的影响相互混淆。由于使用 dropless 策略，不同 Balance BSZ 设置间的 FLOPs 计算量一致（所有 token 都被处理），但 global-batch balance 可能导致局部负载不均（某些 GPU 处理远超均值的 token 数），引起 ~5.8% 的速度下降。通过额外加微量 micro-batch LBL 可缓解。
- **与 capacity-based routing 的权衡**：Dropless routing 避免了信息损失（无 token 被丢弃），但可能在某些 GPU 上产生局部计算热点，导致训练速度波动（straggler effect）。Capacity-based routing 保证计算负载可预测（有利于 Expert Parallelism），但牺牲信息完整性。

涉及论文标题：
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- MegaBlocks: Efficient Sparse Training with Mixture-of-Experts
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts（使用 dropless MoE 训练，基于 gpt-neox + MegaBlocks 实现）

## Expert Trimming (MoE 专家修剪)

术语解释
Expert Trimming 是对 MoE 模型进行结构化压缩的一大类方法，通过识别和移除冗余的结构化模块（experts、MoE layers、transformer blocks）来减少参数数量、内存占用和计算开销。由 He et al. (2025) 在统一 MoE 压缩框架中提出，与 Expert Slimming（压缩单个 expert 内部权重）互补。

术语是什么？
Expert Trimming 的通用形式：T ← T'，其中 T 为原始专家/层/块集合，T' 为保留的子集。根据移除粒度分为三个层次：
1. **Expert Drop**（fine-grained）：移除部分不重要 expert，保留 router 对剩余 expert 的选择
2. **Layer Drop**（medium-grained）：移除整个 MoE 层（含对应 Norm 模块），消除该层的所有 expert 计算和通信
3. **Block Drop**（coarse-grained）：移除整个 Transformer block（Attention + MoE layer + Norms），同时消除 Attention 计算和 KV-Cache

Expert Drop 虽减少参数量，但保留的 MoE 层内仍有 costly computation 和 complex communication（分布式 All-to-All），speedup 微乎其微（<1% at 12.5% experts dropped）。Layer Drop 和 Block Drop 通过粗粒度移除彻底消除对应计算和通信，speedup 显著提升。

从算法pipeline角度拆解术语：
```
=== Expert Drop ===
# 重要性评分 S(E_i) = (1/|X|) * Σ G_i(x)
# Layer-wise: T'(l) = {E_t^(l)} where S(E_t) ∈ TopK({S(E_i)}, n')
# Global: T'(l) = {E_t^(l)} where S(E_t) ∈ TopK(∪_{j}{S(E_i^(j))}, n'*L)

=== Layer Drop ===
# 移除完整 MoE layer + Norm
S^(NM)_l = mean(cos_sim(x', x' + MoE_l(Norm_l(x'))))  # 层冗余度
# 按 S^(NM) 降序排列，移除 Top-K 层

=== Block Drop ===
# 移除完整 Transformer block (Attention + MoE + Norms)
S^(NM)_l = mean(cos_sim(x^l, y^l))  # block 级冗余度
# 按 S^(NM) 降序排列，移除 Top-K blocks
```

关键发现：MoE 层比 Dense 层更冗余——同深度 Mixtral-8×7B (MoE) vs Mistral-7B (Dense)，Drop 8 layers 时 MoE -7.0 vs Dense -24.3 MMLU。

术语一般如何实现？如何使用？
- Expert Drop：基于路由分选择保留 expert，不需要任何训练；可选 post-finetuning 恢复性能
- Layer Drop / Block Drop：用 calibration data（如 128 samples from C4, seq_len=2048）计算每层/块的 cosine similarity，相似度越高 → 冗余越大 → 优先 drop
- Drop 模式：深层 layers/blocks 优先被 drop（与 Xu et al. 2024 / Men et al. 2024 一致），因为深层更冗余
- 鲁棒性：相似度对 calibration 数据选择鲁棒——不同样本量（128 vs 更多）和不同数据集（C4/Lima/MetaMathQA）的相似度模式一致
- 集成策略：先 Expert Slimming（quantization）后 Expert Trimming（Layer/Block Drop），即"S+T" order，性能稍优于 "T+S"
- 效果：AWQ + Block Drop B5/32 on Mixtral-8×7B: 5.94× speedup, 21.9GB memory, Avg=68.0 (95.1% of baseline)

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

## Expert Slimming (MoE 专家瘦身)

术语解释
Expert Slimming 是对 MoE 模型中单个 expert 内部权重进行压缩的技术，通过应用压缩变换 f(W) 减少每个 expert 的冗余，创建轻量化的"slim experts"。由 He et al. (2025) 在 MoE 统一压缩框架中提出，与 Expert Trimming（移除结构化模块）互补。

术语是什么？
Expert Slimming 专注于单个 expert 的权重变换，不改变 expert 数量。主要方法：
1. **Pruning（剪枝）**：f(W) = M ⊙ W，通过 binary mask M 置零不重要权重。分为 unstructured（任意位置，效果最好但硬件不友好）、semi-structured（如 2:4，每 4 个值中保留 2 个，硬件友好但性能损失大）、structured（整行/列移除）
2. **Quantization（量化）**：f(W) = Quant(W)，将 FP16/FP32 权重转换为 INT4/INT8 等低精度表示，减少内存但保持 FLOPs

论文对比发现：量化优于剪枝——unstructured pruning (50%) 虽能保持 >95% 性能但无法硬件加速，semi-structured (2:4) 硬件友好但性能损失显著。4-bit 量化 (AWQ) 实现 >98% 性能 + 5.08× speedup (Mixtral-8×7B) 且硬件可加速。

从算法pipeline角度拆解术语：
```
=== 统一框架中的 Expert Slimming ===
# 压缩后的 MoE 输出
y = Σ_{i∈T'} G_i · E_i(x | f(W_i))
# T': Expert Trimming 保留的 expert 子集
# f(W_i): Expert Slimming 压缩后的权重

# Pruning
W_i_pruned = M_i ⊙ W_i,   M_i ∈ {0,1}^{d×d_h}  # 二值 mask
半结构化 2:4: 每 4 个连续元素中最多 2 个非零

# Quantization (AWQ)
W_i_quant = AWQ_quantize(W_i, bits=4, group_size=128)
# 推理时: y = x @ W_i_quant, 使用 INT4 GEMM kernel
```

Shared Expert 不可压缩性发现：DeepSeek-MoE-16B 残差 MoE（2 shared + 64 routed）中，pruning 不包含 shared experts 时：Wanda +3.6%, SparseGPT +1.5% 平均精度提升。说明 shared experts 承载更关键的通用知识。

术语一般如何实现？如何使用？
- Pruning: 用 Wanda（activation-based magnitude）或 SparseGPT（Hessian-aware）在 calibration data（128 C4 samples）上一次性剪枝
- Quantization: 用 GPTQ/AWQ 进行 weight-only 4-bit 量化。GPTQ default: 128 Alpaca samples；AWQ default: 128 Pile samples。Group size: 128 (Mixtral) 或 64 (DeepSeek)
- 最佳组合：先 Expert Slimming 后 Expert Trimming ("S+T" order)，量化保持性能 + Layer/Block Drop 增加效率
- 综合效果：AWQ 4-bit + Block Drop B5/32, Mixtral-8×7B: 6.05× speedup, 20GB memory, 92.4% performance

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

## Cosine Similarity Metric for MoE Layer Redundancy (MoE 层冗余的余弦相似度度量)

术语解释
Cosine Similarity Metric for MoE Layer Redundancy 是一种无训练的度量方法，通过计算 Transformer block 的输入输出隐状态之间的 cosine similarity 来评估层的冗余程度，进而决定哪些层/块可以被安全移除。由 He et al. (2025) 提出，用于 MoE 模型的 Layer Drop 和 Block Drop。

术语是什么？
核心思想：如果一层的输出与输入高度相似（cosine similarity ≈ 1），则该层对 token hidden state 的"变换"很小，可能是冗余的。具体定义为：

- S^(M) = (x · y) / (||x|| · ||y||)，where y = MoE(x)：仅 MoE 层的输入输出相似度
- S^(NM) = (x' · y') / (||x'|| · ||y'||)，where y' = x' + MoE(Norm(x'))：含 Norm + MoE + residual connection 的完整效果

论文发现 S^(NM) 比 S^(M) 更能准确反映层冗余度，因为在 Transformer 中 Norm 和 residual connection 是关键组件。单独的 S^(M) 相似度低（移除仅 MoE 不可行），但 S^(NM) 相似度高（Norm+MoE+Residual 整体冗余，可移除）。

从算法pipeline角度拆解术语：
```
# 计算流程
def compute_layer_similarity(model, calibration_data, device):
    for each layer l in range(L):
        similarities = []
        for each batch x in calibration_data (128 samples, seq_len=2048):
            # 记录 block 输入
            x_input = x.clone()  # 残差连接前的 hidden state

            # Forward through Norm + MoE + Residual
            x_norm = layer_norm[l](x)
            x_moe = moe_layer[l](x_norm)
            x_output = x_input + x_moe  # residual connection

            # 计算相似度
            sim = cosine_similarity(x_input.flatten(), x_output.flatten())
            similarities.append(sim)

        S_l = mean(similarities)  # 层的平均冗余度

    # 按 S 降序排列 → 高相似度层优先被 drop
    drop_order = argsort(S, descending=True)
    return drop_order

# cosine_similarity(a, b) = (a·b) / (||a||·||b||)
```

鲁棒性验证：
- 样本数：32→128→1024 samples from C4，相似度模式稳定
- 数据集：C4 (pretraining)、Lima (instruction tuning)、MetaMathQA (math) 三种分布的相似度模式一致 → 度量对数据分布鲁棒

术语一般如何实现？如何使用？
- Calibration: 128 random samples from C4, sequence length=2048，sufficient for stable similarity estimation
- 应用流程: 加载模型 → forward calib data 收集每层 hidden states → 计算每层平均 S^(NM) → 按 S^(NM) 排序 → 移除 Top-K 层/块
- 与 MoE routing 的关系：S^(NM) 仅依赖 hidden states，不依赖 router 决策；drop 后剩余层的 router 无需修改
- Drop 模式发现：深层 layers/blocks 优先被 drop（与 Xu et al., ShortGPT, Men et al., 2024 一致），因为深层主要负责高层语义抽象，冗余度更高
- 拓展：可推广到 dense 模型，但 MoE 模型 dropping 后性能衰减更小（same depth: MoE -7.0 vs Dense -24.3 MMLU at 8 layers dropped）

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

## Post-Compression Fine-tuning (MoE 压缩后微调恢复)

术语解释
Post-Compression Fine-tuning 是对 Expert Trimming 压缩后的 MoE 模型进行轻量 fine-tuning 以恢复性能的技术。压缩后的模型（尤其是 Layer/Block Drop 后）在残差连接路径上出现结构不连续，fine-tuning 帮助剩余 layers 适应新的"跳过连接"模式。

术语是什么？
与完整预训练不同，Post-Compression Fine-tuning 仅需少量高质量指令数据，使用标准 LM loss 继续训练压缩后的模型几个 epoch。由于压缩后的模型已经保留了大部分原始模型的知识，fine-tuning 主要是"适应"而非从头学习。

He et al. (2025) 的设置：在 Alpaca-GPT4 数据集上 full-finetune 3 epochs，lr=8e-6，warmup ratio=0.03，cosine schedule，global batch size=32。

效果：DeepSeek-MoE-16B Block Drop B4/28 后 fine-tuning，性能 gap 从 -5.5% 缩小至 -0.6%（接近原始模型）。Layer Drop L4/28 从 -6.5% 恢复至 -1.0%。

术语一般如何实现？如何使用？
- 数据量需求小（Alpaca-GPT4 ~52K samples），不需要大规模预训练数据
- Full fine-tuning 比 LoRA 更有效（因为层结构发生了变化，需要全参数适应）
- Warmup ratio 0.03 + cosine schedule 避免初始训练不稳定
- 压缩比越高，fine-tuning 恢复效果越显著，但绝对性能仍随压缩率下降
- 适用于 Expert Drop / Layer Drop / Block Drop 所有三种 Expert Trimming 方法

涉及论文标题：
- Demystifying the Compression of Mixture-of-Experts Through a Unified Framework

## Dense Backpropagation / EMA Default Vector for MoE Router

术语解释
Dense Backpropagation 是一种 MoE 路由器训练技术，通过为每个 expert 维护其历史输出 EMA（指数移动平均）的 default vector，在反向传播时为 Router 提供来自所有 N 个 experts 的"dense"梯度信号，而非仅有 Top-K 激活 expert 的稀疏梯度，同时保持前向传播的稀疏计算特性。

术语是什么？
Standard TopK MoE 的 Router 梯度为 ∂y/∂π_i = E_i(x) if i∈TopK else 0，N-K 个未激活 expert 的 Router 行不接收梯度更新。Dense Backpropagation 的目标是近似完整的 dense gradient ∂y/∂π = [E_1(x), ..., E_N(x)]^T（即 Straight-Through Estimator 的理论梯度），而无需实际计算所有 expert 的前向输出。

核心机制：
1. **EMA Default Vector**: 为每个 expert i 维护 Ê_i = β·Ê_i^{(t-1)} + (1-β)·E_i(x)（仅对激活的 expert 更新），近似 expert 输出的期望值 E[E_i(x)]
2. **Dense Forward**: y = Σ π_i · (E_i(x) if i∈TopK else Ê_i)，default vector 参与前向组合
3. **Dense Backward**: ∂y/∂π_i = E_i(x) for i∈TopK, Ê_i for i∉TopK → Router 所有行接收梯度
4. **Error Correction**: 相对于 true dense gradient 的误差 ε_default = (∂L/∂y) Σ_{i∉A} (E_i(x) - E[E_i(x)]) · ∂π_i/∂W，期望为 0

从算法pipeline角度拆解术语：
```
# DefaultMoE Forward + EMA Update (8 experts, TopK=1)
Input: x [B, H], router W [N, H], experts E_0..E_{N-1}
State: EMA_buf [N, H]  # default vectors

# 1. Router forward
pi = Softmax(W @ x)            # [B, N]
A = TopK(pi, K=1)              # indices of selected experts

# 2. Sparse expert computation (only K experts)
y = zeros(B, H)
for i in A:
    activated_x = x[mask[:,i]]  # tokens routed to expert i
    y_i = E_i(activated_x)      # [num_activated_i, H]
    y += gather(pi[:,i]) * scatter(y_i, mask[:,i])
    
    # 3. EMA update with router-weighted average
    weighted_y = pi[mask[:,i], i].unsqueeze(-1) * y_i
    mean_output = weighted_y.sum(dim=0) / pi[mask[:,i], i].sum()
    EMA_buf[i] = beta * EMA_buf[i] + (1-beta) * mean_output

# 4. Dense combination (EMA for non-activated)
for i not in A:
    y += pi[:,i].unsqueeze(-1) * EMA_buf[i]  # [1, H] broadcast

# 5. Backward: dense gradient signal
# dL/d(pi_i) = dL/dy * (E_i(x) for i in A, EMA_buf[i] for i not in A)
# dL/dW[i,:] = sum_b( dL/d(pi_{b,i}) * x[b] )  for ALL i in 0..N-1
```

术语一般如何实现？如何使用？
- **超参数 β**: sparser MoE 需要更低 β（如 32c1: β=0.65, 32c4: β=0.999），因为每个 expert 接收更少 token，default vector 需更快适应
- **Weighted EMA Update**: 按 Router probability 加权更新 EMA，消除 β 的敏感度。不加权时 β=0.9 与 β=0.999 性能差异显著；加权后多个 β 值收敛到相同性能
- **Forward EMA 必要性**: 仅在后向传递中注入 default vector 不如前向+后向都使用。原因是前向使用 default vector 参与模型输出计算使梯度误差项（Eq.9）被 loss 缩小
- **EMA 初始化**: 零初始化优于随机初始化（避免早期噪声信号误导 Router）
- **开销**: O(1) memory per expert × hidden_dim（如 1024 维 × 8 experts × 16 layers ≈ 0.03% 参数增量），throughput 下降 <2%（小模型）或 <0.2%（大模型）
- **训练框架**: gpt-neox + MegaBlocks + liger kernel (Triton)，dropless MoE，AdamW optimizer

涉及论文标题：
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts

---

## Load Balancing Loss / Auxiliary Loss in MoE Training

术语解释
Load Balancing Loss（又称 Auxiliary Loss / Load Balance Loss）是 MoE 训练中附加在主要语言建模损失上的辅助损失项，鼓励 Router 将 token 均匀分配到各 expert，防止某些 expert 被过度使用而其他 expert 完全不参与训练（dead experts）。

术语是什么？
标准形式（Switch Transformer, Fedus et al. 2022）：
$$L_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 f_i = 路由到 expert i 的 token 比例，P_i = expert i 的平均 gating probability，N = expert 总数，α = 辅助损失系数。当 f_i 和 P_i 均为 1/N（完全均匀）时 L_aux 取得最小值 α。

从算法pipeline角度拆解术语：
```
# Standard Auxiliary Loss (per micro-batch)
def load_balancing_loss(gate_probs, topk_indices, N_experts, alpha=0.01):
    # gate_probs: [B, N] softmax router outputs
    # topk_indices: [B, K] selected expert indices
    
    # Expert selection frequency f_i
    mask = one_hot(topk_indices, N_experts).sum(dim=-2)  # [B, N]
    f_i = mask.sum(dim=0) / mask.sum()                    # [N]
    
    # Average gating probability P_i
    P_i = gate_probs.mean(dim=0)                          # [N]
    
    # Load balancing loss
    L_aux = alpha * N_experts * sum(f_i * P_i)
    return L_aux

# Total training loss
L_total = L_lm + L_aux
```

**Globally Reduced Auxiliary Loss (Qiu et al., 2025)**: 将 f_i 的计算从 micro-batch 级别改为 global-batch 级别。跨所有 Data Parallel ranks 同步 f_i（仅 N_E 维向量，通信量极小），用全局 f̄_i 替换本地 f_i。这放松了均衡约束——允许每个 micro-batch 内 expert 使用不均，但整个 global batch 均衡——从而促进 expert domain specialization。

术语一般如何实现？如何使用？
- **系数 α**: 典型值 0.01（Switch Transformer 风格）。α 过小 → expert 负载不均或 dead expert；α 过大 → router 过度均匀分配，抑制 expert specialization
- **变体**: 
  - z-loss: 额外的门控 logit 正则项，稳定训练（DeepSeek-V2/V3）
  - Auxiliary-Loss-Free 策略 (Wang et al., 2024): 通过可学习 expert bias b_i 动态调整路由，完全替代 auxiliary loss
  - Group-Level Load Balancing (ARIA): 对 expert 组施加约束（详见对应条目）
- **DefaultMoE 的使用**: α=0.01，使用 globally reduced auxiliary loss（跨节点计算），不使用 z-loss 和 jitter（该规模下无益）。global-batch LBL 使 baseline 性能显著提升，因此 prior routing method 声称的"free lunch"改进被削弱
- **与 Dense Backpropagation 的关系**: DefaultMoE 的 dense gradient 通过减少 Router 梯度误差进一步提升训练稳定性，允许使用更大 learning rate（9×10⁻⁴ vs baseline 的 7×10⁻⁴）而不出现 loss spike
- **Duo-LLM 的 Budget Loss 变体**: 不同于 per-layer 负载均衡，Duo-LLM 使用全局 Budget Loss 约束所有层的 big 模块总使用比例：L_budget = (mean(P_big across all layers) - target_budget)²。这允许 router 跨层灵活分配计算——某些层可以更多使用 big 模块，另一些层更多使用 small 模块——只要全局满足预算。配合 soft routing（温度 τ 逐渐增大实现硬分配），router 被鼓励发现跨层的复杂路由模式而非 per-layer 均匀分配。

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts
- Dense Backpropagation Improves Training for Sparse Mixture-of-Experts
- Demons in the Detail: On Implementing Load Balancing Loss for Training Specialized Mixture-of-Expert Models
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models
- Duo-LLM: A Framework for Studying Adaptive Computation in Large Language Models
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

**CuMo 中的 bzloss 使用**：CuMo 使用标准 load balancing loss (α_b=0.1) + router z-loss (α_z=0.01)，合称 "bzloss"，分别独立应用于 MLP connector、CLIP vision encoder 和 LLM 的每个 MoE 块。总损失 L = L_ce + 0.1·L_b + 0.01·L_z。CuMo 的消融实验（Table 3）表明加入 bzloss 后在 MMVet 上取得明显提升（32.3 → 33.1），验证了负载均衡对 MoE 多模态训练的正面影响。α_b=0.1 相比标准 α=0.01 更高，论文未解释原因。

**DS-MoE 中的 MI Loss**：DS-MoE (Pan et al., 2024) 引入基于信息论的 Mutual Information (MI) Loss 作为负载均衡替代方案。与 switch loss 的双线性形式不同，MI Loss 分为两项：
1. **最大化 expert 分布熵** H(e) = -Σ p(e) log p(e)：促进全局负载均衡（所有 expert 被均匀使用）
2. **最小化条件熵** H(e|X) = -Σ p(e|x) log p(e|x)：鼓励 Router 对每个 token 产生集中的、确定性的 expert 分配（sparse concentration）
总损失：L_MI = -H(e) + (1/|X|) Σ H(e|X)，总训练 loss：L = L_LM + α·L_MI，其中 α 控制 sparsity 程度。

MI Loss 的特殊优势：(a) 不需要 fixed K——支持训练后灵活选择 inference sparsity 级别；(b) 自我平衡——H(e) 推动均衡、H(e|X) 推动集中，两者形成对抗平衡；(c) 训练后 Router 自然产生 sparsity，可仅凭阈值 ε 或 TopK 选择激活 expert。DS-MoE 的 α 参数：MoA 层 3.5e-4 (1B) / 2e-4 (3B/6B)，MLP 层 6.3e-4 (1B) / 4e-4 (3B) / 2e-4 (6B)。α 越大 → sparsity 越高（模型在高 sparsity 下性能保持更好），但可能在低 sparsity 下性能略差。

---

## DS-MoE (Dense Training, Sparse Inference for MoE)

术语解释
DS-MoE 是 Pan et al. (2024) 提出的 MoE 训练范式创新——训练阶段所有 expert 全激活（dense training），推理阶段仅激活 top-K 或超阈值 expert（sparse inference），搭配 Mutual Information (MI) Loss 实现负载均衡和 sparsity 塑造。核心发现：传统 sparse training 中仅 top-K expert 接收梯度导致 Router 参数更新不完整，是 MoE 参数效率低的根本原因。

术语是什么？
DS-MoE 重新定义了 MoE 的训练-推理关系：
- **Dense Training**：前向时计算所有 N 个 expert 的输出 O = Σ_{i=1..N} S_i · E_i(X)；反向时 Router 梯度包含所有 expert 贡献 ∇S = [E_1(X), ..., E_N(X)]^T ∇O，每个 expert 梯度为 ∇e_i(X) = S_i ∇O。训练框架：PyTorch + FSDP + activation checkpointing。
- **Sparse Inference**：Router 计算 scores S 后，仅激活 top-K 或超阈值 expert：O = Σ_{i∈A} S_i · E_i(X) where A = {i | topK(S, K) or p_i > ε}。使用 SimpleMoE 的 ParallelLinear 操作执行稀疏 expert computation。
- **与 Sparse Upcycling 的区别**：Sparse Upcycling 从 dense checkpoint 初始化后转为 sparse training（train dense → train sparse）；DS-MoE 始终保持 dense training（train dense → deploy sparse）。DS-MoE 的参数效率来自训练阶段的完整梯度信号，而非已有 dense checkpoint 的"遗产"。
- **与 DefaultMoE 的区别**：DefaultMoE 使用 EMA default vector 近似 dense gradient 同时保持前向 sparse；DS-MoE 在前向直接执行 dense computation（所有 expert 全激活）。DS-MoE 更精确（无 EMA 近似误差），但训练开销更大。

从算法pipeline角度拆解术语：
```
# DS-MoE Training Pipeline (per layer, per token)
# 1. Router
S = Softmax(h(X))              # [N], Router 计算所有 expert scores

# 2. Dense Forward (ALL experts)
O = zeros(d_h)
for i in 1..N:
    # Expert FFN: GeLU(X @ W_up_i) @ W_down_i
    E_i = GeLU(X @ W_up_i + b_up_i) @ W_down_i + b_down_i
    O += S[i] * E_i            # weighted sum of ALL experts

# 3. Dense Backward
# Router gradient (dense, no mask)
dL/dS = [E_1, ..., E_N]^T @ dL/dO   # all N experts contribute
# Expert gradient
for i in 1..N:
    dL/dW_up_i, dL/dW_down_i = backprop through E_i, scaled by S[i]

# 4. MI Loss (per batch)
P = mean(S, dim=0)                     # [N], expert probability per batch
H_e = -sum(P * log(P))                 # expert entropy (maximize for balance)
H_cond = mean(-sum(S * log(S), dim=-1)) # per-token conditional entropy (minimize for concentration)
L_MI = -H_e + H_cond                   # MI Loss
L_total = L_LM + alpha * L_MI

# 5. Sparse Inference (post-training)
S = Softmax(h(X))
# Option A: Fixed TopK
A = topK(S, K)                # K = 4 or 6 depending on model/sparsity
# Option B: Threshold
p_norm = S * N                # normalized probability
A = where(p_norm > epsilon)   # epsilon = 0.48 (default)
# Sparse compute
O = sum_{i in A} S[i] * E_i(X)  # ParallelLinear (SimpleMoE)
```

术语一般如何实现？如何使用？
- **训练配置**：DS-MoE-1B (1067M params, N_ffd=32, D_ffd=256, N_att=16), DS-MoE-3B (2846M, N_ffd=32, D_ffd=384, N_att=8), DS-MoE-6B (6343M, N_ffd=32, D_ffd=512, N_att=8)
- **训练成本**：H100×8 (1B, 24h), H100×32 (3B, 64h / 6B, 124h)。训练数据 30B tokens (1B) / 100B tokens (3B/6B)
- **推理 sparsity**：DS-MoE-1B 激活 41% hidden, DS-MoE-3B 激活 34%, DS-MoE-6B 激活 29%（趋势：更大模型 → 更高 sparsity）
- **推理性能**：DS-MoE-6B 在 vLLM 上达到 2.00 req/s (A100), 2.30 req/s (H100)，比 Mistral-7B 快 1.86×，比 DeepSeekMoE-16B 快 1.50×，GPU 内存仅 12.6 GiB
- **代码未开源**（截至 2024.4），使用 SimpleMoE (开源) 进行稀疏推理

涉及论文标题：
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

---

## Mutual Information (MI) Loss for MoE Expert Routing

术语解释
由 Shen et al. (2023, ModuleFormer) 提出并由 Pan et al. (2024, DS-MoE) 用于 Dense Training MoE 的核心损失函数。基于信息论中 Mutual Information 的概念，通过最大化 expert 分布的边际熵 H(e) 和最小化条件熵 H(e|X) 来实现 expert 负载均衡和专家集中。

术语是什么？
MI Loss 的数学形式：L_MI = -H(e) + (1/|X|) Σ_{x∈X} H(e|x)，其中：
- H(e) = -Σ_{i=1..N} p(e_i) log p(e_i)：expert 的边际熵。p(e_i) = mean_{batch}(S_i) 为 batch 内 expert i 的平均 Router 概率。最大化 H(e) → 所有 N 个 expert 被平均使用（负载均衡）。
- H(e|x) = -Σ_{i=1..N} S_i log S_i：给定 token x 条件下 expert 的条件熵。最小化 H(e|x) → Router 对每个 token 产生集中的概率分布（expert concentration / sparsity）。
两项形成"对抗平衡"：负载均衡（maximize H(e)）vs. 专家集中（minimize H(e|x)）。

与标准 switch loss 的区别：
| 维度 | Switch Loss (Fedus 2022) | MI Loss (Shen 2023 / Pan 2024) |
|------|--------------------------|-------------------------------|
| 形式 | L = α·N·Σ f_i·P_i (双线性乘积) | L = -H(e) + (1/|X|)·Σ H(e|x) (信息熵差) |
| 所需信息 | f_i (路由频率) + P_i (平均概率) | 仅 Router scores S |
| K 固定 | 通常需 fixed K | 支持 flexible/inference-time K |
| Sparsity 控制 | 隐式（通过 α 和 K） | 显式（H 项间平衡） |

从算法pipeline角度拆解术语：
```
# MI Loss computation (per micro-batch)
def mi_loss(router_scores, alpha, N_experts):
    # router_scores: [B, N] softmax outputs per token
    p_e = router_scores.mean(dim=0)              # [N], marginal P(e)
    H_e = -sum(p_e * log(p_e))                   # expert entropy
    H_cond = -sum(router_scores * log(router_scores), dim=-1).mean()  # conditional entropy
    L_mi = -H_e + H_cond                         # MI Loss
    return alpha * L_mi

# Total loss
L_total = cross_entropy(logits, labels) + mi_loss(router_scores, alpha, N)
```

术语一般如何实现？如何使用？
- **α 调参**：α 控制 sparsity 程度。DS-MoE 验证 α 越大 → 模型在高 sparsity 下性能保持更好，但可能在低 sparsity 下性能略差。需要 α 在"稀疏度"和"整体性能"间平衡。
- **DS-MoE α 值**：MoA 层 3.5e-4 (1B) / 2e-4 (3B/6B), MLP 层 6.3e-4 (1B) / 4e-4 (3B) / 2e-4 (6B)
- **训练后 sparsity 调整**：MI Loss 训练出的 Router 可灵活切换推理 sparsity 级别（调整 K 或 ε），无需重新训练。DS-MoE-6B 在 24% active hidden (vs default 29%) 时仍可通过调大训练 α 保持性能。
- **源工作**：ModuleFormer (Shen et al. 2023) 使用 MI Loss 训练模块化 LLM；DS-MoE 将其应用于 MoE 大规模预训练场景

涉及论文标题：
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

---

## Expert Sampling Strategy (Threshold / TopK / Threshold-TopK for MoE)

术语解释
DS-MoE 推理阶段从 dense training 的"全激活"切换到 sparse inference 的"部分激活"的策略选择方法。三种方法在 sparsity-性能 trade-off 和部署实用性之间提供不同平衡点。

术语是什么？
三种 expert sampling 策略定义了推理时如何从 N 个 expert 中选择 K 个激活：

1. **Threshold**（自适应阈值）：对每个 token 独立计算归一化概率 p_norm_i = S_i · N。选择所有 p_norm_i > ε 的 expert。优点：per-token 自适应——难度高的 token 自然激活更多 expert，简单 token 激活更少。缺点：batch 中不同 token 激活不同数量的 expert，需要 padding 或复杂调度，不利于 GPU 并行和 batch inference。

2. **TopK**（固定数量）：每层每 token 激活固定 K 个 expert（最高分的 K 个）。优点：batch 内所有 token 激活相同数量 expert，GPU 调度简单高效，适合生产部署。缺点：无法自适应——简单 token 浪费计算，复杂 token 可能不够。

3. **Threshold-TopK**（混合策略）：先用 Threshold 计算每个 token 应激活的 expert 数，再取 batch 内平均值作为统一 K 值进行 TopK 选择。优点：兼顾自适应（per-batch 调整）和 batch 效率（所有 token 统一 K）。缺点：需要先 forward 一次 Router 再决定 K 值。

从算法pipeline角度拆解术语：
```
# 1. Threshold Sampling
S = Softmax(h(X))                   # [B, N]
p_norm = S * N                       # normalized [B, N]
active_mask = p_norm > epsilon       # [B, N] bool
A = where(active_mask)               # variable-length per token
O = weight_sum([E_i(X) for i in A], [S[A[idx]] for idx])  # irregular computation

# 2. TopK Sampling (deployment-friendly)
S = Softmax(h(X))                    # [B, N]
A = topK(S, K)                       # [B, K], same K for all tokens
O = ParallelLinear(X, A, all_expert_weights)  # regular, efficient

# 3. Threshold-TopK (hybrid)
S = Softmax(h(X))                    # [B, N]
p_norm = S * N
per_token_K = sum(p_norm > epsilon, dim=-1)  # [B]
avg_K = round(mean(per_token_K))     # scalar
A = topK(S, avg_K)                   # [B, avg_K]
O = ParallelLinear(X, A, all_expert_weights)
```

术语一般如何实现？如何使用？
- **DS-MoE 默认**：使用 ε=0.48 的 Threshold 策略进行评估（追求最优 PPL 权衡）。部署时使用 TopK 或 Threshold-TopK。
- **性能比较**：DS-MoE-3B WikiText PPL → Threshold (best) > Threshold-TopK (practical) > TopK (deploy-friendly) at same active params
- **实际部署建议**：attention 层使用 dense（sparsity<40% 时 sparse overhead > dense），MLP 层使用 ParallelLinear + TopK
- **K 值选择**：DS-MoE-3B 使用 K=6, DS-MoE-6B 使用 K=4；更大模型可承受更高 sparsity（更小 K）

涉及论文标题：
- Dense Training, Sparse Inference Rethinking Training of Mixture-of-Experts Language Models

## Differentiable Expert Pruning (DiEP)

术语解释
DiEP（Differentiable Expert Pruning）是一种将 MoE 专家剪枝重新表述为连续优化问题的框架。通过定义可学习的 intra-layer importance scores（α）和 inter-layer importance scores（β），将原本指数级增长的离散专家搜索空间转化为可微的连续空间，利用梯度下降实现全局最优的非均匀专家剪枝。由 Bai et al. (2025) 在 NeurIPS 2025 提出，是首个将 differentiable architecture search 思想应用于 MoE 架构的方法。

术语是什么？
DiEP 的核心思想是将专家选择从离散的 binary mask m_i^(l) ∈ {0,1} 转化为连续的加权聚合：

y'^(l+1) = β^(l) · Σ_i ᾱ_i^(l) · FFN_i(x^(l))

其中 ᾱ_i^(l) = softmax(α_i^(l)) 为层内专家重要性（归一化后），β^(l) 为跨层重要性标量。目标函数：

L(α, β) = L_ce(y, F'(x; α, β)) + λ · ∥F'(x; α, β) − F(x)∥_F

包含两部分：(1) Cross-entropy loss 保持任务性能；(2) Reconstruction Regularization（Frobenius norm）鼓励剪枝后模型输出与完整模型一致。λ 为平衡系数（Mixtral 上 λ=0.01）。

优化采用 Alternating Update Strategy：以 α:β = 3:1 的比例交替更新，解耦两个参数组的梯度路径，避免 DiffPruning 等先前方法中的 gradient conflict 问题。

收敛后，全局重要性 s_i^(l) = α_i^(l) · β^(l)，对所有 L×N 个专家统一排序，按 sparsity ratio r 删除底部 K = N·L·r 个最不重要专家。这实现了自动的非均匀剪枝：浅层（1-15 层）因 β 和 α 值更高自然保留更多专家，深层冗余大的层剪去更多专家。

从算法pipeline角度拆解术语：
```
# DiEP Algorithm Pipeline
Input: Full MoE model F, calibration data D_cal (128 samples)
       α_i^(l) = 1 ∀i,l,  β^(l) = 1 ∀l,  λ = 0.01

# Phase 1: Differentiable Search (10 epochs)
for epoch in 1..10:
  for batch in D_cal (batch_size=16):
    # Forward with continuous relaxation
    for layer l in 1..L:
      ᾱ_i^(l) = softmax(α_i^(l))              # ∈ R^N
      for expert i in 1..N:
        h_i = FFN_i(x^(l))                      # expert forward
      y'^(l+1) = β^(l) · Σ_i ᾱ_i^(l) · h_i    # weighted aggregation (Eq.5)
    
    F'(x) = full forward with weighted experts
    L = L_ce(y, F'(x)) + λ · ∥F'(x) − F(x)∥_F  # Eq.7
    
    # Alternating updates (3:1 ratio)
    for step in 1..3:                             # α updates
      α ← α − η_α · ∇_α L(α, β)                 # η_α = 5e-3, cosine schedule
    for step in 1..1:                             # β updates  
      β ← β − η_β · ∇_β L(α, β)                 # η_β = 5e-3

# Phase 2: Global Pruning
K = N_layers × N_experts × r                      # e.g., 32×8×0.5=128 experts to prune
for each expert (l, i):
  s_i^(l) = α_i^(l) · β^(l)                       # global importance (Eq.10)
P = argsort(s)[:K]                                # bottom-K least important
for (l, i) in P:
  remove expert i from layer l                    # permanent pruning

# Phase 3: Optional Merging
for each pruned expert e_p:
  e_retained = argmax CKA(e_p, e_j) over retained experts
  merge e_p into e_retained with CKA-based weight
```
关键维度：α ∈ R^(L×N)，β ∈ R^L。额外参数量仅 ~0.01%。Mixtral 8×7B pruning time: 0.23h（vs NAEE exhaustive search 1.31h）。Deepseek-MoE-16B (64 experts/layer) pruning: 0.28h（vs NAEE ≈94000 days 因搜索空间爆炸不可行）。

术语一般如何实现？如何使用？
- 实现依赖：HuggingFace Transformers + lm-eval-harness。Calibration 仅需 128 C4 samples
- 超参数：epochs=10, batch_size=16, lr=5e-3 (cosine schedule), λ=0.01 (Mixtral), α:β update ratio=3:1
- 剪枝后模型标准 HuggingFace 格式，可直接加载推理
- 论文未提供开源代码，但方法可基于标准 PyTorch 复现（核心仅 ~300 行参数更新逻辑）
- 支持 optional expert merging（CKA-based）进一步提升性能
- 论文未在 DeepSeek-V3/Qwen2.5-Max 等更大模型上验证（计算资源限制）

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

---

## Non-uniform Expert Pruning

术语解释
非均匀专家剪枝是指在不同 MoE 层使用不同剪枝比例的专家剪枝策略。与传统的 uniform pruning（每层删除相同数量专家）不同，非均匀剪枝根据各层专家冗余程度自适应调整。例如在 Mixtral 8×7B 中，浅层（1-15）专家重要性高（处理多样化的低层语言特征如词性标注、局部词序），应保留更多专家；深层（16-32）处理全局语义信息，冗余度更高，可剪去更多专家。

术语是什么？
MoE 模型中不同层的专家冗余程度存在显著差异：CKA 可视化分析显示，浅层 expert-pair similarity 矩阵呈现更复杂的块状结构（专家间分工明确），深层相似度更高（功能趋于同质化）。Uniform pruning 忽略这种差异，对所有层应用相同的 expert sparsity ratio，导致浅层剪枝过度（丢失关键语言特征处理能力）或深层剪枝不足（浪费参数）。

非均匀剪枝的核心挑战是搜索空间巨大：L 层每层 N 个专家，搜索每层保留不同数量专家的组合数呈指数增长。DiEP 通过 differentiable optimization 解决了这个问题：学习 per-layer 重要性 β^(l) 和 per-expert 重要性 α_i^(l)，全局排序后自然产生非均匀剪枝分布。

从算法pipeline角度拆解术语：
```
# Uniform vs Non-uniform Pruning Comparison

# Uniform Pruning (e.g., NAEE, M-SMoE)
for each layer l:
  experts[l] = top-k_experts_in_layer(l, k = N×(1-r))
# 每层保留相同数量专家，忽略跨层差异

# Non-uniform Pruning (DiEP)
scores = []  # global list
for layer l:
  for expert i:
    scores.append((l, i, α_i^(l) × β^(l)))
sorted_scores = sort(scores, by=score, descending=True)
keep = sorted_scores[:N_total×(1-r)]  # 全局选择
# 结果：layer 1 可能保留 7/8 experts, layer 30 可能仅保留 3/8
```
实验结果：Mixtral 8×7B 50% sparsity 下，DiEP 的 MMLU avg 57.9 vs uniform baseline 47.3-54.6，提升 3.3-10.6 个百分点。验证了非均匀剪枝在保持模型性能方面的关键作用。

术语一般如何实现？如何使用？
- DiEP 方法：通过 differentiable search 自动学习非均匀分布
- 也可采用 heuristic：人工设定浅层高 β、深层低 β（如 β=2 for layers 1-16, β=1 for 17-32），但论文表明这种方式无法泛化到不同 MoE 架构
- 剪枝后模型直接运行，无需特殊 runtime 支持
- 可与其他压缩方法（merging, quantization）正交组合

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

---

## Reconstruction Regularization for Expert Pruning

术语解释
重建正则化（Reconstruction Regularization）是一种用于 MoE 专家剪枝的训练目标组件，定义为 Φ(α, β) = ∥F'(x; α, β) − F(x)∥_F，其中 F' 为应用 continuous relaxation 后的 pruned model 输出，F 为完整原始模型输出，∥·∥_F 为 Frobenius 范数。该正则项鼓励剪枝后模型的 token-level hidden states 与原始完整模型保持一致，相当于一种知识蒸馏的形式（无需单独的 teacher forward）。

术语是什么？
在 DiEP 中，Reconstruction Regularization 是与 Cross-Entropy Loss 共同优化的一项，总目标为 L = L_ce + λ·Φ。其作用机制：
1. 原始模型 F(x) 在每个 MoE 层使用全部专家的 FFN 输出加权求和
2. Pruned model F'(x) 使用 ᾱ_i^(l)（softmax 后的可学习重要性）和 β^(l) 对各专家输出加权
3. Φ 计算两者在 hidden state 空间的 Frobenius 距离
4. 梯度反向传播更新 α 和 β，使 F' 的中间表示逼近 F

λ 控制重建正则化的强度。论文在 Mixtral 架构上使用 λ=0.01。消融实验（Figure 8a）显示 λ∈{0.005, 0.01, 0.015, 0.02, 0.03} 中 λ=0.01 最优。

从算法pipeline角度拆解术语：
```
# Reconstruction Regularization computation
def reconstruction_loss(model_full, model_pruned, x, alpha, beta):
    """
    x: input hidden states [batch, seq_len, d_model]
    alpha: intra-layer scores [L, N]
    beta: inter-layer scores [L]
    """
    h_full = x
    h_pruned = x
    total_loss = 0.0
    
    for layer l in range(L):
        # Full model forward (all experts, uniform routing)
        h_full = full_moe_layer(h_full, l)
        
        # Pruned model forward (α-weighted experts)
        h_pruned_layer = 0
        for expert i in range(N):
            expert_out = FFN_i(h_pruned)
            h_pruned_layer += softmax(alpha[l])[i] * expert_out
        h_pruned = beta[l] * h_pruned_layer + h_pruned  # residual
        
        # Layer-wise or end-to-end
        total_loss += frobenius_norm(h_pruned - h_full)
    
    return total_loss

# Full objective
loss_total = cross_entropy(y_pred, y_true) + lambda_reg * total_loss
```

术语一般如何实现？如何使用？
- 计算开销：需额外完整模型前向传播一次（或预先缓存完整模型的 hidden states）
- 校准数据仅需 128 samples，正则化使得在小样本下也不会过拟合
- 类似于 Knowledge Distillation 的 feature-level alignment，但不依赖 teacher soft labels
- 可视为一种 self-distillation：完整模型自身作为 teacher，pruned version 作为 student
- 论文附录验证：即使只用 32 calibration samples，DiEP 也能避免性能崩溃（归因于 reconstruction regularization 的约束作用）

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

---

## Centered Kernel Alignment (CKA) for Neural Network Representation Similarity

术语解释
CKA（Centered Kernel Alignment）由 Kornblith et al. (ICML 2019) 提出，是一种用于比较神经网络层间表示相似度的度量方法。CKA 基于 Hilbert-Schmidt Independence Criterion (HSIC)，对正交变换和神经元置换具有不变性——这是比较神经网络表示的关键特性，因为神经元排列不应影响网络功能。DiEP 论文使用 CKA 在两个场景：(1) 可视化不同 MoE 层内和层间的 expert-pair 相似度，作为非均匀剪枝的 motivation；(2) 计算 expert skipping 中 γ₂ 参数（专家输出相似度与平均相似度的比值）。

术语是什么？
Linear CKA 的计算公式：

CKA_linear(X, Y) = ∥X Y^T∥_F² / (∥X X^T∥_F · ∥Y Y^T∥_F)

其中 X ∈ R^(n×p₁)、Y ∈ R^(n×p₂) 为两个层的激活矩阵（n 样本数，p 特征维度）。完整 CKA（含 RBF kernel）流程：
1. 计算 Gram 矩阵 K = XX^T, L = YY^T
2. 使用 centering matrix H = I_n − (1/n)11^T 中心化
3. HSIC(K, L) = tr(K H L H) / (n−1)²
4. CKA(K, L) = HSIC(K, L) / √(HSIC(K,K) · HSIC(L,L))

CKA ∈ [0, 1]，1 表示完全相同的表示结构，0 表示正交。

从算法pipeline角度拆解术语（DiEP 中的使用）：
```
# CKA-based Expert Similarity in DiEP

# 1. Intra-layer CKA (每个 layer 内的 expert-expert 相似度矩阵)
for layer l in 1..L:
    for expert_i, expert_j in pairs(N):
        # 在校准数据上收集 expert 输出
        X_i = collect_expert_outputs(expert_i, D_cal)  # [n_samples, d_model]
        X_j = collect_expert_outputs(expert_j, D_cal)
        # Linear CKA
        CKA[l][i][j] = ∥X_i X_j^T∥_F² / (∥X_i X_i^T∥_F · ∥X_j X_j^T∥_F)

# 2. Inter-layer CKA (相邻层 expert 之间的相似度)
for layer l in 1..L-1:
    for expert_i in layer l, expert_j in layer l+1:
        CKA_inter[l][i][j] = cka_similarity(E_i^(l), E_j^(l+1))

# 3. Adaptive Skipping γ₂ 计算
γ₂ = ρ(y_e0, y_e1) / mean(ρ(y_ei, y_ej))  # 专家输出 CKA 相似度比
# 其中 ρ = linear CKA between expert outputs
# γ₂ > 1 → 两专家特别相似 → 更有可能跳过 e1
```

术语一般如何实现？如何使用？
- PyTorch 实现可参考 github.com/RistoAle97/centered-kernel-alignment
- DiEP 中使用 CKA 做 pre-pruning analysis（可视化动机），不参与 training 计算（仅推理时 skipping 使用）
- CKA 优点：对正交变换不变、可跨维度比较、比 CCA 特异性更好
- CKA 局限：对低方差主成分不敏感（Ding et al. 2021 指出 CKA 主要反映高方差维度的相似性）
- 在 MoE 剪枝上下文中，也可用 cosine similarity 替代（如 Expert Trimming 论文），但 CKA 能捕捉更丰富的结构性相似关系

涉及论文标题：
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

## Dual Sparsity in MoE (MoE 双重稀疏性)

术语解释
MoE 推理中的"双重稀疏性"指 tensor-level sparsity（张量级，即 Top-K 专家选择的不均衡）和 neuron-level sparsity（神经元级，即 SwiGLU FFN 内部激活值分布不均）在不同粒度上同时存在，共同决定计算效率与精度。每个 token 的 FFN 输出由 gating score × activation value 联合调制，两种稀疏性协调利用可在 ~25% drop rate 下仅损失 0.08-0.28% accuracy。

术语是什么？
DualSparse-MoE 论文的核心观察：对 MoE 模型做一次推理前向，沿 expert × neuron 维度可视化 accumulated absolute activation values，发现 (1) Tensor-Level Sparsity（y 轴）：不同 expert 被激活频率极不均衡（高负载 expert 处理大量 token，低负载仅处理极少 token）；(2) Neuron-Level Sparsity（x 轴）：每个 expert 内部 neuron 的 |Swish(x·W₁) ⊙ (x·W₃)| 高度不均（少数 neuron 贡献大部分 output magnitude，大量 neuron 接近零但非硬零，因 SwiGLU 无 ReLU 般硬零）。Tensor-level 用于 coarse-grained expert dropping（1T-Drop/2T-Drop），Neuron-level 用于 fine-grained major/minor reconstruction。Profiling 发现低负载 expert 有大量负 accumulated gate value，而高负载 expert 罕见 → 暗示两种稀疏性存在内在关联。

从算法pipeline角度拆解术语：
```
# Double-sparse observation
For each MoE layer:
  heatmap = zeros(E, d_ffn)
  For each token t:
    s = TopK(Softmax(t·W_g), K)
    For each activated e:
      gate_out = Swish(t·W1_e)
      up_out = t·W3_e
      heatmap[e, :] += |gate_out ⊙ up_out|
  # heatmap rows → tensor-level imbalance
  # heatmap columns within a row → neuron-level imbalance

# Two-level exploitation
L1 (Tensor-level): 1T-Drop via normalized gating score threshold T_drop^1
L2 (Neuron-level): 2T-Drop with T_major^2 < T_minor^2
  - score < T_major^2: skip expert entirely
  - T_major^2 ≤ score ≤ T_minor^2: compute major sub-expert only
  - score > T_minor^2: compute full expert
```

术语一般如何实现？如何使用？
- 观察：在 calibration data (MMLU/C4) 上前向传播，记录 per-expert per-neuron 的 accumulated |gate·up|
- Tensor-level 利用：expert partition 增加专家粒度 → fine-grained dropping；1T-Drop/2T-Drop 按归一化 gating score 阈值丢弃
- Neuron-level 利用：importance profiling → major/minor split → 2T-Drop 的中间档仅计算 major half
- Profiling 选择：Mixtral+OLMoE 适用 accumulated abs gate value；DeepSeek-V2-Lite-Chat 适用 accumulated abs gate-up value
- 跨任务泛化性：gating score 分布在不同 benchmark 间高度一致（图 6c），保证 threshold-based dropping 的泛化性

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

## Expert Partition in MoE (MoE 专家划分)

术语解释
Expert Partition 是在 post-training 阶段将预训练 MoE 模型中每个 expert 划分为 P 个更细粒度 experts，通过增加 tensor-level sparsity 提升 fine-tuning quality 和 inference efficiency，而无需重新预训练。含 Complete Transformation（数学等价变换，适用 fine-tuning 提升）和 Partial Transformation（保持 gating network，适用系统效率优化）。

术语是什么？
MoE 的 expert granularity 在 pre-training 时确定无法后续改变，但 prior work 证明 finer-grained experts 在相同 per-token FLOPs 下可降低 pre-training loss。Expert Partition 在 post-training 阶段实现等效效果。Complete Transformation 三步：重复 W_g 中每个 expert-specific vector P 次 + Top-K → Top-(K×P) + 均分每个 expert 的 neurons 为 P 份 + 将每个新 expert 的 W₂ 乘以 P（补偿因 gating score 变为 1/P 倍导致的输出缩放）。数学验证：s_{e,p} = s_e/P (各子 expert gating score 相等)，输出 y_i^P = y_i（W₂ 补偿后等价）。Partial Transformation 两步：仅重复 gating scores + remap expert indices（contiguous mapping），不修改 W_g 参数和 W₂ 权重。

从算法pipeline角度拆解术语：
```
=== Complete Transformation (P=2, E=2→4) ===
W_g^new = [h_0|h_0|h_1|h_1]  (repeat P times)
Top-K → Top-2K (Top-4)
For each original expert e:
  W1_new^{e,0} = W1_e[:, :d_ffn/2], W1_new^{e,1} = W1_e[:, d_ffn/2:]
  W2_new^{e,0} = W2_e[:d_ffn/2, :] × P,  W2_new^{e,1} = W2_e[d_ffn/2:, :] × P
  W3_new^{e,0} = W3_e[:, :d_ffn/2], W3_new^{e,1} = W3_e[:, d_ffn/2:]

Output consistency:
  s_{e,p} = s_e / P  (repeated W_g vectors → equal logits)
  Σ_{p} s_{e,p} · f_{e,p} = (s_e/P) · Σ_{p} f_{e,p} = s_e · f_e / P
  After W₂×P: s_e · f_e ✓

=== Partial Transformation ===
Gating scores repeated P times, expert indices remapped contiguously.
W₂ NOT scaled. Output: s_e · Σ f_{e,p} = s_e · f_e ✓ (no compensation needed)
```

术语一般如何实现？如何使用？
- Complete Transformation：fine-tuning 前使用 → Mixtral 8→32: downstream accuracy +0.59%；与现有 MoE 框架原生兼容（输出即标准 MoE）
- Partial Transformation：(a) S-ETP 通信优化（AlltoAll 替代 AlltoAll+AllGather）；(b) DualSparse-MoE 推理加速（更细 dropping 粒度）；(c) EP scale-up（更多 experts 分布到更多 devices）
- 局限性：P 过大→marginal benefit 递减 + compute intensity 下降 + gating overhead 增加

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

## Token-Expert Computation Dropping (Token-Expert 计算丢弃, 1T-Drop / 2T-Drop)

术语解释
MoE 推理加速策略：对每 token 在每层 MoE 中，按归一化 gating scores 选择性丢弃 token-expert FFN 计算。1T-Drop 用单一阈值丢低 score 的 token-expert 对；2T-Drop 引入双阈值 + expert reconstruction，对 major (高重要性)/minor (低重要性) sub-expert 采用不同丢弃策略，trade off accuracy 与 speedup。

术语是什么？
观察：不同 benchmark 的 gating score 分布高度一致（图 6b/c），为跨任务 threshold-based dropping 提供泛化基础。极低阈值（~0.05）的丢弃甚至略提升 accuracy（低 score experts 输出可能是噪声）。1T-Drop 归一化 gating scores 后丢弃低于阈值的计算。2T-Drop 先 expert partition + neuron reconstruction → 对 major sub-expert 使用低阈值 T²_major（保守保留），对 minor sub-expert 使用高阈值 T²_minor（激进丢弃）→ 中间档的 expert 仅计算 major half。

从算法pipeline角度拆解术语：
```
=== 1T-Drop ===
s_norm[j] = s_selected[j] / Σ_k s_selected[k]   # normalize
mask[j] = s_norm[j] >= T_drop^1
y = Σ_{j: mask[j]} s_selected[j] · FFN_{e_j}(x)

=== 2T-Drop with Reconstruction ===
# Offline: Importance_e[n] = Σ|Swish(x·W1^n) ⊙ (x·W3^n)|
# Sort → major_expert (top 50%), minor_expert (bottom 50%)

# Online:
For each activated expert e_j:
  if s_norm[j] < T_major^2: skip
  elif s_norm[j] < T_minor^2: compute FFN_major only
  else: compute FFN_full

# Threshold setting: T_major^2 = T_drop^1 - 0.01, T_minor^2 = T_drop^1 + 0.01
# Keeps similar drop rate but higher accuracy
```
22-27% drop rate → 1.17-1.23× MoE speedup, 1.07-1.12× end-to-end. Tensor-level dropping 适配 GPU grouped-GEMM，区别于 neuron-level sparsity 在低稀疏率下难转换为实际 speedup。

术语一般如何实现？如何使用？
- SGLang 框架实现：gating function 后添加 normalize+threshold+mask 逻辑；Triton grouped-GEMM kernel 集成 skip/major-only/full 变粒度模式
- Calibration: MMLU 做 neuron importance profiling；threshold sweep 确定 optimal value
- 部署：支持 single GPU, TP, EP；drop rate 直接 proportional 转换为 speedup
- 局限：GSM8K 对 drop rate 最敏感；drop rate-threshold 非线性；per-layer threshold 待探索

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction

## Neuron Importance Profiling for MoE (MoE 专家神经元重要性分析)

术语解释
在 calibration samples 上对 MoE 每个 SwiGLU FFN neuron 进行重要性度量（四种方法），用于指导 neuron 按重要性排序重构为 major + minor sub-expert，支持 2T-Drop 的细粒度计算丢弃。

术语是什么？
四种 profiling 方法（在 calibration samples 上累积）：(1) Σ Swish(x·W₁^n)；(2) Σ |Swish(x·W₁^n)|；(3) Σ Swish(x·W₁^n) ⊙ (x·W₃^n)；(4) Σ |Swish(x·W₁^n) ⊙ (x·W₃^n)|。实验：(a) 绝对值方法优于非绝对值（避免正负抵消）；(b) 不同模型 affinity 不同（Mixtral+OLMoE：方法2最佳；DeepSeek：方法4最佳，因其含 shared expert 结构）；(c) 低负载 expert 出现大量负 gate value，高负载 expert 罕见。

从算法pipeline角度拆解术语：
```
For each expert e in MoE:
  importance = zeros(d_ffn)
  For each sample x in calibration (MMLU):
    importance += |Swish(x·W1_e) ⊙ (x·W3_e)|  # method 4 (best for DeepSeek)
  sorted_idx = argsort(importance, desc=True)
  major_idx = sorted_idx[:d_ffn/2]; minor_idx = sorted_idx[d_ffn/2:]
```

术语一般如何实现？如何使用？
- Calibration: MMLU 做 profiling（泛化性强），一次前向传播即可完成
- 与 expert partition 结合：先 partition (E→E×P)，再每 finer expert 做 profiling → 总 sub-expert = 2×E×P
- 局限：静态 profiling 无法捕捉 runtime dynamic patterns（但 gating score 跨任务稳定佐证其有效性）；profile method 需 per-model 选择；>2 split 可能进一步改善但降低 compute intensity

涉及论文标题：
- DualSparse-MoE: Coordinating Tensor/Neuron-Level Sparsity with Expert Partition and Reconstruction
- DiEP: Adaptive Mixture-of-Experts Compression through Differentiable Expert Pruning

## Diffusion Large Language Models (dLLMs) with Block-based Parallel Decoding

术语解释
Diffusion Large Language Models (dLLMs) 是一类原生并行解码的 LLM，使用 block-based parallel decoding 替代传统自回归（AR）逐 token 生成。通过同时处理多个 masked tokens，在单次前向传播中生成整个 token block，平衡生成质量和吞吐。

术语是什么？
dLLM 的核心思想是 block diffusion：将输入文本分割为固定大小的 blocks（如 16/32/64 tokens），每个 block 内的 tokens 从 [MASK] 状态开始，经多步 denoising 逐步 unmask。与 AR decoding（token-by-token）和 speculative decoding（draft+verify）不同，dLLM 无需 verification step 即可直接并行生成。

关键架构组件：
- **Block-based generation**: 输入=N tokens，输出=N tokens，一次 forward pass 生成整个 block
- **Mask prediction**: 每个 token position 预测 masked token 的概率分布
- **Confidence-based sampling**: 通过 confidence threshold 决定 token 是否 finalize
- **KV Cache with bidirectional context**: 使用近似 KV cache 策略（如 Fast-dLLM）支持 block 内双向注意力

代表模型：LLaDA (Zhu et al., 2025), LLaDA2.0 (Bie et al., 2025), Dream (Ye et al., 2025), Block Diffusion (Arriola et al., 2025)

从算法pipeline角度拆解术语：
```
# Block-based Parallel Decoding Pipeline
for each block position p in [0, seq_len, block_size]:
    # Step 1: Initialize block
    block_tokens = [MASK] * block_size  # 全 mask 初始状态
    prefix_context = X[:p]               # 已解码的前缀
    
    # Step 2: Iterative denoising
    for step in range(max_steps):
        input_tokens = concat(prefix_context, block_tokens)
        logits = dLLM.forward(input_tokens)[-block_size:]  # 仅取 block 位置
        probs = softmax(logits)
        
        # Confidence-based finalization
        confidence = max(probs, dim=-1)
        for i where confidence[i] > threshold:
            block_tokens[i] = argmax(probs[i])  # finalize
        
        if all(finalized):
            break
    
    # Step 3: Append decoded block to context
    X = concat(X, block_tokens)
```
关键张量维度：输入 [batch, seq_len+block_size, d_model]，输出 [batch, block_size, vocab_size]。

与 MoE 结合的特殊问题：每 token 独立路由 → unique expert load 随 block_size N 线性增长（"expert explosion"）。

术语一般如何实现？如何使用？
- dInfer 框架（Ma et al., 2025）：专门为 dLLM 设计的推理框架
- Fast-dLLM (Wu et al., 2025)：训练无关的 KV cache + 并行解码加速
- SGLang 已规划 dLLM 支持（roadmap 2026 S1）
- 典型配置：block_size=32（16 prefix + 16 suffix cache tokens），confidence threshold=0.9
- 主要 trade-off：block size 增大提升吞吐 but 降低 generation quality；较小 block 保持 AR 级别质量但速度增益有限

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

## Expert Explosion in MoE dLLMs

术语解释
Expert Explosion 是 MoE dLLM 并行解码中的一种现象：随着并行 token 数（block size N）增加，unique activated experts 数量近乎线性增长，导致 HBM→SRAM weight fetching 成本主导延迟，使推理陷入 memory-bound 状态。

术语是什么？
在 MoE dLLM 中，每个 token 通过独立 routing 函数选择 Top-K experts。当 N 个 tokens 并行处理时，unique expert load = |∪_{n=1}^N S_n|，其中 S_n 是第 n 个 token 选择的高分 expert 集合。在均匀路由假设下，|∪| 期望值为 M·(1-(1-K/M)^N)，随 N 增长迅速接近 M。

延迟模型：L_MoE = b·|∪_{n=1}^N S_n| + a·(N·K)，其中 b 是 HBM→SRAM weight fetching cost（主导项），a 是 marginal compute cost。N 增大时 b 项呈线性/次线性增长，导致总延迟上升。

Roofline 分析：MoE dLLM 的 operational intensity 低于同容量的 dense 模型（因为 sparse activation 降低了计算密度），使其更 memory-bound。现代 GPU 的 FLOPs/byte ratio 增速超过 memory bandwidth 增速（Ma & Patterson, 2026），加剧此瓶颈。

从算法pipeline角度拆解术语：
```
# Expert Explosion 量化
M = 128  # total experts
K = 8    # top-k per token
for block_size N in [8, 16, 32, 64]:
    # 每 token 独立选择
    expert_sets = []
    for token n in range(N):
        S_n = TopK(router_gate(token_n), K)  # |S_n| = K
        expert_sets.append(S_n)
    
    unique_experts = len(set.union(*expert_sets))  # ≈ M*(1-(1-K/M)^N)
    # N=8:  unique≈47, N=16: unique≈72, N=32: unique≈98, N=64: unique≈118
    memory_traffic = unique_experts * expert_size_bytes  # dominates latency
```
实验结果（LLaDA2.0-Mini 16B, N=32）：vanilla 产生 ~84 unique experts/layer，expert weight footprint ~0.98 GB/layer。

术语一般如何实现？如何使用？
- 识别：通过 MoE kernel latency profiling（Nsight Systems）观测 HBM traffic 与 unique expert count 的线性关系
- 缓解方向：
  - Dynamic Expert Sharing (DES)：序列级 coreset selection 减少 |∪S_n|
  - Expert offloading/CXL-NDP：将冷 experts 卸载到外部存储
  - Expert quantization：降低每 expert 的 weight footprint
  - 与 AR batching 中 expert popularity skew 的问题不同，expert explosion 是 dLLM 并行解码独有的

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

## Dynamic Expert Sharing (DES)

术语解释
Dynamic Expert Sharing (DES) 是一种将 MoE 优化从 token-centric pruning 转变为 sequence-level coreset selection 的技术。通过识别紧凑、高效用的 expert 子集（coreset）服务整个并行解码 block，最大化 expert 复用，减少 HBM→SRAM weight fetching cost。

术语是什么？
DES 的核心公式：定义 Coreset Selection Function Φ: I → C，将运行时信息 I（如 router logits 或 hidden states）映射到共享 expert coreset C ⊂ {E_1, ..., E_M}。优化目标为：Φ* = argmin |Φ(I)|，满足 A(Φ(I)) ≥ A_base - ε。

DES 算法（Algorithm 1）：
**Stage 1: Sequence-level Consensus** — 通过 Φ 识别 compact high-utility expert set C。
**Stage 2: Constrained Local Routing** — 每 token 仅在 C 内进行 Top-K selection，重新归一化 gate weights。

延迟模型简化为：L_MoE(Φ) ≤ b·|Φ(I)| + a·(N·K)，unique expert weight cost 从 |∪S_n|（随 N 增长）降低为 |C|（与 N 解耦的可控变量）。

两种具体策略：DES-Seq 和 DES-Vote。

从算法pipeline角度拆解术语：
```
# DES Algorithm (Algorithm 1 from paper)
Input: I (sequence info: router logits N×M), Φ (coreset function), σ (activation), K
Output: Y (layer output N×d)

# Stage 1: Sequence-level Consensus
C = Φ(I)  # C ⊂ {1..M}, |C| << M, e.g. |C| ≈ β×M

# Stage 2: Constrained Local Routing
for each token n = 1..N:
    # Route within coreset only
    S_n = TopK(I_n[i] for i in C, K)     # top-K experts from C
    g_n = σ(I_n[i] for i in S_n)          # re-normalize gate weights
    y_n = Σ_{i∈S_n} g_{n,i} · E_i(x_n)   # weighted expert sum

return Y = [y_1, ..., y_N]
```

关键结果（LLaDA2.0-Mini 16B, N=32）：
- DES-Vote (β=0.15): unique experts 84→38 (-55%), MoE layer latency -38.0%, relative accuracy 99.5%
- DES-Vote (β=0.10): unique experts 84→25 (-70%), MoE layer latency 进一步降低, relative accuracy 96.4%

术语一般如何实现？如何使用？
- 无训练（training-free）：直接修改 inference 时的 routing 逻辑，无需重新训练模型
- 参数化：DES-Vote 用 budget factor β 控制 coreset size M_core = β×M；DES-Seq 用 local selection count k < K
- 超参数：β 越小→coreset 越小→memory 节省越大→accuracy 可能降低。β 调节灵活（连续值）
- 系统集成：在 dInfer 等 dLLM 框架中的每 MoE 层插入 coreset selection step
- 自定义 fused kernel 可消除算子碎片化开销
- 发现"re-activation"效应：从 coreset 中重新激活 expert（即使非原始 Top-K）几乎无 marginal cost，可恢复 accuracy

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

## Saliency-Aware Voting (DES-Vote)

术语解释
DES-Vote 是 DES 的 coreset selection 策略。所有并行 tokens 按加权 router saliency 投票选举共享 expert coreset，克服 DES-Seq 的两大局限：无法显式最大化 expert 共享，以及使用固定阈值 k 忽略 token 间 expert 重要性差异。

术语是什么？
DES-Vote 核心流程（Algorithm 3）：
1. **Mask**: 对每个 token 的 router logits，保留 local Top-K 权重，其余置零 → I_m
2. **Aggregate**: 跨所有 token 聚合加权投票 → V_i = Σ_{n=1}^N I_{m,n,i}，即 expert i 从所有 token 收到的总 saliency
3. **Select**: Top-M_core experts by total vote → C = TopK(V, M_core)

关键洞察来自 **Expert Importance Map**（Figure 4）：raw gating weights 与实际 expert 重要性高度相关，因此用 router scores 作为 voting weights 比 uniform voting 更有效。

DES-Vote 优于 DES-Seq 的原因：
- 解决局限 (1)：全局 voting → 自然形成跨 token 共识 → 最大化 expert 共享
- 解决局限 (2)：weighted voting → collective importance 自然决定保留哪些 expert → 自动处理 token 间 expert 重要性差异
- 连续 β → 绕过 DES-Seq 每 token 至少 1 expert 的下限 → 支持更小的 coreset

从算法pipeline角度拆解术语：
```
# DES-Vote Algorithm (Algorithm 3 from paper)
Input: I (router logits N×M), M_core (target coreset size), K (local top-k)
Output: C (coreset expert indices)

# Step 1: Keep only local top-K weights, mask others
I_m = zeros_like(I)
for n in 1..N:
    topk_indices = TopK_indices(I[n], K)  # per-token top-K
    I_m[n, topk_indices] = I[n, topk_indices]  # keep weights, rest = 0

# Step 2: Aggregate weighted votes across sequence
V = sum(I_m, dim=0)  # V ∈ R^M: total saliency per expert

# Step 3: Select top M_core experts by total vote
C = TopK_indices(V, M_core)  # C ⊂ {1..M}, |C| = M_core

# Subsequent Constrained Local Routing (same as DES):
for each token n:
    S_n = TopK(I[n, C], K)  # Route within C
    ...
```

M_core 由 budget factor β 控制：M_core = β × M。例如 β=0.15, M=128 → M_core=19。

实验证据：DES-Vote 在相同 coreset size 下比 DES-Seq 实现更高 Top-K recall（保留更多 ground truth expert selections）和更低 residual reconstruction loss。

术语一般如何实现？如何使用？
- 参数选择：β ∈ (0, 1]，需按模型/任务 tuning。典型配置：LLaDA2.0-Mini β=0.15, LLaDA-MoE-7B β=0.6
- Mask 操作是必需的：不去除 low-rank experts 的 noise weights 会降低 voting 质量
- 与 DES-Seq 对比：DES-Vote 在 Top-K hit rate、reconstruction loss、最终 accuracy 上全面优于 DES-Seq
- 适用场景：并行度越高（block size 大），DES-Vote 优势越明显（more tokens = better voting consensus）

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs

## Intra-Sequence Sharing (DES-Seq)

术语解释
DES-Seq 是 DES 的一种直接 coreset selection 策略：对每个并行 token 取 Top-k 个最 salient experts（k < K），取所有 token 的并集作为共享 coreset。

术语是什么？
DES-Seq 的 coreset 构建：C_DES-Seq = ∪_{n=1}^N TopK(I_n, k)，其中 k 是超参数（满足 k < K，K 为 vanilla routing 的每 token expert 数）。这一策略最早在 AR 模型的 batch-level optimization（OEA, Oncescu et al., 2025）中探索，DES 将其适配到 dLLM 的 intra-sequence level。

从算法pipeline角度拆解术语：
```
# DES-Seq Algorithm (Algorithm 2 from paper)
Input: I (router logits N×M), k (local selection count, k < K)
Output: C (coreset expert indices)

C = ∅
for n = 1 to N:
    topk_n = TopK_indices(I[n], k)  # per-token top-k experts
    C = C ∪ topk_n                   # union across tokens

return C
```

局限性：
1. **不显式最大化 sharing**：仅减少 local budget（k < K），不寻求跨 token 共识 → 可能产生低效 coreset
2. **固定 k 忽略 expert 重要性差异**：第 2 名 expert 对 token A 可能比对 token B 关键得多，uniform k 无法捕捉

术语一般如何实现？如何使用？
- k 值：典型配置 k=2, k=3（vs vanilla K=8）
- coreset size 下限：每 token 至少 1 expert，即 |C| ≥ 1（当所有 token 共享同一 top-1 expert 时）
- 与 DES-Vote 对比：DES-Seq 在 accuracy-efficiency Pareto frontier 上处于 DES-Vote 之下
- 适用场景：当 voting overhead 需要避免时作为简单 baseline；极低 latency 场景
- k=2 配置（LLaDA2.0-Mini）：unique experts 84→34 (-60%), relative accuracy 95.7%

涉及论文标题：
- Dynamic Expert Sharing: Decoupling Memory from Parallelism in Mixture-of-Experts Diffusion LLMs


## Multi-head Latent Attention (MLA, 多头潜在注意力)

术语解释
MLA 是 DeepSeek-V2 提出的注意力机制，通过对 Key 和 Value 进行低秩联合压缩（low-rank joint compression），将 KV Cache 从每头存储完整 K/V 向量压缩为存储一个低维 latent vector，大幅减少推理时的 KV Cache 内存占用。

术语是什么？
传统 MHA 对每个 token 需要缓存所有头的 K 和 V 矩阵，KV Cache 大小随序列长度和头数线性增长。MLA 通过将 K 和 V 投影到低维 latent 空间：输入 h_t 经下投影矩阵 W^{DKV} 压缩为 latent vector c_t^{KV}，再分别经 W^{UK} 和 W^{UV} 上投影还原 K 和 V。推理时仅需缓存 c_t^{KV}（而非完整 K/V），实现 93.3% KV Cache 减少（DeepSeek-V2 报告）。

从算法pipeline角度拆解术语：

```
=== MLA 前向传播 ===
# Step 1: Q 投影（标准）
q_t = W_Q @ h_t  # [n_heads * d_head]

# Step 2: KV 低秩压缩（MLA 核心）
c_t_KV = W_DKV @ h_t  # [d_latent], d_latent << n_heads*d_head

# Step 3: K, V 还原
k_t_C = W_UK @ c_t_KV  # [n_heads * d_head]
v_t_C = W_UV @ c_t_KV  # [n_heads * d_head]

# Step 4: RoPE 仅部分维度
k_t_R = RoPE(W_KR @ h_t)
k_t = concat([k_t_C, k_t_R])

# Step 5: Attention 标准计算
output = softmax(q_t @ K_cache.T / sqrt(d_head)) @ V_cache

# KV Cache 对比：
# MHA: n_heads × d_head × 2(K+V) → e.g., 64KB/token
# MLA: d_latent × 1(latent) → e.g., 1KB/token (64x压缩)
```

术语一般如何实现？如何使用？
- DeepSeek-V2 提出，21B 激活参数达到 Llama3-70B 水平
- KV Cache 减少 93.3%，推理吞吐提升 5.76x
- EPS-MoE 因 MLA 使用 DP+EP（非 TP+EP），避免 MLA 的额外 TP 通信开销
- 适用：长上下文（128K token）内存受限推理场景

涉及论文标题：
- EPS-MoE: Expert Pipeline Scheduler for Cost-Efficient MoE Inference

## Evolutionary Strategy for Model Compression (进化策略模型压缩)

术语是什么？
进化策略（Evolutionary Strategy, ES）是一种基于种群的随机优化算法，不需要梯度信息。在模型压缩中，ES 通过维护一组候选压缩方案（个体），每代通过选择（Selection）、交叉（Crossover）和变异（Mutation）操作迭代优化，在无梯度约束下搜索最优的剪枝/合并配置。

在 EEP 中，ES 被用于搜索最优的 Expert Pruning 和 Expert Merging 配置：
- **个体表示**：每个个体是在所有 MoE 层上的 Router Mapping 矩阵集合 W={W^l_RM, W^l_EM}_{l=1..L}（或 Pruning Phase 中 W_RM = W_EM）
- **Fitness Function**：F(W) = 模型在训练子集上的下游任务准确率（generation-based evaluation），仅需做推理即可评估
- **选择（Selection）**：按 fitness 排名，前 M_CP 个个体进入候选父代集 CP（elitism selection）
- **交叉（Crossover）**：随机从 CP 中采样两个父代 W_f 和 W_m，随机组合两者的 merging coefficients（沿 retained expert 维度交叉），或以一定概率直接选择单一父代的全部矩阵
- **变异（Mutation）**：Pruning Phase 随机替换 pruned expert；Merging Phase 对 merging coefficients 逐元素加入 Gaussian noise N(0, σ²)
- **世代更替**：每代将变异后代 NG 加入种群 P ← P ∪ NG，不淘汰旧个体

从算法pipeline角度拆解术语。
```
# EEP 进化搜索伪代码
Input: Θ = all expert weights, F = evaluator, Epochs, M_CP, Iters
Output: optimal W*

1: P ← {random one-hot initialization W_init with F(W_init)}
2: for phase in {Pruning, Merging}:
3:   for t = 1..Iters:
4:     NG ← ∅
5:     for i = 1..Epochs:
6:       CP ← {W_i | F(W_i·Θ) ranks top min(M_CP, |P|)}
7:       W_f, W_m ← RandomSample(CP)           # 选择
8:       W_new ← Mutate(Crossover(W_f, W_m))    # 交叉+变异
9:       NG ← NG ∪ {(W_new, F(W_new))}          # 评估
10:    P ← P ∪ NG                                # 更新种群
11: return argmin_W F(W)
```

为什么需要？/解决什么痛点？
- LLM 上的梯度计算需要大量 GPU 显存（至少 2× 模型大小），基于梯度的 fine-tuning 对大多数用户不可行
- ES 只需推理即可评估 fitness，可在推理设备上运行，无需反向传播的额外显存
- ES 适用于离散搜索空间（expert 选择和 one-hot 约束），梯度方法难以直接处理离散决策
- ES 天然支持并行评估（种群中所有个体可并行推理），可利用多 GPU/多节点

术语一般如何实现？如何使用？
- EEP 的 ES 超参数：Pruning Phase 40 iterations, Merging Phase 160 iterations, population size 取决于评估预算
- 为减少搜索参数，expert weights 按深度分组（4 groups 或 32 groups），组内共享 merging coefficients
- 适用场景：任何需要搜索离散/连续混合优化空间且梯度不可得或代价过高的模型压缩任务
- 局限性：搜索过程需要大量推理调用（每代每个个体一次完整模型推理），搜索成本随种群规模线性增长
- 相关方法：Model Soup (uniform averaging), Evolutionary Model Merging (Akiba et al. 2024)
- 代码：https://github.com/imagination-research/EEP

涉及论文标题：
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

## Gradient-Free Post-Training Compression (无梯度后训练压缩)

术语是什么？
无梯度后训练压缩是一类不依赖反向传播和梯度计算的模型压缩方法的统称，其核心思想是在模型训练完成后，仅使用推理能力对模型进行压缩。与传统压缩范式（Prune → Fine-tune with SGD）不同，无梯度方法不更新或仅通过 weight averaging/merging 方式更新模型参数。

EEP 将无梯度压缩推进为一种完整的 paradigm：(1) 通过进化搜索实现 expert 选择（pruning），(2) 通过 continuous weight merging 实现知识恢复（analogous to fine-tuning but without gradients）。整个过程不涉及任何 Pytorch backward() 调用。
- 传统范式 I：Importance-based selection + SGD fine-tuning（需要 GPU ≥ 2× model size）
- 传统范式 II：Selection + distillation（需要 teacher model 做额外推理）
- EEP 范式 III：Evolutionary selection + Weight merging（仅需推理设备，单次评估仅做 forward pass）

从算法pipeline角度拆解术语。
```
# 传统有梯度压缩 vs 无梯度压缩（EEP）
# 传统: Prune + Fine-tune
W_pruned = prune_by_importance(W, sparsity)    # e.g., magnitude pruning
for epoch in range(fine_tune_epochs):
    loss = CrossEntropy(model(W_pruned, x), y)
    loss.backward()                              # 需要 ≥2× model size GPU memory
    optimizer.step()

# EEP: Search + Merge (gradient-free)
for iter in range(search_iterations):
    W_candidate = Mutate(Crossover(select(P)))  # ES operations
    accuracy = model.forward(W_candidate, x, y) # inference only, no backward()
    P.append((W_candidate, accuracy))
W_final = best_individual(P)
W_merged = continuous_weighted_average(W_final)  # merging phase
```

为什么需要？/解决什么痛点？
- **降低压缩门槛**：传统 fine-tuning 需要大显存 GPU（如 Mixtral 8×7B FP16 约 94GB，SGD 需要 >188GB），EEP 仅需推理显存（~44GB for fp16 或通过量化进一步降低）
- **设备灵活性**：可在推理专用硬件（如推理卡、边缘设备）上执行压缩
- **避免灾难性遗忘**：不使用梯度更新，通过 weight averaging 保留原始训练知识
- **适用于 downstream tasks**：EEP 针对特定下游任务数据集搜索，同时保持 zero-shot 泛化能力（MMLU OOD 实验验证）

术语一般如何实现？如何使用？
- EEP 实现：在 HuggingFace 框架上仅使用 model.generate() 和 accuracy computation，不调用 backward()。搜索完成后导出为标准 HF 格式模型权重。
- 搜索过程 40+160=200 iterations，每个个体一次推理（约数秒到数分钟），总搜索时间取决于种群规模和并行度
- 适用范围：适合无法负担 fine-tuning 计算资源的用户/场景，尤其适合针对特定下游任务数据集定制的模型压缩
- 限制：搜索成本仍不可忽视（论文标注为 limitation），无梯度搜索可能不如梯度-based fine-tuning 在极低 sparsity 下精确
- 代码：https://github.com/imagination-research/EEP

涉及论文标题：
- Efficient Expert Pruning for Sparse Mixture-of-Experts Language Models: Enhancing Performance and Reducing Inference Costs

## Retrieval Augmented Generation (RAG) for Low-Resource Data Preprocessing

术语解释
RAG（Retrieval Augmented Generation）是利用检索系统从外部知识库中获取相关信息并作为上下文提供给 LLM 以提升生成质量的方法。MELD（KDD '24）提出增强型 RAG 系统用于跨域检索、自标注（self-annotation）和训练数据扩增。

术语是什么？
MELD 的增强型 RAG 包含三个关键组件：
1. **Entry Alignment**：对结构化/半结构化数据，结构相似性与语义相似性同等重要。为每个 query q 构建正例集 P_q（对齐 entries）和负例集 N_q（未对齐 entries）。
2. **Fine-tuning RAG Model**：使用 sentence-bert（bge-large-en）作 backbone，contrastive loss 微调。负例集为空时做 hard negative mining。
3. **Self-Annotation**：微调后的 RAG 模型自动为未标注 query 检索最相似 instance 并生成伪标签，扩大训练集。还可通过 query transformation 实现跨任务数据增强（如 EM query → DI query）。

从算法pipeline角度拆解术语。
```
Input: few-shot labeled data X_i, unlabeled data X̃_i

// Step 1: Entry Alignment
for each query q:
    Serialize q to dict (tuple + meta: table title, column headers)
    Build P_q (positive), N_q (negative)

// Step 2: Fine-tune RAG
M_RAG = bge-large-en; τ=0.02
loss = -log(exp(cos(emb_q,emb_p)/τ) / Σ_{p'} exp(cos(emb_q,emb_p')/τ))

// Step 3: Self-Annotation
for each unlabeled q_i:
    q_j = argmax cos(M_RAG(q_i), M_RAG(q))
    Annotate q_i with label from q_j
    // Cross-task: EM→DI transformation via masking
```

术语一般如何实现？如何使用？
- MELD 使用 bge-large-en 作为 RAG backbone，temperature τ=0.02
- 与 meta-path 数据增强协同：RAG 负责跨域检索和自标注，meta-path 负责结构化增强
- Ablation 显示 w/o RAG 在所有数据集上性能显著下降（如 Semi-Text-Computer F1 86.46→42.02）

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

## Meta-path Search for Expert-based Data Augmentation

术语解释
Meta-path over Experts 是 MELD 提出的基于 expert 序列的数据增强策略。给定 fixed experts E={e_1,...,e_n}，meta-path E_i={e_{j1},...,e_{jm}} 是一个有序 expert 序列，沿序列依次查询 experts 对训练数据进行增强。

术语是什么？
- Meta-path 将多个 DP task 的 experts 串联成流水线，利用前序 expert 输出为后续 expert 提供额外特征
- 搜索算法：贪心搜索，目标 argmax_{E_i} Eval(e_i, X_i^{E_i})，用户定义 sub-optimal paths 缩减搜索空间

从算法pipeline角度拆解术语。
以 EM task meta-path E_EM = {e_Blocking, e_DI, e_AVE, e_EM} 为例：
```
q = (t1="Apple iPhone 13", t2="iPhone 13 by Apple")
→ e_Blocking(q): 候选对筛选 → 过滤噪声
→ e_DI(t1): 填补缺失属性 → t1' (enriched)
→ e_AVE(t1'): 提取关键属性值 → 附加特征
→ e_EM(t1', t2): 最终match/mismatch
```
半结构化数据提升显著（Semi-Text-Watch F1: 55.07→70.78）。

术语一般如何实现？如何使用？
- 贪心搜索避免 expert 组合穷举
- 用户定义 sub-optimal paths 基于领域知识（如 EM 常用 {e_Blocking, e_EM}）
- 搜索完成后 meta-path 固定，训练/推理时直接使用

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

## Information Bottleneck Guided Expert Refinement

术语解释
Information Bottleneck（信息瓶颈，IB）是一种平衡表示复杂度和预测能力的理论框架。MELD 将 IB 用于指导 expert 精炼和 router training，在 few-shot 场景防止过拟合。

术语是什么？
IB 原理：min I(X;Z) 且 max I(Y;Z)，X=输入，Y=标签，Z=表示。
- min I(X;Z)：压缩噪声、捕获高层特征
- max I(Y;Z)：保留足够的预测信息

从算法pipeline角度拆解术语。
Expert 训练优化函数：
```
arg min_{θ_M_RAG} max_{θ_M_G} I(M_G(X_i); M_G(RAG(X_i)))
```
- max θ_M_G：LoRA fine-tune，最大化输出与标签 Y_i 的互信息
- min θ_M_RAG：控制 RAG 采样和 meta-path，最小化原始数据与增强数据的互信息

Router 优化函数：
```
max Σ_{e_i∈N(q_u)} I(e_i(q_u^i); l_u^i)    // 相关性
min Σ_{e_i≠e_j} I(e_i(q_u^i); e_j(q_u^j))  // 多样性
```
实践中用对比学习近似实现，精炼迭代 σ=3 轮。

术语一般如何实现？如何使用？
- Expert 精炼在 LoRA fine-tune 基础上进行，不改 base model weights
- Router 用对比学习近似互信息计算
- IB 同时支撑 Theorem 2（MoE error bound）和 Theorem 3（router 收敛性）

涉及论文标题：
- Efficient Mixture of Experts based on Large Language Models for Low-Resource Data Preprocessing

## Mixture-of-Agents (MoA, 混合代理)

术语是什么？
Mixture-of-Agents (MoA) 是一种多 LLM 协作推理范式，将多个 LLM 作为并行 proposer agents 各自生成答案，再由 aggregator agent 融合这些输出产生最终响应。MoA 可组织为多层结构——相邻层间的 agent 通过"输出融合"建立依赖关系：设前驱 agents A1, A2, A3 输出 o1, o2, o3，后继 aggregator Aagg 的输入 prompt 为 S(Aagg) = ∪(S_prefix, o1, o2, o3, S_suffix)。MoA 在推理、QA 和代码生成等任务上展示了显著的实证收益，但在系统效率上面临两个核心挑战：(1) 全连接拓扑导致冗余 agent 间通信；(2) agent 间异构延迟和复杂数据依赖使得现有 LLM serving 框架（PD disaggregation）无法有效支持——前驱 agent 解码与后继 agent prefilling 被视作严格串行。

从算法pipeline角度拆解：
```
# All-to-All MoA 推理 (两层)
Input: user query Q
Layer 1 (N proposer agents, 并行):
  for each agent a_{1,i}:
    o_{1,i} = LLM_i(prompt_template(Q))
  # 各 agent 用不同 LLM 骨干，推理延迟异构

Layer 2 (aggregator):
  S_agg = concat(S_prefix, o_{1,1}, ..., o_{1,N}, S_suffix)
  o_final = LLM_agg(S_agg)
```
延迟分析：T_total = max_i(t_{1,i}) + t_prefill_agg + t_decode_agg，瓶颈为最慢 proposer 的完成时间和 aggregator 的 prefill（不与其他计算重叠）。

术语一般如何实现？如何使用？
- 基于开源 LLM serving 框架（SGLang、vLLM）部署，每 agent 可选不同模型骨干
- 现有系统多用 all-to-all 连接（Multi-Agent Debate、Reconcile），但存在冗余连接和低 GPU 利用率问题
- 适用于多视角推理任务：复杂数学、科学 QA、代码生成、指令遵循

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

## Hierarchical Tree-Structured Agent Topology (层次化树状 Agent 拓扑)

术语是什么？
将 MoA 的全连接 agent 交互图替换为层次化树结构。典型 9-3-1 三层配置：Layer 1 的 9 个 leaf agents 分为 3 clusters（每 cluster 含 4B/8B/32B 三模型），Layer 2 每 agent 仅连接对应 cluster（|C(a)|=3），Layer 3 root 聚合所有 Layer 2 输出。延迟优化核心：T_ℓ^tree ≈ max_{a_{ℓ,j}} max_{c∈C(a_{ℓ,j})} t_c，远小于 all-to-all 的 T_ℓ^all = max_i t_i；子树间互不阻塞，straggler 影响局限于其子树。

从算法pipeline角度拆解：
```
Input: query Q
Layer 1 (9 agents, 3 clusters):
  // 3 clusters 独立并发
  Cluster k: {a_{1,3k-2}(4B), a_{1,3k-1}(8B), a_{1,3k}(32B)} 并行

Layer 2 (3 agents):
  a_{2,k} ← 仅聚合 Cluster k 的 3 个输出
  // 输入上下文 = prefix + 3×output (vs all-to-all 的 9×output)

Layer 3 (root):
  a_{3,1} ← 聚合全部 Layer 2 输出 → final answer
```
优势：(1) 上下文缩短（prefill 成本线性降）；(2) 子树并发（互不阻塞）；(3) straggler 隔离。

术语一般如何实现？如何使用？
- 修改 MoA orchestration 层的 agent 依赖图
- 聚类可异构分组（每 cluster 含小/中/大模型）
- 需 Shell Router 或编排器管理依赖

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

## Dynamic Agent Early-Exit via Semantic Similarity and Confidence (基于语义相似度与置信度的动态 Agent 早退)

术语是什么？
在 MoA 推理每层内，利用已完成 agent 输出计算早退概率 Q，按 Q 终止未完成的 agent（尤其是大模型），避免等待 straggler。计算流程：(1) 置信度 C_ℓ = exp((1/n_a)·Σ log p_i)（几何平均），C̄ = RMS 历史；(2) 语义相似度：用共享 embedding model 提取 T_i，构建 U=T_i^T·T_i，计算 FrobCosSim(U,V)；(3) 置信度加权 P = (1/W)·Σ C_i·C_j·Sim[i,j]；(4) 校准 B = 1-|P-τ|/τ（τ=0.7）；(5) Q = √(C̄·B)^(1/τ)。引入约 5% 额外延迟，带来 10-50% E2E 减少。

从算法pipeline角度拆解：
```
for each completed agent i in layer ℓ:
  C_i = geometric_mean(token_log_probs)  // token 级置信度
  T_i = EmbedModel(O_i)                   // Qwen3-Embedding-4B
  for each j < i:
    U_j = T_j^T·T_j, U_i = T_i^T·T_i     // correlation matrices
    Sim[j,i] = FrobCosSim(Corr(U_j), Corr(U_i))

P = weighted_avg(Sim, weights=C_i·C_j)
B = 1 - |P - 0.7| / 0.7                   // 校准：偏好适度一致
Q = sqrt(C̄ · B)                            // 合成质量分数

if random() < Q: early_exit()             // 概率性终止剩余 agent
```
核心设计：小模型先完成 → 若输出高置信且语义一致 → 大模型输出可能冗余 → 早退，节省延迟。难任务置信度低 → Q 低 → 继续等待。

术语一般如何实现？如何使用？
- 每层 agent 完成时触发，依赖共享 embedding model
- τ=0.7 经验设定，可网格搜索调整
- 适合 agent 池异构场景（小模型先完成，大模型慢）

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

## Frobenius Cosine Similarity (FCS, Frobenius 余弦相似度)

术语是什么？
标准余弦相似度在矩阵空间的推广。对于 U,V∈R^{h×h}：FrobCosSim(U,V) = ⟨Corr(U), Corr(V)⟩_F / (||Corr(U)||_F · ||Corr(V)||_F)。其中 Frobenius 内积 ⟨A,B⟩_F = trace(A^T·B)，Corr(U)_ij = U_ij/√(U_ii·U_jj)（消除尺度影响）。在 MoA 中用于衡量不同 agent 输出的语义相似度：将 last-layer hidden states T∈R^{n×h} 转为特征维相关矩阵 U=T^T·T∈R^{h×h}，再计算 FCS——解决了不同长度输出（n 不同）之间的相似度比较问题（通过 T^T×T 折叠 token 维）。

从算法pipeline角度拆解：
```
T_i = Embed(O_i) ∈ R^{n_i×h}    // 第 i 个 agent 的输出嵌入
T_j = Embed(O_j) ∈ R^{n_j×h}    // 可能不同长度

U = T_i^T × T_i ∈ R^{h×h}       // 折叠 token 维，保留特征维
V = T_j^T × T_j ∈ R^{h×h}

// 去尺度化
Corr(U)_ij = U_ij / sqrt(U_ii * U_jj)

// Frobenius 余弦
FCS = trace(Corr(U)^T · Corr(V)) / (||Corr(U)||_F · ||Corr(V)||_F)
```
选择特征维相关矩阵而非 token 维的原因：h（hidden dim）对所有输出固定，n（token 数）随输出长度变化——T^T×T 将可变维度折叠为固定 h×h。

术语一般如何实现？如何使用？
- 依赖共享 embedding model（如 Qwen3-Embedding-4B）确保跨模型可比
- 标准线性代数操作，高效实现
- 适用：多模型输出一致性评估、ensemble diversity 度量

涉及论文标题：
- Efficient Mixture-of-Agents Serving via Tree-Structured Routing, Adaptive Pruning, and Dependency-Aware Prefill-Decode Overlap

## Activated Expert Replicas (激活的专家副本)

术语解释
在 Expert Parallelism MoE 推理中，Activated Expert Replicas 指当前 batch 中实际有 token 需要处理的 expert replicas。与 total expert replicas 不同，activated replicas 仅包括那些收到至少一个 token 的 replica。METRO 论文的关键发现：在 memory-bound decode 阶段，GPU 的 MoE layer runtime 由 activated expert replicas 数量决定，而非由 token 数量决定。

术语是什么？
在 EP 部署中，每个 expert 可以有多个 replicas（副本）分布在不同 GPU 上。当 batch 中的 token 经过 router 选择 top-k experts 后，仅部分 replica 会收到 token——这些收到 token 的 replica 即为 "activated"。例如：expert e 有 3 个 replicas 在 GPU 0/3/5，但当前 batch 仅 2 个 token 选中 e——EPLB routing 可能将 2 个 token 分散到 2 个 replicas（激活 2 个），而 METRO routing 将 2 个 token 集中到 1 个 replica（仅激活 1 个）。在 memory-bound 下，每多激活一个 expert replica 就需要额外加载该 expert 的全部 FFN weight（~200MB）从 HBM 到 Tensor Core——这是延迟的主要来源。

从算法pipeline角度拆解术语：
Activated expert replicas 与 decode latency 的关系：

```
=== 8 GPUs, 128 experts, 1.5x replication, decode batch 256 tokens ===

# EPLB token-balancing routing:
for each expert e with T[e] > 0:
    # 将 T[e] 均匀分配到 e 的所有 replicas
    tokens_per_replica = ceil(T[e] / R[e])
    for g in GPUs hosting e replicas:
        if tokens_assigned_to_g < tokens_per_replica:
            y[e][g] = 1  # 激活该 replica
    
# 结果: 若 expert e 有 3 replicas 和 9 tokens
# → 每个 replica 各 3 tokens → 3 activated replicas
# → 全局: max_g Σ_i y[i][g] ≈ 高 (如 20 activated experts per GPU)

# 延迟构成 (memory-bound):
latency_EPLB = max_g (Σ_{i: y[i][g]=1} load_weight(expert_i, HBM→TC) + compute)
             ≈ max_g (num_activated_experts × weight_load_time_per_expert)
             ≈ 20 × 80μs = 1600μs (weight loading)
             + compute (~100μs, 可忽略)


# METRO expert-minimizing routing:
for each expert e with T[e] > 0:
    G_e = GPUs hosting e replicas (from placement matrix A)
    g* = argmin_{g in G_e} activated_count[g]  # 选 activated experts 最少的 GPU
    y[e][g*] = 1  # 仅激活一个 replica
    x[e][g*] = T[e]  # 所有 token 路由到该 replica

# 结果: 若 expert e 有 3 replicas 和 9 tokens
# → 9 tokens 全部路由到 activated_count 最小的 GPU
# → 仅 1 activated replica
# → 全局: max_g Σ_i y[i][g] ≈ 低 (如 12 activated experts per GPU)

# 延迟构成:
latency_METRO = max_g (num_activated_experts × weight_load_time)
              ≈ 12 × 80μs = 960μs (weight loading, -40% vs EPLB)
              + compute (~100μs) + routing_kernel (~26μs)
              ≈ 1086μs (净节省 ~514μs/layer)
```

术语一般如何实现？如何使用？
- Activated experts 在 decode batch 中通常远少于 total experts（batch size 小，top-k 选择导致各 expert 的 tokens 稀疏分布）
- 减少 activated experts 的方法：(a) **METRO greedy routing**: 将每个 expert 的所有 tokens 集中到单一 replica；(b) **减少 replication factor**: 少 replicas → 少可激活的 replica 数，但会损害 prefill 性能；(c) **Expert pruning**: 跳过不重要的 expert
- METRO 实验：activated experts 在最优解的 10.9% 以内，比 EPLB 降低 up to 42.3%
- 为什么不简单地不复制（1.0x replication）？因为 replication 对 compute-bound prefill 有显著提升（-17% TTFT），但会损害 memory-bound decode。METRO 用 expert-minimizing routing 消除 replication 对 decode 的副作用

涉及论文标题：
- Efficient MoE Serving in the Memory-Bound Regime Balance Activated Experts, Not Tokens

---

## Noise Top-k Gating (噪声 Top-k 门控)

术语解释
Noise Top-k Gating 是 MoE router 的一种门控机制，由 Shazeer et al. (2017) 提出。在标准 Top-k 选择前向 gate logits 加入可训练的 Gaussian noise，使得负载均衡通过噪声扰动自然实现而非完全依赖 auxiliary loss。

术语是什么？
公式定义：
$$H(x)_i = (x \cdot W_g)_i + \text{StandardNormal}() \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)$$
$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$
$$y = \sum_{i=1}^{N} G(x)_i E_i(x)$$

核心组成：
- **Gate logits**: `x · W_g` 产生每个 expert 的基本得分
- **Noise term**: `StandardNormal() · Softplus(x · W_noise)` 对 logits 加入可训练的高斯噪声，`W_noise` 可学习噪声幅度
- **KeepTopK**: 保留 top-k logits，其余设为 `-∞`（softmax 后概率为 0）
- **Softmax + Weighted Sum**: 归一化后加权聚合 expert 输出

噪声作用：(1) 在训练中通过随机扰动打破固定 routing 模式，鼓励更多样化的 expert 选择；(2) 配合 auxiliary loss 实现负载均衡；(3) 使 gate probability 分布趋于平滑，减少 router 坍缩到少数 expert 的风险。

从算法pipeline角度拆解术语：
```
# Noise Top-k Gating Forward Pass (per token)
def noise_topk_gating(x, W_g, W_noise, k, N):
    # x: [d] input token embedding
    # W_g: [d, N] gate weight
    # W_noise: [d, N] noise weight
    
    # Step 1: Clean logits
    clean_logits = x @ W_g              # [N]
    
    # Step 2: Noise logits (trainable)
    noise_std = softplus(x @ W_noise)   # [N], always positive
    noise = randn(N) * noise_std        # [N], Gaussian noise
    H = clean_logits + noise            # [N], noisy logits
    
    # Step 3: Top-k selection
    topk_vals, topk_idx = topk(H, k)    # select k experts
    mask = -inf * ones(N)
    mask[topk_idx] = H[topk_idx]
    
    # Step 4: Softmax + aggregate
    G = softmax(mask)                   # [N], sparse gate probs
    y = sum(G[i] * E_i(x) for i where G[i] > 0)
    return y
```
训练时 `W_g` 和 `W_noise` 均可学习；推理时噪声可关闭以确定性选择 expert。

术语一般如何实现？如何使用？
- Llama-MoE (Zhu et al., 2024) 使用 Noise Top-k Gating 作为其 router 机制
- 噪声项通过 `Softplus` 保证标准差始终为正，允许模型学习每个 expert 的噪声幅度
- 在 KD 场景中，SAR 方法更新 `W_g` 和 `W_noise`（仅 router 部分）使其更适应 student 的学习需求
- 负载均衡：Noise Top-k 配合 auxiliary loss `L_b = CV(m)^2 + CV(P)^2` 实现 balanced routing

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Knowledge Augmentation (KA) for MoE Knowledge Distillation

术语解释
Knowledge Augmentation (KA) 是 Kim et al. (2025) 提出的 MoE 专用知识蒸馏方法。核心思想：在蒸馏过程中，对同一输入进行 M 次教师前向传播，每次以概率 λ 从 gate probability 分布中随机采样 N-1 个 expert（以 1-λ 概率取 Top N-1），从而增广来自不同 expert 组合的多样化知识。解决传统 KD 仅使用 Top-k activated experts 而遗漏 non-activated experts 中知识的问题。

术语是什么？
KA 基于以下观察设计：(1) MoE 教师中 non-activated experts 的 gate probabilities 总和超过 50%，即大部分 expert 知识未被传统 KD 利用；(2) 增加 activated experts 数量（k→N-1）可提升 student 性能但不一定提升 teacher 性能，说明 non-activated experts 有独特知识；(3) Load balancing 使同一输入在不同迭代可能激活不同 expert 集合，知识分散在多个 expert 中。

KA 机制：使用 N-1 个 expert（而非原始 Top-k），通过混合策略（以概率 λ 采样、概率 1-λ 取 Top N-1）平衡 knowledge diversity 和 consistency。

从算法pipeline角度拆解术语：
```
# KA Distillation Pipeline (per training step)
def ka_distillation_step(x, teacher_moe, student, M, lambda_, N):
    # x: input request
    # teacher_moe: MoE teacher with Noise Top-k Gating
    # M: number of augmented forward passes
    # lambda_: sampling probability (typ. 0.05)
    
    # Student generates pseudo-target (on-policy)
    y_pseudo = student.generate(x)
    
    # Step 1: KA-based teacher forward (M times)
    teacher_logits_list = []
    for m in range(M):
        # Choose expert selection strategy
        if random() < lambda_:
            # Random sampling from gate prob distribution
            gate_logits = teacher_moe.compute_gate_logits(x)  # H(x)
            gate_probs = softmax(gate_logits)
            E_selected = sample_without_replacement(gate_probs, N-1)
        else:
            # Top N-1 selection (deterministic)
            gate_logits = teacher_moe.compute_gate_logits(x)
            E_selected = topk_indices(gate_logits, N-1)
        
        # Forward with selected experts
        KA_logits = compute_KA_logits(gate_logits, E_selected)
        G_KA = softmax(KA_logits)
        y_teacher = sum(G_KA[i] * E_i(x) for i in E_selected)
        teacher_logits_list.append(y_teacher)
    
    # Step 2: Distillation loss (reverse KL, M times)
    for teacher_logits in teacher_logits_list:
        L = KL_div(student(y_pseudo|x) || teacher_logits)  # reverse KL
        student.backward(L)
    
    return student
```
超参数：M（增广次数，典型值 2）、λ（采样概率，典型值 0.05）。M 过大会导致过度多样化的知识（"nonsense knowledge"），反而降低性能。

术语一般如何实现？如何使用？
- 适用于 MoE teacher → dense student 的蒸馏场景
- 需要 teacher 支持灵活修改 activated expert 数量（从 k 到 N-1）
- 与 GKD (Agarwal et al., 2024) 框架兼容：使用 student on-policy 生成 + reverse KL divergence
- 计算开销：M 次额外教师前向（M=2 时约 2× teacher forward cost per step）
- KA 在 Llama-MoE (3.5B/3.0B) → Sheared-Llama (1.3B) 蒸馏上实现 +4.8 ROUGE-L over conventional KD

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Student-Aware Router (SAR) for MoE Knowledge Distillation

术语解释
Student-Aware Router (SAR) 是 Kim et al. (2025) 提出的第二种 MoE 专用知识蒸馏方法。与 KA 的"采样增广"策略不同，SAR 直接优化 MoE 教师的路由器，使其根据 student 的反馈调整 expert 权重，从而为 student 提供更优的知识聚合。SAR 的核心创新在于将"student-friendly teacher"思想应用于 MoE 的 routing 机制。

术语是什么？
SAR 的每次迭代包含两个阶段：
1. **Router 更新阶段**：使用 student 反馈（reverse KL divergence + auxiliary load balancing loss β·L_b）仅更新 MoE 教师的路由器参数 `W_g` 和 `W_noise`，所有其他参数冻结，所有 expert 全激活
2. **知识蒸馏阶段**：使用更新后的路由器激活所有 expert 并加权聚合输出，通过 reverse KL divergence 蒸馏到 student

SAR 的 motivation：简单激活所有 expert（ALL baseline）已优于传统 KD，但不如 SAR，说明通过 student feedback 调整 expert 权重比简单全激活更有效。

从算法pipeline角度拆解术语：
```
# SAR Training Pipeline (per step)
def sar_step(x, teacher_moe, student, beta):
    N = teacher_moe.num_experts
    
    # Student generates pseudo-target (on-policy)
    y_pseudo = student.generate(x)
    
    # === Phase 1: Router Update ===
    # Teacher forward with ALL experts (full activation)
    gate_logits = teacher_moe.compute_gate_logits(x)  # H(x)
    gate_probs = softmax(gate_logits)                 # no top-k masking
    y_teacher_all = sum(gate_probs[i] * teacher_moe.E_i(x) for i in range(N))
    
    # Router loss: reverse KL + load balancing
    L_router = KL_div(student(y_pseudo|x) || y_teacher_all) + beta * L_b
    
    # Update only router parameters (W_g, W_noise)
    teacher_moe.W_g -= lr * grad(L_router, teacher_moe.W_g)
    teacher_moe.W_noise -= lr * grad(L_router, teacher_moe.W_noise)
    
    # === Phase 2: Knowledge Distillation ===
    # Teacher forward with updated router, ALL experts
    gate_logits_new = teacher_moe.compute_gate_logits(x)
    gate_probs_new = softmax(gate_logits_new)
    y_teacher_sar = sum(gate_probs_new[i] * teacher_moe.E_i(x) for i in range(N))
    
    # Student update
    L_student = KL_div(student(y_pseudo|x) || y_teacher_sar)
    student.backward(L_student)

# Auxiliary Load Balancing Loss
def L_b(m, P):
    # m: token counts per expert [N]
    # P: summed router probabilities per expert [N]
    CV = lambda v: std(v) / mean(v)
    return CV(m)^2 + CV(P)^2
```
β = 0.01 (遵循 Llama-MoE 原始设置)。

术语一般如何实现？如何使用？
- 适用于 MoE teacher (Noise Top-k Gating) → dense student 蒸馏
- Router 更新仅修改 W_g 和 W_noise，不影响 expert 参数
- 效果验证：SAR 的 KL divergence (原始 router vs 更新后 router) 随层深增加而增加→深层 expert 权重变化更显著→student 获益更大
- SAR 计算开销：每次迭代多一次 router 前向更新
- 在有 16 expert 的 Llama-MoE-3.5B 上，SAR 实现 25.91 avg ROUGE-L (vs KD 20.92)

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Gate Probability in Mixture-of-Experts

术语解释
Gate probability（门控概率）是 MoE router 经过 softmax 归一化后对每个 expert 的分配概率，反映 router 认为每个 expert 对当前 token 的有用程度。Gate probability 是理解 MoE expert 利用率和知识分布的核心指标。

术语是什么？
给定 token x，router 先计算 gate logits H(x)（含噪声），经 softmax 得到 gate probability：
$$G_i(x) = \frac{\exp(H(x)_i)}{\sum_{j=1}^N \exp(H(x)_j)}$$

Top-k routing 仅激活 gate probability 最高的 k 个 expert，其余 expert 的 gate probability 被遮蔽（masked to 0 after KeepTopK）。

关键发现 (Kim et al., 2025)：在 MoE KD 过程中，所有层中 activated experts 的 gate probability 之和通常低于 50%。这意味着超过一半的"router 信心"分配给了 non-activated experts，但这些专家的知识未被传统 KD 利用。

从算法pipeline角度拆解术语：
```
# Gate Probability Analysis
for each training sample during KD:
    for each MoE layer l:
        gate_probs = softmax(router(x_l))             # [N]
        activated_probs = gate_probs[topk_indices]    # [K]
        non_activated_probs = gate_probs[~topk_indices]  # [N-K]
        
        sum_activated = sum(activated_probs)          # < 0.5 in most layers
        sum_non_activated = sum(non_activated_probs)  # > 0.5
        
        # Observation: non-activated experts collectively 
        # have higher router confidence than activated ones
```

Gate probability 的两重角色：
1. **Expert 选择**：决定哪些 expert 被激活（top-k selection）
2. **输出加权**：聚合 expert 输出时的权重（weighted sum）

术语一般如何实现？如何使用？
- Softmax 归一化确保所有 expert 概率和为 1
- 负载均衡辅助损失最小化 gate probability 分布的 CV（变异系数），使分布更均匀
- 在 KD 中，SAR 通过 student feedback 微调 gate probability，使知识传递更有效
- Gate probability 是 MoE 知识分布的可视化和诊断工具

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Generalized Knowledge Distillation (GKD)

术语解释
Generalized Knowledge Distillation (GKD) 是 Agarwal et al. (2024, ICLR) 提出的知识蒸馏框架，统一了 on-policy vs off-policy data 和 forward vs reverse KL divergence 的选择，通过广义 Jensen-Shannon (JS) divergence 提供灵活的 teacher-student 知识迁移机制。

术语是什么？
GKD 的两个核心维度：
1. **数据选择**：支持 teacher-generated (off-policy)、固定数据集 (fixed) 或 student-generated (on-policy) 的序列
2. **目标函数**：广义 JS divergence `D_JS^β(p||q) = β·D_KL(p||m) + (1-β)·D_KL(q||m)` where `m = β·p + (1-β)·q`。当 β=0 → forward KL (mode-seeking)，β=1 → reverse KL (mode-covering)，β=0.5 → 原始 JS divergence

GKD 的关键创新：使用 student-generated on-policy 输出替代教师输出进行蒸馏，解决 exposure bias——即 student 在推理时遇到训练中未见过的自回归错误积累。

从算法pipeline角度拆解术语：
```
# GKD (student on-policy, reverse KL, β=1)
def gkd_step(x, teacher, student):
    # 1. Student generates on-policy response
    y_student = student.generate(x, max_new_tokens=512)
    
    # 2. Teacher evaluates student-generated sequence
    teacher_logits = teacher(x, y_student)     # teacher distribution p(y|x)
    student_logits = student(x, y_student)     # student distribution qθ(y|x)
    
    # 3. Reverse KL divergence (or JS divergence)
    L = KL_div(student_logits || teacher_logits)  # reverse KL
    # or L = JS_div(teacher_logits, student_logits, β)
    
    student.backward(L)
```

术语一般如何实现？如何使用？
- Kim et al. (2025) 的 KA 和 SAR 均基于 GKD 的 on-policy + reverse KL 框架构建
- 对比 baseline KD (forward KL + teacher-generated data)，GKD 在 dense teacher 蒸馏中明显更好
- GKD 在 MoE teacher 上的表现不如 KA/SAR，验证了 MoE 专用 KD 方法的必要性
- 实现需要 teacher 和 student 共享 tokenizer（用于 token-level distribution matching）

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Reverse KL Divergence in Knowledge Distillation

术语解释
Reverse KL divergence 是 KD 中的一种分布匹配目标，形式为 `D_KL(qθ || p)` 而非传统的正向 `D_KL(p || qθ)`。其中 p 是教师分布，qθ 是 student 分布。Reverse KL 具有"mode-seeking"行为：student 倾向于聚焦教师的少数高概率 mode，而非尝试覆盖教师的所有 mode（forward KL 的"mean-seeking"行为）。

术语是什么？
正向 KL vs 反向 KL：
- **Forward KL**: `D_KL(p || qθ) = Σ p(y) log(p(y)/qθ(y))` → 惩罚 qθ 在 p 有概率处给低概率 → mean-seeking → student 覆盖教师所有 mode → 可能生成"平均化"输出
- **Reverse KL**: `D_KL(qθ || p) = Σ qθ(y) log(qθ(y)/p(y))` → 惩罚 qθ 在 p 低概率处给高概率 → mode-seeking → student 聚焦教师的高置信度 mode → 生成更精准的输出

在 LLM 文本生成 KD 中，reverse KL 通常优于 forward KL，因为：生成任务有大量低概率但合理的长尾 token，forward KL 会迫使 student 学习这些长尾分布，导致"模糊"或"保守"的生成。

从算法pipeline角度拆解术语：
```
# Reverse KL KD (on-policy, per token)
def reverse_kl_kd_step(x, teacher, student):
    # Student autoregressive generation
    y_tokens = []
    for t in range(max_len):
        student_probs = student(x + y_tokens)         # qθ over vocab
        y_t = sample(student_probs)                    # on-policy sampling
        y_tokens.append(y_t)
    
    # Teacher evaluation on student-generated tokens
    teacher_probs = teacher(x + y_tokens)             # p over vocab
    
    # Reverse KL: D_KL(qθ || p) per token position
    loss = 0
    for t in range(len(y_tokens)):
        loss += sum(student_probs[t][w] * log(student_probs[t][w] / teacher_probs[t][w])
                    for w in vocab)
    return loss
```

术语一般如何实现？如何使用？
- MiniLLM (Gu et al., 2024) 使用 Policy Gradient 优化 reverse KL（因 student 采样不可微）
- GKD (Agarwal et al., 2024) 直接使用 reverse KL 作为可微损失（因 teacher 和 student 共享 tokenizer，在相同 token 序列上计算分布差异）
- KA/SAR (Kim et al., 2025) 使用 reverse KL + student on-policy generation
- 与 forward KL (Sanh, 2019) 不同，reverse KL 要求 KD 在 response tokens 上而非 full sequence 上计算，因为 prompt 部分 teacher 和 student 分布不具有可比性

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## On-policy Knowledge Distillation (在策略知识蒸馏)

术语解释
On-policy KD 是使用 student 模型自身生成的输出（而非 teacher 生成的或数据集中固定的输出）作为蒸馏目标的训练策略。与 off-policy KD（蒸馏数据由 teacher 生成或来自固定数据集）相比，on-policy KD 直接针对 student 的推理分布进行优化，有效缓解 exposure bias。

术语是什么？
Off-policy vs On-policy：
- **Off-policy**: 蒸馏数据来自 teacher 输出或固定数据集 → student 的训练分布 ≠ 推理分布 → exposure bias
- **On-policy**: 蒸馏数据来自 student 自身采样 → student 在训练中遇到的错误模式与其推理时一致 → 蒸馏目标与推理分布一致

On-policy KD 流程（以 GKD 为例）：
1. Student 自回归生成 pseudo-target 序列 `y ~ qθ(·|x)`
2. Teacher 在相同 (x, y) 上计算 logits `p(y|x)`
3. Student 在相同 (x, y) 上计算 logits `qθ(y|x)`
4. 计算分布损失：`L = D_KL(qθ || p)` 或 `L = D_JS(qθ, p)`（使用 student 生成的 y 上的 token-level 分布）

从算法pipeline角度拆解术语：
```
# On-policy vs Off-policy KD comparison
def kd_comparison(teacher, student, x_dataset):
    # Off-policy (traditional KD, Sanh 2019)
    for x, y_golden in x_dataset:                # fixed data
        loss = KL_div(teacher(x, y_golden) || student(x, y_golden))
    
    # Off-policy (teacher-generated)
    for x in x_dataset:
        y_teacher = teacher.generate(x)           # teacher output
        loss = KL_div(teacher(x, y_teacher) || student(x, y_teacher))
    
    # On-policy (GKD, KA, SAR)
    for x in x_dataset:
        y_student = student.generate(x)           # student output!
        loss = KL_div(student(x, y_student) || teacher(x, y_student))
        # teacher evaluates student's own generation
```

术语一般如何实现？如何使用？
- GKD 使用 fixed + on-policy 混合数据训练
- KA 和 SAR 使用纯 on-policy 数据（student 生成 pseudo-target）
- On-policy 训练的关键：teacher 和 student 必须使用相同 tokenizer 以在相同 token 序列上进行分布比较
- 限制：on-policy 生成增加训练计算开销（每次迭代需 student autoregressive 生成）

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Exposure Bias in Sequence Generation (序列生成中的曝光偏差)

术语解释
Exposure bias（曝光偏差）是自回归序列生成模型的训练-推理不一致问题：训练时使用 ground-truth token 作为输入（teacher forcing），推理时使用自身生成的 token 作为输入。当 student 在推理时生成了一个训练中从未见过的错误 token，后续 token 的生成会基于这个错误前缀，导致错误级联放大 (error accumulation)。

术语是什么？
具体表现：
- 训练：`P_θ(y_t | y_{<t}^{gt})`——以 ground-truth history 为条件
- 推理：`P_θ(y_t | y_{<t}^{gen})`——以 self-generated history 为条件
- 差距：`y_{<t}^{gt}` ≠ `y_{<t}^{gen}` 导致分布的系统性偏移

在 KD 中的影响：如果 student 仅在 teacher 生成的 token 序列上训练（off-policy），它习惯于看到"完美"的 context，推理时遇到自己的"非完美"生成就会产生偏离。On-policy KD 通过让 student 在自己的生成序列上训练来消除此偏差。

从算法pipeline角度拆解术语：
```
# Exposure Bias Illustration
# Training (teacher forcing with ground truth):
#   Student sees:  "The cat sat on the..."
#   All previous tokens are correct from ground truth

# Inference (autoregressive generation):
#   Step 1: Student generates "The" ✓
#   Step 2: Student generates "dog" ✗ (wrong token!)
#   Step 3: Student now conditions on "...The dog..." → cascade of errors
#   Student has never trained on sequences starting with its own errors
```

```
# Mitigation via On-policy KD
for x in training_data:
    y_student_tokens = []
    for t in range(max_len):
        probs = student(x + y_student_tokens)    # student's own distribution
        next_tok = sample(probs)
        y_student_tokens.append(next_tok)         # student's own generation
    
    # Now train on student's own (possibly imperfect) generation
    teacher_probs = teacher(x + y_student_tokens)
    loss = KL_div(student(x + y_student_tokens) || teacher_probs)
```

术语一般如何实现？如何使用？
- 解决方案：(a) on-policy KD (GKD, KA, SAR)；(b) Scheduled Sampling——训练时以概率 ε 使用模型自身生成的 token 替代 ground truth；(c) Reverse KL divergence 的 mode-seeking 属性降低对"平均"分布的依赖
- KA 和 SAR 通过 student 自生成 pseudo-target 实现 on-policy 训练
- Exposure bias 在长序列生成中尤为严重（错误随序列长度累积）

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Llama-MoE

术语解释
Llama-MoE 是 Zhu et al. (EMNLP 2024) 提出的一种从 dense LLaMA 模型通过 continual pre-training 构建 MoE 模型的方法。通过将 LLaMA 的 FFN 层转换为 MoE 层并继续训练，在保持原始 LLaMA 能力的同时获得 MoE 架构的计算效率优势。

术语是什么？
Llama-MoE 的构建方法：
- 从 LLaMA checkpoints 出发，将部分 FFN 层替换为 MoE 层
- 使用 Noise Top-k Gating 作为路由机制
- 通过 continual pre-training 训练新增参数（router + experts），同时保留原始权重
- 支持多种配置：k/N 表示每 token 从 N 个 experts 中选 k 个（如 4/16、2/8、2/16）
- 激活参数远少于总参数（sparse activation）

Kim et al. (2025) 使用的变体：Llama-MoE-3.5B (4/16)、Llama-MoE-3.5B (2/8)、Llama-MoE-3.0B (2/16)，其中 3.5B/3.0B 为总参数量，括号内为 (激活 expert 数 / 总 expert 数)。

术语一般如何实现？如何使用？
- Continual pre-training 使用公开语料（如 RedPajama）
- Router 使用 Noise Top-k Gating + auxiliary load balancing loss
- 在 KD 场景中作为 MoE teacher，distill 到 dense student (Sheared-Llama)
- 与 Sheared-Llama 共享 LLaMA tokenizer，满足 KD 中 teacher/student 共享 tokenizer 的要求

涉及论文标题：
- Every Expert Matters: Towards Effective Knowledge Distillation for Mixture-of-Experts Language Models

---

## Stochastic Routing Warmup

术语解释
一种 MoE 训练早期的路由稳定性机制。在训练初期以线性衰减的权重将受控随机噪声混入路由器 logits，迫使 token 均匀分布到所有 expert，防止路由初始化导致的个别 expert 过载或崩溃。由 Ling 团队提出。

术语是什么？
MoE 训练早期（尤其是 fine-grained experts 场景）路由器随机初始化会导致 token 分布极度不均衡——某些 expert 接收远多于容量的 token，造成 OOM 或 expert 负载崩溃。该机制通过混合随机噪声和 learned logits 渐进过渡解决此问题。

公式：ŝ_t = α · s_t + (1 - α) · (μ_s + σ_s · ε)，α = min(i/W, 1.0)，ε ~ N(0, I)。s_t 为路由器线性投影的原始 logits，μ_s/σ_s 为运行时均值和标准差，W 为预热步数。

从算法pipeline角度拆解术语：
```
def moe_warmup_forward(x, step, W):
    s_t = router_linear(x)              # [batch, N_experts]
    if step <= W:
        alpha = step / W
        mu_s, sigma_s = running_mean(s_t), running_std(s_t)
        s_t = alpha * s_t + (1-alpha) * (mu_s + sigma_s * randn_like(s_t))
    return TopK(SoftMax(s_t), k=top_k)
```
α=0 时路由完全随机（所有 expert 等概率），α=1 时完全由学习到的分布控制。

术语一般如何实现？如何使用？
- 预热步数 W 为超参数；μ_s/σ_s 通过 EMA 维护
- 与 dropless routing + load balance loss + z-loss 联合使用
- 可在训练早期完全消除 expert 崩溃问题

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---

## NormHead

术语解释
对 LLM 输出投影层权重进行 L2 归一化后再用于 token 预测的训练稳定性技术。Ling 团队在 MoE 训练中使用，抑制 loss spike 期间的输出 norm 波动。

术语是什么？
Ling 团队发现 LM-Head 的输出 norm 在 loss spike 期间不稳定——权重范数波动放大梯度异常。NormHead 强制 W_lm_head 每行范数为 1 以消除此效应。

公式：h_o = W_lm_head / ||W_lm_head||₂ · h

术语一般如何实现？如何使用？
- 在 LM-Head forward 中插入 F.normalize(weight, p=2, dim=1)
- 与 router z-loss 协同（分别稳定输出层和路由层）
- MoE 训练受益更显著

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---

## Skip Loss Spikes & Sample Retry

术语解释
训练异常自动处理策略：检测到 loss spike 时跳过当前更新，将触发数据随机重注入后续 batch；持续 spike 则自动降学习率。

术语是什么？
MoE 训练中 loss spike 分 narrow（数步影响小）和 wide（多步可致 benchmark 随机水平）。策略：skip→save→re-inject→retry→降 lr 级联。

从算法pipeline角度拆解术语：
```
if is_spike(loss, loss_ema):
    skip_update(); save_data(batch)
    inject_to_future_randomly(batch)
    if retry_count > 0 and is_spike(loss, loss_ema):
        lr *= decay_factor
    retry_count += 1
else:
    backward(); step(); retry_count = 0
```

术语一般如何实现？如何使用？
- spike 检测基于 loss 偏离 EMA 的倍数阈值
- Ling 在 DLRover 中实现，配合自动 checkpoint recovery

涉及论文标题：
- Every FLOP Counts: Scaling a 300B Mixture-of-Experts LING LLM without Premium GPUs

---

## Expert Evolution

术语解释
EvoMoE 提出的一种 MoE expert 初始化策略：从单个可训练 FFN 通过动态混合先验参数和梯度更新，逐步演化出多个功能多样的 MoE expert，替代传统的"复制 FFN 初始化"方式。

术语是什么？
Expert Evolution 解决 MoE-tuning 中的 **Expert Uniformity** 问题——传统方法直接复制（replicate）dense model 的 FFN 参数来初始化多个 expert，导致 expert 在训练后趋同，失去 MoE 架构的多样化优势。Expert Evolution 的核心是 EMA（指数移动平均）形式的参数演化：

$$\theta_n \leftarrow \beta \cdot \theta_1 + (1 - \beta) \cdot \nabla \theta_1, \quad n = 2, 3, \dots, N$$

其中 $\theta_1$ 是唯一可训练的 Expert 1（使用 Stage I 输出初始化），$\theta_n$ 是演化生成的专家，$\nabla \theta_1$ 是 Expert 1 的梯度更新，$\beta \in [0,1]$ 为演化率。不同 expert 使用不同的 $\beta$ 范围（如 [0.9,0.99]、[0.8,0.89]、[0.7,0.79]），从而以不同速率吸收梯度信息，自然产生功能分化。演化后的 expert 和所有其他 LLM/MLP 参数保持冻结。

实验验证：独立评估每个演化后的 expert（不使用 router）发现 Expert 2/3/4 在多个 benchmark 上一致优于 Expert 1（原始 FFN），即使 $\beta=0.9$（仅保留 10% 梯度更新）也能提升性能，证明演化产生的多样性是有效的前非随机的。

从算法pipeline角度拆解术语：
```
# Stage II: Expert Evolution
# 输入：Stage I 输出的密集模型，θ_1 = FFN 参数（Expert 1）
# N = 4 个 expert，top-1 routing

for step = 1 to total_steps:
    # 前向：仅使用 θ_1 作为活跃 expert（Stage II 不开 MoE）
    h = MSA(LN(x)) + x
    y = FFN_1(LN(h)) + h   # FFN_1 即 θ_1
    loss = L_regressive + α * L_aux

    # 反向：仅更新 θ_1
    ∇θ_1 = backward(loss)
    θ_1 = optimizer_step(θ_1, ∇θ_1)

    # 演化其他 expert（每个 step 都执行）：
    for n in [2, 3, 4]:
        β_n = random_uniform(low_n, high_n)
        θ_n ← β_n * θ_1 + (1 - β_n) * ∇θ_1  # EMA 混合

# 输出：4 个具有功能差异的 FFN expert 参数
```

关键设计：
1. 仅 Expert 1 有 optimizer state 和梯度，极大减少训练参数量（vs 全部 expert 参与训练）
2. β 值每步随机采样，增强泛化性
3. β > 0.5 时 expert 才表现差异化，β < 0.5 时趋近于 β=0（纯梯度更新）导致退化
4. 与"加噪声初始化"、"Dropout"、"对比损失"等其他 diversity 策略对比，Expert Evolution 在所有 benchmark 上均优

术语一般如何实现？如何使用？
- 实现于 MoE-tuning 框架的 Stage II，替换原有的 FFN 复制步骤
- 使用 PyTorch 实现：Expert 1 正常参与 backward → optimizer.step()，Expert 2-4 通过 `param.data = beta * expert1_param.data + (1-beta) * expert1_param.grad` 在每个 optimizer step 后更新
- β 从多个预定义范围随机采样，同一范围的 β 通常对应一个特定的 expert（如 Expert 2→[0.9,0.99]）
- 可扩展至任意数量 expert（论文测试 2/4 expert），每增加一个 expert 只需分配一个新的 β 范围

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---

## Dynamic Token-aware Router (DTR)

术语解释
EvoMoE 提出的一种基于 hypernetwork 的 MoE 路由机制：使用两个独立的 hypernetwork 分别处理视觉 token 和文本 token，动态生成每个 token 专用的路由参数，替代传统的静态线性 router。

术语是什么？
DTR 解决 MoE-tuning 中的 **Router Rigidity** 问题：传统 MoE-tuning 使用共享的线性 router (`W_r * h`) 对所有 token 做统一的 top-k expert 选择，无法区分视觉 token 和文本 token 的模态差异，导致 router 输出对输入不敏感（KDE 图显示两种模态的 logit 分布高度重叠）。

DTR 的核心由三个组件构成：
1. **Hypernetwork H_V 和 H_T**：各含两个 MLP layer，接收 token hidden state 作为输入，动态生成 up-sampling 和 down-sampling 层的权重矩阵
2. **动态投影层**：使用 hypernetwork 生成的权重做 down-projection → SwiGLU → up-projection
3. **最终 router φ**：一个 MLP layer，将投影后的特征映射为 expert 概率分布

$$\Theta_{\text{up}}^{\tau}, \Theta_{\text{down}}^{\tau} = \mathcal{H}^{\tau}(z^{\tau\prime}), \quad \tau \in \{V, T\}$$

$$\mathcal{E}^{\tau} = \Theta_{\text{up}}^{\tau} \left( \text{SwiGLU} \left( \Theta_{\text{down}}^{\tau} \left( z^{\tau \prime} \right) \right) \right)$$

$$\rho^{\tau} = \phi(\mathcal{E}^{\tau})$$

其中 $\mathcal{H}^V$ 处理视觉 token，$\mathcal{H}^T$ 处理文本 token。训练时仅 $\mathcal{H}^V$、$\mathcal{H}^T$ 和 $\phi$ 可训练，expert 参数冻结。每个 token 通过 top-1 选择激活概率最高的 expert。

消融实验显示：(1) 模态特定 router（无共享）优于单 router，(2) HyperNet 进一步改善注意力于输入分布，(3) 添加加权共享 router 反而降低性能，(4) DTR（HyperNet + 无共享）结构最优。

从算法pipeline角度拆解术语：
```
# Stage III: DTR Training（每个 MoE decoder layer）
# 输入：MSA 输出 z'（visual tokens V 和 text tokens T 分别处理）
# experts=4, top-k=1，experts 参数冻结

for τ in {V, T}:
    # Step 1: Hypernetwork 生成动态参数
    # H^τ 含两个 MLP：(w1, b1) 和 (w2, b2)
    Θ_up^τ, Θ_down^τ = H^τ(z'^τ)  # 输出 δ/2 → δ 和 δ → δ/2 维矩阵

    # Step 2: Token-aware 动态投影
    h_down = Θ_down^τ @ z'^τ          # down-projection
    h_act = SwiGLU(h_down)           # 门控激活
    h_up = Θ_up^τ @ h_act             # up-projection
    E^τ = h_up                        # Token-specific feature

    # Step 3: 最终 router 预测
    ρ^τ = softmax(φ(E^τ))            # [batch, seq, num_experts]

    # Step 4: Top-1 expert selection
    for each token:
        expert_idx = argmax(ρ^τ)
        output = FFN_{expert_idx}(LN(z'^τ)) + z'^τ

# 损失：L_total = L_regressive + 0.001 * L_aux
```

术语一般如何实现？如何使用？
- 在 MoE decoder layer 中将原始 linear router 替换为 DTR 模块
- DTR 引入的参数增量极小（约 34760 额外参数，仅占模型总参数 < 0.5%）
- 需要两张 embedding table 区分视觉/文本 token 的 modality routing
- 训练使用 DeepSpeed ZeRO-2_offload（Stage III 显存开销较高，因 expert 参数虽冻结仍需驻留）
- 可与 Expert Evolution 的自由演化 expert 配合（Stage II 的产品作为 Stage III 的初始化）

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---

## MoE-tuning

术语解释
一种将 dense MLLM 逐步转换为 sparse MoE 架构的三阶段微调框架：第一阶段预训练对齐跨模态表示，第二阶段扩展为 MoE 结构（复制 FFN + 训练 router），第三阶段 instruction tuning。由 MoE-LLaVA [25] 首次提出，EvoMoE 在此基础上改进 Stage II/III。

术语是什么？
MoE-tuning 解决的核心问题是：直接将 dense LLM 同时转换为 vision-language model 和 sparse MoE 会导致显著的性能下降。因此采用分阶段策略：

- **Stage I (Pre-training + Alignment)**：仅训练 MLP Projector，将视觉 token 映射至 LLM 的语义空间。使用混合多模态数据集（MIMIC-IT、LRV、SViT、LVIS），建立基础的视觉-语言理解能力。
- **Stage II (MoE Initialization)**：将 LLM 中 alternating decoder layer 的 FFN 替换为 MoE 层。传统方式是将原始 FFN 复制 N 份作为 N 个 expert 的初始化，训练线性 router（top-2 selection）和所有参数。EvoMoE 改进为 Expert Evolution（仅训练 1 个 expert + 演化）。
- **Stage III (Instruction Tuning)**：使用 LLaVA-mix-665k 进行指令微调，进一步提升多模态任务表现。EvoMoE 改进为训练 DTR 替代线性 router。

MoE-tuning 的损失函数：
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{regressive}} + \alpha \cdot \mathcal{L}_{\text{aux}}, \quad \alpha = 0.001$$

其中 L_aux 为负载均衡损失：
$$\mathcal{L}_{\text{aux}} = E \cdot \sum_{i=1}^{E} \mathcal{F}_i \cdot \mathcal{G}_i$$

F_i 为每个 expert 处理的 token 比例，G_i 为每个 expert 的平均路由概率。

从算法pipeline角度拆解术语：
```
# MoE-tuning 三阶段（MoE-LLaVA 原始版本）
# Stage I: Pretraining
for epoch in pretrain_epochs:
    img_tokens = VisionEncoder(image)        # CLIP-L
    proj_tokens = MLP_Projector(img_tokens)  # MLP with GeLU
    text_tokens = Tokenizer(text)
    all_tokens = concat(proj_tokens, text_tokens)
    loss = LM_Head(LLM(all_tokens))
    update(MLP_Projector.params)

# Stage II: MoE Initialization
# 将选定的 FFN layers 替换为 MoE：
for layer in MoE_layers:
    layer.experts = [copy(layer.FFN) for _ in range(N)]  # 复制初始化
    layer.router = Linear(hidden_dim, N)                  # 线性 router
# 训练所有参数（除 vision encoder）
for data in LLaVA_mix_665k:
    router_logits = layer.router(token_hidden)
    top_k_indices, top_k_probs = topk(softmax(router_logits), k=2)
    expert_outputs = sum(prob * layer.experts[idx](token) for ...)
    loss = LM_loss + 0.001 * load_balance_loss(L_aux)
    update(all_params)

# Stage III: Instruction Tuning
# 同上，使用更大的数据量进行最终的指令微调
```

术语一般如何实现？如何使用？
- 基于 LLaVA 1.5 代码库实现（https://github.com/haotian-liu/LLaVA）
- MoE-LLaVA 开源仓库：https://github.com/PKU-YuanGroup/MoE-LLaVA
- 使用 DeepSpeed ZeRO-2 进行分布式训练
- MoE layer 采用 alternating placement（每隔若干层放一个 MoE layer）而非全部层替换
- 支持多种 LLM backbone：Qwen 系列、StableLM、Phi-2、OpenChat 等
- 训练硬件：8x A100-80G，bf16 精度
- MoE-tuning 的局限性（由 EvoMoE 揭示并解决）：复制初始化导致 Expert Uniformity、线性 Router 导致 Router Rigidity

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---

## Expert Uniformity

术语解释
MoE tuning 中因复制 FFN 参数初始化 expert 导致的专家同质化现象：所有 expert 从相同的起点出发，经过训练后仍保持高度相似，并未发展出各自特化的功能，违背 MoE 架构的"不同 expert 专精不同任务"的核心设计理念。

术语是什么？
Expert Uniformity 由 EvoMoE 论文通过实验系统性揭示：在 MoE-tuning 训练完成后，随机打乱（shuffle）router 的各层 logits 进行推理，平均性能无明显下降（shuffle 5 次、8 次的 AVG 性能分别为 65.5 和 65.2，与原始 65.5 相当）。这说明 expert 之间没有实质性的功能差异——router 选哪个 expert 对结果影响不大。

根本原因：MoE-tuning 的 expert 初始化方式是"复制原始 dense model 的 FFN 参数 N 份"。所有 expert 起点相同、接收相同梯度、在相同数据上训练 → 训练后趋同。

EvoMoE 通过 Expert Evolution 解决：不同 β 值使各 expert 以不同速率吸收梯度更新 → 自然产生参数分化 → 功能分化。

从算法pipeline角度拆解术语：
Expert Uniformity 的验证实验：
```
# 推理时随机打乱 router logits 评估
for shuffle_trial in [1..8]:
    for layer in MoE_layers:
        # 原本每个 token 的 router logits 对应确定 expert
        logits = layer.router(token_hidden)  # [B, S, N]
        # 随机打乱 logits（在 expert 维度内）
        perm = random_permutation(N)
        logits = logits[:, :, perm]
        selected_expert = argmax(logits)
    # 评估：性能几乎不变 → Expert Uniformity 存在
```

在 MoE LLM 中也可通过以下指标诊断：
- Expert 间的参数 cosine similarity（越接近 1 越均匀）
- Expert 的激活分布差异（KL divergence）
- 不同 expert 在特定 benchmark 上的独立性能差异

术语一般如何实现？如何使用？
- Expert Uniformity 是一个需要诊断和避免的问题，而非可用的方法
- 诊断：shuffle router test、expert 相似度分析、独立 expert 评估
- 解决方案：Expert Evolution（EvoMoE）、Noise Initialization（效果不佳）、Dropout（效果不佳）、对比损失（NCE loss，各 benchmark 表现不一致）、local loss 增加 router entropy（效果有限）
- Expert Uniformity 与 Expert Collapse 不同：Collapse 指少数 expert 接收几乎所有 token（负载不均衡），Uniformity 指 expert 间参数趋同但负载可能均衡

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

---

## Router Rigidity

术语解释
MoE-tuning 中因使用共享静态线性 router 导致的 router 输出僵化现象：router 对所有类型的输入 token（视觉/文本）产生几乎相同的 expert 分布，无法根据输入模态和内容做针对性路由，限制了 MoE 在多模态场景的适应性。

术语是什么？
Router Rigidity 由 EvoMoE 论文通过 KDE（核密度估计）分析揭示：对线性 router 产生的视觉 token logits 和文本 token logits 分别做密度估计，发现两种模态的 logit 分布高度重叠，表明 router 对模态变化不敏感——无论输入是图像 token 还是文本 token，router 给出的 expert 分配几乎一样。

根本原因：传统 MoE-tuning 使用单一线性层 `router = Linear(hidden_dim, num_experts)` 做 expert selection。该线性 router 的参数在训练后固化，对所有 token 使用相同的 W_r 矩阵——没有机制区分 token 来自视觉编码器还是文本 tokenizer。

对 MLLM 的影响：
- 多模态 MoE 的核心价值在于"不同的 expert 处理不同类型的输入"——视觉 expert 处理图像、语言 expert 处理文本
- Router Rigidity 使得这一目标无法实现，所有 token 被均匀分配到所有 expert
- DTR 解决 Router Rigidity 后，可视化显示 visual expert 和 text expert 的激活模式明显分化

从算法pipeline角度拆解术语：
```
# Router Rigidity 诊断（原始 MoE-tuning 的线性 router）
# 输入：visual tokens V ∈ R^{P×C}，text tokens T ∈ R^{M×C}

# 线性 router forward：
logits_V = W_r @ V  # [P, N_experts]  ← 使用相同的 W_r
logits_T = W_r @ T  # [M, N_experts]  ← 使用相同的 W_r

# KDE 分析：
# 对 logits_V 和 logits_T 分别做核密度估计
# 发现两个分布几乎完全重叠 → Router Rigidity
# Expert 分配在不同模态间无差异

# DTR 解决方式（对比）：
Θ_up^V, Θ_down^V = H_V(V)  # 视觉专用 hypernetwork
Θ_up^T, Θ_down^T = H_T(T)  # 文本专用 hypernetwork
# 不同模态走不同的参数生成路径，router 输出自然分化
```

Router Rigidity 不同于 Router Collapse（router 总是选择同一 expert），也不同于 Expert Uniformity（expert 参数趋同）。三者可同时存在但属于不同层面的问题：Router Rigidity 是 router 层的问题，Expert Uniformity 是 expert 层的问题。

术语一般如何实现？如何使用？
- Router Rigidity 是需要诊断和避免的问题
- 诊断：KDE plot（论文方法）、模态间 expert 分配的 Jensen-Shannon 距离、token 来源和 expert 选择的互信息
- 解决：DTR（EvoMoE，hypernetwork 动态生成 router 参数）、modality-specific router（无 hypernetwork 也有改善）、RoE [ICLR 2025]（adapter-based layer skipping）
- 在纯文本 LLM MoE 中不存在此问题（所有 token 同质），仅在 MLLM 中有意义

涉及论文标题：
- EvoMoE: Expert Evolution in Mixture of Experts for Multimodal Large Language Models

## GrAP (Grouped Average Pooling)

术语解释
GrAP 是 LocMoE/ETR 论文提出的用于 MoE 路由层的新型特征提取层，替代传统的全连接 MLP Router。GrAP 按 expert 数量 n 对输入 hidden dimension d 进行分组平均池化，生成对角稀疏亲和力矩阵 W_aff ∈ R^{d×n}，参数仅为传统 Router 的 1/n.

术语是什么？
传统 MLP Router W_g ∈ R^{d×n} 是稠密参数矩阵，计算和存储开销为 O(d²)，且随机初始化的权重向量 w_i 之间不正交，容易导致多个 expert 学到相似路由模式（expert homogenization）。GrAP 将 d 维 token hidden state 按 expert 数 n 均分为 n 组，每组内取平均值，构成对角稀疏 W_aff：

$$W_{\text{aff}} = \operatorname{diag}(w_1, w_2, \ldots, w_n), \quad w_i[j] = \frac{n}{d} \text{ for } j \in [\frac{i \cdot d}{n}, \frac{(i+1) \cdot d}{n})$$

每个 w_i 仅在其分组内非零，不同 w_i 天然正交 (w_i · w_j = 0 for i ≠ j)。然后通过 cosine similarity δ_{t,i} = cos(x_t, w_i) 计算 token-expert 亲和力分数。

从算法pipeline角度拆解：
```
Input: x ∈ R^{s×d} (s tokens, d hidden dim)
Output: δ ∈ R^{s×n} (affinity scores)

# GrAP 前向: 对角稀疏，仅 d 个非零参数
W_aff = zeros(d, n)
for i in range(n):
    start = i * d // n
    end = (i+1) * d // n
    W_aff[start:end, i] = n / d

# 亲和力分数 (cosine similarity)
x_norm = L2_normalize(x, dim=-1)       # O(sd)
w_norm = L2_normalize(W_aff, dim=0)    # O(d)
delta = x_norm @ w_norm                # O(sd), 等价于分组平均池化 + cosine
```
对比传统 MLP Router: W_g ∈ R^{d×n} 全参数矩阵, O(s·d·n) 计算。GrAP 仅 O(s·d)，参数量降为 1/n。

术语一般如何实现？如何使用？
GrAP 在华为 MindSpeed-LLM 框架中实现，运行于 Ascend NPU 的 AI VECTOR CORE（cosine similarity 是向量操作）。分组数通常等于 expert 数 n。正交 gating 权重将 token space 按角度划分为 n 个扇区，每个 expert 对应一个扇区——等价于隐式 spherical k-means 聚类。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Token-Choice Routing (TCR)

术语解释
Token-Choice Routing 是 MoE 中最基础的路由策略：每个 token 独立选择 top-k 个 expert 进行处理，由 Shazeer et al. (2017) 在 Sparsely-Gated MoE 中首次提出。

术语是什么？
TCR 流程: (1) Router 对每个 token x_t 计算 gate scores; (2) 取 top-k (通常 k=1 或 2) 最高分 expert; (3) token 被 dispatch 至选中 expert 进行 FFN 计算。TCR 优势：token 有充分自由选择最适合的 expert。缺陷：load imbalance——某些 expert 收到远多于 capacity 的 token。ETR 论文证明 (Theorem 5)：早期训练阶段 (class-irrelevant token 呈各向同性分布，q_i = Θ(1))，TCR 训练成功率为 Θ(C·Σp_i/s)，显著优于 ECR 的指数衰减率 e^{-s}。

从算法pipeline角度拆解：
```
def TCR_route(scores, k, C):
    # scores: s×n, k: top-k, C: expert capacity
    topk_val, topk_idx = TopK(scores, k)  # s×k
    capacity = zeros(n)
    dispatch = {i: [] for i in range(n)}
    for t in range(s):
        for expert_id in topk_idx[t]:
            if capacity[expert_id] < C:
                dispatch[expert_id].append(t)
                capacity[expert_id] += 1
            # else: token dropped → residual bypass
    return dispatch
```

术语一般如何实现？如何使用？
TCR 在现代 MoE 框架 (Megatron-LM、DeepSpeed-MoE、MindSpeed-LLM) 中广泛实现。使用 All-to-All 通信在 Expert Parallelism 维度上完成 token dispatch/combine。是 GShard、Switch Transformer、Mixtral、DeepSeek-V3 的默认路由策略。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Bidirectional Routing (TCR+ECR Hybrid) / Affinity-Driven Active Selection

术语解释
Bidirectional Routing 是 ETR 的核心创新：在 MoE 路由中同时使用 TCR 和 ECR，让 token 和 expert 双向主动选择，形成"共振效应"。整个流程由 cosine similarity 亲和力分数统一驱动 (Affinity-Driven Active Selection)。

术语是什么？
ETR 双向路由分两阶段: (1) TCR: 每个 token 按 GrAP 亲和力分数 δ_{t,i} = cos(x_t, w_i) 选 top-ℓ experts; (2) ECR: 每个 expert i 从已分配 token 中按其 δ 选 top-C tokens (Bottom-C 保留最高分数)。动态过渡：早期训练 TCR 更优 (q_i ≈ Θ(1))，后期 ECR 更优 (q_i << 1, 接近 100% 成功率)。Theorem 5 提供了全程最大化成功率的最优过渡策略的理论依据。

从算法pipeline角度拆解：
```
def ETR_bidirectional(x, W_aff, k, C):
    # Step 1: 亲和力分数
    delta = cosine_similarity(x, W_aff)  # s×n

    # Step 2: TCR — token 选 top-k experts
    tcr_assign = defaultdict(list)
    for t in range(s):
        for expert_id in TopK(delta[t, :], k):
            tcr_assign[expert_id].append(t)

    # Step 3: ECR — expert 选 top-C tokens
    ecr_assign = {}
    for i in range(n):
        candidates = tcr_assign[i]
        if len(candidates) <= C:
            ecr_assign[i] = candidates
        else:
            scores = [delta[t, i] for t in candidates]
            ecr_assign[i] = BottomC(candidates, scores, c=C)

    return ecr_assign
```

术语一般如何实现？如何使用？
在 Ascend NPU 上通过 MindSpeed-LLM 实现。TCR 阶段用 TopK 算子，ECR 阶段用 BottomC/IndexPutV2 做 token rearrangement。引入的 TopK/IndexPutV2 开销较小，但使 FFN MatMul 获 17× 加速 (相对 baseline)，因为只计算高亲和力 token-expert 对。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Expert Capacity / Adaptive Capacity in MoE

术语解释
Expert Capacity 是 MoE 路由中每个 expert 能处理的最大 token 数。传统方法用固定容量 C = capacity_factor × s/n。ETR 提出自适应容量策略，根据训练进度和亲和力分数动态调整 C。

术语是什么？
Expert capacity 决定每个 expert 一次前向中能处理多少 token。容量过低→过多 token drop (影响质量)，过高→浪费计算和通信 padding。ETR 证明自适应容量可将下界降低最多 40%:

$$C_{\min} = \frac{1}{n} \exp\left(\frac{d \cdot \delta_{\max}^2}{2 - \delta_{\max}^2}\right)$$

δ_max 为 gating weight 与 token 间的最大角度偏差。训练早期 δ_max 大 (token 分布分散)，C 需较大；后期 token 特征收敛，δ_max 减小，C 可显著降低。

术语一般如何实现？如何使用？
在 ETR 中，C 每 step 动态计算：统计当前 batch 的亲和力分数分布，计算 δ_max，代入 C_min 公式，取 max(C_min, s/n) 为最终容量。降低 C 直接减少 expert FFN 计算所需的中间 buffer，减少显存 4.57%-16.27%。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Locality Loss (MoE 局部性损失)

术语解释
Locality Loss 是 LocMoE/ETR 系列提出的负载均衡辅助损失，在 traditional auxiliary loss 基础上引入数据局部性约束：通过 KL 散度惩罚 token 被路由到非本地节点 expert，鼓励同节点路由，减少跨节点 All-to-All 通信。

术语是什么：
L_loc = μ · KL(D_c || D_l)，D_c 为当前 token-to-expert 分配的经验分布，D_l 为全局部化分布 (所有 token 仅分配至本地节点 expert)。当 expert 数 ≥ 节点数时效果最优——每个节点至少有一个 expert。论文验证: 32N/64N 配置下 locality loss 加速效果最显著。

术语一般如何实现？如何使用？
在 MindSpeed-LLM 中作为附加 loss 项实现：根据 expert-to-node 映射表构建 D_l 分布，计算 KL 散度加到总 loss。Locality Loss 仅影响路由决策 (gradient 通过 router 反向传播)，不影响 expert FFN 权重。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Expert Homogenization (专家同质化)

术语解释
Expert Homogenization 是 MoE 训练中的典型退化现象：多个 expert 学到相似特征表示，导致 MoE 退化为近似 dense model，丧失稀疏激活的效率和多样性优势。

术语是什么？
表现: (1) 不同 expert FFN 权重趋同 (cosine similarity 接近 1); (2) router 分配概率接近均匀 (失去区分能力); (3) 随机 shuffle router 分配不影响模型性能。ETR 通过 GrAP 的正交 gating weight 天然防止同质化——每个 w_i 对应 hidden space 的不同扇区，expert 被迫学习其扇区内 token 的专用表征。

同质化的恶性循环:
```
Router随机初始 → 部分expert被过度选择 (rich-get-richer)
→ 被选expert梯度更新更多 → expert能力分化不足
→ router难以区分expert差异 → 继续随机/偏向性路由
→ expert进一步同质化
```

ETR 的打断机制:
```
GrAP正交权重 → 每个expert对应不相交hidden space扇区
→ TCR确保token选最匹配扇区
→ ECR确保expert只处理其扇区内高亲和力token
→ expert被迫在其扇区内专业化 → 正向反馈循环
```

术语一般如何实现？如何使用？
检测: (1) Calinski-Harabasz (CH) Index 测量 expert 间 token 聚类质量 (ETR 使用); (2) expert FFN 权重 pairwise cosine similarity; (3) shuffle router 分配后的性能下降幅度。防止: 正交 router (GrAP)、contrastive loss、mutual distillation loss、expert dropout。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Training Success Rate in MoE

术语解释
Training Success Rate 是 ETR 论文为定量比较 TCR 和 ECR 而定义的理论指标：一次训练 step 中，输入样本 x 的 class-discriminative pattern o_i 被正确分发到第 i 个 expert 的概率。

术语是什么？
定义 (Definition 2): 给定 x ∈ R^{s×d} (s tokens, 含 1 个 class-discriminative pattern o_i 和 s-1 个 class-irrelevant pattern r ∼ N)，若 o_i 被正确 dispatch 到第 i 个 expert，则 x 在此 step "训练成功"。Training Success Rate = P(x succeed in training)。

关键理论结果 (Theorem 5 + Corollary 6):
- TCR 成功率: Θ(C·Σp_i/s), p_i = P(δ_{o_i,i} ≥ δ_{x_j,i}) ≥ 1/n
- ECR 成功率: 当 C ≤ (s-1)q_i/2 时 ≤ (1/n)·Σe^{-(s-1)q_i/8}; 当 C ≥ 2s·q_i 时 ≥ 1-e^{-3C/16}
- q_i = P(r 的分数 > o_i 的分数) 衡量 expert 判别能力

术语一般如何实现？如何使用？
纯理论概念，不在代码中直接实现。价值在于提供 TCR→ECR 过渡的理论依据：早期 q_i ≈ Θ(1)→TCR 更优需 C=Θ(s)；后期 q_i << 1→ECR 更优仅需 C=Θ(1)，容量降低 ~40%。

涉及论文标题：
- Expert-Token Resonance Redefining MoE Routing through Affinity-Driven Active Selection

## Inter-Layer Expert Affinity (跨层专家亲和性)

术语解释
Inter-Layer Expert Affinity（跨层专家亲和性）是 ExFlow 论文提出的概念：在 pre-trained GPT MoE 模型中，给定一个 token 在 layer i 被 route 到某个 expert，该 token 在 layer i+1 及其后续层中被 route 到特定 experts 的条件概率并非均匀分布，而是表现出强烈的非随机模式——某些 expert pairs 在跨层 routing 中具有显著更高的共现概率。这种跨层 conditional routing probability 即为 Expert Affinity。

术语是什么？
Expert Affinity 的数学形式为 conditional probability $P(E_{p,j+1}|E_{i,j})$，即在 layer j 被 route 到 expert $E_i$ 的 token，在 layer j+1 被 route 到 expert $E_p$ 的概率。ExFlow 论文通过采样 Pile 数据集的 token 并 trace 其在每层的 routing 决策，构建了完整的 cross-layer routing heatmap（Fig. 2），证实了所有 layers 对之间均存在显著的 expert affinity，且该 affinity 是模型固有属性——在 OOD 数据集（C4、Dolma、Yelp Reviews）上归一化 affinity 保持 0.989-1.005 的高度一致性。

ExFlow 进一步将 expert affinity 分为两级：
- **Intra-GPU Affinity**（第一级）：expert 对其在后续层中与该 expert 同 GPU 的 affiliated expert 的偏好
- **Intra-Node Affinity**（第二级）：expert 对其在后续层中与该 expert 同节点但不同 GPU 的 expert 的偏好

从算法pipeline角度拆解术语：
Expert Affinity 通过以下算法流程捕捉和利用：

```
# ===== 1. Profiling: 捕捉 Expert Affinity =====
# 从 Pile 随机采样 N=3000 tokens
route_log = []  # route_log[k][j] = expert_idx

for token k in sampled_tokens:
    hidden = embedding(token)
    for layer j in 0..L-1:
        gate_scores = softmax(hidden @ W_gate[j])  # Top-1 gating
        expert_idx = argmax(gate_scores)
        route_log[k][j] = expert_idx
        hidden = expert_compute(expert_idx, hidden)
        hidden = attention(hidden, context)

# ===== 2. 构建 Conditional Probability Matrix =====
# P[j][i][p] = P(E_{p,j+1} | E_{i,j})
for layer j in 0..L-2:
    for token k in 0..N-1:
        i = route_log[k][j]      # token k 在 layer j 的 expert
        p = route_log[k][j+1]    # token k 在 layer j+1 的 expert
        count[j][i][p] += 1

for layer j, expert i, expert p:
    P[j][i][p] = count[j][i][p] / sum_over_p(count[j][i][:])

# ===== 3. Combined Affinity for Multi-Expert GPU =====
# GPU with capacity C_1 experts/层: holding experts {x_1,...,x_C1} at layer j
# Find experts {y_1,...,y_C1} at layer j+1 maximizing:
# score = sum_{k} sum_{p=1..C1} sum_{q=1..C1} P(E_{y_q,j+1} | E_{x_p,j}, T_k)
```

Expert affinity 的演化特性（Fig. 12）：
- 训练初期（iteration 0-200）：expert routing 高度不平衡（少数 expert 被频繁激活），apparent affinity 极高
- 过渡期（iteration 200-2000）：GShard load balancing loss 生效，expert 激活趋于均匀，affinity 下降
- 稳定期（iteration 2000+）：expert 逐渐变得 domain-specific，affinity 稳步上升并趋于稳定

术语一般如何实现？如何使用？
- **Profiling 成本极低**：仅需 1000-3000 个 token 即可精确捕捉 expert affinity（MoE-8 仅需 1000，MoE-64 需 3000）
- **Offline 使用**：在推理前一次性完成 profiling 和 ILP 求解，结果直接用于模型加载时的 expert placement
- **无需修改模型**：不改变 gating function、不添加 training loss、不需要 fine-tuning
- **硬件拓扑无关**：expert affinity 是 model-intrinsic 属性，placement 算法可适应任意硬件拓扑
- **与 expert popularity 的区别**：之前的工作（Lina、EdgeMoE）仅考虑 per-layer 的 expert popularity（单一层的热门 expert），而 Expert Affinity 是 cross-layer 的 conditional relationship

涉及论文标题：
- Exploiting Inter-Layer Expert Affinity for Accelerating Mixture-of-Experts Model Inference

## Task Sensitivity to Token-to-Expert Routing Accuracy (任务对 Token-to-Expert 路由准确度的敏感度)

术语解释
Task Sensitivity to Token-to-Expert Routing Accuracy 是指不同 NLP 任务对 MoE 模型中 expert routing 错误的容忍度存在显著差异的观察。分类和语义相似度任务即使使用随机 routing 仍能保持高输出质量，而对话和摘要等开放式任务对 routing accuracy 高度敏感。这一发现是 task-aware expert loading 等系统优化的算法基础。

术语是什么？
MoE 模型的 router gate 决定每个 token 由哪些 expert 处理。当部分 MoE layer 的 routing 不准确（token 被路由到非最优 expert）时，不同任务的输出质量退化程度不同。eMoE 通过 progressive inaccurate routing 实验（从靠近 input 的层开始逐层应用 random routing，测量输出与 full-model ground truth 的 BERT semantic similarity）：

**实验结果（eMoE §2.2.3，Figure 5）**：
- **Classification / Comparison tasks**：即使 100% layers routing 不准确，similarity 仍 >90%
- **Conversation / Summarization tasks**：75% layers 准确时 similarity 已 <80%
- **QA tasks**：50% layers 准确时 similarity 仍在 >80%
- **Summarization tasks**：50% layers 准确时 similarity <80%

根因分析：靠近 input 的神经网络层倾向于学习 general representations，靠近 output 的层 specialize 为 task-specific representations。任务对 input-side layers 的 routing accuracy 的依赖度不同——开放式生成任务需要更精确的语义分解，分类任务仅依赖高层语义特征。

从算法pipeline角度拆解术语：
MoE 推理中 token routing 与 task sensitivity 的关系：

```
=== MoE Layer 的 Token-to-Expert Routing ===

Input: token embedding x ∈ R^d
For each MoE layer L_i (i = 1..m):
  # Step 1: Gate computation
  g = softmax(W_g · x)  # W_g ∈ R^{d × N_experts}
  
  # Step 2: Top-K selection (accurate routing)
  top_k_indices = argtopK(g, k)  # e.g., k=2
  top_k_weights = g[top_k_indices]
  
  # Step 3: Expert FFN computation
  output = Σ_{j∈top_k} top_k_weights[j] · Expert_j(x)
  # Expert_j(x) = W_out_j · σ(W_in_j · x)

=== Inaccurate Routing (simulated in sensitivity experiment) ===
For first L inaccurate layers (L = 0%..100% of total MoE layers):
  # Replace accurate routing with random selection
  random_k_indices = random_sample(N_experts, k)
  random_k_weights = uniform(k)  # equal weights
  output = Σ_{j∈random} random_k_weights[j] · Expert_j(x)

For remaining (m-L) accurate layers:
  output = standard top-K routing  # as above

=== Task Sensitivity Metric ===
sensitivity(task, layer_range) = 
  BERT_similarity(output_with_inaccurate_routing[0:layer_range], 
                  output_with_full_accurate_routing)

# Classification: s ≈ 0.9+ even when layer_range = 100%
# Conversation: s < 0.8 when layer_range = 75%
```

术语一般如何实现？如何使用？
- Sensitivity profiling 是 offline 一次性操作：对每个 task type 在目标 MoE model 上运行 progressive inaccurate routing 实验
- 结果存储为 per-task per-layer sensitivity matrix：`s[T][L] ∈ {0, 1}`（1=sensitive, 0=insensitive），threshold 通常设为 85% similarity
- 应用：Task-aware Expert Loading（eMoE）、task-aware model compression、task-specific expert pruning
- 与 Layer-wise Importance 的区别：layer importance（如 early exit 文献）通常仅考虑 layer depth，不考虑 task type 维度
- 与 existing work 的联系：Zeiler & Fergus (ECCV 2014) 发现 lower layers learn general features, higher layers learn task-specific features —— 这一神经网络原理在 MoE routing 语境下被 eMoE 扩展到 task type 维度

涉及论文标题：
- eMoE: Task-aware Memory Efficient Mixture-of-Experts-Based (MoE) Model Inference

---

## Penalty-Balanced Top-k Routing (PBTk, 惩罚均衡Top-k路由)

术语解释
Penalty-Balanced Top-k (PBTk) Routing 是 MoE 中最主流的负载均衡路由策略：在语言模型主损失之上添加辅助惩罚项（Auxiliary Loss + Z-Loss），通过梯度反向传播间接约束 router 将 token 均匀分配到各 expert，而非在路由前显式修改路由决策。PBTk 被 DeepSeek-V2/V3、Qwen-MoE、Mixtral 等 SOTA MoE 广泛采用。

术语是什么？
PBTk 路由组合使用两种惩罚损失：
1. **Auxiliary Loss (L_aux)**：L_aux = α · E · Σᵢ fᵢ · Pᵢ，其中 fᵢ = 分配给 expert i 的 token 比例，Pᵢ = router 对 expert i 的平均 softmax 概率。α 典型值 0.01（本文及 ST-MoE）。该损失在 fᵢ = Pᵢ = 1/E 时最小，鼓励均匀分配。
2. **Z-Loss (L_z)**：L_z = λ_z · log²(Σⱼ exp(logitⱼ))，λ_z 典型值 0.001。惩罚 router logit 的过大值，提升训练数值稳定性（Zoph et al., 2022）。

总损失：L_total = L_LM + 0.01 · L_aux + 0.001 · L_z

PBTk 的核心权衡：α 太小 → 路由崩溃（仅少数 expert 被使用）；α 太大 → 梯度干扰主任务，降低模型性能。DeepSeek-V3 后来引入 auxiliary-loss-free 策略（expert-level bias 动态调整）以消除此权衡。

从算法pipeline角度拆解术语：
```python
# PBTk Router forward
def pbtk_router_forward(x, W_router, experts, shared_expert=None):
    # x: [B, S, H] batch of token hidden states
    logits = x @ W_router                    # [B, S, E], E=num experts
    probs = softmax(logits, dim=-1)          # [B, S, E]

    # Top-k selection (no modification to probs)
    topk_probs, topk_indices = topk(probs, k=K)  # K=1 for Switch, K=3 for Granular

    # Expert computation
    output = zeros_like(x)
    if shared_expert is not None:
        output += shared_expert(x)           # shared expert always active

    for i, expert_idx in enumerate(topk_indices):
        expert_out = experts[expert_idx](x)  # GEGLU FFN
        output += topk_probs[i] * expert_out
    output /= sum(topk_probs)                # normalize

    return output, logits, topk_indices

# Loss computation (at training step)
def compute_pbtk_loss(lm_loss, router_logits, topk_indices, batch_size):
    # Auxiliary loss
    f = zeros(E)     # fraction of tokens per expert
    P = zeros(E)     # avg router probability per expert
    for layer_logits, layer_indices in zip(all_router_logits, all_topk_indices):
        probs = softmax(layer_logits, dim=-1)
        for e in range(E):
            f[e] += count(layer_indices == e)
            P[e] += probs[:, :, e].sum()
    f /= total_tokens; P /= total_tokens
    L_aux = 0.01 * E * sum(f[e] * P[e] for e in range(E))

    # Z-loss
    L_z = 0.001 * sum(logsumexp(logits).square().mean() for logits in all_router_logits)

    return lm_loss + L_aux + L_z
```

术语一般如何实现？如何使用？
- **系数选择**：本文使用 α=0.01, λ_z=0.001，与 ST-MoE (Zoph et al., 2022) 和 DeepSeekMoE (Dai et al., 2024) 一致
- **CPT 中的行为**：PBTk 在分布偏移时经历短暂的 MRI spike（路由不均衡激增），但在 ~500 steps 内恢复到比 SBTk 更低的 MRI 水平。说明 PBTk 的路由对分布偏移具有"恢复性鲁棒"（resiliently robust）而非"固有鲁棒"（inherently robust）
- **与 SBTk 的差异**：PBTk 在稳定状态下的 MRI 低于 SBTk（更好的负载均衡），但分布偏移时需要短暂适应期
- **Decayed vs Non-decayed checkpoint**：从衰减 checkpoint 开始 CPT 的 PBTk，在分布偏移后的 MRI spike 稍高（~2-3%），但仍能快速恢复

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Sinkhorn-Balanced Top-k Routing (SBTk, Sinkhorn均衡Top-k路由)

术语解释
Sinkhorn-Balanced Top-k (SBTk) Routing 是将 MoE 的 token-to-expert 分配建模为最优传输（Optimal Transport）问题并通过 Sinkhorn-Knopp 算法迭代求解的路由策略。与 PBTk 的"事后惩罚"不同，SBTk 在路由决策阶段显式调整 routing probabilities 以达到负载均衡。首次由 Clark et al. (2022) 引入 MoE 路由，后由 Anthony et al. (2024) 通过有利初始条件加速收敛。

术语是什么？
SBTk 将路由建模为带熵正则的 Kantorovich 最优传输问题：在满足每 token 概率和为 1、每 expert 接收等量 token 的约束下，最小化路由代价。Sinkhorn-Knopp 算法通过迭代行归一化和列归一化给出近似解：
```
K = exp(logits / ε)          # 将 logits 转为正矩阵，ε 控制熵正则强度
repeat until convergence:
    K = row_normalize(K)     # 每 token 的概率和为 1
    K = col_normalize(K)     # 每 expert 接收等量 token
probs = K                    # 最终的均衡路由概率
```
然后从调整后的 probs 中选 top-k experts。Anthony et al. (2024) 使用分组均值作为初始条件加速收敛。

关键区别：SBTk 的均衡步骤在推理时不可用（Sinkhorn 需要整 batch 的 token 信息，与自回归生成逐 token 解码不兼容），因此推理时必须移除 Sinkhorn 步骤（fall back to greedy top-k），导致推理时的 MRI 高于训练时。

从算法pipeline角度拆解术语：
```python
def sinkhorn_routing(logits, n_iters=5, tol=0.01):
    # logits: [T, E]  T=num_tokens, E=num_experts
    # Anthony et al. (2024) initialization: group-mean of logits
    K = exp(logits)  # exponentiate
    for _ in range(n_iters):
        K = K / K.sum(dim=1, keepdim=True)   # row normalize (token sum=1)
        K = K / K.sum(dim=0, keepdim=True)   # col normalize (expert receives 1/E)
    return K  # doubly stochastic routing matrix

# During training:
probs = sinkhorn_routing(logits)
topk_probs, topk_indices = topk(probs, k=K)

# During inference (auto-regressive):
probs = softmax(logits)      # NO Sinkhorn step
topk_probs, topk_indices = topk(probs, k=K)
```

术语一般如何实现？如何使用？
- **收敛性**：Sinkhorn 迭代收敛到满足双随机约束的矩阵，tolerance 0.01（本文设置）
- **CPT 中的行为**：SBTk 对分布偏移具有"固有鲁棒性"——分布偏移时 MRI 几乎不变（因为显式均衡步骤强制保证），但稳定状态的 MRI 高于 PBTk（均衡不如 PBTk 精细）
- **推理不兼容**：Sinkhorn 需要 batch-level 统计信息，与自回归生成不兼容。推理时 MRI 高于训练时 MRI
- **计算开销**：SBTk 的 forward 和 backward 时间均高于 PBTk（本文 Table 2：SB Granular MoE ~1789ms/step vs PB Granular MoE ~1680ms/step）

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Continual Pre-training (CPT) of LLMs (大语言模型持续预训练)

术语解释
Continual Pre-training (CPT) 是指在已有预训练模型基础上，用大规模新数据（>100B tokens）继续训练以扩展模型能力，而非从头重新训练。与 fine-tuning 的关键区别在于数据规模：fine-tuning 通常使用 MB~GB 级数据，CPT 使用 >100B tokens。CPT 已经在 dense LLM 中被证明可以有效替代 full re-training（Ibrahim et al., 2024），但在 MoE 中的行为此前未被系统研究。

术语是什么？
CPT 的核心挑战是灾难性遗忘（catastrophic forgetting）：模型在学习新分布时丢失旧分布上的能力。Ibrahim et al. (2024) 建立了 dense LLM CPT 的三项核心技术：
1. **LR Re-warming + Re-decaying**：从衰减 checkpoint 开始时，需重新 warm up LR 到 η_max 再 cosine decay 到 η_min
2. **Infinite LR Schedule (CosineInf)**：预训练时就使用不终止的 LR 方案，CPT 时从 constant phase 平滑过渡，无需 re-warming
3. **Replay**：CPT 时混合一定比例的旧数据以减缓遗忘

本文首次将这三项技术应用到 MoE，并证明了 MoE 在 CPT 中：1) 保持 sample efficiency 优势；2) 路由算法对分布偏移具有鲁棒性；3) 可以匹配 full re-training 性能（仅 ~1/3 成本）。

从算法pipeline角度拆解术语：
```
# CPT Pipeline (本文: FineWeb→German, 40% replay, CosineInf)
# Phase 1: Pre-training (FineWeb, 400B tokens)
model = init_moe_or_dense()
scheduler = CosineInf(total=192720, warmup=1%, const=10%, cooldown=70%)
for step in range(192720):
    batch = sample(FineWeb, bs=1024, seq=2048)
    loss = model(batch)
    optimizer.step()

save_checkpoint(model, phase="const", lr=1.65e-4)  # save at constant phase

# Phase 2: CPT (FineWeb→German, 200B tokens)
load_checkpoint(model, phase="const")
scheduler = CosineInf(total=95370, warmup=1%, const=80%)  # no cooldown yet
for step in range(95370):
    batch_replay = sample(FineWeb, bs=410, seq=2048)   # 40% replay
    batch_new = sample(GermanCC, bs=614, seq=2048)      # 60% new
    batch = concat([batch_replay, batch_new])
    loss = model(batch)
    optimizer.step()
```

术语一般如何实现？如何使用？
- **Compute-equivalent replay**：replay 比例增加时不增加总 token 预算，而是减少新数据量，保证不同 replay 比例的 compute 可比
- **Replay 比例**：本文 German CPT 用 40%，Stack CPT 用 30%（遵循 DeepSeek-CoderV2 的设定）
- **Overtraining regime**：本文 600B tokens 训练 570M/2B 模型，对应 dense 的 ~40× Chinchilla optimal，MoE 的 ~10×，代表真实应用场景
- **与 full re-training 的对比**：CPT 仅消耗 ~1/3 FLOPs（因为仅训练 200B 而非 600B tokens），但性能匹配或超越 full re-training

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Maximum Routing Imbalance (MRI, 最大路由不均衡度)

术语解释
Maximum Routing Imbalance (MRI) 是本文提出的 MoE 负载均衡度量指标：在给定 MoE 层 j 和 batch B 中，路由到单个 expert 的最大 token 占比。MRI 是 MoE 推理最坏情况延迟的代理（proxy）：由于 expert parallelism 下每个 accelerator 的计算量正比于其上的 expert 负载，MRI 越高意味着最忙的 accelerator 延迟越大，硬件利用率越低。

术语是什么？
对于 MoE 层 j，训练迭代步 t，MRI 定义为：
$$MRI(t, j) = \max_{i \in [1,\dots,E]} \left[ \frac{\sum_{x \in B} \mathbb{1}\{i \in I_k(x)\}}{|B|} \right]$$

其中 E 是 routed experts 数量，B 是 batch 中所有 token 集合，I_k(x) 是 token x 的 top-k 选中 expert 索引集合，𝟙 是指示函数。

MRI 取值范围：[1/E, 1]（从完美均衡到单 expert 垄断）。MRI 越高 → 最坏情况延迟越大 → 硬件利用率越低。

与延迟的关系：MRI 不直接报告延迟，而是作为延迟模型的输入。相比硬件和实现特定的延迟指标，MRI 是跨部署场景可比较的行为指标。

从算法pipeline角度拆解术语：
```python
def compute_mri_layer(layer, batch_tokens):
    """Compute MRI for a single MoE layer"""
    E = layer.num_experts
    expert_loads = zeros(E)
    for token in batch_tokens:
        topk_indices = layer.route(token)  # top-k selected experts
        for idx in topk_indices:
            expert_loads[idx] += 1
    expert_loads /= len(batch_tokens)  # normalize to proportions
    return expert_loads.max()          # MRI for this layer

# Layer-wise MRI analysis (as in paper):
def compute_mri_model(model, test_dataset, num_tokens=20_000_000):
    """Compute per-layer MRI on test set"""
    mri_per_layer = defaultdict(list)
    for batch in test_dataset.iterate(num_tokens):
        for j, layer in enumerate(model.moe_layers):
            mri_per_layer[j].append(compute_mri_layer(layer, batch))
    return {j: median(mris) for j, mris in mri_per_layer.items()}
```

术语一般如何实现？如何使用？
- **训练中监控**：MRI 在训练过程中实时计算（per step/per layer），用于检测分布偏移时的路由崩溃
- **最终 checkpoint 评估**：在 20M token 测试集上计算 per-layer MRI，评估最终模型的负载均衡质量
- **OOD 检测**：MRI 在 out-of-distribution 数据上显著升高（如非 German 模型在 German 数据上的 MRI），可作为分布偏移检测信号
- **Early layer 关注**：Switch MoE 中 early layers (0-6) 的 MRI 始终最高，是推理延迟的瓶颈层
- **推理预估算**：MRI × accelerator_count × expert_compute_time ≈ 最坏情况推理延迟（作为延迟模型的简化输入）

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## CosineInf Schedule (无限余弦学习率调度)

术语解释
CosineInf (Cosine Infinite) Schedule 是由 Ibrahim et al. (2024) 提出的用于持续预训练（CPT）的学习率调度方案。与标准 Cosine Annealing（需要预先指定总训练步数）不同，CosineInf 在 cosine decay 后进入恒定学习率阶段（constant phase），允许无限期继续训练。当需要部署时，从 constant phase 执行 annealing（衰减到 η_min），然后可以从 pre-annealed constant-phase checkpoint 恢复继续训练。

术语是什么？
CosineInf 包含四个阶段：
$$ \eta(t) = \begin{cases} \text{Linear warmup} & 0 \le t < N_w \\ \text{Cosine cooldown: } \eta_{const} + (\eta_{max} - \eta_{const}) \cdot \frac{1 + \cos(\pi(t-N_w)/(N_c-N_w))}{2} & N_w \le t < N_c \\ \eta_{const} & N_c \le t < N_d \\ \text{Exponential decay to } \eta_{min} & t \ge N_d \end{cases} $$

本文的参数：
- Pre-training: total_iters=192720, η_max=3e-4, η_min=3e-5, η_const=1.65e-4, warmup=1%, cooldown=70%, const=10%
- CPT: total_iters=95370, η_max=3e-4, η_min=3e-5, η_const=1.65e-4, warmup=1%, const=80%, cooldown=0%

与标准 Cosine Decay 的关键差异：
- Cosine Decay（full re-training）：total_iters=288090，warmup 1% → decay to η_min，需预先知道总步数
- CosineInf（CPT）：不随时间衰减到 η_min，保持 constant phase 允许无限继续

从算法pipeline角度拆解术语：
```python
def cosine_inf_schedule(step, total_iters, eta_max, eta_min, eta_const,
                        warmup_pct, cooldown_pct, const_pct):
    N_w = int(total_iters * warmup_pct)
    N_c = int(total_iters * cooldown_pct)
    N_d = int(total_iters * (warmup_pct + cooldown_pct + const_pct))

    if step < N_w:                                    # Phase 1: Warmup
        return eta_max * step / N_w
    elif step < N_c:                                  # Phase 2: Cooldown
        progress = (step - N_w) / (N_c - N_w)
        return eta_const + (eta_max - eta_const) * (1 + cos(pi * progress)) / 2
    elif step < N_d:                                  # Phase 3: Constant
        return eta_const
    else:                                             # Phase 4: Annealing (optional)
        return eta_min + (eta_const - eta_min) * exp_decay(step - N_d)
```

术语一般如何实现？如何使用？
- **CPT 从 constant phase 恢复**：在 constant phase 保存 checkpoint（η=η_const），CPT 时直接从此 checkpoint 以 η_const 继续训练，无需 re-warming → 避免 re-warming 引起的遗忘
- **优于 Cosine Decay**：从衰减 checkpoint 开始的 Cosine Decay CPT 需要在 η_min 上 re-warm 到 η_max → 学习率大幅波动 → 遗忘更严重（本文 Figure 2 验证）
- **Smooth transition**：CosineInf 在 pre-training 和 CPT 之间的 LR 过渡平滑（η_const → η_const），无需大幅调整学习率
- **与 Cosine Decay CPT 的对比**：本文 ablation (Sec 5.1) 显示 CosineInf 从 non-decayed checkpoint 的 CPT 在 FineWeb 遗忘上显著优于 Cosine Decay 从 decayed checkpoint 的 CPT

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Replay in Continual Pre-training (CPT中的数据回放)

术语解释
Replay 是持续学习（Continual Learning）中缓解灾难性遗忘的经典技术：在训练新任务/分布时，将一定比例的旧数据混合到训练 batch 中。在 LLM CPT 语境下，Replay 由 Ibrahim et al. (2024) 验证为 CPT 中最重要的防遗忘技术之一。本文验证了 Replay 对 MoE CPT 同样有效。

术语是什么？
Replay 的具体实现：每个 training batch 中 X% 的 samples 从旧分布（pre-training data）采样，(100-X)% 从新分布（CPT data）采样。例如 "40% Replay" = 每 batch 中 410 samples 来自 FineWeb，614 samples 来自 German CC（batch_size=1024）。

**Compute-equivalent Replay**：为保证不同 replay 比例下的计算量可比，增加 replay 时不增加总 token 预算，而是减少新数据量。例如：200B German CPT at 0% replay = 200B German tokens；at 40% replay = 120B German + 80B FineWeb tokens（总计仍 200B）。

从算法pipeline角度拆解术语：
```python
# CPT with Replay (40% for German)
replay_pct = 0.4
for step in range(95370):
    # Replay portion
    n_replay = int(batch_size * replay_pct)      # 410
    batch_replay = sample(fineweb_loader, n_replay)

    # New data portion
    n_new = batch_size - n_replay                 # 614
    batch_new = sample(german_loader, n_new)

    # Mixed batch
    batch = concat([batch_replay, batch_new])
    # Shuffle batch if needed
    loss = model(batch)
    optimizer.step()
```

术语一般如何实现？如何使用？
- **Replay 比例调优**：本文测试 0%/10%/40%（German）和 30%（Stack）。更高的 replay → 更好的防遗忘（FineWeb val loss 更低），但牺牲部分 adaptation（German/Stack val loss 略高）
- **MoE 与 Dense 的 replay 效果相同**：MoE 的 replay 行为与 FLOP-matched dense 模型一致（本文 Figure 6）
- **Replay 对 MRI 的影响**：Replay 对 SBTk 的 MRI 几乎无影响（SBTk 本身已经固有鲁棒）；对 PBTk 仅轻微减小分布偏移时的 MRI spike（因为 PBTk 恢复很快，replay 的边际收益不大）
- **数据比例**：DeepSeek-CoderV2 使用较高的 replay 比例（30-40%），本文遵循此设定

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Router Saturation (路由饱和率)

术语解释
Router Saturation (RS) 是 Muennighoff et al. (2024) 在 OLMoE 中提出的 MoE 路由稳定性指标：衡量两个训练 checkpoint 之间路由决策的一致性。RS 越高 = 两个 checkpoint 的路由决策越相似 = 路由变化越小。本文将其扩展到持续预训练场景，称为 Continual Router Saturation (CRS)。

术语是什么？
对于两个 checkpoint（分别对应 task h 和 task j 的最终状态），在 N 个 tokens 上的 Router Saturation 定义为：
$$\text{CRS}(h, j) = \frac{1}{N} \sum_{i=1}^{N} \frac{|E_i^{(h)} \cap E_i^{(j)}|}{k}$$

其中 E_i^{(h)} 是 token i 在 checkpoint h 中被选中的 k 个 experts 的集合。RS ∈ [0, 1]，值越低 = 路由决策变化越大。

从算法pipeline角度拆解术语：
```python
def continual_router_saturation(checkpoint_h, checkpoint_j, test_tokens):
    """Compute RS between two checkpoints on test tokens"""
    matches = 0
    for token in test_tokens:
        experts_h = set(route(token, checkpoint_h))  # top-k experts @ ckpt h
        experts_j = set(route(token, checkpoint_j))  # top-k experts @ ckpt j
        matches += len(experts_h & experts_j)
    return matches / (len(test_tokens) * k)
```

术语一般如何实现？如何使用？
- **CPT 分析用途**：RS 帮助识别哪些 MoE layers 的路由在 CPT 中变化最大。本文发现 layers 0-2 和 layers 13-23 的 RS 最低（路由变化最大），且 0% replay 的 checkpoint 在所有层都低于 40% replay
- **与遗忘的关系**：仅有 0% replay checkpoint 出现严重 FineWeb 遗忘，其 RS 在所有层都显著低于 40% replay counterpart，特别是早期层 - 说明早期层的剧烈路由变化与遗忘相关
- **Layer-wise 趋势**：RS 在 layers 2-13 最高（路由最稳定），然后在 layer 13+ 逐渐下降

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Vocabulary Specialization in MoE (MoE词汇专精化)

术语解释
Vocabulary Specialization (VS) 是 Muennighoff et al. (2024) 提出的指标：衡量词汇表中每个 token 被路由到特定 expert 的频率集中度。VS 越高 = 词汇 token 的路由越集中 = expert 的语言学专业化程度越高。本文将其扩展到 CPT 场景，称为 Continual Vocabulary Specialization (CVS)。

术语是什么：
对于 MoE layer j，expert E_i，token x：
$$\text{VS}(j, E_i, x) = \frac{N_{j,x,E_i}^{(k)}}{N_{j,x}}$$

其中 N_{j,x,E_i}^{(k)} 是 token x 在 checkpoint j 下被路由到 expert E_i 的次数，N_{j,x} 是 token x 的总出现次数。

为比较跨 checkpoint 的 specialization 变化：固定 pre-training checkpoint 的 expert-token 映射（将每个 token 分配给最常处理它的 expert），然后在新 checkpoint 上用同一映射计算 VS。如果新 checkpoint 的 VS 显著低于 pre-training checkpoint，说明路由模式发生了变化。

从算法pipeline角度拆解术语：
```python
def continual_vocab_specialization(layer, checkpoint_h, checkpoint_j, test_tokens):
    """CVS: fix mapping from ckpt_h, compute VS using ckpt_j"""
    # Step 1: Create one-to-many mapping from checkpoint_h
    token_to_expert_map = {}
    for token_id in range(vocab_size):
        expert_counts = count_expert_assignments(token_id, checkpoint_h)
        token_to_expert_map[token_id] = argmax(expert_counts)

    # Step 2: Compute CVS using checkpoint_j's routing
    total_vs = 0
    for token in test_tokens:
        assigned_expert = token_to_expert_map[token.id]
        vs = prob_routed_to(token, assigned_expert, checkpoint_j)
        total_vs += vs
    return total_vs / len(test_tokens)
```

术语一般如何实现？如何使用？
- **CPT 分析**：CVS 在 early layers (0-4) 显著降低（路由模式变化大），layers 5-23 几乎不变。0% replay 的 early layers VS 最低 → 与 FineWeb 遗忘相关
- **分布依赖**：同一 token 的 VS 在不同分布上不同（例如 "for" 在 English 和 Code 中的上下文表示不同，路由也不同）
- **架构差异**：Granular MoE 和 Switch MoE 的 VS 模式相似，表明 VS 主要受 CPT 策略（replay）而非架构影响

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

---

## Switch MoE (Switch Transformer 架构)

术语解释
Switch MoE 是由 Fedus et al. (2022) 提出的简化 MoE 架构：每个 MoE layer 仅激活 1 个 expert (K=1)，相比 K≥2 的方案大幅简化路由和负载均衡。Switch Transformer 证明了 K=1 仍能保证有效的路由梯度（因为 batch 内不同 tokens 选择不同 experts），并成功扩展到 1T+ 参数规模。

术语是什么？
Switch MoE 的关键特征：
- K=1：每 token 仅路由到 1 个 expert
- Full-size experts：每个 expert 的 FFN intermediate size 与 dense model 的 FFN 相同（不使用 fine-grained 拆分）
- 无 shared expert：所有 expert 均为 routed experts
- Token dropping：原始 Switch Transformer 使用 capacity factor 控制 expert 最大容量，超限 token 被 drop（本文不使用 token dropping）

本文的 Switch MoE 配置（570M active / 2B total）：E=8 routed experts, K=1, FFN intermediate=2816 (GEGLU)，无 shared expert。

从算法pipeline角度拆解术语：
```python
# Switch MoE layer forward (K=1)
def switch_moe_forward(x, W_router, experts):
    logits = x @ W_router              # [B, S, 8]
    probs = softmax(logits, dim=-1)    # [B, S, 8]
    top1_idx = argmax(probs, dim=-1)   # [B, S], pick SINGLE expert
    top1_prob = probs[top1_idx]        # scalar per token

    output = zeros_like(x)
    for e in range(8):
        mask = (top1_idx == e)
        if mask.any():
            output[mask] = top1_prob[mask] * experts[e](x[mask])
    return output
```

术语一般如何实现？如何使用？
- **CPT 中的表现**：Switch MoE 的 validation loss 和 benchmark 均弱于 Granular MoE（因为 expert 数量少且无 shared expert）
- **MRI 特征**：Switch MoE 的 early layers（0-6）MRI 显著高于 Granular MoE，且与训练/测试分布无关 → 早期层的路由不稳定可能是 Switch MoE 性能较差的根本原因
- **Code 任务**：Switch MoE 在 HumanEval 上意外优于 Granular MoE（可能因 K=1 的路由更简单，code 数据下 specialization 更清晰）
- **架构比较**：本文的 Granular MoE (E=31, K=3) 在所有主要指标上优于 Switch MoE (E=8, K=1)

涉及论文标题：
- Continual Pre-training of MoEs How robust is your router

## Co-Upcycling

术语解释
Co-Upcycling 是 CuMo 论文提出的稀疏 MoE 初始化策略：在多模态 LLM 的视觉编码器（CLIP ViT）和 MLP 连接器中，将每个 dense MLP 块的权重同时（co-）复制为对应 MoE 专家的初始权重，而非随机初始化专家。这是对 Sparse Upcycling 的扩展——不仅 upcycle 单个模块，而是跨多个模块（视觉编码器 + MLP 连接器）协同 upcycle。

术语是什么？
CuMo 在视觉指令微调阶段引入 Co-Upcycling：(1) 预训练 MLP 连接器和预微调全模型（无 MoE 块）；(2) 将每个 dense MLP 块替换为 Top-2-in-4 稀疏 MoE 块；(3) 将预训练/预微调阶段的同位置 MLP 权重复制到对应 MoE 块中的每个 expert；(4) Router 网络（Top-K gating）随机初始化，从头训练。

```
# Co-Upcycling: 同时 upcycle 多个模块的 MLP → MoE
def co_upcycle(pretrained_model):
    # Step 1: Upcycle MLP connector (两层 MLP)
    for mlp_block in pretrained_model.mlp_connector:
        moe_block = TopK_MoE(num_experts=4, top_k=2)
        for expert in moe_block.experts:
            expert.weight = mlp_block.weight.clone()  # 从预训练 MLP 复制
        moe_block.router = Router(num_experts=4)       # 随机初始化
        replace(mlp_block, moe_block)

    # Step 2: Co-Upcycle CLIP vision encoder 的每个 transformer 层
    for layer in pretrained_model.clip_vit.layers:
        moe_block = TopK_MoE(num_experts=4, top_k=2)
        for expert in moe_block.experts:
            expert.weight = layer.mlp.weight.clone()   # 从同层 MLP 复制
        moe_block.router = Router(num_experts=4)
        layer.mlp = moe_block

    return pretrained_model  # 继续 visual instruction tuning
```

从算法pipeline角度拆解术语：
Co-Upcycling 位于三阶段训练的第三阶段（视觉指令微调）开始时。前两阶段（MLP connector 预训练 + 全参数预微调）产生 warm-up 后的模型参数。第三阶段开始时，dense MLP → MoE 替换发生，expert 初始化来自 warm-up MLP 权重。这避免了从头训练 MoE 的不稳定性——论文报告若从随机初始化训练 MoE blocks，模型无法收敛，即使降低学习率也无法达到 baseline 性能。

关键对比：
- Sparse Upcycling：仅 upcycle LLM 的 MLP → MoE（如 Mistral-7B → Mistral-7B-MoE）
- Co-Upcycling：同时 upcycle CLIP ViT + MLP connector 的 MLP → MoE，而 LLM 使用 pre-trained Mixtral-8×7B（因为论文实验表明 upcycled LLM-MoE 效果不如 pre-trained LLM-MoE）

术语一般如何实现？如何使用？
- 前提：已有完成 MLP connector 预训练和全参数预微调的 dense checkpoint
- CuMo 实验表明 Co-Upcycling 比随机初始化 MoE 专家显著更好（Table 3-4），甚至随机初始化导致模型不收敛
- 学习率：Co-Upcycling 后的视觉指令微调使用 2e-6 ~ 4e-6，比常规微调更低
- 配合辅助损失 bzloss（L_balance α=0.1 + L_z α=0.01）维持 expert 负载均衡

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

---

## CLIP-MoE (Vision Encoder MoE)

术语解释
CLIP-MoE 是 CuMo 论文提出的将 Top-K 稀疏门控 MoE 块集成到 CLIP ViT（视觉编码器）每个 transformer encoder 层的设计。具体是将 CLIP ViT-L 中每层的 dense MLP（两个线性层 + GELU）替换为 Top-2-in-4 稀疏 MoE 块，保持 skip connection 不变。

术语是什么？
CLIP ViT-L 的标准结构为交替的 Multi-Head Self-Attention + dense MLP blocks。CLIP-MoE 仅替换 MLP blocks：每个 MLP 变为 1 个 Router（线性层 → Softmax → Top-K）+ 4 个 expert MLP（每个与原始 MLP 结构相同）。Router 对每个 visual token 选择 Top-2 experts，输出为 2 个选中 expert 输出的加权和。

```
# CLIP-MoE 单层 forward（替换原来 ViT 的 MLP block）
def clip_moe_layer(x):  # x: [N, d] visual tokens
    # 1. Multi-Head Self-Attention（保持不变）
    x = x + MHSA(LayerNorm(x))

    # 2. MoE block 替换 dense MLP（核心改动）
    residual = x
    x = LayerNorm(x)
    W = Softmax(Linear_router(x))           # [N, 4]
    W_K_values, W_K_indices = TopK(W, K=2) # 选 top-2 experts
    W_K = Softmax(W_K_values)              # [N, 2]

    out = zeros_like(x)
    for i in range(2):
        expert_idx = W_K_indices[:, i]
        expert_weight = W_K[:, i:i+1]
        out += expert_weight * ExpertMLP_i(x)  # 仅通过选中的 2/4 experts

    x = residual + out
    return x
```

从算法pipeline角度拆解术语：
CLIP-MoE 位于多模态 LLM pipeline 的**视觉编码阶段**。输入图像 → CLIP-MoE（每层 top-2 routing 处理 visual tokens）→ 输出 visual tokens → MLP-MoE 连接器 → LLM。CLIP-MoE 激活 0.50B / 总 0.91B 参数（仅激活 2/4 experts）。论文发现增加 experts 数量到 8（Top-2-in-8）反而略有性能下降（Table 4），推测是有限的视觉指令微调数据不足以训练 8 个鲁棒且均衡的 experts。

术语一般如何实现？如何使用？
- CLIP-MoE experts 使用 Co-Upcycling 初始化（从预微调后的 dense CLIP MLP 权重复制）
- 随机初始化 CLIP-MoE 专家 → 模型不收敛
- 训练时 unfreeze CLIP（预训练阶段 freeze）+ 降低学习率至 2e-5
- 配合 bzloss 维持 expert 负载均衡
- 推理时 expert distribution 在各层间均匀分布（Figure 5 验证）

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

---

## MLP-MoE (Vision-Language Connector MoE)

术语解释
MLP-MoE 是 CuMo 论文提出的将 Top-K 稀疏门控 MoE 块集成到多模态 LLM 中 vision-language MLP 连接器的设计。MLP 连接器通常为两层线性 MLP，将 visual tokens 投影到 word embedding 空间。MLP-MoE 将此 MLP 替换为 Top-2-in-4 稀疏 MoE 块。

术语是什么？
标准 MLP connector：`visual_tokens → Linear1 → GELU → Linear2 → word_embedding_tokens`。MLP-MoE 将其替换为：Router（线性层 → Softmax → Top-2）选择 2/4 experts（每个 expert 同为两层 MLP），加权求和输出。

```
# MLP-MoE connector forward
def mlp_moe_connector(visual_tokens):  # [N, d_v]
    W = Softmax(Linear_router(visual_tokens))  # [N, 4]
    W_K_values, W_K_indices = TopK(W, K=2)
    W_K = Softmax(W_K_values)                  # [N, 2]

    word_embeddings = zeros(N, d_llm)
    for i in range(2):
        expert_idx = W_K_indices[:, i]
        expert_out = ExpertMLP_i(visual_tokens[expert_idx])
        word_embeddings[expert_idx] += W_K[expert_idx, i] * expert_out

    return word_embeddings  # 输入给 LLM
```

从算法pipeline角度拆解术语：
MLP-MoE 位于视觉编码器（CLIP-MoE）和 LLM 之间，负责维度转换。每个 expert 仅包含两个线性层（参数量小），因此 MLP-MoE 的总参数仅 0.10B（激活 0.05B）。CuMo 的消融实验（Table 3）表明：(1) 随机初始化 MLP-MoE → 性能明显下降；(2) Upcycling 初始化 → 边际提升；(3) 加入 bzloss → MMVet 明显提升；(4) Top-2-in-8 → 性能略降（数据不足以训练 8 个均衡专家）。

术语一般如何实现？如何使用？
- 微小的额外参数成本：总参数从 7.25B → 7.65B（+0.40B），激活参数从 7.25B → 7.60B（+0.35B）
- Co-Upcycling 从预训练 MLP connector 权重初始化
- 配合 bzloss（与 CLIP-MoE 的 bzloss 独立应用）
- 推理时 Router 仅增加极小的计算开销（一个 Linear 层 + Softmax + Top-K）

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

---

## Visual Instruction Tuning

术语解释
Visual Instruction Tuning 是将预训练 LLM 转换为多模态 LLM 的核心训练范式，由 LLaVA (Liu et al., 2023) 提出。核心思想是将图像-文本对数据转换为 instruction-following 格式（类似 NLP 的 instruction tuning），让 LLM 学会遵循包含图像的指令并生成文本回复。

术语是什么？
流程：(1) 图像通过冻结的视觉编码器提取 visual tokens；(2) 可训练的 MLP 连接器将 visual tokens 投影到 word embedding 空间；(3) visual tokens + text tokens 拼接后输入 LLM；(4) LLM 仅对 text tokens 计算自回归交叉熵损失（visual tokens 不计算损失）。

```
# Visual Instruction Tuning 的训练样本格式
# 单轮对话:
# <image> USER: <question> ASSISTANT: <answer>
# 多轮对话:
# <image> USER: <q1> ASSISTANT: <a1> USER: <q2> ASSISTANT: <a2>

def visual_instruction_tuning_step(image, conversation):
    visual_tokens = vision_encoder(image)          # CLIP ViT
    word_embed_tokens = mlp_connector(visual_tokens)  # 投影到 LLM 空间
    
    # 拼接 visual tokens + text tokens
    text_tokens = tokenize(conversation)             # "USER: ... ASSISTANT: ..."
    input_ids = concat([visual_tokens, text_tokens])
    
    # 仅对 ASSISTANT 回复部分计算 loss
    logits = LLM(input_ids)
    loss = CrossEntropy(logits[assistant_mask], labels[assistant_mask])
    loss.backward()
```

从算法pipeline角度拆解术语：
Visual instruction tuning 位于多模态 LLM 训练 pipeline 的核心阶段。数据来源于各种 VQA 和看图理解数据集，统一转换为 "USER: <question about image> ASSISTANT: <answer>" 格式。CuMo 在此阶段加入 Co-Upcycled MoE blocks，训练数据混合包括 LLaVA-665K、ShareGPT4V、DocVQA、ChartQA 等约 1.65M 样本。

术语一般如何实现？如何使用？
- 基础：LLaVA 系列（v1, v1.5, NeXT）
- 数据来源：开源 VQA 数据集（VQAv2, GQA, TextVQA 等）+ GPT-4V 生成的高质量指令数据（ShareGPT4V, ALLaVA）
- 典型超参数：学习率 2e-5 ~ 4e-6，batch size 128-256，使用 DeepSpeed ZeRO-3
- 评估：贪心解码，multiple choice / GPT-API 评分（LLaVA-Wild 用 gpt-4-0613, MathVista 用 gpt-3.5-turbo）
- CuMo 的扩展：在 visual instruction tuning 阶段引入 Co-Upcycled MoE blocks + bzloss

涉及论文标题：
- CuMo: Scaling Multimodal LLM with Co-Upcycled Mixture-of-Experts

## Matryoshka Weight Quantization (MWQ / 嵌套权重量化)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Matryoshka Weight Quantization (MWQ) 是 D2MoE 提出的多步量化技术，通过渐进式嵌套压缩，使不同 bit-width 的量化权重可以共享存储。其核心思想来源于套娃的嵌套结构：高 bit-width 权重在存储上天然包含低 bit-width 权重，无需为每个 bit-width 独立存储一份完整权重。

MWQ 分两步执行：
1. **Asymmetric Quantization 到最低 bit-width b₁**（如 INT2）：以 group-wise（group_size=128）的方式，最小化量化后输出误差 ∥WX - Ŵ_{b₁}X∥₂²，得到量化权重 Q_W_{b₁}、scale factor s_{b₁}、zero-point z_{b₁}
2. **Binary Residual Quantization 渐进增加 bit-width**：对残差 R_{b₁} = W - Ŵ_{b₁}，逐步将其量化为 +1/-1 的 binary 增量权重 Q_W_{b_k}，每步增加 1 bit，最终 b_K = b₁ + (K-1)。反量化时 Ŵ_{b_k} = Ŵ_{b₁} + Σ_{i=2}^{b_k} s_{b_i} · Q_W_{b_i}

这意味着存储 INT2/3/4 时仅需：一份 INT2 base + 两个 1-bit residual + 对应 scale factors，存储量接近 INT4 而非 INT2+INT3+INT4 之和。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 D2MoE-V1 (b₁=2, b_K=4, group_size=128) 为例：

```
=== MWQ 量化 Pipeline ===
输入: FP16 expert weight W ∈ R^{s×h}, calibration data X ∈ R^{h×r},
      Hessian regularizer λ, block size γ (e.g., 128)

Step 1: Cholesky decomposition for error compensation
    H^c = Cholesky((2XX^T + λI)^{-1})  # GPTQ-style correction matrix

Step 2: Asymmetric quantization to b₁=2
    对每组尺寸为 128 的元素:
        z_b1, s_b1 = argmin ||W_group · X_group - Ŵ_b1 · X_group||₂²
        Q_W_{b1} = round(W / s_b1 + z_b1)  # 量化到 INT2
        Ŵ_{b1} = (Q_W_{b1} - z_b1) · s_b1  # 反量化
        R_b1 = W - Ŵ_{b1}                  # 残差
    对后续 block 用 H^c 进行 block-wise error compensation (类似 GPTQ)

Step 3: Binary residual quantization: b₁=2 → b₂=3
    对每组:
        s_b2 = argmin ||R_b1 · X_group - s_b2 · Q_W_{b2} · X_group||₂²
        Q_W_{b2} = round(R_b1 / s_b2)  # 得到 +1/-1 的 binary 权重
        Ŵ_{b2} = (Q_W_{b1} - z_b1) · s_b1 + s_b2 · Q_W_{b2}  # INT3 重构
    
Step 4: Binary residual quantization: b₂=3 → b₃=4
    同上，对 R_b2 = W - Ŵ_{b2} 执行:
        s_b3 = argmin, Q_W_{b3} = round(R_b2 / s_b3)
        Ŵ_{b3} = Ŵ_{b1} + s_b2·Q_W_{b2} + s_b3·Q_W_{b3}  # INT4 重构

输出: {Q_W_{b1}, z_b1, s_b1} ∪ {(Q_W_{b_i}, s_b_i)}_{i=2}^{K}
```

**存储对比**：
- 传统方法（INT2+3+4 独立）：存储 INT2 完整权重 + INT3 完整权重 + INT4 完整权重
- MWQ：存储 1 份 INT2 base + (K-1) 组 1-bit residual + scale factors
- 例如 LLaMA-MoE-3.5B：传统方法 ~9.62GB（INT2/3/4），MWQ ~4.48GB（接近 INT4）

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 使用 PyTorch + CUDA 实现。MWQ 属于离线预处理阶段（deployment 前执行一次），使用 C4 calibration dataset（128 random 2048-token segments）。MWQ 借鉴 GPTQ 的 block-wise error compensation，但去掉了 column-level error correction 以降低计算开销。离线阶段在 GPU server (2×A6000) 上执行：LLaMA-MoE-3.5B MWQ 耗时 ~10 min (batch_size=16)，Mixtral 8×7B 耗时 ~20 min (batch_size=4)。

MWQ 的核心使用场景：需要在端侧设备上同时支持多种 bit-width 的 MoE 推理，且内存极度受限（6GB-64GB）时，避免多版本权重存储爆炸。MWQ 的嵌套结构还天然支持低 bit-width 权重的高频复用——多个需要不同 bit-width 的请求可以共享 base 权重，仅额外加载各自需要的 residual bits。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

## Token-Adaptive Bit-Width Selection (基于 Token 自适应的位宽选择)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Token-Adaptive Bit-Width Selection 是 D2MoE 提出的动态量化 router 训练方法，使每个 MoE expert 可以根据当前输入 token 的表示动态选择最合适的量化 bit-width。它基于 Observation：不同 token 对同一 expert 的量化敏感度不同（例如 expert 4 layer 1 量化到 INT1 在样本 1 上损失 0.5% 精度，在样本 10 上损失 0.2%）。

核心设计包括两个机制：
1. **Quantized Expert Capacity**：为每个 bit-width expert 设定 token 容量上限 c_k·T（如 D2MoE-V1 中 {0.3, 0.4, 0.3} 对应 INT2/3/4），超限 token 随机丢弃，防止训练时 bit-width router 坍塌到某一固定 bit-width
2. **Dynamic Bit-Width Selection Loss**：Loss = (1/T) Σ [CE(p(x), q(x)) + (α/L) Σ p_k^l(x) · b_k]，其中 CE 项保持精度（倾向高 bit-width），正则项 p_k^l(x)·b_k 促选低 bit-width（b_k 越小越好），α 平衡精度与效率

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
以 LLaMA-MoE-3.5B 推理时的 bit-width 选择流程为例：

```
=== Token-Adaptive Bit-Width Selection (Inference) ===
输入: 一个 transformer block 内 T 个 token 的 hidden states {h_t}，
      每 expert j 的 bit-width router R_j (已微调)，
      候选 bit-width {b_1=2, b_2=3, b_3=4}，容量 {c_1=0.3, c_2=0.4, c_3=0.3}
      Top-2 expert gating 结果: 每个 token t 选择 2 个 experts

for each expert j that has tokens routed to it:
    tokens_to_this_expert = {t | expert_j selected for token t}
    for each token t in tokens_to_this_expert:
        # Router 输出 K 个 bit-width 的 logits
        logits = R_j(h_t)  # 轻量化 MLP, 输入 hidden_dim, 输出 K
        probs = softmax(logits)  # p_k^l(x): 第 k 个 bit-width 的概率
        selected_bitwidth = top1(probs)
    
    # 容量约束：如果某 bit-width 超出 c_k·T，超限 token 跳过该 expert
    for each bitwidth k:
        if count(selected_bitwidth == k) > c_k * T:
            randomly drop excess tokens (skip expert computation)

输出: 每 token 每 expert 的 selected_bitwidth
      → 用于后续 MWQ 反量化 + GEMM
```

**训练时的 Dynamic Bit-Width Selection Loss**：
```
Loss = (1/T) Σ_t [CE(p_t, q_t) + (α/L) Σ_l Σ_k p_k^l(x_t) · b_k]

其中:
  p_t, q_t: D2MoE 模型和 FP16 基准模型的 logits (after LM head)
  p_k^l(x_t): token t 在 layer l 选中 bit-width k 的概率
  b_k: 第 k 个 bit-width 的数值（如 2, 3, 4）
  CE 项→ 保证精度（本质促选高 bit-width）
  正则项→ 促选低 bit-width（概率分配越小越好）
  α 控制精度-效率权衡
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
D2MoE 中，Bit-Width Router 是一个轻量化 MLP（参数占比 <0.5%），放置在每个 expert 之前。使用 C4 通用数据集（2048 random 2048-token segments）微调，batch_size=64。LLaMA-MoE-3.5B 微调耗时 ~2 小时（2×A6000），Mixtral 8×7B 耗时 ~4 小时。运行时额外开销：计算 <0.28%，内存 <0.53%，延迟 <1.67%（主要是 router 中 softmax 操作）。

对比 EdgeMoE（离线 calibration 固定 bit-width）和 MC-MoE（固定 activation frequency 分配），Token-Adaptive Bit-Width Selection 可以随 token 动态调整，在相同精度下节省 33%-53% 峰值内存。

涉及论文标题：
- D2MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving

## Auxiliary Loss Trio for MoE Load Balance (MoE三级辅助负载均衡损失)

术语解释
DeepSeek-V2 提出的一套三层辅助损失函数体系，分别从 Expert 级、Device 级和 Communication 级三个粒度控制 MoE 训练中的负载均衡，配合 Token-Dropping Strategy 实现软硬结合的负载管理。

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
DeepSeek-V2 训练时采用 8-way Expert Parallelism (D=8)，160 个 routed experts 均匀分配到 8 个设备上，每个 token 激活 K_r=6 个 experts。三层辅助损失分别为：

**(1) Expert-Level Balance Loss (L_ExpBal)**：经典 MoE balance loss（Fedus et al. 2021, Lepikhin et al. 2021），用于防止 routing collapse。L_ExpBal = α1 × Σ(f_i × P_i)，其中 f_i 为 expert i 的实际负载占比，P_i 为 expert i 的平均路由概率。DeepSeek-V2 中 α1=0.003。

**(2) Device-Level Balance Loss (L_DevBal)**：DeepSeek-V2 新增设计，确保各 device 计算量均衡。将 routed experts 分区 D 组 {E_1,...,E_D}，每组部署在一个 device。L_DevBal = α2 × Σ(f'_i × P'_i)，其中 f'_i 为 device i 上所有 experts 的平均负载，P'_i 为 device i 上所有 experts 的总路由概率。DeepSeek-V2 中 α2=0.05（权重最高，因 device 级均衡对计算效率最关键）。

**(3) Communication Balance Loss (L_CommBal)**：DeepSeek-V2 新增设计，确保各 device 收发 token 量均衡。虽然 Device-Limited Routing (M=3) 限制了发送量，但若某 device 收到远超平均的 token，all-to-all 通信效率仍受影响。L_CommBal = α3 × Σ(f''_i × P''_i)，其中 f''_i 为归一化的 device i 接收 token 占比。DeepSeek-V2 中 α3=0.02。

为什么需要三层？单层 expert-level loss 不感知分布式拓扑——expert 级均衡不等于 device 级均衡（一个 device 上多个 expert 可能整体偏载）。device 级均衡不保证通信均衡（发送 bounded ≠ 接收均衡）。三层各司其职。

从算法pipeline角度拆解术语：
```
=== Auxiliary Loss Computation (per training step) ===

Input: batch of T tokens, N_r=160 experts, D=8 devices, K_r=6

// Expert-Level Balance Loss
for expert i in 1..160:
    f_i = (160 / (6*T)) * count(token selects expert i)  // actual load ratio
    P_i = (1/T) * sum_t s_{i,t}                           // mean routing prob
L_ExpBal = 0.003 * sum_i f_i * P_i

// Device-Level Balance Loss  
for device d in 1..8:
    f'_d = (1/20) * sum_{i in E_d} f_i                    // avg on-device expert load
    P'_d = sum_{i in E_d} P_i                              // total routing prob
L_DevBal = 0.05 * sum_d f'_d * P'_d

// Communication Balance Loss
for device d in 1..8:
    f''_d = (8 / (3*T)) * count(token received by device d)
    P''_d = sum_{i in E_d} P_i
L_CommBal = 0.02 * sum_d f''_d * P''_d

L_total = L_main + L_ExpBal + L_DevBal + L_CommBal
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
所有三轮损失在 HAI-LLM 训练框架中实现，与 expert parallel all-to-all 通信层、device-limited routing 和 token dropping 配合构成完整的负载管理方案。α2=0.05 高出一个数量级以上（device 均衡是分布式训练的关键性能瓶颈）。DeepSeek-V3 后续改为 Auxiliary-Loss-Free Load Balancing（bias-based），取消了此三层损失机制。DeepSeek-V2-Lite 仅用简化的 expert-level loss（α1=0.001，无 device/comm loss），因其所有 experts 部署在同一 device。

涉及论文标题：
- DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model
