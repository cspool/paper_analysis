## Upcycling Large Language Models into Mixture of Experts

- 属于算法pipeline的实现是什么？实验比较什么？
  - 实现：将预训练稠密 LLM 转换为稀疏 MoE 模型的 **upcycling** 训练算法，包括：(1) **Virtual Group Initialization（虚拟组初始化）**——针对 fine-grained MoE（granular upcycling），将稠密 MLP 的 FFN 权重按 hidden dimension 切分为 G 个 shard，复制 E 次形成 N=E×G 个 expert，路由器权重也分组复制，保证初始状态下 Router TopK 恰好能选到每个 shard 的一份拷贝，使 MoE 输出与稠密模型功能等价；(2) **Weight Scaling（权重缩放）**——对 expert MLP 的 W1 和 W2 权重同时缩放，缩放因子 = ³√(E×G²/T)，对 squared-relu 推导，对 SwiGLU 也有效，同时适用于 coarse-grained 和 fine-grained MoE；(3) **Softmax-then-TopK Routing**——采用先 softmax 后 topK 的 Router 顺序（而非 Mixtral 的 topK-then-softmax），保留 Router 输出的绝对值信息；(4) **学习率重置策略**——upcycling 时从预训练最低学习率重新提高到峰值学习率（如 2e-4），配合 cosine decay，帮助模型逃离稠密模型的局部最小值、促进 expert 分化；(5) **大批量训练**——batch size 增大至 4M tokens 以降低 expert 梯度噪声和负载均衡损失噪声。
  - 实验比较：(a) Upcycling vs 续训稠密模型——Nemotron 2B 在 0.1T tokens 下，upcycling loss 比续训低 1.1%，且续训迅速 plateau（Figure 4a）；(b) Upcycling vs 从头训练——在固定 compute budget 下 upcycling 显著优于 from scratch（Figure 4b）；(c) 学习率调度消融——constant LR (2e-5) vs 重置 LR (warmup to 2e-4) vs 重置 LR (warmup to 1e-4)，验证重置 LR 优于 constant LR（Figure 5），且权重 cosine similarity 从 ≈1 降至 0.6-0.7（Figure 6）；(d) Batch size 消融——512 (2M tokens) vs 1024 (4M tokens) vs 8192 (32M tokens)，4M tokens batch 最优（Figure 7）；(e) Softmax-TopK 顺序——softmax-then-topK 优于 topK-then-softmax（Section 3.4）；(f) Weight Scaling 消融——w/ vs w/o weight scaling，weight scaling 带来 1.5% loss 改善（Nemotron-4 15B E8G1T1, Figure 9），多种替代方法（MoE output scaling、post expert layernorm）均不如 weight scaling；(g) Granularity 消融——8/64/128/256 experts iso-FLOP 对比，64 experts 最优，128/256 experts 有 diminishing returns（Figure 10）；(h) TopK 消融——Top-1 vs Top-2 (E8G1)，Top-2 优于 Top-1 但计算量加倍（Figure 11）；(i) Shared experts（Deepseek-MoE 风格）——8 shared + 64 experts top-8 vs iso-FLOP 64 experts top-16，性能持平（Figure 13）；(j) 大规模最终实验——Nemotron-4 15B upcycling on 1T tokens：E8G8T8 (64 experts top-8) val loss 1.320 / MMLU 66.2，E8G1T2 (8 experts top-2) val loss 1.306 / MMLU 67.6，均优于续训稠密模型的 val loss 1.377 / MMLU 65.3（Table 1）。

- 硬件平台是什么，配置是什么。
  - NVIDIA GPU（论文未明确说明具体 GPU 型号和数量），使用 Megatron-LM 进行分布式训练，支持 data parallelism + tensor parallelism + expert parallelism。Codebase：Megatron-LM（https://github.com/NVIDIA/Megatron-LM），训练框架还包括 NeMo。

- 模型是什么。数据集和bench分别是什么。
  - 模型：(1) **Nemotron 2B**——decoder-only Transformer，SwiGLU 激活，RoPE，max seq len 4096，no dropout，no bias，untied embedding，vocab size 256K，预训练 1.1T tokens；(2) **Nemotron-4 15B**——15B 参数多语言 LLM，预训练 8T tokens，vocab size 256K。
  - MoE 变体（消融用）：E8G1T1/T2、E8G8T8/T16、E64G8T8、E128G8T8、E256G8T8 等多种配置（E=experts数、G=granularity、T=topK）。
  - 数据集：(a) Nemotron 2B 消融——使用预训练数据（已见过的数据），110B tokens（约 10% 预训练 token 数）；(b) Nemotron-4 15B 消融——续训数据 blend（与预训练数据分布不同），0.1T tokens；(c) Nemotron-4 15B 大规模实验——续训数据 blend，1T tokens。Validation loss 在 1% held-out 数据上测量。
  - Benchmarks：MMLU (5-shot)。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  - 开源：代码位于 Megatron-LM 仓库的 moe/upcycling 分支：https://github.com/NVIDIA/Megatron-LM/tree/0431153bf1b5c405057b158189c260107d8b7c3a/megatron/core/transformer/moe#upcycling
  - 算法 Pipeline（Upcycling 完整流程）：

```
# ===============================
# Step 1: Coarse-Grained Upcycling (E experts, Top-K routing)
# ===============================
# 稠密 FFN: y = W2(activation(W1(x))) = W2(sigma(W1(x)))

# 复制 MLP 权重 -> 每个 expert 初始化相同
experts = [copy.deepcopy(dense_ffn) for _ in range(E)]

# Router 随机初始化
W_r = random_init(shape=(d_model, E))  # 路由器权重矩阵

# Weight Scaling: 对每个 expert 的 W1 和 W2 缩放
# 缩放因子 (针对 Squared-ReLU 推导, SwiGLU 也适用):
#   scale = (E * G^2 / T)^{1/3}
# 对 E8G1T1: scale = (8 * 1 / 1)^{1/3} = 2.0
# 对 E8G8T8: scale = (8 * 64 / 8)^{1/3} = 4.0
for expert in experts:
    expert.W1 *= scale
    expert.W2 *= scale

# ===============================
# Step 2: Softmax-Then-TopK Routing (Forward Pass)
# ===============================
# 输入: x in R^{S x d_model} (S tokens)
r_logits = x @ W_r                          # (S, E), Router logits
r_probs = softmax(r_logits, dim=-1)         # (S, E), Router probabilities
topk_probs, topk_idx = topk(r_probs, T)     # (S, T), select top-T experts

# Expert 计算
MoE_output = zeros_like(x)
for t in range(T):
    expert_idx = topk_idx[:, t]             # (S,)
    expert_weight = topk_probs[:, t]        # (S,)
    # gather tokens to each expert, compute FFN
    MoE_output += expert_weight * gather_expert_ffn(x, expert_idx)

# ===============================
# Step 3: Fine-Grained Upcycling (Virtual Group Init)
# ===============================
# 例子: E8G8T8 = 64 experts, shard by G=8, route to T=8 experts
# 原始 FFN hidden dim = H -> expert hidden dim = H/G

# Shard dense FFN weights by intermediate dim
# W1 in R^{d_model x H} -> shard into G parts: {W1_0, ..., W1_{G-1}}
# W2 in R^{H x d_model} -> shard into G parts: {W2_0, ..., W2_{G-1}}
shards_W1 = W1.split(G, dim=1)   # shape per part: (d_model, H/G)
shards_W2 = W2.split(G, dim=0)   # shape per part: (H/G, d_model)

# 复制 shards 形成 E copies per shard, total N=64 experts
experts = []
for i in range(G):
    for _ in range(E):
        experts.append((shards_W1[i], shards_W2[i]))
# expert_0..expert_7 都是 shard_0, expert_8..expert_15 都是 shard_1, ...

# Virtual Group Router 初始化
# Router weights W_r in R^{d_model x N}, N=E*G=64
# 将 W_r 分为 G 个 group, 每个 group 内的 E 列相同
W_r = random_init((d_model, E))  # 只生成 E 组权重
W_r_full = zeros(d_model, N)
for g in range(G):
    W_r_full[:, g*E : (g+1)*E] = W_r  # copy same weights
# 初始化后 router 在每组内相同, 保证 top-T 均匀覆盖 G 个 group

# ===============================
# Step 4: Loss Computation & Training
# ===============================
# Load balancing aux loss (ST-MoE / Switch Transformer):
# f_e = 1/T * sum_{token} 1_{token routed to expert e}
# P_e = 1/T * sum_{token} softmax_prob[token, e]
# L_aux = E * sum_e f_e * P_e
# Total loss: L = L_LM + alpha_aux * L_aux  (alpha_aux = 1e-2)

# 优化器: Adam (Megatron-LM default)
# 学习率: warmup -> peak (2e-4 or 1e-4) -> cosine decay -> min (2e-5)
# Batch size: 1024 (4M tokens) for Nemotron 2B
```

  - 关键张量计算流（以 E8G1T2 单个 token 为例）:
    ```
    x in R^{d_model} (e.g., 4096)
    
    # Attention 输出 -> MoE Router
    r = W_r^T @ x          # (8,) Router logits
    s = softmax(r)          # (8,) Router probabilities
    [p1, p2], [e1, e2] = top2(s)  # 选 top-2 experts
    
    # Expert 1 计算
    h1 = W1_{e1} @ x       # (H,) 第一线性投影
    a1 = sigma(h1)         # SwiGLU activation
    o1 = W2_{e1} @ a1      # (d_model,) 第二线性投影
    
    # Expert 2 计算
    h2 = W1_{e2} @ x
    a2 = sigma(h2)
    o2 = W2_{e2} @ a2
    
    # 加权输出
    y = p1 * o1 + p2 * o2   # (d_model,)
    ```

  - 大规模实验最终参数（1T tokens upcycling of Nemotron-4 15B）:
    - E8G8T8: peak LR = 3e-4, cosine decay to 1/100 of pretraining min LR
    - E8G1T2: same LR schedule, different batch size (top-2 per-expert more tokens)
