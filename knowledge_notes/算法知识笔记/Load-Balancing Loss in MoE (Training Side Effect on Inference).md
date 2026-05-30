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
