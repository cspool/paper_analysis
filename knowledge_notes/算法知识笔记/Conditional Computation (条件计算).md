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
