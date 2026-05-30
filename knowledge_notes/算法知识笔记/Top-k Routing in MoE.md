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
