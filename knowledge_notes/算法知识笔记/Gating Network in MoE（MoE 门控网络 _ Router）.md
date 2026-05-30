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
