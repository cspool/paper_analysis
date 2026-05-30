## Token-level Expert Gating / Context Switching

术语解释
Token-level Expert Gating 是 BTS 中 Stitch Layer 门控机制实现的推理行为：每个 token 处理时，softmax/sigmoid gate 根据当前 Hub hidden state 重新计算各 Expert 的贡献权重，使得模型能在同一序列内动态切换不同 Expert 的主导地位。这与 Expert Routing baseline（整个 prompt 选一个模型处理全部 token）形成对比。

术语是什么？
BTS 的 token-level gating 机制：
- Gate 计算基于 Hub 的当前 hidden state $h_0$（每个 token 不同）：$g = \text{softmax/sigmoid}(\text{dropout}(w_{\text{gate}}(h_0)))$
- Expert 贡献随 token 变化 → 同一 prompt 内，不同任务段自动激活不同 Expert
- Gate 值可视化验证：
  - Math 任务（GSM8K）：Math Expert gate → 1，其他 Expert gate → 0
  - Translation 任务（Flores）：Multilingual Expert + Seed 交替主导，gate 值在 prompt 和 generation 阶段动态变化
  - Context-switching 场景（Flores→GSM8K→TriviaQA 拼接）：Multilingual Expert gate 在 Flores 段高 → Math Expert gate 在 GSM8K 段接管 → Seed gate 在 TriviaQA 段最高

从系统架构角度拆解术语。
```
# BTS Token-level Gating 推理流程
for each token t in sequence:
    h_hub[t] = HubModel.forward(x[t])
    h_experts[t] = [Expert_i.forward(x[t]) for i in 1..n]
    
    # Stitch layer gate (每 token 独立计算)
    g[t] = softmax/sigmoid(w_gate(h_hub[t]))
    
    if merge_into_hub:
        h_hub[t] = weighted_merge(h_hub[t], h_experts[t], g[t])
    else:
        h_experts[t] = [gated_update(h_experts[t][i], h_hub[t], g[t][i])]
```

对比 Expert Routing baseline（序列级决策）：
```
# Expert Routing: 整个序列固定使用同一模型
prompt_embed = mean([Embed(token) for token in prompt])
model_idx = argmax(Router(prompt_embed))  # 一次性决策
for token in sequence:
    output[token] = models[model_idx].forward(token)  # 全序列固定模型
```

术语一般如何实现？如何使用？
- BTS 中自动实现：每个 token 经过 Stitch Layer 时重新计算 gate，无需显式设计 context-switching 逻辑
- 可视化分析：提取最终 Experts-into-Hub Stitch Layer 的 gate values，按 token 位置作图
- 优势：无需 prompt 级别路由决策，支持混合任务输入；Gate 值提供模型决策的可解释性
- 局限：所有 Expert 均需全程参与前向传播（与 MoE 的稀疏激活不同），推理计算量等于所有 Expert 之和

涉及论文标题：
- BTS Harmonizing Specialized Experts into a Generalist LLM

---
