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
