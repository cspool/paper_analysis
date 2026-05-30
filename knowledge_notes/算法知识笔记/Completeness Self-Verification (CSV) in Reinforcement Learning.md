## Completeness Self-Verification (CSV) in Reinforcement Learning

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Completeness Self-Verification (CSV, 完备性自验证) 是一种 annotation-free 的中间步骤监督机制，作为 GRPO outcome-only reward 的补充。核心理念：让 policy model 自己验证中间搜索步骤是否充分 —— 搜索到的帧是否包含足够信息正确回答问题。CSV 通过将搜索帧集单独取出、禁用新搜索、要求模型重新回答，间接评估搜索质量。CSV 是弱监督过程奖励 (weakly-supervised process reward) 的一种形式：不标注"哪些帧应被搜索到"，而是通过结果（重新回答的正确性）反向验证过程的充分性。关键设计：(1) 禁止工具调用（CSV prompt 不含 tools），(2) 允许 "I don't know"（诚实承认证据不足），(3) 条件奖励（仅正确轨迹施加，避免强化错误搜索策略）。

从算法pipeline角度拆解术语：
```
# CSV 执行流程
# 输入: C (交错CoT), Q (问题), π_θ (policy model)
V_c = ∪{V_i for (T_i, V_i) in C}  # 提取所有搜索帧+时间戳

# CSV prompt (与主推理 prompt 完全不同)
csv_prompt = "You are a helpful assistant. Please answer visual 
  questions as briefly as possible. When you don't have enough 
  visual information, please say 'I don't know'."

csv_input = concat(csv_prompt, V_c_frames_with_timestamps, Q)

# CSV 推理: π_θ 仅基于 V_c 回答，禁止搜索
A_c = π_θ.generate(csv_input, blocked_tokens=[<tool_call>])

# CSV reward: 仅原始答案正确时计算
R_c = 1[Acc(A, A*) > 0.5] · Acc(A_c, A*)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
CSV 不需要额外的 reward model —— 复用 policy model 自身做重新回答，保持训练简洁。CSV 仅在 RL 训练阶段使用，推理时无需额外的 CSV forward pass。Ablation 显示：无 CSV 时 GRPO 训练约 300 step 崩塌（模型停止搜索，completeness 降为零）；CSV + accuracy reward 组合实现最佳 QA（VideoMME 66.6%）。适用场景：任何 multi-turn tool-calling RL 训练，其中中间检索/搜索步骤质量难以直接评估。

涉及论文标题：
- TimeSearch-R__Adaptive_Temporal_Search_for_Long-Form_Video_Understanding_via_Self-Verification_Reinforcement_Learning
