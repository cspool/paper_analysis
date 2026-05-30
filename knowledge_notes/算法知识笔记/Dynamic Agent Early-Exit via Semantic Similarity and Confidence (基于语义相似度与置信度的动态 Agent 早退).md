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
