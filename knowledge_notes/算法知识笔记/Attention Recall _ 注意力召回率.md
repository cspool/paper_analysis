## Attention Recall / 注意力召回率

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Attention Recall 是量化稀疏注意力质量的诊断指标，定义为稀疏注意力所选 token 子集捕获的真实注意力分布的占比。公式：$R_i = \frac{\sum (\text{softmax}(W_i)[\rho])}{\sum (\text{softmax}(W_i))}$，其中 $W_i = \frac{Q_i K_i^T}{\sqrt{d}}$ 为 head i 的 pre-softmax attention scores，ρ 为稀疏注意力选择的 token 子集（|ρ| = k < L_kv）。R_i 取值范围 [0, 1]，越高表示选中的 token 越准确捕获了 full attention 的信息。在推理任务（reasoning）中，高 attention recall 是保持准确率的必要条件（非充分条件），因为即使小选择误差在数千步 decoding 中也会累积为逻辑不一致。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

Attention Recall 的计算过程（以单 head 单 decoding step 为例）：
```
输入: Q_i (1 x d), K_i (L_kv x d), V_i (L_kv x d), 选中索引 rho (k 个)
1. 计算完整 attention scores: W_i = Q_i @ K_i^T / sqrt(d)  # [1, L_kv]
2. 计算完整 attention weights: A_i = softmax(W_i)           # [1, L_kv]
3. 计算选中部分的 attention mass: mass_retained = sum(A_i[rho])  # 标量
4. 计算全部 attention mass: mass_total = sum(A_i)               # = 1.0 (softmax)
5. Attention Recall: R_i = mass_retained / mass_total
```

LessIsMore 论文中用 Running Average Attention Recall 追踪长程 decoding 中的累积质量：每 N 步采样一次 R_i，对全部 head 取平均。Running average 可平滑单步波动，显示 recall 随 generation length 的增长趋势。论文图 1a 显示：StreamingLLM 和 TidalDecode 在 32K decoding 过程中 recall 从 ~90% 分别退化至 ~65% 和 ~75%，而 LessIsMore 稳定在 ~90%。

术语一般如何实现？如何使用？

Attention Recall 是离线分析工具，不参与在线推理。主要用于：(1) 对比不同稀疏注意力方法的 token 选择质量；(2) 诊断推理过程中的 selection error 累积（recall 退化 vs generation length 的曲线）；(3) 验证设计选择的合理性（如 CUSA 的跨头统一选择 vs per-head 独立的 recall 对比，图 4）。LessIsMore 论文利用 attention recall 的关键发现：即使在推理任务（AIME）和检索任务（NiTH）上使用相同 token budget，稀疏注意力在推理上的 recall 远低于检索（图 1b），因为推理需要更多 decoding step，selection error 累积更大。

涉及论文标题：
- Less Is More: Fast and Accurate Reasoning with Cross-Head Unified Sparse Attention
