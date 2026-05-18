## Layer Similarity Heatmap（层间相似度热力图）

术语是什么？通过联网搜索让回答具体和精准。

Layer Similarity Heatmap 是 LEGO 提出的一种 LLM 层间知识相似度分析方法，用于指导 resource-oriented layer-skipping adaptor 选择要跳过的连续 Transformer 层段。它通过计算所有 Transformer layer 对之间的输出 tensor cosine similarity，构建一个 M×M 热力图矩阵（M 为 Transformer layer 数量），可视化各层输出表示的相似程度。热力图的对角线反映不同跳层配置的候选层段。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

构建流程：
```
输入: LLM model (M transformer layers), dataset D (如 WebInstruct)
输出: M×M similarity heatmap H

for each sample x in D:
    前向传播，提取每层 L_i 的输出 tensor T_i (i ∈ [1, M])
    for each layer pair (i, j):
        提取 T_i 和 T_j 的 hidden states（取前 16 个 output token，与游戏输出长度一致）
        H[i][j] += cosine_similarity(T_i, T_j)

H = H / |D|  // 平均所有样本
```

热力图的关键观察（论文 Fig.8）：
1. **对角线反映跳层配置**：对角线 H[i][j] where j-i=N 代表所有"N 跳层"候选项——即跳过从 L_i 到 L_j 的层段
2. **后层高相似度**：Llama3-8B 和 Mistral-7B 的 latter layers 普遍高相似度（>0.8），说明后续层引入的新信息较少，可安全跳过
3. **最后层与倒数第二层低相似度**：最后 transformer layer 编码与 output layer 对接的关键知识，不应跳过
4. **初始层低相似度**：early layers 输出差异大，跳过会丢失基础语义信息

使用热力图选择跳层配置：
- 跳 4 层时（Llama3-8B）：沿 j-i=4 对角线找到相似度最高的区间 → L25-L29
- 跳 8 层时（Llama3-8B）：沿 j-i=8 对角线找到相似度最高的区间 → L23-L31
- 选择连续区间而非离散层：论文实验显示跳过离散层比跳过连续层造成更大的性能退化（因为 knowledge is distributed both within individual layers and across inter-layer connections）

与 LLM-Streamline 的区别：LLM-Streamline 也提出替换连续 transformer layers，但 LEGO 认为其推理过程不充分——"90% 连续层对超过 80% 相似度"不足以证明可以安全跳过，离散跳层比连续跳层造成的知识损失更大。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现要点：
- **相似度度量**：使用 cosine similarity（而非欧氏距离或其他度量），因为 transformer hidden states 的方向（而非幅值）承载更多语义信息
- **采样策略**：使用 2400 samples from WebInstruct，每个 sample 取前 16 个 output token 的 hidden states（与游戏场景 LLM 输出长度一致）
- **计算成本**：O(M² × |D| × d)，M 为层数、d 为 hidden dimension。对 32 层 Llama3-8B，该计算在单 GPU 上数小时内完成
- **通用性**：论文在 Llama3-8B、Mistral-7B、DeepSeek-V2-Lite（MoE 28层）、Mixtral-8x7B（MoE 32层）上均构建了 heatmap，验证了方法的跨模型适用性
- **一次构建，多次复用**：同一 Game-LLM 对的 heatmap 只需构建一次，后续所有跳层配置的选择均基于此 heatmap

涉及论文标题：
- LEGO: Supporting LLM-enhanced Games with One Gaming GPU

