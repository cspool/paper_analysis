## MemIndex (Fast Memory Indexing)（快速记忆索引）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
MemIndex 是 FlexMem 提出的快速记忆索引方法，解决 encoding-based reading 的"每换一个问题需重新 encoding 全视频"的限制。MemIndex 将 memory reading 与 memory encoding 解耦：(1) encoding 阶段不传入 Tq，仅做视觉 KV cache 压缩，同时以更高压缩比生成 compact visual index tensor（k×d 维，k=5 个 token，远小于原始 |Vi|×d）；(2) reading 阶段，通过轻量 statistical fitting 来近似 encoding-based reading 的 relevance score。具体使用 linear regression 学习函数 σ(r̂_i) = Σ α^l·r̂_i^l 来拟合 ground-truth g_i，并基于 learned α^l 权重选择 top-K 个最重要的 cache 层（K=3）。问题编码为 Q_{Tq}[-1]（最后一个 token 的 query embedding），视觉索引为 per-layer top-k salient key vectors。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# === MemIndex: Training ===
data = []
for video, Tq in training_set:
  g_i = encoding_based_reading(M_bank, Tq)  # teacher signal
  for each layer l in 3..L:
    q = MLLM.encode(Tq)[-1]  # 问题: 最后token的Q
    K* = topK_salient_keys(Mi, k=5)  # 视觉: top-k显著keys
    r̂_i^l = dot_product_attention(q, K*)
  data.append(([r̂_i^3..r̂_i^L], g_i))

alpha = LinearRegression().fit(data)  # 学习层权重
H = topK_indices(alpha, K=3)  # 选最重要K层

# === MemIndex: Inference ===
q = MLLM.encode(Tq)[-1]
for Mi in M_bank:
  r̂_i = sum(alpha[l] * attention(q, Mi.memindex[l]) for l in H)
  relevance[i] = sigma(r̂_i)
Y = MLLM.decode(M_bank[topK(relevance, na)], Tq)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
MemIndex 适用于：(a) Streaming QA——监控/直播场景，同一视频多次提问无需重新 encoding；(b) 多问题场景——一次性 encoding 全视频后，每个问题仅需轻量匹配（单次 attention + 加权求和）。在 OVOBench streaming QA 上，FlexMem + MemIndex 达 54.4% 平均性能，显著超过 Flash-VStream (27.4%) 和 VideoLLM-online (36.1%)。劣势：需要额外训练数据（以 encoding-based reading 为教师信号），且 statistical fitting 存在一定精度损失。

涉及论文标题：
- Scaling the Long Video Understanding of Multimodal Large Language Models via Visual Memory Mechanism
