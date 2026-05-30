## Two-Step VLM Candidate Clustering

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Two-Step VLM Candidate Clustering 是 Mordal 避免 O(m²n²) pair-wise CKA 的聚类策略。直接对所有 m×n VLM 候选两两计算 CKA 需 O(m²n²) 次评估。两步策略：(1) 先在 VE 空间做 CKA 聚类（O(m²) 次）；(2) 再在每个 VE cluster 内基于 medoid VE 输出对 LLM 做 CKA 聚类（O(C_ve·n²) 次，C_ve≪m²）；(3) Cartesian product 合成 VLM candidate clusters。避免了计算 dissimilar VE 与不同 LLM 组合间的 CKA——不同 VE cluster 的候选性能差异大，不需要 cluster 间细粒度比较。

从算法pipeline角度拆解术语，给出具体例子。
```
Algorithm 2:
Step 1 - VE Clustering:
  for each pair in 7 VEs: dist = 1 - CKA(VE_A, VE_B)
  C_ve = HierarchicalClustering(dist, t_ve=0.7)
  // e.g., 3 clusters: {CLIP,SigLIP,DFN}, {InternViT,DINOv2}, {EVA-CLIP,ConvNeXt}

Step 2 - LLM Clustering (per VE cluster):
  for each VE_cluster:
      medoid = PickMostCentral(VE_cluster)
      fixed_out = WarmupProjector(medoid(images))  // 10 rounds
      for each pair in 7 LLMs: dist = 1 - CKA(LLM_A.last_hidden, LLM_B.last_hidden)
      C_llm = HierarchicalClustering(dist, t_llm=0.8)
      // e.g., {Vicuna,Llama-2}, {Llama-3,Mistral,Qwen}, {Phi-3,Gemma}

Step 3 - Cartesian: C_vlm = {VE_c × LLM_c for each combination}
// Total: ~9-15 candidate clusters (vs 49 individuals)
```

术语一般如何实现？如何使用？
使用 MinibatchCKA + `scipy.cluster.hierarchy`。关键超参：t_ve=0.7, t_llm=0.8。消融（Table 5）：t_ve=0.5→τ=0.52（太粗），t_ve=0.9→τ=0.86 但 1041h（太细）。LLM 聚类使用 last hidden state（最佳聚类性能）。Warmup round=10 确保 medoid projector 充分训练产生有意义的 LLM 输入表示。

涉及论文标题：
- Mordal: Automated Pretrained Model Selection for Vision Language Models

---
