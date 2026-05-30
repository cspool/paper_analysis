## Locality Sensitive Hashing (LSH)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Locality Sensitive Hashing (LSH) 是一族哈希函数，其核心性质是：相似的输入向量以更高的概率被映射到相同的哈希码，而不相似的向量以更低的概率碰撞。LSH使用两个超参数(K, L)：L张哈希表独立构建，每张表使用K个独立的随机哈希函数将高维向量投影到整数哈希码。LSH最初用于近似最近邻搜索(ANN)，能够在亚线性时间内检索与查询向量相似的数据点。MagicPIG首次将LSH用于decoder-only LLM的self-attention采样估计。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**LSH在MagicPIG attention估计中的pipeline**：

```
// 预处理阶段：为每个attention head构建L张哈希表
For each head:
  K_cache_centered = K - mean(K)  // centering
  For table l in 1..L:
    For each key k_i in K_cache_centered:
      hash_code = Sign(k_i @ W_l)  // W_l ∈ R^{d×K}, K-bit hash
      HT[l][hash_code].append(i)   // 存储key索引

// 解码阶段：每步基于LSH采样估计attention output
Input: q ∈ R^{1×d}, W ∈ R^{d×(K×L)}, HT (L hash tables)
Output: attention output estimate ō

// Step 1: GPU计算query哈希码
q_code = Sign(q @ W)  // K×L bit

// Step 2: CPU查询哈希表，收集采样集合S
S = {}
For each table l in 1..L:
  candidates = HT[l][q_code[l*K:(l+1)*K]]
  For each idx in candidates:
    collision_count[idx] += 1
For each idx where collision_count[idx] >= 2:  // 至少2表碰撞
  S.insert(idx)

// Step 3: 计算采样概率（基于SimHash碰撞概率）
For each i in S:
  p_i = 1 - (1/π) * arccos(q·k_i / (|q|·|k_i|))
  u_i = 1 - (1-p_i^K)^L - L·p_i^K·(1-p_i^K)^{L-1}

// Step 4: Self-normalized Importance Sampling估计
w_S = q @ K[S]^T / √d
ō = Softmax(w_S - log(u)) @ V[S]
```

术语一般如何实现？如何使用？

LSH的典型超参数选择：K=8~10（手动ablation确定），L基于目标计算预算调整（如K=10时L=150对应2%计算量）。K控制空间划分精度——K太小则采样过多不相关key（增加计算），K太大则碰撞概率低。L增加可以弥补K较大时碰撞概率低的问题，但增加DRAM开销（哈希表内存随L线性增长，如Llama-3.1-8B 96K context下(10,150)配置需14GB）。SimHash是目前最常用的余弦LSH家族；更高级的LSH如Cross-polytope hash可进一步减少哈希表大小。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
