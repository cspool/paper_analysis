## Hash-bit Key Clustering（哈希比特键聚类）

术语是什么？通过联网搜索让回答具体和精准。

Hash-bit Key Clustering是V-Rex ReSV算法的核心组件，通过随机hyperplane projection将高维key向量降维并二值化为hash-bit（仅≤0.5%原始dimension），用XOR+popcount计算Hamming distance替代cosine similarity做token聚类。利用视频相邻帧key token的高时空相似性（cosine similarity热力图验证），将相似token归入同一cluster。Hamming distance与cosine similarity相关性约0.8。关键优势：(1) bit-wise操作为硬件友好，避免浮点乘加；(2) 保留原始token value供后续attention（区别于merge/replace方法）；(3) 聚类后仅对representative KeyCluster而非完整key cache做后续computation。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
// Step 1: Hash-bit Generation (每个新frame执行一次)
Hyperplanes = RandomMatrix(N_hp=32, key_dim)
for each key token t in current frame:
    Key_hp[t] = Key[t] @ Hyperplanes^T     // N_hp=32维投影
    Hash_bit[t][i] = (Key_hp[t][i] > 0) ? 1 : 0  // 二值化

// Step 2: Hamming Distance Clustering
HC_table = load_existing_clusters()  // {cluster_id, KeyCluster, hash, token_count}
for each current token t:
    curr_hash = Hash_bit[t]
    for each existing cluster c:
        dist = popcount(curr_hash XOR HC_table[c].hash)
        if dist < Th_hd (论文设7):
            assign t to cluster c
            HC_table[c].token_count++
            HC_table[c].KeyCluster = mean(tokens in cluster c)
    if no cluster matched:
        create new cluster with token t

// HC_table metadata overhead: avg 1.67% of full KV cache
// avg 32 tokens per cluster
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

在V-Rex中，HCU（Hash-bit Cluster Unit）硬件加速clustering：current hash-bit memory + key cache hash-bit memory + NHCU_h个并行XOR accumulator（NHCU_w=16 inputs）。LEE的VPE生成hash-bit→HCU读取已有cluster hash-bit→XOR accumulator并行计算Hamming distance→与Th_hd比较→HC table updater更新metadata。通用软件实现可用NumPy bitwise XOR: `numpy.bitwise_xor(curr, cluster).sum(axis=-1)` 计算Hamming distance，配合threshold筛选cluster。论文未开源。

涉及论文标题：
- V-Rex: Real-Time Streaming Video LLM Acceleration via Dynamic KV Cache Retrieval

---

