## Learning-to-Hash for Top-k Attention in LLMs (面向LLM Top-k注意力的学习哈希)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Learning-to-Hash for Top-k Attention 是一种将学习式哈希（learning-to-hash）技术集成到LLM top-k attention过程中的方法。与传统的Locality-Sensitive Hashing（LSH，基于随机投影）不同，learning-to-hash通过可训练的hash函数h(x)=2·Sigmoid(σ·xW_H)-1将高维query/key向量映射为紧凑的二进制hash codes（如128-bit），使得相似向量被赋予Hamming距离小的hash codes。

核心逻辑链：(1) Top-k attention只需知道哪些keys与当前query最相关（序数比较），而非精确的qk score数值；(2) learning-to-hash通过优化min Σs_i||h(q)-h(k_i)||² + balance/uncorrelation约束，将连续向量空间的相似性保持映射到Hamming空间；(3) 在推理时，仅需bitwise_xor+popc计算Hamming距离（O(s×rbit/32)），选出top-k最近keys进行sparse attention。

HATA（ACL 2025 Findings）是该方法的代表性工作，训练数据由prefill阶段的qk pairs采样构建（top 10%为正样本，线性衰减标签[1,20]；90%为负样本，标签-1），每attention head独立训练hash权重W_H∈R^{d×rbit}。与MagicPIG（LSH-based, 1500-bit）相比，HATA仅需128-bit的learned hash codes即可达到near-lossless精度。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Learning-to-Hash Training Pipeline（HATA）：**

```
# === Training Data Construction ===
for each sequence in calibration set:
    Q, K = prefill(sequence)                  # [n, d]
    for head in 1..H:
        m = random(n/2, n)                    # sample query position
        Score = Q[m] @ K[:m]^T                # causal qk scores
        top10 = TopK(Score, 0.1*m)
        labels = -1 * ones(m)                  # negative samples
        labels[top10] = linspace(20, 1)        # positive: 20→1 decay
        store(q_m, K[:m], labels)

# === Per-Head Hash Training (SGD) ===
W_H = init(d, 128)                             # rbit=128
for epoch in 1..15:
    for batch in training_chunks(32K):
        h_q = 2 * Sigmoid(0.1 * q @ W_H) - 1  # relaxed hash
        h_k = 2 * Sigmoid(0.1 * k_batch @ W_H) - 1
        loss = 0.01 * Σ s_i * ||h_q - h_k_i||²    # similarity
             + 2.0 * ||Σ h_k_i||²                  # bits balance
             + 1.0 * ||W_H^T @ W_H - I||            # uncorrelation
        W_H = SGD(lr=0.1, momentum=0.9)(loss)
```

**推理时的Hash-based Key Retrieval：**
```
# HATA Decode (per head)
Q_H = BitPack(Sign(Q @ W_H))                 # [1, 4] INT32
S = bitcount(bitwise_xor(Q_H, K_H_cache))     # Hamming distances [1, s]
Idx = TopK(-S, N)                             # smallest distances = most similar
O = FlashAttention(Q, K_cache[Idx], V_cache[Idx])
```

术语一般如何实现？如何使用？

HATA开源在https://github.com/gpzlx1/HATA。Hash权重W_H离线训练后作为固定模型参数加载（训练集150K-300K qk pairs）。每head独立训练W_H，支持MHA（32 heads → 32个W_H）和GQA（8 KV heads → 8个W_H per layer）。适用于长上下文（≥32K）或大batch LLM推理加速，与KVCache compression/offloading方法正交可组合。前两层保留vanilla attention（attention outlier layers）。

涉及论文标题：
- HATA__Trainable_and_Hardware-Efficient_Hash-Aware_Top-k_Attention_for_Scalable_Large_Model_Inference

---
