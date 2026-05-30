## Redundancy-aware KV Cache Compression for Reasoning Models (R-KV)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

R-KV 是一种面向推理模型（如 DeepSeek-R1）、训练无关的 KV Cache 压缩方法。核心思路：现有 attention-based KV cache eviction 方法（如 SnapKV）仅依赖 attention score 判断 token 重要性，但推理模型的长 CoT 输出中存在大量重复的自反射和自我验证内容，这些冗余 token 因"self-attend 到自己"同样获得高 attention score，导致关键推理 token 被错误淘汰。R-KV 显式引入 redundancy estimation——通过 key vector 余弦相似度测量 token 间的语义冗余——并用 joint selection score Z = λ·I − (1−λ)·R 同时平衡重要性（I）和去冗余性（R）。

R-KV 三组件：(1) Importance Scoring：基于最后 α 个 observation tokens 的 attention weight，对 GQA 使用 max-pooling (而非 SnapKV 的 mean-pooling) 聚合 query head attention（§3.2）；(2) Redundancy Estimation：对 key vectors 做 L2 归一化后计算余弦相似度矩阵 S = K̄·K̄^T，对角线置零，保留最近 β 个高相似 token（不被标记冗余），剩余高相似 token 通过 softmax 归一化获得 redundancy score R_i^h（§3.3）；(3) Joint Selection：Z_i^h = λ·I_i^h − (1−λ)·R_i^h，λ=0.1（§3.4）。

R-KV 是 decoding-time 压缩：每 B_buffer=128 tokens 触发一次压缩，始终保留最后 α 个 observation tokens。在 AIME24 上，R-KV 以 10% KV cache budget 达到与 FullKV 持平（lossless compression），16% budget 时 even surpass FullKV by 5%（R1-Llama-8B）。进行固定 budget 分析：1024 budget @16K generation → 13.4× larger batch size, 9.2× throughput。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# R-KV decoding-time compression pipeline
# 每生成 B_buffer=128 tokens 后触发一次压缩

def R_KV_compress(K_full, V_full, L_full, B_budget, B_buffer, alpha=8, lambda_=0.1):
    L_budget = len(K_cache)  # 当前保留的KV cache长度
    
    # 检查是否触发压缩
    if L_full - L_budget < B_buffer:
        return K_full, V_full  # 不足一次压缩周期，不压缩
    
    # Step 1: 提取observation tokens + candidate tokens
    K_obs, V_obs = last_alpha(K_full, V_full)  # α=8，始终保留
    K_cand, V_cand = first_N(K_full, V_full, L_full - alpha)
    N_c = L_full - alpha  # 候选token数
    
    if N_c <= B_budget:
        return K_full, V_full  # 候选不足，不压缩
    
    # Step 2: Per-head Importance Scoring
    for h in range(H):  # H个attention heads
        Q_obs_h = Q_obs[:, h, :]  # [α, d_head]
        K_cand_h = K_cand[:, h, :]   # [N_c, d_head]
        
        # GQA: 同组query heads各自计算attention后max-pooling聚合
        A_h = softmax(Q_obs_h @ K_cand_h.T / sqrt(d))  # [α, N_c], Eq.(1)-(3)
        
        # 稳定化：滑动窗口max-pooling (窗口2W)
        A_tilde = sliding_window_maxpool(A_h, window=2W)
        
        # Per-token importance: 沿query维度取均值
        I_h[k] = mean(A_tilde[:, k]) for k in 0..N_c-1  # Eq.(4)
    
    # Step 3: Per-head Redundancy Estimation
    for h in range(H):
        # L2归一化key vectors
        K_norm_h = K_cand_h / (norm(K_cand_h, dim=-1) + 1e-8)  # [N_c, d]
        
        # 余弦相似度矩阵
        S_h = K_norm_h @ K_norm_h.T  # [N_c, N_c], Eq.(5)
        diag(S_h).fill_(0)  # 抑制自相似
        
        # 保留最近β个高相似token（largest indices）
        for i in range(N_c):
            similar_j = where(S_h[:, i] > T)  # 相似度阈值T
            recent_beta = similar_j.topk(k=beta, largest=True)
            S_h[recent_beta, i] = 0  # 不标记为冗余
        
        # 平均相似度 → softmax归一化 → redundancy score
        S_bar_h[i] = mean(S_h[:, i])  # Eq.(6)
        R_h = softmax(S_bar_h)  # [N_c], Eq.(6)
    
    # Step 4: Joint Selection + 跨head聚合
    for h in range(H):
        for k in range(N_c):
            Score_h[k] = lambda_ * I_h[k] - (1-lambda_) * R_h[k]  # Eq.(7)
    
    AggScore[k] = mean_h(Score_h[k])  # 跨head均值聚合
    
    # Step 5: Top-B_budget选择 + 拼接observation tokens
    top_idx = argmax(AggScore, k=B_budget)
    K_comp = cat([K_cand[top_idx], K_obs])
    V_comp = cat([V_cand[top_idx], V_obs])
    
    return K_comp, V_comp  # 压缩后长度 = B_budget + α
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

R-KV 开源：https://github.com/Zefan-Cai/R-KV。PyTorch 实现，基于 HuggingFace Transformers，在模型 forward pass 中插入 compression 逻辑。使用时设置超参：B_budget（KV cache budget）、B_buffer=128（压缩周期）、α=8（observation tokens）、λ=0.1（importance vs redundancy 权重）、T（similarity threshold，论文未明确给出值）、β（最近保留的高相似token数，论文未明确给出值）。R-KV 是 training-free 和 model-agnostic，可直接适用于任何使用 MHA/GQA 的 LLM。局限性：当前不支持 paged attention，且在没有 KV compression 专用接口的 serving 框架中需要 reallocate memory 引入开销（Appendix D）。

涉及论文标题：
- R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration
