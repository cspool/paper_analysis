## Sparse-Linear Attention (SLA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SLA是Zhang et al. (2025, Tsinghua & UC Berkeley) 提出的可训练混合注意力机制，融合稀疏和线性注意力来加速DiT模型。核心洞察：注意力权重可分解为P = (P⊙M) + (P⊙(1-M))，前者（~8%）高rank需O(N²)，后者（~92%）极低rank可用线性注意力。SLA通过压缩mask预测将注意力块分三级：Critical (top k_h%, 默认5%) → O(N²) FlashAttention；Marginal (中间~85%) → O(N) 线性注意力（预计算后仅矩阵加法）；Negligible (bottom k_l%, 默认10%) → 跳过。关键设计：可学习投影Proj(O^l)减少分布不匹配，fine-tuning仅需2000步使模型自适应。

从算法pipeline角度拆解术语：
```
SLA Pipeline per attention forward:
1. P_c = Softmax(pool(Q)pool(K)^T/√d)            // compressed N/b_q × N/b_kv
2. M_c = classify(P_c, k_h=5%, k_l=10%)           // +1/0/-1 per block
3. Precompute: h_j=φ(K_j)^T V_j, z_j=rowsum(φ(K_j)^T)
4. Fused loop: for i,j blocks:
     M_c[i,j]==+1 → FlashAttention(O(N²))         // ~5% blocks
     M_c[i,j]==0  → H_i+=h_j; Z_i+=z_j            // ~85% blocks (O(N))
     M_c[i,j]==-1 → skip                          // ~10% blocks
5. O = O_s + Proj(O_l)                            // fused output
```
SLA在95% sparsity下FLOPs=2.73T (19.3× vs full 52.75T)，VA=76.96≈Full 76.78，远优于Sparse Only 85% (VA=64.00, 7.91T) 和Linear Only (VA=0.042, 0.10T)。

术语一般如何实现？如何使用？
代码：https://github.com/thu-ml/SLA。单fused CUDA kernel实现前向+反向。使用流程：加载DiT → 替换注意力层 → fine-tune 2000步 → SLA kernel推理。推荐超参数：k_h=5%, b_q=b_{kv}=64, φ=softmax。RTX 5090上：13.7× kernel加速 vs FlashAttention2，2.2×端到端加速（Wan2.1-1.3B视频生成）。额外效率优化：Lookup table（sparsity>90%）、Pre-aggregation（减法替代加法）、Method of Four Russians。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
