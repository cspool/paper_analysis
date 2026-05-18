## Vector-wise Similarity Matching

术语是什么？通过联网搜索让回答具体和精准。
Vector-wise Similarity Matching是Focus提出的细粒度冗余检测技术，将token embedding（典型维度如3584）切分为多个32维vectors，在small spatiotemporal block内对vectors做localized cosine similarity comparison。与token-wise matching（比较整个3584维token embedding）相比，vector-wise matching能揭示更多冗余：论文对LLaVA-OneVision的MLVU数据集分析显示，64%的8维vectors cosine similarity超过0.9，而仅18%的3584维full tokens超过0.9。这是因为motion导致的partial alignment（一个token可能部分匹配多个邻近token的不同部分）只能通过sub-token级别的比较捕获。Focus用2×2×2时空block限定比较范围，每个block内最高index vector作为key与其他7个vectors比较，匹配则记录代表vector的index。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Vector-wise similarity matching流程：
```
# After GEMM produces an m×n output tile (m=1024, n=32)
vectors = tile.reshape(1024, 32)  # 1024 vectors of 32 dimensions each
# Build 2×2×2 spatiotemporal blocks
for t in [frame_A, frame_B]:
    for h in 0..H-1 step 1:
        for w in 0..W-1 step 1:
            block = [vectors[t,h,w], vectors[t,h,w+1],
                     vectors[t,h+1,w], vectors[t,h+1,w+1],
                     vectors[other_frame_t,h,w], ...]  # 8 vectors
            key = block[-1]  # highest index vector
            for v in block[0:7]:
                dot_prod = sum(key[p] * v[p] for p in 0..31)
                cos_sim = dot_prod / (L2_norm[key] * L2_norm[v])
                if cos_sim > 0.9:
                    similarity_map[v_idx] = representative_idx(key)
                    # v is redundant, reuse key's index
deduplicated = vectors[unique_indices]  # p vectors, p < 1024
```
L2-norm per vector可precompute并存储在buffer中，使matching仅需1次dot-product和少量element-wise operation。

Granularity trade-off：更小vector（如8-dim）揭示更多冗余但增加comparison和metadata开销；更大vector（如128-dim）减少comparison但降低sparsity。Focus选vector length=32作为综合平衡点。

涉及论文标题：
- Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

---

