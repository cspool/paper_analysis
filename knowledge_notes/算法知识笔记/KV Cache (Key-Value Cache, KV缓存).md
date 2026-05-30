## KV Cache (Key-Value Cache, KV缓存)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
KV Cache（Key-Value Cache）是 Transformer 自回归推理中用于存储每层注意力机制中所有已生成 token 的 Key 和 Value 张量的高速缓存。在 prefill 阶段，输入 prompt 经各层权重 $W_K, W_V$ 投影得到 $X_K, X_V \in \mathbb{R}^{b \times l_{prompt} \times d}$，存入 KV Cache。在 decoding 阶段，每生成一个新 token $t$，计算 $t_K = tW_K, t_V = tW_V$ 后 Concat 到 KV Cache：$X_K \leftarrow \operatorname{Concat}(X_K, t_K)$。然后通过 $A = \operatorname{Softmax}(t_Q X_K^\top)$ 和 $t_O = A X_V$ 完成注意力计算。KV Cache 避免了每一 decoding step 都重新计算所有历史 token 的 Key/Value，将计算复杂度从 $O(l^2 d)$ 降至 $O(ld)$。但代价是内存占用巨大：KV Cache 形状为 $b \times (l_{prompt} + l_{gen}) \times d$，在 OPT-175B、b=512、l=544 时可达 1.2TB（3.8× 模型权重）。KIVI 论文针对 KV Cache 内存瓶颈提出了 2bit 量化方案。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
KV Cache 在 LLM 推理 pipeline 中的生命周期：

```
# === Prefill Phase ===
输入: X ∈ R^{b × l_prompt × d}
for layer in layers:
    X_K = X @ W_K    # [b, l_prompt, d]
    X_V = X @ W_V    # [b, l_prompt, d]
    KV_cache[layer] = (X_K, X_V)  # 存储全精度 KV
    # ... attention computation ...
    X = output  # 传给下一层

# === Decoding Phase (per token) ===
输入: t ∈ R^{b × 1 × d}
for layer in layers:
    t_K = t @ W_K,  t_V = t @ W_V
    X_K, X_V = KV_cache[layer]           # 从内存加载
    X_K = Concat([X_K, t_K], dim=token)  # 追加新 token
    X_V = Concat([X_V, t_V], dim=token)
    KV_cache[layer] = (X_K, X_V)          # 更新缓存
    
    t_Q = t @ W_Q
    A = Softmax(t_Q @ X_K^T / sqrt(d))   # attention scores
    t_O = A @ X_V                         # attention output
    # ... FFN ...
    t = output
```

内存分析：KV Cache 总大小为 $2 \times n_{layers} \times b \times (l_{prompt} + l_{gen}) \times d_{head} \times n_{heads}$ bytes（FP16 时乘 2）。KIVI 将 Key/Value 压缩到 2bit 后减少了约 8× 的 KV Cache 内存。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
HuggingFace Transformers 中 `model.generate(use_cache=True)`（默认）自动管理 KV Cache。vLLM 中使用 PagedAttention 将 KV Cache 分页管理以消除碎片化。FlexGen 通过 offloading 将 KV Cache 转移到 CPU/磁盘。KIVI 通过量化将 KV Cache 压缩到 2bit 以减少 GPU 内存占用，兼容 weight-only 量化和 PagedAttention。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---
