## Residual Token Merging (残差 Token 合并 for KV Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Residual Token Merging 是 ZSMerge 提出的 KV Cache 压缩中的核心机制——将"被驱逐的 token"通过相似度驱动的增量均值聚合动态合并入有限个残差 slot，而非永久丢弃。与纯驱逐方法的根本区别在于：(1) 驱逐方法永久删除 KV 对，导致不可逆信息损失和 attention distribution drift；(2) 残差合并将驱逐转换为压缩编码——用 Br 个 slot 表示大量被驱逐 token 的聚合信息。

合并流程三步（per evicted token (k_t, v_t))：
1. **slot 选择**（Eq. 6）：通过最大 key 内积选择最兼容的 residual slot：$\hat{r} = \arg\max_{r \in \{1,\dots,B_r\}} \mathbf{k}_r^\top \mathbf{k}_t$。内积度量 key 向量的方向相似度。
2. **增量均值聚合**（Eq. 7）：$\mathbf{k}_{\hat{r}} \leftarrow \frac{w_{\hat{r}}\mathbf{k}_{\hat{r}} + \mathbf{k}_t}{w_{\hat{r}} + 1}, \quad \mathbf{v}_{\hat{r}} \leftarrow \frac{w_{\hat{r}}\mathbf{v}_{\hat{r}} + \mathbf{v}_t}{w_{\hat{r}} + 1}$——滑动平均格式，O(d) 增量更新无需存储历史 token。
3. **权重递增**：$w_{\hat{r}} \leftarrow w_{\hat{r}} + 1$，记录 slot r 已合并的 token 数量。

每个 attention head 独立维护 B_r 个残差 slot（K_r, V_r ∈ R^{B_r×d}）和权重向量 w ∈ R^{B_r}。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

```
# 初始化（per attention head）
K_r = zeros(B_r, d)  # 残差 key cache
V_r = zeros(B_r, d)  # 残差 value cache
w = zeros(B_r)       # 合并计数

# 每个解码步 T，处理被驱逐 token
def merge_evicted_token(k_t, v_t):
    # Step 1: 选择最兼容 slot
    scores = K_r @ k_t         # [B_r]，key 内积
    r_hat = argmax(scores)     # 最大内积 → 最相似

    # Step 2: 增量均值更新
    K_r[r_hat] = (w[r_hat] * K_r[r_hat] + k_t) / (w[r_hat] + 1)
    V_r[r_hat] = (w[r_hat] * V_r[r_hat] + v_t) / (w[r_hat] + 1)

    # Step 3: 权重递增
    w[r_hat] += 1

# 拼接压缩 cache
K_B = concat([K_p, K_c, K_r])  # proximity + context + residual
V_B = concat([V_p, V_c, V_r])
```

ZSMerge 实验验证：残差合并比纯驱逐在 ≤20% cache size 下减少 attention 输出误差 37-89%。Br=0 时退化为纯驱逐策略。

术语一般如何实现？如何使用？

ZSMerge 在 HuggingFace Transformers 中实现，替换 `scaled_dot_product_attention` 函数。每个 attention head 独立维护 B_r 个残差 slot（通常 B_r=2，为极小分配）。合并计算量 O(B_r·d) per evicted token，相比 full attention O(T²) 可忽略。残差 slot 在每次解码步动态更新，无需额外存储历史 token。代码开源：https://github.com/SusCom-Lab/ZSMerge。

涉及论文标题：
- ZSMerge: Zero-Shot KV Cache Compression for Memory-Efficient Long-Context LLMs
