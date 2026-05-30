## Salience-Determined Bit Allocation (SBA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
SBA 是 SliM-LLM (ICML 2025) 提出的 group-wise 混合精度 bit-width 分配算法。核心思想：利用 LLM 中 salience weight 在 channel 维度上呈现 spatial clustering 的现象（由 activation outlier channels 驱动，Theorem 1 证明：x_{:,p}^* >> x_{:,j} → H_{p,p} >> H_{j,j} → δ_{:,p} > δ_{:,k}），按 group 平均 salience 排序后，通过双指针搜索最优混合精度配置。目标函数为 KL divergence D_KL(xW^T || xŴ_sba^T)，从信息熵角度对齐量化前后输出分布。约束条件为 |G_{N-1}| = |G_{N+1}|（等量低/高精度 group 维持 target 平均 bit-width）。例如 2-bit 场景：高 salience group 给 3-bit，等量低 salience 给 1-bit，其余 2-bit。搜索空间有限（LLaMA-7B 仅需 16 iterations）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# 输入: 权重W (n×m), 校准数据x (t×m), 逆Hessian H^in, group_size β, target bit N
# 输出: 每个group的bit-width G[0..k-1]

# Step 1: 计算每个group的平均salience
k = m / β  # group数量 (e.g. 4096/128=32)
for i in range(k):
    w_g = W[:, i*β : (i+1)*β]
    S[i] = mean(w_g² / diag(H^in[i*β:(i+1)*β])²)

# Step 2: 按salience排序groups (ascending)
sorted_idx = argsort(S)

# Step 3: 双指针搜索最优混合精度比
best_kl = INF
for p in range(1, k//2 + 1):
    # lowest p groups → (N-1)-bit, highest p → (N+1)-bit, middle → N-bit
    Ŵ_mixed = fakequant(low_sal, N-1) ∪ fakequant(high_sal, N+1) ∪ fakequant(mid, N)
    kl = D_KL(softmax(x·W^T) || softmax(x·Ŵ_mixed^T))
    if kl < best_kl: best_kl, best_p = kl, p

# 分配: G[sorted_idx[0:best_p]] = N-1; G[sorted_idx[k-best_p:k]] = N+1; G[middle] = N
```
与 ILP（整数线性规划，HAWQ v2 方法）对比：ILP 在 {1,2,3}bit 候选空间中有更宽搜索范围，但 SBA 基于输出 KL 散度的双指针搜索在固定整数 bit-width 下实现更优匹配。LLaMA-7B 2-bit WikiText2 PPL: ILP=17.55 vs SBA=14.58。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
SBA 实现集成在 GPTQ 或 OmniQuant 的量化流程中，在权重量化前执行。校准数据 forward pass 获取每层输入 x 计算 Hessian 后，逐层运行 SBA 确定各 group 的 bit-width 配置，随后 GPTQ/OmniQuant 按配置对各 group 使用对应精度量化。SBA 双指针搜索复杂度 O(k²)，但 k 很小（LLaMA-7B 中 k=32，每层仅 16 次迭代），开销可忽略。开源代码：https://github.com/Aaronhuang-778/SliM-LLM。

涉及论文标题：
- SliM-LLM Salience-Driven Mixed-Precision Quantization for Large Language Models

---
