## Co-activation Matrix & Gate Neurons (共激活矩阵与门神经元)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Co-activation Matrix 是 MoE-Prism 中用于量化 MoE expert 内部 neuron 间功能相似性的矩阵 C_co ∈ R^{C×C}（C 为 neuron 总数），定义为 C_co = B^T·B，其中 B ∈ {0,1}^{B×C} 是二值化激活矩阵——对每个 token，激活幅度处于 top-k_a 的 neuron 标记为 1（活跃），其余为 0。C_co[i,j] 表示 neuron i 和 j 在同一 token 下同时活跃的 token 总数，用作功能共激活频率的度量。Gate Neurons 是从每个 sub-expert 中选出的 r 个最具代表性的 neuron，作为该 sub-expert 的"功能中心"——通过计算每个 neuron 在该 sub-expert 内与其它 neuron 的累积共激活次数（centrality），取 top-r 得最高 centrality 的 neuron。Gate neuron 的激活 L1 范数被用作该 sub-expert 整体输出范数的轻量级代理估计，使 router 无需执行所有 sub-expert 即可判断哪些 sub-expert 对当前 token 最有用。

从算法pipeline角度拆解术语：
```
# Co-activation matrix construction
M = activation_matrix  # [B, C], from Neuron Activation Profiler
k_a = C * 3 // 4  # top 3/4 neurons considered "active"
B = zeros(B, C)  # binary activation matrix
for t in range(B):
    threshold = top_k(|M[t,:]|, k_a)
    B[t,:] = (|M[t,:]| >= threshold).astype(int)

C_co = B.T @ B  # [C, C], symmetric

# Gate neuron selection for sub-expert S_n
centrality = []
for neuron_i in S_n:
    co_sum = sum(C_co[neuron_i, neuron_j] for neuron_j in S_n)
    centrality.append((neuron_i, co_sum))
gate_neurons_S_n = top_r(centrality, r=4)  # by centrality desc
```
匹配案例：在 NAACL 2025 论文 "A Closer Look into Mixture-of-Experts in LLMs" 中，研究确认 MoE 中 neuron 行为类似 fine-grained expert，gate projection matrix 负责选择哪些 neuron 激活，且不同 expert 间的 neuron co-activation 模式在深层有更大的多样性。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- MoE-Prism 中 r=4（每个 sub-expert 选 4 个 gate neuron），使用 top 3/4 的激活 neuron 构建 B 矩阵，在 Wikitext-2-raw-v1 校准集上计算 C_co。
- 推理时仅需计算 gate neurons 的中间激活（O(r·B·d_model) vs 完整 sub-expert 的 O(C·B·d_model)），用 gate neuron 的平均 L1 norm 作为代理分数，router 据此选 top-k sub-experts。
- 这一机制是 training-free proxy gating 的基础，无需微调即可实现有效路由，PPL 接近原始模型。

涉及论文标题：
- MoE-Prism: Disentangling Monolithic Experts for Elastic MoE Services via Model-System Co-Designs

---
