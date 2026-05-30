## Viterbi Algorithm in Trellis Coded Quantization（格状编码量化中的维特比算法）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Viterbi 算法（Forney, 1973）是一种在 trellis 图上寻找最优路径的动态规划算法。在 TCQ 中，Viterbi 算法求解最小化问题：minimize Σ_{i=1}^{T/V} ||C_{x_i} - s_i||² over x_1,...,x_{T/V} 为图 G 上的 walk。核心是价值函数 V_t(x) = 以节点 x 结束的、前 t 步的最优累计失真。更新规则：V_t(y) = min_{(x,y)∈G} V_{t-1}(x) + ||C_y - s_t||²。通过回溯指针记录每个状态的最优前驱，最终从最小 V_T 的状态反向追踪得到最优路径。复杂度 O(2^L T)，与比特率 k 无关。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
在 QTIP 的 PyTorch/CUDA 实现中（量化 T=256 维序列，L=16, k=2, V=1）：
```
输入: s[0..T-1] ∈ R^T, codebook C ∈ R^{2^L}
初始化:
  V[0..2^L-1] ← ||C[0..2^L-1] - s[0]||²
  ptr = zeros(T, 2^L)  # 回溯指针
for t = 1 to T-1:
  V_new = ones(2^L) × INF
  for each transition (i→j) in G:  # bitshift: j = ((i<<kV) & (2^L-1)) + c
    err = ||C[j] - s[t]||²
    if V[i] + err < V_new[j]:
      V_new[j] = V[i] + err
      ptr[t, j] = i
  V ← V_new
# 回溯
best = argmin(V); path[T-1] = best
for t = T-2 down to 0:
  path[t] = ptr[t+1, path[t+1]]
输出: 编码 = [起始状态(L bits)] + [每步 c 值(k bits) × T]
```
在 GPU 上，每步的 min 操作可高度并行（每个节点独立），实际实现使用 CUDA reduce 优化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Viterbi 算法在 TCQ 中的使用仅限离线量化阶段，推理时不需要（直接 bitshift 解码）。因复杂度 O(2^L T)，L>20 时不实用（L=16 时 2^16=65K 状态已可接受）。BCJR-QAT (2026) 提出用 BCJR forward-backward 算法替代 Viterbi 实现可微量化感知训练（QAT），将硬量化松弛为 soft 期望，进一步扩展了 trellis 量化的应用范围。

涉及论文标题：
- QTIP: Quantization with Trellises and Incoherence Processing
