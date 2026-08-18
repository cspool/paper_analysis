## 注意力重要性分数与 recent-k 窗口（Attention-based Token Importance & Recent-k Window，H2O 式）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 注意力重要性分数量化每个已缓存 token 对后续生成的贡献，依据是注意力模式具有 locality 与稀疏性（H2O [10] 提出 heavy-hitter token 概念：少数 token 获得主导注意力）。SingularBit-KV 用它决定每个 KV token 的量化位宽：先对当前注意力图逐 attention head 做 max pooling（head-wise max pooling，因为不同 head 关注互补信息，保留任一 head 需要的 token 高精度）并归一化，再维护一个 recent-k 窗口（k=128）记录近 k 步的注意力分布以平滑瞬态模式，最终每 token 重要性取窗口内近 k 步 query 方向的最大值 $\mathcal{I}_i=\max_j\tilde{a}_j[i]$——用 max 而非 mean 是因为在线量化不可逆（一旦低比特存储，恢复高精度需重算 KV，开销大），必须保守保留 token 的峰值需求。
- 从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- pipeline（论文 Sec-IV-B）：①更新：$\tilde{a}_t=\mathrm{normalize}(\max_{h}\mathrm{softmax}(Q_tK^T)[h,:])$，$\tilde{a}_t\in\mathbb{R}^{N_t}$，窗口 $M_t\leftarrow[M_t;\tilde{a}_t]$ 淘汰最旧行；②打分：$\mathcal{I}_i=\max_{j\in[t-k+1,t]}M_t[j,i]$；③分配：把 $[0,1]$ 归一化重要性按线性容量递增的边界 $\{s_0..s_5\}$（$l_i2^{b+i}=m\cdot i+c$）映射到 5 级位宽 b~b+4。伪代码：
  ```
  # 头维最大池化（保留任何 head 需要的 token）
  def agg_attn(A_t):                 # A_t: (H, N_t)
      a = A_t.max(dim=0)             # 逐 head 取最大 → (N_t,)
      return (a - a.min()) / (a.max() - a.min())   # 归一化到 [0,1]
  M_t = deque(maxlen=k)              # recent-k = 128 窗口
  M_t.append(agg_attn(A_t))
  I_i = max(row[i] for row in M_t)   # 近 k 步 query 最大值（保守）
  b_i = threshold_map(I_i, s_0..s_5) # 容量线性递增阈值 → 5 级位宽
  ```
- 术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：算法层是轻量统计（逐层维护 k×N 窗口，复杂度 O(k·N·H) 远小于 attention 本体）；软件上可用 PyTorch 实现或结合 H2O（https://github.com/weizhehuang/H2O）的 heavy-hitter 打分；硬件上 SingularBit 用压缩引擎的 max-tracking 逻辑（maximum-tracking logic）在线实现 head-wise max pooling、归一化、recent-k 窗口与阈值映射，无需 CPU 干预。与 H2O/ZipCache 的差异：H2O 二值化逐出 token（剪枝），ZipCache 按显著性分位宽但只 token 一维；SingularBit-KV 把重要性细化为 5 级（捕捉中间重要度）并叠加 rank 维。论文数据：recent-k=128 窗口 + max 聚合使 KV2 下 CoQA 仅掉 2.0%，而二值化/单维方法在 2-bit 普遍掉 20%+。

涉及论文标题：
- SingularBit: Exploiting Synergy of Singular Value Decomposition and Low-Bit Quantization for Weight and KV Compression in LLM Inference
