## R-KV: Redundancy-aware KV Cache Compression for Training-Free Reasoning Models Acceleration

- 属于算法pipeline的实现是什么？实验比较什么？
  R-KV 提出一种面向推理模型（如 DeepSeek-R1）的训练无关、冗余感知的 KV Cache 压缩算法。核心思路：现有 attention-based KV cache 压缩方法（如 SnapKV）仅依赖 attention score 判断 token 重要性，但推理模型的长 CoT（Chain-of-Thought）输出中存在大量自反射和自我重复内容，这些冗余 token 同样获得高 attention score，导致缓存被冗余内容填满而关键推理 token 被错误淘汰。R-KV 通过三个核心组件联合解决：(1) Importance Scoring（重要性评分，§3.2）：基于最后 α 个 observation tokens 的 attention weight，对 MHA 直接计算 softmax 后的注意力矩阵，对 GQA 则额外对同一 KV head 组内各 query head 的 attention score 做 max-pooling 聚合，再对每个 token 的 per-query importance 做滑动窗口 max-pooling（窗口大小 2W）稳定化后取均值得到 per-head importance score I_i^h；(2) Redundancy Estimation（冗余估计，§3.3）：对各 head 内的 key vectors 做 L2 归一化后计算余弦相似度矩阵 S^h = K̄^h (K̄^h)^T ∈ R^{n×n}，抑制自相似（对角线置零）和最近 β 个高相似 token 的链接（保留最新出现的高相似 token 以避免丢失近期信息），再通过 softmax 归一化得到 per-head redundancy score R_i^h；(3) Joint Selection Strategy（联合选择，§3.4）：最终 selection score Z_i^h = λ·I_i^h − (1−λ)·R_i^h，取 top-B_budget tokens 保留。解码时每 B_buffer 步压缩一次，始终保留最后 α 个 observation tokens。

  实验比较：(a) R-KV vs SnapKV vs FullKV on MATH-500 和 AIME 2024，不同 KV cache budget（128/256/512/768/1024/1536/2048/2560/3072/4096），模型为 DeepSeek-R1-Distill-Llama-8B 和 DeepSeek-R1-Distill-Qwen-14B，pass@1 指标（每问题采样 64 responses）；(b) Throughput 和 Memory Saving 对比：R-KV vs FullKV vs SnapKV，8K 和 16K 生成长度，fixed budget (1024/1536/3072) 和 ratio budget (10%/34%/54%)；(c) λ 消融实验：λ ∈ {0, 0.01, 0.05, 0.1, 0.5, 1} on MATH-500，证明 λ=0.1 最优；(d) Token selection 可视化（Fig. 7）：R-KV vs SnapKV 在相同输入下的 selected KV token 分布对比。

- 硬件平台是什么，配置是什么。
  NVIDIA A100 80GB GPU。模型：DeepSeek-R1-Distill-Llama-8B（32 layers, 32 heads, GQA, hidden_size=4096），DeepSeek-R1-Distill-Qwen-14B（40 layers, 40 heads, GQA, hidden_size=5120）。超参：B_buffer=128, α=8, λ=0.1, similarity threshold T（论文未明确给出具体值），β（论文未明确给出具体值），max-pooling 滑动窗口 W（论文未明确给出具体值）。

- 模型是什么。数据集和bench分别是什么。
  模型：DeepSeek-R1-Distill-Llama-8B（简称 R1-Llama-8B），DeepSeek-R1-Distill-Qwen-14B（简称 R1-Qwen-14B）。数据集：(a) MATH-500（Hendrycks et al., 2021，最大生成长度 16384 tokens），(b) AIME 2024（MAA, 2024，最大生成长度 32768 tokens）。评估方式：pass@1 使用非确定性采样（temperature=0.6, top-p=0.95），每问题采样 64 responses。Baselines: SnapKV（原为 prefilling 设计，适配到 decoding 使用相同压缩间隔），FullKV（无压缩，作为 golden standard）。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码开源：https://github.com/Zefan-Cai/R-KV。论文页面：https://zefan-cai.github.io/R-KV.page/。项目为 PyTorch 实现。

  算法 pipeline（decoding-time KV cache compression）：

  **Step 1 - Decoding-Time Compression 触发判断**：
  设 KV cache 总长度为 L_full，budget 为 B_budget，buffer 为 B_buffer。每生成 B_buffer 个 token 后触发压缩。压缩时保留最后 α 个 tokens 作为 observation tokens，将现有 B_budget 个 cache tokens 与前 (B_buffer − α) 个 buffer tokens 合并为 N_c = B_budget + B_buffer − α 个候选 KV tokens。

  **Step 2 - Importance Scoring via Attention**（per head）：
  ```python
  # Q_obs ∈ R^{α × d}, K_cand ∈ R^{N_c × d} per head h
  A = softmax(Q_obs @ K_cand.T / sqrt(d))  # [α, N_c], Eq. (1)
  # 若为 GQA 则先对各 query head 独立计算再 max-pooling 聚合: Eqs. (2)-(3)
  # 稳定化：滑动窗口 max-pooling + 均值
  for j in range(α):
      A_tilde[j, i] = max(A[j, i-W:i+W])  # 窗口大小 2W, Eq. (4)
  I_i = mean(A_tilde[:, i])  # Eq. (4), per-token importance
  ```

  **Step 3 - Redundancy Estimation via Cosine Similarity**（per head）：
  ```python
  K_norm = K_cand / (norm(K_cand, dim=-1, keepdim=True) + 1e-8)  # Eq. (5)
  S = K_norm @ K_norm.T  # [N_c, N_c], 余弦相似度矩阵
  S.diagonal().fill_(0)  # 抑制自相似
  # 保留最近 β 个高相似 token 不被标记为冗余
  for i in range(N_c):
      similar = where(S[:, i] > T)  # similarity threshold T
      recent_beta = topk_indices(similar, k=β, largest=True)  # 最近 β 个
      S[recent_beta, i] = 0
  S_bar_i = mean(S[:, i])  # 平均相似度
  R = softmax(S_bar)  # [N_c], 归一化冗余分数, Eq. (6)
  ```

  **Step 4 - Joint Selection**（per head）：
  ```python
  Z_i = λ * I_i - (1-λ) * R_i  # Eq. (7), λ=0.1
  # 跨 head 聚合：AggScore_k = mean_h(Z_{k,h})
  # 选择 AggScore 最高的 B_budget tokens
  topk_indices = argmax(AggScore, k=B_budget)
  # 拼接 selected tokens + α observation tokens
  K_comp = cat([K_cand[topk_indices], K_obs])
  V_comp = cat([V_cand[topk_indices], V_obs])
  ```

  **关键数值**：以 R1-Llama-8B on AIME24 为例，平均生成长度 ~15,536 tokens。10% ratio budget → B_budget=1,536, B_buffer=128。压缩后仅保留约 1554 KV tokens vs 15536 FullKV，节省 ~90% KV cache 内存，batch=1 时 throughput 从 75.44 tok/s (FullKV) 提升至 80.46 tok/s（略提升），最大 batch 从 62 提升至 402（6.5×），端到端 throughput 从 849 tok/s 提升至 3252 tok/s（3.8×）。16K 生成长度下优势更明显：最大 batch 从 30 提升至 402（13.4×），throughput 从 347 tok/s 提升至 3189 tok/s（9.2×）。
