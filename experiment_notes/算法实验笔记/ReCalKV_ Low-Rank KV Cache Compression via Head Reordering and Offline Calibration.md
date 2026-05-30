## ReCalKV: Low-Rank KV Cache Compression via Head Reordering and Offline Calibration

- 属于算法pipeline的实现是什么？实验比较什么？
  ReCalKV 是一种后训练（post-training）低秩 KV Cache 压缩方法，通过分别针对 Key 和 Value 的不同特性设计差异化压缩策略来减少 KV Cache 的 hidden dimension 大小。具体实现包括三个核心组件：
  (1) **Fisher Information 引导的压缩率分配**：使用标定数据计算每层的 Fisher Information 分数（继承自 Palu），按层重要性分配不同的压缩率，重要层保留更多 rank。
  (2) **Head-wise Similarity-aware Reordering (HSR) for Keys**：先计算每层所有 attention head 之间的 CKA（Centered Kernel Alignment）相似度矩阵 S ∈ R^{h×h}，贪心地将相似度最高的 head 分组（每组 s=4 heads），然后在每组内分别进行 SVD 低秩分解。Key projection matrix W_k ∈ R^{m×n}（n = h·d_h）被按列分成 h 个 head-wise 子矩阵，每组 s 个 head 拼接后进行 group SVD：W_{g_j} ≈ L_{g_j} R_{g_j}，其中 L_{g_j} ∈ R^{d×r_g}，R_{g_j} ∈ R^{r_g×(d_h·s)}。推理时先计算共享 latent z_{g_j} = x L_{g_j}，再逐 head 重建 [y_{j,1},...,y_{j,s}] = z_{g_j} R_{g_j}。HSR 之后需要 inverse reordering 恢复原始 head 顺序以保证解码等价性。
  (3) **Offline Value Calibration (OVC) for Values**：对 Value projection matrix W_v ∈ R^{m×n} 直接进行全矩阵 SVD 分解：W_v ≈ L_v R_v，然后用标定数据 X 对 L_v 和 R_v 进行闭式校准，最小化近似误差 E = ||L_v R_v X - W_v X||_F^2。校准后通过 Matrix Fusion 将 R_v 融合进 output projection W_o：W̃_o = R_v·W_o，推理时无需显式重建 Value cache。

  实验比较：(a) 与 Palu (G-LRD, group size=4) 对比，在 LLaMA-7B、LLaMA-2-7B、Mistral-7B-Instruct-v0.2、LongChat-7B-v1.5-32k、LLaMA-2-13B-Chat 上评测 50%/60%/70% 三种压缩率下的语言建模困惑度（WikiText2, PTB, C4）和 6 项零样本 QA 准确率（OBQA, HellaSwag, PIQA, ARC-e, ARC-c, Winogrande）；(b) LongBench 长文本 benchmark（8 项任务）下的平均准确率；(c) 集成 KV Cache 量化（4-bit/3-bit per-token quantization + Hadamard transform）后的组合压缩效果；(d) 消融实验：HSR 和 OVC 各自的贡献（80% 压缩率）；(e) 推理效率：Triton 自定义 fused attention kernel 在 4K/16K/65K prompt 下的延迟加速比。

- 硬件平台是什么，配置是什么。
  单张 NVIDIA A800 GPU（80GB 显存）。推理效率评测同样使用 A800。所有实验基于 PyTorch 和 HuggingFace Transformers 实现。标定数据集：从 WikiText-2 随机选取 256 个样本。SVD 前应用 whitening 变换（参考 SVD-LLM 的设置）。

- 模型是什么。数据集和bench分别是什么。
  模型：LLaMA-7B、LLaMA-2-7B、LLaMA-2-13B-Chat（MHA）、Mistral-7B-Instruct-v0.2（GQA）、LongChat-7B-v1.5-32k（MHA, 32K context length）。LLaMA-3.1 结果在补充材料中。
  数据集/Benchmark：(a) 语言建模困惑度：WikiText2、Penn Treebank (PTB)、C4 子集；(b) 零样本 QA 准确率：OBQA、HellaSwag、PIQA、ARC-e、ARC-c、Winogrande；(c) 长文本理解：LongBench（Qasper, QMSum, MultiNews, TREC, TriviaQA, SAMSum, LCC, RepoBench-P 共 8 项任务）；(d) 标定数据：WikiText-2 中 256 个随机样本。

- 开源情况。基于开源文档和论文，使用例子解释，解释算法pipeline，至少具体到伪代码或张量计算。
  代码和模型将在 https://github.com/XIANGLONGYAN/ReCalKV 发布（论文标注 "will be available"）。完整算法伪代码见论文 Algorithm 1，核心流程：

  **Step 1 - Fisher Information 计算与压缩率分配**：
  ```python
  # 使用标定数据 X_calib 计算每层的 Fisher Information
  F = calculate_fisher_info(model, X_calib)  # {layer_idx: fisher_score}
  R = allocate_compression_ratio(model, target_ratio, F)
  # 重要层（高 Fisher）分配更多 rank，不重要层分配更少 rank
  ```

  **Step 2 - HSR: Key Cache 压缩**（per Key projection layer）：
  ```python
  # W_k: Key projection matrix, shape [d, h*d_h], h=32 heads, d_h=head_dim
  # Step 2a: 计算 head 间 CKA 相似度矩阵
  S = zeros(h, h)
  for i, j in range(h), range(h):
      H_i = W_k[:, i*d_h:(i+1)*d_h]  # head i 的投影子矩阵
      H_j = W_k[:, j*d_h:(j+1)*d_h]  # head j 的投影子矩阵
      G_i = H_i @ H_i.T  # Gram matrix
      G_j = H_j @ H_j.T
      # Centering: H_c = I - (1/d)11^T
      G_i_centered = H_c @ G_i @ H_c
      G_j_centered = H_c @ G_j @ H_c
      S[i,j] = Tr(G_i_centered @ G_j_centered) / sqrt(
               Tr(G_i_centered @ G_i_centered) * Tr(G_j_centered @ G_j_centered))

  # Step 2b: Greedy head reordering (group_size=4, 32 heads -> 8 groups)
  groups = [[] for _ in range(8)]
  remaining = set(range(32))
  while remaining:
      i, j = argmax_{i,j in remaining, i<j} S[i,j]  # 最高相似度对
      assign i, j to same group (greedy, 填满 group_size)
      remaining.remove(i); remaining.remove(j)
  # 剩余未分配 head 填入有空位的组

  # Step 2c: Group-wise SVD with whitening
  for group_j in groups:
      W_gj = concat([W_k[:, head*d_h:(head+1)*d_h] for head in group_j])  # [d, s*d_h]
      W_gj_whitened = apply_whitening(W_gj, X_calib)  # 参考 SVD-LLM
      U, Sigma, Vt = SVD(W_gj_whitened)
      r = R[layer_idx]  # 该层分配的 rank (按 Fisher 信息比例)
      L_gj = U[:, :r] @ sqrt(Sigma[:r, :r])  # [d, r]
      R_gj = sqrt(Sigma[:r, :r]) @ Vt[:r, :]  # [r, s*d_h]
  ```

  **Step 3 - OVC: Value Cache 压缩**（per Value projection layer）：
  ```python
  # W_v: Value projection matrix, shape [d, h*d_h]
  # Step 3a: SVD
  U, Sigma, Vt = SVD(W_v_whitened)
  r = R_v[layer_idx]
  L_v = U[:, :r] @ sqrt(Sigma[:r, :r])  # [d, r]
  R_v = sqrt(Sigma[:r, :r]) @ Vt[:r, :]  # [r, h*d_h]

  # Step 3b: Offline calibration on L_v
  # minimize E = ||L_v R_v X - W_v X||_F^2
  # Closed-form: dE/dL_v = 0 =>
  L_v = W_v @ X @ X.T @ R_v.T @ inv(R_v @ X @ X.T @ R_v.T)

  # Step 3c: Offline calibration on R_v
  # dE/dR_v = 0 =>
  R_v = inv(L_v.T @ L_v) @ L_v.T @ W_v

  # Step 3d: Matrix Fusion (offline, no inference overhead)
  W_o_new = R_v @ W_o  # fuse R_v into output projection
  # 推理时: Output = Attention(Q, K, X @ L_v) @ W_o_new
  # 无需重建 X @ L_v @ R_v，直接使用 fused output projection
  ```

  **Step 4 - 推理时**：
  ```python
  # Key 路径（每 token, 有 HSR 在线重排开销）:
  z_gj = x @ L_k_gj              # 共享 latent, [1, r_k]
  # 每 head 独立重建:
  [y_j1, y_j2, y_j3, y_j4] = z_gj @ R_k_gj  # [1, 4*d_h]
  # inverse reordering 恢复原始 head 顺序
  # 应用 RoPE 位置编码

  # Value 路径（每 token, 无重建开销 — Matrix Fusion 已消除）:
  z_v = x @ L_v                 # [1, r_v], 存入 KV cache
  # ... attention computation ...
  output = softmax(QK^T/sqrt(d)) @ (z_v) @ W_o_fused  # W_o_fused = R_v @ W_o
  ```
  压缩比 = r/n（Key）或 r/n（Value），50% KV cache 压缩比意味着 KV cache 总大小减半。
