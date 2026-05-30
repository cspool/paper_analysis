## xKV: Cross-Layer SVD for KV-Cache Compression

- 属于算法pipeline的实现是什么？实验比较什么？
  提出 xKV，一种无需训练的 plug-and-play KV-Cache 压缩方法：将多个相邻层的 KV-Cache 水平拼接后执行跨层 SVD，提取共享的 left singular vectors（共享基 A）和各层独立的 reconstruction matrices（B_ℓ_i），从而将多层 KV-Cache 压缩到同一个低秩子空间中。实验比较：vs MiniCache（inter-layer SLERP merging）、vs Single SVD（per-layer SVD）。在 RULER（64K context）上评估 Llama-3.1-8B-Instruct、Qwen2.5-7B/14B-Instruct-1M；在 RepoBench-P 和 LCC 上评估 DeepSeek-Coder-V2-Lite-Instruct（MLA 架构）。指标：各子任务 accuracy 及 compression rate。xKV 在 8× 压缩比下比 MiniCache 达 6.8× 更高压缩率且 accuracy 提升 2.7%；在 MLA 架构上进一步实现 3× 压缩。

- 硬件平台是什么，配置是什么。
  论文仅提及使用 HuggingFace 实现，未明确说明 GPU 型号或硬件配置。prefill 阶段在线 SVD 开销在 128K context 下 <10% of prefill time。

- 模型是什么。数据集和bench分别是什么。
  模型：Llama-3.1-8B-Instruct (8 KV heads, GQA)、Qwen2.5-7B-Instruct-1M (4 KV heads)、Qwen2.5-14B-Instruct-1M (8 KV heads)、DeepSeek-Coder-V2-Lite-Instruct (16B MoE, 2.4B activated, MLA)。
  数据集/Benchmark：RULER（subtasks: NIAH-S1/S2/MK1/MK2/MQ/MV, QA-1/2, VT, FWE）、RepoBench-P、LCC（from LongBench）。

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  已开源：https://github.com/abdelfattah-lab/xKV

  **算法 Pipeline (伪代码)**：
  ```python
  # 跨层 SVD 压缩
  # 输入: 一组层的 KV-Cache [X_l1, X_l2, ..., X_l|G|] 各 ∈ R^{L×d}
  # G: 组大小(如2或4), L: 序列长度, d: 隐藏维度

  # 1. 水平拼接
  X_cat = concat_horizontal([X_l for l in group])  # shape: [L, |G| * d]

  # 2. SVD 分解, 保留前 r 个秩
  U, S, Vt = SVD(X_cat)
  U_r = U[:, :r]        # [L, r]
  S_r = S[:r, :r]       # [r, r]
  Vt_r = Vt[:r, :]      # [r, |G| * d]

  # 3. 矩阵融合: A = U_r @ S_r, B_li = Vt_r 的对应列块
  A = U_r @ S_r                      # 共享基, [L, r]
  B = [Vt_r[:, i*d:(i+1)*d] for i in range(|G|)]  # 各层重构矩阵, [r, d]

  # 4. 存储: A + {B_li} for each group
  # 原始存储: |G| * L * d
  # 压缩后: L * r + |G| * r * d
  # 压缩率 ≈ (|G| * L * d) / (L * r + |G| * r * d)
  # 当 L >> r*d 时近似于 L/r

  # 5. Decode 重构
  # X_l_i ≈ A @ B_li = [U_r @ S_r] @ B_li
  ```

  **分组策略**：Stride-based contiguous grouping（N 层分 N/G 组，每组 G 层相邻）。
  **Key/Value 差异处理**：keys 和 values 压缩敏感度不同，固定 rank ratio key:value = 1:1.5（如 key rank=96, value rank=144）。
  **RoPE 处理**：对 pre-RoPE key states 执行 SVD，decode 时重新应用 RoPE。
  **MLA 处理**：对 non-RoPE latent representations 应用 xKV，解耦的 RoPE keys 不压缩。
  **在线分解**：prefill 阶段按请求在线执行 SVD（非离线统计），更好捕捉上下文动态。
