## xKV: Cross-Layer SVD for KV-Cache Compression

- baseline方法是什么？
  Baseline 分为两类：(1) **MiniCache**：利用相邻层 KV-Cache 的 token-wise cosine similarity 假设，通过 SLERP 合并相邻层的 KV 对来实现跨层压缩。执行流程：从 Transformer 中间层到末尾层，对半数的相邻层对执行 SLERP merging；每对合并后共享一份 KV-Cache。(2) **Single SVD (per-layer SVD)**：对每一层的 KV-Cache 独立执行 SVD 分解，保留 top-r 奇异值/向量，压缩每层的 KV-Cache，但未利用跨层冗余。

  **全栈执行例子（Baseline: MiniCache）**：
  - 算法层：计算相邻层 KV-Cache 的 token-wise cosine similarity → SLERP 插值合并 → 用合并后 KV 替换原两个 layer 的 cache
  - Serving层：HuggingFace 推理，prefill 后执行合并，decode 阶段直接使用合并后的 KV-Cache（论文未说明修改特定 Serving 框架）
  - 编译框架：未涉及
  - Kernel调度：未涉及
  - 硬件架构：未涉及

  **Baseline 痛点**：MiniCache 依赖 token-wise cosine similarity 假设，但实际中相邻层 embedding 的 token 级相似度很低（Figure 2a），导致高压缩比下准确率急剧下降（Qwen2.5-7B 上 1.3× 压缩就崩到 avg 5.7%）；Single SVD 到 8× 压缩时同样发生灾难性性能退化（Llama-3.1-8B 上 avg 仅 35.3%）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  xKV 发现：尽管 token-wise cosine similarity 很小，但相邻层 KV-Cache 的**主导奇异向量（dominant left singular vectors）**高度对齐（通过 CKA 度量验证，Figure 2b）。基于此，xKV 将多层 KV-Cache 水平拼接后做一次统一的跨层 SVD，提取共享的 left singular vectors（共享基 A），各层只保留独立的 reconstruction matrix（B_ℓ_i）。

  **全栈执行例子（xKV）**：
  - 算法层：Stride-based 分组（相邻 G 层一组）→ 每层 pre-RoPE key/value states 水平拼接 → 对拼接矩阵做 SVD → 保留共享基 A (= U_r @ S_r) 和层独立 B_ℓ_i（Vt_r 分块）→ decode 时 A @ B_ℓ_i 重构 → 对重构 key 重新施加 RoPE
  - Serving层：HuggingFace 推理实现，prefill 阶段在线 SVD 分解（<10% prefill time at 128K），decode 阶段逐 token 重构并查询（论文未说明修改特定 Serving 框架）
  - 编译框架：未涉及
  - Kernel调度：未涉及（论文未实现 custom kernel，仅用 PyTorch/HuggingFace 原生算子）
  - 硬件架构：未涉及

  **关键设计 vs Baseline 缺陷映射**：
  1. **MiniCache 的 token-wise cosine 假设不成立** → xKV 改用 CKA 发现跨层奇异向量对齐，用 SVD 共享子空间替代 token 级合并
  2. **Single SVD 仅利用单层低秩性** → xKV 通过跨层拼接 SVD 利用多层共享基，相同 rank 下保留更多信息（Figure 2c: 层越多所需相对 rank 越低）
  3. **跨层合并受限于 pairwise 操作（2 层一组）** → xKV 的 stride-based grouping 支持任意组大小 G（论文验证了 G=2,4），组越大共享子空间越丰富
  4. **离线统计无法适配不同上下文** → xKV 采用在线 SVD（per-request），捕捉上下文动态变化
  5. **MLA 架构压缩困难** → xKV 直接对 MLA latent representations 做跨层 SVD（non-RoPE 部分），在已压缩的 latent cache 上再获得 3× 压缩
