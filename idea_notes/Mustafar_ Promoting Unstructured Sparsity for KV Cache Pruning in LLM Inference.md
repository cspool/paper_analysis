## Mustafar: Promoting Unstructured Sparsity for KV Cache Pruning in LLM Inference

- baseline方法是什么？
  Baseline 是 ThinK [44] 结构化剪枝（per-channel structured pruning of KV cache）。ThinK 以整个 channel 为粒度剪枝 Key cache（per-channel, output-aware scoring），只剪枝 Key cache 无法有效剪枝 Value cache（Value cache 元素分布均匀，无显著 channel-wise outliers）。ThinK 报告的 Value cache 剪枝上限仅为 30% 稀疏度。此外，ThinK 结构化剪枝的稀疏 pattern 受限于 channel 对齐，导致大量冗余元素被迫保留，且 GPU 上无法将 channel-wise 剪枝直接转换为内存带宽节省（需要实际减少矩阵维度）。

  全栈执行例子（ThinK baseline, Llama-3-8B-Instruct, T=4096, RTX 6000 Ada）：

  - **算法层**：Key cache 剪枝：对每个 channel c，计算 S_c = Σ_{t} |Q_t| · |K[:, c]|（最近32 Q 的 L1 累加 × channel K），保留 top-k channels。Value cache 不剪枝（或仅 30% 稀疏度）。结构化剪枝后 Key cache 矩阵维度从 R^{T×d} 降为 R^{T×d'}（d' = d×(1-s)），仍为稠密矩阵——本质上仍是 dense 矩阵乘法。核心缺陷：(1) Value cache 几乎无法剪枝，KV cache 总体压缩率受限于 Key-only 剪枝；(2) channel-wise 剪枝忽略 token 内不同元素的差异，一个 channel 整体被保留/丢掉，粒度太粗；(3) 结构化稀疏的限制：即使剪枝 70% channels，dense 矩阵仍需整体加载进行计算。

  - **Serving/框架层**：使用 HuggingFace Transformers 推理。论文未修改 serving 框架调度。ThinK 在 attention 计算前对 KV cache 执行 channel-wise 选择，不影响计算 graph 其余部分。

  - **kernel调度层**：标准 PyTorch/cuBLAS batch GEMV 或 FlashAttention decode kernel。ThinK 减少 K 的 channel 维度后，QK^T 计算仍是标准 dense matmul（维度减小但仍是稠密计算），无自定义 kernel。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 核心缺陷总结：
  1. Value cache 剪枝困难——均匀分布无 channel outliers，结构化剪枝仅 30% 稀疏度上限
  2. Channel-wise 粒度过粗——整 channel 剪枝丢弃了 channel 内有价值的元素，同时被迫保留冗余元素
  3. 结构化稀疏无法直接转换为 GPU memory bandwidth 节省——剪枝后仍以 dense matmul 计算
  4. 需要 output-awareness 计算额外 pruning score 开销

- 论文方法是什么？如何对应解决Baseline的缺陷？
  Mustafar 提出 per-token magnitude-based unstructured pruning + bitmap sparse format + custom CUDA SpMV kernel，实现端到端的非结构化 KV cache 剪枝加速。

  Mustafar 全栈执行例子（Llama-3-8B-Instruct, T=4096, K_s=0.5, V_s=0.5, RTX 6000 Ada）：

  - **算法层**：
    1. **Per-token magnitude-based pruning**：对每个 token 的 KV vector 独立按元素绝对值排序，保留 top-(1-s) 元素。Key cache 受益于 outlier channels（高 magnitude 元素集中在特定 channel），Value cache 虽分布均匀但 per-token 按 magnitude 剪枝等价于 per-token output-aware 剪枝（因 attention 中 V 每个元素乘以同一个 attention score）。
    2. **Key cache 结论**：无结构约束的非结构化剪枝在 70% 稀疏度下精度优于 ThinK 50% 结构化剪枝。output-awareness 带来微小提升但 magnitude-only 已足够。
    3. **Value cache 结论**：per-token magnitude-based 在 70% 稀疏度保持精度，远超 ThinK 的 30% 结构化上限。per-channel output-aware 可达到近似精度但需额外重算 attention scores（FlashAttention 不物化完整 attention matrix）。
    4. **Local dense window**：最近 32 token 保留稠密不剪枝，确保近期上下文的完整 attention 质量。
    5. **模块化兼容**：per-token 粒度允许与 token eviction (H2O) 无缝整合（evict token 后，剩余 token 各自独立剪枝）；与 KV cache quantization (KIVI) 叠加（先 prune 再 quantize）。

  - **kernel调度层**：
    1. **Triton 压缩 kernel**：GPU 并行将稀疏 KV cache 实时压缩为 bitmap-based 格式（每 1×64 tile 一个 bitmap + nonzeros）。
    2. **Custom CUDA SpMV kernel**：基于 Coruscant 的 bitmap sparse format，采用 FlashLLM 的 load-as-compressed, compute-as-dense 范式——在 GPU SM 上完成 compressed→register→shared memory decompress→Tensor Core dense compute pipeline。Memory-bound decode attention 中 global memory 数据搬移量大幅减少（仅加载非零元素 + bitmap + offset，而非 full dense tile）。
    3. **Decode attention 重新分拆**：SpMV 处理压缩历史 KV cache + dense MV 处理 local window，两部分结果 concat 后 softmax 再分别加权求和。
    4. **KV cache tile 管理**：Key cache column-tile 沿 token 维度，Value cache column-tile 沿 channel 维度；channel-major 遍历确保新 token 压缩数据可尾部追加。

  - **Serving/框架层**：基于 PyTorch + Triton + CUDA 实现，作为可插拔 attention 后端。Prefill 使用 FlashAttention，Decode 使用 Mustafar kernel。未修改 serving 调度逻辑。

  - **编译框架层/硬件架构层/芯片设计层**：论文未明确说明。

  Baseline 缺陷 → Mustafar 设计映射：
  1. **Value cache 无法剪枝（ThinK 仅 30%）→ Per-token unstructured pruning（70% 保持精度）**：非结构化剪枝解除 channel 对齐约束，允许在每个 token 内独立选择最佳元素保留。即使 Value cache 分布均匀，per-token magnitude 等价于 output-aware，因每个 value 元素在 attention output 中的贡献正比于其 magnitude。
  2. **Channel-wise 粒度过粗 → Element-wise per-token pruning**：以元素而非 channel 为剪枝单位，实现真正细粒度选择。实验证明在同等 70% 稀疏度下（减少 70% 元素），非结构化剪枝精度（LongBench avg 41.55-42.84）远超结构化剪枝（26.55-38.53）。
  3. **结构化稀疏无法转换为 bandwidth 节省 → Bitmap compressed format + custom SpMV kernel**：将无规则稀疏 pattern 压缩为 tile-wise bitmap 表示，SpMV kernel 以压缩格式加载、解压后 dense 计算，memory-bound attention 中 HBM 数据搬运量减少（50% sparsity: 65% compression ratio, 70% sparsity: 45% compression ratio）。
  4. **Output-awareness 计算开销 → Magnitude-only 剪枝避免额外计算**：Key cache 无需 output-awareness 即可达到 competitive 精度；Value cache per-token magnitude 天然等于 output-aware。避免了 ThinK 的 per-channel output-aware score 计算和 attention score 重算。
  5. **与正交方法兼容性差 → Per-token granularity 的模块化设计**：与 token eviction (H2O) 和 quantization (KIVI) 无缝叠加，允许不同程度的联合压缩（如 H2O 20% budget + Mustafar 50% sparsity）。
  6. **Batch=1 下 GPU 利用率不足**：论文明确指出的当前限制——小 batch 下 SpMV kernel 的 threadblock 数少于 SM 数量导致 SM underutilization。在 batch≥4 时性能优势显著（batch=8 时 2.23× throughput）。
