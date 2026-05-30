## Rectified Sparse Attention

- baseline方法是什么？
  Baseline 是直接使用 block-sparse attention 进行全长度稀疏解码（如 Quest [23]、ClusterKV [18]、InfLLM [24] 等 query-aware training-free sparse attention 方法）。核心缺陷：

  **KV Cache 误差累积**：Sparse decoding 每一步都基于近似 attention 计算，产生的预测 token 及其 KV cache 条目包含近似误差。这些误差随 decoding 步数累积在 KV cache 中，导致后续 attention 计算基于越来越不准确的 KV cache，形成"误差累积"恶性循环。如图 1 所示，sparse decoding 性能随 decoding length 增长持续下降。Quest 尝试通过跳过前两层的策略缓解，但效果有限（Table 1 中 Sparse_dense2 vs Sparse 差异不显著）。

  全栈执行例子（Quest-style Sparse Decoding, Qwen2.5 7B, p=0.9, A100）：
  - **算法层**：prefill 后用 dense attention 构建初始 KV cache。Decode 阶段每一步用 query-aware block-sparse attention 近似 full attention——将 KV cache 划分为 block，用 min/max 描述符做近似匹配，每步选择 top-n block attended。新生成的 token 追加到 KV cache。全过程无 rectification，KV cache 的误差随步数单调递增。
  - **系统框架层**：基于 PyTorch 实现，无特殊框架修改。可与 vLLM 等 serving 框架集成。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用 Flash Decoding kernel 进行 sparse attention 的 split-execution。无 rectification 操作，kernel 专注 sparse attention computation。
  - **硬件架构层**：NVIDIA A100-80G GPU。Sparse attention 减少 HBM 访问，但 KV cache 质量退化导致 math reasoning 准确率下降（Qwen2.5 7B avg: Dense 60.72, Sparse 57.72, gap=3.0）。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ReSA 通过引入周期性 dense rectification 直接解决 KV cache 误差累积问题，核心设计如下：

  **(1) 算法层——Group Block Sparse Attention + Dense Rectification**：

  **GBSA**：继承 Quest 的 block-sparse attention（block 描述符、query-dependent top-n selection），新增 GQA group 内共享 attention pattern（来自 NSA 的 shared grouping），减少 block selection 开销。因为同一个 GQA group 的 query heads 共享 KV，复用同一组 block indices 避免了为每个 head 单独做 block selection 的重复计算和 memory access。

  **Dense Rectification**：每 f=32 个 decode step 后，将这 f 个最近生成的 token 批量通过 dense attention 并行重编码，刷新 KV cache 和 block key cache。这从根本上约束了稀疏误差的累积范围——每 f 步后 KV cache 被"校正"到 dense 精度，误差窗口被限制在 f 以内（而非随时间无限累积）。

  **Decoding Procedure**（Algorithm 1）：
  ```
  Prefill(P) → dense KV cache K
  for i in 1..T:
      t = SparseForward(G[i-1], K, B)  # 快速稀疏生成
      append t, update K and B
      if i % f == 0:
          K, B = DenseForward(G[i-f:i], K, B)  # 批量 dense rectification
  ```

  - vs Baseline 的无界误差累积：ReSA 将误差窗口限制在 f 以内。即使 f=128 仍保留大部分性能增益（Fig 9），表明 rectification 对频率不敏感，鲁棒性高。
  - vs Quest 的跳过前两层：ReSA 在所有层都应用 sparse attention + rectification，无需特殊处理特定层。实验显示 Sparse_dense2（前两层 dense）改善不显著，而 ReSA 显著改善。
  - vs Self-Speculation：ReSA 无需 per-token accept/reject 决策和 resampling，比 sparse KV-based self-speculation 平均快 1.92×（Table 3）。

  **(2) Memory Access 模型**：
  公式 Avg(mem) = mem(KV cache) × (1/b + p + 1/f)，显式地量化了 block 粒度 b、sparsity ratio p、rectification frequency f 对 memory access 的影响，提供了理论加速上界。

  **(3) Kernel调度层——Flash Decoding + Block-Sparse Kernel**：
  Custom kernel 采用 GQA-aware SM 分配和 block-level workload splitting，在每个 SM 上独立 fetch 其负责的 block subset 并执行 sparse attention。关键优化：
  - Sparse attention loop 仅遍历 selected blocks（由 GBSA block selection 产生），而非全部 KV blocks。
  - Intra-GQA key 共享减少 HBM 访问：同一 group 的 query heads 复用加载的 KV 数据。
  - Block key cache 在线增量更新，新 token 追加时仅更新对应 block 的 min/max 描述符，O(1) per token。

  实验表明 sparse estimation 和 attention computation 耗时相当（均 ≈ mem(KV cache) × 0.9），这是 kernel 效率设计的关键平衡点。

  **(4) 系统框架兼容性**：
  Rectification 天然兼容 continuous batching 和 chunked prefill（如 Sarathi、DeepSpeed-FastGen），仅需周期性批量重编码，无需引入特殊同步屏障。

  全栈执行例子（ReSA, Qwen2.5 7B, p=0.9, f=32, A100-80G）：
  - **算法层**：Prefill (dense) → decode (GBSA, p=0.9) × 32 steps → rectification (dense forward over last 32 tokens) → decode × 32 → ... 循环。Math reasoning avg accuracy: 60.52 vs Dense 60.72 (gap=0.20)，近乎无损。
  - **系统框架层**：基于 PyTorch，可通过 continuous batching 实现 rectification 的批处理。256K context 下 INT4 end-to-end 2.44× speedup。
  - **编译框架层**：TileLang 库辅助实现 GBSA kernel（Acknowledgments 中致谢）。
  - **kernel调度层**：Custom Flash Decoding kernel with block-sparse support。256K 下 rectification overhead 仅 32.7%，延迟随序列长度接近线性而非二次。
  - **硬件架构层**：NVIDIA A100-80G GPU。Sparse + 量化 (INT4 Marlin) 正交组合，256K 达 2.44× speedup。

- baseline方法是什么？
  Baseline 是 Palu（Chang et al., 2024）的 G-LRD（Grouped Low-Rank Decomposition）变体，即对 KV projection 矩阵直接做 group-wise SVD 低秩分解来压缩 KV cache hidden dimension。Palu 的核心缺陷：
  (a) **忽略 Key-Value 不对称性**：Palu 对 Key 和 Value 使用相同的 SVD 压缩策略，未区分 Keys 承载位置编码（RoPE）信息需重建 → 有额外计算开销，Values 承载语义信息 Fisher Information 显著更高 → 需更高保真度保留。
  (b) **Head 分组随机**：Palu 的 group-wise SVD 将相邻 head 按物理索引简单分组（无相似性考量），不同 head 的 left singular subspace 差异大 → 组内 SVD 近似误差高。
  (c) **Value SVD 无校准**：标准 SVD 分解未优化重建误差 E = ||L_v R_v X - W_v X||_F^2，在高压缩率下精度下降显著。Fisher Information 分析显示 Value projection 矩阵的 Fisher Information 显著高于 Key projection，表明 Value 对模型行为更关键，简单的 SVD 截断会引入较大性能退化。

  全栈执行例子（Palu G-LRD, LLaMA-2-7B, MHA, 70% 压缩率, A800）：
  - **算法层**：对 Key 和 Value projection 按物理相邻 4 个 head 分组，每组做 SVD 低秩分解（group_size=4），压缩后 KV cache hidden dim 减至 30% 原始大小。Key 和 Value 使用相同的压缩率和相同的 group-wise SVD 策略，未区分两者不同特性。70% 压缩率下 WikiText2 困惑度从 5.47 升至 8.62。
  - **系统框架层**：基于 PyTorch + HuggingFace Transformers，在模型加载后 offline 修改 Key/Value projection 权重矩阵，推理时无额外框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：使用标准 FlashAttention kernel，不对低秩 Key/Value 路径做 kernel 级融合优化。Key 重建和 Value 重建在 kernel 外部完成。
  - **硬件架构层**：NVIDIA A800 GPU。KV cache 内存占用减少但无针对性 kernel 优化。

- 论文方法是什么？如何对应解决Baseline的缺陷？
  ReCalKV 通过分析 Key 和 Value 在注意力机制中的不同角色，分别设计了 HSR（针对 Key）和 OVC（针对 Value）两种差异化压缩策略：

  **(1) 算法层——差异化 Key-Value 压缩策略**：

  **HSR (Keys)**：利用 CKA 相似度衡量 head 的表征子空间相似性，将相似度高的 head 通过贪心算法分为一组（而非简单按物理索引分组），组内做 group SVD。相似的 head 共享更多表征成分 → SVD 低秩近似误差更低。推理时需执行在线 inverse reordering 恢复原始 head 顺序。
  - vs Palu 的随机分组：HSR 将 CKA 相似度最高的 head 聚为一组，lower approximation error from shared left singular subspace。
  - 消融实验（80% 压缩率）：HSR alone 将 WikiText2 困惑度从 9.34 降至 9.01，LongBench Avg Acc 从 9.01% 升至 12.44%。

  **OVC (Values)**：先对 Value projection 做 SVD，再用标定数据 X 按闭式解校准 L_v 和 R_v，最小化重建误差 E。校准后 R_v 通过 Matrix Fusion 融合进 W_o 中，推理时无需在线重建 Value。
  - vs Palu 的未校准 SVD：OVC 的闭式校准直接最小化 ||L_v R_v X - W_v X||_F^2，比标准 SVD 截断更精确。
  - 消融实验（80% 压缩率）：OVC alone 将 WikiText2 困惑度从 9.34 降至 8.91，LongBench Avg Acc 从 9.01% 升至 13.09%。
  - HSR + OVC 联合：WikiText2 困惑度降至 8.48，LongBench 升至 15.40%。

  **(2) Fisher Information 引导的压缩率分配**：
  借鉴 Palu 的 Fisher Information 策略，按每层的重要性分配不同的压缩 rank。高 Fisher 层保留更多 rank，低 Fisher 层可更激进压缩。这确保关键层（如承载长程依赖的中间层）的近似质量优先保证。

  **(3) Kernel调度层——Triton Fused Attention Kernel**：
  自定义 Triton fused attention kernel 将 HSR 的在线 head permutation 和 OVC 的 Matrix Fusion 整合到单一 kernel 执行路径中。Key 路径：X·L_k → K_latent → 重建 → inverse reorder → RoPE → attention scores。Value 路径：X·L_v → V_latent（存入 cache）→ fused output。Kernel 融合避免中间结果的 HBM 往返，70% 压缩率下 65K prompt 达到 1.80× 加速。

  **(4) 正交兼容性**：
  ReCalKV 与量化技术正交——可与 4-bit/3-bit per-token quantization + Hadamard transform 组合进一步压缩（Section 4.4），70% low-rank + 3-bit quant 仍维持 7.01 WikiText2 困惑度。

  全栈执行例子（ReCalKV, LLaMA-2-7B, MHA, 70% 压缩率, A800）：
  - **算法层**：Fisher 分配 rank → Key 做 HSR (CKA → greedy grouping → group SVD) → Value 做 OVC (SVD → closed-form L_v/R_v calibration on 256 WikiText2 samples → Matrix Fusion)。70% 压缩率下 WikiText2 困惑度 6.75（vs Palu 8.62），零样本 QA 平均准确率 59.90%（vs Palu 52.14%）。
  - **系统框架层**：基于 PyTorch + HuggingFace Transformers，offline 修改模型权重。无额外 serving 框架修改。
  - **编译框架层**：论文未明确说明。
  - **kernel调度层**：Triton fused attention kernel，集成 HSR 在线置换和离线 Matrix Fusion。70% + 65K 加速 1.80×。测试在 A800 上进行。
  - **硬件架构层**：NVIDIA A800 GPU (80GB)。低秩 KV cache 减少 HBM 占用和带宽压力，长 prompt 下效果更显著。
