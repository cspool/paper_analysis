## Focus: A Streaming Concentration Architecture for Efficient Vision-Language Models

- baseline方法是什么？
  Baseline是两类：**算法baseline** (FrameFusion token merging、AdapTiV/CMC token pruning) 和**硬件baseline** (vanilla systolic array、AdapTiV accelerator、CMC accelerator、GPU+FrameFusion)。

  **算法Baseline缺陷分析**：
  - FrameFusion [20]：merge temporally redundant tokens跨帧，但产生不规则稀疏模式，GPU Tensor Core难以高效利用，且runtime overhead增加up to 36.8%
  - AdapTiV [70]：intra-frame token-level similarity，仅支持静态图像，忽略视频language interaction；用sign-bit做轻量相似度检查，细粒度不足
  - CMC [56]：video-codec-inspired inter-frame redundancy search，利用H.264类压缩，但忽略language inputs；global token-wise操作，需等完整token输出写回DRAM后再压缩，带来高带宽和差locality。CMC虽有46% sparsity但仍有79% dense DRAM traffic
  - 三者共同缺陷：都仅关注visual redundancy（ViT导向），忽略cross-modal semantic intent（prompt对token重要性的影响）；都做token-level coarse granularity操作，无法捕获motion引起的sub-token partial alignment

  **全栈执行例子（CMC baseline, LLaVA-OneVision-7B在VideoMME上推理）**：
  - 算法层：CMC做inter-frame token-level similarity search → 46% sparsity但accuracy 62.11 vs original 63.32；AdapTiV做intra-frame token merging → 39.55% sparsity但accuracy 62.22；FrameFusion merge temporally redundant tokens → 固定70% sparsity但accuracy降至62.54
  - 系统框架层：CMC/adapTiV算法在PyTorch中实现 → GPU上运行 → token pruning/merging结果转换为不规则稀疏mask → 难以映射到高效batch GEMM
  - 编译框架层：论文未明确说明
  - kernel调度层：CMC hardware accelerator用global token-wise方法：systolic array输出完整M×N token矩阵→全部写回DRAM→外部codec unit读回DRAM做token-wise similarity search→压缩结果写回DRAM→下次GEMM前读回。Global执行阻止fine-grained scheduling，增加memory pressure。CMC需1.4MB额外buffer
  - 硬件架构层：Vanilla systolic array (32×32 PE, FP16, weight stationary) → 无compression → 全量tokens参与GEMM → 高DRAM traffic。AdapTiV accelerator集成token merging但still coarse token-pair操作，transfer uncompressed tokens before processing

- 论文方法是什么？如何对应解决Baseline的缺陷？
  提出Focus，通过**Multilevel Streaming Concentration**和**hardware-algorithm co-design**解决baseline四大缺陷：

  **缺陷1：忽略cross-modal semantic intent → static token importance metrics不准**
  → **Focus方案**：SEC从attention SoftMax(QK^T)中提取text-to-image (T×M) cross-modal attention block，计算每image token从所有text tokens和heads接收到的最大attention score。prompt问"dog"时attention集中在狗，问"flower color"时转向花位置。SEC基于实际cross-modal attention做prompt-aware pruning，不再依赖静态magnitude/saliency。

  **缺陷2：token-level coarse granularity → 无法捕获sub-token redundancy和motion-caused partial alignment**
  → **Focus方案**：SIC将token embedding分为32-dim vectors做vector-wise cosine similarity matching。2×2×2时空block内localized comparison（8 vectors per block）。64%的8-dim vectors超过cosine similarity 0.9（vs仅18%的3584-dim full tokens），实现82.8% sparsity（vs token-wise 73.0%）。Vector-level匹配允许每个vector匹配多个candidates，捕获richer sub-token similarity。

  **缺陷3：global token-wise compression → high DRAM traffic, poor locality, non-streamable**
  → **Focus方案**：所有compression on-chip、tile-local、streaming。SIC不等整层token就绪，每个GEMM m×n tile产生后立即压缩（vector-level similarity matching + deduplication），仅deduplicated vectors和similarity map写回DRAM。SEC top-k sorter与image attention GEMM完全重叠，不占用critical path。CMC 46% sparsity → 79% dense DRAM traffic vs Focus ~81% sparsity → 21% bandwidth。

  **缺陷4：irregular sparsity不利于硬件利用 → GPU/accelerator无法高效执行**
  → **Focus方案**：将算法稀疏性转化为hardware-friendly structured/tile-local稀疏。GEMM tile对齐（m=1024, n=32）、convolution-style layouter的conflict-free bank mapping（Bank=f%2×4+r%2×2+c%2）支持2×2×2 block内8 vectors无conflict并行读取。Similarity Scatter用2a-wide accumulator做concurrent reconstruction。SpMM-like execution：GEMM对compact vectors (p<1024)执行，按similarity map scatter partial sums恢复full output。Focus Unit面积仅3.21 mm²（2.7% of SA），功耗736 mW（0.9% of SA）。

  论文方法全栈执行例子（以LLaVA-Video-7B在VideoMME上推理为例）：
  - 算法层（核心创新）：Multilevel Concentration。Layer 3 attention: SEC提取text-to-image attention→importance vector→top-k sorter选40% tokens (约2500/6272)→保留tokens继续P(i)×V。Layer 9: SEC选30%。Layer 18: SEC选20%。Layer 26: SEC选10%。所有FC layers: SIC在每个GEMM tile后做2×2×2 block cosine similarity matching (threshold=0.9)→deduplicate vectors→similarity map记录映射。Scatter恢复full output→下一tile的Gather再压缩。最终avg sparsity 82.82%，accuracy 62.74 vs original 64.15 (-1.41%)
  - 系统框架层：Focus Unit作为modular component嵌入systolic-array accelerator memory interface → 拦截GEMM output和attention SoftMax → 不修改core compute pipeline。SEC在attention layer集成 → SIC在FC/O projection/PV GEMM层集成。Similar to pooling/activation function —— modular, scalable
  - 编译框架层：论文未明确说明
  - kernel调度层（核心创新）：Streaming dataflow。t=1: Attention SoftMax → SEC importance analyzer (parallel max units) → a-way bubble sorter与image attention GEMM (Q(i)K^T)重叠→sorting在attention GEMM完成前结束。t=2: SEC pruning + offset encoding → P(i)×V仅加载保留tokens。t=3: FC GEMM tile (1024×32) → convolution-style layouter重组为FHW布局 → SIC similarity matcher (7 pairwise cosine per key, 8×m cycles max vs GEMM 112×m cycles) → deduplicated vectors + similarity map写回DRAM。t=4: 下一层GEMM对p个compact vectors计算 → Similarity Scatter根据similarity map复制partial sums到full output positions → 2a-wide accumulator并发累加 → tile完成后Similarity Gather再次压缩。Scatter-Gather循环贯穿所有FC层
  - 硬件架构层（核心创新）：Focus Unit (SEC+SIC) integrated near systolic array memory interface。SEC: importance analyzer (并行max units + on-chip 25KB buffer) + a-way bubble sorter + offset encoder (lightweight registers)。SIC: convolution-style layouter (16KB buffer for 256-vector window, conflict-free bank mapping) + similarity matcher (dot-product unit + L2-norm precompute buffer, <1% SA area) + similarity map buffer + scatter accumulator (2a-wide=64)。Total on-chip buffer 734KB。TSMC 28nm: 3.21 mm², 736 mW —— 4.47× speedup vs SA, 7.90× vs GPU, DRAM traffic 4.9× reduction vs dense SA
