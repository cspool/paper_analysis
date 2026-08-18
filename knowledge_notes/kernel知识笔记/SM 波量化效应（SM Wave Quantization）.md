## SM 波量化效应（SM Wave Quantization）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SM wave quantization（SM 波量化效应，NVIDIA 官方文档 [1] 的 wave quantization 概念）指 GPU 上 SM（Streaming Multiprocessor）数量有限导致的调度粒度量化：一个 GEMM 的 thread block（CTA）数不是 SM 数整数倍时，最后一"波"（wave）的 SM 利用率不足，出现部分 SM 空闲的尾波效应。tile 越小、CTA 数越少，波量化损失占比越高——小 GEMM 场景尤其明显。
- 在 LoKA 中的角色：BlockNorm 设计时考虑"小 batch 大输出维"情形——为让单 thread block 装下整行做归一化统计，可把 batch 维 tile 缩小，但这会加剧 SM wave quantization（CTA 数下降、尾波空闲）并降低 W 矩阵的 L2 缓存命中率（W tile 复用减少），端到端加速比趋近于零——这正是论文选择"固定块 BlockNorm 放松数学等价性"而非"调 tile 适配归一化"的原因：后者需对每个新 shape 手工调 tile，牺牲通用性。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 量化损失计算：设 GPU 有 S 个 SM，GEMM 按 tile 得 C=ceil(N_blocks/S) 波；利用率 = N_blocks/(C·S)。例：S=132（H100），若 tile 设计使 CTA 数=130 → 2 波但第二波只占 2/132，利用率≈50%；若 CTA=264 → 2 满波，利用率 100%。LoKA 的 Case 2 情景：为容纳整行归一化缩小 batch tile → CTA 数从 264 降到 ~130 → 波量化损失 50%，且每 CTA 的 W 复用变小、L2 命中率下降。
- 与归一化融合的权衡：论文 Fig.7(c) 展示该路径（缩 batch tile）端到端加速比趋零；Fig.7(d) 最终选择 BlockNorm（固定 256 块，不依赖 tile 适配），在任意 shape 下鲁棒。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现/使用：kernel 设计时选择 tile 尺寸使 CTA 数尽量为 SM 数整数倍（或满足最小化波量化）；NVIDIA 性能指南明确该效应（docs.nvidia.com 的 dl-performance-matrix-multiplication wave-quant 一节）；实际框架（cuBLAS/Triton heuristics）自动在 tile 选择中权衡。在 LoKA 场景中它是设计约束而非优化目标：促使归一化设计向"不依赖全局统计/不依赖 tile 适配"的 BlockNorm 收敛。关联概念：tile 尺寸、CTA/SM 调度、L2 缓存命中率。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
