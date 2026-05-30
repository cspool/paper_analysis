## Group of Pictures (GOP) in Visual Representation Learning

术语是什么？
GOP 是视频编解码标准中的基本编码单元：一个 I-frame（独立编码的完整帧）+ 若干 P/B-frame（依赖参考帧的预测帧）。OV-Encoder 将 GOP 引入视觉表示学习：每个 GOP 的 I-frame 全量编码（256 patches 建立空间上下文），P-frames 仅保留运动+残差显著 patches（3.1%-25%）。GOP 的关键作用不是编解码效率，而是结构化时空分解：I-frame 提供"what is where" 空间锚点，P-frame 提供"what changed" 时序更新。

从算法pipeline角度拆解术语：
OV-Encoder 配置：64帧, GOP=32 → 2个GOP。每个 GOP: 1 I-frame (全256 patches) + 31 P-frames (仅选显著)。Token budget 2048 = 2×256 (I-frames) + 1536 (P-frames top-K)。关键：budget 跨越 GOP 全局排序（非 per-GOP），确保最优分配。GOP size 超参数：太小 → I-frame 频繁刷新占 token；太大 → 长时间无空间上下文刷新，累积预测误差。

术语一般如何实现？如何使用？
GOP 结构在 HEVC 解码时自然产生（编解码器原生支持），不需额外处理。解码器直接按 GOP 输出 I/P 帧及其 motion vectors/residuals。训练和推理使用相同 GOP 配置。限制：非 HEVC 视频需先转码。

涉及论文标题：
- OneVision-Encoder__Codec-Aligned_Sparsity_as_a_Foundational_Principle_for_Multimodal_Intelligence
