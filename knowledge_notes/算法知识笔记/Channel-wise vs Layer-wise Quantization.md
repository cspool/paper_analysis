## Channel-wise vs Layer-wise Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Channel-wise 为权重张量每个输出通道独立分配量化参数（s_c, z_c）；Layer-wise 为整个层使用单一参数组。Channel-wise 适应不同通道分布差异精度高但存储稍增；Layer-wise 参数少实现简单但精度低。APHQ-ViT 中权重使用 channel-wise 量化，激活使用 layer-wise 量化——精度与效率的标准折中。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Channel-wise (权重): W[C_out, C_in]
for c in range(C_out):
    s_c = (max(W[c]) - min(W[c])) / (2^N - 1)
    W_int[c] = round(clip(W[c]) / s_c)

# Layer-wise (激活): X[B, C, H, W]
s = (max(X) - min(X)) / (2^N - 1)
X_int = round(clip(X) / s)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
PyTorch: `PerChannelMinMaxObserver` (channel-wise), `MinMaxObserver` (layer-wise)。APHQ-ViT 在 block 重建中同时使用两者，量化参数校准阶段通过 min-max 确定后固定，重建阶段通过 AdaRound 优化舍入方向。

涉及论文标题：
- APHQ-ViT: Post-Training Quantization with Average Perturbation Hessian Based Reconstruction for Vision Transformers
- I&S-ViT: An Inclusive & Stable Method for Pushing the Limit of Post-Training ViTs Quantization
- Squat (EdgeQAT): Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

I&S-ViT 首次系统分析了不同量化粒度组合对 loss landscape 的影响：channel-wise 权重量化 + layer-wise 激活量化 → rugged、高 loss landscape（不利于优化）；全精度权重 + channel-wise 激活量化 → 平滑、低 loss landscape（利于优化）。基于此分析提出 SOS 三阶段策略——先在 channel-wise 激活量化下优化（平滑 landscape），再通过 scale reparameterization 无损转 layer-wise（保持推理效率），最后量化权重微调。

在 Squat 中，粗粒度逐层量化（layer-wise quantization）是核心设计约束。Squat明确批评之前QAT工作（LLM-QAT、EfficientQAT、TSLD）采用channel-wise/token-wise细粒度量化——这些方法虽然在精度上有优势，但每个矩阵内有多个scaling factor，无法在移动端SIMD的GeMM kernel上高效执行。标准SIMD-based GeMM kernel（如gemmlowp/QNNPACK）仅支持每矩阵单scale的整数MAC操作。因此Squat坚持逐层粗粒度量化以保证移动端部署兼容性，通过熵损失+分布损失+token自适应量化来弥补粗粒度量化的精度劣势。
