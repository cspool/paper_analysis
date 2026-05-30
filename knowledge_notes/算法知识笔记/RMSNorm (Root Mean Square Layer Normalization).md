## RMSNorm (Root Mean Square Layer Normalization)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
RMSNorm (Root Mean Square Layer Normalization, Zhang & Sennrich, 2019) 是一种简化版的 Layer Normalization，去除了均值中心化（mean subtraction），仅通过 root mean square 统计量进行缩放：RMSNorm(x) = x / RMS(x) ⊙ g，其中 RMS(x) = sqrt(mean(x²) + ε)，g 为可学习增益参数。相比 LayerNorm（需计算均值和方差），RMSNorm 省去了均值计算，在 GPU 上约快 7-15%。Transformer 架构中（LLaMA、Qwen、Gemma 等）和 SSM/RNN 架构中普遍采用。xLSTM 7B 的实验证实（Fig. 9, App. C.2）：使用 LayerNorm 作为 pre-norm 在 1.4B 参数规模导致极大的梯度 norm 和验证 loss 发散，而 RMSNorm 训练稳定。对于 head-wise state norm（Eq. 6），RMSNorm 和 LayerNorm 均表现稳定。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
// RMSNorm 前向计算
输入: x ∈ R^d (沿最后一维归一化)
参数: g ∈ R^d (可学习增益), ε = 1e-6 (通常)

rms = sqrt(mean(x²) + ε)   // 仅需一次平方+均值+sqrt
output = (x / rms) ⊙ g      // element-wise 缩放

// 对比 LayerNorm:
μ = mean(x)                  // 额外计算均值
σ² = mean((x - μ)²)         // 需先减均值再平方
output = (x - μ) / sqrt(σ² + ε) ⊙ g + b  // 额外 bias 参数

// xLSTM 7B 中 RMSNorm 的使用位置:
// 1. Pre-norm: 每个 block 进入 mLSTM 前: z_norm = RMSNorm(z)
// 2. Pre-norm: 每个 block 进入 SwiGLU MLP 前: z_norm2 = RMSNorm(z2)
// 3. Head-wise state norm 仍用 LayerNorm (Eq. 6 中的 Norm(h̃_t))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- PyTorch 无原生 RMSNorm，常用实现：
  - LLaMA-Factory/Torch: 自定义 `RMSNorm` 继承 `nn.Module`
  - HuggingFace: 多数 LLM 使用 `LlamaRMSNorm` 或其他等价实现
- 适合替代任何网络中的 LayerNorm，尤其大规模训练中对速度敏感的场景
- 对于递归架构（xLSTM/Mamba/RWKV），推荐在 pre-norm 位置使用 RMSNorm 以获得更好的训练稳定性
- 在 flash-attention 或 fused kernel 中可直接将 RMSNorm 融合进前一层的 output 计算

涉及论文标题：
- xLSTM_7B__A_Recurrent_LLM_for_Fast_and_Efficient_Inference

---
