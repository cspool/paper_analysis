## Diffusion Transformer (DiT)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Diffusion Transformer (DiT) 是 Peebles & Xie (2022, ICCV 2023 Best Paper) 提出的扩散模型架构，将传统U-Net骨干替换为Vision Transformer (ViT)。核心流程：(1) VAE编码器将输入图像/视频压缩到latent space；(2) 将latent切分为不重叠的patches并投影到token embedding；(3) 通过标准Transformer blocks（self-attention + MLP，每层注入时间步t和条件c的adaLN-Zero调制参数）处理token序列；(4) 输出head预测噪声或速度。DiT-XL/2（675M参数）在ImageNet 256×256上FID=2.27，比ADM-U U-Net高效约6×（118.6 vs 742 Gflops）。在视频生成中（Wan2.1、Sora、CogVideoX），视频作为3D volume (T×H×W)处理，patches变为spacetime cubes，序列长度达10K-100K tokens，self-attention的O(N²)成为主要瓶颈。DiT的成功基于将视觉生成建模为序列建模——与LLMs同构，使FlashAttention、tensor parallelism等Transformer基础设施可无缝迁移。

从算法pipeline角度拆解术语：
```
DiT Video Generation per Denoising Step t:
Input: Noisy latent z_t ∈ R^{T×H×W×C}, timestep t, text condition c
→ Patch Embedding: z_t → tokens X ∈ R^{N×d}
→ For each DiT Block:
    (t,c) → MLP → (γ₁,β₁,γ₂,β₂,α₁,α₂)  // adaLN-Zero modulation
    X = X + α₁ ⊙ Attention(LN(X)×(1+γ₁)+β₁)   // O(N²) bottleneck
    X = X + α₂ ⊙ MLP(LN(X)×(1+γ₂)+β₂)
→ Output Head: noise/velocity prediction
```

在Wan2.1-1.3B (N≈30K)中，注意力占单步52.75T FLOPs，SLA降至2.73T（95%稀疏度），实现2.2×端到端加速。

术语一般如何实现？如何使用？
主流开源实现：DiT (https://github.com/facebookresearch/DiT)、Wan2.1 (https://github.com/Wan-Video/Wan2.1)、CogVideoX。SLA使用Wan2.1-1.3B（视频，30K tokens）和LightningDiT-1p0B/1（图像，ImageNet 512×512）为实验模型。FlashAttention、tensor parallelism等LLM基础设施可直接用于DiT推理加速。视频DiT通常使用spatiotemporal patches（如16×16×4 tubelets）和causal temporal attention（每帧仅关注前序帧）降低有效N。

涉及论文标题：
- SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention
