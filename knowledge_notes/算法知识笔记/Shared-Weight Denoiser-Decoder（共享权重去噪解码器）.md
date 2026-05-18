## Shared-Weight Denoiser-Decoder（共享权重去噪解码器）

术语是什么？通过联网搜索让回答具体和精准。
Shared-Weight Denoiser-Decoder是ELF提出的核心架构设计：单一网络（DiT backbone）在所有time steps t∈[0,1)作为denoiser（预测clean embeddings x̂，用MSE loss训练），在final step t=1作为decoder（预测clean embeddings并通过unembedding layer映射为token logits，用CE loss训练）。关键创新在于：通过共享网络权重、联合训练两个目标，消除了对单独训练decoder的需求。网络上通过binary "mode" token（"denoise" vs "decode"）区分两种操作模式。训练时80% steps分配为denoising mode、20%为decoding mode。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

ELF的Shared-Weight Denoiser-Decoder训练流程：
```
// 联合训练（单batch内混合两种mode）
1: x = T5_encode(tokens)                  // 获得clean embeddings
2: 
3: // === Denoising Branch (80% prob) ===
4: t ~ logit_normal(P_mean=-1.5, P_std=0.8)
5: ε ~ N(0, I)
6: z_t = t·x + (1-t)·ε                    // linear interpolation
7: x̂ = DiT(z_t, t, mode="denoise")       // 网络预测clean embeddings
8: L_MSE = ||(x̂ - x)/(1-t)||²             // x-prediction MSE loss
9:
10: // === Decoding Branch (20% prob, t=1) ===
11: p ~ logit_normal(P_mean=0.8, P_std=0.8) // per-token corruption level
12: ε ~ N(0, I), noise_scale = 5
13: z̃ = p·x + (1-p)·ε                      // 模拟不完美denoiser输出
14: h = DiT(z̃, t=1, mode="decode")         // 同一网络，decode mode
15: logits = W·h                            // unembedding: R^d → R^|V|
16: L_CE = CrossEntropy(logits, s)          // token-level cross-entropy

// 推理期
1: z_0 ~ N(0,I)
2: for t in [0, t_1, t_2, ..., t_{T-1}]:    // ODE/SDE denoising steps
3:     x̂ = DiT(z_t, t, mode="denoise")      // denoise mode
4:     v = (x̂ - z_t) / (1-t)
5:     z_{t+dt} = z_t + dt·v
6: // Final step (t=1):
7: h = DiT(z_T, t=1, mode="decode")         // decode mode
8: tokens = argmax(W·h)                      // 离散化输出
```

解码分支的per-token corruption设计（不同token有不同p值）使网络学会从受污染的embeddings中恢复——模拟推理时denoiser的imperfect outputs。Noise scale=5（OWT）使decode mode对residual errors更加鲁棒。

与baseline的对比：
- Latent Diffusion LMs (LD4LG等)：需要单独训练decoder（AR decoder / NAR decoder），增加参数量和训练阶段
- Per-step discretization DLMs (FLM, Diffusion-LM等)：每步做token prediction+CE loss，denoising trajectory受token-level constraint限制
- ELF：Shared-weight design + 仅在最后步CE loss = minimal treatment of discretization

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Shared-Weight Denoiser-Decoder的实现关键：(1) 网络架构：标准DiT (Diffusion Transformer) with SwiGLU + RMSNorm + RoPE + qk-norm；(2) Mode conditioning：4个可学习的mode tokens（denoise/decode）作为in-context conditioning prepend到输入序列；(3) Unembedding layer：可学习矩阵W ∈ R^{d_model × |V|}，仅在decode mode下使用，与网络联合训练；(4) Denoising mode probability=0.8提供最佳trade-off——过低（0.5）导致denoising训练不足，过高则decoding监督不足。Ablation显示Shared-Weight design比Two-Stage (separate encoder→decoder→denoiser) slightly better，且简化了pipeline（无需预训练decoder的额外stage）。该设计使ELF-B从148M参数（若用adaLN-Zero conditioning）降至105M参数（用in-context conditioning），同时保持competitive性能。

涉及论文标题：
- ELF: Embedded Language Flows
