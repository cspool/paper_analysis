## Elucidated Diffusion Models (EDM)（阐明扩散模型）

术语是什么？通过联网搜索让回答具体和精准。

Elucidated Diffusion Models (EDM) 是 Karras et al. (2022) 提出的扩散模型统一框架，将训练和采样从离散时间步重新参数化到连续噪声空间。与DDPM使用离散timestep t∈{1,...,T}不同，EDM使用噪声标准差σ作为连续参数来描述数据损坏程度：x_σ = x₀ + σ·ε, ε~N(0,I)。训练目标为L_EDM = E_{x₀,ε,σ}[λ(σ)·||ε − ε_θ(x_σ,σ)||²₂]，其中weighting function λ(σ)在噪声尺度间平衡贡献。EDM的关键洞察是将扩散模型视作学习scale-dependent denoisers的连续统(continuum)，而非离散的逐步去噪过程。EDM支持两类formulation：Variance-Preserving (VP) 和 Variance-Exploding (VE)，分别对应DDPM++和NCSN++架构变体。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

EDM训练与采样（以FFHQ 64×64 unconditional + VP-EDM为例）：

```
// 训练：连续噪声尺度采样
1: sample x_0 ~ p_data, ε ~ N(0,I)
2: sample σ ~ p_train(σ)           // log-normal分布采样噪声尺度
3: x_σ = x_0 + σ·ε                 // 连续加噪（无离散t）
4: loss = λ(σ)·||ε − ε_θ(x_σ, σ)||²₂  // σ为网络conditioning输入
5: θ ← θ − η·∇_θ loss

// 采样：连续确定性路径（DDIM-like）
1: x_N ~ N(0, σ_max²·I)            // 从最大噪声尺度开始
2: for i = N-1, ..., 0:            // N为离散化采样步数
3:     σ_i = schedule[i]           // 预定义噪声尺度序列
4:     denoised = x_{i+1} − σ_{i+1}·ε_θ(x_{i+1}, σ_{i+1})
5:     x_i = denoised + σ_i/σ_{i+1}·(x_{i+1} − denoised)  // 确定性更新
6: return x_0
```

EDM vs DDPM的关键差异：(1) DDPM使用离散timestep t + fixed noise schedule β_t, ᾱ_t = Π(1-β_s)，训练L = ||ε − ε_θ(x_t, t)||² (无λ加权)；(2) EDM使用连续σ + 灵活noise distribution p_train(σ) + 可调λ(σ)，λ(σ)在中等σ区域赋予更高权重（该区域对应最informative的denoising level）。EDM的reweighting λ(σ)使Spectral Regularization论文能在fine-tuning时选择"weighted"（λ=λ_EDM(σ)）或"unweighted"（λ=1），前者保留EDM的per-noise-level重要性分布，后者对所有噪声水平等同对待。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

NVIDIA官方EDM实现（https://github.com/NVlabs/edm）提供pretrained checkpoints for CIFAR-10/FFHQ/AFHQv2等数据集（VP和VE变体），预训练权重以.pkl格式分发。在Spectral Regularization论文中，EDM作为fine-tuning backbone：加载预训练checkpoint→仅5步fine-tuning with spectral auxiliary loss→采样与标准EDM完全相同。PyTorch实现仅需~50行代码的auxiliary loss计算，computational overhead negligible。

涉及论文标题：
- Spectral Regularization for Diffusion Models

---

