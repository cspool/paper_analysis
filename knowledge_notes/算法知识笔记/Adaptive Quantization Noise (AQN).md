## Adaptive Quantization Noise (AQN)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
AQN 是 QeRL (Huang et al., NVIDIA, 2025) 提出的动态噪声注入机制。核心设计：(1) 对每个量化线性层采样随机噪声 Z_noisy ~ N(0,σ²I)，Z_noisy ∈ R^{1×d}；(2) 噪声通过 RMSNorm scale 参数注入——additive noise 等价转换为 multiplicative noise：(Z_noisy/w+I)⊙\hat{W}；(3) 噪声强度 σ 按指数衰减：σ(k)=σ_start×(σ_end/σ_start)^((k-1)/(K-1))，K=10 阶段，σ_start=1e-2, σ_end=5e-4。Stage 0 仅有量化噪声（σ=0），后续逐步降低注入噪声。关键洞察：量化噪声的静态/确定性特性对 RL 后期不利，AQN 通过动态控制实现探索到利用的过渡——初始利用量化噪声高熵探索，后期降低噪声稳定收敛。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# AQN 嵌入 GRPO/DAPO 的 RL pipeline
K = 10; σ_start = 1e-2; σ_end = 5e-4
steps_per_stage = total_steps // K

for step in range(total_steps):
    k = step // steps_per_stage
    σ = 0 if k==0 else σ_start*(σ_end/σ_start)^((k-1)/(K-1))
    for each RMSNorm before QKV/gate-up:
        Z_noisy ~ N(0, σ²)         # [1, d]
        w_noise = w_orig + Z_noisy  # 融入 RMSNorm scale
    # forward: x_noisy·\hat{W}^T + LoRA
    # backward: gradient only to LoRA A,B
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
开源：https://github.com/NVlabs/QeRL。噪声共享：Q/K/V 共享同 RMSNorm noise，gate/up 共享另一，因其 LLM 架构中分别前置同一 RMSNorm。乘法噪声等效：(Z_noisy/w+I) 作用为 row-wise multiplicative noise on \hat{W}，证明在 QeRL Appendix G。初始化 σ=1e-2 而非传统 noisy network 的 1e-1，因 LLM 对乘法噪声更敏感。

涉及论文标题：
- QeRL Beyond Efficiency - Quantization-enhanced Reinforcement Learning for LLMs

---
