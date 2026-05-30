## Feature Rank (r_eff) in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Feature Rank（r_eff，有效特征秩）是 LatentMoE 论文引入的信息论概念，定义为：对于给定的推理任务，保留任务相关信息所需的最小自由度（degrees of freedom）。r_eff 构成了 MoE routed expert 输入维度 d 的信息论下界——将 d 压缩到低于 r_eff 会导致任务相关信息的不可逆丢失，从而造成精度塌缩。

r_eff 的作用是为 latent space compression 提供理论边界：只要压缩后的 latent dim ℓ ≥ r_eff，信息损失 negligible。LatentMoE 通过压缩比 sweep 实验（α=1,2,4,8）验证 r_eff ≈ d/4（即 α=4 为 safe compression ratio）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

```
Feature Rank Determination Pipeline (Design Principle IV + Empirical Validation):

1. Hypothesis:
   For a given task T and model dimension d:
   ∃ r_eff(T) such that:
   - Compression to ℓ ≥ r_eff → information loss negligible
   - Compression to ℓ < r_eff → accuracy collapse

2. Empirical Validation (LatentMoE 16BT-2BA ablation):
   Fix architecture, sweep compression ratio α = d/ℓ:
   
   α=1 (baseline, ℓ=d=2048): validation loss = L_base
   α=2 (ℓ=1024):          validation loss ≈ L_base (marginal)
   α=4 (ℓ=512):           validation loss ≈ L_base (acceptable)
   α=8 (ℓ=256):           validation loss >> L_base (collapse!)

   → r_eff ≈ d/4 = 512 for this model/task configuration

3. Pipeline Integration:
   if ℓ = d/α ≥ r_eff:
       use LatentMoE with compression ratio α
   else:
       reduce α until ℓ ≥ r_eff (safety margin)

4. Task Dependence:
   - r_eff is task-specific (different tasks extract different information)
   - Larger models may have larger r_eff
   - More diverse training data may require larger r_eff
```

LatentMoE 进一步验证 r_eff 在更大规模（95B）上仍然有效——即对于更大的模型，α=4 的压缩仍然 safe。这表明 r_eff 可能更多是任务驱动的而非模型规模驱动的。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

r_eff 本身是一个理论概念，实践中通过以下方式使用：
- 决定 LatentMoE 的压缩比 α：α_max = d / r_eff（最大安全压缩比）
- 通过实验 sweep 确定：对给定模型/任务配置 sweep α=1,2,3,4,6,8... 找到精度开始下降的拐点
- 类似于 quantization 中的 effective bit-width 概念：不同 tensor/operator 需要不同精度
- 当前需要 per-configuration 实验验证，未来可通过理论分析/小规模 proxy 实验预测

涉及论文标题：
- LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts
