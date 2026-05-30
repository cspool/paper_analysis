## Expert Distribution Gate (专家分布门控)

术语解释
Expert Distribution Gate（δ, 专家分布门控）是 BuddyMoE 的第二个 safety gate，在 batch 级别评估 CPU-resident expert 的比例，防止在过多 expert 同时 offload 时进行广泛替换造成级联误差。

术语是什么？
对于 micro-batch B 在 layer ℓ：δ_ℓ(B) = |{i∈R_ℓ(B) : loc_ℓ(i)=CPU}| / |R_ℓ(B)|。给定阈值 β∈[0,1]，当 δ_ℓ(B) ≥ β 时 bypass replacement（禁止所有替换），当 δ_ℓ(B) < β 时允许 targeted replacement。设计直觉：当许多 requested experts 在 CPU 时，broad replacement 会同时影响许多 tokens，可能 compounding errors；当只有少数在 CPU 时，targeted replacement 可避免 bursty CPU→GPU traffic 且 limited accuracy exposure。

从系统架构角度拆解术语：
Distribution gate 在 system-level batch processing 中的位置：每个 micro-batch进入 MoE layer → Router 计算所有 token 的 expert assignment → 统计 batch-level CPU residency ratio δ → if δ ≥ β: 跳过 buddy replacement，使用 original on-demand loading → else: 逐个 token 执行 TAE gate + buddy selection。β 可关联传输预算 B_PCIe：选择 β 使 n̂_cpu(ℓ) · w̄ ≤ B_PCIe（n̂_cpu 为无替换时的 CPU-only invocation 估计，w̄ 为平均 bytes/expert），使系统保持在 PCIe bandwidth budget 内。在 tensor/pipeline parallel 设置中，δ_ℓ per partition 独立计算并局部应用。

术语一般如何实现？如何使用？
- 固定 β per deployment tier 足够（如 β=0.5），adaptive β based on running bandwidth meter 是 drop-in extension
- 是三个 safety gate 中的第二个（在 TAE gate 之后执行），batch-level 决策（非 per-token）
- 论文未明确指定具体 β 值，建议按 deployment tier 配置

涉及论文标题：
- BuddyMoE Exploiting Expert Redundancy to Accelerate Memory-Constrained Mixture-of-Experts Inference
