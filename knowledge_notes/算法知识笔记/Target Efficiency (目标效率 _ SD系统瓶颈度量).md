## Target Efficiency (目标效率 / SD系统瓶颈度量)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Target Efficiency 是 MoESD 提出的一个新系统指标，用于衡量目标模型架构和工作负载（而非 draft 模型算法质量）对 Speculative Decoding speedup 的影响。定义为 Target Efficiency = T_T(B, 1) / T_T(B, γ)，即目标模型单 token 前向时间与多 token（γ 个 draft tokens）验证前向时间的比值。该比值越接近 1，说明验证多 token 的时间与单 token 解码时间接近，SD 的系统开销越低。与之互补的传统指标 acceptance rate α 衡量 draft 模型的算法准确性，但无法解释系统瓶颈——即使 α 相同，在不同 batch size 或不同模型架构下 SD speedup 可能差异巨大。Target Efficiency 将系统因素（batch size、模型架构、memory-bound vs compute-bound）从算法因素中解耦，帮助研究者独立理解"目标模型和 workload 对 SD 是否友好"。

从算法pipeline角度拆解术语：
Target Efficiency 直接取值于 SD 执行过程中的实测时间，计算简单但含义丰富：

```
# SD 一轮的组成
T_SD_round = γ × T_D(B, 1) + T_T(B, γ) + T_reject

# Speedup 公式（MoESD Eq. 4）
Speedup = σ × (γ + 1) / (γ × T_D(B,1)/T_T(B,1) + T_T(B,γ)/T_T(B,1) + T_reject/T_T(B,1))

# Target Efficiency 定义
Target_Efficiency = T_T(B, 1) / T_T(B, γ)
```

Target Efficiency 反映两种导致 T_T(B,γ)/T_T(B,1) 增大的因素：
- **(1) Compute-boundness**：大 batch 下模型进入 compute-bound，T_T(B,γ) ∝ γ → Target Efficiency → 1/γ（低）
- **(2) Extra memory loads**：小 batch 下 MoE 验证 γ tokens 激活更多 expert → 参数加载增加 → T_T(B,γ) > T_T(B,1) → Target Efficiency 下降

在中等 batch size（所有 expert 已激活但未 compute-bound）下，Target Efficiency ≈ 1（最优）。MoESD 实验（Fig. 2）验证 Target Efficiency 与 end-to-end SD speedup 趋势高度一致，而 acceptance rate 仅在小范围内波动，无法解释 speedup 大幅变化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：从 vLLM runtime logs 中提取 T_T(B,1) 和 T_T(B,γ)（分别来自单 token AR 解码和多 token SD 验证的计时），直接计算比值。vLLM 的 cudagraph optimization 和详细 timing report 使这一提取可行。
- 使用场景：(a) 评估 SD 在不同 batch size 下的适用性——Target Efficiency 曲线与 speedup 曲线高度相关，无需实际运行完整 SD 即可预测趋势；(b) 比较不同模型架构对 SD 的友好程度——MoE vs dense 的 Target Efficiency 对比揭示系统性优势；(c) 指导 SD 部署决策——若 Target Efficiency 持续 < 0.5，SD 可能不适合当前 workload。
- 局限：仅反映 target model 的时间比例，不捕获 T_D（draft model 开销）和 acceptance rate 的影响——完整 speedup 仍需三者结合。论文旨在通过 Target Efficiency 补充（而非替代）acceptance rate，提供更全面的 SD 加速理解。

涉及论文标题：
- MoESD: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE
