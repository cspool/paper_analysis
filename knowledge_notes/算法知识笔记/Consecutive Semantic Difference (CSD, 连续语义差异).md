## Consecutive Semantic Difference (CSD, 连续语义差异)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
CSD 是 Oracle-MoE 提出的衡量 MoE routing 中连续 token 之间 expert 选择变化程度的指标。定义：CSD = Σ_{t=2}^{T} Δe_t，其中 Δe_t = |e_t \ e_{t-1}| 为连续 token 的激活专家集合的对称差。在固定硬件和 swapping 算法下，总延迟 L_total = Σ(L_compute + l_swap · Δe_t)，因此 CSD 直接决定总延迟。

从算法pipeline角度拆解术语：

Token-level MoE：CSD_token ≈ Σ C(W_g, k) ||t_t - t_{t-1}||，由于 token embedding 受 token-identity 主导、方差大，CSD_token 高。
Oracle-MoE：CSD_oracle ≈ Σ ||z_{S(t)} - z_{S(t-1)}||，语义组嵌入在 Oracle Space 中平滑变化，CSD_oracle 低。

Theorem 1: 以高概率 CSD_token > CSD_oracle。实验验证：Oracle-MoE 激活不一致性 4-6 per 100 tokens，Switch Transformer 为 53-82。

术语一般如何实现？如何使用？
- CSD 用作 memory-constrained MoE 推理中 latency 的代理指标，无需实际测量 I/O 即可评估 routing 策略的 swapping 友好程度。
- 将 latency 优化转化为语义空间连续性问题，可从理论上分析和比较不同 routing 策略。

涉及论文标题：
- Oracle-MoE: Locality-preserving Routing in the Oracle Space for Memory-constrained Large Language Model Inference

---
