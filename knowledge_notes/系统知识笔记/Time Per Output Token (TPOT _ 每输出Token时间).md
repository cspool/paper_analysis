## Time Per Output Token (TPOT / 每输出Token时间)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Time Per Output Token (TPOT) 是 LLM 推理 decode 阶段的关键延迟指标，定义为生成单个输出 token 所需的平均时间。TPOT 直接决定了用户可感知的 token 生成速率（tokens/second = 1/TPOT），是评估 LLM 服务质量的 SLO（Service Level Objective）核心参数。论文中给出了 TPOT 的详细分解公式：$\text{TPOT}(B, L) = n_{\text{decoder}} \cdot \left( \frac{M_{\text{attn}} \cdot deg_{\text{DP}} + M_{\text{MoE}}}{n_{\text{acc}} \cdot BW_{\text{Mem}}} + \delta(B, L) \right)$，其中第一项是模型权重的加载延迟（memory-bound），第二项 $\delta(B, L)$ 包括 KV$ 访问时间、激活内存访问、通信开销和剩余计算时间。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
TPOT 约束直接决定最大可行 batch size $B_{\text{SLO}}$：

$$\delta_{\min}(B, L) \ge B \cdot \left(\frac{M_{\text{KV}} \cdot L + M_{\text{act}}(L)}{n_{\text{acc}} \cdot BW_{\text{Mem}}}\right) + \text{Comm}(B, L)$$

当 batch size B > $B_{\text{SLO}}$ 时，无论系统如何优化，TPOT 都无法满足 SLO 时间限制。论文通过此公式分析了：(1) MLA 减小 $M_{\text{KV}}$ 和 $M_{\text{attn}}$ 从而降低 $\delta_{\min}$，增大 $B_{\text{SLO}}$；(2) MoE 增大 $M_{\text{MoE}}$ 增加模型加载时间，压缩 $\delta_{\min}$ 的时间 budget，减小 $B_{\text{SLO}}$；(3) 通信时间 Comm(B,L) 与互联带宽成反比，低带宽互联显著降低 $B_{\text{SLO}}$。论文中假设 TPOT$_{\text{SLO}} = 50$ms 作为 SLO 阈值。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
TPOT 是 LLM serving 系统的关键工程指标。主流推理框架（vLLM、SGLang、TensorRT-LLM）均以 TPOT 作为核心监控指标。实际部署中需结合 TTFT (Time To First Token) 共同评估用户体验。论文通过 TPOT-throughput 曲线图（Figure 9）评估模型在不同 batch size 下的权衡关系。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
