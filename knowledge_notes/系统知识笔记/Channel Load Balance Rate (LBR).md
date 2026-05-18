## Channel Load Balance Rate (LBR)

术语是什么？

Channel Load Balance Rate (LBR) 是 RoMe 论文提出的衡量粗粒度（4KB）内存访问下跨 HBM channel 数据分布均匀性的指标。由于 RoMe 将访问粒度从传统 cache-line 大小（如 HBM4 的 32B）提升到 4KB，若数据跨 channel 分布不均，某些 channel 可能承载过多传输量而另一些闲置，导致有效带宽低于理论值。LBR 量化每个 channel 在 workload 执行期间承载的数据传输量比例（归一化到 baseline HBM4 的 LBR ≈ 1）。LBR 值越接近 1 表示分布越均匀，RoMe 越能充分利用其增加的 channel 带宽（36 channels/cube vs baseline 32 channels/cube）。

从系统架构角度拆解术语：

LBR 在 LLM serving 场景下分别对 Attention 层和 FFN 层计算，因为两层的数据访问模式和数据量差异显著。LBRAttn 的影响因素：(1) tensor parallelism degree——TP 将 weight 沿 attention head 维度分片到不同 accelerator，减少单 accelerator 内单次访问的 weight 数据量，可能降低 LBRAttn；但 data parallelism 下 weight 不分片，LBRAttn 更高；(2) hidden dimension size——大 hidden dim（Llama 3-405B 的 16,384）使 weight 占数据移动主导，即使 TP 下 LBRAttn 仍高；(3) KV-cache size——batch size 增大时，KV-cache 和 activation footprint 增大，更多 channel 被均匀利用，LBRAttn 改善。LBRFFN 的影响因素：(1) MoE vs dense——MoE 中只激活 top-k experts，小 batch 下 expert 选择高度偏斜，LBRFFN 低；batch 增大后更多 experts 被选中，LBRFFN 改善（如 DeepSeek-V3 的 8/256 experts 约在 batch=64 时所有 experts 开始被选中）；(2) 中间层维度——DeepSeek-V3 小中间维度（2048）导致 LBRFFN 偏低，Grok 1（32768）和 Llama 3（53248）则 LBRFFN 高。论文实验中 RoMe 的 LBR 在不同 batch size 和模型下维持在 0.85-1.05 范围。

术语一般如何实现？如何使用？

LBR 通过 LLMSimulator + Ramulator 2.0 cycle-accurate simulation 离线计算：记录每个 decode step 每个 channel 的数据传输量 → 计算 channel-wise data volume distribution → LBR = (average channel load) / (max channel load) 或类似均匀性指标 → 归一化到 HBM4 baseline。LBR 是 RoMe 特有的评估指标，用于验证 4KB 粗粒度访问不会因 data placement 不均衡导致严重 bandwidth underutilization。在系统部署中，LBR 可指导 weight/KV-cache 的 channel-level data placement 策略：通过调整 address mapping 和 data sharding 策略，使数据尽可能均匀分布到所有 channel，最大化有效带宽。

涉及论文标题：
- RoMe: Row Granularity Access Memory System for Large Language Models
