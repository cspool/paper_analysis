## Watermark-based KV-Cache Scaling

术语是什么？

Watermark-based KV-Cache Scaling是SLINFER中的动态KV-cache内存管理策略：通过watermark参数w（默认25%）控制KV-cache的scale-up和scale-down时机，实现early scale-up（预留空间给未来请求和长输出）和lazy scale-down（减少短时波动引起的ping-pong效应）。与简单的on-demand scaling（按当前需求立即分配/释放）不同，watermark策略在scale-up时多分配w%空间(Mrecommend = Mrequire · (1+w%))，在scale-down时延迟释放直到Mrecommend · (1+w%) < Mcur，从而在利用率和scaling overhead间取得平衡。

从系统架构角度拆解术语：

Watermark scaling在paged-attention KV-cache上的运转：

1. **Memory需求估计**：Mrequire = C · max(Σ(Ir + max(Or, Ō)), Lmin)，其中C=per-token KV-cache size，Ir=第r个请求的input length，Or=已生成token数，Ō=历史平均output length，Lmin=下限（设为最大context length）。由于output length不可预知，用Ō做保守估计。

2. **Scale-up决策**（添加新请求时）：若Mcur < Mrequire → 直接scale-up到Mrecommend = Mrequire · (1+w%)。这预留了watermark空间用于：(a) 吸收bursty的新请求到达；(b) 处理超出Ō的长输出请求（一个长输出请求可"steal"其他请求的reserved space）。

3. **Scale-down决策**（请求完成时）：推迟scale-down，除非Mrecommend · (1+w%) < Mcur。这避免了频繁的short-term fluctuation导致的resizing ping-pong。

4. **Scaling overhead**：基于paged-attention的KV-cache resize操作：(a) 分配新cache blocks；(b) 将used cache从旧blocks逐页copy到新blocks；(c) 删除旧blocks。GPU上32GB→64GB scale-up需1.9s，32GB→16GB scale-down需0.3s（Figure 17）。

5. **Watermark sensitivity**：w=0%（禁用，on-demand scaling）→instance花11.3% lifetime在scaling上（frequent adjustments）；w=25%→overhead降至1.4%，request migration rate（因underestimation）仅0-0.3%；w=50%→overhead略降但memory utilization降低。

术语一般如何实现？如何使用？

实现要点：
- **Watermark作为hyperparameter**：推荐w=25%，对workload不敏感
- **兼容paged-attention**：在vLLM的block-based KV-cache管理之上增加watermark逻辑，不修改底层paging机制
- **与orchestration配合**：scale-up前需通过hazard-aware orchestration（见独立条目）的budget检查，可能被reject或compromise
- **Underestimation处理**：尽管有watermark和Ō估计，极少数情况下output远超预期→触发二次scale-up尝试→若失败则evict最长headroom请求并reschedule

涉及论文标题：
- Towards Resource-Efficient Serverless LLM Inference with SLINFER
