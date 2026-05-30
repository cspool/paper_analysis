## Expert Load Imbalance at Accelerator Level（加速器级专家负载不均衡 / $\Gamma_{imb}^{acc}$）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Expert Load Imbalance at Accelerator Level ($\Gamma_{imb}^{acc}$) 是 MoE 推理中因专家路由分布偏斜导致的加速器间负载不均的量化指标。在 Expert Parallelism (EP) 中，不同 expert 被分布到不同加速器上。由于实际请求的 token 路由服从非均匀分布（如 Zipfian 分布），部分"热门"expert 收到大量 token，而"冷门"expert 收到很少 token。$\Gamma_{imb}^{acc}(a)$ 定义为加速器 a 上所有 expert 实际处理的 token 总数与理想均匀分布下 token 数的比值。论文建立的通信时间模型直接关联 $\Gamma_{imb}^{acc}$：$\text{Comm}_{\text{MoE}}(B) = 2 \cdot \max_a(\Gamma_{imb}^{acc}(a)) \cdot M_{\text{token}} \cdot n_k \cdot B / (BW_{\text{Int}} \cdot n_{\text{acc}}) + \alpha$。偏斜越严重，通信时间越长，因为最繁忙的加速器发送/接收的数据量远大于其他加速器。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。通过联网搜索让回答具体和精准。
论文通过 Zipfian 分布的 skewness 参数 s 控制专家路由偏斜程度：

- **s=0.0 (均匀)**：每个 expert 处理 B×nk/ne 个 token
- **s=0.8 (强偏斜)**：热门 expert 处理远超均值，冷门 expert 几近空闲

论文的关键发现：(1) 偏斜导致热 expert 在总 batch size 达到 $B_{\text{MoE}}$ 之前就已饱和，冷 expert ArI 低，整体吞吐量下降；(2) **小粒度部署（如 32 GPU×8）比大粒度部署（256 GPU monolithic）更抗偏斜**——当 s=0.8 时，256 GPU 的 $\Gamma_{imb}^{acc}$ 是 32 GPU×8 的 6.13×，因为 256 GPU 中每 GPU 仅处理 1 个 expert，偏斜直接映射为 GPU 级不均；32 GPU×8 中每 GPU 管理 8 个 expert，自然提供了负载均衡。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
缓解措施包括：(1) 部署粒度选择——使用较小的 EP group；(2) expert 冗余放置——热门 expert 复制到多个 GPU；(3) 动态 expert 迁移——运行时根据负载重新分配 expert；(4) 负载感知路由——在 router 中加入负载均衡约束。论文建议通过合理的部署粒度（如 32 GPU×8 而非 256 GPU monolithic）在架构层面降低偏斜影响，同时保持成本和互联可行性。

涉及论文标题：
- Rethinking LLM Inference Bottlenecks: Insights from Latent Attention and Mixture-of-Experts
- Toward Cost-Efficient Serving of Mixture-of-Experts with Asynchrony
