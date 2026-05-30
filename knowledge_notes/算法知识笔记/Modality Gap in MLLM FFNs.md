## Modality Gap in MLLM FFNs

术语是什么？
Modality Gap是MoDES发现的MLLMs中不同模态token在FFN层的行为差异：(1) t-SNE可视化显示text/vision token的FFN输入表示跨所有层存在一致分布差异；(2) vision token在FFN前后的余弦相似度高于text token——FFN对vision token更新幅度更小；(3) vision token与FFN权重的夹角更接近90°（正交），减弱更新量；(4) 降低vision token的top-k对性能影响更小——vision expert冗余度更高。该发现直接motivate DMT的设计。

从算法pipeline角度拆解术语：
```
度量1: cos_sim(x_pre, x_post)  → text较低(大更新) vs vision较高(小更新)
度量2: angle(x, W_FFN)         → text较小 vs vision接近90°(弱交互)
度量3: sensitivity to k_reduction:
    Δacc(text, k↓) > Δacc(vision, k↓)  → vision冗余度更高
```
这些度量在GQA数据集上使用Kimi-VL-A3B-Instruct验证。

术语一般如何实现？如何使用？
通过t-SNE降维可视化和余弦相似度分析在GQA数据集上验证。这是MoDES区别于text-only LLM expert skipping方法的根本motivation。解释了MoDES在高跳过率（>80%）下仍保持性能的原因——vision token的expert可被大幅跳过而不影响输出质量。该insight也预测了DMT中τ_v < τ_t的设计合理性。

涉及论文标题：
- MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping
