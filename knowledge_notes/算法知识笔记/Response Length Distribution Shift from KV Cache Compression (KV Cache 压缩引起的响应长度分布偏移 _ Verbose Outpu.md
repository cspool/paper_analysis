## Response Length Distribution Shift from KV Cache Compression (KV Cache 压缩引起的响应长度分布偏移 / Verbose Output)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

KV cache 压缩引起的响应长度分布偏移是论文 "Rethinking KV Cache Compression" 首次系统揭示的现象：有损 KV cache 压缩（量化或稀疏 eviction）会导致 LLM 生成比 FP16 baseline 更长的响应（verbose output），且高压缩比加剧此效应。论文通过定义响应长度差异 $D = (L^{un} - L^{cs})/L^{un}$（$L^{un}$ = 未压缩时的响应长度，$L^{cs}$ = 压缩后的响应长度），负值表示压缩导致更长输出。论文用 ShareGPT 1000 样本和 LLaMA-3.1-8B-instruct 测量发现：KIVI/GEAR/H2O/StreamingLLM 均导致 >20% 样本的输出长度增加 ≥50%（1.55-1.76× 平均 length increase）。语义相似度测试（Table 4）进一步表明：压缩后的更长输出并非质量提升，而是在相似或略低的语义质量下更 verbose。这一发现对实际部署有直接含义：即使压缩提升了 tokens/s 吞吐，更长的输出可能完全抵消甚至逆转端到端延迟收益。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**Verbose Output 判定流程（论文 Section 4.3）**：
```
# 给定：同一 prompt，FP16 baseline 和 压缩算法分别生成
response_fp16 = generate(LLM, prompt, compression=None)
response_comp = generate(LLM, prompt, compression=algo)

Q_fp16 = semantic_score(response_fp16, reference)
Q_comp = semantic_score(response_comp, reference)
L_fp16 = len(response_fp16)
L_comp = len(response_comp)

# Verbose 判定：
is_verbose = (Q_comp <= Q_fp16) AND (L_comp >= L_fp16)
# 即：质量没提升（甚至下降），但输出更长

# 长度差异度量：
D = (L_fp16 - L_comp) / L_fp16
# D < 0 → 压缩导致更长输出
# D > 0 → 压缩导致更短输出
```

**Annotations**: 论文 Table 4 显示：FP16 semantic score=49.6，KIVI=50.7（略高），GEAR/H2O/Stream=46.2-46.3（略低），但所有压缩算法的 length increase 均为 1.55-1.76×。温度参数 T=0.9 和 T=1.1 分别导致 ~45% 样本变长和 ~20% 样本变短——大致对称。而 KV cache 压缩则显著非对称地偏向更长输出（>20% samples with 1.5×+ length increase）。

术语一般如何实现？如何使用？

论文提出 **Length Predictor** 作为工具来预测给定压缩算法下某 prompt 的可能响应长度：使用 LongFormer (max_seq_len=4096) 作为 BERT-based classifier，输入为 prompt text，输出为 response_length/prompt_length ratio，训练数据来自压缩算法在 ShareGPT 上的实际生成。精度 >85%（Table 6/10）。在请求路由器中，length predictor 结合 throughput predictor 估计每请求的端到端延迟，用于路由决策。论文 Table 8 显示：仅用 throughput predictor 可加速 1.18-1.48×，加上 length predictor 后可进一步提升至 1.45-1.80× E2E latency speedup。

涉及论文标题：
- Rethinking Key-Value Cache Compression Techniques for Large Language Model Serving
