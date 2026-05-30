## Effective Receptive Field (ERF / 有效感受野)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
ERF 度量神经网络中 token 间信息传播的有效距离。在 LLM 中定义为所有层所有 head 中最后 token 到之前 token 的归一化 attention score 的加权平均距离：$ERF \approx \sum_n \sum_h \sum_s \frac{2 M^h(S,s) \cdot (S-s) \cdot (N-n+1)}{H N (N+1)}$。Hymba 用 ERF 分析不同架构的信息传播效率。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Hymba ERF 分析结论：(1) Llama3 ERF 最大但 cache 代价最高；(2) Parallel hybrid-head 的 ERF 比 sequential hybrid 大一个数量级，cache 相当；(3) 纯 Mamba ERF 最小。这直接解释了 parallel 结构 recall 优势（+4.74%）。

```
ERF = 0
for n in 0..N:
    for h in 0..H-1:
        attn = attention_map[layer=n][head=h][-1, :]
        for s in range(S+1):
            ERF += 2 * attn[s] * (S-s) * (N-n+1) / (H * N * (N+1))
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
ERF 主要作为架构分析工具：(1) 比较架构信息传播效率；(2) 指导 attention 模式设计；(3) 与 task accuracy 交叉验证。

涉及论文标题：
- Hymba: A Hybrid-head Architecture for Small Language Models

---
