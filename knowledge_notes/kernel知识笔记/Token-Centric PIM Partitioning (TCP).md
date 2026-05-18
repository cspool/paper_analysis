## Token-Centric PIM Partitioning (TCP)

术语是什么？通过联网搜索让回答具体和精准。

Token-Centric PIM Partitioning（TCP）是PIMphony提出的PIM workload映射策略，将长上下文LLM decoding中Attention（QK^T和SV）的并行维度从传统的head/batch维度转为token维度。在单个PIM module内，TCP沿token sequence方向切分：对于QK^T，每个channel处理一段token的Key cache，与同一query做部分dot-product，结果在module内经PIM HUB/EPU拼接后进入Softmax；对于SV，每个channel处理一段score/value的partial context，随后通过module内GPR-based inter-channel reduction得到完整context vector。TCP仅在单个PIM module内切token，不跨module，因此避免跨module同步开销（论文报告SV的module内reduction开销在LLM-7B 16K tokens下<0.2% attention latency）。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

```
// TCP QK^T执行（per attention head, per PIM module with N channels）:
// Input: Query q[dh], Key cache K[T][dh], token length T
// Output: Score[T] per channel, concatenated in HUB

for each channel c in 0..N-1:
    // 每个channel处理 T/N 个token的Key segment
    t_start = c * (T / N)
    t_end = (c + 1) * (T / N) - 1

    // Step 1: WR-INP — broadcast query to GBuf
    WR_INP(GBuf[0], q[0:dh])

    // Step 2: MAC — per-token dot product with local Key segment
    for t in t_start..t_end:
        MAC(GBuf[0], K[t][0:dh], OBuf[t - t_start])
        // OBuf[t - t_start] += dot(q, K[t])

    // Step 3: RD-OUT — read partial scores
    for t in 0..(T/N - 1):
        RD_OUT(OBuf[t], Score_c[t])

// HUB: concatenate Score_0..Score_{N-1} → Score[0..T-1]
// EPU: Softmax(Score) → Score_norm[0..T-1]

// TCP SV执行:
for each channel c in 0..N-1:
    t_start = c * (T / N)
    t_end = (c + 1) * (T / N) - 1

    for t in t_start..t_end:
        // 用score scalar缩放V[t]并累加
        MAC(Score_norm[t], V[t][0:dh], OBuf_partial[c])
        // OBuf_partial[c] += Score_norm[t] * V[t]

// Inter-channel reduction in GPR:
// context = sum_{c=0}^{N-1} OBuf_partial[c]
```

TCP的关键特性：(1) 并行度来自token维度——长上下文token数量大（32K-1M），远多于head数（32-64），确保每个channel有充足工作；(2) 在16-channel/16-bank配置下，QK^T token length>256、SV token length>32即可full channel activation；(3) 不跨module同步——避免了分布式reduction的通信开销；(4) 与batch size解耦——即使batch=1也能利用所有channel。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

TCP由PIMphony的MLIR compiler在编译时自动生成partition scheme：compiler根据模型config（num_heads、head_dim、PIM module count、channel count）计算per-channel token segment range，在生成的PIM instruction sequences中嵌入token range metadata。Runtime IREE HAL根据当前请求的token length将对应segment的指令分发到各channel。TCP适合长上下文场景（token数大→并行度高），短上下文收益降低但论文仍报告256 tokens下有2.1× speedup。

涉及论文标题：
- PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System

