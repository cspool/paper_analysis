## 分层蝴蝶分解（Hierarchical Butterfly Decomposition / Hierarchical BSMM）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- 分层蝴蝶分解是 MLX 对全局蝴蝶分解的改进：把权重矩阵 W 划分为 (D/B)×(D/B) 个 B×B 局部 tile，只在每个 tile 内应用蝴蝶因子（而非对整个 D×D 矩阵做全局分解）。总蝴蝶参数计算量从全局 O(D log D) 降到 (D/B)²·O(B log B)=O((D²/B)·log B)；复杂度比从 O(log D / D) 变为 O(log B / B)。B 是第二个可调精度-效率旋钮：B 越大结构化稀疏越强（复杂度比 O(log B/B) 更小、算得更省）但近似误差越大。论文在 B∈{16,32,64} 上做敏感度评估：更大 B 线性层 FLOP 削减更多但精度损失更大，长上下文设置下 B=32 最佳；B 还可与 FFT 压缩率 s 联合调节。
- 结构意义：该分解天然形成两级数据流——tile 间（inter-tile）按粗粒度 blocked-GEMM 数据流执行，tile 内（intra-tile）BSMM 实现细粒度结构化蝴蝶数据流；与语义感知傅里叶压缩（序列维 N）正交，在隐藏维 D 上暴露并行性，二者共同构成"混合化蝴蝶 kernel"（Table I：FFT-CMP 用于 Attn./KV Cache，hierarchical BSMM 用于 QKV/FFN）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
分层蝴蝶投影 pipeline（D=4096、B=32、tile 数 (D/B)²=16384）：
```
# 离线：对每个 (i,j) tile 做蝴蝶分解
for i in 0..D/B-1, j in 0..D/B-1:
    W[iB:(i+1)B, jB:(j+1)B]  ≈  ∏_{k=1}^{log2 B} B_B^(k)   # 32 点蝴蝶，5 层，2×32×5 参数/tile
# 推理：Y = X @ W
for i in 0..D/B-1:                             # tile 间：coarse blocked-GEMM 数据流
    for j in 0..D/B-1:
        Y[iB:(i+1)B, j] += X[:, jB:(j+1)B] @ B_tile(i,j)   # tile 内：蝴蝶稀疏数据流
```
复杂度对比：全局分解 O(D log D)=O(4096·12)≈49k 参数单位 vs 分层 O((D²/B)log B)=O((4096²/32)·5)≈2.6M——分层参数更多但分解收敛更容易、误差更小（论文核心论点：块结构把蝴蝶稀疏局部化到小子矩阵，使分解更易收敛、精度损失更小）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现与使用：在 Llama2-7B/InternLM2-7B 的 QKV/FFN 投影中替换稠密权重为 B×B 块内蝴蝶因子，配合 FFT-CMP（s=0.75/0.5）在 >60% 层上应用并 LoRA 微调，QKV+Attention 计算削减 57%-72%、整体精度降 <1.45%（Winogrande-xl/Wikitext-2/103/Ada-LEval 评估）；H100 decode 阶段结合块 BSMM 1.4-1.9× 端到端加速（减少 KV-cache 流量）。在 MLX 硬件上，B×B tile 内蝴蝶 = 闭环 CDC（n/B 个不相交 closed set），配合闭集局部性重排（I/O shuffle 把长 stride 交换转成紧凑本地数据流 + 有界次数的 stage 间交换）。
- 涉及论文标题：MLX: Multi-Layer Execution for Structured LLM Workload Acceleration on Spatial Architectures
