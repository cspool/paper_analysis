## MM-sc / MM-ss（spike-continuous / spike-spike 矩阵乘法）与 SNN 非线性算子（ssoftmax / slayernorm）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- SNN 的矩阵乘法与 ANN 不同：MM-sc（spike-continuous）是一个操作数为脉冲（{-1,0,1}）、另一个为连续值（权重/膜）的矩阵乘，用于脉冲卷积与线性层；MM-ss（spike-spike）是两个操作数都是脉冲的矩阵乘，用于脉冲注意力（QK^T 与 AV 的 spike 版本）。由于两个脉冲操作数直接相乘大多是 0（稀疏），ELSA/SpikeZIP-TF 按 SpikeZIP-TF 的做法把 MM-ss 用两个 MM-sc 实现：把 spike tracer 当作连续操作数参与计算。除 MM 外，SNN 还需要杂项算子：ssoftmax（spiking softmax）、slayernorm（spiking layer normalization）、残差加、im2col，均来自 SpikeZIP-TF 的整数实现。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 算子在推理 pipeline 中的计算：
```
MM-sc:  out[x,:] += Σ_{spike (x,y,q)} (q? -1:1) * W[y,:]   # 每 spike 触发一行权重累加
MM-ss:  用 tracer 当连续数：先算 Q_trace · K^T 得注意力分数（MM-sc），
        再对分数做 ssoftmax，最后 score · V_trace（MM-sc）
ssoftmax: 整数指数/求和近似 softmax，输出仍为脉冲
slayernorm: 整数均值/方差归一化，输出脉冲
im2col: 卷积输入按核窗口展开（router 侧广播）
```
- 例（ELSA Fig.10c）：MM-sc 中 spike batch (0,1),(0,3) → 读 W 第 2、4 行 → 加法树累加出膜行 [3,5,4,4]；MM-ss 在 ViT-S 的注意力中占主导（Tab.II 中 ViT-S 的 #Sops 90.74G 远大于 #Ops 8.50G，即 spike-spike 计算量的放大）。
- Annotations：q 为脉冲极性位（q=1 负、q=0 正）；负 spike 时权重行取二补码；MM-ss 的"tracer 作连续数"是 SpikeZIP-TF 的关键技巧（论文脚注 4）；ssoftmax 中生成单个 token 需要全部 query/key token 就绪，故 ELSA 在该处停顿流水（token-wise pipeline 的 stall 点）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 算法侧：SpikeZIP-TF（arXiv 2406.03470）定义这些算子并保证数值等价于 QANN 的对应算子；SpikingJelly 的 SpikeZIPTFQANNRecipe 提供实现。硬件侧：ELSA 的 Tab.I 列出支持的算子集——PE 执行 MM-sc（mini-batch Gustavson 数据流）；路由器内置 SSoftmax Unit 与 SLayerNorm Unit（各含少量 ST-BIF 电路与 tracer/膜存储，占 ELSA 面积 6.72%）执行 ssoftmax/slayernorm；im2col 与残差加在路由器侧以广播实现。Tab.III 显示 SSoftmax/SLayerNorm 单元占 ELSA 面积 3.45%/3.27%。

涉及论文标题：
- ELSA: An ELastic SNN Inference Architecture for Efficient Neuromorphic Computing
