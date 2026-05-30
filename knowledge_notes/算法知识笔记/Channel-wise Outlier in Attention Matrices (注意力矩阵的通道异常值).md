## Channel-wise Outlier in Attention Matrices (注意力矩阵的通道异常值)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Channel-wise Outlier in Attention Matrices 是 Attention 中 K 矩阵特定 channel 数值远大于其他 channel 的现象。SageAttention 通过可视化 Unidiffuser/CogvideoX 的 Q,K,V 分布发现：K 存在显著 channel-wise large bias（各 token 一致），V 较轻，Q 相对均匀。根因可能与 transformer attention sink / no-op head 机制相关——某些 head 学会将特定 K 维度推向大值。对 INT8 量化影响：outlier channel 主导 per-token scale，非 outlier channel 信号被压入极小区间被噪声淹没。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# K[:,c] ≈ 100 (outlier), K[:,其他] ∈ [-1,1]
# per-token INT8: δ_K = max(|K[t,:]|)/127 ≈ 100/127 ≈ 0.787
# → 正常 signal (e.g. 0.5) 量化误差 ≈ round(0.5/0.787)-0.5/0.787 ≈ 0.635 >> signal
# → QK^T 精度崩溃
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
处理方法：(1) SageAttention Smooth K——利用 softmax 常数偏移不变性，零精度损失、<0.2% overhead；(2) SmoothQuant per-channel scaling——attention 中不适用（无对应权重维度）；(3) LLM.int8() 混合精度——可行但降低效率。SageAttention 的关键贡献是发现 attention softmax 的特有不变性实现零代价 outlier 平滑。

涉及论文标题：
- SageAttention2 Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization
