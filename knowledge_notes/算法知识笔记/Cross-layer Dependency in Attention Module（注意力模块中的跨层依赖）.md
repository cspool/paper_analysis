## Cross-layer Dependency in Attention Module（注意力模块中的跨层依赖）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Cross-layer Dependency in Attention Module 指 Transformer 的 attention 模块中 Q（Query）、K（Key）、V（Value）三个投影层之间的强相互依赖关系。在 self-attention 中，Q 和 K 的矩阵乘法结果经 softmax 产生 attention map A，A 再与 V 相乘得到 attention 输出。这意味着：(1) Q 的量化误差会通过 softmax 非线性放大并传播给 V；(2) K 的量化误差同样影响 attention map，进而影响 V 的有效输入；(3) V 的量化误差被 attention map A 加权。三层共同的输出是 `SA(Q,K,V) = A·V`，任何一层的误差都会通过非线性交互影响最终 attention 输出。传统的 layer-wise PTQ（AdaRound、OPTQ）假设层间独立，忽略这种依赖，导致低比特下性能退化严重。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
跨层依赖在 attention 计算中的传播路径：
```
# Self-Attention 前向: 三层之间的信息流
Q = X @ W_Q     # Query projection, 形状 [B, H, L, d_h]
K = X @ W_K     # Key projection
V = X @ W_V     # Value projection

# 依赖点 1: Q-K 交互
S = Q @ K.T / sqrt(d_h)    # [B, H, L, L] 注意力分数
A = softmax(S)              # 注意力权重 → Q/K 误差经 softmax 非线性放大

# 依赖点 2: A-V 交互  
O_attn = A @ V              # [B, H, L, d_h] → V 误差被 A 加权组合
```
Layer-wise 方法独立量化 W_Q、W_K、W_V 时，各自仅最小化 `||ΔW_Q·X||^2`、`||ΔW_K·X||^2`、`||ΔW_V·X||^2`，完全忽略了上述交互路径。aespa 通过 attention-wise reconstruction 捕获这些依赖：W_V 的 Hessian `H_V = 2E[XA^TAX^T]` 通过 A 矩阵将 Q 和 K 的信息注入 V 的量化过程；W_Q 的损失引入 `E[K^TK]` 项以感知 K 的信息；W_K 的损失引入 `E[Q^TQ]` 项以感知 Q 的信息。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
捕获跨层依赖需要突破传统 layer-wise 方法的独立 Hessian 假设。aespa 的实现方式：(1) 在全精度下前向一次得到 attention map A；(2) 预计算 attention-aware 统计量 `E[XA^TAX^T]`、`E[K^TK]`、`E[Q^TQ]`；(3) 在后续量化迭代中重用这些统计量，无需重复执行 attention forward。这种预计算方法同时解决了效率问题（避免每轮 attention computation）和依赖建模问题（统计量携带跨投影信息）。该思路可推广到其他含非线性交互的模块（如 FFN 中 gate/up 投影的 SiLU 激活交互），但需要更复杂的数学推导。

涉及论文标题：
- Towards Next-Level Post-Training Quantization of Hyper-Scale Transformers
