## DCCA (Descend-Communicate-Communicate-Ascend)

术语解释
DCCA 是 BigMac 论文提出的低维通信策略，将 fine-grained MoE 的执行顺序从 CDAC（先通信后降维）重新排列为先降维后通信，使 All-to-All 通信在压缩后的低维空间进行，从而大幅减少通信量。DCCA 是 BigMac 的核心创新。

术语是什么？
DCCA 将一个 MoE 层的执行拆分为四个连续阶段：

1. **Descend (降维)**：$x' = xW'_{\downarrow}$，将输入 token x ∈ R^h 通过 descending projection $W'_{\downarrow} \in \mathbb{R}^{h \times (r \cdot h)}$ 压缩到低维空间 x' ∈ R^{r·h}。论文设 downscaling factor r = 0.25。

2. **Communicate (All-to-All Dispatch)**：将压缩后的 token x'（维度 r·h）通过 All-to-All 分发到各 expert 所在设备。通信量 = $2 \times top\_k \times \frac{ep-1}{ep} \times b \times s \times (r \cdot h)$，是 CDAC 的 r 倍（-75%）。

3. **Communicate (All-to-All Combine)** & Expert Computation：各 expert 执行 BigMac Expert 的计算后，All-to-All 将输出汇集回源设备（同样在低维 r·h 进行）。

4. **Ascend (升维)**：$y = y'W'_{\uparrow}$, 将 combined output y' ∈ R^{r·h} 通过 ascending projection $W'_{\uparrow} \in \mathbb{R}^{(r \cdot h) \times h}$ 恢复到原始维度 h。

Gate 路由仍使用降维前的 full-dimension x（而非压缩后的 x'），以保证路由精度。

关键：DCCA 仅增加了两个 projection 矩阵（$W'_{\downarrow}$ 和 $W'_{\uparrow}$），仅带来 +1.35% 参数和 +4.54% FLOPs 的额外开销，却换来 -75% 的通信量削减。

从算法pipeline角度拆解术语：

```
# DCCA MoE layer forward (BigMac)
def dcca_moe_forward(x):              # x: [batch, seq, h]
    # Step 1: Gating at FULL dimension
    gate = SoftMax(x @ W_gate)
    topk_w, topk_idx = TopK(gate, k=top_k)

    # Step 2: DESCEND — compress to r·h
    x_low = x @ W_down_prime           # [batch, seq, h] → [batch, seq, r·h]

    # Step 3: All-to-All DISPATCH (low-dim: r·h)
    dispatched = alltoall_scatter(x_low, topk_idx)

    # Step 4: Expert computation + All-to-All COMBINE (low-dim: r·h)
    # Each expert E_i: σ(x @ W_i↑) @ W_i↓  (先升后降)
    combined = expert_compute_and_alltoall_gather(dispatched)

    # Step 5: ASCEND — restore to h
    y = combined @ W_up_prime          # [batch, seq, r·h] → [batch, seq, h]
    return y
```

给定 GPT3-XL (h=2048, r=0.25, ep=32, top_k=8)：
- CDAC: All-to-All = 1,488 GB, 占总时间 91.8%
- DCCA: All-to-All = 372 GB (-75%), FLOPs +4.54%

术语一般如何实现？如何使用？
- 实现为模型结构变更（添加 projection 层并调整 expert 内部结构），无需修改通信框架
- 与 Megatron、Tutel、DeepSpeed-Inference 等现有框架兼容
- r 是超参数（论文设 0.25），需根据通信/计算 trade-off 调整
- 适合 expert parallelism degree 较大的大规模 MoE 训练和推理

涉及论文标题：
- BigMac A Communication-Efficient Mixture-of-Experts Model Structure for Fast Training and Inference

---
