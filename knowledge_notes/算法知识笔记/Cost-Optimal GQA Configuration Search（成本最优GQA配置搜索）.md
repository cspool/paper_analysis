## Cost-Optimal GQA Configuration Search（成本最优GQA配置搜索）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Cost-Optimal GQA Configuration Search 是本文提出的三步搜索过程，用于在给定目标 loss L* 和推理上下文长度 T 的条件下，找到推理成本最低的 GQA 配置 (n_h, n_kv, N)。不同于传统方法将 n_h 由 d 唯一确定、n_kv 固定为 8、N 独立选择，该方法联合搜索这三个超参数以最小化硬件感知推理成本。

从算法pipeline角度拆解术语：

**三步搜索过程**：

```
// Step 1: Candidate Selection
// 定义候选 GQA 配置集
max_d = 1536  // 最大模型的 hidden size
H_cand = []
for nh in {1, 2, 4, 8, 16, 32}:
    for nkv in {1, 2, 4, 8, 16, 32}:
        if nkv <= nh:
            H_cand.append((nh, nkv))
// |H_cand| = 21（max(d)/d_h = 32, d_h=64）

// Step 2: Scaling Curves Fitting（T=8K 短上下文训练）
for each H in H_cand:
    for N in [3M, 19M, 85M, 150M, 200M, 470M, 680M, 1.2B]:
        model = build(N, H.nh, H.nkv)
        loss = train(model, SlimPajama, ratio=20:1 tokens/param)
    // 拟合 power-plus-constant 函数
    L(N; H) = (a_H / N)^{b_H} + E
    // R² > 0.999
    // E 为语言自然熵，跨配置共享

// Step 3: Cost Minimization
Input: target loss L*, context length T
for each H in H_cand:
    // 从 scaling curve 反求满足 L* 的最小 N
    N*(H) = a_H / (L* - E)^{1/b_H}
    // 计算推理成本
    C_infer = 2N* + 4TL·d_h·H.nh
    M_infer = N* + 2TL·d_h·H.nkv
    // 硬件感知综合成本 (λ=0.9 偏重 memory)
    Z(H) = 0.9 · M_infer^{1/2} + 0.1 · C_infer^{1/3}

H* = argmin Z(H)
return (N*(H*), H*.nh, H*.nkv)
// N* 为连续值，通过线性插值确定 (L,d)，实际部署取最接近整数配置
```

**为什么可以用 T=8K 外推至 T=128K**：
实验验证 T 对 loss 的影响与 N 和 H 相独立（Section 5.7）——相对 loss 差异 ΔL(T) 在 T>8K 后波动 <1%。因此 Step 2 仅需在 T=8K 下训练小模型，Step 3 将 T 代入成本公式即可外推。

**核心发现**：
- 长上下文下应使用更少的 head + 更大的模型（T=128K, L*=2.615 → H*=(8,1), N*=1.8B）
- Llama-3 GQA (d/dh, 8) 仅对特定 (L*, T) 最优，多数情况下 suboptimal
- n_h 比 n_kv 对 loss 更重要（相同参数增量下 n_h 增加带来更大 loss 降低）
- 对齐训练 FLOPs 时，用更少 head 可获更多训练数据，优势更大

术语一般如何实现？如何使用？

实际部署步骤：(1) 离线运行 Step 1-3 获得 (N*, n_h*, n_kv*)；(2) 选择与 N* 最接近的实际配置（通过 Table 7 的预定义 aspect ratio 插值）；(3) 用该配置从头训练模型（或从已有模型 up-training）。该方法与现有 serving 系统完全兼容——仅改变模型配置，无需修改框架或 kernel。代码开源：https://github.com/THUNLP/cost-optimal-gqa。

涉及论文标题：
- Cost-Optimal Grouped-Query Attention for Long-Context LLMs

---
