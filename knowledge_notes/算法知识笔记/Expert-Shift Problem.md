## Expert-Shift Problem

术语解释
Expert-Shift 是 MoE-LLM 低比特量化后出现的路由器输出偏移问题：量化引入的噪声导致 MoE router 的 expert 选择概率分布发生偏移，使模型选错 expert，造成显著的性能退化。这是 MoE 量化中除权重重构误差之外的第二个关键退化因素。

术语是什么？
在 EAC-MoE 论文中，作者通过受控实验分离了两个退化因素：(1) 量化本身导致的权重重构误差；(2) 量化引起的 expert-shift。实验将 FP16 模型和 3-bit 量化模型在 WikiText-2 上记录所有输入的 expert 选择及对应分数，交叉施加条件后测量 PPL：

| 条件 | Mixtral-8x7B PPL | Deepseek-moe-16b PPL |
|------|-----------------|---------------------|
| FP16 + 正确选择 | 3.84 | 6.51 |
| FP16 + Expert-Shift | 4.17 | 6.76 |
| 3-bit + 正确选择 | 4.21 | 6.81 |
| 3-bit + Expert-Shift | 4.65 | 7.17 |

结果表明：(1) 单纯的权重重构误差使 PPL 从 3.84 升至 4.21（+0.37）；(2) Expert-Shift 更进一步恶化至 4.65（+0.44，占总退化的 ~54%）；(3) 即使 FP16 模型被强制使用量化模型的错误 expert 选择，PPL 也显著退化（3.84→4.17）。

Expert-Shift 的根因：量化后的 MHSA 和已量化 expert 的激活值（x̂）与原全精度激活值（x）存在偏差，导致 router 计算 W_r·x̂ 偏离 W_r·x，Softmax 后的概率分布改变，top-K 选择结果发生变化。由于逐层传播，expert-shift 会在深层累积放大。

从算法pipeline角度拆解术语：
```
=== Expert-Shift 如何发生（逐层传播）===
输入: token x (FP16)
Layer l:
    1. x_out = Quantized_MHSA(x)        # MHSA 被量化，输出激活有偏差
    2. logits = Router_W @ x_out         # [num_experts]，但 x_out 已含噪声
    3. probs = Softmax(logits)            # 概率分布因噪声偏移
    4. selected = TopK(probs, K)          # 可能选错 expert（shifted experts）
       # 95.9% 的 shifted expert 仍在 top-16 概率内（64 expert 中）
    5. output = Σ probs[i] * Quantized_Expert_i(x_out)  # 用错误的 expert 计算
    → 第 l+1 层继承错误的 hidden state，expert-shift 继续传播
```

术语一般如何实现？如何使用？
- Expert-Shift 是 MoE-LLM 量化特有的退化机制，dense LLM 不存在此问题
- 量化位宽越低，expert-shift 越严重（2-bit >> 3-bit >> 4-bit）
- EAC-MoE 的 QESC 方法通过 TopK-MSE Loss 逐层校准 router 来缓解 expert-shift
- 在量化校准中保持 router 全精度（router 仅占 <0.03% 参数，不增加显著内存开销）
- 量化 MHSA 的位宽从 2→4→8 bit 提升会显著降低 expert-shift rate（MHSA 4-bit 以上变化趋缓）

涉及论文标题：
- EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
