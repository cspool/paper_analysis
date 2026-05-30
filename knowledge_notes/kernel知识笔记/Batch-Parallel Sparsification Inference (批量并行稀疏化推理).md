## Batch-Parallel Sparsification Inference (批量并行稀疏化推理)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Batch-Parallel Sparsification Inference 是 Dynamic-LLaVA 为实现 mini-batch 内变长 token 集合的 GPU 并行预测和计算而设计的优化策略。挑战：不同样本的 image/text token 数量不同（变长），且 predictor 的稀疏化使 token 集合长度进一步分化，传统 padding-to-max 方法会导致大量无效计算。Dynamic-LLaVA 通过 Left Padding（零填充在左侧）+ TopkArgmax（基于 predictor score 保留固定比例 token）实现批量并行。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。

**Batch-Parallel Prefill 流程（Eq. 11）**：

```
// 输入: B 个样本，image token 数量分别为 N_l^{I(1)}, N_l^{I(2)}, ..., N_l^{I(B)}
// max(N_l^I) = 最大图像 token 数

// Step 1: Left Padding 对齐
for b in 1..B:
    pad_len = max(N_l^I) - N_l^{I(b)}
    S_l^{I(b)}_padded = [zeros(pad_len, d); S_l^{I(b)}]  // 零填充在左侧
// S_l^I: [B, max(N_l^I), d] 连续 tensor, GPU 友好

// Step 2: 批量 predictor 推理
D^I = P^I(S_l^I)                                    // [B, max(N_l^I), 2]
scores = D^I[:, :, 1]                                // 第二维做 keep score

// Step 3: TopkArgmax——按比例保留（而非全局 max）
for b in 1..B:
    k_b = floor(r^I * N_l^{I(b)})                   // 每样本保留数量
    // 仅对非 padding 区域取 top-k
    valid_scores = scores[b, pad_len:]               // 去除 left padding
    topk_idx = TopkArgmax(valid_scores, k_b)         // 取分数最高的 k_b 个
    S_l^{I*(b)} = S_l^{I(b)}[topk_idx]              // 保留的 tokens

// Step 4: 再次 Left Padding 对齐缩减后的 token 集
max_len = max(|S_l^{I*(b)}| for b in 1..B)
S_l^{P*} = [LPadding(S_l^{I*(b)} ∪ S_l^{T(b)}) for b in 1..B]  // [B, max_len, d]

// Step 5: 后续层正常批量计算
for l in l+1..L:
    S_{l+1}^{P*} = TransformerLayer(S_l^{P*})       // 标准 batch forward
```

**Batch-Parallel Decoding w/ KV Cache 流程（Eq. 12）**：

```
// 每个 sample 的 KV cache 独立存储
KV_batch = {{S_l^{K(b)}, S_l^{V(b)}} | b=1..B}

// 对每个 batch 的当前 token
S_l^{OT} = LPadding([S_l^{OT(1)}, ..., S_l^{OT(B)}])  // [B, max(N^{OT}), d]
D^{OT} = P^{OT}(S_l^{OT})                              // [B, max(N^{OT}), 2]
M^{OT(b)} = argmax(D^{OT(b)})                          // 批量预测

// KV cache 更新: padded KV 用于 Attention
S_l^{K} = LPadding([S_l^{K(1)}, ..., S_l^{K(B)}])     // [B, max_K_len, d]
S_l^{V} = LPadding([S_l^{V(1)}, ..., S_l^{V(B)}])
O = Attention(Q, S_l^{K}, S_l^{V})                     // batch attention
```

术语一般如何实现？如何使用？

Left Padding vs Right Padding 的选择：Left Padding 确保实际 token 在张量右侧连续排列，便于去除 padding 后做 TopkArgmax（仅取有效区域的 score）。训练时通过约束正则项 R（Eq. 10）使每个样本的保留比例接近 r^I 和 r^OT，从而推理时 mini-batch 内各样本的实际 token 数量相差不大，减少 padding 浪费。实测 batch=8 的并行效率在 A100 80G 上可充分利用 GPU 并行度。

涉及论文标题：
- Dynamic-LLaVA: Efficient Multimodal Large Language Models via Dynamic Vision-language Context Sparsification
