## Retaining Heads / KV Cache Compression with Learned Importance (LOCRET)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Retaining Heads 是 LOCRET（Huang et al., 2024）提出的可训练 KV cache 重要性评分模块。它在每层 transformer 注入一个小型 MLP（两层线性变换，中间维度 $d_{\mathbf{R}}=1024$），接收当前层 $[\mathbf{Q}, \mathbf{K}, \mathbf{V}]$ 的拼接作为输入，输出每个 token 在每个 KV head 的 **Causal Importance Score (CIS)** $\tilde{\mathbf{S}}[k] \in \mathbb{R}^{h/g}$。CIS 反映该 KV cache unit 对未来 token 理解的重要性——高分 token 被保留在 KV cache budget 中，低分 token 在 chunked prefill 过程中被 evict。

核心设计：$\tilde{\mathbf{S}} = \mathbf{R}([\mathbf{Q}, \mathbf{K}, \mathbf{V}]) = \sigma([\mathbf{Q}, \mathbf{K}, \mathbf{V}]\mathbf{W}_1)\mathbf{W}_2$，其中 $\mathbf{W}_1 \in \mathbb{R}^{(d_m + 2d_{kv}) \times d_{\mathbf{R}}}$ 和 $\mathbf{W}_2 \in \mathbb{R}^{d_{\mathbf{R}} \times h/g}$。对 GQA 模型，重要性估计跨 head 联合进行但输出 per-head 分数。训练时 LLM backbone 完全冻结，仅训练 retaining head 参数（占模型总参数 2.5%-8%），训练开销 < 1 GPU 小时。

Ground truth CIS 定义：$\mathbf{S}[k]_j := \max_p (\mathbf{Q}_j \mathbf{K}_j^T)_{p,k}$，即所有 answer token 对该 prefix token k 的最大 pre-softmax attention score。训练 Loss：$\text{Smooth-}\mathcal{L}_1(\tilde{\mathbf{S}}, \mathbf{S}) + \alpha \mathcal{L}_2(\tilde{\mathbf{S}}[k], \tilde{\mathbf{S}}[k+1])$，后者为相邻 token 平滑正则化。训练数据为任意 long-context QA SFT 数据集（LongAlpaca/LongAlign/Anti-Haystack 均可，性能差异极小）。

与 H2O/A2S/SNAPKV 不同，retaining heads 的 CIS 评分是 **causal** 的——仅依赖当前及之前 token，不依赖后续 token。这使其在 chunked prefill 中评分始终准确（无 local-global discrepancy），且与 FlashAttention 完全兼容（不需要 materialize attention matrix）。推理开销极低（Table 20: w/ R 19153 tok/s vs w/o R 20304 tok/s at 4096 ctx，差距来自系统波动）。

从算法pipeline角度拆解术语。

**LOCRET 原始论文中的 Retaining Head 训练与推理完整流程**：

```
// ============ 训练阶段 ============
// 1. 注入 retaining head
for layer i in 1..L:
    R_i = MLP(d_R=1024)   // W1 ∈ R^{(d_m+2d_kv)×d_R}, W2 ∈ R^{d_R×h/g}

// 2. 前向传播收集 ground truth CIS
Q_i, K_i, V_i = layer_i(H_{i-1})   // 正常 attention 前向
// 对每个 prefix token k:
S_i[k]_j = max_p (Q_i_j @ K_i_j^T)_{p,k}
// p 遍历所有 answer token

// 3. Retaining head 预测
Ŝ_i = R_i([Q_i, K_i, V_i])          // 同时输入所有 head 的信息

// 4. 训练（backbone 冻结）
Loss = Smooth-L1(Ŝ_i, S_i) + α * L2(Ŝ_i[k], Ŝ_i[k+1])
// 仅更新 W1, W2

// ============ 推理阶段（chunked prefill + eviction）============
// Hyperparameters: b (budget), B (chunk size), n_s (stabilizers), n_loc (local)
for chunk in chunks(0, L - n_loc, B):
    K_chunk, V_chunk, score_chunk = forward_with_retaining_heads(chunk)
    K_cache = concat(K_cache, K_chunk)
    V_cache = concat(V_cache, V_chunk)
    score_cache = concat(score_cache, score_chunk)
    if not last_chunk:
        score_cache[-n_s:] = +inf     // stabilizers protection
    indices = topk(score_cache, b)    // keep highest CIS
    K_cache, V_cache, score_cache = K_cache[indices], ...

// 处理最后 n_loc tokens（保证不被 evict）
K_cache, V_cache = forward_final(local_tokens, K_cache, V_cache)
output = model.generate(K_cache, V_cache)
```

**LOCRET-Q 变体（query-aware）**：训练时将 query 最后 $l_a$ 个 token 前置到训练序列首部，使 CIS labels 感知 query。推理时 query 在序列首部，确保所有 eviction 感知 query。这使得 LOCRET-Q 在 RULER 等 query-driven benchmark 上可用（75.54% vs LOCRET 34.33%）。

**训练配置**（LOCRET 原始）：
- 数据：LongAlpaca（默认），LongAlign/Anti-Haystack 也可
- 步数：3000 steps, batch_size=1, max_seq_len=10240
- 优化器：AdamW (lr=5e-4, linear scheduler, warmup=2000)
- Loss: Smooth-L1 + α·L2（α=0.0025）
- 训练开销：Phi-3-mini-128K 0.47h, Llama-3.1-8B 0.80h（单 A800）
- Retaining head 参数占比：8% (Phi-3-mini) / 2.5% (Llama-3.1-8B)

术语一般如何实现？如何使用？

Retaining heads 以即插即用方式注入每层 transformer 的 attention block 之后。训练时 backbone 冻结，仅 MLP 参数更新。推理时 retaining head 在每个 chunked prefill step 执行一次 MLP forward，开销可忽略。与 FlashAttention 完全兼容。保留 pre-RoPE KV cache 并从起始位置重新分配连续 position embedding 以增强上下文连续性。支持 MHA 和 GQA 架构。开源：LOCRET 原始 https://github.com/huangyuxiang03/Locret；APB 复用实现 https://github.com/thunlp/APB。

涉及论文标题：
- APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs
- LOCRET: Enhancing Eviction in Long-Context LLM Inference with Trained Retaining Heads on Consumer-Grade Devices

---
