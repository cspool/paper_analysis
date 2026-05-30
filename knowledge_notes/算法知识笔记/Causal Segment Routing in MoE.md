## Causal Segment Routing in MoE

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Causal Segment Routing 是 Lory 提出的段级路由策略，用于在自回归语言模型中实现可微 MoE 的高效训练。核心思想：将 token 级路由替换为段级路由——将输入序列分为固定长度 T=256 的段，每段仅做一次路由决策和专家合并，使用前一段的隐藏表示计算当前段的路由权重（因果性），避免信息泄露。

动机：如果对每个 token 做一次专家合并（naive extension of SMEAR），合并计算开销为 O(L · E · d · d_ffn)，对于 L=4096 训练序列不切实际。段级路由将合并次数从 L 降为 L/T（对 T=256 为 16 次），额外 FLOPs 仅 E/T。

Causal shift 机制确保自回归因果性：
- Segment S_k (k>1)：使用 S_{k-1} 的隐藏表示平均值 h̄_{k-1} 计算路由权重，然后合并 FFN 处理 S_k
- Segment S_1：使用自身表示 h̄_1 计算路由，但施加 stop-gradient 防止信息泄露
- 推理时：仅用 prompt 做一次路由决策，后续生成使用同一合并 FFN

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Causal Segment Routing 的 PyTorch 风格伪代码（论文 Algorithm 1）：

```python
# input: x (B, L, d), segment size T
# R: routing network (linear layer)
N = L // T  # number of segments

# Step 1: Split into segments and compute segment representations
seg_x = x.view(B*N, T, d)            # (B*N, T, d)
repr = mean(seg_x, dim=1)            # (B*N, d) avg per segment

# Step 2: Compute routing weights for ALL segments (non-causal)
e = softmax(R(repr), dim=-1)         # (B*N, E)

# Step 3: Make routing causal by shifting
e_first = e.view(B, N, E)[:, 0]      # first segment routing
e = roll(e, 1)                        # shift: segment k uses segment k-1's routing
e = e.view(B, N, E)
e[:, 0] = stop_grad(e_first)         # first segment uses own repr (no leakage)
e = e.view(B*N, E)

# Step 4: Expert merging + FFN computation
seg_y = moe_ffn(seg_x, e)            # merged FFN per segment (see Fully Differentiable MoE entry)
y = seg_y.view(B, L, d)              # reshape back
```

**推理时的 Prompt-only Routing**：
```python
# input prompt: x_prompt (1, L_prompt, d)
repr_prompt = mean(x_prompt, dim=1)  # (1, d)
e = softmax(R(repr_prompt), dim=-1)  # (1, E) single routing decision
theta_merged = sum(e[i] * expert[i].params for i in range(E))
# All subsequent generated tokens use theta_merged (same as Dense FFN)
```

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现关键：
- **段大小 T=256**：论文通过实验选定，平衡合并效率（更大 T 减少合并次数）和路由粒度（更小 T 使路由更细粒度）。T=256 在 L=4096 下产生 16 段。
- **Stop-gradient**：第一段使用自身表示的路由权重来自 stop_grad(e_first)，防止该段 token "看到未来"信息。这是 causal 属性的关键技术细节。
- **Segment representation**：使用段内所有 token 的 hidden state 平均值，而非 [CLS] 或其他聚合方式。均值操作使模型在推理时能适应不同长度的 prompt。
- **与 Prefix Routing 对比**：Prefix routing 仅用第一个段路由整个序列（类似 SMEAR），性能显著差于 causal segment routing（图 3），证明每个段提供路由训练信号的重要性。
- **推理 train-test gap**：segment-level routing 和 prompt-only routing 在下游任务上差异不显著（Table 9），前者为训练设计，后者为推理简化。
- **与 Token-level MoE routing 对比**：Token-level routing（如 Expert Choice）学到的是浅层词法特征（标点、冠词），segment routing 学到的是领域级特征（arXiv, Python, Books 等）。

涉及论文标题：
- Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training
