## Lory: Fully Differentiable Mixture-of-Experts for Autoregressive Language Model Pre-training

- 属于算法pipeline的实现是什么？实验比较什么？
  - Lory 提出三种算法 pipeline 创新，实现完全可微的 MoE 自回归语言模型预训练：
    1. **Fully Differentiable Expert Merging**：在参数空间（而非激活空间）对专家进行软合并。给定路由权重 e_i，合并后的 FFN 参数为 θ̄ = Σ_i e_i · θ_i，然后用合并后的 FFN 处理输入：o_x = FFN(h_x; θ̄)。与稀疏 MoE 的 top-k 离散路由不同，整个过程完全可微，无需离散路由决策和辅助负载均衡损失。
    2. **Causal Segment Routing**：将输入序列分为 T=256 token 的固定长度段。对于第 k 段 S_k (k>1)，使用前一段 S_{k-1} 的隐藏表示平均值 h̄_{k-1} 计算路由权重并合并专家 FFN，然后用合并后的 FFN 处理当前段所有 token。对于第一段 S_1，使用自身段表示但施加 stop-gradient 防止信息泄露。推理时仅用 prompt 做一次路由决策，后续生成全程使用合并后的 FFN。
    3. **Similarity-based Data Batching**：使用 Contriever 计算文档相似度，通过贪心搜索算法将语义相似的文档拼接成训练实例，使相邻段来自相似领域，促进专家按领域/主题专业化。
  - 实验比较：
    - Lory MoE 模型 vs 参数匹配的 Dense 模型（0.3B/1.5B active params, 8/16/32 experts）
    - Lory vs Expert Choice (EC) MoE（段级路由和 token 级路由两种变体，capacity factor=1）
    - Ablation：causal segment routing vs prefix routing；similarity-based batching vs random batching
    - 扩展实验：7B/4E 模型（无 similarity batching）
    - 主要结果：0.3B/32E 在 Books 上 perplexity 改善 +13.9%，下游任务 averaged improvement +3.7% (commonsense), +3.3% (reading), +1.5% (QA), +11.1% (classification)

- 硬件平台是什么，配置是什么。
  - **训练**：最多 64 块 NVIDIA A100 GPU
  - **分布式训练**：数据并行 + ZeRO 优化（Rajbhandari et al., 2020）
  - 训练吞吐量（Table 3，A100）：0.3B dense 29,000 tokens/s/gpu；0.3B/8E 24,500；0.3B/16E 22,900；0.3B/32E 20,800
  - 软件环境：论文未明确说明具体 CUDA/PyTorch 版本

- 模型是什么。数据集和bench分别是什么。
  - **模型架构**（Table 4）：
    - 0.3B：24 layers, hidden dim 1024, 16 attention heads。MoE 变体 0.3B/8E (1.8B total), 0.3B/16E (3.5B), 0.3B/32E (6.8B)
    - 1.5B：48 layers, hidden dim 1536, 24 attention heads。MoE 变体 1.5B/8E (7.8B total), 1.5B/16E (15.0B), 1.5B/32E (29.5B)
    - 7B (extended)：32 layers, hidden dim 4096, 32 attention heads。7B/4E (19.7B total)
    - 解码器仅 Transformer，所有 FFN 层替换为 MoE 层，SwiGLU 激活，LLaMA tokenizer，context window 4096
  - **训练数据集**：CommonCrawl 的 150B token 随机子集（Wenzek et al., 2019），使用 similarity-based batching 构造训练实例。7B 实验使用 LLaMA2 的多语料混合的 200B token 子集，使用随机 batching。
  - **评估 Benchmark**：
    - **语言建模（Perplexity）**：arXiv, Books, Wikipedia, C4, Python code（各 1000 样本，4096 tokens/sample）
    - **Commonsense Reasoning**：BoolQ, PIQA, SIQA, HellaSwag, WinoGrande
    - **Reading Comprehension**：RACE-m, RACE-h, ARC-easy, ARC-challenge
    - **Closed-book QA**：Natural Questions, TriviaQA
    - **Text Classification**：AGNews, SST-2, Amazon, Yelp, FEVER, MRPC
  - 训练配置：AdamW (β1=0.9, β2=0.95), lr=2e-4, cosine schedule, batch size 1M tokens, 前 5% 训练步为 dense warmup 阶段（先训练参数匹配的 dense 模型，再复制 FFN 层初始化 MoE）

- 开源情况。基于开源文档和论文，使用例子解释算法pipeline，至少具体到伪代码或张量计算。
  - **未开源**：论文未提供代码仓库链接，arXiv 页面和相关搜索均未找到开源实现。
  - **Causal Segment Routing 伪代码**（论文 Algorithm 1, Appendix A）：
```python
# B: batch size, L: sequence length, d: hidden dim
# E: number of experts, T: segment length (256)
# R: routing network (linear layer)
input: x  # (B, L, d)
N = L // T                     # number of segments per sample
seg_x = x.view(B*N, T, d)      # split into segments
repr = mean(seg_x, dim=1)      # segment representations (B*N, d)
e = softmax(R(repr), dim=-1)   # routing weights (B*N, E)
e_first = e.view(B, N, E)[:, 0] # first segment routing
e = roll(e, 1)                  # shift by 1 -> causal
e = e.view(B, N, E)
e[:, 0] = stop_grad(e_first)   # first segment uses own repr (no leakage)
e = e.view(B*N, E)
seg_y = moe_ffn(seg_x, e)      # merged FFN computed per segment
y = seg_y.view(B, L, d)        # back to instance view
```
  - **moe_ffn 函数实现（参数空间合并）**：
```python
def moe_ffn(seg_x, e):
    # seg_x: (B*N, T, d), e: (B*N, E)
    # experts: list of E expert FFNs, each = (W_gate, W_up, W_down)
    merged_W_gate = sum(e[:, i] * expert[i].W_gate for i in range(E))
    merged_W_up   = sum(e[:, i] * expert[i].W_up   for i in range(E))
    merged_W_down = sum(e[:, i] * expert[i].W_down for i in range(E))
    # SwiGLU FFN:
    gate = silu(matmul(seg_x, merged_W_gate))
    up   = matmul(seg_x, merged_W_up)
    out  = matmul(gate * up, merged_W_down)
    return out
```
  - 关键张量流（单 MoE 层，L=4096, T=256, E=32）：
    1. 输入 x (4096 tokens) → 分为 16 段 → 每段 256 tokens
    2. Segment 0 的 avg representation h̄_0 计算路由权重 e_0 (32-dim softmax) → stop_gradient → 合并 FFN_0 → 处理 Segment 0
    3. Segment 0 的 avg representation h̄_0 计算路由权重 e_1 → 合并 FFN_1 → 处理 Segment 1（causal shift）
    4. 依此类推，每个 segment 使用前一段的表示计算路由
    5. 合并操作 FLOPs 开销：E/T × (FFN FLOPs)，E=32, T=256 → ~12.5% 额外计算 vs Dense（MoE 层）。总模型开销更小（~15-28% 训练减速，Table 3）
  - 推理时：给定 prompt → 每层用 prompt 的平均隐藏表示计算一次路由权重 → 合并 FFN → 后续所有生成 token 使用该合并 FFN，与 Dense 模型推理效率相同
  - 合并操作仅每段执行一次（L/T 次），而非每 token 执行一次，这是 segment-level routing 的关键效率优势
