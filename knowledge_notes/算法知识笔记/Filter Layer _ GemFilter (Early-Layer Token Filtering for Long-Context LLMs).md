## Filter Layer / GemFilter (Early-Layer Token Filtering for Long-Context LLMs)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Filter Layer（过滤层）是 GemFilter 论文提出的核心概念：利用 LLM 的早期 Transformer 层作为输入 token 重要性过滤器，在仅运行前 r 层（而非全部 m 层）后即可识别与 query 相关的重要 token，从而将长上下文输入从 n 个 token 压缩至 k 个 token（如 128K→1024，约 1000× 压缩率），再将压缩后的 token 子序列送入完整 LLM 进行生成。

核心发现：LLM 在早期层（如 LLaMA 3.1 8B 的第 13 层）的 attention 矩阵中就能定位 answer-related tokens——即模型在生成答案之前就已"知道"哪些输入 token 对回答是重要的。这一发现将 prompt computation 的计算量从 Θ(mhn²d) 降至 Θ(rhn²d)（r << m），同时将第二遍推理的序列长度从 n 降至 k，全流程显著加速。

GemFilter 是一个 training-free 方法，不需要任何微调或额外的模型参数，与任何 Transformer LLM 兼容。其算法流程由两次前向传递组成：(1) **第一遍（Filter Pass）**：仅运行前 r 层，在第 r 层提取所有 attention head 的最后一 query token 对全部 key token 的 attention scores，跨 head 求和后取 top-k 最高分的 token 索引；(2) **第二遍（Generation Pass）**：将选中的 k 个 token（按原始顺序排列）送入完整 m 层 LLM 做标准 generation。

与 SnapKV/H2O 的关键差异：GemFilter 在 prompt computation 阶段就减少了计算量（仅运行前 r 层），而 SnapKV/H2O 在 prompt computation 阶段仍需处理全部 m 层和全部 n 个 token。此外 GemFilter 使用单一的全局 token 索引集 J（可打印供人类审查），而 SnapKV/H2O 使用 m·h 套索引。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**GemFilter 两遍推理伪代码（对应 Algorithm 1）**：

```
// ====== Input ======
// tokens T ∈ V^n (n=128K), filter layer index r=13, topk k=1024
// F_{1:m}: m-layer transformer (m=32 for LLaMA 3.1 8B)

// ====== First Pass: Filter (仅前 r 层, Prompt Computation) ======
F_{1:r}(T) → 获取第 r 层输出

// 提取第 r 层的 query 和 key（多 head）
Q^{(r)} = [Q^{(r,1)}, ..., Q^{(r,h)}]   // 每 head: [n, d_h]
K^{(r)} = [K^{(r,1)}, ..., K^{(r,h)}]   // 每 head: [n, d_h]

// 取最后一 query token Q_n^{(r)} ∈ R^{h×d_h}
// 计算对全部 key token 的 attention scores（跨所有 head 求和）
for j in 1..h:
    scores_j = Q_n^{(r,j)} @ K^{(r,j)^T}   // [1, n] per head

total_scores = sum(scores_j for j in 1..h)   // [1, n]
total_scores = avg_pool1d(total_scores, kernel=5)  // 聚类平滑

J = topk_index(total_scores, k)
J = sort(J)   // 按原始顺序：确保 <bos> 在最前

// ====== Second Pass: Full Generation (完整 m 层) ======
T_J = T[J]   // 仅 k=1024 个 token（vs 原始 128K）
output = Gen(F_{1:m}, T_J)   // 标准 greedy generation
// RoPE 位置编码重新计算，最大距离 = k（而非 n）
```

**时间复杂度对比（Theorem 3.3）**：

| Phase | Standard Attention | SnapKV/H2O | GemFilter |
|-------|-------------------|-----------|-----------|
| Prompt Computation | Θ(mhn²d) | Θ(mhn²d) | **Θ(rhn²d)** |
| Iterative Generation | Θ(mh(nt+t²)d) | Θ(mh(kt+t²)d) | Θ(mh(k²+t²)d) |

**GPU 内存对比（Theorem 3.3）**：

| Phase | Standard | SnapKV/H2O | GemFilter |
|-------|----------|-----------|-----------|
| Prompt Comp | mw + 2mhnd | mw + 2hnd + 2mhkd | **rw + 2hnd** |
| Iterative Gen | mw + 2mh(n+t)d | mw + 2mh(k+t)d | mw + 2mh(k+t)d |

n=128K, k=t=1024, r=13, m=32 时：Prompt Time = Standard:SnapKV:GemFilter = 32:32:13 → ~60% 减少；Prompt Memory = mw+2mhnd : mw+2hnd+2mhkd : rw+2hnd。

术语一般如何实现？如何使用？

基于 HuggingFace Transformers v4.43 PyTorch 实现，仅需在 attention forward 中添加 `find_context` 调用。核心函数：(1) `find_context()`: 在 filter layer 提取 last-query-key scores → topk → sort；(2) `top_index()`: Q_n^T K 跨 head 求和 + avg_pool1d + topk。依赖 `transformers==4.43.3` 和 `flash-attn==2.6.3`。

使用示例：`python needle_eval.py --model <hf_id> --modified gemfilter --topk 1024 --ctx_len 32000`

Filter Layer 选择：LLaMA 3.1 8B (32 layers): r=13; Mistral Nemo 12B (40 layers): r=19; Phi 3.5 Mini 3.8B (32 layers): r=19。消融显示 layer 13-25 之间性能鲁棒。

核心性能（LLaMA 3.1 8B, H100-80GB）：2.4× speedup vs SnapKV，GPU 内存 -30% vs SnapKV / -70% vs Standard。Needle in a Haystack (128K) 上显著优于 Standard 和 SnapKV。LongBench 上与 SnapKV/H2O 可比。

代码开源：https://github.com/SalesforceAIResearch/GemFilter

涉及论文标题：
- Discovering the Gems in Early Layers: Accelerating Long-Context LLMs with 1000x Input Token Reduction

---
