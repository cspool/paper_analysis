## PyramidKV

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

PyramidKV（Cai et al., 2024）是首个提出跨层不均匀 KV cache budget 分配的 KV cache 压缩方法。核心创新源于 Pyramidal Information Funneling 的发现——LLM 底层注意力分散（broad-spectrum），中层逐步收窄（localized），顶层集中在少量关键 token（massive attention）。基于此，PyramidKV 提出：(1) 动态跨层 budget 分配——底层多分配 cache、顶层少分配 cache，按算术序列递减；(2) 基于 instruction tokens（最后 α=8 个 token）的 attention score 进行 token 选择（与 SnapKV 共享基本重要性评估框架）。

原始论文 budget 公式：k^{m-1} = k^{total}/(β·m)（顶层），k^0 = 2·k^{total}/m - k^{m-1}（底层），中间层 k^l = k^0 - (k^0 - k^{m-1})/(m-1) × l（arithmetic sequence）。超参 β=20 控制金字塔陡峭程度，α=8 为各层固定保留的 instruction token 数。该公式确保 Σ_l k^l = k^{total}，各层 budget 严格单调递减。

与后续工作在 PruLong 中的变体——使用 γ 参数化的 budget[l] = base_budget × (1 - l/L)^γ——不同，原始 PyramidKV 的算术序列由 total budget 和 β 解析确定，无需 base_budget 比例。

从算法pipeline角度拆解术语。

**原始 PyramidKV（Cai et al., 2024）Budget Allocation + Selection 流程**：
```
// === Step 1: Budget Allocation (预计算，一次性) ===
// m = 总层数, k_total = 总 KV cache budget
// β = 20 (超参，控制金字塔形状), α = 8 (instruction tokens)

k_top = k_total / (β * m)             // 顶层 budget (最少)
k_bottom = 2 * k_total / m - k_top    // 底层 budget (最多)
Δ = (k_bottom - k_top) / (m - 1)      // 层间递减步长

for l in 0..m-1:
    k_l = k_bottom - Δ * l            // arithmetic sequence
    // e.g., L=32, k_total=2048: k_0≈100, k_31≈10 (不含instruction tokens)

// === Step 2: Attention Score Calculation ===
// Prefill 阶段，对每层每 head:
A = softmax(Q @ K.T / sqrt(d_k))       // [seq_len, seq_len]
for h in 0..H-1:
    s_h = sum(A_h[-α:, :], dim=0)      // 最后α个token对各key的attention sum
    // s_h[i] = Σ_{j ∈ [n-α, n]} A_ij^h

// === Step 3: KV Selection ===
for l in 0..m-1:
    retain instruction tokens (最后α个)
    remaining_budget = k_l - α
    for h in 0..H-1:
        top_indices = topk(s_h, remaining_budget)
        K_selected[l,h] = K[l,h, cat([instruction_indices, top_indices])]
        V_selected[l,h] = V[l,h, cat([instruction_indices, top_indices])]

// === vLLM集成 (Appendix R) ===
// Per-layer block table: 每个sequence的block table扩展为每层独立
// 解决uniform eviction在小budget下的fragmentation
```

**Ablation: 算术 vs 几何 vs 指数衰退**（Table 4, LLaMa-3-8B, KV size=64）：
- Linear (PyramidKV): LongBench avg 34.76
- Geometric decay: 34.36
- Exponential decay: 34.23
- Entropy-based adaptive: 32.71
- Gini-based adaptive: 32.58
结论：线性算术序列最匹配观察到的注意力渐进收窄模式，且计算开销最小。

术语一般如何实现？如何使用？

开源：https://github.com/Zefan-Cai/PyramidKV（官方实现），支持 Flash Attention v2 和 SDPA attention，包含 PyramidKV、SnapKV、H2O、StreamingLLM 四种方法的统一实现。支持 LLaMA-3-8B/70B-Instruct、Mistral-7B 等模型。

使用方法（from official repo README）：`python -m longbench.pred --model llama3-8b --method pyramidkv --env_conf config/llama3-8b/pyramidkv.json`。配置文件指定 KV cache size、α (window size)、β (pyramid steepness)。

PruLong 论文后续实现了其 Chunked Eviction 变体——Patched PyramidKV + mean-pool 在 RAG（<34% KV footprint）上取得最优结果。该变体使用 (1 - l/L)^γ 参数化（而非原始算术序列），always-retained window=64。

WindowKV（Zuo et al., 2025）将 PyramidKV 的逐层 budget 分配扩展到**组级别（group-level）**：将 m 层分为 H=m/γ 组，使用相同算术序列公式跨组分配 budget，组内各层均匀共享。同时将 token 级选择替换为 window 级选择，并引入任务自适应分类器决定每窗口中保留的 token 比例 p（localization: p=ω；aggregation: p<ω）。

涉及论文标题：
- PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling
- Cache Me If You Can: How Many KVs Do You Need for Effective Long-Context LMs
- WindowKV: Task-Adaptive Group-Wise KV Cache Window Selection for Efficient LLM Inference

---
