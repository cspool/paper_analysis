## Native Sparse Attention (NSA)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Native Sparse Attention (NSA) 是 DeepSeek-AI 提出的一种可原生训练的稀疏注意力机制，通过动态分层稀疏策略替代 Transformer 中标准 Full Attention 的 O(t²) 密集计算。NSA 将每个 query 的 key/value 序列按时间分块（temporal blocks），经三条并行注意力路径处理：(1) **Token Compression（粗粒度压缩）**：通过可学习 MLP φ（含 intra-block position encoding）将连续 key/value block 压缩为块级紧凑表示 $\tilde{K}_t^{\text{cmp}}, \tilde{V}_t^{\text{cmp}}$（block length l=32, stride d=16），捕获全局高层语义，计算成本 O(t·l/d²)≈O(t/16)；(2) **Blockwise Token Selection（细粒度选择）**：利用压缩注意力的中间 softmax 分数 $\mathbf{p}_t^{\text{cmp}}$（免费获得）推导 selection block（l'=64）的重要性分数，经 Top-n（n=16）选出最重要的连续 token block 保留精细信息；(3) **Sliding Window（局部窗口）**：独立 512 token 局部窗口分支隔离局部模式学习，防止 shortcuts 压制全局路径。三条路径输出通过输入依赖的可学习门控 $g_t^c = \text{Sigmoid}(\text{MLP}_g(\mathbf{q}_t))$ 融合，且使用独立 K, V 投影矩阵（共 3 组而非 1 组）防止跨路径梯度干扰。

关键特性：(a) **全生命周期覆盖**——训练/prefilling/decoding 三个阶段均支持稀疏加速，不同于 H2O/Quest 等方法仅支持推理稀疏；(b) **端到端可训练**——所有算子可微，Top-n selection 在 forward 做离散选择、backward 仅对选中 block 的非零 attention 传梯度，形成隐式 straight-through estimation；(c) **硬件对齐**——blockwise 连续内存访问匹配 Tensor Core 需求，GQA group 内跨 head 共享 KV block 选择消除冗余传输。论文报告在 27B GQA+MoE 模型上：pretrain loss 低于 Full Attention，9 个通用 benchmark 中 7 个超越 Full Attention，LongBench 平均分 0.469（超 Full Attention +0.032），64k Needle-in-a-Haystack 完美检索，AIME 推理 +0.054~0.075。Kernel 层面：64k forward 9.0×/backward 6.0× speedup（Triton vs FA2 Triton），解码预期 11.6× speedup。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。

**NSA 完整 Attention 计算流程（per query token t）**：

```
输入: q_t ∈ R^{d}, K_cache, V_cache ∈ R^{t×d}  (t 为历史序列长度)
超参: l=32 (compression block), d=16 (stride), l'=64 (selection block), n=16, w=512

// === Branch 1: Token Compression ===
num_comp_blocks = floor((t - l) / d) + 1
for i in range(num_comp_blocks):
    K_block = K_cache[i*d : i*d+l]  // [l, d_k]
    K_cmp[i] = phi(K_block)          // [1, d_k], phi 为 MLP + intra-block PE
// 同理得 V_cmp ∈ R^{num_comp_blocks × d_v}
p_cmp = softmax(q_t @ K_cmp^T / sqrt(d_k))         // [1, num_comp_blocks]
o_cmp = p_cmp @ V_cmp                               // [1, d_v]
// ← p_cmp 被免费复用于 Step 2 的 block 重要性推导

// === Branch 2: Blockwise Token Selection ===
// 由 p_cmp 推导 selection block 重要性（共享 blocking scheme 时 p_slc = p_cmp）
p_slc = aggregate(p_cmp, stride=l'/d)               // [1, t/l']
// GQA: 跨 group 内所有 heads 聚合
p_slc_shared = sum_{h=1..H} p_slc^(h)              // [1, t/l']
I_t = topk_indices(p_slc_shared, n)                // 选 n=16 个 block
K_sel = concat([K_cache[i*l' : (i+1)*l'] for i in I_t])  // [nl', d_k]
V_sel = concat([V_cache[i*l' : (i+1)*l'] for i in I_t])
s_sel = q_t @ K_sel^T / sqrt(d_k)                   // [1, nl']
o_sel = softmax(s_sel) @ V_sel                      // [1, d_v]

// === Branch 3: Sliding Window ===
K_win = K_cache[t-w:t]  // [w, d_k]
V_win = V_cache[t-w:t]  // [w, d_v]
o_win = softmax(q_t @ K_win^T / sqrt(d_k)) @ V_win  // [1, d_v]

// === Gated Fusion ===
g_cmp, g_sel, g_win = sigmoid(MLP_gate(q_t))  // 各 ∈ [0,1]
o_t = g_cmp * o_cmp + g_sel * o_sel + g_win * o_win
// 总 KV 访问量 ≈ t/16 + 1024 + 512 ≪ t (长序列)
```

**Prefill 阶段**（训练时并行处理所有 t）：所有 query positions 共享同一套 compression K/V（仅需计算一次），selection block 索引 per-position 不同（需各算各的），window 天然 per-position。压缩和 window 分支复用 FlashAttention-2 kernel，selection 分支用 NSA 专用 group-centric kernel。

**解码阶段**（自回归）：每步只需加载 ~t/16 + nl' + w 个等效 token 量的 KV cache（64k 时 ≈5632 vs Full Attention 65536），memory access 量降 11.6×。

术语一般如何实现？如何使用？

NSA 在 Triton 上实现。Compression attention 和 window attention 直接复用 FlashAttention-2 kernel。Selection attention 使用 NSA 专用 group-centric kernel（详见 kernel调度 分层）。训练时 selection 的 Top-n 操作 forward 做离散 mask（仅计算选中 block 的 attention），backward 时因非零 attention score 梯度自然传播（隐式 STE），无需额外辅助 loss。压缩 MLP φ 与 backbone 联合训练。门控 MLP_g 输出经 sigmoid 约束在 [0,1]。

论文开源情况：DeepSeek-AI 出品，arXiv:2502.11089（ACL 2025 Best Paper），已在 DeepSeek V3.2-Exp 中采用。实现与标准 Transformer 训练流程兼容，可替换现有 attention 层。

涉及论文标题：
- Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention

---
