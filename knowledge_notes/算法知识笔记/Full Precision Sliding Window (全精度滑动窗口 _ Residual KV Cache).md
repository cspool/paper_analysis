## Full Precision Sliding Window (全精度滑动窗口 / Residual KV Cache)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Full Precision Sliding Window（Residual KV Cache）是 KIVI 中保留最近 R 个 token 的 Key/Value 在 FP16 精度、不进行量化的设计。KIVI 将 KV cache 分为 grouped 部分（量化）和 residual 部分（FP16），residual 部分大小上限为 R（通常 R=128），形成大小为 R/2（key）和 R（value）的全精度滑动窗口。

该设计的关键作用：对于 GSM8K 等数学推理等硬任务，fake 2bit 全量化准确率显著下降（Llama-2-7B GSM8K: 16bit=13.50, fake 2bit=5.76），但 KIVI-2（含 residual window）可达 12.74（仅约 1% 下降）。全精度滑动窗口保留局部重要 token 的精确 attention 计算能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Residual KV Cache 的管理逻辑：

```
初始化: R = 128 (residual length), G = 32 (group size)
new_tokens = 0

# 每次 decoding step:
X_K_r = Concat([X_K_r, t_K])   # new token added in FP16
X_V_r = Concat([X_V_r, t_V])   # newly arrived, keep FP16

if len(X_K_r) == R:
    # Residual满了, 将整组R个token量化并移入grouped
    Q(X_K_r) = KeyQuant(X_K_r)              # per-channel quant
    Q(X_K_g) = Concat([Q(X_K_g), Q(X_K_r)]) # merge into grouped
    X_K_r = empty                            # reset residual

if len(X_V_r) > R:
    # Value: 保留最近R个, 超出的量化移入grouped
    Q_outdated = GroupQuant(X_V_r[:-R], dim=token)
    Q(X_V_g) = Concat([Q(X_V_g), Q_outdated])
    X_V_r = X_V_r[-R:]   # keep latest R tokens FP16

# Attention: combined from both parts
A = Concat([t_Q @ Dequant(Q(X_K_g))^T,  t_Q @ X_K_r^T])
# Softmax split → weighted sum from quantized V_g + FP16 V_r
```

记忆开销分析：当 R ≤ 128 且序列长度 l ≫ R 时，residual FP16 部分的额外内存占总 KV Cache 比例 ≈ 128/l。例如 l=8192, R=128 时额外开销仅 1.6%，远小于量化带来的 8× 内存节省。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
KIVI 中 R 默认为 128。ablation 实验显示 R∈{32, 96, 128} 效果相近但都远好于无 residual（fake quantization），说明一定量的全精度窗口对于保持精度至关重要。实现时 residual 和 grouped 通过 tiled matrix multiplication 组合计算 attention。类似设计也被 StreamingLLM 等使用（保留最近的 token 维持精度），但目的不同（StreamingLLM 是 evict 而非 quantize）。

涉及论文标题：
- KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

---
