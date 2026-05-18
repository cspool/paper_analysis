## Residual KV Cache with Block Size Alignment (Nr = Pn × Wn × R)

术语是什么？通过联网搜索让回答具体和精准。

Residual KV Cache 是 BitDecoding 中为对齐 Tensor Cores warp-tiling 而引入的 KV cache 分区策略。将 total KV cache X ∈ R^{L×d} 分为两部分：X_pack = X[: L-Nr]（量化后 packed 存储）和 X_res = X[L-Nr:]（保留 FP16 精度的 residual cache）。Residual block size Nr = Pn × Wn × R，其中 Pn 是单个 warp tile 在 N 维度处理的元素数（如 mma.m16n8k16 下 Pn=8），Wn 是 N 维度 warp 数，R = ω/β 是 packing ratio（如 INT4→INT16, R=4）。Decoding 时新生成的 K/V tokens 先追加到 FP16 residual cache，当累计达 Nr 时触发 Residual Kernel 将其批量量化写入 packed cache。该设计保证每个 Tensor Cores fragment 被完整填充（无 underfill 导致的 compute waste），同时 residual cache overhead 极小（Nr 通常 <256，seq_len >> Nr 时仅占很小比例——paper Fig 14 显示 128K 下 overhead < 0.02ms）。

从 kernel 调度角度拆解术语：

```
// === Residual KV Cache 调度流程（Decode Step） ===
// 状态: residual_len 当前residual cache中FP16 token数

// Step 1: 新生成K/V追加到residual cache
FP16 K_new[d], V_new[d];  // 本轮decode新生成
residual_K[residual_len] = K_new;
residual_V[residual_len] = V_new;
residual_len++;

// Step 2: Packing Kernel执行attention
// 同时使用packed low-bit cache (L - residual_len tokens) 
// 和 residual FP16 cache (residual_len tokens)
packing_kernel_attention(Q, K_packed, V_packed, 
                         residual_K, residual_V, residual_len);

// Step 3: 若residual_len == Nr，触发Residual Kernel
if (residual_len == Nr):
    residual_kernel(residual_K, residual_V, Nr);  // 量化+pack → K_packed/V_packed
    residual_len = 0;  // 清空residual cache
```

Nr 计算示例（INT4, mma.m16n8k16, Wn=4）：Nr = 8 × 4 × 4 = 128。即每128个新token触发一次批量量化。

术语一般如何实现？如何使用？

Residual block size Nr 由 hardware instruction configuration 自动推导：根据 GPU 架构确定 mma variant（Ampere: m16n8k16, Hopper: wgmma.m64n64k16）→ 得到 Pn → 用户配置量化 bit-width β → 自动计算 R → 根据经验或 tuning 确定 Wn → 计算 Nr。Residual cache 存储为 pre-allocated FP16 buffer（size = Nr × d × 2 for K and V）。与 KIVI 的 per-token quantization（每 decode step 都量化，更高 overhead）和 continuous-packing baseline（每次 quantization 需 layout transform）相比，residual block 策略通过批量量化 amortize overhead 并用 ldmatrix 消除 layout transform。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache

