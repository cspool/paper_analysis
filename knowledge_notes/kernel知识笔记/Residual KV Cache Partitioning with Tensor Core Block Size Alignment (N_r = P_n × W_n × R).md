## Residual KV Cache Partitioning with Tensor Core Block Size Alignment (N_r = P_n × W_n × R)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

Residual KV Cache Partitioning 是 BitDecoding 的 KV cache 管理策略：将 KV cache 分为 packed low-bit 区（X_pack）和 FP16 residual 区（X_res），以 Tensor Cores tiling 粒度 N_r 为基本对齐单元。Residual block size N_r = P_n × W_n × R，其中 P_n 为 mma tile 沿 N 的元素数（如 m16n8k16 → 8），W_n 为沿 N 维的 warp 数，R = 16/β 为 packing ratio。这种对齐确保：(1) 每个低比特 fragment 完全填充 TC tile，饱和 TC；(2) Quantization 以 N_r 为单位执行，与 TC 计算自然的批粒度对齐；(3) Residual buffer 很小（N_r < 256），开销可忽略。

从kernel调度角度拆解术语。

```
// KV Cache Partitioning 伪代码
// 输入：Prefill 后 FP16 KV cache X ∈ R^{L×d}, bit-width β
R = 16 / β                      // e.g., 4-bit → R=4, 2-bit → R=8
N_r = 8 × W_n × R               // P_n=8 (m16n8k16), e.g., W_n=4, β=4 → N_r=128
N_pack = L - (L mod N_r)        // 对齐 packed 部分
res_len = L mod N_r             // residual 部分 < N_r

X_pack = X[:N_pack]             // 量化+packed low-bit KV cache
X_res  = X[N_pack:]             // FP16 residual KV cache (< N_r tokens)

// Decode step: 每个新 token append 到 X_res
// 当 res_len == N_r → 触发 Residual Kernel: 量化+pack → 追加到 X_pack
// 清空 X_res → 循环

// 保证: 每个 packed tile 精确填充 TC fragment (无 padding / zero-padding waste)
```

术语一般如何实现？如何使用？

N_r 由 hardware instruction configuration 自动推导：根据 GPU 架构确定 mma variant → 得到 P_n → 根据 β 自动计算 R → 根据经验或 tuning 确定 W_n → 计算 N_r。Residual cache 存储为 pre-allocated FP16 buffer（size = N_r × d × 2 for K+V）。Residual overhead 极小（seq_len >> N_r），以 seq_len=32K, N_r=128 为例：overhead = 128/32000 = 0.4%。

涉及论文标题：
- BitDecoding: Unlocking Tensor Cores for Long-Context LLMs Decoding with Low-Bit KV Cache

---
