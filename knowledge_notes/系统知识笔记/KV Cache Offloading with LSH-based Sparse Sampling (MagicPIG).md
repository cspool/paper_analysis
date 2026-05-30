## KV Cache Offloading with LSH-based Sparse Sampling (MagicPIG)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。

MagicPIG的KV Cache Offloading是将完整的KV cache从GPU HBM移至CPU DRAM，同时在CPU侧维护LSH哈希表（存储每个key的哈希码和索引），GPU仅保留sink tokens和local tokens的KV（on-device cache）。通过LSH采样实现5-10×稀疏性（仅2-5%的KV参与attention计算），弥补CPU DRAM带宽（100-200 GB/s）与GPU HBM带宽（~1 TB/s）的10-20%差距。CPU上的attention计算使用FBGEMM（bfloat16）执行。

从系统架构角度拆解术语，比如术语如何在系统架构中发挥作用，给出术语在系统架构中运转流程的具体例子。

**MagicPIG GPU-CPU异构系统架构流程**：

```
┌───────────────────────────────────────────────────────────┐
│                    CPU (DRAM)                              │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Full KV Cache: {K_ℓ, V_ℓ} for all layers        │    │
│  │  Size: O(n × d × layers × heads)                 │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  LSH Hash Tables (per KV head, L tables each)     │    │
│  │  Each entry: hash_code → [key_indices]            │    │
│  │  Size: e.g., 14GB for 8B model, 96K ctx (10,150) │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  FBGEMM (bfloat16): Sparse Attention Computation  │    │
│  │  K_S = K[S], V_S = V[S]  →  ō_cpu = attn(q,K_S,V_S) │
│  └──────────────────────────────────────────────────┘    │
│                          │                                │
│         PCIe: q_code (K×L bits) + new KV → CPU            │
│                ō_cpu ← GPU (recursive attention merge)    │
│                          │                                │
└──────────────────────────┼────────────────────────────────┘
                           │
┌──────────────────────────┼────────────────────────────────┐
│              GPU (HBM)                                     │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Model Weights + MLP + Linear Projections         │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  HashEncode: q_code = Sign(q @ W)                │    │
│  │  W ∈ R^{d×(K×L)}, shared across all heads        │    │
│  │  Compute overhead: 1.8-8.5% of GPU linear proj   │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  On-device Cache: sink tokens (first few) +       │    │
│  │  local tokens (recent 64/24) KV cache             │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  Recursive Attention Merge: ō = merge(ō_cpu, ō_gpu)  │
│  └──────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────┘
```

执行顺序：GPU (1)线性投影 → (3)HashEncode → 数据传输到CPU → CPU (4)哈希查询+注意力 → 结果传回GPU → GPU (2)Recursive Attention Merge。

术语一般如何实现？如何使用？

MagicPIG的offloading使batch size可达GPU全注意力baseline的12×以上（因KV cache不占用GPU显存）。三种硬件场景验证：A100-80GB + CodeLlama-34B (16K): 1.5× throughput；L20-48GB + CodeLlama-13B (16K): 5.0× throughput；模拟RTX 4090-24GB + Llama-3.1-8B (96K): 3.3× throughput, 单请求54ms解码延迟。限制：(1) 需要足够的CPU DRAM存储哈希表+KV cache；(2) PCIe带宽需传输hash codes和新KV；(3) 尚未实现prefill阶段的offloading。

涉及论文标题：
- MagicPIG: LSH Sampling for Efficient LLM Generation

---
