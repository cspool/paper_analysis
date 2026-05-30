## Sequence Parallelism for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Sequence Parallelism for Attention 是将 Transformer 的 self-attention 计算在序列维度上跨多个设备（GPU）并行化的一类技术。核心思路是将长序列的 K,V 分片到多个设备，避免单个设备内存无法容纳完整 KV cache。不同于 Tensor Parallelism（沿 head/隐藏维度切分）和 Pipeline Parallelism（沿层切分），Sequence Parallelism 沿 token 序列维度切分工作和数据。

主要方法包括：
- **Ring Attention**（Liu et al., 2023）：KV 在 GPU 间 P2P 环形传递，每 GPU 依次处理所有 chunk。
- **Tree Attention**（本论文）：KV 不移动，通过 AllReduce 归约部分结果，通信步数 O(log p)。
- **Ulysses/DeepSpeed-Ulysses**：通过 All-to-All 在序列维度和 head 维度之间转换分片方式。
- **Star Attention**：两阶段——blockwise-local attention + distributed query-anchor softmax。

从算法pipeline角度拆解术语。
Sequence Parallelism 中不同方法的 pipeline 对比（以 p=4 GPU 解码为例）：
```
Ring Attention (O(p) 通信步):
  GPU_0: [q, K_0, V_0] → attn → send(K_0,V_0)→GPU_1, recv(K_1,V_1)←GPU_3
  GPU_1: [q, K_1, V_1] → attn → send(K_1,V_1)→GPU_2, recv(K_2,V_2)←GPU_0
  ...循环 p 次...
  每次传输 2btd elements (K+V chunk)
  总通信量: p × 2btd

Tree Attention (O(log p) 通信步):
  GPU_0..3: [q, K_i, V_i] → FlashAttn2 → (o_i, lse_i)
  AllReduce(max, lse_i)           # tree, O(log p) 步, 1 elem
  AllReduce(sum, n_i)             # tree, O(log p) 步, d_h elems
  AllReduce(sum, d_i)             # tree, O(log p) 步, 1 elem
  总通信量: 2(p-1)/p × (d_h + 2)

Ulysses (All-to-All):
  GPU 间通过 All-to-All 在 seq 维度和 head 维度间转换
  Attention 在 head-parallel 模式下计算
  每个 attention 需要 2 次 All-to-All
```

术语一般如何实现？如何使用？
实现：Ring Attention 通过 NCCL P2P send/recv。Tree Attention 通过 NCCL AllReduce（JAX `lax.pmax`/`lax.psum`）。Ulysses 通过 NCCL All-to-All（DeepSpeed 或 PyTorch `dist.all_to_all`）。选择哪种方法取决于：(a) 硬件拓扑——homogeneous 带宽适合 Ring，两层拓扑适合 Tree/AllReduce；(b) 序列长度——长序列 Ring 通信量大，Tree 通信量与序列长度无关；(c) 是否训练或仅解码——训练时有多个 query，Ring 可 overlap，解码时单 query 无法 overlap。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters

---
