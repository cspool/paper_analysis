## Topology-Aware AllReduce for Attention

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。
Topology-Aware AllReduce for Attention 是 Tree Attention 中利用 NCCL 集体通信库的拓扑感知 AllReduce 操作来实现跨 GPU 注意力归约的技术。现代 GPU 集群具有两层网络拓扑：(a) intra-node：NVLink 4.0（900 GBps, all-to-all 拓扑），同一节点内 GPU 间带宽极高；(b) inter-node：InfiniBand NDR（400 Gbps per link, ~50 GBps），跨节点带宽显著低于 intra-node。NCCL 的 AllReduce 自动检测此拓扑——intra-node 使用 ring reduce（利用 NVLink 高带宽），inter-node 使用 tree reduce（减少跨节点数据传输，仅传递部分归约结果）。

从kernel调度角度拆解术语。
Tree Attention 中 AllReduce 的执行流程（8 GPU, 1 DGX node, intra-node）：
```
# 8 H100 GPU 同节点，NVLink 4.0 (900 GBps) all-to-all topology

# === AllReduce Step 1: Reduce-Scatter (ring within node) ===
# NCCL 自动选择 ring reduce algorithm for intra-node
# 8 GPU 分成 2 组 (NVSwitch fully-connected 允许任意分组)
for step in 0..7:
    partial_max = max(local_max, received_max)
    send(partial_max) → next_GPU_in_ring

# === AllReduce Step 2: AllGather (from ring to all) ===
for step in 0..7:
    recv(reduced_chunk) ← prev_GPU_in_ring
    send(reduced_chunk) → next_GPU_in_ring

# 时间: ~1-2 μs per scalar AllReduce (intra-node NVLink)
# 对比: P2P send K,V chunk (80K×128×2 bytes ≈ 20MB) 需要 ~0.02ms
```

跨节点场景（16 nodes × 8 GPU, inter-node via InfiniBand）：
```
# NCCL uses hierarchical algorithm:
# Phase 1: intra-node ring reduce (NVLink, fast)
# Phase 2: inter-node tree reduce (InfiniBand, slow — but only scalar results)
# Phase 3: intra-node broadcast (NVLink, fast)

# 关键：只有部分归约结果（标量级）跨越 InfiniBand
# 避免像 Ring Attention 那样每个 K,V chunk 都跨节点传输
```

与 Ring Attention 的 kernel 级对比：
```
Ring Attention P2P communication:
  rank_send/recv(K_chunk[t, d_h], V_chunk[t, d_h])  # 2bt×d_h elements
  → 每个 P2P step 传输 20MB (for t=80K, d_h=128, BF16)
  → inter-node bandwidth (50 GBps) 下每步 ~0.4ms
  → p=8 个 step → ~3.2ms 纯通信（不可 overlap with decode compute ~10μs）

Tree Attention AllReduce communication:
  AllReduce(max, lse):          1 element   → ~1μs (intra-node)
  AllReduce(sum, n_local):      d_h elements → ~1μs (intra-node)
  AllReduce(sum, d_local):      1 element   → ~1μs (intra-node)
  → Total: ~3μs (intra-node) vs ~3.2ms (Ring intra-node P2P)
  → 通信量: ~130 elements vs ~20M elements (Tree vs Ring)
```

对于 AllReduce 的 `sum` reduction，使用 reduce-scatter + all-gather 算法（ring 变体）。对于 `max` reduction，使用 reduce + broadcast 算法（tree 变体）。NCCL 自动选择最优算法。

术语一般如何实现？如何使用？
实现：JAX 中通过 `lax.pmax` (max reduction) 和 `lax.psum` (sum reduction) 调用 NCCL。在 PyTorch 中等效为 `torch.distributed.all_reduce`。关键实现细节：Tree Attention 的 JAX 实现（Appendix D）使用 shard_map 在序列轴（'i'）上分片 K,V，然后调用 lax.pmax 和 lax.psum。

使用方式：被 Tree Attention 的 `tree_flash_decode` 函数替代 Ring Attention 的 P2P 通信模式。无需修改 NCCL 配置——NCCL 自动检测网络拓扑并选择最优算法。

涉及论文标题：
- Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters
