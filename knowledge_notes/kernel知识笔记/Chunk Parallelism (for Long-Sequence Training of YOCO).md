## Chunk Parallelism (for Long-Sequence Training of YOCO)

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Chunk Parallelism 是 YOCO 为分布式长序列训练提出的并行策略，利用 Cross-Decoder 解耦注意力依赖的特性来减少 GPU 通信开销。在标准序列并行中，序列被分割到多个设备，每层 self-attention 都需要 all-gather 通信来交换 KV。YOCO 的 Chunk Parallelism 将序列切分为多个 chunks 分配到不同 GPU：Self-Decoder 仅需在相邻设备间传递边界信息（如 gated retention 的 recurrent state S 或 sliding-window 的边界 tokens）；Cross-Decoder 的 K̂,V̂ 则仅需**一次** all-gather（而非每层一次），因为所有 cross-decoder 层共享同一组缓存。这大幅降低了通信频率、减少了 GPU memory fragmentation，使 YOCO 在极长序列训练时具可扩展性优势。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
Chunk Parallelism 在两 GPU 上的训练流程：

```
Sequence [x_1, ..., x_{2N}] → Split into two chunks

GPU 0: [x_1, ..., x_N]        GPU 1: [x_{N+1}, ..., x_{2N}]

=== Self-Decoder (per-device, with boundary communication) ===
for layer in self_decoder_layers:     # layers 1..L/2
    # GPU 0 sends last tokens of its chunk to GPU 1 (for window/recurrent state)
    # GPU 1 receives boundary state from GPU 0
    # Each GPU computes efficiently within its chunk
    X_0 = SelfDecoderLayer(X_0)       # local chunk
    X_1 = SelfDecoderLayer(X_1)       # local chunk
    # Communication volume: O(C*d) for sliding-window or O(d²) for retention

# Output: M_0 ∈ R^{N×d}, M_1 ∈ R^{N×d}

=== Generate Global KV Cache (one-time all-gather) ===
K̂_0 = proj_K(M_0), V̂_0 = proj_V(M_0)  # local computation
K̂_1 = proj_K(M_1), V̂_1 = proj_V(M_1)

# All-gather K̂, V̂ across all devices — ONLY ONCE!
K̂ = AllGather([K̂_0, K̂_1])  # concatenated [2N, d] on both GPUs
V̂ = AllGather([V̂_0, V̂_1])

=== Cross-Decoder (K̂,V̂ already replicated on all devices) ===
# No further communication needed for attention!
for layer in cross_decoder_layers:    # layers L/2+1..L
    Q = proj_Q(X)
    O = CrossAttention(Q, K̂, V̂)       # local, K̂,V̂ already complete
    X = SwiGLU(O)
    # Collect output only at classification head
```

**Annotations**: Self-Decoder 的通信量受高效 attention 限制（sliding-window: O(C×d), gated retention: O(d²)），远小于全局 attention 的 O(N×d)。Cross-Decoder 的 all-gather 仅传输 K̂,V̂（O(N×d)），仅一次。对比标准 Transformer 每层都需要 all-gather Q,K,V（O(L×N×d)），chunk parallelism 减少了约 L× 的通信量。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Chunk Parallelism 的实现和使用：(1) 基于 SuperScaler 或 Megatron-LM 的序列并行框架实现；(2) 适用于极长序列训练场景（百万 token 级别）；(3) Self-Decoder 的边界通信可以使用 P2P send/recv（比 all-gather 更高效）；(4) Chunk 数量可以动态调整——chunk 越多则每设备序列越短（内存节省），但边界通信总量增加。论文未开源 Chunk Parallelism 的实现代码，仅描述了算法原理。限制：需要 YOCO 架构（Cross-Decoder 共享 KV cache）；Self-Decoder 边界通信对于 sliding-window 很简单（直接传 tokens），对于 gated retention 需要传 state S（O(d²) which 在 head_dim 较大时可能成为瓶颈）。

涉及论文标题：
- Efficient implementations for emerging model architectures (YOCO: You Only Cache Once)
