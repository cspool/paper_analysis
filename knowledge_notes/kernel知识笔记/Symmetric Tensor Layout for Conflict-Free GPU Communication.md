## Symmetric Tensor Layout for Conflict-Free GPU Communication

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。

Symmetric Tensor Layout (L ∈ R^{P×R×B×E×C×H}) 是 FlashMoE 提出的多维张量布局，实现 PGAS 模型中 write-write conflict-free 的跨 GPU one-sided 通信。各维度: P=world_size, R=通信轮次 (DISPATCH/COMBINE, 共 2), B=staging buffer (OUTGOING/INCOMING, 每 round 2 个), E=local experts, C=upscaled expert capacity, H=embedding dim。核心 insight: 将 source GPU rank p_s 嵌入 L 第一维索引，使不同 source GPU 的 one-sided write 天然写入不同内存位置 (L[p_s1,...] ≠ L[p_s2,...]) → 无需锁同步。Temporal buffering (每 round 独立 buffer) 隔离 dispatch 和 combine 并发访问，实现 fully non-blocking memory access。

从kernel调度角度拆解术语，比如术语所在kernel调度的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。

Theorem 3.1 证明 (sketch): 任两 write w1(ps1,pt1,i1), w2(ps2,pt2,i2) 若冲突 → pt1=pt2 且 i1=i2。i1=(p1=ps1,r1,b1=1,e1,c1), i2=(p2=ps2,r2,b2=1,e2,c2)。若 i1=i2 则 p1=p2 → ps1=ps2，但冲突定义要求 ps1≠ps2 → 矛盾 ✓。实际: GPU 0 Processor dispatch expert 3 tile to GPU 2 → 写 L[0][DISPATCH][INCOMING][2][3][15]。GPU 1 Processor dispatch → 写 L[1][DISPATCH][INCOMING][2][x][y]。第一维 p 不同 → 无冲突。Size(L) ≈ 4×Size(T) (2R×2B), 实际 overhead ≤2% of inference memory。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。

实现: (1) 每 GPU 用 `nvshmem_malloc()` 分配等量 L，建立对称地址映射；(2) Expert capacity C 对齐到 tile height bM=128 倍数；(3) Temporal buffering: B=2 double-buffering 隔离 dispatch output 和 combine input；(4) Memory overhead 实测: Mixtral 8x7B 2.15%, DeepSeek-V3 0.11%。

涉及论文标题：
- FlashMoE: Fast Distributed MoE in a Single Kernel
