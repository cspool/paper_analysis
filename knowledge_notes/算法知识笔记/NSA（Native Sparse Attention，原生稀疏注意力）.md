## NSA（Native Sparse Attention，原生稀疏注意力）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
NSA（Native Sparse Attention，DeepSeek 2025）是"硬件对齐且原生可训练"的稀疏注意力：把注意力分解为压缩（compress，聚合粗粒度 token 块）、选择（select，按重要性选细粒度块）、滑动窗口（sliding window，局部窗口）三条并行分支，压缩/选择分支的块稀疏结构可与硬件对齐（如块粒度稀疏、压缩 block 紧凑布局），保持训练效率的同时降低长上下文注意力成本。QiMeng-Tensify（ISCA'26）把 NSA 列为最新 benchmark 子图（Table VII，Arch. 列 Transformer）：它是稀疏、非均匀负载的算子，验证框架对新算子的泛化能力。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
NSA 子图的稀疏注意力计算骨架：
```
for q_block in query_blocks:
    # 三条并行分支：
    o_c = attention(q, K_compressed, V_compressed)   # 压缩分支（粗粒度）
    o_s = attention(q, K_selected,   V_selected)     # 选择分支（细粒度，块稀疏）
    o_w = attention(q, K_window,     V_window)       # 滑动窗口分支（局部）
    o   = fuse(o_c, o_s, o_w)                        # 输出融合
```
在 QiMeng-Tensify 中该子图作为输入 TensorIR 被图重写/MCTS 优化：结果相对 Triton 快 1.51×、相对 Reasoning Compiler 快 1.18×（FP16，A100），略低于专家手写 FlashAttention 但证明对稀疏非均匀新算子可泛化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
实现：作为 benchmark 子图（Table VII），与 FlashAttention 同为"最新算子"代表（论文 E 节：benchmark 含 Mirage 的子图集 + LayerNorm + 最新算子 FlashAttention/NSA）；FP16 TensorCore 评估对比 Triton/Reasoning Compiler 等。使用方式：验证 QiMeng-Tensify 对"尚未有成熟手写实现/模板"的新算子的自动优化能力（专家仅 FlashAttention 领先，其余自动方法均落后），说明其一般化范式可覆盖稀疏非均匀负载。

涉及论文标题：
- QiMeng-Tensify Scaling up Tensor Computation Optimization via Architecture-Aware LLM-Guided MCTS
