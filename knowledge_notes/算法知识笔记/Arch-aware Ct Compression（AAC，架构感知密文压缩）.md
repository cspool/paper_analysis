## Arch-aware Ct Compression（AAC，架构感知密文压缩）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- AAC（Architecture-aware Ciphertext Compression）是 FEnc² 的第二个组件：一种跨层槽利用率优化机制，在通道/特征缩减层（如 MobileNet/SqueezeNet/ResNet 的 1×1 卷积把通道 N_in 压到 N_DS）之后，用轻量 rot-mask-add 序列把稀疏填充的密文压实成满装密文，恢复密文密度、减少后续层密文数，且不改变打包格式、自动适配所有中间形状。
- 动机：CKKS 对所有槽做 SIMD 运算，N_DS<α 时稀疏密文浪费计算并迫使后续层用更多密文覆盖 N_out 输出通道（Fig.5 瓶颈 8→2→8：无 AAC 需 4 个 25% 利用率的稀疏密文、4× HE 计算；有 AAC 只需 1 个满装密文）。Fhelipe 等虽做层后槽合并，但忽略下一层计算模式，布局次优。
- 关键设计：AAC 不增加乘法深度——其 0/1 明文掩码不要求比卷积权重更高的明文 scale（标准 CKKS 每 PMult 用统一 scale Δ，权重与掩码分别编码在 Δ₁、Δ₂ 且 Δ₁·Δ₂=Δ），两个连续乘法后只做一次 rescale，不额外消耗 level。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 瓶颈块 8→2→8 通道的 AAC 压实流程（本论文 Fig.5）：
```
Step1: 1×1 缩减层 8→2 通道 → 密文含 8 槽但仅 2 槽有效（25% 利用率）
Step2: 施加 0/1 掩码的 rot-mask-add：
       ct_dense = Σ_{valid} Rot(PMult(ct_reduced, mask_c), offset_c)
       即对每个有效通道，掩码提取（PMult）+ 旋转对齐（Rot）+ 累加（Add）
       —— 2 个有效通道压实到同一密文的相邻槽
Step3: 扩展层 8 输出通道只需 1 个满装密文（无 AAC 需 4 个）
       scale：权重 PMult 用 Δ₁、掩码 PMult 用 Δ₂，Δ₁·Δ₂=Δ → 只做一次 Rescale
```
- Annotations：rot-mask-add 是 FHE 中的"数据复制/重排"标准模式（Fhelipe/Pantheon/Coeus 也用于复制或通信），AAC 把它专门用于跨层保密度；步骤中旋转数受 Conv-aware Encoding 的旋转上界约束（AAC 不破坏布局的旋转保证）。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在推理图编译阶段，FEnc² 识别所有通道缩减-扩张模式（fire-module、residual-shortcut 等），对每个缩减点自动插入 AAC 重打包；运行期执行 mask-rotate-add 的密文操作。效果（Table X）：fire-module 各层加速 1.47×-4.68×（合计 2.09×），residual-shortcut 各层 1.016×-1.75×（合计 1.64×），slot 利用率从 0.02-0.5 恢复到 1.0；端到端重打包开销仅占总延迟 0.42%-3.7%。
