## BSGS（Baby-Step-Giant-Step）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- BSGS（Baby-Step-Giant-Step）原是数论中解离散对数的时空权衡算法，在 HE 推理中被借用来分解多通道卷积的嵌套循环、降低旋转复杂度：把卷积按核/通道维度拆成"小步（baby step）"与"大步（giant step）"两组，用预旋转副本 + 分组聚合替代逐元素旋转，是 SOTA HE 推理系统（Orion、Multiplexed、HEAR 等）普遍采用的技巧（本论文 Sec. III-A 默认 HE 多通道卷积使用 BSGS）。
- 本质：把需要 K² 次内旋转的空间卷积，与需要 (α−1) 次外旋转的通道聚合解耦，避免把 K²×α 的旋转全部串行化；通过预先旋转并缓存输入密文的若干副本（giant step），后续只需较少组合（baby step 增量旋转）即可覆盖所有窗口/通道对齐，以内存换旋转次数。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 一个 3×3 核、α 通道打包的卷积 BSGS 分解示意：
```
# Giant step：预旋转输入副本（一次性代价）
for k in 0..g-1:  X_giant[k] = Rot(X, giant_offset(k))     # g 个大步旋转
# Baby step：对每个输出位置做小步增量
for b in 0..b-1:
    Y_b = Σ_k  PMult(X_giant[k], W_k)      # 组合大步副本
    Y_b = Σ_b'  Rot(Y_b, baby_offset(b'))   # 小步聚合
```
- Annotations：总旋转从 K²×α 量级降为 g+b 量级（g·b≈K²）；FEnc² 的 Conv-aware Encoding 与 BSGS 正交可叠加——块分解（fragment）进一步降低每密文需预旋转的副本数，二者共同把旋转复杂度压到 O(K)；本论文表 III 中 Orion/HELayers 的复杂度即含 BSGS 的基线，FEnc² 优于它们。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：在 HE 推理框架的卷积内核中，先做输入密文的多偏移预旋转并缓存（消耗内存），再按核权重做明文乘与累加，最后用少量旋转对齐输出。使用场景：任意 HE 多通道卷积/全连接层（对角线矩阵乘也可用 BSGS）；本论文 FC 层即"对角线矩阵乘法 + BSGS"。局限：预旋转副本占用内存、且打包布局差时收益有限——这正是 FEnc² 强调"布局决定 BSGS 效率"的原因。
- HE² 补充视角（ISCA'26，BSGS 与 hoisting 的硬件权衡）：在 EVF 单体 ASIC（SHARP）中 bs 与 gs 相等（均 8）时计算代价最低，但 baby-step 密文超片上容量，故取 bs=4；在 IRF 异构 + hoisting 场景下，bs 与 gs 差距更大反而暴露更多 keyswitch 并行、减少计算与通信，但增大 evk 存储需求（可能超 HBM 容量，Fig. 7）。BSGS 把一个并行度 D、入/出度为 1 的 PKB 拆成两个串行 PKB（PKB1 并行度 bs、PKB2 并行度 gs=D/bs），降低 keyswitch 并行度、提高子图出入度。HE² 的 HERO 框架按内存约束选择 BSGS 配置：8 GB HBM 足够时禁用 BSGS（C2S/S2C），内存受限时偏好 bs 与 gs 差距大的配置；BERT 首 FFT 阶段因高层级仍保留 bs=2/gs=32（见"PKB 与 PKB 融合"条目）。
- HyperDrive 补充视角（ISCA'26，BSGS PCMM 与密文复用 PMAC）：在 CKKS bootstrapping 的 CtS/StC 相位，BSGS 把明文-密文矩阵向量乘（PCMM）的旋转数从 O(N) 降到 O(√N)（bs×gs 个非零对角、bs+gs 量级旋转），配合 hoisting（复用 BS 相位的 ModUp 输出）与 double hoisting [7]（PMAC 中间量全程留在 R_{PQ_ℓ}，省掉大部分 ModDown）。BS 相位只做一次预旋转（ModUp）并在 BS-Rots 间复用，故 (NTT2-IP) 协同优化无处可用；GS 相位对密文 b(X) 做 ModDown 回 R_{Q_ℓ} 后接标准 KeySwitch。HyperDrive 的 CRPMAC 把全部 GS-Rots 推迟到末尾、按 GS 方向批处理 PMAC——单个 baby-step 密文乘以一批明文（一次 GMEM 读复用 gs 次），对比 [22] 按 BS 方向批处理的内存足迹更小；另对 bootstrapping 做操作重排：GS 相位 ModDown 紧跟 ModUp 时把 EWSub 提前，使 ModDown INTT 与 ModUp INTT 可批处理并合并进 (BConv2-NTT1) kernel。

涉及论文标题：
- FEnc2: Unifying Data Packing for Efficient Private Inference via Convolution and Architecture-Aware Fragment Encoding
- HE^2: A Communication-Light Heterogeneous Architecture for Efficient Fully Homomorphic Encryption
- HyperDrive: Hierarchical Exploitation of Memory Efficiency for GPU-Based FHE Acceleration
