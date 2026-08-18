## Hard Swish（硬 Swish / h-swish 激活函数）

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
- Hard Swish 是 Swish 的分段线性近似：h-swish(x)=x·ReLU6(x+3)/6，其中 ReLU6(x)=min(max(x,0),6)。Swish（SiLU）x·σ(x) 依赖指数运算，在低精度下指数放大离群元素、压缩小元素，量化损失剧增且计算开销大。Hard Swish 消除指数运算，分段线性天然适合低精度（范围有界、可精确表示），表示能力对推荐模型典型输入范围与 Swish 相当。
- LoKA 用它替换 LRM 中重度使用的 sigmoid 型激活（Swish x·σ(x)、SwishNorm x·σ(Norm(x))）——LoKA Probe 识别出 sigmoid 不稳定是三类低精度隐患之一。Hard Swish 与 BlockNorm 天然可融合进同一 kernel（归一化+激活+量化全在 epilogue）。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
- 计算过程：输入 x → ReLU6(x+3)=clamp(x+3,0,6) → 乘 x → 除 6。分段表达：x≤−3 时输出 0；−3<x<3 时 x(x+3)/6（二次）；x≥3 时输出 x。全程无指数、无分支发散（可用 clamp+乘加实现），FP8 低精度下数值稳定。
- 示例（张量计算）：x=[-4, -1, 2, 5] → ReLU6(x+3)=[0, 2, 5, 6] → h-swish=[0, -0.333, 1.667, 5]。对比 Swish：σ(-4)≈0.018、σ(-1)≈0.269、σ(2)≈0.881、σ(5)≈0.993 → swish=[-0.072,-0.269,1.762,4.966]，在典型输入范围内近似。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
- 实现：作为模型激活替换（把 nn.SiLU 换为 hard-swish 自定义 autograd 或融合进 GEMM epilogue kernel）；训练/推理同定义（无需特殊处理）。使用场景：低精度（FP8/INT）训练与推理中替代指数型激活，与块归一化、量化融合减少 kernel 数与 HBM 流量。注意：Hard Swish 首次广泛用于 MobileNetV3（效率优先），LoKA 将其用于低精度 LRM 稳定性；tradeoff 是略简化激活动力学换取低精度稳定性。

涉及论文标题：
- LoKA: Low-precision Kernel Applications for Recommendation Models At Scale
