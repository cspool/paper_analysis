## Identity Mapping Property (in Residual Connections)

术语是什么？

Identity Mapping 是残差连接的设计原则：shallower layer 信号直接、不经修改地映射到 deeper layer。标准残差连接递归展开为 $\mathbf{x}_L = \mathbf{x}_l + \sum_{i=l}^{L-1} \mathcal{F}(\mathbf{x}_i, \mathcal{W}_i)$，$\mathbf{x}_l$ 项体现 identity mapping。前向保证浅层信号 norm 不因残差结构本身变化；反向保证梯度有直接路径 $\frac{\partial \mathcal{L}}{\partial \mathbf{x}_l} \supset \frac{\partial \mathcal{L}}{\partial \mathbf{x}_L}$，避免梯度消失（He et al., 2016b）。

从算法pipeline角度拆解：

HC 破坏了 identity mapping——递归展开中的 $\mathbf{x}_l$ 被 $(\prod \mathcal{H}_{L-i}^{\text{res}}) \mathbf{x}_l$ 替代，无约束的 $\mathcal{H}^{\text{res}}$ 乘积可能极大（~3000×）或极小，导致信号爆炸/消失。mHC 通过双随机约束恢复：乘积仍为双随机 + 谱范数 ≤ 1 + 凸组合保持均值。

术语一般如何实现？如何使用？

微架构实现方式：Pre-Norm Transformer（Layer Norm 在 sublayer 之前）、ReZero（零初始化残差分支）、mHC（流形约束投影）。核心目标是保证梯度能无阻碍回传。

涉及论文标题：
- mHC Manifold-Constrained Hyper-Connections

---
