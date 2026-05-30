## Givens Rotation for LLM Weight Quantization

术语是什么？回答尽量完整，回答逻辑链中每一步都解释出来。通过联网搜索让回答具体和精准。
Givens 旋转（Givens Rotation）是一种正交线性变换，在二维平面内旋转两个坐标轴方向的值，保持其他维度不变。数学上，G(i, j, θ) 是一个 n×n 单位矩阵，仅在 (i,i)、(i,j)、(j,i)、(j,j) 四个位置替换为 cosθ、-sinθ、sinθ、cosθ。其核心性质：(1) 正交性——G^T G = I，保证范数不变，数值稳定（不引入舍入误差放大）；(2) 仅修改两行/两列，计算成本 O(1) 而非全矩阵乘法的 O(n²)。在 ParoQuant 中，Givens 旋转被用于对 LLM 权重矩阵的输入通道进行成对旋转以抑制离群通道：旋转一对通道 i 和 j 后，W'[i,:]=cosθ·W[i,:]-sinθ·W[j,:]，W'[j,:]=sinθ·W[i,:]+cosθ·W[j,:]。这一操作将离群通道的大值通过正交混合分散到正常通道中，收窄每个量化组内的动态范围。参数由 θ（旋转角度，可学习）和 (i,j)（通道对索引）组成，仅需 1 个 float + 2 个 int 参数 per pair。

从算法pipeline角度拆解术语，比如术语所在pipeline的伪代码或具体计算过程，给出具体例子。通过联网搜索让回答具体和精准。
```
# Givens 旋转作用于权重 (变换 W，离线)
def givens_rotate_weight(W, i, j, theta):
    c, s = cos(theta), sin(theta)
    W[i,:], W[j,:] = c*W[i,:] - s*W[j,:], s*W[i,:] + c*W[j,:]

# Givens 逆旋转作用于激活 (推理时)
def givens_inverse_rotate_activation(X, i, j, theta):
    # 逆变换: angle = -theta, 即 cos(-θ)=cosθ, sin(-θ)=-sinθ
    c, s = cos(theta), sin(theta)
    X[:,i], X[:,j] = c*X[:,i] + s*X[:,j], -s*X[:,i] + c*X[:,j]
```
在 ParoQuant 中，128 通道的每个 group 内，K=8 个 independent rotation 各含 64 个互不重叠的 Givens 旋转。推理时每对旋转仅需 4 次 FMA（每个 token），所有 pair 间无数据依赖，可完全并行化。

术语一般如何实现？如何使用？通过联网搜索让回答具体和精准。
Givens 旋转在量化中的优势：(1) 相比全旋转矩阵（O(n²) 参数），Givens 旋转只需 O(1) 参数 per pair，内存占用极小；(2) 旋转角度通过 AdamW 逐层学习（初始化为 0，对应恒等变换），避免 Hadamard 变换的固定/随机性质；(3) 推理时逆变换在 fused CUDA kernel 中执行，角度和配对索引存储在寄存器中，激活在 shared memory 中完成全部旋转。与 Hadamard 的关键区别：Hadamard 是固定变换有全局依赖（Butterfly 结构，需 O(n log n) 步），Givens 是局部的、可学习的、完全并行的。

涉及论文标题：
- ParoQuant Pairwise Rotation Quantization for Efficient Reasoning LLM Inference

---
